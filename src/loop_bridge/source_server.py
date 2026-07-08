"""In-process Source Bus presence for the Vega RobotEnv.

Presents a Vega robot — single- or bimanual — as ONE ``robot-obs`` source and
executes ONE ``robot-action``, per the robot source contract. The combiner of the
arms lives here (the robot's own repo), not in loop or loop-sdk.

- ``_LockedStepService`` is the upstream ``VegaRobotEnvService`` plus one fix: it
  serializes ``Step`` on the upstream ``_cmd_lock`` (upstream guards only ``Reset``),
  so the bus action lane can't race a Reset on shared IK/filter state.
- ``LoopRobotEnv`` owns the bus I/O over N arm services that share ONE hardware unit:
  an **obs poll** reads each arm's ``_create_observation`` on a clock and publishes
  the merged ``robot-obs`` (Vega computes obs only inside ``_create_observation``, so
  ``LoopRobotEnv`` must drive it — else teleop, which needs obs for a delta, and obs,
  driven by the resulting action's Step, deadlock at startup); an **action lane**
  subscribes ``robot-action`` and dispatches each arm's slice to that arm's Step.

A bimanual robot is ONE ``Robot`` exposing both arms; two per-arm services share it
(``VegaRobot``/service take an injected ``robot``), reusing every per-arm
gain/frame/interpolation/IK/gripper path verbatim.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import sys
import threading
from concurrent import futures
from typing import Any, Sequence

import grpc
from loop_sdk import (
    LoopRobotClient,
    RobotConfig,
    RobotConfigOptions,
)

# Importing the upstream module runs its sys.path setup and binds the proto stubs.
from dexcontrol.core.robotenv_vega import server as _vega_server
from loop_bridge.obs_publisher import merge_observations
from loop_bridge.robot_action import HOME, decode_action
from loop_bridge.robot_obs import DEFAULT_ARM_PREFIX

LOGGER = logging.getLogger("loop_bridge.vega")

DEFAULT_ACTION_SPACE = "target_cartesian_delta"
# Fallback obs publish rate when the action lane is idle. In steady state obs is now
# driven by robot-action arrivals (each Step is followed by a fresh post-step obs), so
# this rate governs only the boot lull and any teleop-hold gaps where actions stop
# flowing. Kept at the engine's default control_hz so obs never falls below the rate
# the engine expects to sample — dropping under control_hz would stall the engine's
# per-tick obs de-dup guard. See ``LoopRobotClient.run`` for the model.
DEFAULT_HEARTBEAT_HZ = 20.0


class _BusStepContext:
    """Minimal gRPC servicer context for replaying Step in-process.

    The Vega ``Step`` happy path never touches the context; we still surface an
    ``abort`` (which it would only call on a hard error) as an exception so the
    action lane skips that tick rather than silently succeeding.
    """

    def set_code(self, code: Any) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details

    def abort(self, code: Any, details: str) -> None:
        raise RuntimeError(f"Step aborted: {code} {details}")


class _StepApplier:
    """Adapts one arm service's ``Step`` to the action consumer's ``step(...)`` seam."""

    def __init__(self, service: Any) -> None:
        self._service = service

    def step(
        self, action: list[float], action_space: str, gripper_action_space: str
    ) -> None:
        request = _vega_server.robotenv_pb2.StepRequest(
            action=list(action),
            action_space=action_space,
            gripper_action_space=gripper_action_space,
        )
        response = self._service.Step(request, _BusStepContext())
        # Step swallows hardware faults into a non-SUCCESS StepResponse.status rather
        # than raising, so the action lane would otherwise treat a stalled command
        # (joint limit, IK failure, comms) as success. Surface it.
        status = getattr(response, "status", "") or ""
        if status and status != "SUCCESS":
            LOGGER.warning(
                "robot-action Step returned %s: %s",
                status,
                getattr(response, "message", ""),
            )
            raise RuntimeError(
                f"robot-action Step returned {status}: {getattr(response, 'message', '')}"
            )

    def home(self) -> None:
        """Home this arm: ``Reset(mode="home")`` — moves it to its home pose.

        The operational counterpart to ``step``; surfaces a non-SUCCESS reset the
        same way (the bus would otherwise treat a stalled home as success).
        """
        request = _vega_server.robotenv_pb2.ResetRequest(mode="home", params={})
        response = self._service.Reset(request, _BusStepContext())
        status = getattr(response, "status", "") or ""
        if status and status != "SUCCESS":
            raise RuntimeError(
                f"home Reset returned {status}: {getattr(response, 'message', '')}"
            )


class _LockedStepService(_vega_server.VegaRobotEnvService):
    """``VegaRobotEnvService`` whose ``Step`` is serialized on the upstream ``_cmd_lock``.

    Upstream takes ``_cmd_lock`` only in ``Reset`` — safe when Step had a single
    caller. The bus action lane is a second concurrent Step source, so without this
    an action-lane Step can race a Reset (or another Step) on shared IK / filter
    state and command a real arm a corrupted pose. ``Step`` never takes the lock
    itself, so this cannot self-deadlock.
    """

    def Step(self, request, context):
        with self._cmd_lock:
            return super().Step(request, context)


class LoopRobotEnv:
    """A Vega RobotEnv presented on loop: one robot-obs out, one robot-action in, over N arm services sharing a robot.

    Obs cadence is now driven by robot-action arrivals — each incoming action is Stepped
    and the post-step observation is republished immediately, so obs rate collapses to
    control rate rather than the previous 100 Hz free-run. ``heartbeat_hz`` is the
    fallback rate for the boot lull (before any action) and for teleop-hold gaps where
    no action is flowing; when actions arrive at or above that rate the heartbeat obs
    is suppressed. See ``LoopRobotClient.run`` in loop-sdk.
    """

    def __init__(
        self,
        arm_services: Sequence[tuple[str, Any]],
        *,
        loop_addr: str,
        action_space: str = DEFAULT_ACTION_SPACE,
        gripper_action_space: str = "",
        heartbeat_hz: float = DEFAULT_HEARTBEAT_HZ,
        enable_action: bool = True,
        action_space_options: Sequence[str] = (),
    ) -> None:
        if heartbeat_hz <= 0:
            raise ValueError(f"heartbeat_hz must be > 0, got {heartbeat_hz}")
        if not arm_services:
            raise ValueError("at least one (arm_prefix, service) is required")
        self._arm_services = tuple(arm_services)
        arm_prefixes = [arm_prefix for arm_prefix, _ in self._arm_services]
        if len(set(arm_prefixes)) != len(arm_prefixes):
            raise ValueError(f"duplicate arm prefixes: {arm_prefixes}")

        self._action_space = action_space
        self._gripper_action_space = gripper_action_space
        self._heartbeat_hz = heartbeat_hz  # fallback obs rate when the action lane is idle
        self._lane_stop = threading.Event()

        # Advertise only what this robot owns — the action space. control_hz is NOT the
        # robot's axis anymore: obs publishes fast and the RCI engine owns the control rate
        # (picked on the robot-control panel), so the robot must not advertise it.
        options = RobotConfigOptions(
            action_space=tuple(action_space_options) or (action_space,),
        )

        def apply_config(config: RobotConfig) -> RobotConfig:
            self.reconfigure(
                control_hz=config.control_hz, action_space=config.action_space
            )
            return config

        # One bus object owns the whole link: publish robot-obs + (when enabled)
        # consume robot-action + robot-command. Source ids are pinned by the SDK facade
        # (our lane convention); LoopRobotEnv owns the per-arm action decode. The obs poll
        # pulls poll_action()/drain_commands() each tick and Steps/homes each arm via a
        # per-arm _StepApplier. Obs-only (enable_action=False) wires neither input lane.
        self._loop_robot_client = LoopRobotClient(
            loop_addr,
            options=options,
            apply_config_callback=apply_config,
            enable_action=enable_action,
        )

        self._appliers: dict[str, _StepApplier] = {}
        if enable_action:
            self._appliers = {
                arm_prefix: _StepApplier(service)
                for arm_prefix, service in self._arm_services
            }

        # The SDK owns the loop. Vega is in-process with the RobotEnv gRPC server, so
        # ``run`` goes on a daemon thread (the main thread serves Step +
        # waits for shutdown). It publishes the bootstrap obs on connect (that first
        # publish is what breaks the obs/action startup cycle), then republishes obs
        # after each Step / home, and falls back to ``heartbeat_hz`` when actions stall.
        # Obs-only mode (``enable_action=False``) has no action lane, so we still fall
        # back to the legacy clock-driven ``run`` in that mode — see ``_run_loop_client``.
        loop_thread = threading.Thread(
            target=self._run_loop_client,
            kwargs=dict(enable_action=enable_action),
            name="robot-obs-poll",
            daemon=True,
        )
        loop_thread.start()
        self._threads: list[threading.Thread] = [loop_thread]

        LOGGER.info(
            "loop robot env enabled: robot-obs %r (%s, heartbeat=%.1f Hz, action-driven) -> %s%s",
            LoopRobotClient.OBS_SOURCE_ID,
            arm_prefixes,
            heartbeat_hz,
            loop_addr,
            f"; robot-action {LoopRobotClient.ACTION_SOURCE_ID!r} -> Step({action_space})"
            if enable_action
            else "",
        )

    def reconfigure(
        self, control_hz: float | None = None, action_space: str = ""
    ) -> None:
        """Apply a Source-Bus-selected config: re-target Step's action space.

        Called from the obs sender's ``apply_config`` when the recorder picks a config.
        The selected ``control_hz`` paces the RCI ENGINE, not this robot: the obs publish
        rate stays fixed (fast) so the engine always samples a fresh pose — re-pacing obs
        down to control_hz would re-introduce the stale-feedback servo shake. Only the
        action space is re-targeted here; it is read by each Step apply next tick.
        """
        del control_hz  # engine's clock, not the robot's obs rate
        if action_space:
            self._action_space = action_space

    def _run_loop_client(self, *, enable_action: bool) -> None:
        """Thread body: drive the SDK loop, surfacing a fatal exit (live robot — a silent
        daemon-thread death would freeze obs/action with no signal).

        With the action lane enabled we take the event-driven path (``LoopRobotClient.run``)
        so obs cadence follows each incoming ``robot-action``. Obs-only mode has no action
        lane to hang the cadence on, so we fall back to the clock-driven ``stream_obs`` at
        ``heartbeat_hz``.
        """
        try:
            if enable_action:
                self._loop_robot_client.run(
                    publish_obs_callback=self._read_obs,
                    apply_action_callback=self._apply_action,
                    handle_command_callback=self._apply_command,
                    heartbeat_hz=self._heartbeat_hz,
                    stop=self._lane_stop,
                )
                return
            self._loop_robot_client.stream_obs(
                publish_obs_callback=self._read_obs,
                hz=self._heartbeat_hz,
                stop=self._lane_stop,
            )
        except Exception:
            LOGGER.exception("loop robot env run() thread exited with an error")

    def _read_obs(self) -> dict[str, Any]:
        """``publish_obs_callback``: read every arm's observation (paired) and merge into one robot-obs.

        Returns the merged payload for ``run()`` to publish. The SDK stamps the
        timestamp itself — the per-arm sample timestamp is not preserved on the wire.
        """
        observations: dict[str, Any] = {}
        for arm_prefix, service in self._arm_services:
            observation, _sample_ts = service._create_observation()
            observations[arm_prefix] = observation
        return merge_observations(observations)

    def _apply_command(self, command: dict[str, Any]) -> None:
        """``handle_command_callback``: home each arm on a HOME command (unknown/failed logged + skipped)."""
        if not self._appliers:
            return
        if command.get("command") != HOME:
            LOGGER.warning("ignoring unknown robot-command %r", command)
            return
        for arm_prefix, applier in self._appliers.items():
            try:
                applier.home()
            except Exception as exc:
                LOGGER.warning("home failed for %s; skipping: %s", arm_prefix, exc)

    def _apply_action(self, payload: dict[str, Any]) -> None:
        """``poll_action_callback``: decode each arm's vector from the raw robot-action and Step it."""
        if not self._appliers:
            return
        for arm_prefix, applier in self._appliers.items():
            action = decode_action(payload, arm_prefix, self._action_space)
            if action is None:
                continue
            try:
                applier.step(action, self._action_space, self._gripper_action_space)
            except (
                Exception
            ) as exc:  # _StepApplier raises on non-SUCCESS Step; skip the tick
                LOGGER.warning(
                    "robot-action Step failed for %s; skipping: %s", arm_prefix, exc
                )

    def close(self) -> None:
        """Stop the lane and close the bus link (BEFORE closing the robot)."""
        self._lane_stop.set()
        for thread in self._threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                LOGGER.warning(
                    "lane thread %r did not stop within timeout", thread.name
                )
        with contextlib.suppress(Exception):
            self._loop_robot_client.disconnect()  # stops sender + action/command subscribe threads


def _install_signal_shutdown(cleanup) -> None:
    def shutdown_handler(signum, frame):
        del signum, frame
        LOGGER.info("Shutting down Vega RobotEnv+Loop server")
        with contextlib.suppress(Exception):
            cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)


def serve_with_loop(
    *,
    loop_addr: str,
    grpc_port: int = 50061,
    arm_prefix: str = DEFAULT_ARM_PREFIX,
    action_space: str = DEFAULT_ACTION_SPACE,
    gripper_action_space: str = "",
    heartbeat_hz: float = DEFAULT_HEARTBEAT_HZ,
    enable_action: bool = True,
    action_space_options: Sequence[str] = (),
    **service_kwargs: Any,
) -> None:
    """Single-arm: a RobotEnv gRPC server that also bridges one robot-obs/robot-action."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    service = _LockedStepService(**service_kwargs)  # builds its own one-arm Robot
    env = LoopRobotEnv(
        [(arm_prefix, service)],
        loop_addr=loop_addr,
        action_space=action_space,
        gripper_action_space=gripper_action_space,
        heartbeat_hz=heartbeat_hz,
        enable_action=enable_action,
        action_space_options=action_space_options,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    _vega_server.robotenv_pb2_grpc.add_RobotEnvServicer_to_server(service, server)
    server.add_insecure_port(f"0.0.0.0:{grpc_port}")
    server.start()
    LOGGER.info(
        "Vega RobotEnv+Loop server on 0.0.0.0:%d (robot-obs=%r)",
        grpc_port,
        LoopRobotClient.OBS_SOURCE_ID,
    )

    def cleanup() -> None:
        # Lanes first (no Step/obs read in flight), then control loop, then robot.
        env.close()
        for teardown in (service._stop_control_loop, service._robot.close):
            with contextlib.suppress(Exception):
                teardown()
        server.stop(grace=5)

    _install_signal_shutdown(cleanup)
    server.wait_for_termination()


_SERIAL_GRIPPERS = ("robotiq", "sr_gripper")


def _dual_arm_comports(
    service_kwargs: dict[str, Any],
    left_robotiq_comport: str | None,
    right_robotiq_comport: str | None,
) -> tuple[str | None, str | None]:
    """Resolve each arm's gripper comport for dual-arm, rejecting the same-port footgun.

    Per-arm overrides win; both fall back to the shared ``robotiq_comport``. A serial
    gripper (robotiq/sr_gripper) is one physical device per port — two arms on the
    SAME port would corrupt comms, so that is rejected. Distinct ports are fine: each
    arm's VegaRobot opens its own gripper independent of the shared arm hardware.
    """
    base = service_kwargs.get("robotiq_comport")
    left = left_robotiq_comport or base
    right = right_robotiq_comport or base
    gripper = service_kwargs.get("gripper_type", "default")
    if gripper == "default":
        gripper = service_kwargs.get("hand_type", "default")
    if gripper in _SERIAL_GRIPPERS and left == right:
        raise ValueError(
            f"dual-arm with a serial gripper ({gripper!r}) needs a DISTINCT comport per arm; "
            f"both arms resolved to {left!r}. Pass --robotiq-comport-left / --robotiq-comport-right."
        )
    return left, right


def serve_dual_arm(
    *,
    loop_addr: str,
    arm_prefixes: tuple[str, str] = ("robot0", "robot1"),
    action_space: str = DEFAULT_ACTION_SPACE,
    gripper_action_space: str = "",
    heartbeat_hz: float = DEFAULT_HEARTBEAT_HZ,
    enable_action: bool = True,
    action_space_options: Sequence[str] = (),
    left_robotiq_comport: str | None = None,
    right_robotiq_comport: str | None = None,
    **service_kwargs: Any,
) -> None:
    """Bimanual: ONE Vega robot, both arms, presented as one robot-obs/robot-action.

    Builds the left service (which constructs the one ``Robot`` with both arms), then
    a right service that SHARES that Robot (injected), so both arms run over one
    hardware connection. Each service keeps its own per-arm IK/filter/gripper.

    A serial gripper (robotiq/sr_gripper) is a separate device per arm — each arm's
    ``VegaRobot`` opens its OWN gripper on its OWN comport, independent of the shared
    arm hardware — so bimanual serial grippers work as long as each arm gets a
    DISTINCT comport (``left_robotiq_comport`` / ``right_robotiq_comport``). The same
    comport on both arms is the real footgun and is rejected.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    left_prefix, right_prefix = arm_prefixes

    left_comport, right_comport = _dual_arm_comports(
        service_kwargs, left_robotiq_comport, right_robotiq_comport
    )
    left = _LockedStepService(arm_side="left", **{**service_kwargs, "robotiq_comport": left_comport})
    shared_robot = left._robot.robot  # the one hardware unit (both arms) left built
    right = _LockedStepService(
        arm_side="right", robot=shared_robot, **{**service_kwargs, "robotiq_comport": right_comport}
    )

    env = LoopRobotEnv(
        [(left_prefix, left), (right_prefix, right)],
        loop_addr=loop_addr,
        action_space=action_space,
        gripper_action_space=gripper_action_space,
        heartbeat_hz=heartbeat_hz,
        enable_action=enable_action,
        action_space_options=action_space_options,
    )
    LOGGER.info(
        "Vega dual-arm robot env running: arms=%s robot-obs=%r",
        list(arm_prefixes),
        LoopRobotClient.OBS_SOURCE_ID,
    )

    def cleanup() -> None:
        # Lanes first (no Step/obs read in flight), then per-arm control loops, then
        # close each VegaRobot — VegaRobot.close() stops that arm's gripper worker AND
        # calls the shared Robot.shutdown() (idempotent, so the second call is a no-op
        # but still stops the right arm's gripper). Using the raw shared_robot.close()
        # would only release the comm node, leaving both arms energized + threads leaked.
        env.close()
        for service in (left, right):
            with contextlib.suppress(Exception):
                service._stop_control_loop()
        for service in (left, right):
            with contextlib.suppress(Exception):
                service._robot.close()

    _install_signal_shutdown(cleanup)
    # No gRPC server here: actions arrive via the bus, obs leave via the bus. The
    # the loop robot env owns the lifetime; block until a shutdown signal.
    threading.Event().wait()
