"""In-process Source Bus bridge for the Vega RobotEnv server.

Presents a Vega robot — single- or bimanual — as ONE ``robot-obs`` source and
executes ONE ``robot-action``, per the robot source contract. The combiner of the
arms lives here (the robot's own repo), not in loop or loop-sdk.

- ``_LockedStepService`` is the upstream ``VegaRobotEnvService`` plus one fix: it
  serializes ``Step`` on the upstream ``_cmd_lock`` (upstream guards only ``Reset``),
  so the bus action lane can't race a Reset on shared IK/filter state.
- ``LoopBridge`` owns the bus I/O over N arm services that share ONE hardware unit:
  an **obs poll** reads each arm's ``_create_observation`` on a clock and publishes
  the merged ``robot-obs`` (Vega computes obs only inside ``_create_observation``, so
  the bridge must drive it — else teleop, which needs obs for a delta, and obs,
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
from typing import Any, Optional, Sequence

import grpc
from loop_sdk import (
    HOME,
    RobotActionReceiver,
    RobotCommandReceiver,
    RobotConfig,
    RobotConfigOptions,
)

# Importing the upstream module runs its sys.path setup and binds the proto stubs.
from dexcontrol.core.robotenv_vega import server as _vega_server
from loop_bridge import lanes
from loop_bridge.obs_publisher import (
    DEFAULT_OBS_SOURCE_ID,
    DEFAULT_OBS_SOURCE_NAME,
    RobotObsPublisher,
)
from loop_bridge.robot_obs import DEFAULT_ARM_PREFIX

LOGGER = logging.getLogger("loop_bridge.vega")

DEFAULT_ACTION_SOURCE_ID = "robot-action"
DEFAULT_COMMAND_SOURCE_ID = "robot-command"
DEFAULT_ACTION_SPACE = "target_cartesian_delta"
DEFAULT_OBS_HZ = 20.0


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


class LoopBridge:
    """Bus I/O (one robot-obs + one robot-action) over N arm services sharing a robot."""

    def __init__(
        self,
        arm_services: Sequence[tuple[str, Any]],
        *,
        loop_addr: str,
        obs_source_id: str = DEFAULT_OBS_SOURCE_ID,
        obs_source_name: str = DEFAULT_OBS_SOURCE_NAME,
        action_source_id: str = DEFAULT_ACTION_SOURCE_ID,
        command_source_id: str = DEFAULT_COMMAND_SOURCE_ID,
        action_space: str = DEFAULT_ACTION_SPACE,
        gripper_action_space: str = "",
        obs_hz: float = DEFAULT_OBS_HZ,
        enable_action: bool = True,
        control_hz_options: Sequence[int] = (),
        action_space_options: Sequence[str] = (),
    ) -> None:
        if obs_hz <= 0:
            raise ValueError(f"obs_hz must be > 0, got {obs_hz}")
        if not arm_services:
            raise ValueError("at least one (arm_prefix, service) is required")
        self._arm_services = tuple(arm_services)
        arm_prefixes = [arm_prefix for arm_prefix, _ in self._arm_services]
        if len(set(arm_prefixes)) != len(arm_prefixes):
            raise ValueError(f"duplicate arm prefixes: {arm_prefixes}")

        self._action_space = action_space
        self._gripper_action_space = gripper_action_space
        self._period_s = 1.0 / obs_hz  # mutable: reconfigure() re-paces the obs poll
        self._lane_stop = threading.Event()

        # Advertise the configs this robot can open with (control_hz / action_space);
        # default to the configured values. apply_config re-paces the live bridge.
        options = RobotConfigOptions(
            control_hz=tuple(control_hz_options) or (int(obs_hz),),
            action_space=tuple(action_space_options) or (action_space,),
        )

        def apply_config(config: RobotConfig) -> RobotConfig:
            self.reconfigure(
                control_hz=config.control_hz, action_space=config.action_space
            )
            return config

        self._obs_publisher = RobotObsPublisher.connect(
            loop_addr=loop_addr,
            source_id=obs_source_id,
            name=obs_source_name,
            arm_prefixes=arm_prefixes,
            options=options,
            apply_config=apply_config,
        )

        # Action consume is a loop-sdk RobotActionReceiver (its own subscribe +
        # reconnect thread); the obs poll pulls its latest() each tick and Steps each
        # arm. One per-arm Step applier; latest() is {} when nothing fresh -> hold.
        self._action_receiver: Any = None
        self._command_receiver: Any = None
        self._appliers: dict[str, _StepApplier] = {}
        if enable_action:
            self._appliers = {
                arm_prefix: _StepApplier(service)
                for arm_prefix, service in self._arm_services
            }
            self._action_receiver = RobotActionReceiver(
                loop_addr,
                action_source_id,
                arms=tuple(arm_prefixes),
                action_space=action_space,
            )
            self._action_receiver.connect()
            # Operational commands (home between episodes, ...) share the per-arm
            # appliers to drive the robot, so the lane runs only when actions do.
            self._command_receiver = RobotCommandReceiver(loop_addr, command_source_id)
            self._command_receiver.connect()

        obs_thread = threading.Thread(
            target=lanes.run_obs_poll,
            kwargs=dict(
                stop_event=self._lane_stop,
                read_observation=self._read_observations,
                publish=self._obs_publisher.publish,
                period_s=lambda: self._period_s,
                apply_actions=self._apply_bus if enable_action else None,
            ),
            name="robot-obs-poll",
            daemon=True,
        )
        obs_thread.start()
        self._threads: list[threading.Thread] = [obs_thread]

        LOGGER.info(
            "loop bridge enabled: robot-obs %r (%s, %.1f Hz) -> %s%s",
            obs_source_id,
            arm_prefixes,
            obs_hz,
            loop_addr,
            f"; robot-action {action_source_id!r} -> Step({action_space})"
            if enable_action
            else "",
        )

    def reconfigure(
        self, control_hz: float | None = None, action_space: str = ""
    ) -> None:
        """Apply a Source-Bus-selected config: re-pace the obs poll / re-target Step.

        Called from the obs sender's ``apply_config`` when the recorder picks a config.
        ``_period_s`` is read by the obs poll each tick (via a callable) and
        ``_action_space`` by each Step apply, so the change takes effect next tick.
        """
        if control_hz and control_hz > 0:
            self._period_s = 1.0 / control_hz
        if action_space:
            self._action_space = action_space

    def _read_observations(self) -> tuple[dict[str, Any], int]:
        """Read every arm's observation in one tick (paired) → {prefix: obs}, timestamp."""
        observations: dict[str, Any] = {}
        timestamp_us: Optional[int] = None
        for arm_prefix, service in self._arm_services:
            observation, sample_ts = service._create_observation()
            observations[arm_prefix] = observation
            if timestamp_us is None:
                timestamp_us = sample_ts
        return observations, int(timestamp_us or 0)

    def _apply_bus(self) -> None:
        """One tick of bus inputs: run operational commands, then Step actions.

        Commands first so a home is honored before the (held) action of that tick.
        """
        self._apply_commands()
        self._apply_actions()

    def _apply_commands(self) -> None:
        """Run any operational commands pulled this tick (home each arm on HOME).

        Each drained command runs once; unknown commands and a failed home are
        logged and skipped, never fatal to the lane.
        """
        if self._command_receiver is None:
            return
        for command in self._command_receiver.drain():
            if command != HOME:
                LOGGER.warning("ignoring unknown robot-command %r", command)
                continue
            for arm_prefix, applier in self._appliers.items():
                try:
                    applier.home()
                except Exception as exc:
                    LOGGER.warning("home failed for %s; skipping: %s", arm_prefix, exc)

    def _apply_actions(self) -> None:
        """Step each arm with the freshest pulled action ({} when nothing fresh → hold)."""
        if self._action_receiver is None:
            return
        for arm_prefix, action in self._action_receiver.latest().items():
            applier = self._appliers.get(arm_prefix)
            if applier is None:
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
        """Stop the lane + action receiver and close the publisher (BEFORE closing the robot)."""
        self._lane_stop.set()
        if self._action_receiver is not None:
            with contextlib.suppress(Exception):
                self._action_receiver.disconnect()  # stops its subscribe thread
        if self._command_receiver is not None:
            with contextlib.suppress(Exception):
                self._command_receiver.disconnect()
        for thread in self._threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                LOGGER.warning(
                    "lane thread %r did not stop within timeout", thread.name
                )
        self._obs_publisher.close()


def _install_signal_shutdown(cleanup) -> None:
    def shutdown_handler(signum, frame):
        del signum, frame
        LOGGER.info("Shutting down Vega RobotEnv+Loop bridge")
        with contextlib.suppress(Exception):
            cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)


def serve_with_loop(
    *,
    loop_addr: str,
    grpc_port: int = 50061,
    obs_source_id: str = DEFAULT_OBS_SOURCE_ID,
    obs_source_name: str = DEFAULT_OBS_SOURCE_NAME,
    arm_prefix: str = DEFAULT_ARM_PREFIX,
    action_source_id: str = DEFAULT_ACTION_SOURCE_ID,
    command_source_id: str = DEFAULT_COMMAND_SOURCE_ID,
    action_space: str = DEFAULT_ACTION_SPACE,
    gripper_action_space: str = "",
    obs_hz: float = DEFAULT_OBS_HZ,
    enable_action: bool = True,
    control_hz_options: Sequence[int] = (),
    action_space_options: Sequence[str] = (),
    **service_kwargs: Any,
) -> None:
    """Single-arm: a RobotEnv gRPC server that also bridges one robot-obs/robot-action."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    service = _LockedStepService(**service_kwargs)  # builds its own one-arm Robot
    bridge = LoopBridge(
        [(arm_prefix, service)],
        loop_addr=loop_addr,
        obs_source_id=obs_source_id,
        obs_source_name=obs_source_name,
        action_source_id=action_source_id,
        command_source_id=command_source_id,
        action_space=action_space,
        gripper_action_space=gripper_action_space,
        obs_hz=obs_hz,
        enable_action=enable_action,
        control_hz_options=control_hz_options,
        action_space_options=action_space_options,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    _vega_server.robotenv_pb2_grpc.add_RobotEnvServicer_to_server(service, server)
    server.add_insecure_port(f"0.0.0.0:{grpc_port}")
    server.start()
    LOGGER.info(
        "Vega RobotEnv+Loop server on 0.0.0.0:%d (robot-obs=%r)",
        grpc_port,
        obs_source_id,
    )

    def cleanup() -> None:
        # Lanes first (no Step/obs read in flight), then control loop, then robot.
        bridge.close()
        for teardown in (service._stop_control_loop, service._robot.close):
            with contextlib.suppress(Exception):
                teardown()
        server.stop(grace=5)

    _install_signal_shutdown(cleanup)
    server.wait_for_termination()


def serve_dual_arm(
    *,
    loop_addr: str,
    arm_prefixes: tuple[str, str] = ("robot0", "robot1"),
    obs_source_id: str = DEFAULT_OBS_SOURCE_ID,
    obs_source_name: str = DEFAULT_OBS_SOURCE_NAME,
    action_source_id: str = DEFAULT_ACTION_SOURCE_ID,
    command_source_id: str = DEFAULT_COMMAND_SOURCE_ID,
    action_space: str = DEFAULT_ACTION_SPACE,
    gripper_action_space: str = "",
    obs_hz: float = DEFAULT_OBS_HZ,
    enable_action: bool = True,
    control_hz_options: Sequence[int] = (),
    action_space_options: Sequence[str] = (),
    **service_kwargs: Any,
) -> None:
    """Bimanual: ONE Vega robot, both arms, presented as one robot-obs/robot-action.

    Builds the left service (which constructs the one ``Robot`` with both arms), then
    a right service that SHARES that Robot (injected), so both arms run over one
    hardware connection. Each service keeps its own per-arm IK/filter/gripper.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    left_prefix, right_prefix = arm_prefixes

    # A serial gripper (robotiq/sr_gripper) is one COM port; both arms sharing it
    # would corrupt gripper comms. Bimanual needs the built-in per-arm grippers
    # (or per-arm comports, not yet wired) — fail fast rather than open it twice.
    # VegaRobot resolves ``hand_type`` to ``gripper_type`` when the latter is
    # "default", so a programmatic caller can request a serial gripper via either
    # key — guard both so neither path slips past.
    serial_grippers = ("robotiq", "sr_gripper")
    if (
        service_kwargs.get("gripper_type", "default") in serial_grippers
        or service_kwargs.get("hand_type") in serial_grippers
    ):
        requested = service_kwargs.get("gripper_type", "default")
        if requested == "default":
            requested = service_kwargs.get("hand_type")
        raise ValueError(
            f"dual-arm needs per-arm grippers; a shared serial gripper "
            f"({requested!r}) can't be opened by both arms"
        )

    left = _LockedStepService(arm_side="left", **service_kwargs)
    shared_robot = left._robot.robot  # the one hardware unit (both arms) left built
    right = _LockedStepService(arm_side="right", robot=shared_robot, **service_kwargs)

    bridge = LoopBridge(
        [(left_prefix, left), (right_prefix, right)],
        loop_addr=loop_addr,
        obs_source_id=obs_source_id,
        obs_source_name=obs_source_name,
        action_source_id=action_source_id,
        command_source_id=command_source_id,
        action_space=action_space,
        gripper_action_space=gripper_action_space,
        obs_hz=obs_hz,
        enable_action=enable_action,
        control_hz_options=control_hz_options,
        action_space_options=action_space_options,
    )
    LOGGER.info(
        "Vega dual-arm bridge running: arms=%s robot-obs=%r",
        list(arm_prefixes),
        obs_source_id,
    )

    def cleanup() -> None:
        # Lanes first (no Step/obs read in flight), then per-arm control loops, then
        # close each VegaRobot — VegaRobot.close() stops that arm's gripper worker AND
        # calls the shared Robot.shutdown() (idempotent, so the second call is a no-op
        # but still stops the right arm's gripper). Using the raw shared_robot.close()
        # would only release the comm node, leaving both arms energized + threads leaked.
        bridge.close()
        for service in (left, right):
            with contextlib.suppress(Exception):
                service._stop_control_loop()
        for service in (left, right):
            with contextlib.suppress(Exception):
                service._robot.close()

    _install_signal_shutdown(cleanup)
    # No gRPC server here: actions arrive via the bus, obs leave via the bus. The
    # bridge owns the lifetime; block until a shutdown signal.
    threading.Event().wait()
