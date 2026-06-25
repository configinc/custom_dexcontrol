"""In-process Source Bus bridge for the Vega RobotEnv server.

``LoopVegaRobotEnvService`` subclasses the upstream ``VegaRobotEnvService`` and
adds the bus I/O edge, in the same process that owns the hardware:

  - an **obs poll** thread reads the current observation on a clock and publishes
    it as ``robot-obs`` (Vega only computes obs when ``_create_observation`` runs,
    so the bridge must drive it — otherwise teleop, which needs obs to compute a
    delta, and obs, driven by the resulting action's Step, deadlock at startup);
  - an **action lane** thread subscribes ``robot-action`` and replays each frame
    through the service's own ``Step`` (so the teleop-gain / frame-transform /
    interpolation logic is reused verbatim, not duplicated).

The robot's control/hardware logic and the RobotEnv gRPC contract are untouched;
living in our own module keeps the upstream-sync merge surface minimal.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import sys
import threading
from concurrent import futures
from typing import Any, Optional

import grpc

# Importing the upstream module runs its sys.path setup and binds the proto stubs.
from dexcontrol.core.robotenv_vega import server as _vega_server
from loop_bridge import lanes
from loop_bridge.action_consumer import RobotActionConsumer
from loop_bridge.obs_publisher import (
    DEFAULT_OBS_SOURCE_ID,
    DEFAULT_OBS_SOURCE_NAME,
    RobotObsPublisher,
)
from loop_bridge.robot_action import DEFAULT_ACTION_SPACE
from loop_bridge.robot_obs import DEFAULT_ARM_PREFIX

LOGGER = logging.getLogger("loop_bridge.vega")

DEFAULT_ACTION_SOURCE_ID = "robot-action"
DEFAULT_OBS_HZ = 20.0


class _BusStepContext:
    """Minimal gRPC servicer context for replaying Step in-process.

    The Vega ``Step`` happy path never touches the context; we still record an
    ``abort`` (which it would only call on a hard error) and surface it as an
    exception so the action lane skips that tick rather than silently succeeding.
    """

    def set_code(self, code: Any) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details

    def abort(self, code: Any, details: str) -> None:
        raise RuntimeError(f"Step aborted: {code} {details}")


class _StepApplier:
    """Adapts the service's ``Step`` to the action consumer's ``step(...)`` seam."""

    def __init__(self, service: "LoopVegaRobotEnvService") -> None:
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
        # Step swallows hardware faults into a non-SUCCESS StepResponse.status
        # rather than raising, so the action lane would otherwise treat a stalled
        # command (joint limit, IK failure, comms) as success. Surface it.
        status = getattr(response, "status", "") or ""
        if status and status != "SUCCESS":
            LOGGER.warning(
                "robot-action Step returned %s: %s",
                status,
                getattr(response, "message", ""),
            )


class LoopVegaRobotEnvService(_vega_server.VegaRobotEnvService):
    """Vega RobotEnv server that also streams ``robot-obs`` and executes ``robot-action``."""

    def __init__(
        self,
        *,
        loop_addr: str,
        obs_source_id: str = DEFAULT_OBS_SOURCE_ID,
        obs_source_name: str = DEFAULT_OBS_SOURCE_NAME,
        arm_prefix: str = DEFAULT_ARM_PREFIX,
        action_source_id: str = DEFAULT_ACTION_SOURCE_ID,
        action_space: str = DEFAULT_ACTION_SPACE,
        gripper_action_space: str = "",
        obs_hz: float = DEFAULT_OBS_HZ,
        enable_action: bool = True,
        **service_kwargs: Any,
    ) -> None:
        if obs_hz <= 0:
            raise ValueError(f"obs_hz must be > 0, got {obs_hz}")

        # Connect the publisher BEFORE launching the robot so the obs lane (started
        # after the hardware is up) always has a live publisher. Tear it down if
        # the hardware launch fails.
        self._obs_publisher = RobotObsPublisher.connect(
            loop_addr=loop_addr,
            source_id=obs_source_id,
            name=obs_source_name,
            arm_prefix=arm_prefix,
        )
        try:
            super().__init__(**service_kwargs)
        except BaseException:
            self._obs_publisher.close()
            raise

        self._lane_stop = threading.Event()
        self._action_consumer: Optional[RobotActionConsumer] = None
        self._consumer_lock = threading.Lock()
        self._threads: list[threading.Thread] = []

        obs_thread = threading.Thread(
            target=lanes.run_obs_poll,
            kwargs=dict(
                stop_event=self._lane_stop,
                read_observation=self._create_observation,
                publish=self._obs_publisher.publish,
                period_s=1.0 / obs_hz,
            ),
            name="robot-obs-poll",
            daemon=True,
        )
        obs_thread.start()
        self._threads.append(obs_thread)

        if enable_action:
            action_thread = threading.Thread(
                target=lanes.run_action_lane,
                kwargs=dict(
                    stop_event=self._lane_stop,
                    loop_addr=loop_addr,
                    applier=_StepApplier(self),
                    action_source_id=action_source_id,
                    arm_prefix=arm_prefix,
                    action_space=action_space,
                    gripper_action_space=gripper_action_space,
                    register_consumer=self._register_consumer,
                ),
                name="robot-action-lane",
                daemon=True,
            )
            action_thread.start()
            self._threads.append(action_thread)

        LOGGER.info(
            "loop bridge enabled: robot-obs %r (%s, %.1f Hz) -> %s%s",
            obs_source_id,
            arm_prefix,
            obs_hz,
            loop_addr,
            f"; robot-action {action_source_id!r} -> Step({action_space})"
            if enable_action
            else "",
        )

    def Step(self, request, context):
        """Serialize bus-action / gRPC Steps against Reset (and each other).

        Upstream takes ``_cmd_lock`` only in ``Reset`` — safe when Step had a
        single caller. The bus action lane is a second concurrent Step source, so
        without this an action-lane Step can race a Reset (or another Step) on the
        shared IK / smoothing-filter / motion-manager state and command a real arm
        a corrupted pose. ``Step`` itself never takes the lock, so this cannot
        self-deadlock.
        """
        with self._cmd_lock:
            return super().Step(request, context)

    def _register_consumer(self, consumer: Optional[RobotActionConsumer]) -> None:
        with self._consumer_lock:
            self._action_consumer = consumer

    def close_loop_bridge(self) -> None:
        """Stop both lanes and close the publisher. Safe to call once.

        Stops + joins the lane threads (so no Step/obs read is in flight) before
        the caller closes the robot — call this BEFORE ``_robot.close()``.
        """
        self._lane_stop.set()
        # Cancel any in-flight subscribe so the action thread unwinds promptly.
        with self._consumer_lock:
            consumer = self._action_consumer
        if consumer is not None:
            with contextlib.suppress(Exception):
                consumer.close()
        for thread in self._threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                LOGGER.warning(
                    "lane thread %r did not stop within timeout", thread.name
                )
        self._obs_publisher.close()


def serve_with_loop(
    *,
    loop_addr: str,
    grpc_port: int = 50061,
    obs_source_id: str = DEFAULT_OBS_SOURCE_ID,
    obs_source_name: str = DEFAULT_OBS_SOURCE_NAME,
    arm_prefix: str = DEFAULT_ARM_PREFIX,
    action_source_id: str = DEFAULT_ACTION_SOURCE_ID,
    action_space: str = DEFAULT_ACTION_SPACE,
    gripper_action_space: str = "",
    obs_hz: float = DEFAULT_OBS_HZ,
    enable_action: bool = True,
    **service_kwargs: Any,
) -> None:
    """Start a Vega RobotEnv gRPC server that also bridges robot-obs / robot-action.

    Mirrors the upstream ``server.serve()`` gRPC boilerplate (it hardcodes the
    plain service class, so we cannot reuse it directly) and instantiates the
    bus-bridging subclass instead. Extra keyword args are forwarded verbatim to
    ``VegaRobotEnvService`` (arm_side, gripper_type, control_hz, ...).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    service = LoopVegaRobotEnvService(
        loop_addr=loop_addr,
        obs_source_id=obs_source_id,
        obs_source_name=obs_source_name,
        arm_prefix=arm_prefix,
        action_source_id=action_source_id,
        action_space=action_space,
        gripper_action_space=gripper_action_space,
        obs_hz=obs_hz,
        enable_action=enable_action,
        **service_kwargs,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    _vega_server.robotenv_pb2_grpc.add_RobotEnvServicer_to_server(service, server)

    server_address = f"0.0.0.0:{grpc_port}"
    server.add_insecure_port(server_address)
    server.start()
    LOGGER.info(
        "Vega RobotEnv+Loop server started on %s (robot-obs=%r, robot-action=%r)",
        server_address,
        obs_source_id,
        action_source_id if enable_action else None,
    )

    def shutdown_handler(signum, frame):
        del signum, frame
        LOGGER.info("Shutting down Vega RobotEnv+Loop server")
        # Stop the bus lanes FIRST (no Step / obs read can then be in flight),
        # then the control loop, then close the robot — otherwise an action-lane
        # Step could touch the arm after _robot.close() (use-after-close).
        for teardown in (
            service.close_loop_bridge,
            service._stop_control_loop,
            service._robot.close,
        ):
            try:
                teardown()
            except Exception:
                LOGGER.exception("teardown step failed")
        server.stop(grace=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    server.wait_for_termination()
