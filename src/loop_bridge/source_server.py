"""In-process Source Bus bridge for the Vega RobotEnv server.

``LoopVegaRobotEnvService`` subclasses the upstream ``VegaRobotEnvService`` and
adds one thing: every observation the server computes is also published to
Loop's Source Bus as ``robot-obs``. The robot's control/hardware logic and the
RobotEnv gRPC contract (Step/Reset/...) are untouched — this only adds the bus
I/O edge, in the same process that owns the hardware.

Living in our own module (rather than editing the upstream-synced ``server.py``)
keeps the upstream-sync merge surface minimal.
"""

from __future__ import annotations

import logging
import signal
import sys
from concurrent import futures
from typing import Any

import grpc

# Importing the upstream module runs its sys.path setup and binds robotenv_pb2_grpc.
from dexcontrol.core.robotenv_vega import server as _vega_server
from loop_bridge.obs_publisher import (
    DEFAULT_OBS_SOURCE_ID,
    DEFAULT_OBS_SOURCE_NAME,
    RobotObsPublisher,
)

LOGGER = logging.getLogger("loop_bridge.vega")


class LoopVegaRobotEnvService(_vega_server.VegaRobotEnvService):
    """Vega RobotEnv server that also publishes ``robot-obs`` to the Source Bus."""

    def __init__(
        self,
        *,
        loop_addr: str,
        obs_source_id: str = DEFAULT_OBS_SOURCE_ID,
        obs_source_name: str = DEFAULT_OBS_SOURCE_NAME,
        **service_kwargs: Any,
    ) -> None:
        # Connect the publisher BEFORE launching the robot so any control-loop
        # thread spawned by super().__init__ can never observe a missing
        # publisher. Tear it down if the hardware launch fails.
        self._obs_publisher = RobotObsPublisher.connect(
            loop_addr=loop_addr, source_id=obs_source_id, name=obs_source_name
        )
        try:
            super().__init__(**service_kwargs)
        except BaseException:
            self._obs_publisher.close()
            raise
        LOGGER.info(
            "robot-obs publishing enabled: source=%r -> %s", obs_source_id, loop_addr
        )

    def _create_observation(self):
        observation, timestamp_us = super()._create_observation()
        # The bus edge must never affect the RobotEnv contract: a publish/
        # validation fault is logged and dropped, never propagated into the
        # Step/Reset RPC path (which would turn a successful command into an
        # error response).
        try:
            self._obs_publisher.publish(observation, timestamp_us)
        except Exception:
            LOGGER.exception("robot-obs publish failed; observation not published")
        return observation, timestamp_us

    def close_loop_bridge(self) -> None:
        self._obs_publisher.close()


def serve_with_loop(
    *,
    loop_addr: str,
    grpc_port: int = 50061,
    obs_source_id: str = DEFAULT_OBS_SOURCE_ID,
    obs_source_name: str = DEFAULT_OBS_SOURCE_NAME,
    **service_kwargs: Any,
) -> None:
    """Start a Vega RobotEnv gRPC server that also publishes robot-obs.

    Mirrors the upstream ``server.serve()`` gRPC boilerplate (it hardcodes the
    plain service class, so we cannot reuse it directly) and instantiates the
    bus-publishing subclass instead. Extra keyword args are forwarded verbatim to
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
        **service_kwargs,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    _vega_server.robotenv_pb2_grpc.add_RobotEnvServicer_to_server(service, server)

    server_address = f"0.0.0.0:{grpc_port}"
    server.add_insecure_port(server_address)
    server.start()
    LOGGER.info(
        "Vega RobotEnv+Loop server started on %s (robot-obs=%r)",
        server_address,
        obs_source_id,
    )

    def shutdown_handler(signum, frame):
        del signum, frame
        LOGGER.info("Shutting down Vega RobotEnv+Loop server")
        for teardown in (
            service._stop_control_loop,
            service._robot.close,
            service.close_loop_bridge,
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
