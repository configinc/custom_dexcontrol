"""CLI launcher: Vega RobotEnv server with robot-obs publishing to Loop.

Mirrors the common ``dexcontrol.core.robotenv_vega.server`` arguments plus the
Loop options. Run co-located with the robot (in-process — same env that runs the
plain RobotEnv server, since Vega supports Python 3.10+):

    python -m loop_bridge \
        --loop-addr loop-host:50051 \
        --arm-side left --gripper-type robotiq --robotiq-comport /dev/ttyUSB0
"""

from __future__ import annotations

import argparse

from loop_bridge.obs_publisher import (
    DEFAULT_OBS_SOURCE_ID,
    DEFAULT_OBS_SOURCE_NAME,
)
from loop_bridge.source_server import serve_with_loop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vega RobotEnv gRPC server + robot-obs publishing to Loop Source Bus"
    )
    # Loop Source Bus options.
    parser.add_argument("--loop-addr", required=True, help="Loop Source Bus host:port")
    parser.add_argument(
        "--obs-source-id", default=DEFAULT_OBS_SOURCE_ID, help="robot-obs source id"
    )
    parser.add_argument(
        "--obs-source-name",
        default=DEFAULT_OBS_SOURCE_NAME,
        help="robot-obs source name",
    )
    # Common RobotEnv server options (forwarded to VegaRobotEnvService).
    parser.add_argument(
        "--grpc-port", type=int, default=50061, help="RobotEnv gRPC service port"
    )
    parser.add_argument("--robot-model", default="vega_1", help="Robot model")
    parser.add_argument(
        "--arm-side",
        default="left",
        choices=["left", "right"],
        help="Which arm this server controls",
    )
    parser.add_argument(
        "--gripper-type", default="default", help="Gripper type (e.g. robotiq)"
    )
    parser.add_argument(
        "--robotiq-comport", default="/dev/ttyUSB0", help="Robotiq serial port"
    )
    parser.add_argument(
        "--control-hz", type=int, default=20, help="Control frequency in Hz"
    )

    args = parser.parse_args()

    serve_with_loop(
        loop_addr=args.loop_addr,
        grpc_port=args.grpc_port,
        obs_source_id=args.obs_source_id,
        obs_source_name=args.obs_source_name,
        robot_model=args.robot_model,
        arm_side=args.arm_side,
        gripper_type=args.gripper_type,
        robotiq_comport=args.robotiq_comport,
        control_hz=args.control_hz,
    )


if __name__ == "__main__":
    main()
