"""CLI launcher: Vega RobotEnv server bridged to Loop (robot-obs out, robot-action in).

Mirrors the common ``dexcontrol.core.robotenv_vega.server`` arguments plus the
Loop options. Run co-located with the robot (in-process — same env that runs the
plain RobotEnv server, since Vega supports Python 3.10+):

    python -m loop_bridge \
        --loop-addr loop-host:50051 \
        --arm-side left --gripper-type robotiq --robotiq-comport /dev/ttyUSB0

Publishes ``robot-obs`` and (unless ``--no-action``) consumes ``robot-action``,
replaying each action through the RobotEnv ``Step`` path.
"""

from __future__ import annotations

import argparse

from loop_bridge.obs_publisher import (
    DEFAULT_OBS_SOURCE_ID,
    DEFAULT_OBS_SOURCE_NAME,
)
from loop_bridge.robot_obs import DEFAULT_ARM_PREFIX
from loop_bridge.source_server import (
    DEFAULT_ACTION_SOURCE_ID,
    DEFAULT_ACTION_SPACE,
    DEFAULT_OBS_HZ,
    serve_dual_arm,
    serve_with_loop,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vega RobotEnv gRPC server bridged to the Loop Source Bus (robot-obs out, robot-action in)"
    )
    # Loop Source Bus options.
    parser.add_argument("--loop-addr", required=True, help="Loop Source Bus host:port")
    parser.add_argument(
        "--arm-prefix",
        default=DEFAULT_ARM_PREFIX,
        help="Channel arm prefix (e.g. robot0)",
    )
    parser.add_argument(
        "--obs-source-id", default=DEFAULT_OBS_SOURCE_ID, help="robot-obs source id"
    )
    parser.add_argument(
        "--obs-source-name",
        default=DEFAULT_OBS_SOURCE_NAME,
        help="robot-obs source name",
    )
    parser.add_argument(
        "--obs-hz",
        type=float,
        default=DEFAULT_OBS_HZ,
        help="robot-obs publish rate (Hz)",
    )
    parser.add_argument(
        "--action-source-id",
        default=DEFAULT_ACTION_SOURCE_ID,
        help="robot-action source id",
    )
    parser.add_argument(
        "--action-space",
        default=DEFAULT_ACTION_SPACE,
        help="RobotEnv action space the bus action lane carries (e.g. target_cartesian_delta)",
    )
    parser.add_argument(
        "--gripper-action-space",
        default="",
        help="Gripper action space; empty lets the server infer it from --action-space",
    )
    parser.add_argument(
        "--no-action",
        action="store_true",
        help="Publish robot-obs only; do not consume/execute robot-action",
    )
    parser.add_argument(
        "--dual-arm",
        action="store_true",
        help="Bimanual: drive BOTH arms of one Vega as one robot-obs (robot0+robot1)",
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
    parser.add_argument(
        "--control-hz-options",
        default="",
        help="Comma-separated control_hz the source advertises for negotiation (default: --obs-hz)",
    )
    parser.add_argument(
        "--action-space-options",
        default="",
        help="Comma-separated action spaces the source advertises (default: --action-space)",
    )

    args = parser.parse_args()

    loop_kwargs = dict(
        loop_addr=args.loop_addr,
        obs_source_id=args.obs_source_id,
        obs_source_name=args.obs_source_name,
        action_source_id=args.action_source_id,
        action_space=args.action_space,
        gripper_action_space=args.gripper_action_space,
        obs_hz=args.obs_hz,
        enable_action=not args.no_action,
        control_hz_options=tuple(
            int(v) for v in args.control_hz_options.split(",") if v.strip()
        ),
        action_space_options=tuple(
            v.strip() for v in args.action_space_options.split(",") if v.strip()
        ),
    )
    service_kwargs = dict(
        robot_model=args.robot_model,
        gripper_type=args.gripper_type,
        robotiq_comport=args.robotiq_comport,
        control_hz=args.control_hz,
    )

    if args.dual_arm:
        # One Vega, both arms, one robot-obs (robot0+robot1). No gRPC server — the
        # bridge owns the lifetime and actions arrive via the bus.
        serve_dual_arm(**loop_kwargs, **service_kwargs)
    else:
        serve_with_loop(
            grpc_port=args.grpc_port,
            arm_prefix=args.arm_prefix,
            arm_side=args.arm_side,
            **loop_kwargs,
            **service_kwargs,
        )


if __name__ == "__main__":
    main()
