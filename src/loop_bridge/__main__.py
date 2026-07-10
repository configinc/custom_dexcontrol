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

from loop_bridge.robot_obs import DEFAULT_ARM_PREFIX
from loop_bridge.source_server import (
    DEFAULT_ACTION_SPACE,
    DEFAULT_HEARTBEAT_HZ,
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
    # Source ids/name are pinned by the SDK (our lane convention) — not CLI-configurable.
    parser.add_argument(
        "--heartbeat-hz",
        type=float,
        default=DEFAULT_HEARTBEAT_HZ,
        help=(
            "Fallback robot-obs publish rate (Hz) when the action lane is idle. "
            "In steady state obs cadence follows each robot-action's post-step; this "
            "rate only paces the boot lull and teleop-hold gaps."
        ),
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
        "--robotiq-comport", default="/dev/ttyUSB0", help="Robotiq serial port (single-arm)"
    )
    parser.add_argument(
        "--robotiq-comport-left", default=None,
        help="Dual-arm: left arm's Robotiq serial port (distinct from right)",
    )
    parser.add_argument(
        "--robotiq-comport-right", default=None,
        help="Dual-arm: right arm's Robotiq serial port (distinct from left)",
    )
    parser.add_argument(
        "--control-hz", type=int, default=20, help="Control frequency in Hz"
    )
    # --- VegaRobotEnvService tuning (forwarded verbatim, mirrors server.py) ---
    # These reach VegaRobotEnvService unchanged; defaults match server.py so an
    # omitted flag behaves exactly as the standalone server would.
    parser.add_argument("--frame-type", default="vega_mobile_base",
                        choices=["vega_mobile_base", "vega_table_mount", "vega_custom"],
                        help="Robot mounting frame type")
    parser.add_argument("--use-velocity-feedforward", action="store_true",
                        help="Send pos+vel feedforward instead of position-only arm commands")
    parser.add_argument("--base-frame-rotation", type=float, nargs=3, default=None,
                        metavar=("ROLL", "PITCH", "YAW"), help="Custom base-frame rotation (deg)")
    parser.add_argument("--ik-solver", dest="ik_solver_type", default="pink",
                        choices=["pink", "placo"], help="IK solver backend")
    parser.add_argument("--gripper-iface", default=None,
                        help="EtherCAT iface for SR gripper; overrides --robotiq-comport when set")
    parser.add_argument("--ema-alpha", type=float, default=0.0,
                        help="Joint-command smoothing responsiveness (0=disabled)")
    parser.add_argument("--ik-damping-default", type=float, default=1e-3)
    parser.add_argument("--ik-damping-torso", type=float, default=30000.0)
    parser.add_argument("--ik-damping-arm-j2", type=float, default=100.0)
    parser.add_argument("--ik-damping-arm-j3", type=float, default=50.0)
    parser.add_argument("--interpolation-method", default="none",
                        choices=["none", "linear", "cubic"], help="Input→control-rate upsampling")
    parser.add_argument("--interpolation-history", type=int, default=4)
    parser.add_argument("--control-loop-hz", type=int, default=0,
                        help="Control loop frequency for interpolation upsampling (0=off)")
    parser.add_argument("--filter-type", default="none",
                        choices=["none", "butterworth", "ema"], help="Output filter")
    parser.add_argument("--filter-cutoff-freq", type=float, default=10.0)
    parser.add_argument("--filter-order", type=int, default=2)
    parser.add_argument("--filter-ema-alpha", type=float, default=0.1)
    parser.add_argument("--vel-smoothing-alpha", type=float, default=0.3)
    parser.add_argument("--hw-correction-alpha", type=float, default=0.7)
    parser.add_argument("--max-delta-scale", type=float, default=1.0)
    parser.add_argument("--max-jerk", type=float, default=0.25)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--vel-ratio", type=float, default=1.0)
    parser.add_argument("--vel-damp-thresh", type=float, default=0.05)
    parser.add_argument(
        "--action-space-options",
        default="",
        help="Comma-separated action spaces the source advertises (default: --action-space)",
    )
    parser.add_argument(
        "--gripper-type-options",
        default="",
        help="Comma-separated gripper types the source advertises (empty = no menu)",
    )
    parser.add_argument(
        "--finger-type-options",
        default="",
        help="Comma-separated finger types the source advertises (empty = no menu)",
    )
    parser.add_argument(
        "--robot-type-options",
        default="",
        help="Comma-separated robot types the source advertises (empty = no menu)",
    )
    parser.add_argument(
        "--robot-firmware-version-options",
        default="",
        help="Comma-separated robot firmware versions the source advertises (empty = no menu)",
    )

    args = parser.parse_args()

    def _csv(value: str) -> tuple[str, ...]:
        return tuple(v.strip() for v in value.split(",") if v.strip())

    loop_kwargs = dict(
        loop_addr=args.loop_addr,
        action_space=args.action_space,
        gripper_action_space=args.gripper_action_space,
        heartbeat_hz=args.heartbeat_hz,
        enable_action=not args.no_action,
        action_space_options=_csv(args.action_space_options),
        gripper_type_options=_csv(args.gripper_type_options),
        finger_type_options=_csv(args.finger_type_options),
        robot_type_options=_csv(args.robot_type_options),
        robot_firmware_version_options=_csv(args.robot_firmware_version_options),
    )
    # --gripper-iface (SR EtherCAT) takes precedence over --robotiq-comport, both
    # feed the one "where is the gripper" slot (mirrors server.py).
    gripper_addr = args.gripper_iface or args.robotiq_comport
    service_kwargs = dict(
        robot_model=args.robot_model,
        gripper_type=args.gripper_type,
        frame_type=args.frame_type,
        control_hz=args.control_hz,
        use_velocity_feedforward=args.use_velocity_feedforward,
        base_frame_rotation=args.base_frame_rotation,
        ik_solver_type=args.ik_solver_type,
        robotiq_comport=gripper_addr,
        ema_alpha=args.ema_alpha,
        ik_damping_default=args.ik_damping_default,
        ik_damping_torso=args.ik_damping_torso,
        ik_damping_arm_j2=args.ik_damping_arm_j2,
        ik_damping_arm_j3=args.ik_damping_arm_j3,
        interpolation_method=args.interpolation_method,
        interpolation_history=args.interpolation_history,
        control_loop_hz=args.control_loop_hz,
        filter_type=args.filter_type,
        filter_cutoff_freq=args.filter_cutoff_freq,
        filter_order=args.filter_order,
        filter_ema_alpha=args.filter_ema_alpha,
        vel_smoothing_alpha=args.vel_smoothing_alpha,
        hw_correction_alpha=args.hw_correction_alpha,
        max_delta_scale=args.max_delta_scale,
        max_jerk=args.max_jerk,
        rot_sensitivity=args.rot_sensitivity,
        vel_ratio=args.vel_ratio,
        vel_damp_thresh=args.vel_damp_thresh,
    )

    if args.dual_arm:
        # One Vega, both arms, one robot-obs (robot0+robot1). No gRPC server — the
        # bridge owns the lifetime and actions arrive via the bus. Each arm opens
        # its own gripper on its own comport (distinct ports required for serial).
        serve_dual_arm(
            left_robotiq_comport=args.robotiq_comport_left,
            right_robotiq_comport=args.robotiq_comport_right,
            **loop_kwargs,
            **service_kwargs,
        )
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
