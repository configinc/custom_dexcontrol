"""CLI for the external dual-arm Vega Robot Node."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from loop_sdk import NodeConnectionConfig

from loop_bridge.source_server import serve_dual_arm


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Expose both arms of one Vega as a Loop Node Graph Robot Node"
    )
    parser.add_argument("--node-id", required=True)
    parser.add_argument(
        "--loop-endpoint",
        default=os.environ.get("LOOP_NODE_GRAPH_NODE_ENDPOINT") or "tcp/127.0.0.1:7448",
        help="Loop address (default: LOOP_NODE_GRAPH_NODE_ENDPOINT or tcp/127.0.0.1:7448)",
    )
    parser.add_argument("--status-period-ms", type=_positive_int, default=250)
    parser.add_argument("--control-request-capacity", type=_positive_int, default=16)
    parser.add_argument("--data-request-capacity", type=_positive_int, default=16)

    parser.add_argument("--robot-model", default="vega_1", help="Robot model")
    parser.add_argument(
        "--gripper-type", default="default", help="Gripper type (e.g. robotiq)"
    )
    parser.add_argument(
        "--robotiq-comport",
        default="/dev/ttyUSB0",
        help="Fallback serial port; dual serial grippers should use the per-arm flags",
    )
    parser.add_argument(
        "--robotiq-comport-left",
        default=None,
        help="Left arm's Robotiq serial port",
    )
    parser.add_argument(
        "--robotiq-comport-right",
        default=None,
        help="Right arm's Robotiq serial port",
    )
    parser.add_argument("--control-hz", type=int, default=20)
    parser.add_argument(
        "--frame-type",
        default="vega_mobile_base",
        choices=["vega_mobile_base", "vega_table_mount", "vega_custom"],
    )
    parser.add_argument("--use-velocity-feedforward", action="store_true")
    parser.add_argument(
        "--base-frame-rotation",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
    )
    parser.add_argument(
        "--ik-solver", dest="ik_solver_type", default="pink", choices=["pink", "placo"]
    )
    parser.add_argument("--gripper-iface", default=None)
    parser.add_argument("--ema-alpha", type=float, default=0.0)
    parser.add_argument("--ik-damping-default", type=float, default=1e-3)
    parser.add_argument("--ik-damping-torso", type=float, default=30000.0)
    parser.add_argument("--ik-damping-arm-j2", type=float, default=100.0)
    parser.add_argument("--ik-damping-arm-j3", type=float, default=50.0)
    parser.add_argument(
        "--interpolation-method",
        default="none",
        choices=["none", "linear", "cubic"],
    )
    parser.add_argument("--interpolation-history", type=int, default=4)
    parser.add_argument("--control-loop-hz", type=int, default=0)
    parser.add_argument(
        "--filter-type",
        default="none",
        choices=["none", "butterworth", "ema"],
    )
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
    args = parser.parse_args(argv)

    gripper_addr = args.gripper_iface or args.robotiq_comport
    connection = NodeConnectionConfig(
        loop_endpoint=args.loop_endpoint,
        status_period_ms=args.status_period_ms,
        control_request_capacity=args.control_request_capacity,
        data_request_capacity=args.data_request_capacity,
    )
    serve_dual_arm(
        node_id=args.node_id,
        connection=connection,
        left_robotiq_comport=args.robotiq_comport_left,
        right_robotiq_comport=args.robotiq_comport_right,
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


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


if __name__ == "__main__":
    main()
