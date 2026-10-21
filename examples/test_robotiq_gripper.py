#!/usr/bin/env python3
"""Test Robotiq 2F-85 gripper control via the Vega RobotEnv gRPC server.

Cycles through open → partial close → full close → open, printing gripper
position at each step so you can verify the gripper moves correctly.

Usage:
  # Server must be running with --gripper-type robotiq, e.g.:
  #   python src/dexcontrol/core/robotenv_vega/server.py \
  #       --arm-side left --gripper-type robotiq

  python examples/test_robotiq_gripper.py
  python examples/test_robotiq_gripper.py --port 50062 --steps 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "src"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

from dexcontrol.core.robot_env_client import RobotEnvClient


def get_obs(client) -> dict:
    obs, _ = client.get_observation()
    return obs["robot_state"]


def move_gripper(
    client,
    target: float,
    current: float,
    action_space: str,
    steps: int,
    hz: float,
) -> float:
    """Ramp gripper from current to target over `steps` steps.

    For joint_position space the arm action is zeros (hold position).
    Returns the final observed gripper position.
    """
    dt = 1.0 / hz
    targets = np.linspace(current, target, steps)

    for g in targets:
        if action_space.startswith("joint"):
            # 7 arm joints + 1 gripper
            action = [0.0] * 7 + [float(g)]
        else:
            # 6 cartesian + 1 gripper
            action = [0.0] * 6 + [float(g)]
        client.step(action)
        time.sleep(dt)

    state = get_obs(client)
    return float(state["gripper_position"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Robotiq gripper via RobotEnv gRPC")
    parser.add_argument("--host", default="localhost", help="RobotEnv server host")
    parser.add_argument("--port", type=int, default=50061, help="RobotEnv server port")
    parser.add_argument(
        "--action-space",
        default="joint_position",
        choices=["joint_position", "joint_delta", "cartesian_delta"],
        help="Action space (default: joint_position)",
    )
    parser.add_argument("--hz", type=float, default=20.0, help="Control frequency in Hz")
    parser.add_argument("--steps", type=int, default=20, help="Steps per gripper motion")
    parser.add_argument("--no-reset", action="store_true", help="Skip initial reset")
    args = parser.parse_args()

    print(f"Connecting to RobotEnv at {args.host}:{args.port} ...")
    client = RobotEnvClient(
        robot_ip=args.host,
        robot_port=args.port,
        action_space=args.action_space,
        do_reset=not args.no_reset,
    )

    if not args.no_reset:
        print("Reset done. Waiting 1s ...")
        time.sleep(1.0)

    state = get_obs(client)
    gripper_pos = float(state["gripper_position"])
    print(f"\nInitial gripper position: {gripper_pos:.3f}  (0=open, 1=closed)")

    # Sequence: open → quarter → half → full close → open
    sequence = [
        ("Open (0.0)",         0.0),
        ("Quarter close (0.25)", 0.25),
        ("Half close (0.5)",   0.5),
        ("Full close (1.0)",   1.0),
        ("Open again (0.0)",   0.0),
    ]

    print(f"\nGripper sequence ({args.steps} steps each, {args.hz:.0f} Hz):")
    for name, _ in sequence:
        print(f"  {name}")

    input("\nPress Enter to start ...")

    for name, target in sequence:
        print(f"\n>>> {name}")
        state = get_obs(client)
        current = float(state["gripper_position"])
        final = move_gripper(
            client,
            target=target,
            current=current,
            action_space=args.action_space,
            steps=args.steps,
            hz=args.hz,
        )
        print(f"    target={target:.2f}  final={final:.3f}  error={abs(final - target):.3f}")
        time.sleep(0.3)

    print("\nDone.")
    client.close()
    print("Connection closed.")


if __name__ == "__main__":
    main()
