#!/usr/bin/env python3
"""Test compound arm motions with Robotiq 2F-85 gripper open/close.

Runs a series of arm trajectories while simultaneously commanding the gripper,
simulating pick-and-place style sequences:
  1. Reach down  → close gripper (grasp)
  2. Rise up     → hold closed
  3. Move lateral → open gripper (release)
  4. Return home

Phase 1 (interactive): step through each segment manually.
Phase 2 (continuous):  run a looping pick-place-like cycle automatically.

Usage:
  # Server must be running with --gripper-type robotiq, e.g.:
  #   python src/dexcontrol/core/robotenv_vega/server.py \
  #       --arm-side left --gripper-type robotiq

  python examples/test_robotiq_with_arm.py
  python examples/test_robotiq_with_arm.py --port 50062 --hz 20 --delta-m 0.03
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


def fmt(v, prec=3):
    return np.array2string(np.asarray(v), precision=prec, suppress_small=True, separator=", ")


def get_state(client) -> dict:
    obs, _ = client.get_observation()
    return obs["robot_state"]


def get_cart(client) -> np.ndarray:
    return np.asarray(get_state(client)["cartesian_position"], dtype=np.float64)


def run_segment(
    client,
    name: str,
    actions: list[list[float]],
    hz: float,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Execute actions at hz. Returns (cart_before, cart_after, elapsed)."""
    dt = 1.0 / hz
    cart_before = get_cart(client)
    t0 = time.time()
    for action in actions:
        t_cmd = time.time()
        client.step(action)
        remaining = dt - (time.time() - t_cmd)
        if remaining > 0:
            time.sleep(remaining)
    elapsed = time.time() - t0
    cart_after = get_cart(client)

    if verbose:
        d = cart_after - cart_before
        state = get_state(client)
        g = float(state["gripper_position"])
        print(
            f"  {name:35s}  dt={elapsed:.2f}s  "
            f"Δxyz={fmt(d[:3], 4)}m  gripper={g:.2f}"
        )
    return cart_before, cart_after, elapsed


def make_segment(
    n_steps: int,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    dr: float = 0.0,
    dp: float = 0.0,
    dyaw: float = 0.0,
    gripper_start: float = 0.5,
    gripper_end: float = 0.5,
) -> list[list[float]]:
    """Build a segment that linearly interpolates gripper while moving the arm."""
    gripper_vals = np.linspace(gripper_start, gripper_end, n_steps)
    return [
        [dx, dy, dz, dr, dp, dyaw, float(g)]
        for g in gripper_vals
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compound arm + Robotiq gripper test via RobotEnv gRPC"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50061)
    parser.add_argument("--hz", type=float, default=20.0, help="Control frequency in Hz")
    parser.add_argument("--delta-m", type=float, default=0.03, help="Position step per action (m)")
    parser.add_argument("--steps", type=int, default=20, help="Steps per segment")
    parser.add_argument("--cycles", type=int, default=3, help="Phase 2 repeat cycles")
    parser.add_argument("--no-reset", action="store_true", help="Skip initial reset")
    args = parser.parse_args()

    dm = args.delta_m
    N = args.steps

    print(f"\n=== Robotiq + Arm Compound Motion Test ===")
    print(f"Host: {args.host}:{args.port}  Hz: {args.hz}  delta: {dm}m  steps/seg: {N}")

    client = RobotEnvClient(
        robot_ip=args.host,
        robot_port=args.port,
        action_space="cartesian_delta",
        do_reset=not args.no_reset,
    )

    if not args.no_reset:
        print("Reset done. Waiting 1s ...")
        time.sleep(1.0)

    home_cart = get_cart(client)
    print(f"Home: xyz={fmt(home_cart[:3])}  rpy={fmt(np.rad2deg(home_cart[3:6]))}°")

    # ── Phase 1: Interactive pick-and-place segments ──
    #
    # Simulate: reach down (close gripper) → lift (hold) → move over (open) → return
    #
    segments_phase1 = [
        (
            "1. Reach down (gripper opens)",
            make_segment(N, dz=-dm, gripper_start=0.5, gripper_end=0.0),
        ),
        (
            "2. Grasp (close gripper, hold position)",
            make_segment(N, gripper_start=0.0, gripper_end=1.0),
        ),
        (
            "3. Lift up (gripper closed)",
            make_segment(N, dz=dm, gripper_start=1.0, gripper_end=1.0),
        ),
        (
            "4. Move lateral +Y (gripper closed)",
            make_segment(N, dy=dm, gripper_start=1.0, gripper_end=1.0),
        ),
        (
            "5. Release (open gripper, hold position)",
            make_segment(N, gripper_start=1.0, gripper_end=0.0),
        ),
        (
            "6. Return -Y (gripper open)",
            make_segment(N, dy=-dm, gripper_start=0.0, gripper_end=0.0),
        ),
        (
            "7. Lower back -Z (gripper open)",
            make_segment(N, dz=-dm, gripper_start=0.0, gripper_end=0.0),
        ),
        (
            "8. Rise to home +Z",
            make_segment(N, dz=dm, gripper_start=0.0, gripper_end=0.5),
        ),
    ]

    print(f"\n{'='*60}")
    print("PHASE 1: Interactive pick-and-place — press Enter for each segment")
    print(f"{'='*60}")
    print("\nPlanned segments:")
    for name, actions in segments_phase1:
        g_start = actions[0][-1]
        g_end = actions[-1][-1]
        print(f"  {name:40s}  gripper {g_start:.1f} → {g_end:.1f}  ({len(actions)} steps)")

    for name, actions in segments_phase1:
        input(f"\n  [{name}] Press Enter to run ...")
        run_segment(client, name, actions, args.hz)

    print(f"\nPhase 1 complete. Resetting ...")
    client.reset()
    time.sleep(1.0)

    # ── Phase 2: Continuous looping cycle ──
    #
    # A tighter pick-place loop: down+close → up → sweep → release → return
    # Uses slightly smaller delta to stay within joint limits across cycles.
    dm2 = dm * 0.8
    N2 = max(10, N // 2)

    cycle_segments = [
        ("Reach down + grasp",   make_segment(N2, dz=-dm2, gripper_start=0.3, gripper_end=1.0)),
        ("Lift + sweep +X",      make_segment(N2, dz=dm2, dx=dm2, gripper_start=1.0, gripper_end=1.0)),
        ("Release over target",  make_segment(N2, gripper_start=1.0, gripper_end=0.0)),
        ("Return -X + lower",    make_segment(N2, dx=-dm2, dz=-dm2 * 0.5, gripper_start=0.0, gripper_end=0.3)),
    ]

    print(f"\n{'='*60}")
    print(f"PHASE 2: Continuous loop — {args.cycles} cycles")
    print(f"{'='*60}")
    print("\nCycle segments:")
    for name, actions in cycle_segments:
        print(f"  {name:35s}  ({len(actions)} steps, ~{len(actions)/args.hz:.1f}s)")
    print(f"\nTotal cycle time: ~{sum(len(a) for _, a in cycle_segments) / args.hz:.1f}s")

    input("\nPress Enter to start continuous cycles ...")

    for cycle_idx in range(1, args.cycles + 1):
        print(f"\n--- Cycle {cycle_idx}/{args.cycles} ---")
        for name, actions in cycle_segments:
            run_segment(client, name, actions, args.hz)
        client.reset()
        time.sleep(0.5)

    # ── Phase 3: Diagonal arm sweep with gripper sync ──
    print(f"\n{'='*60}")
    print("PHASE 3: Diagonal sweep — gripper tracks arm Z height")
    print(f"{'='*60}")

    # As arm moves diagonally forward+down, gripper closes; reverse opens it.
    sweep_N = N * 2
    sweep_out = make_segment(sweep_N, dx=dm * 0.5, dz=-dm * 0.5, dy=dm * 0.3,
                             gripper_start=0.0, gripper_end=1.0)
    sweep_back = make_segment(sweep_N, dx=-dm * 0.5, dz=dm * 0.5, dy=-dm * 0.3,
                              gripper_start=1.0, gripper_end=0.0)

    input("\nPress Enter to run diagonal sweep ...")
    run_segment(client, "Sweep out (close gripper)", sweep_out, args.hz)
    run_segment(client, "Sweep back (open gripper)", sweep_back, args.hz)

    print(f"\nAll phases complete. Final state:")
    state = get_state(client)
    cart = np.asarray(state["cartesian_position"])
    print(f"  xyz={fmt(cart[:3])}  rpy={fmt(np.rad2deg(cart[3:6]))}°  gripper={state['gripper_position']:.3f}")

    client.close()
    print("\nConnection closed.")


if __name__ == "__main__":
    main()
