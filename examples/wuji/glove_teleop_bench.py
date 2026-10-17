#!/usr/bin/env python3
"""M3.5 — drive a Wuji hand straight from the Rokoko gloves. No arm, no gRPC.

Glove stream -> HandSolver (21 MediaPipe keypoints) -> RetargetSession (20 rad)
-> WujiHandAdapter. The shortest possible path from a human hand to the robot
hand, so a mismatch here is in the solve/retarget/driver chain and not in the
arm, the controller, or the network.

Run it on the machine the hand is plugged into (the robot). The Rokoko driver
normally runs on the teleop host, so point --rokoko-host at that machine; the
RGMP port is a plain TCP stream and the client is the same either way.

The solver and retargeter live in robot-control-interface. Copy that package to
this machine once and pass its root:

    rsync -a ubuntu@<teleop-host>:~/robot-control-interface/core \\
             ubuntu@<teleop-host>:~/robot-control-interface/config \\
             ~/rci-rokoko/

Usage:
    python examples/wuji/glove_teleop_bench.py --handedness left \\
        --rokoko-host 192.168.5.10 --rci-path ~/rci-rokoko [--effort-limit 1.0]

Safety: the hand ramps from its current pose to the first glove pose over
--ramp seconds instead of jumping, holds its last pose when tracking drops,
and returns to the open pose before disabling. Keep --effort-limit low.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dexcontrol.core.wuji_hand import WujiHandAdapter  # noqa: E402

CONTROL_HZ = 60.0
STALE_S = 0.2          # glove sample older than this counts as tracking loss
REPORT_EVERY_S = 1.0


def _load_rokoko(rci_path: str):
    """Import the solver/retarget package from a robot-control-interface tree."""
    root = Path(os.path.expanduser(rci_path)).resolve()
    if not (root / "core" / "controllers" / "rokoko").is_dir():
        raise SystemExit(
            f"No core/controllers/rokoko under {root}. Copy it from the teleop "
            f"host (see the module docstring) or pass --rci-path."
        )
    sys.path.insert(0, str(root))
    from core.controllers.rokoko.calibration import load_calibration
    from core.controllers.rokoko.hand_solver import HandSolver
    from core.controllers.rokoko.retargeting import make_retargeter
    from core.controllers.rokoko.rgmp_client import RgmpClient

    return RgmpClient, HandSolver, make_retargeter, load_calibration


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handedness", required=True, choices=["left", "right"])
    ap.add_argument("--sn", default=None, help="Explicit hand serial number")
    ap.add_argument("--rokoko-host", default="127.0.0.1",
                    help="Host running the Rokoko SDK driver")
    ap.add_argument("--rokoko-port", type=int, default=12276)
    ap.add_argument("--rci-path", default="~/rci-rokoko",
                    help="robot-control-interface tree holding core/controllers/rokoko")
    ap.add_argument("--effort-limit", type=float, default=1.0, help="Amps")
    ap.add_argument("--duration", type=float, default=120.0, help="Seconds to run")
    ap.add_argument("--ramp", type=float, default=2.0,
                    help="Seconds to blend from the current pose into the glove pose")
    args = ap.parse_args()

    RgmpClient, HandSolver, make_retargeter, load_calibration = _load_rokoko(args.rci_path)
    side = args.handedness

    client = RgmpClient(host=args.rokoko_host, port=args.rokoko_port)
    client.start()
    print(f"Waiting for the {side} glove at {args.rokoko_host}:{args.rokoko_port} ...")
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and client.get_glove(side) is None:
        time.sleep(0.25)
    if client.get_glove(side) is None:
        client.stop()
        print(f"No {side} glove stream. Is the driver running on that host?")
        return 1

    calibration = load_calibration() or {}
    if not calibration.get(side):
        print(f"WARNING: no calibration for the {side} glove — poses will be "
              f"less accurate. Run scripts/rokoko_calibrate.py calibrate.")
    solver = HandSolver(is_right=(side == "right"),
                        calibration_offsets=calibration.get(side))

    hand = WujiHandAdapter(handedness=side, sn=args.sn,
                           effort_limit=args.effort_limit)
    print(f"Connected: model={hand.model} sn={hand.serial_number}")
    # Retarget for the generation actually attached — a v1 retargeter driving
    # a v2 hand is exactly the silent mismatch the server refuses to start on.
    retargeter = make_retargeter(side, hand.model)
    print(f"Driving the {side} hand from the {side} glove for {args.duration:.0f}s. "
          f"Ctrl-C to stop.")

    open_pose = hand.get_predefined_pose("open")
    start_pose = hand.get_joint_pos()
    if start_pose is None or not np.isfinite(start_pose).all():
        start_pose = open_pose.copy()
    target = start_pose.copy()

    period = 1.0 / CONTROL_HZ
    t0 = time.monotonic()
    next_report = t0
    tracked_frames = lost_frames = 0
    try:
        while time.monotonic() - t0 < args.duration:
            loop_start = time.monotonic()
            entry = client.get_glove(side)
            fresh = entry is not None and entry[1] <= STALE_S
            if fresh:
                sample, _ = entry
                result = solver.solve(sample.quats, sample.positions,
                                      sample.coil_present())
                kp = result.keypoints
                if np.isfinite(kp).all():
                    qpos = retargeter.step(kp)
                    if np.isfinite(qpos).all():
                        # Blend in over --ramp seconds so the hand eases into
                        # the operator pose instead of snapping to it.
                        elapsed = loop_start - t0
                        alpha = min(1.0, elapsed / args.ramp) if args.ramp > 0 else 1.0
                        target = (1.0 - alpha) * start_pose + alpha * qpos
                        tracked_frames += 1
            else:
                lost_frames += 1  # hold the last target

            hand.set_joint_pos(target)

            if loop_start >= next_report:
                actual = hand.get_joint_pos()
                err = (float(np.max(np.abs(actual - target)))
                       if actual is not None and np.isfinite(actual).all() else float("nan"))
                print(f"[{loop_start - t0:5.1f}s] tracked={fresh} "
                      f"thumb={np.round(target[0:4], 2)} "
                      f"index={np.round(target[4:8], 2)} "
                      f"max|cmd-actual|={err:.3f}")
                next_report = loop_start + REPORT_EVERY_S

            time.sleep(max(0.0, period - (time.monotonic() - loop_start)))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Ease back to open before releasing the motors.
        try:
            for a in np.linspace(0.0, 1.0, int(CONTROL_HZ)):
                hand.set_joint_pos((1.0 - a) * target + a * open_pose)
                time.sleep(period)
        except Exception:
            pass
        hand.shutdown()
        client.stop()

    total = tracked_frames + lost_frames
    if total:
        print(f"Frames: {tracked_frames} tracked / {lost_frames} lost "
              f"({100.0 * lost_frames / total:.1f}% held last pose)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
