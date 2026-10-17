#!/usr/bin/env python3
"""Wuji Hand bench test (M1) — exercises WujiHandAdapter without a robot.

Connects the requested hand, streams state at 50 Hz, runs a slow sinusoid on
one finger's flex joints, dumps tactile frames (to pin the per-model
hand_tactile schema), then disables cleanly. Unplug the USB/Ethernet cable
mid-run to verify the watchdog pauses the command stream and recovers.

Usage:
    python examples/wuji/bench_test.py --handedness right [--sn SN]
        [--effort-limit 1.0] [--duration 20] [--no-motion]

Run one process per hand (left/right) to verify both hands drive side by side.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dexcontrol.core.wuji_hand import WujiHandAdapter  # noqa: E402

STATE_HZ = 50.0
SINE_PERIOD_S = 4.0
SINE_AMPLITUDE_RAD = 0.6
SINE_FINGER = 1  # index finger
REPORT_EVERY_S = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handedness", required=True, choices=["left", "right"])
    parser.add_argument("--sn", default=None, help="Explicit device serial number")
    parser.add_argument("--effort-limit", type=float, default=1.0,
                        help="Amps; bench default is lower than the 1.5 runtime default")
    parser.add_argument("--duration", type=float, default=20.0, help="Seconds to run")
    parser.add_argument("--no-motion", action="store_true",
                        help="State/tactile streaming only, no joint commands")
    args = parser.parse_args()

    hand = WujiHandAdapter(
        handedness=args.handedness,
        sn=args.sn,
        effort_limit=args.effort_limit,
    )
    print(f"Connected: model={hand.model} sn={hand.serial_number} "
          f"tactile={hand.has_tactile}")
    print(f"Joint limits (rad):\n{np.round(hand.joint_limits, 3)}")

    open_pose = hand.get_predefined_pose("open")
    target = open_pose.copy()
    t0 = time.monotonic()
    next_report = t0
    period = 1.0 / STATE_HZ

    try:
        if not args.no_motion:
            hand.open_hand()
        while time.monotonic() - t0 < args.duration:
            tick = time.monotonic()
            elapsed = tick - t0

            if not args.no_motion:
                # Slow sinusoid on one finger's flex joints (j1 + j3).
                curl = SINE_AMPLITUDE_RAD * 0.5 * (1 - np.cos(2 * np.pi * elapsed / SINE_PERIOD_S))
                target[:] = open_pose
                target[SINE_FINGER * 4 + 0] = curl
                target[SINE_FINGER * 4 + 2] = curl
                hand.set_joint_pos(target)

            if tick >= next_report:
                pos = hand.get_joint_pos()
                tactile = hand.get_hand_tactile()
                print(
                    f"[{elapsed:5.1f}s] connected={hand.is_connected} "
                    f"finger{SINE_FINGER + 1}={np.round(pos[SINE_FINGER * 4:SINE_FINGER * 4 + 4], 3)} "
                    f"tactile={'None' if tactile is None else f'shape={tactile.shape} max={np.nanmax(tactile):.3f}'}"
                )
                if tactile is not None and elapsed < REPORT_EVERY_S * 2:
                    # First full dumps — used to pin the per-model schema.
                    print(f"  tactile full: {np.round(tactile, 4).tolist()}")
                next_report = tick + REPORT_EVERY_S

            dt = time.monotonic() - tick
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if not args.no_motion:
            hand.open_hand(wait_time=0.5)
        hand.shutdown()
        print("Shut down cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
