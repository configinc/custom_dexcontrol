#!/usr/bin/env python3
"""One-time per-unit setup: map each Wuji hand serial number to left/right.

Writes ~/.wuji/hands.json (override with WUJI_HAND_MAP), which WujiHandAdapter
reads to bind each arm's server to the correct hand. Run this once per robot,
and again whenever a hand is swapped.

Why it exists: shipped serials are plain STM32 UIDs (e.g. "367A39773134") with
no handedness encoded, so nothing in the repo can know which hand is which on
a given unit. Serials are unit-specific, so they must never be committed to a
shared playbook — they live on the robot host instead.

Modes:
  --auto     trust the SDK's reported handedness (fails if it reports nothing
             or if both hands claim the same side)
  (default)  wiggle each hand in turn and ask the operator which one moved —
             works regardless of what the firmware reports

Usage:
    python examples/wuji/identify_hands.py                  # guided
    python examples/wuji/identify_hands.py --auto           # SDK-reported
    python examples/wuji/identify_hands.py --show           # print current map
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dexcontrol.core.wuji_hand import _HAND_MAP_PATH  # noqa: E402
from dexcontrol.core.wuji_hand import WujiHandAdapter  # noqa: E402

WIGGLE_SECONDS = 4.0
WIGGLE_AMPLITUDE_RAD = 0.5


def _scan() -> list:
    from wuji_sdk import DeviceType, SdkManager

    hand_types = (DeviceType.WujiHand, DeviceType.WujiHand2)
    devices = [d for d in SdkManager.instance().scan() if d.device_type in hand_types]
    return sorted(devices, key=lambda d: d.sn)


def _wiggle(sn: str, effort_limit: float) -> None:
    """Open/close one hand so the operator can see which physical hand it is."""
    import numpy as np

    # handedness is a required argument but irrelevant here: we pass an
    # explicit sn, which takes precedence over any handedness-based selection.
    hand = WujiHandAdapter(handedness="right", sn=sn, effort_limit=effort_limit)
    try:
        open_pose = hand.get_predefined_pose("open")
        close_pose = hand.get_predefined_pose("close")
        span = close_pose - open_pose
        t0 = time.monotonic()
        while time.monotonic() - t0 < WIGGLE_SECONDS:
            phase = 0.5 - 0.5 * np.cos(2 * np.pi * (time.monotonic() - t0) / 1.5)
            hand.set_joint_pos(open_pose + WIGGLE_AMPLITUDE_RAD * phase * span)
            time.sleep(0.02)
        hand.set_joint_pos(open_pose)
        time.sleep(0.3)
    finally:
        hand.shutdown()


def _ask_side(sn: str) -> str:
    while True:
        answer = input(f"  Which hand moved? [l]eft / [r]ight / [s]kip: ").strip().lower()
        if answer in ("l", "left"):
            return "left"
        if answer in ("r", "right"):
            return "right"
        if answer in ("s", "skip"):
            return ""
        print("  Please answer l, r or s.")


def _write(mapping: dict) -> Path:
    path = Path(_HAND_MAP_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(mapping, indent=2) + "\n")
    os.replace(tmp, path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--auto", action="store_true",
                    help="Use the handedness the SDK reports instead of asking")
    ap.add_argument("--show", action="store_true", help="Print the current map and exit")
    ap.add_argument("--effort-limit", type=float, default=0.6,
                    help="Amps for the identification wiggle (keep low)")
    args = ap.parse_args()

    if args.show:
        try:
            print(f"{_HAND_MAP_PATH}:")
            print(Path(_HAND_MAP_PATH).read_text())
        except FileNotFoundError:
            print(f"No hand map at {_HAND_MAP_PATH} (auto-selection in use).")
        return 0

    devices = _scan()
    if not devices:
        print("No Wuji hands found. Check USB/Ethernet and power, then retry.")
        return 1
    print(f"Found {len(devices)} hand(s): {[d.sn for d in devices]}\n")

    mapping: dict[str, str] = {}
    for dev in devices:
        if args.auto:
            hand = WujiHandAdapter(handedness="right", sn=dev.sn,
                                   effort_limit=args.effort_limit)
            try:
                side = hand.reported_handedness
            finally:
                hand.shutdown()
            if side is None:
                print(f"{dev.sn}: the hand does not report its side "
                      "(handedness SDO reads Unknown) — rerun without --auto.")
                return 1
            print(f"{dev.sn}: reports {side}")
        else:
            print(f"{dev.sn}: wiggling for {WIGGLE_SECONDS:.0f}s — watch the hands.")
            _wiggle(dev.sn, args.effort_limit)
            side = _ask_side(dev.sn)
            if not side:
                print(f"  skipped {dev.sn}")
                continue
        if side in mapping:
            print(f"Both {mapping[side]} and {dev.sn} identified as {side} — "
                  "rerun and answer carefully (or check the hardware).")
            return 1
        mapping[side] = dev.sn

    if not mapping:
        print("Nothing identified; hand map not written.")
        return 1

    path = _write(mapping)
    print(f"\nWrote {path}: {mapping}")
    missing = {"left", "right"} - mapping.keys()
    if missing:
        print(f"Note: no serial recorded for {sorted(missing)} — plug that hand in "
              "and rerun to complete the map.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
