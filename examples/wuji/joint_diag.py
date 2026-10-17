#!/usr/bin/env python3
"""Per-joint health check for a Wuji hand — no motion commanded.

Connects one hand through the SAME selection path as the servers
(hand map / SN convention / SDO), enables it, then reports one row per
expected joint: whether the joint ever appeared in the diagnostics and
state streams (a dead/offline joint simply never appears), its motor
ext_state (2 = Enabled — anything else means the motor is limp), and its
current position. Use it when a finger segment is limp or unresponsive,
to tell a dead motor from a retargeting/command problem.

Usage:
    python examples/wuji/joint_diag.py --handedness left [--seconds 5]
    python examples/wuji/joint_diag.py --handedness left --sn WH2JA01260801003

Interpretation:
    seen=NO                -> joint never reported: offline (wiring/motor/board)
    ext_state != 2         -> motor not Enabled: faulted or enable failed (limp)
    enabled but not moving -> mechanics/transmission, or the command side;
                              cross-check with bench_test.py's index sinusoid
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dexcontrol.core.wuji_hand import (  # noqa: E402
    _FINGERS,
    _JOINTS_PER_FINGER,
    _N_JOINTS,
    _detect_nid_scheme,
    _nid_to_joint,
    WujiHandAdapter,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handedness", required=True, choices=["left", "right"])
    ap.add_argument("--sn", default=None, help="Explicit device serial number")
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="How long to collect diagnostics/state frames")
    ap.add_argument("--watch", type=float, default=0.0, metavar="SECONDS",
                    help="After the table, stream positions live for this "
                         "long. Move the suspect joint BY HAND: a changing "
                         "value = encoder alive (motor/enable problem); a "
                         "frozen value = dead joint.")
    args = ap.parse_args()

    hand = WujiHandAdapter(handedness=args.handedness, sn=args.sn,
                           effort_limit=0.5)
    print(f"Connected: model={hand.model} sn={hand.serial_number}")
    if not hand._is_hand2:
        print("NOTE: joint_diagnostics ext_state is a Hand 2 stream; on v1 "
              "this script only reports positions.")

    seen_diag: dict[int, object] = {}
    seen_state: set[int] = set()
    diag_sub = None
    if hand._is_hand2:
        diag_sub = hand._hand.joint_diagnostics().subscribe()
    state_sub = hand._hand.joint_states().subscribe()

    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            time.sleep(0.05)
            if diag_sub is not None:
                while True:
                    f = diag_sub.recv()
                    if f is None:
                        break
                    for e in f.joints:
                        seen_diag[e.nid] = e
            while True:
                f = state_sub.recv()
                if f is None:
                    break
                for e in f.joints:
                    seen_state.add(e.nid)
    finally:
        if diag_sub is not None:
            diag_sub.close()
        state_sub.close()

    # The raw nid set is printed as-is — it is the ground truth when the
    # display and the physical hand seem to disagree. The vendor's device
    # snapshot confirms stride-5 finger groups on Hand 2 (see
    # _detect_nid_scheme); contiguous schemes are handled as fallbacks.
    nids = sorted(set(seen_diag) | seen_state)
    if not nids:
        print("No joint frames received at all — link/stream problem, not a "
              "per-joint fault.")
        hand.shutdown()
        return 1
    print(f"\nraw nids seen ({len(nids)}): {nids}")
    scheme = _detect_nid_scheme(nids)
    print(f"nid scheme: {scheme}")
    nid_of = {}
    for n in range(5 * _N_JOINTS):
        j = _nid_to_joint(n, scheme)
        if j >= 0:
            nid_of[j] = n

    pos = hand.get_joint_pos()
    pos = np.full(_N_JOINTS, np.nan) if pos is None else np.asarray(pos)

    bad = []
    print(f"\n{'joint':12s} {'nid':>4s} {'seen':>5s} {'ext_state':>9s} {'pos':>8s}")
    for j in range(_N_JOINTS):
        finger = _FINGERS[j // _JOINTS_PER_FINGER]
        label = f"F{j // _JOINTS_PER_FINGER + 1}J{j % _JOINTS_PER_FINGER + 1} {finger}"
        nid = nid_of[j]
        in_stream = nid in seen_state or nid in seen_diag
        ext = "-"
        entry = seen_diag.get(nid)
        if entry is not None:
            ext = str(getattr(getattr(entry, "status_word", None),
                              "ext_state", "?"))
        healthy = in_stream and (ext in ("2", "-") if diag_sub is None else ext == "2")
        if not healthy:
            bad.append(label)
        mark = "" if healthy else "   <-- CHECK"
        print(f"{label:12s} {nid:4d} {('yes' if in_stream else 'NO'):>5s} "
              f"{ext:>9s} {pos[j]:8.3f}{mark}")

    if bad:
        print(f"\nUNHEALTHY: {', '.join(bad)}")
        print("seen=NO -> joint offline (never reported: wiring/motor/board).")
        print("ext_state!=2 -> motor not Enabled -> limp; try power-cycling "
              "the hand, then rerun. If it persists, capture this output for "
              "the vendor (RMA evidence).")
    else:
        print("\nAll joints online and Enabled. A finger that still does not "
              "track the glove points at retargeting/command mapping, not "
              "hardware — cross-check with bench_test.py (index sinusoid).")

    if args.watch > 0:
        print(f"\nWatching positions for {args.watch:.0f}s — move the suspect "
              "joint by hand. '*' marks joints that moved >0.02 rad from the "
              "baseline (encoder alive).")
        base = np.array(pos)
        end = time.monotonic() + args.watch
        while time.monotonic() < end:
            time.sleep(0.25)
            cur = hand.get_joint_pos()
            if cur is None:
                continue
            cur = np.asarray(cur)
            moved = np.abs(cur - base) > 0.02
            cells = []
            for f in range(len(_FINGERS)):
                seg = cur[f * _JOINTS_PER_FINGER:(f + 1) * _JOINTS_PER_FINGER]
                mk = moved[f * _JOINTS_PER_FINGER:(f + 1) * _JOINTS_PER_FINGER]
                cells.append(_FINGERS[f][:2] + ":" + ",".join(
                    f"{v:+.2f}{'*' if m else ' '}" for v, m in zip(seg, mk)))
            print("\r" + "  ".join(cells), end="", flush=True)
        print()

    hand.shutdown()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
