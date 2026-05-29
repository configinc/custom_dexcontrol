# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""SR gripper adapter for dexcontrol.

Wraps `sr_gripper_controller.SrGripper` to expose the same interface as
DexGripper/Hand so it can be used as a drop-in replacement in VegaRobot.

SR-firmware specifics worth knowing (verified upstream by
sr_gripper_controller test_stall_cause_sweep.py 2026-05-04):

  * A single isolated `goto + sendCommand` stalls 3-13% of the time on
    close-direction motions — the encoder freezes mid-motion. The same
    target re-streamed at 50-200 ms cadence stalls 0/30 times. The
    worker therefore re-issues the latest target on a steady 100 ms
    tick whenever there's nothing new in the queue. Continuous
    trajectory callers do this implicitly; we do it explicitly so a
    one-off close doesn't get stuck.

  * Two failure modes constrain the wire-level send cadence; we have to
    sit between them:
      - Too aggressive (50 Hz of identical packets): every
        `client.sendCommand()` triggers a `_clear_error_latch()` SDO
        write when `dcu_error != 0` (the motion-done latch sits at
        0x0002 right after arrival). 50 Hz of latch-clear + replay
        cycles the firmware's motion-done state machine, and after
        ~28 s the gripper stops accepting new commands (observed:
        unresponsive after ~2-3 reset() calls).
      - Too sparse (purely deduped silence at endpoint hold): the SR
        firmware has no internal position-hold loop (Robotiq does, SR
        does not — see commit 09ce9590). It needs periodic
        application-level "command update events" or its application
        state machine wedges. Observed 2026-05-10: 53 s silent at
        TARGET_OPEN -> motor reports `dcu_status=8 (moving)` to new
        commands but encoder doesn't budge.
    Resolution: the worker dedups identical packets but force-resends
    once every _FORCE_RESEND_PERIOD_S (100 ms) so the firmware always
    sees a fresh command event within its wedge window, while the
    cadence stays at 1/5 of the rate that triggers the release/re-grip
    cycle. While the slew target is still moving (delta > slew limit),
    each tick already produces a different packet so dedup is naturally
    bypassed and stall-avoidance is preserved. The inner upstream
    `_pdo_worker` continues to stream the output buffer at ~1 kHz to
    keep the EtherCAT bus connection itself alive — that is independent
    of our re-issue cadence.

    Even with the cadence sitting in the safe band, individual physical
    units occasionally still wedge (per-unit firmware/mechanical
    variance — left arm dies while right arm survives the same hold).
    `reconnect()` exposes the proven empirical recovery path: tear
    down the EtherCAT master and re-run full_init's activation
    sequence (calibrate=False so no extra motion). Wire it into the
    higher-level reset() flow if a single arm wedges occasionally.

  * Real position feedback comes back at 1 kHz from the upstream PDO
    worker. We rely on the worker thread to refresh `_cached_pos` and
    deliberately do *not* optimistically write the commanded target
    into the cache — fighting the real measurement with an optimistic
    overwrite produced visible jumps in `get_joint_pos()`.

  * Convention: this adapter exposes the SR driver's *native* frame
    (0 m = open, stroke m = closed), matching the dexcontrol Robotiq
    wrapper so VegaRobot pairs `open`/`close` symbolically the same
    way across both grippers. (The franka-controller variant flips to
    the Robotiq external frame internally because its callers use that
    convention; here we don't.)

Requirements:
    pip install -e sr_gripper_controller/
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

import numpy as np
from loguru import logger

# Ensure the submodule takes precedence over any namespace package with the
# same name that may shadow it.
_SUBMODULE = Path(__file__).resolve().parents[3] / "sr_gripper_controller"
if _SUBMODULE.exists() and str(_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE))

try:
    from sr_gripper_controller import SrGripper
    from sr_gripper_controller._ethercat import (
        build_ctrl_packet,
        CTRL_POSITION,
        TARGET_OPEN,
        TARGET_CLOSE,
    )
except ImportError as e:
    raise ImportError(
        "sr_gripper_controller is not installed. "
        "Run: pip install -e sr_gripper_controller/"
    ) from e

# SR gripper stroke in metres. Driver native frame: 0 m = open, stroke = closed.
_STROKE_M = 0.11

# Predefined positions in metres (driver native). Mirrors the dexcontrol
# Robotiq wrapper's symbolic mapping so VegaRobot pairs `open`/`close` the
# same way across both grippers.
_POSE_POOL = {
    "open": np.array([0.0], dtype=np.float64),
    "close": np.array([_STROKE_M], dtype=np.float64),
}

# Per-call slew limit. See module docstring rationale.
SLEW_PER_CALL_M = 0.022

# Worker tick period. 20 ms = 50 Hz. Slew progresses one tick per cycle and
# the queue drain runs at this rate.
_RESTREAM_PERIOD_S = 0.02

# Force-resend floor. Even if the wire packet is identical to the last one
# sent (dedup hit), we re-issue once every _FORCE_RESEND_PERIOD_S so the SR
# firmware keeps seeing application-level "command update events". Without
# this the firmware wedges on long endpoint holds (verified 2026-05-10:
# 53 s silent at TARGET_OPEN -> motor stops accepting commands even though
# dcu_status replies "moving"). 100 ms matches upstream goto_blocking's
# restream_dt and is well below the wedge time we observed; the 10 Hz
# cadence is also 1/5 of the 50 Hz that triggers the latch release/re-grip
# cycle, so we get the keep-alive without the cycling.
_FORCE_RESEND_PERIOD_S = 0.1


class SrGripperAdapter:
    """SR gripper controller compatible with the VegaRobot hand interface.

    Joint space is a single scalar in metres [0, stroke] in the SR driver's
    native frame: 0 m = open, stroke m = closed (matches the dexcontrol
    Robotiq wrapper's exposed convention).

    Uses a background worker thread so that EtherCAT I/O does not block the
    caller. `_cached_pos` is the worker's measured readback only; commands
    are sent synchronously through a dedup + force-resend committer and
    never overwrite the cache.
    """

    def __init__(
        self,
        comport: str = "eth0",
        init_timeout: float = 20.0,
    ) -> None:
        """Connect to the gripper and run full initialisation.

        Args:
            comport: EtherCAT network interface the gripper is on.
            init_timeout: Seconds to wait for the gripper to become ready.
        """
        # Saved for reconnect(); both fields are read-only after __init__.
        self._comport = comport
        self._init_timeout = init_timeout

        # Locks survive across reconnect() so any external holders don't
        # see a torn reference.
        self._gripper_io_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._gripper: SrGripper | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._bringup(calibrate=True)

    def _bringup(self, calibrate: bool) -> None:
        """Open EtherCAT, run full_init, start worker. Shared by __init__ and reconnect()."""
        logger.info("Connecting to SR gripper on {} ...", self._comport)
        self._gripper = SrGripper(comport=self._comport, stroke=_STROKE_M)
        self._gripper.full_init(timeout=self._init_timeout, calibrate=calibrate)

        # Cached state in driver native frame (0 = open, stroke = closed).
        self._cached_pos = float(self._gripper.get_pos())

        # Slew limit tracker — last target we *issued* to the firmware
        # (clipped if needed). The caller's requested target may be further
        # away; subsequent set_joint_pos calls continue converging at
        # SLEW_PER_CALL_M per tick.
        self._last_slew_target: float | None = None
        # Adapter-level dedup + worker re-stream source. Mirrors upstream
        # SrGripper.sendCommand()'s dedup guard but adds a periodic
        # force-resend floor (_FORCE_RESEND_PERIOD_S) to keep the SR
        # firmware's application state machine alive on long holds. The
        # worker re-streams this packet from `_resend_active`; the dedup
        # window prevents the 50 Hz cycling that wedges the firmware.
        self._last_sent_packet: bytes = b""
        self._last_send_t: float = 0.0

        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="sr-gripper-worker",
            daemon=True,
        )
        self._worker.start()

        logger.info("SR gripper ready (async worker started).")

    def _pos_to_raw_target(self, pos_m: float) -> int:
        """Map driver-native metres to driver raw int target.

        Driver native: 0 m (open) -> raw TARGET_OPEN, stroke (closed) -> raw TARGET_CLOSE.
        """
        # Same map as SrGripper._pos_to_raw_target (kept inlined so we
        # don't have to call into the driver at every control tick).
        return int(round(
            TARGET_OPEN
            + (TARGET_CLOSE - TARGET_OPEN) * (pos_m / _STROKE_M)
        ))

    def _commit_target(self, pos_m: float, vel: float, force: float) -> None:
        """Build a fresh control packet for a new target and send it.

        Runtime path. Deliberately avoids `SrGripper.goto()` — that
        mutates rACT/rGTO/rPR/_last_packet and is the wrong shape for a
        streaming control loop. `goto()` belongs to init/calibration
        only. Here we just rebuild the 14-byte position packet and ship
        it on the wire (matches the validated upstream pattern in
        tests/test_dual_realtime_plot.py:80-89, 192-193).
        """
        raw_target = self._pos_to_raw_target(pos_m)
        packet = build_ctrl_packet(CTRL_POSITION, raw_target)
        now = time.time()
        # Dedup: identical packet within the force-resend window -> skip.
        # Identical packet outside that window -> resend (firmware keep-alive).
        # Different packet -> always send (slew progressing or new caller target).
        if (packet == self._last_sent_packet
                and (now - self._last_send_t) < _FORCE_RESEND_PERIOD_S):
            return
        with self._gripper_io_lock:
            ok = self._gripper.client.sendCommand(packet)
        if ok:
            self._last_sent_packet = packet
            self._last_send_t = now

    def _resend_active(self) -> None:
        """Re-stream the last-sent packet, gated by the force-resend window.

        The worker calls this every _RESTREAM_PERIOD_S (20 ms). The
        force-resend gate keeps the effective wire cadence at ~10 Hz on
        steady caller targets, which is the safe band: fast enough that
        the SR firmware stays out of the silent-hold wedge, slow enough
        that the per-send `_clear_error_latch` SDO write does not cycle
        the motion-done state machine into the unresponsive wedge.
        """
        if not self._last_sent_packet:
            return
        now = time.time()
        if (now - self._last_send_t) < _FORCE_RESEND_PERIOD_S:
            return
        with self._gripper_io_lock:
            ok = self._gripper.client.sendCommand(self._last_sent_packet)
        if ok:
            self._last_send_t = now

    def _worker_loop(self) -> None:
        """Background loop: re-assert the latest commit every tick + refresh state.

        Caller-driven set_joint_pos / open_hand / close_hand still send
        synchronously through `_commit_target`. The worker re-streams the
        same packet via `_resend_active` only when the force-resend
        window has elapsed, so caller-driven sends and worker re-streams
        share a single ~10 Hz wire cadence on steady targets. The SR
        firmware needs that periodic command-update event because it
        has no internal position-hold loop. Every tick also pulls a
        fresh status frame so get_joint_pos() reflects real motor
        position.
        """
        while not self._stop_event.is_set():
            time.sleep(_RESTREAM_PERIOD_S)
            try:
                self._resend_active()
            except Exception as e:
                logger.warning("SR worker re-stream error: {}", e)

            try:
                with self._gripper_io_lock:
                    status_ok = self._gripper.getStatus()
                if status_ok:
                    with self._state_lock:
                        self._cached_pos = float(self._gripper.get_pos())
            except Exception as e:
                logger.warning("SR worker state refresh error: {}", e)

    # ------------------------------------------------------------------
    # VegaRobot hand interface
    # ------------------------------------------------------------------

    def get_joint_pos(self) -> np.ndarray:
        """Return current measured gripper position (1-element array in metres).

        Driver-native frame: 0 = open, stroke = closed. Non-blocking
        (returns cached state from the worker thread).
        """
        with self._state_lock:
            return np.array([self._cached_pos], dtype=np.float64)

    def set_joint_pos(
        self,
        joint_pos,
        wait_time: float = 0.0,
        **_kwargs,
    ) -> None:
        """Command gripper to a position in metres.

        Sends synchronously so a teleop trajectory isn't downsampled to
        the worker's idle-hold tick. The worker only kicks in for the
        re-stream when no fresh command is arriving.

        Args:
            joint_pos: Target position. Scalar, list, or 1-element array in
                metres, clipped to [0, stroke]. Driver native:
                0 = open, stroke = closed.
            wait_time: Seconds to sleep after sending the command.
        """
        pos_m = float(np.asarray(joint_pos, dtype=np.float64).flat[0])
        pos_m = float(np.clip(pos_m, 0.0, _STROKE_M))

        # Slew limit anchored on *measured* position, not on the previous
        # commanded target. The user-visible failure mode is the command
        # racing far ahead of where the motor actually is — once command
        # and measured diverge by more than the firmware can chase in one
        # tick, a sharp reversal sweeps the full stroke. Clamping the
        # issued target to (measured ± SLEW_PER_CALL_M) keeps the
        # firmware tracking instead of bang-banging, even when the caller
        # waves the gripper through aggressive zigzags.
        with self._state_lock:
            measured = self._cached_pos
        if measured != measured:  # NaN guard
            measured = pos_m
        delta = pos_m - measured
        if delta > SLEW_PER_CALL_M:
            issued = measured + SLEW_PER_CALL_M
        elif delta < -SLEW_PER_CALL_M:
            issued = measured - SLEW_PER_CALL_M
        else:
            issued = pos_m
        issued = float(np.clip(issued, 0.0, _STROKE_M))
        self._last_slew_target = issued

        try:
            self._commit_target(issued, 0.05, 95)
        except Exception as e:
            logger.warning("SR set_joint_pos send error: {}", e)
        # No optimistic _cached_pos update — the worker fights it with
        # the real 1 kHz PDO measurement otherwise, producing visible jumps.
        if wait_time > 0.0:
            time.sleep(wait_time)

    def open_hand(self, wait_time: float = 0.0, **_kwargs) -> None:
        """Open the gripper fully (driver native: 0 m)."""
        self.set_joint_pos(0.0, wait_time=wait_time)

    def close_hand(self, wait_time: float = 0.0, **_kwargs) -> None:
        """Close the gripper fully (driver native: stroke)."""
        self.set_joint_pos(_STROKE_M, wait_time=wait_time)

    def get_predefined_pose(self, name: str) -> np.ndarray:
        """Return a predefined pose by name ('open' or 'close')."""
        if name not in _POSE_POOL:
            raise KeyError(f"Unknown predefined pose '{name}'. Available: {list(_POSE_POOL)}")
        return _POSE_POOL[name].copy()

    def reconnect(self) -> None:
        """Tear down and re-establish the SR EtherCAT connection.

        Empirically equivalent to restarting the controller: when the SR
        firmware enters a wedge state where the slave still replies but
        the motor stops responding to new position targets, re-running
        the EtherCAT bring-up + full_init activation sequence cleanly
        recovers it. Calls into upstream's connectToDevice -> SAFEOP ->
        OP -> _clear_error_latch -> full_init path, just like a fresh
        __init__.

        Skips the open-close-open calibration sweep so the gripper does
        not physically move during recovery; the per-unit encoder
        limits learned at the initial __init__ are saved and restored
        across the bring-up so get_pos() stays accurate. ~2 s end-to-end.

        Caller should not be issuing commands concurrently — set_joint_pos
        calls during reconnect will be silently dropped (the worker is
        being torn down). Safe to invoke from inside reset() flows.
        """
        # Preserve per-unit calibration: a fresh SrGripper would otherwise
        # fall back to the module defaults (~9 counts off in the field).
        saved_enc_open = self._gripper.enc_open
        saved_enc_closed = self._gripper.enc_closed

        logger.info("SR reconnect: tearing down ...")
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        try:
            self._gripper.shutdown()
        except Exception:
            pass

        self._bringup(calibrate=False)

        self._gripper.enc_open = saved_enc_open
        self._gripper.enc_closed = saved_enc_closed
        logger.info("SR reconnect: done.")

    def shutdown(self) -> None:
        """Stop worker thread and disconnect from the gripper."""
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        try:
            self._gripper.shutdown()
        except Exception:
            pass
