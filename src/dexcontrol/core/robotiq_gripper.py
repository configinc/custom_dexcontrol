# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Robotiq 2F-85 gripper adapter for dexcontrol.

Wraps the robotiq_2f_85_controller package to expose the same interface as
DexGripper/Hand so it can be used as a drop-in replacement in VegaRobot.

Uses a background worker thread for non-blocking gripper I/O. The cached
position is updated only from real Modbus reads in the worker — `set_joint_pos`
deliberately does *not* optimistically write the commanded target into the
cache. State must reflect what the hardware actually does, not what we told
it to do; the latter conflates measured and commanded values and produces
state that snaps back to reality on contact stalls (see franka-controller
core/robotiq_gripper.py).

Requirements:
    pip install -e robotiq_2f_85_controller/
    (or pip install robotiq-2f-85-controller)
"""

from __future__ import annotations

import sys
import time
import threading
from queue import Queue, Empty, Full
from pathlib import Path

import numpy as np
from loguru import logger

# Ensure the submodule takes precedence over any namespace package with the
# same name that may shadow it (e.g. when /home/dexmate/robotiq_2f_85_controller
# is not on sys.path but a .pth editable install points elsewhere).
_SUBMODULE = Path(__file__).resolve().parents[3] / "robotiq_2f_85_controller"
if _SUBMODULE.exists() and str(_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE))

try:
    from robotiq_2f_85_controller import Robotiq2FingerGripper
except ImportError as e:
    raise ImportError(
        "robotiq_2f_85_controller is not installed. "
        "Run: pip install -e robotiq_2f_85_controller/ "
        "(or pip install robotiq-2f-85-controller)"
    ) from e

# Robotiq 2F-85 stroke in metres
_STROKE_M = 0.085

# Predefined positions in metres (open = max stroke, close = 0). Kept for
# the VegaRobot hand interface, which addresses the gripper through
# symbolic pose names.
_POSE_POOL = {
    "open": np.array([_STROKE_M], dtype=np.float64),
    "close": np.array([0.0], dtype=np.float64),
}


class RobotiqGripper:
    """Robotiq 2F-85 gripper controller compatible with the VegaRobot hand interface.

    Joint space is a single scalar in metres [0, 0.085] where 0.085 is fully
    open and 0.0 is fully closed, matching Robotiq2FingerGripper.get_pos().

    Uses a background worker thread so that Modbus serial I/O does not block
    the caller. `_cached_pos` is the worker's measured readback; commands are
    enqueued non-blockingly and never overwrite the cache.
    """

    def __init__(
        self,
        comport: str = "/dev/ttyUSB0",
        init_timeout: float = 15.0,
    ) -> None:
        """Connect to the gripper and run full initialisation.

        Args:
            comport: Serial port the gripper is connected to.
            init_timeout: Seconds to wait for the gripper to become ready.
        """
        logger.info("Connecting to Robotiq 2F-85 on {} ...", comport)
        self._gripper = Robotiq2FingerGripper(comport=comport)
        self._gripper.full_init(timeout=init_timeout)

        # Locks
        self._gripper_io_lock = threading.Lock()
        self._state_lock = threading.Lock()

        # Cached state (updated by worker thread only — measured, never commanded)
        self._cached_pos = self._gripper.get_pos()

        # Worker thread
        self._command_queue: Queue[tuple[float, float, float]] = Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="robotiq-gripper-worker",
            daemon=True,
        )
        self._worker.start()

        logger.info("Robotiq 2F-85 ready (async worker started).")

    def _worker_loop(self) -> None:
        """Background loop: execute queued commands and refresh measured state."""
        while not self._stop_event.is_set():
            # Drain queue — only keep the latest command
            latest_cmd: tuple[float, float, float] | None = None
            try:
                latest_cmd = self._command_queue.get(timeout=0.02)
                # Drain any newer commands
                while True:
                    latest_cmd = self._command_queue.get_nowait()
            except Empty:
                pass

            # Execute command
            if latest_cmd is not None:
                pos_m, vel, force = latest_cmd
                try:
                    with self._gripper_io_lock:
                        self._gripper.goto(pos=pos_m, vel=vel, force=force)
                        self._gripper.sendCommand()
                except Exception as e:
                    logger.warning("Robotiq worker command error: {}", e)

            # Refresh cached state from real Modbus read
            try:
                with self._gripper_io_lock:
                    status_ok = self._gripper.getStatus()
                if status_ok:
                    with self._state_lock:
                        self._cached_pos = self._gripper.get_pos()
            except Exception as e:
                logger.warning("Robotiq worker state refresh error: {}", e)

    def _enqueue_command(self, pos_m: float, vel: float = 0.05, force: float = 95) -> None:
        """Non-blocking enqueue. Drops stale command if queue is full."""
        cmd = (pos_m, vel, force)
        try:
            self._command_queue.put_nowait(cmd)
        except Full:
            try:
                self._command_queue.get_nowait()
            except Empty:
                pass
            try:
                self._command_queue.put_nowait(cmd)
            except Full:
                pass

    # ------------------------------------------------------------------
    # VegaRobot hand interface
    # ------------------------------------------------------------------

    def get_joint_pos(self) -> np.ndarray:
        """Return current measured gripper position (1-element array in metres)."""
        with self._state_lock:
            return np.array([self._cached_pos], dtype=np.float64)

    def set_joint_pos(
        self,
        joint_pos,
        wait_time: float = 0.0,
        **_kwargs,
    ) -> None:
        """Command gripper to a position in metres (non-blocking).

        Args:
            joint_pos: Target position. Scalar, list, or 1-element array in
                metres, clipped to [0, stroke].
            wait_time: Seconds to sleep after sending the command.
        """
        pos_m = float(np.asarray(joint_pos, dtype=np.float64).flat[0])
        pos_m = float(np.clip(pos_m, 0.0, _STROKE_M))
        self._enqueue_command(pos_m)
        if wait_time > 0.0:
            time.sleep(wait_time)

    def open_hand(self, wait_time: float = 0.0, **_kwargs) -> None:
        """Open the gripper fully (non-blocking)."""
        self._enqueue_command(_STROKE_M)
        if wait_time > 0.0:
            time.sleep(wait_time)

    def close_hand(self, wait_time: float = 0.0, **_kwargs) -> None:
        """Close the gripper fully (non-blocking)."""
        self._enqueue_command(0.0)
        if wait_time > 0.0:
            time.sleep(wait_time)

    def get_predefined_pose(self, name: str) -> np.ndarray:
        """Return a predefined pose by name ('open' or 'close').

        Args:
            name: Pose name.

        Returns:
            1-element numpy array with position in metres.

        Raises:
            KeyError: If name is not a known predefined pose.
        """
        if name not in _POSE_POOL:
            raise KeyError(f"Unknown predefined pose '{name}'. Available: {list(_POSE_POOL)}")
        return _POSE_POOL[name].copy()

    def shutdown(self) -> None:
        """Stop worker thread and disconnect from the gripper."""
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        try:
            self._gripper.shutdown()
        except Exception:
            pass
