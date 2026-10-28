"""Exclusive arm/body control. Transport independent; clock and hardware are injectable."""
from __future__ import annotations

import threading
import time

import numpy as np


class ControlConflict(RuntimeError):
    pass


class BodyControl:
    PERIOD = 0.02
    TIMEOUT = 0.4

    def __init__(self, hardware, clock=time.monotonic):
        self.hardware = hardware
        self.clock = clock
        self.lock = threading.RLock()
        self.mode = "hold"
        self.error = None
        self.command = np.zeros(3)  # base vx/vy/wz; torso uses an absolute shoulder target
        self.deadline = float("inf")
        self._last_body_time = self.clock()
        self.hardware.hold()

    def _hold(self, error=None):
        self.mode = "hold"
        self.command[:] = 0
        self.deadline = float("inf")
        self.error = error
        self.hardware.hold()

    def stop(self):
        with self.lock:
            self._hold()
            return self.state()

    def state(self):
        with self.lock:
            return dict(mode=self.mode, error=self.error,
                        limits=self.hardware.limits(), **self.hardware.snapshot())

    def select(self, mode):
        if mode not in ("body", "prepare_arms"):
            raise ValueError("Expected body or prepare_arms")
        with self.lock:
            if self.mode == "transition":
                raise ControlConflict("Mode transition already in progress")
            self._hold()
            self.mode = "transition"
        # Require several fresh stopped samples, not just a stop acknowledgement.
        until = self.clock() + 3.0
        still_since = None
        while self.clock() < until:
            with self.lock:
                if self.mode != "transition":
                    raise ControlConflict("Mode transition was stopped")
                if self.hardware.stationary():
                    still_since = self.clock() if still_since is None else still_since
                    if self.clock() - still_since >= 0.15:
                        if mode == "prepare_arms":
                            try:
                                self.hardware.prepare_arms()
                            except Exception as exc:
                                self._hold(str(exc))
                                raise
                        self.mode = "body" if mode == "body" else "arm_ready"
                        self._last_body_time = self.clock()
                        self.deadline = self.clock() + (self.TIMEOUT if mode == "body" else 5.0)
                        return self.state()
                else:
                    still_since = None
            time.sleep(self.PERIOD)
        with self.lock:
            if self.mode != "transition":
                raise ControlConflict("Mode transition was stopped")
            self._hold("Robot did not stop within 3 seconds")
        raise ControlConflict("Robot did not stop within 3 seconds")

    def enable_arms(self):
        with self.lock:
            if self.mode != "arm_ready" or self.clock() >= self.deadline:
                raise ControlConflict("Prepare arms and reanchor before enabling")
            if not self.hardware.stationary():
                raise ControlConflict("Robot is still moving")
            self.mode = "arms"
            self.deadline = float("inf")
            self.hardware.start_arms()
            return self.state()

    def accept_arm(self, side):
        # The caller holds its arm's command lock. Mode transitions close
        # admission before draining those locks and clearing buffered targets.
        if self.mode != "arms":
            raise ControlConflict("Arm commands are blocked in this mode")

    def set_command(self, values, shoulder_target=None):
        values = np.asarray(values, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("Expected three finite base velocity values")
        limits = np.asarray(self.hardware.limits()["command_velocity"][:3])
        if np.any(np.abs(values) > limits):
            raise ValueError("Command exceeds configured velocity limits")
        with self.lock:
            if self.mode != "body" or self.clock() >= self.deadline:
                raise ControlConflict("Body control is not active")
            if shoulder_target is not None:
                self.hardware.set_shoulder_target(shoulder_target)
            self.command = values
            self.deadline = self.clock() + self.TIMEOUT

    def tick(self):
        with self.lock:
            try:
                now = self.clock()
                if now >= self.deadline:
                    self._hold("Control input expired; select a mode to resume")
                if self.mode == "body":
                    dt = now - self._last_body_time
                    self._last_body_time = now
                    if dt > 0:
                        self.hardware.move_body(self.command, dt)
                elif self.mode in ("hold", "transition", "arm_ready"):
                    self.hardware.maintain_hold()
            except Exception as exc:
                # A fault enters hold. A new explicit mode selection is required.
                self._hold(str(exc))
                raise
