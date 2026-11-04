"""Asynchronous collision/stability approval."""

from time import monotonic

from .motion_guard import MotionGuard


class MotionSafety:
    def __init__(
        self,
        devices,
        config,
        collision_model=None,
        clock=monotonic,
        worker=None,
    ):
        self.devices = devices
        self.config = config
        self.clock = clock
        self.collision = collision_model
        enabled = any(
            config.safety[name].enabled
            for name in ("self_collision", "static_stability")
        )
        if enabled and collision_model is None:
            raise ValueError("Enabled model safety requires a robot model")
        names = [joint for device in devices.values() for joint in device.joint_name]
        if len(names) != len(set(names)):
            raise ValueError("Managed components have duplicate joint names")
        self.guard = (
            MotionGuard(devices, collision_model, config, clock, worker)
            if enabled
            else None
        )

    def reset(self):
        if self.guard:
            self.guard.reset()

    def close(self):
        if self.guard:
            self.guard.close()

    def snapshot(self, check=None):
        return self.guard.snapshot(check) if self.guard else {"state": "disabled"}

    def observe(self, owner):
        if self.guard:
            self.guard.observe(owner)

    def filter(self, commands, effects, owner, arm_mode=False):
        if self.guard:
            self.guard.filter(commands, effects, owner, arm_mode)
            return self.guard.stop_event
        return ""
