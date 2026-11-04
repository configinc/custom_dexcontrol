"""Continuous position-offset torque relief at the managed output rate."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArmTorqueConfig:
    torque_constant_nm_per_a: list[float]
    current_sign: list[float]
    torque_limit_nm: list[float]
    gain_rad_per_nm_s: list[float]
    max_relief_speed_rad_s: list[float]
    recovery_speed_rad_s: list[float]

    def __post_init__(self):
        for name, values in vars(self).items():
            value = np.asarray(values)
            if (
                value.shape != (7,)
                or value.dtype.kind not in "fi"
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Arm torque {name} requires seven finite numbers")
            if name == "current_sign":
                if not np.isin(value, (-1, 1)).all():
                    raise ValueError("Arm torque current_sign must be +1 or -1")
            elif np.any(value <= 0):
                raise ValueError(f"Arm torque {name} must be positive")


@dataclass(frozen=True)
class TorqueLimitConfig:
    feedback_timeout_s: float
    arms: dict[str, ArmTorqueConfig]

    def __post_init__(self):
        if (
            isinstance(self.feedback_timeout_s, bool)
            or not isinstance(self.feedback_timeout_s, (int, float))
            or not 0 < self.feedback_timeout_s < float("inf")
        ):
            raise ValueError(
                "Arm torque feedback_timeout_s must be positive and finite"
            )
        if set(self.arms) != {"left_arm", "right_arm"}:
            raise ValueError("Arm torque configuration requires left_arm and right_arm")


class ContinuousTorqueOffset:
    """One independent offset per joint; user position changes and FF pass through."""

    def __init__(self, config):
        self.settings = {
            key: np.asarray(value, dtype=float) for key, value in vars(config).items()
        }
        self.offset = np.zeros(7)

    def reset(self):
        self.offset[:] = 0

    def update(self, position, torque, dt, limits=None):
        c, previous = self.settings, self.offset.copy()
        headroom = c["torque_limit_nm"] + np.sign(previous) * torque
        recovery_speed = np.minimum(
            c["recovery_speed_rad_s"],
            c["gain_rad_per_nm_s"] * np.maximum(headroom, 0),
        )
        recovered = previous - np.sign(previous) * np.minimum(
            np.abs(previous), recovery_speed * dt
        )
        relief_speed = -np.sign(torque) * np.minimum(
            c["max_relief_speed_rad_s"],
            c["gain_rad_per_nm_s"]
            * np.maximum(np.abs(torque) - c["torque_limit_nm"], 0),
        )
        self.offset = recovered + relief_speed * dt
        target = position + self.offset
        if limits is not None:
            bounded = np.clip(target, limits[:, 0], limits[:, 1])
            saturated = bounded != target
            # Preserve the SDK range clamp and do not accumulate relief beyond it.
            self.offset[saturated] = bounded[saturated] - position[saturated]
            target = bounded
        return target


class ArmTorqueLimiter:
    """Correct arm candidates before collision/stability approval; no robot I/O waits."""

    def __init__(self, devices, config, enabled, period, clock):
        self.devices, self.config = devices, config
        self.enabled, self.period, self.clock = enabled, period, clock
        self.limiters = {
            name: ContinuousTorqueOffset(settings)
            for name, settings in config.arms.items()
        }
        if enabled:
            for name in self.limiters:
                if name not in devices or len(devices[name].joint_name) != 7:
                    raise ValueError(
                        f"Arm torque limiting requires seven joints for {name}"
                    )
        self.reset()

    def reset(self):
        for limiter in self.limiters.values():
            limiter.reset()
        self.targets, self.feedback, self.details = {}, {}, {}
        self.state = "idle" if self.enabled else "disabled"

    def suspend(self):
        """A model hold supersedes torque motion; do not integrate unsent relief."""
        for limiter in self.limiters.values():
            limiter.reset()
        if self.enabled:
            self.state = "model_hold"

    def _current(self, name, now):
        sample = self.devices[name].get_state()
        current = np.asarray(sample.get("cur"), dtype=float)
        stamp = sample.get("timestamp_ns")
        if (
            current.shape != (7,)
            or not np.isfinite(current).all()
            or isinstance(stamp, bool)
            or not isinstance(stamp, (int, np.integer))
            or stamp < 0
        ):
            raise ValueError(f"Invalid arm torque current feedback for {name}")
        previous, changed_at = self.feedback.get(name, (stamp, now))
        if stamp < previous:
            raise ValueError(
                f"Arm torque feedback timestamp moved backwards for {name}"
            )
        if stamp != previous:
            changed_at = now
        self.feedback[name] = (stamp, changed_at)
        if now - changed_at > self.config.feedback_timeout_s:
            raise TimeoutError(f"Arm torque current feedback expired for {name}")
        return current

    def filter(self, commands):
        if not self.enabled:
            return []
        now = self.clock()
        violations = []
        for name, limiter in self.limiters.items():
            if name in commands:
                if "pos" not in commands[name]:
                    raise ValueError(
                        f"Arm torque limiting requires a position target for {name}"
                    )
                # Keep raw targets separate from corrected/approved output. Sparse
                # action/body commands still receive relief on every output tick.
                self.targets[name] = {
                    key: value.copy() for key, value in commands[name].items()
                }
            if name not in self.targets:
                continue
            current = self._current(name, now)
            settings = limiter.settings
            torque = (
                current
                * settings["torque_constant_nm_per_a"]
                * settings["current_sign"]
            )
            if not np.isfinite(torque).all():
                raise ValueError(f"Invalid estimated arm torque for {name}")
            raw = self.targets[name]
            corrected = limiter.update(
                raw["pos"], torque, self.period, self.devices[name].joint_pos_limit
            )
            # Retain the exact existing velocity/feedforward and other fields.
            commands[name] = {**raw, "pos": corrected}
            over = np.abs(torque) > settings["torque_limit_nm"]
            self.details[name] = {
                "component": name,
                "joint_names": list(self.devices[name].joint_name),
                "current_a": current.tolist(),
                "estimated_torque_nm": torque.tolist(),
                "torque_limit_nm": settings["torque_limit_nm"].tolist(),
                "requested_position_rad": raw["pos"].tolist(),
                "candidate_position_rad": corrected.tolist(),
                "candidate_offset_rad": limiter.offset.tolist(),
                "over_limit": over.tolist(),
            }
            if np.any(over):
                violations.append(
                    f"{name} torque exceeds limit at joints {(np.flatnonzero(over) + 1).tolist()}"
                )
        self.state = "active" if self.targets else "idle"
        return violations

    def record_output(self, commands):
        for name, details in self.details.items():
            if name in commands and "pos" in commands[name]:
                details["output_position_rad"] = commands[name]["pos"].tolist()

    def snapshot(self):
        return {"state": self.state, "arms": list(self.details.values())}
