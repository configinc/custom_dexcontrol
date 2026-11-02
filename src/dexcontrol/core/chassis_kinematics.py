"""Bounded steering for a wheel whose drive can reverse."""
import numpy as np


def bounded_wheel_control(vector, current_angle, limits):
    vector = np.asarray(vector, dtype=float)
    lower, upper = limits
    if vector.shape != (2,) or not np.all(np.isfinite([*vector, current_angle, lower, upper])) or lower >= upper:
        raise ValueError("Invalid wheel vector or steering limits")
    speed = float(np.linalg.norm(vector))
    if speed < 1e-6:
        return float(np.clip(0., lower, upper)), 0.
    base = float(-np.arctan2(vector[1], vector[0]))
    alternate = base + np.pi if base < 0 else base - np.pi
    candidates = [(angle, velocity) for angle, velocity in ((alternate, -speed), (base, speed))
                  if lower <= angle <= upper]
    if not candidates:
        raise ValueError("Requested wheel direction is outside steering limits")
    return min(candidates, key=lambda candidate: abs(candidate[0] - current_angle))
