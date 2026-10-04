"""Frame-explicit Vega external-wrench conversion."""

from __future__ import annotations

import numpy as np


def sensor_frame_wrench_to_wrist(
    wrench: np.ndarray, *, sensor_to_wrist_yaw_degrees: float
) -> np.ndarray:
    """Rotate a sensor-frame wrench into the company wrist convention.

    The configured angle is an active +Z yaw from sensor coordinates to wrist
    coordinates. Force and torque rotate identically and remain referenced at
    the physical F/T sensor measurement origin.
    """
    array = np.asarray(wrench, dtype=np.float64)
    yaw = float(sensor_to_wrist_yaw_degrees)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ValueError("wrench must contain exactly six finite values")
    if not np.isfinite(yaw):
        raise ValueError("sensor_to_wrist_yaw_degrees must be finite")

    angle = np.deg2rad(yaw)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return np.concatenate((rotation @ array[:3], rotation @ array[3:]))
