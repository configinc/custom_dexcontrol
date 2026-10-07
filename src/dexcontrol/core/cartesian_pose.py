"""Cartesian pose composition helpers shared by Vega action logging paths."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def add_poses(delta, source) -> np.ndarray:
    """Left-compose an xyz/rpy pose delta with an absolute xyz/rpy pose."""
    delta_arr = np.asarray(delta, dtype=np.float64)
    source_arr = np.asarray(source, dtype=np.float64)
    if delta_arr.shape != (6,) or source_arr.shape != (6,):
        raise ValueError(
            "delta and source must both be 6D xyz/rpy poses, got "
            f"{delta_arr.shape} and {source_arr.shape}"
        )

    result = source_arr.copy()
    result[:3] += delta_arr[:3]
    result[3:6] = (
        R.from_euler("xyz", delta_arr[3:6]) * R.from_euler("xyz", source_arr[3:6])
    ).as_euler("xyz")
    return result
