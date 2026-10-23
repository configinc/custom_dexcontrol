"""Absolute Cartesian target -> bounded per-step Cartesian delta.

Kept free of robot imports so it can be unit-tested without the SDK.
Same contract as the Franka server's ``target_cartesian``: each step moves
toward the target by at most ``max_linear_delta`` / ``max_rotation_delta``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > limit > 0.0:
        return vector * (limit / norm)
    return vector


def absolute_target_to_delta(
    target_pose: np.ndarray,
    current_pose: np.ndarray,
    max_linear_delta: float,
    max_rotation_delta: float,
) -> np.ndarray:
    """``[x, y, z, roll, pitch, yaw]`` target and current pose -> 6D delta.

    Position delta is the difference, norm-clipped. Rotation delta is the XYZ
    Euler of the relative rotation ``R_target R_current^T`` (left-multiplied
    onto the current orientation, the convention ``cartesian_delta`` uses),
    clipped on its rotation-vector norm so the bound is a physical angle.
    """
    target = np.asarray(target_pose, dtype=np.float64).reshape(-1)
    current = np.asarray(current_pose, dtype=np.float64).reshape(-1)
    if target.shape != (6,) or current.shape != (6,):
        raise ValueError("target_pose and current_pose must each have 6 values")
    if not (np.all(np.isfinite(target)) and np.all(np.isfinite(current))):
        raise ValueError("target_pose and current_pose must be finite")
    if max_linear_delta <= 0.0 or max_rotation_delta <= 0.0:
        raise ValueError("max_linear_delta and max_rotation_delta must be positive")

    delta_xyz = _clip_norm(target[:3] - current[:3], max_linear_delta)
    relative = R.from_euler("xyz", target[3:6]) * R.from_euler("xyz", current[3:6]).inv()
    delta_rotvec = _clip_norm(relative.as_rotvec(), max_rotation_delta)
    delta_rpy = R.from_rotvec(delta_rotvec).as_euler("xyz")
    return np.concatenate([delta_xyz, delta_rpy])
