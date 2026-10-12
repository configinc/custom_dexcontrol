"""Frame-explicit Vega external-wrench conversion."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


VEGA_TCP_FROM_COMPANY_WRIST_YAW_DEGREES = 165.0


def vega_wrench_frame_rotations(
    cartesian_position: np.ndarray, *, gripper_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return producer-owned sensor-to-wrist/world rotations.

    The installed Vega F/T sensor is mechanically aligned with Vega TCP.
    The company wrist frame for RobotiQ is +165 degrees about TCP Z.  The
    Cartesian pose is already expressed in RobotEnv's configured world frame.
    """
    pose = np.asarray(cartesian_position, dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("cartesian_position must contain six finite values")
    if not str(gripper_type).lower().split("@", 1)[0].startswith("robotiq"):
        raise ValueError("Vega wrench frame contract is only validated for RobotiQ")
    tcp_from_wrist = Rotation.from_euler(
        "z", VEGA_TCP_FROM_COMPANY_WRIST_YAW_DEGREES, degrees=True
    ).as_matrix()
    world_from_sensor = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    wrist_from_sensor = tcp_from_wrist.T
    return wrist_from_sensor, world_from_sensor


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
