"""Frame-explicit Vega external-wrench conversion."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


VEGA_TCP_FROM_COMPANY_WRIST_YAW_DEGREES = 165.0

# RQ_Vega_adapter_v2 places the RobotiQ mounting plane 18.5 mm farther along
# the physical sensor/link-8 +Z tool axis. Cartesian control remains on the
# zero-translation L_ee/R_ee frame; this offset changes only the wrench origin.
VEGA_COMPANY_WRIST_ORIGIN_FROM_SENSOR_M = np.array(
    [0.0, 0.0, 0.0185], dtype=np.float64
)

# Dexmate's Vega URDF defines the physical F/T measurement axes on arm link 8.
# Cartesian control/recording remains on L_ee/R_ee, whose fixed joint is yawed
# relative to link 8 by -90 degrees on the left arm and +90 degrees on the
# right arm.  These are sensor-to-TCP mounting transforms, not the independent
# TCP-to-company-wrist convention above.
VEGA_LINK8_FROM_TCP_YAW_DEGREES = {
    "left": -90.0,
    "right": 90.0,
}


def vega_wrench_frame_rotations(
    cartesian_position: np.ndarray, *, arm_side: str, gripper_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return producer-owned sensor-to-wrist/world rotations.

    Cartesian pose is the recorded/control L_ee or R_ee frame.  The physical
    F/T sensor axes follow Dexmate's corresponding arm link-8 frame.  The
    company wrist frame for RobotiQ remains +165 degrees about TCP Z.
    """
    pose = np.asarray(cartesian_position, dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("cartesian_position must contain six finite values")
    if not str(gripper_type).lower().split("@", 1)[0].startswith("robotiq"):
        raise ValueError("Vega wrench frame contract is only validated for RobotiQ")
    side = str(arm_side).strip().lower()
    if side not in VEGA_LINK8_FROM_TCP_YAW_DEGREES:
        raise ValueError("arm_side must be 'left' or 'right'")
    link8_from_tcp = Rotation.from_euler(
        "z", VEGA_LINK8_FROM_TCP_YAW_DEGREES[side], degrees=True
    ).as_matrix()
    tcp_from_wrist = Rotation.from_euler(
        "z", VEGA_TCP_FROM_COMPANY_WRIST_YAW_DEGREES, degrees=True
    ).as_matrix()
    world_from_tcp = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    tcp_from_sensor = link8_from_tcp.T
    world_from_sensor = world_from_tcp @ tcp_from_sensor
    wrist_from_sensor = tcp_from_wrist.T @ tcp_from_sensor
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


def shift_wrench_reference(
    wrench: np.ndarray, *, target_from_source_translation: np.ndarray
) -> np.ndarray:
    """Move a wrench to a new origin without changing its coordinate axes."""
    array = np.asarray(wrench, dtype=np.float64)
    translation = np.asarray(target_from_source_translation, dtype=np.float64)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ValueError("wrench must contain exactly six finite values")
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError(
            "target_from_source_translation must contain three finite values"
        )
    return np.concatenate(
        (array[:3], array[3:] - np.cross(translation, array[:3]))
    )
