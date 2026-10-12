import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexcontrol.core.robotenv_vega.wrench_contract import (
    sensor_frame_wrench_to_wrist,
    vega_wrench_frame_rotations,
)


def test_sensor_to_wrist_yaw_rotates_force_and_torque() -> None:
    np.testing.assert_allclose(
        sensor_frame_wrench_to_wrist(
            np.array([1.0, 0.0, 3.0, 2.0, 0.0, 4.0]),
            sensor_to_wrist_yaw_degrees=90.0,
        ),
        np.array([0.0, 1.0, 3.0, 0.0, 2.0, 4.0]),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "wrench",
    ([1.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0, float("nan")]),
)
def test_sensor_to_wrist_rejects_malformed_wrench(wrench) -> None:
    with pytest.raises(ValueError, match="six finite values"):
        sensor_frame_wrench_to_wrist(
            np.asarray(wrench), sensor_to_wrist_yaw_degrees=0.0
        )


def test_sensor_to_wrist_rejects_nonfinite_yaw() -> None:
    with pytest.raises(ValueError, match="yaw_degrees must be finite"):
        sensor_frame_wrench_to_wrist(
            np.ones(6), sensor_to_wrist_yaw_degrees=float("nan")
        )


def test_vega_robot_server_owns_company_wrist_and_world_rotations() -> None:
    pose = np.array([0.1, -0.2, 0.3, 0.2, -0.1, 0.4])
    wrist_from_sensor, world_from_sensor = vega_wrench_frame_rotations(
        pose, gripper_type="robotiq"
    )

    expected_world = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    expected_wrist = Rotation.from_euler("z", -165.0, degrees=True).as_matrix()
    np.testing.assert_allclose(world_from_sensor, expected_world, atol=1e-12)
    np.testing.assert_allclose(wrist_from_sensor, expected_wrist, atol=1e-12)
