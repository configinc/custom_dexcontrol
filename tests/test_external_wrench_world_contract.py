from types import SimpleNamespace

import numpy as np

from dexcontrol.core.robotenv_vega.server import VegaRobotEnvService


def _service() -> VegaRobotEnvService:
    service = VegaRobotEnvService.__new__(VegaRobotEnvService)
    service.robot_model = "vega_1"
    service.arm_side = "left"
    service.gripper_type = "default"
    service.frame_type = "vega_mobile_base"
    service.control_hz = 20
    service.R_robot_to_world = None
    return service


def test_observation_contract_uses_external_wrench_world_only() -> None:
    spec = _service().GetObservationSpec(None, None)

    assert "external_wrench_world" in spec.fields
    assert "wrench_state" not in spec.fields
    assert list(spec.fields["external_wrench_world"].shape) == [6]


def test_observation_preserves_external_world_wrench_values() -> None:
    expected = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
        "external_wrench_world": expected,
    }
    service = _service()
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123})
    )

    observation, timestamp_us = service._create_observation()

    assert timestamp_us == 123
    assert "wrench_state" not in observation
    assert list(observation["external_wrench_world"].float_array.values) == list(
        expected
    )
