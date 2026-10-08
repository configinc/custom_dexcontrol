import sys
import types
from types import SimpleNamespace

import numpy as np

# The production server imports the full robot/teleop stack at module import
# time.  These contract tests exercise observation serialization only, so keep
# hardware and HID libraries out of the test process.
robot_module = types.ModuleType("core.vega.robot")
robot_module.CommunicationFailedError = type("CommunicationFailedError", (Exception,), {})
robot_module.IKFailedError = type("IKFailedError", (Exception,), {})
robot_module.JointLimitExceededError = type("JointLimitExceededError", (Exception,), {})
robot_module.VegaRobot = object
sys.modules.setdefault("core.vega.robot", robot_module)

from dexcontrol.core.robotenv_vega.server import VegaRobotEnvService  # noqa: E402


def _service_with_state(extra):
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
        "wrench_state": np.zeros(6),
        **extra,
    }
    service = VegaRobotEnvService.__new__(VegaRobotEnvService)
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123})
    )
    service.R_robot_to_world = None
    return service


def test_observation_exposes_raw_sensor_and_shared_body_state():
    service = _service_with_state({
        "external_wrench_sensor_frame": np.arange(6, dtype=float),
        "torso_joint_positions": np.array([0.1, 0.2, 0.3]),
        "head_joint_positions": np.array([-0.1, -0.2, -0.3]),
    })

    observation, timestamp_us = service._create_observation()

    assert timestamp_us == 123
    assert list(observation["external_wrench_sensor_frame"].float_array.values) == list(range(6))
    assert list(observation["torso_joint_positions"].float_array.values) == [0.1, 0.2, 0.3]
    assert list(observation["head_joint_positions"].float_array.values) == [-0.1, -0.2, -0.3]


def test_observation_does_not_fabricate_optional_sensor_or_body_state():
    observation, _ = _service_with_state({})._create_observation()

    assert "external_wrench_sensor_frame" not in observation
    assert "torso_joint_positions" not in observation
    assert "head_joint_positions" not in observation
