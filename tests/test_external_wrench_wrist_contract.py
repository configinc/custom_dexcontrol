from types import SimpleNamespace

import numpy as np

from dexcontrol.core.arm import ArmWrenchSensor
from dexcontrol.core.component import RobotComponent
from dexcontrol.core.robotenv_vega.server import (
    VegaRobotEnvService,
    sensor_frame_wrench_to_wrist,
)
from dexcontrol.core.vega.robot import VegaRobot


def _service() -> VegaRobotEnvService:
    service = VegaRobotEnvService.__new__(VegaRobotEnvService)
    service.robot_model = "vega_1"
    service.arm_side = "left"
    service.gripper_type = "default"
    service.frame_type = "vega_mobile_base"
    service.control_hz = 20
    service.R_robot_to_world = None
    service.external_wrench_sensor_to_wrist_yaw_degrees = None
    return service


def test_observation_contract_uses_external_wrench_wrist_only() -> None:
    spec = _service().GetObservationSpec(None, None)

    assert "external_wrench_wrist" in spec.fields
    assert "external_wrench_world" not in spec.fields
    assert "wrench_state" not in spec.fields
    assert list(spec.fields["external_wrench_wrist"].shape) == [6]
    assert spec.fields["external_wrench_wrist_reference_point"].dtype == "string"


def test_sensor_boundary_preserves_source_wrench_order_and_sign(monkeypatch) -> None:
    upstream_wrench_on_robot = [1.0, -2.0, 3.0, -0.1, 0.2, -0.3]
    monkeypatch.setattr(
        RobotComponent,
        "get_state",
        lambda _sensor: {"wrench": upstream_wrench_on_robot},
    )
    sensor = ArmWrenchSensor.__new__(ArmWrenchSensor)

    np.testing.assert_array_equal(
        sensor.get_wrench_state(),
        np.asarray(upstream_wrench_on_robot, dtype=np.float32),
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


def test_observation_publishes_configured_external_wrist_wrench() -> None:
    source = np.array([1.0, 0.0, 3.0, 2.0, 0.0, 4.0])
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
        "external_wrench_sensor_frame": source,
    }
    service = _service()
    service.external_wrench_sensor_to_wrist_yaw_degrees = 90.0
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123})
    )

    observation, timestamp_us = service._create_observation()

    assert timestamp_us == 123
    assert "wrench_state" not in observation
    np.testing.assert_allclose(
        observation["external_wrench_wrist"].float_array.values,
        [0.0, 1.0, 3.0, 0.0, 2.0, 4.0],
        atol=1e-12,
    )
    assert (
        observation["external_wrench_wrist_reference_point"].string_value
        == "physical F/T sensor measurement origin"
    )


def test_robot_state_keeps_source_wrench_internal_until_server_conversion() -> None:
    expected = np.array([4.0, -5.0, 6.0, -0.4, 0.5, -0.6])
    robot = VegaRobot.__new__(VegaRobot)
    robot.arm = SimpleNamespace(
        get_joint_pos=lambda: np.zeros(7),
        get_joint_vel=lambda: np.zeros(7),
        get_joint_torque=lambda: np.zeros(7),
        get_timestamp_ns=lambda: 123_456_000,
        wrench_sensor=SimpleNamespace(get_wrench_state=lambda: expected.copy()),
    )
    robot.hand = None
    robot._get_cartesian_pose = lambda *, joint_positions: np.zeros(6)
    robot._prev_controller_latency_ms = 0.0
    robot._prev_command_successful = True
    robot._prev_gripper_command_successful = True

    state, _ = robot.get_robot_state()

    assert "wrench_state" not in state
    assert "external_wrench_world" not in state
    assert "external_wrench_wrist" not in state
    np.testing.assert_array_equal(state["external_wrench_sensor_frame"], expected)


def test_unconfigured_yaw_omits_canonical_wrist_wrench() -> None:
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
        "external_wrench_sensor_frame": np.ones(6),
    }
    service = _service()
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123})
    )

    observation, _ = service._create_observation()

    assert "external_wrench_wrist" not in observation
    assert "external_wrench_wrist_reference_point" not in observation


def test_malformed_source_wrench_omits_optional_wrist_wrench() -> None:
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
        "external_wrench_sensor_frame": np.ones(5),
    }
    service = _service()
    service.external_wrench_sensor_to_wrist_yaw_degrees = 0.0
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123})
    )

    observation, _ = service._create_observation()

    assert "external_wrench_wrist" not in observation
    assert "external_wrench_wrist_reference_point" not in observation


def test_sensor_absence_omits_optional_external_wrench() -> None:
    robot = VegaRobot.__new__(VegaRobot)
    robot.arm = SimpleNamespace(
        get_joint_pos=lambda: np.zeros(7),
        get_joint_vel=lambda: np.zeros(7),
        get_joint_torque=lambda: np.zeros(7),
        get_timestamp_ns=lambda: 123_456_000,
        wrench_sensor=None,
    )
    robot.hand = None
    robot._get_cartesian_pose = lambda *, joint_positions: np.zeros(6)
    robot._prev_controller_latency_ms = 0.0
    robot._prev_command_successful = True
    robot._prev_gripper_command_successful = True

    state, _ = robot.get_robot_state()
    assert "external_wrench_sensor_frame" not in state

    service = _service()
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123})
    )
    observation, _ = service._create_observation()
    assert "external_wrench_wrist" not in observation
    assert "external_wrench_wrist_reference_point" not in observation
