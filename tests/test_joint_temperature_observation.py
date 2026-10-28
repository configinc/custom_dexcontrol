"""Arm motor temperatures are an optional, additive observation field."""
import logging
import math
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np

# Keep this contract test independent of the optional DualSense/hidapi stack.
_base_arm_teleop = ModuleType("base_arm_teleop")
_base_arm_teleop.BaseIKController = object
sys.modules.setdefault("base_arm_teleop", _base_arm_teleop)

from dexcontrol.core.robotenv_vega.server import VegaRobotEnvService  # noqa: E402
from dexcontrol.core.robot_env_client import RobotEnvClient  # noqa: E402
from dexcontrol.core.vega.robot import VegaRobot  # noqa: E402
from dexcontrol.exceptions import ServiceUnavailableError  # noqa: E402

NAMES = [f"L_arm_j{i}" for i in range(1, 8)]


class FakeTemperatureSensor:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def get_temperatures(self):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _robot(sensor=None) -> VegaRobot:
    robot = VegaRobot.__new__(VegaRobot)
    robot.arm = SimpleNamespace(
        get_joint_pos=lambda: np.zeros(7),
        get_joint_vel=lambda: np.zeros(7),
        get_joint_torque=lambda: np.zeros(7),
        get_timestamp_ns=lambda: 123_000,
        wrench_sensor=None,
        temperature_sensor=sensor,
    )
    robot.robot = SimpleNamespace()
    robot.hand = None
    robot._arm_joint_names = NAMES
    robot._temperature_sensor = sensor
    robot._joint_temperatures_available = None
    robot._prev_controller_latency_ms = 0.0
    robot._prev_command_successful = True
    robot._prev_gripper_command_successful = True
    robot._get_cartesian_pose = lambda *, joint_positions: np.zeros(6)
    return robot


def _service(robot) -> VegaRobotEnvService:
    service = VegaRobotEnvService.__new__(VegaRobotEnvService)
    service._shutdown_started = threading.Event()
    service.robot_model = "vega_1"
    service.arm_side = "left"
    service.gripper_type = "robotiq"
    service.frame_type = "vega_mobile_base"
    service.control_hz = 20
    service.R_robot_to_world = None
    service.external_wrench_sensor_to_wrist_yaw_degrees = None
    service._force_calibration_applier_lock = threading.RLock()
    service._force_calibration_applier = None
    service._force_calibration_expires_at_timestamp_us = 0
    service._robot = robot
    return service


def test_no_sensor_keeps_state_and_observation_keys_unchanged() -> None:
    robot = _robot(sensor=None)
    state, _ = robot.get_robot_state()
    observation, _ = _service(robot)._create_observation()
    spec = _service(robot).GetObservationSpec(None, None)

    assert "joint_temperatures" not in state
    assert "joint_temperatures" not in observation
    assert "joint_temperatures" not in spec.fields


def test_full_data_follows_joint_name_order_not_message_order() -> None:
    temps = {name: {"motor": 30.0 + i} for i, name in enumerate(NAMES)}
    shuffled = dict(reversed(list(temps.items())))
    robot = _robot(FakeTemperatureSensor(shuffled))

    state, _ = robot.get_robot_state()
    observation, _ = _service(robot)._create_observation()
    spec = _service(robot).GetObservationSpec(None, None)

    expected = [30.0 + i for i in range(7)]
    assert state["joint_temperatures"] == expected
    assert list(observation["joint_temperatures"].float_array.values) == expected
    assert list(spec.fields["joint_temperatures"].shape) == [7]
    assert spec.fields["joint_temperatures"].required is False


def test_missing_joint_is_nan_and_non_motor_source_falls_back() -> None:
    temps = {name: {"motor": 40.0} for name in NAMES}
    del temps["L_arm_j3"]
    temps["L_arm_j5"] = {"driver": 45.0}
    temps["L_arm_j6"] = {}
    state, _ = _robot(FakeTemperatureSensor(temps)).get_robot_state()

    values = state["joint_temperatures"]
    assert len(values) == 7
    assert math.isnan(values[2])
    assert values[4] == 45.0
    assert math.isnan(values[5])
    assert values[0] == values[6] == 40.0


def test_unavailable_data_omits_key_without_raising_or_per_step_logs(caplog) -> None:
    sensor = FakeTemperatureSensor(ServiceUnavailableError("no data yet"))
    robot = _robot(sensor)

    with caplog.at_level(logging.WARNING, logger="robotenv_vega"):
        for _ in range(50):
            state, _ = robot.get_robot_state()
            assert "joint_temperatures" not in state
        sensor.result = {}
        observation, _ = _service(robot)._create_observation()

    assert "joint_temperatures" not in observation
    assert sensor.calls == 51
    warnings = [r for r in caplog.records if "temperatures unavailable" in r.message]
    assert len(warnings) == 1


def test_recovery_after_unavailable_adds_key() -> None:
    sensor = FakeTemperatureSensor(ServiceUnavailableError("no data yet"))
    robot = _robot(sensor)
    assert "joint_temperatures" not in robot.get_robot_state()[0]

    sensor.result = {name: {"motor": 33.0} for name in NAMES}
    assert robot.get_robot_state()[0]["joint_temperatures"] == [33.0] * 7


def test_local_client_passes_temperatures_through() -> None:
    client = RobotEnvClient.__new__(RobotEnvClient)
    with_key = client._build_robot_state({"joint_temperatures": np.arange(7.0)})
    without_key = client._build_robot_state({})

    assert with_key["joint_temperatures"] == list(range(7))
    assert "joint_temperatures" not in without_key
