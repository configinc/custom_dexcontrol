import inspect
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np

# Keep this contract test independent of the optional DualSense/hidapi stack.
_base_arm_teleop = ModuleType("base_arm_teleop")
_base_arm_teleop.BaseIKController = object
sys.modules.setdefault("base_arm_teleop", _base_arm_teleop)

from dexcontrol.core.arm import ArmWrenchSensor  # noqa: E402
from dexcontrol.core.component import RobotComponent  # noqa: E402
import dexcontrol.core.robotenv_vega.server as server_module  # noqa: E402
from dexcontrol.core.robotenv_vega.server import VegaRobotEnvService  # noqa: E402
from dexcontrol.core.robot_env_client import RobotEnvClient  # noqa: E402
from dexcontrol.core.robotenv_vega.wrench_contract import (  # noqa: E402
    sensor_frame_wrench_to_wrist,
)
from dexcontrol.core.vega.robot import VegaRobot  # noqa: E402
from dexcontrol.exceptions import ServiceUnavailableError  # noqa: E402


def _service() -> VegaRobotEnvService:
    service = VegaRobotEnvService.__new__(VegaRobotEnvService)
    service._shutdown_started = threading.Event()
    service._transport_rpc_condition = threading.Condition()
    service._active_transport_rpcs = 0
    service.robot_model = "vega_1"
    service.arm_side = "left"
    service.gripper_type = "robotiq"
    service._robot = SimpleNamespace(
        _hand_dof=lambda: 1,
        hand=SimpleNamespace(),
    )
    service.frame_type = "vega_mobile_base"
    service.control_hz = 20
    service.R_robot_to_world = None
    service.external_wrench_sensor_to_wrist_yaw_degrees = None
    return service


def test_robot_server_does_not_default_a_canonical_wrist_transform() -> None:
    signature = inspect.signature(VegaRobotEnvService)

    assert (
        signature.parameters[
            "external_wrench_sensor_to_wrist_yaw_degrees"
        ].default
        is None
    )


def test_observation_contract_exposes_raw_and_optional_calibrated_wrenches() -> None:
    spec = _service().GetObservationSpec(None, None)

    assert "external_wrench_wrist" in spec.fields
    assert "external_wrench_sensor_raw" in spec.fields
    assert "external_wrench_world" in spec.fields
    assert "wrench_state" not in spec.fields
    assert list(spec.fields["external_wrench_sensor_raw"].shape) == [6]
    assert spec.fields["external_wrench_sensor_raw_reference_point"].dtype == "string"
    assert spec.fields["external_wrench_sensor_raw_timestamp_ns"].dtype == "int64"
    assert list(spec.fields["external_wrench_wrist_from_sensor_rotation"].shape) == [9]
    assert list(spec.fields["external_wrench_world_from_sensor_rotation"].shape) == [9]
    assert spec.fields["external_wrench_wrist"].required is False
    assert spec.fields["external_wrench_world"].required is False
    assert spec.metadata["force_signal_provider_kind"] == "wrist_force_torque"
    assert spec.metadata["force_signal_provider_schema_version"] == "2"


def test_observation_spec_transport_access_is_tracked_until_complete() -> None:
    service = _service()
    entered = threading.Event()
    release = threading.Event()

    def blocking_hand_dof():
        entered.set()
        assert release.wait(timeout=1.0)
        return 1

    service._robot._hand_dof = blocking_hand_dof
    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault("spec", service.GetObservationSpec(None, None))
    )
    thread.start()
    assert entered.wait(timeout=1.0)
    assert service._active_transport_rpcs == 1
    service._shutdown_started.set()
    release.set()
    thread.join(timeout=1.0)

    assert "spec" in result
    assert service._active_transport_rpcs == 0


def test_sensor_boundary_preserves_source_wrench_order_and_sign(monkeypatch) -> None:
    upstream_wrench_on_robot = [1.0, -2.0, 3.0, -0.1, 0.2, -0.3]
    monkeypatch.setattr(
        RobotComponent,
        "get_state",
        lambda _sensor: {"wrench": upstream_wrench_on_robot, "timestamp_ns": 123},
    )
    sensor = ArmWrenchSensor.__new__(ArmWrenchSensor)

    np.testing.assert_array_equal(
        sensor.get_wrench_state(),
        np.asarray(upstream_wrench_on_robot, dtype=np.float32),
    )
    _, timestamp_ns = sensor.get_wrench_sample()
    assert timestamp_ns == 123


def test_local_robotenv_client_preserves_external_wrench_contract_fields() -> None:
    client = RobotEnvClient.__new__(RobotEnvClient)
    raw = np.arange(6.0)
    calibrated = np.arange(6.0) + 10.0
    state = client._build_robot_state(
        {
            "external_wrench_sensor_raw": raw,
            "external_wrench_sensor_raw_timestamp_ns": 123_000,
            "external_wrench_sensor_raw_reference_point": "sensor-origin",
            "external_wrench_world": calibrated,
            "external_wrench_world_reference_point": "tcp-anchor",
        }
    )

    assert state["external_wrench_sensor_raw"] == raw.tolist()
    assert state["external_wrench_sensor_raw_timestamp_ns"] == 123_000
    assert state["external_wrench_sensor_raw_reference_point"] == "sensor-origin"
    assert state["external_wrench_world"] == calibrated.tolist()
    assert state["external_wrench_world_reference_point"] == "tcp-anchor"


def test_sensor_to_wrist_yaw_rotates_force_and_torque() -> None:
    np.testing.assert_allclose(
        sensor_frame_wrench_to_wrist(
            np.array([1.0, 0.0, 3.0, 2.0, 0.0, 4.0]),
            sensor_to_wrist_yaw_degrees=90.0,
        ),
        np.array([0.0, 1.0, 3.0, 0.0, 2.0, 4.0]),
        atol=1e-12,
    )


def test_observation_keeps_configured_rotation_out_of_canonical_fields() -> None:
    source = np.array([1.0, 0.0, 3.0, 2.0, 0.0, 4.0])
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
    }
    service = _service()
    service.external_wrench_sensor_to_wrist_yaw_degrees = 90.0
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123}),
        arm=SimpleNamespace(
            wrench_sensor=SimpleNamespace(
                get_wrench_sample=lambda: (source.copy(), 123_000)
            )
        ),
    )

    observation, timestamp_us = service._create_observation()

    assert timestamp_us == 123
    assert "wrench_state" not in observation
    np.testing.assert_allclose(
        observation["external_wrench_sensor_raw"].float_array.values,
        source,
        atol=1e-12,
    )
    assert observation["external_wrench_sensor_raw_timestamp_ns"].int_value == 123_000
    assert (
        observation["external_wrench_sensor_raw_reference_point"].string_value
        == "physical F/T sensor measurement origin"
    )
    assert "external_wrench_wrist" not in observation
    assert "external_wrench_wrist_reference_point" not in observation
    wrist_from_sensor = np.asarray(
        observation["external_wrench_wrist_from_sensor_rotation"].float_array.values
    ).reshape(3, 3)
    world_from_sensor = np.asarray(
        observation["external_wrench_world_from_sensor_rotation"].float_array.values
    ).reshape(3, 3)
    expected = np.array(
        [
            [np.cos(np.deg2rad(165.0)), np.sin(np.deg2rad(165.0)), 0.0],
            [-np.sin(np.deg2rad(165.0)), np.cos(np.deg2rad(165.0)), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(wrist_from_sensor, expected, atol=1e-12)
    np.testing.assert_allclose(world_from_sensor, np.eye(3), atol=1e-12)


def test_observation_publishes_both_calibrated_frames_only_with_active_applier() -> None:
    source = np.arange(6.0)
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
    }

    class Applier:
        def measured_sensor_raw_to_calibrated_wrist(self, *args, **kwargs):
            assert service._force_calibration_applier_lock.locked()
            assert kwargs["reference_point"] == "physical F/T sensor measurement origin"
            return np.ones(6)

        def measured_sensor_raw_to_calibrated_world(self, *args, **kwargs):
            assert service._force_calibration_applier_lock.locked()
            return np.full(6, 2.0)

    service = _service()
    service._force_calibration_applier_lock = threading.Lock()
    service._force_calibration_applier = Applier()
    service._force_calibration_expires_at_timestamp_us = 2**63 - 1
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123}),
        arm=SimpleNamespace(
            wrench_sensor=SimpleNamespace(
                get_wrench_sample=lambda: (source.copy(), 123_000)
            )
        ),
    )

    observation, _ = service._create_observation()

    np.testing.assert_array_equal(
        observation["external_wrench_wrist"].float_array.values, np.ones(6)
    )
    np.testing.assert_array_equal(
        observation["external_wrench_world"].float_array.values, np.full(6, 2.0)
    )
    assert (
        observation["external_wrench_wrist_reference_point"].string_value
        == "configured TCP anchor (rotation only)"
    )

    service._force_calibration_expires_at_timestamp_us = 1
    expired_observation, _ = service._create_observation()
    assert "external_wrench_wrist" not in expired_observation
    assert "external_wrench_world" not in expired_observation
    assert service._force_calibration_applier is None


def test_misaligned_sample_keeps_raw_but_omits_pose_dependent_wrench() -> None:
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
    }
    service = _service()
    service._force_calibration_applier_lock = threading.Lock()
    service._force_calibration_applier = SimpleNamespace()
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 1_000_000}),
        arm=SimpleNamespace(
            wrench_sensor=SimpleNamespace(
                get_wrench_sample=lambda: (np.arange(6.0), 500_000_000)
            )
        ),
    )

    observation, _ = service._create_observation()

    np.testing.assert_array_equal(
        observation["external_wrench_sensor_raw"].float_array.values,
        np.arange(6.0),
    )
    assert observation["external_wrench_sensor_raw_timestamp_ns"].int_value == 500_000_000
    assert "external_wrench_world_from_sensor_rotation" not in observation
    assert "external_wrench_world" not in observation
    assert "external_wrench_wrist" not in observation


def test_stalled_sample_pair_stops_canonical_publication(monkeypatch) -> None:
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
    }

    class Applier:
        def measured_sensor_raw_to_calibrated_wrist(self, *args, **kwargs):
            return np.ones(6)

        def measured_sensor_raw_to_calibrated_world(self, *args, **kwargs):
            return np.ones(6)

    service = _service()
    service._force_calibration_applier_lock = threading.Lock()
    service._force_calibration_applier = Applier()
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 1_000_000}),
        arm=SimpleNamespace(
            wrench_sensor=SimpleNamespace(
                get_wrench_sample=lambda: (np.arange(6.0), 1_000_000_000)
            )
        ),
    )
    monotonic_values = iter((10.0, 10.3))
    monkeypatch.setattr(server_module.time, "monotonic", lambda: next(monotonic_values))

    fresh, _ = service._create_observation()
    stalled, _ = service._create_observation()

    assert "external_wrench_world" in fresh
    assert "external_wrench_sensor_raw" in stalled
    assert "external_wrench_world" not in stalled
    assert "external_wrench_wrist" not in stalled


def test_robot_state_does_not_read_or_leak_source_wrench() -> None:
    robot = VegaRobot.__new__(VegaRobot)
    robot.arm = SimpleNamespace(
        get_joint_pos=lambda: np.zeros(7),
        get_joint_vel=lambda: np.zeros(7),
        get_joint_torque=lambda: np.zeros(7),
        get_timestamp_ns=lambda: 123_456_000,
        wrench_sensor=SimpleNamespace(
            get_wrench_state=lambda: (_ for _ in ()).throw(
                AssertionError("robot state must not read optional F/T sensor")
            ),
            get_wrench_sample=lambda: (_ for _ in ()).throw(
                AssertionError("robot state must not read optional F/T sensor")
            ),
        ),
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
    assert "external_wrench_sensor_frame" not in state


def test_unconfigured_yaw_preserves_raw_sensor_and_omits_wrist_wrench() -> None:
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
    }
    service = _service()
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123}),
        arm=SimpleNamespace(
            wrench_sensor=SimpleNamespace(
                get_wrench_sample=lambda: (np.arange(6.0), 123_000)
            )
        ),
    )

    observation, _ = service._create_observation()

    assert "external_wrench_wrist" not in observation
    assert "external_wrench_wrist_reference_point" not in observation
    np.testing.assert_array_equal(
        observation["external_wrench_sensor_raw"].float_array.values,
        np.arange(6.0),
    )
    assert (
        observation["external_wrench_sensor_raw_reference_point"].string_value
        == "physical F/T sensor measurement origin"
    )


def test_malformed_source_wrench_omits_optional_wrist_wrench() -> None:
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
    }
    service = _service()
    service.external_wrench_sensor_to_wrist_yaw_degrees = 0.0
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123}),
        arm=SimpleNamespace(
            wrench_sensor=SimpleNamespace(
                get_wrench_sample=lambda: (np.ones(5), 123_000)
            )
        ),
    )

    observation, _ = service._create_observation()

    assert "external_wrench_wrist" not in observation
    assert "external_wrench_wrist_reference_point" not in observation
    assert "external_wrench_sensor_raw" not in observation
    assert "external_wrench_sensor_raw_reference_point" not in observation


def test_silent_sensor_omits_wrench_without_breaking_observation() -> None:
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
    }
    service = _service()
    service.external_wrench_sensor_to_wrist_yaw_degrees = 0.0
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123}),
        arm=SimpleNamespace(
            wrench_sensor=SimpleNamespace(
                get_wrench_sample=lambda: (_ for _ in ()).throw(
                    ServiceUnavailableError("no sensor sample")
                )
            )
        ),
    )

    observation, timestamp_us = service._create_observation()

    assert timestamp_us == 123
    assert "joint_positions" in observation
    assert "external_wrench_wrist" not in observation
    assert "external_wrench_sensor_raw" not in observation


def test_sensor_payload_missing_wrench_omits_only_optional_signal() -> None:
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
    }
    service = _service()
    service.external_wrench_sensor_to_wrist_yaw_degrees = 0.0
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123}),
        arm=SimpleNamespace(
            wrench_sensor=SimpleNamespace(
                get_wrench_sample=lambda: (_ for _ in ()).throw(KeyError("wrench"))
            )
        ),
    )

    observation, timestamp_us = service._create_observation()

    assert timestamp_us == 123
    assert "joint_positions" in observation
    assert "external_wrench_wrist" not in observation
    assert "external_wrench_sensor_raw" not in observation


def test_unexpected_sensor_failure_omits_only_optional_signal() -> None:
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
    }
    service = _service()
    service.external_wrench_sensor_to_wrist_yaw_degrees = 0.0
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123}),
        arm=SimpleNamespace(
            wrench_sensor=SimpleNamespace(
                get_wrench_sample=lambda: (_ for _ in ()).throw(
                    RuntimeError("sensor transport failed")
                )
            )
        ),
    )

    observation, timestamp_us = service._create_observation()

    assert timestamp_us == 123
    assert "joint_positions" in observation
    assert "external_wrench_wrist" not in observation
    assert "external_wrench_sensor_raw" not in observation


def test_unavailable_sensor_warning_is_rate_limited(caplog) -> None:
    state = {
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_torques_computed": np.zeros(7),
        "gripper_position": 0.0,
        "cartesian_position": np.zeros(6),
    }
    service = _service()
    service.external_wrench_sensor_to_wrist_yaw_degrees = 0.0
    service._robot = SimpleNamespace(
        get_robot_state=lambda: (state, {"robot_timestamp_us": 123}),
        arm=SimpleNamespace(
            wrench_sensor=SimpleNamespace(
                get_wrench_sample=lambda: (_ for _ in ()).throw(
                    ServiceUnavailableError("no sensor sample")
                )
            )
        ),
    )

    with caplog.at_level("WARNING", logger="robotenv_vega"):
        service._create_observation()
        service._create_observation()

    records = [
        record
        for record in caplog.records
        if "Omitting unavailable or malformed" in record.message
    ]
    assert len(records) == 1


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
    assert "external_wrench_sensor_raw" not in observation
    assert "external_wrench_sensor_raw_reference_point" not in observation
