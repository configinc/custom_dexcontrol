import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexcontrol.core.robotenv_vega.force_calibration.wrist_ft import (
    GRAVITY_M_S2,
    WristFtCalibrationApplier,
    WristFtCalibrationHold,
    WristFtCalibrationParameters,
    WristFtCalibrationQualityError,
    fit_wrist_ft_calibration,
    wrist_ft_calibration_from_snapshot,
)

GRAVITY_WORLD = np.array([0.0, 0.0, -1.0])
MASS_KG = 1.2
COM_SENSOR_M = np.array([0.03, -0.02, 0.08])
FORCE_OFFSET = np.array([0.4, -0.25, 0.15])
TORQUE_OFFSET = np.array([0.02, -0.03, 0.01])
TORQUE_GAIN = 0.97
REFERENCE_POINT = "physical F/T sensor measurement origin"


def _orientations():
    angles_degrees = (
        (0, 0, 0),
        (45, 0, 0),
        (-45, 0, 0),
        (0, 45, 0),
        (0, -45, 0),
        (40, 35, 0),
        (-40, 35, 0),
        (35, -40, 20),
        (-35, -40, -20),
        (55, 20, 35),
        (-55, 20, -35),
        (20, -55, 45),
    )
    return [Rotation.from_euler("xyz", value, degrees=True).as_matrix() for value in angles_degrees]


def _raw_static_wrench(world_from_ee, *, external_sensor=None):
    gravity_sensor = world_from_ee.T @ GRAVITY_WORLD
    gravity_force = MASS_KG * GRAVITY_M_S2 * gravity_sensor
    external = (
        np.zeros(6, dtype=np.float64) if external_sensor is None else np.asarray(external_sensor, dtype=np.float64)
    )
    force = gravity_force + external[:3] + FORCE_OFFSET
    torque = TORQUE_GAIN * (np.cross(COM_SENSOR_M, gravity_force) + external[3:]) + TORQUE_OFFSET
    return np.concatenate([force, torque])


def _holds():
    return tuple(
        WristFtCalibrationHold(
            pose_id=f"pose-{index}",
            measured_sensor_wrench=_raw_static_wrench(world_from_ee),
            world_from_sensor_rotation=world_from_ee,
        )
        for index, world_from_ee in enumerate(_orientations())
    )


def _fit(*, arm_name="left"):
    return fit_wrist_ft_calibration(
        _holds(),
        arm_name=arm_name,
        sensor_frame_id="left_ft_sensor",
        reference_point=REFERENCE_POINT,
        world_frame_id="world",
        gravity_world_unit=GRAVITY_WORLD,
        gripper_mass_kg=MASS_KG,
        gripper_com_sensor_m=COM_SENSOR_M,
    )


def test_fit_recovers_static_sensor_model_and_removes_gravity():
    artifact = _fit()

    assert artifact.as_parameters()["frame_rotations_owner"] == "robot_server"
    np.testing.assert_allclose(artifact.force_decoupling_inverse, np.eye(3), atol=2e-5)
    np.testing.assert_allclose(artifact.force_offset_sensor, FORCE_OFFSET, atol=2e-5)
    np.testing.assert_allclose(artifact.torque_offset_sensor, TORQUE_OFFSET, atol=2e-5)
    assert artifact.torque_gain == pytest.approx(TORQUE_GAIN, abs=2e-5)
    assert artifact.quality["passed"] is True

    applier = WristFtCalibrationApplier(artifact)
    for hold in _holds():
        calibrated = applier.measured_sensor_raw_to_calibrated_world(
            hold.measured_sensor_wrench,
            hold.world_from_sensor_rotation,
            reference_point=REFERENCE_POINT,
        )
        np.testing.assert_allclose(calibrated, np.zeros(6), atol=3e-5)


def test_common_robust_refit_drops_two_contaminated_vega_poses_and_keeps_eight():
    clean = list(_holds()[:10])
    contaminated = []
    for index, hold in enumerate(clean):
        wrench = hold.measured_sensor_wrench.copy()
        if index in {1, 8}:
            wrench[0] += 8.0 if index == 1 else -8.0
        contaminated.append(
            WristFtCalibrationHold(
                hold.pose_id, wrench, hold.world_from_sensor_rotation
            )
        )

    parameters = fit_wrist_ft_calibration(
        contaminated,
        arm_name="left",
        sensor_frame_id="left_ft_sensor",
        reference_point=REFERENCE_POINT,
        world_frame_id="world",
        gravity_world_unit=GRAVITY_WORLD,
        gripper_mass_kg=MASS_KG,
        gripper_com_sensor_m=COM_SENSOR_M,
    )

    assert parameters.quality["hold_count"] == 8
    assert set(parameters.quality["dropped_pose_ids"]) == {"pose-1", "pose-8"}


def test_applier_rotates_external_force_and_torque_to_world_at_same_origin():
    artifact = _fit()
    applier = WristFtCalibrationApplier(artifact)
    world_from_ee = _orientations()[5]
    external_sensor = np.array([2.0, -1.0, 0.5, 0.2, -0.1, 0.05])
    raw = _raw_static_wrench(world_from_ee, external_sensor=external_sensor)

    calibrated = applier.measured_sensor_raw_to_calibrated_world(raw, world_from_ee, reference_point=REFERENCE_POINT)

    np.testing.assert_allclose(calibrated[:3], world_from_ee @ external_sensor[:3], atol=3e-5)
    np.testing.assert_allclose(calibrated[3:], world_from_ee @ external_sensor[3:], atol=3e-5)


def test_fit_consumes_robot_server_world_from_sensor_rotation_directly():
    true_ee_from_sensor = Rotation.from_euler("z", 4.0, degrees=True).as_matrix()
    holds = []
    for index, world_from_ee in enumerate(_orientations()):
        world_from_sensor = world_from_ee @ true_ee_from_sensor
        gravity_sensor = world_from_sensor.T @ GRAVITY_WORLD
        gravity_force = MASS_KG * GRAVITY_M_S2 * gravity_sensor
        raw = np.concatenate(
            [
                gravity_force + FORCE_OFFSET,
                TORQUE_GAIN * np.cross(COM_SENSOR_M, gravity_force) + TORQUE_OFFSET,
            ]
        )
        holds.append(
            WristFtCalibrationHold(
                pose_id=f"pose-{index}",
                measured_sensor_wrench=raw,
                world_from_sensor_rotation=world_from_sensor,
            )
        )

    parameters = fit_wrist_ft_calibration(
        holds,
        arm_name="right",
        sensor_frame_id="right_ft_sensor",
        reference_point=REFERENCE_POINT,
        world_frame_id="world",
        gravity_world_unit=GRAVITY_WORLD,
        gripper_mass_kg=MASS_KG,
        gripper_com_sensor_m=COM_SENSOR_M,
    )

    assert parameters.as_parameters()["frame_rotations_owner"] == "robot_server"


def _holds_with_force_decoupling(cross_axis_term, diagonal=(1.0, 1.0, 1.0)):
    decoupling = np.diag(diagonal)
    decoupling[0, 1] = cross_axis_term
    holds = []
    for index, world_from_sensor in enumerate(_orientations()):
        gravity_sensor = world_from_sensor.T @ GRAVITY_WORLD
        gravity_force = MASS_KG * GRAVITY_M_S2 * gravity_sensor
        raw = np.concatenate(
            [
                decoupling @ gravity_force + FORCE_OFFSET,
                TORQUE_GAIN * np.cross(COM_SENSOR_M, gravity_force) + TORQUE_OFFSET,
            ]
        )
        holds.append(
            WristFtCalibrationHold(
                pose_id=f"pose-{index}",
                measured_sensor_wrench=raw,
                world_from_sensor_rotation=world_from_sensor,
            )
        )
    return tuple(holds)


def test_fit_accepts_force_cross_axis_term_below_point_two():
    parameters = fit_wrist_ft_calibration(
        _holds_with_force_decoupling(0.19),
        arm_name="right",
        sensor_frame_id="right_ft_sensor",
        reference_point=REFERENCE_POINT,
        world_frame_id="world",
        gravity_world_unit=GRAVITY_WORLD,
        gripper_mass_kg=MASS_KG,
        gripper_com_sensor_m=COM_SENSOR_M,
    )

    assert parameters.quality["max_force_decoupling_off_diagonal"] == pytest.approx(0.19)


def test_fit_rejects_force_cross_axis_term_above_point_two():
    with pytest.raises(WristFtCalibrationQualityError, match="cross-axis term exceeds 0.2"):
        fit_wrist_ft_calibration(
            _holds_with_force_decoupling(0.21),
            arm_name="right",
            sensor_frame_id="right_ft_sensor",
            reference_point=REFERENCE_POINT,
            world_frame_id="world",
            gravity_world_unit=GRAVITY_WORLD,
            gripper_mass_kg=MASS_KG,
            gripper_com_sensor_m=COM_SENSOR_M,
        )


def test_fit_accepts_fleet_gain_and_rejects_outside_twenty_percent():
    parameters = fit_wrist_ft_calibration(
        _holds_with_force_decoupling(0.0, diagonal=(1.0, 1.16, 1.0)),
        arm_name="right",
        sensor_frame_id="right_ft_sensor",
        reference_point=REFERENCE_POINT,
        world_frame_id="world",
        gravity_world_unit=GRAVITY_WORLD,
        gripper_mass_kg=MASS_KG,
        gripper_com_sensor_m=COM_SENSOR_M,
    )
    assert parameters.quality["force_decoupling_diagonal"][1] == pytest.approx(1.16)

    with pytest.raises(WristFtCalibrationQualityError, match="outside \\[0.80, 1.20\\]"):
        fit_wrist_ft_calibration(
            _holds_with_force_decoupling(0.0, diagonal=(1.0, 1.21, 1.0)),
            arm_name="right",
            sensor_frame_id="right_ft_sensor",
            reference_point=REFERENCE_POINT,
            world_frame_id="world",
            gravity_world_unit=GRAVITY_WORLD,
            gripper_mass_kg=MASS_KG,
            gripper_com_sensor_m=COM_SENSOR_M,
        )


def test_applier_rejects_a_different_wrench_reference_point():
    with pytest.raises(ValueError, match="reference point"):
        WristFtCalibrationApplier(_fit()).measured_sensor_raw_to_calibrated_world(
            _raw_static_wrench(np.eye(3)),
            np.eye(3),
            reference_point="tool tip",
        )


def test_fit_rejects_insufficient_pose_coverage():
    repeated = tuple(
        WristFtCalibrationHold(
            pose_id=f"pose-{index}",
            measured_sensor_wrench=_raw_static_wrench(np.eye(3)),
            world_from_sensor_rotation=np.eye(3),
        )
        for index in range(8)
    )

    with pytest.raises(WristFtCalibrationQualityError, match="spread") as exc_info:
        fit_wrist_ft_calibration(
            repeated,
            arm_name="right",
            sensor_frame_id="right_ft_sensor",
            reference_point=REFERENCE_POINT,
            world_frame_id="world",
            gravity_world_unit=GRAVITY_WORLD,
            gripper_mass_kg=MASS_KG,
            gripper_com_sensor_m=COM_SENSOR_M,
        )
    assert "metrics=" in str(exc_info.value)
    assert "force_decoupling_diagonal" in str(exc_info.value)


def _parameters_for_runtime():
    return _fit()


def test_accepted_snapshot_round_trips_through_runtime_validation():
    parameters = _parameters_for_runtime()
    assert parameters.as_parameters()["schema_version"] == 5
    assert parameters.as_parameters()["frame_rotations_owner"] == "robot_server"
    restored = wrist_ft_calibration_from_snapshot(parameters.as_parameters())
    assert restored.as_parameters() == parameters.as_parameters()

    forged_quality = parameters.as_parameters()
    forged_quality["quality"] = {"passed": True}
    with pytest.raises(ValueError, match="reconcile"):
        wrist_ft_calibration_from_snapshot(forged_quality)

    missing_evidence = parameters.as_parameters()
    missing_evidence.pop("pose_evidence")
    with pytest.raises(ValueError, match="pose evidence"):
        wrist_ft_calibration_from_snapshot(missing_evidence)

    wrong = parameters.as_parameters()
    wrong["input_field"] = "external_wrench_world_raw"
    with pytest.raises(ValueError, match="input field"):
        wrist_ft_calibration_from_snapshot(wrong)

    legacy = parameters.as_parameters()
    legacy["schema_version"] = 4
    legacy.pop("frame_rotations_owner")
    with pytest.raises((TypeError, ValueError)):
        wrist_ft_calibration_from_snapshot(legacy)


def test_parameters_cannot_be_forged_without_the_validated_fitter():
    with pytest.raises(TypeError, match="created by fit_wrist_ft_calibration"):
        WristFtCalibrationParameters(
            arm_name="left",
            sensor_frame_id="left_ft_sensor",
            reference_point=REFERENCE_POINT,
            world_frame_id="world",
            gravity_world_unit=GRAVITY_WORLD,
            gripper_mass_kg=MASS_KG,
            gripper_com_sensor_m=COM_SENSOR_M,
            force_decoupling_inverse=np.eye(3),
            force_offset_sensor=FORCE_OFFSET,
            torque_gain=TORQUE_GAIN,
            torque_offset_sensor=TORQUE_OFFSET,
            quality={"passed": True, "hold_count": 12, "reasons": []},
            pose_evidence=(),
        )
