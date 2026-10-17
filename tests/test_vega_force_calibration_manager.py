from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from dexcontrol.core.robotenv_vega.force_calibration import WristFtCalibrationHold
from dexcontrol.core.robotenv_vega.force_calibration_lifecycle import (
    BilateralCalibrationPlan,
    ForceCalibrationLifecycle,
    LifecycleError,
)
from dexcontrol.core.robotenv_vega.vega_force_calibration import (
    CalibrationCancelled,
    VegaForceCalibrationManager,
    VegaRobotCalibrationAdapter,
    _minimum_jerk_rotation_steps,
)


def _plan(tmp_path: Path, stage_count: int = 2) -> BilateralCalibrationPlan:
    stages = []
    for index in range(stage_count):
        stages.append(
            {
                "stage_id": f"stage-{index}",
                "arms": {
                    "left": {"kind": "pose", "value": index},
                    "right": {"kind": "pose", "value": -index},
                },
            }
        )
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {"schema_version": 1, "plan_id": "test-plan", "stages": stages},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return BilateralCalibrationPlan.load(path, "left")


def _lifecycle(tmp_path: Path, stage_count: int = 2) -> ForceCalibrationLifecycle:
    return ForceCalibrationLifecycle(
        plan=_plan(tmp_path, stage_count),
        server_identity={"robot": "vega", "arm": "left"},
        storage_directory=tmp_path / "state",
        calibration_kind="vega_wrist_ft_v1",
        reference_point_id="sensor-origin",
    )


class FakeParameters:
    quality = {"passed": True}

    def as_parameters(self):
        return {"schema_version": 5, "fake": True}


class FakeAdapter:
    def __init__(self) -> None:
        self.executions: list[str] = []
        self.installed = []
        self.safe_returns = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False

    def execute_stage(self, stage_id, payload, cancelled, set_phase):
        self.executions.append(stage_id)
        self.entered.set()
        if self.block:
            while not self.release.wait(0.005):
                if cancelled.is_set():
                    raise CalibrationCancelled()
        set_phase("COLLECTING", stage_id)
        return WristFtCalibrationHold(
            pose_id=stage_id,
            measured_sensor_wrench=np.zeros(6),
            world_from_sensor_rotation=np.eye(3),
            measured_sensor_wrench_samples=np.zeros((2, 6)),
        )

    def safe_return(self, cancelled=None):
        self.safe_returns += 1

    def fit(self, holds):
        assert len(holds) == 2
        return FakeParameters()

    def prepare_install(self, parameters):
        return parameters

    def install_prepared(self, prepared):
        self.installed.append(prepared)

    def commit_install(self, prepared, commit):
        status = commit()
        self.install_prepared(prepared)
        return status


class Claim:
    def __init__(self) -> None:
        self.owned = False
        self.claims = 0
        self.releases = 0

    def acquire(self) -> bool:
        if self.owned:
            return False
        self.owned = True
        self.claims += 1
        return True

    def release(self) -> None:
        assert self.owned
        self.owned = False
        self.releases += 1


def _manager(tmp_path: Path):
    adapter = FakeAdapter()
    claim = Claim()
    manager = VegaForceCalibrationManager(
        lifecycle=_lifecycle(tmp_path),
        adapter=adapter,
        claim_robot=claim.acquire,
        release_robot=claim.release,
    )
    return manager, adapter, claim


def _wait_state(manager, state: str, timeout: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.status()
        if status.state == state:
            return status
        time.sleep(0.005)
    raise AssertionError(f"did not reach {state}; last={manager.status()}")


def test_start_claims_robot_without_motion(tmp_path: Path) -> None:
    manager, adapter, claim = _manager(tmp_path)
    status = manager.start("run-1", True)
    assert status.state == "READY_FOR_STAGE"
    assert adapter.executions == []
    assert claim.owned is True


def test_advance_executes_each_stage_once_and_stages_candidate(tmp_path: Path) -> None:
    manager, adapter, _ = _manager(tmp_path)
    manager.start("run-1", True)
    manager.advance("run-1", 0, "stage-0")
    _wait_state(manager, "READY_FOR_STAGE")
    retry = manager.advance("run-1", 0, "stage-0")
    assert retry.completed_stages == 1
    assert adapter.executions == ["stage-0"]
    manager.advance("run-1", 1, "stage-1")
    status = _wait_state(manager, "READY_TO_APPLY")
    assert status.completed_stages == 2
    assert adapter.executions == ["stage-0", "stage-1"]


def test_commit_applies_candidate_and_releases_robot(tmp_path: Path) -> None:
    manager, adapter, claim = _manager(tmp_path)
    manager.start("run-1", True)
    manager.advance("run-1", 0, "stage-0")
    _wait_state(manager, "READY_FOR_STAGE")
    manager.advance("run-1", 1, "stage-1")
    ready = _wait_state(manager, "READY_TO_APPLY")
    status = manager.commit("run-1", ready.plan_sha256)
    assert status.state == "SUCCEEDED"
    assert len(adapter.installed) == 1
    assert claim.owned is False
    assert claim.releases == 1


def test_commit_validates_runtime_candidate_before_durable_success(tmp_path: Path) -> None:
    manager, adapter, claim = _manager(tmp_path)
    manager.start("run-1", True)
    manager.advance("run-1", 0, "stage-0")
    _wait_state(manager, "READY_FOR_STAGE")
    manager.advance("run-1", 1, "stage-1")
    ready = _wait_state(manager, "READY_TO_APPLY")

    def reject(_parameters):
        raise ValueError("invalid runtime parameters")

    adapter.prepare_install = reject
    with pytest.raises(ValueError, match="invalid runtime parameters"):
        manager.commit("run-1", ready.plan_sha256)
    assert manager.status().state == "READY_TO_APPLY"
    assert claim.owned is True


def test_cancel_stops_worker_returns_safe_and_releases(tmp_path: Path) -> None:
    manager, adapter, claim = _manager(tmp_path)
    adapter.block = True
    manager.start("run-1", True)
    manager.advance("run-1", 0, "stage-0")
    assert adapter.entered.wait(1.0)
    status = manager.request_cancel("run-1")
    assert status.state == "CANCELLING"
    _wait_state(manager, "CANCELLED")
    assert adapter.safe_returns == 1
    assert claim.owned is False


def test_cancel_during_fit_finishes_cancelled_not_failed(tmp_path: Path) -> None:
    manager, adapter, claim = _manager(tmp_path)
    fit_entered = threading.Event()
    fit_release = threading.Event()
    original_fit = adapter.fit

    def delayed_fit(holds):
        fit_entered.set()
        assert fit_release.wait(1.0)
        return original_fit(holds)

    adapter.fit = delayed_fit
    manager.start("run-1", True)
    manager.advance("run-1", 0, "stage-0")
    _wait_state(manager, "READY_FOR_STAGE")
    manager.advance("run-1", 1, "stage-1")
    assert fit_entered.wait(1.0)
    assert manager.request_cancel("run-1").state == "CANCELLING"
    fit_release.set()
    _wait_state(manager, "CANCELLED")
    assert claim.owned is False


def test_rotation_profile_uses_euler_deltas_and_returns_exactly() -> None:
    target = np.radians(np.array([25.0, -30.0, 40.0]))
    forward, reverse = _minimum_jerk_rotation_steps(target, 20.0)
    np.testing.assert_allclose(np.sum(forward, axis=0), target, atol=1e-12)
    np.testing.assert_allclose(
        np.sum((*forward, *reverse), axis=0), np.zeros(3), atol=1e-12
    )


class _FakeSensor:
    def __init__(self, timestamp_ns: int) -> None:
        self.timestamp_ns = timestamp_ns

    def get_wrench_sample(self):
        return np.arange(6, dtype=np.float64), self.timestamp_ns


class _FakeArm:
    def __init__(self, joints=None, sensor_timestamp_ns: int = 1_000_000_000) -> None:
        self._joints = np.zeros(7) if joints is None else np.asarray(joints, dtype=float)
        self.wrench_sensor = _FakeSensor(sensor_timestamp_ns)

    def get_joint_pos(self):
        return self._joints.copy()


class _FakeRobot:
    def __init__(self, joints=None, sensor_timestamp_ns: int = 1_000_000_000) -> None:
        self.arm = _FakeArm(joints, sensor_timestamp_ns)

    def get_robot_state(self):
        return (
            {
                "cartesian_position": np.zeros(6),
                "joint_positions": self.arm.get_joint_pos(),
                "gripper_position": 0.0,
            },
            {"robot_timestamp_us": 1_000_000},
        )

    def update_command(self, *_args, **_kwargs):
        return None


class _FakeService:
    gripper_type = "robotiq"
    arm_side = "left"
    frame_type = "world"
    control_hz = 20.0
    R_robot_to_world = None
    R_world_to_robot = None

    def __init__(self, joints=None, sensor_timestamp_ns: int = 1_000_000_000) -> None:
        self._robot = _FakeRobot(joints, sensor_timestamp_ns)
        self._cmd_lock = threading.Lock()

    def _execute_reset_sequence(self, target, **_kwargs):
        self._robot.arm._joints = np.asarray(target, dtype=float).copy()

    @staticmethod
    def _timestamp_us(timestamp):
        return timestamp["robot_timestamp_us"]


def _physical_adapter(service: _FakeService) -> VegaRobotCalibrationAdapter:
    return VegaRobotCalibrationAdapter(service, 1.2, np.array([0.0, 0.0, 0.072]))


def test_sample_uses_sensor_timestamp_and_rejects_cross_stream_skew() -> None:
    adapter = _physical_adapter(_FakeService(sensor_timestamp_ns=1_000_000_000))
    assert adapter._sample()[3] == pytest.approx(1.0)

    stale = _physical_adapter(_FakeService(sensor_timestamp_ns=500_000_000))
    with pytest.raises(RuntimeError, match="not time aligned"):
        stale._sample()


def test_safe_return_verification_rejects_unreached_joint_target() -> None:
    adapter = _physical_adapter(_FakeService(joints=np.ones(7)))
    with pytest.raises(RuntimeError, match="safe return joint error"):
        adapter._verify_joint_return(np.zeros(7))


def test_real_adapter_prepare_completes_against_lifecycle(tmp_path: Path) -> None:
    plan_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "dexcontrol"
        / "core"
        / "robotenv_vega"
        / "force_calibration"
        / "vega_bilateral_plan_v1.json"
    )
    lifecycle = ForceCalibrationLifecycle(
        plan=BilateralCalibrationPlan.load(plan_path, "left"),
        server_identity={"robot": "vega", "arm": "left"},
        storage_directory=tmp_path / "state",
        calibration_kind="vega_wrist_ft_v1",
        reference_point_id="sensor-origin",
    )
    adapter = _physical_adapter(_FakeService())
    lifecycle.start("run-1", force_recalibrate=True)
    _, payload = lifecycle.advance("run-1", 0, "prepare")
    adapter.execute_stage(
        "prepare",
        payload,
        threading.Event(),
        lambda state, message: lifecycle.set_phase("run-1", state, message),
    )
    assert lifecycle.complete_stage("run-1", 0).state == "READY_FOR_STAGE"


def test_real_adapter_cancel_during_pose_preserves_cancel_signal() -> None:
    adapter = _physical_adapter(_FakeService())
    adapter._start_joints = np.zeros(7)
    cancelled = threading.Event()
    phases = []

    def set_phase(state, _message):
        phases.append(state)
        if state == "SETTLING":
            cancelled.set()

    with pytest.raises(CalibrationCancelled):
        adapter.execute_stage(
            "pose-1",
            {"kind": "pose", "rpy_degrees": [0.0, 0.0, 0.0]},
            cancelled,
            set_phase,
        )
    assert phases == ["MOVING", "SETTLING"]


def test_second_start_is_rejected_while_first_run_owns_robot(tmp_path: Path) -> None:
    manager, _, _ = _manager(tmp_path)
    manager.start("run-1", True)
    with pytest.raises(LifecycleError, match="busy"):
        manager.start("run-2", True)


def test_packaged_plan_has_shared_stage_ids_and_mirrored_arm_payloads() -> None:
    plan_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "dexcontrol"
        / "core"
        / "robotenv_vega"
        / "force_calibration"
        / "vega_bilateral_plan_v1.json"
    )
    left = BilateralCalibrationPlan.load(plan_path, "left")
    right = BilateralCalibrationPlan.load(plan_path, "right")

    assert left.plan_sha256 == right.plan_sha256
    assert [stage.stage_id for stage in left.stages] == [
        stage.stage_id for stage in right.stages
    ]
    assert len(left.stages) == 11
    np.testing.assert_allclose(
        left.stages[1].arm_payload["rpy_degrees"], [-30.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        right.stages[1].arm_payload["rpy_degrees"], [30.0, 0.0, 0.0]
    )
