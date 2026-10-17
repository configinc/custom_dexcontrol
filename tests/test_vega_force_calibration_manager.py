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

    def install(self, parameters):
        self.installed.append(parameters)


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
