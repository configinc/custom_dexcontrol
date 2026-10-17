from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

import dexcontrol.core.robotenv_vega.force_calibration_lifecycle as lifecycle_module
from dexcontrol.core.robotenv_vega.force_calibration_lifecycle import (
    BilateralCalibrationPlan,
    ForceCalibrationLifecycle,
    LifecycleError,
    _next_kst_boundary_us,
    bilateral_results_compatible,
)


class Clock:
    def __init__(self, timestamp_us: int) -> None:
        self.timestamp_us = timestamp_us

    def __call__(self) -> int:
        return self.timestamp_us


def _timestamp_us(value: str) -> int:
    return int(datetime.fromisoformat(value).astimezone(timezone.utc).timestamp() * 1_000_000)


def _write_plan(path: Path) -> bytes:
    raw = (
        '{"schema_version":1,"plan_id":"vega-bilateral-v1","stages":['
        '{"stage_id":"tilt-x","arms":{"left":{"rpy_deg":[-30,0,0]},'
        '"right":{"rpy_deg":[30,0,0]}}},'
        '{"stage_id":"tilt-y","arms":{"left":{"rpy_deg":[0,-45,0]},'
        '"right":{"rpy_deg":[0,-45,0]}}}]}'
    ).encode()
    path.write_bytes(raw)
    return raw


@pytest.fixture
def plan(tmp_path: Path) -> BilateralCalibrationPlan:
    path = tmp_path / "plan.json"
    raw = _write_plan(path)
    loaded = BilateralCalibrationPlan.load(path, "left")
    assert loaded.plan_sha256 == sha256(raw).hexdigest()
    return loaded


@pytest.fixture
def clock() -> Clock:
    return Clock(_timestamp_us("2026-10-17T00:00:00+00:00"))


def _lifecycle(tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock) -> ForceCalibrationLifecycle:
    return ForceCalibrationLifecycle(
        plan=plan,
        server_identity={"unit_id": "ds-teleop-0001-04", "firmware": "1.2.3"},
        storage_directory=tmp_path / "cache",
        calibration_kind="wrist_ft_bias",
        reference_point_id="configured_tcp",
        command_timeout_seconds=5,
        now_us=clock,
    )


def _collect_all(lifecycle: ForceCalibrationLifecycle, run_id: str) -> None:
    for index, stage in enumerate(lifecycle.plan.stages):
        status, payload = lifecycle.advance(run_id, index, stage.stage_id)
        assert status.state == "MOVING"
        assert payload == stage.arm_payload
        lifecycle.set_phase(run_id, "COLLECTING")
        lifecycle.complete_stage(run_id, index)


def test_prepare_stage_can_return_directly_from_moving(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-prepare", force_recalibrate=True)
    lifecycle.advance("run-prepare", 0, "tilt-x")
    status = lifecycle.set_phase("run-prepare", "RETURNING")
    assert status.state == "RETURNING"


def test_plan_is_one_exact_bilateral_artifact(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    _write_plan(path)
    left = BilateralCalibrationPlan.load(path, "left")
    right = BilateralCalibrationPlan.load(path, "right")
    assert left.plan_sha256 == right.plan_sha256
    assert [stage.stage_id for stage in left.stages] == ["tilt-x", "tilt-y"]
    assert left.stages[0].arm_payload != right.stages[0].arm_payload

    document = json.loads(path.read_text())
    del document["stages"][0]["arms"]["right"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="left and right"):
        BilateralCalibrationPlan.load(path, "left")

    path.write_text(
        '{"schema_version":1,"plan_id":"duplicate","plan_id":"hidden",'
        '"stages":[]}'
    )
    with pytest.raises(ValueError, match="strict JSON"):
        BilateralCalibrationPlan.load(path, "left")


def test_plan_validates_unselected_arm_payload(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(
        '{"schema_version":1,"plan_id":"bad-right","stages":['
        '{"stage_id":"one","arms":{"left":{"rpy_deg":[0,0,0]},'
        '"right":{"rpy_deg":[NaN,0,0]}}}]}'
    )
    with pytest.raises(ValueError, match="strict JSON"):
        BilateralCalibrationPlan.load(path, "left")


def test_stage_barrier_rejects_skip_mismatch_and_conflicting_retry(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    first = lifecycle.start("run-1", force_recalibrate=True)
    assert first.state == "READY_FOR_STAGE"
    assert first.current_stage_index == 0

    with pytest.raises(LifecycleError) as skipped:
        lifecycle.advance("run-1", 1, "tilt-y")
    assert skipped.value.code == "FAILED_PRECONDITION"
    with pytest.raises(LifecycleError, match="stage_id mismatch"):
        lifecycle.advance("run-1", 0, "tilt-y")
    with pytest.raises(LifecycleError, match="conflicting Start retry"):
        lifecycle.start("run-1", force_recalibrate=False)

    _, payload = lifecycle.advance("run-1", 0, "tilt-x")
    assert payload == {"rpy_deg": [-30, 0, 0]}
    retry_in_flight, retry_payload = lifecycle.advance("run-1", 0, "tilt-x")
    assert retry_in_flight.state == "MOVING"
    assert retry_payload is None
    payload["rpy_deg"][0] = 999
    assert lifecycle.plan.stages[0].arm_payload["rpy_deg"][0] == -30
    lifecycle.set_phase("run-1", "COLLECTING")
    lifecycle.complete_stage("run-1", 0)
    retry, duplicate_payload = lifecycle.advance("run-1", 0, "tilt-x")
    assert retry.current_stage_index == 1
    assert duplicate_payload is None


def test_staged_candidate_survives_restart_and_commit_is_idempotent(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    _collect_all(lifecycle, "run-1")
    with pytest.raises(LifecycleError, match="quality gate"):
        lifecycle.stage_candidate(
            "run-1", parameters={"bias": [0] * 6}, quality={"passed": False}
        )
    ready = lifecycle.stage_candidate(
        "run-1",
        parameters={"bias": [0.1] * 6},
        quality={"passed": True, "rmse": 0.2},
    )
    assert ready.state == "READY_TO_APPLY"
    assert ready.command_deadline_timestamp_us == 0

    restarted = _lifecycle(tmp_path, plan, clock)
    assert restarted.server_instance_id != lifecycle.server_instance_id
    assert restarted.status().state == "READY_TO_APPLY"
    committed = restarted.commit("run-1", plan.plan_sha256)
    assert committed.state == "SUCCEEDED"
    assert restarted.commit("run-1", plan.plan_sha256) == committed
    current = restarted.get_result()
    assert current["available"] is True
    assert current["was_committed"] is True
    assert current["source_run_id"] == "run-1"

    restarted_again = _lifecycle(tmp_path, plan, clock)
    assert restarted_again.status().state == "SUCCEEDED"
    assert restarted_again.commit("run-1", plan.plan_sha256).state == "SUCCEEDED"


def test_staged_candidate_can_commit_after_daily_boundary(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    clock.timestamp_us = _timestamp_us("2026-10-16T21:59:59.999999+00:00")
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("boundary-run", force_recalibrate=True)
    _collect_all(lifecycle, "boundary-run")
    lifecycle.stage_candidate(
        "boundary-run", parameters={"bias": [0] * 6}, quality={"passed": True}
    )

    clock.timestamp_us = _timestamp_us("2026-10-16T22:00:00+00:00")
    restarted = _lifecycle(tmp_path, plan, clock)
    assert restarted.status().state == "READY_TO_APPLY"
    assert restarted.commit("boundary-run", plan.plan_sha256).state == "SUCCEEDED"
    assert restarted.get_result()["available"] is False


def test_cache_reuse_preserves_source_run_and_history(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("source-run", force_recalibrate=True)
    _collect_all(lifecycle, "source-run")
    lifecycle.stage_candidate(
        "source-run", parameters={"bias": [0] * 6}, quality={"passed": True}
    )
    lifecycle.commit("source-run", plan.plan_sha256)

    reused = lifecycle.start("reuse-run", force_recalibrate=False)
    assert reused.state == "SUCCEEDED"
    assert reused.cache_reused is True
    current = lifecycle.get_result()
    assert current["run_id"] == "reuse-run"
    assert current["source_run_id"] == "source-run"
    historical = lifecycle.get_result("source-run")
    assert historical["run_id"] == "source-run"


def test_bilateral_gate_rejects_partial_commit_cache_pair() -> None:
    base = {
        "available": True,
        "was_committed": True,
        "run_id": "reuse-run",
        "source_run_id": "source-run",
        "plan_id": "plan",
        "plan_sha256": "abc",
        "calibration_kind": "wrist_ft_bias",
        "reference_point_id": "configured_tcp",
        "expires_at_timestamp_us": 200,
    }
    left = {**base, "arm_name": "left"}
    right = {**base, "arm_name": "right"}
    assert bilateral_results_compatible(left, right, now_us=100)
    assert not bilateral_results_compatible(
        left, {**right, "source_run_id": "different-source"}, now_us=100
    )
    assert not bilateral_results_compatible(
        left, {**right, "reference_point_id": "sensor"}, now_us=100
    )
    assert not bilateral_results_compatible(left, right, now_us=200)


def test_daily_kst_boundary_and_invalid_cache_fail_closed(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    before = _timestamp_us("2026-10-16T21:59:59.999999+00:00")  # 06:59:59.999999 KST
    boundary = _timestamp_us("2026-10-16T22:00:00+00:00")  # 07:00 KST
    assert _next_kst_boundary_us(before) == boundary
    assert _next_kst_boundary_us(boundary) == _timestamp_us("2026-10-17T22:00:00+00:00")

    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    _collect_all(lifecycle, "run-1")
    lifecycle.stage_candidate(
        "run-1", parameters={"bias": [0] * 6}, quality={"passed": True}
    )
    lifecycle.commit("run-1", plan.plan_sha256)
    clock.timestamp_us = lifecycle.get_result()["expires_at_timestamp_us"]
    assert lifecycle.get_result()["available"] is False
    assert lifecycle.start("run-2", force_recalibrate=False).state == "READY_FOR_STAGE"

    cache = json.loads(lifecycle.active_cache_path.read_text())
    cache["calibrated_at_timestamp_us"] = clock.timestamp_us + 1
    lifecycle.active_cache_path.write_text(json.dumps(cache))
    assert lifecycle.get_result()["available"] is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("calibration_kind", "wrong-kind"),
        ("reference_point_id", "sensor-origin"),
    ],
)
def test_cache_rejects_calibration_contract_mismatch(
    tmp_path: Path,
    plan: BilateralCalibrationPlan,
    clock: Clock,
    field: str,
    replacement: str,
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    _collect_all(lifecycle, "run-1")
    lifecycle.stage_candidate(
        "run-1", parameters={"bias": [0] * 6}, quality={"passed": True}
    )
    lifecycle.commit("run-1", plan.plan_sha256)
    record = json.loads(lifecycle.active_cache_path.read_text())
    record[field] = replacement
    lifecycle.active_cache_path.write_text(json.dumps(record))
    assert lifecycle.get_result()["available"] is False


def test_commit_revalidates_staged_quality(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    _collect_all(lifecycle, "run-1")
    lifecycle.stage_candidate(
        "run-1", parameters={"bias": [0] * 6}, quality={"passed": True}
    )
    record = json.loads(lifecycle.staged_candidate_path.read_text())
    record["quality"]["passed"] = False
    lifecycle.staged_candidate_path.write_text(json.dumps(record))
    with pytest.raises(LifecycleError, match="unavailable"):
        lifecycle.commit("run-1", plan.plan_sha256)


def test_deadline_requests_safe_cancel_but_ready_candidate_never_expires(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    ready = lifecycle.start("run-1", force_recalibrate=True)
    clock.timestamp_us = ready.command_deadline_timestamp_us
    assert lifecycle.expire_command_deadline().state == "CANCELLING"
    assert lifecycle.finish_cancel("run-1").state == "CANCELLED"

    lifecycle.start("run-2", force_recalibrate=True)
    _collect_all(lifecycle, "run-2")
    lifecycle.stage_candidate(
        "run-2", parameters={"bias": [0] * 6}, quality={"passed": True}
    )
    clock.timestamp_us += 10**9
    assert lifecycle.expire_command_deadline().state == "READY_TO_APPLY"


def test_late_advance_is_rejected_without_motion(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    ready = lifecycle.start("run-1", force_recalibrate=True)
    clock.timestamp_us = ready.command_deadline_timestamp_us
    with pytest.raises(LifecycleError, match="not ready for release"):
        lifecycle.advance("run-1", 0, "tilt-x")
    assert lifecycle.status().state == "CANCELLING"


def test_long_running_worker_phase_has_no_coordinator_deadline(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    moving, _ = lifecycle.advance("run-1", 0, "tilt-x")
    assert moving.command_deadline_timestamp_us == 0
    clock.timestamp_us += 60_000_000
    assert lifecycle.expire_command_deadline().state == "MOVING"
    lifecycle.set_phase("run-1", "COLLECTING")
    clock.timestamp_us += 60_000_000
    assert lifecycle.expire_command_deadline().state == "COLLECTING"


def test_failed_staged_run_does_not_resurrect_after_restart(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    _collect_all(lifecycle, "run-1")
    lifecycle.stage_candidate(
        "run-1", parameters={"bias": [0] * 6}, quality={"passed": True}
    )
    assert lifecycle.fail("run-1", "bilateral peer failed").state == "FAILED"
    assert not lifecycle.staged_candidate_path.exists()

    restarted = _lifecycle(tmp_path, plan, clock)
    assert restarted.status().state == "FAILED"
    with pytest.raises(LifecycleError, match="not ready to apply"):
        restarted.commit("run-1", plan.plan_sha256)


def test_ready_candidate_cancel_survives_restart(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    _collect_all(lifecycle, "run-1")
    lifecycle.stage_candidate(
        "run-1", parameters={"bias": [0] * 6}, quality={"passed": True}
    )
    assert lifecycle.request_cancel("run-1").state == "CANCELLED"
    assert _lifecycle(tmp_path, plan, clock).status().state == "CANCELLED"


def test_new_run_removes_orphaned_staged_candidate(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    _collect_all(lifecycle, "run-1")
    lifecycle.stage_candidate(
        "run-1", parameters={"bias": [1] * 6}, quality={"passed": True}
    )
    staged = lifecycle.staged_candidate_path.read_bytes()
    lifecycle.fail("run-1", "peer failed")
    lifecycle.staged_candidate_path.write_bytes(staged)  # crash-window residue

    restarted = _lifecycle(tmp_path, plan, clock)
    assert restarted.status().state == "FAILED"
    restarted.start("run-2", force_recalibrate=True)
    assert not restarted.staged_candidate_path.exists()
    assert _lifecycle(tmp_path, plan, clock).status().state == "IDLE"


def test_committed_run_id_cannot_be_reused(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    _collect_all(lifecycle, "run-1")
    lifecycle.stage_candidate(
        "run-1", parameters={"bias": [1] * 6}, quality={"passed": True}
    )
    lifecycle.commit("run-1", plan.plan_sha256)
    lifecycle.start("run-2", force_recalibrate=True)
    lifecycle.request_cancel("run-2")
    lifecycle.finish_cancel("run-2")
    with pytest.raises(LifecycleError, match="immutable result"):
        lifecycle.start("run-1", force_recalibrate=False)


@pytest.mark.parametrize("terminal", ["FAILED", "CANCELLED"])
def test_failed_and_cancelled_run_ids_cannot_be_recycled(
    tmp_path: Path,
    plan: BilateralCalibrationPlan,
    clock: Clock,
    terminal: str,
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("recycled", force_recalibrate=True)
    if terminal == "FAILED":
        lifecycle.fail("recycled", "failed")
    else:
        lifecycle.request_cancel("recycled")
        lifecycle.finish_cancel("recycled")
    historical = lifecycle.get_result("recycled")
    assert historical == {
        "available": False,
        "run_id": "recycled",
        "message": "failed" if terminal == "FAILED" else "calibration cancelled",
    }

    lifecycle.start("intervening", force_recalibrate=True)
    lifecycle.request_cancel("intervening")
    lifecycle.finish_cancel("intervening")
    with pytest.raises(LifecycleError, match="immutable result"):
        lifecycle.start("recycled", force_recalibrate=True)


@pytest.mark.parametrize("operation", ["commit", "cancel"])
def test_terminal_write_failure_is_retryable_and_restart_safe(
    tmp_path: Path,
    plan: BilateralCalibrationPlan,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    _collect_all(lifecycle, "run-1")
    lifecycle.stage_candidate(
        "run-1", parameters={"bias": [0] * 6}, quality={"passed": True}
    )
    original_write = lifecycle_module._atomic_write_json
    failed_once = False

    def fail_terminal_once(path: Path, payload: object) -> None:
        nonlocal failed_once
        if path == lifecycle.terminal_state_path and not failed_once:
            failed_once = True
            raise OSError("injected terminal write failure")
        original_write(path, payload)  # type: ignore[arg-type]

    monkeypatch.setattr(lifecycle_module, "_atomic_write_json", fail_terminal_once)
    with pytest.raises(OSError, match="injected"):
        if operation == "commit":
            lifecycle.commit("run-1", plan.plan_sha256)
        else:
            lifecycle.request_cancel("run-1")
    assert lifecycle.status().state == "READY_TO_APPLY"

    if operation == "commit":
        assert lifecycle.commit("run-1", plan.plan_sha256).state == "SUCCEEDED"
        expected = "SUCCEEDED"
    else:
        assert lifecycle.request_cancel("run-1").state == "CANCELLED"
        expected = "CANCELLED"
    assert lifecycle.terminal_state_path.exists()
    assert _lifecycle(tmp_path, plan, clock).status().state == expected


def test_empty_cancel_message_is_rejected_without_state_change(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    lifecycle.request_cancel("run-1")
    with pytest.raises(ValueError, match="non-empty"):
        lifecycle.finish_cancel("run-1", message="")
    assert lifecycle.status().state == "CANCELLING"


def test_failed_start_cleanup_can_be_retried_without_false_success(
    tmp_path: Path,
    plan: BilateralCalibrationPlan,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    lifecycle.request_cancel("run-1")
    lifecycle.finish_cancel("run-1")
    original_unlink = lifecycle_module._durable_unlink
    failed_once = False

    def fail_once(path: Path) -> None:
        nonlocal failed_once
        if path == lifecycle.terminal_state_path and not failed_once:
            failed_once = True
            raise OSError("injected cleanup failure")
        original_unlink(path)

    monkeypatch.setattr(lifecycle_module, "_durable_unlink", fail_once)
    with pytest.raises(OSError, match="cleanup"):
        lifecycle.start("run-2", force_recalibrate=True)
    assert lifecycle.status().run_id == "run-1"
    assert lifecycle.status().state == "CANCELLED"
    assert lifecycle.start("run-2", force_recalibrate=True).state == "READY_FOR_STAGE"


@pytest.mark.parametrize("failure_target", ["active", "terminal", "history"])
def test_cache_reuse_partial_write_is_retryable(
    tmp_path: Path,
    plan: BilateralCalibrationPlan,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
    failure_target: str,
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("source", force_recalibrate=True)
    _collect_all(lifecycle, "source")
    lifecycle.stage_candidate(
        "source", parameters={"bias": [0] * 6}, quality={"passed": True}
    )
    lifecycle.commit("source", plan.plan_sha256)
    original_write = lifecycle_module._atomic_write_json
    target = {
        "active": lifecycle.active_cache_path,
        "terminal": lifecycle.terminal_state_path,
        "history": lifecycle._history_path("reuse"),
    }[failure_target]
    failed_once = False

    def fail_once(path: Path, payload: object) -> None:
        nonlocal failed_once
        if path == target and not failed_once:
            failed_once = True
            raise OSError(f"injected {failure_target} write failure")
        original_write(path, payload)  # type: ignore[arg-type]

    monkeypatch.setattr(lifecycle_module, "_atomic_write_json", fail_once)
    with pytest.raises(OSError, match="injected"):
        lifecycle.start("reuse", force_recalibrate=False)
    assert lifecycle.status().state == "IDLE"
    if failure_target == "terminal":
        assert lifecycle._history_path("reuse").exists()
    else:
        assert not lifecycle._history_path("reuse").exists()
    restarted = _lifecycle(tmp_path, plan, clock)
    assert restarted.start("reuse", force_recalibrate=False).state == "SUCCEEDED"
    assert restarted.get_result("reuse")["available"] is True


def test_partial_successful_start_cannot_be_recycled_after_intervening_run(
    tmp_path: Path,
    plan: BilateralCalibrationPlan,
    clock: Clock,
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("source", force_recalibrate=True)
    _collect_all(lifecycle, "source")
    lifecycle.stage_candidate(
        "source", parameters={"bias": [0] * 6}, quality={"passed": True}
    )
    lifecycle.commit("source", plan.plan_sha256)
    lifecycle.start("reuse", force_recalibrate=False)
    lifecycle.start("intervening", force_recalibrate=False)
    with pytest.raises(LifecycleError, match="immutable result"):
        lifecycle.start("reuse", force_recalibrate=True)


@pytest.mark.parametrize("failure_target", ["history", "terminal"])
def test_partial_cache_reuse_rejects_conflicting_force_retry(
    tmp_path: Path,
    plan: BilateralCalibrationPlan,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
    failure_target: str,
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("source", force_recalibrate=True)
    _collect_all(lifecycle, "source")
    lifecycle.stage_candidate(
        "source", parameters={"bias": [0] * 6}, quality={"passed": True}
    )
    lifecycle.commit("source", plan.plan_sha256)
    original_write = lifecycle_module._atomic_write_json
    failed_once = False

    target = {
        "history": lifecycle._history_path("reuse"),
        "terminal": lifecycle.terminal_state_path,
    }[failure_target]

    def fail_once(path: Path, payload: object) -> None:
        nonlocal failed_once
        if path == target and not failed_once:
            failed_once = True
            raise OSError(f"injected {failure_target} failure")
        original_write(path, payload)  # type: ignore[arg-type]

    monkeypatch.setattr(lifecycle_module, "_atomic_write_json", fail_once)
    with pytest.raises(OSError, match="injected"):
        lifecycle.start("reuse", force_recalibrate=False)
    restarted = _lifecycle(tmp_path, plan, clock)
    with pytest.raises(LifecycleError, match="conflicting Start retry"):
        restarted.start("reuse", force_recalibrate=True)
    assert restarted.start("reuse", force_recalibrate=False).state == "SUCCEEDED"


@pytest.mark.parametrize("failure_target", ["history", "terminal"])
def test_cancel_after_partial_commit_repairs_success(
    tmp_path: Path,
    plan: BilateralCalibrationPlan,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
    failure_target: str,
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    _collect_all(lifecycle, "run-1")
    lifecycle.stage_candidate(
        "run-1", parameters={"bias": [0] * 6}, quality={"passed": True}
    )
    original_write = lifecycle_module._atomic_write_json
    target = {
        "history": lifecycle._history_path("run-1"),
        "terminal": lifecycle.terminal_state_path,
    }[failure_target]
    failed_once = False

    def fail_once(path: Path, payload: object) -> None:
        nonlocal failed_once
        if path == target and not failed_once:
            failed_once = True
            raise OSError(f"injected {failure_target} failure")
        original_write(path, payload)  # type: ignore[arg-type]

    monkeypatch.setattr(lifecycle_module, "_atomic_write_json", fail_once)
    with pytest.raises(OSError, match="injected"):
        lifecycle.commit("run-1", plan.plan_sha256)
    repaired = lifecycle.request_cancel("run-1")
    assert repaired.state == "SUCCEEDED"
    assert lifecycle.get_result()["available"] is True
    assert lifecycle.get_result("run-1")["available"] is True
    assert _lifecycle(tmp_path, plan, clock).status().state == "SUCCEEDED"


def test_stale_run_cancel_and_result_are_isolated(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    with pytest.raises(LifecycleError) as stale:
        lifecycle.request_cancel("run-old")
    assert stale.value.code == "NOT_FOUND"
    with pytest.raises(LifecycleError) as missing:
        lifecycle.get_result("run-old")
    assert missing.value.code == "NOT_FOUND"


def test_cancel_wins_race_against_late_worker_update(
    tmp_path: Path, plan: BilateralCalibrationPlan, clock: Clock
) -> None:
    lifecycle = _lifecycle(tmp_path, plan, clock)
    lifecycle.start("run-1", force_recalibrate=True)
    lifecycle.advance("run-1", 0, "tilt-x")
    assert lifecycle.request_cancel("run-1").state == "CANCELLING"
    with pytest.raises(LifecycleError, match="stale calibration worker"):
        lifecycle.set_phase("run-1", "COLLECTING")
    assert lifecycle.status().state == "CANCELLING"
