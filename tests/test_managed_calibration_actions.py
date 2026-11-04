"""Managed calibration keeps its RPC protocol inside the common action lifecycle."""

import threading
from types import MethodType

import pytest
from test_control_runtime import Hardware
from test_managed_arm_output import CalibrationContext, service, start_calibration
from test_vega_force_calibration_manager import FakeAdapter, _lifecycle, _wait_state

from dexcontrol.core.control.fsm import ControlConflict
from dexcontrol.core.robotenv_vega.control_node import VegaControlNode
from dexcontrol.core.robotenv_vega.managed_hardware import ManagedVegaHardware
from dexcontrol.core.robotenv_vega.vega_force_calibration import (
    VegaForceCalibrationManager,
    VegaRobotCalibrationAdapter,
)
from proto import robotenv_pb2 as pb

RUN = "managed-calibration"


@pytest.fixture
def calibration(tmp_path):
    hardware = Hardware()
    node = VegaControlNode(hardware, hardware.config)
    node.start()
    node.mode = "arms"
    arms, managers, adapters = [], [], []
    for side in ("left", "right"):
        directory = tmp_path / side
        directory.mkdir()
        arm = service(node)
        arm.arm_side = side
        arm.buffer = None
        device = getattr(hardware, side)

        def reset(arm=arm):
            arm.buffer = None

        def tick(arm=arm, device=device, **kwargs):
            if arm.buffer is not None:
                device.command(arm.buffer)

        arm._robot.reset_filter_state = reset
        arm._robot.execute_interpolated_tick = tick
        adapter = FakeAdapter()
        manager = VegaForceCalibrationManager(
            lifecycle=_lifecycle(directory),
            adapter=adapter,
            claim_robot=arm._claim_robot_for_force_calibration,
            release_robot=arm._release_robot_from_force_calibration,
        )
        arm._force_calibration_manager = manager
        arms.append(arm)
        managers.append(manager)
        adapters.append(adapter)
    hardware.services = arms
    hardware.calibration_tick = MethodType(
        ManagedVegaHardware.calibration_tick, hardware
    )
    yield node, hardware, arms, managers, adapters
    node.stop()
    for adapter in adapters:
        adapter.release.set()
    node.actions.join()
    for manager in managers:
        manager.shutdown()


def start_both(arms):
    for arm in arms:
        response, context = start_calibration(arm)
        assert context.code is None, context.details
        assert response.state == pb.FORCE_CALIBRATION_STATE_READY_FOR_STAGE


def advance(arm, index):
    context = CalibrationContext()
    arm.AdvanceForceCalibration(
        pb.AdvanceForceCalibrationRequest(
            run_id=RUN, stage_index=index, stage_id=f"stage-{index}"
        ),
        context,
    )
    assert context.code is None, context.details


def prepare_candidate(arm, manager):
    advance(arm, 0)
    _wait_state(manager, "READY_FOR_STAGE")
    advance(arm, 1)
    return _wait_state(manager, "READY_TO_APPLY")


def commit(arm, manager):
    context = CalibrationContext()
    response = arm.CommitForceCalibration(
        pb.CommitForceCalibrationRequest(
            run_id=RUN, plan_sha256=manager.lifecycle.plan.plan_sha256
        ),
        context,
    )
    assert context.code is None, context.details
    assert response.state == pb.FORCE_CALIBRATION_STATE_SUCCEEDED


def test_bilateral_calibration_waits_for_both_commits_and_returns_to_hold(calibration):
    node, _, arms, managers, adapters = calibration
    start_both(arms)
    assert node.state()["action"] == {
        "name": "force_calibration",
        "state": "running",
        "error": "",
    }
    with pytest.raises(ControlConflict):
        node.start_action("initial_pose")
    with pytest.raises(ControlConflict):
        node.select("body")

    for adapter in adapters:
        adapter.block = True
    for arm in arms:
        advance(arm, 0)
    # Neither arm has to finish its stage before the other can enter it.
    assert all(adapter.entered.wait(1) for adapter in adapters)
    for adapter in adapters:
        adapter.release.set()
    for manager in managers:
        _wait_state(manager, "READY_FOR_STAGE")
    for arm, manager in zip(arms, managers):
        advance(arm, 1)
        _wait_state(manager, "READY_TO_APPLY")

    commit(arms[0], managers[0])
    node.tick()
    assert node.mode == "action"
    assert node.state()["action"]["state"] == "running"
    commit(arms[1], managers[1])
    node.actions.join()
    node.tick()
    assert node.mode == "hold"
    assert node.state()["action"]["state"] == "succeeded"
    assert all(adapter.normal_returns == 1 for adapter in adapters)
    assert all(manager.result(RUN)["available"] for manager in managers)


@pytest.mark.parametrize("phase", ["waiting", "moving", "ready"])
def test_stop_cancels_both_participants_without_return_motion(calibration, phase):
    node, hardware, arms, managers, adapters = calibration
    start_both(arms)
    if phase == "moving":
        for arm, adapter in zip(arms, adapters):
            adapter.block = True
            advance(arm, 0)
            assert adapter.entered.wait(1)
    elif phase == "ready":
        for arm, manager in zip(arms, managers):
            prepare_candidate(arm, manager)
    for arm in arms:
        arm.buffer = 0.4
    node.tick()
    assert hardware.left.sent[-1]["pos"][0] == 0.4
    node.stop()
    node.actions.join()
    node.tick()
    assert node.state()["control_state"] == "idle"
    assert node.state()["action"]["state"] == "cancelled"
    assert all(not manager.owns_robot for manager in managers)
    assert all(adapter.safe_returns == 0 for adapter in adapters)
    assert hardware.left.sent[-1]["pos"][0] == 0
    assert hardware.right.sent[-1]["pos"][0] == 0
    for manager in managers:
        assert manager.status().state == (
            "READY_TO_APPLY" if phase == "ready" else "CANCELLED"
        )
    if phase == "ready":
        node.start()
        node.start_action("initial_pose")
        for arm, manager in zip(arms, managers):
            commit(arm, manager)
        # A late bilateral commit changes the saved calibration only.
        assert node.mode == "action"
        assert node.state()["action"]["name"] == "initial_pose"


def test_failure_in_one_arm_faults_action_and_cancels_the_other(calibration):
    node, _, arms, managers, adapters = calibration
    start_both(arms)
    adapters[1].block = True
    advance(arms[1], 0)
    assert adapters[1].entered.wait(1)

    def fail_fit(holds):
        raise RuntimeError("Calibration quality rejected")

    adapters[0].fit = fail_fit
    advance(arms[0], 0)
    _wait_state(managers[0], "READY_FOR_STAGE")
    advance(arms[0], 1)
    _wait_state(managers[0], "FAILED")
    node.tick()
    node.actions.join()
    assert node.state()["control_state"] == "fault"
    assert node.state()["fault_reason"] == "action_error"
    assert node.state()["action"]["state"] == "failed"
    assert "Calibration quality rejected" in node.state()["action"]["error"]
    assert managers[1].status().state == "CANCELLED"
    assert adapters[1].safe_returns == 0


def test_safety_rejection_cancels_calibration_and_preserves_running_authorization(
    calibration,
):
    node, hardware, arms, managers, adapters = calibration
    start_both(arms)
    for arm, adapter in zip(arms, adapters):
        adapter.block = True
        advance(arm, 0)
        assert adapter.entered.wait(1)
    hardware.clear_motion_targets = lambda rejected: None

    def reject(commands, effects, owner, arm_mode=False):
        assert not arm_mode  # Calibration is an action, not manual arm control.
        return "rejected"

    node.motion_safety.filter = reject
    node.motion_safety.snapshot = lambda *args: {
        "violations": [],
        "reason": "Unsafe endpoint",
    }
    node.tick()
    node.actions.join()
    assert node.mode == "hold"
    assert node.state()["control_state"] == "running"
    assert node.state()["action"]["state"] == "cancelled"
    assert all(manager.status().state == "CANCELLED" for manager in managers)
    assert all(adapter.safe_returns == 0 for adapter in adapters)


def test_rpc_cancel_retains_existing_safe_return_behavior(calibration):
    node, _, arms, managers, adapters = calibration
    start_calibration(arms[0])
    adapters[0].block = True
    advance(arms[0], 0)
    assert adapters[0].entered.wait(1)
    context = CalibrationContext()
    arms[0].CancelForceCalibration(
        pb.CancelForceCalibrationRequest(run_id=RUN), context
    )
    assert context.code is None, context.details
    _wait_state(managers[0], "CANCELLED")
    node.actions.join()
    node.tick()
    assert node.mode == "hold"
    assert node.state()["action"]["state"] == "cancelled"
    assert adapters[0].safe_returns == 1


def test_direct_calibration_reset_suspends_only_that_arms_interpolation(calibration):
    node, hardware, arms, _, _ = calibration
    start_both(arms)
    arms[0].buffer = 0.4
    arms[1].buffer = -0.4
    entered, release = threading.Event(), threading.Event()

    def reset_sequence(target, **kwargs):
        hardware.left.command(0.2)
        entered.set()
        assert release.wait(1)

    arms[0]._execute_reset_sequence = reset_sequence
    adapter = VegaRobotCalibrationAdapter(arms[0], 1.2, (0, 0, 0.072))
    action = arms[0]._force_calibration_manager.action
    worker = action.worker(
        target=adapter._execute_reset, args=([0.0] * 7, threading.Event())
    )
    worker.start()
    try:
        assert entered.wait(1)
        node.tick()
        assert hardware.left.sent[-1]["pos"][0] == 0.2
        assert hardware.right.sent[-1]["pos"][0] == -0.4
    finally:
        release.set()
        worker.join(1)
    node.tick()
    assert hardware.left.sent[-1]["pos"][0] == 0.2


def test_stop_waits_for_calibration_cleanup_before_restart(calibration):
    node, _, arms, managers, adapters = calibration
    start_calibration(arms[0])
    entered, release = threading.Event(), threading.Event()

    def slow_stage(stage_id, payload, cancelled, set_phase):
        entered.set()
        cancelled.wait(1)
        assert release.wait(1)

    adapters[0].execute_stage = slow_stage
    advance(arms[0], 0)
    assert entered.wait(1)
    try:
        node.stop()
        with pytest.raises(ControlConflict):
            node.start()
        assert node.state()["action"]["state"] == "cancelling"
    finally:
        release.set()
    node.actions.join()
    assert managers[0].status().state == "CANCELLED"
    node.start()
    assert node.state()["control_state"] == "running"
