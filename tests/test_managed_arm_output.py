"""Arm output remains independent of slow RPCs and body scheduling."""
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import grpc

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'src/dexcontrol')]
from dexcontrol.core.robotenv_vega.body_server import ManagedArm
from dexcontrol.core.robotenv_vega.server import VegaRobotEnvService
from dexcontrol.core.robotenv_vega.control_node import VegaControlNode
from test_control_runtime import Hardware, arm_mode
from dexcontrol.core.robotenv_vega.vega_force_calibration import VegaForceCalibrationManager
from proto import robotenv_pb2
from test_vega_force_calibration_manager import FakeAdapter, _lifecycle, _wait_state


def service(control):
    arm = ManagedArm.__new__(ManagedArm)
    arm.control = control
    arm.arm_side = 'left'
    arm._cmd_lock = threading.RLock()
    arm._control_loop_hz = 200
    arm._control_loop_thread = None
    arm._control_loop_stop = threading.Event()
    arm._cancel_move = threading.Event()
    arm._force_calibration_admission_lock = threading.Lock()
    arm._force_calibration_owns_robot = threading.Event()
    arm.sent = []
    arm._robot = SimpleNamespace(execute_interpolated_tick=lambda **kwargs: arm.sent.append(time.monotonic()))
    return arm


def test_slow_arm_step_does_not_block_either_output_loop(monkeypatch):
    hardware = Hardware()
    control = VegaControlNode(hardware, hardware.config)
    arm_mode(control)
    left, right = service(control), service(control)
    right.arm_side = 'right'
    entered, release, right_done = threading.Event(), threading.Event(), threading.Event()
    def step(self, request, context):
        if self.arm_side == 'left':
            entered.set()
            release.wait(2)
        else:
            right_done.set()
        return SimpleNamespace(status='SUCCESS')
    monkeypatch.setattr(VegaRobotEnvService, 'Step', step)
    stop_tick = threading.Event()
    hardware.arm_tick = lambda: [arm._robot.execute_interpolated_tick(raise_errors=True) for arm in (left, right)]
    def tick():
        while not stop_tick.wait(.005):
            control.tick()
    output = threading.Thread(target=tick)
    output.start()
    blocked = threading.Thread(target=left.Step, args=(None, None))
    other = threading.Thread(target=right.Step, args=(None, None))
    try:
        blocked.start()
        assert entered.wait(1)
        counts = len(left.sent), len(right.sent)
        other.start()
        assert right_done.wait(.5)
        time.sleep(.1)
        assert len(left.sent) > counts[0] + 5
        assert len(right.sent) > counts[1] + 5
    finally:
        release.set();blocked.join(2);other.join(2)
        stop_tick.set();output.join(2)
        control.stop()


def test_arm_step_failure_retains_original_response(monkeypatch):
    hardware = Hardware()
    control = VegaControlNode(hardware, hardware.config)
    arm_mode(control)
    arm = service(control)
    failure = SimpleNamespace(status='IK_FAILED', message='Unreachable target')
    monkeypatch.setattr(VegaRobotEnvService, 'Step', lambda *args: failure)
    assert arm.Step(None, None) is failure
    assert control.mode == 'arms'


class CalibrationContext:
    code = None
    details = ''

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def calibration_service(tmp_path, mode='arms'):
    hardware = Hardware()
    control = None if mode is None else VegaControlNode(hardware, hardware.config)
    if control is not None:
        control.start()
        control.mode = mode
    arm = service(control)
    adapter = FakeAdapter()
    manager = VegaForceCalibrationManager(
        lifecycle=_lifecycle(tmp_path),
        adapter=adapter,
        claim_robot=arm._claim_robot_for_force_calibration,
        release_robot=arm._release_robot_from_force_calibration,
    )
    arm._force_calibration_manager = manager
    return arm, manager, adapter


def start_calibration(arm):
    context = CalibrationContext()
    response = arm.StartForceCalibration(
        robotenv_pb2.StartForceCalibrationRequest(
            run_id='managed-calibration', force_recalibrate=True,
            gripper_mass_kg=1.2, gripper_com_sensor_m=[0, 0, .072],
        ), context,
    )
    return response, context


@pytest.mark.parametrize('mode', ['hold', 'body', 'transition', 'arm_ready'])
def test_calibration_start_is_rejected_outside_arm_mode(tmp_path, mode):
    arm, manager, adapter = calibration_service(tmp_path, mode)
    try:
        _, context = start_calibration(arm)
        assert context.code == grpc.StatusCode.FAILED_PRECONDITION
        assert not manager.owns_robot
        assert adapter.executions == []
    finally:
        manager.shutdown()


@pytest.mark.parametrize('mode', ['arms', None])
def test_calibration_can_claim_arms_and_restore_before_control_assignment(tmp_path, mode):
    arm, manager, adapter = calibration_service(tmp_path, mode)
    try:
        response, context = start_calibration(arm)
        assert context.code is None
        assert response.state == robotenv_pb2.FORCE_CALIBRATION_STATE_READY_FOR_STAGE
        assert manager.owns_robot
        assert adapter.executions == []
    finally:
        manager.shutdown()


def test_stop_cancels_calibration_owning_arm_lock_without_return_motion(tmp_path):
    arm, manager, adapter = calibration_service(tmp_path)
    adapter.block = True
    execute = adapter.execute_stage

    def execute_locked(*args):
        with arm._cmd_lock:
            return execute(*args)

    adapter.execute_stage = execute_locked
    try:
        start_calibration(arm)
        manager.advance('managed-calibration', 0, 'stage-0')
        assert adapter.entered.wait(1)
        arm.control.mode = 'hold'
        arm._cancel_move.set()
        arm._cancel_force_calibration_motion()
        _wait_state(manager, 'CANCELLED')
        assert adapter.safe_returns == 0
        assert not manager.owns_robot
        assert arm._cmd_lock.acquire(blocking=False)
        arm._cmd_lock.release()
    finally:
        adapter.release.set()
        manager.shutdown()


def test_stop_preserves_completed_calibration_for_bilateral_commit(tmp_path):
    arm, manager, adapter = calibration_service(tmp_path)
    try:
        start_calibration(arm)
        manager.advance('managed-calibration', 0, 'stage-0')
        _wait_state(manager, 'READY_FOR_STAGE')
        manager.advance('managed-calibration', 1, 'stage-1')
        ready = _wait_state(manager, 'READY_TO_APPLY')
        arm.control.mode = 'hold'
        arm._cancel_force_calibration_motion()
        assert manager.status().state == 'READY_TO_APPLY'
        committed = manager.commit('managed-calibration', ready.plan_sha256)
        assert committed.state == 'SUCCEEDED'
        assert adapter.safe_returns == 0
    finally:
        manager.shutdown()
