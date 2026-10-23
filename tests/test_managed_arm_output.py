"""Arm output remains independent of slow RPCs and body scheduling."""
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip('dexmotion')
ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'src/dexcontrol')]
from core.robotenv_vega.body_server import ManagedArm, arm_options, parse_args
from core.robotenv_vega.server import VegaRobotEnvService
from core.robotenv_vega.body_control import BodyControl


def service(control):
    arm = ManagedArm.__new__(ManagedArm)
    arm.control = control
    arm.arm_side = 'left'
    arm._cmd_lock = threading.RLock()
    arm._control_loop_hz = 200
    arm._control_loop_thread = None
    arm._control_loop_stop = threading.Event()
    arm.sent = []
    arm._robot = SimpleNamespace(execute_interpolated_tick=lambda: arm.sent.append(time.monotonic()))
    return arm


def test_slow_arm_step_does_not_block_either_output_loop(monkeypatch):
    control = BodyControl.__new__(BodyControl)
    control.mode = 'arms'
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
    left._start_control_loop();right._start_control_loop()
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
        left._stop_control_loop();right._stop_control_loop()


def test_arm_step_failure_retains_original_response(monkeypatch):
    control = BodyControl.__new__(BodyControl)
    control.mode = 'arms'
    arm = service(control)
    failure = SimpleNamespace(status='IK_FAILED', message='Unreachable target')
    monkeypatch.setattr(VegaRobotEnvService, 'Step', lambda *args: failure)
    assert arm.Step(None, None) is failure
    assert control.mode == 'arms'
