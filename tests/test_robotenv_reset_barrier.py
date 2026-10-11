import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

# Keep this server test independent of the optional DualSense/hidapi stack.
_base_arm_teleop = ModuleType("base_arm_teleop")
_base_arm_teleop.BaseIKController = object
sys.modules.setdefault("base_arm_teleop", _base_arm_teleop)

from dexcontrol.core.robotenv_vega.server import VegaRobotEnvService


class _Robot:
    interpolation_enabled = False

    def __init__(self):
        self.commands = []

    def update_command(self, action, **kwargs):
        self.commands.append((np.asarray(action).copy(), kwargs))


def _service():
    service = VegaRobotEnvService.__new__(VegaRobotEnvService)
    service._cmd_lock = threading.Lock()
    service._command_epoch_lock = threading.Lock()
    service._command_epoch = 0
    service._reset_in_progress = threading.Event()
    service._control_loop_hz = 0
    service._robot = _Robot()
    return service


def test_step_dispatch_uses_normal_path_when_no_reset_owns_robot():
    service = _service()

    service._dispatch_step_command(
        np.ones(7),
        action_space="cartesian_delta",
        gripper_action_space="position",
        command_epoch=0,
    )

    assert len(service._robot.commands) == 1


def test_reset_epoch_rejects_step_that_was_queued_before_reset():
    service = _service()
    entered = threading.Event()
    outcome = SimpleNamespace(error=None)
    service._cmd_lock.acquire()

    def dispatch():
        entered.set()
        try:
            service._dispatch_step_command(
                np.ones(7),
                action_space="cartesian_delta",
                gripper_action_space="position",
                command_epoch=0,
            )
        except Exception as exc:
            outcome.error = exc

    worker = threading.Thread(target=dispatch)
    worker.start()
    assert entered.wait(timeout=1.0)
    service._reset_in_progress.set()
    with service._command_epoch_lock:
        service._command_epoch += 1
    service._cmd_lock.release()
    worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert isinstance(outcome.error, RuntimeError)
    assert "superseded" in str(outcome.error)
    assert service._robot.commands == []


def test_step_is_rejected_while_reset_is_in_progress():
    service = _service()
    service._reset_in_progress.set()

    with pytest.raises(RuntimeError, match="superseded"):
        service._dispatch_step_command(
            np.ones(7),
            action_space="cartesian_delta",
            gripper_action_space="position",
            command_epoch=0,
        )

    assert service._robot.commands == []
