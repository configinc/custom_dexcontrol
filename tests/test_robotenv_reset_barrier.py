import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import grpc

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
    service.arm_side = "left"
    service._cmd_lock = threading.Lock()
    service._command_epoch_lock = threading.Lock()
    service._command_epoch = 0
    service._reset_in_progress = threading.Event()
    service._reset_state_lock = threading.Lock()
    service._active_reset_count = 0
    service._active_reset_cancel = None
    service._reset_restart_control_loop = False
    service._control_loop_hz = 0
    service._robot = _Robot()
    service._control_loop_thread = None
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


def test_cancelled_step_is_rejected_even_if_it_captures_post_reset_epoch():
    service = _service()
    service._command_epoch = 7
    rpc_cancelled = threading.Event()
    rpc_cancelled.set()

    with pytest.raises(RuntimeError, match="superseded"):
        service._dispatch_step_command(
            np.ones(7),
            action_space="cartesian_delta",
            gripper_action_space="position",
            command_epoch=7,
            rpc_cancelled=rpc_cancelled,
        )

    assert service._robot.commands == []


def test_uncancelled_step_admitted_during_reset_cannot_run_after_reset():
    service = _service()
    service._command_epoch = 1
    service._reset_in_progress.set()
    service._cmd_lock.acquire()
    outcome = SimpleNamespace(error=None)

    def dispatch():
        try:
            service._dispatch_step_command(
                np.ones(7),
                action_space="cartesian_delta",
                gripper_action_space="position",
                command_epoch=1,
                reset_active_at_entry=True,
            )
        except Exception as exc:
            outcome.error = exc

    worker = threading.Thread(target=dispatch)
    worker.start()
    service._reset_in_progress.clear()
    service._cmd_lock.release()
    worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert isinstance(outcome.error, RuntimeError)
    assert service._robot.commands == []


def test_step_admitted_during_reset_fails_before_reading_robot_state():
    service = _service()
    service._reset_in_progress.set()
    service._robot.hand = None
    service._robot.get_robot_state = lambda: (_ for _ in ()).throw(
        AssertionError("reset-active Step must fail before touching robot state")
    )
    request = SimpleNamespace(
        action_space="cartesian_delta",
        action=np.zeros(7),
        gripper_action_space="position",
    )
    context = SimpleNamespace(add_callback=lambda _callback: True)

    response = service.Step(request, context)

    assert response.status == "ERROR"
    assert "reset was active" in response.message
    assert service._robot.commands == []


def test_cancelled_reset_rpc_aborts_before_motion_after_lock_wait():
    class Context:
        def __init__(self):
            self.callback = None

        def add_callback(self, callback):
            self.callback = callback
            return True

        def set_code(self, _code):
            raise AssertionError("cancelled RPC must retain its cancellation status")

        def set_details(self, _details):
            raise AssertionError("cancelled RPC must retain its cancellation details")

    service = _service()
    service._create_observation = lambda: ({}, 1)
    context = Context()
    result = SimpleNamespace(response=None)
    service._cmd_lock.acquire()

    def reset():
        result.response = service.Reset(SimpleNamespace(mode="current", params={}), context)

    worker = threading.Thread(target=reset)
    worker.start()
    for _ in range(100):
        if context.callback is not None and service._reset_in_progress.is_set():
            break
        threading.Event().wait(0.01)
    assert context.callback is not None
    context.callback()
    service._cmd_lock.release()
    worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert result.response.status == ""
    assert service._reset_in_progress.is_set() is False


def test_overlapping_resets_keep_loop_paused_until_newest_reset_finishes():
    class Context:
        def add_callback(self, callback):
            self.callback = callback
            return True

        def set_code(self, _code):
            pass

        def set_details(self, _details):
            pass

    class RunningThread:
        def is_alive(self):
            return True

    service = _service()
    service._control_loop_thread = RunningThread()
    stop_calls = []
    start_calls = []
    entered_observation = threading.Event()
    release_observation = threading.Event()

    def stop_loop():
        stop_calls.append(True)
        service._control_loop_thread = None

    def start_loop():
        start_calls.append(True)
        service._control_loop_thread = RunningThread()

    def observation():
        entered_observation.set()
        assert service._reset_in_progress.is_set()
        assert start_calls == []
        assert release_observation.wait(timeout=1.0)
        return {}, 1

    service._stop_control_loop = stop_loop
    service._start_control_loop = start_loop
    service._create_observation = observation
    service._cmd_lock.acquire()
    results = []

    def reset(context):
        results.append(service.Reset(SimpleNamespace(mode="current", params={}), context))

    first = threading.Thread(target=reset, args=(Context(),))
    second = threading.Thread(target=reset, args=(Context(),))
    first.start()
    for _ in range(100):
        if service._active_reset_count == 1:
            break
        threading.Event().wait(0.01)
    second.start()
    for _ in range(100):
        if service._active_reset_count == 2:
            break
        threading.Event().wait(0.01)
    service._cmd_lock.release()
    assert entered_observation.wait(timeout=1.0)
    release_observation.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert first.is_alive() is False and second.is_alive() is False
    assert len(stop_calls) == 1
    assert len(start_calls) == 1
    assert service._reset_in_progress.is_set() is False


def test_superseded_reset_returns_aborted_instead_of_empty_success():
    class Context:
        def __init__(self):
            self.callback = None
            self.code = None
            self.details = None

        def add_callback(self, callback):
            self.callback = callback
            return True

        def set_code(self, code):
            self.code = code

        def set_details(self, details):
            self.details = details

    service = _service()
    service._create_observation = lambda: ({}, 1)
    first_context = Context()
    second_context = Context()
    service._cmd_lock.acquire()
    responses = {}

    first = threading.Thread(
        target=lambda: responses.setdefault(
            "first",
            service.Reset(SimpleNamespace(mode="current", params={}), first_context),
        )
    )
    second = threading.Thread(
        target=lambda: responses.setdefault(
            "second",
            service.Reset(SimpleNamespace(mode="current", params={}), second_context),
        )
    )
    first.start()
    for _ in range(100):
        if service._active_reset_count == 1:
            break
        threading.Event().wait(0.01)
    second.start()
    for _ in range(100):
        if service._active_reset_count == 2:
            break
        threading.Event().wait(0.01)
    service._cmd_lock.release()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert first_context.code == grpc.StatusCode.ABORTED
    assert "superseded" in first_context.details
    assert responses["first"].status == ""
    assert second_context.code is None
    assert responses["second"].status == "SUCCESS"
