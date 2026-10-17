import sys
import threading
import logging
from types import ModuleType, SimpleNamespace

import grpc
import numpy as np
import pytest

# Keep this server test independent of the optional DualSense/hidapi stack.
_base_arm_teleop = ModuleType("base_arm_teleop")
_base_arm_teleop.BaseIKController = object
sys.modules.setdefault("base_arm_teleop", _base_arm_teleop)

import dexcontrol.core.robotenv_vega.server as server_module  # noqa: E402
from dexcontrol.core.robotenv_vega.server import (  # noqa: E402
    VegaRobotEnvService,
    _shutdown_service_until_safe,
)


def test_serve_defers_startup_signal_until_service_has_cleanup_owner(
    monkeypatch, tmp_path
):
    handlers = {}
    calls = []

    class Service:
        def __init__(self, **_kwargs):
            calls.append("service-created")
            handlers[server_module.signal.SIGTERM](server_module.signal.SIGTERM, None)

        def shutdown(self):
            calls.append("service-shutdown")

    monkeypatch.setattr(server_module, "VegaRobotEnvService", Service)
    monkeypatch.setattr(
        server_module.signal, "getsignal", lambda signum: f"old-{signum}"
    )
    monkeypatch.setattr(
        server_module.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(
        server_module.signal, "pthread_sigmask", lambda *_args: set()
    )
    monkeypatch.setattr(
        server_module.os.path,
        "expanduser",
        lambda _path: str(tmp_path / "robotenv_vega.log"),
    )

    with pytest.raises(SystemExit):
        server_module.serve(arm_side="left")

    assert calls == ["service-created", "service-shutdown"]


def test_serve_does_not_acquire_hardware_after_bootstrap_shutdown(
    monkeypatch, tmp_path
):
    handlers = {}
    calls = []

    class Handler(logging.Handler):
        def __init__(self, _path):
            super().__init__()
            handlers[server_module.signal.SIGTERM](server_module.signal.SIGTERM, None)

    class Service:
        def __init__(self, **_kwargs):
            calls.append("service-created")

    circular_module = sys.modules["core.robotenv_vega.circular_file_handler"]
    monkeypatch.setattr(circular_module, "CircularFileHandler", Handler)
    monkeypatch.setattr(server_module, "VegaRobotEnvService", Service)
    monkeypatch.setattr(
        server_module.signal, "getsignal", lambda signum: f"old-{signum}"
    )
    monkeypatch.setattr(
        server_module.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(
        server_module.signal, "pthread_sigmask", lambda *_args: set()
    )
    monkeypatch.setattr(
        server_module.os.path,
        "expanduser",
        lambda _path: str(tmp_path / "robotenv_vega.log"),
    )

    with pytest.raises(SystemExit):
        server_module.serve(arm_side="left")

    assert calls == []


class _Robot:
    interpolation_enabled = False

    def __init__(self):
        self.commands = []
        self.closed = False

    def update_command(self, action, **kwargs):
        self.commands.append((np.asarray(action).copy(), kwargs))

    def close(self):
        self.closed = True


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
    service._shutdown_started = threading.Event()
    service._shutdown_lock = threading.Lock()
    service._shutdown_completed = False
    service._transport_rpc_condition = threading.Condition()
    service._active_transport_rpcs = 0
    service._control_loop_hz = 0
    service._robot = _Robot()
    service._control_loop_thread = None
    service._force_calibration_owner_lock = threading.Lock()
    service._force_calibration_owns_robot = threading.Event()
    return service


class _UnstoppableThread:
    def __init__(self):
        self.join_calls = []

    def is_alive(self):
        return True

    def join(self, timeout=None):
        self.join_calls.append(timeout)


def test_control_loop_stop_keeps_live_thread_handle_and_fails_closed():
    service = _service()
    service._control_loop_stop = threading.Event()
    thread = _UnstoppableThread()
    service._control_loop_thread = thread

    with pytest.raises(RuntimeError, match="did not stop"):
        service._stop_control_loop()

    assert service._control_loop_thread is thread
    assert service._control_loop_stop.is_set()
    assert thread.join_calls == [2.0]


@pytest.mark.parametrize(
    "control_hz", [0, -20, 0.5, 20.5, float("nan"), False, np.bool_(True)]
)
def test_service_rejects_invalid_control_hz_before_robot_creation(
    monkeypatch, control_hz
):
    robot_calls = []
    monkeypatch.setattr(
        server_module, "VegaRobot", lambda **kwargs: robot_calls.append(kwargs)
    )

    with pytest.raises(ValueError, match="control_hz must be a finite positive integer"):
        VegaRobotEnvService(arm_side="left", control_hz=control_hz)

    assert robot_calls == []


def test_service_uses_normalized_control_hz_after_validation(monkeypatch):
    calls = []

    class StopConstruction(Exception):
        pass

    class Robot:
        def __init__(self, **kwargs):
            calls.append(kwargs["control_hz"])
            raise StopConstruction

    monkeypatch.setattr(server_module, "VegaRobot", Robot)

    with pytest.raises(StopConstruction):
        VegaRobotEnvService(arm_side="left", control_hz="20")

    assert calls == [20]


@pytest.mark.parametrize(
    "control_loop_hz",
    [-1, 0.5, float("nan"), float("inf"), False, np.bool_(True)],
)
def test_service_rejects_invalid_control_loop_hz_before_robot_creation(
    monkeypatch, control_loop_hz
):
    robot_calls = []
    monkeypatch.setattr(
        server_module,
        "VegaRobot",
        lambda **kwargs: robot_calls.append(kwargs),
    )

    with pytest.raises(
        ValueError, match="control_loop_hz must be a finite non-negative integer"
    ):
        VegaRobotEnvService(arm_side="left", control_loop_hz=control_loop_hz)

    assert robot_calls == []


@pytest.mark.parametrize("control_loop_hz", ["0", "100", 100.0, np.int64(100)])
def test_service_normalizes_control_loop_hz_before_robot_creation(
    monkeypatch, control_loop_hz
):
    class StopConstruction(Exception):
        pass

    class Robot:
        def __init__(self, **kwargs):
            raise StopConstruction

    monkeypatch.setattr(server_module, "VegaRobot", Robot)

    with pytest.raises(StopConstruction):
        VegaRobotEnvService(arm_side="left", control_loop_hz=control_loop_hz)


@pytest.mark.parametrize("ema_alpha", [1.0, "1.0", 2.0])
def test_service_normalizes_ema_alpha_before_robot_creation(monkeypatch, ema_alpha):
    calls = []

    class StopConstruction(Exception):
        pass

    class Robot:
        def __init__(self, **kwargs):
            calls.append(kwargs["ema_alpha"])
            raise StopConstruction

    monkeypatch.setattr(server_module, "VegaRobot", Robot)

    with pytest.raises(StopConstruction):
        VegaRobotEnvService(arm_side="left", ema_alpha=ema_alpha)

    assert calls == [0.99]


@pytest.mark.parametrize("ema_alpha", [float("nan"), float("inf"), "bad"])
def test_service_rejects_invalid_ema_alpha_before_robot_creation(
    monkeypatch, ema_alpha
):
    robot_calls = []
    monkeypatch.setattr(
        server_module,
        "VegaRobot",
        lambda **kwargs: robot_calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="ema_alpha must be finite"):
        VegaRobotEnvService(arm_side="left", ema_alpha=ema_alpha)

    assert robot_calls == []


def test_force_calibration_claim_rolls_back_if_control_loop_does_not_stop():
    service = _service()
    service._control_loop_stop = threading.Event()
    service._control_loop_thread = _UnstoppableThread()

    with pytest.raises(RuntimeError, match="did not stop"):
        service._claim_robot_for_force_calibration()

    assert service._force_calibration_owns_robot.is_set() is False
    assert service._force_calibration_restart_control_loop is False


def test_shutdown_closes_admission_before_draining_and_rejects_motion_rpcs():
    service = _service()
    service._control_loop_stop = threading.Event()
    manager_calls = []
    service._force_calibration_manager = SimpleNamespace(
        shutdown=lambda: manager_calls.append(service._shutdown_started.is_set())
    )
    context = SimpleNamespace(code=None, details="")
    context.set_code = lambda code: setattr(context, "code", code)
    context.set_details = lambda details: setattr(context, "details", details)

    service.shutdown()

    assert manager_calls == [True]
    assert service._shutdown_completed is True
    assert service._robot.closed is True
    assert service._claim_robot_for_force_calibration() is False

    assert service.Reset(SimpleNamespace(mode="home", params={}), context).status == ""
    assert context.code == grpc.StatusCode.UNAVAILABLE
    context.code = None
    assert service.Step(SimpleNamespace(), context).status == ""
    assert context.code == grpc.StatusCode.UNAVAILABLE


def test_shutdown_waits_for_admitted_current_reset_observation_before_close():
    class Context:
        def __init__(self):
            self.code = None
            self.details = ""

        def set_code(self, code):
            self.code = code

        def set_details(self, details):
            self.details = details

    service = _service()
    service._control_loop_stop = threading.Event()
    service._force_calibration_manager = SimpleNamespace(shutdown=lambda: None)
    service._force_calibration_owns_robot.set()
    observation_started = threading.Event()
    release_observation = threading.Event()

    def observation():
        observation_started.set()
        assert release_observation.wait(timeout=1.0)
        return {}, 1

    service._create_observation = observation
    reset_response = {}
    reset_thread = threading.Thread(
        target=lambda: reset_response.setdefault(
            "value",
            service.Reset(SimpleNamespace(mode="current", params={}), Context()),
        )
    )
    shutdown_thread = threading.Thread(target=service.shutdown)

    reset_thread.start()
    assert observation_started.wait(timeout=1.0)
    shutdown_thread.start()
    threading.Event().wait(0.05)
    assert service._shutdown_started.is_set()
    assert service._robot.closed is False
    assert shutdown_thread.is_alive()

    release_observation.set()
    reset_thread.join(timeout=1.0)
    shutdown_thread.join(timeout=1.0)

    assert reset_response["value"].status == "SUCCESS"
    assert service._robot.closed is True
    assert service._active_transport_rpcs == 0


def test_shutdown_runner_retries_without_returning_after_fail_closed_error():
    calls = []
    sleeps = []

    class Service:
        def shutdown(self):
            calls.append("shutdown")
            if len(calls) == 1:
                raise RuntimeError("worker still active")

    _shutdown_service_until_safe(
        Service(), retry_delay_s=0.25, sleep_fn=sleeps.append
    )

    assert calls == ["shutdown", "shutdown"]
    assert sleeps == [0.25]


def test_step_dispatch_uses_normal_path_when_no_reset_owns_robot():
    service = _service()

    service._dispatch_step_command(
        np.ones(7),
        action_space="cartesian_delta",
        gripper_action_space="position",
        command_epoch=0,
    )

    assert len(service._robot.commands) == 1


def test_step_dispatch_is_rejected_while_force_calibration_owns_robot():
    service = _service()
    service._force_calibration_owns_robot.set()

    with pytest.raises(RuntimeError, match="force calibration"):
        service._dispatch_step_command(
            np.ones(7),
            action_space="cartesian_delta",
            gripper_action_space="position",
            command_epoch=0,
        )

    assert service._robot.commands == []


def test_motion_reset_is_rejected_while_force_calibration_owns_robot():
    service = _service()
    service._force_calibration_owns_robot.set()
    context = SimpleNamespace(code=None, details="")
    context.set_code = lambda code: setattr(context, "code", code)
    context.set_details = lambda details: setattr(context, "details", details)

    response = service.Reset(SimpleNamespace(mode="home", params={}), context)

    assert response.status == ""
    assert context.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "force calibration" in context.details
    assert service._reset_in_progress.is_set() is False


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


def test_current_reset_remains_available_without_waiting_for_motion_lock():
    class Context:
        def __init__(self):
            self.callback = None

        def add_callback(self, callback):
            self.callback = callback
            return True

        def set_code(self, code):
            self.code = code

        def set_details(self, details):
            self.details = details

    service = _service()
    service._force_calibration_owns_robot.set()
    service._create_observation = lambda: ({}, 1)
    context = Context()
    result = SimpleNamespace(response=None)
    service._cmd_lock.acquire()

    def reset():
        result.response = service.Reset(SimpleNamespace(mode="current", params={}), context)

    worker = threading.Thread(target=reset)
    worker.start()
    worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert result.response.status == "SUCCESS"
    assert service._reset_in_progress.is_set() is False
    service._cmd_lock.release()


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


def test_reset_superseded_while_building_observation_cannot_return_success():
    class Context:
        def __init__(self):
            self.code = None
            self.details = None

        def add_callback(self, _callback):
            return True

        def set_code(self, code):
            self.code = code

        def set_details(self, details):
            self.details = details

    service = _service()
    observation_entered = threading.Event()
    release_first_observation = threading.Event()
    observation_calls = 0
    observation_lock = threading.Lock()

    def observation():
        nonlocal observation_calls
        with observation_lock:
            observation_calls += 1
            call = observation_calls
        if call == 1:
            observation_entered.set()
            assert release_first_observation.wait(timeout=1.0)
        return {}, call

    service._create_observation = observation
    first_context = Context()
    second_context = Context()
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
    assert observation_entered.wait(timeout=1.0)
    second.start()
    for _ in range(100):
        if service._active_reset_count == 2:
            break
        threading.Event().wait(0.01)
    assert service._active_reset_count == 2
    release_first_observation.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert first_context.code == grpc.StatusCode.ABORTED
    assert "superseded" in first_context.details
    assert responses["first"].status == ""
    assert second_context.code is None
    assert responses["second"].status == "SUCCESS"
