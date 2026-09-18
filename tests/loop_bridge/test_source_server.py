"""Tests for source_server glue that would otherwise import live dexcontrol deps."""

from __future__ import annotations

import importlib
import sys
import threading
import time
import types
from contextlib import contextmanager

import pytest
from conftest import make_observation
from loop_sdk import ReceivedMessage, RobotCommand

from loop_bridge.contracts import (
    ROBOT_OBSERVATION_CONTRACT,
    float64_tensor,
    float64_values,
)


class _ProtoMap(dict[str, object]):
    def __missing__(self, key: str) -> object:
        value = _FakeValue()
        self[key] = value
        return value


class _StepRequest:
    def __init__(
        self,
        *,
        action: list[float],
        action_space: str,
        gripper_action_space: str,
    ) -> None:
        self.action = action
        self.action_space = action_space
        self.gripper_action_space = gripper_action_space
        self.pre_action_state = _ProtoMap()


class _ResetRequest:
    def __init__(self, *, mode: str, params: object) -> None:
        self.mode = mode
        self.params = params


class _FakeFloatArray:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _FakeValue:
    """Stand-in for robotenv_pb2.Value: exposes the set field + ``WhichOneof('kind')``."""

    def __init__(
        self,
        *,
        float_value: float | None = None,
        float_array: list[float] | _FakeFloatArray | None = None,
        int_value: int | None = None,
        string_value: str | None = None,
    ) -> None:
        self.float_value = float_value
        self.float_array = (
            float_array
            if isinstance(float_array, _FakeFloatArray)
            else _FakeFloatArray(float_array)
            if float_array is not None
            else None
        )
        self.int_value = int_value
        self.string_value = string_value
        self._kind = next(
            (
                name
                for name, val in (
                    ("float_value", float_value),
                    ("float_array", self.float_array),
                    ("int_value", int_value),
                    ("string_value", string_value),
                )
                if val is not None
            ),
            None,
        )

    def WhichOneof(self, oneof: str) -> str | None:
        assert oneof == "kind"
        return self._kind

    def CopyFrom(self, other: _FakeValue) -> None:
        self.float_value = other.float_value
        self.float_array = other.float_array
        self.int_value = other.int_value
        self.string_value = other.string_value
        self._kind = other._kind


class _FakeService:
    def __init__(
        self,
        *,
        status: str = "SUCCESS",
        message: str = "",
        action_info: dict[str, _FakeValue] | None = None,
    ) -> None:
        self.status = status
        self.message = message
        self.action_info = action_info or {}
        self.requests: list[_StepRequest] = []
        self.resets: list[_ResetRequest] = []
        self.paused = True
        self.closed = False
        self.queued_commands = []
        self.control_ticks = 0

    def resume(self, control_frequency_hz):
        self.paused = False
        self.control_frequency_hz = control_frequency_hz

    def execute_control_tick(self):
        self.control_ticks += 1

    def capture_state(self):
        from loop_bridge.robot_obs import observation_state

        return observation_state(make_observation())

    def observation_from_state(self, state):
        return state

    def pause(self):
        self.paused = True
        self.queued_commands.clear()

    def close(self):
        self.pause()
        self.closed = True

    def _create_observation(self) -> tuple[dict[str, object], int]:
        return make_observation(), 123

    def Step(self, request: _StepRequest, context: object) -> object:
        del context
        self.requests.append(request)
        return types.SimpleNamespace(
            status=self.status, message=self.message, action_info=self.action_info
        )

    def Reset(self, request: _ResetRequest, context: object) -> object:
        if not self.status:
            context.set_details(self.message)
        self.resets.append(request)
        return types.SimpleNamespace(status=self.status, message=self.message)


def _import_source_server(monkeypatch: pytest.MonkeyPatch) -> object:
    fake_server = types.ModuleType("dexcontrol.core.robotenv_vega.server")
    fake_server.robotenv_pb2 = types.SimpleNamespace(
        StepRequest=_StepRequest,
        ResetRequest=_ResetRequest,
        FloatArray=_FakeFloatArray,
        Value=_FakeValue,
    )
    fake_server.robotenv_pb2_grpc = types.SimpleNamespace(
        add_RobotEnvServicer_to_server=lambda *_: None
    )
    fake_server.VegaRobotEnvService = object

    fake_vega_robot = types.ModuleType("dexcontrol.core.vega.robot")
    fake_vega_robot.SUPPORTED_ACTION_SPACES = ("target_cartesian_delta",)

    monkeypatch.setitem(sys.modules, "dexcontrol", types.ModuleType("dexcontrol"))
    monkeypatch.setitem(
        sys.modules, "dexcontrol.core", types.ModuleType("dexcontrol.core")
    )
    monkeypatch.setitem(
        sys.modules,
        "dexcontrol.core.robotenv_vega",
        types.ModuleType("dexcontrol.core.robotenv_vega"),
    )
    monkeypatch.setitem(
        sys.modules, "dexcontrol.core.robotenv_vega.server", fake_server
    )
    monkeypatch.setitem(
        sys.modules, "dexcontrol.core.vega", types.ModuleType("dexcontrol.core.vega")
    )
    monkeypatch.setitem(sys.modules, "dexcontrol.core.vega.robot", fake_vega_robot)
    sys.modules.pop("loop_bridge.source_server", None)
    return importlib.import_module("loop_bridge.source_server")


def test_step_applier_sends_successful_step(monkeypatch: pytest.MonkeyPatch) -> None:
    source_server = _import_source_server(monkeypatch)
    service = _FakeService(status="SUCCESS")

    source_server._StepApplier(service).step(
        [1.0, 2.0], "target_cartesian_delta", "position"
    )

    assert len(service.requests) == 1
    assert service.requests[0].action == [1.0, 2.0]
    assert service.requests[0].action_space == "target_cartesian_delta"
    assert service.requests[0].gripper_action_space == "position"


def test_step_applier_forwards_the_exact_pre_action_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_server = _import_source_server(monkeypatch)
    service = _FakeService(status="SUCCESS")

    source_server._StepApplier(service).step(
        [0.0] * 7,
        "target_cartesian_delta",
        "",
        pre_apply_obs={
            "cartesian_position": [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            "gripper_position": 0.75,
        },
    )

    request = service.requests[0]
    assert request.pre_action_state["cartesian_position"].float_array.values == [
        1.0,
        2.0,
        3.0,
        0.1,
        0.2,
        0.3,
    ]
    assert request.pre_action_state["gripper_position"].float_value == 0.75


def test_step_applier_returns_decoded_action_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_server = _import_source_server(monkeypatch)
    service = _FakeService(
        status="SUCCESS",
        action_info={
            "desired_velocity": _FakeValue(float_array=[0.1, 0.2, 0.3]),
            "gripper_delta": _FakeValue(float_value=0.5),
            # state.* entries duplicate the published obs snapshot — must be dropped.
            "state.cartesian_position": _FakeValue(float_array=[9.0, 9.0]),
        },
    )

    result = source_server._StepApplier(service).step(
        [1.0], "target_cartesian_delta", ""
    )

    assert result == {"desired_velocity": [0.1, 0.2, 0.3], "gripper_delta": 0.5}


def test_step_applier_raises_on_non_success_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_server = _import_source_server(monkeypatch)
    service = _FakeService(status="IK_FAILED", message="unreachable target")

    with pytest.raises(RuntimeError, match="IK_FAILED"):
        source_server._StepApplier(service).step([1.0], "target_cartesian_delta", "")


def test_step_applier_home_sends_reset_home(monkeypatch: pytest.MonkeyPatch) -> None:
    source_server = _import_source_server(monkeypatch)
    service = _FakeService(status="SUCCESS")

    source_server._StepApplier(service).home()

    assert len(service.resets) == 1
    assert service.resets[0].mode == "home"


def test_step_applier_home_raises_on_non_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_server = _import_source_server(monkeypatch)
    service = _FakeService(status="HOME_FAILED", message="estopped")

    with pytest.raises(RuntimeError, match="HOME_FAILED"):
        source_server._StepApplier(service).home()


def test_action_uses_pre_action_state_and_observations_include_latest_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_server = _import_source_server(monkeypatch)
    left = _FakeService(
        action_info={"delta_action": _FakeValue(float_array=[0.01] * 7)}
    )
    right = _FakeService(
        action_info={"delta_action": _FakeValue(float_array=[0.02] * 7)}
    )
    node = source_server.VegaRobotNode([("left", left), ("right", right)])
    published: list[dict[str, object]] = []
    node.publish_observation = lambda payload, timestamp_ns: published.append(  # type: ignore[method-assign]
        dict(payload)
    )
    node._active = True

    left_delta = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    right_delta = (-0.1, -0.2, -0.3, -0.4, -0.5, -0.6)
    node._apply_action(
        ReceivedMessage(
            timestamp_ns=100,
            sequence=7,
            received_at_ns=110,
            payload={
                "left.target_cartesian_delta": float64_tensor(left_delta, shape=(6,)),
                "left.gripper_position": 0.25,
                "right.target_cartesian_delta": float64_tensor(right_delta, shape=(6,)),
                "right.gripper_position": 0.75,
            },
        )
    )

    assert left.requests[0].action == [*left_delta, 0.25]
    assert right.requests[0].action == [*right_delta, 0.75]
    assert left.requests[0].pre_action_state["joint_positions"].float_array.values == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]
    assert right.requests[0].pre_action_state["gripper_position"].float_value == 0.5
    assert published == []
    node._publish_snapshot(
        source_server._ObservationSnapshot(
            123,
            {"left": left.capture_state(), "right": right.capture_state()},
            node._latest_action_info,
            time.monotonic_ns(),
        )
    )
    assert len(published) == 1
    assert published[0]["left.gripper_position"] == 0.5
    assert published[0]["right.gripper_position"] == 0.5
    assert float64_values(
        published[0]["received_action"],  # type: ignore[arg-type]
        field_name="received_action",
        shape=(14,),
    ) == (*left_delta, 0.25, *right_delta, 0.75)
    assert (
        float64_values(
            published[0]["left.action.delta_action"],  # type: ignore[arg-type]
            field_name="left.action.delta_action",
            shape=(7,),
        )
        == (0.01,) * 7
    )
    assert (
        float64_values(
            published[0]["right.action.delta_action"],  # type: ignore[arg-type]
            field_name="right.action.delta_action",
            shape=(7,),
        )
        == (0.02,) * 7
    )


def test_dual_arm_node_home_uses_both_existing_reset_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_server = _import_source_server(monkeypatch)
    left = _FakeService()
    right = _FakeService()
    node = source_server.VegaRobotNode([("left", left), ("right", right)])
    node.publish_observation = lambda payload, timestamp_ns: None  # type: ignore[method-assign]
    node._active = True

    node._handle_command(source_server.RobotCommand.HOME)

    assert [reset.mode for reset in left.resets] == ["home"]
    assert [reset.mode for reset in right.resets] == ["home"]


class _InputBinding:
    dropped_count = 0

    def start(self, receiver, policy):
        self.receiver = receiver

    def close(self):
        pass


def test_lifecycle_opens_on_start_pauses_and_reuses_then_closes(monkeypatch):
    from loop_node.runtime import NodeRuntime

    module = _import_source_server(monkeypatch)
    left, right = _FakeService(), _FakeService()
    opened = []

    @contextmanager
    def factory(config):
        opened.append(config)
        try:
            yield [("left", left), ("right", right)]
        finally:
            right.close()
            left.close()

    node = module.VegaRobotNode(service_factory=factory)
    runtime = NodeRuntime(node)
    assert not opened
    try:
        runtime.configure({})
        runtime.start({"action_command": _InputBinding()})
        assert len(opened) == 1
        assert not left.paused and not right.paused
        left.queued_commands.append("old target")
        runtime.stop()
        assert left.paused and right.paused
        assert not left.queued_commands
        assert not left.closed and not right.closed
        runtime.configure({"control_frequency_hz": 100})
        runtime.start({"action_command": _InputBinding()})
        assert len(opened) == 1
        assert not left.queued_commands
        assert left.control_frequency_hz == right.control_frequency_hz == 100
    finally:
        runtime.shutdown()
    assert left.closed and right.closed


@pytest.mark.parametrize(
    "settings",
    [
        {"frame_type": "vega-1-pro_torso_frame_v2"},
        {
            "gripper_type": "sr_gripper",
            "left_gripper_device": "enp1s0",
            "right_gripper_device": "enp2s0",
            "action_frequency_hz": 30,
        },
    ],
)
def test_reconfigure_reopens_hardware_with_new_settings(
    monkeypatch,
    settings,
):
    from loop_node.runtime import NodeRuntime

    module = _import_source_server(monkeypatch)
    created = []

    def make_service(**kwargs):
        service = _FakeService()
        service._robot = types.SimpleNamespace(robot=object())
        created.append((kwargs, service))
        return service

    monkeypatch.setattr(module, "_LockedStepService", make_service)
    node = module.VegaRobotNode(service_factory=module._open_arm_services)
    runtime = NodeRuntime(node)
    try:
        runtime.configure({})
        assert created == []
        runtime.start({"action_command": _InputBinding()})
        runtime.stop()
        runtime.configure(settings)
        assert len(created) == 2
        runtime.start({"action_command": _InputBinding()})
        assert all(service.closed for _, service in created[:2])
        left, right = created[2:]
        config = module.VegaRobotNodeConfig(**settings)
        assert left[0]["robotiq_comport"] == config.left_gripper_device
        assert right[0]["robotiq_comport"] == config.right_gripper_device
        assert (
            left[0]["gripper_type"] == right[0]["gripper_type"] == config.gripper_type
        )
        assert (
            left[0]["control_hz"]
            == right[0]["control_hz"]
            == config.action_frequency_hz
        )
        assert left[0]["frame_type"] == right[0]["frame_type"] == config.frame_type
        assert right[0]["robot"] is left[1]._robot.robot
    finally:
        runtime.shutdown()
    assert all(service.closed for _, service in created)


@pytest.mark.parametrize("status", ["HOME_FAILED", ""])
def test_home_failure_reaches_caller_and_faults_both_arms(monkeypatch, status):
    from loop_node.runtime import NodeRuntime
    from loop_sdk import LifecycleState

    module = _import_source_server(monkeypatch)
    left = _FakeService(status=status, message="estopped")
    right = _FakeService()
    node = module.VegaRobotNode([("left", left), ("right", right)])
    runtime = NodeRuntime(node)
    try:
        runtime.configure({})
        runtime.start({"action_command": _InputBinding()})
        with pytest.raises(RuntimeError, match="estopped"):
            node._handle_command(module.RobotCommand.HOME)
        assert runtime.status.lifecycle is LifecycleState.FAULT
        assert left.paused and right.paused
        assert not right.resets
        runtime.reset_fault()
        runtime.configure({})
        runtime.start({"action_command": _InputBinding()})
        assert runtime.status.lifecycle is LifecycleState.ACTIVE
    finally:
        runtime.shutdown()


def test_failed_start_releases_opened_hardware(monkeypatch):
    from loop_node.lifecycle import NodeOperationError
    from loop_node.runtime import NodeRuntime

    module = _import_source_server(monkeypatch)
    left, right = _FakeService(), _FakeService()
    right.resume = lambda hz: (_ for _ in ()).throw(RuntimeError("right unavailable"))
    node = module.VegaRobotNode([("left", left), ("right", right)])
    runtime = NodeRuntime(node)
    with pytest.raises(NodeOperationError, match="right unavailable"):
        runtime.configure({})
        runtime.start({"action_command": _InputBinding()})
    assert left.closed and right.closed
    runtime.shutdown()


def test_right_initialization_failure_closes_left(monkeypatch):
    module = _import_source_server(monkeypatch)
    left = _FakeService()
    left._robot = types.SimpleNamespace(robot=object())

    def make_service(**kwargs):
        if kwargs["arm_side"] == "right":
            raise RuntimeError("right initialization failed")
        return left

    monkeypatch.setattr(module, "_LockedStepService", make_service)
    with pytest.raises(RuntimeError, match="right initialization failed"):
        with module._open_arm_services(module.VegaRobotNodeConfig()):
            pytest.fail("Initialization should fail before yielding services")
    assert left.closed


def test_failed_registration_never_opens_hardware_and_closes_node(monkeypatch):
    module = _import_source_server(monkeypatch)
    events = []

    class Host:
        is_running = False

        def __init__(self, **kwargs):
            events.append("created")

        def start(self, **kwargs):
            raise RuntimeError("session failed")

        def _on_shutdown(self):
            events.append("cleanup")

        def close(self):
            events.append("closed")

    monkeypatch.setattr(module, "VegaRobotNode", Host)
    monkeypatch.setattr(
        module,
        "_open_arm_services",
        lambda **kwargs: pytest.fail("Hardware opened before Start"),
    )
    with pytest.raises(RuntimeError, match="session failed"):
        module.serve_dual_arm(
            node_id="robot",
            connection=module.NodeConnectionConfig(loop_endpoint="tcp/127.0.0.1:7448"),
        )
    assert events == ["created", "cleanup", "closed"]


def test_action_failure_pauses_both_arms_without_dispatching_the_other(monkeypatch):
    from loop_node.runtime import NodeRuntime
    from loop_sdk import LifecycleState

    module = _import_source_server(monkeypatch)
    left = _FakeService(status="IK_FAILED", message="unreachable")
    right = _FakeService()
    node = module.VegaRobotNode([("left", left), ("right", right)])
    runtime = NodeRuntime(node)
    try:
        runtime.configure({})
        runtime.start({"action_command": _InputBinding()})
        message = ReceivedMessage(
            timestamp_ns=1,
            sequence=1,
            received_at_ns=2,
            payload={
                f"{arm}.{field}": value
                for arm in ["left", "right"]
                for field, value in [
                    ("target_cartesian_delta", float64_tensor([0.0] * 6, shape=(6,))),
                    ("gripper_position", 0.5),
                ]
            },
        )
        with pytest.raises(RuntimeError, match="IK_FAILED"):
            node._apply_action(message)
        assert runtime.status.lifecycle is LifecycleState.FAULT
        assert not right.requests
        assert left.paused and right.paused
    finally:
        runtime.shutdown()


@pytest.mark.parametrize("failure", ["capture", "publish", "control"])
def test_worker_failure_stops_ticks_and_faults_both_arms(monkeypatch, failure):
    from loop_node.runtime import NodeRuntime
    from loop_sdk import LifecycleState

    module = _import_source_server(monkeypatch)
    left, right = _FakeService(), _FakeService()
    node = module.VegaRobotNode([("left", left), ("right", right)])
    runtime = NodeRuntime(node)
    failed = threading.Event()

    def fail(*args, **kwargs):
        failed.set()
        raise RuntimeError("disconnected")

    try:
        if failure == "capture":
            left.capture_state = fail
        elif failure == "publish":
            node.publish_observation = fail
        else:
            service = module._LockedStepService.__new__(module._LockedStepService)
            service.arm_side = "left"
            service._robot = types.SimpleNamespace(
                execute_interpolated_tick=lambda: (failed.set(), True)[1],
                _prev_command_successful=False,
            )
            left.execute_control_tick = service.execute_control_tick
        runtime.configure({})
        runtime.start({"action_command": _InputBinding()})
        assert failed.wait(2)
        assert node._worker_stop.wait(2)
        node.check_health()
        assert runtime.status.lifecycle is LifecycleState.FAULT
        assert left.paused and right.paused
        assert node._control_thread is None and node._observation_thread is None
    finally:
        runtime.shutdown()


def test_control_ticks_continue_without_targets_or_waiting_for_publication(monkeypatch):
    from loop_node.runtime import NodeRuntime

    module = _import_source_server(monkeypatch)
    left, right = _FakeService(), _FakeService()
    node = module.VegaRobotNode([("left", left), ("right", right)])
    runtime = NodeRuntime(node)
    publishing = threading.Event()
    release = threading.Event()
    advanced = threading.Event()
    resumed_publish = threading.Event()
    published = []
    tick_order = []

    def capture(arm, service):
        tick_order.append(f"capture {arm}")
        state = _FakeService.capture_state(service)
        state["gripper_position"] = service.control_ticks
        return state

    def tick(arm, service):
        tick_order.append(f"command {arm}")
        service.control_ticks += 1
        if arm == "right" and publishing.is_set() and service.control_ticks >= 5:
            advanced.set()

    def publish(payload, timestamp_ns):
        published.append((payload, timestamp_ns))
        if len(published) == 1:
            publishing.set()
            assert release.wait(2)
        else:
            resumed_publish.set()

    left.capture_state = lambda: capture("left", left)
    right.capture_state = lambda: capture("right", right)
    left.execute_control_tick = lambda: tick("left", left)
    right.execute_control_tick = lambda: tick("right", right)
    node.publish_observation = publish
    try:
        runtime.configure({})
        runtime.start({"action_command": _InputBinding()})
        assert publishing.wait(2), "No observations before the first action"
        # Action/Home serialization must not hold up ordinary control ticks.
        with node._device_lock:
            assert advanced.wait(2), "Publication blocked motor control"
        with node._tick_lock:
            assert len(published) == 1
            assert (
                tick_order[:12]
                == [
                    "capture left",
                    "capture right",
                    "command left",
                    "command right",
                ]
                * 3
            )
            assert published[0][0]["left.gripper_position"] == 0
            release.set()
            assert resumed_publish.wait(2)
            assert published[1][0]["left.gripper_position"] == left.control_ticks - 1
            assert published[1][0]["right.gripper_position"] == right.control_ticks - 1
            assert published[1][1] > published[0][1]
    finally:
        release.set()
        runtime.shutdown()
    assert node._pending_snapshot is None


@pytest.mark.parametrize("stop_after_commands", [1, 3])
def test_stop_between_arms_prevents_the_remaining_command(
    monkeypatch, stop_after_commands
):
    module = _import_source_server(monkeypatch)
    left, right = _FakeService(), _FakeService()
    node = module.VegaRobotNode([("left", left), ("right", right)])
    commands = []

    def tick(arm):
        commands.append(arm)
        if len(commands) == stop_after_commands:
            node._worker_stop.set()

    left.execute_control_tick = lambda: tick("left")
    right.execute_control_tick = lambda: tick("right")
    node._control_loop(200)

    assert commands == ["left", "right", "left"][:stop_after_commands]
    assert node._worker_error is None


def test_home_excludes_interpolation_and_control_resumes_after_both_arms(monkeypatch):
    from loop_node import Message, OverflowPolicy, PolledInputPort
    from loop_node.runtime import NodeRuntime
    from loop_node.testing import InMemoryRequest, InMemoryStream

    module = _import_source_server(monkeypatch)
    from loop_bridge.source_server import _ObservationSnapshot

    left, right = _FakeService(), _FakeService()
    node = module.VegaRobotNode([("left", left), ("right", right)])
    runtime = NodeRuntime(node)
    actions, observations = InMemoryStream(), InMemoryStream()
    commands = InMemoryRequest()
    output = actions.output_binding()
    captured = PolledInputPort(
        "captured",
        ROBOT_OBSERVATION_CONTRACT,
        capacity=100,
        overflow=OverflowPolicy.DROP_OLDEST,
    )
    captured.bind(observations.polled_input_binding())
    home_entered = threading.Event()
    finish_home = threading.Event()
    ticked = threading.Event()
    errors = []
    action_applied = threading.Event()
    old_snapshot = _ObservationSnapshot(
        123,
        {arm: service.capture_state() for arm, service in node._arm_services},
        {},
        time.monotonic_ns(),
    )
    payload = {
        "left.target_cartesian_delta": float64_tensor((0.0,) * 6, shape=(6,)),
        "left.gripper_position": 0.2,
        "right.target_cartesian_delta": float64_tensor((0.0,) * 6, shape=(6,)),
        "right.gripper_position": 0.2,
    }

    def step(request, context):
        result = _FakeService.Step(right, request, context)
        action_applied.set()
        return result

    def home(request, context):
        home_entered.set()
        assert finish_home.wait(2)
        return _FakeService.Reset(left, request, context)

    def tick():
        assert not home_entered.is_set() or finish_home.is_set()
        _FakeService.execute_control_tick(right)
        ticked.set()

    def request_home():
        try:
            commands.client_binding().request(RobotCommand.HOME.to_payload())
        except Exception as error:
            errors.append(error)

    left.Reset = home
    right.execute_control_tick = tick
    right.Step = step
    home_thread = threading.Thread(target=request_home)
    try:
        runtime.configure({})
        runtime.start(
            {
                "action_command": actions.callback_input_binding(),
                "robot_command": commands.server_binding(),
                "robot_observation": observations.output_binding(),
            }
        )
        assert ticked.wait(2)
        home_thread.start()
        assert home_entered.wait(2)
        ticked.clear()
        output.publish(Message(timestamp_ns=1, sequence=0, payload=payload))
        stale_action = ReceivedMessage(
            timestamp_ns=1,
            sequence=1,
            payload=payload,
            received_at_ns=time.monotonic_ns(),
        )
        assert not ticked.wait(0.03), "Interpolation commands overlapped Home"
        finish_home.set()
        home_thread.join(timeout=2)
        assert not home_thread.is_alive()
        assert not errors
        assert len(left.resets) == len(right.resets) == 1
        assert ticked.wait(2)
        node._apply_action(stale_action)
        assert not action_applied.wait(0.03), (
            "An action queued during Home was replayed"
        )
        assert left.requests == right.requests == []
        node._publish_snapshot(old_snapshot)
        while (message := captured.poll_next()) is not None:
            assert message.timestamp_ns != 123, (
                "A pre-Home snapshot was published after Home"
            )
        output.publish(Message(timestamp_ns=2, sequence=2, payload=payload))
        assert action_applied.wait(2), "Fresh actions did not resume after Home"
        assert len(left.requests) == len(right.requests) == 1
        assert node._worker_error is None
    finally:
        finish_home.set()
        if home_thread.ident is not None:
            home_thread.join(timeout=2)
        runtime.shutdown()
        captured.unbind()
