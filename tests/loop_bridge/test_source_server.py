"""Tests for source_server glue that would otherwise import live dexcontrol deps."""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager

import pytest
from conftest import make_observation
from loop_sdk import ReceivedMessage

from loop_bridge.contracts import float64_tensor, float64_values


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
        self.control_error = None
        self.queued_commands = []

    def resume(self):
        self.paused = False

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


def test_dual_arm_node_preserves_legacy_action_and_observation_pairing(
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


# --- dual-arm per-arm gripper comports -------------------------------------


def test_dual_arm_comports_distinct_robotiq_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    source_server = _import_source_server(monkeypatch)
    kwargs = {"gripper_type": "robotiq", "robotiq_comport": "/dev/ttyUSB0"}
    left, right = source_server._dual_arm_comports(
        kwargs, "/dev/ttyUSB1", "/dev/ttyUSB0"
    )
    assert (left, right) == ("/dev/ttyUSB1", "/dev/ttyUSB0")


def test_dual_arm_comports_same_serial_port_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_server = _import_source_server(monkeypatch)
    kwargs = {"gripper_type": "robotiq", "robotiq_comport": "/dev/ttyUSB0"}
    # Both arms falling back to the same shared port is the footgun → reject.
    with pytest.raises(ValueError, match="DISTINCT comport"):
        source_server._dual_arm_comports(kwargs, None, None)


def test_dual_arm_comports_non_serial_gripper_unrestricted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_server = _import_source_server(monkeypatch)
    # Built-in (non-serial) grippers share no port → same/empty comport is fine.
    kwargs = {"gripper_type": "default"}
    assert source_server._dual_arm_comports(kwargs, None, None) == (None, None)


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
    def factory():
        opened.append(True)
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
        assert opened == [True]
        assert not left.paused and not right.paused
        left.queued_commands.append("old target")
        runtime.stop()
        assert left.paused and right.paused
        assert not left.queued_commands
        assert not left.closed and not right.closed
        runtime.configure({})
        runtime.start({"action_command": _InputBinding()})
        assert opened == [True]
        assert not left.queued_commands
    finally:
        runtime.shutdown()
    assert left.closed and right.closed


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
    right.resume = lambda: (_ for _ in ()).throw(RuntimeError("right unavailable"))
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
        with module._open_arm_services():
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


def test_heartbeat_failure_can_pause_its_own_worker(monkeypatch):
    import threading

    from loop_node.runtime import NodeRuntime
    from loop_sdk import LifecycleState

    module = _import_source_server(monkeypatch)
    left, right = _FakeService(), _FakeService()
    node = module.VegaRobotNode([("left", left), ("right", right)])
    runtime = NodeRuntime(node)
    paused = threading.Event()
    original_pause = right.pause

    def pause():
        original_pause()
        paused.set()

    right.pause = pause
    try:
        runtime.configure({"heartbeat_frequency_hz": 100})
        runtime.start({"action_command": _InputBinding()})
        left._create_observation = lambda: (_ for _ in ()).throw(
            RuntimeError("disconnected")
        )
        assert paused.wait(2), (
            "Observation worker could not finish its own fault cleanup"
        )
        assert runtime.status.lifecycle is LifecycleState.FAULT
        assert left.paused and right.paused
    finally:
        runtime.shutdown()


def test_interpolation_failure_reaches_node_and_pauses_both_arms(monkeypatch):
    import threading

    from loop_node.runtime import NodeRuntime
    from loop_sdk import LifecycleState

    module = _import_source_server(monkeypatch)
    service = module._LockedStepService.__new__(module._LockedStepService)
    service.arm_side = "left"
    service._control_loop_hz = 200
    service._control_loop_stop = threading.Event()
    service._robot = types.SimpleNamespace(
        execute_interpolated_tick=lambda: True,
        _prev_command_successful=False,
    )
    service._control_loop_run()

    left, right = _FakeService(), _FakeService()
    node = module.VegaRobotNode([("left", left), ("right", right)])
    runtime = NodeRuntime(node)
    try:
        runtime.configure({})
        runtime.start({"action_command": _InputBinding()})
        left.control_error = service.control_error
        node.check_health()
        assert runtime.status.lifecycle is LifecycleState.FAULT
        assert left.paused and right.paused
        assert service._control_loop_stop.is_set()
    finally:
        runtime.shutdown()
