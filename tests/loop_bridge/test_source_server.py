"""Tests for source_server glue that would otherwise import live dexcontrol deps."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


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
        float_array: list[float] | None = None,
        int_value: int | None = None,
        string_value: str | None = None,
    ) -> None:
        self.float_value = float_value
        self.float_array = _FakeFloatArray(float_array) if float_array is not None else None
        self.int_value = int_value
        self.string_value = string_value
        self._kind = next(
            (
                name
                for name, val in (
                    ("float_value", float_value),
                    ("float_array", float_array),
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

    def Step(self, request: _StepRequest, context: object) -> object:
        del context
        self.requests.append(request)
        return types.SimpleNamespace(
            status=self.status, message=self.message, action_info=self.action_info
        )

    def Reset(self, request: _ResetRequest, context: object) -> object:
        del context
        self.resets.append(request)
        return types.SimpleNamespace(status=self.status, message=self.message)


def _import_source_server(monkeypatch: pytest.MonkeyPatch) -> object:
    fake_server = types.ModuleType("dexcontrol.core.robotenv_vega.server")
    fake_server.robotenv_pb2 = types.SimpleNamespace(
        StepRequest=_StepRequest, ResetRequest=_ResetRequest
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


# --- dual-arm per-arm gripper comports -------------------------------------


def test_dual_arm_comports_distinct_robotiq_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    source_server = _import_source_server(monkeypatch)
    kwargs = {"gripper_type": "robotiq", "robotiq_comport": "/dev/ttyUSB0"}
    left, right = source_server._dual_arm_comports(kwargs, "/dev/ttyUSB1", "/dev/ttyUSB0")
    assert (left, right) == ("/dev/ttyUSB1", "/dev/ttyUSB0")


def test_dual_arm_comports_same_serial_port_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    source_server = _import_source_server(monkeypatch)
    kwargs = {"gripper_type": "robotiq", "robotiq_comport": "/dev/ttyUSB0"}
    # Both arms falling back to the same shared port is the footgun → reject.
    with pytest.raises(ValueError, match="DISTINCT comport"):
        source_server._dual_arm_comports(kwargs, None, None)


def test_dual_arm_comports_non_serial_gripper_unrestricted(monkeypatch: pytest.MonkeyPatch) -> None:
    source_server = _import_source_server(monkeypatch)
    # Built-in (non-serial) grippers share no port → same/empty comport is fine.
    kwargs = {"gripper_type": "default"}
    assert source_server._dual_arm_comports(kwargs, None, None) == (None, None)
