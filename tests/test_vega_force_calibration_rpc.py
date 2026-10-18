from __future__ import annotations

import json
import sys
import threading
from types import ModuleType

import grpc

# Keep this RPC test independently collectible without the optional hidapi stack.
_base_arm_teleop = ModuleType("base_arm_teleop")
_base_arm_teleop.BaseIKController = object
sys.modules.setdefault("base_arm_teleop", _base_arm_teleop)

from dexcontrol.core.robotenv_vega.force_calibration_lifecycle import (  # noqa: E402
    CalibrationStatusSnapshot,
    LifecycleError,
)
from dexcontrol.core.robotenv_vega.server import VegaRobotEnvService  # noqa: E402
from proto import robotenv_pb2  # noqa: E402


class Context:
    def __init__(self) -> None:
        self.code = None
        self.details = ""

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def _status(state="READY_FOR_STAGE"):
    return CalibrationStatusSnapshot(
        state=state,
        run_id="run-1",
        current_stage_index=0,
        completed_stages=0,
        total_stages=2,
        message="ready",
        arm_name="left",
        plan_id="plan",
        plan_sha256="a" * 64,
        stage_id="stage-0",
        server_instance_id="server",
        cache_reused=False,
        command_deadline_timestamp_us=123,
    )


class Manager:
    def __init__(self) -> None:
        self.calls = []

    def start(self, run_id, force, fit_inputs=None):
        self.calls.append(("start", run_id, force, fit_inputs))
        return _status()

    def advance(self, run_id, index, stage_id):
        self.calls.append(("advance", run_id, index, stage_id))
        return _status("MOVING")

    def commit(self, run_id, digest):
        self.calls.append(("commit", run_id, digest))
        return _status("SUCCEEDED")

    def status(self, run_id):
        self.calls.append(("status", run_id))
        return _status()

    def request_cancel(self, run_id):
        self.calls.append(("cancel", run_id))
        return _status("CANCELLING")

    def result(self, run_id):
        self.calls.append(("result", run_id))
        return {
            "available": True,
            "calibrated_at_timestamp_us": 10,
            "expires_at_timestamp_us": 20,
            "calibration_kind": "vega_wrist_ft_v1",
            "parameters": {"schema_version": 5},
            "quality": {"passed": True},
            "message": "",
            "run_id": "run-1",
            "schema_version": 1,
            "reference_point_id": "sensor-origin",
            "arm_name": "left",
            "plan_id": "plan",
            "plan_sha256": "a" * 64,
            "server_instance_id": "server",
            "source_run_id": "run-1",
            "was_committed": True,
        }


def _service(manager=None):
    service = VegaRobotEnvService.__new__(VegaRobotEnvService)
    service._force_calibration_manager = manager or Manager()
    service._shutdown_started = threading.Event()
    return service


def test_rpc_methods_forward_exact_contract_fields() -> None:
    manager = Manager()
    service = _service(manager)
    context = Context()

    started = service.StartForceCalibration(
        robotenv_pb2.StartForceCalibrationRequest(
            run_id="run-1",
            force_recalibrate=True,
            gripper_mass_kg=1.2,
            gripper_com_sensor_m=[0.01, 0.02, 0.03],
        ),
        context,
    )
    moved = service.AdvanceForceCalibration(
        robotenv_pb2.AdvanceForceCalibrationRequest(
            run_id="run-1", stage_index=0, stage_id="stage-0"
        ),
        context,
    )
    assert started.state == robotenv_pb2.FORCE_CALIBRATION_STATE_READY_FOR_STAGE
    assert started.plan_sha256 == "a" * 64
    assert moved.state == robotenv_pb2.FORCE_CALIBRATION_STATE_MOVING
    assert manager.calls[0][:3] == ("start", "run-1", True)
    assert manager.calls[0][3].gripper_mass_kg == 1.2
    assert manager.calls[0][3].gripper_com_sensor_m == (0.01, 0.02, 0.03)
    assert manager.calls[1] == ("advance", "run-1", 0, "stage-0")


def test_result_serializes_inspectable_json() -> None:
    response = _service().GetForceCalibrationResult(
        robotenv_pb2.GetForceCalibrationResultRequest(run_id="run-1"), Context()
    )
    assert response.available is True
    assert json.loads(response.parameters_json) == {"schema_version": 5}
    assert json.loads(response.quality_json) == {"passed": True}
    assert response.was_committed is True


def test_lifecycle_error_maps_to_grpc_status() -> None:
    class RejectingManager(Manager):
        def start(self, run_id, force, fit_inputs=None):
            raise LifecycleError("FAILED_PRECONDITION", "busy")

    context = Context()
    response = _service(RejectingManager()).StartForceCalibration(
        robotenv_pb2.StartForceCalibrationRequest(
            run_id="run-2",
            gripper_mass_kg=1.2,
            gripper_com_sensor_m=[0.01, 0.02, 0.03],
        ),
        context,
    )
    assert response == robotenv_pb2.ForceCalibrationStatus()
    assert context.code == grpc.StatusCode.FAILED_PRECONDITION
    assert context.details == "busy"


def test_empty_cancel_is_not_found_without_touching_manager() -> None:
    manager = Manager()
    context = Context()
    response = _service(manager).CancelForceCalibration(
        robotenv_pb2.CancelForceCalibrationRequest(), context
    )
    assert response == robotenv_pb2.ForceCalibrationStatus()
    assert context.code == grpc.StatusCode.NOT_FOUND
    assert manager.calls == []


def test_shutdown_barrier_rejects_mutating_calibration_rpcs() -> None:
    manager = Manager()
    service = _service(manager)
    service._shutdown_started.set()

    requests = (
        (service.StartForceCalibration, robotenv_pb2.StartForceCalibrationRequest(run_id="r")),
        (
            service.AdvanceForceCalibration,
            robotenv_pb2.AdvanceForceCalibrationRequest(
                run_id="r", stage_index=0, stage_id="stage-0"
            ),
        ),
        (
            service.CommitForceCalibration,
            robotenv_pb2.CommitForceCalibrationRequest(run_id="r", plan_sha256="a" * 64),
        ),
    )
    for method, request in requests:
        context = Context()
        assert method(request, context) == robotenv_pb2.ForceCalibrationStatus()
        assert context.code == grpc.StatusCode.UNAVAILABLE
        assert context.details == "robot server is shutting down"

    assert manager.calls == []
