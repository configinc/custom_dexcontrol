from concurrent import futures

import grpc
import pytest

from proto import robotenv_pb2, robotenv_pb2_grpc


def test_force_calibration_rpc_v1_descriptor() -> None:
    service = robotenv_pb2.DESCRIPTOR.services_by_name["RobotEnv"]
    expected_methods = {
        "StartForceCalibration": (
            "StartForceCalibrationRequest",
            "ForceCalibrationStatus",
        ),
        "AdvanceForceCalibration": (
            "AdvanceForceCalibrationRequest",
            "ForceCalibrationStatus",
        ),
        "CommitForceCalibration": (
            "CommitForceCalibrationRequest",
            "ForceCalibrationStatus",
        ),
        "GetForceCalibrationStatus": (
            "GetForceCalibrationStatusRequest",
            "ForceCalibrationStatus",
        ),
        "CancelForceCalibration": (
            "CancelForceCalibrationRequest",
            "ForceCalibrationStatus",
        ),
        "GetForceCalibrationResult": (
            "GetForceCalibrationResultRequest",
            "ForceCalibrationResult",
        ),
    }
    methods = {method.name: method for method in service.methods}
    for name, (request_name, response_name) in expected_methods.items():
        assert name in methods
        assert methods[name].input_type.name == request_name
        assert methods[name].output_type.name == response_name

    state = robotenv_pb2.ForceCalibrationState.DESCRIPTOR
    assert {value.name: value.number for value in state.values} == {
        "FORCE_CALIBRATION_STATE_UNSPECIFIED": 0,
        "FORCE_CALIBRATION_STATE_IDLE": 1,
        "FORCE_CALIBRATION_STATE_READY_FOR_STAGE": 2,
        "FORCE_CALIBRATION_STATE_MOVING": 3,
        "FORCE_CALIBRATION_STATE_SETTLING": 4,
        "FORCE_CALIBRATION_STATE_COLLECTING": 5,
        "FORCE_CALIBRATION_STATE_FITTING": 6,
        "FORCE_CALIBRATION_STATE_APPLYING": 7,
        "FORCE_CALIBRATION_STATE_RETURNING": 8,
        "FORCE_CALIBRATION_STATE_SUCCEEDED": 9,
        "FORCE_CALIBRATION_STATE_FAILED": 10,
        "FORCE_CALIBRATION_STATE_CANCELLING": 11,
        "FORCE_CALIBRATION_STATE_CANCELLED": 12,
        "FORCE_CALIBRATION_STATE_READY_TO_APPLY": 13,
    }


def test_force_calibration_rpc_v1_field_numbers() -> None:
    messages = {
        robotenv_pb2.StartForceCalibrationRequest: {
            "run_id": 1,
            "force_recalibrate": 2,
            "gripper_mass_kg": 3,
            "gripper_com_sensor_m": 4,
        },
        robotenv_pb2.AdvanceForceCalibrationRequest: {
            "run_id": 1,
            "stage_index": 2,
            "stage_id": 3,
        },
        robotenv_pb2.CommitForceCalibrationRequest: {
            "run_id": 1,
            "plan_sha256": 2,
        },
        robotenv_pb2.GetForceCalibrationStatusRequest: {"run_id": 1},
        robotenv_pb2.CancelForceCalibrationRequest: {"run_id": 1},
        robotenv_pb2.GetForceCalibrationResultRequest: {"run_id": 1},
        robotenv_pb2.ForceCalibrationStatus: {
            "state": 1,
            "run_id": 2,
            "current_stage_index": 3,
            "completed_stages": 4,
            "total_stages": 5,
            "message": 6,
            "arm_name": 7,
            "plan_id": 8,
            "plan_sha256": 9,
            "stage_id": 10,
            "server_instance_id": 11,
            "cache_reused": 12,
            "command_deadline_timestamp_us": 13,
        },
        robotenv_pb2.ForceCalibrationResult: {
            "available": 1,
            "calibrated_at_timestamp_us": 2,
            "expires_at_timestamp_us": 3,
            "calibration_kind": 4,
            "parameters_json": 5,
            "quality_json": 6,
            "message": 7,
            "run_id": 8,
            "schema_version": 9,
            "reference_point_id": 10,
            "arm_name": 11,
            "plan_id": 12,
            "plan_sha256": 13,
            "server_instance_id": 14,
            "source_run_id": 15,
            "was_committed": 16,
        },
    }
    for message_type, expected in messages.items():
        assert {
            field.name: field.number
            for field in message_type.DESCRIPTOR.fields
        } == expected


def test_generated_servicer_exposes_force_calibration_methods() -> None:
    servicer = robotenv_pb2_grpc.RobotEnvServicer()
    for method_name in (
        "StartForceCalibration",
        "AdvanceForceCalibration",
        "CommitForceCalibration",
        "GetForceCalibrationStatus",
        "CancelForceCalibration",
        "GetForceCalibrationResult",
    ):
        assert callable(getattr(servicer, method_name))


def test_generated_registration_keeps_new_methods_unimplemented_by_default() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    robotenv_pb2_grpc.add_RobotEnvServicer_to_server(
        robotenv_pb2_grpc.RobotEnvServicer(), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = robotenv_pb2_grpc.RobotEnvStub(channel)
        with pytest.raises(grpc.RpcError) as error:
            stub.GetForceCalibrationStatus(
                robotenv_pb2.GetForceCalibrationStatusRequest(), timeout=1
            )
        assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED
    finally:
        channel.close()
        server.stop(grace=None)
