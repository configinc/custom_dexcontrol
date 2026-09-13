"""Canonical Loop Node contracts for one dual-arm Vega robot."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Mapping

from loop_sdk import FieldContract, PayloadContract, TensorValue, ValueKind

LEFT_ARM = "left"
RIGHT_ARM = "right"
ARM_NAMES = (LEFT_ARM, RIGHT_ARM)

FLOAT64_DTYPE = "float64"

_OBSERVATION_TENSOR_SHAPES: Mapping[str, tuple[int, ...]] = {
    "joint_positions": (7,),
    "cartesian_position": (6,),
    "joint_velocities": (7,),
    "joint_torques_computed": (7,),
    "wrench_state": (6,),
}

_ACTION_INFO_TENSOR_SHAPES: Mapping[str, tuple[int, ...]] = {
    "cartesian_velocity": (6,),
    "target_cartesian_delta": (7,),
    "delta_action": (7,),
    "cartesian_position": (6,),
    "joint_position": (7,),
    "joint_velocity": (7,),
}

_ACTION_INFO_SCALARS = ("gripper_position", "gripper_delta")


def _tensor_contract(shape: tuple[int, ...], *, required: bool = True) -> FieldContract:
    return FieldContract(
        kind=ValueKind.TENSOR,
        dtype=FLOAT64_DTYPE,
        shape=shape,
        required=required,
    )


def _observation_fields() -> dict[str, FieldContract]:
    fields: dict[str, FieldContract] = {}
    for arm in ARM_NAMES:
        for field, shape in _OBSERVATION_TENSOR_SHAPES.items():
            fields[f"{arm}.{field}"] = _tensor_contract(shape)
        fields[f"{arm}.gripper_position"] = FieldContract(kind=ValueKind.SCALAR)

        for field, shape in _ACTION_INFO_TENSOR_SHAPES.items():
            fields[f"{arm}.action.{field}"] = _tensor_contract(shape, required=False)
        for field in _ACTION_INFO_SCALARS:
            fields[f"{arm}.action.{field}"] = FieldContract(
                kind=ValueKind.SCALAR,
                required=False,
            )

    # The legacy LoopRobotClient appended the exact opaque vector it received to
    # every action-paired observation. Keep the same diagnostic value, now with a
    # fixed bimanual shape instead of an undeclared list.
    fields["received_action"] = _tensor_contract((14,), required=False)
    return fields


ROBOT_OBSERVATION_CONTRACT = PayloadContract(fields=_observation_fields())

ROBOT_ACTION_CONTRACT = PayloadContract(
    fields={
        "left.target_cartesian_delta": _tensor_contract((6,)),
        "left.gripper_position": FieldContract(kind=ValueKind.SCALAR),
        "right.target_cartesian_delta": _tensor_contract((6,)),
        "right.gripper_position": FieldContract(kind=ValueKind.SCALAR),
    }
)


def float64_tensor(values: Iterable[float], *, shape: tuple[int, ...]) -> TensorValue:
    frozen = tuple(float(value) for value in values)
    expected_size = math.prod(shape)
    if len(frozen) != expected_size:
        raise ValueError(f"tensor shape {shape} requires {expected_size} values")
    return TensorValue(
        dtype=FLOAT64_DTYPE,
        shape=shape,
        data=memoryview(struct.pack(f"<{len(frozen)}d", *frozen)),
    )


def float64_values(
    tensor: TensorValue,
    *,
    field_name: str,
    shape: tuple[int, ...],
) -> tuple[float, ...]:
    if tensor.dtype != FLOAT64_DTYPE:
        raise ValueError(f"{field_name} must use {FLOAT64_DTYPE}")
    if tensor.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    value_count = math.prod(shape)
    expected_bytes = value_count * 8
    if tensor.data.nbytes != expected_bytes:
        raise ValueError(f"{field_name} requires {expected_bytes} data bytes")
    values = struct.unpack(f"<{value_count}d", tensor.data)
    return values


def action_info_shape(field: str) -> tuple[int, ...] | None:
    return _ACTION_INFO_TENSOR_SHAPES.get(field)


def is_action_info_scalar(field: str) -> bool:
    return field in _ACTION_INFO_SCALARS
