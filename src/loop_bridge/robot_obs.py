"""Project Vega RobotEnv observations onto the dual-arm Robot Node payload."""

from __future__ import annotations

from typing import Any, Mapping, cast

from loop_bridge.contracts import float64_tensor

_OBS_FIELDS: tuple[tuple[str, int, bool, bool], ...] = (
    ("joint_positions", 7, False, True),
    ("gripper_position", 1, True, True),
    ("cartesian_position", 6, False, True),
    ("joint_velocities", 7, False, True),
    ("joint_torques_computed", 7, False, True),
    ("external_wrench_world", 6, False, False),
)


def observation_state(observation: Mapping[str, Any]) -> dict[str, float | list[float]]:
    """Decode one RobotEnv proto observation without changing its values."""

    state: dict[str, float | list[float]] = {}
    for field, count, scalar, required in _OBS_FIELDS:
        if not required and field not in observation:
            continue
        value = observation[field]
        if scalar:
            state[field] = float(value.float_value)
            continue
        values = [float(item) for item in value.float_array.values]
        if len(values) != count:
            raise ValueError(
                f"robot observation field {field!r} carries {len(values)} values, expected {count}"
            )
        state[field] = values
    return state


def observation_payload(observation: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Encode one arm under explicit ``left.*`` or ``right.*`` Node fields."""

    return state_payload(observation_state(observation), arm)


def state_payload(state: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Encode a measured arm state directly, without a RobotEnv proto round trip."""
    payload: dict[str, Any] = {}
    for field, count, scalar, required in _OBS_FIELDS:
        if not required and field not in state:
            continue
        value = state[field]
        if scalar:
            payload[f"{arm}.{field}"] = cast(float, value)
            continue
        payload[f"{arm}.{field}"] = float64_tensor(
            cast(list[float], value), shape=(count,)
        )
    return payload
