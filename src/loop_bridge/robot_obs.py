"""Project Vega RobotEnv observations onto the dual-arm Robot Node payload."""

from __future__ import annotations

from typing import Any, Mapping, cast

from loop_bridge.contracts import float64_tensor

_OBS_FIELDS: tuple[tuple[str, int, bool], ...] = (
    ("joint_positions", 7, False),
    ("gripper_position", 1, True),
    ("cartesian_position", 6, False),
    ("joint_velocities", 7, False),
    ("joint_torques_computed", 7, False),
    ("wrench_state", 6, False),
)


def observation_state(observation: Mapping[str, Any]) -> dict[str, float | list[float]]:
    """Decode one RobotEnv proto observation without changing its values."""

    state: dict[str, float | list[float]] = {}
    for field, count, scalar in _OBS_FIELDS:
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

    state = observation_state(observation)
    payload: dict[str, Any] = {}
    for field, count, scalar in _OBS_FIELDS:
        value = state[field]
        if scalar:
            payload[f"{arm}.{field}"] = cast(float, value)
            continue
        payload[f"{arm}.{field}"] = float64_tensor(
            cast(list[float], value), shape=(count,)
        )
    return payload
