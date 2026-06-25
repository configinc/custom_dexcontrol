"""robot-obs channel layout for the Vega RobotEnv server (Loop Source Bus).

This module owns the contract between Vega's RobotEnv observation and the
``robot-obs`` source the server publishes onto Loop's Source Bus:

  - ``build_obs_channels()`` declares the ordered ``ChannelSpec`` layout once.
  - ``observation_to_step()`` projects one RobotEnv observation onto that exact
    layout, producing the ``{channel_key: reading}`` dict ``RobotStepSender``
    streams (the dict-payload wire format — named channels, not a bare vector).

Channel keys follow the RCI wire convention ``<arm>.observation.robot_state.``
``<field>[i]`` (a sibling of the ``<arm>.action.<space>[i]`` keys the action lane
carries), so observation and action keys never collide when the loop pairs them
into a robot-step. The arm prefix (default ``robot0``) namespaces one arm's
channels; a dual-arm unit runs one bridge per arm.

The values are the SAME numbers the RobotEnv ``Step``/``Reset`` RPC already
returns (``VegaRobotEnvService._create_observation``); only the shape changes
(named ``Value`` map -> named-channel dict). No hardware logic lives here, so
both functions are pure and testable without a robot or a running Source Bus.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from loop_sdk import ChannelRole, ChannelSpec

# Namespace infix mirroring the RobotEnv obs_dict shape (obs["robot_state"][field]).
_OBS_NAMESPACE = "observation.robot_state"

# Default arm prefix for a single-arm bridge. Dual-arm units run one bridge per
# arm (robot0 / robot1) and pass the matching prefix.
DEFAULT_ARM_PREFIX = "robot0"

# Ordered (proto_key, count, scalar, role) projection of the RobotEnv observation
# map onto the robot-obs channels. ``scalar`` reads ``Value.float_value``;
# otherwise ``Value.float_array.values`` (which must hold exactly ``count``).
# Every walk over the layout goes through ``_channel_key`` so the declared layout
# and the streamed dict can never drift.
_OBS_FIELDS: tuple[tuple[str, int, bool, ChannelRole], ...] = (
    ("joint_positions", 7, False, ChannelRole.CORE),
    ("gripper_position", 1, True, ChannelRole.CORE),
    ("cartesian_position", 6, False, ChannelRole.CORE),
    ("joint_velocities", 7, False, ChannelRole.AUX),
    ("joint_torques_computed", 7, False, ChannelRole.AUX),
    ("wrench_state", 6, False, ChannelRole.AUX),
)

# Per-index units for the mixed-unit cartesian pose (xyz metres, rpy radians)
# and the wrench (force N, torque N·m). Other fields are uniform-unit.
_CARTESIAN_UNITS = ("m", "m", "m", "rad", "rad", "rad")
_WRENCH_UNITS = ("N", "N", "N", "N·m", "N·m", "N·m")

# Per-field unit for the uniform-unit fields, keyed by proto field name.
_FIELD_UNITS = {
    "joint_positions": "rad",
    "gripper_position": "normalized",
    "joint_velocities": "rad/s",
    "joint_torques_computed": "N·m",
}


def _channel_key(arm_prefix: str, field: str, index: Optional[int]) -> str:
    """One namespaced channel key. Scalar fields omit the ``[i]`` suffix."""
    base = f"{arm_prefix}.{_OBS_NAMESPACE}.{field}"
    return base if index is None else f"{base}[{index}]"


def _unit_for(field: str, index: int) -> str:
    if field == "cartesian_position":
        return _CARTESIAN_UNITS[index]
    if field == "wrench_state":
        return _WRENCH_UNITS[index]
    return _FIELD_UNITS.get(field, "")


def build_obs_channels(arm_prefix: str = DEFAULT_ARM_PREFIX) -> tuple[ChannelSpec, ...]:
    """Declare the ordered robot-obs channel layout for a single Vega arm.

    CORE = the essential measured state RCI consumes to close the loop (joint
    positions, gripper, end-effector pose). AUX = supplementary provenance
    (velocities, torques, wrench). The order matches ``observation_to_step``.
    """
    channels: list[ChannelSpec] = []
    for field, count, scalar, role in _OBS_FIELDS:
        if scalar:
            channels.append(
                ChannelSpec(
                    key=_channel_key(arm_prefix, field, None),
                    role=role,
                    unit=_unit_for(field, 0),
                    range=(0.0, 1.0) if field == "gripper_position" else None,
                )
            )
            continue
        channels.extend(
            ChannelSpec(
                key=_channel_key(arm_prefix, field, i),
                role=role,
                unit=_unit_for(field, i),
            )
            for i in range(count)
        )
    return tuple(channels)


def obs_dim() -> int:
    """Number of robot-obs channels (== len(build_obs_channels()))."""
    return sum(count for _, count, _, _ in _OBS_FIELDS)


def observation_to_step(
    observation: Mapping[str, Any], arm_prefix: str = DEFAULT_ARM_PREFIX
) -> dict[str, float]:
    """Project one RobotEnv observation map onto the robot-obs step dict.

    ``observation`` is the ``dict[str, robotenv_pb2.Value]`` produced by
    ``VegaRobotEnvService._create_observation``. Returns ``{channel_key: reading}``
    in ``build_obs_channels()`` order, ready for ``RobotStepSender.send``. Raises
    ``KeyError`` if a declared field is absent and ``ValueError`` if an array
    field's length disagrees with its declared count — both are contract
    violations the caller should fix, not silently paper over.
    """
    step: dict[str, float] = {}
    for field, count, scalar, _ in _OBS_FIELDS:
        value = observation[field]
        if scalar:
            step[_channel_key(arm_prefix, field, None)] = float(value.float_value)
            continue
        array = list(value.float_array.values)
        if len(array) != count:
            raise ValueError(
                f"robot-obs field {field!r} carries {len(array)} values, expected {count}"
            )
        for i, item in enumerate(array):
            step[_channel_key(arm_prefix, field, i)] = float(item)
    return step
