"""robot-obs channel layout for the Vega RobotEnv server (Loop Source Bus).

This module owns the contract between Vega's RobotEnv observation and the
``robot-obs`` source the server publishes onto Loop's Source Bus:

  - ``build_obs_channels()`` declares the ordered ``ChannelSpec`` layout once.
  - ``flatten_observation()`` projects one RobotEnv observation onto that exact
    order, producing the flat ``values`` vector the bus carries.

The values are the SAME numbers the RobotEnv ``Step``/``Reset`` RPC already
returns (``VegaRobotEnvService._create_observation``); only the shape changes
(named ``Value`` map -> ordered float vector, ``values[i]`` aligned to
``build_obs_channels()[i]``). No hardware logic lives here, so both functions
are pure and testable without a robot or a running Source Bus.
"""

from __future__ import annotations

from typing import Any, Mapping

from loop_sdk import ChannelRole, ChannelSpec

# Ordered (proto_key, count, scalar) projection of the RobotEnv observation map
# onto the robot-obs value vector. ``scalar`` reads ``Value.float_value``;
# otherwise ``Value.float_array.values`` (which must hold exactly ``count``).
# Order here is authoritative — build_obs_channels() and flatten_observation()
# both walk it, so they can never drift.
_OBS_FIELDS: tuple[tuple[str, int, bool], ...] = (
    ("joint_positions", 7, False),
    ("gripper_position", 1, True),
    ("cartesian_position", 6, False),
    ("joint_velocities", 7, False),
    ("joint_torques_computed", 7, False),
    ("wrench_state", 6, False),
)

# Per-index units for the mixed-unit cartesian pose (xyz metres, rpy radians)
# and the wrench (force N, torque N·m). Other fields are uniform-unit.
_CARTESIAN_UNITS = ("m", "m", "m", "rad", "rad", "rad")
_WRENCH_UNITS = ("N", "N", "N", "N·m", "N·m", "N·m")


def build_obs_channels() -> tuple[ChannelSpec, ...]:
    """Declare the ordered robot-obs channel layout for a single Vega arm.

    CORE = the essential measured state RCI consumes to close the loop (joint
    positions, gripper, end-effector pose). AUX = supplementary provenance
    (velocities, torques, wrench) carried alongside. ``values[i]`` from
    ``flatten_observation`` aligns to index ``i`` here.
    """
    channels: list[ChannelSpec] = []

    # CORE — measured arm state.
    channels.extend(
        ChannelSpec(key=f"joint_positions[{i}]", role=ChannelRole.CORE, unit="rad")
        for i in range(7)
    )
    channels.append(
        ChannelSpec(
            key="gripper_position",
            role=ChannelRole.CORE,
            unit="normalized",
            range=(0.0, 1.0),
        )
    )
    channels.extend(
        ChannelSpec(key=f"cartesian_position[{i}]", role=ChannelRole.CORE, unit=unit)
        for i, unit in enumerate(_CARTESIAN_UNITS)
    )

    # AUX — provenance carried alongside the core state.
    channels.extend(
        ChannelSpec(key=f"joint_velocities[{i}]", role=ChannelRole.AUX, unit="rad/s")
        for i in range(7)
    )
    channels.extend(
        ChannelSpec(
            key=f"joint_torques_computed[{i}]", role=ChannelRole.AUX, unit="N·m"
        )
        for i in range(7)
    )
    channels.extend(
        ChannelSpec(key=f"wrench_state[{i}]", role=ChannelRole.AUX, unit=unit)
        for i, unit in enumerate(_WRENCH_UNITS)
    )

    return tuple(channels)


def obs_dim() -> int:
    """Expected length of the robot-obs value vector (== len(build_obs_channels()))."""
    return sum(count for _, count, _ in _OBS_FIELDS)


def flatten_observation(observation: Mapping[str, Any]) -> tuple[float, ...]:
    """Project one RobotEnv observation map onto the robot-obs value vector.

    ``observation`` is the ``dict[str, robotenv_pb2.Value]`` produced by
    ``VegaRobotEnvService._create_observation``. Returns floats in
    ``build_obs_channels()`` order. Raises ``KeyError`` if a declared field is
    absent and ``ValueError`` if an array field's length disagrees with the
    declared channel count — both are contract violations the caller should fix,
    not silently paper over.
    """
    values: list[float] = []
    for key, count, scalar in _OBS_FIELDS:
        value = observation[key]
        if scalar:
            values.append(float(value.float_value))
            continue
        array = list(value.float_array.values)
        if len(array) != count:
            raise ValueError(
                f"robot-obs field {key!r} carries {len(array)} values, expected {count}"
            )
        values.extend(float(item) for item in array)
    return tuple(values)
