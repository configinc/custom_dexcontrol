"""robot-obs projection for the Vega RobotEnv server (Loop Source Bus).

``observation_to_step()`` projects one RobotEnv observation onto the ``robot-obs``
step dict the bridge hands to ``RobotStepSender.send`` — ``{<arm>.observation.
state.<field>[i]: reading}``. That's it: the SDK declares the layout from the first
frame's keys and the recorder takes its columns from the streamed keys, so there
is no ``ChannelSpec`` layout to build/declare here.

Channel keys follow the RCI wire convention ``<arm>.observation.state.<field>[i]``
(a sibling of the ``<arm>.action.<space>[i]`` keys), arm-prefixed so observation
and action keys never collide when the loop pairs them into a robot-step. The
values are the SAME numbers ``VegaRobotEnvService._create_observation`` returns;
only the shape changes (named ``Value`` map -> named-channel dict). Pure and
testable: no robot, no Source Bus.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# Wire-key infix. The recorder stores obs columns as ``observation.state.*`` (its
# old transform renamed ``robot_state``->``state``); now that the recorder no
# longer transforms, we emit that final column name directly. The native RobotEnv
# obs_dict is still read by field name, independent of this wire infix.
_OBS_NAMESPACE = "observation.state"

# Default arm prefix for a single-arm bridge. Dual-arm units run one bridge with
# two services (robot0 / robot1) and pass the matching prefix.
DEFAULT_ARM_PREFIX = "robot0"

# Ordered (proto_key, count, scalar) projection of the RobotEnv observation map.
# ``scalar`` reads ``Value.float_value``; otherwise ``Value.float_array.values``
# (which must hold exactly ``count``).
_OBS_FIELDS: tuple[tuple[str, int, bool], ...] = (
    ("joint_positions", 7, False),
    ("gripper_position", 1, True),
    ("cartesian_position", 6, False),
    ("joint_velocities", 7, False),
    ("joint_torques_computed", 7, False),
    ("wrench_state", 6, False),
)


def _channel_key(arm_prefix: str, field: str, index: Optional[int]) -> str:
    """One namespaced channel key. Scalar fields omit the ``[i]`` suffix."""
    base = f"{arm_prefix}.{_OBS_NAMESPACE}.{field}"
    return base if index is None else f"{base}[{index}]"


def observation_to_step(
    observation: Mapping[str, Any], arm_prefix: str = DEFAULT_ARM_PREFIX
) -> dict[str, float]:
    """Project one RobotEnv observation map onto the robot-obs step dict.

    ``observation`` is the ``dict[str, robotenv_pb2.Value]`` produced by
    ``VegaRobotEnvService._create_observation``. Returns ``{channel_key: reading}``
    ready for ``RobotStepSender.send``. Raises ``KeyError`` if a projected field is
    absent and ``ValueError`` if an array field's length disagrees with its declared
    count — both are contract violations to fix, not silently paper over.
    """
    step: dict[str, float] = {}
    for field, count, scalar in _OBS_FIELDS:
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
