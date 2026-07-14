"""Decode the device-side ``robot-action`` payload — the consume mirror of robot_obs.

``robot-action`` samples are ``RobotActionFrame`` vectors. The vector is opaque:
the source controller forwards it as-is and producers (teleop / policy) and this
robot server agree on its layout out-of-band (via ``action_space``). By that
agreement the vector is a plain concatenation of per-arm blocks in arm order,
each ``arm_dim`` long; this decoder slices one arm's block out by index.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping

# Wire value of the home/reset command on the robot-command lane (our convention).
HOME = "home"


# Per-arm action vector dimension for each Vega action space (from
# dexcontrol.core.robotenv_vega.server.Step's action shape gate at
# action_space.startswith("joint")). Joint spaces = 7 joints + 1 gripper;
# cartesian spaces = 6 cartesian + 1 gripper.
_ARM_DIM_BY_ACTION_SPACE: Mapping[str, int] = {
    "target_cartesian_delta": 7,
    "cartesian_delta": 7,
    "cartesian_velocity": 7,
    "target_cartesian": 7,
    "target_joint_position": 8,
    "target_joint_velocity": 8,
    "target_joint_effort": 8,
}


def arm_dim_for_action_space(action_space: str) -> int:
    """Per-arm action vector dimension for a given Vega action_space, or
    raises ``KeyError`` for an unknown space. The dim covers the arm's motor
    channels plus its gripper slot (the Vega Step wire concatenates them)."""
    return _ARM_DIM_BY_ACTION_SPACE[action_space]


def decode_action(values: Sequence[float], arm_index: int, arm_dim: int) -> list[float] | None:
    """Pull one arm's block out of an opaque ``RobotActionFrame.values`` vector.

    The vector is a plain concatenation of per-arm blocks in arm order, each
    ``arm_dim`` long (agreed out-of-band with the producers). Arm ``arm_index``
    owns ``values[arm_index * arm_dim : (arm_index + 1) * arm_dim]``. Returns that
    slice, or ``None`` when the vector is too short to cover this arm (e.g. a
    bootstrap / partial frame) so the caller skips this arm for the tick.
    """
    start = arm_index * arm_dim
    end = start + arm_dim
    if end > len(values):
        return None
    return [float(value) for value in values[start:end]]
