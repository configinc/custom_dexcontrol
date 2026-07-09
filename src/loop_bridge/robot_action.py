"""Decode the device-side ``robot-action`` payload — the consume mirror of robot_obs.

Post vector-symmetric refactor, ``robot-action`` samples are ``RobotActionFrame``
vectors aligned to the robot server's declared ``action_channels`` (see
``build_action_channels`` below). This decoder pulls one arm's slice out of that
vector using the declared layout — pure index arithmetic, no wire keys.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

# Wire value of the home/reset command on the robot-command lane (our convention).
HOME = "home"


def build_action_channels(
    arm_prefixes: Sequence[str], action_space: str, arm_dim: int
) -> tuple[str, ...]:
    """Return the ordered channel keys the robot server declares on its robot-obs
    source's ``action_channels``. Convention: per-arm the channel key is
    ``<arm_prefix>.action.<action_space>[i]`` for i in [0, arm_dim). Arms are
    concatenated in ``arm_prefixes`` order — this ordering defines the wire
    layout every ``teleop-action`` / ``policy-action`` / ``robot-action`` vector
    must match."""
    return tuple(
        f"{arm_prefix}.action.{action_space}[{index}]"
        for arm_prefix in arm_prefixes
        for index in range(arm_dim)
    )


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


def decode_action(
    values: Sequence[float],
    arm_channels: Sequence[str],
    arm_prefix: str,
    action_space: str,
) -> list[float] | None:
    """Pull one arm's action vector out of a ``RobotActionFrame.values`` vector.

    ``arm_channels`` is the source's declared ``action_channels`` — the vector
    positions this frame's values map to. Returns the contiguous slice matching
    ``<arm_prefix>.action.<action_space>[i]`` keys, or ``None`` when this arm has
    no such channels declared. Raises ``ValueError`` on a discontiguous slice
    (a producer wiring bug).
    """
    prefix = f"{arm_prefix}.action.{action_space}["
    hits = [i for i, channel in enumerate(arm_channels) if channel.startswith(prefix) and channel.endswith("]")]
    if not hits:
        return None
    if hits != list(range(hits[0], hits[0] + len(hits))):
        raise ValueError(f"non-contiguous action indices for {prefix!r} in declared action_channels")
    if hits[-1] + 1 > len(values):
        return None
    return [float(values[i]) for i in hits]


def decode_action_from_dict(
    state: Mapping[str, Any], arm_prefix: str, action_space: str
) -> list[float] | None:
    """Legacy dict-path decoder — kept for the migration window where callers
    still poll robot-action via the dict API (``LoopRobotClient.poll_action``).
    Reads ``<arm_prefix>.action.<action_space>`` as a native vector channel or
    flattened ``...[i]`` indices."""
    base = f"{arm_prefix}.action.{action_space}"
    native = state.get(base)
    if isinstance(native, (list, tuple)):
        return [float(value) for value in native]
    values: list[float] = []
    index = 0
    while f"{base}[{index}]" in state:
        values.append(float(state[f"{base}[{index}]"]))
        index += 1
    if not values:
        return None
    if f"{base}[{index + 1}]" in state:
        raise ValueError(f"non-contiguous action indices for {base!r}")
    return values
