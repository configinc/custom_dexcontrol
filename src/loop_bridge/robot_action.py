"""robot-action decode for the Vega RobotEnv bridge (Loop Source Bus).

The mirror of ``robot_obs``: the loop publishes a ``robot-action`` source whose
channels follow the RCI wire convention ``<arm>.action.<space>[i]`` (a sibling of
the ``<arm>.observation.robot_state.<field>[i]`` obs keys). This module turns one
``RobotFrame.state`` dict back into the flat action vector the RobotEnv ``Step``
path takes, in the negotiated ``action_space`` (e.g. ``target_cartesian_delta``:
6 cartesian terms + 1 gripper).

Pure and testable: no robot, no Source Bus. ``action_from_state`` returns ``None``
for a tick that carries no action for this arm (a "no command" tick the caller
skips) and raises ``ValueError`` on a malformed layout (non-contiguous indices,
or a width that disagrees with the action space) — refusing to command a real arm
a wrong-length vector rather than silently truncating it.
"""

from __future__ import annotations

from typing import Mapping, Optional

from loop_sdk import RobotStateValue

from loop_bridge.robot_obs import DEFAULT_ARM_PREFIX

# The action space the teleop/RCI lane emits by default — Vega end-effector delta
# with teleop gains applied (6 cartesian terms + 1 gripper).
DEFAULT_ACTION_SPACE = "target_cartesian_delta"

# Expected action-vector width per known action space. Cartesian spaces are 6
# pose terms + 1 gripper; joint spaces are 7 joints + 1 gripper. A decoded vector
# whose length disagrees is rejected rather than executed. Spaces absent from this
# map are not width-checked.
_EXPECTED_DIM = {
    "cartesian_velocity": 7,
    "cartesian_delta": 7,
    "target_cartesian_delta": 7,
    "joint_position": 8,
    "joint_velocity": 8,
    "joint_delta": 8,
}


def _action_prefix(arm_prefix: str, action_space: str) -> str:
    return f"{arm_prefix}.action.{action_space}"


def _as_float(value: RobotStateValue) -> float:
    """Coerce one channel reading to a scalar float (channels are per-index)."""
    if isinstance(value, bool):  # bool is an int subclass; reject the ambiguity
        raise ValueError(f"action channel carries a bool, expected a scalar: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 1:
        return float(value[0])
    raise ValueError(f"action channel carries {value!r}, expected a scalar")


def action_from_state(
    state: Mapping[str, RobotStateValue],
    arm_prefix: str = DEFAULT_ARM_PREFIX,
    action_space: str = DEFAULT_ACTION_SPACE,
) -> Optional[list[float]]:
    """Project one ``robot-action`` frame's state onto the flat action vector.

    Reads the contiguous ``<arm>.action.<space>[0..n]`` channels present in
    ``state``. Returns the ordered ``[v0, v1, ...]`` vector, or ``None`` if this
    frame carries no action channel for ``arm_prefix``/``action_space`` (e.g. the
    loop is holding — the caller skips the tick). Raises ``ValueError`` if the
    present indices are not a contiguous ``0..n`` run, or if the decoded width
    disagrees with the action space's expected dimension.
    """
    base = _action_prefix(arm_prefix, action_space)
    indexed: dict[int, float] = {}
    for key, value in state.items():
        if not key.startswith(base + "[") or not key.endswith("]"):
            continue
        if value is None:  # "no reading this tick" for a declared channel
            continue
        index_text = key[len(base) + 1 : -1]
        try:
            index = int(index_text)
        except ValueError as error:
            raise ValueError(f"malformed action channel key {key!r}") from error
        if index < 0:
            raise ValueError(f"action channel {key!r} has a negative index")
        indexed[index] = _as_float(value)

    if not indexed:
        return None

    highest = max(indexed)
    missing = [i for i in range(highest + 1) if i not in indexed]
    if missing:
        raise ValueError(
            f"action {base!r} has non-contiguous indices; missing {missing} (got {sorted(indexed)})"
        )
    vector = [indexed[i] for i in range(highest + 1)]

    expected = _EXPECTED_DIM.get(action_space)
    if expected is not None and len(vector) != expected:
        raise ValueError(
            f"action {base!r} decoded {len(vector)} values, expected {expected} for action_space {action_space!r}"
        )
    return vector
