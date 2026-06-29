"""Decode the device-side ``robot-action`` payload — the consume mirror of robot_obs.

loop-sdk is a generic dict-in/dict-out transport: ``LoopRobotClient.poll_action()``
hands back the freshest ``robot-action`` frame as a RAW ``{key: value}`` dict, and
turning that into a per-arm control vector is OUR contract, not the SDK's. This is
that decode — the consume counterpart of ``robot_obs.observation_to_step`` (which
builds the ``robot-obs`` keys). No hardware, no bus: testable on a plain dict.
"""

from __future__ import annotations

from typing import Any, Mapping

# Wire value of the home/reset command on the robot-command lane (our convention).
HOME = "home"


def decode_action(state: Mapping[str, Any], arm_prefix: str, action_space: str) -> list[float] | None:
    """Pull one arm's action vector out of a raw ``robot-action`` payload.

    Reads ``<arm_prefix>.action.<action_space>`` — either the native vector channel
    (a whole ``list``) or the flattened ``...[i]`` indices. Returns the vector, or
    ``None`` if this arm carries no action this frame. Raises ``ValueError`` on a gap
    in the indices (a producer wiring bug).
    """
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
