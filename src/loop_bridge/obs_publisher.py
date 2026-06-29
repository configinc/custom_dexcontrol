"""Project Vega RobotEnv observations into one ``robot-obs`` sample.

Per the robot source contract, a Vega robot (single- or dual-arm) is ONE source:
each arm's observation is projected onto its ``robotN.`` namespace and the arms
merge into one ``{channel_key: reading}`` dict in the RCI wire format. The bus side
(publish/receive) is owned by loop-sdk's ``LoopRobotClient``; this module is just
the device-side obs merge, testable with no robot and no bus.
"""

from __future__ import annotations

from typing import Any, Mapping

from loop_bridge.robot_obs import observation_to_step


def merge_observations(observations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Merge each arm's RobotEnv observation into one namespaced ``robot-obs`` dict.

    ``observations`` maps each ``robotN.`` arm prefix to that arm's observation map;
    the result is the merged sample to hand to ``LoopRobotClient.publish_obs``.
    """
    step: dict[str, Any] = {}
    for arm_prefix, observation in observations.items():
        step.update(observation_to_step(observation, arm_prefix))
    return step
