"""Loop Source Bus bridge for the Vega controller.

Bridges Vega's RobotEnv to Loop's Source Bus in-process with the RobotEnv server:
it publishes observations as ``robot-obs`` and executes ``robot-action`` it
consumes from the bus through the RobotEnv ``Step`` path. See
``source_server.LoopVegaRobotEnvService`` and the ``python -m loop_bridge`` launcher.

Note: importing ``source_server`` pulls in the in-process ``dexcontrol`` robot
stack; the bus-only modules below do not, so they stay unit-testable on their own.
"""

from __future__ import annotations

from loop_bridge.action_consumer import RobotActionConsumer
from loop_bridge.obs_publisher import RobotObsPublisher
from loop_bridge.robot_action import action_from_state
from loop_bridge.robot_obs import (
    build_obs_channels,
    obs_dim,
    observation_to_step,
)

__all__ = [
    "RobotActionConsumer",
    "RobotObsPublisher",
    "action_from_state",
    "build_obs_channels",
    "obs_dim",
    "observation_to_step",
]
