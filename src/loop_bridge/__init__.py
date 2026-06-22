"""Loop Source Bus bridge for the Vega controller.

Publishes Vega's RobotEnv observations onto Loop's Source Bus as ``robot-obs``,
in-process with the RobotEnv server. See ``source_server.LoopVegaRobotEnvService``
and the ``python -m loop_bridge`` launcher.
"""

from __future__ import annotations

from loop_bridge.obs_publisher import RobotObsPublisher
from loop_bridge.robot_obs import (
    build_obs_channels,
    flatten_observation,
    obs_dim,
)

__all__ = [
    "RobotObsPublisher",
    "build_obs_channels",
    "flatten_observation",
    "obs_dim",
]
