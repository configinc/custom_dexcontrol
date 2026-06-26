"""Loop Source Bus bridge for the Vega controller.

Bridges Vega's RobotEnv to Loop's Source Bus in-process with the RobotEnv server:
it publishes observations as ``robot-obs`` and executes ``robot-action`` it pulls
from the bus (via loop-sdk's ``RobotActionReceiver``) through the RobotEnv ``Step``
path. See ``source_server`` (``serve_with_loop`` / ``serve_dual_arm``) and the
``python -m loop_bridge`` launcher.

Note: importing ``source_server`` pulls in the in-process ``dexcontrol`` robot
stack; the bus-only modules below do not, so they stay unit-testable on their own.
"""

from __future__ import annotations

from loop_bridge.obs_publisher import RobotObsPublisher
from loop_bridge.robot_obs import observation_to_step

__all__ = [
    "RobotObsPublisher",
    "observation_to_step",
]
