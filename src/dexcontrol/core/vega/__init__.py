"""Vega robot interfaces."""

from dexcontrol.core.vega.robot import (
    CommunicationFailedError,
    IKFailedError,
    JointLimitExceededError,
    VegaRobot,
)

__all__ = [
    "VegaRobot",
    "JointLimitExceededError",
    "IKFailedError",
    "CommunicationFailedError",
]
