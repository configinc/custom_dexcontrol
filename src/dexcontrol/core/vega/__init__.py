"""Vega robot interfaces."""

from .robot import (
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
