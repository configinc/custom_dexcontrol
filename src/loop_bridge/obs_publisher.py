"""Publish Vega RobotEnv observations onto Loop's Source Bus as ``robot-obs``.

``RobotObsPublisher`` wraps a loop-sdk ``RobotStepSender`` (the ergonomic front
door over the raw producer) and turns each RobotEnv observation map into one
``robot-obs`` sample — a ``{channel_key: reading}`` dict in the RCI wire format
(named, namespaced channels; the SDK assigns sequence numbers and omits absent
readings). It owns no hardware and only depends on loop-sdk, so it is testable
with a fake sender — no robot and no running Source Bus required.

The subclass in ``source_server.py`` wires this to the live RobotEnv server;
this module keeps the bus-facing logic isolated and unit-testable.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Mapping, Optional, Protocol

from loop_sdk import RobotConfig, RobotConfigOptions, RobotStepSender

from loop_bridge.robot_obs import (
    DEFAULT_ARM_PREFIX,
    build_obs_channels,
    observation_to_step,
)

DEFAULT_OBS_SOURCE_ID = "robot-obs"
DEFAULT_OBS_SOURCE_NAME = "vega robot state"

ApplyConfig = Callable[[RobotConfig], Optional[RobotConfig]]


class _Sender(Protocol):
    """The slice of ``RobotStepSender`` this publisher needs (eases testing)."""

    def send(
        self, timestamp_us: int, step: Mapping[str, Any], *, sequence: int | None = ...
    ) -> bool: ...

    def disconnect(self) -> None: ...


class RobotObsPublisher:
    """Publishes RobotEnv observations to a single ``robot-obs`` source.

    The wrapped ``RobotStepSender`` owns sequence numbering and the (declared
    once) channel layout. The lock guards concurrent publishes because the obs
    poll loop and any other caller may publish from different threads.
    """

    def __init__(self, sender: _Sender, arm_prefix: str = DEFAULT_ARM_PREFIX) -> None:
        self._sender = sender
        self._arm_prefix = arm_prefix
        self._lock = threading.Lock()

    @classmethod
    def connect(
        cls,
        loop_addr: str,
        source_id: str = DEFAULT_OBS_SOURCE_ID,
        name: str = DEFAULT_OBS_SOURCE_NAME,
        arm_prefix: str = DEFAULT_ARM_PREFIX,
        options: Optional[RobotConfigOptions] = None,
        apply_config: Optional[ApplyConfig] = None,
    ) -> RobotObsPublisher:
        """Open a Source Bus sender and declare the robot-obs channel layout.

        Declaring up front (rather than letting the first ``send`` do it) lets the
        Source Bus negotiate ``options`` and run ``apply_config`` before any data
        flows. Pass ``options`` to advertise the configs (e.g. control rates) this
        arm can open with; omit it for no negotiation.
        """
        sender = RobotStepSender(
            loop_addr, source_id, name=name, options=options, apply_config=apply_config
        )
        sender.connect()
        sender.declare(build_obs_channels(arm_prefix))
        return cls(sender, arm_prefix)

    def publish(self, observation: Mapping[str, Any], timestamp_us: int) -> bool:
        """Project one observation onto the step dict and send it as ``robot-obs``.

        Returns whether the send was accepted (the sender drops, never raises, on
        a transport hiccup so the poll loop keeps running).
        """
        step = observation_to_step(observation, self._arm_prefix)
        with self._lock:
            return bool(self._sender.send(timestamp_us, step))

    def close(self) -> None:
        self._sender.disconnect()
