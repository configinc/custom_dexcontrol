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
from typing import Any, Mapping, Protocol, Sequence

from loop_sdk import RobotStepSender

from loop_bridge.robot_obs import (
    DEFAULT_ARM_PREFIX,
    build_obs_channels,
    observation_to_step,
)

DEFAULT_OBS_SOURCE_ID = "robot-obs"
DEFAULT_OBS_SOURCE_NAME = "vega robot state"


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

    def __init__(
        self, sender: _Sender, arm_prefixes: Sequence[str] = (DEFAULT_ARM_PREFIX,)
    ) -> None:
        self._sender = sender
        self._arm_prefixes = tuple(arm_prefixes)
        self._lock = threading.Lock()

    @classmethod
    def connect(
        cls,
        loop_addr: str,
        source_id: str = DEFAULT_OBS_SOURCE_ID,
        name: str = DEFAULT_OBS_SOURCE_NAME,
        arm_prefixes: Sequence[str] = (DEFAULT_ARM_PREFIX,),
    ) -> RobotObsPublisher:
        """Open a Source Bus sender and declare every arm's robot-obs channels.

        Declares the full layout up front (every arm's channels concatenated) so the
        source reaches "ready" before any data flows. (Config negotiation —
        ``RobotStepSender``'s ``options``/``apply_config`` — is not wired here yet;
        add it when a caller needs to advertise control rates.)
        """
        channels: list[Any] = []
        for arm_prefix in arm_prefixes:
            channels.extend(build_obs_channels(arm_prefix))
        sender = RobotStepSender(loop_addr, source_id, name=name)
        sender.connect()
        sender.declare(tuple(channels))
        return cls(sender, arm_prefixes)

    def publish(
        self, observations: Mapping[str, Mapping[str, Any]], timestamp_us: int
    ) -> bool:
        """Merge each arm's observation into one ``robot-obs`` sample and send it.

        ``observations`` maps each ``robotN.`` arm prefix to that arm's RobotEnv
        observation map. Returns whether the send was accepted (the sender drops,
        never raises, on a transport hiccup so the poll loop keeps running).
        """
        step: dict[str, float] = {}
        for arm_prefix, observation in observations.items():
            step.update(observation_to_step(observation, arm_prefix))
        with self._lock:
            return bool(self._sender.send(timestamp_us, step))

    def close(self) -> None:
        self._sender.disconnect()
