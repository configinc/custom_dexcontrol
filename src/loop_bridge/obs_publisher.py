"""Publish Vega RobotEnv observations onto Loop's Source Bus as ``robot-obs``.

``RobotObsPublisher`` wraps a loop-sdk ``SourceProducer`` and turns each
RobotEnv observation map into one ``robot-obs`` sample (flat value vector +
monotonic sequence). It owns no hardware and only depends on loop-sdk, so it is
testable with a fake producer — no robot and no running Source Bus required.

The subclass in ``source_server.py`` wires this to the live RobotEnv server;
this module keeps the bus-facing logic isolated and unit-testable.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Protocol

from loop_sdk import ChannelSpec, RobotStreamSchema, SourceProducer, SourceSchema

from loop_bridge.robot_obs import build_obs_channels, flatten_observation

DEFAULT_OBS_SOURCE_ID = "robot-obs"
DEFAULT_OBS_SOURCE_NAME = "vega robot state"


class _Producer(Protocol):
    """The slice of ``SourceProducer`` this publisher needs (eases testing)."""

    def send_robot(
        self,
        source_id: str,
        timestamp_us: int,
        sequence: int,
        values: tuple[float, ...],
    ) -> None: ...

    def close(self) -> None: ...


class RobotObsPublisher:
    """Publishes RobotEnv observations to a single ``robot-obs`` source.

    Sequence numbers are assigned monotonically per published sample, guarded by
    a lock because the RobotEnv server serves Step/Reset/status on a gRPC thread
    pool and may publish concurrently.
    """

    def __init__(
        self, producer: _Producer, source_id: str = DEFAULT_OBS_SOURCE_ID
    ) -> None:
        self._producer = producer
        self._source_id = source_id
        self._sequence = 0
        self._lock = threading.Lock()

    @classmethod
    def connect(
        cls,
        loop_addr: str,
        source_id: str = DEFAULT_OBS_SOURCE_ID,
        name: str = DEFAULT_OBS_SOURCE_NAME,
        channels: tuple[ChannelSpec, ...] | None = None,
    ) -> RobotObsPublisher:
        """Open a Source Bus producer declaring one robot-obs source."""
        declared = channels if channels is not None else build_obs_channels()
        producer = SourceProducer.connect(
            loop_addr=loop_addr,
            schema=SourceSchema(
                robot=(
                    RobotStreamSchema(
                        source_id=source_id, name=name, channels=declared
                    ),
                )
            ),
        )
        return cls(producer, source_id)

    def publish(self, observation: Mapping[str, Any], timestamp_us: int) -> int:
        """Flatten one observation and send it as the next robot-obs sample.

        Returns the sequence number assigned to this sample.
        """
        values = flatten_observation(observation)
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
        self._producer.send_robot(
            self._source_id, timestamp_us=timestamp_us, sequence=sequence, values=values
        )
        return sequence

    def close(self) -> None:
        self._producer.close()
