"""Shared fakes for loop_bridge tests — no hardware, no running Source Bus."""

from __future__ import annotations

from typing import Any


class FakeFloatArray:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class FakeValue:
    """Duck-typed stand-in for robotenv_pb2.Value (only fields flatten reads)."""

    def __init__(
        self, float_value: float = 0.0, float_array: FakeFloatArray | None = None
    ) -> None:
        self.float_value = float_value
        self.float_array = float_array


def arr(values: list[float]) -> FakeValue:
    return FakeValue(float_array=FakeFloatArray(list(values)))


def scalar(value: float) -> FakeValue:
    return FakeValue(float_value=value)


def make_observation() -> dict[str, Any]:
    """A complete, well-formed Vega RobotEnv observation map with distinct values
    so ordering bugs in flatten are detectable."""
    return {
        "joint_positions": arr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        "gripper_position": scalar(0.5),
        "cartesian_position": arr([10.0, 11.0, 12.0, 0.1, 0.2, 0.3]),
        "joint_velocities": arr([21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0]),
        "joint_torques_computed": arr([31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0]),
        "wrench_state": arr([41.0, 42.0, 43.0, 0.4, 0.5, 0.6]),
        # extra RobotEnv fields the bus does not carry — must be ignored:
        "timestamp": scalar(999.0),
        "prev_command_successful": scalar(1.0),
    }


class FakeProducer:
    """Records send_robot / close calls in place of a loop-sdk SourceProducer."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    def send_robot(self, source_id, timestamp_us, sequence, values) -> None:
        self.sent.append(
            {
                "source_id": source_id,
                "timestamp_us": timestamp_us,
                "sequence": sequence,
                "values": values,
            }
        )

    def close(self) -> None:
        self.closed = True
