"""Tests for RobotObsPublisher — the bus-facing publish logic (no hardware)."""

from __future__ import annotations

from conftest import FakeProducer, make_observation

from loop_bridge.obs_publisher import RobotObsPublisher
from loop_bridge.robot_obs import flatten_observation


def test_publish_sends_flattened_values():
    producer = FakeProducer()
    publisher = RobotObsPublisher(producer, source_id="robot-obs")
    obs = make_observation()

    publisher.publish(obs, timestamp_us=111)

    assert len(producer.sent) == 1
    call = producer.sent[0]
    assert call["source_id"] == "robot-obs"
    assert call["timestamp_us"] == 111
    assert call["sequence"] == 0
    assert call["values"] == flatten_observation(obs)


def test_publish_assigns_monotonic_sequence():
    producer = FakeProducer()
    publisher = RobotObsPublisher(producer, source_id="robot-obs")

    seqs = [
        publisher.publish(make_observation(), timestamp_us=ts) for ts in (10, 20, 30)
    ]

    assert seqs == [0, 1, 2]
    assert [c["sequence"] for c in producer.sent] == [0, 1, 2]
    assert [c["timestamp_us"] for c in producer.sent] == [10, 20, 30]


def test_publish_returns_assigned_sequence():
    producer = FakeProducer()
    publisher = RobotObsPublisher(producer, source_id="robot-obs")
    assert publisher.publish(make_observation(), timestamp_us=1) == 0
    assert publisher.publish(make_observation(), timestamp_us=2) == 1


def test_close_closes_producer():
    producer = FakeProducer()
    publisher = RobotObsPublisher(producer, source_id="robot-obs")
    publisher.close()
    assert producer.closed is True


def test_custom_source_id_is_used():
    producer = FakeProducer()
    publisher = RobotObsPublisher(producer, source_id="cell3/robot-obs")
    publisher.publish(make_observation(), timestamp_us=5)
    assert producer.sent[0]["source_id"] == "cell3/robot-obs"
