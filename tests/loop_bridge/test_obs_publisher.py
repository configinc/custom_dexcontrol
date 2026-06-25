"""Tests for RobotObsPublisher — the bus-facing publish logic (no hardware)."""

from __future__ import annotations

from conftest import FakeSender, make_observation

from loop_bridge.obs_publisher import RobotObsPublisher
from loop_bridge.robot_obs import observation_to_step


def test_publish_sends_projected_step():
    sender = FakeSender()
    publisher = RobotObsPublisher(sender, arm_prefixes=["robot0"])
    obs = make_observation()

    assert publisher.publish({"robot0": obs}, timestamp_us=111) is True

    call = sender.sent[0]
    assert call["timestamp_us"] == 111
    assert call["sequence"] == 0
    assert call["step"] == observation_to_step(obs, "robot0")


def test_publish_merges_two_arms_into_one_sample():
    sender = FakeSender()
    publisher = RobotObsPublisher(sender, arm_prefixes=["robot0", "robot1"])
    obs0, obs1 = make_observation(), make_observation()

    publisher.publish({"robot0": obs0, "robot1": obs1}, timestamp_us=5)

    step = sender.sent[0]["step"]
    assert step == {
        **observation_to_step(obs0, "robot0"),
        **observation_to_step(obs1, "robot1"),
    }
    assert any(k.startswith("robot0.") for k in step)
    assert any(k.startswith("robot1.") for k in step)


def test_publish_lets_sender_assign_monotonic_sequence():
    sender = FakeSender()
    publisher = RobotObsPublisher(sender, arm_prefixes=["robot0"])

    for ts in (10, 20, 30):
        publisher.publish({"robot0": make_observation()}, timestamp_us=ts)

    assert [c["sequence"] for c in sender.sent] == [0, 1, 2]
    assert [c["timestamp_us"] for c in sender.sent] == [10, 20, 30]


def test_publish_uses_configured_arm_prefix():
    sender = FakeSender()
    publisher = RobotObsPublisher(sender, arm_prefixes=["robot1"])
    publisher.publish({"robot1": make_observation()}, timestamp_us=5)
    assert all(
        k.startswith("robot1.observation.robot_state.") for k in sender.sent[0]["step"]
    )


def test_close_disconnects_sender():
    sender = FakeSender()
    publisher = RobotObsPublisher(sender, arm_prefixes=["robot0"])
    publisher.close()
    assert sender.disconnected is True
