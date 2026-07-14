"""Tests for merge_observations — the device-side robot-obs merge (no hardware/bus)."""

from __future__ import annotations

from conftest import make_observation

from loop_bridge.obs_publisher import merge_observations
from loop_bridge.robot_obs import observation_to_step


def test_merge_single_arm():
    obs = make_observation()
    assert merge_observations({"robot0": obs}) == observation_to_step(obs, "robot0")


def test_merge_two_arms_into_one_sample():
    obs0, obs1 = make_observation(), make_observation()
    step = merge_observations({"robot0": obs0, "robot1": obs1})
    assert step == {
        **observation_to_step(obs0, "robot0"),
        **observation_to_step(obs1, "robot1"),
    }
    assert any(k.startswith("robot0.") for k in step)
    assert any(k.startswith("robot1.") for k in step)


def test_merge_namespaces_by_arm_prefix():
    step = merge_observations({"robot1": make_observation()})
    assert all(k.startswith("robot1.observation.state.") for k in step)
