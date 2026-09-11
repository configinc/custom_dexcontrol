"""Tests for merging both arm observations into one Robot Node payload."""

from __future__ import annotations

from conftest import make_observation

from loop_bridge.obs_publisher import merge_observations
from loop_bridge.robot_obs import observation_payload


def test_merge_two_arms_into_one_payload() -> None:
    left = make_observation()
    right = make_observation()

    payload = merge_observations({"left": left, "right": right})

    assert payload == {
        **observation_payload(left, "left"),
        **observation_payload(right, "right"),
    }
    assert any(field.startswith("left.") for field in payload)
    assert any(field.startswith("right.") for field in payload)
