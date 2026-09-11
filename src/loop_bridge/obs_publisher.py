"""Merge both per-arm Vega observations into one Robot Node payload."""

from __future__ import annotations

from typing import Any, Mapping

from loop_bridge.robot_obs import observation_payload


def merge_observations(observations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for arm, observation in observations.items():
        payload.update(observation_payload(observation, arm))
    return payload
