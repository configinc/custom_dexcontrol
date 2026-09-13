"""Tests for the value-preserving RobotEnv-to-Node observation projection."""

from __future__ import annotations

import pytest
from conftest import arr, make_observation, scalar

from loop_bridge.contracts import float64_values
from loop_bridge.robot_obs import observation_payload, observation_state


def test_observation_state_preserves_every_legacy_value() -> None:
    state = observation_state(make_observation())

    assert state == {
        "joint_positions": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "gripper_position": 0.5,
        "cartesian_position": [10.0, 11.0, 12.0, 0.1, 0.2, 0.3],
        "joint_velocities": [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0],
        "joint_torques_computed": [31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0],
        "wrench_state": [41.0, 42.0, 43.0, 0.4, 0.5, 0.6],
    }


def test_observation_payload_uses_canonical_arm_fields_and_tensors() -> None:
    payload = observation_payload(make_observation(), "left")

    assert set(payload) == {
        "left.joint_positions",
        "left.gripper_position",
        "left.cartesian_position",
        "left.joint_velocities",
        "left.joint_torques_computed",
        "left.wrench_state",
    }
    assert payload["left.gripper_position"] == 0.5
    assert float64_values(
        payload["left.cartesian_position"],  # type: ignore[arg-type]
        field_name="left.cartesian_position",
        shape=(6,),
    ) == (10.0, 11.0, 12.0, 0.1, 0.2, 0.3)


def test_observation_state_ignores_undeclared_robotenv_fields() -> None:
    state = observation_state(make_observation())
    assert "timestamp" not in state
    assert "prev_command_successful" not in state


def test_observation_state_raises_on_wrong_array_length() -> None:
    observation = make_observation()
    observation["joint_positions"] = arr([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="joint_positions"):
        observation_state(observation)


def test_observation_state_raises_on_missing_field() -> None:
    observation = make_observation()
    del observation["wrench_state"]
    with pytest.raises(KeyError):
        observation_state(observation)


def test_observation_state_reads_gripper_scalar() -> None:
    observation = make_observation()
    observation["gripper_position"] = scalar(0.9)
    assert observation_state(observation)["gripper_position"] == 0.9
