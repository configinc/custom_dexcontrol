"""Tests for the Vega robot-obs observation projection."""

from __future__ import annotations

import pytest
from conftest import arr, make_observation, scalar

from loop_bridge.robot_obs import observation_to_step

_PREFIX = "robot0.observation.state"

# joint_positions(7) + gripper(1) + cartesian(6) + joint_velocities(7) + joint_torques_computed(7) + wrench_state(6)
_EXPECTED_KEYS = {
    *[f"{_PREFIX}.joint_positions[{i}]" for i in range(7)],
    f"{_PREFIX}.gripper_position",
    *[f"{_PREFIX}.cartesian_position[{i}]" for i in range(6)],
    *[f"{_PREFIX}.joint_velocities[{i}]" for i in range(7)],
    *[f"{_PREFIX}.joint_torques_computed[{i}]" for i in range(7)],
    *[f"{_PREFIX}.wrench_state[{i}]" for i in range(6)],
}


def test_step_keys_are_the_projected_layout():
    assert set(observation_to_step(make_observation())) == _EXPECTED_KEYS


def test_step_projects_values_in_field_layout():
    step = observation_to_step(make_observation())
    assert [step[f"{_PREFIX}.joint_positions[{i}]"] for i in range(7)] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]
    assert step[f"{_PREFIX}.gripper_position"] == 0.5
    assert [step[f"{_PREFIX}.cartesian_position[{i}]"] for i in range(6)] == [
        10.0,
        11.0,
        12.0,
        0.1,
        0.2,
        0.3,
    ]
    assert [step[f"{_PREFIX}.joint_torques_computed[{i}]"] for i in range(7)] == [
        31,
        32,
        33,
        34,
        35,
        36,
        37,
    ]
    assert [step[f"{_PREFIX}.wrench_state[{i}]"] for i in range(6)] == [
        41.0,
        42.0,
        43.0,
        0.4,
        0.5,
        0.6,
    ]


def test_step_values_are_plain_floats():
    assert all(
        isinstance(v, float) for v in observation_to_step(make_observation()).values()
    )


def test_arm_prefix_namespaces_every_key():
    keys = observation_to_step(make_observation(), arm_prefix="robot1")
    assert all(k.startswith("robot1.observation.state.") for k in keys)
    assert all(".action." not in k for k in keys)  # obs/action are sibling namespaces


def test_step_ignores_undeclared_fields():
    step = observation_to_step(make_observation())
    assert not any("timestamp" in k for k in step)
    assert len(step) == len(_EXPECTED_KEYS)


def test_step_raises_on_wrong_array_length():
    obs = make_observation()
    obs["joint_positions"] = arr([1.0, 2.0, 3.0])  # 3, expected 7
    with pytest.raises(ValueError, match="joint_positions"):
        observation_to_step(obs)


def test_step_raises_on_missing_field():
    obs = make_observation()
    del obs["wrench_state"]
    with pytest.raises(KeyError):
        observation_to_step(obs)


def test_step_reads_gripper_from_scalar_field():
    obs = make_observation()
    obs["gripper_position"] = scalar(0.9)
    assert observation_to_step(obs)[f"{_PREFIX}.gripper_position"] == 0.9
