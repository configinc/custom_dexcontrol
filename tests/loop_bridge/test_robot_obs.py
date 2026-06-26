"""Tests for the robot-obs channel layout and observation projection."""

from __future__ import annotations

import pytest
from conftest import arr, make_observation, scalar
from loop_sdk import ChannelRole

from loop_bridge.robot_obs import (
    build_obs_channels,
    obs_dim,
    observation_to_step,
)

_PREFIX = "robot0.observation.state"


def test_obs_dim_matches_channels_and_field_layout():
    channels = build_obs_channels()
    # 7 joints + 1 gripper + 6 cartesian + 7 vel + 7 torque + 6 wrench
    assert obs_dim() == 34
    assert len(channels) == obs_dim()


def test_core_channels_are_the_essential_measured_state():
    core_keys = [c.key for c in build_obs_channels() if c.role is ChannelRole.CORE]
    assert core_keys == [
        *[f"{_PREFIX}.joint_positions[{i}]" for i in range(7)],
        f"{_PREFIX}.gripper_position",
        *[f"{_PREFIX}.cartesian_position[{i}]" for i in range(6)],
    ]


def test_aux_channels_are_provenance():
    aux_keys = [c.key for c in build_obs_channels() if c.role is ChannelRole.AUX]
    assert aux_keys == [
        *[f"{_PREFIX}.joint_velocities[{i}]" for i in range(7)],
        *[f"{_PREFIX}.joint_torques_computed[{i}]" for i in range(7)],
        *[f"{_PREFIX}.wrench_state[{i}]" for i in range(6)],
    ]


def test_arm_prefix_namespaces_every_channel():
    keys = [c.key for c in build_obs_channels(arm_prefix="robot1")]
    assert all(k.startswith("robot1.observation.state.") for k in keys)
    assert all(".action." not in k for k in keys)  # obs/action are sibling namespaces


def test_units_and_gripper_range():
    by_key = {c.key: c for c in build_obs_channels()}
    assert by_key[f"{_PREFIX}.joint_positions[0]"].unit == "rad"
    assert by_key[f"{_PREFIX}.joint_velocities[0]"].unit == "rad/s"
    assert by_key[f"{_PREFIX}.joint_torques_computed[0]"].unit == "N·m"
    assert by_key[f"{_PREFIX}.cartesian_position[0]"].unit == "m"
    assert by_key[f"{_PREFIX}.cartesian_position[3]"].unit == "rad"
    assert by_key[f"{_PREFIX}.wrench_state[0]"].unit == "N"
    assert by_key[f"{_PREFIX}.wrench_state[3]"].unit == "N·m"
    assert by_key[f"{_PREFIX}.gripper_position"].unit == "normalized"
    assert by_key[f"{_PREFIX}.gripper_position"].range == (0.0, 1.0)


def test_step_keys_match_declared_channels():
    step = observation_to_step(make_observation())
    assert set(step) == {c.key for c in build_obs_channels()}


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
    step = observation_to_step(make_observation())
    assert all(isinstance(v, float) for v in step.values())


def test_step_ignores_undeclared_fields():
    step = observation_to_step(make_observation())
    assert not any("timestamp" in k for k in step)
    assert len(step) == obs_dim()


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
