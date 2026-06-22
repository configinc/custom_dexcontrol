"""Tests for the robot-obs channel layout and observation flattening."""

from __future__ import annotations

import pytest
from conftest import arr, make_observation, scalar
from loop_sdk import ChannelRole

from loop_bridge.robot_obs import (
    build_obs_channels,
    flatten_observation,
    obs_dim,
)


def test_obs_dim_matches_channels_and_field_layout():
    channels = build_obs_channels()
    # 7 joints + 1 gripper + 6 cartesian + 7 vel + 7 torque + 6 wrench
    assert obs_dim() == 34
    assert len(channels) == obs_dim()


def test_core_channels_are_the_essential_measured_state():
    channels = build_obs_channels()
    core_keys = [c.key for c in channels if c.role is ChannelRole.CORE]
    assert core_keys == [
        *[f"joint_positions[{i}]" for i in range(7)],
        "gripper_position",
        *[f"cartesian_position[{i}]" for i in range(6)],
    ]


def test_aux_channels_are_provenance():
    channels = build_obs_channels()
    aux_keys = [c.key for c in channels if c.role is ChannelRole.AUX]
    assert aux_keys == [
        *[f"joint_velocities[{i}]" for i in range(7)],
        *[f"joint_torques_computed[{i}]" for i in range(7)],
        *[f"wrench_state[{i}]" for i in range(6)],
    ]


def test_units_and_gripper_range():
    by_key = {c.key: c for c in build_obs_channels()}
    assert by_key["joint_positions[0]"].unit == "rad"
    assert by_key["joint_velocities[0]"].unit == "rad/s"
    # cartesian: xyz metres, rpy radians
    assert by_key["cartesian_position[0]"].unit == "m"
    assert by_key["cartesian_position[3]"].unit == "rad"
    # wrench: force N, torque N·m
    assert by_key["wrench_state[0]"].unit == "N"
    assert by_key["wrench_state[3]"].unit == "N·m"
    assert by_key["gripper_position"].unit == "normalized"
    assert by_key["gripper_position"].range == (0.0, 1.0)


def test_flatten_projects_values_in_channel_order():
    values = flatten_observation(make_observation())
    assert values == (
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,  # joint_positions
        0.5,  # gripper_position (scalar)
        10.0,
        11.0,
        12.0,
        0.1,
        0.2,
        0.3,  # cartesian_position
        21.0,
        22.0,
        23.0,
        24.0,
        25.0,
        26.0,
        27.0,  # joint_velocities
        31.0,
        32.0,
        33.0,
        34.0,
        35.0,
        36.0,
        37.0,  # joint_torques_computed
        41.0,
        42.0,
        43.0,
        0.4,
        0.5,
        0.6,  # wrench_state
    )
    assert len(values) == obs_dim()


def test_flatten_returns_plain_floats():
    values = flatten_observation(make_observation())
    assert all(isinstance(v, float) for v in values)


def test_flatten_raises_on_wrong_array_length():
    obs = make_observation()
    obs["joint_positions"] = arr([1.0, 2.0, 3.0])  # 3, expected 7
    with pytest.raises(ValueError, match="joint_positions"):
        flatten_observation(obs)


def test_flatten_raises_on_missing_field():
    obs = make_observation()
    del obs["wrench_state"]
    with pytest.raises(KeyError):
        flatten_observation(obs)


def test_flatten_reads_gripper_from_scalar_field():
    obs = make_observation()
    obs["gripper_position"] = scalar(0.9)
    values = flatten_observation(obs)
    assert values[7] == 0.9  # index right after the 7 joint positions
