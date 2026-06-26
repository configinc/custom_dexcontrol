"""Tests for robot-action decode (RobotFrame.state -> flat action vector)."""

from __future__ import annotations

import pytest

from loop_bridge.robot_action import DEFAULT_ACTION_SPACE, action_from_state

_CART = DEFAULT_ACTION_SPACE  # target_cartesian_delta -> expects 7
_FREE = "freeform"  # not width-checked


def _delta(values, space=_FREE, arm="robot0"):
    return {f"{arm}.action.{space}[{i}]": v for i, v in enumerate(values)}


def test_decodes_full_cartesian_vector():
    state = _delta([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0], space=_CART)
    assert action_from_state(state) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]


def test_returns_none_when_no_action_for_arm():
    state = _delta([float(i) for i in range(7)], space=_CART, arm="robot1")
    assert action_from_state(state, arm_prefix="robot0") is None


def test_filters_to_requested_arm():
    state = {**_delta([1.0, 2.0]), **_delta([99.0], arm="robot1")}
    assert action_from_state(state, arm_prefix="robot0", action_space=_FREE) == [
        1.0,
        2.0,
    ]


def test_returns_none_on_empty_state():
    assert action_from_state({}) is None


def test_ignores_unrelated_keys_and_other_spaces():
    state = {
        **_delta([1.0, 2.0, 3.0]),
        "robot0.observation.state.joint_positions[0]": 7.0,
        "robot0.action.joint_position[0]": 5.0,
    }
    assert action_from_state(state, action_space=_FREE) == [1.0, 2.0, 3.0]


def test_single_element_list_value_is_coerced():
    state = {f"robot0.action.{_FREE}[0]": [1.5], f"robot0.action.{_FREE}[1]": 2.5}
    assert action_from_state(state, action_space=_FREE) == [1.5, 2.5]


def test_skips_none_valued_trailing_channel():
    state = _delta([1.0, 2.0, 3.0])
    state[f"robot0.action.{_FREE}[3]"] = None
    assert action_from_state(state, action_space=_FREE) == [1.0, 2.0, 3.0]


def test_raises_on_non_contiguous_indices():
    state = {f"robot0.action.{_FREE}[0]": 1.0, f"robot0.action.{_FREE}[2]": 3.0}
    with pytest.raises(ValueError, match="non-contiguous"):
        action_from_state(state, action_space=_FREE)


def test_raises_on_negative_index():
    state = {f"robot0.action.{_FREE}[-1]": 1.0}
    with pytest.raises(ValueError, match="negative index"):
        action_from_state(state, action_space=_FREE)


def test_rejects_bool_value():
    state = {f"robot0.action.{_FREE}[0]": True}
    with pytest.raises(ValueError, match="bool"):
        action_from_state(state, action_space=_FREE)


def test_raises_on_truncated_cartesian_vector():
    state = _delta([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], space=_CART)  # 6, expected 7
    with pytest.raises(ValueError, match="expected 7"):
        action_from_state(state)


def test_raises_on_truncated_joint_vector():
    state = _delta(
        [float(i) for i in range(7)], space="joint_position"
    )  # 7, expected 8
    with pytest.raises(ValueError, match="expected 8"):
        action_from_state(state, action_space="joint_position")


def test_custom_action_space_full_width():
    state = _delta([float(i) for i in range(8)], space="joint_position")
    assert action_from_state(state, action_space="joint_position") == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]
