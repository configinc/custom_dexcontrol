"""Tests for Robot.set_joint_target() validation against managed components.

Note: Robot.set_joint_target wraps all exceptions raised during validation
in DexcontrolError (with the original as __cause__), matching the existing
behaviour for validate_component_names. Tests expect the wrapped form.
"""

import numpy as np
import pytest

from dexcontrol.core.component import (
    ManagedJointComponent,
    MotionPluginManaged,
    RobotJointComponent,
)
from dexcontrol.exceptions import DexcontrolError


class _FakeArm(ManagedJointComponent):
    """Stand-in for Arm that skips real initialisation (no node, no zenoh)."""

    def __init__(self):
        self._joint_name = ["j0"]

    def set_joint_target(self, pos, scale=None, relative=False, tracked=False):
        return None


class _FakeHand(RobotJointComponent):
    """Stand-in for Hand that skips real initialisation."""

    def __init__(self):
        self._joint_name = ["j0"]


def test_validate_managed_targets_rejects_non_managed(monkeypatch):
    """Robot.set_joint_target should raise DexcontrolError naming the bad component."""
    from dexcontrol.robot import Robot

    # Build a Robot stub that returns our fakes from get_controllable_component_map.
    robot = Robot.__new__(Robot)
    arm = _FakeArm()
    hand = _FakeHand()
    monkeypatch.setattr(
        robot,
        "get_controllable_component_map",
        lambda: {"left_arm": arm, "left_hand": hand},
    )

    with pytest.raises(DexcontrolError) as exc_info:
        robot.set_joint_target({"left_hand": np.zeros(1)})

    # The underlying validation error is a ValueError.
    assert isinstance(exc_info.value.__cause__, ValueError)
    msg = str(exc_info.value)
    assert "left_hand" in msg
    assert "left_arm" in msg  # supported components listed


def test_validate_managed_targets_accepts_managed(monkeypatch):
    """When only managed components are targeted, no validation error is raised."""
    from dexcontrol.robot import Robot

    robot = Robot.__new__(Robot)
    arm = _FakeArm()
    monkeypatch.setattr(
        robot, "get_controllable_component_map", lambda: {"left_arm": arm}
    )

    # Should not raise. Result is None because _FakeArm.set_joint_target returns None
    # and tracked defaults to False.
    result = robot.set_joint_target({"left_arm": np.zeros(1)})
    assert result is None


def test_fake_arm_is_managed():
    """Sanity: _FakeArm classifies as ManagedJointComponent + MotionPluginManaged."""
    arm = _FakeArm()
    assert isinstance(arm, ManagedJointComponent)
    assert isinstance(arm, MotionPluginManaged)
    assert isinstance(arm, RobotJointComponent)
