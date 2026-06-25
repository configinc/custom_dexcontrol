"""Tests for RobotActionConsumer — subscribe robot-action, apply via the applier."""

from __future__ import annotations

import pytest
from conftest import FakeApplier, FakeConsumer
from loop_sdk import RobotFrame

from loop_bridge.action_consumer import RobotActionConsumer
from loop_bridge.robot_action import DEFAULT_ACTION_SPACE

_BASE = f"robot0.action.{DEFAULT_ACTION_SPACE}"


def _cart(gripper=1.0, sequence=0):
    """A valid 7-wide target_cartesian_delta frame (6 pose terms + 1 gripper)."""
    state = {
        f"{_BASE}[{i}]": v
        for i, v in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, gripper])
    }
    return RobotFrame(
        source_id="robot-action", timestamp_us=sequence, sequence=sequence, state=state
    )


def _consumer(frames=None, fault=None, applier=None, **kwargs):
    return RobotActionConsumer(
        FakeConsumer(frames=frames, fault=fault), applier or FakeApplier(), **kwargs
    )


def test_applies_each_action_via_step():
    applier = FakeApplier()
    rc = _consumer(frames=[_cart(1.0), _cart(0.0, 1)], applier=applier)

    rc.run()

    assert [s["action"] for s in applier.steps] == [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],
    ]
    assert applier.steps[0]["action_space"] == DEFAULT_ACTION_SPACE


def test_passes_gripper_action_space_through():
    applier = FakeApplier()
    rc = _consumer(frames=[_cart()], applier=applier, gripper_action_space="velocity")
    rc.run()
    assert applier.steps[0]["gripper_action_space"] == "velocity"


def test_subscribes_to_configured_source_id():
    consumer = FakeConsumer(frames=[])
    rc = RobotActionConsumer(consumer, FakeApplier(), source_id="cell3/robot-action")
    rc.run()
    assert consumer.subscribed == ["cell3/robot-action"]


def test_skips_frame_with_no_action_for_arm():
    applier = FakeApplier()
    other = RobotFrame(
        source_id="robot-action",
        timestamp_us=0,
        sequence=0,
        state={"robot1.action.target_cartesian_delta[0]": 1.0},
    )
    rc = _consumer(frames=[other], applier=applier, arm_prefix="robot0")
    rc.run()
    assert applier.steps == []


def test_ignores_non_robot_frames():
    applier = FakeApplier()
    rc = _consumer(frames=[object()], applier=applier)
    rc.run()
    assert applier.steps == []


def test_malformed_frame_is_skipped_not_fatal():
    applier = FakeApplier()
    bad = RobotFrame(
        source_id="robot-action",
        timestamp_us=0,
        sequence=0,
        state={f"{_BASE}[0]": 1.0, f"{_BASE}[2]": 3.0},  # non-contiguous
    )
    rc = _consumer(frames=[bad, _cart()], applier=applier)

    rc.run()  # must not raise

    assert [s["action"] for s in applier.steps] == [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]]


def test_step_error_is_skipped_and_loop_continues():
    applier = FakeApplier(fail_times=1)  # first step raises, rest succeed
    rc = _consumer(frames=[_cart(1.0), _cart(0.0, 1)], applier=applier)

    rc.run()  # must not raise

    assert applier.steps[0] == {"failed": True}
    assert applier.steps[1]["action"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]


def test_subscription_fault_reraises_when_not_closed():
    rc = _consumer(fault=RuntimeError("bus dropped"))
    with pytest.raises(RuntimeError, match="bus dropped"):
        rc.run()


def test_fault_is_swallowed_after_close():
    consumer = FakeConsumer(fault=RuntimeError("cancelled"))
    rc = RobotActionConsumer(consumer, FakeApplier())

    rc.close()
    rc.run()  # the fault from the cancelled subscription is expected -> swallowed

    assert consumer.closed is True


def test_close_cancels_consumer():
    consumer = FakeConsumer(frames=[])
    rc = RobotActionConsumer(consumer, FakeApplier())
    rc.close()
    assert consumer.closed is True
