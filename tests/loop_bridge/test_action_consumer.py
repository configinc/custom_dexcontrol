"""Tests for ArmActionBackend + RobotActionConsumer (subscribe robot-action, apply)."""

from __future__ import annotations

import pytest
from conftest import FakeApplier, FakeConsumer
from loop_sdk import RobotFrame

from loop_bridge.action_consumer import ArmActionBackend, RobotActionConsumer
from loop_bridge.robot_action import DEFAULT_ACTION_SPACE

_BASE = f"robot0.action.{DEFAULT_ACTION_SPACE}"


def _frame(values, prefix="robot0", sequence=0, state=None):
    base = f"{prefix}.action.{DEFAULT_ACTION_SPACE}"
    state = dict(state or {})
    state.update({f"{base}[{i}]": v for i, v in enumerate(values)})
    return RobotFrame(
        source_id="robot-action", timestamp_us=sequence, sequence=sequence, state=state
    )


def _cart(gripper=1.0, prefix="robot0", sequence=0):
    return _frame(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, gripper], prefix=prefix, sequence=sequence
    )


def _backend(applier, *, arm_prefix="robot0", gripper_action_space=""):
    return ArmActionBackend(
        applier,
        arm_prefix=arm_prefix,
        action_space=DEFAULT_ACTION_SPACE,
        gripper_action_space=gripper_action_space,
    )


def _consumer(frames=None, fault=None, backends=None):
    return RobotActionConsumer(
        FakeConsumer(frames=frames, fault=fault), backends or [_backend(FakeApplier())]
    )


def test_applies_each_action_via_step():
    applier = FakeApplier()
    rc = _consumer(
        frames=[_cart(1.0), _cart(0.0, sequence=1)], backends=[_backend(applier)]
    )

    rc.run()

    assert [s["action"] for s in applier.steps] == [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],
    ]
    assert applier.steps[0]["action_space"] == DEFAULT_ACTION_SPACE


def test_dispatches_each_arm_to_its_backend():
    a0, a1 = FakeApplier(), FakeApplier()
    frame = _frame([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], prefix="robot0")
    frame = _frame(
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], prefix="robot1", state=frame.state
    )
    rc = _consumer(
        frames=[frame],
        backends=[_backend(a0, arm_prefix="robot0"), _backend(a1, arm_prefix="robot1")],
    )

    rc.run()

    assert a0.steps[0]["action"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert a1.steps[0]["action"] == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]


def test_passes_gripper_action_space_through():
    applier = FakeApplier()
    rc = _consumer(
        frames=[_cart()], backends=[_backend(applier, gripper_action_space="velocity")]
    )
    rc.run()
    assert applier.steps[0]["gripper_action_space"] == "velocity"


def test_subscribes_to_configured_source_id():
    consumer = FakeConsumer(frames=[])
    rc = RobotActionConsumer(
        consumer, [_backend(FakeApplier())], source_id="cell3/robot-action"
    )
    rc.run()
    assert consumer.subscribed == ["cell3/robot-action"]


def test_skips_arm_with_no_action_in_frame():
    applier = FakeApplier()
    other = RobotFrame(
        source_id="robot-action",
        timestamp_us=0,
        sequence=0,
        state={"robot1.action.target_cartesian_delta[0]": 1.0},
    )
    rc = _consumer(frames=[other], backends=[_backend(applier, arm_prefix="robot0")])
    rc.run()
    assert applier.steps == []


def test_ignores_non_robot_frames():
    applier = FakeApplier()
    rc = _consumer(frames=[object()], backends=[_backend(applier)])
    rc.run()
    assert applier.steps == []


def test_malformed_frame_is_skipped_not_fatal():
    applier = FakeApplier()
    bad = RobotFrame(
        source_id="robot-action",
        timestamp_us=0,
        sequence=0,
        state={f"{_BASE}[0]": 1.0, f"{_BASE}[2]": 3.0},
    )
    rc = _consumer(frames=[bad, _cart()], backends=[_backend(applier)])

    rc.run()  # must not raise

    assert [s["action"] for s in applier.steps] == [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]]


def test_step_error_is_skipped_and_loop_continues():
    applier = FakeApplier(fail_times=1)
    rc = _consumer(
        frames=[_cart(1.0), _cart(0.0, sequence=1)], backends=[_backend(applier)]
    )

    rc.run()  # must not raise

    assert applier.steps[0] == {"failed": True}
    assert applier.steps[1]["action"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]


def test_subscription_fault_reraises_when_not_closed():
    rc = _consumer(fault=RuntimeError("bus dropped"))
    with pytest.raises(RuntimeError, match="bus dropped"):
        rc.run()


def test_fault_is_swallowed_after_close():
    consumer = FakeConsumer(fault=RuntimeError("cancelled"))
    rc = RobotActionConsumer(consumer, [_backend(FakeApplier())])
    rc.close()
    rc.run()
    assert consumer.closed is True


def test_close_cancels_consumer():
    consumer = FakeConsumer(frames=[])
    rc = RobotActionConsumer(consumer, [_backend(FakeApplier())])
    rc.close()
    assert consumer.closed is True
