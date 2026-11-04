"""Continuous torque relief and its managed output/lifecycle contracts."""

from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from test_control_runtime import Device, Hardware

from dexcontrol.core.control.config import load_config
from dexcontrol.core.control.output import OutputGate
from dexcontrol.core.control.torque_limit import ContinuousTorqueOffset
from dexcontrol.core.robotenv_vega.control_node import VegaControlNode


class Arm(Device):
    def __init__(self, side):
        super().__init__(side)
        self.joint_name = [f"{side}_j{i}" for i in range(1, 8)]
        self.joint_pos_limit = np.tile([-2.0, 2.0], (7, 1))
        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.current = np.zeros(7)
        self.stamp = 1

    def get_state(self):
        return {"cur": self.current.copy(), "timestamp_ns": self.stamp}

    def command(self, q, vel=0.0):
        self.sink({"pos": np.full(7, q), "vel": np.full(7, vel)})


class TorqueHardware(Hardware):
    def __init__(self):
        super().__init__()
        self.gate = OutputGate()
        self.left, self.right = Arm("left"), Arm("right")
        self.gate.attach("left_arm", self.left)
        self.gate.attach("right_arm", self.right)
        rules = dict(self.config.safety)
        rules["arm_torque"] = replace(rules["arm_torque"], enabled=True)
        self.config = replace(self.config, safety=rules)
        self.now = 0.0

    def arm_tick(self):
        self.left.command(self.targets[0], 0.42)
        self.right.command(self.targets[1], -0.31)

    def clear_motion_targets(self, rejected=False):
        self.targets = [self.left.q[0], self.right.q[0]]


@pytest.fixture
def runtime():
    hardware = TorqueHardware()
    node = VegaControlNode(hardware, hardware.config, clock=lambda: hardware.now)
    node.start()
    # The timed handover itself is covered by the lifecycle tests.
    node.mode = "arm_ready"
    node.enable_arms()
    yield node, hardware
    node.stop()
    node.actions.join()
    node.close()


def settings():
    return replace(
        load_config().arm_torque.arms["left_arm"], torque_limit_nm=[20.0] * 7
    )


def test_relief_direction_recovery_and_user_motion_are_independent():
    limiter = ContinuousTorqueOffset(settings())
    requested = np.zeros(7)
    torque = np.array([30, -30, 20, -20, 0, 19, -19])
    corrected = limiter.update(requested, torque, 0.005)
    np.testing.assert_allclose(corrected, [-0.00075, 0.00075, 0, 0, 0, 0, 0])
    # At the limit the correction stays; a new user command passes through.
    corrected = limiter.update(requested + 0.4, np.sign(torque) * 20, 0.005)
    np.testing.assert_allclose(corrected[:2], [0.39925, 0.40075])
    # Recovery ends at zero instead of overshooting to the opposite side.
    corrected = limiter.update(requested + 0.4, np.zeros(7), 0.01)
    np.testing.assert_array_equal(corrected, np.full(7, 0.4))


def test_existing_absolute_joint_bounds_stop_offset_windup():
    limiter = ContinuousTorqueOffset(settings())
    bounds = np.tile([-1.0, 1.0], (7, 1))
    for _ in range(100):
        sent = limiter.update(np.full(7, -0.99), np.full(7, 100), 0.005, bounds)
    np.testing.assert_array_equal(sent, np.full(7, -1.0))
    # Only 0.01 rad of real correction remains, not all attempted relief.
    for _ in range(14):
        sent = limiter.update(np.full(7, -0.99), np.zeros(7), 0.005, bounds)
    np.testing.assert_array_equal(sent, np.full(7, -0.99))


def test_normal_targets_and_velocity_feedforward_are_unchanged(runtime):
    node, hw = runtime
    hw.left.current[:] = 0.1
    hw.right.current[:] = -0.1
    for target in [0.0, 1.0, -0.7, 0.2]:
        hw.targets = [target, -target]
        node.tick()
        np.testing.assert_array_equal(hw.left.sent[-1]["pos"], np.full(7, target))
        np.testing.assert_array_equal(hw.right.sent[-1]["pos"], np.full(7, -target))
        np.testing.assert_array_equal(hw.left.sent[-1]["vel"], np.full(7, 0.42))
        np.testing.assert_array_equal(hw.right.sent[-1]["vel"], np.full(7, -0.31))


def test_currents_use_per_joint_constants_and_all_joints_are_independent(runtime):
    node, hw = runtime
    constants = np.array(hw.config.arm_torque.arms["left_arm"].torque_constant_nm_per_a)
    limits = np.array(hw.config.arm_torque.arms["left_arm"].torque_limit_nm)
    hw.left.current = (limits + 10) / constants
    hw.right.current = -hw.left.current
    node.tick()
    np.testing.assert_allclose(hw.left.sent[-1]["pos"], np.full(7, -0.00075))
    np.testing.assert_allclose(hw.right.sent[-1]["pos"], np.full(7, 0.00075))
    state = node.state()["arm_torque"]
    assert state["state"] == "active"
    assert all(state["arms"][0]["over_limit"])
    np.testing.assert_allclose(state["arms"][0]["estimated_torque_nm"], limits + 10)
    assert node.state()["control_state"] == "running"


def test_sparse_targets_continue_relief_without_reapplying_previous_correction(runtime):
    node, hw = runtime
    hw.targets = [0.5, -0.2]
    hw.left.current[:] = 100
    node.tick()
    hw.arm_tick = lambda: None
    for _ in range(9):
        node.tick()
    np.testing.assert_allclose(hw.left.sent[-1]["pos"], np.full(7, 0.4925))
    np.testing.assert_array_equal(hw.left.sent[-1]["vel"], np.full(7, 0.42))
    np.testing.assert_array_equal(hw.right.sent[-1]["pos"], np.full(7, -0.2))


def test_action_waypoints_use_same_limiter_and_cancel_to_measured_hold(runtime):
    node, hw = runtime
    staged = Event()
    hw.left.current[:] = 100

    def action(cancelled):
        hw.left.command(0.5, 0.1)
        staged.set()
        cancelled.wait(1.0)

    node.start_action("test_waypoint", action)
    assert staged.wait(1.0)
    for _ in range(10):
        node.tick()
    np.testing.assert_allclose(hw.left.sent[-1]["pos"], np.full(7, 0.4925))
    np.testing.assert_array_equal(hw.left.sent[-1]["vel"], np.full(7, 0.1))
    node.halt()
    node.actions.join()
    node.tick()
    np.testing.assert_array_equal(hw.left.sent[-1]["pos"], hw.left.q)


def test_configured_current_sign_reverses_relief_direction():
    hw = TorqueHardware()
    torque = hw.config.arm_torque
    arms = dict(torque.arms)
    arms["left_arm"] = replace(arms["left_arm"], current_sign=[-1] * 7)
    node = VegaControlNode(
        hw, replace(hw.config, arm_torque=replace(torque, arms=arms))
    )
    try:
        node.start()
        node.mode = "arm_ready"
        node.enable_arms()
        hw.left.current[:] = 100
        node.tick()
        np.testing.assert_allclose(hw.left.sent[-1]["pos"], np.full(7, 0.00075))
    finally:
        node.stop()
        node.close()


@pytest.mark.parametrize("operation", ["stop", "halt", "fault"])
def test_lifecycle_hold_drops_old_offsets_and_targets(runtime, operation):
    node, hw = runtime
    hw.targets = [1.0, 0.5]
    hw.left.current[:] = 100
    node.tick()
    if operation == "fault":
        node.fault("execution_error", "test fault")
    else:
        getattr(node, operation)()
    for _ in range(4):
        node.tick()
        np.testing.assert_array_equal(hw.left.sent[-1]["pos"], hw.left.q)
        np.testing.assert_array_equal(hw.left.sent[-1]["vel"], np.zeros(7))
    assert node.state()["arm_torque"] == {"state": "idle", "arms": []}
    if operation == "fault":
        node.reset_fault()
    if operation != "halt":
        node.start()
    node.mode = "arm_ready"
    node.enable_arms()
    hw.targets = [0.0, 0.0]
    hw.left.current[:] = 0
    node.tick()
    np.testing.assert_array_equal(hw.left.sent[-1]["pos"], np.zeros(7))


@pytest.mark.parametrize("bad_current", [None, [0.0] * 6, [float("nan")] * 7])
def test_invalid_current_faults_whole_batch_and_can_still_hold(runtime, bad_current):
    node, hw = runtime
    hw.targets = [0.5, 0.5]
    hw.right.get_state = lambda: {"cur": bad_current, "timestamp_ns": 1}
    before = len(hw.left.sent)
    with pytest.raises(ValueError, match="current feedback"):
        node.tick()
    assert node.state()["fault_reason"] == "execution_error"
    for sent in hw.left.sent[before:]:
        np.testing.assert_array_equal(sent["pos"], hw.left.q)
    node.tick()  # FAULT hold does not depend on a usable current packet.


def test_frozen_feedback_expires_but_fresh_timestamp_refreshes_deadline(runtime):
    node, hw = runtime
    node.tick()
    hw.now = 0.19
    hw.left.stamp += 1
    hw.right.stamp += 1
    node.tick()
    hw.now = 0.38
    node.tick()
    hw.now = 0.40
    with pytest.raises(TimeoutError, match="expired"):
        node.tick()
    assert node.state()["control_state"] == "fault"


def test_disabled_limiting_needs_no_current_and_preserves_output():
    hw = TorqueHardware()
    rules = dict(hw.config.safety)
    rules["arm_torque"] = replace(rules["arm_torque"], enabled=False)
    hw.left.get_state = lambda: pytest.fail("Disabled limiter read current")
    node = VegaControlNode(hw, replace(hw.config, safety=rules))
    try:
        node.start()
        node.mode = "arm_ready"
        node.enable_arms()
        hw.targets = [0.8, -0.7]
        node.tick()
        np.testing.assert_array_equal(hw.left.sent[-1]["pos"], np.full(7, 0.8))
        assert node.state()["arm_torque"]["state"] == "disabled"
    finally:
        node.stop()
        node.close()


def test_model_approval_sees_corrected_target_and_can_override_with_hold(runtime):
    node, hw = runtime
    hw.targets = [0.5, 0.3]
    hw.left.current[:] = 100
    seen = []

    def reject(commands, effects, owner, arm_mode=False):
        seen.append(commands["left_arm"]["pos"].copy())
        for name, device in hw.gate.devices.items():
            commands[name] = {"pos": device.q.copy(), "vel": np.zeros(7)}
        return ""

    # An endpoint is pending approval, so no correction may bypass its hold.
    node.motion_safety.guard = SimpleNamespace(
        guarded=True,
        active=None,
        reset=lambda: None,
        close=lambda: None,
        snapshot=lambda check: {"state": "checking"},
    )
    node.motion_safety.filter = reject
    for _ in range(3):
        node.tick()
        np.testing.assert_array_equal(hw.left.sent[-1]["pos"], hw.left.q)
    for candidate in seen:
        np.testing.assert_allclose(candidate, np.full(7, 0.49925))
    state = node.state()["arm_torque"]
    assert state["state"] == "model_hold"
    np.testing.assert_array_equal(state["arms"][0]["output_position_rad"], hw.left.q)


@pytest.mark.parametrize(
    "field,value",
    [
        ("torque_limit_nm", [1] * 6),
        ("torque_constant_nm_per_a", [0] * 7),
        ("gain_rad_per_nm_s", [float("nan")] * 7),
        ("current_sign", [0] * 7),
    ],
)
def test_joint_configuration_rejects_invalid_settings(field, value):
    with pytest.raises(ValueError):
        replace(settings(), **{field: value})
