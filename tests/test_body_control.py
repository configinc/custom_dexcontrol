"""Behavioral contracts, without opening a robot connection."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "dexcontrol"))
from core.robotenv_vega.body_control import BodyControl, ControlConflict
from core.chassis_kinematics import bounded_wheel_control


class Hardware:
    def __init__(self):
        self.stopped = True
        self.sent = []
        self.gripper = .7
    def hold(self): self.sent.append("hold")
    def maintain_hold(self): self.sent.append("hold_tick")
    def snapshot(self): return {"gripper": self.gripper}
    def limits(self): return {"command_velocity": [.15, .15, .3, .05, .05, .05]}
    def stationary(self): return self.stopped
    def move_body(self, command, dt): self.sent.append(command.copy())
    def set_shoulder_target(self, pose): self.target = pose
    def start_arms(self): self.sent.append("arms")
    def prepare_arms(self): self.sent.append("prepare_arms")


@pytest.fixture
def setup():
    hardware = Hardware()
    return hardware, BodyControl(hardware)


def test_mode_switch_blocks_arms_and_requires_reanchor_ack(setup):
    hw, control = setup
    control.select("body")
    with pytest.raises(ControlConflict): control.accept_arm("left")
    with pytest.raises(ControlConflict): control.enable_arms()
    control.select("prepare_arms")
    assert hw.sent[-1] == "prepare_arms"
    control.enable_arms()
    control.accept_arm("left")
    assert control.mode == "arms"
    assert hw.gripper == .7


def test_model_refresh_failure_keeps_commands_blocked(setup):
    hw, control = setup
    def fail():
        raise RuntimeError("Cannot build arm model")
    hw.prepare_arms = fail
    with pytest.raises(RuntimeError, match="Cannot build arm model"):
        control.select("prepare_arms")
    assert control.mode == "hold"
    assert control.state()["error"] == "Cannot build arm model"
    with pytest.raises(ControlConflict):
        control.enable_arms()
    with pytest.raises(ControlConflict):
        control.accept_arm("left")


def test_timeout_stops_and_requires_mode_selection(setup):
    hw, control = setup
    control.select("body")
    control.set_command([.1, 0, 0])
    control.tick()
    assert hw.sent[-1][0] == .1
    control.clock = lambda: control.deadline + 1 if np.isfinite(control.deadline) else 1e10
    control.tick()
    assert control.mode == "hold"
    with pytest.raises(ControlConflict): control.set_command([0]*3)
    assert hw.sent[-1] == "hold_tick"


def test_invalid_commands_do_not_change_motion(setup):
    hw, control = setup
    control.select("body")
    control.set_command([.1, 0, 0])
    for values in ([float("nan")]*3, [.3, 0, 0], [0]*2):
        with pytest.raises(ValueError): control.set_command(values)
    assert control.command[0] == .1


def test_explicit_stop_blocks_pending_arm_enable(setup):
    hw, control = setup
    control.select("prepare_arms")
    control.stop()
    with pytest.raises(ControlConflict): control.enable_arms()
    assert control.mode == "hold"


def test_arm_mode_keeps_running_without_body_watchdog(setup):
    hw, control = setup
    control.select("prepare_arms")
    control.enable_arms()
    control.clock = lambda: 1e12
    control.tick()
    control.accept_arm("left")
    assert control.mode == "arms"


def test_bounded_steering_preserves_direction_at_every_heading():
    for heading in np.linspace(-np.pi, np.pi, 721):
        vector = .12*np.array([np.cos(heading), np.sin(heading)])
        for current in (-2.35, 0., 2.35):
            angle, speed = bounded_wheel_control(vector, current, (-2.35, 2.35))
            assert -2.35 <= angle <= 2.35
            np.testing.assert_allclose([speed*np.cos(angle), -speed*np.sin(angle)], vector, atol=1e-12)


def test_unreachable_direction_is_rejected_instead_of_clipped():
    with pytest.raises(ValueError): bounded_wheel_control([0, 1], 0, (-.3, .3))
    with pytest.raises(ValueError): bounded_wheel_control([float("nan"), 0], 0, (-2.35, 2.35))


@pytest.mark.parametrize("intervals", [[.02]*10, [.105, .035, .06]])
def test_absolute_path_progress_depends_on_elapsed_time(intervals):
    from test_body_hardware import make_hardware
    hw = make_hardware()
    hw.stationary = lambda: True
    control = BodyControl(hw)
    control.select("body")
    now = [0.]
    control.clock = lambda: now[0]
    control._last_body_time = 0.
    goal = hw.shoulder_target + [0, -.03, 0]
    control.set_command([0]*3, goal)
    expected, _ = hw.torso_path.sample(sum(intervals))
    for interval in intervals:
        now[0] += interval
        control.set_command([0]*3, goal)
        control.tick()
    np.testing.assert_allclose(hw.torso_command, expected, atol=1e-12)
    np.testing.assert_array_equal(hw.shoulder_target, goal)
