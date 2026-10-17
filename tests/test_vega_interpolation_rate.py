import threading
import sys
from types import ModuleType

import numpy as np
import pytest

# Keep the rate tests independent of the optional DualSense/hidapi stack.
_base_arm_teleop = ModuleType("base_arm_teleop")
_base_arm_teleop.BaseIKController = object
sys.modules.setdefault("base_arm_teleop", _base_arm_teleop)

from dexcontrol.core.vega.robot import VegaRobot, _resample_smoothing_alpha


def test_resampled_smoothing_alpha_preserves_wall_clock_omega():
    source_alpha = 0.1
    source_hz = 20.0
    target_hz = 100.0

    target_alpha = _resample_smoothing_alpha(
        source_alpha, source_hz, target_hz
    )

    source_omega = source_alpha * source_hz / (1.0 - source_alpha)
    target_omega = target_alpha * target_hz / (1.0 - target_alpha)
    assert target_alpha < source_alpha
    assert target_omega == pytest.approx(source_omega)


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 0.99])
def test_resampled_smoothing_alpha_is_unchanged_at_same_rate(alpha):
    assert _resample_smoothing_alpha(alpha, 20.0, 20.0) == pytest.approx(alpha)


def test_interpolated_tick_passes_execution_rate_to_arm_and_hand():
    robot = VegaRobot.__new__(VegaRobot)
    robot._interp_lock = threading.Lock()
    robot._latest_target_joint_pos = np.ones(7)
    robot._latest_hand_action = np.array([1.0])
    robot._latest_gripper_action = 1.0
    robot._latest_gripper_action_space = "velocity"
    robot._interpolator = type(
        "Interpolator",
        (),
        {"interpolate": lambda self, now, compute_velocity: (np.ones(7), np.ones(7))},
    )()
    robot.use_velocity_feedforward = True
    robot._output_filter = None
    robot._interpolation_ema_alpha = 0.0
    robot.interpolation_control_hz = 100
    robot._prev_command_successful = False
    calls = []
    robot.update_joints = lambda *args, **kwargs: calls.append(("arm", kwargs))
    robot._apply_hand_action = lambda *args, **kwargs: calls.append(("hand", kwargs))

    assert robot.execute_interpolated_tick()
    assert calls[0][1]["execution_hz"] == 100
    assert calls[1][1]["execution_hz"] == 100


def test_joint_velocity_feedforward_uses_interpolation_tick_rate():
    sent = []
    arm = type(
        "Arm",
        (),
        {
            "joint_pos_limit": None,
            "get_joint_pos": lambda self: np.zeros(7),
            "set_joint_pos_vel": lambda self, pos, vel, relative: sent.append(
                (np.asarray(pos), np.asarray(vel), relative)
            ),
        },
    )()
    robot = VegaRobot.__new__(VegaRobot)
    robot.arm_side = "left"
    robot.arm = arm
    robot.control_hz = 20
    robot._last_cmd_joint_pos = None
    robot._max_delta_scale = 1.0
    robot._MOTOR_MAX_JERK_RAD = 0.0
    robot._prev_cmd_delta = None
    robot._HW_CORRECTION_OUTLIER_THRESH = 0.5
    robot._HW_CORRECTION_ALPHA = 0.7
    robot.use_velocity_feedforward = True
    robot._prev_joint_vel = None
    robot._vel_smoothing_alpha = 1.0
    robot._vel_damp_thresh = 1e-6
    robot._vel_ratio = 1.0
    robot._vel_log_file = None
    robot.validate_joint_limits = lambda target: None
    robot._agent_debug_log = lambda **kwargs: None
    robot.sync_motion_manager_with_arm = lambda target: None

    target = np.full(7, 0.001)
    robot.update_joints(target, execution_hz=100)

    assert len(sent) == 1
    assert sent[0][0] == pytest.approx(target)
    assert sent[0][1] == pytest.approx(np.full(7, 0.1))


def test_scalar_gripper_velocity_uses_interpolation_tick_rate():
    queued = []
    robot = VegaRobot.__new__(VegaRobot)
    robot.hand = object()
    robot.control_hz = 20
    robot.get_cached_gripper_position = lambda: 0.2
    robot._enqueue_gripper_target = queued.append
    robot._prev_gripper_command_successful = False

    robot.update_gripper(1.0, velocity=True, execution_hz=100)

    assert queued == pytest.approx([0.21])


def test_multi_hand_velocity_uses_interpolation_tick_rate():
    sent = []
    hand = type(
        "Hand",
        (),
        {
            "model": "wuji_hand_v2",
            "get_joint_pos": lambda self: np.zeros(3),
            "set_joint_pos": lambda self, target, wait_time: sent.append(
                (np.asarray(target), wait_time)
            ),
        },
    )()
    robot = VegaRobot.__new__(VegaRobot)
    robot.hand = hand
    robot.control_hz = 20
    robot._gripper_io_lock = threading.Lock()
    robot._prev_gripper_command_successful = False

    robot._apply_hand_action(
        np.ones(3), "velocity", blocking=False, execution_hz=100
    )

    assert sent[0][0] == pytest.approx(np.full(3, 0.01))
    assert sent[0][1] == 0.0


def test_interpolated_scalar_gripper_velocity_resolves_once_per_input_step():
    class Interpolator:
        def add_point(self, timestamp, target):
            self.target = np.asarray(target)

    robot = VegaRobot.__new__(VegaRobot)
    robot._interpolator = Interpolator()
    robot._interp_lock = threading.Lock()
    robot.control_hz = 20
    robot._latest_target_joint_pos = None
    robot._latest_hand_action = None
    robot._latest_gripper_action = 0.0
    robot._latest_gripper_action_space = "position"
    robot.get_robot_state = lambda: ({"joint_positions": np.zeros(7)}, {})
    robot._hand_dof = lambda: 1
    robot.get_cached_gripper_position = lambda: 0.2

    command = np.concatenate([np.zeros(7), [1.0]])
    robot.add_command_point(
        command, "joint_position", gripper_action_space="velocity"
    )
    assert robot._latest_hand_action == pytest.approx([0.25])
    assert robot._latest_gripper_action_space == "position"

    # A second 20 Hz input step integrates from the prior commanded target,
    # not from the still-lagging 20 Hz measured/cache value.
    robot.add_command_point(
        command, "joint_position", gripper_action_space="velocity"
    )
    assert robot._latest_hand_action == pytest.approx([0.30])
    assert robot._latest_gripper_action_space == "position"


def test_interpolated_multi_hand_velocity_resolves_once_per_input_step():
    class Interpolator:
        def add_point(self, timestamp, target):
            self.target = np.asarray(target)

    hand = type(
        "Hand", (), {"get_joint_pos": lambda self: np.full(3, 0.1)}
    )()
    robot = VegaRobot.__new__(VegaRobot)
    robot.hand = hand
    robot._interpolator = Interpolator()
    robot._interp_lock = threading.Lock()
    robot.control_hz = 20
    robot._latest_target_joint_pos = None
    robot._latest_hand_action = None
    robot._latest_gripper_action = 0.0
    robot._latest_gripper_action_space = "position"
    robot.get_robot_state = lambda: ({"joint_positions": np.zeros(7)}, {})
    robot._hand_dof = lambda: 3

    command = np.concatenate([np.zeros(7), np.ones(3)])
    robot.add_command_point(
        command, "joint_position", gripper_action_space="velocity"
    )
    assert robot._latest_hand_action == pytest.approx(np.full(3, 0.15))
    assert robot._latest_gripper_action_space == "position"
