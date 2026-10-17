import threading
import sys
from types import ModuleType

import numpy as np
import pytest

# Keep the rate tests independent of the optional DualSense/hidapi stack.
_base_arm_teleop = ModuleType("base_arm_teleop")
_base_arm_teleop.BaseIKController = object
sys.modules.setdefault("base_arm_teleop", _base_arm_teleop)

import dexcontrol.core.vega.robot as robot_module
from dexcontrol.core.vega.robot import (
    VegaRobot,
    _resample_first_order_alpha,
    _resample_smoothing_alpha,
)
from dexcontrol.utils.trajectory_interpolator import TrajectoryInterpolator


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


def test_first_order_alpha_preserves_decay_over_one_input_period():
    source_alpha = 0.3
    target_alpha = _resample_first_order_alpha(source_alpha, 20.0, 200.0)
    assert (1.0 - target_alpha) ** 10 == pytest.approx(1.0 - source_alpha)


def test_interpolated_tick_passes_execution_rate_to_arm_and_hand():
    robot = VegaRobot.__new__(VegaRobot)
    robot._interp_lock = threading.Lock()
    robot._latest_target_joint_pos = np.ones(7)
    robot.control_hz = 20
    robot._latest_arm_feedforward_velocity = np.ones(7)
    robot._latest_command_monotonic = 1.0
    robot._latest_hand_action = np.array([1.0])
    robot._latest_gripper_action = 1.0
    robot._latest_gripper_action_space = "velocity"
    robot._interpolator = type(
        "Interpolator",
        (),
        {
            "interpolate": lambda self, now, compute_velocity: (
                np.ones(8),
                np.ones(8),
            )
        },
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
    robot.update_joints(
        target,
        execution_hz=100,
        feedforward_velocity=np.full(7, 0.2),
    )

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
    robot._latest_arm_feedforward_velocity = None
    robot._latest_command_monotonic = None
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
    robot._latest_arm_feedforward_velocity = None
    robot._latest_command_monotonic = None
    robot._latest_hand_action = None
    robot._latest_gripper_action = 0.0
    robot._latest_gripper_action_space = "position"
    robot._gripper_io_lock = threading.Lock()
    robot.get_robot_state = lambda: ({"joint_positions": np.zeros(7)}, {})
    robot._hand_dof = lambda: 3

    command = np.concatenate([np.zeros(7), np.ones(3)])
    robot.add_command_point(
        command, "joint_position", gripper_action_space="velocity"
    )
    assert robot._latest_hand_action == pytest.approx(np.full(3, 0.15))
    assert robot._latest_gripper_action_space == "position"


def test_interpolated_joint_velocity_keeps_requested_feedforward():
    class Interpolator:
        def add_point(self, timestamp, target):
            self.target = np.asarray(target)

    robot = VegaRobot.__new__(VegaRobot)
    robot._interpolator = Interpolator()
    robot._interp_lock = threading.Lock()
    robot.control_hz = 20
    robot._latest_target_joint_pos = None
    robot._latest_arm_feedforward_velocity = None
    robot._latest_command_monotonic = None
    robot._latest_hand_action = None
    robot._latest_gripper_action = 0.0
    robot._latest_gripper_action_space = "position"
    robot.get_robot_state = lambda: ({"joint_positions": np.zeros(7)}, {})
    robot._hand_dof = lambda: 1
    robot.get_cached_gripper_position = lambda: 0.0

    command = np.concatenate([np.ones(7), [0.0]])
    robot.add_command_point(command, "joint_velocity")

    assert robot._latest_target_joint_pos == pytest.approx(np.full(7, 0.05))
    assert robot._latest_arm_feedforward_velocity == pytest.approx(np.ones(7))


def test_hand_velocity_accumulator_clips_and_reverses_without_windup():
    class Interpolator:
        def add_point(self, timestamp, target):
            pass

    robot = VegaRobot.__new__(VegaRobot)
    robot._interpolator = Interpolator()
    robot._interp_lock = threading.Lock()
    robot.control_hz = 20
    robot._latest_target_joint_pos = None
    robot._latest_arm_feedforward_velocity = None
    robot._latest_command_monotonic = None
    robot._latest_hand_action = None
    robot._latest_gripper_action = 0.0
    robot._latest_gripper_action_space = "position"
    robot.get_robot_state = lambda: ({"joint_positions": np.zeros(7)}, {})
    robot._hand_dof = lambda: 1
    robot.get_cached_gripper_position = lambda: 0.99

    positive = np.concatenate([np.zeros(7), [1.0]])
    for _ in range(20):
        robot.add_command_point(
            positive, "joint_position", gripper_action_space="velocity"
        )
    assert robot._latest_hand_action == pytest.approx([1.0])

    negative = np.concatenate([np.zeros(7), [-1.0]])
    robot.add_command_point(
        negative, "joint_position", gripper_action_space="velocity"
    )
    assert robot._latest_hand_action == pytest.approx([0.95])


def test_reset_filter_state_discards_persistent_hand_and_arm_targets():
    robot = VegaRobot.__new__(VegaRobot)
    robot._filter_pos = np.ones(7)
    robot._filter_vel = np.ones(7)
    robot._last_cmd_joint_pos = np.ones(7)
    robot._prev_cmd_delta = np.ones(7)
    robot._prev_joint_vel = np.ones(7)
    robot._latest_target_joint_pos = np.ones(7)
    robot._latest_arm_feedforward_velocity = np.ones(7)
    robot._latest_command_monotonic = 1.0
    robot._latest_hand_action = np.ones(3)
    robot._latest_gripper_action = 1.0
    robot._latest_gripper_action_space = "velocity"
    robot._interpolator = None
    robot._output_filter = None

    robot.reset_filter_state()

    assert robot._latest_target_joint_pos is None
    assert robot._prev_joint_vel is None
    assert robot._latest_arm_feedforward_velocity is None
    assert robot._latest_command_monotonic is None
    assert robot._latest_hand_action is None
    assert robot._latest_gripper_action == 0.0
    assert robot._latest_gripper_action_space == "position"


def test_reset_command_state_drains_gripper_worker_before_clearing_targets():
    robot = VegaRobot.__new__(VegaRobot)
    calls = []
    robot._gripper_worker = type(
        "Worker", (), {"is_alive": lambda self: True}
    )()
    robot._stop_gripper_worker = lambda: calls.append("stop")
    robot.reset_filter_state = lambda: calls.append("clear")
    robot._start_gripper_worker = lambda: calls.append("start")

    robot.reset_command_state()

    assert calls == ["stop", "clear", "start"]


def test_interpolated_hand_velocity_rejects_nonfinite_before_storage():
    robot = VegaRobot.__new__(VegaRobot)
    robot._interpolator = type(
        "Interpolator", (), {"add_point": lambda self, timestamp, target: None}
    )()
    robot._interp_lock = threading.Lock()
    robot.control_hz = 20
    robot._latest_target_joint_pos = None
    robot._latest_arm_feedforward_velocity = None
    robot._latest_command_monotonic = None
    robot._latest_hand_action = None
    robot._latest_gripper_action = 0.0
    robot._latest_gripper_action_space = "position"
    robot.get_robot_state = lambda: ({"joint_positions": np.zeros(7)}, {})
    robot._hand_dof = lambda: 1

    command = np.concatenate([np.zeros(7), [np.nan]])
    with pytest.raises(ValueError, match="non-finite"):
        robot.add_command_point(
            command, "joint_position", gripper_action_space="velocity"
        )
    assert robot._latest_target_joint_pos is None
    assert robot._latest_arm_feedforward_velocity is None


def test_nonfinite_measured_hand_anchor_rejects_entire_command():
    points = []
    hand = type(
        "Hand",
        (),
        {
            "model": "wuji_hand_v2",
            "get_joint_pos": lambda self: np.full(3, np.nan),
        },
    )()
    robot = VegaRobot.__new__(VegaRobot)
    robot.hand = hand
    robot._interpolator = type(
        "Interpolator",
        (),
        {"add_point": lambda self, timestamp, target: points.append(target)},
    )()
    robot._interp_lock = threading.Lock()
    robot._gripper_io_lock = threading.Lock()
    robot.control_hz = 20
    robot._latest_target_joint_pos = None
    robot._latest_arm_feedforward_velocity = None
    robot._latest_command_monotonic = None
    robot._latest_hand_action = None
    robot._latest_gripper_action = 0.0
    robot._latest_gripper_action_space = "position"
    robot.get_robot_state = lambda: ({"joint_positions": np.zeros(7)}, {})
    robot._hand_dof = lambda: 3

    command = np.concatenate([np.ones(7), np.ones(3)])
    with pytest.raises(ValueError, match="Resolved hand target"):
        robot.add_command_point(
            command, "joint_position", gripper_action_space="velocity"
        )

    assert points == []
    assert robot._latest_target_joint_pos is None


def test_delayed_interpolation_produces_motion_and_zeroes_feedforward_at_hold(
    monkeypatch,
):
    times = iter([10.0, 10.05, 10.075, 10.16])
    monkeypatch.setattr(robot_module._time, "perf_counter", lambda: next(times))
    robot = VegaRobot.__new__(VegaRobot)
    robot._interpolator = TrajectoryInterpolator(method="linear", history_size=3)
    robot._interp_lock = threading.Lock()
    robot.control_hz = 20
    robot.interpolation_control_hz = 200
    robot.use_velocity_feedforward = True
    robot._output_filter = None
    robot._interpolation_ema_alpha = 0.0
    robot._latest_target_joint_pos = None
    robot._latest_arm_feedforward_velocity = None
    robot._latest_command_monotonic = None
    robot._latest_hand_action = None
    robot._latest_gripper_action = 0.0
    robot._latest_gripper_action_space = "position"
    robot.get_robot_state = lambda: ({"joint_positions": np.zeros(7)}, {})
    robot._hand_dof = lambda: 1
    robot.get_cached_gripper_position = lambda: 0.0
    dispatched = []
    dispatched_hand = []
    robot.update_joints = lambda pos, **kwargs: dispatched.append(
        (np.asarray(pos), np.asarray(kwargs["feedforward_velocity"]))
    )
    robot._apply_hand_action = lambda action, *args, **kwargs: dispatched_hand.append(
        np.asarray(action)
    )

    zero = np.concatenate([np.zeros(7), [0.0]])
    one = np.concatenate([np.ones(7), [1.0]])
    robot.add_command_point(zero, "joint_position")
    robot.add_command_point(one, "joint_position")
    assert robot.execute_interpolated_tick()
    assert dispatched[-1][0] == pytest.approx(np.full(7, 0.5), abs=0.01)
    assert dispatched[-1][1] == pytest.approx(np.full(7, 20.0), rel=0.02)
    assert dispatched_hand[-1] == pytest.approx([0.5], abs=0.01)

    assert robot.execute_interpolated_tick()
    assert dispatched[-1][0] == pytest.approx(np.ones(7))
    assert dispatched[-1][1] == pytest.approx(np.zeros(7))
    assert dispatched_hand[-1] == pytest.approx([1.0])


def test_feedforward_deadman_bypasses_velocity_ema():
    sent = []
    arm = type(
        "Arm",
        (),
        {
            "joint_pos_limit": None,
            "get_joint_pos": lambda self: np.zeros(7),
            "set_joint_pos_vel": lambda self, pos, vel, relative: sent.append(
                np.asarray(vel)
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
    robot._prev_joint_vel = np.ones(7)
    robot._vel_smoothing_alpha = 0.3
    robot._interpolation_vel_smoothing_alpha = 0.03
    robot._vel_damp_thresh = 1.0
    robot._vel_ratio = 1.0
    robot._vel_log_file = None
    robot.validate_joint_limits = lambda target: None
    robot._agent_debug_log = lambda **kwargs: None
    robot.sync_motion_manager_with_arm = lambda target: None

    robot.update_joints(
        np.full(7, 0.05),
        execution_hz=200,
        feedforward_velocity=np.zeros(7),
        force_zero_feedforward=True,
    )

    assert sent[-1] == pytest.approx(np.zeros(7))
    assert robot._prev_joint_vel == pytest.approx(np.zeros(7))


def test_wuji_velocity_uses_measured_anchor_without_host_clipping():
    class Interpolator:
        def add_point(self, timestamp, target):
            pass

    measurements = iter([np.full(3, 0.09), np.full(3, 0.11)])
    hand = type(
        "Hand",
        (),
        {
            "model": "wuji_hand_v1",
            "joint_limits": np.column_stack((np.zeros(3), np.full(3, 0.1))),
            "get_joint_pos": lambda self: next(measurements),
        },
    )()
    robot = VegaRobot.__new__(VegaRobot)
    robot.hand = hand
    robot._gripper_io_lock = threading.Lock()
    robot._interpolator = Interpolator()
    robot._interp_lock = threading.Lock()
    robot.control_hz = 20
    robot._latest_target_joint_pos = None
    robot._latest_arm_feedforward_velocity = None
    robot._latest_command_monotonic = None
    robot._latest_hand_action = None
    robot._latest_gripper_action = 0.0
    robot._latest_gripper_action_space = "position"
    robot.get_robot_state = lambda: ({"joint_positions": np.zeros(7)}, {})
    robot._hand_dof = lambda: 3
    command = np.concatenate([np.zeros(7), np.ones(3)])

    robot.add_command_point(
        command, "joint_position", gripper_action_space="velocity"
    )
    assert robot._latest_hand_action == pytest.approx(np.full(3, 0.14))
    robot.add_command_point(
        command, "joint_position", gripper_action_space="velocity"
    )
    assert robot._latest_hand_action == pytest.approx(np.full(3, 0.16))
