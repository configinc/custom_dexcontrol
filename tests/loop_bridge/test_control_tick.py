"""Motor dispatch must not change an in-progress IK calculation."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from dexmotion.motion_manager import MotionManager

from dexcontrol.core.robotenv_vega.server import _INIT_JOINTS
from dexcontrol.core.vega.ik_controller import BaseIKController
from dexcontrol.core.vega.robot import VegaRobot
from dexcontrol.utils.trajectory_interpolator import TrajectoryInterpolator


@pytest.mark.parametrize("arm_side", ["left", "right"])
def test_motor_tick_during_ik_preserves_zero_action_target(arm_side, monkeypatch):
    # Build only the kinematics model; VegaRobot.__init__ connects to hardware.
    robot = VegaRobot.__new__(VegaRobot)
    robot.arm_side = arm_side
    robot.control_hz = 20
    robot._ik_solver_type = "pink"
    initial = {
        f"{side[0].upper()}_arm_j{i + 1}": float(value)
        for side, positions in _INIT_JOINTS.items()
        for i, value in enumerate(positions)
    }
    manager = MotionManager(
        robot_type="vega_1",
        initial_joint_configuration_dict=initial,
        init_global_ik=False,
        local_ik_solver_type="pink",
        custom_local_ik_config=robot._build_ik_config(),
        init_planner=False,
        init_trajectory_generator=False,
        init_trajectory_smoother=False,
        init_visualizer=False,
    )
    controller = BaseIKController.__new__(BaseIKController)
    controller.arm_dof = 7
    controller.motion_manager = manager
    robot.ik_controller = controller
    robot._arm_joint_names = [f"{arm_side[0].upper()}_arm_j{i + 1}" for i in range(7)]
    measured = _INIT_JOINTS[arm_side].copy()
    pending = measured.copy()
    pending[0] += 0.1
    sent = []
    robot.arm = SimpleNamespace(
        get_joint_pos=lambda: measured.copy(),
        joint_pos_limit=None,
        set_joint_pos_vel=lambda positions, velocities, relative: sent.append(
            positions.copy()
        ),
    )
    robot.hand = None
    robot._agent_debug_log = lambda **kwargs: None
    robot._last_cmd_joint_pos = None
    robot._prev_cmd_delta = None
    robot._max_delta_scale = 3.0
    robot._MOTOR_MAX_JERK_RAD = 0.0
    robot.use_velocity_feedforward = True
    robot._prev_joint_vel = None
    robot._vel_ratio = 1.0
    robot._vel_damp_thresh = 0.05
    robot._vel_log_file = None
    robot._interp_lock = threading.Lock()
    robot._interpolator = TrajectoryInterpolator(method="linear")
    robot._interpolator.add_point(time.perf_counter(), pending)
    robot._latest_target_joint_pos = pending
    robot._latest_gripper_action = 0.0
    robot._latest_gripper_action_space = "position"
    robot._output_filter = None
    robot._ema_alpha = 0.0

    solve_ik = manager.local_ik_solver.solve_ik

    def solve_with_motor_tick(target_poses):
        # Interleave a real dispatch after the Cartesian target was calculated.
        assert robot.execute_interpolated_tick()
        assert robot._prev_command_successful
        return solve_ik(target_poses)

    monkeypatch.setattr(manager.local_ik_solver, "solve_ik", solve_with_motor_tick)
    target = controller.move_delta_cartesian(np.zeros(3), np.zeros(3), arm_side)

    assert len(sent) == 1
    np.testing.assert_allclose(sent[0], pending)
    np.testing.assert_allclose(target, measured, atol=1e-8)
