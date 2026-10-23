from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "dexcontrol"))
from core.robotenv_vega.body_hardware import VegaHardware
from core.robotenv_vega.torso_kinematics import TorsoKinematics
from core.chassis_kinematics import bounded_wheel_control


class Component:
    def __init__(self, q):
        self.q=np.array(q,dtype=float);self.vel=np.zeros_like(self.q);self.stamp=0
        self.joint_name=["test"]*len(q)
    def get_joint_pos(self):return self.q.copy()
    def get_joint_vel(self):return self.vel.copy()
    def get_timestamp_ns(self):return self.stamp
    def set_joint_pos_vel(self,q,vel,wait_time=0):self.q=np.array(q);self.command_velocity=np.array(vel)


class ArmComponent(Component):
    def set_joint_pos_vel(self, q, vel, relative=False):
        self.q = np.array(q)
        self.command_velocity = np.array(vel)


def make_hardware():
    h=VegaHardware.__new__(VegaHardware)
    h.torso=Component([.8,1.6,.8])
    h.torso_model=TorsoKinematics([[-.235,.248],[.396,.082],[-.40718,.09764],[-.05908,.44528]],
                                [[0,1.57],[0,3.141],[-1.57,1.57]],[.2]*3)
    h.torso_limits=h.torso_model.limits
    steer,drive=Component([0,0]),Component([0,0])
    def command(angles,speeds,wait_time=0):
        h.last_wheel_command=(np.array(angles),np.array(speeds))
    h.chassis=SimpleNamespace(chassis_steer=steer,chassis_drive=drive,steering_angle=np.zeros(2),
                             _center_to_wheel_axis_dist=.219,_half_wheels_dist=.225,max_lin_vel=.8,
                             steering_limits=np.array([[-2.35,2.35]]*2),set_motion_state=command,
                             _compute_wheel_control=bounded_wheel_control)
    h.services=[]
    for _ in range(2):
        arm=ArmComponent([.1]*7)
        robot=SimpleNamespace(arm=arm,_stop_gripper_worker=lambda:None,reset_filter_state=lambda:None)
        h.services.append(SimpleNamespace(_robot=robot, _cmd_lock=threading.RLock(),
                                          _stop_control_loop=lambda:None,
                                          _cancel_move=threading.Event(),
                                          _cancel_force_calibration_motion=lambda:None))
    h.command_limits=np.array([.15,.15,.3,.02,.02,.05]);h.wheel_command=np.zeros(2);h.hold()
    return h


def test_body_motion_keeps_arm_joints_and_steering_bounded():
    h=make_hardware();before=h.services[0]._robot.arm.q.copy();start=h.torso_model.forward(h.torso.q)
    h.set_shoulder_target(start+[.01,0,0])
    for _ in range(100):h.move_body(np.zeros(3),.02)
    end=h.torso_model.forward(h.torso.q)
    np.testing.assert_equal(h.services[0]._robot.arm.q,before)
    assert end[0]>start[0]
    np.testing.assert_allclose(end[1:],start[1:],atol=1e-9)
    assert np.all(np.abs(h.torso.command_velocity)<=.2)


def test_turning_starts_only_after_wheels_stop_then_align():
    h=make_hardware();h.chassis.chassis_drive.vel[:]=.1
    h.move_body(np.array([0,.1,0]),.02)
    np.testing.assert_equal(h.last_wheel_command[0],[0,0])
    np.testing.assert_equal(h.last_wheel_command[1],[0,0])
    h.chassis.chassis_drive.vel[:]=0
    h.move_body(np.array([0,.1,0]),.02)
    angles=h.last_wheel_command[0]
    assert np.all(np.abs(angles)<=2.35) and np.all(np.abs(angles)>1)
    np.testing.assert_equal(h.last_wheel_command[1],[0,0])
    h.chassis.steering_angle=angles
    h.move_body(np.array([0,.1,0]),.02)
    assert np.any(h.last_wheel_command[1]!=0)


def test_stop_confirmation_requires_new_sensor_samples():
    h=make_hardware()
    assert not h.stationary()
    for c in h.components():c.stamp+=1
    assert h.stationary()


def test_stop_cancels_both_reset_moves_before_waiting_for_arm_commands():
    h = make_hardware()
    entered = [threading.Event(), threading.Event()]
    saw_bilateral_stop = []
    for service in h.services:
        service._cancel_move.clear()

    def reset_worker(index):
        service = h.services[index]
        with service._cmd_lock:
            entered[index].set()
            # A reset can own the arm command lock until cancellation arrives.
            stopped = all(s._cancel_move.wait(1) for s in h.services)
            saw_bilateral_stop.append(stopped)
            service._robot.arm.q[:] = .3 + index * .1

    workers = [threading.Thread(target=reset_worker, args=(i,)) for i in range(2)]
    for worker in workers:
        worker.start()
    try:
        assert all(event.wait(1) for event in entered)
        h.hold()
    finally:
        for service in h.services:
            service._cancel_move.set()
        for worker in workers:
            worker.join(2)
    assert saw_bilateral_stop == [True, True]
    for index, service in enumerate(h.services):
        np.testing.assert_allclose(h.hold_joints[index], .3 + index * .1)
        np.testing.assert_array_equal(service._robot.arm.command_velocity, np.zeros(7))


def test_release_keeps_target_when_measured_pose_lags_or_drifts():
    h = make_hardware()
    target = h.torso_target.copy()
    commands = []
    # Keep feedback separate from the commands, as on a delayed real actuator.
    h.torso.set_joint_pos_vel = lambda q, vel, wait_time=0: commands.append(
        (np.array(q).copy(), np.array(vel).copy()))
    for command in (np.zeros(3), np.array([.05, 0, 0])):
        h.torso.q -= np.array([.08, .08, 0.])
        h.move_body(command, .02)
        np.testing.assert_array_equal(h.torso_target, target)
        np.testing.assert_array_equal(commands[-1][0], target)
        np.testing.assert_array_equal(commands[-1][1], np.zeros(3))
    # An explicit Stop still discards the old target and holds measured joints.
    measured = h.torso.q.copy()
    h.hold()
    np.testing.assert_array_equal(h.torso_target, measured)


def test_absolute_goal_is_fixed_while_path_runs_and_reaches_goal():
    h = make_hardware()
    goal = h.shoulder_target + [0, -.04, 0]
    start = h.torso_command.copy()
    h.set_shoulder_target(goal)
    duration = h.torso_path.duration
    for _ in range(int(duration/.02)+2):
        previous = h.torso_command.copy()
        h.move_body(np.zeros(3), .02)
        np.testing.assert_array_equal(h.snapshot()["shoulder_target"], goal)
        assert np.all(np.abs(h.torso_command-previous) <= .2*.02+1e-10)
        assert np.all(np.abs(h.torso.command_velocity) <= .2+1e-10)
        pose = h.torso_model.forward(h.torso_command)
        np.testing.assert_allclose(pose[[0,2]], goal[[0,2]], atol=1e-5)
    assert not np.array_equal(start, h.torso_command)
    np.testing.assert_allclose(h.torso_model.forward(h.torso_command), goal, atol=1e-10)


def test_rejected_goal_preserves_accepted_goal_and_trajectory():
    h = make_hardware()
    goal = h.shoulder_target + [.02, 0, 0]
    h.set_shoulder_target(goal)
    path = h.torso_path
    for rejected in ([10,10,0], [float('nan'),0,0]):
        with pytest.raises(ValueError):h.set_shoulder_target(rejected)
        np.testing.assert_array_equal(h.shoulder_target, goal)
        assert h.torso_path is path
    h.move_body(np.zeros(3), .02)
    assert h.torso_elapsed == .02
    # Duplicate target packets keep the existing plan and progress.
    h.set_shoulder_target(goal)
    assert h.torso_path is path and h.torso_elapsed == .02
    h.hold()
    assert h.torso_path is None
    np.testing.assert_array_equal(h.torso_target, h.torso.get_joint_pos())


def test_retargeting_starts_from_current_command_without_a_jump():
    h = make_hardware()
    h.set_shoulder_target(h.shoulder_target+[.02,-.02,.02])
    h.move_body(np.zeros(3),.105)
    before=h.torso_command.copy()
    goal=h.torso_model.forward(before)+[-.01,.005,-.02]
    h.set_shoulder_target(goal)
    np.testing.assert_array_equal(h.torso_command,before)
    h.torso.q += [.01,-.01,.01]  # feedback does not move the accepted goal
    h.move_body(np.zeros(3),.035)
    assert np.all(np.abs(h.torso_command-before)<=.2*.035+1e-10)
    np.testing.assert_array_equal(h.shoulder_target,goal)


def test_delayed_torso_reaches_goal_after_planned_time_in_both_directions():
    h = make_hardware()
    initial = h.torso.q.copy()
    # Torso actuators use positive speed magnitudes to approach position
    # targets. Model slower tracking separately from commanded positions.
    def command(q, speed, wait_time=0):
        h.torso.command_pos = np.array(q).copy()
        h.torso.command_velocity = np.array(speed).copy()
    h.torso.set_joint_pos_vel = command
    goals = [h.shoulder_target + [0, -.04, 0], h.shoulder_target.copy()]
    for goal in goals:
        start = h.torso.q.copy()
        h.set_shoulder_target(goal)
        duration = h.torso_path.duration
        for tick in range(int(duration/.02)+100):
            h.move_body(np.zeros(3), .02)
            speed = h.torso.command_velocity
            assert np.all(speed >= 0) and np.all(speed <= .2)
            # Only move at half the requested speed, simulating actuator lag.
            h.torso.q += np.clip(h.torso.command_pos-h.torso.q, -speed*.01, speed*.01)
            np.testing.assert_array_equal(h.shoulder_target, goal)
            if tick == int(duration/.02):
                assert np.max(np.abs(h.torso.q-h.torso_target)) > .005
        assert not np.array_equal(h.torso.q, start)
        np.testing.assert_allclose(h.torso_model.forward(h.torso.q), goal, atol=1e-9)
    np.testing.assert_allclose(h.torso.q, initial, atol=1e-9)
    h.hold()
    np.testing.assert_array_equal(h.torso.command_velocity, np.zeros(3))


def test_paused_feedback_does_not_run_reference_to_distant_goal():
    h = make_hardware()
    start = h.torso.q.copy()
    goal = h.shoulder_target + [.02, -.04, .05]
    h.set_shoulder_target(goal)
    def command(q, speed, wait_time=0):
        h.torso.command_pos = np.array(q).copy()
        h.torso.command_velocity = np.array(speed).copy()
    h.torso.set_joint_pos_vel = command
    for _ in range(250):
        h.move_body(np.zeros(3), .02)
        np.testing.assert_array_equal(h.shoulder_target, goal)
        assert np.all(np.abs(h.torso.command_pos-start) <= .2*h.TORSO_LOOKAHEAD+1e-10)
    assert np.max(np.abs(h.torso.command_pos-start)) > .01
    # Once feedback moves again, follow the same path to the unchanged goal.
    # J2/J3 are slower than J1, as on the physical torso.
    moving = np.zeros(3, dtype=bool)
    for _ in range(1500):
        previous = h.torso_command.copy()
        h.move_body(np.zeros(3), .02)
        assert np.all(np.abs(h.torso_command-previous) <= .2*.02+1e-10)
        step = np.array([.2, .04, .04])*.02
        moving |= np.abs(h.torso.command_pos-h.torso.q) > .01
        h.torso.q += np.where(moving, np.clip(h.torso.command_pos-h.torso.q, -step, step), 0.)
    np.testing.assert_allclose(h.torso_model.forward(h.torso.q), goal, atol=1e-9)
