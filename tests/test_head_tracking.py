"""Measured torso compensation, explicit head initialization, and stop behavior."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_control_runtime import Device, Hardware

from dexcontrol.core.robotenv_vega.control_node import VegaControlNode
from dexcontrol.core.robotenv_vega.managed_hardware import ManagedVegaHardware
from dexcontrol.core.torso import Torso
from dexcontrol.robot import Robot


class HeadDevice(Device):
    def __init__(self):
        super().__init__("head_j1")
        self.joint_name = ["head_j1", "head_j2", "head_j3"]
        self.q = np.array([1.22, 0.0, -0.3])
        self.dq = np.zeros(3)
        self.joint_pos_limit = np.array(
            [[-1.483, 1.483], [-2.792, 2.792], [-1.378, 1.483]]
        )

    def set_joint_pos(self, target):
        self.sink({"pos": np.asarray(target), "vel": np.sign(target - self.q) * 0.3})


class TrackingHardware(Hardware):
    capture_head_reference = ManagedVegaHardware.capture_head_reference
    head_tick = ManagedVegaHardware.head_tick
    initial_pose = ManagedVegaHardware.initial_pose

    def __init__(self):
        super().__init__()
        self.head = HeadDevice()
        self.torso_q = np.array([0.78, 1.57, 0.14])
        torso = Torso.__new__(Torso)
        torso.get_joint_pos = lambda: self.torso_q.copy()
        self.robot = SimpleNamespace(
            head=self.head,
            torso=torso,
            robot_model="vega_1",
            _head_init_pos=np.array([2.0, 0.0, -0.3]),
        )
        self.robot.compensate_torso_pitch = lambda q, part: (
            Robot.compensate_torso_pitch(self.robot, q, part)
        )
        self.gate.attach("head", self.head)
        self.services = []
        self.capture_head_reference()

    def hold(self):
        super().hold()
        self.head_hold = self.head.get_joint_pos()
        self.head.set_joint_pos(self.head_hold)

    def maintain_hold(self):
        super().maintain_hold()
        self.head.set_joint_pos(self.head_hold)

    def move_body(self, command, dt):
        pass


@pytest.fixture
def runtime():
    hardware = TrackingHardware()
    node = VegaControlNode(hardware, hardware.config)
    yield node, hardware
    node.stop()
    node.actions.join()
    node.close()


def initialize(node, hardware):
    node.start()
    node.start_action("initial_pose", hardware.initial_pose)
    node.actions.join()
    node.tick()
    node.tick()


def test_initial_pose_completes_with_a_stalled_head_and_continues_tracking(runtime):
    node, hardware = runtime
    initialize(node, hardware)
    assert node.state()["action"]["state"] == "succeeded"
    assert node.state()["control_state"] == "running" and node.mode == "hold"
    np.testing.assert_allclose(hardware.head.sent[-1]["pos"], [1.35, 0.0, -0.3])
    # Measured head remains at 1.22; it is not a completion/fault condition.
    for _ in range(20):
        node.tick()
    assert node.state()["control_state"] == "running"
    assert hardware.head.q[0] == 1.22


@pytest.mark.parametrize("mode", ["arms", "body", "action", "hold"])
def test_head_tracks_measured_torso_in_each_active_mode(runtime, mode):
    node, hardware = runtime
    initialize(node, hardware)
    if mode == "action":
        node.start_action("fold", lambda cancelled: cancelled.wait(2))
    elif mode == "arms":
        node.mode = "arm_ready"
        node.enable_arms()
    else:
        node.mode = mode
    # An unexecuted torso goal must not change the viewing direction.
    hardware.torso_target = np.array([0.0, 0.0, -1.5])
    node.tick()
    np.testing.assert_allclose(hardware.head.sent[-1]["pos"], [1.35, 0.0, -0.3])
    hardware.torso_q[2] -= 0.2
    node.tick()
    np.testing.assert_allclose(hardware.head.sent[-1]["pos"], [1.15, 0.0, -0.3])


def test_head_saturation_keeps_reference_for_later_torso_motion(runtime):
    node, hardware = runtime
    initialize(node, hardware)
    hardware.torso_q[2] += 0.5
    node.tick()
    assert hardware.head.sent[-1]["pos"][0] == 1.483
    hardware.torso_q[2] -= 0.7
    node.tick()
    np.testing.assert_allclose(hardware.head.sent[-1]["pos"], [1.15, 0.0, -0.3])


@pytest.mark.parametrize("stop", ["halt", "stop", "fault"])
def test_stop_holds_head_and_does_not_resume_the_initial_target(runtime, stop):
    node, hardware = runtime
    initialize(node, hardware)
    if stop == "fault":
        node.fault("execution_error", "Test fault")
    else:
        getattr(node, stop)()
    node.tick()
    np.testing.assert_array_equal(hardware.head.sent[-1]["pos"], hardware.head.q)
    if stop != "halt":
        hardware.torso_q[2] -= 0.2
        node.tick()
        np.testing.assert_array_equal(hardware.head.sent[-1]["pos"], hardware.head.q)
        if stop == "fault":
            node.reset_fault()
        node.start()
        node.tick()
        np.testing.assert_allclose(hardware.head.sent[-1]["pos"], hardware.head.q)


def test_head_tracking_uses_collision_approval_and_preserves_actuator_speed(tmp_path):
    from test_motion_safety import PolicyWorker

    from dexcontrol.core.control.collision import CoalCollisionModel

    urdf, srdf = tmp_path / "head.urdf", tmp_path / "head.srdf"
    urdf.write_text("""<robot name="head">
      <link name="base"><collision><origin xyz="0 1 0"/>
        <geometry><sphere radius="0.1"/></geometry></collision></link>
      <link name="head1"><collision><origin xyz="1 0 0"/>
        <geometry><sphere radius="0.1"/></geometry></collision></link>
      <link name="head2"/><link name="head3"/>
      <joint name="head_j1" type="revolute"><parent link="base"/><child link="head1"/>
        <axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="1" velocity="3"/></joint>
      <joint name="head_j2" type="revolute"><parent link="head1"/><child link="head2"/>
        <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="3"/></joint>
      <joint name="head_j3" type="revolute"><parent link="head2"/><child link="head3"/>
        <axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="1" velocity="3"/></joint>
    </robot>""")
    srdf.write_text('<robot name="head"/>')
    model = CoalCollisionModel(urdf, srdf)
    hardware = TrackingHardware()
    hardware.head.q[0] = 1.35
    hardware.create_collision_model = lambda config: model
    hardware.clear_motion_targets = lambda rejected: None
    rules = dict(hardware.config.safety)
    rules["self_collision"] = replace(rules["self_collision"], enabled=True)
    config = replace(hardware.config, safety=rules)
    now = [0.0]
    node = VegaControlNode(
        hardware, config, clock=lambda: now[0], model_worker=PolicyWorker(model, config)
    )

    def tick(count):
        for _ in range(count):
            now[0] += 0.005
            node.tick()

    try:
        node.start()
        tick(10)
        hardware.torso_q[2] += 0.2  # Following would move the head into the obstacle.
        tick(20)
        assert node.state()["collision"]["blocked"]
        assert node.state()["control_state"] == "running"
        assert max(command["pos"][0] for command in hardware.head.sent) <= 1.35
        hardware.torso_q[2] -= 0.4  # The opposite head motion increases clearance.
        tick(20)
        assert hardware.head.sent[-1]["pos"][0] < 1.35
        assert hardware.head.sent[-1]["vel"][0] < 0
    finally:
        node.stop()
        node.close()
