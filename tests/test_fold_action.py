"""Fold ordering, feedback completion and cancellation with staged output."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest

from dexcontrol.core.control.actions import ActionCancelled
from dexcontrol.core.control.config import load_config
from dexcontrol.core.control.output import OutputGate
from dexcontrol.core.robotenv_vega import managed_hardware
from dexcontrol.core.arm import Arm
from dexcontrol.core.torso import Torso


class Device:
    def __init__(self, names, q):
        self.joint_name = names
        self.q = np.array(q, dtype=float)
        self.dq = np.zeros_like(self.q)
        self.joint_pos_limit = np.tile([-4.0, 4.0], (len(q), 1))
        self.joint_vel_limit = np.full(len(q), 1.0)
        self.command = None
        self.stuck = False

    def bind_command_sink(self, sink):
        self.sink = sink

    def get_joint_pos(self):
        return self.q.copy()

    def get_joint_vel(self):
        return self.dq.copy()

    def set_joint_pos_vel(self, pos, vel):
        self.sink({"pos": np.array(pos), "vel": np.array(vel)})

    def publish_staged_command(self, command):
        self.command = command


class Simulation:
    def __init__(self, monkeypatch):
        self.now = 0.0
        self.cancelled = False
        self.cancel_when = lambda: False
        self.records = []
        self.prepared = False
        hw = managed_hardware.ManagedVegaHardware.__new__(
            managed_hardware.ManagedVegaHardware
        )
        hw.config = load_config()
        hw.control_hz = 200
        hw.gate = OutputGate()
        hw.torso_model = SimpleNamespace(
            velocity_limits=np.full(3, 0.2), forward=lambda q: q.copy()
        )
        hw.prepare_arms = lambda: setattr(self, "prepared", True)
        for name, prefix, size in (
            ("left_arm", "L_arm", 7),
            ("right_arm", "R_arm", 7),
            ("torso", "torso", 3),
        ):
            device = Device(
                [f"{prefix}_j{i}" for i in range(1, size + 1)], np.zeros(size)
            )
            sdk_type = Torso if name == "torso" else Arm
            device.set_joint_pos_vel = create_autospec(
                sdk_type.set_joint_pos_vel.__get__(device),
                side_effect=device.set_joint_pos_vel,
            )
            hw.gate.attach(name, device)
        self.hardware = hw
        self.devices = hw.gate.devices
        self.torso_start = self.devices["torso"].q.copy()
        self.owner = object()
        hw.gate.select(self.owner)
        self.check = lambda commands, effects: None
        monkeypatch.setattr(managed_hardware, "monotonic", lambda: self.now)

    def is_set(self):
        return self.cancelled

    def wait(self, dt):
        before = {name: d.q.copy() for name, d in self.devices.items()}
        commands = self.hardware.gate.commit(self.check)
        self.records.append((self.now, before, commands))
        for name, d in self.devices.items():
            d.dq[:] = 0
            if d.command is not None and not d.stuck:
                speed = 0.2 if name == "torso" else 0.5
                step = np.clip(d.command["pos"] - d.q, -speed * dt, speed * dt)
                d.q += step
                d.dq = step / dt
        self.now += dt
        self.cancelled |= self.cancel_when()
        return self.cancelled

    def run(self):
        with self.hardware.gate.producing(self.owner):
            self.hardware.action("fold")(self)


def test_fold_finishes_left_then_right_then_torso_and_bounds_reference_speed(
    monkeypatch,
):
    sim = Simulation(monkeypatch)
    sim.run()
    pose = sim.hardware.config.poses["folded"]
    settings = sim.hardware.config.actions["fold"]
    for name, device in sim.devices.items():
        np.testing.assert_allclose(
            device.q, pose[name], atol=settings.position_tolerance_rad
        )
    first_torso = next(
        i
        for i, (_, _, commands) in enumerate(sim.records)
        if np.any(commands["torso"]["vel"] > 0)
    )
    first_right = next(
        i
        for i, (_, _, commands) in enumerate(sim.records)
        if np.any(commands["right_arm"]["pos"] != 0)
    )
    assert 0 < first_right < first_torso
    np.testing.assert_allclose(
        sim.records[first_right][1]["left_arm"],
        pose["left_arm"],
        atol=settings.position_tolerance_rad,
    )
    for name in ("left_arm", "right_arm"):
        np.testing.assert_allclose(
            sim.records[first_torso][1][name],
            pose[name],
            atol=settings.position_tolerance_rad,
        )
    assert first_torso > 0 and sim.prepared
    for name in sim.devices:
        refs = np.array([commands[name]["pos"] for _, _, commands in sim.records])
        times = np.array([t for t, _, _ in sim.records])
        assert (
            np.max(np.abs(np.diff(refs, axis=0)) / np.diff(times)[:, None])
            <= (0.2 if name == "torso" else 0.5) + 1e-9
        )
        if name != "torso":
            assert (
                np.max(np.abs(np.diff(refs, axis=0)) / np.diff(times)[:, None]) > 0.45
            )
    assert all(
        np.all(commands["torso"]["vel"] == 0)
        for _, _, commands in sim.records[:first_torso]
    )
    assert all(
        np.all(commands["torso"]["vel"] <= 0.2) for _, _, commands in sim.records
    )


@pytest.mark.parametrize("feedback", ["stalled", "moving"])
def test_unfinished_arm_never_advances_to_torso(monkeypatch, feedback):
    sim = Simulation(monkeypatch)
    sim.hardware.config = replace(
        sim.hardware.config,
        actions={
            "fold": replace(sim.hardware.config.actions["fold"], stage_timeout_s=0.5)
        },
    )
    for name in ("left_arm", "right_arm"):
        sim.devices[name].q[:] = sim.hardware.config.poses["folded"][name]
    if feedback == "stalled":
        sim.devices["left_arm"].q[0] -= 0.1
        sim.devices["left_arm"].stuck = True
    else:
        sim.devices["left_arm"].get_joint_vel = lambda: np.full(7, 0.1)
    with pytest.raises(TimeoutError, match="Fold left_arm"):
        sim.run()
    assert not sim.prepared
    assert all(
        np.array_equal(commands["torso"]["pos"], sim.torso_start)
        for _, _, commands in sim.records
    )


@pytest.mark.parametrize("stage", ["left_arm", "right_arm", "torso"])
def test_cancelling_either_stage_stops_sequence(monkeypatch, stage):
    sim = Simulation(monkeypatch)
    sim.cancel_when = (
        (lambda: sim.now >= 0.1)
        if stage == "left_arm"
        else (lambda: np.any(sim.devices["right_arm"].q != 0))
        if stage == "right_arm"
        else (lambda: not np.array_equal(sim.devices["torso"].q, sim.torso_start))
    )
    with pytest.raises(ActionCancelled, match=f"Fold {stage}"):
        sim.run()
    assert not sim.prepared
    if stage != "torso":
        np.testing.assert_array_equal(sim.devices["torso"].q, sim.torso_start)


def test_invalid_fold_target_is_rejected_before_sending_any_command(monkeypatch):
    sim = Simulation(monkeypatch)
    pose = dict(sim.hardware.config.poses["folded"])
    pose["torso"] = [0, 0, -10]
    sim.hardware.config = replace(sim.hardware.config, poses={"folded": pose})
    with pytest.raises(ValueError, match="torso exceeds joint limits"):
        sim.hardware.action("fold")
    assert not sim.hardware.gate.pending
    assert not sim.records


def test_fold_uses_model_safety_and_cancels_before_torso_on_block(monkeypatch):
    from dexbot_utils.configs import get_robot_config
    from test_motion_safety import PolicyWorker

    from dexcontrol.core.control.collision import CoalCollisionModel
    from dexcontrol.core.control.motion_safety import MotionSafety

    sim = Simulation(monkeypatch)
    config = sim.hardware.config
    # Make the endpoint violate a configured 25 mm margin (folded arms: 20 mm).
    config = replace(config, collision=replace(config.collision, margin_m=0.025))
    sim.hardware.config = config
    robot = get_robot_config("vega_1")
    for name in ("left_arm", "right_arm"):
        sim.devices[name].q[:] = robot.components[name].pose_pool["L_shape"]
    sim.hardware.gate.attach(
        "head", Device([f"head_j{i}" for i in range(1, 4)], [0, 0, 0])
    )
    model = CoalCollisionModel.for_robot(
        robot.urdf_path,
        excluded_links=config.collision.excluded_links,
        max_outer_error_m=config.collision.max_outer_error_m,
        robotiq_sides=("left", "right"),
        model_config=config.model,
    )
    safety = MotionSafety(
        sim.devices,
        config,
        model,
        clock=lambda: sim.now,
        worker=PolicyWorker(model, config),
    )

    def check(commands, effects):
        if safety.filter(commands, effects, sim.owner) == "rejected":
            sim.cancelled = True

    sim.check = check
    try:
        with pytest.raises(ActionCancelled):
            sim.run()
        assert safety.snapshot("self_collision")["blocked"]
        np.testing.assert_array_equal(sim.devices["torso"].q, sim.torso_start)
        assert not sim.prepared
    finally:
        safety.close()
