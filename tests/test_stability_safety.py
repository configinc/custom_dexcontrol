"""Static tipping protection through the managed writer and model worker."""

import time
from dataclasses import replace

import numpy as np
import pytest
from test_control_runtime import Device
from test_motion_safety import activate, make_node, run_arm_ticks

from dexcontrol.core.control.collision import CoalCollisionModel
from dexcontrol.core.control.config import StabilityConfig, load_config
from dexcontrol.core.control.model_worker import ModelPolicy, ModelWorker
from dexcontrol.core.control.stability import StabilityPolicy, StaticStability


@pytest.fixture
def mass_model(tmp_path):
    # Three 1 kg bodies. Two revolute axes move CoM to (sin(left)/3, sin(right)/3).
    # Collision spheres are separated vertically so support can be tested alone.
    urdf = tmp_path / "mass.urdf"
    links = []
    for name, z in (("base", 10), ("left", 0), ("right", 5)):
        center_z = 0 if name == "base" else 1
        links.append(f'''<link name="{name}">
          <inertial><mass value="1"/><origin xyz="0 0 {center_z}"/>
            <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
          <collision><origin xyz="0 0 {z}"/><geometry><sphere radius="0.1"/></geometry></collision>
        </link>''')
    joints = []
    for name, axis in (("left", "0 1 0"), ("right", "-1 0 0")):
        joints.append(f'''<joint name="{name}_j1" type="revolute">
          <parent link="base"/><child link="{name}"/><axis xyz="{axis}"/>
          <limit lower="-4" upper="4" effort="1" velocity="1"/>
        </joint>''')
    urdf.write_text('<robot name="mass">' + "".join(links + joints) + "</robot>")
    srdf = tmp_path / "mass.srdf"
    srdf.write_text('<robot name="mass"/>')
    return CoalCollisionModel(urdf, srdf)


@pytest.fixture
def policy(mass_model):
    config = load_config()
    return StabilityPolicy(
        StaticStability(mass_model.urdf_path, config.model), config.stability
    )


def q(x, y=0):
    return {"left_j1": float(np.arcsin(x)), "right_j1": float(np.arcsin(y))}


def test_support_margin_blocks_crossing_and_permits_recovery_and_hold(policy):
    assert not policy.evaluate(q(0.45), q(0.54)).accepted
    assert policy.evaluate(q(0.54), q(0.51)).accepted  # 5 -> 15 mm, still below 20.
    assert not policy.evaluate(q(0.54), q(0.57)).accepted
    assert not policy.evaluate(q(0.54), q(0.54, 0.03)).accepted
    assert policy.evaluate(q(0.54), q(0.54)).accepted
    assert policy.evaluate(q(0.6), q(0.57)).accepted  # Recover while still outside.
    assert not policy.evaluate(q(0.54), q(0.51), base_motion=True).accepted
    assert not policy.evaluate(q(0.54), q(0.54), non_position_motion=True).accepted


def test_recovery_cannot_trade_one_unsafe_edge_for_another(policy):
    current, target = q(0.6, 0.75), q(0.606, 0.72)
    assert (
        policy.model.evaluate(target)["margin_m"]
        > policy.model.evaluate(current)["margin_m"]
    )
    assert not policy.evaluate(current, target).accepted


def test_support_approach_hysteresis(policy):
    assert not policy.evaluate(q(0.3)).guarded
    assert not policy.evaluate(q(0.39)).guarded
    assert policy.evaluate(q(0.42)).guarded
    assert policy.evaluate(q(0.39)).guarded
    assert not policy.evaluate(q(0.36)).guarded


def test_tip_block_holds_whole_batch_then_allows_recovery_without_reset(mass_model):
    node, hardware = make_node(mass_model, stability=True, collision=False)
    hardware.left.q[:] = np.arcsin(0.45)
    activate(node)
    hardware.targets = np.arcsin([0.54, 0.1])
    effects = []
    with node.gate.producing(node.normal_owner):
        node.gate.defer("hand", lambda: effects.append("moved"))
    run_arm_ticks(node, 3)
    assert not effects
    assert hardware.left.sent[-1]["pos"][0] == pytest.approx(np.arcsin(0.45))
    assert hardware.right.sent[-1]["pos"][0] == 0
    state = node.state()
    assert state["stability"]["blocked"]
    assert state["stability"]["margin_m"] == pytest.approx(0.035)
    assert state["stability"]["target_margin_m"] == pytest.approx(0.005)
    assert state["collision"]["state"] == "disabled"
    assert state["control_state"] == "running" and state["mode"] == "arms"
    hardware.targets = np.arcsin([0.42, 0])
    run_arm_ticks(node, 30)
    assert hardware.left.sent[-1]["pos"][0] == pytest.approx(np.arcsin(0.42))
    assert not node.state()["fault_reason"]


def test_base_velocity_is_zeroed_below_margin_even_with_posture_recovery(mass_model):
    node, hardware = make_node(mass_model, stability=True, collision=False)
    drive = Device("drive_j1")
    node.gate.attach("drive", drive)
    hardware.left.q[:] = np.arcsin(0.54)
    hardware.targets = np.arcsin([0.51, 0])
    arms = hardware.arm_tick

    def tick():
        arms()
        drive.sink({"vel": np.array([0.2])})

    hardware.arm_tick = tick
    activate(node)
    run_arm_ticks(node, 4)
    assert drive.sent[-1]["vel"][0] == 0
    assert "Base motion" in node.state()["stability"]["reason"]


def test_collision_clearance_cannot_override_tip_rejection(mass_model):
    policy = ModelPolicy(mass_model, load_config())
    result = policy.evaluate(np.arcsin([0.45, 0]), np.arcsin([0.54, 0]))
    assert result.collision.accepted
    assert not result.accepted
    assert result.violations == ("static_stability",)


def test_spawned_worker_computes_mass_and_support_from_same_config(mass_model):
    config = load_config()
    worker = ModelWorker(mass_model, config)
    try:
        worker.submit(np.arcsin([0.45, 0]), np.arcsin([0.54, 0]))
        deadline = time.monotonic() + 5
        result = None
        while result is None and time.monotonic() < deadline:
            result = worker.poll()
            time.sleep(0.005)
        assert result is not None
        assert not result.accepted
        assert result.stability.margin_m == pytest.approx(0.035)
        assert result.stability.mass_kg == 3
    finally:
        worker.close()


@pytest.mark.parametrize(
    "settings", [(0.02, 0.01, 0.06), (-0.01, 0.05, 0.06), (0.02, 0.05, float("nan"))]
)
def test_invalid_support_thresholds_are_rejected(settings):
    with pytest.raises(ValueError, match="Stability"):
        StabilityConfig(*settings)


def test_stability_can_be_disabled_without_disabling_collision(mass_model):
    config = load_config()
    rules = dict(config.safety)
    rules["static_stability"] = replace(rules["static_stability"], enabled=False)
    policy = ModelPolicy(mass_model, replace(config, safety=rules))
    result = policy.evaluate(np.arcsin([0.45, 0]), np.arcsin([0.54, 0]))
    assert result.accepted and result.stability is None


def test_idle_diagnostics_refresh_static_margin_from_measured_pose(mass_model):
    node, hardware = make_node(mass_model, stability=True, collision=False)
    run_arm_ticks(node, 4)
    first = node.state()["stability"]
    assert first["margin_m"] > 0.02
    hardware.left.q[:] = np.arcsin(0.6)
    run_arm_ticks(node, 4)
    current = node.state()["stability"]
    assert current["margin_m"] < 0
    assert current["target_margin_m"] is None
    assert current["result_age_s"] < 0.03
    assert node.state()["control_state"] == "idle"
    assert node.mode == "hold"
