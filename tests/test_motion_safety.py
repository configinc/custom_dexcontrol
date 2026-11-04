"""Model-clearance contracts without robot I/O."""

import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from test_control_runtime import Hardware

from dexcontrol.core.control.collision import CoalCollisionModel
from dexcontrol.core.control.config import load_config
from dexcontrol.core.control.model_worker import EndpointPolicy, ModelPolicy
from dexcontrol.core.robotenv_vega.control_node import VegaControlNode


@pytest.fixture
def collision_model(tmp_path):
    # A one-meter rotating arm passes a fixed sphere at (0, 1, 0).
    # A second independently commanded arm is far away.
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("""<robot name="test">
      <link name="base"><collision><origin xyz="0 1 0"/>
        <geometry><sphere radius="0.1"/></geometry></collision></link>
      <link name="left"><collision><origin xyz="1 0 0"/>
        <geometry><sphere radius="0.1"/></geometry></collision></link>
      <link name="right"><collision><geometry><sphere radius="0.1"/></geometry></collision></link>
      <joint name="left_j1" type="revolute"><parent link="base"/><child link="left"/>
        <axis xyz="0 0 1"/><limit lower="-4" upper="4" effort="1" velocity="3"/></joint>
      <joint name="right_j1" type="revolute"><parent link="base"/><child link="right"/>
        <origin xyz="10 0 0"/><axis xyz="0 0 1"/>
        <limit lower="-4" upper="4" effort="1" velocity="3"/></joint>
    </robot>""")
    srdf = tmp_path / "robot.srdf"
    srdf.write_text('<robot name="test"/>')
    return CoalCollisionModel(urdf, srdf)


class PolicyWorker:
    """Deterministic transport: computation becomes visible on the next poll."""

    def __init__(self, model, config):
        self.policy = ModelPolicy(model, config)
        self.pending = None
        self.requests = []

    def submit(
        self,
        measured,
        target=None,
        non_position_motion=False,
        base_motion=False,
        arm_mode=False,
    ):
        assert self.pending is None
        self.requests.append(
            (measured.copy(), None if target is None else target.copy())
        )
        self.pending = self.policy.evaluate(
            measured, target, non_position_motion, base_motion, arm_mode
        )

    def poll(self):
        result, self.pending = self.pending, None
        return result

    def close(self):
        pass


def run_arm_ticks(node, count):
    for _ in range(count):
        node.advance()
        node.tick()


def activate(node):
    # Exercise lifecycle/admission without waiting for a timed mode transition.
    node.start()
    node.mode = "arm_ready"
    node.enable_arms()


def make_node(model=None, stability=False, collision=True):
    hardware = Hardware()
    for device in (hardware.left, hardware.right):
        device.joint_pos_limit = np.array([[-4.0, 4.0]])
    rules = dict(hardware.config.safety)
    rules["self_collision"] = replace(
        rules["self_collision"], enabled=model is not None and collision
    )
    rules["static_stability"] = replace(rules["static_stability"], enabled=stability)
    config = replace(
        hardware.config,
        safety=rules,
    )
    hardware.create_collision_model = lambda settings: model
    worker = PolicyWorker(model, config) if model else None
    node = VegaControlNode(hardware, config, model_worker=worker)
    if model:
        clock = [time.monotonic()]
        node.clock = node.motion_safety.guard.clock = lambda: clock[0]
        node.advance = lambda seconds=0.005: clock.__setitem__(0, clock[0] + seconds)
        hardware.clear_motion_targets = lambda rejected: None
    return node, hardware


def test_collision_distances_and_midpath_crossing(collision_model):
    m = collision_model
    start = m.positions({"left_j1": 0.0, "right_j1": 0.0})
    end = m.positions({"left_j1": np.pi, "right_j1": 0.0})
    assert m.clearances(start).min() == pytest.approx(np.sqrt(2) - 0.2)
    assert m.violation(start, 0.01) is None
    assert m.violation(end, 0.01) is None
    violation = m.check_path(start, end, 0.01)
    assert violation.pair == ("base", "left")
    assert violation.distance_m == pytest.approx(-0.2)
    assert m.check_path(start, start + 0.1, 0.01) is None


def test_missing_or_invalid_joint_feedback_is_not_assumed_zero(collision_model):
    with pytest.raises(ValueError, match="Missing"):
        collision_model.positions({"left_j1": 0})
    with pytest.raises(ValueError, match="Invalid"):
        collision_model.positions({"left_j1": np.nan, "right_j1": 0})


def test_uncertified_path_is_rejected_instead_of_silently_skipping(collision_model):
    start, end = np.array([0.0, 0.0]), np.array([1.35, 0.0])
    assert collision_model.violation(end, 0.01) is None
    assert collision_model.check_path(start, end, 0.01) is None
    limited = collision_model.check_path(start, end, 0.01, max_checks=1)
    assert "cannot certify" in limited.detail


def test_collision_blocks_whole_batch_and_gripper_without_latching_fault(
    collision_model,
):
    node, hardware = make_node(collision_model)
    hardware.left.q[:] = 1.35  # 20 mm clearance: guarded region.
    activate(node)
    hardware.targets = [1.5, 0.3]
    sent = []
    with node.gate.producing(node.normal_owner):
        node.gate.defer("hand", lambda: sent.append(True))
    run_arm_ticks(node, 3)
    assert not sent
    assert hardware.left.sent[-1]["pos"][0] == pytest.approx(1.35)
    assert hardware.right.sent[-1]["pos"][0] == 0
    state = node.state()
    assert state["control_state"] == "running"
    assert state["mode"] == "arms"
    assert state["collision"]["blocked"]
    assert not state["fault_reason"]


def test_far_region_preserves_normal_feedforward_and_does_not_certify_path(
    collision_model,
):
    node, hardware = make_node(collision_model)
    activate(node)
    hardware.arm_tick = lambda: hardware.left.sink(
        {"pos": np.array([np.pi]), "vel": np.array([4.0])}
    )
    run_arm_ticks(node, 3)
    assert hardware.left.sent[-1]["vel"][0] == 4
    assert hardware.left.sent[-1]["pos"][0] == pytest.approx(np.pi)
    assert node.state()["collision"]["state"] == "monitoring"


def test_guarded_motion_keeps_idle_arm_at_limit_without_faulting_on_feedback_noise(
    collision_model,
):
    node, hardware = make_node(collision_model)
    hardware.left.q[:] = 1.35
    hardware.right.q[:] = -4.00003
    activate(node)
    # Initial Pose moves one arm while the other arm has no new command.
    hardware.arm_tick = lambda: hardware.left.command(1.3)
    run_arm_ticks(node, 20)
    assert node.state()["control_state"] == "running"
    assert hardware.left.sent[-1]["pos"][0] < 1.35
    assert hardware.right.sent[-1]["pos"][0] == -4.0
    worker = node.motion_safety.guard.worker
    # Collision geometry must continue to use actual feedback, not the clamp.
    assert all(measured[1] == -4.00003 for measured, _ in worker.requests)
    assert all(target[1] == -4.0 for _, target in worker.requests if target is not None)


def test_generated_velocity_endpoint_uses_sdk_position_range(collision_model):
    node, hardware = make_node(collision_model)
    hardware.left.q[:] = 1.35
    hardware.right.q[:] = 3.99
    activate(node)

    def inputs():
        hardware.left.command(1.3)
        # The 20 Hz approval endpoint would be 4.09 rad, above the 4 rad bound.
        hardware.right.sink({"vel": np.array([2.0])})

    hardware.arm_tick = inputs
    run_arm_ticks(node, 20)
    assert node.state()["control_state"] == "running"
    assert 3.99 < hardware.right.sent[-1]["pos"][0] <= 4.0


def test_one_shot_command_and_hand_effect_survive_initial_monitor_hold(collision_model):
    node, hardware = make_node(collision_model)
    activate(node)
    hardware.arm_tick = lambda: None
    effects = []
    with node.gate.producing(node.normal_owner):
        hardware.left.command(0.1)
        node.gate.defer("hand", lambda: effects.append("sent"))
    run_arm_ticks(node, 1)
    assert effects == []
    run_arm_ticks(node, 8)
    assert effects == ["sent"]
    assert hardware.left.sent[-1]["pos"][0] == pytest.approx(0.1)


def test_recovery_from_overlap_starts_and_moves_without_fault_reset(collision_model):
    node, hardware = make_node(collision_model)
    hardware.left.q[:] = 1.5
    activate(node)
    hardware.targets = [1.4, 0.0]
    run_arm_ticks(node, 10)
    target = hardware.left.sent[-1]
    assert 1.4 <= target["pos"][0] < 1.5
    assert target["vel"][0] == 0
    assert node.state()["control_state"] == "running"
    assert node.state()["mode"] == "arms"


def test_approval_cannot_authorize_a_newer_unchecked_target(collision_model):
    node, hardware = make_node(collision_model)
    hardware.left.q[:] = 1.35
    activate(node)
    hardware.targets = [1.3, 0]
    run_arm_ticks(node, 1)
    hardware.targets = [1.5, 0]
    run_arm_ticks(node, 6)
    assert hardware.left.sent[-1]["pos"][0] <= 1.35
    assert all(c["pos"][0] <= 1.35 for c in hardware.left.sent[1:])


def test_repeated_approvals_do_not_restart_interpolation_at_lagging_feedback(
    collision_model,
):
    node, hardware = make_node(collision_model)
    hardware.left.q[:] = 1.35
    activate(node)
    hardware.targets = [1.3, 0]
    run_arm_ticks(node, 50)
    targets = np.array([c["pos"][0] for c in hardware.left.sent[1:]])
    assert np.all(np.diff(targets) <= 1e-12)
    assert targets[-1] == pytest.approx(1.3, abs=1e-4)


def test_restart_requires_fresh_monitor_before_releasing_output(collision_model):
    node, hardware = make_node(collision_model)
    activate(node)
    hardware.targets = [0.1, 0]
    run_arm_ticks(node, 1)  # Pending result says far; now revoke it.
    node.stop()
    hardware.left.q[:] = 1.35
    activate(node)
    hardware.targets = [1.5, 0]
    run_arm_ticks(node, 3)
    assert hardware.left.sent[-1]["pos"][0] == pytest.approx(1.35)
    assert node.state()["collision"]["blocked"]


def test_stop_discards_late_approval(collision_model):
    node, hardware = make_node(collision_model)
    hardware.left.q[:] = 1.35
    activate(node)
    hardware.targets = [1.3, 0]
    run_arm_ticks(node, 1)
    node.stop()
    run_arm_ticks(node, 3)
    assert hardware.left.sent[-1]["pos"][0] == pytest.approx(1.35)
    assert node.state()["control_state"] == "idle"


def test_stationary_arm_target_does_not_treat_hand_motion_as_a_hold(collision_model):
    node, hardware = make_node(collision_model)
    hardware.left.q[:] = 1.5
    hardware.targets = [1.5, 0]
    activate(node)
    effects = []
    with node.gate.producing(node.normal_owner):
        node.gate.defer("hand", lambda: effects.append("moved"))
    run_arm_ticks(node, 3)
    assert not effects
    assert node.state()["collision"]["blocked"]
    assert node.state()["control_state"] == "running"


def test_missing_monitor_result_faults_and_holds(collision_model):
    node, hardware = make_node(collision_model)
    activate(node)
    node.motion_safety.guard.worker.poll = lambda: None
    run_arm_ticks(node, 1)
    node.advance(0.11)
    with pytest.raises(TimeoutError, match="expired"):
        node.tick()
    assert node.state()["fault_reason"] == "execution_error"
    assert hardware.left.sent[-1]["pos"][0] == 0


def test_rejected_action_is_cancelled_without_collision_fault(collision_model):
    node, hardware = make_node(collision_model)
    hardware.left.q[:] = 1.35
    node.start()
    ready = threading.Event()

    def action(cancelled):
        hardware.left.command(1.5)
        ready.set()
        assert cancelled.wait(2)

    node.start_action("test", action)
    try:
        assert ready.wait(1)
        run_arm_ticks(node, 3)
        node.actions.join()
        assert node.state()["control_state"] == "running"
        assert node.actions.snapshot()["state"] == "cancelled"
    finally:
        node.stop()
        node.actions.join()


def test_pairwise_recovery_cannot_hide_worsening_secondary_pair():
    class Model:
        DISTANCE_TOLERANCE_M = 1e-6
        link_pairs = (("a", "b"), ("c", "d"))

        def clearances(self, q):
            return np.array(q)

    policy = EndpointPolicy(Model(), load_config().collision)
    assert not policy.evaluate([-0.01, 0.005], [-0.005, 0.004]).accepted
    assert policy.evaluate([-0.01, 0.005], [-0.005, 0.005]).accepted
    assert not policy.evaluate([-0.01, 0.02], [-0.005, 0.009]).accepted
    assert not policy.evaluate([-0.01, 0.02], [-0.01, 0.03]).accepted


def test_monitor_hysteresis():
    class Model:
        DISTANCE_TOLERANCE_M = 1e-6
        link_pairs = (("a", "b"),)

        def clearances(self, q):
            return np.array(q)

    policy = EndpointPolicy(Model(), load_config().collision)
    assert not policy.evaluate([0.05]).guarded
    assert not policy.evaluate([0.035]).guarded
    assert policy.evaluate([0.025]).guarded
    assert policy.evaluate([0.035]).guarded
    assert not policy.evaluate([0.045]).guarded


def test_arm_mode_excludes_only_configured_gripper_pair():
    class Model:
        DISTANCE_TOLERANCE_M = 1e-6
        link_pairs = (
            ("R_robotiq", "L_robotiq"),
            ("L_robotiq", "R_arm_l7"),
            ("L_robotiq", "torso_l3"),
        )

        def clearances(self, q):
            return np.array(q)

    policy = EndpointPolicy(Model(), load_config().collision)
    measured, target = [-0.01, 0.1, 0.1], [-0.02, 0.1, 0.1]
    arms = policy.evaluate(measured, target, arm_mode=True)
    assert arms.accepted and not arms.guarded
    assert arms.pair != Model.link_pairs[0]
    assert not policy.evaluate(measured, [-0.02, 0.005, 0.1], arm_mode=True).accepted
    assert not policy.evaluate(measured, [-0.02, 0.1, 0.005], arm_mode=True).accepted
    other = policy.evaluate(measured, target, arm_mode=False)
    assert not other.accepted and other.guarded
    assert other.pair == Model.link_pairs[0]
    assert policy.evaluate(measured, [-0.005, 0.1, 0.1], arm_mode=False).accepted


def test_leaving_arm_mode_discards_pending_gripper_exclusion(collision_model, tmp_path):
    urdf = tmp_path / "grippers.urdf"
    urdf.write_text(
        collision_model.urdf_path.read_text()
        .replace('"base"', '"R_robotiq"')
        .replace('"left"', '"L_robotiq"')
    )
    model = CoalCollisionModel(urdf, collision_model.srdf_path)
    node, hardware = make_node(model)
    hardware.left.q[:] = 1.35
    activate(node)
    hardware.targets = [1.4, 0]
    run_arm_ticks(node, 10)
    assert hardware.left.sent[-1]["pos"][0] == 1.4
    assert node.state()["collision"]["state"] == "monitoring"
    guard = node.motion_safety.guard
    # Leave an arm-mode monitor reply in flight, then change only collision scope.
    while guard.request is None:
        run_arm_ticks(node, 1)
    owner = guard.owner
    for _ in range(20):
        node.advance()
        commands = {"left": {"pos": np.array([1.4])}}
        guard.filter(commands, {}, owner, arm_mode=False)
        assert commands["left"]["pos"][0] == 1.35
    assert guard.snapshot("self_collision")["blocked"]
    assert set(guard.snapshot("self_collision")["pair"]) == {"L_robotiq", "R_robotiq"}


def mesh_scene(
    tmp_path, fixed, moving, *, scale="1 1 1", origin="0 0 0", joint_origin="0 0 0"
):
    fixed.export(tmp_path / "fixed.obj")
    moving.export(tmp_path / "moving.obj")
    urdf = tmp_path / "mesh.urdf"
    urdf.write_text(f'''<robot name="mesh">
      <link name="base"><collision><geometry><mesh filename="fixed.obj"/></geometry></collision></link>
      <link name="arm"><collision><origin xyz="{origin}"/>
        <geometry><mesh filename="moving.obj" scale="{scale}"/></geometry></collision></link>
      <joint name="arm_j1" type="revolute"><parent link="base"/><child link="arm"/>
        <origin xyz="{joint_origin}"/><axis xyz="0 0 1"/>
        <limit lower="-4" upper="4" effort="1" velocity="3"/></joint>
    </robot>''')
    srdf = tmp_path / "mesh.srdf"
    srdf.write_text('<robot name="mesh"/>')
    return CoalCollisionModel(urdf, srdf)


def test_mesh_clearance_respects_scale_origins_and_nearest_points(tmp_path):
    import trimesh

    fixed = trimesh.creation.box(extents=[0.4] * 3)
    fixed.apply_translation([1, 0, 0])
    moving = trimesh.creation.box(extents=[0.2] * 3)
    model = mesh_scene(
        tmp_path, fixed, moving, scale="2 1 1", origin="0.1 0 0", joint_origin="0.2 0 0"
    )
    assert model.clearances([0])[0] == pytest.approx(0.3, abs=1e-6)
    points = model.nearest_points(0)
    assert np.linalg.norm(points[1] - points[0]) == pytest.approx(0.3, abs=1e-6)
    assert model.clearances([np.pi / 2])[0] == pytest.approx(0.5, abs=1e-6)


def test_disconnected_mesh_pieces_preserve_free_space(tmp_path):
    import trimesh

    left, right = (trimesh.creation.box(extents=[0.2] * 3) for _ in range(2))
    left.apply_translation([-1, 0, 0])
    right.apply_translation([1, 0, 0])
    fixed = trimesh.util.concatenate([left, right])
    moving = trimesh.creation.box(extents=[0.2] * 3)
    model = mesh_scene(tmp_path, fixed, moving)
    assert model.clearances([0])[0] == pytest.approx(0.8, abs=1e-6)


def test_mesh_containment_is_a_collision_without_surface_intersection(tmp_path):
    import trimesh

    model = mesh_scene(
        tmp_path,
        trimesh.creation.box(extents=[2] * 3),
        trimesh.creation.box(extents=[0.2] * 3),
    )
    assert model.clearances([0])[0] < 0
    assert model.violation([0], 0.01).pair == ("base", "arm")


def test_rotating_mesh_is_checked_between_clear_endpoints_and_midpoint(tmp_path):
    import trimesh

    fixed = trimesh.creation.box(extents=[0.1] * 3)
    fixed.apply_translation([0.64, 0.64, 0])
    moving = trimesh.creation.box(extents=[2, 0.04, 0.04])
    model = mesh_scene(tmp_path, fixed, moving)
    for angle in (0, np.pi / 2, np.pi):
        assert model.violation([angle], 0.01) is None
    assert model.check_path([0], [np.pi], 0.01) is not None


def test_srdf_exclusion_is_used_without_pose_based_pair_removal(
    collision_model, tmp_path
):
    srdf = tmp_path / "excluded.srdf"
    srdf.write_text(
        '<robot name="test"><disable_collisions link1="base" link2="left" reason="test"/></robot>'
    )
    model = CoalCollisionModel(collision_model.urdf_path, srdf)
    assert model.check_path([0, 0], [np.pi, 0], 0.01) is None


def test_primitive_dimensions_are_independent_for_each_link(collision_model, tmp_path):
    urdf = tmp_path / "different_radii.urdf"
    urdf.write_text(
        collision_model.urdf_path.read_text().replace('radius="0.1"', 'radius="0.2"', 1)
    )
    model = CoalCollisionModel(urdf, tmp_path / "robot.srdf")
    assert model.clearances([0, 0]).min() == pytest.approx(np.sqrt(2) - 0.3)


@pytest.mark.parametrize("margin", [-1, float("nan"), float("inf")])
def test_invalid_clearance_margin_is_rejected(collision_model, margin):
    with pytest.raises(ValueError, match="margin"):
        collision_model.violation([0, 0], margin)
    with pytest.raises(ValueError, match="margin"):
        collision_model.check_path([0, 0], [0.1, 0], margin)


@pytest.mark.parametrize("lifecycle", ["idle", "running", "fault"])
def test_hold_refreshes_measured_clearance_without_approving_motion(
    collision_model, lifecycle
):
    node, hardware = make_node(collision_model)
    if lifecycle == "running":
        node.start()
    elif lifecycle == "fault":
        node.fault("execution_error", "Test fault")
    run_arm_ticks(node, 4)
    first = node.state()["collision"]
    assert first["clearance_m"] == pytest.approx(np.sqrt(2) - 0.2)
    assert first["required_margin_m"] == 0.01
    hardware.left.q[:] = 1.5
    run_arm_ticks(node, 4)
    current = node.state()["collision"]
    assert current["clearance_m"] < 0
    assert current["result_age_s"] < 0.03
    assert current["state"] == "paused" and not current["blocked"]
    assert node.state()["control_state"] == lifecycle
    assert node.mode == "hold"
    assert all(target is None for _, target in node.motion_safety.guard.worker.requests)
    assert all(c["pos"][0] in (0.0, 1.5) for c in hardware.left.sent)


@pytest.mark.parametrize("lifecycle", ["idle", "running", "fault"])
def test_hold_resamples_after_delayed_tick_without_starting_motion(
    collision_model, lifecycle
):
    node, hardware = make_node(collision_model)
    if lifecycle == "running":
        node.start()
    elif lifecycle == "fault":
        node.fault("execution_error", "Existing fault")
    run_arm_ticks(node, 4)
    # IK preparation during a handover can delay the stopped controller.
    node.advance(0.5)
    hardware.left.q[:] = 0.2
    node.tick()
    assert node.state()["control_state"] == lifecycle
    assert node.state()["collision"]["result_age_s"] >= 0.5
    run_arm_ticks(node, 4)
    assert node.state()["collision"]["result_age_s"] < 0.03
    assert node.mode == "hold"
    assert hardware.left.sent[-1]["pos"][0] == pytest.approx(0.2)
    assert all(target is None for _, target in node.motion_safety.guard.worker.requests)
