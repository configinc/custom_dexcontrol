"""Tool collision coverage and static support geometry, without robot I/O."""

import numpy as np
import pytest

from dexcontrol.core.control.collision import CoalCollisionModel
from dexcontrol.core.control.config import load_config
from dexcontrol.core.control.stability import StaticStability


@pytest.fixture(scope="module")
def tool_model():
    from dexbot_utils.configs import get_robot_config

    robot = get_robot_config("vega_1")
    config = load_config()
    return CoalCollisionModel.for_robot(
        robot.urdf_path,
        excluded_links=config.collision.excluded_links,
        max_outer_error_m=config.collision.max_outer_error_m,
        robotiq_sides=("left", "right"),
        model_config=config.model,
    )


def test_robot_collision_model_includes_gripper_to_gripper_and_body_pairs(tool_model):
    pairs = {frozenset(pair) for pair in tool_model.link_pairs}
    assert frozenset(("L_robotiq", "R_robotiq")) in pairs
    for gripper, opposite_arm in (("L_robotiq", "R_arm_l7"), ("R_robotiq", "L_arm_l7")):
        assert frozenset((gripper, "base")) in pairs
        assert frozenset((gripper, "torso_l3")) in pairs
        assert frozenset((gripper, opposite_arm)) in pairs
    assert (
        len(tool_model.joint_names) == 20
    )  # Full-stroke envelopes need no jaw feedback.
    assert all(mesh.is_watertight for mesh in tool_model.shape_meshes)


def test_grippers_use_symmetric_mounts_and_sensor_frame_mass_center(tool_model):
    settings = load_config().model
    stability = StaticStability(tool_model.urdf_path, settings)
    result = stability.evaluate(dict.fromkeys(tool_model.joint_names, 0.0))
    for side, prefix in (("left", "L"), ("right", "R")):
        frames = stability.model.frames
        gripper_id = stability.model.getFrameId(f"{prefix}_robotiq")
        sensor_id = stability.model.getFrameId(f"{prefix}_arm_l8")
        mount = stability.data.oMf[sensor_id].inverse() * stability.data.oMf[gripper_id]
        np.testing.assert_allclose(
            mount.translation, settings["robotiq_mount_xyz_m"], atol=1e-12
        )
        expected_x = [0, 1 if side == "left" else -1, 0]
        np.testing.assert_allclose(mount.rotation[:, 0], expected_x, atol=1e-12)
        index = stability.frames.index(gripper_id)
        sensor_com = mount.act(stability.centers[index])
        np.testing.assert_allclose(
            sensor_com, settings["robotiq_com_sensor_m"], atol=1e-12
        )
        assert frames[gripper_id].name == f"{prefix}_robotiq"
    assert result["mass_kg"] == pytest.approx(143.31356)


def test_fixed_base_mass_is_included_in_com_and_signed_support_margin(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot name="test">
      <link name="base"><inertial><mass value="10"/><origin xyz="0 0 0.1"/>
        <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial></link>
      <link name="arm"><inertial><mass value="1"/><origin xyz="2 0 0.1"/>
        <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial></link>
      <joint name="arm_j1" type="revolute"><parent link="base"/><child link="arm"/>
        <axis xyz="0 0 1"/><limit lower="-4" upper="4" effort="1" velocity="1"/></joint>
    </robot>""")
    settings = {
        "support_points_xy_m": [[-1, -1], [0.15, -1], [0.15, 1]],
        "ground_z_m": 0,
    }
    model = StaticStability(path, settings)
    outside = model.evaluate({"arm_j1": 0})
    assert outside["mass_kg"] == 11
    np.testing.assert_allclose(outside["com_m"], [2 / 11, 0, 0.1])
    assert outside["margin_m"] == pytest.approx(0.15 - 2 / 11)
    assert not outside["inside_support"]
    inside = model.evaluate({"arm_j1": np.pi})
    assert inside["inside_support"]
    assert inside["support_xy_m"] == outside["support_xy_m"]
    assert inside["margin_m"] > 0
