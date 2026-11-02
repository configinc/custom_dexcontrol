"""Real DexMotion regression checks; no Robot or hardware connection is opened."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("dexmotion")
pin = pytest.importorskip("pinocchio")
ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/dexcontrol"), str(ROOT / "examples/teleop")]
from dexbot_utils.configs import get_robot_config
from dexbot_utils.robot_info import RobotInfo
from core.vega.robot import VegaRobot
from base_arm_teleop import BaseIKController
from core.robotenv_vega.torso_kinematics import TorsoKinematics


ARM_POSES = {
    "left": [-1.4234, 1.3524, 2.8707, -1.981, .6751, -.1662, .068],
    "right": [1.4234, -1.3524, -2.8707, -1.981, -.1515, .1662, -.068],
}
TORSO_POSES = [[.7791149616, 1.5699236393, .1396263391], [.8, 1.6, .8], [.5, 1.3, .25]]


def make_controller(monkeypatch):
    monkeypatch.delenv("ROBOT_CONFIG", raising=False)
    info = RobotInfo(configs=get_robot_config("vega_1"))
    names = {side: [f"{side[0].upper()}_arm_j{i+1}" for i in range(7)] for side in ARM_POSES}
    torso_names = [f"torso_j{i+1}" for i in range(3)]
    measured = dict(zip(torso_names, TORSO_POSES[0]))
    for side, pose in ARM_POSES.items():
        measured.update(zip(names[side], pose))
    bot = SimpleNamespace(
        _robot_info=info, heartbeat=SimpleNamespace(pause=lambda: None),
        has_component=lambda name: name == "torso",
        get_joint_pos_dict=lambda **kwargs: measured.copy(),
        torso=SimpleNamespace(joint_name=torso_names),
        left_arm=SimpleNamespace(joint_name=names["left"]),
        right_arm=SimpleNamespace(joint_name=names["right"]),
    )
    config = VegaRobot._build_ik_config(SimpleNamespace(_ik_solver_type="pink", control_hz=20))
    return BaseIKController(bot=bot, visualize=False, custom_local_ik_config=config), measured


def full_fk(manager, measured, frame):
    model = pin.buildModelFromUrdf(str(manager.urdf_path))
    data = model.createData()
    q = pin.neutral(model)
    for name, value in {**manager.locked_joints_dict, **measured}.items():
        if model.existJointName(name):
            q[model.joints[model.getJointId(name)].idx_q] = value
    pin.framesForwardKinematics(model, data, q)
    return data.oMf[model.getFrameId(frame)].homogeneous.copy()


@pytest.mark.parametrize("torso", TORSO_POSES)
def test_refresh_preserves_arm_only_ik_and_matches_full_robot_fk(monkeypatch, torso):
    controller, measured = make_controller(monkeypatch)
    measured.update(zip(controller.bot.torso.joint_name, torso))
    controller.refresh_fixed_torso(measured)
    mm = controller.motion_manager
    arm_names = {name for side in ("left_arm", "right_arm")
                 for name in getattr(controller.bot, side).joint_name}
    assert set(mm.get_joint_pos_dict()) == arm_names
    for name in controller.bot.torso.joint_name:
        assert mm.locked_joints_dict[name] == measured[name]
    for side, frame in [("left", "L_ee"), ("right", "R_ee")]:
        # The public arm observation must use the newly fixed torso geometry.
        wrapper = VegaRobot.__new__(VegaRobot)
        wrapper.ik_controller = controller
        wrapper._arm_joint_names = getattr(controller.bot, side+"_arm").joint_name
        wrapper.arm_side = side
        obs = wrapper._forward_kinematics(np.array(ARM_POSES[side]))
        from scipy.spatial.transform import Rotation
        actual = np.eye(4)
        actual[:3, :3] = Rotation.from_euler("xyz", obs[3:]).as_matrix()
        actual[:3, 3] = obs[:3]
        np.testing.assert_allclose(actual, full_fk(mm, measured, frame), atol=1e-9)

        original = mm.get_joint_pos_dict()
        zero = controller.move_delta_cartesian(np.zeros(3), np.zeros(3), side)
        np.testing.assert_allclose(zero, ARM_POSES[side], atol=1e-5)
        # Exercise translation AND rotation in each direction at the actual init
        # arm pose, not just the benign synthetic pose used in the earlier check.
        for axis in range(6):
            for sign in (-1, 1):
                mm.set_joint_pos(original)
                delta = np.zeros(6); delta[axis] = sign*.002
                result = controller.move_delta_cartesian(delta[:3], delta[3:], side)
                assert result.shape == (7,) and np.all(np.isfinite(result))
                assert np.max(np.abs(result - ARM_POSES[side])) < .1
                assert set(mm.get_joint_pos_dict()) == arm_names
                for name in controller.bot.torso.joint_name:
                    assert mm.locked_joints_dict[name] == measured[name]
        mm.set_joint_pos(original)


def test_shoulder_geometry_matches_installed_full_robot_model(monkeypatch):
    controller, measured = make_controller(monkeypatch)
    mm = controller.motion_manager
    kinematics = TorsoKinematics.from_urdf(controller.bot._robot_info.urdf_path,
        np.array([[0, 1.57], [0, 3.141], [-1.57, 1.57]]), np.ones(3))
    for torso in TORSO_POSES:
        measured.update(zip(controller.bot.torso.joint_name, torso))
        transform = full_fk(mm, measured, "arm_center")
        pose = kinematics.forward(torso)
        np.testing.assert_allclose(pose[:2], transform[[0, 2], 3], atol=1e-9)
        np.testing.assert_allclose([np.cos(pose[2]), np.sin(pose[2])],
                                  transform[[0, 0], [0, 2]], atol=1e-9)


@pytest.mark.parametrize("torso", TORSO_POSES)
def test_body_mode_observations_follow_measured_torso_without_changing_ik(monkeypatch, torso):
    controller, measured = make_controller(monkeypatch)
    mm = controller.motion_manager
    locked_before = dict(mm.locked_joints_dict)
    arm_before = mm.get_joint_pos_dict()
    torso_model = TorsoKinematics.from_urdf(controller.bot._robot_info.urdf_path,
        np.array([[0, 1.57], [0, 3.141], [-1.57, 1.57]]), np.ones(3))
    measured.update(zip(controller.bot.torso.joint_name, torso))
    controller.bot.torso.get_joint_pos = lambda: np.array(torso)
    for side, frame in [("left", "L_ee"), ("right", "R_ee")]:
        wrapper = VegaRobot.__new__(VegaRobot)
        wrapper.ik_controller = controller
        wrapper.robot = controller.bot
        wrapper.body_torso_kinematics = torso_model
        wrapper._arm_joint_names = getattr(controller.bot, side+"_arm").joint_name
        wrapper.arm_side = side
        obs = wrapper._forward_kinematics(np.array(ARM_POSES[side]))
        from scipy.spatial.transform import Rotation
        actual = np.eye(4)
        actual[:3, :3] = Rotation.from_euler("xyz", obs[3:]).as_matrix()
        actual[:3, 3] = obs[:3]
        np.testing.assert_allclose(actual, full_fk(mm, measured, frame), atol=1e-9)
    assert dict(mm.locked_joints_dict) == locked_before
    assert mm.get_joint_pos_dict() == arm_before
