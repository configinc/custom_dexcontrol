"""Observation snapshots and FK must stay independent of live state and IK."""

from types import SimpleNamespace

import numpy as np
import pinocchio as pin
from dexmotion.motion_manager import MotionManager
from dexmotion.utils.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation

from loop_bridge.state_reader import ArmStateReader


def test_snapshot_fk_matches_controller_without_using_its_working_data():
    model = pin.buildSampleModelManipulator()
    model.addFrame(
        pin.Frame("L_ee", model.njoints - 1, pin.SE3.Identity(), pin.FrameType.OP_FRAME)
    )
    pin_robot = RobotWrapper(model, pin.GeometryModel())
    manager = MotionManager.__new__(MotionManager)
    manager.pin_robot = pin_robot
    manager._current_qpos = pin.neutral(model)
    joints = list(model.names)[1:]
    positions = np.linspace(-0.3, 0.3, model.nq)
    state = {"pos": positions.copy(), "vel": positions * 2, "torque": positions * 3}
    robot = SimpleNamespace(
        arm_side="left",
        _arm_joint_names=joints,
        ik_controller=SimpleNamespace(motion_manager=manager),
        arm=SimpleNamespace(get_state=lambda: state),
        hand=None,
    )
    reader = ArmStateReader(robot)
    snapshot = reader.capture()
    expected = manager.fk(["L_ee"], qpos=positions, update_robot_state=False)["L_ee"].np

    # New hardware data and an IK/FK update must not change the captured pose.
    state["pos"][:] = 0.8
    state["vel"][:] = 0.9
    manager.fk(["L_ee"], qpos=np.ones(model.nq), update_robot_state=False)
    controller_pose = pin_robot.data.oMf[model.getFrameId("L_ee")].homogeneous.copy()
    observation = reader.complete(snapshot)
    np.testing.assert_allclose(observation["joint_positions"], positions)
    np.testing.assert_allclose(observation["joint_velocities"], positions * 2)
    cartesian = observation["cartesian_position"]
    np.testing.assert_allclose(cartesian[:3], expected[:3, 3])
    np.testing.assert_allclose(
        Rotation.from_euler("xyz", cartesian[3:]).as_matrix(),
        expected[:3, :3],
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        pin_robot.data.oMf[model.getFrameId("L_ee")].homogeneous, controller_pose
    )
