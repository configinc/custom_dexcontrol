"""Capture measured state cheaply; compute observation FK on a separate worker."""

from typing import Any

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation


class ArmStateReader:
    """Own FK working data so observation calculation cannot race the IK solver."""

    def __init__(self, robot: Any) -> None:
        self._robot = robot
        manager = robot.ik_controller.motion_manager
        self._model = manager.pin_robot.model
        self._data = self._model.createData()
        self._qpos = manager.get_joint_pos().copy()
        self._joint_indices = [
            self._model.joints[self._model.getJointId(name)].idx_q
            for name in robot._arm_joint_names
        ]
        frame = "L_ee" if robot.arm_side == "left" else "R_ee"
        self._frame_id = self._model.getFrameId(frame)

    def capture(self) -> dict[str, Any]:
        """Copy one cached arm sample without FK or blocking gripper I/O."""
        robot = self._robot
        state = robot.arm.get_state()
        wrench = getattr(robot.arm, "wrench_sensor", None)
        return {
            "joint_positions": np.array(state["pos"], dtype=np.float64),
            "joint_velocities": np.array(state["vel"], dtype=np.float64),
            "joint_torques_computed": np.array(
                state.get("torque", np.zeros(7)), dtype=np.float64
            ),
            "gripper_position": (
                robot.get_cached_gripper_position() if robot.hand is not None else 0.0
            ),
            "wrench_state": (
                np.array(wrench.get_wrench_state(), dtype=np.float64)
                if wrench is not None
                else np.zeros(6)
            ),
        }

    def complete(self, state: dict[str, Any]) -> dict[str, Any]:
        """Compute Cartesian pose from the captured joints, using private FK data."""
        qpos = self._qpos.copy()
        qpos[self._joint_indices] = state["joint_positions"]
        pin.framesForwardKinematics(self._model, self._data, qpos)  # pyright: ignore[reportAttributeAccessIssue]
        pose = self._data.oMf[self._frame_id]
        cartesian = np.concatenate(
            (pose.translation, Rotation.from_matrix(pose.rotation).as_euler("xyz"))
        )
        return {**state, "cartesian_position": cartesian}
