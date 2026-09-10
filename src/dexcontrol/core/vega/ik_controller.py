# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Arm inverse kinematics shared by the Vega runtime and teleop examples."""

import numpy as np
import pytransform3d.rotations as pr
from dexmotion.motion_manager import MotionManager
from dexmotion.utils import robot_utils
from loguru import logger

from dexcontrol.robot import Robot


class BaseIKController:
    """Unified controller for arm inverse kinematics operations.

    This class supports both joint-space and Cartesian-space control of robot arms
    using the MotionManager for kinematics calculations.

    Attributes:
        bot (Robot): Robot instance to control.
        motion_manager (MotionManager): MotionManager instance for IK solving.
        arm_dof (int): Number of degrees of freedom in the robot arms.
        visualize (bool): Whether to enable visualization.
    """

    def __init__(
        self,
        bot: Robot | None = None,
        visualize: bool = True,
        ik_solver_type: str = "pink",
        custom_local_ik_config=None,
    ):
        """Initialize the arm IK controller.

        Args:
            bot (Robot | None): Robot instance to control. If None, a new Robot
                instance will be created.
            visualize (bool): Whether to enable visualization. Defaults to True.
            ik_solver_type (str): IK solver backend — "pink" (QP-based) or "placo".
            custom_local_ik_config: Optional IKConfig to override default solver config.

        Raises:
            RuntimeError: If config path can't be found or Motion Manager
                initialization fails.
        """
        # Initialize robot if not provided
        if bot is None:
            self.bot = Robot()
        else:
            self.bot = bot

        self.visualize = visualize

        self.arm_dof = 7  # Number of degrees of freedom in the arms

        # Get initial joint positions
        try:
            if self.bot.has_component("torso"):
                joint_pos_dict = self.bot.get_joint_pos_dict(
                    component=["left_arm", "right_arm", "torso"]
                )
            else:
                joint_pos_dict = self.bot.get_joint_pos_dict(
                    component=["left_arm", "right_arm"]
                )
            logger.info("Initial joint positions retrieved successfully")
        except Exception as e:
            logger.error(f"Error getting joint positions: {e}")
            raise

        # Initialize MotionManager for IK solving

        try:
            self.bot.heartbeat.pause()

            self.motion_manager = MotionManager(
                init_visualizer=self.visualize,
                initial_joint_configuration_dict=joint_pos_dict,
                local_ik_solver_type=ik_solver_type,
                custom_local_ik_config=custom_local_ik_config,
                init_planner=False,
            )

            logger.success("Motion Manager setup complete")
        except Exception as e:
            logger.error(f"Error initializing Motion Manager: {e}")
            raise

    def move_delta_joint_space(
        self, new_joint_pos_dict: dict[str, float]
    ) -> dict[str, float]:
        """Apply a delta movement to specified joints.

        Args:
            new_joint_pos_dict (dict[str, float]): Dictionary of joint names and
                target positions.

        Returns:
            dict[str, float]: Dictionary of updated joint positions.

        Raises:
            RuntimeError: If robot or IK solver is not initialized.
        """
        if self.motion_manager.pin_robot is None:
            raise RuntimeError("Robot is not initialized")
        if self.motion_manager.local_ik_solver is None:
            raise RuntimeError("Local IK solver is not initialized")

        current_qpos_dict = self.motion_manager.get_joint_pos_dict()

        # Create a new dictionary with updated values
        updated_new_joint_pos_dict = dict(current_qpos_dict)
        updated_new_joint_pos_dict.update(new_joint_pos_dict)

        updated_new_joint_pos = robot_utils.get_qpos_from_joint_dict(
            self.motion_manager.pin_robot, updated_new_joint_pos_dict
        )

        qpos_dict, is_in_collision, is_within_joint_limits = (
            self.motion_manager.local_ik_solver.solve_ik(
                target_configuration=updated_new_joint_pos
            )
        )
        self.motion_manager.set_joint_pos(qpos_dict)

        # Update visualizer if available
        if self.motion_manager.visualizer is not None:
            qpos_array = robot_utils.get_qpos_from_joint_dict(
                self.motion_manager.pin_robot, qpos_dict
            )
            self.motion_manager.visualizer.update_motion_plan(
                qpos_array.reshape(1, -1),
                robot_utils.get_joint_names(self.motion_manager.pin_robot),
                duration=0.4,
            )

        return qpos_dict

    def move_delta_cartesian(
        self, delta_xyz: np.ndarray, delta_rpy: np.ndarray, side: str
    ) -> np.ndarray:
        """Move the specified arm by a delta in Cartesian space.

        Args:
            delta_xyz (np.ndarray): Translation delta in x, y, z (meters).
            delta_rpy (np.ndarray): Rotation delta in roll, pitch, yaw (radians).
            side (str): Which arm to move ("left" or "right").

        Returns:
            np.ndarray: Target joint positions for the specified arm.

        Raises:
            ValueError: If an invalid arm side is specified.
            RuntimeError: If robot or IK solver is not initialized.
        """
        if self.motion_manager.pin_robot is None:
            raise RuntimeError("Robot is not initialized")
        if self.motion_manager.local_ik_solver is None:
            raise RuntimeError("Local IK solver is not initialized")

        current_qpos = self.motion_manager.get_joint_pos()
        assert isinstance(current_qpos, np.ndarray)

        ee_pose = self.motion_manager.fk(
            frame_names=self.motion_manager.target_frames,
            qpos=current_qpos,
            update_robot_state=False,
        )

        left_ee_pose = ee_pose["L_ee"]
        right_ee_pose = ee_pose["R_ee"]

        if side == "left":
            ee_pose_np = left_ee_pose.np.copy()  # type: ignore
            ee_pose_np[:3, :3] = (
                pr.matrix_from_euler(delta_rpy, 0, 1, 2, True) @ ee_pose_np[:3, :3]
            )
            ee_pose_np[:3, 3] += delta_xyz
            target_poses_dict = {
                "L_ee": ee_pose_np,
                "R_ee": right_ee_pose.np,
            }  # type: ignore
        elif side == "right":
            ee_pose_np = right_ee_pose.np.copy()  # type: ignore
            ee_pose_np[:3, 3] += delta_xyz
            ee_pose_np[:3, :3] = (
                pr.matrix_from_euler(delta_rpy, 0, 1, 2, True) @ ee_pose_np[:3, :3]
            )
            target_poses_dict = {
                "L_ee": left_ee_pose.np,
                "R_ee": ee_pose_np,
            }  # type: ignore
        else:
            raise ValueError(f"Invalid arm side: {side}. Must be 'left' or 'right'.")

        qpos_result, is_in_collision, is_within_joint_limits = (
            self.motion_manager.local_ik_solver.solve_ik(target_poses_dict)
        )

        if isinstance(qpos_result, dict):
            qpos_dict = qpos_result
        else:
            joint_names = robot_utils.get_joint_names(self.motion_manager.pin_robot)
            qpos_dict = dict(zip(joint_names, qpos_result.tolist()))

        self.motion_manager.set_joint_pos(qpos_dict)

        # Update visualizer if available
        if self.motion_manager.visualizer is not None:
            qpos_array = robot_utils.get_qpos_from_joint_dict(
                self.motion_manager.pin_robot, qpos_dict
            )
            self.motion_manager.visualizer.update_motion_plan(
                qpos_array.reshape(1, -1),
                robot_utils.get_joint_names(self.motion_manager.pin_robot),
                duration=0.4,
            )

        # Extract joint positions for the requested arm
        arm_prefix = side[0].upper()
        arm_joint_indices = [f"{arm_prefix}_arm_j{i + 1}" for i in range(self.arm_dof)]
        return np.array([qpos_dict[joint_name] for joint_name in arm_joint_indices])
