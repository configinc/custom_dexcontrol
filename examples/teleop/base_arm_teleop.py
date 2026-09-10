# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Base teleop class with shared functionality for different teleop modes.

This module provides base classes for robot teleoperation with inverse kinematics
control support for multiple teleoperation modes (joint space, Cartesian space).
"""

import os
import threading

from dexcomm import RateLimiter
from loguru import logger

from dexcontrol.apps.dualsense_teleop_base import DualSenseTeleopBase
from dexcontrol.core.vega.ik_controller import BaseIKController as BaseIKController


def is_running_remote() -> bool:
    """Check if the script is running in a remote environment.

    Returns:
        bool: True if running in a remote environment (no display), False otherwise.
    """
    return "DISPLAY" not in os.environ


class BaseArmTeleopNode(DualSenseTeleopBase):
    """Base teleop node with shared functionality for arm control.

    Attributes:
        side (str): Currently active arm ("left" or "right").
        arms (dict): Dictionary of robot arm interfaces.
        arm_target_qpos (dict): Target joint positions for each arm.
        arm_motion_lock (threading.Lock): Thread lock for synchronizing arm updates.
        arm_control_thread (threading.Thread): Thread for running the arm control loop.
    """

    def __init__(
        self,
        control_hz: int = 200,
        button_update_hz: int = 20,
        device_index: int = 0,
    ):
        """Initialize the base teleop node.

        Args:
            control_hz (int): Control loop frequency in Hz. Defaults to 200.
            button_update_hz (int): Button update frequency in Hz. Defaults to 20.
            device_index (int): Index of the DualSense controller device. Defaults to 0.
        """
        super().__init__(control_hz, button_update_hz, device_index)
        self.side = "left"
        self.arm_motion_lock = threading.Lock()

        # Real robot interface
        self.arms = {"left": self.bot.left_arm, "right": self.bot.right_arm}

        # Target joint positions for continuous motion
        self.arm_target_qpos = {
            "left": self.arms["left"].get_joint_pos(),
            "right": self.arms["right"].get_joint_pos(),
        }

        # Initialize the arm control thread
        self.arm_control_thread = threading.Thread(target=self.arm_control_loop)
        self.arm_control_thread.daemon = True

        # Set initial controller feedback
        self.update_controller_lightbar()

    def update_controller_lightbar(self):
        """Update the controller lightbar color based on active arm."""
        if self.side == "left":
            # Magenta for left arm
            self.dualsense.lightbar.set_color(255, 0, 255)
        else:
            # Cyan for right arm
            self.dualsense.lightbar.set_color(0, 255, 255)

    def toggle_arm(self):
        """Toggle between left and right arm control."""
        self.side = "left" if self.side == "right" else "right"
        with self.arm_motion_lock:
            self.arm_target_qpos[self.side] = self.arms[self.side].get_joint_pos()
        logger.info(f"Active arm: {self.side}")
        self.update_controller_lightbar()

    def arm_control_loop(self):
        """Control loop for smooth arm motion.

        This method runs in a separate thread to continuously apply smoothed
        motion to the robot arms based on the current target positions.
        """

        limiter = RateLimiter(self.control_hz)
        while self.is_running:
            if self.safe_pressed:
                with self.arm_motion_lock:
                    # Snapshot `side` under the lock so toggle_arm() can't
                    # pair this arm with the other arm's target between
                    # the lookup and the publish.
                    side = self.side
                    target_qpos = self.arm_target_qpos[side].copy()
                self.arms[side].set_joint_pos(target_qpos)
            limiter.sleep()

    def update_motion(self) -> None:
        """Update the robot's motion based on current controller state.

        This method must be implemented by subclasses to handle specific
        motion control logic for different teleop modes.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def stop_all_motion(self) -> None:
        """Stop all ongoing robot motion and reset motion commands."""
        try:
            # Reset smoothers and target positions
            for side, arm in self.arms.items():
                current_pos = arm.get_joint_pos()
                with self.arm_motion_lock:
                    self.arm_target_qpos[side] = current_pos
            logger.info("All arm motion stopped")
        except Exception as e:
            logger.error(f"Error stopping motion: {e}")

    def _cleanup(self):
        """Clean up resources when shutting down."""
        self.dualsense.lightbar.set_color_white()
        self.dualsense.deactivate()
        logger.info("Exiting teleop node")
