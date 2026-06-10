# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example: tracked arm control using set_joint_target().

Demonstrates using the set_joint_target(tracked=True) API to send target positions
and wait for the motion plugin to complete each motion before moving on.

This is the recommended pattern for scripted motions: send a target, wait for
completion, then send the next target. The motion plugin handles trajectory
smoothing, gravity compensation, and convergence detection.

Comparison with set_joint_pos():
    - set_joint_pos():     Low-level direct commands to control/ topic. You must
                           send smooth, continuous commands at 100-500 Hz.
    - set_joint_target():    Goal-oriented commands to the motion plugin. Send
                           arbitrary targets at any rate. The plugin does the rest.

Requirements:
    - robot-server running with the motion plugin loaded
    - ROBOT_NAME and ZENOH_CONFIG environment variables set

Usage:
    python move_arm_target.py
    python move_arm_target.py --side left
    python move_arm_target.py --step-size 0.4
"""

from typing import Literal

import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main(
    side: Literal["right", "left"] = "right",
    step_size: float = 0.2,
) -> None:
    """Move arm joints one by one using tracked set_joint_target().

    Each joint is moved to +step_size then back to zero. The motion plugin
    smooths the trajectory and adds gravity compensation. We wait for each
    motion to complete before sending the next.

    Args:
        side: Which arm to move ("right" or "left").
        step_size: Magnitude of joint movement in radians.
    """
    logger.warning("Warning: Be ready to press e-stop if needed!")
    logger.warning("This example requires the motion plugin to be running.")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    robot = Robot()
    arm = robot.left_arm if side == "left" else robot.right_arm

    logger.info(f"Using set_joint_target(tracked=True) on {side} arm")

    try:
        # Move to zero position and wait for completion
        logger.info("Moving to zero position")
        handle = arm.set_joint_target(np.zeros(7), tracked=True)
        result = handle.wait(timeout=10.0)
        logger.info(f"Zero position reached (state={result})")

        # Sequentially move each joint
        for joint_idx in range(7):
            logger.info(f"Moving joint {joint_idx} to {step_size:.2f} rad")

            target_pos = np.zeros(7)
            target_pos[joint_idx] = step_size
            handle = arm.set_joint_target(target_pos, tracked=True)
            result = handle.wait(timeout=10.0)
            logger.info(f"Joint {joint_idx} reached target (state={result})")

            logger.info(f"Moving joint {joint_idx} back to zero")
            handle = arm.set_joint_target(np.zeros(7), tracked=True)
            result = handle.wait(timeout=10.0)
            logger.info(f"Joint {joint_idx} returned to zero (state={result})")

        logger.info("Movement sequence completed")
    except TimeoutError as e:
        logger.error(f"Motion timed out: {e}")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
    finally:
        logger.info("Shutting down")
        robot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
