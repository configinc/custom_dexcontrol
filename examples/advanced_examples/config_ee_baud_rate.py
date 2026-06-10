# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""End-effector baud rate configuration.

This script provides commands to query and set the RS485 baud rate
for the end-effector communication on the robot arms.

Usage:
    python config_ee_baud_rate.py get                          # Read baud rate for both arms
    python config_ee_baud_rate.py get --side right             # Read baud rate for right arm
    python config_ee_baud_rate.py set --baud-rate 921600       # Set left arm baud rate
    python config_ee_baud_rate.py set --side both --baud-rate 3000000  # Set both arms
"""

from typing import Literal

import tyro
from loguru import logger

from dexcontrol.robot import Robot


def get(side: Literal["left", "right", "both"] = "both") -> None:
    """Get current end-effector RS485 baud rate.

    Args:
        side: Which arm to query ('left', 'right', or 'both').
    """
    with Robot() as bot:
        if side in ("left", "both"):
            result = bot.left_arm.get_ee_baud_rate()
            if result.get("success"):
                logger.info(f"Left arm:  {result.get('baud_rate')}")
            else:
                logger.error(f"Left arm:  Error - {result.get('message')}")

        if side in ("right", "both"):
            result = bot.right_arm.get_ee_baud_rate()
            if result.get("success"):
                logger.info(f"Right arm: {result.get('baud_rate')}")
            else:
                logger.error(f"Right arm: Error - {result.get('message')}")


def set_baud_rate(
    side: Literal["left", "right", "both"] = "left",
    baud_rate: int = 115200,
) -> None:
    """Set end-effector RS485 baud rate.

    Args:
        side: Which arm to configure ('left', 'right', or 'both').
        baud_rate: Baud rate to set. Common values: 115200, 460800, 921600,
            1000000, 3000000.
    """
    with Robot() as bot:
        if side in ("left", "both"):
            result = bot.left_arm.set_ee_baud_rate(baud_rate)
            if not result.get("success"):
                logger.error(f"Left arm:  Failed - {result.get('message')}")

        if side in ("right", "both"):
            result = bot.right_arm.set_ee_baud_rate(baud_rate)
            if not result.get("success"):
                logger.error(f"Right arm: Failed - {result.get('message')}")

        logger.info("Verifying new baud rate...")
        if side in ("left", "both"):
            result = bot.left_arm.get_ee_baud_rate()
            if result.get("success"):
                logger.info(f"Left arm:  {result.get('baud_rate')}")
            else:
                logger.error(f"Left arm:  Error - {result.get('message')}")

        if side in ("right", "both"):
            result = bot.right_arm.get_ee_baud_rate()
            if result.get("success"):
                logger.info(f"Right arm: {result.get('baud_rate')}")
            else:
                logger.error(f"Right arm: Error - {result.get('message')}")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"get": get, "set": set_baud_rate})
