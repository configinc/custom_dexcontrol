# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Check whether 'cur' and 'torque' fields are populated from the firmware.

Reads raw state from each arm and reports which fields are present and
what values they contain.
"""

import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot


def _check_field(component_name: str, state: dict, field: str, unit: str) -> None:
    if field not in state:
        logger.warning(f"[{component_name}] '{field}' field is MISSING in state.")
        return

    values = state[field]
    if not values:
        logger.warning(f"[{component_name}] '{field}' field is EMPTY.")
        return

    arr = np.array(values, dtype=np.float32)
    logger.info(
        f"[{component_name}] '{field}' OK — "
        f"shape={arr.shape}, values={np.round(arr, 4)} {unit}"
    )


def main() -> None:
    """Read arm states and check cur/torque field availability."""
    with Robot() as bot:
        arms = {"left_arm": bot.left_arm, "right_arm": bot.right_arm}

        for name, arm in arms.items():
            state = arm.get_state()
            logger.info(f"[{name}] available state keys: {list(state.keys())}")
            _check_field(name, state, "cur", "A")
            _check_field(name, state, "torque", "N·m")


if __name__ == "__main__":
    tyro.cli(main)
