"""Exercise command clearing without connecting to robot hardware."""

import threading
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from dexcontrol.core.vega.robot import VegaRobot
from dexcontrol.utils.trajectory_interpolator import TrajectoryInterpolator


@pytest.mark.parametrize("gripper_stuck", [False, True])
def test_pause_discards_old_commands_and_holds_arm_even_if_gripper_stalls(
    gripper_stuck,
):
    robot = VegaRobot.__new__(VegaRobot)
    held = []
    robot.arm = SimpleNamespace(
        get_joint_pos=lambda: np.arange(7, dtype=float),
        set_joint_pos=lambda positions, wait_time: held.append(positions),
    )
    robot._gripper_command_queue = Queue(maxsize=1)
    robot._gripper_command_queue.put(0.9)
    robot._gripper_stop_event = threading.Event()
    robot._gripper_worker = (
        SimpleNamespace(is_alive=lambda: True, join=lambda timeout: None)
        if gripper_stuck
        else None
    )
    robot._interp_lock = threading.Lock()
    robot._interpolator = TrajectoryInterpolator(method="linear")
    robot._interpolator.add_point(1.0, np.full(7, 8.0))
    robot._latest_target_joint_pos = np.full(7, 8.0)
    robot._output_filter = None

    with (
        pytest.raises(RuntimeError, match="Gripper worker did not stop")
        if gripper_stuck
        else nullcontext()
    ):
        robot.pause_commands()

    assert robot._gripper_command_queue is None or robot._gripper_command_queue.empty()
    assert not robot.execute_interpolated_tick()
    assert robot._interpolator.interpolate(2.0) == (None, None)
    np.testing.assert_array_equal(held[0], np.arange(7, dtype=float))
