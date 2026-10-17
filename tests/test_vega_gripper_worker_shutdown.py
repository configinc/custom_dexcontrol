import sys
import threading
from queue import Queue
from types import ModuleType

import numpy as np
import pytest

# Keep this worker test independent of the optional DualSense/hidapi stack.
_base_arm_teleop = ModuleType("base_arm_teleop")
_base_arm_teleop.BaseIKController = object
sys.modules.setdefault("base_arm_teleop", _base_arm_teleop)

from dexcontrol.core.robotenv_vega.server import VegaRobot  # noqa: E402


class _BlockingHand:
    def __init__(self, shutdown_failures=0):
        self.command_started = threading.Event()
        self.release_command = threading.Event()
        self.shutdown_calls = 0
        self.shutdown_failures = shutdown_failures

    def open_hand(self, *, wait_time):
        del wait_time
        self.command_started.set()
        assert self.release_command.wait(timeout=2.0)

    def shutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_calls <= self.shutdown_failures:
            raise RuntimeError("hand busy")


class _Transport:
    def __init__(self, failures=0):
        self.shutdown_calls = 0
        self.failures = failures

    def shutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_calls <= self.failures:
            raise RuntimeError("transport busy")


def _robot_with_blocking_gripper():
    robot = VegaRobot.__new__(VegaRobot)
    robot.arm_side = "left"
    robot.gripper_type = "robotiq"
    robot.hand = _BlockingHand()
    robot.robot = _Transport()
    robot._vel_log_file = None
    robot._gripper_worker = None
    robot._gripper_command_queue = Queue(maxsize=1)
    robot._gripper_stop_event = threading.Event()
    robot._gripper_poll_interval_s = 0.01
    robot._gripper_io_lock = threading.Lock()
    robot._gripper_state_lock = threading.Lock()
    robot._gripper_open_pos = np.array([0.0])
    robot._gripper_close_pos = np.array([1.0])
    robot._prev_gripper_command_successful = True
    robot._hand_shutdown_completed = False
    robot._robot_shutdown_completed = False
    return robot


def test_close_keeps_live_gripper_worker_and_refuses_transport_teardown():
    robot = _robot_with_blocking_gripper()
    worker = threading.Thread(
        target=robot._gripper_worker_loop,
        name="blocking-gripper-test",
        daemon=True,
    )
    robot._gripper_worker = worker
    worker.start()
    robot._gripper_command_queue.put(0.0)
    assert robot.hand.command_started.wait(timeout=1.0)

    with pytest.raises(RuntimeError, match="gripper worker did not stop"):
        robot.close()

    assert robot._gripper_worker is worker
    assert worker.is_alive()
    assert robot.hand.shutdown_calls == 0
    assert robot.robot.shutdown_calls == 0

    robot.hand.release_command.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()

    robot.close()
    assert robot._gripper_worker is None
    assert robot._gripper_command_queue is None
    assert robot.hand.shutdown_calls == 1
    assert robot.robot.shutdown_calls == 1


def test_close_reports_transport_failure_and_retries_only_incomplete_teardown():
    robot = _robot_with_blocking_gripper()
    robot.robot = _Transport(failures=1)

    with pytest.raises(RuntimeError, match="robot shutdown failed: transport busy"):
        robot.close()

    assert robot.hand.shutdown_calls == 1
    assert robot._hand_shutdown_completed is True
    assert robot.robot.shutdown_calls == 1
    assert robot._robot_shutdown_completed is False

    robot.close()

    assert robot.hand.shutdown_calls == 1
    assert robot.robot.shutdown_calls == 2
    assert robot._robot_shutdown_completed is True


def test_base_transport_waits_until_required_hand_teardown_succeeds():
    robot = _robot_with_blocking_gripper()
    robot.hand = _BlockingHand(shutdown_failures=1)

    with pytest.raises(RuntimeError, match="hand shutdown failed: hand busy"):
        robot.close()

    assert robot.hand.shutdown_calls == 1
    assert robot.robot.shutdown_calls == 0
    assert robot._hand_shutdown_completed is False
    assert robot._robot_shutdown_completed is False

    robot.close()

    assert robot.hand.shutdown_calls == 2
    assert robot.robot.shutdown_calls == 1
    assert robot._hand_shutdown_completed is True
    assert robot._robot_shutdown_completed is True
