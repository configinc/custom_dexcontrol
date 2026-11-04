"""Connection, explicit activation, and stop behavior using a Modbus fake."""

import threading
import time

import pytest

from dexcontrol.core import robotiq_gripper as module


class Driver:
    def __init__(self, **kwargs):
        self.ready = False
        self.status_ok = True
        self.commands = []
        self.pending = None

    def getStatus(self):
        return self.status_ok

    def get_pos(self):
        return 0.02

    def is_ready(self):
        return self.ready

    def goto(self, **kwargs):
        self.pending = ("move", kwargs["pos"])

    def stop(self):
        self.pending = ("stop",)

    def deactivate_emergency_release(self):
        pass

    def deactivate_gripper(self):
        self.pending = ("deactivate",)

    def sendCommand(self):
        self.commands.append(self.pending)

    def shutdown(self):
        pass

    def full_init(self, **kwargs):
        raise AssertionError("Unexpected startup motion")


@pytest.fixture
def gripper(monkeypatch):
    monkeypatch.setattr(module, "Robotiq2FingerGripper", Driver)
    hand = module.RobotiqGripper(side="left", initialize=False)
    yield hand, hand._gripper
    hand.shutdown()


def wait_for(check):
    until = time.monotonic() + 1
    while not check():
        assert time.monotonic() < until
        time.sleep(0.005)


def test_connect_and_stop_do_not_activate_or_open_gripper(gripper):
    hand, driver = gripper
    hand.stop_motion()
    assert driver.commands == []
    assert not driver.ready
    assert hand.get_joint_pos()[0] == 0.02


def test_stop_discards_pending_travel_without_opening(gripper):
    hand, driver = gripper
    driver.ready = True
    hand.set_joint_pos([0.03])
    wait_for(lambda: bool(driver.commands))
    hand.set_joint_pos([0.07])
    hand.stop_motion()
    count = len(driver.commands)
    time.sleep(0.06)
    assert driver.commands[-1] == ("stop",)
    assert len(driver.commands) == count


def test_io_failure_is_reported_until_explicit_reset(gripper):
    hand, driver = gripper
    driver.status_ok = False
    wait_for(lambda: hand._worker_error is not None)
    driver.status_ok = True
    with pytest.raises(RuntimeError, match="Robotiq I/O failed"):
        hand.check_error()
    hand.reset_error()
    hand.check_error()


def test_cancelled_activation_does_not_send_commands(gripper):
    hand, driver = gripper
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        hand.initialize(cancelled, lambda operation: operation())
    assert driver.commands == []
