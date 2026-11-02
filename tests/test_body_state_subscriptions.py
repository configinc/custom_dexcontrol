"""Managed server state stays available without observation polling."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "dexcontrol"))
from core import subscription_policy
from core.robotenv_vega.body_hardware import keep_state_subscriptions_active
from core.subscription_policy import SubscriptionPolicyManager, SubscriptionPolicyMixin


class StateStream:
    def __init__(self):
        self.paused = True
        self.value = None

    def pause(self):
        self.paused = True
        self.value = None

    def resume(self):
        self.paused = False

    def is_paused(self):
        return self.paused

    def is_active(self, window=1.):
        return not self.paused and self.value is not None

    def get_latest(self):
        return self.value

    def publish(self, value):
        if not self.paused:
            self.value = value


class StateComponent(SubscriptionPolicyMixin):
    def __init__(self, name, available=True):
        self.stream = StateStream()
        self.available = available
        self._policy_manager = SubscriptionPolicyManager(self.stream, name)
        self._subcomponents = {}

    def wait_for_active(self):
        if self.available:
            self.stream.publish(1)
        return self._policy_manager.is_active()

    def get_state(self):
        value = self._policy_manager.get_latest_managed()
        if value is None:
            raise RuntimeError("No state data")
        return value


def state_robot():
    states = {name: StateComponent(name) for name in (
        "torso", "head", "chassis_steer", "chassis_drive",
        "left_arm", "right_arm", "left_wrench", "right_wrench",
    )}
    robot = SimpleNamespace(**states)
    robot.chassis = SimpleNamespace(chassis_steer=states["chassis_steer"],
                                    chassis_drive=states["chassis_drive"])
    for side in ("left", "right"):
        states[f"{side}_arm"].wrench_sensor = states[f"{side}_wrench"]
    return robot, states


def test_both_arm_readers_receive_current_state_after_idle_periods(monkeypatch):
    now = [0.]
    monkeypatch.setattr(subscription_policy, "time", SimpleNamespace(monotonic=lambda: now[0]))
    robot, states = state_robot()
    keep_state_subscriptions_active(robot)

    def observation():
        return [component.get_state() for component in states.values()]

    with ThreadPoolExecutor(max_workers=2) as readers:
        for timestamp, value in ((6., 2), (60., 3), (600., 4)):
            now[0] = timestamp
            for component in states.values():
                component._policy_manager.check_idle()
                component.stream.publish(value)
            left, right = readers.submit(observation), readers.submit(observation)
            assert left.result(timeout=1) == [value] * len(states)
            assert right.result(timeout=1) == [value] * len(states)


@pytest.mark.parametrize("missing", ["torso", "head", "left_wrench"])
def test_missing_startup_state_identifies_the_component(missing):
    robot, states = state_robot()
    states[missing].available = False
    with pytest.raises(RuntimeError, match=f"No state data received for {missing} during startup"):
        keep_state_subscriptions_active(robot)


def test_optional_sensors_and_unrelated_subscriptions_keep_their_contract():
    robot, states = state_robot()
    robot.head = None
    robot.left_arm.wrench_sensor = None
    camera = StateComponent("camera")
    temperature = StateComponent("temperature")
    robot.sensors = SimpleNamespace(camera=camera)
    robot.left_arm._subcomponents["temperature"] = temperature

    keep_state_subscriptions_active(robot)

    assert states["torso"].get_state() == 1
    assert states["right_wrench"].get_state() == 1
    assert camera.stream.is_paused()
    assert temperature.stream.is_paused()
