from types import SimpleNamespace

import pytest

import dexcontrol.robot as robot_module
from dexcontrol.robot import Robot


class _Component:
    def __init__(self):
        self.stop_calls = 0
        self.shutdown_calls = 0

    def stop(self):
        self.stop_calls += 1

    def shutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_calls == 1:
            raise RuntimeError("component busy")


def test_robot_shutdown_retries_incomplete_component_before_shared_transport(monkeypatch):
    robot = Robot.__new__(Robot)
    robot._shutdown_called = False
    component = _Component()
    robot._components = [component]
    robot.sensors = None
    shared_calls = []
    robot._release_shared_node = lambda: shared_calls.append("node")
    monkeypatch.setattr(
        robot_module, "cleanup_session", lambda: shared_calls.append("session")
    )

    with pytest.raises(RuntimeError, match="component busy"):
        robot.shutdown()

    assert robot.is_shutdown() is False
    assert component.stop_calls == 1
    assert component.shutdown_calls == 1
    assert shared_calls == []

    robot.shutdown()

    assert robot.is_shutdown() is True
    assert component.stop_calls == 1
    assert component.shutdown_calls == 2
    assert shared_calls == ["node", "session"]
