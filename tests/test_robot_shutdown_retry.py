from types import SimpleNamespace

import pytest

import dexcontrol.robot as robot_module
import dexcontrol.core.shared_node as shared_node_module
import dexcontrol.core.robot_query_interface as query_module
from dexcontrol.core.robot_query_interface import RobotQueryInterface
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
    robot._release_shared_node = lambda: shared_calls.append("node") or True
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


def test_robot_shutdown_retries_last_owner_node_without_double_decrement(monkeypatch):
    class FailingNode:
        def __init__(self):
            self.calls = 0

        def shutdown(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("node busy")

    node = FailingNode()
    monkeypatch.setattr(shared_node_module, "_shared_node", node)
    monkeypatch.setattr(shared_node_module, "_owner_count", 1)
    monkeypatch.setattr(shared_node_module, "_shutdown_pending", False)
    monkeypatch.setattr(robot_module, "cleanup_session", lambda: None)
    robot = Robot.__new__(Robot)
    robot._shutdown_called = False
    robot._node_released = False
    robot._node_owner_released = False
    robot._components = []
    robot.sensors = None

    with pytest.raises(RuntimeError, match="shared node cleanup failed: node busy"):
        robot.shutdown()

    assert shared_node_module._owner_count == 0
    assert shared_node_module._shutdown_pending is True
    assert robot._node_owner_released is True
    assert robot._node_released is False

    robot.shutdown()

    assert node.calls == 2
    assert shared_node_module._owner_count == 0
    assert shared_node_module._shared_node is None
    assert shared_node_module._shutdown_pending is False
    assert robot._node_released is True
    assert robot.is_shutdown() is True


def test_non_final_owner_does_not_cleanup_process_global_session(monkeypatch):
    class Node:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    node = Node()
    monkeypatch.setattr(shared_node_module, "_shared_node", node)
    monkeypatch.setattr(shared_node_module, "_owner_count", 2)
    monkeypatch.setattr(shared_node_module, "_shutdown_pending", False)
    session_calls = []
    monkeypatch.setattr(
        robot_module, "cleanup_session", lambda: session_calls.append("session")
    )

    robots = []
    for _ in range(2):
        robot = Robot.__new__(Robot)
        robot._shutdown_called = False
        robot._node_released = False
        robot._node_owner_released = False
        robot._node_session_cleanup_required = False
        robot._components = []
        robot.sensors = None
        robots.append(robot)

    robots[0].shutdown()
    assert shared_node_module._owner_count == 1
    assert node.shutdown_calls == 0
    assert session_calls == []

    robots[1].shutdown()
    assert shared_node_module._owner_count == 0
    assert node.shutdown_calls == 1
    assert session_calls == ["session"]


def test_query_constructor_releases_owner_when_client_creation_fails(monkeypatch):
    class Node:
        def create_service_client(self, **_kwargs):
            raise RuntimeError("client setup failed")

    release_calls = []
    monkeypatch.setattr(query_module, "acquire_shared_node", lambda: Node())
    monkeypatch.setattr(
        shared_node_module,
        "shutdown_shared_node",
        lambda *, release_owner=True: release_calls.append(release_owner) or False,
    )
    monkeypatch.setattr(query_module, "cleanup_session", lambda: None)
    config = SimpleNamespace(
        querables={
            "hand_info": "hand",
            "soc_ntp": "ntp",
            "version_info": "version",
            "status_info": "status",
            "reboot": "reboot",
            "clear_error": "clear",
        }
    )

    with pytest.raises(RuntimeError, match="client setup failed"):
        RobotQueryInterface(config)

    assert release_calls == [True]
