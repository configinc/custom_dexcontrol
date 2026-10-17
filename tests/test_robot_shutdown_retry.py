from types import SimpleNamespace
import threading

import pytest

import dexcontrol.robot as robot_module
import dexcontrol.core.shared_node as shared_node_module
import dexcontrol.core.robot_query_interface as query_module
from dexcontrol.core.robot_query_interface import RobotQueryInterface
from dexcontrol.robot import Robot
from dexcontrol.sensors.manager import Sensors
from dexcontrol.core.chassis import Chassis
from dexcontrol.core.arm import Arm
from dexcontrol.core.hand import HandF5D6V2
from dexcontrol.core.head import Head
from dexcontrol.core.misc import Battery, EStop, Heartbeat
from dexcontrol.core.torso import Torso
from dexcontrol.sensors.imu.zed_imu import ZedIMUSensor
from dexcontrol.sensors.lidar.lidar_3d import Lidar3DSensor
from dexcontrol.sensors.lidar.rplidar import RPLidarSensor
from dexcontrol.sensors.ultrasonic import UltrasonicSensor


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
        shared_node_module,
        "cleanup_session",
        lambda: shared_calls.append("session"),
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
    assert shared_calls == ["node"]


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
    monkeypatch.setattr(shared_node_module, "_node_shutdown_completed", False)
    monkeypatch.setattr(shared_node_module, "cleanup_session", lambda: None)
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
    monkeypatch.setattr(shared_node_module, "_node_shutdown_completed", False)
    session_calls = []
    monkeypatch.setattr(
        shared_node_module,
        "cleanup_session",
        lambda: session_calls.append("session"),
    )

    robots = []
    for _ in range(2):
        robot = Robot.__new__(Robot)
        robot._shutdown_called = False
        robot._node_released = False
        robot._node_owner_released = False
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


def test_final_cleanup_blocks_new_owner_until_session_transaction_finishes(monkeypatch):
    class Node:
        def shutdown(self):
            return None

    old_node = Node()
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    acquired = threading.Event()
    monkeypatch.setattr(shared_node_module, "_shared_node", old_node)
    monkeypatch.setattr(shared_node_module, "_owner_count", 1)
    monkeypatch.setattr(shared_node_module, "_shutdown_pending", False)
    monkeypatch.setattr(shared_node_module, "_node_shutdown_completed", False)
    monkeypatch.setattr(shared_node_module, "Node", lambda **_kwargs: Node())

    def cleanup():
        cleanup_entered.set()
        assert release_cleanup.wait(timeout=1.0)

    monkeypatch.setattr(shared_node_module, "cleanup_session", cleanup)
    closing = threading.Thread(target=shared_node_module.shutdown_shared_node)
    closing.start()
    assert cleanup_entered.wait(timeout=1.0)

    acquiring = threading.Thread(
        target=lambda: (shared_node_module.acquire_shared_node(), acquired.set())
    )
    acquiring.start()
    assert acquired.wait(timeout=0.05) is False

    release_cleanup.set()
    closing.join(timeout=1.0)
    acquiring.join(timeout=1.0)
    assert acquired.is_set()


def test_robot_close_then_shutdown_cleans_global_session_once(monkeypatch):
    class Node:
        def shutdown(self):
            return None

    calls = []
    monkeypatch.setattr(shared_node_module, "_shared_node", Node())
    monkeypatch.setattr(shared_node_module, "_owner_count", 1)
    monkeypatch.setattr(shared_node_module, "_shutdown_pending", False)
    monkeypatch.setattr(shared_node_module, "_node_shutdown_completed", False)
    monkeypatch.setattr(
        shared_node_module, "cleanup_session", lambda: calls.append("session")
    )
    robot = Robot.__new__(Robot)
    robot._shutdown_called = False
    robot._node_released = False
    robot._node_owner_released = False
    robot._components = []
    robot.sensors = None

    robot.close()
    robot.shutdown()

    assert calls == ["session"]


def test_concurrent_close_of_one_owner_does_not_release_peer(monkeypatch):
    class Node:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    node = Node()
    monkeypatch.setattr(shared_node_module, "_shared_node", node)
    monkeypatch.setattr(shared_node_module, "_owner_count", 2)
    monkeypatch.setattr(shared_node_module, "_shutdown_pending", False)
    monkeypatch.setattr(shared_node_module, "_node_shutdown_completed", False)
    owner = RobotQueryInterface.__new__(RobotQueryInterface)
    owner._node_released = False
    owner._node_owner_released = False
    owner._node_release_lock = threading.Lock()

    threads = [threading.Thread(target=owner.close) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    assert shared_node_module._owner_count == 1
    assert node.shutdown_calls == 0


def test_partial_component_creation_is_immediately_owned_for_rollback(monkeypatch):
    class Config:
        enabled = True

    class ComponentClass:
        def __init__(self, *, name, robot_info):
            del robot_info
            self.name = name
            if name == "second":
                raise RuntimeError("second failed")

        def stop(self):
            return None

        def shutdown(self):
            return None

    robot = Robot.__new__(Robot)
    robot._shutdown_called = True
    robot._components = []
    robot._initialized_component_ids = set()
    robot._configs = SimpleNamespace(
        components={"first": Config(), "second": Config()}
    )
    robot._robot_info = object()
    robot._hand_types = {}
    robot.query_hand_type = lambda: {}
    monkeypatch.setattr(robot_module, "get_component_config_map", lambda: {Config: ComponentClass})
    monkeypatch.setattr(robot_module, "runtime_override_robot_config", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="second failed"):
        robot._initialize_robot_components()

    assert [component.name for component in robot._components] == ["first", "second"]


def test_partial_component_shutdown_runs_even_when_graceful_stop_fails():
    class PartialComponent:
        def __init__(self):
            self.shutdown_calls = 0

        def stop(self):
            raise AttributeError("later constructor field is missing")

        def shutdown(self):
            self.shutdown_calls += 1

    component = PartialComponent()
    robot = Robot.__new__(Robot)
    robot._shutdown_called = False
    robot._shutdown_lock = threading.RLock()
    robot._components = [component]
    robot._initialized_component_ids = set()
    robot.sensors = None
    robot._node_released = True
    robot._node_owner_released = True
    robot._node_session_cleanup_required = False
    robot._node_release_lock = threading.Lock()

    with pytest.raises(RuntimeError, match="later constructor field is missing"):
        robot.shutdown()

    assert component.shutdown_calls == 1


def test_initialized_component_stop_failure_does_not_release_transport():
    class MovingComponent:
        def __init__(self):
            self.stop_calls = 0
            self.shutdown_calls = 0

        def stop(self):
            self.stop_calls += 1
            raise RuntimeError("motion stop failed")

        def shutdown(self):
            self.shutdown_calls += 1

    component = MovingComponent()
    robot = Robot.__new__(Robot)
    robot._shutdown_called = False
    robot._shutdown_lock = threading.RLock()
    robot._components = [component]
    robot._initialized_component_ids = {id(component)}
    robot.sensors = None
    robot._node_released = True
    robot._node_owner_released = True
    robot._node_session_cleanup_required = False
    robot._node_release_lock = threading.Lock()

    with pytest.raises(RuntimeError, match="motion stop failed"):
        robot.shutdown()

    assert component.stop_calls == 1
    assert component.shutdown_calls == 0
    assert robot._shutdown_called is False


def test_partial_chassis_releases_first_child_when_second_child_init_failed():
    class Child:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    steer = Child()
    chassis = Chassis.__new__(Chassis)
    chassis.chassis_steer = steer
    chassis.chassis_drive = None
    chassis.steer_temperature_sensor = None
    chassis.drive_temperature_sensor = None
    chassis._shutdown_child_ids = set()
    chassis._initialization_complete = False

    chassis.shutdown()

    assert steer.shutdown_calls == 1


@pytest.mark.parametrize(
    "component_class",
    [Arm, Head, Torso, HandF5D6V2, Battery, EStop, Heartbeat],
)
def test_mapped_component_shutdown_is_safe_before_base_init(component_class):
    component_class.__new__(component_class).shutdown()


def test_sensor_constructor_retries_partial_cleanup_before_raising(monkeypatch):
    calls = []

    class Sensor:
        def __init__(self):
            self.attempts = 0

        def shutdown(self):
            self.attempts += 1
            calls.append("shutdown")
            if self.attempts == 1:
                raise RuntimeError("sensor busy")

    sensor = Sensor()

    def create(self, _config, name):
        if name == "second":
            raise RuntimeError("second failed")
        return sensor

    monkeypatch.setattr(Sensors, "_create_sensor", create)
    monkeypatch.setattr("dexcontrol.sensors.manager.time.sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match="second failed"):
        Sensors({"first": object(), "second": object()})

    assert calls == ["shutdown", "shutdown"]


def test_sensor_start_failure_cleans_the_not_yet_published_sensor(monkeypatch):
    class Config:
        enabled = True

    class Sensor:
        instance = None

        def __init__(self, **_kwargs):
            self.shutdown_calls = 0
            Sensor.instance = self

        def start(self):
            raise RuntimeError("start failed")

        def shutdown(self):
            self.shutdown_calls += 1

    monkeypatch.setattr(
        "dexcontrol.core.config.get_sensor_mapping", lambda: {Config: Sensor}
    )

    with pytest.raises(RuntimeError, match="start failed"):
        Sensors({"sensor": Config()})

    assert Sensor.instance.shutdown_calls == 1


def test_sensor_init_failure_cleans_the_partially_constructed_sensor(monkeypatch):
    class Config:
        enabled = True

    class Sensor:
        instance = None

        def __init__(self, **_kwargs):
            self.resource = object()
            self.shutdown_calls = 0
            Sensor.instance = self
            raise RuntimeError("later stream failed")

        def shutdown(self):
            self.shutdown_calls += 1
            self.resource = None

    monkeypatch.setattr(
        "dexcontrol.core.config.get_sensor_mapping", lambda: {Config: Sensor}
    )

    with pytest.raises(RuntimeError, match="later stream failed"):
        Sensors({"sensor": Config()})

    assert Sensor.instance.shutdown_calls == 1
    assert Sensor.instance.resource is None


@pytest.mark.parametrize(
    "sensor_class",
    [UltrasonicSensor, Lidar3DSensor, RPLidarSensor, ZedIMUSensor],
)
def test_sensor_shutdown_is_safe_before_subscriber_creation(sensor_class):
    sensor_class.__new__(sensor_class).shutdown()


def test_concurrent_robot_shutdown_runs_component_teardown_once():
    class Component:
        def __init__(self):
            self.stop_calls = 0
            self.shutdown_calls = 0

        def stop(self):
            self.stop_calls += 1

        def shutdown(self):
            self.shutdown_calls += 1

    component = Component()
    robot = Robot.__new__(Robot)
    robot._shutdown_called = False
    robot._shutdown_lock = threading.RLock()
    robot._components = [component]
    robot.sensors = None
    robot._node_released = True
    robot._node_owner_released = True
    robot._node_session_cleanup_required = False
    robot._node_release_lock = threading.Lock()

    threads = [threading.Thread(target=robot.shutdown) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    assert component.stop_calls == 1
    assert component.shutdown_calls == 1
