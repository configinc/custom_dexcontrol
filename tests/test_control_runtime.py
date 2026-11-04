"""Managed lifecycle and checked output without robot connections."""

import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dexcontrol.core.control.config import load_config
from dexcontrol.core.control.fsm import ControlConflict
from dexcontrol.core.control.output import OutputGate
from dexcontrol.core.robotenv_vega.control_node import VegaControlNode


class Device:
    def __init__(self, name):
        self.joint_name = [name]
        self.joint_pos_limit = np.array([[-1.0, 1.0]])
        self.sent = []
        self.q = np.zeros(1)
        self.dq = np.zeros(1)

    def get_joint_pos(self):
        return self.q.copy()

    def get_joint_vel(self):
        return self.dq.copy()

    def bind_command_sink(self, sink):
        self.sink = sink

    def publish_staged_command(self, command):
        self.sent.append(command)

    def command(self, q):
        self.sink({"pos": np.array([q])})


class Hardware:
    control_hz = 200

    def __init__(self):
        self.gate = OutputGate()
        self.left, self.right = Device("left_j1"), Device("right_j1")
        self.gate.attach("left", self.left)
        self.gate.attach("right", self.right)
        self.targets = [0.0, 0.0]
        self.prepared = 0
        self.cancel_seen = threading.Event()
        config = load_config()
        rules = dict(config.safety)
        for name in ("self_collision", "static_stability", "arm_torque"):
            rules[name] = replace(rules[name], enabled=False)
        self.config = replace(config, safety=rules)

    def hold(self):
        self.left.command(self.left.q[0])
        self.right.command(self.right.q[0])

    maintain_hold = hold

    def limits(self):
        return {"command_velocity": [0.15, 0.15, 0.3, 0.02, 0.02, 0.05]}

    def snapshot(self):
        return {}

    def stationary(self):
        return True

    def validate_state(self):
        pass

    def reset_fault(self):
        pass

    def prepare_arms(self):
        self.prepared += 1

    def start_arms(self):
        pass

    def capture_head_reference(self):
        pass

    def head_tick(self):
        return False

    def arm_tick(self):
        self.left.command(self.targets[0])
        self.right.command(self.targets[1])

    calibration_tick = arm_tick

    def action(self, name):
        if name != "initial_pose":
            raise ValueError("Unknown action")

        def run(cancelled):
            self.left.command(0.2)
            while not cancelled.wait(0.01):
                pass
            self.cancel_seen.set()
            # A late producer cannot overwrite the stop command.
            self.right.command(0.8)

        return run


@pytest.fixture
def runtime():
    hardware = Hardware()
    control = VegaControlNode(hardware, hardware.config)
    yield control, hardware
    control.stop()
    control.actions.join()


def arm_mode(control):
    control.start()
    control.select("prepare_arms")
    control.enable_arms()


def test_start_is_explicit_and_does_not_run_initial_pose(runtime):
    node, hardware = runtime
    assert node.state()["control_state"] == "idle"
    with pytest.raises(ControlConflict):
        node.select("body")
    with pytest.raises(ControlConflict):
        node.start_action("initial_pose")
    node.start()
    assert node.state()["action"]["state"] == "idle"
    assert all(command["pos"][0] == 0 for command in hardware.left.sent)


def test_one_bad_arm_blocks_complete_normal_batch(runtime):
    node, hardware = runtime
    arm_mode(node)
    hardware.targets = [0.3, float("nan")]
    before = len(hardware.left.sent)
    with pytest.raises(ValueError):
        node.tick()
    assert node.state()["control_state"] == "fault"
    assert node.state()["fault_reason"] == "execution_error"
    assert all(c["pos"][0] == 0 for c in hardware.left.sent[before:])
    with pytest.raises(ControlConflict):
        node.select("body")
    node.stop()
    assert node.state()["control_state"] == "fault"
    assert "Nonfinite command for right" in node.state()["error"]
    node.reset_fault()
    assert node.state()["control_state"] == "idle"
    with pytest.raises(ControlConflict):
        node.accept_arm("left")


def test_cancel_rejects_old_action_commands(runtime):
    node, hardware = runtime
    node.start()
    node.start_action("initial_pose")
    node.stop()
    node.actions.join()
    assert hardware.cancel_seen.is_set()
    assert node.state()["action"]["state"] == "cancelled"
    node.tick()
    assert all(c["pos"][0] != 0.8 for c in hardware.right.sent)


def test_unknown_action_leaves_current_mode_unchanged(runtime):
    node, _ = runtime
    arm_mode(node)
    with pytest.raises(ValueError):
        node.start_action("not_configured")
    assert node.mode == "arms"


def test_fault_reset_checks_actual_stopped_state(runtime):
    node, hardware = runtime
    node.fault("execution_error", "failed")
    hardware.stationary = lambda: False
    with pytest.raises(ControlConflict):
        node.reset_fault()
    assert node.state()["control_state"] == "fault"


def test_action_failure_is_reported_and_latched(runtime):
    node, _ = runtime
    node.start()

    def fail(cancelled):
        raise TimeoutError("Target not reached")

    node.start_action("initial_pose", fail)
    node.actions.join()
    node.tick()
    assert node.state()["fault_reason"] == "action_error"
    assert node.state()["action"]["error"] == "Target not reached"


def test_unowned_sdk_output_is_rejected_and_gripper_effect_is_not_sent(runtime):
    node, hardware = runtime
    with pytest.raises(ControlConflict):
        hardware.left.command(0.5)
    arm_mode(node)
    effects = []
    with node.gate.producing(node.normal_owner):
        hardware.right.sink({"pos": np.array([0.1, 0.2])})
        node.gate.defer("gripper", lambda: effects.append("sent"))
    with pytest.raises(ValueError):
        node.gate.commit(node._check_commands)
    assert effects == []
    assert node.gate.pending == {}


def test_halt_preserves_start_authorization_but_stop_revokes_it(runtime):
    node, _ = runtime
    arm_mode(node)
    node.halt()
    assert node.state()["control_state"] == "running"
    node.select("prepare_arms")
    node.enable_arms()
    node.stop()
    assert node.state()["control_state"] == "idle"
    with pytest.raises(ControlConflict):
        node.select("prepare_arms")


def test_failed_state_read_cancels_pending_output_and_holds(runtime):
    node, hardware = runtime
    arm_mode(node)
    hardware.targets = [0.3, 0.4]
    hardware.validate_state = lambda: (_ for _ in ()).throw(
        RuntimeError("State unavailable")
    )
    before = len(hardware.left.sent)
    with pytest.raises(RuntimeError, match="State unavailable"):
        node.tick()
    assert node.state()["fault_reason"] == "execution_error"
    assert all(c["pos"][0] == 0 for c in hardware.left.sent[before:])


def test_timeout_latches_fault_even_after_fresh_input(runtime):
    node, _ = runtime
    node.start()
    node.select("body")
    node.clock = lambda: node.deadline + 1
    node.tick()
    assert node.state()["fault_reason"] == "input_timeout"
    with pytest.raises(ControlConflict):
        node.set_command([0, 0, 0])


def test_previous_calibration_release_cannot_cancel_new_action(runtime):
    node, _ = runtime
    arm_mode(node)
    service = SimpleNamespace(
        arm_side="left",
        _cancel_force_calibration_motion=lambda: participant.finish(cancelled=True),
    )
    participant = node.claim_calibration(service)
    node.halt()
    node.start_action("initial_pose")
    participant.finish()
    assert node.mode == "action"
    assert node.state()["action"]["name"] == "initial_pose"


def test_lifecycle_and_actions_over_grpc(runtime):
    from concurrent.futures import ThreadPoolExecutor

    import grpc
    from test_body_grpc import Hardware as WireHardware

    from dexcontrol.core.robotenv_vega.body_grpc import BodyService
    from proto import vega_body_pb2 as pb
    from proto import vega_body_pb2_grpc as rpc

    node, hardware = runtime
    wire = WireHardware()
    hardware.limits = wire.limits
    hardware.snapshot = wire.snapshot
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    rpc.add_BodyControlServicer_to_server(BodyService(node), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = rpc.BodyControlStub(channel)
            assert stub.GetState(pb.Empty(), timeout=1).control_state == "idle"
            with pytest.raises(grpc.RpcError) as exc:
                stub.StartAction(pb.ActionRequest(name="initial_pose"), timeout=1)
            assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION
            stub.Start(pb.Empty(), timeout=1)
            running = stub.StartAction(pb.ActionRequest(name="initial_pose"), timeout=1)
            assert running.mode == pb.ACTION and running.action.state == "running"
            assert stub.Halt(pb.Empty(), timeout=1).control_state == "running"
            node.actions.join()
            node.fault("execution_error", "Lost feedback")
            assert stub.Stop(pb.Empty(), timeout=1).control_state == "fault"
            assert stub.ResetFault(pb.Empty(), timeout=1).control_state == "idle"
            with pytest.raises(grpc.RpcError):
                stub.EnableArms(pb.Empty(), timeout=1)
    finally:
        server.stop(0).wait()
