"""BodyControl wire contracts using real gRPC and hardware-free control."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

import grpc
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src' / 'dexcontrol'))
from proto import vega_body_pb2 as pb, vega_body_pb2_grpc as rpc
from core.robotenv_vega.body_control import BodyControl
from core.robotenv_vega.body_grpc import BodyService


class Hardware:
    def __init__(self):
        self.velocity = np.zeros(3)
        self.target = [-.28,1.40,.01]
    def hold(self): self.velocity[:] = 0
    def maintain_hold(self): self.hold()
    def stationary(self): return True
    def start_arms(self): pass
    def prepare_arms(self): pass
    def set_shoulder_target(self, target):
        if not np.all(np.isfinite(target)) or np.max(np.abs(target)) > 5:
            raise ValueError("Shoulder target unreachable")
        self.target = list(target)
    def move_body(self, command, dt): self.velocity = command.copy()
    def limits(self):
        return dict(steering=[[-2.35, 2.35]]*2, torso=[[0, 1.570], [0, 3.141], [-1.570, 1.570]],
                    command_velocity=[.15, .15, .3, .02, .02, .05], torso_joint_velocity=[.2]*3,
                    shoulder_frame='arm_center', reference_frame='base', watchdog_seconds=.4)
    def snapshot(self):
        return dict(torso=[.8, 1.6, .8], shoulder=[-.29, 1.39, 0], torso_target=[.81, 1.62, .8], shoulder_target=self.target, steering=[0, 0], wheel_velocity=[0, 0])


@pytest.fixture
def connection():
    hardware = Hardware()
    control = BodyControl(hardware)
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    rpc.add_BodyControlServicer_to_server(BodyService(control), server)
    port = server.add_insecure_port('127.0.0.1:0')
    server.start()
    with grpc.insecure_channel(f'127.0.0.1:{port}') as channel:
        yield rpc.BodyControlStub(channel), control, hardware
    server.stop(0).wait()


def test_body_commands_stop_and_hold_blocks_motion(connection):
    stub, control, hardware = connection
    state = stub.GetState(pb.Empty(), timeout=1)
    assert state.mode == pb.HOLD
    assert state.shoulder.z == 1.39
    assert state.shoulder_target.z == 1.40
    assert list(state.torso_target) == [.81, 1.62, .8]
    assert state.limits.command_velocity.shoulder_vx == .02
    state = stub.SelectMode(pb.SelectModeRequest(mode=pb.BODY), timeout=1)
    command = pb.BodyCommand(velocity=pb.Velocity(base_vx=.1))
    stub.Command(command, timeout=1)
    control.tick()
    assert hardware.velocity[0] == .1
    stopped = stub.Stop(pb.Empty(), timeout=1)
    assert stopped.mode == pb.HOLD
    assert not hardware.velocity.any()
    with pytest.raises(grpc.RpcError) as exc:
        stub.Command(command, timeout=1)
    assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_prepare_requires_enable_and_arm_mode_stays_active(connection):
    stub, control, _ = connection
    ready = stub.SelectMode(pb.SelectModeRequest(mode=pb.ARMS), timeout=1)
    assert ready.mode == pb.ARM_READY
    arms = stub.EnableArms(pb.Empty(), timeout=1)
    assert arms.mode == pb.ARMS
    control.clock = lambda: 1e12
    control.tick()
    assert stub.GetState(pb.Empty(), timeout=1).mode == pb.ARMS


def test_stop_interrupts_a_pending_mode_transition(connection):
    stub, control, hardware = connection
    hardware.stationary = lambda: False
    pending = stub.SelectMode.future(pb.SelectModeRequest(
        mode=pb.BODY), timeout=2)
    until = time.monotonic() + 1
    while control.mode != 'transition' and time.monotonic() < until:
        time.sleep(.005)
    assert control.mode == 'transition'
    stopped = stub.Stop(pb.Empty(), timeout=1)
    assert stopped.mode == pb.HOLD
    with pytest.raises(grpc.RpcError) as exc:
        pending.result(timeout=1)
    assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert control.mode == 'hold'


@pytest.mark.parametrize('velocity', [None, pb.Velocity(base_vx=float('nan')), pb.Velocity(shoulder_vz=.5)])
def test_invalid_velocity_is_rejected(connection, velocity):
    stub, control, _ = connection
    state = stub.SelectMode(pb.SelectModeRequest(mode=pb.BODY), timeout=1)
    request = pb.BodyCommand()
    if velocity is not None:
        request.velocity.CopyFrom(velocity)
    with pytest.raises(grpc.RpcError) as exc:
        stub.Command(request, timeout=1)
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert not control.command.any()


def test_absolute_goal_round_trip_and_rejection(connection):
    stub, control, hardware = connection
    stub.SelectMode(pb.SelectModeRequest(mode=pb.BODY), timeout=1)
    command = pb.BodyCommand(velocity=pb.Velocity(), shoulder_target=pb.ShoulderPose(x=-.2,z=1.3,pitch=.1))
    accepted = stub.Command(command,timeout=1)
    assert accepted.shoulder_target == command.shoulder_target
    for bad in [pb.ShoulderPose(x=10,z=10), pb.ShoulderPose(x=float('nan'))]:
        with pytest.raises(grpc.RpcError) as exc:
            stub.Command(pb.BodyCommand(velocity=pb.Velocity(base_vx=.1),shoulder_target=bad),timeout=1)
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert stub.GetState(pb.Empty(),timeout=1).shoulder_target == command.shoulder_target
        assert not control.command.any()


@pytest.mark.parametrize("clearance", [None, 0.0, -0.001])
def test_collision_diagnostics_preserve_unavailable_and_zero_clearance(connection, clearance):
    stub, control, hardware = connection
    snapshot = hardware.snapshot
    hardware.snapshot = lambda: dict(snapshot(), collision={
        "state": "blocked", "blocked": True, "reason": "Target reduces clearance",
        "clearance_m": clearance, "pair": ["base", "L_robotiq"],
        "result_age_s": None, "check_ms": 1.5,
    })
    state = stub.GetState(pb.Empty(), timeout=1)
    assert state.collision.blocked
    assert state.collision.reason == "Target reduces clearance"
    assert list(state.collision.pair) == ["base", "L_robotiq"]
    assert state.collision.HasField("clearance_m") == (clearance is not None)
    if clearance is not None:
        assert state.collision.clearance_m == clearance
    assert not state.collision.HasField("result_age_s")


@pytest.mark.parametrize("margin", [None, 0.0, -0.001])
def test_static_stability_diagnostics_over_grpc(connection, margin):
    stub, control, hardware = connection
    snapshot = hardware.snapshot
    hardware.snapshot = lambda: dict(snapshot(), stability={
        "state": "blocked", "blocked": True, "reason": "Target reduces support clearance",
        "margin_m": margin, "target_margin_m": None, "com_m": [0.2, 0.0, 0.5],
        "mass_kg": 143.3, "required_margin_m": 0.02, "result_age_s": None, "check_ms": 1.8,
    })
    state = stub.GetState(pb.Empty(), timeout=1).stability
    assert state.blocked
    assert state.HasField("margin_m") == (margin is not None)
    if margin is not None:
        assert state.margin_m == margin
    assert list(state.com_m) == [0.2, 0.0, 0.5]
    assert state.mass_kg == 143.3 and state.required_margin_m == 0.02
    assert not state.HasField("target_margin_m")
    assert not state.HasField("result_age_s")


def test_torque_diagnostics_preserve_candidate_and_final_output(connection):
    stub, control, hardware = connection
    snapshot = hardware.snapshot
    arm = dict(
        component="left_arm", joint_names=[f"left_j{i}" for i in range(1, 8)],
        current_a=[5.0] * 7, estimated_torque_nm=[21.275] * 7,
        torque_limit_nm=[20.0] * 7, requested_position_rad=[0.5] * 7,
        candidate_position_rad=[0.499] * 7, candidate_offset_rad=[-0.001] * 7,
        output_position_rad=[0.0] * 7, over_limit=[True] * 7,
    )
    hardware.snapshot = lambda: dict(
        snapshot(), arm_torque={"state": "model_hold", "arms": [arm]}
    )
    state = stub.GetState(pb.Empty(), timeout=1).arm_torque
    assert state.state == "model_hold"
    output = state.arms[0]
    assert output.component == "left_arm"
    for name, values in arm.items():
        if name != "component":
            assert list(getattr(output, name)) == values
