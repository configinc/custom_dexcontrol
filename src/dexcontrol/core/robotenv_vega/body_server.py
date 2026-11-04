"""Run two RobotEnv gRPC services and one exclusive body-control gRPC service.

python -m dexcontrol.core.robotenv_vega.body_server --help
"""
from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import grpc

from proto import vega_body_pb2_grpc

from ..control.config import load_config
from .body_control import ControlConflict
from .body_grpc import BodyService
from .body_hardware import keep_state_subscriptions_active
from .control_node import VegaControlNode
from .managed_hardware import ManagedVegaHardware
from .server import VegaRobotEnvService, robotenv_pb2, robotenv_pb2_grpc

LOG = logging.getLogger(__name__)


class ManagedArm(VegaRobotEnvService):
    control = None
    _calibration_direct_reset = False

    @contextmanager
    def calibration_reset(self):
        # Drain an in-flight calibration interpolation tick before direct joint
        # commands begin. The other arm keeps its own interpolation running.
        with self.control.gate.lock:
            self._calibration_direct_reset = True
            self._robot.reset_filter_state()
        try:
            yield
        finally:
            with self.control.gate.lock:
                self._robot.reset_filter_state()
                self._calibration_direct_reset = False

    def _start_control_loop(self):
        # All managed output is scheduled by VegaControlNode.
        return

    def _claim_robot_for_force_calibration(self):
        if self.control is None:
            return super()._claim_robot_for_force_calibration()
        try:
            # Keep the node lock through admission: Stop must see the fully
            # attached participant and its cancellation event.
            with self.control.lock:
                action = self.control.claim_calibration(self)
                if action is None:
                    return False
                acquired = super()._claim_robot_for_force_calibration()
                if acquired:
                    self._force_calibration_manager.action = action
                else:
                    action.finish(cancelled=True)
                return acquired
        except ControlConflict:
            return False

    def _release_robot_from_force_calibration(self):
        super()._release_robot_from_force_calibration()
        manager = getattr(self, "_force_calibration_manager", None)
        if manager is not None and manager.action is not None:
            status = manager.lifecycle.status()
            manager.action.finish(
                error=status.message if status.state == "FAILED" else None,
                cancelled=status.state == "CANCELLED",
            )

    def _cancel_force_calibration_motion(self):
        # Signal the stage immediately. Lifecycle cancellation may need the
        # manager lock, so it must not block the control/output thread.
        if not self._force_calibration_owns_robot.is_set():
            return
        self._force_calibration_manager.interrupt_motion()

    def Step(self, request, context):
        with self._cmd_lock:
            try:
                with self.control.arm_input(self.arm_side):
                    return super().Step(request, context)
            except ControlConflict as exc:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

    def Reset(self, request, context):
        if request.mode == "current":
            return super().Reset(request, context)
        mode = request.mode or "home"
        try:
            if mode == "home":
                target = self.reset_joints
            elif mode == "target":
                target = self._extract_target_joints(request)
            elif mode == "random":
                target = self._sample_random_target()
            else:
                raise ValueError(f"Unknown reset mode: {mode}")
            skip = bool(request.params["skip_intermediate_pose"].int_value) if "skip_intermediate_pose" in request.params else False
            def move(cancelled):
                self._cancel_move.clear()
                self.control.hardware.initialize_gripper(self, cancelled)
                with self._cmd_lock:
                    self._execute_reset_sequence(target, cancelled=cancelled,
                        interruptible_gripper=True, skip_intermediate_pose=skip)
            self.control.start_action(f"initial_pose_{self.arm_side}", move)
            worker = self.control.actions.worker
            while worker.is_alive():
                worker.join(.05)
                if not context.is_active():
                    self.control.stop()
            # Finish the action before accepting another arm's Reset.
            self.control.finish_action()
            status = self.control.actions.snapshot()
            if status["state"] != "succeeded":
                raise RuntimeError(status["error"] or f"Action {status['state']}")
            observation, timestamp = self._create_observation()
            return robotenv_pb2.ResetResponse(observation=observation, status="SUCCESS",
                message=f"Reset to {mode}", timestamp_us=timestamp)
        except ControlConflict as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def GetConfig(self, request, context):
        result = super().GetConfig(request, context)
        result.metadata["body_control_required"] = "true"
        result.metadata["body_control_lifecycle"] = "true"
        result.metadata["body_control_port"] = str(self.body_port)
        return result



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-config", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--body-port", type=int, default=50065)
    parser.add_argument("--left-port", type=int, default=50061)
    parser.add_argument("--right-port", type=int, default=50063)
    parser.add_argument("--robot-model", default="vega_1")
    parser.add_argument("--gripper-type", choices=["default", "robotiq"], default="robotiq")
    parser.add_argument("--left-gripper", default=None,
                        help="Left Robotiq USB serial port; omit for arm EE RS485 pass-through")
    parser.add_argument("--right-gripper", default=None,
                        help="Right Robotiq USB serial port; omit for arm EE RS485 pass-through")
    parser.add_argument("--action-hz", type=int, default=20)
    # Match the deployed Vega arm launch profile; body timing is independent.
    parser.add_argument("--control-loop-hz", type=int, default=200)
    parser.add_argument("--use-velocity-feedforward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interpolation-method", choices=["linear", "cubic"], default="linear")
    parser.add_argument("--interpolation-history", type=int, default=3)
    parser.add_argument("--vel-smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--hw-correction-alpha", type=float, default=0.5)
    parser.add_argument("--max-delta-scale", type=float, default=3.0)
    parser.add_argument("--max-jerk", type=float, default=0.0)
    parser.add_argument("--rot-sensitivity", type=float, default=2.0)
    parser.add_argument("--vel-ratio", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.control_loop_hz <= 0 or args.action_hz <= 0:
        parser.error("Managed input and output frequencies must be positive")
    return args


def arm_options(args):
    return dict(control_hz=args.action_hz, control_loop_hz=args.control_loop_hz,
                use_velocity_feedforward=args.use_velocity_feedforward,
                interpolation_method=args.interpolation_method,
                interpolation_history=args.interpolation_history,
                vel_smoothing_alpha=args.vel_smoothing_alpha,
                hw_correction_alpha=args.hw_correction_alpha,
                max_delta_scale=args.max_delta_scale, max_jerk=args.max_jerk,
                rot_sensitivity=args.rot_sensitivity, vel_ratio=args.vel_ratio)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.control_config)

    def shutdown(signum, frame):
        raise KeyboardInterrupt

    # tmux restart sends SIGHUP; process cleanup sends SIGTERM. Both must
    # run the same stop/hold cleanup as Ctrl+C.
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    services, servers = [], []
    stop = threading.Event()
    control = worker = None
    try:
        for side, device in (("left", args.left_gripper), ("right", args.right_gripper)):
            service = ManagedArm(robot_model=args.robot_model, arm_side=side,
                gripper_type=args.gripper_type, robotiq_comport=device,
                **arm_options(args), auto_start_control_loop=False,
                robot=services[0]._robot.robot if services else None)
            services.append(service)
        keep_state_subscriptions_active(services[0]._robot.robot)
        control = VegaControlNode(ManagedVegaHardware(services, config), config)
        for service, port in zip(services, (args.left_port, args.right_port)):
            service.control = control
            service.body_port = args.body_port
            server = grpc.server(ThreadPoolExecutor(max_workers=4))
            robotenv_pb2_grpc.add_RobotEnvServicer_to_server(service, server)
            if not server.add_insecure_port(f"{args.host}:{port}"):
                raise RuntimeError(f"Cannot bind arm port {port}")
            server.start()
            servers.append(server)
        body = grpc.server(ThreadPoolExecutor(max_workers=4))
        vega_body_pb2_grpc.add_BodyControlServicer_to_server(BodyService(control), body)
        if not body.add_insecure_port(f"{args.host}:{args.body_port}"):
            raise RuntimeError(f"Cannot bind body port {args.body_port}")
        body.start()
        servers.append(body)

        def run():
            next_tick = time.monotonic()
            while not stop.is_set():
                try:
                    control.tick()
                except Exception:
                    LOG.exception("Control stopped after fault")
                    # Fault remains latched until an explicit reset and start.
                next_tick += control.PERIOD
                now = time.monotonic()
                next_tick = max(next_tick, now)
                if stop.wait(next_tick - now):
                    break
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        LOG.info("Body gRPC listening on %s:%s; initial mode HOLD", args.host, args.body_port)
        body.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for server in servers:
            server.stop(0).wait()
        if worker:
            worker.join(timeout=2)
        try:
            if control:
                control.stop()
                control.actions.join()
                control.close()
        finally:
            for service in reversed(services):
                service._force_calibration_manager.shutdown()
                service._stop_control_loop()
                service._robot.close()


if __name__ == "__main__":
    main()
