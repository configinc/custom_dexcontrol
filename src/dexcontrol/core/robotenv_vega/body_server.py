"""Run two RobotEnv gRPC services and one exclusive body-control gRPC service.

python -m dexcontrol.core.robotenv_vega.body_server --help
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import signal
import threading
import time

import grpc

from .server import VegaRobotEnvService, robotenv_pb2, robotenv_pb2_grpc
from .body_control import BodyControl, ControlConflict
from .body_hardware import VegaHardware
from .body_grpc import BodyService
from proto import vega_body_pb2_grpc

LOG = logging.getLogger(__name__)


class ManagedArm(VegaRobotEnvService):
    control = None

    def Step(self, request, context):
        with self._cmd_lock:
            try:
                self.control.accept_arm(self.arm_side)
            except ControlConflict as exc:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            return super().Step(request, context)

    def Reset(self, request, context):
        with self._cmd_lock:
            if request.mode != "current":
                try:
                    self.control.accept_arm(self.arm_side)
                except ControlConflict as exc:
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            return super().Reset(request, context)

    def GetConfig(self, request, context):
        result = super().GetConfig(request, context)
        result.metadata["body_control_required"] = "true"
        result.metadata["body_control_port"] = str(self.body_port)
        return result



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--body-port", type=int, default=50065)
    parser.add_argument("--left-port", type=int, default=50061)
    parser.add_argument("--right-port", type=int, default=50063)
    parser.add_argument("--robot-model", default="vega_1")
    parser.add_argument("--gripper-type", choices=["default", "robotiq"], default="robotiq")
    parser.add_argument("--left-gripper", default="/dev/ttyUSB0")
    parser.add_argument("--right-gripper", default="/dev/ttyUSB1")
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
    return parser.parse_args(argv)


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
        control = BodyControl(VegaHardware(services))
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
                    # Retry holding, never motion. A new explicit mode is required.
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
        finally:
            for service in reversed(services):
                service._force_calibration_manager.shutdown()
                service._stop_control_loop()
                service._robot.close()


if __name__ == "__main__":
    main()
