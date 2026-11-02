"""Typed BodyControl transport; hardware and mode logic live in BodyControl."""
import logging

import grpc
from proto import vega_body_pb2 as pb, vega_body_pb2_grpc as rpc

from .body_control import ControlConflict

LOG = logging.getLogger(__name__)
MODES = {"hold": pb.HOLD, "transition": pb.TRANSITION, "body": pb.BODY,
         "arm_ready": pb.ARM_READY, "arms": pb.ARMS}
VELOCITY_FIELDS = ("base_vx", "base_vy", "base_wz", "shoulder_vx", "shoulder_vz", "pitch_rate")


def state_message(state):
    limits = state["limits"]
    return pb.BodyState(
        mode=MODES[state["mode"]],
        error=state["error"] or "", torso=state["torso"],
        shoulder=pb.ShoulderPose(**dict(zip(("x", "z", "pitch"), state["shoulder"]))),
        torso_target=state["torso_target"],
        shoulder_target=pb.ShoulderPose(**dict(zip(("x", "z", "pitch"), state["shoulder_target"]))),
        steering=state["steering"], wheel_velocity=state["wheel_velocity"],
        limits=pb.Limits(
            steering=[pb.JointRange(lower=a, upper=b) for a, b in limits["steering"]],
            torso=[pb.JointRange(lower=a, upper=b) for a, b in limits["torso"]],
            command_velocity=pb.Velocity(**dict(zip(VELOCITY_FIELDS, limits["command_velocity"]))),
            torso_joint_velocity=limits["torso_joint_velocity"],
            shoulder_frame=limits["shoulder_frame"], reference_frame=limits["reference_frame"],
            watchdog_seconds=limits["watchdog_seconds"]))


class BodyService(rpc.BodyControlServicer):
    def __init__(self, control):
        self.control = control

    def _call(self, context, operation):
        try:
            result = operation()
            return pb.Empty() if result is None else state_message(result)
        except ControlConflict as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except (ValueError, TypeError) as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:
            LOG.exception("Body RPC failed")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def GetState(self, request, context):
        return self._call(context, self.control.state)

    def SelectMode(self, request, context):
        def select():
            if request.mode not in (pb.BODY, pb.ARMS):
                raise ValueError("Select BODY or ARMS")
            return self.control.select("body" if request.mode == pb.BODY else "prepare_arms")
        return self._call(context, select)

    def EnableArms(self, request, context):
        return self._call(context, self.control.enable_arms)

    def Command(self, request, context):
        def command():
            if not request.HasField("velocity"):
                raise ValueError("Velocity is required")
            if any(getattr(request.velocity, name) != 0 for name in VELOCITY_FIELDS[3:]):
                raise ValueError("Torso velocity is unsupported; send an absolute shoulder_target")
            target = ([request.shoulder_target.x, request.shoulder_target.z, request.shoulder_target.pitch]
                      if request.HasField("shoulder_target") else None)
            self.control.set_command([getattr(request.velocity, name) for name in VELOCITY_FIELDS[:3]], target)
            return self.control.state()
        return self._call(context, command)

    def Stop(self, request, context):
        return self._call(context, self.control.stop)
