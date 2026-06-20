"""Vega RobotEnv gRPC Service."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Any, Optional

import grpc
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add package root for local imports.
# server.py lives at: <repo>/src/dexcontrol/core/robotenv_vega/server.py
#   parents: [0]=robotenv_vega, [1]=core, [2]=dexcontrol, [3]=src, [4]=<repo>
_this = Path(__file__).resolve()
sys.path.insert(0, str(_this.parents[2]))  # src/dexcontrol/ -> "from core.vega..."
sys.path.insert(0, str(_this.parents[4]))  # <repo>/          -> "from proto..."

from core.vega.robot import (  # noqa: E402
    CommunicationFailedError,
    IKFailedError,
    JointLimitExceededError,
    VegaRobot,
)
from proto import robotenv_pb2, robotenv_pb2_grpc  # noqa: E402


LOGGER = logging.getLogger("robotenv_vega")


def _to_proto_value(val: Any) -> robotenv_pb2.Value:
    """Convert a Python scalar / list / ndarray to a proto Value."""
    if isinstance(val, (list, tuple, np.ndarray)):
        return robotenv_pb2.Value(
            float_array=robotenv_pb2.FloatArray(values=[float(v) for v in val])
        )
    if isinstance(val, (int, bool, np.integer, np.bool_)):
        return robotenv_pb2.Value(float_value=float(val))
    if isinstance(val, (float, np.floating)):
        return robotenv_pb2.Value(float_value=float(val))
    # Fallback: stringify
    return robotenv_pb2.Value(string_value=str(val))


# Per-arm init (home) and reset middle waypoints.
# Left→Right mirroring: [-v0, -v1, -v2, v3, -v4, -v5, -v6]
_INIT_JOINTS = {
    "left":  np.array([-1.4234,  1.3524,  2.8707, -1.981,   0.6751, -0.1662,  0.068]),
    "right": np.array([ 1.4234, -1.3524, -2.8707, -1.981,  -0.1515,  0.1662, -0.068]),
}
_RESET_MIDDLE_JOINTS = {
    "left":  np.array([-2.7592,   1.3579,   2.8643, -1.8855, 0.6702, -0.1592, 0.2338]),
    "right": np.array([ 2.7592,   -1.3579,   -2.8643, -1.8855, -0.6702, 0.1592, -0.2338]),
}


class VegaRobotEnvService(robotenv_pb2_grpc.RobotEnvServicer):
    """RobotEnv service implementation for one Vega arm."""

    def __init__(
        self,
        robot_model: str = "vega_1",
        arm_side: str = "left",
        gripper_type: str = "default",
        frame_type: str = "vega_mobile_base",
        control_hz: int = 20,
        use_velocity_feedforward: bool = False,
        base_frame_rotation: Optional[list[float]] = None,
        ik_solver_type: str = "pink",
        robotiq_comport: str = "/dev/ttyUSB0",
        ema_alpha: float = 0.0,
        ik_damping_default: float = 1e-3,
        ik_damping_torso: float = 30000.0,
        ik_damping_arm_j2: float = 100.0,
        ik_damping_arm_j3: float = 50.0,
        interpolation_method: str = "none",
        interpolation_history: int = 4,
        control_loop_hz: int = 0,
        filter_type: str = "none",
        filter_cutoff_freq: float = 10.0,
        filter_order: int = 2,
        filter_ema_alpha: float = 0.1,
        vel_smoothing_alpha: float = 0.3,
        hw_correction_alpha: float = 0.7,
        max_delta_scale: float = 1.0,
        max_jerk: float = 0.25,
        rot_sensitivity: float = 1.0,
        vel_ratio: float = 1.0,
        vel_damp_thresh: float = 0.05,
        head_init_pos: list[float] | None = None,  # default: [yaw=2.0, pitch=0.0, roll=-0.3]
        **kwargs,
    ):
        hand_type = kwargs.pop("hand_type", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if hand_type is not None and gripper_type == "default":
            gripper_type = hand_type
        if head_init_pos is None:
            head_init_pos = [2.0, 0.0, -0.3]

        self.robot_model = robot_model
        self.arm_side = arm_side
        self.gripper_type = gripper_type
        self.frame_type = frame_type
        self.control_hz = int(control_hz)
        self.use_velocity_feedforward = bool(use_velocity_feedforward)
        self.base_frame_rotation = base_frame_rotation
        self._max_lin_delta, self._max_rot_delta = self._compute_cartesian_delta_limits(
            self.control_hz, rot_sensitivity=rot_sensitivity,
        )

        self._robot = VegaRobot(
            robot_model=robot_model,
            arm_side=arm_side,
            control_hz=control_hz,
            use_velocity_feedforward=use_velocity_feedforward,
            gripper_type=gripper_type,
            ik_solver_type=ik_solver_type,
            robotiq_comport=robotiq_comport,
            ema_alpha=ema_alpha,
            interpolation_method=interpolation_method,
            interpolation_history=interpolation_history,
            filter_type=filter_type,
            filter_cutoff_freq=filter_cutoff_freq,
            filter_order=filter_order,
            filter_ema_alpha=filter_ema_alpha,
            vel_smoothing_alpha=vel_smoothing_alpha,
            hw_correction_alpha=hw_correction_alpha,
            max_delta_scale=max_delta_scale,
            max_jerk=max_jerk,
            vel_ratio=vel_ratio,
            vel_damp_thresh=vel_damp_thresh,
            head_init_pos=head_init_pos,
        )
        self._robot.launch_robot()

        # --- Background control loop for interpolation-based upsampling ---
        # When interpolation is active, gRPC Step() buffers commands at
        # input rate and this thread dispatches them to hardware at
        # control_loop_hz (typically 100 Hz).
        self._control_loop_hz = control_loop_hz if control_loop_hz > 0 else 0
        self._control_loop_thread: Optional[threading.Thread] = None
        self._control_loop_stop = threading.Event()
        if self._robot.interpolation_enabled and self._control_loop_hz > 0:
            self._start_control_loop()
            LOGGER.info(
                "Control loop thread started: %d Hz (input≈%d Hz → control=%d Hz)",
                self._control_loop_hz,
                control_hz,
                self._control_loop_hz,
            )

        # Override home position with per-arm init joints.
        self.reset_joints = _INIT_JOINTS[arm_side].copy()
        self.reset_middle_joints = _RESET_MIDDLE_JOINTS[arm_side].copy()
        self.safe_transit_pose = self._robot.safe_transit_pose.copy()

        if base_frame_rotation is not None:
            rot = R.from_euler("XYZ", base_frame_rotation, degrees=True).as_matrix()
            self.R_world_to_robot = rot.T
            self.R_robot_to_world = rot
        else:
            self.R_world_to_robot = None
            self.R_robot_to_world = None

        # Serialize all robot commands — prevents concurrent Reset/Step from
        # sending conflicting position commands to the same arm.
        self._cmd_lock = threading.Lock()
        # Set by a new Reset to cancel any in-progress _move_incremental loop.
        self._cancel_move = threading.Event()

        LOGGER.info(
            "Initialized VegaRobotEnvService model=%s arm=%s gripper=%s frame=%s hz=%s",
            robot_model,
            arm_side,
            gripper_type,
            frame_type,
            control_hz,
        )
        LOGGER.info(
            "Joint command mode: %s",
            "position+velocity feedforward" if self.use_velocity_feedforward else "position-only",
        )
        LOGGER.info(
            "IK config: solver=%s damping(default=%.6f torso=%.1f arm_j2=%.1f arm_j3=%.1f)",
            ik_solver_type,
            ik_damping_default,
            ik_damping_torso,
            ik_damping_arm_j2,
            ik_damping_arm_j3,
        )
        LOGGER.info(
            "Cartesian velocity normalization enabled: max_lin_delta=%.6f max_rot_delta=%.6f",
            self._max_lin_delta,
            self._max_rot_delta,
        )
        if ema_alpha > 0:
            _omega = ema_alpha / ((1.0 / max(1, control_hz)) * (1.0 - ema_alpha))
            LOGGER.info(
                "2nd-order smoothing: enabled (alpha=%.3f, omega=%.1f rad/s, critically damped)",
                ema_alpha, _omega,
            )
        else:
            LOGGER.info("Smoothing: disabled (alpha=0.0)")

        # Move to init position on startup.
        LOGGER.info("Moving to init position on startup (arm=%s)", arm_side)
        self._execute_reset_sequence(self.reset_joints)

    # ------------------------------------------------------------------
    # Background control loop (for interpolation upsampling)
    # ------------------------------------------------------------------

    def _start_control_loop(self) -> None:
        """Start the background control-loop thread."""
        if self._control_loop_thread is not None:
            return
        self._control_loop_stop.clear()
        self._control_loop_thread = threading.Thread(
            target=self._control_loop_run,
            name=f"ctrl-loop-{self.arm_side}",
            daemon=True,
        )
        self._control_loop_thread.start()

    def _stop_control_loop(self) -> None:
        """Stop the background control-loop thread."""
        self._control_loop_stop.set()
        if self._control_loop_thread is not None and self._control_loop_thread.is_alive():
            self._control_loop_thread.join(timeout=2.0)
        self._control_loop_thread = None

    def _control_loop_run(self) -> None:
        """Main loop: dispatch interpolated commands at control rate."""
        dt = 1.0 / self._control_loop_hz
        LOGGER.info("Control loop running at %d Hz", self._control_loop_hz)
        while not self._control_loop_stop.is_set():
            t0 = time.perf_counter()
            try:
                self._robot.execute_interpolated_tick()
            except Exception as exc:
                LOGGER.error("Control loop tick error: %s", exc)
            elapsed = time.perf_counter() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    @staticmethod
    def _compute_cartesian_delta_limits(
        control_hz: int,
        rot_sensitivity: float = 1.0,
    ) -> tuple[float, float]:
        """Compute per-step Cartesian delta limits from control frequency."""
        baseline_hz = 10.0
        scale = 1.0 - (float(control_hz) - baseline_hz) / 80.0
        # Keep behavior safe even when control_hz is unexpectedly high.
        scale = float(np.clip(scale, 0.1, 1.5))
        return 0.075 * scale, 0.3 * scale * rot_sensitivity

    def _cartesian_velocity_to_delta(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized 6D Cartesian velocity command to per-step delta."""
        converted = np.asarray(action, dtype=np.float64).copy()
        if converted.shape[0] < 6:
            return converted

        lin_vel = converted[:3]
        rot_vel = converted[3:6]

        lin_norm = float(np.linalg.norm(lin_vel))
        rot_norm = float(np.linalg.norm(rot_vel))
        if lin_norm > 1.0:
            lin_vel = lin_vel / lin_norm
        if rot_norm > 1.0:
            rot_vel = rot_vel / rot_norm

        converted[:3] = lin_vel * self._max_lin_delta
        converted[3:6] = rot_vel * self._max_rot_delta
        return converted

    def GetObservationSpec(self, request, context):
        del request, context
        spec = robotenv_pb2.ObservationSpec(
            num_arms=1,
            metadata={
                "robot_type": "vega",
                "robot_model": self.robot_model,
                "arm_side": self.arm_side,
                "gripper_type": self.gripper_type,
                "frame_type": self.frame_type,
                "control_hz": str(self.control_hz),
            },
        )

        spec.fields["joint_positions"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[7],
                required=True,
                description="Joint positions in radians",
            )
        )
        spec.fields["joint_velocities"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[7],
                required=False,
                description="Joint velocities in rad/s",
            )
        )
        spec.fields["joint_torques_computed"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[7],
                required=False,
                description="Estimated joint torques in Nm",
            )
        )
        spec.fields["prev_joint_torques_computed"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[7],
                required=False,
                description="Previous joint torques computed",
            )
        )
        spec.fields["prev_joint_torques_computed_safened"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[7],
                required=False,
                description="Previous joint torques computed (safened)",
            )
        )
        spec.fields["motor_torques_measured"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[7],
                required=False,
                description="Motor torques measured",
            )
        )
        spec.fields["gripper_position"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[],
                required=True,
                description="Scalar gripper position in [0,1]",
            )
        )
        spec.fields["cartesian_position"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[6],
                required=True,
                description="End-effector pose [x, y, z, roll, pitch, yaw]",
            )
        )
        spec.fields["wrench_state"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[6],
                required=False,
                description="Wrench state [fx, fy, fz, tx, ty, tz]",
            )
        )
        spec.fields["prev_controller_latency_ms"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[],
                required=False,
                description="Controller latency reported by robot stack",
            )
        )
        spec.fields["prev_command_successful"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="int64",
                shape=[],
                required=False,
                description="1 if previous command succeeded, otherwise 0",
            )
        )
        spec.fields["prev_gripper_command_successful"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="int64",
                shape=[],
                required=False,
                description="1 if previous gripper command succeeded, otherwise 0",
            )
        )
        spec.fields["timestamp"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="int64",
                shape=[],
                required=True,
                description="Robot timestamp in microseconds",
            )
        )
        return spec

    def GetConfig(self, request, context):
        del request, context
        return robotenv_pb2.RobotConfig(
            gripper_type=self.gripper_type,
            frame_type=self.frame_type,
            dof=[7],
            supported_action_spaces=[
                "joint_position",
                "joint_velocity",
                "joint_delta",
                "cartesian_velocity",
                "cartesian_delta",
                "target_cartesian_delta",
            ],
            metadata={
                "robot_model": self.robot_model,
                "control_hz": str(self.control_hz),
                "arm_side": self.arm_side,
            },
        )

    def Reset(self, request, context):
        mode = request.mode or "home"
        LOGGER.info("Reset requested: mode=%s arm=%s", mode, self.arm_side)

        # Cancel any in-progress _move_incremental from a previous Reset.
        self._cancel_move.set()

        # Pause the control loop during reset to prevent conflicting commands.
        was_running = self._control_loop_thread is not None and self._control_loop_thread.is_alive()
        if was_running:
            self._stop_control_loop()

        with self._cmd_lock:
            self._cancel_move.clear()
            try:
                if mode == "home":
                    target_joints = self.reset_joints
                    self._execute_reset_sequence(target_joints)
                elif mode == "target":
                    target_joints = self._extract_target_joints(request)
                    self._execute_reset_sequence(target_joints)
                elif mode == "random":
                    target_joints = self._sample_random_target()
                    self._execute_reset_sequence(target_joints)
                elif mode == "current":
                    pass
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(f"Unknown reset mode: {mode}")
                    return robotenv_pb2.ResetResponse()

                observation, timestamp_us = self._create_observation()
                return robotenv_pb2.ResetResponse(
                    observation=observation,
                    status="SUCCESS",
                    message=f"Reset to {mode}",
                    timestamp_us=timestamp_us,
                )
            except Exception as exc:
                LOGGER.exception("Reset failed (mode=%s): %s", mode, exc)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Reset failed: {exc}")
                return robotenv_pb2.ResetResponse()
            finally:
                # Restart control loop if it was running before reset.
                if was_running:
                    self._start_control_loop()

    def Step(self, request, context):
        action_space = request.action_space
        action_space_for_robot = action_space
        action = np.asarray(request.action, dtype=np.float64)
        gripper_action_space = request.gripper_action_space
        if not gripper_action_space:
            gripper_action_space = "velocity" if "velocity" in action_space else "position"

        try:
            # -- gripper debug --
            if self._robot.hand is not None:
                _cur_grip_joint = self._robot.get_cached_gripper_joint_pos()
                _cur_grip_raw = float(_cur_grip_joint[0]) if _cur_grip_joint.size > 0 else 0.0
                _cur_grip_norm = float(self._robot.get_cached_gripper_position())
            else:
                _cur_grip_raw = 0.0
                _cur_grip_norm = 0.0
            if action_space.startswith("joint"):
                _grip_cmd = float(action[7]) if action.shape[0] > 7 else 0.0
            else:
                _grip_cmd = float(action[6]) if action.shape[0] > 6 else 0.0
            LOGGER.info(
                "[Gripper] cmd=%.4f  space=%s  gripper_space=%s  cur_raw=%.4f  cur_norm=%.4f",
                _grip_cmd, action_space, gripper_action_space, _cur_grip_raw, _cur_grip_norm,
            )
            # -- end gripper debug --

            t_step_start = time.time()
            if self.R_world_to_robot is not None and (
                "cartesian" in action_space or action_space == "target_cartesian_delta"
            ):
                action = self._transform_action_to_robot_frame(action)
            if action_space == "cartesian_velocity":
                raw_action = action.copy()
                raw_lin_norm = float(np.linalg.norm(raw_action[:3]))
                raw_rot_norm = float(np.linalg.norm(raw_action[3:6]))
                action = self._cartesian_velocity_to_delta(action)
                action_space_for_robot = "cartesian_delta"
                LOGGER.info(
                    "Converted cartesian_velocity -> cartesian_delta: lin_norm=%.4f rot_norm=%.4f max_lin_delta=%.6f max_rot_delta=%.6f",
                    raw_lin_norm, raw_rot_norm, self._max_lin_delta, self._max_rot_delta,
                )
            elif action_space == "target_cartesian_delta":
                # Recover cartesian_velocity by re-applying teleop gains,
                # then convert to motor delta via the standard path.
                from dexcontrol.core.vega.robot import (
                    _TELEOP_POS_ACTION_GAIN,
                    _TELEOP_ROT_ACTION_GAIN,
                )
                action[:3] *= _TELEOP_POS_ACTION_GAIN
                action[3:6] *= _TELEOP_ROT_ACTION_GAIN
                action = self._cartesian_velocity_to_delta(action)
                action_space_for_robot = "cartesian_delta"
            t_after_xform = time.time()

            # Get robot_state BEFORE executing action (needed for create_action_dict)
            pre_action_state, _ = self._robot.get_robot_state()

            if self._robot.interpolation_enabled and self._control_loop_hz > 0:
                # Interpolation mode: buffer command; control loop dispatches.
                self._robot.add_command_point(
                    action,
                    action_space=action_space_for_robot,
                    gripper_action_space=gripper_action_space,
                )
            else:
                # Legacy synchronous path.
                self._robot.update_command(
                    action,
                    action_space=action_space_for_robot,
                    gripper_action_space=gripper_action_space,
                    blocking=False,
                )
            t_after_cmd = time.time()

            # Build comprehensive action_info using the original request action_space
            # (before any velocity->delta conversion) so the dict contains all
            # representations (cartesian_velocity, delta_action, joint_velocity, etc.)
            try:
                action_dict = self._robot.create_action_dict(
                    np.asarray(request.action, dtype=np.float64),
                    action_space=action_space,
                    gripper_action_space=gripper_action_space,
                    robot_state=pre_action_state,
                    max_lin_delta=self._max_lin_delta,
                    max_rot_delta=self._max_rot_delta,
                )
                action_info = self._action_dict_to_proto(action_dict)
            except Exception as exc:
                LOGGER.warning("create_action_dict failed, sending empty action_info: %s", exc)
                action_info = {}
            t_after_ainfo = time.time()

            observation, timestamp_us = self._create_observation()
            t_after_obs = time.time()
            total_ms = (t_after_obs - t_step_start) * 1000
            if total_ms > 20:
                LOGGER.warning(
                    "[Step SLOW] total=%.1fms (xform=%.1f cmd=%.1f ainfo=%.1f obs=%.1f)",
                    total_ms,
                    (t_after_xform - t_step_start) * 1000,
                    (t_after_cmd - t_after_xform) * 1000,
                    (t_after_ainfo - t_after_cmd) * 1000,
                    (t_after_obs - t_after_ainfo) * 1000,
                )
            return robotenv_pb2.StepResponse(
                observation=observation,
                status="SUCCESS",
                message="",
                timestamp_us=timestamp_us,
                action_info=action_info,
            )
        except JointLimitExceededError as exc:
            observation, timestamp_us = self._safe_observation()
            return robotenv_pb2.StepResponse(
                observation=observation,
                status="JOINT_LIMIT_EXCEEDED",
                message=str(exc),
                timestamp_us=timestamp_us,
            )
        except IKFailedError as exc:
            observation, timestamp_us = self._safe_observation()
            return robotenv_pb2.StepResponse(
                observation=observation,
                status="IK_FAILED",
                message=str(exc),
                timestamp_us=timestamp_us,
            )
        except CommunicationFailedError as exc:
            observation, timestamp_us = self._safe_observation()
            return robotenv_pb2.StepResponse(
                observation=observation,
                status="COMMUNICATION_FAILED",
                message=str(exc),
                timestamp_us=timestamp_us,
            )
        except Exception as exc:
            LOGGER.exception("Step failed: %s", exc)
            observation, timestamp_us = self._safe_observation()
            return robotenv_pb2.StepResponse(
                observation=observation,
                status="ERROR",
                message=str(exc),
                timestamp_us=timestamp_us,
            )

    def HealthCheck(self, request, context):
        del request, context
        try:
            self._robot.get_robot_state()
            version_info = self._load_version_info()
            version_str = version_info.get("tag", "1.0.0-vega")
            return robotenv_pb2.HealthCheckResponse(
                status="HEALTHY",
                message=f"Vega {self.arm_side} arm operational",
                version=version_str,
            )
        except Exception as exc:
            return robotenv_pb2.HealthCheckResponse(
                status="UNHEALTHY",
                message=str(exc),
                version="1.0.0-vega",
            )

    def GetVersion(self, request, context):
        """Return deployed code version info."""
        del request, context
        info = self._load_version_info()
        return robotenv_pb2.VersionInfo(
            branch=info.get("branch", "unknown"),
            commit=info.get("commit", "unknown"),
            tag=info.get("tag", "unknown"),
        )

    def _load_version_info(self) -> dict[str, str]:
        """Load version info from version_info.txt."""
        info = {"branch": "unknown", "commit": "unknown", "tag": "unknown"}
        # server.py is at <repo>/src/dexcontrol/core/robotenv_vega/server.py
        # version_info.txt is at <repo>/version_info.txt
        version_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "..", "version_info.txt",
        )
        if os.path.exists(version_file):
            with open(version_file, "r", encoding="utf-8") as f:
                for line in f.read().splitlines():
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        if key in info:
                            info[key] = value
        return info

    def _create_observation(self) -> tuple[dict[str, Any], int]:
        state_dict, timestamp_dict = self._robot.get_robot_state()
        timestamp_us = self._timestamp_us(timestamp_dict)

        cartesian_position = np.asarray(state_dict["cartesian_position"], dtype=np.float64)
        if self.R_robot_to_world is not None:
            cartesian_position = self._transform_state_to_env_frame(cartesian_position)

        observation = {
            "joint_positions": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict["joint_positions"]).tolist())
            ),
            "joint_velocities": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict["joint_velocities"]).tolist())
            ),
            "joint_torques_computed": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict["joint_torques_computed"]).tolist())
            ),
            "prev_joint_torques_computed": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict.get("prev_joint_torques_computed", [0]*7)).tolist())
            ),
            "prev_joint_torques_computed_safened": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict.get("prev_joint_torques_computed_safened", [0]*7)).tolist())
            ),
            "motor_torques_measured": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict.get("motor_torques_measured", [0]*7)).tolist())
            ),
            "gripper_position": robotenv_pb2.Value(
                float_value=float(state_dict["gripper_position"])
            ),
            "cartesian_position": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=cartesian_position.tolist())
            ),
            "wrench_state": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=np.asarray(state_dict["wrench_state"]).tolist())
            ),
            "prev_controller_latency_ms": robotenv_pb2.Value(
                float_value=float(state_dict.get("prev_controller_latency_ms", 0.0))
            ),
            "prev_command_successful": robotenv_pb2.Value(
                int_value=int(bool(state_dict.get("prev_command_successful", True)))
            ),
            "prev_gripper_command_successful": robotenv_pb2.Value(
                int_value=int(bool(state_dict.get("prev_gripper_command_successful", True)))
            ),
            "timestamp": robotenv_pb2.Value(
                int_value=int(timestamp_us)
            ),
        }
        return observation, int(timestamp_us)

    @staticmethod
    def _action_dict_to_proto(action_dict: dict) -> dict:
        """Convert create_action_dict() output to proto map<string, Value>.

        Handles: float scalars, list/ndarray of floats, and nested robot_state
        dict (which is serialised as a flat set of 'state.<key>' entries so the
        client can reconstruct it).
        """
        proto_map: dict[str, Any] = {}

        for key, val in action_dict.items():
            if key == "robot_state":
                # Flatten robot_state into "state.<subkey>" entries
                if isinstance(val, dict):
                    for sk, sv in val.items():
                        proto_map[f"state.{sk}"] = _to_proto_value(sv)
                continue
            proto_map[key] = _to_proto_value(val)

        return proto_map

    def _safe_observation(self) -> tuple[dict[str, Any], int]:
        try:
            return self._create_observation()
        except Exception:
            now_us = int(time.time() * 1_000_000)
            return {}, now_us

    _RESET_TOLERANCE_RAD = 0.05
    _RESET_TIMEOUT_S = 30.0
    _RESET_CMD_HZ = 100.0
    _RESET_MAX_STEP_RAD = 0.2
    _RESET_SETTLE_S = 0.5

    def _execute_reset_sequence(self, target_joints: np.ndarray) -> None:
        t0 = time.time()
        LOGGER.info("Reset[%s]: starting gripper open", self.arm_side)
        self._robot.update_gripper(0.0, velocity=False, blocking=True)
        LOGGER.info("Reset[%s]: gripper done (%.2fs), starting reset motion", self.arm_side, time.time() - t0)
        self._robot.reset_filter_state()

        # Move to middle waypoint to avoid collisions.
        middle_f64 = np.asarray(self.reset_middle_joints, dtype=np.float64)
        LOGGER.info("Reset[%s]: moving to middle waypoint", self.arm_side)
        self._move_incremental(middle_f64)
        LOGGER.info("Reset[%s]: middle waypoint reached (%.2fs)", self.arm_side, time.time() - t0)

        # Then move to the final target (home/init position).
        target_f64 = np.asarray(target_joints, dtype=np.float64)
        LOGGER.info("Reset[%s]: moving to target", self.arm_side)
        self._move_incremental(target_f64)
        LOGGER.info("Reset[%s]: target reached (%.2fs), syncing motion manager", self.arm_side, time.time() - t0)

        self._robot.reset_filter_state()
        self._robot.sync_motion_manager_with_arm(
            np.asarray(self._robot.arm.get_joint_pos(), dtype=np.float64)
        )
        LOGGER.info("Reset[%s]: sequence complete (%.2fs)", self.arm_side, time.time() - t0)

    _RESET_SETTLE_TOL_RAD = 0.15  # Looser tolerance for settle-based exit

    def _move_incremental(
        self,
        target: np.ndarray,
        tolerance: float | None = None,
        timeout: float | None = None,
    ) -> None:
        """Move to target in incremental steps respecting motor delta limit."""
        tol = tolerance if tolerance is not None else self._RESET_TOLERANCE_RAD
        t_max = timeout if timeout is not None else self._RESET_TIMEOUT_S
        settle_tol = self._RESET_SETTLE_TOL_RAD
        target = np.asarray(target, dtype=np.float64)
        dt = 1.0 / self._RESET_CMD_HZ
        max_step = self._RESET_MAX_STEP_RAD
        deadline = time.time() + t_max
        settle_start = None
        prev_actual = None

        while time.time() < deadline:
            if self._cancel_move.is_set():
                LOGGER.info("_move_incremental cancelled by new request")
                return

            actual = np.asarray(self._robot.arm.get_joint_pos(), dtype=np.float64)
            diff = target - actual
            max_err = float(np.max(np.abs(diff)))
            if max_err < tol:
                break

            clipped = np.clip(diff, -max_step, max_step)
            intermediate = actual + clipped
            self._robot.arm._send_position_command(intermediate)

            # Settle detection using previous loop's reading (no extra read).
            if prev_actual is not None:
                max_move = float(np.max(np.abs(actual - prev_actual)))
                if max_move < 0.005 and max_err < settle_tol:
                    if settle_start is None:
                        settle_start = time.time()
                    elif time.time() - settle_start > self._RESET_SETTLE_S:
                        LOGGER.info(
                            "Arm settled with max error %.4f rad (settle_tol %.4f)",
                            max_err, settle_tol,
                        )
                        break
                else:
                    settle_start = None

            prev_actual = actual
            time.sleep(dt)
        else:
            actual = np.asarray(self._robot.arm.get_joint_pos(), dtype=np.float64)
            max_err = float(np.max(np.abs(target - actual)))
            LOGGER.warning(
                "Reset move timed out after %.1fs (max_err=%.4f rad, tol=%.4f)",
                t_max, max_err, tol,
            )

    def _extract_target_joints(self, request) -> np.ndarray:
        if "joint_positions" not in request.params:
            raise ValueError("target reset requires params['joint_positions']")
        joint_values = request.params["joint_positions"].float_array.values
        if len(joint_values) != 7:
            raise ValueError(f"Expected 7 joint values, got {len(joint_values)}")
        target = np.asarray(joint_values, dtype=np.float64)
        limits = self._robot.arm.joint_pos_limit
        if limits is not None:
            low = limits[:, 0].astype(np.float64)
            high = limits[:, 1].astype(np.float64)
            violates = np.logical_or(target < low, target > high)
            if np.any(violates):
                target = np.clip(target, low, high)
                LOGGER.warning(
                    "Reset target joints clipped to limits (violated indices: %s)",
                    np.where(violates)[0].tolist(),
                )
        self._robot.validate_joint_limits(target)
        return target

    def _sample_random_target(self) -> np.ndarray:
        limits = self._robot.arm.joint_pos_limit
        if limits is None:
            return self.reset_joints.copy()

        center = self.reset_joints.astype(np.float64)
        noise = np.random.uniform(-0.15, 0.15, size=7)
        sampled = center + noise
        sampled = np.clip(sampled, limits[:, 0], limits[:, 1])
        return sampled

    def _timestamp_us(self, timestamp_dict: dict[str, Any]) -> int:
        if "robot_timestamp_us" in timestamp_dict:
            return int(timestamp_dict["robot_timestamp_us"])
        sec = int(timestamp_dict.get("robot_timestamp_seconds", int(time.time())))
        nsec = int(timestamp_dict.get("robot_timestamp_nanos", 0))
        return sec * 1_000_000 + (nsec // 1_000)

    def _transform_action_to_robot_frame(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).copy()
        action[:3] = self.R_world_to_robot @ action[:3]
        action[3:6] = self.R_world_to_robot @ action[3:6]
        return action

    def _transform_state_to_env_frame(self, state: np.ndarray) -> np.ndarray:
        # state[3:6] is an absolute Euler pose (from quat_to_euler("xyz")),
        # not a small rotation vector, so R @ v is only a small-angle
        # approximation. Compose rotations instead.
        state = np.asarray(state, dtype=np.float64).copy()
        state[:3] = self.R_robot_to_world @ state[:3]
        R_base = R.from_matrix(self.R_robot_to_world)
        R_state = R.from_euler("xyz", state[3:6])
        state[3:6] = (R_base * R_state).as_euler("xyz")
        return state


def serve(
    grpc_port: int = 50061,
    robot_model: str = "vega_1",
    arm_side: str = "left",
    gripper_type: str = "default",
    frame_type: str = "vega_mobile_base",
    control_hz: int = 20,
    use_velocity_feedforward: bool = False,
    base_frame_rotation: Optional[list[float]] = None,
    ik_solver_type: str = "pink",
    robotiq_comport: str = "/dev/ttyUSB0",
    ema_alpha: float = 0.0,
    ik_damping_default: float = 1e-3,
    ik_damping_torso: float = 30000.0,
    ik_damping_arm_j2: float = 100.0,
    ik_damping_arm_j3: float = 50.0,
    interpolation_method: str = "none",
    interpolation_history: int = 4,
    control_loop_hz: int = 0,
    filter_type: str = "none",
    filter_cutoff_freq: float = 10.0,
    filter_order: int = 2,
    filter_ema_alpha: float = 0.1,
    vel_smoothing_alpha: float = 0.3,
    hw_correction_alpha: float = 0.7,
    max_delta_scale: float = 1.0,
    max_jerk: float = 0.25,
    rot_sensitivity: float = 1.0,
    vel_ratio: float = 1.0,
    vel_damp_thresh: float = 0.05,
    head_init_pos: list[float] | None = None,
    **kwargs,
) -> None:
    """Start Vega RobotEnv gRPC server."""
    hand_type = kwargs.pop("hand_type", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")
    if hand_type is not None and gripper_type == "default":
        gripper_type = hand_type

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    service = VegaRobotEnvService(
        robot_model=robot_model,
        arm_side=arm_side,
        gripper_type=gripper_type,
        frame_type=frame_type,
        control_hz=control_hz,
        use_velocity_feedforward=use_velocity_feedforward,
        base_frame_rotation=base_frame_rotation,
        ik_solver_type=ik_solver_type,
        robotiq_comport=robotiq_comport,
        ema_alpha=ema_alpha,
        ik_damping_default=ik_damping_default,
        ik_damping_torso=ik_damping_torso,
        ik_damping_arm_j2=ik_damping_arm_j2,
        ik_damping_arm_j3=ik_damping_arm_j3,
        interpolation_method=interpolation_method,
        interpolation_history=interpolation_history,
        control_loop_hz=control_loop_hz,
        filter_type=filter_type,
        filter_cutoff_freq=filter_cutoff_freq,
        filter_order=filter_order,
        filter_ema_alpha=filter_ema_alpha,
        vel_smoothing_alpha=vel_smoothing_alpha,
        hw_correction_alpha=hw_correction_alpha,
        max_delta_scale=max_delta_scale,
        max_jerk=max_jerk,
        rot_sensitivity=rot_sensitivity,
        vel_ratio=vel_ratio,
        vel_damp_thresh=vel_damp_thresh,
        head_init_pos=head_init_pos,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    robotenv_pb2_grpc.add_RobotEnvServicer_to_server(service, server)

    server_address = f"0.0.0.0:{grpc_port}"
    server.add_insecure_port(server_address)

    server.start()
    LOGGER.info(
        "Vega RobotEnv server started on %s (arm=%s, model=%s)",
        server_address,
        arm_side,
        robot_model,
    )

    def shutdown_handler(signum, frame):
        del signum, frame
        LOGGER.info("Shutting down Vega RobotEnv server")
        try:
            service._stop_control_loop()
        except Exception:
            pass
        try:
            service._robot.close()
        except Exception:
            pass
        server.stop(grace=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vega RobotEnv gRPC Server")
    parser.add_argument("--grpc-port", type=int, default=50061, help="gRPC service port")
    parser.add_argument(
        "--robot-model",
        type=str,
        default="vega_1",
        help="Robot model (e.g. vega_1, vega_1_f5d6, vega_1u)",
    )
    parser.add_argument(
        "--arm-side",
        type=str,
        required=True,
        choices=["left", "right"],
        help="Which arm this server controls",
    )
    parser.add_argument(
        "--gripper-type",
        type=str,
        default="default",
        help="Gripper type metadata",
    )
    parser.add_argument(
        "--hand-type",
        dest="gripper_type",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--frame-type",
        type=str,
        default="vega_mobile_base",
        choices=["vega_mobile_base", "vega_table_mount", "vega_custom"],
        help="Robot mounting frame type",
    )
    parser.add_argument("--control-hz", type=int, default=20, help="Control frequency in Hz")
    parser.add_argument(
        "--use-velocity-feedforward",
        action="store_true",
        help="Send joint position and velocity feedforward together (pos+vel) instead of position-only arm commands",
    )
    parser.add_argument(
        "--base-frame-rotation",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Custom base-frame rotation in degrees",
    )
    parser.add_argument(
        "--ik-solver",
        type=str,
        default="pink",
        choices=["pink", "placo"],
        help="IK solver backend (default: pink)",
    )
    parser.add_argument(
        "--robotiq-comport",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port for Robotiq gripper when --gripper-type=robotiq (default: /dev/ttyUSB0). "
             "For SR gripper, prefer --gripper-iface; this flag is still accepted as a fallback.",
    )
    parser.add_argument(
        "--gripper-iface",
        type=str,
        default=None,
        help="EtherCAT network interface for SR gripper (e.g. enx00e04c680093). "
             "Ignored for non-EtherCAT grippers.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.0,
        help="Smoothing responsiveness for joint commands (0.0=disabled, 0.3~0.8=typical). "
             "Uses a critically damped 2nd-order filter: higher = faster tracking, "
             "lower = smoother. No overshoot at any setting. (default: 0.0)",
    )
    parser.add_argument(
        "--ik-damping-default",
        type=float,
        default=1e-3,
        help="Pink IK default damping weight (default: 1e-3)",
    )
    parser.add_argument(
        "--ik-damping-torso",
        type=float,
        default=30000.0,
        help="Pink IK damping override for torso_j1~j3 (default: 30000)",
    )
    parser.add_argument(
        "--ik-damping-arm-j2",
        type=float,
        default=100.0,
        help="Pink IK damping override for L/R_arm_j2 (default: 100)",
    )
    parser.add_argument(
        "--ik-damping-arm-j3",
        type=float,
        default=50.0,
        help="Pink IK damping override for L/R_arm_j3 (default: 50)",
    )
    # --- Interpolation & filtering ---
    parser.add_argument(
        "--interpolation-method",
        type=str,
        default="none",
        choices=["none", "linear", "cubic"],
        help="Trajectory interpolation method for input→control rate upsampling. "
             "'cubic' uses PCHIP (monotone, C1-continuous, no overshoot). "
             "(default: none = legacy synchronous path)",
    )
    parser.add_argument(
        "--interpolation-history",
        type=int,
        default=4,
        help="Number of past command points to keep for interpolation (default: 4)",
    )
    parser.add_argument(
        "--control-loop-hz",
        type=int,
        default=0,
        help="Control loop frequency for interpolation upsampling. "
             "Set to e.g. 100 when using --interpolation-method. "
             "0 = no background loop (default: 0)",
    )
    parser.add_argument(
        "--filter-type",
        type=str,
        default="none",
        choices=["none", "butterworth", "ema"],
        help="Output filter applied after interpolation/EMA smoothing. "
             "(default: none)",
    )
    parser.add_argument(
        "--filter-cutoff-freq",
        type=float,
        default=10.0,
        help="Butterworth filter cutoff frequency in Hz (default: 10.0)",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=2,
        help="Butterworth filter order (default: 2)",
    )
    parser.add_argument(
        "--filter-ema-alpha",
        type=float,
        default=0.1,
        help="EMA filter alpha (0=fully smooth, 1=no smoothing) when --filter-type=ema (default: 0.1)",
    )
    parser.add_argument(
        "--vel-smoothing-alpha",
        type=float,
        default=0.3,
        help="Velocity feedforward EMA factor (0=fully smooth, 1=no smoothing, default: 0.3)",
    )
    parser.add_argument(
        "--hw-correction-alpha",
        type=float,
        default=0.7,
        help="HW feedback correction blend (0=ignore hw, 1=snap to hw, default: 0.7)",
    )
    parser.add_argument(
        "--max-delta-scale",
        type=float,
        default=1.0,
        help="Scale factor for per-joint max delta clipping (>1=faster, <1=slower, default: 1.0)",
    )
    parser.add_argument(
        "--max-jerk",
        type=float,
        default=0.25,
        help="Max jerk (rad/step^2) for acceleration limiting (0=disable, default: 0.25)",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="RPY rotation sensitivity multiplier (>1=more sensitive, default: 1.0)",
    )
    parser.add_argument(
        "--vel-ratio",
        type=float,
        default=1.0,
        help="Velocity feedforward ratio (0.5=half speed, 1.0=full, default: 1.0)",
    )
    parser.add_argument(
        "--vel-damp-thresh",
        type=float,
        default=0.05,
        help="Velocity damping threshold in rad. Velocity tapers linearly from 0→100%% over this distance to target (default: 0.05). Smaller=less damping, larger=more damping.",
    )
    parser.add_argument(
        "--head-init-pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("YAW", "PITCH", "ROLL"),
        help="Head joint init position in radians [yaw, pitch, roll]. "
             "If omitted, defaults to [2.0, 0.0, -0.3].",
    )
    args = parser.parse_args()

    # Both flags feed the same downstream "where is the gripper attached"
    # slot. --gripper-iface is the canonical name for SR; --robotiq-comport
    # is kept for backwards compatibility and as a generic fallback.
    gripper_addr = args.gripper_iface or args.robotiq_comport

    serve(
        grpc_port=args.grpc_port,
        robot_model=args.robot_model,
        arm_side=args.arm_side,
        gripper_type=args.gripper_type,
        frame_type=args.frame_type,
        control_hz=args.control_hz,
        use_velocity_feedforward=args.use_velocity_feedforward,
        base_frame_rotation=args.base_frame_rotation,
        ik_solver_type=args.ik_solver,
        robotiq_comport=gripper_addr,
        ema_alpha=args.ema_alpha,
        ik_damping_default=args.ik_damping_default,
        ik_damping_torso=args.ik_damping_torso,
        ik_damping_arm_j2=args.ik_damping_arm_j2,
        ik_damping_arm_j3=args.ik_damping_arm_j3,
        interpolation_method=args.interpolation_method,
        interpolation_history=args.interpolation_history,
        control_loop_hz=args.control_loop_hz,
        filter_type=args.filter_type,
        filter_cutoff_freq=args.filter_cutoff_freq,
        filter_order=args.filter_order,
        filter_ema_alpha=args.filter_ema_alpha,
        vel_smoothing_alpha=args.vel_smoothing_alpha,
        hw_correction_alpha=args.hw_correction_alpha,
        max_delta_scale=args.max_delta_scale,
        max_jerk=args.max_jerk,
        rot_sensitivity=args.rot_sensitivity,
        vel_ratio=args.vel_ratio,
        vel_damp_thresh=args.vel_damp_thresh,
        head_init_pos=args.head_init_pos,
    )


if __name__ == "__main__":
    main()
