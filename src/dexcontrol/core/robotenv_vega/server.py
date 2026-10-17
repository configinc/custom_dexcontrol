"""Vega RobotEnv gRPC Service."""

from __future__ import annotations

import argparse
import functools
import logging
import os
import signal
import sys
import threading
import time
from concurrent import futures
from contextlib import nullcontext
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

from core.robotenv_vega.force_calibration import (  # noqa: E402
    wrist_ft_calibration_from_snapshot,
)
from core.robotenv_vega.force_calibration_lifecycle import (  # noqa: E402
    BilateralCalibrationPlan,
    ForceCalibrationLifecycle,
)
from core.robotenv_vega.vega_force_calibration import (  # noqa: E402
    VegaForceCalibrationManager,
    VegaRobotCalibrationAdapter,
    parameters_json,
    quality_json,
)
from core.robotenv_vega.wrench_contract import (  # noqa: E402
    vega_wrench_frame_rotations,
)
from core.vega.robot import (  # noqa: E402
    CommunicationFailedError,
    IKFailedError,
    JointLimitExceededError,
    VegaRobot,
)

from proto import robotenv_pb2, robotenv_pb2_grpc  # noqa: E402

LOGGER = logging.getLogger("robotenv_vega")


def _track_transport_rpc(response_type):
    """Keep the robot transport alive for the full lifetime of an RPC."""

    def decorate(method):
        @functools.wraps(method)
        def wrapped(self, request, context):
            condition = self._transport_rpc_condition
            with condition:
                if self._shutdown_started.is_set():
                    context.set_code(grpc.StatusCode.UNAVAILABLE)
                    context.set_details("robot server is shutting down")
                    return response_type()
                self._active_transport_rpcs += 1
            try:
                return method(self, request, context)
            finally:
                with condition:
                    self._active_transport_rpcs -= 1
                    if self._active_transport_rpcs == 0:
                        condition.notify_all()

        return wrapped

    return decorate


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
    "left":  np.array([-0.9548,  0.9862, -0.3738, -1.4169,  0.6624, -0.2987, -0.0650]),
    "right": np.array([ 0.9548, -0.9862,  0.3738, -1.4169, -0.6624,  0.2987,  0.0650]),
}


class VegaRobotEnvService(robotenv_pb2_grpc.RobotEnvServicer):
    """RobotEnv service implementation for one Vega arm."""

    _WRENCH_WARNING_INTERVAL_SECONDS = 10.0
    _WRENCH_ROBOT_MAX_SKEW_SECONDS = 0.25
    _WRENCH_SAMPLE_MAX_STALL_SECONDS = 0.25

    def __init__(
        self,
        robot_model: str = "vega_1",
        arm_side: str = "left",
        gripper_type: str = "default",
        frame_type: str = "vega_mobile_base",
        control_hz: int = 20,
        use_velocity_feedforward: bool = False,
        base_frame_rotation: Optional[list[float]] = None,
        external_wrench_sensor_to_wrist_yaw_degrees: float | None = None,
        ik_solver_type: str = "pink",
        robotiq_comport: str = "/dev/ttyUSB0",
        wuji_sn: Optional[str] = None,
        wuji_effort_limit: Optional[float] = None,
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
        head_init_pos: tuple[float, ...] | list[float] = (2.0, 0.0, -0.3),  # head_j1 limit: ±1.483 rad
        force_calibration_gripper_mass_kg: float = 1.2,
        force_calibration_gripper_com_m: tuple[float, ...] | list[float] = (0.0, 0.0, 0.072),
        force_calibration_storage_directory: str | None = None,
        **kwargs,
    ):
        hand_type = kwargs.pop("hand_type", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if hand_type is not None and gripper_type == "default":
            gripper_type = hand_type

        # The F5D6 5-finger hand is an integrated robot component (left_hand/
        # right_hand) that only exists in the *_f5d6 robot config. Robotiq/SR/
        # Wuji grippers are separate hardware and use the base vega_1 config.
        # So when the hand is vega_hand, ensure robot_model carries the f5d6
        # variant — otherwise has_component('left_hand') is False, the hand is
        # never wired up, and the hand DoF collapses to 1.
        if gripper_type == "vega_hand" and "f5d6" not in robot_model:
            robot_model = f"{robot_model}_f5d6"
            LOGGER.info("[ModelResolve] gripper_type=vega_hand -> robot_model=%s", robot_model)

        self.robot_model = robot_model
        self.arm_side = arm_side
        self.gripper_type = gripper_type
        self.frame_type = frame_type
        self.control_hz = int(control_hz)
        self.use_velocity_feedforward = bool(use_velocity_feedforward)
        self.base_frame_rotation = base_frame_rotation
        self.external_wrench_sensor_to_wrist_yaw_degrees = (
            None
            if external_wrench_sensor_to_wrist_yaw_degrees is None
            else float(external_wrench_sensor_to_wrist_yaw_degrees)
        )
        if (
            self.external_wrench_sensor_to_wrist_yaw_degrees is not None
            and not np.isfinite(self.external_wrench_sensor_to_wrist_yaw_degrees)
        ):
            raise ValueError(
                "external_wrench_sensor_to_wrist_yaw_degrees must be finite"
            )
        if self.external_wrench_sensor_to_wrist_yaw_degrees is None:
            LOGGER.info(
                "publishing source-native external_wrench_sensor_raw; "
                "RobotEnv publishes calibrated wrist/world after calibration"
            )
        else:
            LOGGER.warning(
                "ignoring deprecated sensor-to-wrist yaw %.6f degrees; "
                "RobotEnv owns the calibrated wrist/world transform",
                self.external_wrench_sensor_to_wrist_yaw_degrees,
            )
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
            wuji_sn=wuji_sn,
            wuji_effort_limit=wuji_effort_limit,
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
        self._command_epoch_lock = threading.Lock()
        self._force_calibration_owner_lock = threading.Lock()
        self._force_calibration_owns_robot = threading.Event()
        self._force_calibration_applier_lock = threading.Lock()
        self._force_calibration_applier = None
        self._command_epoch = 0
        self._reset_in_progress = threading.Event()
        self._reset_state_lock = threading.Lock()
        self._active_reset_count = 0
        self._active_reset_cancel = None
        self._reset_restart_control_loop = False
        self._shutdown_started = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._shutdown_completed = False
        self._transport_rpc_condition = threading.Condition()
        self._active_transport_rpcs = 0
        if self._robot.interpolation_enabled and self._control_loop_hz > 0:
            self._start_control_loop()
            LOGGER.info(
                "Control loop thread started: %d Hz (input≈%d Hz → control=%d Hz)",
                self._control_loop_hz,
                control_hz,
                self._control_loop_hz,
            )
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

        # Startup is deliberately motionless.  A single arm may restart while
        # its peer is still in a coordinated calibration pose; moving here
        # would bypass the bilateral RCI barrier.  The coordinator performs an
        # explicit paired Reset before teleoperation or recalibration.
        LOGGER.info(
            "Startup leaves arm at its measured pose; waiting for an explicit "
            "coordinated Reset (arm=%s)",
            arm_side,
        )

        # Fetch and cache robot firmware version once at startup.
        self._firmware_version = self._fetch_firmware_version()

        plan_path = (
            Path(__file__).resolve().parent
            / "force_calibration"
            / "vega_bilateral_plan_v1.json"
        )
        calibration_storage = Path(
            force_calibration_storage_directory
            or Path.home()
            / ".config"
            / "dexcontrol"
            / "force-calibration"
            / arm_side
        ).expanduser()
        plan = BilateralCalibrationPlan.load(plan_path, arm_side)
        lifecycle = ForceCalibrationLifecycle(
            plan=plan,
            server_identity={
                "robot_model": self.robot_model,
                "arm_side": self.arm_side,
                "gripper_type": self.gripper_type,
                "frame_type": self.frame_type,
                "firmware_version": self._firmware_version,
                "sensor_to_wrist_yaw_degrees": 165.0,
                "gripper_mass_kg": float(force_calibration_gripper_mass_kg),
                "gripper_com_sensor_m": [
                    float(value) for value in force_calibration_gripper_com_m
                ],
            },
            storage_directory=calibration_storage,
            calibration_kind="vega_wrist_ft_v1",
            reference_point_id="physical_ft_sensor_measurement_origin",
            validate_parameters=lambda value: wrist_ft_calibration_from_snapshot(value),
        )
        adapter = VegaRobotCalibrationAdapter(
            service=self,
            gripper_mass_kg=force_calibration_gripper_mass_kg,
            gripper_com_sensor_m=np.asarray(
                force_calibration_gripper_com_m, dtype=np.float64
            ),
        )
        self._force_calibration_manager = VegaForceCalibrationManager(
            lifecycle=lifecycle,
            adapter=adapter,
            claim_robot=self._claim_robot_for_force_calibration,
            release_robot=self._release_robot_from_force_calibration,
        )

    # ------------------------------------------------------------------
    # Background control loop (for interpolation upsampling)
    # ------------------------------------------------------------------

    def _start_control_loop(self) -> None:
        """Start the background control-loop thread."""
        if self._control_loop_thread is not None:
            if self._control_loop_thread.is_alive():
                if self._control_loop_stop.is_set():
                    raise RuntimeError("Vega control loop is still stopping")
                return
            self._control_loop_thread = None
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
        thread = self._control_loop_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("Vega control loop did not stop within 2 seconds")
        self._control_loop_thread = None

    def _control_loop_run(self) -> None:
        """Main loop: dispatch interpolated commands at control rate."""
        dt = 1.0 / self._control_loop_hz
        LOGGER.info("Control loop running at %d Hz", self._control_loop_hz)
        while not self._control_loop_stop.is_set():
            t0 = time.perf_counter()
            try:
                if (
                    not self._reset_in_progress.is_set()
                    and not self._force_calibration_owns_robot.is_set()
                ):
                    self._robot.execute_interpolated_tick()
            except Exception as exc:
                LOGGER.error("Control loop tick error: %s", exc)
            elapsed = time.perf_counter() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _claim_robot_for_force_calibration(self) -> bool:
        with self._force_calibration_owner_lock:
            if self._shutdown_started.is_set():
                return False
            if self._force_calibration_owns_robot.is_set() or self._reset_in_progress.is_set():
                return False
            if not self._cmd_lock.acquire(blocking=False):
                return False
            try:
                if self._reset_in_progress.is_set():
                    return False
                self._force_calibration_owns_robot.set()
                with self._command_epoch_lock:
                    self._command_epoch += 1
                self._force_calibration_restart_control_loop = bool(
                    self._control_loop_thread is not None
                )
                if self._force_calibration_restart_control_loop:
                    try:
                        self._stop_control_loop()
                    except Exception:
                        self._force_calibration_restart_control_loop = False
                        self._force_calibration_owns_robot.clear()
                        raise
                return True
            finally:
                self._cmd_lock.release()

    def _release_robot_from_force_calibration(self) -> None:
        with self._force_calibration_owner_lock:
            self._force_calibration_owns_robot.clear()
            if getattr(self, "_force_calibration_restart_control_loop", False):
                self._force_calibration_restart_control_loop = False
                if not self._shutdown_started.is_set():
                    self._start_control_loop()

    def _set_force_calibration_applier(self, applier) -> None:
        with self._force_calibration_applier_lock:
            self._force_calibration_applier = applier

    def _reject_if_shutting_down(self, context) -> bool:
        shutting_down = getattr(self, "_shutdown_started", None)
        if shutting_down is None or not shutting_down.is_set():
            return False
        context.set_code(grpc.StatusCode.UNAVAILABLE)
        context.set_details("robot server is shutting down")
        return True

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
                "force_signal_provider_kind": "wrist_force_torque",
                "force_signal_provider_schema_version": "2",
                "force_signal_stream_id": f"{self.robot_model}:{self.arm_side}:wrist_ft",
                "external_wrench_world_frame_id": self.frame_type,
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
        # Multi-DoF hand → joint vector; scalar gripper → unchanged scalar spec.
        _hand_dof = self._robot._hand_dof()
        if _hand_dof > 1:
            spec.fields["gripper_position"].CopyFrom(
                robotenv_pb2.FieldSpec(
                    dtype="float64",
                    shape=[_hand_dof],
                    required=True,
                    description="Hand joint positions in radians (finger-major)",
                )
            )
        else:
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
        spec.fields["external_wrench_sensor_raw"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64",
                shape=[6],
                required=False,
                description=(
                    "Source-native uncalibrated F/T sample expressed in sensor axes "
                    "at the physical sensor origin [fx, fy, fz, tx, ty, tz] "
                    "(N, N, N, Nm, Nm, Nm); wrench_on_robot sign"
                ),
            )
        )
        spec.fields["external_wrench_sensor_raw_reference_point"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="string",
                shape=[],
                required=False,
                description="Physical reference point of external_wrench_sensor_raw",
            )
        )
        spec.fields["external_wrench_sensor_raw_timestamp_ns"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="int64",
                shape=[],
                required=False,
                description=(
                    "Acquisition timestamp from the same source message as "
                    "external_wrench_sensor_raw"
                ),
            )
        )
        for field_name, description in (
            (
                "external_wrench_wrist_from_sensor_rotation",
                "3x3 row-major rotation from physical sensor axes to company wrist axes",
            ),
            (
                "external_wrench_world_from_sensor_rotation",
                "3x3 row-major rotation from physical sensor axes to configured world axes",
            ),
        ):
            spec.fields[field_name].CopyFrom(
                robotenv_pb2.FieldSpec(
                    dtype="float64", shape=[9], required=False, description=description
                )
            )
        for field_name, description in (
            (
                "external_wrench_wrist",
                "Calibrated gripper-compensated wrench in company wrist axes at the configured TCP anchor",
            ),
            (
                "external_wrench_world",
                "Calibrated gripper-compensated wrench in configured world axes at the configured TCP anchor",
            ),
        ):
            spec.fields[field_name].CopyFrom(
                robotenv_pb2.FieldSpec(
                    dtype="float64", shape=[6], required=False, description=description
                )
            )
            spec.fields[f"{field_name}_reference_point"].CopyFrom(
                robotenv_pb2.FieldSpec(
                    dtype="string",
                    shape=[],
                    required=False,
                    description="Configured TCP anchor; rotation only, no moment translation",
                )
            )
        spec.fields["torso_joint_positions"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64", shape=[3], required=False,
                description="Measured torso_j1..torso_j3 positions in radians",
            )
        )
        spec.fields["head_joint_positions"].CopyFrom(
            robotenv_pb2.FieldSpec(
                dtype="float64", shape=[3], required=False,
                description="Measured head_j1..head_j3 positions in radians",
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
        # Tactile: declared only when the hand actually streams it, with the
        # runtime-detected size (wuji_hand_v2 fingertip aggregate = 15,
        # wuji_hand_v1 tactile glove pooled = 40, F5D6_V2 fingertip force = 5).
        # Scalar-gripper specs are unchanged.
        _tactile = None
        if hasattr(self._robot.hand, "get_hand_tactile"):
            _tactile = self._robot.hand.get_hand_tactile()
            if _tactile is None and getattr(self._robot.hand, "has_tactile", False):
                # Stream wired up but no frame yet — report the pinned
                # per-model size instead of dropping the field.
                _tactile = np.zeros(
                    15 if getattr(self._robot.hand, "model", "") == "wuji_hand_v2" else 40
                )
        elif self.gripper_type == "vega_hand" and hasattr(
            self._robot.hand, "get_finger_tip_force"
        ):
            # dexbot F5D6_V2 fingertip force. Gated on the accessor existing:
            # an F5D6 V1 has no tactile at all, and advertising a field the
            # observation never carries makes the spec lie to the client.
            _tactile = np.zeros(5)
        if _tactile is not None:
            spec.fields["hand_tactile"].CopyFrom(
                robotenv_pb2.FieldSpec(
                    dtype="float64",
                    shape=[int(np.asarray(_tactile).size)],
                    required=False,
                    description="Hand tactile vector (per-model layout)",
                )
            )
        # Report the auto-detected hand model (wuji_hand_v1 vs wuji_hand_v2) for
        # logging/debug; absent for non-wuji end-effectors.
        _model = getattr(self._robot.hand, "model", None)
        if _model is not None:
            spec.metadata["hand_model"] = str(_model)
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
                "force_signal_provider_kind": "wrist_force_torque",
                "force_signal_provider_schema_version": "2",
                "force_signal_stream_id": f"{self.robot_model}:{self.arm_side}:wrist_ft",
                "external_wrench_world_frame_id": self.frame_type,
                "external_wrench_sensor_to_wrist_yaw_degrees": (
                    "unset"
                    if self.external_wrench_sensor_to_wrist_yaw_degrees is None
                    else str(self.external_wrench_sensor_to_wrist_yaw_degrees)
                ),
            },
        )

    _FORCE_CALIBRATION_STATE_PROTO = {
        name: getattr(robotenv_pb2, f"FORCE_CALIBRATION_STATE_{name}")
        for name in (
            "IDLE",
            "READY_FOR_STAGE",
            "MOVING",
            "SETTLING",
            "COLLECTING",
            "FITTING",
            "APPLYING",
            "RETURNING",
            "SUCCEEDED",
            "FAILED",
            "CANCELLING",
            "CANCELLED",
            "READY_TO_APPLY",
        )
    }

    @classmethod
    def _force_calibration_status_proto(cls, status):
        return robotenv_pb2.ForceCalibrationStatus(
            state=cls._FORCE_CALIBRATION_STATE_PROTO[status.state],
            run_id=status.run_id,
            current_stage_index=status.current_stage_index,
            completed_stages=status.completed_stages,
            total_stages=status.total_stages,
            message=status.message,
            arm_name=status.arm_name,
            plan_id=status.plan_id,
            plan_sha256=status.plan_sha256,
            stage_id=status.stage_id,
            server_instance_id=status.server_instance_id,
            cache_reused=status.cache_reused,
            command_deadline_timestamp_us=status.command_deadline_timestamp_us,
        )

    @staticmethod
    def _set_force_calibration_rpc_error(context, exc: BaseException) -> None:
        lifecycle_code = getattr(exc, "code", None)
        if isinstance(lifecycle_code, str) and hasattr(grpc.StatusCode, lifecycle_code):
            code = getattr(grpc.StatusCode, lifecycle_code)
        elif isinstance(exc, (TypeError, ValueError)):
            code = grpc.StatusCode.INVALID_ARGUMENT
        else:
            code = grpc.StatusCode.INTERNAL
        context.set_code(code)
        context.set_details(str(exc) or type(exc).__name__)

    def StartForceCalibration(self, request, context):
        if self._reject_if_shutting_down(context):
            return robotenv_pb2.ForceCalibrationStatus()
        try:
            status = self._force_calibration_manager.start(
                request.run_id, bool(request.force_recalibrate)
            )
            return self._force_calibration_status_proto(status)
        except BaseException as exc:
            LOGGER.exception("StartForceCalibration failed: %s", exc)
            self._set_force_calibration_rpc_error(context, exc)
            return robotenv_pb2.ForceCalibrationStatus()

    def AdvanceForceCalibration(self, request, context):
        if self._reject_if_shutting_down(context):
            return robotenv_pb2.ForceCalibrationStatus()
        try:
            status = self._force_calibration_manager.advance(
                request.run_id, int(request.stage_index), request.stage_id
            )
            return self._force_calibration_status_proto(status)
        except BaseException as exc:
            LOGGER.exception("AdvanceForceCalibration failed: %s", exc)
            self._set_force_calibration_rpc_error(context, exc)
            return robotenv_pb2.ForceCalibrationStatus()

    def CommitForceCalibration(self, request, context):
        if self._reject_if_shutting_down(context):
            return robotenv_pb2.ForceCalibrationStatus()
        try:
            status = self._force_calibration_manager.commit(
                request.run_id, request.plan_sha256
            )
            return self._force_calibration_status_proto(status)
        except BaseException as exc:
            LOGGER.exception("CommitForceCalibration failed: %s", exc)
            self._set_force_calibration_rpc_error(context, exc)
            return robotenv_pb2.ForceCalibrationStatus()

    def GetForceCalibrationStatus(self, request, context):
        try:
            return self._force_calibration_status_proto(
                self._force_calibration_manager.status(request.run_id)
            )
        except BaseException as exc:
            self._set_force_calibration_rpc_error(context, exc)
            return robotenv_pb2.ForceCalibrationStatus()

    def CancelForceCalibration(self, request, context):
        if not request.run_id:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("unknown calibration run_id")
            return robotenv_pb2.ForceCalibrationStatus()
        try:
            status = self._force_calibration_manager.request_cancel(request.run_id)
            return self._force_calibration_status_proto(status)
        except BaseException as exc:
            LOGGER.exception("CancelForceCalibration failed: %s", exc)
            self._set_force_calibration_rpc_error(context, exc)
            return robotenv_pb2.ForceCalibrationStatus()

    def GetForceCalibrationResult(self, request, context):
        try:
            result = self._force_calibration_manager.result(request.run_id)
            if not result.get("available"):
                return robotenv_pb2.ForceCalibrationResult(
                    available=False,
                    run_id=result.get("run_id", ""),
                    message=result.get("message", ""),
                )
            return robotenv_pb2.ForceCalibrationResult(
                available=True,
                calibrated_at_timestamp_us=result["calibrated_at_timestamp_us"],
                expires_at_timestamp_us=result["expires_at_timestamp_us"],
                calibration_kind=result["calibration_kind"],
                parameters_json=parameters_json(result),
                quality_json=quality_json(result),
                message=result.get("message", ""),
                run_id=result["run_id"],
                schema_version=result["schema_version"],
                reference_point_id=result["reference_point_id"],
                arm_name=result["arm_name"],
                plan_id=result["plan_id"],
                plan_sha256=result["plan_sha256"],
                server_instance_id=result["server_instance_id"],
                source_run_id=result["source_run_id"],
                was_committed=result["was_committed"],
            )
        except BaseException as exc:
            self._set_force_calibration_rpc_error(context, exc)
            return robotenv_pb2.ForceCalibrationResult()

    @_track_transport_rpc(robotenv_pb2.ResetResponse)
    def Reset(self, request, context):
        if self._reject_if_shutting_down(context):
            return robotenv_pb2.ResetResponse()
        mode = request.mode or "home"
        LOGGER.info("Reset requested: mode=%s arm=%s", mode, self.arm_side)
        calibration_owner = getattr(self, "_force_calibration_owns_robot", None)
        if (
            mode == "current"
            and calibration_owner is not None
            and calibration_owner.is_set()
        ):
            try:
                observation, timestamp_us = self._create_observation()
                return robotenv_pb2.ResetResponse(
                    observation=observation,
                    status="SUCCESS",
                    message="Reset to current",
                    timestamp_us=timestamp_us,
                )
            except Exception as exc:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Reset failed: {exc}")
                return robotenv_pb2.ResetResponse()
        rpc_cancelled = threading.Event()
        motion_cancelled = threading.Event()

        def cancel_reset_motion() -> None:
            rpc_cancelled.set()
            motion_cancelled.set()

        if hasattr(context, "add_callback") and not context.add_callback(cancel_reset_motion):
            cancel_reset_motion()

        # Establish reset ownership before waiting for the command lock. A new
        # reset cancels the previous reset's local motion loop, while the
        # shared in-progress event stays set until every superseded handler has
        # unwound.
        owner_guard = getattr(self, "_force_calibration_owner_lock", None)
        with owner_guard if owner_guard is not None else nullcontext():
            calibration_owner = getattr(self, "_force_calibration_owns_robot", None)
            if (
                mode != "current"
                and calibration_owner is not None
                and calibration_owner.is_set()
            ):
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("force calibration owns the robot command path")
                return robotenv_pb2.ResetResponse()
            with self._reset_state_lock:
                if self._active_reset_cancel is not None:
                    self._active_reset_cancel.set()
                self._active_reset_cancel = motion_cancelled
                self._active_reset_count += 1
                self._reset_in_progress.set()
                with self._command_epoch_lock:
                    self._command_epoch += 1

        was_running = False
        try:
            with self._cmd_lock:
                if self._shutdown_started.is_set():
                    raise RuntimeError("Reset was cancelled because the robot server is shutting down")
                if motion_cancelled.is_set():
                    raise RuntimeError("Reset was cancelled before robot motion started")
                calibration_cancellable = bool(
                    request.params.get("calibration_cancellable", robotenv_pb2.Value()).int_value
                )
                # Pause the control loop while this reset owns robot commands.
                was_running = self._control_loop_thread is not None and self._control_loop_thread.is_alive()
                if was_running:
                    with self._reset_state_lock:
                        self._reset_restart_control_loop = True
                    self._stop_control_loop()
                if mode == "home":
                    target_joints = self.reset_joints
                    self._execute_reset_sequence(
                        target_joints,
                        cancelled=motion_cancelled,
                        interruptible_gripper=calibration_cancellable,
                    )
                elif mode == "target":
                    target_joints = self._extract_target_joints(request)
                    self._execute_reset_sequence(
                        target_joints,
                        cancelled=motion_cancelled,
                        interruptible_gripper=calibration_cancellable,
                    )
                elif mode == "random":
                    target_joints = self._sample_random_target()
                    self._execute_reset_sequence(
                        target_joints,
                        cancelled=motion_cancelled,
                        interruptible_gripper=calibration_cancellable,
                    )
                elif mode == "current":
                    pass
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(f"Unknown reset mode: {mode}")
                    return robotenv_pb2.ResetResponse()
                if motion_cancelled.is_set():
                    raise RuntimeError("Reset was cancelled during robot motion")

                observation, timestamp_us = self._create_observation()
                # Linearize success against registration of a newer Reset.
                # A newer owner either sets our cancellation event before
                # acquiring the command lock or cannot register until after
                # this response has been committed as the current owner.
                with self._reset_state_lock:
                    if (
                        motion_cancelled.is_set()
                        or self._active_reset_cancel is not motion_cancelled
                    ):
                        raise RuntimeError("Reset was superseded before its response was committed")
                    return robotenv_pb2.ResetResponse(
                        observation=observation,
                        status="SUCCESS",
                        message=f"Reset to {mode}",
                        timestamp_us=timestamp_us,
                    )
        except Exception as exc:
            LOGGER.exception("Reset failed (mode=%s): %s", mode, exc)
            if rpc_cancelled.is_set():
                pass
            elif motion_cancelled.is_set():
                context.set_code(grpc.StatusCode.ABORTED)
                context.set_details("Reset was superseded by a newer reset request")
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Reset failed: {exc}")
            return robotenv_pb2.ResetResponse()
        finally:
            restart_control_loop = False
            with self._reset_state_lock:
                self._active_reset_count -= 1
                if self._active_reset_cancel is motion_cancelled:
                    self._active_reset_cancel = None
                if self._active_reset_count == 0:
                    self._reset_in_progress.clear()
                    restart_control_loop = self._reset_restart_control_loop
                    self._reset_restart_control_loop = False
            if restart_control_loop and not self._shutdown_started.is_set():
                try:
                    self._start_control_loop()
                except RuntimeError:
                    # A timed-out old loop still owns the handle.  Keep the
                    # restart intent so the next Step can recover after it exits.
                    with self._reset_state_lock:
                        self._reset_restart_control_loop = True

    @_track_transport_rpc(robotenv_pb2.StepResponse)
    def Step(self, request, context):
        if self._reject_if_shutting_down(context):
            return robotenv_pb2.StepResponse()
        calibration_owner = getattr(self, "_force_calibration_owns_robot", None)
        if calibration_owner is not None and calibration_owner.is_set():
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("force calibration owns the robot command path")
            return robotenv_pb2.StepResponse()
        rpc_cancelled = threading.Event()
        if hasattr(context, "add_callback") and not context.add_callback(rpc_cancelled.set):
            rpc_cancelled.set()
        with self._reset_state_lock:
            reset_active_at_entry = self._reset_in_progress.is_set()
            with self._command_epoch_lock:
                command_epoch = self._command_epoch
        action_space = request.action_space
        action_space_for_robot = action_space
        action = np.asarray(request.action, dtype=np.float64)
        gripper_action_space = request.gripper_action_space
        if not gripper_action_space:
            gripper_action_space = "velocity" if "velocity" in action_space else "position"

        try:
            if reset_active_at_entry:
                raise RuntimeError("Step was admitted while robot reset was active")
            # -- gripper debug --
            if self._robot.hand is not None:
                _cur_grip_joint = self._robot.get_cached_gripper_joint_pos()
                _cur_grip_raw = float(_cur_grip_joint[0]) if _cur_grip_joint.size > 0 else 0.0
                _cur_grip_norm = float(self._robot.get_cached_gripper_position())
            else:
                _cur_grip_raw = 0.0
                _cur_grip_norm = 0.0
            _arm_dof = 7 if action_space.startswith("joint") else 6
            # End-effector slice: scalar gripper (1) or multi-finger hand (N).
            _hand_cmd = action[_arm_dof:].tolist() if action.shape[0] > _arm_dof else []
            LOGGER.debug(
                "[Gripper] cmd=%s  space=%s  gripper_space=%s  cur_raw=%.4f  cur_norm=%.4f",
                _hand_cmd, action_space, gripper_action_space, _cur_grip_raw, _cur_grip_norm,
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
                LOGGER.debug(
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
            LOGGER.debug(
                "[CartesianPos] %s",
                np.round(np.asarray(pre_action_state["cartesian_position"], dtype=np.float64), 4).tolist(),
            )

            self._dispatch_step_command(
                action,
                action_space=action_space_for_robot,
                gripper_action_space=gripper_action_space,
                command_epoch=command_epoch,
                rpc_cancelled=rpc_cancelled,
                reset_active_at_entry=reset_active_at_entry,
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

    def _dispatch_step_command(
        self,
        action: np.ndarray,
        *,
        action_space: str,
        gripper_action_space: str,
        command_epoch: int,
        rpc_cancelled: threading.Event | None = None,
        reset_active_at_entry: bool = False,
    ) -> None:
        """Serialize physical Step dispatch with Reset and reject stale work."""
        with self._cmd_lock:
            with self._command_epoch_lock:
                reset_superseded_command = command_epoch != self._command_epoch
            if (
                self._reset_in_progress.is_set()
                or reset_superseded_command
                or reset_active_at_entry
                or (rpc_cancelled is not None and rpc_cancelled.is_set())
                or (
                    getattr(self, "_force_calibration_owns_robot", None) is not None
                    and self._force_calibration_owns_robot.is_set()
                )
                or self._shutdown_started.is_set()
            ):
                raise RuntimeError(
                    "Step was superseded by reset, force calibration, or shutdown"
                )
            if self._robot.interpolation_enabled and self._control_loop_hz > 0:
                if self._control_loop_thread is None or not self._control_loop_thread.is_alive():
                    self._start_control_loop()
                    with self._reset_state_lock:
                        self._reset_restart_control_loop = False
                elif self._control_loop_stop.is_set():
                    raise RuntimeError("Vega control loop is still stopping")
                self._robot.add_command_point(
                    action,
                    action_space=action_space,
                    gripper_action_space=gripper_action_space,
                )
            else:
                self._robot.update_command(
                    action,
                    action_space=action_space,
                    gripper_action_space=gripper_action_space,
                    blocking=False,
                )

    @_track_transport_rpc(robotenv_pb2.HealthCheckResponse)
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
            firmware_version=self._firmware_version,
        )

    def _fetch_firmware_version(self) -> str:
        """Query robot firmware version via Zenoh at server startup and return it.

        Calls Robot.get_version_info() which queries the hardware's JSON version
        endpoint over DexComm/Zenoh. The result is cached in self._firmware_version
        and served through GetVersion RPC for the lifetime of the server process.

        Returns empty string if the firmware version cannot be retrieved.
        """
        try:
            version_info = self._robot.robot.get_version_info(show=False)
            firmware_version = version_info.get("version", "")
            if firmware_version:
                LOGGER.info("Robot firmware version: %s", firmware_version)
            else:
                LOGGER.warning("Firmware version field missing in version_info response")
            return firmware_version
        except Exception as exc:
            LOGGER.warning("Failed to fetch robot firmware version: %s", exc)
            return ""

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
            # Scalar gripper → float_value; multi-finger hand (joint list) →
            # float_array. Same key either way (the consumer keeps it as-is).
            "gripper_position": (
                robotenv_pb2.Value(
                    float_array=robotenv_pb2.FloatArray(
                        values=[float(v) for v in state_dict["gripper_position"]]
                    )
                )
                if isinstance(state_dict["gripper_position"], (list, tuple, np.ndarray))
                else robotenv_pb2.Value(float_value=float(state_dict["gripper_position"]))
            ),
            "cartesian_position": robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(values=cartesian_position.tolist())
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
        wrench_sensor = getattr(getattr(self._robot, "arm", None), "wrench_sensor", None)
        if wrench_sensor is not None:
            try:
                sensor_wrench_raw, sensor_timestamp_ns = (
                    wrench_sensor.get_wrench_sample()
                )
                sensor_wrench = np.asarray(sensor_wrench_raw, dtype=np.float64)
                if sensor_wrench.shape != (6,) or not np.all(np.isfinite(sensor_wrench)):
                    raise ValueError("sensor wrench must contain six finite values")
                if (
                    isinstance(sensor_timestamp_ns, bool)
                    or not isinstance(sensor_timestamp_ns, (int, np.integer))
                    or int(sensor_timestamp_ns) <= 0
                ):
                    raise ValueError("sensor wrench timestamp must be a positive integer")
                sensor_timestamp_ns = int(sensor_timestamp_ns)
            except Exception as exc:
                now = time.monotonic()
                last_warning = getattr(
                    self, "_last_wrench_warning_monotonic", float("-inf")
                )
                if now - last_warning >= self._WRENCH_WARNING_INTERVAL_SECONDS:
                    LOGGER.warning(
                        "Omitting unavailable or malformed source-native F/T "
                        "sample; expected six finite values (%s: %s)",
                        type(exc).__name__,
                        exc,
                    )
                    self._last_wrench_warning_monotonic = now
            else:
                reference_point = "physical F/T sensor measurement origin"
                observation["external_wrench_sensor_raw"] = robotenv_pb2.Value(
                    float_array=robotenv_pb2.FloatArray(
                        values=sensor_wrench.tolist()
                    )
                )
                observation["external_wrench_sensor_raw_reference_point"] = (
                    robotenv_pb2.Value(
                        string_value=reference_point
                    )
                )
                observation["external_wrench_sensor_raw_timestamp_ns"] = (
                    robotenv_pb2.Value(int_value=sensor_timestamp_ns)
                )
                robot_timestamp_ns = int(timestamp_us) * 1_000
                sample_pair = (sensor_timestamp_ns, robot_timestamp_ns)
                now_monotonic = time.monotonic()
                if getattr(self, "_last_wrench_sample_pair", None) != sample_pair:
                    self._last_wrench_sample_pair = sample_pair
                    self._last_wrench_sample_progress_monotonic = now_monotonic
                stalled_for = now_monotonic - getattr(
                    self,
                    "_last_wrench_sample_progress_monotonic",
                    now_monotonic,
                )
                timing_valid = (
                    robot_timestamp_ns > 0
                    and abs(sensor_timestamp_ns - robot_timestamp_ns)
                    <= int(self._WRENCH_ROBOT_MAX_SKEW_SECONDS * 1_000_000_000)
                    and stalled_for <= self._WRENCH_SAMPLE_MAX_STALL_SECONDS
                )
                if not timing_valid:
                    if now_monotonic - getattr(
                        self, "_last_wrench_timing_warning_monotonic", float("-inf")
                    ) >= self._WRENCH_WARNING_INTERVAL_SECONDS:
                        LOGGER.warning(
                            "Omitting calibrated external wrench because source and "
                            "robot-state timestamps are stale or misaligned"
                        )
                        self._last_wrench_timing_warning_monotonic = now_monotonic
                    continue_wrench_processing = False
                else:
                    continue_wrench_processing = True
                if not continue_wrench_processing:
                    wrist_from_sensor = world_from_sensor = None
                else:
                    try:
                        wrist_from_sensor, world_from_sensor = vega_wrench_frame_rotations(
                            cartesian_position, gripper_type=self.gripper_type
                        )
                    except ValueError as exc:
                        LOGGER.warning("Omitting unsupported wrench frame rotations: %s", exc)
                        wrist_from_sensor = world_from_sensor = None
                if wrist_from_sensor is not None and world_from_sensor is not None:
                    observation["external_wrench_wrist_from_sensor_rotation"] = (
                        robotenv_pb2.Value(
                            float_array=robotenv_pb2.FloatArray(
                                values=wrist_from_sensor.reshape(-1).tolist()
                            )
                        )
                    )
                    observation["external_wrench_world_from_sensor_rotation"] = (
                        robotenv_pb2.Value(
                            float_array=robotenv_pb2.FloatArray(
                                values=world_from_sensor.reshape(-1).tolist()
                            )
                        )
                    )
                    applier_lock = getattr(
                        self, "_force_calibration_applier_lock", None
                    )
                    with applier_lock if applier_lock is not None else nullcontext():
                        applier = getattr(self, "_force_calibration_applier", None)
                        if applier is not None:
                            try:
                                wrist_wrench = (
                                    applier.measured_sensor_raw_to_calibrated_wrist(
                                        sensor_wrench,
                                        world_from_sensor,
                                        wrist_from_sensor,
                                        reference_point=reference_point,
                                    )
                                )
                                world_wrench = (
                                    applier.measured_sensor_raw_to_calibrated_world(
                                        sensor_wrench,
                                        world_from_sensor,
                                        reference_point=reference_point,
                                    )
                                )
                            except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
                                LOGGER.warning(
                                    "Omitting calibrated external wrench: %s", exc
                                )
                            else:
                                anchor = "configured TCP anchor (rotation only)"
                                for name, values in (
                                    ("external_wrench_wrist", wrist_wrench),
                                    ("external_wrench_world", world_wrench),
                                ):
                                    observation[name] = robotenv_pb2.Value(
                                        float_array=robotenv_pb2.FloatArray(
                                            values=values.tolist()
                                        )
                                    )
                                    observation[f"{name}_reference_point"] = (
                                        robotenv_pb2.Value(string_value=anchor)
                                    )
        for field_name in ("torso_joint_positions", "head_joint_positions"):
            if field_name in state_dict:
                observation[field_name] = robotenv_pb2.Value(
                    float_array=robotenv_pb2.FloatArray(
                        values=np.asarray(state_dict[field_name], dtype=np.float64).tolist()
                    )
                )
        # Hand tactile, present only when the hand streams it. Added as a
        # float_array under its own key when get_robot_state() supplied it;
        # scalar grippers never set "hand_tactile" so the observation is
        # byte-identical to before for them. Clients that don't know the key
        # ignore it (proto map).
        if state_dict.get("hand_tactile") is not None:
            observation["hand_tactile"] = robotenv_pb2.Value(
                float_array=robotenv_pb2.FloatArray(
                    values=[float(v) for v in state_dict["hand_tactile"]]
                )
            )
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
    _RESET_GRIPPER_TIMEOUT_S = 5.0
    _RESET_GRIPPER_TOLERANCE = 0.05

    def _execute_reset_sequence(
        self,
        target_joints: np.ndarray,
        *,
        cancelled: threading.Event | None = None,
        interruptible_gripper: bool = False,
    ) -> None:
        t0 = time.time()
        if cancelled is not None and cancelled.is_set():
            return
        LOGGER.info("Reset[%s]: starting gripper open", self.arm_side)
        self._robot.update_gripper(
            0.0,
            velocity=False,
            blocking=not interruptible_gripper,
        )
        if interruptible_gripper:
            gripper_deadline = time.monotonic() + self._RESET_GRIPPER_TIMEOUT_S
            while abs(float(self._robot.get_cached_gripper_position())) > (
                self._RESET_GRIPPER_TOLERANCE
            ):
                if cancelled is not None and cancelled.is_set():
                    return
                if time.monotonic() >= gripper_deadline:
                    raise RuntimeError("Vega gripper did not open before reset motion")
                time.sleep(0.02)
        if cancelled is not None and cancelled.is_set():
            return
        LOGGER.info("Reset[%s]: gripper done (%.2fs), starting reset motion", self.arm_side, time.time() - t0)
        self._robot.reset_filter_state()

        # Move to middle waypoint to avoid collisions.
        middle_f64 = np.asarray(self.reset_middle_joints, dtype=np.float64)
        LOGGER.info("Reset[%s]: moving to middle waypoint", self.arm_side)
        self._move_incremental(middle_f64, cancelled=cancelled)
        LOGGER.info("Reset[%s]: middle waypoint reached (%.2fs)", self.arm_side, time.time() - t0)

        # Then move to the final target (home/init position).
        target_f64 = np.asarray(target_joints, dtype=np.float64)
        LOGGER.info("Reset[%s]: moving to target", self.arm_side)
        self._move_incremental(target_f64, cancelled=cancelled)
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
        cancelled: threading.Event | None = None,
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
        _tick = 0  # [BUG PROBE]

        while time.time() < deadline:
            if cancelled is not None and cancelled.is_set():
                LOGGER.info("_move_incremental cancelled by new request")
                return

            actual = np.asarray(self._robot.arm.get_joint_pos(), dtype=np.float64)
            # [BUG PROBE] 첫 tick 위치를 기록해 stale 여부를 사후 분석에 활용
            if _tick == 0:
                LOGGER.info(
                    "_move_incremental[%s][BUG_PROBE] tick=0 actual=%s target=%s",
                    self.arm_side, actual.round(4).tolist(), target.round(4).tolist(),
                )
            _tick += 1
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

    def shutdown(self) -> None:
        """Safely stop calibration, command loops, and the robot transport."""
        with self._shutdown_lock:
            if self._shutdown_completed:
                return
            # Close admission before inspecting any owner.  In-flight RPCs
            # either observe this flag before physical dispatch or are drained
            # by the ownership barriers below.
            self._shutdown_started.set()
            with self._command_epoch_lock:
                self._command_epoch += 1
            with self._reset_state_lock:
                if self._active_reset_cancel is not None:
                    self._active_reset_cancel.set()
            manager = getattr(self, "_force_calibration_manager", None)
            if manager is not None:
                manager.shutdown()
            with self._transport_rpc_condition:
                while self._active_transport_rpcs:
                    self._transport_rpc_condition.wait()
            # Reset and Step perform physical dispatch while holding this lock.
            # Acquiring it here proves that every admitted command has unwound.
            with self._cmd_lock:
                self._stop_control_loop()
                self._robot.close()
            self._shutdown_completed = True


def serve(
    grpc_port: int = 50061,
    robot_model: str = "vega_1",
    arm_side: str = "left",
    gripper_type: str = "default",
    frame_type: str = "vega_mobile_base",
    control_hz: int = 20,
    use_velocity_feedforward: bool = False,
    base_frame_rotation: Optional[list[float]] = None,
    external_wrench_sensor_to_wrist_yaw_degrees: float | None = None,
    ik_solver_type: str = "pink",
    robotiq_comport: str = "/dev/ttyUSB0",
    wuji_sn: Optional[str] = None,
    wuji_effort_limit: Optional[float] = None,
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
    head_init_pos: tuple[float, ...] | list[float] = (1.48, 0.0, -0.3),
    force_calibration_gripper_mass_kg: float = 1.2,
    force_calibration_gripper_com_m: tuple[float, ...] | list[float] = (0.0, 0.0, 0.072),
    force_calibration_storage_directory: str | None = None,
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

    _log_path = os.path.expanduser("~/custom_dexcontrol/src/dexcontrol/logs/robotenv_vega.log")
    os.makedirs(os.path.dirname(_log_path), exist_ok=True)
    from core.robotenv_vega.circular_file_handler import CircularFileHandler
    _file_handler = CircularFileHandler(_log_path)
    _file_handler.setLevel(logging.getLogger().level)
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s [%(filename)s:%(lineno)d]: %(message)s"
    ))
    logging.getLogger().addHandler(_file_handler)

    service = VegaRobotEnvService(
        robot_model=robot_model,
        arm_side=arm_side,
        gripper_type=gripper_type,
        frame_type=frame_type,
        control_hz=control_hz,
        use_velocity_feedforward=use_velocity_feedforward,
        base_frame_rotation=base_frame_rotation,
        external_wrench_sensor_to_wrist_yaw_degrees=(
            external_wrench_sensor_to_wrist_yaw_degrees
        ),
        ik_solver_type=ik_solver_type,
        robotiq_comport=robotiq_comport,
        wuji_sn=wuji_sn,
        wuji_effort_limit=wuji_effort_limit,
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
        force_calibration_gripper_mass_kg=force_calibration_gripper_mass_kg,
        force_calibration_gripper_com_m=force_calibration_gripper_com_m,
        force_calibration_storage_directory=force_calibration_storage_directory,
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
        # Stop new transport admission before draining in-flight robot work.
        server.stop(grace=0)
        try:
            service.shutdown()
        except Exception:
            LOGGER.exception(
                "Vega shutdown could not complete measured calibration return"
            )
            return
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
        "--external-wrench-sensor-to-wrist-yaw-degrees",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility option. RobotEnv owns the calibrated "
            "wrist/world transform and ignores this override."
        ),
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
        "--wuji-sn",
        type=str,
        default=None,
        help="Wuji hand serial number when --gripper-type=wuji_hand_v1/v2. Defaults to "
             "auto-selection by handedness (SN 4th char J=left/K=right).",
    )
    parser.add_argument(
        "--wuji-effort-limit",
        type=float,
        default=None,
        help="Wuji hand per-joint current limit in Amps "
             "(default: 1.7 for wuji_hand_v2, 1.5 for wuji_hand_v1).",
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
        default=[2.0, 0.0, -0.3],
        metavar=("YAW", "PITCH", "ROLL"),
        help="Head joint init position in radians [yaw, pitch, roll]. "
             "Defaults to [1.48, 0.0, -0.3] (head_j1 limit: ±1.483 rad).",
    )
    parser.add_argument(
        "--force-calibration-gripper-mass-kg",
        type=float,
        default=1.2,
        help="Installed RobotiQ/tool mass used by server-owned force calibration",
    )
    parser.add_argument(
        "--force-calibration-gripper-com-m",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.072],
        metavar=("X", "Y", "Z"),
        help="Installed RobotiQ/tool center of mass in sensor axes, metres",
    )
    parser.add_argument(
        "--force-calibration-storage-directory",
        type=str,
        default=None,
        help="Durable per-arm force-calibration state directory",
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
        external_wrench_sensor_to_wrist_yaw_degrees=(
            args.external_wrench_sensor_to_wrist_yaw_degrees
        ),
        ik_solver_type=args.ik_solver,
        robotiq_comport=gripper_addr,
        wuji_sn=args.wuji_sn,
        wuji_effort_limit=args.wuji_effort_limit,
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
        force_calibration_gripper_mass_kg=args.force_calibration_gripper_mass_kg,
        force_calibration_gripper_com_m=args.force_calibration_gripper_com_m,
        force_calibration_storage_directory=args.force_calibration_storage_directory,
    )


if __name__ == "__main__":
    main()
