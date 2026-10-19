"""Vega motion adapter and asynchronous force-calibration coordinator."""

from __future__ import annotations

import json
import logging
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .force_calibration import (
    WristFtCalibrationApplier,
    WristFtCalibrationHold,
    WristFtCalibrationParameters,
    fit_wrist_ft_calibration,
    wrist_ft_calibration_from_snapshot,
)
from .force_calibration_lifecycle import (
    CalibrationStatusSnapshot,
    ForceCalibrationLifecycle,
    LifecycleError,
)
from .wrench_contract import vega_wrench_frame_rotations

LOGGER = logging.getLogger("robotenv_vega.force_calibration")

MAX_ROTATION_SPEED_DPS = 15.0
POSE_HOLD_SECONDS = 7.0
START_HOLD_SECONDS = 15.0
NEUTRAL_RETURN_HOLD_SECONDS = 1.0
FIT_TAIL_SECONDS = 2.0
SPEED_WINDOW_SECONDS = 0.5
SETTLED_MAX_JOINT_SPEED_DPS = 2.5
REFERENCE_POINT = "physical F/T sensor measurement origin"
SENSOR_ROBOT_MAX_SKEW_SECONDS = 0.25
RETURN_MAX_JOINT_ERROR_RAD = np.radians(5.0)
TERMINAL_PERSIST_RETRY_SECONDS = 1.0
ACTIVE_CALIBRATION_STATES = {
    "READY_FOR_STAGE",
    "MOVING",
    "SETTLING",
    "COLLECTING",
    "RETURNING",
    "FITTING",
    "READY_TO_APPLY",
    "CANCELLING",
}
ROBOT_OWNING_CALIBRATION_STATES = ACTIVE_CALIBRATION_STATES - {"CANCELLING"}


class CalibrationCancelled(RuntimeError):
    """Internal cooperative-cancellation signal."""


@dataclass(frozen=True)
class VegaForceCalibrationFitInputs:
    gripper_mass_kg: float
    gripper_com_sensor_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        mass = float(self.gripper_mass_kg)
        com = np.asarray(self.gripper_com_sensor_m, dtype=np.float64)
        if not isfinite(mass) or mass <= 0:
            raise ValueError("gripper_mass_kg must be finite and positive")
        if com.shape != (3,) or not np.isfinite(com).all():
            raise ValueError("gripper_com_sensor_m must contain three finite values")
        object.__setattr__(self, "gripper_mass_kg", mass)
        object.__setattr__(
            self, "gripper_com_sensor_m", tuple(float(value) for value in com)
        )


class CalibrationAdapter(Protocol):
    def begin_run(self) -> None: ...

    def validate_start(self) -> None: ...

    def execute_stage(
        self,
        stage_id: str,
        payload: Mapping[str, Any],
        cancelled: threading.Event,
        set_phase: Callable[[str, str], None],
    ) -> WristFtCalibrationHold | None: ...

    def safe_return(self, cancelled: threading.Event | None = None) -> None: ...

    def normal_return(self, cancelled: threading.Event | None = None) -> None: ...

    def fit(self, holds: Sequence[WristFtCalibrationHold]) -> WristFtCalibrationParameters: ...

    def prepare_install(self, parameters: WristFtCalibrationParameters | None) -> Any: ...

    def install_prepared(
        self, prepared: Any, *, expires_at_timestamp_us: int | None = None
    ) -> None: ...

    def commit_install(
        self,
        prepared: Any,
        commit: Callable[[], CalibrationStatusSnapshot],
        install_if: Callable[[], bool],
        *,
        expires_at_timestamp_us: int,
    ) -> CalibrationStatusSnapshot: ...


def _minimum_jerk_rotation_steps(
    target_rpy: np.ndarray, hz: float
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    target = np.asarray(target_rpy, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target_rpy must contain three finite values")
    duration = 1.875 * np.degrees(np.linalg.norm(target)) / MAX_ROTATION_SPEED_DPS
    progress = np.linspace(0.0, 1.0, max(1, ceil(duration * hz)) + 1)
    blend = progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)
    rotations = Rotation.from_euler("xyz", blend[:, None] * target).as_matrix()
    forward = rotations[1:] @ np.swapaxes(rotations[:-1], -1, -2)
    reverse = rotations[:-1] @ np.swapaxes(rotations[1:], -1, -2)
    return (
        tuple(Rotation.from_matrix(forward).as_euler("xyz")),
        tuple(Rotation.from_matrix(reverse[::-1]).as_euler("xyz")),
    )


def _settled_tail(
    samples: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, float], ...]:
    """Return the latest two-second settled interval from the complete hold."""
    if not samples:
        raise RuntimeError("calibration hold contains no samples")
    timestamps = np.asarray([item[3] for item in samples], dtype=np.float64)
    joints = np.asarray([item[2] for item in samples], dtype=np.float64)
    if not np.all(np.isfinite(timestamps)):
        raise RuntimeError("calibration hold timestamps contain a non-finite value")
    timestamp_deltas = np.diff(timestamps)
    invalid = np.flatnonzero(timestamp_deltas <= 0)
    if invalid.size:
        index = int(invalid[0] + 1)
        raise RuntimeError(
            "calibration hold timestamps are not strictly increasing "
            f"(sample_index={index}, previous={timestamps[index - 1]:.9f}s, "
            f"current={timestamps[index]:.9f}s, delta={timestamp_deltas[index - 1]:.9f}s)"
        )
    span_epsilon = max(
        1e-9,
        np.finfo(np.float64).eps * float(np.max(np.abs(timestamps))),
    )
    settled = np.zeros(len(samples), dtype=bool)
    speeds_dps = np.full(len(samples), np.nan, dtype=np.float64)
    for index, timestamp in enumerate(timestamps):
        speed_start = int(
            np.searchsorted(timestamps, timestamp - SPEED_WINDOW_SECONDS, side="right")
            - 1
        )
        if speed_start < 0 or speed_start >= index:
            continue
        span = timestamp - timestamps[speed_start]
        if span < SPEED_WINDOW_SECONDS:
            continue
        joint_range_deg = np.degrees(np.ptp(joints[speed_start : index + 1], axis=0))
        speed = float(np.max(joint_range_deg) / span)
        speeds_dps[index] = speed
        settled[index] = speed < SETTLED_MAX_JOINT_SPEED_DPS
    edges = np.flatnonzero(np.diff(np.r_[False, settled, False].astype(np.int8)))
    runs = list(zip(edges[::2], edges[1::2], strict=True))
    spans = [timestamps[end - 1] - timestamps[start] for start, end in runs]
    qualifying = [
        (start, end)
        for (start, end), span in zip(runs, spans, strict=True)
        if span >= FIT_TAIL_SECONDS - span_epsilon
    ]
    if not qualifying:
        finite_speeds = speeds_dps[np.isfinite(speeds_dps)]
        maximum_speed = (
            float(np.max(finite_speeds)) if finite_speeds.size else float("nan")
        )
        raise RuntimeError(
            "calibration hold did not contain two continuous settled seconds "
            f"(samples={len(samples)}, duration={timestamps[-1] - timestamps[0]:.2f}s, "
            f"max_joint_speed={maximum_speed:.3f}deg/s, "
            f"best_settled_span={max(spans, default=0.0):.2f}s, "
            f"threshold={SETTLED_MAX_JOINT_SPEED_DPS:.3f}deg/s)"
        )
    run_start, run_end = qualifying[-1]
    run_end_timestamp = timestamps[run_end - 1]
    tail_start = int(
        np.searchsorted(
            timestamps, run_end_timestamp - FIT_TAIL_SECONDS, side="right"
        )
        - 1
    )
    return tuple(samples[max(tail_start, run_start) : run_end])


@dataclass
class VegaRobotCalibrationAdapter:
    """Physical Vega implementation. The manager serializes calls per run."""

    service: Any
    gripper_mass_kg: float
    gripper_com_sensor_m: np.ndarray

    def __post_init__(self) -> None:
        self.gripper_mass_kg = float(self.gripper_mass_kg)
        self.gripper_com_sensor_m = np.asarray(
            self.gripper_com_sensor_m, dtype=np.float64
        )
        if not isfinite(self.gripper_mass_kg) or self.gripper_mass_kg <= 0:
            raise ValueError("gripper_mass_kg must be finite and positive")
        if self.gripper_com_sensor_m.shape != (3,) or not np.isfinite(
            self.gripper_com_sensor_m
        ).all():
            raise ValueError("gripper_com_sensor_m must contain three finite values")
        self._start_joints: np.ndarray | None = None
        self._gripper_position = 0.0

    def begin_run(self) -> None:
        self._start_joints = None
        self._gripper_position = 0.0

    def configure_fit_inputs(self, inputs: VegaForceCalibrationFitInputs) -> None:
        self.gripper_mass_kg = inputs.gripper_mass_kg
        self.gripper_com_sensor_m = np.asarray(
            inputs.gripper_com_sensor_m, dtype=np.float64
        )

    def validate_start(self) -> None:
        if not str(self.service.gripper_type).lower().split("@", 1)[0].startswith(
            "robotiq"
        ):
            raise LifecycleError(
                "FAILED_PRECONDITION",
                "Vega force calibration is validated only for RobotiQ",
            )
        sensor = getattr(getattr(self.service._robot, "arm", None), "wrench_sensor", None)
        if sensor is None:
            raise LifecycleError(
                "FAILED_PRECONDITION", "Vega wrist F/T sensor is unavailable"
            )

    def _check_cancelled(self, cancelled: threading.Event) -> None:
        if cancelled.is_set():
            raise CalibrationCancelled("force calibration cancelled")

    def _command_delta(self, delta_rotation: np.ndarray) -> None:
        self._command_cartesian_error(
            np.r_[np.zeros(3), np.asarray(delta_rotation, dtype=np.float64)]
        )

    def _command_cartesian_error(self, cartesian_error: np.ndarray) -> None:
        from dexcontrol.core.vega.robot import (
            _TELEOP_POS_ACTION_GAIN,
            _TELEOP_ROT_ACTION_GAIN,
        )

        action = np.r_[
            np.asarray(cartesian_error, dtype=np.float64), self._gripper_position
        ]
        if self.service.R_world_to_robot is not None:
            action[:3] = self.service.R_world_to_robot @ action[:3]
            # Preserve the deployed Step path exactly: its small RPY delta is
            # treated as a three-vector and rotated into the robot frame.
            action[3:6] = self.service.R_world_to_robot @ action[3:6]
        action[:3] *= _TELEOP_POS_ACTION_GAIN
        action[3:6] *= _TELEOP_ROT_ACTION_GAIN
        action = self.service._cartesian_velocity_to_delta(action)
        self.service._robot.update_command(
            action,
            action_space="cartesian_delta",
            gripper_action_space="position",
            blocking=False,
        )

    def _command_cartesian_target(self, target: np.ndarray) -> None:
        actual = self._current_cartesian()
        linear_error = np.asarray(target[:3]) - np.asarray(actual[:3])
        rotation_error = (
            Rotation.from_euler("xyz", target[3:])
            * Rotation.from_euler("xyz", actual[3:]).inv()
        ).as_euler("xyz")
        self._command_cartesian_error(np.r_[linear_error, rotation_error])

    def _sleep_tick(self, started: float) -> None:
        delay = 1.0 / self.service.control_hz - (time.perf_counter() - started)
        if delay > 0:
            time.sleep(delay)

    def _move_profile(
        self,
        steps: Sequence[np.ndarray],
        commanded: np.ndarray,
        cancelled: threading.Event,
    ) -> np.ndarray:
        target = np.asarray(commanded, dtype=np.float64).copy()
        for delta in steps:
            self._check_cancelled(cancelled)
            started = time.perf_counter()
            target[3:] = (
                Rotation.from_euler("xyz", delta)
                * Rotation.from_euler("xyz", target[3:])
            ).as_euler("xyz")
            self._command_cartesian_target(target)
            self._sleep_tick(started)
        return target

    def _hold_cartesian_target(
        self,
        target: np.ndarray,
        seconds: float,
        cancelled: threading.Event,
    ) -> None:
        for _ in range(max(1, round(seconds * self.service.control_hz))):
            self._check_cancelled(cancelled)
            started = time.perf_counter()
            self._command_cartesian_target(target)
            self._sleep_tick(started)

    def _sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        state, timestamps = self.service._robot.get_robot_state()
        cartesian = np.asarray(state["cartesian_position"], dtype=np.float64)
        if self.service.R_robot_to_world is not None:
            cartesian = self.service._transform_state_to_env_frame(cartesian)
        _, world_from_sensor = vega_wrench_frame_rotations(
            cartesian, gripper_type=self.service.gripper_type
        )
        sensor = self.service._robot.arm.wrench_sensor
        wrench_raw, sensor_timestamp_ns = sensor.get_wrench_sample()
        wrench = np.asarray(wrench_raw, dtype=np.float64)
        joints = np.asarray(state["joint_positions"], dtype=np.float64)
        if "robot_timestamp_us" not in timestamps and not {
            "robot_timestamp_seconds",
            "robot_timestamp_nanos",
        }.issubset(timestamps):
            raise RuntimeError("Vega robot-state acquisition timestamp is unavailable")
        robot_timestamp_us = self.service._timestamp_us(timestamps)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise RuntimeError("Vega wrist F/T sample is unavailable")
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise RuntimeError("Vega joint sample is unavailable")
        if sensor_timestamp_ns <= 0 or robot_timestamp_us <= 0:
            raise RuntimeError("Vega acquisition timestamps are unavailable")
        sensor_timestamp = sensor_timestamp_ns * 1e-9
        robot_timestamp = robot_timestamp_us * 1e-6
        if abs(sensor_timestamp - robot_timestamp) > SENSOR_ROBOT_MAX_SKEW_SECONDS:
            raise RuntimeError("Vega wrench and robot-state samples are not time aligned")
        return wrench, world_from_sensor, joints, robot_timestamp

    def _current_cartesian(self) -> np.ndarray:
        state, _ = self.service._robot.get_robot_state()
        cartesian = np.asarray(state["cartesian_position"], dtype=np.float64)
        if cartesian.shape != (6,) or not np.isfinite(cartesian).all():
            raise RuntimeError("Vega Cartesian pose is unavailable")
        if self.service.R_robot_to_world is not None:
            cartesian = self.service._transform_state_to_env_frame(cartesian)
        return cartesian

    def _verify_joint_return(self, target: np.ndarray) -> None:
        actual = np.asarray(self.service._robot.arm.get_joint_pos(), dtype=np.float64)
        if actual.shape != (7,) or not np.isfinite(actual).all():
            raise RuntimeError("Vega joint pose is unavailable after safe return")
        error = float(np.max(np.abs(actual - target)))
        if error > RETURN_MAX_JOINT_ERROR_RAD:
            raise RuntimeError(f"Vega safe return joint error is {error:.4f} rad")

    def _prepare(self, payload: Mapping[str, Any], cancelled: threading.Event) -> None:
        joints = np.asarray(payload.get("start_joints"), dtype=np.float64)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("prepare stage requires seven finite start_joints")
        self._start_joints = joints.copy()
        self.service._execute_reset_sequence(
            joints, cancelled=cancelled, interruptible_gripper=True
        )
        self._check_cancelled(cancelled)
        self._verify_joint_return(joints)
        state, _ = self.service._robot.get_robot_state()
        gripper = np.asarray(state.get("gripper_position", 0.0), dtype=np.float64).reshape(-1)
        if gripper.size != 1 or not np.isfinite(gripper[0]):
            raise RuntimeError("Vega RobotiQ gripper position is unavailable")
        self._gripper_position = float(gripper[0])
        # Preserve the legacy sequence's 15-second active neutral hold after
        # the reset and before the first calibration pose.
        self._hold_cartesian_target(
            self._current_cartesian(), START_HOLD_SECONDS, cancelled
        )

    def _pose(
        self,
        stage_id: str,
        payload: Mapping[str, Any],
        cancelled: threading.Event,
        set_phase: Callable[[str, str], None],
    ) -> WristFtCalibrationHold:
        if self._start_joints is None:
            raise RuntimeError("prepare stage must complete before pose collection")
        degrees = np.asarray(payload.get("rpy_degrees"), dtype=np.float64)
        if degrees.shape != (3,) or not np.isfinite(degrees).all():
            raise ValueError("pose stage requires three finite rpy_degrees")
        forward_steps, reverse_steps = _minimum_jerk_rotation_steps(
            np.radians(degrees), self.service.control_hz
        )
        start_cartesian = self._current_cartesian()
        set_phase("MOVING", f"moving to {stage_id}")
        commanded = self._move_profile(forward_steps, start_cartesian, cancelled)
        samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        try:
            set_phase("SETTLING", f"settling at {stage_id}")
            count = max(1, round(POSE_HOLD_SECONDS * self.service.control_hz))
            settled = None
            for index in range(count):
                self._check_cancelled(cancelled)
                started = time.perf_counter()
                self._command_cartesian_target(commanded)
                samples.append(self._sample())
                try:
                    settled = _settled_tail(samples)
                except RuntimeError as exc:
                    if "did not contain two continuous settled seconds" not in str(exc):
                        raise
                else:
                    set_phase("COLLECTING", f"collecting {stage_id}")
                    # The legacy loop completed the current control tick before
                    # starting the reverse profile, even on early settlement.
                    self._sleep_tick(started)
                    break
                self._sleep_tick(started)
            if settled is None:
                settled = _settled_tail(samples)
            raw_samples = np.asarray([item[0] for item in settled], dtype=np.float64)
            rotation = Rotation.from_matrix([item[1] for item in settled]).mean().as_matrix()
            return WristFtCalibrationHold(
                pose_id=stage_id,
                measured_sensor_wrench=np.mean(raw_samples, axis=0),
                world_from_sensor_rotation=rotation,
                measured_sensor_wrench_samples=raw_samples,
            )
        finally:
            if cancelled.is_set():
                raise CalibrationCancelled("force calibration cancelled")
            set_phase("RETURNING", f"returning from {stage_id}")
            neutral_target = self._move_profile(reverse_steps, commanded, cancelled)
            self._hold_cartesian_target(
                neutral_target, NEUTRAL_RETURN_HOLD_SECONDS, cancelled
            )

    def execute_stage(
        self,
        stage_id: str,
        payload: Mapping[str, Any],
        cancelled: threading.Event,
        set_phase: Callable[[str, str], None],
    ) -> WristFtCalibrationHold | None:
        kind = payload.get("kind")
        with self.service._cmd_lock:
            if kind == "prepare":
                set_phase("MOVING", "moving to calibration start")
                self._prepare(payload, cancelled)
                set_phase("RETURNING", "calibration start reached")
                return None
            if kind == "pose":
                return self._pose(stage_id, payload, cancelled, set_phase)
            raise ValueError(f"unsupported Vega calibration stage kind: {kind!r}")

    def safe_return(self, cancelled: threading.Event | None = None) -> None:
        if self._start_joints is None:
            return
        safe_cancel = threading.Event() if cancelled is None else cancelled
        with self.service._cmd_lock:
            self.service._execute_reset_sequence(
                self._start_joints,
                cancelled=safe_cancel,
                interruptible_gripper=True,
            )
            self._verify_joint_return(self._start_joints)

    def normal_return(self, cancelled: threading.Event | None = None) -> None:
        """Return to the configured inference/reset pose after a successful fit."""
        safe_cancel = threading.Event() if cancelled is None else cancelled
        with self.service._cmd_lock:
            self.service._execute_reset_sequence(
                self.service.reset_joints,
                cancelled=safe_cancel,
                interruptible_gripper=True,
            )
            self._check_cancelled(safe_cancel)
            self._verify_joint_return(self.service.reset_joints)

    def fit(self, holds: Sequence[WristFtCalibrationHold]) -> WristFtCalibrationParameters:
        return fit_wrist_ft_calibration(
            holds,
            arm_name=self.service.arm_side,
            sensor_frame_id=f"{self.service.arm_side}_vega_ft_sensor",
            reference_point=REFERENCE_POINT,
            world_frame_id=self.service.frame_type,
            gravity_world_unit=np.array([0.0, 0.0, -1.0]),
            gripper_mass_kg=self.gripper_mass_kg,
            gripper_com_sensor_m=self.gripper_com_sensor_m,
        )

    def prepare_install(
        self, parameters: WristFtCalibrationParameters | None
    ) -> WristFtCalibrationApplier | None:
        return None if parameters is None else WristFtCalibrationApplier(parameters)

    def install_prepared(
        self,
        prepared: WristFtCalibrationApplier | None,
        *,
        expires_at_timestamp_us: int | None = None,
    ) -> None:
        self.service._set_force_calibration_applier(
            prepared,
            expires_at_timestamp_us=expires_at_timestamp_us,
        )

    def commit_install(
        self,
        prepared: WristFtCalibrationApplier | None,
        commit: Callable[[], CalibrationStatusSnapshot],
        install_if: Callable[[], bool],
        *,
        expires_at_timestamp_us: int,
    ) -> CalibrationStatusSnapshot:
        """Publish runtime state and durable success as one observable transition."""
        with self.service._force_calibration_applier_lock:
            status = commit()
            if install_if():
                self.service._force_calibration_applier = prepared
                self.service._force_calibration_expires_at_timestamp_us = int(
                    expires_at_timestamp_us
                )
            return status


class VegaForceCalibrationManager:
    """Nonblocking RPC-facing orchestration around the durable lifecycle."""

    def __init__(
        self,
        *,
        lifecycle: ForceCalibrationLifecycle,
        adapter: CalibrationAdapter,
        claim_robot: Callable[[], bool],
        release_robot: Callable[[], None],
    ) -> None:
        self.lifecycle = lifecycle
        self.adapter = adapter
        self._claim_robot = claim_robot
        self._release_robot = release_robot
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._deadline_timer: threading.Timer | None = None
        self._terminal_retry_timer: threading.Timer | None = None
        self._cancelled = threading.Event()
        self._cancel_safe_return = True
        self._holds: list[WristFtCalibrationHold] = []
        self._candidate: WristFtCalibrationParameters | None = None
        self._pending_terminal: tuple[str, str | None] | None = None
        self._runtime_install_pending_run_id: str | None = None
        self._fit_inputs: VegaForceCalibrationFitInputs | None = None
        self._owns_robot = False
        self._restore_runtime_state()

    def _cancel_deadline_locked(self) -> None:
        if self._deadline_timer is not None:
            self._deadline_timer.cancel()
            self._deadline_timer = None

    def _schedule_terminal_retry_locked(self) -> None:
        if self._terminal_retry_timer is not None or self._pending_terminal is None:
            return
        timer = threading.Timer(
            TERMINAL_PERSIST_RETRY_SECONDS, self._retry_terminal_persistence
        )
        timer.daemon = True
        self._terminal_retry_timer = timer
        try:
            timer.start()
        except BaseException:
            self._terminal_retry_timer = None
            LOGGER.exception(
                "Vega force calibration terminal retry timer could not start"
            )

    def _clear_terminal_retry_locked(self) -> None:
        if self._terminal_retry_timer is not None:
            self._terminal_retry_timer.cancel()
            self._terminal_retry_timer = None

    def _persist_pending_terminal_locked(self) -> bool:
        if self._pending_terminal is None:
            return True
        run_id, failure_message = self._pending_terminal
        try:
            if failure_message is None:
                self.lifecycle.finish_cancel(
                    run_id, message="force calibration cancelled"
                )
            else:
                self.lifecycle.fail(run_id, failure_message)
        except LifecycleError:
            self._pending_terminal = None
            self._clear_terminal_retry_locked()
            self._release_if_owned()
            return True
        except BaseException:
            LOGGER.exception(
                "Vega force calibration terminal state could not be persisted"
            )
            self._release_if_owned()
            self._schedule_terminal_retry_locked()
            return False
        self._pending_terminal = None
        self._clear_terminal_retry_locked()
        self._release_if_owned()
        return True

    def _retry_terminal_persistence(self) -> None:
        with self._lock:
            self._terminal_retry_timer = None
            self._persist_pending_terminal_locked()

    def _schedule_deadline_locked(self, status: CalibrationStatusSnapshot) -> None:
        self._cancel_deadline_locked()
        if status.state != "READY_FOR_STAGE" or not status.command_deadline_timestamp_us:
            return
        delay = max(
            0.0, status.command_deadline_timestamp_us * 1e-6 - time.time()
        )
        timer = threading.Timer(delay, self._deadline_elapsed, args=(status.run_id,))
        timer.daemon = True
        self._deadline_timer = timer
        timer.start()

    def _deadline_elapsed(self, run_id: str) -> None:
        try:
            with self._lock:
                if self.lifecycle.status().run_id != run_id:
                    return
                status = self.lifecycle.expire_command_deadline()
                if status.state == "CANCELLING":
                    self.request_cancel(status.run_id)
                elif status.state == "READY_FOR_STAGE":
                    # threading.Timer is monotonic while the durable deadline is
                    # wall-clock based.  If a callback wakes a little early,
                    # preserve the server-owned timeout by arming the remainder.
                    self._schedule_deadline_locked(status)
        except BaseException:
            LOGGER.exception("Vega force calibration deadline cleanup failed")

    def _restore_runtime_state(self) -> None:
        status = self.lifecycle.status()
        result: dict[str, Any] = {"available": False}
        if status.state == "READY_TO_APPLY":
            self._owns_robot = self._claim_robot()
            if not self._owns_robot:
                raise RuntimeError("recovered calibration candidate cannot claim robot")
        try:
            if status.state == "READY_TO_APPLY":
                record = self.lifecycle.get_staged_candidate(status.run_id)
                self._candidate = wrist_ft_calibration_from_snapshot(
                    record["parameters"]
                )
            result = self.lifecycle.get_result("")
            if result.get("available"):
                self._install_result_snapshot(result)
        except BaseException:
            if result.get("available") and result.get("run_id"):
                self._runtime_install_pending_run_id = str(result["run_id"])
            self._release_if_owned()
            raise

    @property
    def owns_robot(self) -> bool:
        with self._lock:
            return self._owns_robot

    def _release_if_owned(self) -> None:
        with self._lock:
            if self._owns_robot:
                self._release_robot()
                self._owns_robot = False

    def _bind_fit_inputs_locked(
        self, inputs: VegaForceCalibrationFitInputs | None
    ) -> None:
        if inputs is None:
            return
        configure = getattr(self.adapter, "configure_fit_inputs", None)
        if callable(configure):
            configure(inputs)
        self._fit_inputs = inputs

    def start(
        self,
        run_id: str,
        force_recalibrate: bool,
        fit_inputs: VegaForceCalibrationFitInputs | None = None,
    ) -> CalibrationStatusSnapshot:
        with self._lock:
            current = self.lifecycle.status()
            if run_id == current.run_id:
                if (
                    fit_inputs is not None
                    and self._fit_inputs is not None
                    and fit_inputs != self._fit_inputs
                ):
                    raise LifecycleError(
                        "FAILED_PRECONDITION", "conflicting Start fit inputs"
                    )
                status = self.lifecycle.start(
                    run_id, force_recalibrate=force_recalibrate
                )
                if status.state == "SUCCEEDED":
                    try:
                        self._install_current_result_if_available(run_id)
                    finally:
                        self._release_if_owned()
                elif (
                    self._pending_terminal is not None
                    and self._pending_terminal[0] == run_id
                ):
                    self._schedule_terminal_retry_locked()
                    return status
                elif (
                    status.state in ROBOT_OWNING_CALIBRATION_STATES
                    and not self._owns_robot
                ):
                    acquired = False
                    try:
                        if not self._claim_robot():
                            raise LifecycleError(
                                "FAILED_PRECONDITION", "robot command path is busy"
                            )
                        self._owns_robot = True
                        acquired = True
                        self._bind_fit_inputs_locked(fit_inputs)
                        if (
                            status.state == "READY_FOR_STAGE"
                            and status.completed_stages == 0
                        ):
                            self.adapter.begin_run()
                        if status.state == "READY_FOR_STAGE":
                            self._schedule_deadline_locked(status)
                    except BaseException:
                        if acquired:
                            self._release_if_owned()
                        raise
                elif status.state == "READY_FOR_STAGE":
                    self._schedule_deadline_locked(status)
                return status
            if current.state in ACTIVE_CALIBRATION_STATES:
                raise LifecycleError(
                    "FAILED_PRECONDITION", "robot command path is busy"
                )
            validator = getattr(self.adapter, "validate_start", None)
            if callable(validator):
                validator()
            if not self._claim_robot():
                raise LifecycleError(
                    "FAILED_PRECONDITION", "robot command path is busy"
                )
            self._owns_robot = True
            try:
                self._bind_fit_inputs_locked(fit_inputs)
                self.adapter.begin_run()
                status = self.lifecycle.start(
                    run_id, force_recalibrate=force_recalibrate
                )
                if force_recalibrate and status.state != "SUCCEEDED":
                    self.lifecycle.invalidate_active_result()
                    self.adapter.install_prepared(
                        None, expires_at_timestamp_us=1
                    )
                    self._runtime_install_pending_run_id = None
                self._cancelled = threading.Event()
                self._cancel_safe_return = True
                self._holds = []
                self._candidate = None
                self._pending_terminal = None
                if status.state == "SUCCEEDED":
                    try:
                        self._install_current_result_if_available(run_id)
                    finally:
                        self._release_if_owned()
                else:
                    self._schedule_deadline_locked(status)
                return status
            except BaseException:
                self._release_if_owned()
                raise

    def advance(self, run_id: str, stage_index: int, stage_id: str) -> CalibrationStatusSnapshot:
        with self._lock:
            if not self._owns_robot:
                raise LifecycleError(
                    "FAILED_PRECONDITION", "robot command path is not owned"
                )
            status, payload = self.lifecycle.advance(run_id, stage_index, stage_id)
            if payload is None:
                return status
            self._cancel_deadline_locked()
            worker = threading.Thread(
                target=self._run_stage,
                args=(run_id, stage_index, stage_id, deepcopy(payload)),
                name=f"vega-force-cal-{self.lifecycle.plan.arm_name}",
                daemon=True,
            )
            self._worker = worker
            try:
                worker.start()
            except BaseException as exc:
                self._worker = None
                failure_message = str(exc).strip() or type(exc).__name__
                try:
                    self.adapter.safe_return()
                except BaseException as return_exc:
                    return_detail = (
                        str(return_exc).strip() or type(return_exc).__name__
                    )
                    failure_message = (
                        f"{failure_message}; safe return failed: {return_detail}"
                    )
                self._pending_terminal = (run_id, failure_message)
                self._persist_pending_terminal_locked()
                raise
            return status

    def _run_stage(
        self,
        run_id: str,
        stage_index: int,
        stage_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        def set_phase(state: str, message: str) -> None:
            try:
                self.lifecycle.set_phase(run_id, state, message)
            except LifecycleError:
                if self._cancelled.is_set():
                    raise CalibrationCancelled("force calibration cancelled")
                raise

        try:
            hold = self.adapter.execute_stage(
                stage_id, payload, self._cancelled, set_phase
            )
            if self._cancelled.is_set():
                raise CalibrationCancelled("force calibration cancelled")
            if hold is not None:
                self._holds.append(hold)
            with self._lock:
                status = self.lifecycle.complete_stage(run_id, stage_index)
                if status.state == "READY_FOR_STAGE":
                    self._worker = None
                    self._schedule_deadline_locked(status)
            if status.state == "FITTING":
                parameters = self.adapter.fit(tuple(self._holds))
                if self._cancelled.is_set():
                    raise CalibrationCancelled("force calibration cancelled")
                # Legacy behavior returns to the normal configured pose only
                # after fit/quality validation succeeds. Failures still use the
                # calibration-start safe return in the exception path below.
                self.adapter.normal_return(self._cancelled)
                if self._cancelled.is_set():
                    raise CalibrationCancelled("force calibration cancelled")
                self._candidate = parameters
                parameter_snapshot = parameters.as_parameters()
                with self._lock:
                    if self._cancelled.is_set():
                        raise CalibrationCancelled("force calibration cancelled")
                    self.lifecycle.stage_candidate(
                        run_id,
                        parameters=parameter_snapshot,
                        # as_parameters() deep-thaws the immutable nested
                        # MappingProxyType values used by runtime parameters.
                        # A shallow dict(parameters.quality) leaves nested
                        # proxies that json.dumps cannot serialize.
                        quality=parameter_snapshot["quality"],
                    )
        except CalibrationCancelled:
            self._finish_cancel_safely(run_id)
        except BaseException as exc:
            if self._cancelled.is_set():
                self._finish_cancel_safely(run_id)
                return
            LOGGER.exception("Vega force calibration stage failed: %s", exc)
            try:
                self.adapter.safe_return()
            except BaseException as return_exc:
                exc = RuntimeError(f"{exc}; safe return failed: {return_exc}")
            with self._lock:
                failure_message = str(exc).strip() or type(exc).__name__
                self._pending_terminal = (
                    run_id,
                    failure_message,
                )
                self._persist_pending_terminal_locked()
        finally:
            with self._lock:
                if self._worker is threading.current_thread():
                    self._worker = None

    def _finish_cancel_safely(self, run_id: str) -> None:
        with self._lock:
            if (
                self._pending_terminal is not None
                and self._pending_terminal[0] == run_id
            ):
                self._persist_pending_terminal_locked()
                return
        message = "force calibration cancelled"
        return_failed = False
        with self._lock:
            safe_return = self._cancel_safe_return
        if safe_return:
            try:
                self.adapter.safe_return()
            except BaseException as exc:
                detail = str(exc).strip() or type(exc).__name__
                message = f"force calibration cancelled; safe return failed: {detail}"
                return_failed = True
        with self._lock:
            self._pending_terminal = (
                run_id,
                message if return_failed else None,
            )
            self._persist_pending_terminal_locked()

    def request_cancel(
        self, run_id: str, *, safe_return: bool = True
    ) -> CalibrationStatusSnapshot:
        with self._lock:
            self._cancel_safe_return = self._cancel_safe_return and bool(safe_return)
            try:
                status = self.lifecycle.request_cancel(run_id)
            except BaseException:
                status = self.lifecycle.status()
                if (
                    run_id == status.run_id
                    and status.state in {"CANCELLED", "FAILED", "SUCCEEDED"}
                ):
                    if status.state == "SUCCEEDED":
                        self._install_current_result_and_release(status.run_id)
                    else:
                        self._release_if_owned()
                    return status
                raise
            self._cancel_deadline_locked()
            if status.state == "CANCELLING":
                self._cancelled.set()
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(
                        target=self._finish_cancel_safely,
                        args=(run_id,),
                        name=f"vega-force-cal-cancel-{self.lifecycle.plan.arm_name}",
                        daemon=True,
                    )
                    try:
                        self._worker.start()
                    except BaseException:
                        self._worker = None
                        self._finish_cancel_safely(run_id)
                        raise
            elif status.state in {"CANCELLED", "FAILED", "SUCCEEDED"}:
                if status.state == "SUCCEEDED":
                    self._install_current_result_and_release(run_id)
                else:
                    self._release_if_owned()
            return status

    def expire_deadline(self) -> CalibrationStatusSnapshot:
        with self._lock:
            status = self.lifecycle.expire_command_deadline()
            if status.state == "CANCELLING":
                return self.request_cancel(status.run_id)
            return status

    def _install_result_snapshot(self, result: Mapping[str, Any]) -> None:
        parameters = wrist_ft_calibration_from_snapshot(result["parameters"])
        self.adapter.install_prepared(
            self.adapter.prepare_install(parameters),
            expires_at_timestamp_us=int(result["expires_at_timestamp_us"]),
        )

    def _install_current_result_if_available(self, run_id: str) -> bool:
        result = self.lifecycle.get_result("")
        if not result.get("available") or result.get("run_id") != run_id:
            if self._runtime_install_pending_run_id == run_id:
                self._runtime_install_pending_run_id = None
            return False
        try:
            self._install_result_snapshot(result)
        except BaseException:
            self._runtime_install_pending_run_id = run_id
            raise
        self._runtime_install_pending_run_id = None
        return True

    def _current_result_matches(self, run_id: str) -> bool:
        result = self.lifecycle.get_result("")
        return bool(result.get("available") and result.get("run_id") == run_id)

    def _install_current_result_and_release(self, run_id: str) -> bool:
        try:
            return self._install_current_result_if_available(run_id)
        finally:
            self._release_if_owned()

    def commit(self, run_id: str, plan_sha256: str) -> CalibrationStatusSnapshot:
        with self._lock:
            current = self.lifecycle.status()
            if (
                run_id != current.run_id
                or plan_sha256 != self.lifecycle.plan.plan_sha256
            ):
                raise LifecycleError("FAILED_PRECONDITION", "stale Commit request")
            if current.state == "SUCCEEDED":
                status = self.lifecycle.commit(run_id, plan_sha256)
                self._install_current_result_and_release(run_id)
                return status
            record = self.lifecycle.get_staged_candidate(run_id)
            parameters = self._candidate
            snapshotter = getattr(parameters, "as_parameters", None)
            if (
                not callable(snapshotter)
                or snapshotter() != record["parameters"]
            ):
                parameters = wrist_ft_calibration_from_snapshot(record["parameters"])
            prepared = self.adapter.prepare_install(parameters)

            def commit_or_repair() -> CalibrationStatusSnapshot:
                try:
                    return self.lifecycle.commit(run_id, plan_sha256)
                except BaseException:
                    repaired = self.lifecycle.repair_partial_commit(run_id)
                    if repaired is None:
                        raise
                    return repaired

            try:
                status = self.adapter.commit_install(
                    prepared,
                    commit_or_repair,
                    lambda: self._current_result_matches(run_id),
                    expires_at_timestamp_us=int(
                        record["expires_at_timestamp_us"]
                    ),
                )
            except BaseException:
                if self.lifecycle.status().state == "SUCCEEDED":
                    self._runtime_install_pending_run_id = run_id
                    self._release_if_owned()
                raise
            else:
                self._runtime_install_pending_run_id = None
                self._release_if_owned()
            return status

    def status(self, run_id: str = "") -> CalibrationStatusSnapshot:
        with self._lock:
            status = self.expire_deadline()
            if run_id and run_id != status.run_id:
                raise LifecycleError("NOT_FOUND", "unknown calibration run_id")
            if status.state == "SUCCEEDED" and (
                self._owns_robot
                or self._runtime_install_pending_run_id == status.run_id
            ):
                self._install_current_result_and_release(status.run_id)
            elif status.state in {"CANCELLED", "FAILED"}:
                self._release_if_owned()
            pending_runtime = self._runtime_install_pending_run_id
            if pending_runtime is not None and pending_runtime != status.run_id:
                self._install_current_result_if_available(pending_runtime)
            if self._pending_terminal is not None:
                self._schedule_terminal_retry_locked()
            return status

    def result(self, run_id: str = "") -> dict[str, Any]:
        with self._lock:
            status = self.lifecycle.status()
            if status.state == "SUCCEEDED" and (
                self._owns_robot
                or self._runtime_install_pending_run_id == status.run_id
            ):
                self._install_current_result_and_release(status.run_id)
            elif status.state in {"CANCELLED", "FAILED"}:
                self._release_if_owned()
            pending_runtime = self._runtime_install_pending_run_id
            if pending_runtime is not None and pending_runtime != status.run_id:
                self._install_current_result_if_available(pending_runtime)
            if self._pending_terminal is not None:
                self._schedule_terminal_retry_locked()
            return self.lifecycle.get_result(run_id)

    def shutdown(self) -> None:
        """Cancel active calibration and wait for measured safe return.

        Process shutdown must not close the robot transport while a calibration
        worker still owns or moves the arm.  This intentionally has no short
        timeout: the adapter's own bounded motion checks decide failure, persist
        a terminal state, and release command ownership before shutdown returns.
        """
        with self._lock:
            self._cancel_deadline_locked()
            status = self.lifecycle.status()
            if status.state in ACTIVE_CALIBRATION_STATES:
                self.request_cancel(status.run_id)
            worker = self._worker

        while worker is not None and worker.is_alive():
            worker.join(timeout=0.1)
            with self._lock:
                worker = self._worker

        with self._lock:
            if self._pending_terminal is not None:
                self._persist_pending_terminal_locked()
            self._clear_terminal_retry_locked()
            self._release_if_owned()


def parameters_json(result: Mapping[str, Any]) -> str:
    return json.dumps(result.get("parameters", {}), allow_nan=False, sort_keys=True)


def quality_json(result: Mapping[str, Any]) -> str:
    return json.dumps(result.get("quality", {}), allow_nan=False, sort_keys=True)
