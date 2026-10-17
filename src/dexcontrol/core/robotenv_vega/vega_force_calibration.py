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
FIT_TAIL_SECONDS = 2.0
SPEED_WINDOW_SECONDS = 0.5
SETTLED_MAX_JOINT_SPEED_DPS = 2.5
REFERENCE_POINT = "physical F/T sensor measurement origin"


class CalibrationCancelled(RuntimeError):
    """Internal cooperative-cancellation signal."""


class CalibrationAdapter(Protocol):
    def validate_start(self) -> None: ...

    def execute_stage(
        self,
        stage_id: str,
        payload: Mapping[str, Any],
        cancelled: threading.Event,
        set_phase: Callable[[str, str], None],
    ) -> WristFtCalibrationHold | None: ...

    def safe_return(self, cancelled: threading.Event | None = None) -> None: ...

    def fit(self, holds: Sequence[WristFtCalibrationHold]) -> WristFtCalibrationParameters: ...

    def install(self, parameters: WristFtCalibrationParameters | None) -> None: ...


def _minimum_jerk_rotation_steps(target_rpy: np.ndarray, hz: float) -> tuple[np.ndarray, ...]:
    target = np.asarray(target_rpy, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target_rpy must contain three finite values")
    duration = 1.875 * np.degrees(np.linalg.norm(target)) / MAX_ROTATION_SPEED_DPS
    progress = np.linspace(0.0, 1.0, max(1, ceil(duration * hz)) + 1)
    blend = progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)
    rotations = Rotation.from_euler("xyz", blend[:, None] * target).as_matrix()
    deltas = rotations[1:] @ np.swapaxes(rotations[:-1], -1, -2)
    return tuple(Rotation.from_matrix(deltas).as_rotvec())


def _settled_tail(
    samples: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, float], ...]:
    if len(samples) < 2:
        raise RuntimeError("calibration hold contains too few samples")
    timestamps = np.asarray([item[3] for item in samples], dtype=np.float64)
    joints = np.asarray([item[2] for item in samples], dtype=np.float64)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise RuntimeError("calibration timestamps must be finite and strictly increasing")
    end = float(timestamps[-1])
    start = int(np.searchsorted(timestamps, end - FIT_TAIL_SECONDS, side="left"))
    if start >= len(samples) - 1 or end - float(timestamps[start]) < FIT_TAIL_SECONDS * 0.9:
        raise RuntimeError("calibration hold did not contain two seconds of fresh samples")
    window_start = int(
        np.searchsorted(timestamps, end - FIT_TAIL_SECONDS - SPEED_WINDOW_SECONDS, side="left")
    )
    for index in range(start, len(samples)):
        speed_start = int(
            np.searchsorted(timestamps, timestamps[index] - SPEED_WINDOW_SECONDS, side="left")
        )
        speed_start = max(speed_start, window_start)
        span = timestamps[index] - timestamps[speed_start]
        if span < SPEED_WINDOW_SECONDS * 0.9:
            raise RuntimeError("calibration hold lacks a full joint-speed window")
        joint_range_deg = np.degrees(np.ptp(joints[speed_start : index + 1], axis=0))
        if float(np.max(joint_range_deg) / span) >= SETTLED_MAX_JOINT_SPEED_DPS:
            raise RuntimeError("calibration hold did not settle before measurement")
    return tuple(samples[start:])


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
        action = np.r_[
            np.zeros(3),
            np.asarray(delta_rotation, dtype=np.float64),
            self._gripper_position,
        ]
        if self.service.R_world_to_robot is not None:
            action = self.service._transform_action_to_robot_frame(action)
        self.service._robot.update_command(
            action,
            action_space="cartesian_delta",
            gripper_action_space="position",
            blocking=False,
        )

    def _sleep_tick(self, started: float) -> None:
        delay = 1.0 / self.service.control_hz - (time.perf_counter() - started)
        if delay > 0:
            time.sleep(delay)

    def _move_profile(
        self,
        steps: Sequence[np.ndarray],
        cancelled: threading.Event,
    ) -> None:
        for delta in steps:
            self._check_cancelled(cancelled)
            started = time.perf_counter()
            self._command_delta(delta)
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
        wrench = np.asarray(sensor.get_wrench_state(), dtype=np.float64)
        joints = np.asarray(state["joint_positions"], dtype=np.float64)
        timestamp_us = self.service._timestamp_us(timestamps)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise RuntimeError("Vega wrist F/T sample is unavailable")
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise RuntimeError("Vega joint sample is unavailable")
        if timestamp_us <= 0:
            raise RuntimeError("Vega acquisition timestamp is unavailable")
        return wrench, world_from_sensor, joints, timestamp_us * 1e-6

    def _prepare(self, payload: Mapping[str, Any], cancelled: threading.Event) -> None:
        joints = np.asarray(payload.get("start_joints"), dtype=np.float64)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("prepare stage requires seven finite start_joints")
        self._start_joints = joints.copy()
        self.service._execute_reset_sequence(joints, cancelled=cancelled)
        self._check_cancelled(cancelled)
        state, _ = self.service._robot.get_robot_state()
        gripper = np.asarray(state.get("gripper_position", 0.0), dtype=np.float64).reshape(-1)
        if gripper.size != 1 or not np.isfinite(gripper[0]):
            raise RuntimeError("Vega RobotiQ gripper position is unavailable")
        self._gripper_position = float(gripper[0])

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
        steps = _minimum_jerk_rotation_steps(np.radians(degrees), self.service.control_hz)
        set_phase("MOVING", f"moving to {stage_id}")
        self._move_profile(steps, cancelled)
        samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        try:
            set_phase("SETTLING", f"settling at {stage_id}")
            count = max(1, round(POSE_HOLD_SECONDS * self.service.control_hz))
            collect_after = max(
                0,
                count - round((FIT_TAIL_SECONDS + SPEED_WINDOW_SECONDS) * self.service.control_hz),
            )
            for index in range(count):
                self._check_cancelled(cancelled)
                started = time.perf_counter()
                self._command_delta(np.zeros(3))
                if index >= collect_after:
                    if index == collect_after:
                        set_phase("COLLECTING", f"collecting {stage_id}")
                    samples.append(self._sample())
                self._sleep_tick(started)
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
            set_phase("RETURNING", f"returning from {stage_id}")
            self._move_profile(tuple(-delta for delta in reversed(steps)), cancelled)

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

    def install(self, parameters: WristFtCalibrationParameters | None) -> None:
        self.service._set_force_calibration_applier(
            None if parameters is None else WristFtCalibrationApplier(parameters)
        )


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
        self._cancelled = threading.Event()
        self._holds: list[WristFtCalibrationHold] = []
        self._candidate: WristFtCalibrationParameters | None = None
        self._owns_robot = False
        self._restore_runtime_state()

    def _restore_runtime_state(self) -> None:
        status = self.lifecycle.status()
        if status.state == "READY_TO_APPLY":
            self._owns_robot = self._claim_robot()
            if not self._owns_robot:
                raise RuntimeError("recovered calibration candidate cannot claim robot")
        result = self.lifecycle.get_result("")
        if result.get("available"):
            parameters = wrist_ft_calibration_from_snapshot(result["parameters"])
            self.adapter.install(parameters)

    @property
    def owns_robot(self) -> bool:
        with self._lock:
            return self._owns_robot

    def _release_if_owned(self) -> None:
        with self._lock:
            if self._owns_robot:
                self._owns_robot = False
                self._release_robot()

    def start(self, run_id: str, force_recalibrate: bool) -> CalibrationStatusSnapshot:
        with self._lock:
            current = self.lifecycle.status()
            if run_id == current.run_id:
                return self.lifecycle.start(
                    run_id, force_recalibrate=force_recalibrate
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
                status = self.lifecycle.start(
                    run_id, force_recalibrate=force_recalibrate
                )
                self._cancelled = threading.Event()
                self._holds = []
                self._candidate = None
                if status.state == "SUCCEEDED":
                    result = self.lifecycle.get_result("")
                    self.adapter.install(
                        wrist_ft_calibration_from_snapshot(result["parameters"])
                    )
                    self._release_if_owned()
                return status
            except BaseException:
                self._release_if_owned()
                raise

    def advance(self, run_id: str, stage_index: int, stage_id: str) -> CalibrationStatusSnapshot:
        with self._lock:
            status, payload = self.lifecycle.advance(run_id, stage_index, stage_id)
            if payload is None:
                return status
            if self._worker is not None and self._worker.is_alive():
                raise LifecycleError("FAILED_PRECONDITION", "stage worker is already running")
            worker = threading.Thread(
                target=self._run_stage,
                args=(run_id, stage_index, stage_id, deepcopy(payload)),
                name=f"vega-force-cal-{self.lifecycle.plan.arm_name}",
                daemon=True,
            )
            self._worker = worker
            worker.start()
            return status

    def _run_stage(
        self,
        run_id: str,
        stage_index: int,
        stage_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        def set_phase(state: str, message: str) -> None:
            self.lifecycle.set_phase(run_id, state, message)

        try:
            hold = self.adapter.execute_stage(
                stage_id, payload, self._cancelled, set_phase
            )
            if self._cancelled.is_set():
                raise CalibrationCancelled("force calibration cancelled")
            if hold is not None:
                self._holds.append(hold)
            status = self.lifecycle.complete_stage(run_id, stage_index)
            if status.state == "FITTING":
                parameters = self.adapter.fit(tuple(self._holds))
                self._candidate = parameters
                self.lifecycle.stage_candidate(
                    run_id,
                    parameters=parameters.as_parameters(),
                    quality=dict(parameters.quality),
                )
        except CalibrationCancelled:
            self._finish_cancel_safely(run_id)
        except BaseException as exc:
            LOGGER.exception("Vega force calibration stage failed: %s", exc)
            try:
                self.adapter.safe_return()
            except BaseException as return_exc:
                exc = RuntimeError(f"{exc}; safe return failed: {return_exc}")
            try:
                self.lifecycle.fail(run_id, str(exc) or type(exc).__name__)
            except LifecycleError:
                pass
            self._release_if_owned()

    def _finish_cancel_safely(self, run_id: str) -> None:
        message = "force calibration cancelled"
        try:
            self.adapter.safe_return()
        except BaseException as exc:
            message = f"force calibration cancelled; safe return failed: {exc}"
        try:
            self.lifecycle.finish_cancel(run_id, message=message)
        except LifecycleError:
            pass
        self._release_if_owned()

    def request_cancel(self, run_id: str) -> CalibrationStatusSnapshot:
        with self._lock:
            status = self.lifecycle.request_cancel(run_id)
            if status.state == "CANCELLING":
                self._cancelled.set()
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(
                        target=self._finish_cancel_safely,
                        args=(run_id,),
                        name=f"vega-force-cal-cancel-{self.lifecycle.plan.arm_name}",
                        daemon=True,
                    )
                    self._worker.start()
            elif status.state in {"CANCELLED", "FAILED", "SUCCEEDED"}:
                self._release_if_owned()
            return status

    def expire_deadline(self) -> CalibrationStatusSnapshot:
        status = self.lifecycle.expire_command_deadline()
        if status.state == "CANCELLING":
            return self.request_cancel(status.run_id)
        return status

    def commit(self, run_id: str, plan_sha256: str) -> CalibrationStatusSnapshot:
        with self._lock:
            status = self.lifecycle.commit(run_id, plan_sha256)
            result = self.lifecycle.get_result("")
            parameters = (
                self._candidate
                if self._candidate is not None
                else wrist_ft_calibration_from_snapshot(result["parameters"])
            )
            self.adapter.install(parameters)
            self._release_if_owned()
            return status

    def status(self, run_id: str = "") -> CalibrationStatusSnapshot:
        status = self.expire_deadline()
        if run_id and run_id != status.run_id:
            raise LifecycleError("NOT_FOUND", "unknown calibration run_id")
        return status

    def result(self, run_id: str = "") -> dict[str, Any]:
        return self.lifecycle.get_result(run_id)


def parameters_json(result: Mapping[str, Any]) -> str:
    return json.dumps(result.get("parameters", {}), allow_nan=False, sort_keys=True)


def quality_json(result: Mapping[str, Any]) -> str:
    return json.dumps(result.get("quality", {}), allow_nan=False, sort_keys=True)
