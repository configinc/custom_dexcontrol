"""Multi-pose calibration and runtime compensation for wrist F/T sensors.

The numerical model is adapted from ``core/ft_calibration/{compute,model}.py``
on Seongmin's ``feat/ft-calib-system`` branch.  This version has explicit
sensor/world frame boundaries and returns the final wrench in the world frame.
"""

from __future__ import annotations

import json
from dataclasses import InitVar, dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .held_out_validation import validate_held_out_wrench
from .static_wrench_fit import MIN_CLEAN_HOLDS, robust_static_wrench_refit

GRAVITY_M_S2 = 9.81
MIN_HOLDS = MIN_CLEAN_HOLDS
MIN_POSE_SPREAD = 0.3
MAX_RMS_FORCE_N = 0.5
K_DIAGONAL_RANGE = (0.80, 1.20)
MAX_K_OFF_DIAGONAL = 0.2
_VALIDATED_FIT_TOKEN = object()


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite shape {shape}")
    # Immutable bytes prevent callers from turning WRITEABLE back on and
    # changing runtime math after Recorder captured the parameter snapshot.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(shape)


def _rotation(value: Any, name: str) -> np.ndarray:
    matrix = _finite_array(value, (3, 3), name)
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        raise ValueError(f"{name} must be a proper rotation matrix")
    return matrix


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _deep_thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class WristFtCalibrationHold:
    """Mean raw sensor sample and tool orientation from one settled pose."""

    pose_id: str
    measured_sensor_wrench: np.ndarray
    world_from_sensor_rotation: np.ndarray
    measured_sensor_wrench_samples: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose_id", _nonempty(self.pose_id, "pose_id"))
        object.__setattr__(
            self,
            "measured_sensor_wrench",
            _finite_array(self.measured_sensor_wrench, (6,), "measured_sensor_wrench"),
        )
        object.__setattr__(
            self,
            "world_from_sensor_rotation",
            _rotation(self.world_from_sensor_rotation, "world_from_sensor_rotation"),
        )
        samples = self.measured_sensor_wrench_samples
        if samples is None:
            # Compatibility for synthetic callers. Real collection always
            # provides the settled samples used to derive repeatability.
            samples = np.repeat(self.measured_sensor_wrench[None, :], 2, axis=0)
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] != 6:
            raise ValueError("measured_sensor_wrench_samples must have shape (N>=2, 6)")
        if not np.isfinite(samples).all():
            raise ValueError("measured_sensor_wrench_samples must be finite")
        if not np.allclose(np.mean(samples, axis=0), self.measured_sensor_wrench, rtol=1e-10, atol=1e-12):
            raise ValueError("measured_sensor_wrench must equal the sample mean")
        object.__setattr__(self, "measured_sensor_wrench_samples", samples.copy())


@dataclass(frozen=True)
class WristFtCalibrationParameters:
    """Exact accepted parameters for raw sensor wrench to calibrated world wrench."""

    arm_name: str
    sensor_frame_id: str
    reference_point: str
    world_frame_id: str
    gravity_world_unit: np.ndarray
    gripper_mass_kg: float
    gripper_com_sensor_m: np.ndarray
    force_decoupling_inverse: np.ndarray
    force_offset_sensor: np.ndarray
    torque_gain: float
    torque_offset_sensor: np.ndarray
    quality: Mapping[str, Any]
    pose_evidence: Sequence[Mapping[str, Any]]
    schema_version: int = 5
    _validated_fit_token: InitVar[object] = None

    def __post_init__(self, _validated_fit_token: object) -> None:
        if _validated_fit_token is not _VALIDATED_FIT_TOKEN:
            raise TypeError(
                "WristFtCalibrationParameters must be created by fit_wrist_ft_calibration"
            )
        if self.arm_name not in {"left", "right"}:
            raise ValueError("arm_name must be left or right")
        for name in ("sensor_frame_id", "reference_point", "world_frame_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        gravity = _finite_array(self.gravity_world_unit, (3,), "gravity_world_unit")
        if not np.isclose(np.linalg.norm(gravity), 1.0, atol=1e-6):
            raise ValueError("gravity_world_unit must have unit norm")
        object.__setattr__(self, "gravity_world_unit", gravity)
        if not isfinite(self.gripper_mass_kg) or self.gripper_mass_kg <= 0:
            raise ValueError("gripper_mass_kg must be finite and positive")
        object.__setattr__(
            self,
            "gripper_com_sensor_m",
            _finite_array(self.gripper_com_sensor_m, (3,), "gripper_com_sensor_m"),
        )
        object.__setattr__(
            self,
            "force_decoupling_inverse",
            _finite_array(
                self.force_decoupling_inverse,
                (3, 3),
                "force_decoupling_inverse",
            ),
        )
        object.__setattr__(
            self,
            "force_offset_sensor",
            _finite_array(self.force_offset_sensor, (3,), "force_offset_sensor"),
        )
        if not isfinite(self.torque_gain) or abs(self.torque_gain) < 1e-9:
            raise ValueError("torque_gain must be finite and non-zero")
        object.__setattr__(
            self,
            "torque_offset_sensor",
            _finite_array(self.torque_offset_sensor, (3,), "torque_offset_sensor"),
        )
        if self.schema_version != 5:
            raise ValueError("unsupported wrist F/T calibration schema")
        try:
            quality = json.loads(
                json.dumps(
                    dict(self.quality),
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("quality must be finite JSON") from exc
        if quality.get("passed") is not True:
            raise ValueError("runtime parameters require passed calibration quality")
        object.__setattr__(self, "quality", _deep_freeze_json(quality))
        evidence = tuple(dict(item) for item in self.pose_evidence)
        required_keys = {
            "pose_id",
            "measured_sensor_wrench",
            "measured_sensor_wrench_samples",
            "world_from_sensor_rotation",
        }
        if len(evidence) < MIN_HOLDS:
            raise ValueError("pose_evidence must contain the calibration pose bank")
        if any(set(item) != required_keys for item in evidence):
            raise ValueError("pose_evidence fields do not match the wrist F/T schema")
        if len({item["pose_id"] for item in evidence}) != len(evidence):
            raise ValueError("pose_evidence ids must be unique")
        object.__setattr__(self, "pose_evidence", _deep_freeze_json(list(evidence)))

    def as_parameters(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": "multi_pose_wrist_ft",
            "arm_name": self.arm_name,
            "input_field": "external_wrench_sensor_raw",
            "input_frame": self.sensor_frame_id,
            "output_field": "external_wrench_world",
            "output_fields": ["external_wrench_wrist", "external_wrench_world"],
            "output_frame": self.world_frame_id,
            "reference_point": self.reference_point,
            "frame_rotations_owner": "robot_server",
            "gravity_world_unit": self.gravity_world_unit.tolist(),
            "gripper_mass_kg": float(self.gripper_mass_kg),
            "gripper_com_sensor_m": self.gripper_com_sensor_m.tolist(),
            "force_decoupling_inverse": self.force_decoupling_inverse.tolist(),
            "force_offset_sensor": self.force_offset_sensor.tolist(),
            "torque_gain": float(self.torque_gain),
            "torque_offset_sensor": self.torque_offset_sensor.tolist(),
            "quality": _deep_thaw_json(self.quality),
            "pose_evidence": _deep_thaw_json(self.pose_evidence),
        }


class WristFtCalibrationQualityError(ValueError):
    def __init__(self, reasons: Sequence[str], quality: Mapping[str, Any]):
        self.reasons = tuple(reasons)
        self.quality = dict(quality)
        diagnostics = {
            key: self.quality.get(key)
            for key in (
                "hold_count",
                "dropped_pose_ids",
                "pose_spread",
                "rms_force_n",
                "rms_torque_nm",
                "force_decoupling_diagonal",
                "max_force_decoupling_off_diagonal",
                "torque_gain",
            )
        }
        super().__init__(
            "wrist F/T calibration quality failed: "
            + "; ".join(self.reasons)
            + f"; metrics={json.dumps(diagnostics, sort_keys=True)}"
        )


def _identify(
    gravity_sensor: np.ndarray,
    measured_force: np.ndarray,
    measured_torque: np.ndarray,
) -> dict[str, Any]:
    count = len(gravity_sensor)
    force_design = np.zeros((3 * count, 4))
    force_design[:, 0] = gravity_sensor.ravel()
    force_design[:, 1:] = np.tile(np.eye(3), (count, 1))
    force_solution = np.linalg.lstsq(force_design, measured_force.ravel(), rcond=None)[0]
    weight, force_offset = float(force_solution[0]), force_solution[1:]
    torque_design = np.vstack([np.c_[-_skew(weight * direction), np.eye(3)] for direction in gravity_sensor])
    torque_solution = np.linalg.lstsq(torque_design, measured_torque.ravel(), rcond=None)[0]
    com, torque_offset = torque_solution[:3], torque_solution[3:]
    force_residual = measured_force - (weight * gravity_sensor + force_offset)
    torque_residual = measured_torque - (np.cross(com, weight * gravity_sensor) + torque_offset)
    return {
        "weight": weight,
        "force_offset": force_offset,
        "com": com,
        "torque_offset": torque_offset,
        "rms_force": float(np.sqrt(np.mean(force_residual**2))),
        "rms_torque": float(np.sqrt(np.mean(torque_residual**2))),
    }


def _evaluate_sensor_geometry(
    holds: Sequence[WristFtCalibrationHold],
    gravity_world_unit: np.ndarray,
) -> dict[str, Any]:
    """Evaluate the hardware-declared sensor mount without estimating rotation.

    Vega's F/T sensor is mechanically aligned to its TCP.  Rotation is therefore
    a hardware contract, not a calibration degree of freedom.  Keeping it fixed
    also prevents sensor bias/decoupling error from being misidentified as a
    change in the robot's coordinate frame.
    """
    measured = np.asarray([hold.measured_sensor_wrench for hold in holds])
    world_from_sensor = np.asarray(
        [hold.world_from_sensor_rotation for hold in holds]
    )
    gravity_sensor = np.einsum("nji,j->ni", world_from_sensor, gravity_world_unit)
    fit = _identify(gravity_sensor, measured[:, :3], measured[:, 3:])
    hold_residual = np.linalg.norm(
        measured[:, :3] - (fit["weight"] * gravity_sensor + fit["force_offset"]),
        axis=1,
    )
    fit.update(
        gravity_sensor=gravity_sensor,
        hold_residual=hold_residual,
        spread=np.linalg.svd(gravity_sensor - gravity_sensor.mean(axis=0), compute_uv=False),
    )
    return fit


def _fit_decoupling(
    holds: Sequence[WristFtCalibrationHold],
    gravity_world_unit: np.ndarray,
    mass_kg: float,
    com_sensor_m: np.ndarray,
) -> dict[str, Any]:
    measured = np.asarray([hold.measured_sensor_wrench for hold in holds])
    world_from_sensor = np.asarray(
        [hold.world_from_sensor_rotation for hold in holds]
    )
    gravity_sensor = np.einsum("nji,j->ni", world_from_sensor, gravity_world_unit)
    gravity_force = mass_kg * GRAVITY_M_S2 * gravity_sensor
    force_design = np.c_[gravity_force, np.ones(len(gravity_force))]
    force_solution = np.linalg.lstsq(force_design, measured[:, :3], rcond=None)[0]
    decoupling, force_offset = force_solution[:3].T, force_solution[3]
    force_residual = measured[:, :3] - force_design @ force_solution
    gravity_torque = np.cross(com_sensor_m, gravity_force)
    torque_design = np.c_[gravity_torque.ravel(), np.tile(np.eye(3), (len(gravity_force), 1))]
    torque_solution = np.linalg.lstsq(torque_design, measured[:, 3:].ravel(), rcond=None)[0]
    torque_residual = measured[:, 3:].ravel() - torque_design @ torque_solution
    off_diagonal = decoupling - np.diag(np.diag(decoupling))
    return {
        "decoupling": decoupling,
        "force_offset": force_offset,
        "torque_gain": float(torque_solution[0]),
        "torque_offset": torque_solution[1:],
        "rms_force": float(np.sqrt(np.mean(force_residual**2))),
        "rms_torque": float(np.sqrt(np.mean(torque_residual**2))),
        "max_off_diagonal": float(np.max(np.abs(off_diagonal))),
    }


def _held_out_world_samples(
    hold: WristFtCalibrationHold,
    fit: Mapping[str, Any] | None,
    *,
    gravity_world_unit: np.ndarray,
    mass_kg: float,
    com_sensor_m: np.ndarray,
) -> np.ndarray:
    """Express raw or fold-calibrated held-out samples in the world frame."""
    raw = hold.measured_sensor_wrench_samples
    rotation = hold.world_from_sensor_rotation
    if fit is None:
        force_sensor = raw[:, :3]
        torque_sensor = raw[:, 3:]
    else:
        gravity_sensor = rotation.T @ gravity_world_unit
        gravity_force = mass_kg * GRAVITY_M_S2 * gravity_sensor
        decoupling_inverse = np.linalg.inv(fit["decoupling"])
        force_sensor = (
            (decoupling_inverse @ (raw[:, :3] - fit["force_offset"]).T).T
            - gravity_force
        )
        torque_sensor = (
            (raw[:, 3:] - fit["torque_offset"]) / fit["torque_gain"]
            - np.cross(com_sensor_m, gravity_force)
        )
    return np.c_[
        (rotation @ force_sensor.T).T,
        (rotation @ torque_sensor.T).T,
    ]


def fit_wrist_ft_calibration(
    holds: Sequence[WristFtCalibrationHold],
    *,
    arm_name: str,
    sensor_frame_id: str,
    reference_point: str,
    world_frame_id: str,
    gravity_world_unit: Any,
    gripper_mass_kg: float,
    gripper_com_sensor_m: Any,
) -> WristFtCalibrationParameters:
    """Fit static sensor offsets/decoupling using RobotEnv-owned frame rotations."""
    holds = tuple(holds)
    if len(holds) < MIN_HOLDS:
        raise ValueError(f"at least {MIN_HOLDS} settled pose holds are required")
    if len({hold.pose_id for hold in holds}) != len(holds):
        raise ValueError("each calibration hold must have a unique pose_id")
    gravity = _finite_array(gravity_world_unit, (3,), "gravity_world_unit")
    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm < 1e-12:
        raise ValueError("gravity_world_unit must be non-zero")
    gravity = gravity / gravity_norm
    if not isfinite(gripper_mass_kg) or gripper_mass_kg <= 0:
        raise ValueError("gripper_mass_kg must be finite and positive")
    com = _finite_array(gripper_com_sensor_m, (3,), "gripper_com_sensor_m")
    robust = robust_static_wrench_refit(
        holds,
        lambda selected: _evaluate_sensor_geometry(selected, gravity),
        lambda fit: fit["hold_residual"],
    )
    kept = robust.kept_holds
    geometry = robust.fit
    decoupling_fit = _fit_decoupling(
        kept,
        gravity,
        float(gripper_mass_kg),
        com,
    )

    reasons = []
    spread = geometry["spread"]
    if len(kept) < MIN_HOLDS:
        reasons.append(f"only {len(kept)} uncontaminated pose holds remain")
    if min(spread[:2]) < MIN_POSE_SPREAD:
        reasons.append("pose orientation spread is insufficient")
    if decoupling_fit["rms_force"] > MAX_RMS_FORCE_N:
        reasons.append("force residual exceeds 0.5 N")
    diagonal = np.diag(decoupling_fit["decoupling"])
    if not np.all((diagonal >= K_DIAGONAL_RANGE[0]) & (diagonal <= K_DIAGONAL_RANGE[1])):
        reasons.append("force decoupling diagonal is outside [0.80, 1.20]")
    if decoupling_fit["max_off_diagonal"] > MAX_K_OFF_DIAGONAL:
        reasons.append(
            f"force decoupling cross-axis term exceeds {MAX_K_OFF_DIAGONAL:g}"
        )
    if abs(decoupling_fit["torque_gain"]) < 1e-9:
        reasons.append("torque gain is unobservable")
    held_out_quality: Mapping[str, Any] = {
        "passed": False,
        "status": "not_evaluated_due_to_preliminary_quality_failure",
    }
    if not reasons:
        raw_validation = []
        calibrated_validation = []
        kept_tuple = tuple(kept)
        for held_out_index, held_out in enumerate(kept_tuple):
            training = kept_tuple[:held_out_index] + kept_tuple[held_out_index + 1 :]
            fold_fit = _fit_decoupling(
                training,
                gravity,
                float(gripper_mass_kg),
                com,
            )
            raw_validation.append(
                _held_out_world_samples(
                    held_out,
                    None,
                    gravity_world_unit=gravity,
                    mass_kg=float(gripper_mass_kg),
                    com_sensor_m=com,
                )
            )
            calibrated_validation.append(
                _held_out_world_samples(
                    held_out,
                    fold_fit,
                    gravity_world_unit=gravity,
                    mass_kg=float(gripper_mass_kg),
                    com_sensor_m=com,
                )
            )
        held_out_validation = validate_held_out_wrench(
            raw_validation, calibrated_validation
        )
        held_out_quality = held_out_validation.as_dict()
        if not held_out_validation.passed:
            reasons.append(
                "held-out force/torque correction regresses beyond stationary repeatability"
            )
    quality = {
        "passed": not reasons,
        "hold_count": len(kept),
        "dropped_pose_ids": list(robust.dropped_pose_ids),
        "pose_spread": spread.tolist(),
        "rms_force_n": decoupling_fit["rms_force"],
        "rms_torque_nm": decoupling_fit["rms_torque"],
        "force_decoupling_diagonal": diagonal.tolist(),
        "max_force_decoupling_off_diagonal": decoupling_fit["max_off_diagonal"],
        "torque_gain": decoupling_fit["torque_gain"],
        "held_out_validation": dict(held_out_quality),
        "reasons": reasons,
    }
    if reasons:
        raise WristFtCalibrationQualityError(reasons, quality)
    try:
        decoupling_inverse = np.linalg.inv(decoupling_fit["decoupling"])
    except np.linalg.LinAlgError as exc:
        raise WristFtCalibrationQualityError(("force decoupling matrix is singular",), quality) from exc
    return WristFtCalibrationParameters(
        arm_name=arm_name,
        sensor_frame_id=sensor_frame_id,
        reference_point=reference_point,
        world_frame_id=world_frame_id,
        gravity_world_unit=gravity,
        gripper_mass_kg=float(gripper_mass_kg),
        gripper_com_sensor_m=com,
        force_decoupling_inverse=decoupling_inverse,
        force_offset_sensor=decoupling_fit["force_offset"],
        torque_gain=decoupling_fit["torque_gain"],
        torque_offset_sensor=decoupling_fit["torque_offset"],
        quality=quality,
        pose_evidence=tuple(
            {
                "pose_id": hold.pose_id,
                "measured_sensor_wrench": hold.measured_sensor_wrench.tolist(),
                "measured_sensor_wrench_samples": hold.measured_sensor_wrench_samples.tolist(),
                "world_from_sensor_rotation": hold.world_from_sensor_rotation.tolist(),
            }
            for hold in holds
        ),
        _validated_fit_token=_VALIDATED_FIT_TOKEN,
    )


def wrist_ft_calibration_from_snapshot(snapshot: Mapping[str, Any]) -> WristFtCalibrationParameters:
    """Rehydrate an accepted on-disk snapshot through the runtime validators."""
    if not isinstance(snapshot, Mapping):
        raise TypeError("wrist F/T calibration snapshot must be a mapping")
    if snapshot.get("method") != "multi_pose_wrist_ft":
        raise ValueError("unsupported wrist F/T calibration method")
    if snapshot.get("input_field") != "external_wrench_sensor_raw":
        raise ValueError("wrist F/T calibration input field is incompatible")
    if snapshot.get("output_field") != "external_wrench_world":
        raise ValueError("wrist F/T calibration output field is incompatible")
    if snapshot.get("frame_rotations_owner") != "robot_server":
        raise ValueError("wrist F/T frame rotations must be owned by RobotEnv")
    if snapshot.get("schema_version") != 5:
        raise ValueError("unsupported wrist F/T calibration schema")
    evidence = snapshot.get("pose_evidence")
    if not isinstance(evidence, list):
        raise ValueError("wrist F/T snapshot must contain reproducible pose evidence")
    try:
        holds = tuple(
            WristFtCalibrationHold(
                pose_id=item["pose_id"],
                measured_sensor_wrench=item["measured_sensor_wrench"],
                measured_sensor_wrench_samples=item["measured_sensor_wrench_samples"],
                world_from_sensor_rotation=item["world_from_sensor_rotation"],
            )
            for item in evidence
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("wrist F/T pose evidence is invalid") from exc
    reproduced = fit_wrist_ft_calibration(
        holds,
        arm_name=snapshot.get("arm_name"),
        sensor_frame_id=snapshot.get("input_frame"),
        reference_point=snapshot.get("reference_point"),
        world_frame_id=snapshot.get("output_frame"),
        gravity_world_unit=snapshot.get("gravity_world_unit"),
        gripper_mass_kg=snapshot.get("gripper_mass_kg"),
        gripper_com_sensor_m=snapshot.get("gripper_com_sensor_m"),
    )
    normalized_snapshot = json.loads(json.dumps(dict(snapshot), allow_nan=False))
    if reproduced.as_parameters() != normalized_snapshot:
        raise ValueError("wrist F/T snapshot does not reconcile with pose evidence")
    return reproduced


@dataclass(frozen=True)
class WristFtCalibrationApplier:
    """Apply accepted parameters and express the result in wrist or world."""

    parameters: WristFtCalibrationParameters

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, WristFtCalibrationParameters):
            raise TypeError("parameters must be WristFtCalibrationParameters")

    def _measured_sensor_raw_to_calibrated_sensor(
        self,
        measured_sensor_wrench: Any,
        world_from_sensor_rotation: Any,
        *,
        reference_point: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if _nonempty(reference_point, "reference_point") != self.parameters.reference_point:
            raise ValueError("sensor wrench reference point does not match calibration")
        measured = _finite_array(measured_sensor_wrench, (6,), "measured_sensor_wrench")
        world_from_sensor = _rotation(
            world_from_sensor_rotation, "world_from_sensor_rotation"
        )
        gravity_sensor = world_from_sensor.T @ self.parameters.gravity_world_unit
        gravity_force = self.parameters.gripper_mass_kg * GRAVITY_M_S2 * gravity_sensor
        force_sensor = (
            self.parameters.force_decoupling_inverse @ (measured[:3] - self.parameters.force_offset_sensor)
            - gravity_force
        )
        torque_sensor = (measured[3:] - self.parameters.torque_offset_sensor) / self.parameters.torque_gain - np.cross(
            self.parameters.gripper_com_sensor_m, gravity_force
        )
        return force_sensor, torque_sensor, world_from_sensor

    def measured_sensor_raw_to_calibrated_wrist(
        self,
        measured_sensor_wrench: Any,
        world_from_sensor_rotation: Any,
        wrist_from_sensor_rotation: Any,
        *,
        reference_point: str,
    ) -> np.ndarray:
        force_sensor, torque_sensor, _ = self._measured_sensor_raw_to_calibrated_sensor(
            measured_sensor_wrench,
            world_from_sensor_rotation,
            reference_point=reference_point,
        )
        wrist_from_sensor = _rotation(
            wrist_from_sensor_rotation, "wrist_from_sensor_rotation"
        )
        return np.concatenate([
            wrist_from_sensor @ force_sensor,
            wrist_from_sensor @ torque_sensor,
        ])

    def measured_sensor_raw_to_calibrated_world(
        self,
        measured_sensor_wrench: Any,
        world_from_sensor_rotation: Any,
        *,
        reference_point: str,
    ) -> np.ndarray:
        force_sensor, torque_sensor, world_from_sensor = (
            self._measured_sensor_raw_to_calibrated_sensor(
            measured_sensor_wrench,
            world_from_sensor_rotation,
            reference_point=reference_point,
            )
        )
        return np.concatenate([
            world_from_sensor @ force_sensor,
            world_from_sensor @ torque_sensor,
        ])
