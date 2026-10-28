"""Shared acceptance policy for unloaded Cartesian wrench calibration.

Robot-specific fitters provide leave-one-pose-out raw and calibrated wrench
samples.  This module only decides whether the correction generalizes.  Force
and torque remain in their physical units and are never combined or manually
weighted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _finite_wrench_samples(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 6 or array.shape[0] < 2:
        raise ValueError(f"{name} must have shape (samples>=2, 6)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def _metric_rms(samples: np.ndarray, component: slice) -> float:
    return _rms(samples[:, component])


def _repeatability_margin(samples_by_pose: Sequence[np.ndarray], component: slice) -> float:
    """Return the typical within-hold stationary noise in physical units.

    Each pose contributes the RMS magnitude of its mean-removed samples.  The
    median across poses prevents one drifting or contaminated hold from
    expanding the tolerance for every other held-out pose.
    """
    pose_noise = []
    for samples in samples_by_pose:
        values = samples[:, component]
        pose_noise.append(
            _rms(values - np.mean(values, axis=0, keepdims=True))
        )
    return float(np.median(np.asarray(pose_noise, dtype=np.float64)))


def _classification(delta: float, margin: float) -> str:
    if delta < -margin:
        return "improved"
    if delta <= margin:
        return "equivalent"
    return "regressed"


@dataclass(frozen=True)
class HeldOutWrenchValidation:
    raw_force_rms_n: float
    calibrated_force_rms_n: float
    force_repeatability_margin_n: float
    force_classification: str
    raw_torque_rms_nm: float
    calibrated_torque_rms_nm: float
    torque_repeatability_margin_nm: float
    torque_classification: str
    passed: bool

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "raw_force_rms_n": self.raw_force_rms_n,
            "calibrated_force_rms_n": self.calibrated_force_rms_n,
            "force_repeatability_margin_n": self.force_repeatability_margin_n,
            "force_classification": self.force_classification,
            "raw_torque_rms_nm": self.raw_torque_rms_nm,
            "calibrated_torque_rms_nm": self.calibrated_torque_rms_nm,
            "torque_repeatability_margin_nm": self.torque_repeatability_margin_nm,
            "torque_classification": self.torque_classification,
            "passed": self.passed,
        }


def validate_held_out_wrench(
    raw_samples_by_pose: Sequence[np.ndarray],
    calibrated_samples_by_pose: Sequence[np.ndarray],
) -> HeldOutWrenchValidation:
    """Classify held-out calibration using only pose-paired physical metrics.

    Acceptance requires no meaningful force or torque regression and at least
    one numerically improved metric.  A change inside the repeatability margin
    is treated as equivalent, not as evidence of degradation.
    """
    raw = tuple(
        _finite_wrench_samples(value, f"raw_samples_by_pose[{index}]")
        for index, value in enumerate(raw_samples_by_pose)
    )
    calibrated = tuple(
        _finite_wrench_samples(value, f"calibrated_samples_by_pose[{index}]")
        for index, value in enumerate(calibrated_samples_by_pose)
    )
    if len(raw) < 3 or len(raw) != len(calibrated):
        raise ValueError("held-out validation requires matching samples from at least three poses")
    if any(left.shape != right.shape for left, right in zip(raw, calibrated, strict=True)):
        raise ValueError("raw and calibrated held-out pose samples must align exactly")

    raw_rows = np.concatenate(raw, axis=0)
    calibrated_rows = np.concatenate(calibrated, axis=0)
    raw_force = _metric_rms(raw_rows, slice(0, 3))
    calibrated_force = _metric_rms(calibrated_rows, slice(0, 3))
    raw_torque = _metric_rms(raw_rows, slice(3, 6))
    calibrated_torque = _metric_rms(calibrated_rows, slice(3, 6))
    force_margin = _repeatability_margin(raw, slice(0, 3))
    torque_margin = _repeatability_margin(raw, slice(3, 6))
    force_delta = calibrated_force - raw_force
    torque_delta = calibrated_torque - raw_torque
    force_classification = _classification(force_delta, force_margin)
    torque_classification = _classification(torque_delta, torque_margin)
    passed = bool(
        force_classification != "regressed"
        and torque_classification != "regressed"
        and (force_delta < 0.0 or torque_delta < 0.0)
    )
    return HeldOutWrenchValidation(
        raw_force_rms_n=raw_force,
        calibrated_force_rms_n=calibrated_force,
        force_repeatability_margin_n=force_margin,
        force_classification=force_classification,
        raw_torque_rms_nm=raw_torque,
        calibrated_torque_rms_nm=calibrated_torque,
        torque_repeatability_margin_nm=torque_margin,
        torque_classification=torque_classification,
        passed=passed,
    )
