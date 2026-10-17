"""Shared robust fitting mechanics for unloaded multi-pose wrench calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

import numpy as np

OUTLIER_FACTOR = 3.0
OUTLIER_FLOOR_N = 0.5
OUTLIER_MAX_FRACTION = 0.2
MIN_CLEAN_HOLDS = 8

HoldT = TypeVar("HoldT")
FitT = TypeVar("FitT")


@dataclass(frozen=True)
class RobustStaticFit:
    fit: Any
    kept_holds: tuple[Any, ...]
    dropped_pose_ids: tuple[str, ...]
    initial_force_residual_n: np.ndarray


def robust_static_wrench_refit(
    holds: Sequence[HoldT],
    solve: Callable[[tuple[HoldT, ...]], FitT],
    force_residual: Callable[[FitT], np.ndarray],
) -> RobustStaticFit:
    """Apply Seongmin's one-pass contaminated-hold rejection to any static model.

    Ten input poses allow the common 20% rejection rule to remove two bad poses
    while retaining the required eight clean observations.
    """
    all_holds = tuple(holds)
    if len(all_holds) < MIN_CLEAN_HOLDS:
        raise ValueError(f"at least {MIN_CLEAN_HOLDS} settled pose holds are required")
    if len({hold.pose_id for hold in all_holds}) != len(all_holds):
        raise ValueError("calibration pose ids must be unique")
    initial = solve(all_holds)
    residual = np.asarray(force_residual(initial), dtype=np.float64)
    if residual.shape != (len(all_holds),) or not np.isfinite(residual).all():
        raise ValueError("static calibration force residual must be finite per pose")
    threshold = max(
        OUTLIER_FACTOR * float(np.median(residual)),
        OUTLIER_FLOOR_N,
    )
    candidates = np.flatnonzero(residual > threshold)
    maximum_drop = int(OUTLIER_MAX_FRACTION * len(all_holds))
    dropped = candidates[np.argsort(-residual[candidates])][:maximum_drop]
    dropped_indices = set(int(index) for index in dropped)
    kept = tuple(
        hold for index, hold in enumerate(all_holds) if index not in dropped_indices
    )
    final = solve(kept) if dropped_indices else initial
    return RobustStaticFit(
        fit=final,
        kept_holds=kept,
        dropped_pose_ids=tuple(all_holds[index].pose_id for index in dropped),
        initial_force_residual_n=np.frombuffer(
            residual.tobytes(), dtype=np.float64
        ),
    )
