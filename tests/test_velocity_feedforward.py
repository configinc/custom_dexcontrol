"""Unit tests for target-derivative velocity feedforward."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "dexcontrol"
    / "core"
    / "vega"
    / "velocity_feedforward.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "vega_velocity_feedforward",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

TargetVelocityFeedforward = _MODULE.TargetVelocityFeedforward


class TargetVelocityFeedforwardTests(unittest.TestCase):
    def test_first_target_has_zero_velocity(self) -> None:
        estimator = TargetVelocityFeedforward()

        velocity = estimator.update(np.array([0.2, -0.4]), timestamp=10.0)

        np.testing.assert_allclose(velocity, np.zeros(2))

    def test_uses_target_difference_and_nominal_period(self) -> None:
        estimator = TargetVelocityFeedforward(
            nominal_dt_s=0.05,
            stale_timeout_s=0.2,
        )
        estimator.update(np.array([0.0, 0.2]), timestamp=10.0)

        velocity = estimator.update(
            np.array([0.05, 0.1]),
            timestamp=10.03,
        )

        # Arrival jitter is 30 ms, but the configured input period is 50 ms.
        np.testing.assert_allclose(velocity, [1.0, -2.0])

    def test_applies_velocity_ratio(self) -> None:
        estimator = TargetVelocityFeedforward(
            velocity_ratio=0.5,
            smoothing_alpha=1.0,
            stale_timeout_s=0.2,
        )
        estimator.update(np.zeros(2), timestamp=1.0)

        velocity = estimator.update(
            np.array([0.2, -0.2]),
            timestamp=1.05,
        )

        # Raw [4, -4] rad/s, then vel_ratio=0.5.
        np.testing.assert_allclose(velocity, [2.0, -2.0])

    def test_smoothing_is_applied_between_input_target_velocities(self) -> None:
        estimator = TargetVelocityFeedforward(
            smoothing_alpha=0.5,
            stale_timeout_s=0.2,
        )
        estimator.update(np.array([0.0]), timestamp=2.0)

        first_velocity = estimator.update(
            np.array([0.1]),
            timestamp=2.05,
        )
        second_velocity = estimator.update(
            np.array([0.3]),
            timestamp=2.10,
        )

        np.testing.assert_allclose(first_velocity, [1.0])
        np.testing.assert_allclose(second_velocity, [2.5])

    def test_stale_command_stream_returns_zero(self) -> None:
        estimator = TargetVelocityFeedforward(
            stale_timeout_s=0.1,
        )
        estimator.update(np.array([0.0]), timestamp=3.0)
        estimator.update(np.array([0.1]), timestamp=3.05)

        fresh_velocity = estimator.sample(timestamp=3.10)
        stale_velocity = estimator.sample(timestamp=3.16)

        assert fresh_velocity is not None
        assert stale_velocity is not None
        np.testing.assert_allclose(fresh_velocity, [2.0])
        np.testing.assert_allclose(stale_velocity, [0.0])

    def test_new_target_after_stale_gap_restarts_at_zero(self) -> None:
        estimator = TargetVelocityFeedforward(
            stale_timeout_s=0.1,
        )
        estimator.update(np.array([0.0]), timestamp=4.0)
        estimator.update(np.array([0.1]), timestamp=4.05)

        velocity = estimator.update(np.array([1.0]), timestamp=4.30)

        np.testing.assert_allclose(velocity, [0.0])


if __name__ == "__main__":
    unittest.main()
