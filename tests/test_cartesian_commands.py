"""Unit tests for Vega Cartesian command semantics.

The production package imports robot-only dependencies from ``dexcontrol``'s
top-level module, so load this pure numerical helper directly for host-side
tests.
"""

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
    / "cartesian_commands.py"
)
_SPEC = importlib.util.spec_from_file_location("vega_cartesian_commands", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

clip_physical_cartesian_delta = _MODULE.clip_physical_cartesian_delta
normalized_cartesian_velocity_to_delta = (
    _MODULE.normalized_cartesian_velocity_to_delta
)


class PhysicalCartesianDeltaTests(unittest.TestCase):
    def test_small_physical_error_passes_through_unchanged(self) -> None:
        command = np.array([0.01, -0.02, 0.03, 0.10, -0.05, 0.02, 0.7])

        result = clip_physical_cartesian_delta(
            command,
            max_linear_delta=0.065625,
            max_rotation_delta=0.2625,
        )

        np.testing.assert_allclose(result, command)
        self.assertIsNot(result, command)

    def test_linear_and_rotation_norms_are_clipped_independently(self) -> None:
        command = np.array([3.0, 4.0, 0.0, 0.0, 0.0, -0.5, 0.25])

        result = clip_physical_cartesian_delta(
            command,
            max_linear_delta=0.1,
            max_rotation_delta=0.2,
        )

        np.testing.assert_allclose(result[:3], [0.06, 0.08, 0.0])
        np.testing.assert_allclose(result[3:6], [0.0, 0.0, -0.2])
        self.assertEqual(result[6], command[6])

    def test_zero_limit_zeroes_only_nonzero_cartesian_vectors(self) -> None:
        command = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4])

        result = clip_physical_cartesian_delta(
            command,
            max_linear_delta=0.0,
            max_rotation_delta=0.0,
        )

        np.testing.assert_allclose(result[:6], np.zeros(6))
        self.assertEqual(result[6], command[6])

    def test_invalid_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            clip_physical_cartesian_delta(
                np.zeros(7),
                max_linear_delta=-0.1,
                max_rotation_delta=0.2,
            )


class LegacyCartesianVelocityTests(unittest.TestCase):
    def test_small_normalized_velocity_is_still_scaled(self) -> None:
        command = np.array([0.1, 0.0, 0.0, 0.0, -0.2, 0.0, 0.7])

        result = normalized_cartesian_velocity_to_delta(
            command,
            max_linear_delta=0.065625,
            max_rotation_delta=0.2625,
        )

        np.testing.assert_allclose(result[:3], [0.0065625, 0.0, 0.0])
        np.testing.assert_allclose(result[3:6], [0.0, -0.0525, 0.0])
        self.assertEqual(result[6], command[6])

    def test_normalized_velocity_is_norm_limited_before_scaling(self) -> None:
        command = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 2.0])

        result = normalized_cartesian_velocity_to_delta(
            command,
            max_linear_delta=0.1,
            max_rotation_delta=0.2,
        )

        np.testing.assert_allclose(result[:3], [0.06, 0.08, 0.0])
        np.testing.assert_allclose(result[3:6], [0.0, 0.0, 0.2])


if __name__ == "__main__":
    unittest.main()
