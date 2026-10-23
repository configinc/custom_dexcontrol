"""absolute_target_to_delta: the target_cartesian step bound (no robot SDK needed)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

_PATH = Path(__file__).resolve().parents[1] / "src/dexcontrol/core/vega/cartesian_target.py"
_spec = importlib.util.spec_from_file_location("cartesian_target", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
absolute_target_to_delta = _mod.absolute_target_to_delta

CURRENT = np.array([0.29, 0.2, 0.67, 0.95, 1.18, 0.70])
LIN, ROT = 0.065625, 0.2625  # 20 Hz server limits


def _apply(current, delta):
    pos = current[:3] + delta[:3]
    rot = R.from_euler("xyz", delta[3:6]) * R.from_euler("xyz", current[3:6])
    return pos, rot


def test_small_target_is_reached_in_one_step():
    target = CURRENT.copy()
    target[:3] += [0.01, -0.005, 0.002]
    target[3:6] = (R.from_euler("xyz", [0.02, 0.0, -0.03]) * R.from_euler("xyz", CURRENT[3:6])).as_euler("xyz")
    pos, rot = _apply(CURRENT, absolute_target_to_delta(target, CURRENT, LIN, ROT))
    np.testing.assert_allclose(pos, target[:3], atol=1e-12)
    assert (rot * R.from_euler("xyz", target[3:6]).inv()).magnitude() < 1e-9


def test_far_target_is_clipped_along_the_same_direction():
    target = CURRENT.copy()
    target[:3] += [0.3, 0.0, 0.4]
    target[3:6] = (R.from_rotvec([0.0, 0.0, 1.0]) * R.from_euler("xyz", CURRENT[3:6])).as_euler("xyz")
    delta = absolute_target_to_delta(target, CURRENT, LIN, ROT)
    assert np.linalg.norm(delta[:3]) == pytest.approx(LIN)
    np.testing.assert_allclose(delta[:3] / LIN, [0.6, 0.0, 0.8], atol=1e-12)
    np.testing.assert_allclose(R.from_euler("xyz", delta[3:6]).as_rotvec(), [0.0, 0.0, ROT], atol=1e-12)


def test_current_pose_gives_zero_delta():
    np.testing.assert_allclose(absolute_target_to_delta(CURRENT, CURRENT, LIN, ROT), 0.0, atol=1e-12)


@pytest.mark.parametrize("bad", [np.zeros(5), np.full(6, np.nan)])
def test_malformed_targets_are_rejected(bad):
    with pytest.raises(ValueError):
        absolute_target_to_delta(bad, CURRENT, LIN, ROT)
