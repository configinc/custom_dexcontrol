import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

_MODULE_PATH = (
    Path(__file__).parents[1] / "src" / "dexcontrol" / "core" / "cartesian_pose.py"
)
_SPEC = importlib.util.spec_from_file_location("cartesian_pose", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
add_poses = _MODULE.add_poses


def test_add_poses_left_composes_rotation_delta():
    source = np.array([0.4, -0.2, 1.1, 0.7, -0.4, 0.2])
    delta = np.array([0.03, 0.01, -0.02, -0.3, 0.25, 0.15])

    result = add_poses(delta, source)

    np.testing.assert_allclose(result[:3], source[:3] + delta[:3])
    np.testing.assert_allclose(
        R.from_euler("xyz", result[3:6]).as_matrix(),
        (
            R.from_euler("xyz", delta[3:6]) * R.from_euler("xyz", source[3:6])
        ).as_matrix(),
        atol=1e-12,
    )
    assert not np.allclose(result[3:6], source[3:6] + delta[3:6])


def test_add_poses_rejects_non_pose_vectors():
    with np.testing.assert_raises_regex(ValueError, "must both be 6D"):
        add_poses(np.zeros(7), np.zeros(6))
