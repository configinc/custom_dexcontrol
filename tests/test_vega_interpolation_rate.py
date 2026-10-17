import pytest

from dexcontrol.core.vega.robot import _resample_smoothing_alpha


def test_resampled_smoothing_alpha_preserves_wall_clock_omega():
    source_alpha = 0.1
    source_hz = 20.0
    target_hz = 100.0

    target_alpha = _resample_smoothing_alpha(
        source_alpha, source_hz, target_hz
    )

    source_omega = source_alpha * source_hz / (1.0 - source_alpha)
    target_omega = target_alpha * target_hz / (1.0 - target_alpha)
    assert target_alpha < source_alpha
    assert target_omega == pytest.approx(source_omega)


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 0.99])
def test_resampled_smoothing_alpha_is_unchanged_at_same_rate(alpha):
    assert _resample_smoothing_alpha(alpha, 20.0, 20.0) == pytest.approx(alpha)
