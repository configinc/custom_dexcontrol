"""Current diagnostics consume frames without commanding or sanitizing state."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve().parents[1] / "src/dexcontrol/core/wuji_current.py"
spec = importlib.util.spec_from_file_location("wuji_current_tested", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
CurrentSummary = module.CurrentSummary


def entry(nid=1, current=.2, position=.4):
    return SimpleNamespace(nid=nid, effort=current, position=position)


def test_sampled_peak_sign_and_error_without_mutating_inputs():
    summary = CurrentSummary("left", 1.5, 3., .05)
    target = [.7] * 20
    sample = entry(current=-1.49)
    assert summary.sample([sample], lambda nid: nid - 1, target, 0.) is None
    result = summary.sample([entry(current=.2)], lambda nid: nid - 1, target, 1.)
    assert result['sampled_frames'] == 2
    row = result['joints'][0]
    assert row['peak_abs_a'] == 1.49 and row['current_a'] == .2
    assert row['sent_error_rad'] == pytest.approx(.3)
    assert row['joint'] == 0 and row['nid'] == 1
    assert sample.effort == -1.49 and sample.position == .4 and target == [.7] * 20
    assert result['configured_limit_a'] == 1.5


def test_partial_frames_age_and_window_reset():
    summary = CurrentSummary("right", 1.5, 3., .05)
    summary.sample([entry(1, 1.)], lambda nid: nid - 1, None, 0.)
    result = summary.sample([entry(6, .2)], lambda nid: nid - 1, None, 1.)
    assert [row['age_s'] for row in result['joints']] == [1., 0.]
    assert result['joints'][0]['sent_target_rad'] is None
    result = summary.sample([entry(6, .1)], lambda nid: nid - 1, None, 2.)
    assert len(result['joints']) == 1
    assert result['joints'][0]['peak_abs_a'] == .1


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), None])
def test_invalid_measurements_are_unknown_not_zero(bad):
    summary = CurrentSummary("left", 1.5, 3., .05)
    sample = entry(current=bad, position=bad)
    summary.sample([sample], lambda nid: 0, [float('nan')] * 20, 0.)
    report = summary.sample([sample], lambda nid: 0, None, 1.)
    row = report['joints'][0]
    for key in ('current_a', 'peak_abs_a', 'position_rad', 'sent_error_rad'):
        assert row[key] is None
    json.dumps(report, allow_nan=False)


def test_unknown_nodes_are_not_attributed_to_a_joint():
    summary = CurrentSummary("left", 1.5, 3., .05)
    summary.sample([entry()], lambda nid: -1, None, 0.)
    assert summary.sample([entry()], lambda nid: -1, None, 1.)['joints'] == []
