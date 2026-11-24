"""Official-rule parity, stage switches, index alignment and supervision protection."""
from pathlib import Path
import runpy

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
M = runpy.run_path(str(ROOT / 'lerobot/skillVLA/droid/configs/src/idle_filter.py'))
Selector = M['TrainingIdleFilter']


def config(stage, enabled):
    return {'workflow_domain': 'droid', 'idle_filter': {stage: enabled}}


def reference(v):
    # Independent transcription of the official range-based implementation.
    idle = np.hstack(([False], np.all(np.abs(v[1:] - v[:-1]) < 1e-3, axis=1)))
    diff = np.diff(np.r_[False, idle, False].astype(int))
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    keep = np.ones(len(v), dtype=bool)
    for a, b in zip(starts, ends):
        if b - a >= 7:
            keep[a:b] = False
    diff = np.diff(np.r_[False, keep, False].astype(int))
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    result = np.zeros(len(v), dtype=bool)
    for a, b in zip(starts, ends):
        if b - a >= 16:
            result[a:b-10] = True
    return result


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_official_rule_parity(dtype):
    for seed in range(20):
        v = np.random.default_rng(seed).normal(size=(160, 7)).astype(dtype)
        v[30:51] = 0
        v[75:82] = 0
        v[110:136] = .5  # Official test can reject CONSTANT NONZERO commands too.
        np.testing.assert_array_equal(M['nonidle_start_mask'](v), reference(v))
    assert not M['nonidle_start_mask'](np.zeros((100, 7))).any()


@pytest.mark.parametrize('stage', M['STAGES'])
def test_each_switch_is_independent_and_false_preserves_sampler(stage):
    v = np.zeros((50, 7))
    off = Selector(config(stage, False), stage)
    sentinel = object()
    assert off.sampler(None, base_sampler=sentinel) is sentinel
    if stage == 'FSQ':
        assert off.skill_mask(v, [[0, 25], [25, 50]]).all()
        assert not Selector(config(stage, True), stage).skill_mask(v, [[0, 25], [25, 50]]).any()
    else:
        assert off.frame_mask(v).all()
        enabled = Selector(config(stage, True), stage)
        protected = np.zeros(50, dtype=bool) if stage.endswith('_TERM') else None
        assert not enabled.frame_mask(v, protected=protected).any()
    assert all(not value for key, value in M['validate_settings'](config(stage, True)).items() if key != stage)


def test_mask_selects_starts_only_and_preserves_full_future_chunk():
    v = np.arange(420, dtype=float).reshape(60, 7) / 100
    v[30:50] = 0
    source = np.asarray([i for i in range(60) if i not in (3, 4, 11)])
    targets = v[source].copy()
    eligible = Selector(config('PT_VSA', True), 'PT_VSA').frame_mask(v, source_frame_indices=source)
    np.testing.assert_equal(eligible, reference(v)[source])
    chosen = list(M['MaskedStartSampler'](eligible, shuffle=False))
    assert chosen == np.flatnonzero(eligible).tolist()
    np.testing.assert_equal(targets, v[source])
    # Chunk lookup remains on the unfiltered timeline, rather than eligible rows.
    np.testing.assert_equal(targets[chosen[-1]:chosen[-1]+15], v[source][chosen[-1]:chosen[-1]+15])


def test_term_requires_boundary_protection_and_keeps_tagged_positives():
    s = Selector(config('FT_TERM', True), 'FT_TERM')
    with pytest.raises(ValueError, match='protection'):
        s.frame_mask(np.zeros((40, 7)))
    protection = M['termination_protection_mask'](np.arange(40), np.arange(39, -1, -1), post_end_steps=3)
    keep = s.frame_mask(np.zeros((40, 7)), protected=protection)
    assert keep[:4].all() and keep[-2:].all() and not keep[10:30].any()
    sampler = s.sampler(keep, candidates=[0, 15, 38, (39, True)], shuffle=False)
    assert list(sampler) == [0, 38, (39, True)]


def test_bad_settings_and_empty_selection_fail_instead_of_silent_fallback():
    for value in ('false', 0, None):
        with pytest.raises(ValueError):
            Selector(config('DP', value), 'DP')
    with pytest.raises(ValueError):
        Selector({'workflow_domain': 'abc', 'idle_filter': {'DP': True}}, 'DP')
    with pytest.raises(ValueError):
        Selector(config('PT_PRDE', True), 'PT_PRDE')
    with pytest.raises(ValueError, match='no trainable'):
        M['MaskedStartSampler'](np.zeros(10, dtype=bool))


def test_missing_section_in_old_frozen_global_does_not_turn_filtering_on():
    assert not any(M['validate_settings']({'workflow_domain': 'droid'}).values())
