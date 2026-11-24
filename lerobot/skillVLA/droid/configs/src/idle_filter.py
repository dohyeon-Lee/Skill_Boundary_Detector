"""DROID-only training start selection; never edits trajectories or labels.

Reference: openpi/examples/droid/compute_droid_nonidle_ranges.py.
Its test is the *difference* of commanded joint velocities, not speed magnitude.
Compute on each original episode, then map through source.frame_index. This
avoids changing the rule when the converter removes inactive control samples.
"""
from __future__ import annotations

import numpy as np

STAGES = ('DP', 'FSQ', 'PT_VSA', 'PT_PRED', 'PT_TERM', 'FT_VSA', 'FT_PRED', 'FT_TERM')
RULE = {'name': 'openpi_droid_nonidle_v1', 'velocity_difference_threshold': 1e-3,
        'min_idle_frames': 7, 'min_nonidle_frames': 16, 'trim_range_end_frames': 10}


def validate_settings(config):
    """Missing section in an older frozen config means all off, not new defaults."""
    if config.get('workflow_domain') != 'droid':
        raise ValueError('DROID idle filtering requires workflow_domain: droid')
    switches = config.get('idle_filter', {})
    if not isinstance(switches, dict):
        raise ValueError('idle_filter must map the eight training stage names to true/false')
    unknown = set(switches) - set(STAGES)
    if unknown:
        raise ValueError(f'Unknown idle_filter stages: {sorted(unknown)}; choose {STAGES}')
    for stage, value in switches.items():
        if type(value) is not bool:
            raise ValueError(f'idle_filter.{stage} must be a YAML boolean true/false')
    return {stage: switches.get(stage, False) for stage in STAGES}


def _runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True)


def nonidle_start_mask(joint_velocities):
    """Exact OpenPI default range rule for one RAW episode, CPU only.

    Returns eligible *start indices*. In particular, do not use this mask to
    concatenate a shorter action trajectory or to recut skills. Constant
    nonzero velocity can also satisfy the official idle test; gripper-only
    motion is not protected by that test.
    """
    velocity = np.asarray(joint_velocities)
    if (velocity.ndim != 2 or velocity.shape[1] != 7 or velocity.dtype.kind not in 'fiu'
            or not np.isfinite(velocity).all()):
        raise ValueError('Expected finite, unnormalized RAW joint velocity commands shaped (N, 7)')
    if not len(velocity):
        return np.zeros(0, dtype=bool)
    idle = np.r_[False, np.all(np.abs(np.diff(velocity, axis=0)) < 1e-3, axis=1)]
    active = np.ones(len(velocity), dtype=bool)
    for start, end in _runs(idle):
        if end - start >= 7:
            active[start:end] = False
    starts = np.zeros(len(velocity), dtype=bool)
    for start, end in _runs(active):
        if end - start >= 16:
            starts[start:end - 10] = True
    return starts


def _frame_indices(indices, raw_length):
    if indices is None:
        return np.arange(raw_length, dtype=np.int64)
    value = np.asarray(indices)
    if (value.ndim != 1 or value.dtype.kind not in 'iu'
            or np.any(value < 0) or np.any(value >= raw_length)
            or np.any(np.diff(value.astype(np.int64)) <= 0)):
        raise ValueError('source.frame_index must be unique, increasing original episode indices')
    return value.astype(np.int64, copy=False)


def termination_protection_mask(skill_ds, skill_de, *, post_end_steps=0, positive=None):
    """Protect boundary negatives/positives and configured post-end supervision.

    ds/de describe the EXISTING converted skill labels, not recut trajectories.
    A caller with jittered or different labels should additionally pass its own
    positive mask (or construct the complete protected mask itself).
    """
    ds, de = np.asarray(skill_ds), np.asarray(skill_de)
    if (ds.ndim != 1 or de.shape != ds.shape or ds.dtype.kind not in 'iu'
            or de.dtype.kind not in 'iu' or np.any(ds < 0) or np.any(de < 0)
            or type(post_end_steps) is not int or post_end_steps < 0):
        raise ValueError('Expected nonnegative integer skill_ds/skill_de and post_end_steps')
    protected = (ds <= max(1, post_end_steps)) | (de <= 1)
    if positive is not None:
        protected |= _boolean_mask(positive, len(ds), 'positive')
    return protected


def _boolean_mask(value, length, name):
    value = np.asarray(value)
    if value.dtype != np.bool_ or value.shape != (length,):
        raise ValueError(f'{name} must be a boolean mask with {length} entries')
    return value


class TrainingIdleFilter:
    """Explicit stage selection; no inference from PT/FT paths or model names.

    DP and FSQ launchers resolve this class from their frozen global config.
    Future PT/FT launchers must install frame sampling at the final observation
    selection point (after any jitter), not at a trajectory reader/encoder.
    """

    def __init__(self, global_config, stage):
        switches = validate_settings(global_config)
        if stage not in STAGES:
            raise ValueError(f'Unknown training stage {stage!r}; choose {STAGES}')
        self.stage, self.enabled = stage, switches[stage]

    def contract(self):
        """Save with a future training config/checkpoint; resume uses this copy."""
        return {'stage': self.stage, 'enabled': self.enabled, 'rule': dict(RULE),
                'scope': 'training_sample_starts', 'preserve_trajectory': True,
                'termination_protection_required': self.stage.endswith('_TERM')}

    def frame_mask(self, raw_joint_velocities, *, source_frame_indices=None, protected=None):
        """Mask converted observation starts, preserving full future targets.

        TERM requires an explicit supervision-protection mask so callers cannot
        silently discard stationary positive labels or their boundary negatives.
        Any stage may protect additional examples, e.g. gripper transitions.
        """
        indices = _frame_indices(source_frame_indices, len(raw_joint_velocities))
        if not self.enabled:
            return np.ones(len(indices), dtype=bool)
        if self.stage == 'FSQ':
            raise ValueError('FSQ samples full skills: use skill_mask, not a frame mask')
        if self.stage.endswith('_TERM') and protected is None:
            raise ValueError('TERM idle filtering requires an explicit boundary/positive protection mask')
        keep = nonidle_start_mask(raw_joint_velocities)[indices]
        if protected is not None:
            keep |= _boolean_mask(protected, len(indices), 'protected')
        return keep

    def skill_mask(self, raw_joint_velocities, skill_ranges):
        """FSQ opt-in: select skill occurrences, NEVER remove frames inside one.

        Ranges are [start, end) in original RAW episode coordinates. Retain a
        skill if it contains at least one eligible non-idle start. The default
        FSQ=false keeps every skill, including stationary ones.
        """
        if self.stage != 'FSQ':
            raise ValueError('skill_mask is only for the FSQ training stage')
        spans = np.asarray(skill_ranges)
        if (spans.ndim != 2 or spans.shape[1] != 2 or spans.dtype.kind not in 'iu'
                or np.any(spans[:, 0] < 0) or np.any(spans[:, 1] > len(raw_joint_velocities))
                or np.any(spans[:, 0] >= spans[:, 1])):
            raise ValueError('skill_ranges must contain valid RAW [start, end) pairs')
        if not self.enabled:
            return np.ones(len(spans), dtype=bool)
        eligible = nonidle_start_mask(raw_joint_velocities)
        return np.asarray([eligible[start:end].any() for start, end in spans], dtype=bool)

    def sampler(self, mask, *, base_sampler=None, candidates=None, weights=None, shuffle=True):
        """False returns the existing sampler unchanged, including its RNG path.

        Enabled mode takes final frame/skill indices and optional existing
        sampling weights. Grouped/recurrent sampling must filter its final
        observation choices itself; a batch/list index is intentionally rejected.
        Tagged (frame, True) TERM episode-final positives retain their tag.
        """
        if not self.enabled:
            return base_sampler
        return MaskedStartSampler(mask, candidates=candidates, weights=weights,
                                  shuffle=shuffle, allow_terminal_tags=self.stage.endswith('_TERM'))


class MaskedStartSampler:
    """PyTorch-compatible sampler; dataset indices and action targets stay intact."""

    def __init__(self, mask, *, candidates=None, weights=None, shuffle=True, allow_terminal_tags=False):
        keep = np.asarray(mask)
        if keep.ndim != 1 or keep.dtype != np.bool_:
            raise ValueError('Expected a one-dimensional boolean start mask')
        candidates = list(range(len(keep))) if candidates is None else list(candidates)
        selected = []
        for item in candidates:
            tagged = isinstance(item, tuple)
            if tagged and (not allow_terminal_tags or len(item) != 2 or item[1] is not True):
                raise ValueError('Only TERM (frame, True) terminal-positive tags are supported')
            index = item[0] if tagged else item
            if (not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_))
                    or not 0 <= index < len(keep)):
                raise ValueError('Sampler candidates must be valid scalar indices; grouped batches are not supported')
            selected.append(bool(keep[index]) or tagged)
        selected = np.asarray(selected, dtype=bool)
        self.indices = [item for item, valid in zip(candidates, selected, strict=True) if valid]
        self.weights = None
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if (weights.shape != (len(candidates),) or not np.isfinite(weights).all() or np.any(weights < 0)):
                raise ValueError('Weights must be finite, nonnegative and aligned with original candidates')
            self.weights = weights[selected]
        if not self.indices or (self.weights is not None and not np.any(self.weights > 0)):
            raise ValueError('DROID idle filter leaves no trainable samples; inspect the selection, do not silently disable it')
        self.shuffle = bool(shuffle)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        # Use the existing global PyTorch RNG so full-state checkpoints preserve
        # sampling reproducibility. No private generator is reset on each epoch.
        import torch
        if self.weights is not None:
            order = torch.multinomial(torch.as_tensor(self.weights), len(self), replacement=True)
        else:
            order = torch.randperm(len(self)) if self.shuffle else range(len(self))
        for rank in order:
            yield self.indices[int(rank)]
