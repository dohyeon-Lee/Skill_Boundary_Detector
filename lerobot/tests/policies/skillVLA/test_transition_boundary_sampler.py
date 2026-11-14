from types import SimpleNamespace

import draccus
import numpy as np
import pytest
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.transition_oversampling import TransitionOversamplingConfig
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.skillVLA.transition_boundary_sampler import TransitionBoundarySampler


class Metadata:
    def __init__(self, values):
        self.values = values
        self.column_names = list(values)

    def with_format(self, _, *, columns, output_all_columns):
        assert not output_all_columns
        assert set(columns) <= set(self.values)
        return Metadata({key: self.values[key] for key in columns})

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.values[item]
        return {key: value[item] for key, value in self.values.items()}


class Dataset:
    def __init__(self, lengths, episodes=1):
        rows = []
        for _ in range(episodes):
            for skill, length in enumerate(lengths):
                rows.extend((skill, len(lengths) + 1, lengths + [0], ds, length - ds - 1) for ds in range(length))
        self.hf_dataset = Metadata({name: np.asarray([row[i] for row in rows]) for i, name in enumerate(
            ['skill_index', 'skill_sequence_len', 'skill_length_sequence', 'skill_ds', 'skill_de'])})

    def __len__(self):
        return len(self.hf_dataset.values['skill_index'])

    def __getitem__(self, item):
        raise AssertionError('Sampler must never decode images, draw jitter or inspect gripper validity')


def config(**kwargs):
    return TransitionOversamplingConfig.from_mapping({'enabled': True, **kwargs})


def test_asymmetric_100_50_boundary_and_half_up_rounding():
    sampler = TransitionBoundarySampler(Dataset([100, 50]), config())
    indices = torch.nonzero(sampler.weights == 2).flatten().tolist()
    assert indices == list(range(87, 125)) + list(range(137, 150))  # internal 13+25, final 13
    assert sampler.boundary_frames == 51 and len(sampler) == 150
    assert sampler.episode_end_frames == 13
    assert sampler.weights.device.type == 'cpu'


def test_short_skill_overlaps_are_not_multiplied_and_do_not_cross_episode_starts():
    sampler = TransitionBoundarySampler(Dataset([100, 4, 100], episodes=2), config())
    expected = list(range(99, 105)) + list(range(179, 204)) + list(range(303, 309)) + list(range(383, 408))
    assert torch.nonzero(sampler.weights == 2).flatten().tolist() == expected
    assert sampler.weights.max() == 2
    assert sampler.weights[0] == sampler.weights[204] == 1
    assert sampler.weights[203] == sampler.weights[-1] == 2


def test_single_skill_episode_oversamples_its_own_last_quarter_only():
    sampler = TransitionBoundarySampler(Dataset([100], episodes=2), config())
    assert torch.nonzero(sampler.weights == 2).flatten().tolist() == list(range(75, 100)) + list(range(175, 200))
    assert sampler.episode_end_frames == sampler.boundary_frames == 50


def test_final_window_overlapping_an_internal_window_is_weighted_once():
    sampler = TransitionBoundarySampler(Dataset([100, 4]), config())
    assert sampler.weights[100:].tolist() == [2] * 4
    assert sampler.weights.max() == 2
    assert sampler.episode_end_frames == 1


def test_uniform_weighted_draws_preserve_epoch_size_and_expected_boundary_frequency():
    sampler = TransitionBoundarySampler(Dataset([100, 50]), config())
    torch.manual_seed(123)
    draws = [index for _ in range(100) for index in sampler]
    assert len(draws) == 15000
    actual = sum(87 <= index < 125 or 137 <= index < 150 for index in draws) / len(draws)
    assert abs(actual - 102 / 201) < .02
    assert len(set(draws)) == 150


def test_existing_drop_frame_filter_is_preserved():
    data = Dataset([100, 50])
    allowed = EpisodeAwareSampler([0], [150], drop_n_last_frames=30).indices
    sampler = TransitionBoundarySampler(data, config(), indices=allowed)
    draws = list(sampler)
    assert len(draws) == 120 and max(draws) < 120


def test_terminator_keeps_tagged_post_end_positives_and_boundary_weights():
    from lerobot.policies.skillVLA.newterm_supervision import NewTermFrameSampler

    data = Dataset([100, 50], episodes=2)
    data.hf_dataset.values['episode_index'] = np.repeat([0, 1], 150)
    data.hf_dataset.column_names.append('episode_index')
    original = NewTermFrameSampler(data, 10, shuffle=False)
    sampler = TransitionBoundarySampler(data, config(), post_end_steps=10)
    assert sampler.indices == original.indices
    assert len(sampler) == 320 and sampler.boundary_frames == 102
    assert sampler.weights[:300].tolist() == TransitionBoundarySampler(data, config()).weights.tolist()
    assert sampler.weights[300:].tolist() == [1] * 20
    torch.manual_seed(42)
    draws = [item for _ in range(10) for item in sampler]
    tagged = [item for item in draws if isinstance(item, tuple)]
    assert tagged and set(tagged) == {(149, True), (299, True)}
    assert all(0 <= item < 300 for item in draws if not isinstance(item, tuple))


def test_disabled_and_legacy_checkpoints_keep_sampler_rng_and_defaults_unchanged():
    legacy = draccus.decode(TrainPipelineConfig, {'dataset': {'repo_id': 'test'}})
    assert legacy.transition_oversampling.enabled is False
    before = torch.random.get_rng_state().clone()
    baseline = EpisodeAwareSampler([0], [150], shuffle=True)
    expected = list(baseline)
    torch.random.set_rng_state(before)
    off = TransitionOversamplingConfig.from_mapping({'enabled': False})
    off.validate()
    assert off.suffix == ''
    # Disabled options neither read skill metadata nor consume an extra draw.
    with pytest.raises(ValueError, match='original sampler'):
        TransitionBoundarySampler(SimpleNamespace(), off)
    actual = list(EpisodeAwareSampler([0], [150], shuffle=True))
    assert actual == expected


def test_weights_need_no_new_dataset_or_validity_columns():
    data = Dataset([100, 50])
    sampler = TransitionBoundarySampler(data, config())
    assert sampler.boundary_frames == 51
    data.hf_dataset.column_names.remove('skill_de')
    with pytest.raises(ValueError, match='metadata'):
        TransitionBoundarySampler(data, config())


@pytest.mark.parametrize('kwargs', [{'percent': 0}, {'percent': -1}, {'percent': float('nan')},
                                  {'weight': .5}, {'weight': float('inf')}, {'distribution': 'half_normal'},
                                  {'percent': True}, {'enabled': 'false'}, {'preserve_gripper': True}])
def test_invalid_enabled_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)
