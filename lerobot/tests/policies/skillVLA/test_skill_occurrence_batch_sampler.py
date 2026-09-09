import numpy as np

from lerobot.policies.skillVLA.skill_occurrence_batch_sampler import (
    SkillOccurrenceBatchSampler,
)


class _FakeHFDataset:
    column_names = [
        "episode_index",
        "frame_index",
        "skill_sequence_len",
        "skill_initial_frame",
        "skill_length_sequence",
    ]

    def __init__(self) -> None:
        starts = np.asarray([[0, 3, -1]] * 6, dtype=np.int64)
        lengths = np.asarray([[3, 3, 0]] * 6, dtype=np.int64)
        self.values = {
            "episode_index": np.zeros(6, dtype=np.int64),
            "frame_index": np.arange(6, dtype=np.int64),
            "skill_sequence_len": np.full(6, 3, dtype=np.int64),
            "skill_initial_frame": starts,
            "skill_length_sequence": lengths,
        }

    def with_format(self, *args, **kwargs):
        del args, kwargs
        return self

    def __getitem__(self, index):
        assert isinstance(index, slice)
        return self.values


class _FakeDataset:
    def __init__(self) -> None:
        self.hf_dataset = _FakeHFDataset()
        self.jitter_directional_pmaxes = {
            "early_start": 0,
            "late_start": 0,
            "early_end": 0,
            "late_end": 0,
        }
        self.jitter_distribution = "half_normal"

    def __len__(self) -> int:
        return 6


def test_occurrence_sampler_emits_random_chunks_in_dense_groups() -> None:
    sampler = SkillOccurrenceBatchSampler(
        _FakeDataset(), batch_size=2, samples_per_skill=2, seed=7
    )

    batch = next(iter(sampler))

    assert len(batch) == 4
    assert [sample[-1] for sample in batch] == [0, 0, 1, 1]
    for sample in batch:
        frame_index, pair_id, fallback, selected_skill, offset, _ = sample
        assert 0 <= frame_index < 6
        assert pair_id == -1
        assert fallback is False
        assert selected_skill in {0, 1}
        assert offset == 0

