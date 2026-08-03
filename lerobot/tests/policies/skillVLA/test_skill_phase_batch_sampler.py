import numpy as np
import pytest

from lerobot.policies.skillVLA.skill_phase_batch_sampler import (
    SkillPhaseFocusedBatchSampler,
)


class _FakeHFDataset:
    column_names = ["skill_ds", "skill_de"]

    def __init__(self, ds: np.ndarray, de: np.ndarray) -> None:
        self._data = {"skill_ds": ds, "skill_de": de}

    def with_format(self, *_args, **_kwargs):
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._data
        return {key: value[index] for key, value in self._data.items()}


class _FakeDataset:
    def __init__(self) -> None:
        # 30 early, 30 middle and 30 late rows.
        self.ds = np.concatenate(
            (np.ones(30), np.full(30, 5), np.full(30, 9))
        )
        self.de = np.concatenate(
            (np.full(30, 9), np.full(30, 5), np.ones(30))
        )
        self.hf_dataset = _FakeHFDataset(self.ds, self.de)

    def __len__(self) -> int:
        return len(self.ds)


def test_phase_sampler_guarantees_configured_early_and_late_counts() -> None:
    dataset = _FakeDataset()
    sampler = SkillPhaseFocusedBatchSampler(
        dataset,
        batch_size=20,
        focused_fraction=0.8,
        early_fraction=0.5,
        early_threshold=0.25,
        late_threshold=0.75,
        seed=7,
    )

    batch = next(iter(sampler))
    progress = dataset.ds[batch] / (dataset.ds[batch] + dataset.de[batch])

    assert len(batch) == 20
    assert len(set(batch)) == 20
    assert np.count_nonzero(progress <= 0.25) >= 8
    assert np.count_nonzero(progress >= 0.75) >= 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"focused_fraction": 1.1},
        {"early_fraction": -0.1},
        {"early_threshold": 0.8, "late_threshold": 0.7},
    ],
)
def test_phase_sampler_rejects_invalid_hyperparameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SkillPhaseFocusedBatchSampler(_FakeDataset(), batch_size=20, **kwargs)
