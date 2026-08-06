from __future__ import annotations

import numpy as np
import pytest

from lerobot.policies.skill_aux.endpoint_batch_sampler import (
    TerminatorEndpointBatchSampler,
)


class _FakeHFDataset:
    column_names = ["skill_de"]

    def __init__(self, distance_to_end: np.ndarray) -> None:
        self._data = {"skill_de": distance_to_end}

    def with_format(self, *_args, **_kwargs):
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._data
        return {"skill_de": self._data["skill_de"][index]}


class _FakeDataset:
    def __init__(self) -> None:
        self.distance_to_end = np.concatenate(
            (
                np.zeros(30, dtype=np.int64),
                np.ones(30, dtype=np.int64),
                np.full(30, 2, dtype=np.int64),
                np.full(30, 8, dtype=np.int64),
            )
        )
        self.hf_dataset = _FakeHFDataset(self.distance_to_end)

    def __len__(self) -> int:
        return len(self.distance_to_end)


def test_endpoint_sampler_guarantees_exact_and_near_end_quotas() -> None:
    dataset = _FakeDataset()
    sampler = TerminatorEndpointBatchSampler(
        dataset,
        batch_size=20,
        exact_end_fraction=0.25,
        near_end_fraction=0.25,
        near_end_max_distance=2,
        seed=7,
    )

    batch = next(iter(sampler))
    distance = dataset.distance_to_end[batch]

    assert len(batch) == 20
    assert len(set(batch)) == 20
    assert np.count_nonzero(distance == 0) >= 5
    assert np.count_nonzero((distance > 0) & (distance <= 2)) >= 5
    assert sampler.num_uniform_samples == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exact_end_fraction": -0.1},
        {"near_end_fraction": 1.1},
        {"exact_end_fraction": 0.6, "near_end_fraction": 0.5},
        {"near_end_max_distance": 0},
    ],
)
def test_endpoint_sampler_rejects_invalid_hyperparameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        TerminatorEndpointBatchSampler(_FakeDataset(), batch_size=20, **kwargs)


def test_endpoint_sampler_requires_the_requested_pools() -> None:
    dataset = _FakeDataset()
    dataset.distance_to_end[:] = 8
    dataset.hf_dataset = _FakeHFDataset(dataset.distance_to_end)

    with pytest.raises(ValueError, match="skill_de == 0"):
        TerminatorEndpointBatchSampler(
            dataset,
            batch_size=8,
            exact_end_fraction=0.25,
            near_end_fraction=0.0,
        )
