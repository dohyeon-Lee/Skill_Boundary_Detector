"""Batch sampling that guarantees exact and near skill-end observations."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator

import numpy as np
from torch.utils.data import BatchSampler


log = logging.getLogger(__name__)


class TerminatorEndpointBatchSampler(BatchSampler):
    """Mix exact endpoints, near endpoints, and ordinary uniform samples.

    Pools are defined from the original current-frame ``skill_de`` metadata;
    predictor-side transition jitter, when enabled, is still applied later by
    the dataset after an index has been selected.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        exact_end_fraction: float = 0.25,
        near_end_fraction: float = 0.25,
        near_end_max_distance: int = 2,
        seed: int = 1000,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if not 0.0 <= exact_end_fraction <= 1.0:
            raise ValueError(
                "exact_end_fraction must be in [0, 1], got "
                f"{exact_end_fraction}."
            )
        if not 0.0 <= near_end_fraction <= 1.0:
            raise ValueError(
                "near_end_fraction must be in [0, 1], got "
                f"{near_end_fraction}."
            )
        if exact_end_fraction + near_end_fraction > 1.0:
            raise ValueError(
                "exact_end_fraction + near_end_fraction must be <= 1."
            )
        if near_end_max_distance < 1:
            raise ValueError("near_end_max_distance must be at least 1.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.exact_end_fraction = float(exact_end_fraction)
        self.near_end_fraction = float(near_end_fraction)
        self.near_end_max_distance = int(near_end_max_distance)
        self.seed = int(seed)
        self.epoch = 0

        self.num_exact_end_samples = round(
            self.batch_size * self.exact_end_fraction
        )
        self.num_near_end_samples = round(
            self.batch_size * self.near_end_fraction
        )
        focused = self.num_exact_end_samples + self.num_near_end_samples
        if focused > self.batch_size:
            # Independent rounding can exceed the batch size by one at the
            # boundary even when the real-valued fractions sum to one.
            self.num_near_end_samples -= focused - self.batch_size
        self.num_uniform_samples = (
            self.batch_size
            - self.num_exact_end_samples
            - self.num_near_end_samples
        )
        self._load_metadata()
        if self.num_exact_end_samples > 0 and self._exact_end_indices.size == 0:
            raise ValueError("No dataset rows satisfy skill_de == 0.")
        if self.num_near_end_samples > 0 and self._near_end_indices.size == 0:
            raise ValueError(
                "No dataset rows satisfy 0 < skill_de <= "
                f"{self.near_end_max_distance}."
            )
        log.info(
            "Terminator endpoint sampler: batch=%d exact=%d near=%d uniform=%d "
            "near_distance<=%d",
            self.batch_size,
            self.num_exact_end_samples,
            self.num_near_end_samples,
            self.num_uniform_samples,
            self.near_end_max_distance,
        )

    def _load_metadata(self) -> None:
        hf_dataset = getattr(self.dataset, "hf_dataset", None)
        if hf_dataset is None:
            raise ValueError(
                "Endpoint oversampling requires a frame-level dataset with "
                "hf_dataset metadata."
            )
        if "skill_de" not in hf_dataset.column_names:
            raise ValueError(
                "Endpoint oversampling requires the parquet column 'skill_de'."
            )
        metadata = hf_dataset.with_format(
            "numpy", columns=["skill_de"], output_all_columns=False
        )[:]
        self._distance_to_end = np.asarray(
            metadata["skill_de"], dtype=np.int64
        ).reshape(-1)
        if len(self._distance_to_end) != len(self.dataset):
            raise ValueError(
                "Endpoint sampler metadata length does not match the dataset: "
                f"{len(self._distance_to_end)} != {len(self.dataset)}."
            )
        self._all_indices = np.arange(len(self.dataset), dtype=np.int64)
        self._exact_end_indices = np.flatnonzero(self._distance_to_end == 0)
        self._near_end_indices = np.flatnonzero(
            (self._distance_to_end > 0)
            & (self._distance_to_end <= self.near_end_max_distance)
        )

    @staticmethod
    def _draw(
        rng: np.random.Generator,
        pool: np.ndarray,
        count: int,
        used: set[int],
    ) -> list[int]:
        if count <= 0:
            return []
        used_array = np.fromiter(used, dtype=np.int64)
        available = pool[~np.isin(pool, used_array)]
        source = available if available.size > 0 else pool
        replace = source.size < count
        chosen = rng.choice(source, size=count, replace=replace)
        result = [int(index) for index in np.asarray(chosen).reshape(-1)]
        used.update(result)
        return result

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(len(self)):
            used: set[int] = set()
            batch = self._draw(
                rng,
                self._exact_end_indices,
                self.num_exact_end_samples,
                used,
            )
            batch.extend(
                self._draw(
                    rng,
                    self._near_end_indices,
                    self.num_near_end_samples,
                    used,
                )
            )
            batch.extend(
                self._draw(
                    rng,
                    self._all_indices,
                    self.num_uniform_samples,
                    used,
                )
            )
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)
