"""Batch sampler that emphasizes the beginning and end of complete skills."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator

import numpy as np
from torch.utils.data import BatchSampler


log = logging.getLogger(__name__)


class SkillPhaseFocusedBatchSampler(BatchSampler):
    """Mix guaranteed early/late samples with ordinary random samples.

    Phase is always computed from the original physical skill as
    ``skill_ds / (skill_ds + skill_de)``. Dataset-side transition jitter remains
    independent and is applied after an index is selected.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        focused_fraction: float = 0.75,
        early_fraction: float = 0.5,
        early_threshold: float = 0.25,
        late_threshold: float = 0.75,
        seed: int = 1000,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if not 0.0 <= focused_fraction <= 1.0:
            raise ValueError(
                f"focused_fraction must be in [0, 1], got {focused_fraction}."
            )
        if not 0.0 <= early_fraction <= 1.0:
            raise ValueError(
                f"early_fraction must be in [0, 1], got {early_fraction}."
            )
        if not 0.0 <= early_threshold < late_threshold <= 1.0:
            raise ValueError(
                "Phase thresholds must satisfy "
                f"0 <= early < late <= 1, got {early_threshold}, {late_threshold}."
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.focused_fraction = float(focused_fraction)
        self.early_fraction = float(early_fraction)
        self.early_threshold = float(early_threshold)
        self.late_threshold = float(late_threshold)
        self.seed = int(seed)
        self.epoch = 0

        focused_samples = round(self.batch_size * self.focused_fraction)
        self.num_early_samples = round(focused_samples * self.early_fraction)
        self.num_late_samples = focused_samples - self.num_early_samples
        self.num_random_samples = self.batch_size - focused_samples
        self._load_metadata()
        if self.num_early_samples > 0 and self._early_indices.size == 0:
            raise ValueError("No samples satisfy the configured early threshold.")
        if self.num_late_samples > 0 and self._late_indices.size == 0:
            raise ValueError("No samples satisfy the configured late threshold.")
        log.info(
            "Skill-phase sampler: batch=%d early=%d late=%d random=%d "
            "thresholds=(%.3f, %.3f)",
            self.batch_size,
            self.num_early_samples,
            self.num_late_samples,
            self.num_random_samples,
            self.early_threshold,
            self.late_threshold,
        )

    def _load_metadata(self) -> None:
        columns = ["skill_ds", "skill_de"]
        missing = [
            name for name in columns if name not in self.dataset.hf_dataset.column_names
        ]
        if missing:
            raise ValueError(f"Skill-phase sampler needs parquet columns {missing}.")
        metadata = self.dataset.hf_dataset.with_format(
            "numpy", columns=columns, output_all_columns=False
        )[:]
        ds = np.asarray(metadata["skill_ds"], dtype=np.float64).reshape(-1)
        de = np.asarray(metadata["skill_de"], dtype=np.float64).reshape(-1)
        if len(ds) != len(self.dataset):
            raise ValueError(
                f"Sampler metadata has {len(ds)} rows but dataset has {len(self.dataset)}."
            )
        self._progress = np.clip(ds / np.maximum(ds + de, 1.0), 0.0, 1.0)
        self._all_indices = np.arange(len(self.dataset), dtype=np.int64)
        self._early_indices = np.flatnonzero(
            self._progress <= self.early_threshold
        )
        self._late_indices = np.flatnonzero(
            self._progress >= self.late_threshold
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
        available = pool[~np.isin(pool, np.fromiter(used, dtype=np.int64))]
        replace = available.size < count
        source = available if available.size > 0 else pool
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
                rng, self._early_indices, self.num_early_samples, used
            )
            batch.extend(
                self._draw(rng, self._late_indices, self.num_late_samples, used)
            )
            batch.extend(
                self._draw(rng, self._all_indices, self.num_random_samples, used)
            )
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)
