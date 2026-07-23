"""Mixed random + post-jitter-same-skill/different-task batches for renewed Stage-0."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator

import numpy as np
from torch.utils.data import BatchSampler

from lerobot.policies.skillVLA.skill_jitter import (
    JitterDraw,
    apply_jitter_draw,
    sample_jitter_draw,
)

log = logging.getLogger(__name__)

# Pair slots additionally carry the sampler-resolved (k_prime, offset), preventing the dataset from
# drawing a second independent jitter. Random/fallback slots retain the original three-field index.
GroupedSampleIndex = tuple[int, int, bool] | tuple[int, int, bool, int, int]


class SameSkillDifferentTaskBatchSampler(BatchSampler):
    """Build batches containing paired and ordinary random samples.

    Each valid pair starts from the same original FSQ code, shares one draw from the configured jitter
    law, and is retained only when its resolved post-jitter code also matches. Within other tasks,
    nearby post-jitter progress is sampled preferentially. Ordinary random slots keep independent
    dataset-side jitter, so this changes only the intended pair correlation, not their marginal law.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        grouped_fraction: float = 0.5,
        progress_temperature: float = 0.1,
        progress_candidates: int = 8,
        seed: int = 1000,
    ) -> None:
        if batch_size < 2:
            raise ValueError(f"same-skill batching needs batch_size >= 2, got {batch_size}.")
        if not 0.0 <= grouped_fraction <= 1.0:
            raise ValueError(f"grouped_fraction must be in [0, 1], got {grouped_fraction}.")
        if progress_temperature <= 0.0:
            raise ValueError(f"progress_temperature must be > 0, got {progress_temperature}.")
        if progress_candidates <= 0:
            raise ValueError(f"progress_candidates must be > 0, got {progress_candidates}.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.grouped_fraction = float(grouped_fraction)
        self.progress_temperature = float(progress_temperature)
        self.progress_candidates = int(progress_candidates)
        self.seed = int(seed)
        self.epoch = 0

        # Pair slots are always even.  N=16 and fraction=0.5 gives 4 pairs + 8 random samples.
        self.num_pairs = int(self.batch_size * self.grouped_fraction) // 2
        self.num_grouped_samples = 2 * self.num_pairs
        self.num_random_samples = self.batch_size - self.num_grouped_samples

        self._load_metadata()
        if self.num_pairs > 0 and self._eligible_indices.size == 0:
            raise ValueError(
                "No skill code occurs in more than one task; same-skill/different-task batches "
                "cannot be constructed for this dataset."
            )
        log.info(
            "Same-skill batch sampler: batch=%d, pairs=%d (%d samples), random=%d, "
            "eligible_anchors=%d/%d, progress_temperature=%.3f",
            self.batch_size,
            self.num_pairs,
            self.num_grouped_samples,
            self.num_random_samples,
            self._eligible_indices.size,
            len(self.dataset),
            self.progress_temperature,
        )

    @staticmethod
    def _flat(values, dtype) -> np.ndarray:
        return np.asarray(values, dtype=dtype).reshape(-1)

    def _load_metadata(self) -> None:
        columns = [
            "skill_sequence",
            "skill_sequence_len",
            "skill_length_sequence",
            "skill_initial_frame",
            "skill_index",
            "skill_ds",
            "skill_de",
            "task_index",
        ]
        missing = [name for name in columns if name not in self.dataset.hf_dataset.column_names]
        if missing:
            raise ValueError(f"Same-skill batch sampler needs parquet columns {missing}.")

        # skill_sequence is short (the dataset schema's maximum skills per episode), and the
        # remaining columns are scalars.  Reading only these columns avoids touching image/action data.
        metadata = self.dataset.hf_dataset.with_format(
            "numpy", columns=columns, output_all_columns=False
        )[:]
        skill_sequence = np.asarray(metadata["skill_sequence"], dtype=np.int64)
        if skill_sequence.ndim != 2:
            skill_sequence = np.stack(
                [np.asarray(row, dtype=np.int64).reshape(-1) for row in metadata["skill_sequence"]]
            )
        skill_index = self._flat(metadata["skill_index"], np.int64)
        if len(skill_index) != len(self.dataset):
            raise ValueError(
                f"Sampler metadata has {len(skill_index)} rows but dataset has {len(self.dataset)}."
            )
        if np.any(skill_index < 0) or np.any(skill_index >= skill_sequence.shape[1]):
            raise ValueError("Dataset contains a skill_index outside skill_sequence bounds.")

        rows = np.arange(len(skill_index), dtype=np.int64)
        self._skill = skill_sequence[rows, skill_index]
        self._skill_sequence = skill_sequence
        self._sequence_len = self._flat(metadata["skill_sequence_len"], np.int64)
        self._skill_lengths = np.asarray(metadata["skill_length_sequence"], dtype=np.int64)
        self._skill_starts = np.asarray(metadata["skill_initial_frame"], dtype=np.int64)
        if self._skill_lengths.ndim != 2:
            self._skill_lengths = np.stack(
                [np.asarray(row, dtype=np.int64).reshape(-1) for row in metadata["skill_length_sequence"]]
            )
        if self._skill_starts.ndim != 2:
            self._skill_starts = np.stack(
                [np.asarray(row, dtype=np.int64).reshape(-1) for row in metadata["skill_initial_frame"]]
            )
        self._task = self._flat(metadata["task_index"], np.int64)
        self._ds = self._flat(metadata["skill_ds"], np.int64)
        self._de = self._flat(metadata["skill_de"], np.int64)
        self._skill_index = skill_index
        self._progress = np.clip(
            self._ds / np.maximum(self._ds + self._de, 1.0), 0.0, 1.0)

        # (skill, task) pools are sorted by progress so partner lookup examines only nearby values.
        self._pools: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        eligible_skills: set[int] = set()
        for skill in np.unique(self._skill):
            skill_rows = np.flatnonzero(self._skill == skill)
            by_task: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for task in np.unique(self._task[skill_rows]):
                indices = skill_rows[self._task[skill_rows] == task]
                order = np.argsort(self._progress[indices], kind="stable")
                indices = indices[order]
                by_task[int(task)] = (indices, self._progress[indices])
            self._pools[int(skill)] = by_task
            if len(by_task) >= 2:
                eligible_skills.add(int(skill))
        self._eligible_indices = np.flatnonzero(np.isin(self._skill, list(eligible_skills)))

    def _jitter_outcome(self, index: int, draw: JitterDraw) -> tuple[int, int, int, float]:
        k = int(self._skill_index[index])
        kp, offset = apply_jitter_draw(
            k,
            int(self._ds[index]),
            int(self._de[index]),
            int(self._sequence_len[index]),
            draw,
        )
        code = int(self._skill_sequence[index, kp])
        current_frame = int(self._skill_starts[index, k]) + int(self._ds[index])
        denom = max(int(self._skill_lengths[index, kp]) - 1, 1)
        progress = float(np.clip(
            (current_frame - int(self._skill_starts[index, kp])) / denom, 0.0, 1.0))
        return kp, offset, code, progress

    def _nearest_candidates(
        self, indices: np.ndarray, progress: np.ndarray, target: float
    ) -> tuple[np.ndarray, np.ndarray]:
        pos = int(np.searchsorted(progress, target))
        radius = self.progress_candidates
        lo = max(0, pos - radius)
        hi = min(len(indices), pos + radius)
        candidate_indices = indices[lo:hi]
        candidate_progress = progress[lo:hi]
        if len(candidate_indices) > self.progress_candidates:
            nearest = np.argsort(np.abs(candidate_progress - target), kind="stable")[
                : self.progress_candidates
            ]
            candidate_indices = candidate_indices[nearest]
            candidate_progress = candidate_progress[nearest]
        return candidate_indices, candidate_progress

    @staticmethod
    def _unused_random(rng: np.random.Generator, size: int, used: set[int]) -> int | None:
        for _ in range(32):
            index = int(rng.integers(0, size))
            if index not in used:
                return index
        if len(used) >= size:
            return None
        start = int(rng.integers(0, size))
        for offset in range(size):
            index = (start + offset) % size
            if index not in used:
                return index
        return None

    def _sample_partner(
        self,
        anchor: int,
        draw: JitterDraw,
        rng: np.random.Generator,
        used: set[int],
    ) -> tuple[int, tuple[int, int], tuple[int, int]] | None:
        skill = int(self._skill[anchor])
        anchor_task = int(self._task[anchor])
        task_pools = self._pools[skill]
        other_tasks = [task for task in task_pools if task != anchor_task]
        target_progress = float(self._progress[anchor])
        anchor_kp, anchor_offset, anchor_code, anchor_progress = self._jitter_outcome(anchor, draw)
        all_candidates: list[np.ndarray] = []
        all_progress: list[np.ndarray] = []
        all_kp: list[np.ndarray] = []
        all_offset: list[np.ndarray] = []
        for task in other_tasks:
            indices, progress = task_pools[task]
            candidates, _ = self._nearest_candidates(indices, progress, target_progress)
            keep = np.asarray([int(index) not in used for index in candidates], dtype=bool)
            candidates = candidates[keep]
            outcomes = [self._jitter_outcome(int(index), draw) for index in candidates]
            keep_code = np.asarray([outcome[2] == anchor_code for outcome in outcomes], dtype=bool)
            candidates = candidates[keep_code]
            if candidates.size > 0:
                all_candidates.append(candidates)
                kept = [outcome for outcome, valid in zip(outcomes, keep_code, strict=True) if valid]
                all_progress.append(np.asarray([outcome[3] for outcome in kept], dtype=np.float32))
                all_kp.append(np.asarray([outcome[0] for outcome in kept], dtype=np.int64))
                all_offset.append(np.asarray([outcome[1] for outcome in kept], dtype=np.int64))
        if not all_candidates:
            return None
        candidates = np.concatenate(all_candidates)
        candidate_progress = np.concatenate(all_progress)
        candidate_kp = np.concatenate(all_kp)
        candidate_offset = np.concatenate(all_offset)
        logits = -np.abs(candidate_progress - anchor_progress) / self.progress_temperature
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        selected = int(rng.choice(len(candidates), p=weights))
        return (
            int(candidates[selected]),
            (anchor_kp, anchor_offset),
            (int(candidate_kp[selected]), int(candidate_offset[selected])),
        )

    def __iter__(self) -> Iterator[list[GroupedSampleIndex]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        for _ in range(len(self)):
            batch: list[GroupedSampleIndex] = []
            used: set[int] = set()
            for pair_id in range(self.num_pairs):
                draw = sample_jitter_draw(
                    self.dataset.jitter_pmax,
                    rng,
                    self.dataset.jitter_distribution,
                )
                anchor = partner = None
                anchor_jitter = partner_jitter = None
                for _ in range(32):
                    candidate = int(rng.choice(self._eligible_indices))
                    if candidate in used:
                        continue
                    matched = self._sample_partner(candidate, draw, rng, used | {candidate})
                    if matched is not None:
                        anchor = candidate
                        partner, anchor_jitter, partner_jitter = matched
                        break
                if anchor is None or partner is None:
                    # Retain the requested batch size and explicitly mark the two failed pair slots.
                    for _ in range(2):
                        fallback = self._unused_random(rng, len(self.dataset), used)
                        if fallback is None:
                            fallback = int(rng.integers(0, len(self.dataset)))
                        used.add(fallback)
                        batch.append((fallback, -1, True))
                    continue

                assert anchor_jitter is not None and partner_jitter is not None
                used.update((anchor, partner))
                batch.extend((
                    (anchor, pair_id, False, *anchor_jitter),
                    (partner, pair_id, False, *partner_jitter),
                ))

            for _ in range(self.num_random_samples):
                index = self._unused_random(rng, len(self.dataset), used)
                if index is None:
                    index = int(rng.integers(0, len(self.dataset)))
                used.add(index)
                batch.append((index, -1, False))

            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)
