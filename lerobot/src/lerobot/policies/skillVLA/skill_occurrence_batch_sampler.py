"""Random multi-chunk batches with one shared latent per skill occurrence."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator

import numpy as np
from torch.utils.data import BatchSampler

from lerobot.policies.skillVLA.skill_jitter import sample_p


log = logging.getLogger(__name__)

# ``SkillVLADataset.__getitem__`` interprets this seven-field index as
# (frame, pair_id, pair_fallback, selected_skill, start_offset, latent_group,
# effective_de). Keeping start and end perturbations separate is essential:
# one shared predictor-start observation must coexist with independently
# jittered samples near the occurrence's end.
OccurrenceSampleIndex = tuple[int, int, bool, int, int, int, int]


class SkillOccurrenceBatchSampler(BatchSampler):
    """Sample M random frames from each selected skill occurrence.

    ``batch_size`` counts independent skill occurrences, not flattened frames.
    Every occurrence receives one coherent transition-boundary draw and emits
    exactly ``samples_per_skill`` frame indices.  The flat rows are contiguous,
    and carry a local group id so the Stage-2 model can run the skill-start VLM
    and latent predictor once before broadcasting z back over the M chunks.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        samples_per_skill: int,
        seed: int = 1000,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(
                f"Skill-occurrence batch_size must be positive, got {batch_size}."
            )
        if samples_per_skill <= 1:
            raise ValueError(
                "SkillOccurrenceBatchSampler requires samples_per_skill > 1, "
                f"got {samples_per_skill}."
            )
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.samples_per_skill = int(samples_per_skill)
        self.seed = int(seed)
        self.epoch = 0
        self._load_occurrences()
        if not self._occurrences:
            raise ValueError("The SkillVLA dataset contains no valid skill occurrences.")
        log.info(
            "Skill-occurrence sampler: occurrences=%d, occurrence_batch=%d, "
            "random_chunks_per_skill=%d, flattened_chunks=%d",
            len(self._occurrences),
            self.batch_size,
            self.samples_per_skill,
            self.batch_size * self.samples_per_skill,
        )

    @staticmethod
    def _flat(values, dtype) -> np.ndarray:
        return np.asarray(values, dtype=dtype).reshape(-1)

    @staticmethod
    def _matrix(values, dtype) -> np.ndarray:
        array = np.asarray(values, dtype=dtype)
        if array.ndim != 2:
            array = np.stack(
                [np.asarray(row, dtype=dtype).reshape(-1) for row in values]
            )
        return array

    def _load_occurrences(self) -> None:
        columns = [
            "episode_index",
            "frame_index",
            "skill_sequence_len",
            "skill_initial_frame",
            "skill_length_sequence",
        ]
        missing = [
            name for name in columns if name not in self.dataset.hf_dataset.column_names
        ]
        if missing:
            raise ValueError(
                f"Skill-occurrence sampling needs parquet columns {missing}."
            )
        metadata = self.dataset.hf_dataset.with_format(
            "numpy", columns=columns, output_all_columns=False
        )[:]
        episode = self._flat(metadata["episode_index"], np.int64)
        frame = self._flat(metadata["frame_index"], np.int64)
        sequence_len = self._flat(metadata["skill_sequence_len"], np.int64)
        starts = self._matrix(metadata["skill_initial_frame"], np.int64)
        lengths = self._matrix(metadata["skill_length_sequence"], np.int64)
        if not (
            len(episode)
            == len(frame)
            == len(sequence_len)
            == starts.shape[0]
            == lengths.shape[0]
            == len(self.dataset)
        ):
            raise ValueError("Skill-occurrence sampler metadata length mismatch.")

        self._frame_index: dict[tuple[int, int], int] = {
            (int(ep), int(fr)): index
            for index, (ep, fr) in enumerate(zip(episode, frame, strict=True))
        }
        self._episode_length = {
            int(ep): int(frame[episode == ep].max()) + 1 for ep in np.unique(episode)
        }
        self._occurrences: list[tuple[int, int, int, int]] = []
        for ep in np.unique(episode):
            row = int(np.flatnonzero(episode == ep)[0])
            # skill_sequence_len includes EOS, hence N real skills = len - 1.
            for skill_index in range(max(int(sequence_len[row]) - 1, 0)):
                start = int(starts[row, skill_index])
                length = int(lengths[row, skill_index])
                if start >= 0 and length > 0:
                    self._occurrences.append(
                        (int(ep), skill_index, start, start + length)
                    )

    def _boundary_offset(
        self,
        rng: np.random.Generator,
        negative_limit: int,
        positive_limit: int,
    ) -> int:
        choices = []
        if negative_limit > 0:
            choices.append((-1, negative_limit))
        if positive_limit > 0:
            choices.append((1, positive_limit))
        if not choices:
            return 0
        sign, limit = choices[int(rng.integers(0, len(choices)))]
        return sign * sample_p(
            limit,
            rng,
            self.dataset.jitter_distribution,
        )

    def _sample_occurrence(
        self,
        occurrence_id: int,
        group_id: int,
        rng: np.random.Generator,
    ) -> list[OccurrenceSampleIndex]:
        episode, skill_index, original_start, original_end = self._occurrences[
            occurrence_id
        ]
        limits = self.dataset.jitter_directional_pmaxes
        start_offset = self._boundary_offset(
            rng,
            limits["early_start"],
            limits["late_start"],
        )
        end_offset = self._boundary_offset(
            rng,
            limits["early_end"],
            limits["late_end"],
        )
        episode_length = self._episode_length[episode]
        virtual_start = int(np.clip(original_start + start_offset, 0, episode_length - 1))
        virtual_end = int(np.clip(original_end + end_offset, virtual_start + 1, episode_length))
        frames = rng.choice(
            np.arange(virtual_start, virtual_end, dtype=np.int64),
            size=self.samples_per_skill,
            replace=(virtual_end - virtual_start) < self.samples_per_skill,
        )
        result: list[OccurrenceSampleIndex] = []
        for frame in frames:
            key = (episode, int(frame))
            if key not in self._frame_index:
                raise RuntimeError(
                    "Skill occurrence references a frame missing from the selected "
                    f"dataset: episode={episode}, frame={int(frame)}."
                )
            result.append(
                (
                    self._frame_index[key],
                    -1,
                    False,
                    skill_index,
                    start_offset,
                    group_id,
                    virtual_end - 1 - int(frame),
                )
            )
        return result

    def __iter__(self) -> Iterator[list[OccurrenceSampleIndex]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self._occurrences))
        for start in range(0, len(order), self.batch_size):
            selected = order[start : start + self.batch_size]
            batch: list[OccurrenceSampleIndex] = []
            for group_id, occurrence_id in enumerate(selected.tolist()):
                batch.extend(
                    self._sample_occurrence(occurrence_id, group_id, rng)
                )
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self._occurrences) / self.batch_size)
