"""Reuse occurrence-balanced sampling without expanding native episode metadata."""

from .skill_occurrence_batch_sampler import SkillOccurrenceBatchSampler


class ABCSkillOccurrenceBatchSampler(SkillOccurrenceBatchSampler):
    def _load_occurrences(self):
        dataset = self.dataset
        archive = dataset._skill_annotations["archive"]
        positions = dataset._skill_annotations["episode_position"]
        self._frame_index = {}
        self._episode_length = {}
        self._occurrences = []
        offset = 0
        for row, usable in zip(dataset._episode_rows, dataset._usable_lengths, strict=True):
            episode, length = int(row["episode_index"]), int(usable)
            position = positions[episode]
            starts = archive["skill_initial_frame"][position]
            lengths = archive["skill_length_sequence"][position]
            count = int(archive["skill_sequence_len"][position]) - 1
            self._episode_length[episode] = length
            self._frame_index.update(((episode, frame), offset + frame) for frame in range(length))
            for rank in range(count):
                start, size = int(starts[rank]), int(lengths[rank])
                if start >= 0 and size > 0:
                    self._occurrences.append((episode, rank, start, start + size))
            offset += length
