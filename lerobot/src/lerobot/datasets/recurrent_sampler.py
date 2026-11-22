"""Chronological episode streams for truncated-BPTT GRU training."""
import heapq
import math
import random


def recurrent_episode_ranges(dataset, drop_n_last_frames: int = 0):
    """Find usable episode spans in the dataset's actual local frame index."""
    if drop_n_last_frames < 0:
        raise ValueError("drop_n_last_frames must be nonnegative.")
    if hasattr(dataset, 'episode_sample_ranges'):
        # ABC's native reader already removes incomplete action targets and
        # compacts selected episodes. Do not trim those tails a second time.
        return dataset.episode_sample_ranges

    import numpy as np

    # LeRobot's episode metadata contains global offsets. Reading the loaded
    # index columns also handles episode subsets whose local offsets differ.
    columns = dataset.hf_dataset.select_columns(['episode_index', 'frame_index']).with_format('numpy')[:]
    episodes = np.asarray(columns['episode_index']).reshape(-1)
    frames = np.asarray(columns['frame_index']).reshape(-1)
    if not len(episodes):
        raise ValueError("No frames remain for GRU_epi training.")
    starts = np.r_[0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1]
    lengths = np.diff(np.r_[starts, len(episodes)])
    if len(np.unique(episodes[starts])) != len(starts):
        raise ValueError("GRU_epi requires contiguous episode frames.")
    expected_frames = np.arange(len(frames)) - np.repeat(starts, lengths)
    if not np.array_equal(frames, expected_frames):
        raise ValueError("GRU_epi requires complete chronological episodes starting at frame zero.")
    return [(int(start), int(length - drop_n_last_frames))
            for start, length in zip(starts, lengths, strict=True)
            if length > drop_n_last_frames]


class RecurrentEpisodeBatchSampler:
    """Full episode streams with continuous state across finite loader passes."""

    def __init__(self, episode_ranges, batch_size: int, chunk_length: int, seed: int = 42):
        if batch_size < 1 or chunk_length < 1:
            raise ValueError("Recurrent batch size and chunk length must be positive.")
        self.ranges = [(int(start), int(length)) for start, length in episode_ranges if length > 0]
        if not self.ranges:
            raise ValueError("No complete action targets remain for GRU_epi training.")
        if len(self.ranges) < batch_size:
            raise ValueError("GRU_epi needs at least batch_size usable episodes for distinct full streams.")
        self.batch_size = batch_size
        self.chunk_length = chunk_length
        self.seed = seed if seed is not None else 0
        self.epoch = 0
        self._plan = None
        self._rng = random.Random(self.seed)
        # Prefer episodes with fewer starts. Active episodes remain outside the
        # heap, so a long episode cannot block replenishing other streams or
        # appear twice in one batch.
        self._available = [(0, self._rng.random(), start, length)
                           for start, length in self.ranges]
        heapq.heapify(self._available)
        self._active = []
        self._batches_per_pass = max(
            1, math.ceil(sum(length for _, length in self.ranges) / (chunk_length * batch_size))
        )

    def _batches(self):
        if self._plan is not None:
            return self._plan
        plan = []
        for _ in range(self._batches_per_pass):
            while len(self._active) < self.batch_size:
                visits, _, start, length = heapq.heappop(self._available)
                # Randomize the boundary phase on each episode start. States before
                # the first anchor are encoded too, so causal history is complete.
                anchor = self._rng.randrange(min(self.chunk_length, length))
                self._active.append((start, length, anchor, visits))
            plan.append([start + anchor for start, _length, anchor, _visits in self._active])
            continuing = []
            for start, length, anchor, visits in self._active:
                next_anchor = anchor + self.chunk_length
                if next_anchor < length:
                    continuing.append((start, length, next_anchor, visits))
                else:
                    heapq.heappush(self._available, (visits + 1, self._rng.random(), start, length))
            self._active = continuing
        self._plan = plan
        return plan

    def __len__(self):
        return self._batches_per_pass

    def __iter__(self):
        try:
            yield from self._batches()
        finally:
            self.epoch += 1
            self._plan = None
