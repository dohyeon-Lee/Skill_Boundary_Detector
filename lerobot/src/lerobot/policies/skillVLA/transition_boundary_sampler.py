"""Uniform weights in percentage-jitter windows and the final skill's end.

Only clean parquet metadata is read: no augmented samples, images or gripper
validity checks. With replacement, boundary frames can occur more frequently
while the epoch size, batch size and losses remain unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


class TransitionBoundarySampler(Sampler):
    def __init__(self, dataset, config, *, indices=None, post_end_steps=0):
        config.validate()
        if not config.enabled:
            raise ValueError("Disabled oversampling must use the original sampler.")
        columns = ["skill_index", "skill_sequence_len", "skill_length_sequence", "skill_ds", "skill_de"]
        missing = set(columns) - set(dataset.hf_dataset.column_names)
        if missing:
            raise ValueError(f"Transition oversampling needs clean skill metadata: {sorted(missing)}.")
        values = dataset.hf_dataset.with_format("numpy", columns=columns, output_all_columns=False)[:]
        k = np.asarray(values["skill_index"], dtype=np.int64).reshape(-1)
        seq = np.asarray(values["skill_sequence_len"], dtype=np.int64).reshape(-1)
        ds = np.asarray(values["skill_ds"], dtype=np.int64).reshape(-1)
        de = np.asarray(values["skill_de"], dtype=np.int64).reshape(-1)
        lengths = np.asarray(values["skill_length_sequence"])
        if lengths.ndim != 2:
            lengths = np.stack([np.asarray(row, dtype=np.int64) for row in values["skill_length_sequence"]])
        lengths = lengths.astype(np.int64, copy=False)
        count = len(dataset)
        if (lengths.shape[0] != count or any(len(x) != count for x in [k, seq, ds, de])
                or np.any(k < 0) or np.any(k > seq - 2) or np.any(seq > lengths.shape[1])
                or np.any(ds < 0) or np.any(de < 0)):
            raise ValueError("Invalid original skill-boundary metadata.")

        def limits(selected, ranks):
            chosen = lengths[selected, ranks[selected]]
            if np.any(chosen <= 0):
                raise ValueError("Real adjacent skills must have positive lengths.")
            limit = np.zeros(count, dtype=np.int64)
            limit[selected] = np.maximum(1, np.floor(chosen * config.percent / 100.0 + 0.5).astype(np.int64))
            return limit

        # In skill k: next-skill length governs early switching at its end;
        # previous-skill length governs delayed switching near its start.
        early = limits(k < seq - 2, k + 1)
        late = limits(k > 0, k - 1)
        # The final real skill has no following skill: use its own length to
        # include the episode's last action segment as well.
        terminal = de < limits(k == seq - 2, k)
        boundary = (de < early) | (ds < late) | terminal
        self.indices = np.arange(count, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        if not len(self.indices) or np.any(self.indices < 0) or np.any(self.indices >= count):
            raise ValueError("No valid frame indices for transition oversampling.")
        selected = boundary[self.indices]
        self.weights = torch.as_tensor(np.where(selected, config.weight, 1.0), dtype=torch.double, device="cpu")
        self.boundary_frames = int(selected.sum())
        self.episode_end_frames = int(terminal[self.indices].sum())
        if post_end_steps:
            from lerobot.policies.skillVLA.newterm_supervision import NewTermFrameSampler

            if indices is not None:
                raise ValueError("Post-end Terminator sampling must retain every real frame.")
            # Keep the original tagged final-frame positive population and its
            # supervision semantics. Real boundary/end-window frames get the
            # extra weight; synthetic post-end positives retain unit weight.
            self.indices = NewTermFrameSampler(dataset, post_end_steps, shuffle=False).indices
            repeats = len(self.indices) - count
            self.weights = torch.cat((self.weights, torch.ones(repeats, dtype=torch.double)))
        self.sampler = WeightedRandomSampler(self.weights, len(self.indices), replacement=True)

    def __iter__(self):
        for index in self.sampler:
            item = self.indices[index]
            yield item if isinstance(item, tuple) else int(item)

    def __len__(self):
        return len(self.indices)
