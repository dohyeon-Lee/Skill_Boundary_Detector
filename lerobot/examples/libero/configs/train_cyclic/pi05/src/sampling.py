"""Task-atomic, frame-balanced block cycling, adapted from skillVLA_real/train_pi05_cycle.

No Reptile, extra loss, probe forwards or model/processor changes. Only data order differs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def build_groups(dataset, n_groups: int, seed: int) -> list[dict]:
    if getattr(dataset, "episodes", None) is not None:
        raise ValueError("Block-cyclic PT requires the full offline dataset, not an episode subset.")
    eps = dataset.meta.episodes
    starts = [int(v) for v in eps["dataset_from_index"]]
    ends = [int(v) for v in eps["dataset_to_index"]]
    tasks = []
    for value in eps["tasks"]:
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != 1:
                raise ValueError("Each episode must have exactly one task for task-atomic grouping.")
            value = value[0]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Missing episode task description.")
        tasks.append(value)
    if not starts or len(starts) != len(ends) or len(starts) != len(tasks):
        raise ValueError("Incomplete episode metadata.")
    ranges = sorted(zip(starts, ends, strict=True))
    cursor = 0
    for start, end in ranges:
        if start != cursor or end <= start:
            raise ValueError("Episode ranges must cover the full dataset exactly without gaps or overlaps.")
        cursor = end
    if cursor != len(dataset):
        raise ValueError("Episode frame ranges do not match dataset length.")
    unique = sorted(set(tasks))
    if isinstance(n_groups, bool) or not isinstance(n_groups, int) or not 1 <= n_groups <= len(unique):
        raise ValueError(f"n_groups must be between 1 and the number of tasks ({len(unique)}).")
    task_frames = dict.fromkeys(unique, 0)
    for task, start, end in zip(tasks, starts, ends, strict=True):
        task_frames[task] += end - start
    order = sorted(np.random.default_rng(seed).permutation(unique).tolist(), key=lambda t: -task_frames[t])
    loads = [0] * n_groups
    group_of_task = {}
    for task in order:
        gid = int(np.argmin(loads))
        group_of_task[task] = gid
        loads[gid] += task_frames[task]
    groups = []
    for gid in range(n_groups):
        ep_ids = [i for i, task in enumerate(tasks) if group_of_task[task] == gid]
        groups.append({
            "group_id": gid,
            "tasks": sorted(task for task, g in group_of_task.items() if g == gid),
            "episode_indices": ep_ids,
            "frame_ranges": [[starts[i], ends[i]] for i in ep_ids],
            "frame_indices": np.concatenate([np.arange(starts[i], ends[i]) for i in ep_ids]),
        })
    return groups


class GroupCursor:
    """Continue an unfinished group epoch when revisiting; reshuffle only on exhaustion."""

    def __init__(self, frames, seed: int):
        self.frames = np.asarray(frames, dtype=np.int64)
        if not len(self.frames):
            raise ValueError("Cannot cycle an empty group.")
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.frames)
        self.pos = 0

    def take(self, count: int) -> list[int]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer.")
        out = []
        while count:
            chunk = self.perm[self.pos:self.pos + count]
            out.extend(chunk.tolist())
            self.pos += len(chunk)
            count -= len(chunk)
            if self.pos == len(self.perm):
                self.perm = self.rng.permutation(self.frames)
                self.pos = 0
        return out


class BlockCyclicBatchSampler:
    """Fixed-size batches from ONE group for phase_steps, reshuffling group order each cycle.

    A full iterator describes the remaining training run (not one dataset epoch). Replaying
    just the integer index schedule to start_step restores order exactly on resume, independent
    of DataLoader/Accelerate prefetch. No frame decoding or model forwards are replayed.
    """

    drop_last = False

    def __init__(self, groups, batch_size: int, phase_steps: int, steps: int, seed: int, start_step=0):
        for name, value in (("batch_size", batch_size), ("phase_steps", phase_steps), ("steps", steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not groups or isinstance(start_step, bool) or not isinstance(start_step, int) or not 0 <= start_step <= steps:
            raise ValueError("Invalid groups or start_step.")
        self.groups = groups
        self.batch_size, self.phase_steps, self.steps = batch_size, phase_steps, steps
        self.seed, self.start_step = seed, start_step

    def __len__(self):
        return self.steps - self.start_step

    def __iter__(self):
        cursors = [GroupCursor(g["frame_indices"], self.seed + 1000 + i) for i, g in enumerate(self.groups)]
        step, cycle = 0, 0
        while step < self.steps:
            order = np.random.default_rng(self.seed + 1 + cycle).permutation(len(cursors))
            for gid in order:
                for _ in range(min(self.phase_steps, self.steps - step)):
                    batch = cursors[int(gid)].take(self.batch_size)
                    if step >= self.start_step:
                        yield batch
                    step += 1
                if step >= self.steps:
                    break
            cycle += 1


def save_manifest(output: Path, groups, cfg, settings, start_step: int):
    """Keep human-readable grouping and reject a changed curriculum on resume."""
    compact = [{k: v for k, v in g.items() if k != "frame_indices"} for g in groups]
    payload = {
        "mode": "cyclic", "n_groups": len(groups),
        "phase_steps": settings["cyclic_phase_steps"], "group_seed": settings["cyclic_group_seed"],
        "batch_size": cfg.batch_size, "steps": cfg.steps, "training_seed": cfg.seed,
        "dataset": str(cfg.dataset.repo_id),
        "groups_sha256": hashlib.sha256(json.dumps(compact, sort_keys=True).encode()).hexdigest(),
    }
    path = output / "curriculum.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("Saved cyclic curriculum differs; do not change the schedule/groups on resume.")
    elif start_step:
        raise ValueError("Cyclic resume needs curriculum.json from the original run.")
    group_path = output / "groups.json"
    if group_path.exists() and json.loads(group_path.read_text()) != compact:
        raise ValueError("Saved groups.json differs from the current dataset grouping.")
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    group_path.write_text(json.dumps(compact, indent=2))


def make_loader_factory(settings):
    def factory(cfg, dataset, loader_kwargs, sampler, start_step):
        import torch

        if cfg.dataset.streaming or sampler is not None:
            raise ValueError("Block-cyclic pi05 requires a full, offline frame dataset without another sampler.")
        groups = build_groups(dataset, settings["cyclic_n_groups"], settings["cyclic_group_seed"])
        batches = BlockCyclicBatchSampler(
            groups, cfg.batch_size, settings["cyclic_phase_steps"], cfg.steps,
            settings["cyclic_group_seed"], start_step,
        )
        save_manifest(Path(cfg.output_dir), groups, cfg, settings, start_step)
        return torch.utils.data.DataLoader(**loader_kwargs, batch_sampler=batches)

    return factory
