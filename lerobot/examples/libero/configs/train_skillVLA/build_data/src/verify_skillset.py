#!/usr/bin/env python3
"""Verify skillset completeness; gate the stage and drive retry sweeps.

An (episode, task) pair is DONE if it has a `.done` marker (written by
build_skill_dataset.py --write_done_markers, which covers filtered / 0-skill
episodes) or at least one skill npz. This distinguishes "GPU died before the
episode was reached" (missing → must retry) from "legitimately produced 0 skills"
(done), so retries terminate instead of looping on filtered episodes.

Self-contained (pandas + numpy only) so it runs instantly on the login node for
gating, without importing the heavy skill_divider/torch stack.

Modes:
  (default)               human-readable report, always exit 0
  --check                 exit 0 if every expected episode is done, else exit 1
  --print-missing-tasks   print space-separated task_ids that still have missing
                          episodes (empty if complete); always exit 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_episodes_meta(dataset_dir: Path) -> pd.DataFrame:
    # identical to skill_divider.load_episodes_meta (kept inline to avoid heavy imports)
    files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def task_to_episodes(dataset_dir: Path, task_ids: list[int] | None) -> dict[int, list[int]]:
    """Episode→task mapping, matching build_skill_dataset.py exactly."""
    episodes_meta = load_episodes_meta(dataset_dir)
    tasks_meta = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet").reset_index()
    if task_ids is None:
        task_ids = sorted(int(t) for t in tasks_meta["task_index"].tolist())

    mapping: dict[int, list[int]] = {}
    for task_id in task_ids:
        row = tasks_meta[tasks_meta["task_index"] == task_id]
        if row.empty:
            continue
        target_lang = row.iloc[0]["task"]
        ep_of_task = episodes_meta[episodes_meta["tasks"].apply(
            lambda t: target_lang in (
                [str(x) for x in t] if isinstance(t, (list, np.ndarray)) else [str(t)]
            )
        )]
        mapping[int(task_id)] = [int(e) for e in ep_of_task["episode_index"].tolist()]
    return mapping


def is_done(skills_dir: Path, ep_id: int, task_id: int) -> bool:
    task_dir = skills_dir / f"task{task_id:02d}"
    if (task_dir / f"ep{ep_id:05d}_task{task_id:02d}.done").exists():
        return True
    return any(task_dir.glob(f"ep{ep_id:05d}_task{task_id:02d}_skill*.npz"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_dir", required=True, type=Path)
    ap.add_argument("--skillset_dir", required=True, type=Path,
                    help="directory that contains skills/ (i.e. SKILLSET_DIR)")
    ap.add_argument("--task_ids", type=int, nargs="*", default=None,
                    help="restrict to these task ids (default: all tasks in meta)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true",
                   help="exit 0 if complete else 1 (no output)")
    g.add_argument("--print-missing-tasks", action="store_true",
                   help="print task ids that still have missing episodes")
    args = ap.parse_args()

    skills_dir = args.skillset_dir / "skills"
    mapping = task_to_episodes(args.dataset_dir, args.task_ids)

    per_task: list[tuple[int, int, int, int]] = []
    missing_tasks: list[int] = []
    n_exp = n_done = 0
    for task_id, eps in sorted(mapping.items()):
        miss = [e for e in eps if not is_done(skills_dir, e, task_id)]
        n_exp += len(eps)
        n_done += len(eps) - len(miss)
        if miss:
            missing_tasks.append(task_id)
        per_task.append((task_id, len(eps), len(eps) - len(miss), len(miss)))

    if args.print_missing_tasks:
        print(" ".join(str(t) for t in missing_tasks))
        return 0
    if args.check:
        return 0 if not missing_tasks else 1

    # default: human-readable report
    status = "COMPLETE" if not missing_tasks else f"INCOMPLETE ({len(missing_tasks)} task(s))"
    print(f"Skillset {status}: {n_done}/{n_exp} episodes done")
    print(f"  dir: {skills_dir}")
    for task_id, tot, done, miss in per_task:
        flag = "" if miss == 0 else f"   ← MISSING {miss}"
        print(f"  task{task_id:02d}: {done:3d}/{tot:<3d}{flag}")
    if missing_tasks:
        print("Incomplete tasks:", " ".join(str(t) for t in missing_tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
