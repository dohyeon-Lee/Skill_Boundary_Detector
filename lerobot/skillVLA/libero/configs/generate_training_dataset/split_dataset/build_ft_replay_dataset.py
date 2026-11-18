#!/usr/bin/env python3
# Inputs:
#   PT_dataset (rehearsal source) : --pt-dataset or ftrep_pt_dataset   (e.g. libero_90_full_full)
#   FT_dataset (finetune target)  : --ft-dataset or ftrep_ft_dataset   (e.g. libero_10_full_1)
#   PT episodes/task              : --pt-episodes-per-task (default 1)
# Reference model: none
# Output:
#   name : {FT_dataset}_FT (or --output-name)   path: {project_root}/{dataset_root}/{name}
"""Build a FINETUNE + PT-REPLAY dataset under libero_dataset/.

Merges the WHOLE FT dataset with a small REHEARSAL sample (1 episode per task) from the PT dataset, so
finetuning the new task keeps anchoring the old tasks. Tasks are NOT deduped by language (language is not a
unique task key in LIBERO — same language can map to different scenes/envs, and there is no env column to
tell them apart); every FT and PT task stays distinct, keyed by its source task_index. Stats (state/action
quantiles) are recomputed GLOBALLY over the merged frames; video stats are copied from the FT source.

  python build_ft_replay_dataset.py --pt-dataset libero_90_full_full --ft-dataset libero_10_full_1
  → libero_10_full_1_FT  =  all of libero_10_full_1  +  1 episode/task from libero_90_full_full
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from build_training_dataset import (  # noqa: E402  (reuse the slice tool's dataset I/O + stats)
    compute_stats,
    load_dataset,
    refresh_episode_stats,
    select_episodes,
)
from training_dataset_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    as_bool,
    dataset_root_path,
    get_value,
    load_config,
)


def _prep_data(data: pd.DataFrame, sel_eps: list[int], ep_map: dict[int, int],
               task_map: dict[int, int], cols: list[str]) -> pd.DataFrame:
    """Subset to selected episodes, restrict to common columns, remap episode + task indices."""
    d = data[data["episode_index"].isin(sel_eps)][cols].copy()
    d["episode_index"] = d["episode_index"].map(ep_map)
    d["task_index"] = d["task_index"].map(task_map)
    return d


def _prep_tasks(tasks: pd.DataFrame, ids: list[int], task_map: dict[int, int]) -> pd.DataFrame:
    t = tasks[tasks["task_index"].isin(ids)].copy()
    t["task_index"] = t["task_index"].map(task_map)
    return t


def _prep_eps(eps: pd.DataFrame, sel: list[int], ep_map: dict[int, int], src_name: str,
              data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Episode-meta rows for the selected episodes: keep common cols, stamp provenance, remap index."""
    e = eps[eps["episode_index"].isin(sel)][cols].copy()
    e["source_dataset"] = src_name
    e["source_episode_index"] = e["episode_index"]
    e2t = data[["episode_index", "task_index"]].drop_duplicates().set_index("episode_index")["task_index"]
    e["source_task_index"] = e["source_episode_index"].map(e2t)
    e["episode_index"] = e["episode_index"].map(ep_map)
    return e


def build_ft_replay(pt_src: Path, ft_src: Path, dst: Path, pt_eps_per_task: int,
                    overwrite: bool) -> None:
    pt_info, pt_tasks, pt_data, pt_eps = load_dataset(pt_src)
    ft_info, ft_tasks, ft_data, ft_eps = load_dataset(ft_src)

    # data columns must line up (same LIBERO format); intersect + warn on any mismatch.
    common_data = [c for c in ft_data.columns if c in pt_data.columns]
    mismatch = set(ft_data.columns) ^ set(pt_data.columns)
    if mismatch:
        print(f"[warn] data column mismatch — using {len(common_data)} common cols, dropping: {sorted(mismatch)[:8]}")
    video_keys = [k for k, v in ft_info["features"].items() if v.get("dtype") == "video"]

    # ── selections ──
    ft_task_ids = sorted(int(t) for t in ft_data["task_index"].drop_duplicates())
    pt_task_ids = sorted(int(t) for t in pt_data["task_index"].drop_duplicates())
    ft_sel = select_episodes(ft_data, ft_task_ids, None, allow_fewer=True)            # ALL FT episodes
    pt_sel = select_episodes(pt_data, pt_task_ids, pt_eps_per_task, allow_fewer=True)  # N/task from PT

    # ── unified maps (NO dedup): FT first, PT appended ──
    ft_task_map = {old: new for new, old in enumerate(ft_task_ids)}                   # 0 .. N_ft-1
    pt_task_map = {old: len(ft_task_ids) + i for i, old in enumerate(pt_task_ids)}    # N_ft ..
    ft_ep_map = {old: new for new, old in enumerate(ft_sel)}                          # 0 ..
    pt_ep_map = {old: len(ft_sel) + i for i, old in enumerate(pt_sel)}               # after FT

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {dst}. Pass --overwrite to replace it.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    (dst / "meta").mkdir(exist_ok=True)

    # ── data ──
    merged = pd.concat(
        [_prep_data(ft_data, ft_sel, ft_ep_map, ft_task_map, common_data),
         _prep_data(pt_data, pt_sel, pt_ep_map, pt_task_map, common_data)],
        ignore_index=True,
    )
    merged = merged.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    merged["index"] = np.arange(len(merged), dtype=np.int64)
    (dst / "data/chunk-000").mkdir(parents=True)
    merged.to_parquet(dst / "data/chunk-000/file-000.parquet", index=False)

    # ── tasks (unified; languages may repeat across FT/PT — that is fine, task_index is the key) ──
    unified_tasks = pd.concat(
        [_prep_tasks(ft_tasks, ft_task_ids, ft_task_map),
         _prep_tasks(pt_tasks, pt_task_ids, pt_task_map)]
    ).sort_values("task_index")
    unified_tasks.to_parquet(dst / "meta/tasks.parquet")

    # ── videos (copy from each source, remap file index) ──
    for vk in video_keys:
        (dst / "videos" / vk / "chunk-000").mkdir(parents=True)
        for src_root, eps_df, ep_map in ((ft_src, ft_eps, ft_ep_map), (pt_src, pt_eps, pt_ep_map)):
            for old_ep, new_ep in ep_map.items():
                row = eps_df[eps_df["episode_index"] == old_ep].iloc[0]
                ch = int(row[f"videos/{vk}/chunk_index"]); fi = int(row[f"videos/{vk}/file_index"])
                src_v = src_root / "videos" / vk / f"chunk-{ch:03d}" / f"file-{fi:03d}.mp4"
                shutil.copy2(src_v, dst / "videos" / vk / "chunk-000" / f"file-{new_ep:03d}.mp4")

    # ── episodes meta ──
    common_ep = [c for c in ft_eps.columns if c in pt_eps.columns]
    merged_eps = pd.concat(
        [_prep_eps(ft_eps, ft_sel, ft_ep_map, ft_src.name, ft_data, common_ep),
         _prep_eps(pt_eps, pt_sel, pt_ep_map, pt_src.name, pt_data, common_ep)]
    ).sort_values("episode_index").reset_index(drop=True)

    ranges = (merged.groupby("episode_index")["index"]
              .agg(dataset_from_index="min", dataset_to_index="max").reset_index())
    ranges["dataset_to_index"] += 1
    merged_eps = merged_eps.drop(columns=["dataset_from_index", "dataset_to_index"], errors="ignore")
    merged_eps = merged_eps.merge(ranges, on="episode_index", how="left")
    for vk in video_keys:
        merged_eps[f"videos/{vk}/chunk_index"] = 0
        merged_eps[f"videos/{vk}/file_index"] = merged_eps["episode_index"]
    merged_eps["data/chunk_index"] = 0; merged_eps["data/file_index"] = 0
    merged_eps["meta/episodes/chunk_index"] = 0; merged_eps["meta/episodes/file_index"] = 0
    merged_eps = refresh_episode_stats(merged_eps, merged)
    (dst / "meta/episodes/chunk-000").mkdir(parents=True)
    merged_eps.to_parquet(dst / "meta/episodes/chunk-000/file-000.parquet", index=False)

    # ── stats.json: GLOBAL recompute over merged frames (video stats copied from the FT source) ──
    with open(ft_src / "meta/stats.json") as f:
        ft_stats = json.load(f)
    with open(dst / "meta/stats.json", "w") as f:
        json.dump(compute_stats(merged, ft_stats, video_keys), f, indent=2)

    # ── info.json ──
    new_info = dict(ft_info)
    new_info["total_episodes"] = len(ft_sel) + len(pt_sel)
    new_info["total_frames"] = int(len(merged))
    new_info["total_tasks"] = int(len(unified_tasks))
    new_info["splits"] = {"train": f"0:{len(ft_sel) + len(pt_sel)}"}
    new_info["chunks_size"] = 1000
    new_info["source_ft_dataset"] = ft_src.name
    new_info["source_pt_dataset"] = pt_src.name
    new_info["ft_episodes"] = len(ft_sel)
    new_info["ft_tasks"] = len(ft_task_ids)
    new_info["pt_replay_episodes"] = len(pt_sel)
    new_info["pt_replay_episodes_per_task"] = pt_eps_per_task
    new_info["pt_tasks"] = len(pt_task_ids)
    new_info.pop("data_files_size_in_mb", None)
    new_info.pop("video_files_size_in_mb", None)
    with open(dst / "meta/info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    print(f"Built FT+replay dataset: {dst}")
    print(f"  tasks    : {new_info['total_tasks']}  (FT {len(ft_task_ids)} + PT {len(pt_task_ids)})")
    print(f"  episodes : {new_info['total_episodes']}  (FT {len(ft_sel)} + PT-replay {len(pt_sel)})")
    print(f"  frames   : {new_info['total_frames']}")


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)
    default_root = dataset_root_path(cfg)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter, parents=[pre])
    p.add_argument("--pt-dataset", default=str(get_value(cfg, "ftrep_pt_dataset", "libero_90_full_full")),
                   help="PT (rehearsal) dataset under libero_dataset/")
    p.add_argument("--ft-dataset", default=str(get_value(cfg, "ftrep_ft_dataset", "")),
                   help="FT (finetune target) dataset under libero_dataset/")
    p.add_argument("--pt-episodes-per-task", type=int,
                   default=int(get_value(cfg, "ftrep_pt_episodes_per_task", 1)),
                   help="PT episodes to replay per task (default 1)")
    p.add_argument("--output-name", default=str(get_value(cfg, "ftrep_output_name", "")),
                   help="Override auto name: {FT_dataset}_FT")
    p.add_argument("--src-root", type=Path, default=default_root)
    p.add_argument("--dst-root", type=Path, default=default_root)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction,
                   default=as_bool(get_value(cfg, "ftrep_overwrite", False)))
    args = p.parse_args()

    if not args.ft_dataset:
        p.error("--ft-dataset is required (or set ftrep_ft_dataset in the config yaml)")
    if args.pt_episodes_per_task <= 0:
        p.error("--pt-episodes-per-task must be a positive integer")

    pt_src = args.src_root / args.pt_dataset
    ft_src = args.src_root / args.ft_dataset
    output_name = args.output_name or f"{args.ft_dataset}_FT"
    dst = args.dst_root / output_name

    print("FT + PT-replay dataset generation")
    print(f"  FT (full)      : {ft_src}")
    print(f"  PT (replay {args.pt_episodes_per_task}/task): {pt_src}")
    print(f"  output         : {dst}")

    build_ft_replay(pt_src, ft_src, dst, args.pt_episodes_per_task, args.overwrite)


if __name__ == "__main__":
    main()
