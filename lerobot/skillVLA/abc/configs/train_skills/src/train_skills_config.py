#!/usr/bin/env python3
"""Resolve ABC skill-data settings while reusing the shared SBD implementation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "build_data/build_data_config.yaml"
ABC_NATIVE_FORMAT = "sbd_abc_sim_native/v1"
SBD_PROFILES = {
    "libero_standard": {
        "skillset_smooth_window": 7,
        "skillset_savgol_polyorder": 4,
        "skillset_score_power": 2.0,
        # The score recipe follows LIBERO, but terminal handling must remain
        # ABC-native: demonstrations end with a gripper release and a return
        # to the bimanual home pose.  A fixed tail guard would include part of
        # that reset in some episodes and remove task motion in others.
        "skillset_terminal_mask_frames": 0,
        "skillset_terminal_mask_mode": "abc_terminal_release",
        "skillset_terminal_release_pre_offset_frames": 8,
        "skillset_output_suffix": "",
    },
    "abc_standard": {
        "skillset_smooth_window": 8,
        "skillset_savgol_polyorder": 4,
        "skillset_score_power": 1.0,
        "skillset_terminal_mask_frames": 0,
        "skillset_terminal_mask_mode": "abc_terminal_release",
        "skillset_terminal_release_pre_offset_frames": 8,
        "skillset_output_suffix": "abc-standard",
    },
}


def _shared_config_module() -> ModuleType:
    path = next(
        parent
        / "libero/configs/train_skills/src/train_skills_config.py"
        for parent in Path(__file__).resolve().parents
        if (parent / "libero/configs/train_skills/src/train_skills_config.py").is_file()
    )
    spec = importlib.util.spec_from_file_location("sbd_libero_train_skills_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared train_skills config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _abc_selection(dataset_dir: Path, source_tasks: list[str]) -> tuple[list[int], int]:
    marker_path = dataset_dir / "native_dataset.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"ABC native dataset marker not found: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("format") != ABC_NATIVE_FORMAT:
        raise ValueError(f"Unsupported ABC native format in {marker_path}: {marker.get('format')!r}")
    episode_parts = sorted((dataset_dir / "meta/episodes").rglob("file-*.parquet"))
    if not episode_parts:
        raise FileNotFoundError(f"ABC episode metadata not found: {dataset_dir / 'meta/episodes'}")
    episodes = pd.concat([pd.read_parquet(path) for path in episode_parts], ignore_index=True)
    available = set(episodes["source_task"].astype(str))
    missing = sorted(set(source_tasks) - available)
    if missing:
        raise ValueError(f"Unknown abc_source_tasks {missing}; available={sorted(available)}")
    selected = episodes[episodes["source_task"].astype(str).isin(source_tasks)]
    task_indices = sorted(
        {int(task_index) for indices in selected["task_indices"] for task_index in indices}
    )
    if not task_indices or selected.empty:
        raise ValueError(f"ABC source-task selection is empty: {source_tasks}")
    return task_indices, len(selected)


def resolve_build_parallelism(
    cfg: dict[str, Any], *, task_count: int | None = None
) -> dict[str, int]:
    """Resolve logical curve shards, GPU allocations, and local workers."""
    tasks_per_job = int(cfg.get("skillset_tasks_per_job", 4))
    single_gpu_workers = int(cfg.get("skillset_single_gpu_workers", 0))
    workers_per_gpu = int(cfg.get("skillset_workers_per_gpu", 1))
    max_gpu_jobs = int(cfg.get("skillset_max_gpu_jobs", 0))
    if tasks_per_job < 1:
        raise ValueError(
            f"skillset_tasks_per_job must be >= 1, got {tasks_per_job}."
        )
    if single_gpu_workers < 0:
        raise ValueError(
            "skillset_single_gpu_workers must be >= 0 "
            f"(0=array mode, N>0=one shared GPU), got {single_gpu_workers}."
        )
    if workers_per_gpu < 1:
        raise ValueError(
            f"skillset_workers_per_gpu must be >= 1, got {workers_per_gpu}."
        )
    if max_gpu_jobs < 0:
        raise ValueError(
            f"skillset_max_gpu_jobs must be >= 0, got {max_gpu_jobs}."
        )
    if single_gpu_workers > 0 and (workers_per_gpu != 1 or max_gpu_jobs != 0):
        raise ValueError(
            "skillset_single_gpu_workers cannot be combined with "
            "skillset_workers_per_gpu != 1 or skillset_max_gpu_jobs > 0."
        )
    if max_gpu_jobs > 0:
        if task_count is None or task_count < 1:
            raise ValueError(
                "skillset_max_gpu_jobs requires a positive resolved task_count."
            )
        logical_workers = max_gpu_jobs * workers_per_gpu
        tasks_per_job = max(1, (task_count + logical_workers - 1) // logical_workers)
    return {
        "skillset_tasks_per_job": tasks_per_job,
        "skillset_single_gpu_workers": single_gpu_workers,
        "skillset_workers_per_gpu": workers_per_gpu,
        "skillset_max_gpu_jobs": max_gpu_jobs,
    }


def load_settings(config_path: Path, dataset: str | None = None) -> dict[str, Any]:
    shared = _shared_config_module()
    cfg = shared.load_config(config_path)
    profile = str(cfg.get("skillset_sbd_profile", "libero_standard")).strip().lower()
    if profile not in SBD_PROFILES:
        raise ValueError(
            f"skillset_sbd_profile must be one of {tuple(SBD_PROFILES)}, got {profile!r}."
        )
    # A profile owns the complete post-processing recipe so one YAML choice is
    # enough and stale individual knobs cannot silently create a hybrid build.
    cfg.update(SBD_PROFILES[profile])
    settings = shared.build_data_settings(cfg, dataset=dataset)
    source_tasks = _as_list(cfg.get("abc_source_tasks", ["grab_clutter"]))
    if not source_tasks:
        raise ValueError("abc_source_tasks must contain at least one official ABC Sim task.")
    task_indices, expected_episodes = _abc_selection(
        Path(settings["raw_dataset_dir"]), source_tasks
    )
    if settings["skillset_action_mode"] not in {
        "dataset",
        "anchor_relative",
        "delta_joint",
        "delta_pose",
    }:
        raise ValueError(
            "ABC skillset_action_mode must be dataset, anchor_relative, delta_joint, or delta_pose."
        )
    if (
        settings["skillset_action_mode"] != "dataset"
        and settings["skillset_relative_exclude_joints"] != "gripper"
    ):
        raise ValueError(
            "ABC transformed-action skillsets must keep both gripper dimensions absolute "
            "with skillset_relative_exclude_joints: [gripper]."
        )
    settings.update(
        {
            "abc_source_tasks": source_tasks,
            "abc_source_tasks_csv": ",".join(source_tasks),
            "abc_task_indices": task_indices,
            "abc_task_indices_json": json.dumps(task_indices),
            "abc_task_count": len(task_indices),
            "abc_expected_episodes": expected_episodes,
            "skillset_sbd_profile": profile,
            "skillset_terminal_mask_mode": str(
                cfg["skillset_terminal_mask_mode"]
            ),
            "skillset_terminal_release_pre_offset_frames": int(
                cfg["skillset_terminal_release_pre_offset_frames"]
            ),
            **resolve_build_parallelism(cfg, task_count=len(task_indices)),
        }
    )
    return settings


def load_fsq_settings(config_path: Path, dataset: str | None = None) -> dict[str, Any]:
    """Resolve the shared FSQ trainer with ABC's native/action contracts."""
    shared = _shared_config_module()
    cfg = shared.load_config(config_path)
    settings = shared.train_settings(cfg, dataset=dataset)
    manifest_path = Path(settings["skillset_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_tasks = [str(task) for task in manifest.get("source_tasks", [])]
    requested_tasks = _as_list(cfg.get("abc_source_tasks", manifest_tasks))
    if not requested_tasks:
        raise ValueError("ABC FSQ requires abc_source_tasks or manifest source_tasks.")
    if sorted(requested_tasks) != sorted(manifest_tasks):
        raise ValueError(
            "ABC FSQ task selection must exactly match the selected skillset: "
            f"config={requested_tasks}, manifest={manifest_tasks}."
        )
    dataset_dir = Path(settings["raw_dataset_dir"])
    _abc_selection(dataset_dir, requested_tasks)

    action_contract = manifest.get("action") or {}
    gripper_indices = [
        int(index) for index in action_contract.get("gripper_indices", [6, 13])
    ]
    if gripper_indices != [6, 13]:
        raise ValueError(
            "ABC FSQ expects the canonical bimanual gripper axes [6, 13], got "
            f"{gripper_indices}."
        )
    skill_files = sorted(
        (Path(settings["skillset_dir"]) / "skills").rglob("*.npz")
    )
    if not skill_files:
        raise FileNotFoundError(
            f"No FSQ skill files found under {settings['skillset_dir']}/skills"
        )
    with np.load(skill_files[0], allow_pickle=False) as sample:
        if (
            "actions" not in sample
            or sample["actions"].ndim != 2
            or sample["actions"].shape[1] != 14
        ):
            raise ValueError(
                f"ABC FSQ needs raw 14D absolute actions in each skill file: {skill_files[0]}"
            )

    settings.update(
        {
            "abc_source_tasks": requested_tasks,
            "abc_source_tasks_json": json.dumps(requested_tasks),
            "abc_source_tasks_label": ",".join(requested_tasks),
            "fsq_action_gripper_indices": gripper_indices,
            "fsq_action_gripper_indices_str": " ".join(
                str(index) for index in gripper_indices
            ),
            # SBD provenance may be delta-joint, but build_skillset stores the
            # untouched native action as the FSQ encoder/reconstructor target.
            "fsq_action_space": "absolute_joint",
            "fsq_action_dim": 14,
        }
    )
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--stage", choices=("build_data", "fsq"), default="build_data")
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    shared = _shared_config_module()
    settings = (
        load_fsq_settings(args.config, dataset=args.dataset)
        if args.stage == "fsq"
        else load_settings(args.config, dataset=args.dataset)
    )
    if args.shell:
        shared.print_shell(settings)
    else:
        print(json.dumps(settings, indent=2, default=str))


if __name__ == "__main__":
    main()
