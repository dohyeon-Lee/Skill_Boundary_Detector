#!/usr/bin/env python3
"""Resolve ABC skill-data settings while reusing the shared SBD implementation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "build_data/build_data_config.yaml"
ABC_NATIVE_FORMAT = "sbd_abc_sim_native/v1"


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


def resolve_build_parallelism(cfg: dict[str, Any]) -> dict[str, int]:
    """Resolve ABC curve-build sharding and optional shared-GPU workers."""
    tasks_per_job = int(cfg.get("skillset_tasks_per_job", 4))
    single_gpu_workers = int(cfg.get("skillset_single_gpu_workers", 0))
    if tasks_per_job < 1:
        raise ValueError(
            f"skillset_tasks_per_job must be >= 1, got {tasks_per_job}."
        )
    if single_gpu_workers < 0:
        raise ValueError(
            "skillset_single_gpu_workers must be >= 0 "
            f"(0=array mode, N>0=one shared GPU), got {single_gpu_workers}."
        )
    return {
        "skillset_tasks_per_job": tasks_per_job,
        "skillset_single_gpu_workers": single_gpu_workers,
    }


def load_settings(config_path: Path, dataset: str | None = None) -> dict[str, Any]:
    shared = _shared_config_module()
    cfg = shared.load_config(config_path)
    settings = shared.build_data_settings(cfg, dataset=dataset)
    source_tasks = _as_list(cfg.get("abc_source_tasks", ["grab_clutter"]))
    if not source_tasks:
        raise ValueError("abc_source_tasks must contain at least one official ABC Sim task.")
    task_indices, expected_episodes = _abc_selection(
        Path(settings["raw_dataset_dir"]), source_tasks
    )
    if settings["skillset_action_mode"] not in {"dataset", "anchor_relative", "delta_joint"}:
        raise ValueError(
            "ABC skillset_action_mode must be dataset, anchor_relative, or delta_joint."
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
            **resolve_build_parallelism(cfg),
        }
    )
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--stage", choices=("build_data",), default="build_data")
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    shared = _shared_config_module()
    settings = load_settings(args.config, dataset=args.dataset)
    if args.shell:
        shared.print_shell(settings)
    else:
        print(json.dumps(settings, indent=2, default=str))


if __name__ == "__main__":
    main()
