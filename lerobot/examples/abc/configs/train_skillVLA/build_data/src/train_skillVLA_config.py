#!/usr/bin/env python3
"""Resolve the ABC-native FSQ -> SkillVLA dataset build."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "train_skillVLA_config.yaml"


def _abc_train_skills_config() -> ModuleType:
    path = next(
        parent / "abc/configs/train_skills/src/train_skills_config.py"
        for parent in Path(__file__).resolve().parents
        if (parent / "abc/configs/train_skills/src/train_skills_config.py").is_file()
    )
    spec = importlib.util.spec_from_file_location("sbd_abc_train_skills_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import ABC config helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off", ""}:
        return False
    raise ValueError(f"Expected a boolean, got {value!r}.")


def _path_from_meta(
    value: str,
    *,
    fallback: Path,
    description: str,
) -> Path:
    recorded = Path(str(value)).expanduser() if str(value).strip() else fallback
    if recorded.is_file():
        return recorded.resolve()
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(
        f"{description} not found. Recorded={recorded}; local fallback={fallback}"
    )


def _safe_suffix(value: Any) -> str:
    text = str(value or "").strip().strip("_")
    if text and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", text) is None:
        raise ValueError(f"Invalid skillvla_output_suffix: {value!r}")
    return f"_{text}" if text else ""


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value if str(item).strip()]


def build_settings(cfg: dict) -> dict[str, Any]:
    root = Path(str(cfg["project_root"])).expanduser().resolve()
    dataset_root = root / str(cfg.get("dataset_root", "dataset_ABC"))
    outputs_root = root / str(cfg.get("outputs_root", "outputs_ABC"))
    source_dataset = str(cfg.get("source_dataset", "abc_sim_all"))
    raw_dataset_dir = dataset_root / source_dataset
    if not (raw_dataset_dir / "native_dataset.json").is_file():
        raise FileNotFoundError(f"ABC native source dataset not found: {raw_dataset_dir}")

    data_mode = str(cfg.get("skillvla_data_mode", "pt")).strip().lower()
    if data_mode != "pt":
        raise ValueError(
            "The first ABC port supports PT data only; skillvla_data_mode must be pt."
        )
    fsq_run_name = str(cfg["fsq_run_name"]).strip()
    fsq_checkpoint = int(cfg.get("fsq_checkpoint", 2000))
    if fsq_checkpoint <= 0:
        raise ValueError("fsq_checkpoint must be a positive epoch.")
    fsq_dir = outputs_root / "FSQ" / fsq_run_name
    fsq_meta_path = fsq_dir / "fsq_meta.json"
    if not fsq_meta_path.is_file():
        raise FileNotFoundError(f"FSQ metadata not found: {fsq_meta_path}")
    fsq_meta = json.loads(fsq_meta_path.read_text(encoding="utf-8"))
    target_dataset = str(fsq_meta.get("target_dataset", ""))
    if target_dataset != source_dataset:
        raise ValueError(
            f"FSQ target_dataset={target_dataset!r} does not match {source_dataset!r}."
        )
    source_tasks = [str(value) for value in fsq_meta.get("source_tasks", [])]
    if not source_tasks:
        raise ValueError(f"FSQ metadata has no source_tasks: {fsq_meta_path}")
    levels = [int(value) for value in fsq_meta.get("fsq_levels", [])]
    if not levels or any(value < 2 for value in levels):
        raise ValueError(f"Invalid fsq_levels in {fsq_meta_path}: {levels}")
    if int(fsq_meta.get("action_dim", -1)) != 14:
        raise ValueError("ABC SkillVLA build requires the native 14D action FSQ.")
    if str(fsq_meta.get("action_space", "")) != "absolute_joint":
        raise ValueError("ABC SkillVLA build currently preserves absolute_joint actions.")

    seg_name = str(fsq_meta.get("skillset_seg_name", "")).strip()
    manifest_fallback = (
        dataset_root
        / "FSQ_dataset"
        / source_dataset
        / "FSQ_inputs"
        / seg_name
        / "skillset"
        / "skillset_manifest.json"
    )
    manifest_path = _path_from_meta(
        str(fsq_meta.get("skillset_manifest_path", "")),
        fallback=manifest_fallback,
        description="FSQ training skillset manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sorted(str(value) for value in manifest.get("source_tasks", [])) != sorted(source_tasks):
        raise ValueError("FSQ metadata and skillset manifest select different ABC tasks.")
    skills_dir = manifest_path.parent / "skills"
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"Skill files not found: {skills_dir}")

    recorded_stats = str(fsq_meta.get("stats_path", "")).strip()
    if recorded_stats:
        # Absolute paths in fsq_meta.json belong to the machine that trained the
        # FSQ. Preserve the exact filename when rebuilding on another server.
        stats_fallback = (
            raw_dataset_dir
            / "meta"
            / "selection_stats"
            / Path(recorded_stats).name
        )
    else:
        candidates = []
        if len(source_tasks) == 1:
            candidates = sorted(
                (raw_dataset_dir / "meta" / "selection_stats").glob(
                    f"task-{source_tasks[0]}-*.json"
                )
            )
        stats_fallback = (
            candidates[0] if len(candidates) == 1 else raw_dataset_dir / "meta" / "stats.json"
        )
    stats_path = _path_from_meta(
        recorded_stats,
        fallback=stats_fallback,
        description="ABC task-selection statistics",
    )
    fsq_model_path = fsq_dir / f"FSQ_epoch{fsq_checkpoint:04d}.pt"

    base_run_name = str(
        cfg.get("skillvla_run_name", f"FSQ{''.join(map(str, levels))}_{fsq_checkpoint}_pt")
    ).strip()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", base_run_name) is None:
        raise ValueError(f"Invalid skillvla_run_name: {base_run_name!r}")
    run_name = base_run_name + _safe_suffix(cfg.get("skillvla_output_suffix", ""))
    run_dir = (
        dataset_root
        / str(cfg.get("skillvla_dataset_root", "skillvla"))
        / source_dataset
        / run_name
    )

    jitter_pmax = int(cfg.get("transition_jitter_pmax", 30))
    if jitter_pmax < 0:
        raise ValueError("transition_jitter_pmax must be non-negative.")
    jitter_distribution = str(
        cfg.get("transition_jitter_distribution", "half_normal")
    ).strip().lower().replace("-", "_")
    if jitter_distribution not in {"half_normal", "uniform"}:
        raise ValueError("transition_jitter_distribution must be half_normal|uniform.")
    jitter_max_percent = float(cfg.get("transition_jitter_max_percent", 0.0))
    if jitter_max_percent != 0.0:
        raise ValueError(
            "ABC SkillVLA uses fixed-step transition randomization; "
            "transition_jitter_max_percent must remain 0.0."
        )
    if str(cfg.get("proprio_grounding", "none")).strip().lower() != "none":
        raise ValueError("ABC joint proprio must use proprio_grounding: none.")
    spatial_targets_enabled = _as_bool(cfg.get("spatial_targets_enabled", True))
    spatial_workers = int(
        cfg.get("spatial_workers", cfg.get("skillvla_build_cpus_per_task", 16))
    )
    if spatial_workers < 1:
        raise ValueError("spatial_workers must be positive.")

    partitions = cfg.get("train_partition", ["debug"])
    if isinstance(partitions, str):
        partitions = [partitions]
    train_partition = ",".join(str(value) for value in partitions)
    excludes = _as_string_list(cfg.get("train_exclude_nodes", []))
    cpu_excludes = _as_string_list(cfg.get("cpu_exclude_nodes", []))
    max_order = int(cfg.get("max_order", 0))
    max_length = int(cfg.get("max_length", 0))
    if max_order < 0 or max_length < 0:
        raise ValueError("max_order and max_length must be non-negative.")

    return {
        "project_root": root,
        "lerobot_root": root / "lerobot",
        "source_dataset": source_dataset,
        "raw_dataset_dir": raw_dataset_dir,
        "abc_source_tasks": source_tasks,
        "abc_source_tasks_str": " ".join(source_tasks),
        "abc_source_tasks_json": json.dumps(source_tasks),
        "skillset_manifest_path": manifest_path,
        "skillset_dir": manifest_path.parent,
        "fsq_run_name": fsq_run_name,
        "fsq_checkpoint": fsq_checkpoint,
        "fsq_model_path": fsq_model_path,
        "fsq_meta_path": fsq_meta_path,
        "fsq_stats_path": stats_path,
        "fsq_levels_str": " ".join(str(value) for value in levels),
        "skillvla_run_name": run_name,
        "skillvla_run_dir": run_dir,
        "skillvla_dataset_dir": run_dir / "skillvla",
        "skill_latents_path": run_dir / "skill_latents.npz",
        "iss_npz_path": run_dir / "skill_initial_state.npz",
        "fsq_copy_path": run_dir / "FSQ.pt",
        "skill_annotations_path": run_dir / "skillvla" / "skill_annotations.npz",
        "spatial_targets_enabled": str(spatial_targets_enabled).lower(),
        "spatial_targets_path": run_dir / "skill_spatial_targets.npz",
        "spatial_shard_dir": run_dir / ".spatial_target_shards",
        "spatial_assets_dir": root / "third_party" / "abc" / "abc_sim" / "models" / "assets",
        "spatial_workers": spatial_workers,
        "spatial_keep_shards": str(_as_bool(cfg.get("spatial_keep_shards", False))).lower(),
        "transition_jitter_pmax": jitter_pmax,
        "transition_jitter_distribution": jitter_distribution,
        "transition_jitter_max_percent": jitter_max_percent,
        "transition_jitter_precompute_gripper_validity": str(
            _as_bool(cfg.get("transition_jitter_precompute_gripper_validity", False))
        ).lower(),
        "max_order": max_order,
        "max_length": max_length,
        "cleanup_intermediate": str(_as_bool(cfg.get("cleanup_intermediate", False))).lower(),
        "skillvla_encode_partition": train_partition,
        "skillvla_encode_qos": str(cfg.get("train_qos", "base_qos")),
        "skillvla_encode_gres": str(cfg.get("skillvla_encode_gres", "gpu:1")),
        "skillvla_encode_cpus_per_task": int(cfg.get("skillvla_encode_cpus_per_task", 8)),
        "skillvla_encode_mem": str(cfg.get("skillvla_encode_mem", "64G")),
        "skillvla_encode_time": str(cfg.get("skillvla_encode_time", "4:00:00")),
        "skillvla_encode_nodelist": str(cfg.get("train_nodelist", "")),
        "skillvla_encode_exclude_nodes": ",".join(str(value) for value in excludes if str(value)),
        "skillvla_build_partition": str(cfg.get("cpu_partition", train_partition)),
        "skillvla_build_qos": str(cfg.get("cpu_qos", cfg.get("train_qos", "base_qos"))),
        "skillvla_build_account": str(cfg.get("cpu_account", "")),
        "skillvla_build_cpus_per_task": int(cfg.get("skillvla_build_cpus_per_task", 16)),
        "skillvla_build_mem": str(cfg.get("skillvla_build_mem", "64G")),
        "skillvla_build_time": str(cfg.get("skillvla_build_time", "8:00:00")),
        "skillvla_build_exclude_nodes": ",".join(
            value for value in cpu_excludes if value
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    abc = _abc_train_skills_config()
    shared = abc._shared_config_module()
    settings = build_settings(shared.load_config(args.config))
    if args.shell:
        shared.print_shell(settings)
    else:
        print(json.dumps(settings, indent=2, default=str))


if __name__ == "__main__":
    main()
