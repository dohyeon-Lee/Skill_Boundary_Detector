#!/usr/bin/env python3
"""Resolve paths and settings for the dual-camera attention-target preview."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2] / "src"))

from train_skills_config import as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent / "config.yaml"


def _mapping(config: dict, section: str) -> dict:
    value = config.get(section, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a YAML mapping.")
    return value


def _subsection(config: dict, section: str, subsection: str) -> dict:
    parent = _mapping(config, section)
    value = parent.get(subsection, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{section}.{subsection} must be a YAML mapping.")
    return value


def _resolve(project_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _safe_name(value: object) -> str:
    name = str(value or "").strip()
    if not name or re.fullmatch(r"[A-Za-z0-9._-]+", name) is None:
        raise ValueError(
            "output_name must contain only letters, digits, '.', '_', and '-'."
        )
    return name


def _int_list_or_all(value: object, *, field: str) -> str:
    if isinstance(value, str) and value.strip().lower() == "all":
        return "all"
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be 'all' or a non-empty integer list.")
    result = [int(item) for item in value]
    if any(item < 0 for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique non-negative integers.")
    return json.dumps(result, separators=(",", ":"))


def _optional_int_list(value: object, *, field: str) -> str:
    if value in (None, ""):
        value = []
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an integer list.")
    result = [int(item) for item in value]
    if any(item < 0 for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique non-negative integers.")
    return json.dumps(result, separators=(",", ":"))


def _noise_levels(value: object) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError("input_xyz_noise.std_m must be a non-empty list.")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) or item < 0.0 for item in result):
        raise ValueError("input_xyz_noise.std_m must contain finite non-negative values.")
    if len(result) != len(set(result)):
        raise ValueError("input_xyz_noise.std_m must not contain duplicates.")
    return sorted(result)


def build_settings(config: dict) -> dict:
    project_root = _resolve(Path.cwd(), get_value(config, "project_root"))
    dataset_root = _resolve(project_root, get_value(config, "dataset_root", "dataset"))

    source_dataset = str(get_value(config, "source_dataset", "")).strip()
    run_name = str(get_value(config, "skillvla_run", "")).strip()
    target_task = str(get_value(config, "target_task", "")).strip()
    if not source_dataset or not run_name or not target_task:
        raise ValueError("source_dataset, skillvla_run, and target_task are required.")
    if Path(source_dataset).name != source_dataset or Path(run_name).name != run_name:
        raise ValueError("source_dataset and skillvla_run must be bare folder names.")

    run_dir = dataset_root / "skillvla_dataset" / source_dataset / run_name
    skill_dataset_dir = run_dir / "skillvla"
    latents_path = run_dir / "skill_latents.npz"
    info_path = skill_dataset_dir / "meta" / "info.json"
    exact_path = dataset_root / "skillvla_dataset" / source_dataset / "eval_init_states.npz"
    original_dataset_dir = _resolve(
        project_root,
        get_value(config, "original_dataset_dir", f"libero_original_dataset/{target_task}"),
    )
    required = {
        "SkillVLA dataset": skill_dataset_dir,
        "skill assignments": latents_path,
        "SkillVLA metadata": info_path,
        "episode-exact map": exact_path,
        "original LIBERO dataset": original_dataset_dir,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing attention-preview input(s):\n  " + "\n  ".join(missing)
        )

    task_ids = _int_list_or_all(get_value(config, "task_ids", "all"), field="task_ids")
    episode_ids = _optional_int_list(
        get_value(config, "episode_ids", []), field="episode_ids"
    )
    episodes_per_task = int(get_value(config, "episodes_per_task", 2))
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive.")
    episode_selection = str(get_value(config, "episode_selection", "first")).lower()
    if episode_selection not in {"first", "random"}:
        raise ValueError("episode_selection must be first|random.")

    agent = _subsection(config, "cameras", "agent")
    wrist = _subsection(config, "cameras", "wrist")
    agent_camera = str(agent.get("name", "agentview")).strip()
    agent_video_key = str(
        agent.get("video_key", "observation.images.image")
    ).strip()
    wrist_video_key = str(
        wrist.get("video_key", "observation.images.wrist_image")
    ).strip()
    if not agent_camera or not agent_video_key or not wrist_video_key:
        raise ValueError("Camera names and video keys must be non-empty.")
    if agent_video_key == wrist_video_key:
        raise ValueError("Agent and wrist video keys must be different.")

    attention = _mapping(config, "attention")
    patch_grid = int(attention.get("patch_grid", 14))
    soft_sigma = float(attention.get("soft_sigma", 0.7))
    frames_per_skill = int(attention.get("frames_per_skill", 5))
    if patch_grid <= 0:
        raise ValueError("attention.patch_grid must be positive.")
    if not 0.0 <= soft_sigma < float("inf"):
        raise ValueError("attention.soft_sigma must be finite and non-negative.")
    if frames_per_skill < 2:
        raise ValueError("attention.frames_per_skill must be at least 2.")

    noise = _mapping(config, "input_xyz_noise")
    noise_std_m = _noise_levels(
        noise.get("std_m", [0.0, 0.005, 0.01, 0.02, 0.05])
    )
    noise_samples = int(noise.get("samples_per_level", 32))
    if noise_samples <= 0:
        raise ValueError("input_xyz_noise.samples_per_level must be positive.")

    output_name = _safe_name(
        get_value(config, "output_name", "skill_attention_preview")
    )
    output_dir = _HERE.parent / "outputs" / output_name
    default_partitions = [
        str(value).strip()
        for value in as_list(get_value(config, "train_partition", "debug"))
        if str(value).strip()
    ]
    default_excludes = [
        str(value).strip()
        for value in as_list(get_value(config, "train_exclude_nodes", []))
        if str(value).strip()
    ]
    slurm = _mapping(config, "slurm")
    partitions = [
        str(value).strip()
        for value in as_list(slurm.get("partition", default_partitions))
        if str(value).strip()
    ]
    excludes = [
        str(value).strip()
        for value in as_list(slurm.get("exclude_nodes", default_excludes))
        if str(value).strip()
    ]

    return {
        "project_root": str(project_root),
        "lerobot_root": str(project_root / "lerobot"),
        "skill_dataset_dir": str(skill_dataset_dir),
        "skill_latents_path": str(latents_path),
        "eval_init_states_path": str(exact_path),
        "original_dataset_dir": str(original_dataset_dir),
        "target_task": target_task,
        "task_ids": task_ids,
        "episode_ids": episode_ids,
        "episodes_per_task": episodes_per_task,
        "episode_selection": episode_selection,
        "eval_seed": int(get_value(config, "seed", 42)),
        "agent_camera": agent_camera,
        "agent_video_key": agent_video_key,
        "wrist_video_key": wrist_video_key,
        "patch_grid": patch_grid,
        "soft_sigma": soft_sigma,
        "frames_per_skill": frames_per_skill,
        "noise_std_m": json.dumps(noise_std_m, separators=(",", ":")),
        "noise_samples": noise_samples,
        "preview_output_dir": str(output_dir),
        "preview_partition": ",".join(partitions) or "debug",
        "preview_qos": str(slurm.get("qos", get_value(config, "train_qos", "base_qos"))),
        "preview_nodelist": str(
            slurm.get("nodelist", get_value(config, "train_nodelist", ""))
        ),
        "preview_exclude_nodes": ",".join(excludes),
        "preview_gres": str(slurm.get("gres", "")),
        "preview_cpus": int(slurm.get("cpus", 4)),
        "preview_memory": str(slurm.get("memory", "16G")),
        "preview_time": str(slurm.get("time", "02:00:00")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
