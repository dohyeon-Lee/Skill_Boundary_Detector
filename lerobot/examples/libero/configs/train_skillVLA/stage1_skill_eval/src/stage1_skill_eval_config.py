#!/usr/bin/env python3
"""Resolve the single-model Stage-1 skill-segment evaluation config."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve()
_TRAIN_SKILLVLA = _HERE.parent.parent.parent
_PROJECT_ROOT_DEFAULT = _HERE.parents[7]
sys.path.insert(0, str(_TRAIN_SKILLVLA / "stage1_eval" / "src"))
sys.path.insert(0, str(_TRAIN_SKILLVLA.parent / "train_skills" / "src"))

from stage1_eval_config import (  # noqa: E402
    _checkpoint_contract,
    _relocate_project_path,
    _validate_external_terminator,
)
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage1_skill_eval_config.yaml"


def _at(config: dict, *path: str, default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _safe_name(value: str, *, field: str) -> str:
    value = str(value).strip()
    if not value or value in {".", ".."} or "/" in value or "\0" in value:
        raise ValueError(f"{field} must be a non-empty folder name, got {value!r}.")
    return value


def _clean_label(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-_")
    if not value:
        raise ValueError("model.label must contain at least one safe character.")
    return value


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else project_root / path


def build_settings(config: dict) -> dict:
    project_root = Path(str(get_value(config, "project_root", _PROJECT_ROOT_DEFAULT))).expanduser()
    outputs_root = project_root / str(get_value(config, "outputs_root", "outputs"))
    model = get_value(config, "model", {})
    if not isinstance(model, dict):
        raise ValueError("model must be a YAML mapping.")
    allowed_model_fields = {
        "previous",
        "model_dir",
        "checkpoint",
        "label",
        "terminator_source",
    }
    unknown = sorted(set(model) - allowed_model_fields)
    if unknown:
        raise ValueError(f"Unknown model fields {unknown}; supported={sorted(allowed_model_fields)}.")

    model_dir = _safe_name(model.get("model_dir", ""), field="model.model_dir")
    checkpoint = _safe_name(model.get("checkpoint", ""), field="model.checkpoint")
    previous = as_bool(model.get("previous", False))
    model_root = outputs_root / "skillVLA_stage1" / ("previous" if previous else "")
    policy_path = model_root / model_dir / "checkpoints" / checkpoint / "pretrained_model"
    contract = _checkpoint_contract(policy_path, project_root)

    terminator_source = str(model.get("terminator_source", "own")).strip().lower()
    if terminator_source not in {"own", "external"}:
        raise ValueError("model.terminator_source must be own|external.")
    external_value = str(get_value(config, "external_skill_model", "") or "").strip()
    external_skill_model = (
        _relocate_project_path(project_root, external_value) if external_value else None
    )
    if terminator_source == "own":
        if not contract["has_terminator"]:
            raise ValueError(
                f"terminator_source=own but checkpoint has no trained terminator: {policy_path}"
            )
    else:
        if external_skill_model is None:
            raise ValueError("terminator_source=external requires external_skill_model.")
        _validate_external_terminator(
            external_skill_model,
            target_policy=contract["policy"],
        )

    if not as_bool(get_value(config, "episode_exact", True)):
        raise ValueError(
            "stage1_skill_eval requires episode_exact=true; no approximate fallback exists."
        )
    if not contract["eval_init_states_path"].is_file():
        raise FileNotFoundError(
            "Episode-exact map not found: "
            f"{contract['eval_init_states_path']}. Build it with stage1_eval/oracle_matching."
        )
    if not contract["skill_latents_path"].is_file():
        raise FileNotFoundError(f"Skill occurrence metadata not found: {contract['skill_latents_path']}")

    target_task = str(get_value(config, "target_task", "libero_90")).strip()
    task_ids = get_value(config, "task_ids", [0])
    if isinstance(task_ids, str):
        task_ids = json.loads(task_ids)
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError("task_ids must be a non-empty list.")
    task_ids = [int(value) for value in task_ids]
    if any(value < 0 for value in task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError(f"task_ids must be unique non-negative integers, got {task_ids}.")

    episode_ids = get_value(config, "episode_ids", []) or []
    if isinstance(episode_ids, str):
        episode_ids = json.loads(episode_ids)
    if not isinstance(episode_ids, list):
        raise ValueError("episode_ids must be a list.")
    episode_ids = [int(value) for value in episode_ids]
    if any(value < 0 for value in episode_ids) or len(episode_ids) != len(set(episode_ids)):
        raise ValueError(f"episode_ids must be unique non-negative integers, got {episode_ids}.")

    episode_selection = str(get_value(config, "episode_selection", "first")).strip().lower()
    if episode_selection not in {"first", "random"}:
        raise ValueError("episode_selection must be first|random.")
    episodes_per_task = int(get_value(config, "episodes_per_task", 10))
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive.")
    requested_eval_gpus = int(get_value(config, "eval_num_gpus", 1))
    if requested_eval_gpus <= 0:
        raise ValueError("eval_num_gpus must be positive.")
    selected_episode_upper_bound = (
        len(episode_ids) if episode_ids else len(task_ids) * episodes_per_task
    )
    eval_num_gpus = min(requested_eval_gpus, selected_episode_upper_bound)

    time_shift_offset = int(get_value(config, "time_shift_offset", 15))
    if time_shift_offset <= 0:
        raise ValueError("time_shift_offset must be a positive integer.")
    n_action_steps = int(get_value(config, "n_action_steps", 5))
    chunk_size = int(contract["policy"].get("chunk_size", n_action_steps))
    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError(
            f"n_action_steps={n_action_steps} must be in [1, chunk_size={chunk_size}]."
        )

    end_mode = str(_at(config, "terminator", "end_mode", default="or")).strip().lower()
    if end_mode not in {"termination", "progress", "or", "and"}:
        raise ValueError("terminator.end_mode must be termination|progress|or|and.")
    max_skill_length = int(_at(config, "terminator", "max_skill_length", default=200))
    if max_skill_length <= 0:
        raise ValueError("terminator.max_skill_length must be positive.")

    original_dataset_dir = _resolve_path(
        project_root,
        get_value(config, "original_dataset_dir", f"libero_original_dataset/{target_task}"),
    )
    if not original_dataset_dir.is_dir():
        raise FileNotFoundError(f"Original LIBERO HDF5 folder not found: {original_dataset_dir}")

    label = _clean_label(model.get("label", f"{model_dir}_{checkpoint}"))
    output_name = str(get_value(config, "output_name", "") or "").strip()
    if not output_name:
        output_name = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_name = _safe_name(output_name, field="output_name")

    spec = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in {
            "label": label,
            "model_dir": model_dir,
            "checkpoint": checkpoint,
            "previous_checkpoint": previous,
            "policy_path": policy_path,
            "skill_source": "gt",
            "advance_mode": terminator_source,
            "external_skill_model": external_skill_model or "",
            **{key: value for key, value in contract.items() if key != "policy"},
        }.items()
    }
    eval_dir = _HERE.parent.parent
    settings = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "stage1_skill_eval_dir": eval_dir,
        "spec_json": json.dumps(spec, separators=(",", ":")),
        "policy_path": policy_path,
        "fsq_path": contract["fsq_path"],
        "dino_model_path": contract["dino_model_path"],
        "tokenizer_path": contract["tokenizer_path"],
        "skill_dataset_dir": contract["skill_dataset_dir"],
        "skill_latents_path": contract["skill_latents_path"],
        "eval_init_states_path": contract["eval_init_states_path"],
        "original_dataset_dir": original_dataset_dir,
        "architecture_label": contract["architecture_label"],
        "model_label": label,
        "target_task": target_task,
        "task_ids": json.dumps(task_ids, separators=(",", ":")),
        "episode_ids": json.dumps(episode_ids, separators=(",", ":")),
        "episodes_per_task": episodes_per_task,
        "episode_selection": episode_selection,
        "episode_exact": True,
        "eval_num_gpus": eval_num_gpus,
        "eval_seed": int(get_value(config, "seed", 42)),
        "time_shift_offset": time_shift_offset,
        "n_action_steps": n_action_steps,
        "skill_end_mode": end_mode,
        "skill_end_threshold": float(
            _at(config, "terminator", "end_threshold", default=0.5)
        ),
        "skill_end_progress_threshold": float(
            _at(config, "terminator", "progress_threshold", default=0.95)
        ),
        "inference_skill_max_length": max_skill_length,
        "finish_action_chunk_on_end": as_bool(
            _at(config, "terminator", "finish_action_chunk_on_end", default=True)
        ),
        "video_frame_stride": int(_at(config, "video", "frame_stride", default=2)),
        "video_fps": int(_at(config, "video", "fps", default=10)),
        "eval_resume": as_bool(get_value(config, "resume", False)),
        "eval_out_dir": eval_dir / "outputs" / output_name,
        "eval_partition": ",".join(
            as_list(get_value(config, "train_partition", ["debug"]))
        ) or "debug",
        "eval_qos": str(get_value(config, "train_qos", "base_qos")),
        "eval_gres": str(_at(config, "slurm", "gres", default="gpu:1")),
        "eval_cpus_per_task": int(_at(config, "slurm", "cpus", default=8)),
        "eval_mem": str(_at(config, "slurm", "memory", default="64G")),
        "eval_time": str(_at(config, "slurm", "time", default="12:00:00")),
        "eval_nodelist": str(get_value(config, "train_nodelist", "")),
        "eval_exclude_nodes": ",".join(
            as_list(get_value(config, "train_exclude_nodes", []))
        ),
    }
    if settings["video_frame_stride"] <= 0 or settings["video_fps"] <= 0:
        raise ValueError("video.frame_stride and video.fps must be positive.")
    return settings


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
