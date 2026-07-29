#!/usr/bin/env python3
"""Resolve clean Stage-2 multi-checkpoint evaluation into shell exports."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage2_eval_config.yaml"


def _at(config: dict, *path: str, default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _relocate_project_path(project_root: Path, value: str | Path | None) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return project_root / ".missing-required-path"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return project_root / path
    if path.exists():
        return path
    for anchor in (
        "dataset",
        "dataset_filtered",
        "dataset_ABC",
        "models",
        "outputs",
        "outputs_filtered",
        "outputs_ABC",
    ):
        if anchor in path.parts:
            return project_root.joinpath(*path.parts[path.parts.index(anchor) :])
    return path


def _safe_name(value: str, *, field: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or "/" in value or "\0" in value:
        raise ValueError(f"{field} must be a non-empty folder name, got {value!r}.")
    return value


def _clean_label(value: str) -> str:
    value = value.replace("/", "_").strip()
    if not value:
        raise ValueError("Every Stage-2 eval model needs a non-empty label.")
    return value


def _default_output_name(models: list[dict]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(models) > 1:
        return f"compare_{len(models)}models_{stamp}"
    model = re.sub(r"[^A-Za-z0-9._-]+", "-", models[0]["model_dir"]).strip("-_")
    raw = f"{model}_{models[0]['checkpoint']}_{stamp}"
    return raw if len(raw) <= 200 else f"stage2_{models[0]['checkpoint']}_{stamp}"


def _checkpoint_contract(policy_path: Path, project_root: Path) -> dict:
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    )
    missing = [name for name in required if not (policy_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete Stage-2 checkpoint at {policy_path}: missing {missing}."
        )

    policy = json.loads((policy_path / "config.json").read_text())
    if policy.get("type", policy.get("model_type")) != "skill_vla_stage2":
        raise ValueError(f"Expected a skill_vla_stage2 checkpoint: {policy_path}")
    if not as_bool(policy.get("train_skill_predictor", False)):
        raise ValueError(f"Stage-2 checkpoint has no frozen VLM/predictor: {policy_path}")
    if not as_bool(policy.get("train_terminator", False)):
        raise ValueError(f"Stage-2 checkpoint has no retained terminator: {policy_path}")

    train_config = json.loads((policy_path / "train_config.json").read_text())
    dataset_value = str((train_config.get("dataset") or {}).get("root") or "").strip()
    if not dataset_value:
        raise ValueError(f"Stage-2 train_config has no dataset.root: {policy_path}")
    skill_dataset_dir = _relocate_project_path(project_root, dataset_value)
    if not (skill_dataset_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"Stage-2 SkillVLA dataset not found: {skill_dataset_dir}"
        )
    run_dir = skill_dataset_dir.parent
    source_dir = run_dir.parent
    if len(source_dir.parents) < 2:
        raise ValueError(f"Unexpected Stage-2 dataset layout: {skill_dataset_dir}")

    paths = {
        "fsq_path": _relocate_project_path(project_root, policy.get("fsq_path")),
        "skill_dataset_dir": skill_dataset_dir,
        "eval_init_states_path": source_dir / "eval_init_states.npz",
        "skill_latents_path": run_dir / "skill_latents.npz",
        "raw_dataset_dir": source_dir.parents[1] / source_dir.name,
        "dino_model_path": _relocate_project_path(
            project_root, policy.get("dino_model_path")
        ),
        "terminator_dino_model_path": _relocate_project_path(
            project_root,
            policy.get("terminator_dino_model_path") or policy.get("dino_model_path"),
        ),
        "tokenizer_path": _relocate_project_path(
            project_root, policy.get("tokenizer_path")
        ),
    }
    if not paths["fsq_path"].is_file():
        raise FileNotFoundError(f"Stage-2 FSQ checkpoint not found: {paths['fsq_path']}")
    for key in ("dino_model_path", "terminator_dino_model_path", "tokenizer_path"):
        if not paths[key].is_dir():
            raise FileNotFoundError(f"Stage-2 dependency not found ({key}): {paths[key]}")
    return {"policy": policy, **paths}


def _model_entries(config: dict) -> list[dict]:
    default_checkpoint = str(get_value(config, "checkpoint", "last"))
    default_skill_source = str(get_value(config, "skill_source", "gt")).lower()
    default_advance = str(
        _at(config, "oracle", "advance_mode", default="terminator")
    ).lower()
    models = get_value(config, "models", None)
    if isinstance(models, list) and models:
        raw_entries = models
    else:
        model_dir = str(get_value(config, "model_dir", "") or "")
        if not model_dir:
            raise ValueError("Set models[] or a top-level model_dir in Stage-2 eval config.")
        raw_entries = [{"model_dir": model_dir}]

    entries = []
    for index, raw in enumerate(raw_entries):
        model_dir = _safe_name(str(raw.get("model_dir", "")), field="models[].model_dir")
        checkpoint = _safe_name(
            str(raw.get("checkpoint", default_checkpoint)), field="models[].checkpoint"
        )
        skill_source = str(raw.get("skill_source", default_skill_source)).lower()
        if skill_source not in {"gt", "predictor"}:
            raise ValueError("models[].skill_source must be gt|predictor.")
        advance_mode = str(raw.get("advance_mode", default_advance)).lower()
        if advance_mode not in {"terminator", "gt"}:
            raise ValueError("models[].advance_mode must be terminator|gt.")
        if skill_source == "predictor" and advance_mode != "terminator":
            raise ValueError("skill_source=predictor requires advance_mode=terminator.")
        label = str(raw.get("label", "") or "").strip()
        entries.append(
            {
                "model_dir": model_dir,
                "checkpoint": checkpoint,
                "skill_source": skill_source,
                "advance_mode": advance_mode,
                "label": _clean_label(label or f"model{index + 1}-{skill_source}"),
            }
        )
    labels = [entry["label"] for entry in entries]
    if len(labels) != len(set(labels)):
        raise ValueError(f"models[].label values must be unique, got {labels}.")
    return entries


def build_settings(config: dict) -> dict:
    project_root = Path(str(get_value(config, "project_root"))).expanduser()
    outputs_root = project_root / str(get_value(config, "outputs_root", "outputs"))
    eval_outputs_root = _HERE.parent.parent / "outputs"
    entries = _model_entries(config)
    resolved = []
    for entry in entries:
        policy_path = (
            outputs_root
            / "skillVLA_stage2"
            / entry["model_dir"]
            / "checkpoints"
            / entry["checkpoint"]
            / "pretrained_model"
        )
        resolved.append(
            {
                **entry,
                "policy_path": policy_path,
                **_checkpoint_contract(policy_path, project_root),
            }
        )

    episode_exact = as_bool(_at(config, "oracle", "episode_exact", default=False))
    if episode_exact:
        init_state_paths = {model["eval_init_states_path"].resolve() for model in resolved}
        if len(init_state_paths) != 1:
            raise ValueError(
                "Multi-model episode-exact comparison requires the same source init-state map."
            )
        for model in resolved:
            if not model["eval_init_states_path"].is_file():
                raise FileNotFoundError(
                    "oracle.episode_exact=true requires "
                    f"{model['eval_init_states_path']}. Build the oracle map first."
                )

    end_mode = str(_at(config, "terminator", "end_mode", default="or")).lower()
    if end_mode not in {"termination", "progress", "or", "and"}:
        raise ValueError("terminator.end_mode must be termination|progress|or|and.")
    n_action_steps = int(
        get_value(config, "n_action_steps", resolved[0]["policy"].get("n_action_steps", 10))
    )
    for model in resolved:
        chunk_size = int(model["policy"].get("chunk_size", n_action_steps))
        if not 1 <= n_action_steps <= chunk_size:
            raise ValueError(
                f"n_action_steps={n_action_steps} exceeds {model['label']}'s "
                f"chunk_size={chunk_size}."
            )

    task_ids = get_value(config, "task_ids", list(range(10)))
    if isinstance(task_ids, str):
        task_ids = json.loads(task_ids)
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError("task_ids must be a non-empty JSON/YAML list.")
    task_ids = [int(task_id) for task_id in task_ids]

    output_name = str(get_value(config, "output_name", "") or "").strip()
    output_name = _safe_name(
        output_name or _default_output_name(resolved), field="output_name"
    )
    models_json = json.dumps(
        [
            {
                key: (
                    ""
                    if key == "eval_init_states_path" and not episode_exact
                    else str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in model.items()
                if key != "policy"
            }
            for model in resolved
        ],
        separators=(",", ":"),
    )
    primary = resolved[0]
    settings = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "models_json": models_json,
        "model_count": len(resolved),
        "models_per_row": int(get_value(config, "models_per_row", 2) or 0),
        "policy_path": primary["policy_path"],
        "fsq_path": primary["fsq_path"],
        "skill_dataset_dir": primary["skill_dataset_dir"],
        "eval_init_states_path": primary["eval_init_states_path"] if episode_exact else "",
        "skill_latents_path": primary["skill_latents_path"],
        "raw_dataset_dir": primary["raw_dataset_dir"],
        "dino_model_path": primary["dino_model_path"],
        "terminator_dino_model_path": primary["terminator_dino_model_path"],
        "tokenizer_path": primary["tokenizer_path"],
        "eval_out_dir": eval_outputs_root / output_name,
        "target_task": str(get_value(config, "target_task", "libero_90")),
        "task_ids": json.dumps(task_ids, separators=(",", ":")),
        "eval_num_gpus": int(get_value(config, "eval_num_gpus", 1)),
        "n_episodes": int(get_value(config, "n_episodes", 3)),
        "eval_batch_size": int(get_value(config, "eval_batch_size", 1)),
        "max_parallel_tasks": int(get_value(config, "max_parallel_tasks", 1)),
        "n_action_steps": n_action_steps,
        "skill_end_mode": end_mode,
        "skill_end_threshold": float(
            _at(config, "terminator", "end_threshold", default=0.5)
        ),
        "skill_end_progress_threshold": float(
            _at(config, "terminator", "progress_threshold", default=0.95)
        ),
        "inference_skill_max_length": int(
            _at(config, "terminator", "max_skill_length", default=150)
        ),
        "max_videos_per_task": int(_at(config, "video", "max_per_task", default=3)),
        "video_frame_stride": int(_at(config, "video", "frame_stride", default=2)),
        "video_fps": int(_at(config, "video", "fps", default=10)),
        "skill_html": as_bool(get_value(config, "skill_html", True)),
        "skill_html_train_samples": int(get_value(config, "skill_html_train_samples", 5)),
        "wandb_enable": as_bool(_at(config, "logging", "wandb", "enable", default=True)),
        "wandb_project": str(
            _at(config, "logging", "wandb", "project", default="VLA_stage2_eval")
        ),
        "wandb_run_name": f"S2eval_{output_name}"
        + (f"_{os.environ['TASK_TAG']}" if os.environ.get("TASK_TAG") else ""),
    }
    if settings["n_episodes"] <= 0 or settings["eval_batch_size"] <= 0:
        raise ValueError("n_episodes and eval_batch_size must be positive.")
    if settings["max_parallel_tasks"] != 1:
        raise ValueError("Stage-2 policies are stateful; max_parallel_tasks must remain 1.")
    settings.update(
        {
            "eval_partition": ",".join(
                as_list(get_value(config, "train_partition", ["debug"]))
            )
            or "debug",
            "eval_qos": str(get_value(config, "train_qos", "base_qos")),
            "eval_gres": str(_at(config, "slurm", "gres", default="gpu:1")),
            "eval_cpus_per_task": int(_at(config, "slurm", "cpus", default=8)),
            "eval_mem": str(_at(config, "slurm", "memory", default="64G")),
            "eval_time": str(_at(config, "slurm", "time", default="4:00:00")),
            "eval_nodelist": str(get_value(config, "train_nodelist", "")),
            "eval_exclude_nodes": ",".join(
                as_list(get_value(config, "train_exclude_nodes", []))
            ),
        }
    )
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
