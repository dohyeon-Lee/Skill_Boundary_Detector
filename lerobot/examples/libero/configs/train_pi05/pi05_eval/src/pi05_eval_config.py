#!/usr/bin/env python3
"""Resolve pi05 closed-loop evaluation (configs/train_pi05/pi05_eval).

Mirrors the Stage-1 eval resolver's YAML shape (model_defaults / models / output_name / resume /
video / logging / slurm) without anything skill-specific: a pi05 panel is just a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
from train_pi05_config import (  # noqa: E402
    _auto_labels,
    as_bool,
    as_list,
    load_config,
    print_shell,
    resolve_path,
)

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "pi05_eval_config.yaml"
_STAGE_ROOT_KEYS = {"PT": ("pi05_outputs_root", "pi05_PT"), "FT": ("pi05_ft_outputs_root", "pi05_FT")}
_MODEL_KEYS = {"model_dir", "checkpoint", "label", "stage"}
_TOP_LEVEL_KEYS = {
    "pi05_outputs_root", "pi05_ft_outputs_root", "pi05_tokenizer", "model_defaults", "models",
    "output_name", "resume", "target_task", "task_ids", "episode_offset", "eval_num_gpus",
    "eval_max_workers_per_gpu", "n_episodes", "eval_batch_size", "max_parallel_tasks",
    "n_action_steps", "video", "logging", "slurm", "oracle",
}
_GLOBAL_KEYS = {
    "project_root", "dataset_root", "outputs_root", "train_partition", "train_qos",
    "train_nodelist", "train_exclude_nodes",
}


def _at(config: dict, *path: str, default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _safe_name(value: str, *, field: str) -> str:
    value = str(value).strip()
    if not value or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError(f"{field} may contain only letters, numbers, '.', '_' and '-': {value!r}")
    return value


def panel_dir_name(index: int, label: str) -> str:
    """``00_<label>`` - the same panel folder naming as the Stage-1 evaluator."""
    safe = "".join(character if character.isalnum() or character in "._-" else "-" for character in label)
    return f"{index:02d}_{safe.strip('-_') or 'model'}"


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    """``last`` → the greatest numeric step; otherwise the named step. Returns pretrained_model."""
    checkpoints = run_dir / "checkpoints"
    if str(checkpoint).strip().lower() == "last":
        steps = sorted(p.name for p in checkpoints.glob("*") if p.name.isdigit())
        if not steps:
            raise FileNotFoundError(f"No numeric checkpoints under {checkpoints}")
        checkpoint = steps[-1]
    path = checkpoints / str(checkpoint) / "pretrained_model"
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"pi05 checkpoint not found: {path}")
    return path


def _model_entries(config: dict) -> list[dict]:
    defaults = config.get("model_defaults", {}) or {}
    unknown = set(defaults) - (_MODEL_KEYS - {"model_dir", "label"})
    if unknown:
        raise ValueError(f"Unsupported model_defaults keys: {sorted(unknown)}")
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list of {model_dir, checkpoint?, label?, stage?}.")
    entries = []
    for index, raw in enumerate(models):
        if not isinstance(raw, dict) or set(raw) - _MODEL_KEYS:
            raise ValueError(f"models[{index}] supports only {sorted(_MODEL_KEYS)}, got {raw!r}")
        model_dir = str(raw.get("model_dir", "") or "").strip()
        if not model_dir:
            raise ValueError(f"models[{index}].model_dir is required.")
        stage = str(raw.get("stage", defaults.get("stage", "PT"))).strip().upper()
        if stage not in _STAGE_ROOT_KEYS:
            raise ValueError(f"models[{index}].stage must be PT or FT, got {stage!r}")
        entries.append({
            "model_dir": model_dir,
            "checkpoint": str(raw.get("checkpoint", defaults.get("checkpoint", "last"))),
            "label": str(raw.get("label", "") or "").strip(),
            "stage": stage,
        })
    return entries


def build_settings(config: dict) -> dict[str, Any]:
    unknown = set(config) - _TOP_LEVEL_KEYS - _GLOBAL_KEYS
    if unknown:
        raise ValueError(f"Unsupported pi05 eval config keys: {sorted(unknown)}")
    project_root = Path(str(config["project_root"])).expanduser()
    outputs_root = project_root / str(config.get("outputs_root", "outputs"))
    tokenizer = Path(resolve_path(
        project_root, config.get("pi05_tokenizer", "models/paligemma-3b-pt-224-tokenizer")
    ))
    missing = [n for n in ("config.json", "tokenizer_config.json", "tokenizer.json") if not (tokenizer / n).is_file()]
    if missing:
        raise FileNotFoundError(f"Local PaliGemma tokenizer is incomplete at {tokenizer}: missing {missing}")

    entries = _model_entries(config)
    auto_labels = _auto_labels(
        [f"{e['model_dir']}_{e['checkpoint']}" for e in entries]
        if len({e["model_dir"] for e in entries}) != len(entries)
        else [e["model_dir"] for e in entries]
    )
    n_action_steps = int(config.get("n_action_steps", 5))
    panels, labels = [], []
    for entry, auto in zip(entries, auto_labels, strict=True):
        root_key, root_default = _STAGE_ROOT_KEYS[entry["stage"]]
        run_dir = outputs_root / str(config.get(root_key, root_default)) / entry["model_dir"]
        policy_path = resolve_checkpoint(run_dir, entry["checkpoint"])
        policy = json.loads((policy_path / "config.json").read_text())
        if policy.get("type") != "pi05":
            raise ValueError(f"Not a pi05 checkpoint (type={policy.get('type')!r}): {policy_path}")
        chunk_size = int(policy.get("chunk_size", 50))
        if not 1 <= n_action_steps <= chunk_size:
            raise ValueError(
                f"n_action_steps={n_action_steps} must be within [1, chunk_size={chunk_size}] of {policy_path}"
            )
        label = _safe_name((entry["label"] or auto).replace("/", "_").replace(" ", "_"), field="label")
        if label in labels:
            raise ValueError(f"Duplicate panel label {label!r}; set models[].label explicitly.")
        labels.append(label)
        panels.append({
            "label": label,
            "panel_dir": panel_dir_name(len(panels), label),
            "policy_path": str(policy_path),
            "model_dir": entry["model_dir"],
            "checkpoint": policy_path.parent.name,
            "stage": entry["stage"],
            # Informational: lerobot-eval reads the checkpoint's own field and grounds rollouts itself.
            "proprio_grounding": str(policy.get("proprio_grounding", "none") or "none"),
        })

    task_ids = config.get("task_ids", list(range(10)))
    if isinstance(task_ids, str):
        task_ids = json.loads(task_ids)
    task_ids = [int(value) for value in task_ids]
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be a non-empty list of unique task ids.")
    target_task = str(config.get("target_task", "libero_90"))
    episode_offset = int(config.get("episode_offset", 0))
    n_episodes = int(config.get("n_episodes", 5))
    if min(n_episodes, int(config.get("eval_batch_size", 1)), int(config.get("max_parallel_tasks", 1))) <= 0:
        raise ValueError("n_episodes, eval_batch_size and max_parallel_tasks must be positive.")

    # Episode-exact: start every rollout from a matched dataset demo's exact scene — the same
    # init-state map Stage-1 eval uses, so pi05 and SkillVLA panels are scored on identical scenes.
    oracle = config.get("oracle", {}) or {}
    if not isinstance(oracle, dict) or set(oracle) - {"episode_exact", "dataset_source"}:
        raise ValueError("oracle supports only episode_exact and dataset_source.")
    episode_exact = as_bool(oracle.get("episode_exact", False))
    init_states_path = ""
    if episode_exact:
        source = str(oracle.get("dataset_source", "") or "").strip()
        if not source:
            raise ValueError(
                "oracle.episode_exact=true needs oracle.dataset_source (e.g. libero_90_full_full): "
                "pi05 checkpoints do not record a SkillVLA dataset to infer it from."
            )
        init_states_path = (
            project_root / str(config.get("dataset_root", "dataset")) / "skillvla_dataset"
            / source / "eval_init_states.npz"
        )
        if not init_states_path.is_file():
            raise FileNotFoundError(
                f"oracle.episode_exact=true requires {init_states_path}. "
                f"Build it with stage1_eval/oracle_matching/run.sh {source}."
            )
        if episode_offset != 0:
            raise ValueError(
                "episode_offset selects benchmark init states; it must be 0 with oracle.episode_exact=true."
            )

    video_enable = as_bool(_at(config, "video", "enable", default=True))
    grid_columns = int(_at(config, "video", "grid_columns", default=0))
    if grid_columns < 0:
        raise ValueError("video.grid_columns must be >= 0 (0 = one row).")

    init_tag = "exact" if episode_exact else f"offset{episode_offset}"
    output_name = str(config.get("output_name", "") or "").strip()
    if output_name:
        output_name = _safe_name(output_name, field="output_name")
    elif len(panels) == 1:
        output_name = f"{panels[0]['model_dir']}_{panels[0]['checkpoint']}_{target_task}_{init_tag}"
    else:
        output_name = f"compare_{'_vs_'.join(labels)}_{target_task}_{init_tag}"
    if not video_enable and not str(config.get("output_name", "") or "").strip():
        # Keep summary-only runs apart from full video evals of the same panels.
        output_name += "_graph"
    if len(output_name) > 200:
        raise ValueError(f"Output name is too long ({len(output_name)} chars); set output_name.")

    return {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "pi05_tokenizer_path": tokenizer,
        "models_json": json.dumps(panels, separators=(",", ":")),
        "models_labels": json.dumps(labels, separators=(",", ":")),
        "panel_count": len(panels),
        "panel_summary": ", ".join(
            f"{p['label']}[{p['stage']}:{p['checkpoint']}, grounding={p['proprio_grounding']}]" for p in panels
        ),
        "target_task": target_task,
        "task_ids": json.dumps(task_ids, separators=(",", ":")),
        "eval_expected_tasks": len(task_ids),
        "n_episodes": n_episodes,
        "episode_offset": episode_offset,
        "episode_exact": episode_exact,
        "eval_init_states_path": init_states_path,
        "n_action_steps": n_action_steps,
        "eval_batch_size": int(config.get("eval_batch_size", 1)),
        "max_parallel_tasks": int(config.get("max_parallel_tasks", 1)),
        "video_enable": video_enable,
        # lerobot-eval treats zero as "render nothing" while still writing success metrics.
        "max_videos_per_task": int(_at(config, "video", "max_per_task", default=1)) if video_enable else 0,
        "video_frame_stride": int(_at(config, "video", "frame_stride", default=2)),
        "video_fps": int(_at(config, "video", "fps", default=10)),
        # false: rollout view only (+ SUCCESS/FAIL bar and caption); true: add the wrist camera panel.
        "video_show_wrist": as_bool(_at(config, "video", "show_wrist", default=False)),
        "grid_columns": grid_columns,
        "eval_resume": as_bool(config.get("resume", False)),
        "eval_out_dir": _HERE.parent.parent / "outputs" / output_name,
        "wandb_enable": as_bool(_at(config, "logging", "wandb", "enable", default=True)),
        "wandb_project": str(_at(config, "logging", "wandb", "project", default="VLA_eval")),
        "wandb_run_name": output_name,
        "eval_num_gpus": int(config.get("eval_num_gpus", 1)),
        "eval_max_workers_per_gpu": int(config.get("eval_max_workers_per_gpu", 1)),
        "eval_partition": ",".join(as_list(config.get("train_partition", ["debug"]))) or "debug",
        "eval_qos": str(config.get("train_qos", "base_qos")),
        "eval_nodelist": str(config.get("train_nodelist", "")),
        "eval_exclude_nodes": ",".join(as_list(config.get("train_exclude_nodes", []))),
        "eval_gres": str(_at(config, "slurm", "gres", default="gpu:1")),
        "eval_cpus_per_task": int(_at(config, "slurm", "cpus", default=8)),
        "eval_mem": str(_at(config, "slurm", "memory", default="64G")),
        "eval_time": str(_at(config, "slurm", "time", default="6:00:00")),
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
