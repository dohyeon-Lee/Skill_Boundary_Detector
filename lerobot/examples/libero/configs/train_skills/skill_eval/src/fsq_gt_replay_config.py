#!/usr/bin/env python3
"""Resolve an FSQ-only, episode-exact GT skill replay evaluation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "src"))

from eval_config import _resolve_fsq_artifact  # noqa: E402
from train_skills_config import (  # noqa: E402
    as_bool,
    as_list,
    get_value,
    load_config,
    print_shell,
)

DEFAULT_CONFIG_PATH = _HERE.parent / "fsq_gt_replay_config.yaml"


def _at(config: dict, section: str, key: str, default=None):
    value = config.get(section, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a YAML mapping.")
    return value.get(key, default)


def _json_int_list(value: object, *, field: str, allow_empty: bool) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{field} must be {qualifier}.")
    result = [int(item) for item in value]
    if any(item < 0 for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique non-negative integers: {result}.")
    return result


def _safe_relative_output(value: object, *, default: str) -> Path:
    text = str(value or "").strip() or default
    path = Path(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"output_name must be a safe relative path, got {text!r}.")
    if any(not re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in path.parts):
        raise ValueError(
            "output_name components may contain only letters, digits, '.', '_', and '-': "
            f"{text!r}."
        )
    return path


def _fsq_checkpoints(config: dict) -> list[str]:
    checkpoints = [
        str(value).strip().lower()
        for value in as_list(get_value(config, "fsq_eval_checkpoint", "last"))
        if str(value).strip()
    ]
    if not checkpoints:
        raise ValueError("fsq_eval_checkpoint must contain at least one checkpoint.")
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError(f"fsq_eval_checkpoint contains duplicates: {checkpoints}.")
    return checkpoints


def build_settings(config: dict, *, checkpoint_override: str | None = None) -> dict:
    project_root = Path(str(get_value(config, "project_root"))).expanduser().resolve()
    dataset_root = Path(str(get_value(config, "dataset_root", "dataset"))).expanduser()
    outputs_root = Path(str(get_value(config, "outputs_root", "outputs"))).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    if not outputs_root.is_absolute():
        outputs_root = project_root / outputs_root

    requested_checkpoints = _fsq_checkpoints(config)
    artifacts = []
    skipped_checkpoints = []
    for checkpoint in requested_checkpoints:
        artifact = _resolve_fsq_artifact(
            config,
            dataset_root=dataset_root,
            outputs_root=outputs_root,
            checkpoint=checkpoint,
            missing_ok=True,
        )
        if artifact is None:
            skipped_checkpoints.append(checkpoint)
        else:
            artifacts.append(artifact)
    if not artifacts:
        raise FileNotFoundError(
            f"No fsq_eval_checkpoint entry {requested_checkpoints} has a trained "
            "FSQ checkpoint file yet."
        )
    if skipped_checkpoints:
        print(
            "fsq_gt_replay: skipping not-yet-trained checkpoints "
            f"{skipped_checkpoints}; evaluating "
            f"{[a['fsq_eval_resolved_checkpoint'] for a in artifacts]}.",
            file=sys.stderr,
        )
    resolved_checkpoints = [
        artifact["fsq_eval_resolved_checkpoint"] for artifact in artifacts
    ]
    epoch_tags = [artifact["fsq_eval_epoch_tag"] for artifact in artifacts]
    if len(resolved_checkpoints) != len(set(resolved_checkpoints)):
        raise ValueError(
            "fsq_eval_checkpoint entries resolve to duplicate checkpoints: "
            f"{resolved_checkpoints}."
        )
    if len(epoch_tags) != len(set(epoch_tags)):
        raise ValueError(
            f"fsq_eval_checkpoint entries resolve to duplicate epoch tags: {epoch_tags}."
        )
    selected_checkpoint = str(
        checkpoint_override or resolved_checkpoints[0]
    ).strip().lower()
    selected = [
        artifact
        for artifact in artifacts
        if selected_checkpoint
        in {
            artifact["fsq_eval_selected_checkpoint"],
            artifact["fsq_eval_resolved_checkpoint"],
        }
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Checkpoint override {selected_checkpoint!r} is not one of "
            f"{resolved_checkpoints}."
        )
    artifact = selected[0]
    model_path = Path(artifact["fsq_eval_model_path"])
    latents_path = Path(artifact["fsq_eval_latents_path"])
    skill_dataset_dir = Path(artifact["fsq_eval_dataset_dir"])
    skills_dir = Path(artifact["fsq_eval_skillset_dir"]) / "skills"
    dataset_dirs = {item["fsq_eval_dataset_dir"] for item in artifacts}
    run_names = {item["fsq_eval_run_name"] for item in artifacts}
    if len(dataset_dirs) != 1 or len(run_names) != 1:
        raise ValueError("All replay checkpoints must belong to one FSQ run and dataset.")

    target_dataset = skill_dataset_dir.name
    target_task = str(get_value(config, "target_task", "libero_90")).strip()
    if not target_task:
        raise ValueError("target_task must be non-empty.")
    task_ids = _json_int_list(
        get_value(config, "task_ids", [0]), field="task_ids", allow_empty=False
    )
    episode_ids = _json_int_list(
        get_value(config, "episode_ids", []), field="episode_ids", allow_empty=True
    )
    episodes_per_task = int(get_value(config, "episodes_per_task", 2))
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive.")
    episode_selection = str(
        get_value(config, "episode_selection", "first")
    ).strip().lower()
    if episode_selection not in {"first", "random"}:
        raise ValueError("episode_selection must be first|random.")

    eval_init_states_path = (
        dataset_root / "skillvla_dataset" / target_dataset / "eval_init_states.npz"
    )
    original_dataset_dir = project_root / "libero_original_dataset" / target_task
    for label, path in (
        ("FSQ checkpoint", model_path),
        ("FSQ source dataset", skill_dataset_dir),
        ("episode-exact map", eval_init_states_path),
        ("original LIBERO dataset", original_dataset_dir),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    requested_gpus = int(get_value(config, "eval_num_gpus", 1))
    workers_per_checkpoint = int(get_value(config, "workers_per_checkpoint", 1))
    if requested_gpus <= 0 or workers_per_checkpoint <= 0:
        raise ValueError("eval_num_gpus and workers_per_checkpoint must be positive.")
    selected_episode_upper_bound = (
        len(episode_ids) if episode_ids else len(task_ids) * episodes_per_task
    )
    worker_count = min(workers_per_checkpoint, selected_episode_upper_bound)
    concurrent_jobs = min(requested_gpus, len(artifacts) * worker_count)

    run_name = artifact["fsq_eval_run_name"]
    epoch_tag = artifact["fsq_eval_epoch_tag"]
    default_output = (
        f"{run_name}/{epoch_tags[0]}"
        if len(epoch_tags) == 1
        else f"{run_name}/compare_{'_'.join(epoch_tags)}"
    )
    output_relative = _safe_relative_output(
        get_value(config, "output_name", ""),
        default=default_output,
    )
    eval_dir = _HERE.parent
    collection_dir = eval_dir / "outputs" / "fsq_gt_replay" / output_relative
    checkpoint_output_dir = collection_dir / "checkpoints" / epoch_tag
    exclude = as_list(get_value(config, "train_exclude_nodes", []))
    return {
        "project_root": str(project_root),
        "lerobot_root": str(project_root / "lerobot"),
        "fsq_gt_replay_dir": str(eval_dir),
        "fsq_model_path": str(model_path),
        "fsq_latents_path": str(latents_path),
        "fsq_skills_dir": str(skills_dir),
        "skill_dataset_dir": str(skill_dataset_dir),
        "eval_init_states_path": str(eval_init_states_path),
        "original_dataset_dir": str(original_dataset_dir),
        "fsq_run_name": run_name,
        "fsq_epoch_tag": epoch_tag,
        "fsq_eval_checkpoints": " ".join(resolved_checkpoints),
        "fsq_skipped_checkpoints": " ".join(skipped_checkpoints),
        "fsq_expected_epoch_tags": json.dumps(epoch_tags, separators=(",", ":")),
        "fsq_checkpoint_count": len(artifacts),
        "target_task": target_task,
        "task_ids": json.dumps(task_ids, separators=(",", ":")),
        "episode_ids": json.dumps(episode_ids, separators=(",", ":")),
        "episodes_per_task": episodes_per_task,
        "episode_selection": episode_selection,
        "eval_seed": int(get_value(config, "seed", 42)),
        "eval_num_gpus": concurrent_jobs,
        "eval_worker_count": worker_count,
        "eval_resume": str(as_bool(get_value(config, "resume", False))).lower(),
        "eval_collection_dir": str(collection_dir),
        "eval_out_dir": str(checkpoint_output_dir),
        "eval_partition": ",".join(
            as_list(get_value(config, "train_partition", ["debug"]))
        ) or "debug",
        "eval_qos": str(get_value(config, "train_qos", "base_qos")),
        "eval_gres": str(_at(config, "slurm", "gres", "gpu:1")),
        "eval_cpus_per_task": int(_at(config, "slurm", "cpus", 8)),
        "eval_mem": str(_at(config, "slurm", "memory", "64G")),
        "eval_time": str(_at(config, "slurm", "time", "12:00:00")),
        "eval_nodelist": str(get_value(config, "train_nodelist", "")),
        "eval_exclude_nodes": ",".join(exclude),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(
        load_config(args.config), checkpoint_override=args.checkpoint
    )
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
