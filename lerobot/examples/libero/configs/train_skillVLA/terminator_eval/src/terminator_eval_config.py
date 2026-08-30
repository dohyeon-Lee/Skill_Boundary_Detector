#!/usr/bin/env python3
"""Resolve the multi-terminator, single-skill evaluation config."""

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

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "terminator_eval_config.yaml"


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


def _model_checkpoint_path(
    *,
    outputs_root: Path,
    group: str,
    model_dir: object,
    checkpoint: object,
    field: str,
) -> Path:
    group = _safe_name(group, field=f"{field}.group")
    model_dir = _safe_name(str(model_dir or ""), field=f"{field}.model_dir")
    checkpoint = _safe_name(str(checkpoint or ""), field=f"{field}.checkpoint")
    return outputs_root / group / model_dir / "checkpoints" / checkpoint / "pretrained_model"


def _resolve_external_model(
    config: dict,
    project_root: Path,
    outputs_root: Path,
    fsq_path: Path,
) -> tuple[Path | None, str]:
    raw = get_value(config, "external_skill_model", "")
    if raw is None or raw == "":
        return None, "checkpoint"
    if not isinstance(raw, dict):
        # Backward compatibility for existing full/relative checkpoint paths.
        return _relocate_project_path(project_root, str(raw).strip()), "checkpoint"
    variant = str(raw.get("variant", "checkpoint") or "checkpoint").strip().lower()
    if variant in {"fsq", "fsq_base", "fsq_initial"}:
        unknown = sorted(set(raw) - {"variant"})
        if unknown:
            raise ValueError(
                "external_skill_model variant=fsq_initial uses the selected "
                f"policy's FSQ.pt and accepts no checkpoint fields; unknown={unknown}."
            )
        return fsq_path, "fsq_initial"
    canonical_variant = (
        "state_image"
        if variant in {"checkpoint", "trained"}
        else _normalize_display_variant(variant)
    )
    unknown = sorted(set(raw) - {"variant", "group", "model_dir", "checkpoint"})
    if unknown:
        raise ValueError(
            "external_skill_model supports variant/group/model_dir/checkpoint; "
            f"unknown={unknown}."
        )
    default_group = (
        "skillVLA_stage1"
        if canonical_variant == "state_image"
        else "skillVLA_terminator"
    )
    return (
        _model_checkpoint_path(
            outputs_root=outputs_root,
            group=str(raw.get("group", default_group) or default_group),
            model_dir=raw.get("model_dir"),
            checkpoint=raw.get("checkpoint"),
            field="external_skill_model",
        ),
        "checkpoint" if canonical_variant == "state_image" else canonical_variant,
    )


def _normalize_display_variant(value: object) -> str:
    aliases = {
        "normal": "state_image",
        "state_image": "state_image",
        "state+image": "state_image",
        "image": "image_only",
        "image_only": "image_only",
        "image-only": "image_only",
        "wrist": "wrist_only",
        "wrist_only": "wrist_only",
        "wrist-only": "wrist_only",
        "proprio": "state_only",
        "state_only": "state_only",
        "state-only": "state_only",
        "state_only_terminator": "state_only",
        "rnn": "state_rnn",
        "state_rnn": "state_rnn",
        "state-rnn": "state_rnn",
        "state_rnn_terminator": "state_rnn",
        "fsq": "fsq_initial",
        "fsq_base": "fsq_initial",
        "fsq_initial": "fsq_initial",
    }
    normalized = aliases.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(
            "terminator_models[].variant must be "
            "state_image|image_only|wrist_only|state_only|state_rnn|fsq_initial, "
            f"got {value!r}."
        )
    return normalized


def _resolve_display_models(
    config: dict,
    project_root: Path,
    outputs_root: Path,
) -> list[dict[str, str]]:
    """Resolve trained terminators and the unmodified FSQ terminator."""
    raw = get_value(config, "terminator_models", None)
    legacy_raw = get_value(config, "image_only_terminator_model", None)
    has_raw = raw is not None and raw != ""
    has_legacy_raw = legacy_raw is not None and legacy_raw != ""
    if has_raw and has_legacy_raw:
        raise ValueError(
            "Use terminator_models or legacy image_only_terminator_model, not both."
        )
    legacy_mode = not has_raw
    raw = legacy_raw if legacy_mode else raw
    if raw is None or raw == "" or raw == []:
        raise ValueError("terminator_eval requires top-level terminator_models.")
    entries = raw if isinstance(raw, list) else [raw]
    models: list[dict[str, str]] = []
    labels: set[str] = set()
    default_labels = {
        "state_image": "STATE",
        "image_only": "IMAGE",
        "wrist_only": "WRIST",
        "state_only": "STATE_ONLY",
        "state_rnn": "STATE_RNN",
        "fsq_initial": "FSQ_INIT",
    }
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            unknown = sorted(
                set(entry)
                - {"variant", "path", "label", "model_dir", "checkpoint"}
            )
            if unknown:
                raise ValueError(
                    "terminator_models entries support "
                    "variant/label/model_dir/checkpoint; "
                    f"unknown={unknown}."
                )
            variant = _normalize_display_variant(
                entry.get("variant", "image_only")
            )
            path_value = str(entry.get("path", "") or "").strip()
            label = str(entry.get("label", "") or "").strip()
            if path_value and (entry.get("model_dir") or entry.get("checkpoint")):
                raise ValueError(
                    f"terminator_models[{index}] cannot mix legacy path "
                    "with model_dir/checkpoint."
                )
            if variant == "fsq_initial":
                if path_value or entry.get("model_dir") or entry.get("checkpoint"):
                    raise ValueError(
                        f"terminator_models[{index}] variant=fsq_initial uses the "
                        "selected policy's FSQ.pt and accepts no checkpoint path."
                    )
                resolved_path = None
            elif not path_value:
                resolved_path = _model_checkpoint_path(
                    outputs_root=outputs_root,
                    group="skillVLA_terminator",
                    model_dir=entry.get("model_dir"),
                    checkpoint=entry.get("checkpoint"),
                    field=f"terminator_models[{index}]",
                )
        else:
            variant = "image_only"
            path_value = str(entry or "").strip()
            label = ""
            resolved_path = None
        if not path_value and not isinstance(entry, dict):
            raise ValueError(
                f"terminator_models[{index}] requires a non-empty path."
            )
        if not label:
            base_label = default_labels[variant]
            label = base_label if base_label not in labels else f"{base_label}{index + 1}"
        if "\n" in label or "\r" in label or len(label) > 16:
            raise ValueError(
                f"terminator label must be one line and <=16 characters, got {label!r}."
            )
        if label in labels:
            raise ValueError(f"Duplicate terminator label: {label!r}.")
        labels.add(label)
        models.append(
            {
                "label": label,
                "variant": variant,
                "path": (
                    ""
                    if variant == "fsq_initial"
                    else str(
                        resolved_path
                        if not path_value
                        else _relocate_project_path(project_root, path_value)
                    )
                ),
            }
        )
    return models


def _validate_display_model(model: dict[str, str], *, target_policy: dict) -> None:
    variant = model["variant"]
    if variant == "fsq_initial":
        fsq_path = Path(model["path"])
        if not fsq_path.is_file():
            raise FileNotFoundError(
                f"Raw FSQ terminator checkpoint not found: {fsq_path}"
            )
        return
    checkpoint = Path(model["path"])
    if variant in {"state_image", "image_only"}:
        _validate_external_terminator(
            checkpoint,
            target_policy=target_policy,
            variant=variant,
        )
        return
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"External {variant} terminator config not found: {config_path}"
        )
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"External {variant} terminator weights not found: {weights_path}"
        )
    source = json.loads(config_path.read_text())
    if source.get("type") != "skill_aux":
        raise ValueError(
            f"External {variant} terminator must come from policy.type=skill_aux, "
            f"got {source.get('type')!r} at {checkpoint}."
        )
    train_field = {
        "wrist_only": "train_wrist_only_terminator",
        "state_only": "train_state_only_terminator",
        "state_rnn": "train_state_rnn_terminator",
    }.get(variant)
    if train_field is None:
        raise ValueError(f"Unsupported auxiliary terminator variant: {variant!r}.")
    if not as_bool(source.get(train_field, False)):
        raise ValueError(
            f"External checkpoint has no trained {variant} terminator: {checkpoint}"
        )
    if source.get("skill_fsq_levels") != target_policy.get("skill_fsq_levels"):
        raise ValueError(
            f"External {variant} terminator FSQ mismatch: "
            f"terminator={source.get('skill_fsq_levels')!r}, "
            f"target={target_policy.get('skill_fsq_levels')!r}"
        )


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
    if terminator_source not in {"own", "original", "external"}:
        raise ValueError("model.terminator_source must be own|original|external.")
    external_skill_model = None
    external_skill_model_variant = "checkpoint"
    if terminator_source == "external":
        external_skill_model, external_skill_model_variant = _resolve_external_model(
            config,
            project_root,
            outputs_root,
            fsq_path=Path(contract["fsq_path"]),
        )
    display_terminator_models = _resolve_display_models(
        config,
        project_root,
        outputs_root,
    )
    for display_model in display_terminator_models:
        if display_model["variant"] == "fsq_initial":
            # Resolve the path from the selected action policy rather than from
            # external_skill_model. This keeps the raw FSQ baseline independent
            # from whichever predictor/main terminator overlay is configured.
            display_model["path"] = str(contract["fsq_path"])
    if terminator_source == "own":
        if not contract["has_terminator"]:
            raise ValueError(
                f"terminator_source=own but checkpoint has no trained terminator: {policy_path}"
            )
    elif terminator_source == "original":
        fsq_path = Path(contract["fsq_path"])
        if not fsq_path.is_file():
            raise FileNotFoundError(
                "terminator_source=original could not find the selected policy's "
                f"source FSQ checkpoint: {fsq_path}"
            )
    else:
        if external_skill_model is None:
            raise ValueError("terminator_source=external requires external_skill_model.")
        if external_skill_model_variant == "fsq_initial":
            if not external_skill_model.is_file():
                raise FileNotFoundError(
                    f"Raw FSQ terminator checkpoint not found: {external_skill_model}"
                )
        elif external_skill_model_variant == "checkpoint":
            _validate_external_terminator(
                external_skill_model,
                target_policy=contract["policy"],
            )
        else:
            _validate_display_model(
                {
                    "variant": external_skill_model_variant,
                    "path": str(external_skill_model),
                },
                target_policy=contract["policy"],
            )
    for display_model in display_terminator_models:
        _validate_display_model(
            display_model,
            target_policy=contract["policy"],
        )

    if not as_bool(get_value(config, "episode_exact", True)):
        raise ValueError(
            "terminator_eval requires episode_exact=true; no approximate fallback exists."
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
    eval_max_workers_per_gpu = int(
        get_value(config, "eval_max_workers_per_gpu", 4)
    )
    if not 1 <= eval_max_workers_per_gpu <= 4:
        raise ValueError("eval_max_workers_per_gpu must be between 1 and 4.")
    selected_episode_upper_bound = (
        len(episode_ids) if episode_ids else len(task_ids) * episodes_per_task
    )
    # Every selected episode contains at least one independent skill
    # occurrence, so this is a safe upper bound for logical workers before the
    # exact occurrence list is loaded on the compute node. The shared packing
    # planner may place several such workers on one physical GPU.
    eval_work_unit_count = selected_episode_upper_bound
    eval_num_gpus = min(requested_eval_gpus, eval_work_unit_count)

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
    fixed_max = _at(config, "terminator", "max_skill_length", default=None)
    scaled_max = _at(config, "terminator", "max_skill_length_scale", default=None)
    if fixed_max is not None and scaled_max is not None:
        raise ValueError(
            "terminator cannot set both max_skill_length and max_skill_length_scale."
        )
    if scaled_max is not None:
        max_skill_length = 1  # Imported Stage-1 wrapper requires an integer.
        max_skill_length_mode = "gt_scale"
        max_skill_length_scale = float(scaled_max)
        if max_skill_length_scale < 1.0:
            raise ValueError("terminator.max_skill_length_scale must be >= 1.0.")
    else:
        max_skill_length = int(200 if fixed_max is None else fixed_max)
        max_skill_length_mode = "fixed"
        max_skill_length_scale = 0.0
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
            "external_skill_model_variant": external_skill_model_variant,
            "terminator_models": display_terminator_models,
            **{key: value for key, value in contract.items() if key != "policy"},
        }.items()
    }
    eval_dir = _HERE.parent.parent
    settings = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "terminator_eval_dir": eval_dir,
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
        "terminator_models_json": json.dumps(
            display_terminator_models,
            separators=(",", ":"),
        ),
        "target_task": target_task,
        "task_ids": json.dumps(task_ids, separators=(",", ":")),
        "episode_ids": json.dumps(episode_ids, separators=(",", ":")),
        "episodes_per_task": episodes_per_task,
        "episode_selection": episode_selection,
        "episode_exact": True,
        "eval_num_gpus": eval_num_gpus,
        "eval_max_workers_per_gpu": eval_max_workers_per_gpu,
        "eval_work_unit_count": eval_work_unit_count,
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
        "skill_max_length_mode": max_skill_length_mode,
        "skill_max_length_scale": max_skill_length_scale,
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
