#!/usr/bin/env python3
"""Resolve the multi-policy, single-terminator Stage-1 skill evaluation config."""

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
    _model_entries,
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


def _resolve_end_rule(
    raw: dict,
    *,
    field: str,
    default_mode: str,
    default_end_threshold: float,
    default_progress_threshold: float,
) -> dict[str, str | float]:
    end_mode = str(raw.get("end_mode", default_mode)).strip().lower()
    if end_mode not in {"termination", "progress", "or", "and"}:
        raise ValueError(f"{field}.end_mode must be termination|progress|or|and.")
    end_threshold = float(raw.get("end_threshold", default_end_threshold))
    progress_threshold = float(
        raw.get("progress_threshold", default_progress_threshold)
    )
    for name, value in (
        ("end_threshold", end_threshold),
        ("progress_threshold", progress_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field}.{name} must be in [0, 1], got {value}.")
    return {
        "end_mode": end_mode,
        "end_threshold": end_threshold,
        "progress_threshold": progress_threshold,
    }


def _resolve_terminator_model(
    config: dict,
    *,
    project_root: Path,
    outputs_root: Path,
) -> tuple[Path, str, str, dict[str, str | float]]:
    raw = get_value(config, "terminator_model", None)
    if raw is None:
        # Keep old configs readable while making the new singular role explicit.
        raw = get_value(config, "external_skill_model", None)
    if raw is None or raw == "":
        raise ValueError("Set the shared top-level terminator_model.")
    if not isinstance(raw, dict):
        path = _relocate_project_path(project_root, str(raw).strip())
        return path, "state_image", "shared-terminator", _resolve_end_rule(
            {},
            field="terminator_model",
            default_mode="termination",
            default_end_threshold=0.5,
            default_progress_threshold=0.95,
        )

    unknown = sorted(
        set(raw)
        - {
            "label",
            "variant",
            "path",
            "group",
            "model_dir",
            "checkpoint",
            "end_mode",
            "end_threshold",
            "progress_threshold",
        }
    )
    if unknown:
        raise ValueError(
            "terminator_model supports checkpoint selection plus its own "
            "end_mode/end_threshold/progress_threshold; "
            f"unknown={unknown}."
        )
    aliases = {
        "normal": "state_image",
        "state_image": "state_image",
        "state+image": "state_image",
        "image": "image_only",
        "image_only": "image_only",
        "image-only": "image_only",
    }
    variant = aliases.get(str(raw.get("variant", "state_image")).strip().lower())
    if variant is None:
        raise ValueError("terminator_model.variant must be state_image|image_only.")
    label = _clean_label(str(raw.get("label", "shared-terminator")))
    path_value = str(raw.get("path", "") or "").strip()
    checkpoint_fields = any(raw.get(field) for field in ("group", "model_dir", "checkpoint"))
    if path_value and checkpoint_fields:
        raise ValueError(
            "terminator_model cannot mix path with group/model_dir/checkpoint."
        )
    if path_value:
        path = _relocate_project_path(project_root, path_value)
    else:
        group = _safe_name(
            str(raw.get("group", "skillVLA_stage1")), field="terminator_model.group"
        )
        model_dir = _safe_name(
            str(raw.get("model_dir", "")), field="terminator_model.model_dir"
        )
        checkpoint = _safe_name(
            str(raw.get("checkpoint", "")), field="terminator_model.checkpoint"
        )
        path = outputs_root / group / model_dir / "checkpoints" / checkpoint / "pretrained_model"
    end_rule = _resolve_end_rule(
        raw,
        field="terminator_model",
        default_mode="termination",
        default_end_threshold=0.5,
        default_progress_threshold=0.95,
    )
    return path, variant, label, end_rule


def _resolve_main_terminator(
    config: dict,
    *,
    terminator_variant: str = "state_image",
    terminator_label: str = "shared-terminator",
    terminator_end_rule: dict[str, str | float] | None = None,
) -> dict[str, object]:
    raw = get_value(config, "main_terminator", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("main_terminator must be a YAML mapping.")
    legacy = get_value(config, "terminator", {}) or {}
    if not isinstance(legacy, dict):
        raise ValueError("legacy terminator must be a YAML mapping.")
    raw = dict(raw)
    for key in (
        "end_mode",
        "end_threshold",
        "progress_threshold",
        "max_skill_length",
        "max_skill_length_scale",
        "finish_action_chunk_on_end",
    ):
        if key not in raw and key in legacy:
            raw[key] = legacy[key]
    unknown = sorted(
        set(raw)
        - {
            "label",
            "variant",
            "end_mode",
            "end_threshold",
            "progress_threshold",
            "max_skill_length",
            "max_skill_length_scale",
            "finish_action_chunk_on_end",
        }
    )
    if unknown:
        raise ValueError(
            "main_terminator supports a variant plus stopping-rule fields; "
            f"unknown={unknown}."
        )
    # The short form (only max length / chunk behavior) means that MAIN reuses
    # terminator_model, including its end rule.
    requested_variant = str(raw.get("variant", "terminator_model")).strip().lower()
    fsq_aliases = {"fsq", "fsq_base", "fsq_initial"}
    shared_aliases = {"terminator_model", "shared", "display"}
    trained_aliases = {
        "normal": "state_image",
        "state_image": "state_image",
        "state+image": "state_image",
        "image": "image_only",
        "image_only": "image_only",
        "image-only": "image_only",
    }
    if requested_variant in fsq_aliases:
        variant = "fsq_initial"
        default_label = "FSQ_INIT"
    else:
        variant = (
            terminator_variant
            if requested_variant in shared_aliases
            else trained_aliases.get(requested_variant)
        )
        if variant is None:
            raise ValueError(
                "main_terminator.variant must be fsq_initial|terminator_model|"
                "state_image|image_only."
            )
        if variant != terminator_variant:
            raise ValueError(
                "A trained MAIN shares terminator_model's checkpoint, so "
                "main_terminator.variant must be terminator_model or match "
                f"terminator_model.variant={terminator_variant!r}; got {variant!r}."
            )
        default_label = f"MAIN_{terminator_label}"
    shared_end_rule = terminator_end_rule or {
        "end_mode": "termination",
        "end_threshold": 0.5,
        "progress_threshold": 0.95,
    }
    inherits_shared_rule = variant != "fsq_initial"
    end_rule = _resolve_end_rule(
        raw,
        field="main_terminator",
        default_mode=(
            str(shared_end_rule["end_mode"])
            if inherits_shared_rule
            else "or"
        ),
        default_end_threshold=(
            float(shared_end_rule["end_threshold"])
            if inherits_shared_rule
            else 0.5
        ),
        default_progress_threshold=(
            float(shared_end_rule["progress_threshold"])
            if inherits_shared_rule
            else 0.95
        ),
    )
    has_fixed_max = raw.get("max_skill_length") is not None
    has_scaled_max = raw.get("max_skill_length_scale") is not None
    if has_fixed_max and has_scaled_max:
        raise ValueError(
            "main_terminator cannot set both max_skill_length and "
            "max_skill_length_scale."
        )
    if has_scaled_max:
        max_skill_length = None
        max_skill_length_scale = float(raw["max_skill_length_scale"])
        if max_skill_length_scale < 1.0:
            raise ValueError(
                "main_terminator.max_skill_length_scale must be >= 1.0."
            )
    else:
        max_skill_length = int(raw.get("max_skill_length", 200))
        if max_skill_length <= 0:
            raise ValueError("main_terminator.max_skill_length must be positive.")
        max_skill_length_scale = None
    return {
        "variant": variant,
        "label": _clean_label(str(raw.get("label", default_label))),
        "shares_terminator_model": inherits_shared_rule,
        **end_rule,
        "max_skill_length": max_skill_length,
        "max_skill_length_scale": max_skill_length_scale,
        "finish_action_chunk_on_end": as_bool(
            raw.get("finish_action_chunk_on_end", True)
        ),
    }


def build_settings(config: dict) -> dict:
    project_root = Path(str(get_value(config, "project_root", _PROJECT_ROOT_DEFAULT))).expanduser()
    outputs_root = project_root / str(get_value(config, "outputs_root", "outputs"))
    (
        terminator_path,
        terminator_variant,
        terminator_label,
        terminator_end_rule,
    ) = _resolve_terminator_model(
        config,
        project_root=project_root,
        outputs_root=outputs_root,
    )
    main_terminator = _resolve_main_terminator(
        config,
        terminator_variant=terminator_variant,
        terminator_label=terminator_label,
        terminator_end_rule=terminator_end_rule,
    )

    model_config = dict(config)
    defaults = dict(get_value(config, "model_defaults", {}) or {})
    defaults.setdefault("skill_source", "gt")
    defaults["advance_mode"] = "external"
    defaults["terminator_variant"] = terminator_variant
    model_config["model_defaults"] = defaults
    entries = _model_entries(model_config)
    if any(entry["skill_source"] != "gt" for entry in entries):
        raise ValueError("stage1_skill_eval models always use skill_source=gt.")
    if any(entry["advance_mode"] != "external" for entry in entries):
        raise ValueError(
            "stage1_skill_eval uses the one shared terminator_model for every policy; "
            "models[].advance_mode must be external."
        )
    if any(entry["terminator_variant"] != terminator_variant for entry in entries):
        raise ValueError(
            "models[].terminator_variant cannot override shared terminator_model.variant."
        )

    resolved = []
    for entry in entries:
        model_root = outputs_root / "skillVLA_stage1"
        if entry["previous_checkpoint"]:
            model_root = model_root / "previous"
        policy_path = (
            model_root
            / entry["model_dir"]
            / "checkpoints"
            / entry["checkpoint"]
            / "pretrained_model"
        )
        contract = _checkpoint_contract(policy_path, project_root)
        _validate_external_terminator(
            terminator_path,
            target_policy=contract["policy"],
            variant=terminator_variant,
        )
        resolved.append(
            {
                **entry,
                "policy_path": policy_path,
                "skill_source": "gt",
                "advance_mode": "external",
                "terminator_variant": terminator_variant,
                # MAIN can use either the pristine FSQ terminator or the same
                # trained checkpoint as the display terminator. Their stopping
                # rules remain independent in both cases.
                "external_skill_model": (
                    contract["fsq_path"]
                    if main_terminator["variant"] == "fsq_initial"
                    else terminator_path
                ),
                "external_skill_model_variant": main_terminator["variant"],
                "terminator_models": [
                    {
                        "label": terminator_label,
                        "variant": terminator_variant,
                        "path": str(terminator_path),
                        **terminator_end_rule,
                    }
                ],
                **contract,
            }
        )

    primary = resolved[0]
    shared_artifact_fields = (
        "skill_dataset_dir",
        "skill_latents_path",
        "eval_init_states_path",
        "fsq_path",
    )
    for field in shared_artifact_fields:
        paths = {Path(model[field]).resolve() for model in resolved}
        if len(paths) != 1:
            raise ValueError(
                "Multi-policy skill comparison requires every policy to share "
                f"the same {field}; got {sorted(map(str, paths))}."
            )

    if not as_bool(get_value(config, "episode_exact", True)):
        raise ValueError(
            "stage1_skill_eval requires episode_exact=true; no approximate fallback exists."
        )
    if not primary["eval_init_states_path"].is_file():
        raise FileNotFoundError(
            "Episode-exact map not found: "
            f"{primary['eval_init_states_path']}. Build it with stage1_eval/oracle_matching."
        )
    if not primary["skill_latents_path"].is_file():
        raise FileNotFoundError(f"Skill occurrence metadata not found: {primary['skill_latents_path']}")

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
    # Rollouts are independent for every policy x episode pair.  Cap the
    # Slurm array at that many useful workers instead of at the episode count.
    eval_num_gpus = min(
        requested_eval_gpus,
        len(resolved) * selected_episode_upper_bound,
    )

    time_shift_offset = int(get_value(config, "time_shift_offset", 15))
    if time_shift_offset <= 0:
        raise ValueError("time_shift_offset must be a positive integer.")
    n_action_steps = int(get_value(config, "n_action_steps", 5))
    for model in resolved:
        chunk_size = int(model["policy"].get("chunk_size", n_action_steps))
        if not 1 <= n_action_steps <= chunk_size:
            raise ValueError(
                f"n_action_steps={n_action_steps} must be in [1, "
                f"{model['label']}'s chunk_size={chunk_size}]."
            )

    original_dataset_dir = _resolve_path(
        project_root,
        get_value(config, "original_dataset_dir", f"libero_original_dataset/{target_task}"),
    )
    if not original_dataset_dir.is_dir():
        raise FileNotFoundError(f"Original LIBERO HDF5 folder not found: {original_dataset_dir}")

    output_name = str(get_value(config, "output_name", "") or "").strip()
    if not output_name:
        output_name = (
            f"compare_{len(resolved)}policies_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    output_name = _safe_name(output_name, field="output_name")

    model_specs = [
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in model.items()
            if key != "policy"
        }
        for model in resolved
    ]
    eval_dir = _HERE.parent.parent
    settings = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "stage1_skill_eval_dir": eval_dir,
        "models_json": json.dumps(model_specs, separators=(",", ":")),
        # The CLI needs one concrete policy config for argument construction;
        # each runtime worker loads only its assigned MODELS_JSON policies.
        "policy_path": primary["policy_path"],
        "fsq_path": primary["fsq_path"],
        "dino_model_path": primary["dino_model_path"],
        "tokenizer_path": primary["tokenizer_path"],
        "skill_dataset_dir": primary["skill_dataset_dir"],
        "skill_latents_path": primary["skill_latents_path"],
        "eval_init_states_path": primary["eval_init_states_path"],
        "original_dataset_dir": original_dataset_dir,
        "architecture_label": ",".join(model["architecture_label"] for model in resolved),
        "model_label": f"{len(resolved)} policies",
        "model_count": len(resolved),
        "terminator_model_path": terminator_path,
        "terminator_model_label": terminator_label,
        "terminator_model_variant": terminator_variant,
        "terminator_model_end_mode": terminator_end_rule["end_mode"],
        "terminator_model_end_threshold": terminator_end_rule["end_threshold"],
        "terminator_model_progress_threshold": terminator_end_rule[
            "progress_threshold"
        ],
        "main_terminator_label": main_terminator["label"],
        "main_terminator_variant": main_terminator["variant"],
        "main_terminator_path": primary["external_skill_model"],
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
        "skill_end_mode": main_terminator["end_mode"],
        "skill_end_threshold": main_terminator["end_threshold"],
        "skill_end_progress_threshold": main_terminator["progress_threshold"],
        # The imported Stage1 wrapper requires an integer at construction, but
        # stage1_skill_eval applies its own per-occurrence limit below.
        "inference_skill_max_length": (
            main_terminator["max_skill_length"]
            if main_terminator["max_skill_length"] is not None
            else 1
        ),
        "skill_max_length_mode": (
            "gt_scale"
            if main_terminator["max_skill_length_scale"] is not None
            else "fixed"
        ),
        "skill_max_length_scale": (
            main_terminator["max_skill_length_scale"]
            if main_terminator["max_skill_length_scale"] is not None
            else 0.0
        ),
        "finish_action_chunk_on_end": main_terminator[
            "finish_action_chunk_on_end"
        ],
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
