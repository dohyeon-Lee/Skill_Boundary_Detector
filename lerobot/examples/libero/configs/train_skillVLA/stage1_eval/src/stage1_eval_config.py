#!/usr/bin/env python3
"""Resolve renewed Stage-1 multi-checkpoint evaluation into shell exports."""

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

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage1_eval_config.yaml"

_PREDICTOR_CHECKPOINT_CONTRACT_FIELDS = (
    "skill_vocab_size",
    "skill_fsq_levels",
    "skill_predictor_vlm_variant",
    "skill_predictor_image_size",
    "skill_predictor_reader_tokens",
    "skill_predictor_reader_depth",
    "skill_predictor_reader_heads",
    "skill_predictor_all_layers",
    "skill_predictor_detach_vlm",
    "skill_predictor_lora",
    "skill_predictor_lora_targets",
    "skill_predictor_lora_rank",
    "skill_predictor_lora_alpha",
    "skill_predictor_lora_dropout",
    "skill_predictor_deadzone_frac",
    "skill_predictor_attend_image",
    "skill_predictor_attend_language",
    "tokenizer_max_length",
)


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
    anchors = (
        "dataset",
        "dataset_filtered",
        "dataset_ABC",
        "models",
        "outputs",
        "outputs_filtered",
    )
    for anchor in anchors:
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
        raise ValueError("Every Stage-1 eval model needs a non-empty label.")
    return value


def _default_output_name(models: list[dict]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(models) > 1:
        return f"compare_{len(models)}models_{stamp}"
    model = re.sub(r"[^A-Za-z0-9._-]+", "-", models[0]["model_dir"]).strip("-_")
    raw = f"{model}_{models[0]['checkpoint']}_{stamp}"
    return raw if len(raw) <= 200 else f"stage1_{models[0]['checkpoint']}_{stamp}"


def _visual_crossattn_query_label(*, include_state: bool, include_skill: bool) -> str:
    tokens = []
    if include_state:
        tokens.append("state")
    if include_skill:
        tokens.append("skill")
    return " + ".join((*tokens, "action")) if tokens else "action-only"


VISION_CONDITIONING_MODES = (
    "residual_cross_attention",
    "in_context_tokens",
    "global_visual_adarms",
)


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
            f"Incomplete Stage-1 checkpoint at {policy_path}: missing {missing}."
        )
    policy = json.loads((policy_path / "config.json").read_text())
    if policy.get("type", policy.get("model_type")) != "skill_expert":
        raise ValueError(f"Expected a skill_expert checkpoint: {policy_path}")

    architecture = str(policy.get("architecture", "")).strip().lower()
    if architecture != "vsa_perceiver_crossattn":
        raise ValueError(
            "This branch evaluates only architecture=vsa_perceiver_crossattn; "
            f"legacy Stage-1 checkpoints must be evaluated from their original branch: {policy_path}"
        )
    architecture_revision = str(policy.get("architecture_revision", "")).strip()
    if architecture_revision not in {"", "residual_sa18_v2"}:
        raise ValueError(
            "Unsupported VSA architecture_revision="
            f"{architecture_revision!r} at {policy_path}."
        )
    eval_legacy_vsa = not architecture_revision
    vision_conditioning_mode = str(
        policy.get("vision_conditioning_mode", "residual_cross_attention")
    ).strip().lower()
    if not eval_legacy_vsa and vision_conditioning_mode not in VISION_CONDITIONING_MODES:
        raise ValueError(
            "Unsupported vision_conditioning_mode="
            f"{vision_conditioning_mode!r} at {policy_path}."
        )
    num_visual_latents_per_camera = int(
        policy.get("num_visual_latents_per_camera", 8 if eval_legacy_vsa else 32)
    )
    action_loss_mode = str(policy.get("action_loss_mode", "")).strip().lower()
    if action_loss_mode not in {"flow", "flow_endpoint_xyz"}:
        raise ValueError(
            "Stage-1 checkpoint does not record a supported action objective "
            f"(flow|flow_endpoint_xyz): {policy_path}"
        )
    # These switches belong to the checkpoint architecture. Evaluation must not
    # override them from its own YAML. Checkpoints saved before either switch was
    # introduced used the action-only path, which is represented by false/false.
    include_state_in_visual_crossattn = as_bool(
        policy.get("include_state_in_visual_crossattn", False)
    )
    include_skill_in_visual_crossattn = as_bool(
        policy.get("include_skill_in_visual_crossattn", False)
    )
    visual_crossattn_queries = (
        _visual_crossattn_query_label(
            include_state=include_state_in_visual_crossattn,
            include_skill=include_skill_in_visual_crossattn,
        )
        if vision_conditioning_mode == "residual_cross_attention"
        else "ignored"
    )

    train_config = json.loads((policy_path / "train_config.json").read_text())
    dataset_value = str((train_config.get("dataset") or {}).get("root") or "").strip()
    if not dataset_value:
        raise ValueError(f"Stage-1 train_config has no dataset.root: {policy_path}")
    skill_dataset_dir = _relocate_project_path(project_root, dataset_value)
    if not (skill_dataset_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"Stage-1 SkillVLA dataset not found: {skill_dataset_dir}"
        )
    run_dir = skill_dataset_dir.parent
    source_dir = run_dir.parent
    if len(source_dir.parents) < 2:
        raise ValueError(f"Unexpected Stage-1 dataset layout: {skill_dataset_dir}")

    fsq_path = _relocate_project_path(project_root, policy.get("fsq_path"))
    has_terminator = as_bool(policy.get("train_terminator", False))
    has_predictor = as_bool(policy.get("train_skill_predictor", False)) or str(
        policy.get("training_skill_source", "gt")
    ).strip().lower() == "predictor"
    if has_terminator and not fsq_path.is_file():
        raise FileNotFoundError(f"FSQ checkpoint referenced by Stage 1 not found: {fsq_path}")
    paths = {
        "fsq_path": fsq_path,
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
    if not paths["dino_model_path"].is_dir():
        raise FileNotFoundError(
            f"Stage-1 model directory not found: {paths['dino_model_path']}"
        )
    if has_terminator and not paths["terminator_dino_model_path"].is_dir():
        raise FileNotFoundError(
            "Stage-1 terminator model directory not found: "
            f"{paths['terminator_dino_model_path']}"
        )
    if has_predictor and not paths["tokenizer_path"].is_dir():
        raise FileNotFoundError(f"Stage-1 tokenizer not found: {paths['tokenizer_path']}")
    return {
        "policy": policy,
        "architecture": architecture,
        "architecture_revision": architecture_revision or "legacy_alternating_v1",
        "eval_legacy_vsa": eval_legacy_vsa,
        "vision_conditioning_mode": (
            "legacy_alternating" if eval_legacy_vsa else vision_conditioning_mode
        ),
        "num_visual_latents_per_camera": num_visual_latents_per_camera,
        "include_state_in_visual_crossattn": include_state_in_visual_crossattn,
        "include_skill_in_visual_crossattn": include_skill_in_visual_crossattn,
        "visual_crossattn_queries": visual_crossattn_queries,
        "action_loss_mode": action_loss_mode,
        "has_predictor": has_predictor,
        "has_terminator": has_terminator,
        **paths,
    }


def _external_predictor_contract(
    checkpoint: Path,
    *,
    target_policy: dict,
    project_root: Path,
) -> dict:
    """Validate an eval-time predictor overlay against the target policy."""
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"External predictor config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"External predictor weights not found: {weights_path}")
    source = json.loads(config_path.read_text())
    if source.get("type") != "skill_expert":
        raise ValueError(
            "External predictor must come from policy.type=skill_expert, got "
            f"{source.get('type')!r} at {checkpoint}."
        )
    if not as_bool(source.get("train_skill_predictor", False)):
        raise ValueError(
            f"External predictor checkpoint has no trained predictor: {checkpoint}"
        )
    mismatches = [
        f"{field}: predictor={source.get(field)!r}, target={target_policy.get(field)!r}"
        for field in _PREDICTOR_CHECKPOINT_CONTRACT_FIELDS
        if source.get(field) != target_policy.get(field)
    ]
    if mismatches:
        raise ValueError(
            "External predictor module contract mismatch: " + "; ".join(mismatches)
        )
    tokenizer_path = _relocate_project_path(
        project_root, source.get("tokenizer_path")
    )
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"External predictor tokenizer not found: {tokenizer_path}"
        )
    return {"tokenizer_path": tokenizer_path}


def _external_terminator_contract(
    checkpoint: Path,
    *,
    target_policy: dict,
    project_root: Path,
) -> dict:
    """Validate and relocate an eval-time co-trained terminator source."""
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"External terminator config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"External terminator weights not found: {weights_path}")
    source = json.loads(config_path.read_text())
    if source.get("type") != "skill_expert":
        raise ValueError(
            "External terminator must come from policy.type=skill_expert, got "
            f"{source.get('type')!r} at {checkpoint}."
        )
    if not as_bool(source.get("train_terminator", False)):
        raise ValueError(
            f"External terminator checkpoint has no trained terminator: {checkpoint}"
        )
    if source.get("skill_fsq_levels") != target_policy.get("skill_fsq_levels"):
        raise ValueError(
            "External terminator FSQ mismatch: "
            f"terminator={source.get('skill_fsq_levels')!r}, "
            f"target={target_policy.get('skill_fsq_levels')!r}"
        )
    dino_path = _relocate_project_path(
        project_root,
        source.get("terminator_dino_model_path") or source.get("dino_model_path"),
    )
    if not dino_path.is_dir():
        raise FileNotFoundError(f"External terminator DINO model not found: {dino_path}")
    return {"terminator_dino_model_path": dino_path}


def _model_entries(config: dict) -> list[dict]:
    default_checkpoint = str(get_value(config, "checkpoint", "last"))
    default_skill_source = str(get_value(config, "skill_source", "gt")).lower()
    default_advance = str(
        _at(config, "oracle", "advance_mode", default="own")
    ).lower()
    models = get_value(config, "models", None)
    if isinstance(models, list) and models:
        raw_entries = models
    else:
        model_dir = str(get_value(config, "model_dir", "") or "")
        if not model_dir:
            raise ValueError("Set models[] or a top-level model_dir in Stage-1 eval config.")
        raw_entries = [{"model_dir": model_dir}]

    entries = []
    for index, raw in enumerate(raw_entries):
        obsolete = [
            field
            for field in ("predictor_checkpoint", "terminator_checkpoint")
            if str(raw.get(field, "") or "").strip()
        ]
        if obsolete:
            raise ValueError(
                f"models[] fields {obsolete} were replaced by top-level "
                "external_skill_model; select them with skill_source=external "
                "and/or advance_mode=external."
            )
        model_dir = _safe_name(str(raw.get("model_dir", "")), field="models[].model_dir")
        checkpoint = _safe_name(
            str(raw.get("checkpoint", default_checkpoint)),
            field="models[].checkpoint",
        )
        skill_source = str(raw.get("skill_source", default_skill_source)).lower()
        aliases = {
            "gt": "gt",
            "oracle": "gt",
            "own": "own",
            "pred": "own",
            "predicted": "own",
            "predictor": "own",
            "external": "external",
        }
        skill_source = aliases.get(skill_source, "")
        if not skill_source:
            raise ValueError("models[].skill_source must be external|own|gt.")
        advance_mode = str(raw.get("advance_mode", default_advance)).lower()
        advance_aliases = {
            "gt": "gt",
            "own": "own",
            "terminator": "own",
            "external": "external",
        }
        advance_mode = advance_aliases.get(advance_mode, "")
        if not advance_mode:
            raise ValueError("models[].advance_mode must be external|own|gt.")
        label = str(raw.get("label", "") or "").strip()
        if not label:
            label = f"model{index + 1}-{skill_source}"
        entries.append(
            {
                "model_dir": model_dir,
                "checkpoint": checkpoint,
                "skill_source": skill_source,
                "advance_mode": advance_mode,
                "label": _clean_label(label),
            }
        )
    labels = [entry["label"] for entry in entries]
    if len(labels) != len(set(labels)):
        raise ValueError(f"models[].label values must be unique, got {labels}.")
    return entries


def build_settings(config: dict) -> dict:
    project_root = Path(str(get_value(config, "project_root"))).expanduser()
    eval_outputs_root = _HERE.parent.parent / "outputs"
    outputs_root = project_root / str(get_value(config, "outputs_root", "outputs"))
    obsolete = [
        field
        for field in ("predictor_checkpoint", "terminator_checkpoint")
        if str(get_value(config, field, "") or "").strip()
    ]
    if obsolete:
        raise ValueError(
            f"Top-level fields {obsolete} were replaced by external_skill_model."
        )
    external_skill_model_value = str(
        get_value(config, "external_skill_model", "") or ""
    ).strip()
    external_skill_model = (
        _relocate_project_path(project_root, external_skill_model_value)
        if external_skill_model_value
        else None
    )
    entries = _model_entries(config)
    resolved = []
    for entry in entries:
        policy_path = (
            outputs_root
            / "skillVLA_stage1"
            / entry["model_dir"]
            / "checkpoints"
            / entry["checkpoint"]
            / "pretrained_model"
        )
        contract = _checkpoint_contract(policy_path, project_root)
        if entry["skill_source"] == "own" and not contract["has_predictor"]:
            raise ValueError(
                f"skill_source=own but checkpoint has no trained predictor: {policy_path}"
            )
        tokenizer_path = contract["tokenizer_path"]
        if entry["skill_source"] == "external":
            if external_skill_model is None:
                raise ValueError(
                    "skill_source=external requires top-level external_skill_model."
                )
            external = _external_predictor_contract(
                external_skill_model,
                target_policy=contract["policy"],
                project_root=project_root,
            )
            tokenizer_path = external["tokenizer_path"]
        terminator_dino_model_path = contract["terminator_dino_model_path"]
        if entry["advance_mode"] == "external":
            if external_skill_model is None:
                raise ValueError(
                    "advance_mode=external requires top-level external_skill_model."
                )
            external_terminator = _external_terminator_contract(
                external_skill_model,
                target_policy=contract["policy"],
                project_root=project_root,
            )
            terminator_dino_model_path = external_terminator[
                "terminator_dino_model_path"
            ]
        if entry["advance_mode"] == "own" and not contract["has_terminator"]:
            raise ValueError(
                "advance_mode=own but checkpoint has no trained terminator: "
                f"{policy_path}"
            )
        resolved.append(
            {
                **entry,
                "policy_path": policy_path,
                **contract,
                "external_skill_model": external_skill_model or "",
                "tokenizer_path": tokenizer_path,
                "terminator_dino_model_path": terminator_dino_model_path,
            }
        )

    episode_exact = as_bool(_at(config, "oracle", "episode_exact", default=False))
    if episode_exact:
        init_state_paths = {
            model["eval_init_states_path"].resolve() for model in resolved
        }
        if len(init_state_paths) != 1:
            raise ValueError(
                "Multi-model episode-exact comparison requires every checkpoint "
                "to use the same source dataset/init-state map."
            )
        for model in resolved:
            if not model["eval_init_states_path"].is_file():
                source = model["skill_dataset_dir"].parents[1].name
                raise FileNotFoundError(
                    f"oracle.episode_exact=true requires {model['eval_init_states_path']}. "
                    f"Build it with stage1_eval/oracle_matching/run.sh {source}."
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
                f"n_action_steps={n_action_steps} exceeds {model['label']}'s chunk_size={chunk_size}."
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
        "eval_resume": as_bool(get_value(config, "resume", False)),
        "policy_path": primary["policy_path"],
        "fsq_path": primary["fsq_path"],
        "skill_dataset_dir": primary["skill_dataset_dir"],
        "eval_init_states_path": (
            primary["eval_init_states_path"] if episode_exact else ""
        ),
        "skill_latents_path": primary["skill_latents_path"],
        "raw_dataset_dir": primary["raw_dataset_dir"],
        "dino_model_path": primary["dino_model_path"],
        "terminator_dino_model_path": primary["terminator_dino_model_path"],
        "tokenizer_path": primary["tokenizer_path"],
        "architecture": primary["architecture"],
        "vision_conditioning_mode": primary["vision_conditioning_mode"],
        "include_state_in_visual_crossattn": primary[
            "include_state_in_visual_crossattn"
        ],
        "include_skill_in_visual_crossattn": primary[
            "include_skill_in_visual_crossattn"
        ],
        "visual_crossattn_queries": primary["visual_crossattn_queries"],
        "action_loss_mode": primary["action_loss_mode"],
        "eval_out_dir": eval_outputs_root / output_name,
        "target_task": str(get_value(config, "target_task", "libero_goal")),
        "task_ids": json.dumps(task_ids, separators=(",", ":")),
        "eval_num_gpus": int(get_value(config, "eval_num_gpus", 1)),
        "n_episodes": int(get_value(config, "n_episodes", 3)),
        "eval_batch_size": int(get_value(config, "eval_batch_size", 1)),
        "max_parallel_tasks": int(get_value(config, "max_parallel_tasks", 1)),
        "n_action_steps": n_action_steps,
        "advance_mode": primary["advance_mode"],
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
            _at(config, "logging", "wandb", "project", default="VLA_stage1_eval")
        ),
        "wandb_run_name": f"S1eval_{output_name}"
        + (f"_{os.environ['TASK_TAG']}" if os.environ.get("TASK_TAG") else ""),
    }
    if settings["n_episodes"] <= 0 or settings["eval_batch_size"] <= 0:
        raise ValueError("n_episodes and eval_batch_size must be positive.")
    if settings["max_parallel_tasks"] != 1:
        raise ValueError("Stage-1 policies are stateful; max_parallel_tasks must remain 1.")
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
