#!/usr/bin/env python3
"""Resolve renewed Stage-1 multi-checkpoint evaluation into shell exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from functools import lru_cache
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


def _langgap_env_task_ids(
    exact_map_path: Path,
    dataset_task_ids: list[int],
    *,
    suite_name: str,
) -> list[int]:
    """Map compact LangGap dataset task IDs to sparse benchmark task IDs."""
    diagnostics_path = exact_map_path.with_suffix(".diagnostics.json")
    if not diagnostics_path.is_file():
        raise FileNotFoundError(
            "LangGap exact-map diagnostics not found: "
            f"{diagnostics_path}. Rebuild the map with oracle_matching so the "
            "dataset-task to simulator-task provenance is available."
        )
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    matched = payload.get("matched")
    if not isinstance(matched, list) or not matched:
        raise ValueError(
            f"LangGap exact-map diagnostics has no matched episodes: {diagnostics_path}"
        )
    mapping: dict[int, int] = {}
    for row in matched:
        if not isinstance(row, dict) or str(row.get("suite_name", "")) != suite_name:
            continue
        dataset_task_id = int(row["dataset_task_id"])
        suite_task_id = int(row["suite_task_id"])
        previous = mapping.setdefault(dataset_task_id, suite_task_id)
        if previous != suite_task_id:
            raise ValueError(
                "One LangGap dataset task maps to multiple simulator tasks: "
                f"dataset task {dataset_task_id} -> {previous}, {suite_task_id}."
            )
    missing = sorted(set(dataset_task_ids) - set(mapping))
    if missing:
        raise ValueError(
            f"LangGap exact map has no simulator-task mapping for dataset tasks {missing}."
        )
    return sorted({mapping[task_id] for task_id in dataset_task_ids})


def _resolve_run_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    """Return a run's pretrained_model directory, resolving ``last`` safely."""
    checkpoint_name = _safe_name(str(checkpoint), field="checkpoint")
    checkpoints_dir = run_dir / "checkpoints"
    direct = checkpoints_dir / checkpoint_name / "pretrained_model"
    if checkpoint_name.lower() != "last":
        return direct
    if (direct / "config.json").is_file():
        # Canonicalize an existing ``last`` symlink so the resolved settings
        # record the concrete checkpoint that was actually evaluated.
        return direct.resolve()

    # Training outputs normally use zero-padded numeric checkpoint folders and
    # do not create a ``last`` symlink.  Select the greatest numeric checkpoint
    # only after the caller has identified the exact run directory.
    candidates: list[tuple[int, Path]] = []
    if checkpoints_dir.is_dir():
        for child in checkpoints_dir.iterdir():
            pretrained = child / "pretrained_model"
            if child.is_dir() and child.name.isdigit() and (
                pretrained / "config.json"
            ).is_file():
                candidates.append((int(child.name), pretrained))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return direct


def _resolve_external_terminator_path(
    project_root: Path,
    outputs_root: Path,
    run_or_path: str,
    checkpoint: str,
) -> Path:
    """Resolve a concise skillVLA_terminator run name or a legacy full path."""
    raw = str(run_or_path or "").strip()
    path = Path(raw).expanduser()
    if raw and not path.is_absolute() and len(path.parts) == 1:
        run_name = _safe_name(raw, field="external_terminator_model")
        checkpoint_name = _safe_name(
            str(checkpoint), field="external_terminator_checkpoint"
        )
        resolved = _resolve_run_checkpoint(
            outputs_root / "skillVLA_terminator" / run_name,
            checkpoint_name,
        )
        if (resolved / "config.json").is_file():
            return resolved
        for archive_name in ("PREV", "previous"):
            archived = _resolve_run_checkpoint(
                outputs_root / "skillVLA_terminator" / archive_name / run_name,
                checkpoint_name,
            )
            if (archived / "config.json").is_file():
                return archived
        return resolved
    return _relocate_project_path(project_root, raw)


def _resolve_external_predictor_path(
    project_root: Path,
    outputs_root: Path,
    run_or_path: str,
    checkpoint: str,
) -> Path:
    """Resolve a concise skillVLA_terminator predictor run name or full path."""
    raw = str(run_or_path or "").strip()
    path = Path(raw).expanduser()
    if raw and not path.is_absolute() and len(path.parts) == 1:
        run_name = _safe_name(raw, field="external_predictor_model")
        checkpoint_name = _safe_name(
            str(checkpoint), field="external_predictor_checkpoint"
        )
        resolved = _resolve_run_checkpoint(
            outputs_root / "skillVLA_terminator" / run_name,
            checkpoint_name,
        )
        if (resolved / "config.json").is_file():
            return resolved
        for archive_name in ("PREV", "previous"):
            archived = _resolve_run_checkpoint(
                outputs_root / "skillVLA_terminator" / archive_name / run_name,
                checkpoint_name,
            )
            if (archived / "config.json").is_file():
                return archived
        return resolved
    return _relocate_project_path(project_root, raw)


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


def _default_output_name(
    models: list[dict], *, model_count: int, checkpoint_count: int
) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_count > 1 or checkpoint_count > 1:
        return (
            f"compare_{model_count}models_"
            f"{checkpoint_count}checkpoints_{stamp}"
        )
    model = re.sub(r"[^A-Za-z0-9._-]+", "-", models[0]["model_dir"]).strip("-_")
    raw = f"{model}_{models[0]['checkpoint']}_{stamp}"
    return raw if len(raw) <= 200 else f"stage1_{models[0]['checkpoint']}_{stamp}"


def _checkpoint_list(value: object, *, field: str) -> list[str]:
    """Normalize a scalar/list checkpoint setting while preserving its order."""
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError(f"{field} must contain at least one checkpoint.")
    checkpoints = [
        _safe_name(str(checkpoint), field=field) for checkpoint in values
    ]
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError(f"{field} contains duplicate checkpoints: {checkpoints}.")
    return checkpoints


def _visual_crossattn_query_label(*, include_state: bool, include_skill: bool) -> str:
    tokens = []
    if include_state:
        tokens.append("state")
    if include_skill:
        tokens.append("skill")
    return " + ".join((*tokens, "action")) if tokens else "action-only"


VISION_CONDITIONING_MODES = (
    "uncompressed_visual_kv_self_attention",
    "compressed_visual_kv_self_attention",
    "interleaved_cross_attention",
    "in_context_tokens",
    "global_visual_adarms",
)
VSA_ARCHITECTURE = "vsa_perceiver_crossattn"
VSA_ARCHITECTURE_REVISION = "interleaved_direct1024_v3"
UNCOMPRESSED_VISUAL_KV_REVISION = "visual_kv_uncompressed_v1"
COMPRESSED_VISUAL_KV_REVISION = "visual_kv_perceiver_v1"
LEGACY_RESIDUAL_VSA_REVISION = "residual_sa18_v2"
COND_GEMMA_ARCHITECTURE = "cond_gemma"
COND_GEMMA_ARCHITECTURE_REVISION = "skillvla_real_v1"
COND_GEMMA_ARCHITECTURE_LABELS = {
    COND_GEMMA_ARCHITECTURE_REVISION: "arch0",
    "expert_state_adarms_v1": "arch0_1",
    "cond_expert_state_adarms_v1": "arch0_2",
    "cond_expert_separate_state_adarms_v1": "arch0_2_sep",
    "wrist_cond_expert_state_adarms_v1": "arch0_3",
    "expert_skill_adarms_v1": "arch0_adarms",
    "expert_skill_adarms_zero_v1": "arch0_adarms_zero",
    "expert_skill_token_v1": "arch0_token",
    "expert_skill_token_isolated_v1": "arch0_token_iso",
    "cond_skill_broadcast_v1": "arch0_cond",
    "dual_skill_broadcast_v1": "arch0_both",
    "expert_tokens_uncompressed_v1": "arch1_1",
    "expert_tokens_perceiver_v1": "arch1_2",
}
VSA_ARCHITECTURE_LABELS = {
    "uncompressed_visual_kv_self_attention": "arch1_3",
    "compressed_visual_kv_self_attention": "arch2_1",
    "interleaved_cross_attention": "arch2_2",
    "in_context_tokens": "arch3",
    "global_visual_adarms": "arch4",
}
VSA_REVISION_MODE_LABELS = {
    UNCOMPRESSED_VISUAL_KV_REVISION: {
        "uncompressed_visual_kv_self_attention": "arch1_3"
    },
    COMPRESSED_VISUAL_KV_REVISION: {
        "compressed_visual_kv_self_attention": "arch2_1"
    },
    VSA_ARCHITECTURE_REVISION: {
        "interleaved_cross_attention": "arch2_2",
        "in_context_tokens": "arch3",
        "global_visual_adarms": "arch4",
    },
}
LEGACY_RESIDUAL_VSA_LABELS = {
    "residual_cross_attention": "arch2_2",
    "in_context_tokens": "arch3",
    "global_visual_adarms": "arch4",
}
CONDITIONING_ROUTES = frozenset(
    {
        "state_cond",
        "state_skill_cond",
        "state_skill_only_cond",
        "stateonly_cond",
        "skillonly_cond",
        "visiononly_cond",
    }
)


def _normalize_conditioning_route(value: object) -> str:
    route = str(value or "state_cond").strip().lower()
    return "skillonly_cond" if route == "skill_cond" else route


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

    saved_architecture = str(policy.get("architecture", "")).strip().lower()
    architecture_inferred = not saved_architecture and "conditioning_route" in policy
    architecture = (
        COND_GEMMA_ARCHITECTURE if architecture_inferred else saved_architecture
    )
    if architecture not in {VSA_ARCHITECTURE, COND_GEMMA_ARCHITECTURE}:
        raise ValueError(
            "Unsupported Stage-1 checkpoint architecture="
            f"{saved_architecture or '<missing>'!r} at {policy_path}. "
            "A missing architecture is accepted only for skillVLA_real checkpoints "
            "that record conditioning_route."
        )

    architecture_revision = str(policy.get("architecture_revision", "")).strip()
    if architecture == VSA_ARCHITECTURE:
        if architecture_revision not in {
            "",
            LEGACY_RESIDUAL_VSA_REVISION,
            VSA_ARCHITECTURE_REVISION,
            UNCOMPRESSED_VISUAL_KV_REVISION,
            COMPRESSED_VISUAL_KV_REVISION,
        }:
            raise ValueError(
                "Unsupported VSA architecture_revision="
                f"{architecture_revision!r} at {policy_path}."
            )
        if not architecture_revision:
            eval_vsa_revision = "legacy_alternating_v1"
            resolved_architecture_revision = eval_vsa_revision
            vision_conditioning_mode = "legacy_alternating"
            architecture_label = "arch2_1"
        elif architecture_revision == LEGACY_RESIDUAL_VSA_REVISION:
            eval_vsa_revision = LEGACY_RESIDUAL_VSA_REVISION
            resolved_architecture_revision = architecture_revision
            vision_conditioning_mode = str(
                policy.get("vision_conditioning_mode", "residual_cross_attention")
            ).strip().lower()
            if vision_conditioning_mode not in LEGACY_RESIDUAL_VSA_LABELS:
                raise ValueError(
                    "Unsupported historical vision_conditioning_mode="
                    f"{vision_conditioning_mode!r} at {policy_path}."
                )
            architecture_label = LEGACY_RESIDUAL_VSA_LABELS[
                vision_conditioning_mode
            ]
        else:
            eval_vsa_revision = ""
            resolved_architecture_revision = architecture_revision
            vision_conditioning_mode = str(
                policy.get(
                    "vision_conditioning_mode", "interleaved_cross_attention"
                )
            ).strip().lower()
            revision_modes = VSA_REVISION_MODE_LABELS[architecture_revision]
            if vision_conditioning_mode not in revision_modes:
                raise ValueError(
                    "Unsupported vision_conditioning_mode="
                    f"{vision_conditioning_mode!r} for revision "
                    f"{architecture_revision!r} at {policy_path}."
                )
            architecture_label = revision_modes[
                vision_conditioning_mode
            ]
        eval_legacy_vsa = bool(eval_vsa_revision)
        if architecture_revision == UNCOMPRESSED_VISUAL_KV_REVISION:
            num_visual_latents_per_camera = 197
        else:
            num_visual_latents_per_camera = int(
                policy.get(
                    "num_visual_latents_per_camera",
                    8 if eval_vsa_revision == "legacy_alternating_v1" else 32,
                )
            )
        # Previous VSA checkpoints predate this metadata field. Their resampler
        # tensors unambiguously use the historical 384-wide implementation.
        visual_perceiver_width = int(
            policy.get("visual_perceiver_width", 384 if eval_legacy_vsa else 1024)
        )
        if visual_perceiver_width <= 0:
            raise ValueError(
                f"Invalid visual_perceiver_width={visual_perceiver_width} at {policy_path}."
            )
        conditioning_route = ""
        if not eval_legacy_vsa and visual_perceiver_width != 1024:
            raise ValueError(
                "Current Arch1_3--4 checkpoints require "
                f"visual_perceiver_width=1024 at {policy_path}."
            )
    else:
        if architecture_revision not in {"", *COND_GEMMA_ARCHITECTURE_LABELS}:
            raise ValueError(
                "Unsupported cond_gemma architecture_revision="
                f"{architecture_revision!r} at {policy_path}."
            )
        resolved_architecture_revision = (
            architecture_revision or COND_GEMMA_ARCHITECTURE_REVISION
        )
        eval_legacy_vsa = False
        eval_vsa_revision = ""
        vision_conditioning_mode = "condition_gemma"
        is_perceiver_ablation = (
            resolved_architecture_revision == "expert_tokens_perceiver_v1"
        )
        num_visual_latents_per_camera = (
            int(policy.get("num_visual_latents_per_camera", 32))
            if is_perceiver_ablation
            else 0
        )
        visual_perceiver_width = (
            int(policy.get("visual_perceiver_width", 1024))
            if is_perceiver_ablation
            else 0
        )
        conditioning_route = _normalize_conditioning_route(
            policy.get("conditioning_route", "state_cond")
        )
        if conditioning_route not in CONDITIONING_ROUTES:
            raise ValueError(
                f"Unsupported cond_gemma conditioning_route={conditioning_route!r} "
                f"at {policy_path}."
            )
        architecture_label = COND_GEMMA_ARCHITECTURE_LABELS[
            resolved_architecture_revision
        ]
    saved_architecture_label = str(policy.get("architecture_label", "")).strip().lower()
    historical_arch0_alias = (
        architecture == COND_GEMMA_ARCHITECTURE
        and resolved_architecture_revision == COND_GEMMA_ARCHITECTURE_REVISION
        and saved_architecture_label == "arch1"
    )
    skill_aux_alias = (
        architecture == COND_GEMMA_ARCHITECTURE
        and (
            (
                resolved_architecture_revision
                == COND_GEMMA_ARCHITECTURE_REVISION
                and saved_architecture_label
                in {"arch0_skill", "arch0_skill_chunk"}
            )
            or (
                resolved_architecture_revision == "cond_expert_state_adarms_v1"
                and saved_architecture_label == "arch0_2_skill_chunk"
            )
        )
    )
    historical_arch2_alias = (
        architecture == VSA_ARCHITECTURE
        and resolved_architecture_revision == VSA_ARCHITECTURE_REVISION
        and vision_conditioning_mode == "interleaved_cross_attention"
        and saved_architecture_label == "arch2"
        and architecture_label == "arch2_2"
    )
    if (
        saved_architecture_label
        and saved_architecture_label != architecture_label
        and not historical_arch0_alias
        and not skill_aux_alias
        and not historical_arch2_alias
    ):
        raise ValueError(
            f"Checkpoint architecture_label={saved_architecture_label!r} does not "
            f"match its architecture contract; expected {architecture_label!r} at "
            f"{policy_path}."
        )
    if skill_aux_alias:
        architecture_label = saved_architecture_label
    action_loss_mode = str(policy.get("action_loss_mode", "")).strip().lower()
    if action_loss_mode != "flow":
        raise ValueError(
            "Stage-1 checkpoint action objective must be flow: "
            f"{policy_path}"
        )
    if architecture == VSA_ARCHITECTURE:
        # These switches belong to the VSA checkpoint architecture. Evaluation
        # must not override them from its own YAML.
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
            if vision_conditioning_mode
            in {
                "interleaved_cross_attention",
                "residual_cross_attention",
                "legacy_alternating",
            }
            else (
                "expert queries; visual fixed KV"
                if vision_conditioning_mode
                in {
                    "uncompressed_visual_kv_self_attention",
                    "compressed_visual_kv_self_attention",
                }
                else "ignored"
            )
        )
    else:
        include_state_in_visual_crossattn = False
        include_skill_in_visual_crossattn = False
        visual_crossattn_queries = "not_applicable"

    train_config = json.loads((policy_path / "train_config.json").read_text())
    dataset_value = str((train_config.get("dataset") or {}).get("root") or "").strip()
    if not dataset_value:
        raise ValueError(f"Stage-1 train_config has no dataset.root: {policy_path}")
    fsq_path = _relocate_project_path(project_root, policy.get("fsq_path"))
    skill_dataset_dir = _relocate_project_path(project_root, dataset_value)
    dataset_info_path = skill_dataset_dir / "meta" / "info.json"
    if not dataset_info_path.is_file():
        # Node-local training stages the dataset under /tmp or /dev/shm, so
        # train_config.json can legitimately retain a path that disappears
        # with the training job. FSQ remains the portable dataset provenance.
        portable_dataset_dir = fsq_path.parent / "skillvla"
        portable_info_path = portable_dataset_dir / "meta" / "info.json"
        if not portable_info_path.is_file():
            raise FileNotFoundError(
                "Stage-1 SkillVLA dataset not found at either the recorded or "
                f"portable FSQ location: {skill_dataset_dir}, {portable_dataset_dir}"
            )
        skill_dataset_dir = portable_dataset_dir
        dataset_info_path = portable_info_path
    dataset_info = json.loads(dataset_info_path.read_text())
    dataset_proprio_grounding = str(
        dataset_info.get("proprio_grounding", "none") or "none"
    ).strip().lower().replace("-", "_")
    policy_proprio_grounding = str(
        policy.get("proprio_grounding", "none") or "none"
    ).strip().lower().replace("-", "_")
    supported_proprio_grounding = {"none", "episode_start_xyz"}
    if policy_proprio_grounding not in supported_proprio_grounding:
        raise ValueError(
            "Unsupported checkpoint proprio_grounding="
            f"{policy_proprio_grounding!r} at {policy_path}."
        )
    if dataset_proprio_grounding != policy_proprio_grounding:
        raise ValueError(
            "Stage-1 checkpoint/dataset proprio grounding mismatch: "
            f"checkpoint={policy_proprio_grounding!r}, "
            f"dataset={dataset_proprio_grounding!r} at {dataset_info_path}."
        )
    run_dir = skill_dataset_dir.parent
    source_dir = run_dir.parent
    if len(source_dir.parents) < 2:
        raise ValueError(f"Unexpected Stage-1 dataset layout: {skill_dataset_dir}")

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
        "tokenizer_path": _relocate_project_path(
            project_root, policy.get("tokenizer_path")
        ),
    }
    if not paths["dino_model_path"].is_dir():
        raise FileNotFoundError(
            f"Stage-1 model directory not found: {paths['dino_model_path']}"
        )
    if has_predictor and not paths["tokenizer_path"].is_dir():
        raise FileNotFoundError(f"Stage-1 tokenizer not found: {paths['tokenizer_path']}")
    contract = {
        "policy": policy,
        "architecture": architecture,
        "architecture_label": architecture_label,
        "architecture_revision": resolved_architecture_revision,
        "architecture_inferred": architecture_inferred,
        "eval_legacy_vsa": eval_legacy_vsa,
        "eval_vsa_revision": eval_vsa_revision,
        "vision_conditioning_mode": vision_conditioning_mode,
        "num_visual_latents_per_camera": num_visual_latents_per_camera,
        "visual_perceiver_width": visual_perceiver_width,
        "include_state_in_visual_crossattn": include_state_in_visual_crossattn,
        "include_skill_in_visual_crossattn": include_skill_in_visual_crossattn,
        "visual_crossattn_queries": visual_crossattn_queries,
        "action_loss_mode": action_loss_mode,
        "has_predictor": has_predictor,
        "has_terminator": has_terminator,
        "proprio_grounding": policy_proprio_grounding,
        **paths,
    }
    if architecture == COND_GEMMA_ARCHITECTURE:
        contract["conditioning_route"] = conditioning_route
    return contract


def _policy_code_space_id(policy: dict) -> str:
    value = str(policy.get("skill_code_space_id", "") or "").strip()
    if value:
        return value
    fsq_path = str(policy.get("fsq_path", "") or "").strip()
    return Path(fsq_path).parent.name if fsq_path else ""


def _policies_share_fsq_checkpoint(
    source: dict,
    target: dict,
    *,
    project_root: Path | None,
) -> bool:
    if project_root is None:
        return False
    source_fsq = _relocate_project_path(project_root, source.get("fsq_path"))
    target_fsq = _relocate_project_path(project_root, target.get("fsq_path"))
    if not source_fsq.is_file() or not target_fsq.is_file():
        return False
    return source_fsq.resolve() == target_fsq.resolve() or (
        source_fsq.stat().st_size == target_fsq.stat().st_size
        and _checkpoint_sha256(source_fsq) == _checkpoint_sha256(target_fsq)
    )


def _validate_policy_code_space(
    source: dict,
    target: dict,
    *,
    component: str,
    project_root: Path | None,
) -> None:
    source_space = _policy_code_space_id(source)
    target_space = _policy_code_space_id(target)
    if not source_space or not target_space or source_space == target_space:
        return
    # Human-readable dataset/output suffixes may differ even when the copied
    # quantizer checkpoint is byte-identical. Geometry alone is insufficient:
    # accept the alias only after comparing the actual FSQ checkpoint bytes.
    if _policies_share_fsq_checkpoint(
        source, target, project_root=project_root
    ):
        return
    raise ValueError(
        f"External {component} skill-code space mismatch: "
        f"{component}={source_space!r}, target={target_space!r}."
    )


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
    if source.get("type") not in {"skill_expert", "skill_aux"}:
        raise ValueError(
            "External predictor must come from policy.type=skill_expert or skill_aux, got "
            f"{source.get('type')!r} at {checkpoint}."
        )
    if not as_bool(source.get("train_skill_predictor", False)):
        raise ValueError(
            f"External predictor checkpoint has no trained predictor: {checkpoint}"
        )
    # A Stage-1 target that trained its own predictor owns a fixed module, so the
    # overlay must match it exactly. Predictor-free Stage 1 rebuilds the module
    # and checks only skill geometry. Stage 2 always stores a frozen VLM module,
    # but it may be only the pristine pi0.5 placeholder used by its action path;
    # eval replaces that module from the selected predictor checkpoint. In that
    # case reader/head/LoRA settings may differ, while geometry and the base VLM
    # interface must remain compatible with the trained Stage-2 projection.
    target_has_predictor = as_bool(target_policy.get("train_skill_predictor", False))
    target_type = target_policy.get("type", target_policy.get("model_type"))
    if target_type == "skill_vla_stage2":
        checked_fields = (
            "skill_vocab_size",
            "skill_fsq_levels",
            "skill_predictor_vlm_variant",
            "skill_predictor_image_size",
        )
        mismatch_label = "Stage-2 interface"
    elif target_has_predictor:
        checked_fields = _PREDICTOR_CHECKPOINT_CONTRACT_FIELDS
        mismatch_label = "module contract"
    else:
        checked_fields = ("skill_vocab_size", "skill_fsq_levels")
        mismatch_label = "skill geometry"
    mismatches = [
        f"{field}: predictor={source.get(field)!r}, target={target_policy.get(field)!r}"
        for field in checked_fields
        if source.get(field) != target_policy.get(field)
    ]
    if mismatches:
        raise ValueError(
            "External predictor " + mismatch_label + " mismatch: "
            + "; ".join(mismatches)
        )
    _validate_policy_code_space(
        source,
        target_policy,
        component="predictor",
        project_root=project_root,
    )
    tokenizer_path = _relocate_project_path(
        project_root, source.get("tokenizer_path")
    )
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"External predictor tokenizer not found: {tokenizer_path}"
        )
    return {"tokenizer_path": tokenizer_path}


def _validate_external_terminator(
    checkpoint: Path,
    *,
    target_policy: dict,
    variant: str = "state_image",
    project_root: Path | None = None,
) -> None:
    """Validate an eval-time co-trained terminator source."""
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"External terminator config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"External terminator weights not found: {weights_path}")
    source = json.loads(config_path.read_text())
    allowed_types = {"skill_aux"} if variant == "image_only" else {
        "skill_expert",
        "skill_aux",
    }
    if source.get("type") not in allowed_types:
        raise ValueError(
            f"External {variant} terminator must come from "
            f"policy.type={sorted(allowed_types)}, got "
            f"{source.get('type')!r} at {checkpoint}."
        )
    train_field = (
        "train_image_only_terminator"
        if variant == "image_only"
        else "train_terminator"
    )
    if not as_bool(source.get(train_field, False)):
        raise ValueError(
            f"External checkpoint has no trained {variant} terminator: {checkpoint}"
        )
    if source.get("skill_fsq_levels") != target_policy.get("skill_fsq_levels"):
        raise ValueError(
            "External terminator FSQ mismatch: "
            f"terminator={source.get('skill_fsq_levels')!r}, "
            f"target={target_policy.get('skill_fsq_levels')!r}"
        )
    _validate_policy_code_space(
        source,
        target_policy,
        component="terminator",
        project_root=project_root,
    )


@lru_cache(maxsize=64)
def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_entries(config: dict) -> list[dict]:
    model_defaults = get_value(config, "model_defaults", {}) or {}
    if not isinstance(model_defaults, dict):
        raise ValueError("model_defaults must be a YAML mapping.")
    supported_defaults = {
        "outputs_root",
        "previous",
        "checkpoint",
        "skill_source",
        "advance_mode",
        "terminator_variant",
        "external_skill_model",
        "external_predictor_model",
        "external_predictor_checkpoint",
        "external_terminator_model",
        "external_terminator_checkpoint",
        "latent_source",
        "oracle_latent_target",
        "oracle_latent_grid_size",
        "oracle_latent_timesteps",
    }
    unknown_defaults = sorted(set(model_defaults) - supported_defaults)
    if unknown_defaults:
        raise ValueError(
            f"Unknown model_defaults fields {unknown_defaults}; "
            f"supported={sorted(supported_defaults)}."
        )

    # Per-model values below override this block. The older top-level fields and
    # oracle.advance_mode remain as compatibility fallbacks for saved configs.
    default_previous = as_bool(
        model_defaults.get("previous", get_value(config, "previous", False))
    )
    default_outputs_root = str(model_defaults.get("outputs_root", "") or "").strip()
    default_checkpoints = _checkpoint_list(
        model_defaults.get("checkpoint", get_value(config, "checkpoint", "last")),
        field="model_defaults.checkpoint",
    )
    default_skill_source = str(
        model_defaults.get("skill_source", get_value(config, "skill_source", "gt"))
    ).lower()
    # A models[] entry may name its own external checkpoint, which matters when
    # each target needs a terminator trained on its own FSQ/dataset run.
    # external_skill_model is the shared fallback; the predictor- and
    # terminator-specific keys split it when the two overlays must come from
    # different checkpoints.
    def _external_default(field: str, fallback: str = "") -> str:
        return str(
            model_defaults.get(field, get_value(config, field, fallback)) or ""
        ).strip()

    default_external_skill_model = _external_default("external_skill_model")
    default_external_predictor_model = _external_default(
        "external_predictor_model", default_external_skill_model
    )
    default_external_predictor_checkpoint = _safe_name(
        str(
            model_defaults.get(
                "external_predictor_checkpoint",
                get_value(config, "external_predictor_checkpoint", "last"),
            )
            or "last"
        ),
        field="model_defaults.external_predictor_checkpoint",
    )
    default_external_terminator_model = _external_default(
        "external_terminator_model", default_external_skill_model
    )
    default_external_terminator_checkpoint = _safe_name(
        str(
            model_defaults.get(
                "external_terminator_checkpoint",
                get_value(config, "external_terminator_checkpoint", "last"),
            )
            or "last"
        ),
        field="model_defaults.external_terminator_checkpoint",
    )
    default_advance = str(
        model_defaults.get(
            "advance_mode",
            _at(config, "oracle", "advance_mode", default="own"),
        )
    ).lower()
    default_terminator_variant = str(
        model_defaults.get(
            "terminator_variant",
            _at(config, "terminator", "variant", default="state_image"),
        )
    ).lower()
    default_latent_source = str(
        model_defaults.get(
            "latent_source",
            _at(config, "oracle", "latent_source", default="random"),
        )
        or "random"
    ).strip().lower()
    if default_latent_source in {"gt", "sampled", "predicted"}:
        default_latent_source = (
            "oracle" if default_latent_source == "gt" else "random"
        )
    if default_latent_source not in {"random", "oracle"}:
        raise ValueError(
            "model_defaults.latent_source must be random|oracle "
            "(predicted/sampled and gt are compatibility aliases)."
        )
    default_oracle_latent_target = str(
        model_defaults.get(
            "oracle_latent_target",
            _at(config, "oracle", "latent_target", default="start_chunk"),
        )
        or "start_chunk"
    ).strip().lower()
    if default_oracle_latent_target not in {"start_chunk", "full_skill"}:
        raise ValueError(
            "model_defaults.oracle_latent_target must be start_chunk|full_skill."
        )
    default_oracle_latent_grid_size = int(
        model_defaults.get(
            "oracle_latent_grid_size",
            _at(config, "oracle", "latent_grid_size", default=3),
        )
    )
    default_oracle_latent_timesteps = int(
        model_defaults.get(
            "oracle_latent_timesteps",
            _at(config, "oracle", "latent_timesteps", default=2),
        )
    )
    models = get_value(config, "models", None)
    if isinstance(models, list) and models:
        raw_entries = models
    else:
        model_dir = str(get_value(config, "model_dir", "") or "")
        if not model_dir:
            raise ValueError("Set models[] or a top-level model_dir in Stage-1 eval config.")
        raw_entries = [{"model_dir": model_dir}]

    rows = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise ValueError(f"models[{index}] must be a YAML mapping.")
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
        checkpoints = _checkpoint_list(
            raw.get("checkpoint", default_checkpoints),
            field="models[].checkpoint",
        )
        # Role-specific model fields are selectors as well as paths.  This keeps
        # per-panel YAML concise: naming an external predictor/terminator opts
        # that role into external mode unless the entry explicitly says
        # otherwise.  ``original`` is a terminator-only sentinel which follows
        # the FSQ checkpoint recorded by this Stage-1 model.
        explicit_predictor = str(
            raw.get("external_predictor_model", "") or ""
        ).strip()
        explicit_terminator = str(
            raw.get("external_terminator_model", "") or ""
        ).strip()
        original_terminator = explicit_terminator.lower() == "original"

        skill_source = str(
            raw.get(
                "skill_source",
                "external" if explicit_predictor else default_skill_source,
            )
        ).lower()
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
        inferred_advance = (
            "original"
            if original_terminator
            else "external"
            if explicit_terminator
            else default_advance
        )
        advance_mode = str(raw.get("advance_mode", inferred_advance)).lower()
        advance_aliases = {
            "gt": "gt",
            "own": "own",
            "terminator": "own",
            "external": "external",
            "original": "original",
        }
        advance_mode = advance_aliases.get(advance_mode, "")
        if not advance_mode:
            raise ValueError(
                "models[].advance_mode must be external|own|original|gt."
            )
        if original_terminator and advance_mode != "original":
            raise ValueError(
                "external_terminator_model=original conflicts with explicit "
                f"advance_mode={advance_mode!r}; omit advance_mode or set it to original."
            )
        variant_aliases = {
            "normal": "state_image",
            "state_image": "state_image",
            "state+image": "state_image",
            "image": "image_only",
            "image_only": "image_only",
            "image-only": "image_only",
        }
        terminator_variant = variant_aliases.get(
            str(raw.get("terminator_variant", default_terminator_variant)).lower(), ""
        )
        if not terminator_variant:
            raise ValueError(
                "models[].terminator_variant must be state_image|image_only."
            )
        latent_source = str(
            raw.get("latent_source", default_latent_source) or "random"
        ).strip().lower()
        if latent_source in {"gt", "sampled", "predicted"}:
            latent_source = "oracle" if latent_source == "gt" else "random"
        if latent_source not in {"random", "oracle"}:
            raise ValueError(
                "models[].latent_source must be random|oracle "
                "(predicted/sampled and gt are compatibility aliases)."
            )
        oracle_latent_target = str(
            raw.get("oracle_latent_target", default_oracle_latent_target)
            or "start_chunk"
        ).strip().lower()
        if oracle_latent_target not in {"start_chunk", "full_skill"}:
            raise ValueError(
                "models[].oracle_latent_target must be start_chunk|full_skill."
            )
        oracle_latent_grid_size = int(
            raw.get("oracle_latent_grid_size", default_oracle_latent_grid_size)
        )
        oracle_latent_timesteps = int(
            raw.get("oracle_latent_timesteps", default_oracle_latent_timesteps)
        )
        if oracle_latent_grid_size < 2:
            raise ValueError("models[].oracle_latent_grid_size must be at least 2.")
        if oracle_latent_timesteps <= 0:
            raise ValueError("models[].oracle_latent_timesteps must be positive.")
        if (
            advance_mode in {"own", "original"}
            and terminator_variant == "image_only"
        ):
            raise ValueError(
                "terminator_variant=image_only requires advance_mode=external; "
                "own/original provide only the normal state+image terminator."
            )
        label = str(raw.get("label", "") or "").strip()
        if not label:
            label = f"model{index + 1}-{skill_source}"
        rows.append(
            {
                "model_dir": model_dir,
                # Empty means: use the top-level/global outputs_root resolved by
                # build_settings. A relative override is rooted at project_root.
                "outputs_root_value": str(
                    raw.get("outputs_root", default_outputs_root) or ""
                ).strip(),
                "checkpoints": checkpoints,
                "skill_source": skill_source,
                "advance_mode": advance_mode,
                "terminator_variant": terminator_variant,
                "latent_source": latent_source,
                "oracle_latent_target": oracle_latent_target,
                "oracle_latent_grid_size": oracle_latent_grid_size,
                "oracle_latent_timesteps": oracle_latent_timesteps,
                "label": _clean_label(label),
                "previous_checkpoint": as_bool(
                    raw.get("previous", default_previous)
                ),
                # Raw strings; build_settings resolves them against project_root.
                # A per-entry external_skill_model still covers both overlays,
                # but either role may name its own checkpoint instead.
                "external_predictor_model_value": str(
                    raw.get(
                        "external_predictor_model",
                        raw.get(
                            "external_skill_model", default_external_predictor_model
                        ),
                    )
                    or ""
                ).strip(),
                "external_predictor_checkpoint": _safe_name(
                    str(
                        raw.get(
                            "external_predictor_checkpoint",
                            default_external_predictor_checkpoint,
                        )
                        or default_external_predictor_checkpoint
                    ),
                    field="models[].external_predictor_checkpoint",
                ),
                "external_terminator_model_value": str(
                    ""
                    if original_terminator
                    else raw.get(
                        "external_terminator_model",
                        raw.get(
                            "external_skill_model", default_external_terminator_model
                        ),
                    )
                    or ""
                ).strip(),
                "external_terminator_checkpoint": _safe_name(
                    str(
                        raw.get(
                            "external_terminator_checkpoint",
                            default_external_terminator_checkpoint,
                        )
                        or default_external_terminator_checkpoint
                    ),
                    field="models[].external_terminator_checkpoint",
                ),
            }
        )
    labels = [row["label"] for row in rows]
    if len(labels) != len(set(labels)):
        raise ValueError(f"models[].label values must be unique, got {labels}.")

    checkpoint_sequences = [row["checkpoints"] for row in rows]
    # Preserve the historical comparison where each model may select one
    # different checkpoint. Multi-checkpoint grids, however, need identical
    # ordered rows across every model column.
    is_single_checkpoint_comparison = all(
        len(sequence) == 1 for sequence in checkpoint_sequences
    )
    if not is_single_checkpoint_comparison and any(
        sequence != checkpoint_sequences[0]
        for sequence in checkpoint_sequences[1:]
    ):
        raise ValueError(
            "Every models[] entry must use the same ordered checkpoint list so "
            "side-by-side rows represent the same checkpoints. Put the shared "
            "list in model_defaults.checkpoint."
        )

    checkpoint_count = len(checkpoint_sequences[0])
    entries = []
    # Flatten checkpoint-major: one output row per checkpoint, with all models
    # adjacent horizontally in YAML order.
    for checkpoint_index in range(checkpoint_count):
        for model_index, row in enumerate(rows):
            checkpoint = row["checkpoints"][checkpoint_index]
            model_label = row["label"]
            panel_label = (
                f"{model_label} | ckpt {checkpoint}"
                if checkpoint_count > 1
                else model_label
            )
            entries.append(
                {
                    "model_dir": row["model_dir"],
                    "outputs_root_value": row["outputs_root_value"],
                    "checkpoint": checkpoint,
                    "skill_source": row["skill_source"],
                    "advance_mode": row["advance_mode"],
                    "terminator_variant": row["terminator_variant"],
                    "latent_source": row["latent_source"],
                    "oracle_latent_target": row["oracle_latent_target"],
                    "oracle_latent_grid_size": row[
                        "oracle_latent_grid_size"
                    ],
                    "oracle_latent_timesteps": row[
                        "oracle_latent_timesteps"
                    ],
                    "label": panel_label,
                    "model_label": model_label,
                    "model_index": model_index,
                    "checkpoint_index": checkpoint_index,
                    "previous_checkpoint": row["previous_checkpoint"],
                    "external_predictor_model_value": row[
                        "external_predictor_model_value"
                    ],
                    "external_predictor_checkpoint": row[
                        "external_predictor_checkpoint"
                    ],
                    "external_terminator_model_value": row[
                        "external_terminator_model_value"
                    ],
                    "external_terminator_checkpoint": row[
                        "external_terminator_checkpoint"
                    ],
                }
            )
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
    model_count = 1 + max(entry["model_index"] for entry in entries)
    checkpoint_count = 1 + max(entry["checkpoint_index"] for entry in entries)
    resolved = []
    for entry in entries:
        outputs_root_value = entry.pop("outputs_root_value", "")
        model_outputs_root = (
            _relocate_project_path(project_root, outputs_root_value)
            if outputs_root_value
            else outputs_root
        )
        model_root = model_outputs_root / "skillVLA_stage1"
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
        if entry["previous_checkpoint"]:
            if contract["architecture"] != VSA_ARCHITECTURE:
                raise ValueError(
                    "models[].previous=true is reserved for historical VSA checkpoints: "
                    f"{policy_path}."
                )
            if contract["visual_perceiver_width"] != 384:
                raise ValueError(
                    "A previous VSA checkpoint must use the historical "
                    "visual_perceiver_width=384 contract: "
                    f"{policy_path}."
                )
        if entry["skill_source"] == "own" and not contract["has_predictor"]:
            raise ValueError(
                f"skill_source=own but checkpoint has no trained predictor: {policy_path}"
            )
        tokenizer_path = contract["tokenizer_path"]
        predictor_value = entry.pop("external_predictor_model_value", "")
        terminator_value = entry.pop("external_terminator_model_value", "")
        entry_predictor = (
            _resolve_external_predictor_path(
                project_root,
                outputs_root,
                predictor_value,
                entry["external_predictor_checkpoint"],
            )
            if predictor_value
            else external_skill_model
        )
        entry_terminator = (
            _resolve_external_terminator_path(
                project_root,
                outputs_root,
                terminator_value,
                entry["external_terminator_checkpoint"],
            )
            if terminator_value
            else external_skill_model
        )
        if entry["skill_source"] == "external":
            if entry_predictor is None:
                raise ValueError(
                    f"models[].label={entry['label']!r} uses skill_source=external "
                    "but no external_predictor_model or external_skill_model was "
                    "set on the entry, in model_defaults, or at the top level."
                )
            external = _external_predictor_contract(
                entry_predictor,
                target_policy=contract["policy"],
                project_root=project_root,
            )
            tokenizer_path = external["tokenizer_path"]
        if entry["advance_mode"] == "external":
            if entry_terminator is None:
                raise ValueError(
                    f"models[].label={entry['label']!r} uses advance_mode=external "
                    "but no external_terminator_model or external_skill_model was "
                    "set on the entry, in model_defaults, or at the top level."
                )
            _validate_external_terminator(
                entry_terminator,
                target_policy=contract["policy"],
                variant=entry["terminator_variant"],
                project_root=project_root,
            )
        if entry["advance_mode"] == "original" and not contract["fsq_path"].is_file():
            raise FileNotFoundError(
                "advance_mode=original requires the FSQ checkpoint referenced by "
                f"the Stage-1 policy: {contract['fsq_path']}"
            )
        if entry["advance_mode"] == "own" and not contract["has_terminator"]:
            raise ValueError(
                "advance_mode=own but checkpoint has no trained terminator: "
                f"{policy_path}"
            )
        resolved.append(
            {
                **entry,
                "outputs_root": model_outputs_root,
                "policy_path": policy_path,
                "chunk_size": int(contract["policy"].get("chunk_size", 10)),
                **contract,
                # Kept for logging/back-compat; the two role-specific paths below
                # are what run_eval actually overlays.
                "external_skill_model": entry_terminator or entry_predictor or "",
                "external_predictor_model": entry_predictor or "",
                "external_terminator_model": entry_terminator or "",
                "tokenizer_path": tokenizer_path,
            }
        )

    episode_exact = as_bool(_at(config, "oracle", "episode_exact", default=False))
    oracle_latent_models = [
        model for model in resolved if model["latent_source"] == "oracle"
    ]
    if oracle_latent_models and not episode_exact:
        raise ValueError(
            "latent_source=oracle requires oracle.episode_exact=true so the "
            "aligned GT action chunk is available."
        )
    for model in oracle_latent_models:
        policy = model["policy"]
        if not as_bool(policy.get("skill_flow_latent_best_of_n_enabled", False)):
            raise ValueError(
                f"models[].label={model['label']!r} uses latent_source=oracle, "
                "but its Stage-1 checkpoint has no latent Best-of-N path."
            )
        if int(policy.get("skill_flow_latent_dim", 0)) != 2:
            raise ValueError(
                f"models[].label={model['label']!r} uses latent_source=oracle, "
                "but the grid oracle currently requires skill_flow_latent_dim=2."
            )
        if model["oracle_latent_target"] == "full_skill":
            if str(policy.get("type", "")) != "skill_expert":
                raise ValueError(
                    "oracle_latent_target=full_skill is currently supported only "
                    f"for Stage-1 skill_expert checkpoints, not {policy.get('type')!r}."
                )
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
    gt_termination_min_fraction = float(
        _at(config, "terminator", "gt_termination_min_fraction", default=0.5)
    )
    if not 0.0 <= gt_termination_min_fraction <= 1.0:
        raise ValueError(
            "terminator.gt_termination_min_fraction must be between 0 and 1."
        )
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
    target_task = str(get_value(config, "target_task", "libero_goal"))
    env_task_ids = task_ids
    if target_task.startswith("langgap_"):
        env_task_ids = _langgap_env_task_ids(
            resolved[0]["eval_init_states_path"],
            task_ids,
            suite_name=target_task,
        )

    output_name = str(get_value(config, "output_name", "") or "").strip()
    output_name = _safe_name(
        output_name
        or _default_output_name(
            resolved,
            model_count=model_count,
            checkpoint_count=checkpoint_count,
        ),
        field="output_name",
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
        "model_count": model_count,
        "checkpoint_count": checkpoint_count,
        "panel_count": len(resolved),
        "model_architectures": ", ".join(
            f"{model['label']}={model['architecture_label']}"
            + ("[previous]" if model["previous_checkpoint"] else "")
            for model in resolved
        ),
        # Specs are flattened checkpoint-major, so this many columns produces
        # one row per checkpoint and one column per model.
        "grid_columns": model_count,
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
        "tokenizer_path": primary["tokenizer_path"],
        "architecture": primary["architecture"],
        "architecture_label": primary["architecture_label"],
        "conditioning_route": primary.get("conditioning_route", ""),
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
        "target_task": target_task,
        "dataset_task_ids": json.dumps(task_ids, separators=(",", ":")),
        "task_ids": json.dumps(env_task_ids, separators=(",", ":")),
        # Preserve the full task count before array workers replace TASK_IDS
        # with their own chunk.  Every worker runs the idempotent merge step;
        # the last one therefore annotates and writes the complete report.
        "eval_expected_tasks": len(env_task_ids),
        "eval_num_gpus": int(get_value(config, "eval_num_gpus", 1)),
        "eval_max_workers_per_gpu": int(
            get_value(config, "eval_max_workers_per_gpu", 4)
        ),
        "n_episodes": int(get_value(config, "n_episodes", 3)),
        "eval_batch_size": int(get_value(config, "eval_batch_size", 1)),
        "max_parallel_tasks": int(get_value(config, "max_parallel_tasks", 1)),
        "n_action_steps": n_action_steps,
        "advance_mode": primary["advance_mode"],
        "terminator_variant": primary["terminator_variant"],
        "skill_end_mode": end_mode,
        "skill_end_threshold": float(
            _at(config, "terminator", "end_threshold", default=0.5)
        ),
        "skill_end_progress_threshold": float(
            _at(config, "terminator", "progress_threshold", default=0.95)
        ),
        "gt_termination_min_fraction": gt_termination_min_fraction,
        "immediate_replan_on_skill_end": as_bool(
            _at(
                config,
                "terminator",
                "immediate_replan_on_skill_end",
                default=False,
            )
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
    if settings["eval_num_gpus"] <= 0:
        raise ValueError("eval_num_gpus must be positive.")
    if not 1 <= settings["eval_max_workers_per_gpu"] <= 4:
        raise ValueError("eval_max_workers_per_gpu must be between 1 and 4.")
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
