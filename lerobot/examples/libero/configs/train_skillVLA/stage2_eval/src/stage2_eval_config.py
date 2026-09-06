#!/usr/bin/env python3
"""Resolve Stage-2 evaluation into shell exports.

Stage-2 evaluation assembles a likelihood or DSBC checkpoint with the selected
predictor and either an external or pristine-FSQ terminator. Each configured
checkpoint expands into up to two panels:

* ``stage2``: the complete Stage-2 policy, with its mode read from checkpoint.
* ``prior``: the exact frozen Stage-1 prior recorded in the Stage-2
  checkpoint's config, evaluated standalone.

Both panels run under the same predictor, terminator, and oracle skill map,
and the shared evaluator stitches them into one side-by-side video, so the
difference is exactly what the selected Stage-2 method added.
"""

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
sys.path.insert(0, str(_HERE.parent.parent.parent / "stage1_eval" / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

from stage1_eval_config import (  # noqa: E402
    _checkpoint_contract as _stage1_prior_contract,
    _external_predictor_contract,
    _langgap_env_task_ids,
    _resolve_external_predictor_path,
    _resolve_external_terminator_path,
    _validate_external_terminator,
)

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage2_eval_config.yaml"

_MODES = ("stage2", "prior")
_STAGE2_MODES = ("likelihood", "dsbc")
_DSBC_NOISE_OUTPUT_MODES = ("shared", "per_step")
_PROPRIO_GROUNDING_MODES = {"none", "episode_start_xyz"}
_SKILL_FLOW_ARCHITECTURE_REVISIONS = {
    "arch0_skill": "skillvla_real_v1",
    "arch0_skill_chunk": "skillvla_real_v1",
    "arch0_2_skill_chunk": "cond_expert_state_adarms_v1",
}


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


def _resolve_recorded_stage1_prior(
    project_root: Path, value: str | Path | None
) -> Path:
    """Follow a Stage-1 checkpoint after an old run was archived under PREV."""
    path = _relocate_project_path(project_root, value)
    if (path / "config.json").is_file():
        return path
    parts = path.parts
    if "skillVLA_stage1" not in parts:
        return path
    index = parts.index("skillVLA_stage1") + 1
    for archive_name in ("PREV", "previous"):
        candidate = Path(*parts[:index], archive_name, *parts[index:])
        if (candidate / "config.json").is_file():
            return candidate
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


def _default_output_name(entries: list[dict]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(entries) > 1:
        return f"compare_{len(entries)}models_{stamp}"
    model = re.sub(r"[^A-Za-z0-9._-]+", "-", entries[0]["model_dir"]).strip("-_")
    raw = f"{model}_{entries[0]['checkpoint']}_{stamp}"
    return raw if len(raw) <= 200 else f"stage2_{entries[0]['checkpoint']}_{stamp}"


def _stage2_checkpoint_contract(policy_path: Path, project_root: Path) -> dict:
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
    if str(policy.get("architecture", "")) != "cond_gemma":
        raise ValueError(
            f"Stage-2 checkpoints are built on the cond_gemma prior: {policy_path}"
        )
    architecture_label = str(policy.get("architecture_label", "") or "").strip().lower()
    architecture_revision = str(
        policy.get("architecture_revision", "skillvla_real_v1") or ""
    ).strip()
    expected_revision = _SKILL_FLOW_ARCHITECTURE_REVISIONS.get(architecture_label)
    if expected_revision is not None and architecture_revision != expected_revision:
        raise ValueError(
            f"Stage-2 {architecture_label} architecture contract mismatch at "
            f"{policy_path}: expected revision={expected_revision!r}, got "
            f"{architecture_revision!r}."
        )
    if not as_bool(policy.get("train_skill_predictor", False)):
        raise ValueError(f"Stage-2 checkpoint has no frozen VLM module: {policy_path}")
    # Mode is checkpoint-owned. Legacy checkpoints predate this field and are
    # exactly the likelihood architecture, so their unambiguous default is
    # likelihood rather than an eval-time selector.
    stage2_mode = str(policy.get("stage2_mode", "likelihood")).strip().lower()
    if stage2_mode not in _STAGE2_MODES:
        raise ValueError(
            f"Invalid Stage-2 mode {stage2_mode!r} at {policy_path}; "
            f"expected one of {_STAGE2_MODES}."
        )
    dsbc_noise_output_mode = str(
        policy.get("dsbc_noise_output_mode", "shared")
    ).strip().lower()
    if dsbc_noise_output_mode not in _DSBC_NOISE_OUTPUT_MODES:
        raise ValueError(
            "Invalid DSBC noise output mode "
            f"{dsbc_noise_output_mode!r} at {policy_path}; expected one of "
            f"{_DSBC_NOISE_OUTPUT_MODES}."
        )
    dsbc_frs_num_steps = int(policy.get("dsbc_frs_num_steps", 10))
    dsbc_anchor_seed = int(policy.get("dsbc_anchor_seed", 0))
    dsbc_reader = str(policy.get("dsbc_reader", "final")).strip().lower()
    dsbc_latent_predictor_enabled = as_bool(
        policy.get("dsbc_latent_predictor_enabled", False)
    )
    dsbc_latent_predictor_mode = str(
        policy.get("dsbc_latent_predictor_mode", "skill_start")
    ).strip().lower().replace("-", "_")
    dsbc_latent_supervision = str(
        policy.get("dsbc_latent_supervision", "main_chunk")
    ).strip().lower().replace("-", "_")
    dsbc_latent_loss_weight = float(policy.get("dsbc_latent_loss_weight", 1.0))
    dsbc_latent_timesteps = int(policy.get("dsbc_latent_timesteps", 2))
    if dsbc_frs_num_steps <= 0:
        raise ValueError(f"DSBC FRS steps must be positive at {policy_path}.")
    if dsbc_anchor_seed < 0:
        raise ValueError(f"DSBC anchor seed must be non-negative at {policy_path}.")
    if dsbc_reader not in {"final", "all_layers"}:
        raise ValueError(
            f"Invalid DSBC reader {dsbc_reader!r} at {policy_path}; expected "
            "'final' or 'all_layers'."
        )
    if dsbc_latent_predictor_mode not in {
        "skill_start",
        "per_chunk_final",
        "per_chunk_expert",
    }:
        raise ValueError(
            "Invalid DSBC latent predictor mode "
            f"{dsbc_latent_predictor_mode!r} at {policy_path}; expected "
            "'skill_start', 'per_chunk_final', or 'per_chunk_expert'."
        )
    if dsbc_latent_supervision not in {"main_chunk", "skill_only"}:
        raise ValueError(
            "Invalid DSBC latent supervision "
            f"{dsbc_latent_supervision!r} at {policy_path}; expected "
            "'main_chunk' or 'skill_only'."
        )
    if dsbc_latent_loss_weight <= 0.0:
        raise ValueError(f"DSBC latent loss weight must be positive at {policy_path}.")
    if dsbc_latent_timesteps <= 0:
        raise ValueError(f"DSBC latent timesteps must be positive at {policy_path}.")
    if stage2_mode == "dsbc" and as_bool(
        policy.get("cumulative_xyz_loss_enabled", False)
    ):
        raise ValueError(
            f"DSBC checkpoint cannot enable cumulative_xyz_loss: {policy_path}"
        )
    stage1_prior_path = _resolve_recorded_stage1_prior(
        project_root, policy.get("stage1_checkpoint_path")
    )

    train_config = json.loads((policy_path / "train_config.json").read_text())
    dataset_config = train_config.get("dataset") or {}
    dataset_value = str(dataset_config.get("root") or "").strip()
    fsq_path = _relocate_project_path(project_root, policy.get("fsq_path"))
    candidates: list[Path] = []
    if dataset_value:
        candidates.append(_relocate_project_path(project_root, dataset_value))

    # Node-local training intentionally records /tmp/.../skillvla in
    # train_config.json. Recover the persistent source/run from the portable
    # FSQ lineage and dataset repo_id once that temporary directory disappears.
    if fsq_path.name and len(fsq_path.parents) >= 3:
        source = str(dataset_config.get("repo_id") or "").strip().rstrip("/")
        source = source.rsplit("/", 1)[-1] if source else ""
        if source:
            candidates.append(fsq_path.parents[2] / source / fsq_path.parent.name / "skillvla")
        candidates.append(fsq_path.parent / "skillvla")

    skill_dataset_dir = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "meta" / "info.json").is_file()
        ),
        None,
    )
    if skill_dataset_dir is None:
        rendered = ", ".join(str(candidate) for candidate in candidates) or "<none>"
        raise FileNotFoundError(
            "Stage-2 SkillVLA dataset not found at its recorded or portable "
            f"lineage locations: {rendered}"
        )
    dataset_info_path = skill_dataset_dir / "meta" / "info.json"
    dataset_info = json.loads(dataset_info_path.read_text())
    dataset_proprio_grounding = str(
        dataset_info.get("proprio_grounding", "none") or "none"
    ).strip().lower().replace("-", "_")
    policy_proprio_grounding = str(
        policy.get("proprio_grounding", "none") or "none"
    ).strip().lower().replace("-", "_")
    if policy_proprio_grounding not in _PROPRIO_GROUNDING_MODES:
        raise ValueError(
            "Unsupported Stage-2 checkpoint proprio_grounding="
            f"{policy_proprio_grounding!r} at {policy_path}."
        )
    if dataset_proprio_grounding != policy_proprio_grounding:
        raise ValueError(
            "Stage-2 checkpoint/dataset proprio grounding mismatch: "
            f"checkpoint={policy_proprio_grounding!r}, "
            f"dataset={dataset_proprio_grounding!r} at {dataset_info_path}."
        )
    run_dir = skill_dataset_dir.parent
    source_dir = run_dir.parent
    if len(source_dir.parents) < 2:
        raise ValueError(f"Unexpected Stage-2 dataset layout: {skill_dataset_dir}")

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
    if not paths["fsq_path"].is_file():
        raise FileNotFoundError(f"Stage-2 FSQ checkpoint not found: {paths['fsq_path']}")
    for key in ("dino_model_path", "tokenizer_path"):
        if not paths[key].is_dir():
            raise FileNotFoundError(f"Stage-2 dependency not found ({key}): {paths[key]}")
    return {
        "policy": policy,
        "stage1_prior_path": stage1_prior_path,
        "architecture": str(policy.get("architecture")),
        "architecture_label": architecture_label,
        "architecture_revision": architecture_revision,
        "conditioning_route": str(policy.get("conditioning_route", "state_cond")),
        "num_visual_latents_per_camera": int(
            policy.get("num_visual_latents_per_camera", 32)
        ),
        "visual_perceiver_width": int(policy.get("visual_perceiver_width", 1024)),
        "action_loss_mode": str(policy.get("action_loss_mode", "flow")),
        "stage2_mode": stage2_mode,
        "dsbc_noise_output_mode": dsbc_noise_output_mode,
        "dsbc_frs_num_steps": dsbc_frs_num_steps,
        "dsbc_anchor_seed": dsbc_anchor_seed,
        "dsbc_reader": dsbc_reader,
        "dsbc_latent_predictor_enabled": dsbc_latent_predictor_enabled,
        "dsbc_latent_predictor_mode": dsbc_latent_predictor_mode,
        "dsbc_latent_supervision": dsbc_latent_supervision,
        "dsbc_latent_loss_weight": dsbc_latent_loss_weight,
        "dsbc_latent_timesteps": dsbc_latent_timesteps,
        "proprio_grounding": policy_proprio_grounding,
        **paths,
    }


def _oracle_dataset_contract(config: dict, project_root: Path) -> dict | None:
    """Resolve an optional eval-only GT dataset shared by every panel."""
    dataset_value = str(
        _at(config, "oracle", "skill_dataset_dir", default="") or ""
    ).strip()
    if not dataset_value:
        return None

    skill_dataset_dir = _relocate_project_path(project_root, dataset_value)
    dataset_info_path = skill_dataset_dir / "meta" / "info.json"
    if not dataset_info_path.is_file():
        raise FileNotFoundError(
            "Stage-2 oracle SkillVLA dataset not found: "
            f"{skill_dataset_dir}"
        )
    run_dir = skill_dataset_dir.parent
    source_dir = run_dir.parent
    if len(source_dir.parents) < 2:
        raise ValueError(
            f"Unexpected Stage-2 oracle dataset layout: {skill_dataset_dir}"
        )
    dataset_info = json.loads(dataset_info_path.read_text())
    proprio_grounding = str(
        dataset_info.get("proprio_grounding", "none") or "none"
    ).strip().lower().replace("-", "_")
    if proprio_grounding not in _PROPRIO_GROUNDING_MODES:
        raise ValueError(
            "Unsupported oracle dataset proprio_grounding="
            f"{proprio_grounding!r} at {dataset_info_path}."
        )
    return {
        "skill_dataset_dir": skill_dataset_dir,
        "eval_init_states_path": source_dir / "eval_init_states.npz",
        "skill_latents_path": run_dir / "skill_latents.npz",
        "raw_dataset_dir": source_dir.parents[1] / source_dir.name,
        "proprio_grounding": proprio_grounding,
    }


def _model_entries(config: dict) -> list[dict]:
    model_defaults = get_value(config, "model_defaults", {}) or {}
    if not isinstance(model_defaults, dict):
        raise ValueError("model_defaults must be a YAML mapping.")
    supported_defaults = {
        "outputs_root",
        "checkpoint",
        "outputs_subdir",
        "skill_source",
        "advance_mode",
        "modes",
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

    default_checkpoint = str(
        model_defaults.get("checkpoint", get_value(config, "checkpoint", "last"))
    )
    default_outputs_root = str(model_defaults.get("outputs_root", "") or "").strip()
    default_outputs_subdir = _safe_name(
        str(
            model_defaults.get(
                "outputs_subdir",
                get_value(config, "outputs_subdir", "skillVLA_stage2"),
            )
        ),
        field="outputs_subdir",
    )
    default_skill_source = model_defaults.get(
        "skill_source", get_value(config, "skill_source", None)
    )
    default_advance = model_defaults.get(
        "advance_mode",
        _at(config, "oracle", "advance_mode", default=None),
    )
    default_modes = model_defaults.get(
        "modes", get_value(config, "modes", list(_MODES))
    )
    default_terminator_variant = str(
        model_defaults.get(
            "terminator_variant",
            _at(config, "terminator", "variant", default="state_image"),
        )
    ).lower()

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
    default_latent_source = str(
        model_defaults.get(
            "latent_source",
            _at(config, "oracle", "latent_source", default="predicted"),
        )
        or "predicted"
    ).strip().lower()
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
            raise ValueError("Set models[] or a top-level model_dir in Stage-2 eval config.")
        raw_entries = [{"model_dir": model_dir}]

    entries = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise ValueError(f"models[{index}] must be a YAML mapping.")
        model_dir = _safe_name(str(raw.get("model_dir", "")), field="models[].model_dir")
        outputs_subdir = _safe_name(
            str(raw.get("outputs_subdir", default_outputs_subdir)),
            field="models[].outputs_subdir",
        )
        checkpoint = _safe_name(
            str(raw.get("checkpoint", default_checkpoint)), field="models[].checkpoint"
        )
        selected_predictor = str(
            raw.get(
                "external_predictor_model",
                raw.get("external_skill_model", default_external_predictor_model),
            )
            or ""
        ).strip()
        predictor_selector = selected_predictor.lower()
        selected_terminator = str(
            raw.get(
                "external_terminator_model",
                raw.get("external_skill_model", default_external_terminator_model),
            )
            or ""
        ).strip()
        terminator_selector = selected_terminator.lower()

        legacy_skill_source = raw.get("skill_source", default_skill_source)
        raw_skill_source = (
            str(legacy_skill_source).lower()
            if legacy_skill_source is not None
            else (
                "gt"
                if predictor_selector in {"", "gt"}
                else "own"
                if predictor_selector in {"own", "original"}
                else "external"
            )
        )
        skill_aliases = {
            "gt": "gt",
            "oracle": "gt",
            "own": "own",
            "predictor": "external"
            if predictor_selector not in {"", "gt", "own", "original"}
            else "own",
            "pred": "own",
            "predicted": "own",
            "external": "external",
        }
        skill_source = skill_aliases.get(raw_skill_source, "")
        if not skill_source:
            raise ValueError("models[].skill_source must be gt|own|external.")

        if "advance_mode" in raw:
            raw_advance = str(raw["advance_mode"]).lower()
        elif terminator_selector:
            raw_advance = (
                terminator_selector
                if terminator_selector in {"gt", "original"}
                else "external"
            )
        elif default_advance is not None:
            raw_advance = str(default_advance).lower()
        else:
            raw_advance = "gt"
        advance_aliases = {
            "gt": "gt",
            "external": "external",
            # Historical Stage-2 YAML called an external overlay simply
            # ``terminator``.
            "terminator": "external",
            "original": "original",
        }
        advance_mode = advance_aliases.get(raw_advance, "")
        if not advance_mode:
            raise ValueError(
                "models[].advance_mode must be gt|external|original."
            )
        if terminator_selector == "original" and advance_mode != "original":
            raise ValueError(
                "external_terminator_model=original conflicts with explicit "
                f"advance_mode={advance_mode!r}."
            )
        if terminator_selector == "gt" and advance_mode != "gt":
            raise ValueError(
                "external_terminator_model=gt conflicts with explicit "
                f"advance_mode={advance_mode!r}."
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
            str(raw.get("terminator_variant", default_terminator_variant)).lower(),
            "",
        )
        if not terminator_variant:
            raise ValueError(
                "models[].terminator_variant must be state_image|image_only."
            )
        if advance_mode == "original" and terminator_variant != "state_image":
            raise ValueError(
                "external_terminator_model=original supports only state_image."
            )
        modes = raw.get("modes", default_modes)
        if isinstance(modes, str):
            modes = [modes]
        modes = [str(mode).strip().lower() for mode in modes]
        if not modes or len(modes) != len(set(modes)):
            raise ValueError("models[].modes must be a non-empty unique list.")
        unknown_modes = sorted(set(modes) - set(_MODES))
        if unknown_modes:
            raise ValueError(
                f"models[].modes only accepts {list(_MODES)}, got {unknown_modes}."
            )
        label = str(raw.get("label", "") or "").strip()
        latent_source = str(
            raw.get("latent_source", default_latent_source) or "predicted"
        ).strip().lower()
        if latent_source == "gt":
            latent_source = "oracle"
        if latent_source not in {"predicted", "oracle"}:
            raise ValueError(
                "models[].latent_source must be predicted|oracle (gt is an alias)."
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
        entries.append(
            {
                "model_dir": model_dir,
                # Empty follows the snapshotted global outputs_root. This path
                # applies only to the Stage-2 policy; auxiliary predictor and
                # terminator runs always resolve from the global root.
                "outputs_root_value": str(
                    raw.get("outputs_root", default_outputs_root) or ""
                ).strip(),
                "outputs_subdir": outputs_subdir,
                "checkpoint": checkpoint,
                "skill_source": skill_source,
                "advance_mode": advance_mode,
                "terminator_variant": terminator_variant,
                "modes": modes,
                "latent_source": latent_source,
                "oracle_latent_target": oracle_latent_target,
                "oracle_latent_grid_size": oracle_latent_grid_size,
                "oracle_latent_timesteps": oracle_latent_timesteps,
                "label": _clean_label(
                    label or f"model{index + 1}-{raw_skill_source}"
                ),
                "external_predictor_model_value": (
                    ""
                    if predictor_selector in {"", "gt", "own", "original"}
                    else selected_predictor
                ),
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
                    if terminator_selector in {"", "gt", "original"}
                    else selected_terminator
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
    labels = [entry["label"] for entry in entries]
    if len(labels) != len(set(labels)):
        raise ValueError(f"models[].label values must be unique, got {labels}.")
    return entries


def _panel_spec(
    entry: dict,
    mode: str,
    *,
    policy_path: Path,
    policy_config: dict,
    architecture_fields: dict,
    eval_data_fields: dict,
    tokenizer_path: Path,
    predictor_path: Path | None,
    terminator_path: Path | None,
    terminator_variant: str,
) -> dict:
    if entry["skill_source"] == "gt":
        skill_source = "gt"
    elif mode == "prior" or predictor_path is not None:
        skill_source = "external"
    else:
        # The Stage-2 checkpoint embeds the predictor it trained with.
        skill_source = "own"
    advance_mode = entry["advance_mode"]
    return {
        "model_dir": entry["model_dir"],
        "checkpoint": entry["checkpoint"],
        "mode": mode,
        "label": f"{entry['label']}-{mode}",
        "policy_path": policy_path,
        "chunk_size": int(policy_config.get("chunk_size", 10)),
        "policy": policy_config,
        "skill_source": skill_source,
        # A Stage-1 prior panel has no learned Stage-2 latent predictor. Keep it
        # as the ordinary random-latent baseline even when its paired Stage-2
        # panel requests the hindsight oracle.
        "latent_source": (
            entry["latent_source"] if mode == "stage2" else "predicted"
        ),
        "oracle_latent_target": (
            entry["oracle_latent_target"]
            if mode == "stage2"
            else "start_chunk"
        ),
        "oracle_latent_grid_size": entry["oracle_latent_grid_size"],
        "oracle_latent_timesteps": entry["oracle_latent_timesteps"],
        "advance_mode": advance_mode,
        "terminator_variant": terminator_variant,
        "external_predictor_model": str(predictor_path or ""),
        "external_terminator_model": str(terminator_path or ""),
        "eval_legacy_vsa": False,
        "eval_vsa_revision": "",
        "tokenizer_path": tokenizer_path,
        **architecture_fields,
        **eval_data_fields,
    }


def build_settings(config: dict) -> dict:
    project_root = Path(str(get_value(config, "project_root"))).expanduser()
    outputs_root = _relocate_project_path(
        project_root, get_value(config, "outputs_root", "outputs")
    )
    eval_outputs_root = _HERE.parent.parent / "outputs"

    entries = _model_entries(config)
    oracle_dataset = _oracle_dataset_contract(config, project_root)
    resolved = []
    # Prior panels are identical whenever they share the same frozen Stage-1
    # checkpoint and runtime settings; evaluate each distinct prior only once.
    prior_specs: dict[tuple, dict] = {}
    prior_requests: dict[tuple, int] = {}
    for entry in entries:
        outputs_root_value = entry.pop("outputs_root_value", "")
        model_outputs_root = (
            _relocate_project_path(project_root, outputs_root_value)
            if outputs_root_value
            else outputs_root
        )
        stage2_path = (
            model_outputs_root
            / entry["outputs_subdir"]
            / entry["model_dir"]
            / "checkpoints"
            / entry["checkpoint"]
            / "pretrained_model"
        )
        contract = _stage2_checkpoint_contract(stage2_path, project_root)
        # A saved Stage-2 policy is self-contained: its safetensors already
        # include the frozen Stage-1 prior and from_pretrained deliberately
        # loads it with initialize_from_sources=False.  The original Stage-1
        # directory is needed only when eval explicitly requests a standalone
        # ``prior`` panel.
        if (
            "prior" in entry["modes"]
            and not (contract["stage1_prior_path"] / "config.json").is_file()
        ):
            raise FileNotFoundError(
                "Stage-1 prior recorded in the Stage-2 checkpoint not found: "
                f"{contract['stage1_prior_path']}"
            )
        stage2_policy = contract["policy"]

        predictor_value = entry.pop("external_predictor_model_value", "")
        terminator_value = entry.pop("external_terminator_model_value", "")
        predictor_path = (
            _resolve_external_predictor_path(
                project_root,
                outputs_root,
                predictor_value,
                entry["external_predictor_checkpoint"],
            )
            if predictor_value
            else None
        )
        terminator_path = (
            _resolve_external_terminator_path(
                project_root,
                outputs_root,
                terminator_value,
                entry["external_terminator_checkpoint"],
            )
            if terminator_value
            else None
        )
        terminator_variant = entry["terminator_variant"]

        needs_terminator = entry["advance_mode"] == "external"
        needs_predictor = entry["skill_source"] == "external"
        if needs_terminator:
            if terminator_path is None:
                raise ValueError(
                    f"models[].label={entry['label']!r} uses "
                    "advance_mode=external but no external_terminator_model "
                    "was set."
                )
            _validate_external_terminator(
                terminator_path,
                target_policy=stage2_policy,
                variant=terminator_variant,
                project_root=project_root,
            )
        predictor_contract = None
        if needs_predictor:
            if predictor_path is None:
                raise ValueError(
                    f"models[].label={entry['label']!r} uses "
                    "skill_source=external but no external_predictor_model "
                    "was set."
                )
            predictor_contract = _external_predictor_contract(
                predictor_path,
                target_policy=stage2_policy,
                project_root=project_root,
            )
        elif entry["skill_source"] == "own" and "prior" in entry["modes"]:
            raise ValueError(
                f"models[].label={entry['label']!r} uses skill_source=own, "
                "but a prior panel has no predictor of its own. Select "
                "skill_source=external and provide external_predictor_model, "
                "or evaluate only modes: [stage2]."
            )

        # Both panels share one oracle dataset so their GT maps, init states,
        # and skill traces are identical. By default this is the Stage-2
        # training dataset; oracle.skill_dataset_dir can explicitly select an
        # eval-suite dataset without changing the checkpoint-owned model paths.
        eval_data_fields = {
            "fsq_path": contract["fsq_path"],
            "dino_model_path": contract["dino_model_path"],
            "skill_dataset_dir": contract["skill_dataset_dir"],
            "eval_init_states_path": contract["eval_init_states_path"],
            "skill_latents_path": contract["skill_latents_path"],
            "raw_dataset_dir": contract["raw_dataset_dir"],
            "proprio_grounding": contract["proprio_grounding"],
        }
        if oracle_dataset is not None:
            if oracle_dataset["proprio_grounding"] != contract["proprio_grounding"]:
                raise ValueError(
                    "Stage-2 checkpoint/oracle proprio grounding mismatch: "
                    f"checkpoint={contract['proprio_grounding']!r}, "
                    f"oracle={oracle_dataset['proprio_grounding']!r} at "
                    f"{oracle_dataset['skill_dataset_dir']}."
                )
            eval_data_fields.update(oracle_dataset)
        eval_tokenizer_path = (
            predictor_contract["tokenizer_path"]
            if predictor_contract is not None
            else contract["tokenizer_path"]
        )
        stage2_architecture_fields = {
            "architecture": contract["architecture"],
            "architecture_label": contract["architecture_label"],
            "architecture_revision": contract["architecture_revision"],
            "architecture_inferred": False,
            "conditioning_route": contract["conditioning_route"],
            "num_visual_latents_per_camera": contract["num_visual_latents_per_camera"],
            "visual_perceiver_width": contract["visual_perceiver_width"],
            "action_loss_mode": contract["action_loss_mode"],
            "stage2_mode": contract["stage2_mode"],
            "dsbc_noise_output_mode": contract["dsbc_noise_output_mode"],
            "dsbc_frs_num_steps": contract["dsbc_frs_num_steps"],
            "dsbc_anchor_seed": contract["dsbc_anchor_seed"],
            "dsbc_reader": contract["dsbc_reader"],
            "dsbc_latent_predictor_enabled": contract[
                "dsbc_latent_predictor_enabled"
            ],
            "dsbc_latent_predictor_mode": contract[
                "dsbc_latent_predictor_mode"
            ],
            "dsbc_latent_supervision": contract[
                "dsbc_latent_supervision"
            ],
            "dsbc_latent_loss_weight": contract["dsbc_latent_loss_weight"],
            "dsbc_latent_timesteps": contract["dsbc_latent_timesteps"],
        }
        for mode in entry["modes"]:
            if mode == "stage2":
                resolved.append(
                    _panel_spec(
                        entry,
                        mode,
                        policy_path=stage2_path,
                        policy_config=stage2_policy,
                        architecture_fields=stage2_architecture_fields,
                        eval_data_fields=eval_data_fields,
                        tokenizer_path=eval_tokenizer_path,
                        predictor_path=predictor_path,
                        terminator_path=terminator_path,
                        terminator_variant=terminator_variant,
                    )
                )
                continue
            prior_key = (
                str(contract["stage1_prior_path"]),
                entry["skill_source"],
                entry["advance_mode"],
                str(predictor_path or ""),
                str(terminator_path or ""),
                terminator_variant,
            )
            prior_requests[prior_key] = prior_requests.get(prior_key, 0) + 1
            if prior_key in prior_specs:
                continue
            prior_contract = _stage1_prior_contract(
                contract["stage1_prior_path"], project_root
            )
            if prior_contract["proprio_grounding"] != contract["proprio_grounding"]:
                raise ValueError(
                    "Stage-2 checkpoint/prior proprio grounding mismatch: "
                    f"stage2={contract['proprio_grounding']!r}, "
                    f"prior={prior_contract['proprio_grounding']!r}."
                )
            prior_architecture_fields = {
                "architecture": prior_contract["architecture"],
                "architecture_label": prior_contract["architecture_label"],
                "architecture_revision": prior_contract["architecture_revision"],
                "architecture_inferred": prior_contract["architecture_inferred"],
                "conditioning_route": prior_contract.get(
                    "conditioning_route", "state_cond"
                ),
                "num_visual_latents_per_camera": prior_contract[
                    "num_visual_latents_per_camera"
                ],
                "visual_perceiver_width": prior_contract["visual_perceiver_width"],
                "action_loss_mode": prior_contract["action_loss_mode"],
            }
            prior_spec = _panel_spec(
                entry,
                mode,
                policy_path=contract["stage1_prior_path"],
                policy_config=prior_contract["policy"],
                architecture_fields=prior_architecture_fields,
                eval_data_fields=eval_data_fields,
                tokenizer_path=eval_tokenizer_path,
                predictor_path=predictor_path,
                terminator_path=terminator_path,
                terminator_variant=terminator_variant,
            )
            prior_specs[prior_key] = prior_spec
            resolved.append(prior_spec)

    shared_priors = [
        spec for key, spec in prior_specs.items() if prior_requests[key] > 1
    ]
    for index, spec in enumerate(shared_priors):
        spec["label"] = "prior" if len(shared_priors) == 1 else f"prior{index + 1}"

    labels = [spec["label"] for spec in resolved]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Resolved panel labels must be unique, got {labels}.")

    episode_exact = as_bool(_at(config, "oracle", "episode_exact", default=False))
    oracle_latent_specs = [
        spec for spec in resolved if spec.get("latent_source") == "oracle"
    ]
    if oracle_latent_specs and not episode_exact:
        raise ValueError(
            "latent_source=oracle requires oracle.episode_exact=true so GT "
            "actions can be matched to the selected source episode."
        )
    for spec in oracle_latent_specs:
        if spec.get("mode") != "stage2":
            raise ValueError("Oracle latent is supported only by Stage-2 panels.")
        if spec.get("stage2_mode") != "dsbc" or not spec.get(
            "dsbc_latent_predictor_enabled", False
        ):
            raise ValueError(
                f"{spec['label']} requests oracle latent but its checkpoint is "
                "not latent-enabled DSBC."
            )
    if episode_exact:
        init_state_paths = {spec["eval_init_states_path"].resolve() for spec in resolved}
        if len(init_state_paths) != 1:
            raise ValueError(
                "Multi-model episode-exact comparison requires the same source init-state map."
            )
        for spec in resolved:
            if not spec["eval_init_states_path"].is_file():
                raise FileNotFoundError(
                    "oracle.episode_exact=true requires "
                    f"{spec['eval_init_states_path']}. Build the oracle map first."
                )

    end_mode = str(_at(config, "terminator", "end_mode", default="or")).lower()
    if end_mode not in {"termination", "progress", "or", "and"}:
        raise ValueError("terminator.end_mode must be termination|progress|or|and.")
    gt_termination_min_fraction = float(
        _at(config, "terminator", "gt_termination_min_fraction", default=0.5)
    )
    if not 0.0 <= gt_termination_min_fraction <= 1.0:
        raise ValueError("terminator.gt_termination_min_fraction must be between 0 and 1.")
    n_action_steps = int(
        get_value(config, "n_action_steps", resolved[0]["policy"].get("n_action_steps", 10))
    )
    for spec in resolved:
        chunk_size = int(spec["policy"].get("chunk_size", n_action_steps))
        if not 1 <= n_action_steps <= chunk_size:
            raise ValueError(
                f"n_action_steps={n_action_steps} exceeds {spec['label']}'s "
                f"chunk_size={chunk_size}."
            )

    task_ids = get_value(config, "task_ids", list(range(10)))
    if isinstance(task_ids, str):
        task_ids = json.loads(task_ids)
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError("task_ids must be a non-empty JSON/YAML list.")
    task_ids = [int(task_id) for task_id in task_ids]
    target_task = str(get_value(config, "target_task", "libero_90"))
    env_task_ids = task_ids
    if target_task.startswith("langgap_"):
        env_task_ids = _langgap_env_task_ids(
            resolved[0]["eval_init_states_path"],
            task_ids,
            suite_name=target_task,
        )

    output_name = str(get_value(config, "output_name", "") or "").strip()
    output_name = _safe_name(
        output_name or _default_output_name(entries), field="output_name"
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
                for key, value in spec.items()
                if key != "policy"
            }
            for spec in resolved
        ],
        separators=(",", ":"),
    )
    primary = resolved[0]
    models_per_row = int(get_value(config, "models_per_row", 2) or 0)
    settings = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "models_json": models_json,
        "model_count": len(resolved),
        "grid_columns": models_per_row,
        "eval_resume": as_bool(get_value(config, "resume", False)),
        "policy_path": primary["policy_path"],
        "external_predictor_model": str(
            primary.get("external_predictor_model") or ""
        ),
        "external_terminator_model": str(
            primary.get("external_terminator_model") or ""
        ),
        "fsq_path": primary["fsq_path"],
        "skill_dataset_dir": primary["skill_dataset_dir"],
        "eval_init_states_path": primary["eval_init_states_path"] if episode_exact else "",
        "skill_latents_path": primary["skill_latents_path"],
        "raw_dataset_dir": primary["raw_dataset_dir"],
        "dino_model_path": primary["dino_model_path"],
        "tokenizer_path": primary["tokenizer_path"],
        "eval_out_dir": eval_outputs_root / output_name,
        "target_task": target_task,
        "dataset_task_ids": json.dumps(task_ids, separators=(",", ":")),
        "task_ids": json.dumps(env_task_ids, separators=(",", ":")),
        "eval_expected_tasks": len(env_task_ids),
        "eval_num_gpus": int(get_value(config, "eval_num_gpus", 1)),
        "eval_max_workers_per_gpu": int(
            get_value(config, "eval_max_workers_per_gpu", 4)
        ),
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
        "gt_termination_min_fraction": gt_termination_min_fraction,
        "terminator_variant": primary["terminator_variant"],
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
            _at(config, "logging", "wandb", "project", default="VLA_stage2_eval")
        ),
        "wandb_run_name": f"S2eval_{output_name}"
        + (f"_{os.environ['TASK_TAG']}" if os.environ.get("TASK_TAG") else ""),
    }
    if settings["n_episodes"] <= 0 or settings["eval_batch_size"] <= 0:
        raise ValueError("n_episodes and eval_batch_size must be positive.")
    if settings["eval_num_gpus"] <= 0:
        raise ValueError("eval_num_gpus must be positive.")
    if not 1 <= settings["eval_max_workers_per_gpu"] <= 4:
        raise ValueError("eval_max_workers_per_gpu must be between 1 and 4.")
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
