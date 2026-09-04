#!/usr/bin/env python3
"""Resolve the unified Stage-1 Arch0--4 YAML into shell exports."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import (  # noqa: E402
    as_bool,
    as_list,
    load_config,
    print_shell,
    resolve_path,
    resolve_skillvla_dataset_run,
)

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage1_train_config.yaml"
ARCHITECTURE_LABEL_TO_VISION_MODE = {
    "arch1_3": "uncompressed_visual_kv_self_attention",
    "arch2_1": "compressed_visual_kv_self_attention",
    "arch2_2": "interleaved_cross_attention",
    "arch3": "in_context_tokens",
    "arch4": "global_visual_adarms",
}
VSA_LABEL_TO_REVISION = {
    "arch1_3": "visual_kv_uncompressed_v1",
    "arch2_1": "visual_kv_perceiver_v1",
    "arch2_2": "interleaved_direct1024_v3",
    "arch3": "interleaved_direct1024_v3",
    "arch4": "interleaved_direct1024_v3",
}
COND_GEMMA_LABEL_TO_REVISION = {
    "arch0": "skillvla_real_v1",
    # Same parameter/state-dict contract as Arch0; the difference is the
    # training-only canonical skill-trajectory flow objective.
    "arch0_skill": "skillvla_real_v1",
    # Same Arch0 rollout graph, but the training-only route predicts an
    # extended current-frame action chunk instead of a canonical trajectory.
    "arch0_skill_chunk": "skillvla_real_v1",
    "arch0_1": "expert_state_adarms_v1",
    "arch0_2": "cond_expert_state_adarms_v1",
    # Same Arch0_2 rollout graph. Its auxiliary route keeps Expert-side state
    # AdaRMS while bypassing vision and Cond-Gemma.
    "arch0_2_skill_chunk": "cond_expert_state_adarms_v1",
    "arch0_2_sep": "cond_expert_separate_state_adarms_v1",
    "arch0_3": "wrist_cond_expert_state_adarms_v1",
    "arch0_adarms": "expert_skill_adarms_v1",
    "arch0_adarms_zero": "expert_skill_adarms_zero_v1",
    "arch0_token": "expert_skill_token_v1",
    "arch0_token_iso": "expert_skill_token_isolated_v1",
    "arch0_cond": "cond_skill_broadcast_v1",
    "arch0_both": "dual_skill_broadcast_v1",
    "arch1_1": "expert_tokens_uncompressed_v1",
    "arch1_2": "expert_tokens_perceiver_v1",
}


def _at(config: dict, *path: str, default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _local_model_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return Path(resolve_path(project_root, path))
    if path.exists() or "models" not in path.parts:
        return path
    return project_root.joinpath(*path.parts[path.parts.index("models") :])


def _read_dataset_contract(dataset_dir: Path, run_tag: str) -> dict:
    """Read skill geometry and jitter metadata without opening an FSQ checkpoint."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Stage-1 dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    levels = [int(level) for level in info.get("skill_fsq_levels", [])]
    if not levels or any(level <= 1 for level in levels):
        raise ValueError(f"Invalid skill_fsq_levels in {info_path}: {levels}")
    match = re.search(r"FSQ(\d+)", run_tag)
    if match and [int(digit) for digit in match.group(1)] != levels:
        raise ValueError(
            f"Dataset run says FSQ{match.group(1)}, but {info_path} says levels={levels}."
        )
    features = info.get("features", {})
    state_dim = int(features["observation.state"]["shape"][0])
    action_dim = int(features["action"]["shape"][0])
    jitter_pmax = int(info.get("skill_pmax", 0))
    directional_jitter = {
        name: int(info.get(f"skill_jitter_{name}_pmax", jitter_pmax))
        for name in ("early_start", "late_start", "early_end", "late_end")
    }
    if any(value < 0 or value > jitter_pmax for value in directional_jitter.values()):
        raise ValueError(
            f"Invalid directional jitter contract in {info_path}: "
            f"storage={jitter_pmax}, directional={directional_jitter}."
        )
    proprio_grounding = str(
        info.get("proprio_grounding", "none") or "none"
    ).strip().lower()
    if proprio_grounding not in {"none", "episode_start_xyz"}:
        raise ValueError(
            f"Invalid proprio_grounding in {info_path}: {proprio_grounding!r}."
        )
    return {
        "levels": levels,
        "skill_code_space_id": str(
            info.get("skill_code_space_id", run_tag) or run_tag
        ).strip(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "proprio_grounding": proprio_grounding,
        "skill_observed_max_length": int(
            info.get("skill_observed_max_length", 0)
        ),
        "jitter_pmax": jitter_pmax,
        "jitter_early_start_pmax": directional_jitter["early_start"],
        "jitter_late_start_pmax": directional_jitter["late_start"],
        "jitter_early_end_pmax": directional_jitter["early_end"],
        "jitter_late_end_pmax": directional_jitter["late_end"],
        "jitter_distribution": str(
            info.get("skill_jitter_distribution", "half_normal")
        ).replace("-", "_"),
    }


_PREDICTOR_CONTRACT_FIELDS = (
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


def _predictor_contract_from_checkpoint(
    checkpoint: Path,
    *,
    levels: list[int],
) -> dict:
    """Read the frozen predictor contract; Stage1 never trains this module."""
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-1 predictor config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Stage-1 predictor weights not found: {weights_path}")
    source = json.loads(config_path.read_text())
    if source.get("type") not in {"skill_expert", "skill_aux"}:
        raise ValueError(
            "Predictor checkpoint must be policy.type=skill_expert or skill_aux, got "
            f"{source.get('type')!r} at {checkpoint}."
        )
    if not source.get("train_skill_predictor", False):
        raise ValueError(f"Stage-1 checkpoint has no trained predictor: {checkpoint}")

    expected_geometry = {
        "skill_fsq_levels": levels,
        "skill_vocab_size": math.prod(levels),
    }
    mismatches = [
        f"{field}: checkpoint={source.get(field)!r}, dataset={value!r}"
        for field, value in expected_geometry.items()
        if source.get(field) != value
    ]
    if mismatches:
        raise ValueError("Predictor checkpoint contract mismatch: " + "; ".join(mismatches))
    missing = [field for field in _PREDICTOR_CONTRACT_FIELDS if field not in source]
    if missing:
        raise ValueError(
            "Predictor checkpoint does not record its complete frozen contract: "
            f"missing={missing} at {checkpoint}."
        )
    return {field: source[field] for field in _PREDICTOR_CONTRACT_FIELDS}


def build_settings(config: dict) -> dict:
    removed_sections = {"skill_predictor", "terminator"} & set(config)
    if removed_sections:
        raise ValueError(
            "Stage1 trains only the action model; remove co-training sections: "
            f"{sorted(removed_sections)}."
        )
    run_config = config.get("run", {})
    if not isinstance(run_config, dict):
        raise ValueError("run must be a mapping containing an optional suffix.")
    unknown_run_keys = set(run_config) - {"suffix"}
    if unknown_run_keys:
        raise ValueError(f"Unsupported run settings: {sorted(unknown_run_keys)}.")
    run_suffix = str(run_config.get("suffix", "") or "").strip().strip("_")
    if run_suffix and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_suffix) is None:
        raise ValueError(
            "run.suffix may contain only letters, numbers, '.', '_' and '-', "
            "and must start with a letter or number."
        )
    project_root = Path(str(config["project_root"])).expanduser()
    dataset_root = project_root / str(config.get("dataset_root", "dataset"))
    outputs_root = project_root / str(config.get("outputs_root", "outputs"))
    source = str(_at(config, "dataset", "source"))
    base_run_tag = str(_at(config, "dataset", "run"))
    skillvla_root = dataset_root / str(
        _at(config, "dataset", "skillvla_root", default="skillvla_dataset")
    )
    run_tag, dataset_relabeled = resolve_skillvla_dataset_run(
        skillvla_root / source,
        base_run_tag,
        _at(config, "dataset", "relabeled", default=""),
    )
    dataset_dir = skillvla_root / source / run_tag / "skillvla"
    contract = _read_dataset_contract(dataset_dir, run_tag)

    training_steps = int(
        _at(config, "training", "schedule", "steps", default=50000)
    )
    scheduler_mode = str(
        _at(
            config,
            "training",
            "schedule",
            "lr_mode",
            default="cosine_decay",
        )
    ).strip().lower()
    scheduler_warmup_steps = int(
        _at(config, "training", "schedule", "warmup_steps", default=1000)
    )
    scheduler_decay_steps = int(
        _at(config, "training", "schedule", "lr_decay_steps", default=30000)
    )
    if training_steps <= 0:
        raise ValueError("training.schedule.steps must be positive.")
    if scheduler_mode not in {"cosine_decay", "warmup_constant"}:
        raise ValueError(
            "training.schedule.lr_mode must be 'cosine_decay' or "
            f"'warmup_constant', got {scheduler_mode!r}."
        )
    if scheduler_warmup_steps < 0:
        raise ValueError("training.schedule.warmup_steps must be non-negative.")
    if scheduler_decay_steps <= 0:
        raise ValueError("training.schedule.lr_decay_steps must be positive.")

    pi_base = _local_model_path(
        project_root, str(_at(config, "warm_start", "pi_base", default="models/pi05_base"))
    )
    tokenizer_path = _local_model_path(
        project_root,
        str(
            _at(
                config,
                "warm_start",
                "tokenizer",
                default="models/paligemma-3b-pt-224-tokenizer",
            )
        ),
    )
    dino_model = _local_model_path(
        project_root, str(_at(config, "vision", "dino_model", default="models/dinov3-vitl16"))
    )
    fsq_path = dataset_dir.parent / "FSQ.pt"
    if not (pi_base / "model.safetensors").is_file():
        raise FileNotFoundError(f"pi0.5 base checkpoint not found: {pi_base}")
    if not dino_model.is_dir():
        raise FileNotFoundError(f"DINO model not found: {dino_model}")
    training_skill_source = str(
        _at(config, "action_conditioning", "training_skill_source", default="gt")
    ).strip().lower()
    if training_skill_source not in {"gt", "predictor"}:
        raise ValueError(
            "action_conditioning.training_skill_source must be gt or predictor."
        )
    predictor_checkpoint_value = str(
        _at(config, "warm_start", "predictor_checkpoint", default="") or ""
    ).strip()
    predictor_checkpoint = (
        Path(resolve_path(project_root, predictor_checkpoint_value))
        if predictor_checkpoint_value
        else None
    )
    if training_skill_source == "predictor" and predictor_checkpoint is None:
        raise ValueError(
            "training_skill_source=predictor requires warm_start.predictor_checkpoint."
        )
    uses_skill_predictor = training_skill_source == "predictor"
    if uses_skill_predictor:
        required = ("config.json", "tokenizer_config.json", "tokenizer.json")
        missing = [name for name in required if not (tokenizer_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Local PaliGemma tokenizer is incomplete at {tokenizer_path}: missing {missing}."
            )
    freeze_vision_encoder = as_bool(
        _at(config, "vision", "freeze", default=False)
    )
    expert_variant = str(
        _at(config, "architecture", "expert_variant", default="gemma_300m")
    )
    architecture_label = str(
        _at(config, "architecture", "name", default="arch2_2")
    ).strip().lower()
    architecture_config = config.get("architecture", {})
    if not isinstance(architecture_config, dict):
        raise ValueError("architecture must be a mapping.")
    misplaced_keys = {
        "arch0",
        "arch0_skill",
        "arch0_skill_chunk",
        "arch0_1",
        "arch0_2",
        "arch0_2_skill_chunk",
        "arch0_2_sep",
        "arch0_3",
        "arch0_adarms",
        "arch0_adarms_zero",
        "arch0_token",
        "arch0_token_iso",
        "arch0_cond",
        "arch0_both",
        "arch1",
        "arch1_1",
        "arch1_2",
        "arch1_3",
        "cond_variant",
        "conditioning_route",
        "vision_conditioning_mode",
        "visual_latents_per_camera",
    } & set(architecture_config)
    if misplaced_keys:
        raise ValueError(
            "Arch0--0_3/Arch1_1/Arch1_2 are fixed Cond-Gemma ablations, while VSA-only "
            "settings belong under architecture.vsa; remove architecture keys: "
            f"{sorted(misplaced_keys)}."
        )
    if architecture_label == "arch2":
        raise ValueError(
            "architecture.name='arch2' was split into arch2_1 and arch2_2; "
            "use arch2_2 for the unchanged alternating cross-attention model."
        )
    if architecture_label == "arch1":
        raise ValueError(
            "architecture.name='arch1' was split into arch0, arch1_1, and arch1_2; "
            "use arch0 for the unchanged skillVLA_real Cond-Gemma baseline."
        )
    if architecture_label not in {
        *COND_GEMMA_LABEL_TO_REVISION,
        *ARCHITECTURE_LABEL_TO_VISION_MODE,
    }:
        raise ValueError(
            "architecture.name must be "
            "arch0|arch0_skill|arch0_skill_chunk|arch0_1|arch0_2|arch0_2_skill_chunk|"
            "arch0_2_sep|arch0_3|arch0_adaRMS|arch0_adaRMS_zero|"
            "arch0_token|arch0_token_iso|arch0_cond|arch0_both|"
            "arch1_1|arch1_2|arch1_3|"
            "arch2_1|arch2_2|arch3|arch4, got "
            f"{architecture_label!r}."
        )
    vsa_config = _at(config, "architecture", "vsa", default={})
    if not isinstance(vsa_config, dict):
        raise ValueError("architecture.vsa must be a mapping.")

    if architecture_label in COND_GEMMA_LABEL_TO_REVISION:
        architecture = "cond_gemma"
        architecture_revision = COND_GEMMA_LABEL_TO_REVISION[architecture_label]
        cond_variant = expert_variant
        # Arch0--0_3 and Arch0_adaRMS keep the uncompressed visual Cond-Gemma
        # path and inject the motion-level skill directly into the action expert
        # (layerwise broadcast, or expert AdaRMS for Arch0_adaRMS).
        # Arch1_1/Arch1_2 instead use explicit expert state/skill tokens.
        conditioning_route = (
            "state_cond"
            if architecture_label.startswith("arch0")
            else "state_skill_cond"
        )
        vision_conditioning_mode = "interleaved_cross_attention"
        include_state_in_visual_crossattn = True
        include_skill_in_visual_crossattn = True
        num_visual_latents_per_camera = int(
            vsa_config.get("visual_latents_per_camera", 32)
        )
        if architecture_label == "arch1_2" and not (
            1 <= num_visual_latents_per_camera <= 197
        ):
            raise ValueError(
                "architecture.vsa.visual_latents_per_camera must be between 1 and 197."
            )
        visual_crossattn_query_label = "not used (Cond Gemma family)"
    else:
        architecture = "vsa_perceiver_crossattn"
        architecture_revision = VSA_LABEL_TO_REVISION[architecture_label]
        cond_variant = expert_variant
        conditioning_route = "state_skill_cond"
        if freeze_vision_encoder:
            raise ValueError(
                "vision.freeze=true is unsupported for Arch1_3--4; their shared DINO "
                "must be trainable."
            )
        fixed_query_keys = {
            "include_state_in_visual_crossattn",
            "include_skill_in_visual_crossattn",
        } & set(vsa_config)
        if fixed_query_keys:
            raise ValueError(
                "VSA visual conditioning is fixed by architecture.name; remove "
                f"the YAML keys {sorted(fixed_query_keys)}."
            )
        vision_conditioning_mode = ARCHITECTURE_LABEL_TO_VISION_MODE[
            architecture_label
        ]
        include_state_in_visual_crossattn = True
        include_skill_in_visual_crossattn = True
        if vision_conditioning_mode == "interleaved_cross_attention":
            visual_crossattn_query_label = "state + skill + action"
        elif vision_conditioning_mode in {
            "uncompressed_visual_kv_self_attention",
            "compressed_visual_kv_self_attention",
        }:
            visual_crossattn_query_label = "expert queries; visual fixed KV"
        else:
            visual_crossattn_query_label = "ignored"
        num_visual_latents_per_camera = (
            197
            if architecture_label == "arch1_3"
            else int(vsa_config.get("visual_latents_per_camera", 32))
        )
        if not 1 <= num_visual_latents_per_camera <= 197:
            raise ValueError(
                "architecture.vsa.visual_latents_per_camera must be between 1 and 197."
            )

    max_state_dim = int(_at(config, "architecture", "max_state_dim", default=32))
    max_action_dim = int(_at(config, "architecture", "max_action_dim", default=32))
    if contract["state_dim"] > max_state_dim or contract["action_dim"] > max_action_dim:
        raise ValueError(
            "Dataset state/action dimensions exceed the configured pi0.5 projections: "
            f"dataset=({contract['state_dim']}, {contract['action_dim']}), "
            f"configured=({max_state_dim}, {max_action_dim})."
        )

    chunk_size = int(_at(config, "architecture", "chunk_size", default=10))
    if "loss" in config:
        raise ValueError(
            "Stage1 uses a fixed flow objective; remove the legacy top-level "
            "loss key and control the trajectory auxiliary with "
            "cumulative_xyz_loss.enabled."
        )
    mask_actions_after_skill_end = as_bool(
        config.get("mask_actions_after_skill_end", False)
    )
    cumulative_xyz_config = config.get("cumulative_xyz_loss", {})
    if not isinstance(cumulative_xyz_config, dict):
        raise ValueError("cumulative_xyz_loss must be a mapping.")
    cumulative_xyz_loss_enabled = as_bool(
        cumulative_xyz_config.get("enabled", False)
    )
    cumulative_xyz_loss_weight = float(cumulative_xyz_config.get("weight", 0.5))
    if not math.isfinite(cumulative_xyz_loss_weight) or cumulative_xyz_loss_weight <= 0:
        raise ValueError("cumulative_xyz_loss.weight must be finite and positive.")
    skill_flow_config = config.get("skill_flow", {})
    if not isinstance(skill_flow_config, dict):
        raise ValueError("skill_flow must be a mapping.")
    unknown_skill_flow_keys = set(skill_flow_config) - {
        "weight",
        "chunk_multiplier",
        "latent_best_of_n",
    }
    if unknown_skill_flow_keys:
        raise ValueError(
            f"Unsupported skill_flow settings: {sorted(unknown_skill_flow_keys)}."
        )
    skill_flow_enabled = architecture_label in {
        "arch0_skill",
        "arch0_skill_chunk",
        "arch0_2_skill_chunk",
    }
    skill_flow_weight = float(skill_flow_config.get("weight", 1.0))
    if not math.isfinite(skill_flow_weight) or skill_flow_weight <= 0:
        raise ValueError("skill_flow.weight must be finite and positive.")
    skill_flow_chunk_multiplier = int(skill_flow_config.get("chunk_multiplier", 3))
    if skill_flow_chunk_multiplier <= 0:
        raise ValueError("skill_flow.chunk_multiplier must be positive.")
    latent_best_of_n = skill_flow_config.get("latent_best_of_n", {})
    if not isinstance(latent_best_of_n, dict):
        raise ValueError("skill_flow.latent_best_of_n must be a mapping.")
    unknown_latent_keys = set(latent_best_of_n) - {
        "enabled",
        "candidates",
        "top_k",
        "timesteps",
        "ranking",
        "fp32",
    }
    if unknown_latent_keys:
        raise ValueError(
            "Unsupported skill_flow.latent_best_of_n settings: "
            f"{sorted(unknown_latent_keys)}."
        )
    skill_flow_latent_best_of_n_enabled = as_bool(
        latent_best_of_n.get("enabled", False)
    )
    skill_flow_latent_candidates = int(latent_best_of_n.get("candidates", 5))
    skill_flow_latent_top_k = int(latent_best_of_n.get("top_k", 1))
    skill_flow_latent_assignment_timesteps = int(
        latent_best_of_n.get("timesteps", 2)
    )
    skill_flow_latent_ranking_route = str(
        latent_best_of_n.get("ranking", "main")
    ).strip().lower().replace("-", "_")
    skill_flow_latent_fp32 = as_bool(latent_best_of_n.get("fp32", False))
    if skill_flow_latent_candidates <= 0:
        raise ValueError("skill_flow.latent_best_of_n.candidates must be positive.")
    if not 1 <= skill_flow_latent_top_k <= skill_flow_latent_candidates:
        raise ValueError(
            "skill_flow.latent_best_of_n.top_k must be within [1, candidates]."
        )
    if skill_flow_latent_assignment_timesteps <= 0:
        raise ValueError("skill_flow.latent_best_of_n.timesteps must be positive.")
    if skill_flow_latent_ranking_route not in {"main", "skill_only"}:
        raise ValueError(
            "skill_flow.latent_best_of_n.ranking must be main|skill_only."
        )
    if skill_flow_latent_best_of_n_enabled and architecture_label not in {
        "arch0_skill",
        "arch0_skill_chunk",
        "arch0_2_skill_chunk",
    }:
        raise ValueError(
            "skill_flow.latent_best_of_n is supported only for "
            "architecture.name=arch0_skill|arch0_skill_chunk|arch0_2_skill_chunk."
        )
    if skill_flow_latent_fp32 and not skill_flow_latent_best_of_n_enabled:
        raise ValueError(
            "skill_flow.latent_best_of_n.fp32 requires enabled: true."
        )
    skill_flow_target = (
        "extended_chunk"
        if architecture_label in {"arch0_skill_chunk", "arch0_2_skill_chunk"}
        else "canonical"
    )
    skill_flow_state_conditioned = architecture_label == "arch0_2_skill_chunk"
    skill_flow_max_length = (
        int(contract["skill_observed_max_length"])
        if skill_flow_target == "canonical"
        else chunk_size * skill_flow_chunk_multiplier
    )
    if architecture_label == "arch0_skill":
        if training_skill_source != "gt":
            raise ValueError(
                "architecture.name=arch0_skill currently requires "
                "action_conditioning.training_skill_source=gt."
            )
        if skill_flow_max_length <= 0:
            raise ValueError(
                "architecture.name=arch0_skill requires a positive "
                "skill_observed_max_length in the dataset info.json."
            )
    n_action_steps = int(
        _at(config, "execution", "action_steps", default=chunk_size)
    )
    transition_jitter = as_bool(
        _at(config, "transition_randomization", "enabled", default=True)
    )
    jitter_pmax = contract["jitter_pmax"] if transition_jitter else 0
    directional_jitter = {
        name: contract[f"jitter_{name}_pmax"] if transition_jitter else 0
        for name in ("early_start", "late_start", "early_end", "late_end")
    }
    jitter_distribution = contract["jitter_distribution"] if transition_jitter else "half_normal"
    if jitter_distribution not in {"half_normal", "uniform"}:
        raise ValueError(f"Unsupported dataset jitter distribution: {jitter_distribution!r}")

    batch_size = int(_at(config, "training", "dataloader", "batch_size", default=16))
    num_gpus = int(_at(config, "training", "dataloader", "gpus", default=1))
    base_lr = float(_at(config, "training", "optimizer", "base_lr", default=2.5e-5))
    optimizer_config = _at(config, "training", "optimizer", default={})
    if "dino_lr" in optimizer_config:
        raise ValueError(
            "training.optimizer.dino_lr was replaced by the relative dino_lr_scale."
        )
    if "terminator_lr_scale" in optimizer_config:
        raise ValueError(
            "training.optimizer.terminator_lr_scale was removed; Stage1 trains "
            "only the action model."
        )
    dino_lr_scale = float(
        _at(config, "training", "optimizer", "dino_lr_scale", default=0.1)
    )
    if dino_lr_scale <= 0.0:
        raise ValueError("training.optimizer.dino_lr_scale must be positive.")
    # Muon probe: 2D hidden matrices switch to Muon (AdamW keeps the rest).
    # base_lr/dino_lr_scale are reused unchanged (match_rms_adamw scaling).
    use_muon = as_bool(_at(config, "training", "optimizer", "muon", default=False))

    predictor_contract = {
        "skill_predictor_vlm_variant": "gemma_2b",
        "skill_predictor_image_size": 224,
        "skill_predictor_reader_tokens": 4,
        "skill_predictor_reader_depth": 2,
        "skill_predictor_reader_heads": 8,
        "skill_predictor_all_layers": False,
        "skill_predictor_detach_vlm": True,
        "skill_predictor_lora": False,
        "skill_predictor_lora_targets": "q,k,v,o",
        "skill_predictor_lora_rank": 8,
        "skill_predictor_lora_alpha": 16.0,
        "skill_predictor_lora_dropout": 0.0,
        "skill_predictor_deadzone_frac": 0.0,
        "skill_predictor_attend_image": True,
        "skill_predictor_attend_language": True,
        "tokenizer_max_length": 200,
    }
    if predictor_checkpoint is not None:
        predictor_contract = _predictor_contract_from_checkpoint(
            predictor_checkpoint,
            levels=contract["levels"],
        )
    run_name = f"bs{batch_size}_{source}_{run_tag}_{architecture_label}"
    if training_skill_source == "predictor":
        run_name = f"{run_name}_pretrained_predictor"
    if mask_actions_after_skill_end:
        run_name = f"{run_name}_skillendmask"
    if cumulative_xyz_loss_enabled:
        cumulative_weight_label = f"{cumulative_xyz_loss_weight:g}".replace(
            ".", "p"
        )
        run_name = f"{run_name}_cumxyz{cumulative_weight_label}"
    if skill_flow_latent_best_of_n_enabled:
        run_name = (
            f"{run_name}_zbest{skill_flow_latent_candidates}"
            f"k{skill_flow_latent_top_k}m{skill_flow_latent_assignment_timesteps}"
        )
        if skill_flow_latent_ranking_route == "main":
            run_name = f"{run_name}_rank"
        if skill_flow_latent_fp32:
            run_name = f"{run_name}_zfp32"
    if use_muon:
        # Muon A/B runs must never collide with the AdamW output directory.
        run_name = f"{run_name}_muon"
    if run_suffix:
        # The user suffix is always last so the automatic batch/architecture
        # naming contract stays machine-readable.
        run_name = f"{run_name}_{run_suffix}"

    levels = contract["levels"]
    training_config = config.get("training", {})
    if "vsa_debug_schedule" in training_config or "vsa_debug_steps" in training_config:
        raise ValueError(
            "training.vsa_debug_schedule/vsa_debug_steps were replaced by "
            "training.vsa_debug.{every,initial}."
        )
    vsa_debug_config = training_config.get("vsa_debug", {})
    if not isinstance(vsa_debug_config, dict):
        raise ValueError("training.vsa_debug must be a mapping.")
    vsa_debug_every = int(vsa_debug_config.get("every", 0))
    vsa_debug_initial = [
        int(step) for step in as_list(vsa_debug_config.get("initial", []))
    ]
    if vsa_debug_every < 0:
        raise ValueError("training.vsa_debug.every must be non-negative.")
    if any(step <= 0 for step in vsa_debug_initial):
        raise ValueError("training.vsa_debug.initial entries must be positive.")
    if sorted(set(vsa_debug_initial)) != vsa_debug_initial:
        raise ValueError(
            "training.vsa_debug.initial must be sorted and contain no duplicates."
        )
    periodic_debug_steps = (
        range(vsa_debug_every, training_steps + 1, vsa_debug_every)
        if vsa_debug_every > 0
        else ()
    )
    vsa_debug_schedule = sorted(
        set(vsa_debug_initial).union(periodic_debug_steps)
    )
    settings = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "skillvla_dataset_dir": dataset_dir,
        "dataset_relabeled": dataset_relabeled,
        "repo_id": f"dohyeon/{source}",
        "pi_base": pi_base,
        "fsq_path": fsq_path,
        "tokenizer_path": tokenizer_path,
        "dino_model_path": dino_model,
        "dino_image_size": int(_at(config, "vision", "image_size", default=224)),
        "freeze_vision_encoder": freeze_vision_encoder,
        "dino_lr_scale": dino_lr_scale,
        "architecture": architecture,
        "architecture_label": architecture_label,
        "architecture_revision": architecture_revision,
        "vision_conditioning_mode": vision_conditioning_mode,
        "include_state_in_visual_crossattn": include_state_in_visual_crossattn,
        "include_skill_in_visual_crossattn": include_skill_in_visual_crossattn,
        "num_visual_latents_per_camera": num_visual_latents_per_camera,
        "visual_perceiver_width": 1024,
        "visual_crossattn_queries": visual_crossattn_query_label,
        "action_expert_variant": expert_variant,
        "cond_encoder_variant": cond_variant,
        "conditioning_route": conditioning_route,
        "skill_fsq_levels": "[" + ",".join(str(level) for level in levels) + "]",
        "skill_vocab_size": math.prod(levels),
        "skill_code_space_id": contract["skill_code_space_id"],
        "transition_jitter_pmax": jitter_pmax,
        "transition_jitter_early_start_pmax": directional_jitter["early_start"],
        "transition_jitter_late_start_pmax": directional_jitter["late_start"],
        "transition_jitter_early_end_pmax": directional_jitter["early_end"],
        "transition_jitter_late_end_pmax": directional_jitter["late_end"],
        "transition_jitter_distribution": jitter_distribution,
        "training_skill_source": training_skill_source,
        "skill_predictor_checkpoint_path": predictor_checkpoint or "",
        "skill_predictor_all_layers": predictor_contract["skill_predictor_all_layers"],
        "skill_predictor_detach_vlm": predictor_contract["skill_predictor_detach_vlm"],
        "skill_predictor_lora": predictor_contract["skill_predictor_lora"],
        "skill_predictor_lora_targets": predictor_contract["skill_predictor_lora_targets"],
        "skill_predictor_lora_rank": predictor_contract["skill_predictor_lora_rank"],
        "skill_predictor_lora_alpha": predictor_contract["skill_predictor_lora_alpha"],
        "skill_predictor_lora_dropout": predictor_contract["skill_predictor_lora_dropout"],
        "skill_predictor_reader_tokens": predictor_contract["skill_predictor_reader_tokens"],
        "skill_predictor_reader_depth": predictor_contract["skill_predictor_reader_depth"],
        "skill_predictor_reader_heads": predictor_contract["skill_predictor_reader_heads"],
        "skill_predictor_deadzone_frac": predictor_contract["skill_predictor_deadzone_frac"],
        "skill_predictor_attend_image": predictor_contract["skill_predictor_attend_image"],
        "skill_predictor_attend_language": predictor_contract["skill_predictor_attend_language"],
        "tokenizer_max_length": predictor_contract["tokenizer_max_length"],
        "max_state_dim": max_state_dim,
        "max_action_dim": max_action_dim,
        "proprio_grounding": contract["proprio_grounding"],
        "chunk_size": chunk_size,
        "mask_actions_after_skill_end": mask_actions_after_skill_end,
        "cumulative_xyz_loss_enabled": cumulative_xyz_loss_enabled,
        "cumulative_xyz_loss_weight": cumulative_xyz_loss_weight,
        "skill_flow_enabled": skill_flow_enabled,
        "skill_flow_weight": skill_flow_weight,
        "skill_flow_max_length": skill_flow_max_length if skill_flow_enabled else 0,
        "skill_flow_target": skill_flow_target,
        "skill_flow_state_conditioned": skill_flow_state_conditioned,
        "skill_flow_chunk_multiplier": skill_flow_chunk_multiplier,
        "skill_flow_latent_best_of_n_enabled": skill_flow_latent_best_of_n_enabled,
        "skill_flow_latent_candidates": skill_flow_latent_candidates,
        "skill_flow_latent_top_k": skill_flow_latent_top_k,
        "skill_flow_latent_assignment_timesteps": skill_flow_latent_assignment_timesteps,
        "skill_flow_latent_ranking_route": skill_flow_latent_ranking_route,
        "skill_flow_latent_fp32": skill_flow_latent_fp32,
        # The probe deliberately fixes geometry/scale to avoid extra tuning knobs.
        "skill_flow_latent_dim": 2,
        "skill_flow_latent_distribution": "uniform_square",
        "skill_flow_latent_gain_init": 0.1,
        "n_action_steps": n_action_steps,
        "min_period": float(_at(config, "flow", "min_period", default=4e-3)),
        "max_period": float(_at(config, "flow", "max_period", default=4.0)),
        "time_sampling_beta_alpha": float(
            _at(config, "flow", "beta_alpha", default=1.5)
        ),
        "time_sampling_beta_beta": float(
            _at(config, "flow", "beta_beta", default=1.0)
        ),
        "time_sampling_scale": float(_at(config, "flow", "scale", default=0.999)),
        "time_sampling_offset": float(_at(config, "flow", "offset", default=0.001)),
        "pt_run_name": run_name,
        "pt_output_dir": outputs_root / "skillVLA_stage1" / run_name,
        "batch_size": batch_size,
        "num_workers": int(
            _at(config, "training", "dataloader", "workers", default=2)
        ),
        "num_gpus": num_gpus,
        "gradient_checkpointing": as_bool(
            _at(config, "training", "gradient_checkpointing", default=True)
        ),
        "vsa_debug_steps": 0,
        "vsa_debug_schedule": "["
        + ",".join(str(step) for step in vsa_debug_schedule)
        + "]",
        "lr": base_lr * num_gpus,
        "use_muon": use_muon,
        "steps": training_steps,
        "scheduler_mode": scheduler_mode,
        "scheduler_warmup_steps": scheduler_warmup_steps,
        "scheduler_decay_steps": scheduler_decay_steps,
        "log_freq": int(
            _at(config, "training", "schedule", "log_every", default=100)
        ),
        "save_freq": int(
            _at(config, "training", "schedule", "save_every", default=5000)
        ),
        "wandb_enable": as_bool(
            _at(config, "logging", "wandb", "enable", default=True)
        ),
        "wandb_project": str(
            _at(config, "logging", "wandb", "project", default="VLA_stage1")
        ),
        "train_partition": ",".join(as_list(config.get("train_partition", ["big"]))) or "big",
        "train_qos": str(config.get("train_qos", "big_qos")),
        "train_gres": str(_at(config, "slurm", "gres", default="gpu:1")),
        "train_cpus_per_task": int(_at(config, "slurm", "cpus", default=16)),
        "train_mem": str(_at(config, "slurm", "memory", default="256G")),
        "train_time": str(_at(config, "slurm", "time", default="48:00:00")),
        "train_nodelist": str(config.get("train_nodelist", "")),
        "train_exclude_nodes": ",".join(as_list(config.get("train_exclude_nodes", []))),
    }
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--architecture",
        # architecture.name is lowercased when the YAML is resolved, so accept
        # the arch0_adaRMS spelling used in the docs and submit_train.sh too.
        type=str.lower,
        choices=(
            "arch0",
            "arch0_skill",
            "arch0_skill_chunk",
            "arch0_1",
            "arch0_2",
            "arch0_2_skill_chunk",
            "arch0_2_sep",
            "arch0_3",
            "arch0_adarms",
            "arch0_adarms_zero",
            "arch0_token",
            "arch0_token_iso",
            "arch0_cond",
            "arch0_both",
            "arch1_1",
            "arch1_2",
            "arch1_3",
            "arch2_1",
            "arch2_2",
            "arch3",
            "arch4",
        ),
        help="Override architecture.name without editing the YAML.",
    )
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.architecture:
        config.setdefault("architecture", {})["name"] = args.architecture
    settings = build_settings(config)
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
