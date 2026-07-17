#!/usr/bin/env python3
"""Shared config helpers for train_skills scripts."""

from __future__ import annotations

import argparse
import ast
import math
import os
import shlex
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "train_skills_config.yaml"


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if " #" in text:
        text = text.split(" #", 1)[0].rstrip()
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text.strip("\"'")


def _load_flat_yaml(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- ") and current_list_key is not None:
            out[current_list_key].append(_parse_scalar(line[2:]))
            continue
        if ":" not in line:
            current_list_key = None
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            out[key] = []
            current_list_key = key
        else:
            out[key] = _parse_scalar(value)
            current_list_key = None
    return out


GLOBAL_CONFIG_NAME = "global_config.yaml"


def _read_yaml(config_path: Path) -> dict[str, Any]:
    try:
        import yaml

        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return _load_flat_yaml(config_path)


def _find_global(start: Path) -> Path | None:
    """Walk up from `start` to find the nearest global_config.yaml (stops at first hit)."""
    for d in [start.resolve(), *start.resolve().parents]:
        candidate = d / GLOBAL_CONFIG_NAME
        if candidate.exists():
            return candidate
    return None


def _merge_global(config_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge the nearest global_config.yaml as a base (module cfg keys win)."""
    gpath = _find_global(config_path.parent)
    if gpath is None or gpath.resolve() == config_path.resolve():
        return cfg
    return {**_read_yaml(gpath), **cfg}


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return _merge_global(config_path, _read_yaml(config_path))


def get_value(cfg: dict[str, Any], key: str, default: Any = None, env: str | None = None) -> Any:
    env_key = env or key.upper()
    if env_key in os.environ:
        return os.environ[env_key]
    return cfg.get(key, default)


def resolve_path(project_root: "Path | str", value: Any, default: str = "") -> str:
    """Resolve a config path against project_root for cross-server portability.

    Absolute path → used as-is; relative/bare name → joined under project_root; blank → "".
    Lets configs store e.g. ``models/pi05_base`` (relative) and work on any machine — project_root
    comes from global_config — instead of hardcoding one server's absolute path."""
    s = str(value if value not in (None, "", "null") else default).strip()
    if not s:
        return ""
    p = Path(s).expanduser()
    return str(p if p.is_absolute() else (Path(project_root) / p))


def resolve_skillset_threshold_mode(cfg: dict[str, Any], project_root: Path) -> str:
    """Read the mode from this module, or inherit build_data's selection.

    The build config is the user-facing source of truth.  FSQ/eval configs omit
    this key deliberately, so changing only build_data_config.yaml also makes
    their resolver point at the matching `seg_*` directory.
    """
    value = get_value(cfg, "skillset_boundary_threshold_mode", None)
    if value is None:
        build_cfg = (
            project_root / "lerobot" / "examples" / "libero" / "configs"
            / "train_skills" / "build_data" / "build_data_config.yaml"
        )
        if build_cfg.is_file():
            value = get_value(load_config(build_cfg), "skillset_boundary_threshold_mode", "episode_mean")
    return str(value if value is not None else "episode_mean").strip().lower()


def resolve_skillset_global_threshold_source(cfg: dict[str, Any], project_root: Path) -> str:
    """Resolve an optional fixed global-threshold JSON, inheriting build_data.

    A source is intentionally separate from ``global_mean``: the latter normally
    recomputes a mean for the dataset being built, while a non-empty source
    freezes a previously established cross-task threshold.
    """
    value = get_value(cfg, "skillset_global_threshold_source", None)
    if value is None:
        build_cfg = (
            project_root / "lerobot" / "examples" / "libero" / "configs"
            / "train_skills" / "build_data" / "build_data_config.yaml"
        )
        if build_cfg.is_file():
            value = get_value(load_config(build_cfg), "skillset_global_threshold_source", "")
    return resolve_path(project_root, value)


def resolve_skillset_output_suffix(cfg: dict[str, Any], project_root: Path) -> str:
    """Resolve an optional, manually chosen experiment suffix for a skillset."""
    value = get_value(cfg, "skillset_output_suffix", None)
    if value is None:
        build_cfg = (
            project_root / "lerobot" / "examples" / "libero" / "configs"
            / "train_skills" / "build_data" / "build_data_config.yaml"
        )
        if build_cfg.is_file():
            value = get_value(load_config(build_cfg), "skillset_output_suffix", "")
    raw = str(value if value is not None else "").strip()
    if not raw:
        return ""
    tag = raw[1:] if raw.startswith("_") else raw
    if not tag or not all(char.isalnum() or char in "._-" for char in tag):
        raise ValueError(
            "skillset_output_suffix may contain only letters, digits, '.', '_' and '-', "
            f"got {raw!r}."
        )
    return f"_{tag}"


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    s = str(value).strip()
    if "," in s:
        return [v.strip() for v in s.split(",") if v.strip()]
    return [v.strip() for v in s.split() if v.strip()]


def as_levels(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    cleaned = str(value).replace("[", " ").replace("]", " ").replace(",", " ")
    return tuple(int(v) for v in cleaned.split())


SKILLSET_MODES = ("spherical", "full", "without_gripper", "std")


def skillset_probe_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve one user-facing SBD mode into the internal action-manifold arguments."""
    probe_count = int(get_value(cfg, "skillset_probe_count", 24))
    probe_alpha = float(get_value(cfg, "skillset_probe_alpha", 0.1))
    pca_variance = float(get_value(cfg, "skillset_pca_variance", 0.95))
    pca_stride = int(get_value(cfg, "skillset_pca_stride", 3))
    action_mode = str(get_value(cfg, "skillset_action_mode", "dataset"))
    gripper_mode = str(get_value(cfg, "skillset_gripper_mode", "continuous"))
    relative_exclude = ",".join(
        as_list(get_value(cfg, "skillset_relative_exclude_joints", ["gripper"]))
    )
    gripper_indices = ",".join(
        as_list(get_value(cfg, "skillset_gripper_indices", [-1]))
    )
    gripper_values = ",".join(
        as_list(get_value(cfg, "skillset_gripper_values", [-1.0, 1.0]))
    )
    gripper_threshold = float(get_value(cfg, "skillset_gripper_threshold", 0.0))

    mode_value = get_value(cfg, "skillset_mode", None)
    if mode_value in (None, ""):
        # FSQ/eval configs do not need to duplicate the build choice. Inherit the
        # canonical mode when this resolver is called from another train_skills module.
        project_root = get_value(cfg, "project_root", None)
        if project_root:
            build_cfg = (
                Path(str(project_root)).expanduser()
                / "lerobot/examples/libero/configs/train_skills/build_data/build_data_config.yaml"
            )
            if build_cfg.is_file():
                mode_value = get_value(load_config(build_cfg), "skillset_mode", None)
    if mode_value in (None, ""):
        # Backward compatibility for detailed pre-mode configs and snapshots.
        old_type = str(get_value(cfg, "skillset_probe_type", "spherical_xyz"))
        old_scale = str(get_value(cfg, "skillset_pca_scale_mode", "none")).lower()
        old_exclude = ",".join(
            as_list(get_value(cfg, "skillset_probe_exclude_indices", []))
        )
        if old_type == "spherical_xyz":
            mode_value = "spherical"
        elif old_type == "pca_action" and old_scale == "std":
            mode_value = "std"
        elif old_type == "pca_action" and old_exclude:
            mode_value = "without_gripper"
        else:
            mode_value = "full"
    mode = str(mode_value).strip().lower()
    if mode not in SKILLSET_MODES:
        raise ValueError(f"skillset_mode must be one of {SKILLSET_MODES}, got {mode}")

    if mode == "spherical":
        probe_type, pca_scale_mode, probe_exclude_indices = "spherical_xyz", "none", ""
    elif mode == "full":
        probe_type, pca_scale_mode, probe_exclude_indices = "pca_action", "none", ""
    elif mode == "without_gripper":
        probe_type, pca_scale_mode = "pca_action", "none"
        probe_exclude_indices = gripper_indices
    else:
        probe_type, pca_scale_mode, probe_exclude_indices = "pca_action", "std", ""

    return {
        "skillset_mode": mode,
        "skillset_probe_type": probe_type,
        "skillset_probe_count": probe_count,
        "skillset_probe_alpha": probe_alpha,
        "skillset_pca_variance": pca_variance,
        "skillset_pca_stride": pca_stride,
        "skillset_pca_scale_mode": pca_scale_mode,
        "skillset_probe_exclude_indices": probe_exclude_indices,
        "skillset_action_mode": action_mode,
        "skillset_relative_exclude_joints": relative_exclude,
        "skillset_gripper_mode": gripper_mode,
        "skillset_gripper_indices": gripper_indices,
        "skillset_gripper_values": gripper_values,
        "skillset_gripper_threshold": gripper_threshold,
        "skillset_probe_suffix": f"_{mode}",
    }


DINO_IMAGE_MODEL_DIR = "dinov3-vits16"


def train_settings(cfg: dict[str, Any], dataset: str | None = None) -> dict[str, Any]:
    target_dataset = dataset or str(get_value(cfg, "target_dataset", "libero_90", env="TARGET_DATASET"))
    root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root_name = str(get_value(cfg, "dataset_root", "libero_dataset"))
    dataset_root = root / dataset_root_name
    outputs_root = root / str(get_value(cfg, "outputs_root", "outputs"))
    # Fixed per-stage subdirs under the single outputs root (not configurable in yaml).
    dp_outputs_root = outputs_root / "DP"
    fsq_outputs_root = outputs_root / "FSQ"
    fsq_dataset_root_name = str(get_value(cfg, "fsq_dataset_root", "FSQ_dataset"))
    fsq_dataset_root = dataset_root / fsq_dataset_root_name
    fsq_inputs_name = str(get_value(cfg, "fsq_inputs_name", "FSQ_inputs"))
    fsq_inputs_dir = fsq_dataset_root / target_dataset / fsq_inputs_name

    raw_dataset_dir = dataset_root / target_dataset

    dp_n_obs_steps = int(get_value(cfg, "dp_n_obs_steps", 10))
    dp_horizon = int(get_value(cfg, "dp_horizon", 16))
    train_dp = as_bool(get_value(cfg, "train_DP", get_value(cfg, "train_dp", True), env="TRAIN_DP"))
    dp_vision = str(get_value(cfg, "dp_vision", "state")).strip().lower()
    dp_vision_tag = dp_vision
    dp_policy_template = str(
        get_value(
            cfg,
            "dp_policy_name",
            "dp_{target_dataset}_{dp_vision_tag}_obs{dp_n_obs_steps}_horizon{dp_horizon}",
        )
    )
    dp_policy = dp_policy_template.format(
        target_dataset=target_dataset,
        dp_vision=dp_vision,
        dp_vision_tag=dp_vision_tag,
        dp_n_obs_steps=dp_n_obs_steps,
        dp_horizon=dp_horizon,
    )
    # dp_run_name (or env DP_RUN_NAME) names a trained DP folder DIRECTLY (outputs/DP/<name>).
    # Downstream stages (build_data / FSQ / eval) set it to target a DP without re-deriving the
    # name from vision/grid/n_obs. Empty → use the name generated above.
    _dp_run_name = str(get_value(cfg, "dp_run_name", "")).strip()
    if _dp_run_name:
        dp_policy = _dp_run_name
        if "_state" in _dp_run_name:
            dp_vision = "state"
        elif "_resnet" in _dp_run_name:
            dp_vision = "resnet"
        elif "_dino" in _dp_run_name:
            dp_vision = "dino"
    if dp_vision not in {"state", "resnet", "dino"}:
        raise ValueError(f"dp_vision must be state|resnet|dino, got {dp_vision!r}.")
    dp_checkpoint = str(get_value(cfg, "dp_checkpoint", "100000"))
    probe_settings = skillset_probe_settings(cfg)
    # Boundaries are DP/checkpoint-dependent, so different runs never reuse or clobber a skillset.
    skillset_boundary_threshold_mode = resolve_skillset_threshold_mode(cfg, root)
    if skillset_boundary_threshold_mode not in {"episode_mean", "global_mean"}:
        raise ValueError(
            "skillset_boundary_threshold_mode must be 'episode_mean' or 'global_mean', "
            f"got {skillset_boundary_threshold_mode!r}."
        )
    skillset_global_threshold_source = resolve_skillset_global_threshold_source(cfg, root)
    if skillset_global_threshold_source and skillset_boundary_threshold_mode != "global_mean":
        raise ValueError(
            "skillset_global_threshold_source requires skillset_boundary_threshold_mode=global_mean."
        )
    # Keep the legacy directory for episode_mean. A global threshold produces different
    # labels, so isolate it instead of allowing --resume to mix both segmentations.
    skillset_threshold_suffix = ""
    if skillset_boundary_threshold_mode == "global_mean":
        skillset_threshold_suffix = "_globalref" if skillset_global_threshold_source else "_globalmean"
    skillset_output_suffix = resolve_skillset_output_suffix(cfg, root)
    skillset_suffix = (
        probe_settings["skillset_probe_suffix"]
        + skillset_threshold_suffix
        + skillset_output_suffix
    )
    fsq_seg_dir = fsq_inputs_dir / f"seg_{dp_policy}_ck{dp_checkpoint}{skillset_suffix}"

    fsq_levels = as_levels(get_value(cfg, "fsq_levels", [5, 5, 5]))
    fsq_tag = "fsq" + "".join(str(v) for v in fsq_levels)
    fsq_exp = str(get_value(cfg, "fsq_exp", "")).strip()
    fsq_exp_suffix = f"_{fsq_exp}" if fsq_exp else ""
    # End-weight the sampled FSQ VSA flow loss. The endpoint multiplier is explicit so different
    # curricula cannot auto-resume from one another. Keep the historical `_weighted` tag for 2x.
    weighted_loss = as_bool(get_value(cfg, "weighted_loss", False))
    weighted_loss_end_weight = float(get_value(cfg, "weighted_loss_end_weight", 2.0))
    if weighted_loss_end_weight <= 0:
        raise ValueError(
            "weighted_loss_end_weight must be positive, "
            f"got {weighted_loss_end_weight}."
        )
    if not weighted_loss:
        weighted_suffix = ""
    elif math.isclose(weighted_loss_end_weight, 2.0):
        weighted_suffix = "_weighted"
    else:
        weight_tag = f"{weighted_loss_end_weight:g}".replace("-", "m").replace(".", "p")
        weighted_suffix = f"_weighted{weight_tag}"
    fsq_terminator_arch = str(get_value(cfg, "fsq_terminator_arch", "small"))
    if fsq_terminator_arch not in {"small", "cond"}:
        raise ValueError(f"fsq_terminator_arch must be small|cond, got {fsq_terminator_arch!r}.")
    fsq_terminator_layers = int(get_value(cfg, "fsq_terminator_layers", 2))
    fsq_terminator_heads = int(get_value(cfg, "fsq_terminator_heads", 4))
    if fsq_terminator_layers < 1 or fsq_terminator_heads < 1:
        raise ValueError("fsq_terminator_layers and fsq_terminator_heads must both be >= 1.")
    fsq_cond_encoder_variant = str(get_value(cfg, "fsq_cond_encoder_variant", "gemma_300m"))
    # Depth/head settings are recorded in fsq_meta.json and the checkpoint config, but architecture
    # alone keeps the user-facing run name concise.
    fsq_terminator_tag = fsq_terminator_arch
    fsq_vision_backbone = str(get_value(cfg, "fsq_vision_backbone", "dino"))
    fsq_freeze_vision_encoder = as_bool(get_value(cfg, "fsq_freeze_vision_encoder", True))
    fsq_vision_tag = fsq_vision_backbone + ("_frozen" if fsq_freeze_vision_encoder else "_tuned")
    fsq_encoder_input_mode = str(get_value(cfg, "fsq_encoder_input_mode", "zero_grounded")).strip().lower()
    if fsq_encoder_input_mode not in {"zero_grounded", "raw_state", "optimal"}:
        raise ValueError(
            "fsq_encoder_input_mode must be zero_grounded|raw_state|optimal, "
            f"got {fsq_encoder_input_mode!r}."
        )
    # Preserve the historical zero-grounded name; the other conventions get explicit suffixes so
    # they can never auto-resume from an incompatible encoder checkpoint.
    fsq_encoder_input_suffix = {
        "zero_grounded": "",
        "raw_state": "_rawstate",
        "optimal": "_optimal",
    }[fsq_encoder_input_mode]
    fsq_state_cond_mode = str(get_value(cfg, "fsq_state_cond_mode", "state"))
    if fsq_state_cond_mode not in {"state", "state_skill"}:
        raise ValueError(
            "fsq_state_cond_mode must be state|state_skill, "
            f"got {fsq_state_cond_mode!r}."
        )
    # This affects only the VSA action expert; the terminator always receives both state and z_q.
    # Keep it in every run name so modes cannot accidentally resume from one another.
    fsq_state_cond_suffix = f"_vsa_{fsq_state_cond_mode}"
    # DP tag for the FSQ run name = the DP run with the dataset prefix stripped
    # (libero_90_full_full_state_obs20 → state_obs20), so the FSQ folder shows WHICH DP's skillset it
    # was trained on (for example state_obs20), not just the FSQ architecture.
    dp_tag = dp_policy[len(target_dataset) + 1:] if dp_policy.startswith(f"{target_dataset}_") else dp_policy
    dp_tag += probe_settings["skillset_probe_suffix"]
    if skillset_boundary_threshold_mode == "global_mean":
        dp_tag += "_global"
    dp_tag += skillset_output_suffix
    fsq_run_template = str(
        get_value(
            cfg,
            "fsq_run_name",
            "{target_dataset}_{dp_tag}_{fsq_tag}_{fsq_vision_tag}_{fsq_terminator_tag}"
            "{fsq_encoder_input_suffix}{fsq_state_cond_suffix}{weighted_suffix}{fsq_exp_suffix}",
        )
    )
    fsq_run_name = fsq_run_template.format(
        target_dataset=target_dataset,
        dp_tag=dp_tag,
        fsq_tag=fsq_tag,
        fsq_exp=fsq_exp,
        fsq_exp_suffix=fsq_exp_suffix,
        weighted_suffix=weighted_suffix,
        fsq_vision_tag=fsq_vision_tag,
        fsq_terminator_arch=fsq_terminator_arch,
        fsq_terminator_tag=fsq_terminator_tag,
        fsq_terminator_layers=fsq_terminator_layers,
        fsq_terminator_heads=fsq_terminator_heads,
        fsq_encoder_input_mode=fsq_encoder_input_mode,
        fsq_encoder_input_suffix=fsq_encoder_input_suffix,
        fsq_state_cond_mode=fsq_state_cond_mode,
        fsq_state_cond_suffix=fsq_state_cond_suffix,
    )
    # Slurm partition/qos/nodelist/exclude are canonical (read from global_config.yaml's train_*);
    # output keys below keep their per-job prefix so submit scripts read the same $..._PARTITION vars.
    slurm_partitions = as_list(get_value(cfg, "train_partition", ["debug"]))
    # Pass the FULL partition list (comma-joined), like fsq_train_partition — so Slurm can place the
    # job on a partition where the chosen qos is valid. Pinning to just the first partition (e.g. an
    # a6000) while train_qos=pro6000_qos triggers "Invalid qos specification".
    slurm_partition = ",".join(slurm_partitions) or "debug"

    return {
        "project_root": root,
        "lerobot_root": root / "lerobot",
        "dataset_root": dataset_root,
        "dataset_root_name": dataset_root_name,
        "fsq_dataset_root": fsq_dataset_root,
        "fsq_dataset_root_name": fsq_dataset_root_name,
        "target_dataset": target_dataset,
        "raw_dataset_dir": raw_dataset_dir,
        "data_dir": dataset_root / f"{target_dataset}_data",
        "outputs_root": outputs_root,
        "dp_outputs_root": dp_outputs_root,
        "fsq_outputs_root": fsq_outputs_root,
        "pi_base_model_path": root / "models" / "pi05_base",
        "dp_base_config": root / "lerobot" / str(get_value(cfg, "dp_base_config")),
        "dp_policy": dp_policy,
        "dp_output_dir": dp_outputs_root / dp_policy,
        "dp_policy_path": dp_outputs_root / dp_policy / "checkpoints" / dp_checkpoint / "pretrained_model",
        "dp_checkpoint": dp_checkpoint,
        "dp_vision": dp_vision,
        "dp_vision_backbone": str(get_value(cfg, "dp_vision_backbone", "resnet18")),
        "train_DP": train_dp,
        "dp_n_obs_steps": dp_n_obs_steps,
        # Default = max valid chunk (horizon - n_obs + 1); stays consistent if n_obs/horizon change.
        "dp_n_action_steps": int(get_value(cfg, "dp_n_action_steps", dp_horizon - dp_n_obs_steps + 1)),
        "dp_horizon": dp_horizon,
        "dp_batch_size": int(get_value(cfg, "dp_batch_size", 64)),
        "dp_relative": as_bool(get_value(cfg, "dp_relative", False)),
        "dp_steps": int(get_value(cfg, "dp_steps", 100000)),
        "dp_num_workers": int(get_value(cfg, "dp_num_workers", 4)),
        "dp_save_freq": int(get_value(cfg, "dp_save_freq", 50000)),
        "dp_log_freq": int(get_value(cfg, "dp_log_freq", 200)),
        "dp_eval_freq": int(get_value(cfg, "dp_eval_freq", 0)),
        "dp_seed": int(get_value(cfg, "dp_seed", 42)),
        "dp_wandb_project": str(get_value(cfg, "dp_wandb_project", "DP_train")),
        "dp_wandb_enable": as_bool(get_value(cfg, "dp_wandb_enable", True)),
        "dp_overwrite_output": as_bool(get_value(cfg, "dp_overwrite_output", True)),
        "fsq_levels_str": " ".join(str(v) for v in fsq_levels),
        "fsq_levels_arg": "[" + ",".join(str(v) for v in fsq_levels) + "]",
        "fsq_tag": fsq_tag,
        "fsq_exp": fsq_exp,
        "fsq_exp_suffix": fsq_exp_suffix,
        "fsq_weighted_loss": weighted_loss,
        "fsq_weighted_loss_end_weight": weighted_loss_end_weight,
        "weighted_suffix": weighted_suffix,
        "fsq_encoder_input_suffix": fsq_encoder_input_suffix,
        "fsq_run_name": fsq_run_name,
        "fsq_output_dir": fsq_outputs_root / fsq_run_name,
        "fsq_dim": len(fsq_levels),
        "fsq_num_embeddings": math.prod(fsq_levels),
        "fsq_epoch": str(get_value(cfg, "fsq_epoch", "1000")),
        "fsq_batch_size": int(get_value(cfg, "fsq_batch_size", 256)),
        "fsq_num_workers": int(get_value(cfg, "fsq_num_workers", 8)),
        "fsq_num_epochs": int(get_value(cfg, "fsq_num_epochs", 1000)),
        "fsq_checkpoint_every": int(get_value(cfg, "fsq_checkpoint_every", 500)),
        "fsq_encoder_lr": str(get_value(cfg, "fsq_encoder_lr", "3e-4")),
        "fsq_terminator_lr": str(get_value(cfg, "fsq_terminator_lr", "3e-4")),
        "fsq_expert_lr": str(get_value(cfg, "fsq_expert_lr", "2.5e-5")),
        "fsq_samples_per_skill": int(get_value(cfg, "fsq_samples_per_skill", 2)),
        "fsq_action_expert_variant": str(get_value(cfg, "fsq_action_expert_variant", "gemma_300m")),
        "fsq_encoder_input_mode": fsq_encoder_input_mode,
        "fsq_state_cond_mode": fsq_state_cond_mode,
        "fsq_expert_dtype": str(get_value(cfg, "fsq_expert_dtype", "bfloat16")),
        "fsq_hidden_dim": int(get_value(cfg, "fsq_hidden_dim", 256)),
        "fsq_num_layers": int(get_value(cfg, "fsq_num_layers", 2)),
        "fsq_n_control": int(get_value(cfg, "fsq_n_control", 30)),
        "fsq_terminator_arch": fsq_terminator_arch,
        "fsq_terminator_layers": fsq_terminator_layers,
        "fsq_terminator_heads": fsq_terminator_heads,
        "fsq_vision_backbone": fsq_vision_backbone,
        "fsq_freeze_vision_encoder": fsq_freeze_vision_encoder,
        # Accept a portable project-relative path in YAML (e.g. models/dinov3-vits16), while
        # passing an absolute path to the Slurm job and checkpoint metadata.
        "fsq_dino_model_path": resolve_path(
            root, get_value(cfg, "fsq_dino_model_path", f"models/{DINO_IMAGE_MODEL_DIR}")
        ),
        "fsq_dino_image_size": int(get_value(cfg, "fsq_dino_image_size", 224)),
        "fsq_siglip_image_size": int(get_value(cfg, "fsq_siglip_image_size", 224)),
        "fsq_cond_encoder_variant": fsq_cond_encoder_variant,
        "fsq_chunk_size": int(get_value(cfg, "fsq_chunk_size", 10)),
        "fsq_action_loss_weight": str(get_value(cfg, "fsq_action_loss_weight", 1.0)),
        "fsq_progress_loss_weight": str(get_value(cfg, "fsq_progress_loss_weight", 1.0)),
        "fsq_end_loss_weight": str(get_value(cfg, "fsq_end_loss_weight", 1.0)),
        # best-val SELECTION metric weights — empty → "" (sbatch omits the flag → selection follows loss)
        "fsq_val_select_action_weight": str(get_value(cfg, "fsq_val_select_action_weight", "") or ""),
        "fsq_val_select_progress_weight": str(get_value(cfg, "fsq_val_select_progress_weight", "") or ""),
        "fsq_val_select_end_weight": str(get_value(cfg, "fsq_val_select_end_weight", "") or ""),
        "fsq_end_target_sigma": str(get_value(cfg, "fsq_end_target_sigma", 0.0)),
        "fsq_wandb_project": str(get_value(cfg, "fsq_wandb_project", "VAE_train")),
        "dp_partition": slurm_partition,
        "dp_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "dp_exclude_nodes": as_list(get_value(cfg, "train_exclude_nodes", [])),
        "dp_qos": str(get_value(cfg, "train_qos", "base_qos")),
        "dp_gres": str(get_value(cfg, "dp_gres", "gpu:1")),
        "dp_cpus_per_task": int(get_value(cfg, "dp_cpus_per_task", 8)),
        "dp_mem": str(get_value(cfg, "dp_mem", "64G")),
        "dp_time": str(get_value(cfg, "dp_time", "48:00:00")),
        "skillset_name": str(get_value(cfg, "skillset_name", "skillset")),
        "skillset_dir": fsq_seg_dir / str(get_value(cfg, "skillset_name", "skillset")),
        "skillset_done_path": fsq_seg_dir / str(get_value(cfg, "skillset_name", "skillset")) / ".complete",
        "skillset_tasks_per_job": int(get_value(cfg, "skillset_tasks_per_job", 5)),
        "skillset_wandb_project": str(get_value(cfg, "skillset_wandb_project", "Skill_dataset")),
        "skillset_dn_step": int(get_value(cfg, "skillset_dn_step", 7)),
        "skillset_n_gmm": int(get_value(cfg, "skillset_n_gmm", 5)),
        "skillset_smooth_window": int(get_value(cfg, "skillset_smooth_window", 7)),
        "skillset_savgol_polyorder": int(get_value(cfg, "skillset_savgol_polyorder", 4)),
        "skillset_replan_interval": int(get_value(cfg, "skillset_replan_interval", 3)),
        "skillset_nms_dist": int(get_value(cfg, "skillset_nms_dist", 25)),
        **probe_settings,
        "skillset_boundary_threshold_mode": skillset_boundary_threshold_mode,
        "skillset_global_threshold_source": skillset_global_threshold_source,
        "skillset_output_suffix": skillset_output_suffix,
        "skillset_global_threshold_path": (
            fsq_seg_dir / str(get_value(cfg, "skillset_name", "skillset"))
            / "global_boundary_threshold.json"
        ),
        "skillset_dino_feature_dir": resolve_path(
            root, get_value(cfg, "skillset_dino_feature_dir", "")
        ),
        "skillset_cpus_per_task": int(get_value(cfg, "skillset_cpus_per_task", 4)),
        "skillset_mem": str(get_value(cfg, "skillset_mem", "32G")),
        "skillset_time": str(get_value(cfg, "skillset_time", "4:00:00")),
        "fsq_inputs_name": fsq_inputs_name,
        "fsq_inputs_dir": fsq_inputs_dir,
        "fsq_seg_dir": fsq_seg_dir,
        "slurm_partitions": slurm_partitions,
        "slurm_partition": slurm_partition,
        "slurm_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "slurm_exclude_nodes": as_list(get_value(cfg, "train_exclude_nodes", [])),
        "slurm_qos": str(get_value(cfg, "train_qos", "base_qos")),
        "slurm_gres": str(get_value(cfg, "slurm_gres", "gpu:1")),
        "fsq_train_partition": ",".join(slurm_partitions) or "debug",
        "fsq_train_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "fsq_train_exclude_nodes": as_list(get_value(cfg, "train_exclude_nodes", [])),
        "fsq_train_qos": str(get_value(cfg, "train_qos", "base_qos")),
        "fsq_train_gres": str(get_value(cfg, "fsq_train_gres", "gpu:1")),
        "fsq_train_cpus_per_task": int(get_value(cfg, "fsq_train_cpus_per_task", 12)),
        "fsq_train_mem": str(get_value(cfg, "fsq_train_mem", "64G")),
        "fsq_train_time": str(get_value(cfg, "fsq_train_time", "48:00:00")),
    }


def print_shell(settings: dict[str, Any]) -> None:
    for key, value in settings.items():
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, list):
            value = ",".join(str(v) for v in value)
        print(f"export {key.upper()}={shlex.quote(str(value))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = train_settings(cfg, dataset=args.dataset)
    if args.shell:
        print_shell(settings)
        return
    for key, value in settings.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
