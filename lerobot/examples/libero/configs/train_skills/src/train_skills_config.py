#!/usr/bin/env python3
"""Shared config helpers for train_skills scripts."""

from __future__ import annotations

import argparse
import ast
import json
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


DINO_VISUAL_BACKBONE = "dinov3_vits16"
DINO_IMAGE_MODEL_DIR = "dinov3-vits16"
DINO_FEATURE_TAG = "dinov3_vits16"
DINO_FEATURE_DIM = 384


def infer_source_dataset_name(target_dataset: str) -> str:
    for base in (
        "libero_90",
        "libero_10",
        "libero_spatial_object",
        "libero_spatial",
        "libero_object",
    ):
        if target_dataset == base or target_dataset.startswith(base + "_"):
            return base
    return target_dataset


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

    backbone = DINO_VISUAL_BACKBONE
    dino_feature_tag = DINO_FEATURE_TAG
    dino_patch_grid = int(get_value(cfg, "dino_patch_grid", 8, env="DINO_PATCH_GRID"))
    # DP segments/trains on pg{dino_patch_grid}. FSQ may use a DIFFERENT grid for its skill
    # DINO tokens (e.g. DP on pg8, FSQ on pg14). fsq_patch_grid defaults to dino_patch_grid,
    # so existing single-grid configs are unchanged; set it to decouple the FSQ side.
    fsq_patch_grid = int(get_value(cfg, "fsq_patch_grid", dino_patch_grid, env="FSQ_PATCH_GRID"))
    dino_feature_dataset = str(get_value(cfg, "dino_feature_dataset", "{target_dataset}")).format(
        target_dataset=target_dataset
    )
    configured_source_dataset = str(get_value(cfg, "dino_source_dataset", ""))
    raw_dataset_dir = dataset_root / target_dataset
    inferred_source_dataset = infer_source_dataset_name(target_dataset)
    info_path = raw_dataset_dir / "meta" / "info.json"
    if info_path.exists():
        try:
            with open(info_path) as f:
                inferred_source_dataset = str(json.load(f).get("source_dataset", inferred_source_dataset))
        except Exception:
            inferred_source_dataset = infer_source_dataset_name(target_dataset)
    dino_source_dataset = configured_source_dataset or inferred_source_dataset

    dp_n_obs_steps = int(get_value(cfg, "dp_n_obs_steps", 10))
    dp_horizon = int(get_value(cfg, "dp_horizon", 16))
    train_dp = as_bool(get_value(cfg, "train_DP", get_value(cfg, "train_dp", True), env="TRAIN_DP"))
    # Vision encoder: "dino" (default) or "resnet" (original DP). dp_vision_tag is the name tag —
    # "dino{grid}" for dino (keeps the existing name), bare "resnet" for resnet (which has no DINO
    # grid). dp_policy_name uses {dp_vision_tag} so a resnet DP never clobbers a dino DP.
    dp_vision = str(get_value(cfg, "dp_vision", "dino"))
    dp_vision_tag = f"dino{dino_patch_grid}" if dp_vision == "dino" else dp_vision
    dp_policy_template = str(
        get_value(
            cfg,
            "dp_policy_name",
            "dp_{target_dataset}_{dp_vision_tag}_obs{dp_n_obs_steps}_horizon{dp_horizon}",
        )
    )
    dp_policy = dp_policy_template.format(
        target_dataset=target_dataset,
        dino_feature_dataset=dino_feature_dataset,
        dino_feature_tag=dino_feature_tag,
        dino_patch_grid=dino_patch_grid,
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
        # Infer the DP's vision from its name tag so downstream gates (DINO prepare/check) are
        # correct even when the DP is referenced only by folder name. A state/resnet DP uses no
        # DINO, so build_data must not require it.
        if "_state" in _dp_run_name:
            dp_vision = "state"
        elif "_resnet" in _dp_run_name:
            dp_vision = "resnet"
        elif "_dino" in _dp_run_name:
            dp_vision = "dino"
    dp_checkpoint = str(get_value(cfg, "dp_checkpoint", "100000"))
    # skillset + per-skill DINO tokens are DP-dependent (the boundaries come from the DP),
    # so key them by DP/checkpoint — a different DP/checkpoint never reuses or clobbers
    # another's segmentation. Mirrors train_skillVLA's _work/seg_{dp}_ck{ckpt}/.
    fsq_seg_dir = fsq_inputs_dir / f"seg_{dp_policy}_ck{dp_checkpoint}"

    fsq_levels = as_levels(get_value(cfg, "fsq_levels", [5, 5, 5]))
    fsq_tag = "fsq" + "".join(str(v) for v in fsq_levels)
    fsq_image_token_dim = int(get_value(cfg, "fsq_image_token_dim", 256))
    fsq_exp = str(get_value(cfg, "fsq_exp", "")).strip()
    fsq_exp_suffix = f"_{fsq_exp}" if fsq_exp else ""
    fsq_run_template = str(
        get_value(
            cfg,
            "fsq_run_name",
            "{target_dataset}_{fsq_tag}_pg{fsq_patch_grid}_image{fsq_image_token_dim}{fsq_exp_suffix}",
        )
    )
    fsq_run_name = fsq_run_template.format(
        target_dataset=target_dataset,
        fsq_tag=fsq_tag,
        dino_patch_grid=dino_patch_grid,
        fsq_patch_grid=fsq_patch_grid,
        fsq_image_token_dim=fsq_image_token_dim,
        fsq_exp=fsq_exp,
        fsq_exp_suffix=fsq_exp_suffix,
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
        "dino_source_dataset": dino_source_dataset,
        "dino_feature_dataset": dino_feature_dataset,
        "base_dino_feature_dir": dataset_root / f"{dino_source_dataset}_DINO" / f"pg{dino_patch_grid}",
        "dino_feature_dir": fsq_dataset_root / target_dataset / "DINO" / f"pg{dino_patch_grid}",
        # FSQ-side DINO at its own grid (= DP grid unless fsq_patch_grid is set). The skill-token
        # extractor and FSQ training read these; DP/skillset keep using the dino_* dirs above.
        "fsq_patch_grid": fsq_patch_grid,
        "base_fsq_dino_feature_dir": dataset_root / f"{dino_source_dataset}_DINO" / f"pg{fsq_patch_grid}",
        "fsq_dino_feature_dir": fsq_dataset_root / target_dataset / "DINO" / f"pg{fsq_patch_grid}",
        "dino_visual_backbone": backbone,
        "dino_image_model_dir": DINO_IMAGE_MODEL_DIR,
        "dino_image_model_path": root / "models" / DINO_IMAGE_MODEL_DIR,
        "dino_feature_tag": dino_feature_tag,
        "dino_image_keys": as_list(get_value(cfg, "dino_image_keys", ["observation.images.image"])),
        "dino_patch_grid": dino_patch_grid,
        "dino_image_size": 224,
        "dino_feature_dim": int(get_value(cfg, "dino_feature_dim", DINO_FEATURE_DIM)),
        "dino_visual_feature_dim": int(get_value(cfg, "dino_visual_feature_dim", 256)),
        "dino_transformer_n_layers": int(get_value(cfg, "dino_transformer_n_layers", 1)),
        "dino_transformer_n_heads": int(get_value(cfg, "dino_transformer_n_heads", 4)),
        "dino_cache_size": int(get_value(cfg, "dino_cache_size", 8)),
        "dino_copy_mode": str(get_value(cfg, "dino_copy_mode", "copy")),
        "dino_overwrite_prepared": as_bool(get_value(cfg, "dino_overwrite_prepared", False)),
        "dp_base_config": root / "lerobot" / str(get_value(cfg, "dp_base_config")),
        "dp_policy": dp_policy,
        "dp_output_dir": dp_outputs_root / dp_policy,
        "dp_policy_path": dp_outputs_root / dp_policy / "checkpoints" / dp_checkpoint / "pretrained_model",
        "dp_checkpoint": dp_checkpoint,
        # DP vision encoder: "dino" = precomputed DINO tokens (this project's default);
        # "resnet" = original lerobot Diffusion Policy (ResNet on raw frames, use_dino_features=false).
        "dp_vision": dp_vision,
        "dp_vision_backbone": str(get_value(cfg, "dp_vision_backbone", "resnet18")),
        "train_DP": train_dp,
        "dp_n_obs_steps": dp_n_obs_steps,
        # Default = max valid chunk (horizon - n_obs + 1); stays consistent if n_obs/horizon change.
        "dp_n_action_steps": int(get_value(cfg, "dp_n_action_steps", dp_horizon - dp_n_obs_steps + 1)),
        "dp_horizon": dp_horizon,
        "dp_batch_size": int(get_value(cfg, "dp_batch_size", 64)),
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
        "fsq_run_name": fsq_run_name,
        "fsq_output_dir": fsq_outputs_root / fsq_run_name,
        "fsq_dim": len(fsq_levels),
        "fsq_num_embeddings": math.prod(fsq_levels),
        "fsq_epoch": str(get_value(cfg, "fsq_epoch", "1000")),
        "fsq_batch_size": int(get_value(cfg, "fsq_batch_size", 256)),
        "fsq_num_workers": int(get_value(cfg, "fsq_num_workers", 8)),
        "fsq_num_epochs": int(get_value(cfg, "fsq_num_epochs", 1000)),
        "fsq_checkpoint_every": int(get_value(cfg, "fsq_checkpoint_every", 500)),
        "fsq_lr": str(get_value(cfg, "fsq_lr", "3e-4")),
        "fsq_hidden_dim": int(get_value(cfg, "fsq_hidden_dim", 256)),
        "fsq_num_layers": int(get_value(cfg, "fsq_num_layers", 2)),
        "fsq_n_control": int(get_value(cfg, "fsq_n_control", 30)),
        "fsq_image_token_dim": int(get_value(cfg, "fsq_image_token_dim", 256)),
        "fsq_terminator_use_third": as_bool(get_value(cfg, "fsq_terminator_use_third", True)),
        "fsq_terminator_use_wrist": as_bool(get_value(cfg, "fsq_terminator_use_wrist", True)),
        "fsq_chunk_size": int(get_value(cfg, "fsq_chunk_size", 10)),
        "fsq_delta_loss_weight": str(get_value(cfg, "fsq_delta_loss_weight", 1.0)),
        "fsq_progress_loss_weight": str(get_value(cfg, "fsq_progress_loss_weight", 1.0)),
        "fsq_end_loss_weight": str(get_value(cfg, "fsq_end_loss_weight", 1.0)),
        "fsq_end_target_sigma": str(get_value(cfg, "fsq_end_target_sigma", 0.0)),
        "fsq_wandb_project": str(get_value(cfg, "fsq_wandb_project", "VAE_train")),
        "dp_partition": slurm_partition,
        "dp_nodelist": str(get_value(cfg, "train_nodelist", "")),
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
        "skillset_cpus_per_task": int(get_value(cfg, "skillset_cpus_per_task", 4)),
        "skillset_mem": str(get_value(cfg, "skillset_mem", "32G")),
        "skillset_time": str(get_value(cfg, "skillset_time", "4:00:00")),
        "fsq_inputs_name": fsq_inputs_name,
        "fsq_inputs_dir": fsq_inputs_dir,
        "fsq_seg_dir": fsq_seg_dir,   # DP-keyed: skillset + per-skill dino_tokens live here
        "dino_tokens_path": (
            fsq_seg_dir
            / f"dino_tokens_pg{fsq_patch_grid}.npz"
        ),
        "dino_tokens_wrist_path": (
            fsq_seg_dir
            / f"dino_tokens_wrist_pg{fsq_patch_grid}.npz"
        ),
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
