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


def resolve_skillset_threshold_mode(cfg: dict[str, Any], project_root: Path) -> str:
    """Read the boundary-threshold mode from this module's own config.

    Every config (build_data, FSQ, eval) is self-contained and must carry the
    key itself when it deviates from the default — there is no cross-config
    inheritance, so editing one module's yaml never changes another's resolution.
    """
    return str(
        get_value(cfg, "skillset_boundary_threshold_mode", "episode_mean")
    ).strip().lower()


def resolve_skillset_threshold_scale(cfg: dict[str, Any]) -> float:
    scale = float(get_value(cfg, "skillset_boundary_threshold_scale", 1.0))
    if scale <= 0.0:
        raise ValueError(
            f"skillset_boundary_threshold_scale must be positive, got {scale}."
        )
    return scale


def _scale_percent_tag(scale: float) -> str:
    return f"{scale * 100:g}".replace(".", "p") + "p"


def resolve_skillset_global_threshold_source(cfg: dict[str, Any], project_root: Path) -> str:
    """Resolve an optional fixed global-threshold JSON from this module's config.

    A source is intentionally separate from ``global_mean``: the latter normally
    recomputes a mean for the dataset being built, while a non-empty source
    freezes a previously established cross-task threshold.
    """
    return resolve_path(project_root, get_value(cfg, "skillset_global_threshold_source", ""))


def resolve_skillset_output_suffix(cfg: dict[str, Any], project_root: Path) -> str:
    """Resolve an optional, manually chosen experiment suffix for a skillset."""
    raw = str(get_value(cfg, "skillset_output_suffix", "") or "").strip()
    if not raw:
        return ""
    tag = raw[1:] if raw.startswith("_") else raw
    if not tag or not all(char.isalnum() or char in "._-" for char in tag):
        raise ValueError(
            "skillset_output_suffix may contain only letters, digits, '.', '_' and '-', "
            f"got {raw!r}."
        )
    return f"_{tag}"


def resolve_skillset_min_skills(cfg: dict[str, Any], project_root: Path) -> int:
    """Read the minimum segment count from this module's own config (default 1)."""
    min_skills = int(get_value(cfg, "skillset_min_skills", 1))
    if min_skills < 1:
        raise ValueError(f"skillset_min_skills must be >= 1, got {min_skills}.")
    return min_skills


def resolve_skillset_min_skill_len(cfg: dict[str, Any]) -> int:
    """Minimum frames per final segment after boundary post-processing."""
    min_skill_len = int(get_value(cfg, "skillset_min_skill_len", 10))
    if min_skill_len < 1:
        raise ValueError(
            f"skillset_min_skill_len must be >= 1, got {min_skill_len}."
        )
    return min_skill_len


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


def _selected_skillset(
    cfg: dict[str, Any],
    *,
    dataset_root: Path,
    target_dataset: str,
) -> dict[str, Any] | None:
    """Resolve an existing skillset selected by folder components.

    Build configs still derive ``seg_*`` from DP/detector knobs. FSQ training can
    instead name an immutable artifact directly and take all of its provenance
    from ``skillset_manifest.json``.
    """
    seg_name = str(get_value(cfg, "skillset_seg_name", "") or "").strip()
    if not seg_name:
        return None

    components = {
        "fsq_dataset_root": str(
            get_value(
                cfg,
                "fsq_dataset_root",
                "FSQ_dataset",
                env="FSQ_DATASET_ROOT_NAME",
            )
        ).strip(),
        "target_dataset": str(target_dataset).strip(),
        "fsq_inputs_name": str(get_value(cfg, "fsq_inputs_name", "FSQ_inputs")).strip(),
        "skillset_seg_name": seg_name,
        "skillset_name": str(get_value(cfg, "skillset_name", "skillset")).strip(),
    }
    missing = [key for key, value in components.items() if not value]
    if missing:
        raise ValueError(f"Selected skillset path is missing folder components: {missing}")
    invalid = [
        key
        for key, value in components.items()
        if Path(value).name != value or value in {".", ".."}
    ]
    if invalid:
        raise ValueError(
            "Selected skillset components must be folder names, not paths: "
            f"{invalid}"
        )

    skillset_dir = (
        dataset_root
        / components["fsq_dataset_root"]
        / components["target_dataset"]
        / components["fsq_inputs_name"]
        / components["skillset_seg_name"]
        / components["skillset_name"]
    )
    manifest_path = skillset_dir / "skillset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Selected FSQ skillset manifest not found: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid skillset manifest: {manifest_path}: {error}") from error

    manifest_dataset = str(manifest.get("dataset_name", "")).strip()
    if manifest_dataset != target_dataset:
        raise ValueError(
            "Selected skillset dataset mismatch: "
            f"folder/config={target_dataset!r}, manifest={manifest_dataset!r} "
            f"({manifest_path})"
        )
    policy_raw = str(manifest.get("policy_path", "")).strip()
    policy_path = Path(policy_raw)
    if not policy_raw or len(policy_path.parents) < 3:
        raise ValueError(f"Invalid policy_path in skillset manifest: {manifest_path}")
    dp_policy = policy_path.parents[2].name
    dp_checkpoint = policy_path.parent.name
    if dp_checkpoint.isdigit():
        dp_checkpoint = dp_checkpoint.zfill(6)
    seg_prefix = f"seg_{dp_policy}_ck{dp_checkpoint}"
    if not seg_name.startswith(seg_prefix):
        raise ValueError(
            "Selected seg folder disagrees with its manifest policy/checkpoint: "
            f"expected prefix {seg_prefix!r}, got {seg_name!r}"
        )

    detector = manifest.get("detector") or {}
    mode = str(manifest.get("mode", "")).strip().lower()
    if mode not in SKILLSET_MODES:
        raise ValueError(
            f"Invalid mode in selected skillset manifest: {mode!r} ({manifest_path})"
        )
    threshold_mode = str(
        detector.get("boundary_threshold_mode", "episode_mean")
    ).strip().lower()
    if threshold_mode not in {"episode_mean", "global_mean"}:
        raise ValueError(
            "Invalid detector.boundary_threshold_mode in selected skillset manifest: "
            f"{threshold_mode!r} ({manifest_path})"
        )
    threshold_scale = float(detector.get("boundary_threshold_scale", 1.0))
    if threshold_scale <= 0.0:
        raise ValueError(
            "Invalid detector.boundary_threshold_scale in selected skillset manifest: "
            f"{threshold_scale} ({manifest_path})"
        )

    return {
        **components,
        "skillset_dir": skillset_dir,
        "seg_dir": skillset_dir.parent,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "dp_policy": dp_policy,
        "dp_checkpoint": dp_checkpoint,
        "mode": mode,
        "threshold_mode": threshold_mode,
        "threshold_scale": threshold_scale,
        "min_skills": int(detector.get("min_skills", 1)),
        "min_skill_len": int(detector.get("min_skill_len", 10)),
        # The exact suffix (including globalref/manual tags) is immutable in the
        # selected folder name and is therefore safer than reconstructing it.
        "seg_suffix": seg_name[len(seg_prefix) :],
    }


def train_settings(cfg: dict[str, Any], dataset: str | None = None) -> dict[str, Any]:
    target_dataset = dataset or str(get_value(cfg, "target_dataset", "libero_90", env="TARGET_DATASET"))
    root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root_name = str(
        get_value(cfg, "dataset_root", "libero_dataset", env="DATASET_ROOT_NAME")
    )
    dataset_root = root / dataset_root_name
    outputs_root = root / str(get_value(cfg, "outputs_root", "outputs"))
    # Fixed per-stage subdirs under the single outputs root (not configurable in yaml).
    dp_outputs_root = outputs_root / "DP"
    fsq_outputs_root = outputs_root / "FSQ"
    fsq_dataset_root_name = str(
        get_value(
            cfg,
            "fsq_dataset_root",
            "FSQ_dataset",
            env="FSQ_DATASET_ROOT_NAME",
        )
    )
    fsq_dataset_root = dataset_root / fsq_dataset_root_name
    fsq_inputs_name = str(get_value(cfg, "fsq_inputs_name", "FSQ_inputs"))
    fsq_inputs_dir = fsq_dataset_root / target_dataset / fsq_inputs_name
    selected_skillset = _selected_skillset(
        cfg,
        dataset_root=dataset_root,
        target_dataset=target_dataset,
    )

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
    # lerobot checkpoint folders are zero-padded to 6 digits (050000); normalize numeric
    # values so `50000` and `050000` resolve to the same checkpoint AND seg_* skillset dir.
    if dp_checkpoint.isdigit():
        dp_checkpoint = dp_checkpoint.zfill(6)
    skillset_cfg = cfg
    if selected_skillset is not None:
        dp_policy = selected_skillset["dp_policy"]
        dp_checkpoint = selected_skillset["dp_checkpoint"]
        if "_state" in dp_policy:
            dp_vision = "state"
        elif "_resnet" in dp_policy:
            dp_vision = "resnet"
        elif "_dino" in dp_policy:
            dp_vision = "dino"
        manifest_action = selected_skillset["manifest"].get("action") or {}
        manifest_probe = selected_skillset["manifest"].get("probe") or {}
        skillset_cfg = {
            **cfg,
            "skillset_mode": selected_skillset["mode"],
            "skillset_probe_count": manifest_probe.get("count", 24),
            "skillset_probe_alpha": manifest_probe.get("alpha", 0.1),
            "skillset_pca_variance": manifest_probe.get("pca_variance", 0.95),
            "skillset_pca_stride": manifest_probe.get("pca_stride", 3),
            "skillset_action_mode": manifest_action.get("mode", "dataset"),
            "skillset_relative_exclude_joints": manifest_action.get(
                "relative_exclude_joints", ["gripper"]
            ),
            "skillset_gripper_mode": manifest_action.get("gripper_mode", "continuous"),
            "skillset_gripper_indices": manifest_action.get("gripper_indices", [-1]),
            "skillset_gripper_values": manifest_action.get("gripper_values", [-1.0, 1.0]),
            "skillset_gripper_threshold": manifest_action.get("gripper_threshold", 0.0),
        }
    probe_settings = skillset_probe_settings(skillset_cfg)
    skillset_min_skills = (
        selected_skillset["min_skills"]
        if selected_skillset is not None
        else resolve_skillset_min_skills(cfg, root)
    )
    skillset_min_skill_len = (
        selected_skillset["min_skill_len"]
        if selected_skillset is not None
        else resolve_skillset_min_skill_len(cfg)
    )
    if skillset_min_skills < 1:
        raise ValueError(f"skillset manifest min_skills must be >= 1, got {skillset_min_skills}.")
    if skillset_min_skill_len < 1:
        raise ValueError(
            f"skillset manifest min_skill_len must be >= 1, got {skillset_min_skill_len}."
        )
    # Boundaries are DP/checkpoint-dependent, so different runs never reuse or clobber a skillset.
    skillset_boundary_threshold_mode = (
        selected_skillset["threshold_mode"]
        if selected_skillset is not None
        else resolve_skillset_threshold_mode(cfg, root)
    )
    if skillset_boundary_threshold_mode not in {"episode_mean", "global_mean"}:
        raise ValueError(
            "skillset_boundary_threshold_mode must be 'episode_mean' or 'global_mean', "
            f"got {skillset_boundary_threshold_mode!r}."
        )
    skillset_boundary_threshold_scale = (
        selected_skillset["threshold_scale"]
        if selected_skillset is not None
        else resolve_skillset_threshold_scale(cfg)
    )
    selected_is_globalref = (
        selected_skillset is not None
        and "_globalref_" in selected_skillset["seg_suffix"]
    )
    skillset_global_threshold_source = (
        str(
            (selected_skillset["manifest"].get("detector") or {}).get(
                "global_threshold_path", ""
            )
        )
        if selected_is_globalref
        else (
            ""
            if selected_skillset is not None
            else resolve_skillset_global_threshold_source(cfg, root)
        )
    )
    if skillset_global_threshold_source and skillset_boundary_threshold_mode != "global_mean":
        raise ValueError(
            "skillset_global_threshold_source requires skillset_boundary_threshold_mode=global_mean."
        )
    # Both the threshold scope and scale are always tagged so distinct
    # segmentations cannot share a directory (for example episodemean_80p).
    if selected_is_globalref:
        threshold_scope_tag = "globalref"
    elif skillset_boundary_threshold_mode == "global_mean":
        threshold_scope_tag = (
            "globalref" if skillset_global_threshold_source else "globalmean"
        )
    else:
        threshold_scope_tag = "episodemean"
    threshold_percent_tag = _scale_percent_tag(skillset_boundary_threshold_scale)
    skillset_threshold_name = f"{threshold_scope_tag}_{threshold_percent_tag}"
    skillset_threshold_suffix = f"_{skillset_threshold_name}"
    # min_skills=1 is the normal setting and is omitted. Non-default values
    # remain explicit so experiments with different episode filtering cannot mix.
    skillset_min_skills_suffix = (
        "" if skillset_min_skills == 1 else f"_ms{skillset_min_skills}"
    )
    if selected_skillset is not None:
        canonical_suffix = (
            probe_settings["skillset_probe_suffix"]
            + skillset_threshold_suffix
            + skillset_min_skills_suffix
        )
        if not selected_skillset["seg_suffix"].startswith(canonical_suffix):
            raise ValueError(
                "Selected seg folder identity disagrees with its manifest: "
                f"expected suffix prefix {canonical_suffix!r}, "
                f"got {selected_skillset['seg_suffix']!r}"
            )
        skillset_output_suffix = selected_skillset["seg_suffix"][len(canonical_suffix) :]
    else:
        skillset_output_suffix = resolve_skillset_output_suffix(cfg, root)
    skillset_suffix = (
        probe_settings["skillset_probe_suffix"]
        + skillset_threshold_suffix
        + skillset_min_skills_suffix
        + skillset_output_suffix
    )
    fsq_seg_dir = (
        selected_skillset["seg_dir"]
        if selected_skillset is not None
        else fsq_inputs_dir / f"seg_{dp_policy}_ck{dp_checkpoint}{skillset_suffix}"
    )

    fsq_levels = as_levels(get_value(cfg, "fsq_levels", [5, 5, 5]))
    fsq_tag = "fsq" + "".join(str(v) for v in fsq_levels)
    fsq_exp = str(get_value(cfg, "fsq_exp", "")).strip()
    fsq_exp_suffix = f"_{fsq_exp}" if fsq_exp else ""
    fsq_samples_per_skill = int(get_value(cfg, "fsq_samples_per_skill", 2))
    fsq_lr_schedule = str(get_value(cfg, "fsq_lr_schedule", "cosine")).strip().lower()
    if fsq_lr_schedule not in {"cosine", "constant"}:
        raise ValueError(f"fsq_lr_schedule must be cosine|constant, got {fsq_lr_schedule!r}.")
    fsq_terminator_arch = str(get_value(cfg, "fsq_terminator_arch", "small"))
    if fsq_terminator_arch not in {"small", "cond"}:
        raise ValueError(f"fsq_terminator_arch must be small|cond, got {fsq_terminator_arch!r}.")
    fsq_terminator_layers = int(get_value(cfg, "fsq_terminator_layers", 2))
    fsq_terminator_heads = int(get_value(cfg, "fsq_terminator_heads", 4))
    if fsq_terminator_layers < 1 or fsq_terminator_heads < 1:
        raise ValueError("fsq_terminator_layers and fsq_terminator_heads must both be >= 1.")
    fsq_terminator_termination_only = as_bool(
        get_value(cfg, "fsq_terminator_termination_only", False)
    )
    fsq_reconstructor_only = as_bool(get_value(cfg, "fsq_reconstructor_only", False))
    if fsq_reconstructor_only and fsq_terminator_termination_only:
        raise ValueError(
            "fsq_reconstructor_only and fsq_terminator_termination_only are mutually "
            "exclusive: reconstructor_only builds no terminator at all."
        )
    fsq_terminator_only = as_bool(get_value(cfg, "fsq_terminator_only", False))
    if fsq_reconstructor_only and fsq_terminator_only:
        raise ValueError(
            "fsq_reconstructor_only and fsq_terminator_only are mutually exclusive."
        )
    fsq_cond_encoder_variant = str(get_value(cfg, "fsq_cond_encoder_variant", "gemma_300m"))
    fsq_terminator_tag = fsq_terminator_arch
    fsq_vision_backbone = str(get_value(cfg, "fsq_vision_backbone", "dino"))
    if fsq_vision_backbone not in {"dino", "siglip"}:
        raise ValueError(f"fsq_vision_backbone must be dino|siglip, got {fsq_vision_backbone!r}.")
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
    fsq_skill_cond_mode = str(get_value(cfg, "fsq_skill_cond_mode", "token")).strip().lower()
    if fsq_skill_cond_mode not in {"token", "broadcast"}:
        raise ValueError(
            "fsq_skill_cond_mode must be token|broadcast, "
            f"got {fsq_skill_cond_mode!r}."
        )
    fsq_skill_cond_suffix = "" if fsq_skill_cond_mode == "token" else "_broadcast"
    # DP tag for the FSQ run name = the DP run with the dataset prefix stripped
    # (libero_90_full_full_state_obs20 → state_obs20), so the FSQ folder shows WHICH DP's skillset it
    # was trained on (for example state_obs20), not just the FSQ architecture.
    dp_tag = dp_policy[len(target_dataset) + 1:] if dp_policy.startswith(f"{target_dataset}_") else dp_policy
    if selected_skillset is not None:
        dp_tag += selected_skillset["seg_suffix"]
    else:
        dp_tag += probe_settings["skillset_probe_suffix"]
        dp_tag += skillset_threshold_suffix
        dp_tag += skillset_min_skills_suffix
        dp_tag += skillset_output_suffix
    # Architecture knobs (vision backbone/freeze, terminator, encoder input mode,
    # skill cond mode) are fixed project-wide and deliberately NOT
    # part of the name anymore — use fsq_exp to separate runs if one is ever varied.
    fsq_run_template = str(
        get_value(
            cfg,
            "fsq_run_name",
            "{target_dataset}_{dp_tag}_{fsq_tag}{fsq_exp_suffix}",
        )
    )
    fsq_run_name = fsq_run_template.format(
        target_dataset=target_dataset,
        dp_tag=dp_tag,
        fsq_tag=fsq_tag,
        fsq_exp=fsq_exp,
        fsq_exp_suffix=fsq_exp_suffix,
        fsq_vision_tag=fsq_vision_tag,
        fsq_terminator_arch=fsq_terminator_arch,
        fsq_terminator_tag=fsq_terminator_tag,
        fsq_terminator_layers=fsq_terminator_layers,
        fsq_terminator_heads=fsq_terminator_heads,
        fsq_patch_grid=int(get_value(cfg, "fsq_patch_grid", 8)),
        fsq_skill_cond_mode=fsq_skill_cond_mode,
        fsq_skill_cond_suffix=fsq_skill_cond_suffix,
        fsq_encoder_input_mode=fsq_encoder_input_mode,
        fsq_encoder_input_suffix=fsq_encoder_input_suffix,
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
        "fsq_encoder_input_suffix": fsq_encoder_input_suffix,
        "fsq_skill_cond_mode": fsq_skill_cond_mode,
        "fsq_skill_cond_suffix": fsq_skill_cond_suffix,
        "fsq_run_name": fsq_run_name,
        "fsq_output_dir": fsq_outputs_root / fsq_run_name,
        "fsq_dim": len(fsq_levels),
        "fsq_num_embeddings": math.prod(fsq_levels),
        "fsq_epoch": str(get_value(cfg, "fsq_epoch", "1000")),
        "fsq_patch_grid": int(get_value(cfg, "fsq_patch_grid", 8)),
        "fsq_batch_size": int(get_value(cfg, "fsq_batch_size", 256)),
        "fsq_num_workers": int(get_value(cfg, "fsq_num_workers", 8)),
        "fsq_val_num_workers": int(get_value(cfg, "fsq_val_num_workers", 0)),
        "fsq_val_every": int(get_value(cfg, "fsq_val_every", 1)),
        "fsq_save_best_model": as_bool(get_value(cfg, "fsq_save_best_model", True)),
        "fsq_gradient_checkpointing": as_bool(get_value(cfg, "fsq_gradient_checkpointing", False)),
        "fsq_num_epochs": int(get_value(cfg, "fsq_num_epochs", 1000)),
        "fsq_checkpoint_every": int(get_value(cfg, "fsq_checkpoint_every", 500)),
        "fsq_encoder_lr": str(get_value(cfg, "fsq_encoder_lr", "3e-4")),
        "fsq_terminator_lr": str(get_value(cfg, "fsq_terminator_lr", "3e-4")),
        "fsq_reconstructor_lr": str(get_value(cfg, "fsq_reconstructor_lr", get_value(cfg, "fsq_encoder_lr", "3e-4"))),
        "fsq_lr": str(get_value(cfg, "fsq_lr", get_value(cfg, "fsq_encoder_lr", "3e-4"))),
        "fsq_lr_schedule": fsq_lr_schedule,
        "fsq_samples_per_skill": fsq_samples_per_skill,
        "fsq_encoder_input_mode": fsq_encoder_input_mode,
        "fsq_terminator_arch": fsq_terminator_arch,
        "fsq_terminator_layers": fsq_terminator_layers,
        "fsq_terminator_heads": fsq_terminator_heads,
        "fsq_terminator_termination_only": fsq_terminator_termination_only,
        "fsq_reconstructor_only": fsq_reconstructor_only,
        "fsq_terminator_only": fsq_terminator_only,
        "fsq_vision_backbone": fsq_vision_backbone,
        "fsq_freeze_vision_encoder": fsq_freeze_vision_encoder,
        "fsq_hidden_dim": int(get_value(cfg, "fsq_hidden_dim", 256)),
        "fsq_num_layers": int(get_value(cfg, "fsq_num_layers", 2)),
        "fsq_n_control": int(get_value(cfg, "fsq_n_control", 30)),
        "fsq_image_token_dim": int(get_value(cfg, "fsq_image_token_dim", 128)),
        "fsq_terminator_use_third": as_bool(get_value(cfg, "fsq_terminator_use_third", True)),
        "fsq_terminator_use_wrist": as_bool(get_value(cfg, "fsq_terminator_use_wrist", False)),
        # Accept a portable project-relative path in YAML (e.g. models/dinov3-vits16), while
        # passing an absolute path to the Slurm job and checkpoint metadata.
        "fsq_dino_model_path": resolve_path(
            root, get_value(cfg, "fsq_dino_model_path", f"models/{DINO_IMAGE_MODEL_DIR}")
        ),
        "dino_image_model_path": resolve_path(
            root, get_value(cfg, "fsq_dino_model_path", f"models/{DINO_IMAGE_MODEL_DIR}")
        ),
        "dino_feature_dim": int(get_value(cfg, "dino_feature_dim", get_value(cfg, "fsq_dino_feature_dim", 384))),
        "dino_image_size": int(get_value(cfg, "dino_image_size", get_value(cfg, "fsq_dino_image_size", 224))),
        "fsq_dino_image_size": int(get_value(cfg, "fsq_dino_image_size", 224)),
        "fsq_siglip_image_size": int(get_value(cfg, "fsq_siglip_image_size", 224)),
        "fsq_cond_encoder_variant": fsq_cond_encoder_variant,
        "fsq_chunk_size": int(get_value(cfg, "fsq_chunk_size", 10)),
        "fsq_action_loss_weight": str(get_value(cfg, "fsq_action_loss_weight", 1.0)),
        "fsq_delta_loss_weight": str(
            get_value(cfg, "fsq_delta_loss_weight", get_value(cfg, "fsq_action_loss_weight", 1.0))
        ),
        "fsq_progress_loss_weight": str(get_value(cfg, "fsq_progress_loss_weight", 1.0)),
        "fsq_end_loss_weight": str(get_value(cfg, "fsq_end_loss_weight", 1.0)),
        # best-val SELECTION metric weights — empty → "" (sbatch omits the flag → selection follows loss)
        "fsq_val_select_action_weight": str(get_value(cfg, "fsq_val_select_action_weight", "") or ""),
        "fsq_val_select_delta_weight": str(
            get_value(cfg, "fsq_val_select_delta_weight", get_value(cfg, "fsq_val_select_action_weight", "")) or ""
        ),
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
        "skillset_seg_name": fsq_seg_dir.name,
        "skillset_manifest_path": (
            selected_skillset["manifest_path"]
            if selected_skillset is not None
            else fsq_seg_dir / str(get_value(cfg, "skillset_name", "skillset"))
            / "skillset_manifest.json"
        ),
        "skillset_tasks_per_job": int(get_value(cfg, "skillset_tasks_per_job", 5)),
        "skillset_wandb_project": str(get_value(cfg, "skillset_wandb_project", "Skill_dataset")),
        "skillset_dn_step": int(get_value(cfg, "skillset_dn_step", 7)),
        "skillset_n_gmm": int(get_value(cfg, "skillset_n_gmm", 5)),
        "skillset_smooth_window": int(get_value(cfg, "skillset_smooth_window", 7)),
        "skillset_savgol_polyorder": int(get_value(cfg, "skillset_savgol_polyorder", 4)),
        "skillset_replan_interval": int(get_value(cfg, "skillset_replan_interval", 3)),
        "skillset_nms_dist": int(get_value(cfg, "skillset_nms_dist", 25)),
        "skillset_min_skills": skillset_min_skills,
        "skillset_min_skills_suffix": skillset_min_skills_suffix,
        "skillset_min_skill_len": skillset_min_skill_len,
        **probe_settings,
        "skillset_boundary_threshold_mode": skillset_boundary_threshold_mode,
        "skillset_boundary_threshold_scale": skillset_boundary_threshold_scale,
        "skillset_boundary_threshold_scale_tag": _scale_percent_tag(
            skillset_boundary_threshold_scale
        ),
        "skillset_boundary_threshold_name": skillset_threshold_name,
        "skillset_boundary_threshold_suffix": skillset_threshold_suffix,
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
