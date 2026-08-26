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


def _compact_decimal_tag(value: float) -> str:
    """Filesystem-safe compact decimal tag (0.1 -> 01, 0.25 -> 025)."""
    text = f"{value:.12g}"
    if "e" in text.lower():
        text = f"{value:.12f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


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


def resolve_dp_eef_scales(dataset_dir: Path) -> tuple[float, float]:
    """Load the OSC execution scales from one derived LIBERO dataset contract."""
    contract_path = dataset_dir / "meta" / "action_contract.json"
    stats_path = dataset_dir / "meta" / "relative_action_stats.json"
    missing = [path for path in (contract_path, stats_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "dp_eef_relative=true requires the derived LIBERO dataset contract and stats; "
            f"missing: {', '.join(str(path) for path in missing)}"
        )

    contract = json.loads(contract_path.read_text())
    stats = json.loads(stats_path.read_text())
    expected_contract = {
        "storage_representation": "absolute_eef_command",
        "model_representation": "eef_anchor_relative_so3",
        "rotation_representation": "axis_angle_rotation_vector",
        "rotation_composition": "left_world",
    }
    contract_mismatches = {
        key: (contract.get(key), expected)
        for key, expected in expected_contract.items()
        if contract.get(key) != expected
    }
    if contract_mismatches:
        raise ValueError(
            f"Unsupported EEF action contract in {contract_path}: {contract_mismatches}"
        )

    resolved: list[float] = []
    for key in ("osc_position_scale", "osc_rotation_scale"):
        try:
            contract_value = float(contract[key])
            stats_value = float(stats[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid or missing {key} in {contract_path} or {stats_path}"
            ) from error
        if not math.isfinite(contract_value) or contract_value <= 0.0:
            raise ValueError(f"{key} must be finite and positive, got {contract_value}")
        if not math.isclose(contract_value, stats_value, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(
                f"{key} mismatch inside derived dataset: "
                f"action_contract={contract_value}, relative_stats={stats_value}"
            )
        resolved.append(contract_value)
    return resolved[0], resolved[1]


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
    fsq_frame_cache_enabled = as_bool(
        get_value(cfg, "fsq_frame_cache_enabled", True)
    )
    fsq_frame_cache_stage_local = as_bool(
        get_value(cfg, "fsq_frame_cache_stage_local", True)
    )
    fsq_frame_cache_local_root = str(
        get_value(cfg, "fsq_frame_cache_local_root", "") or ""
    ).strip()
    if fsq_frame_cache_local_root and not Path(
        fsq_frame_cache_local_root
    ).is_absolute():
        raise ValueError(
            "fsq_frame_cache_local_root must be empty or an absolute node-local path."
        )
    fsq_frame_cache_local_reserve_gb = int(
        get_value(cfg, "fsq_frame_cache_local_reserve_gb", 16)
    )
    if fsq_frame_cache_local_reserve_gb < 0:
        raise ValueError("fsq_frame_cache_local_reserve_gb must be >= 0.")
    default_frame_cache_root = (
        root
        / ".cache"
        / "fsq_frame_cache"
        / dataset_root_name
        / target_dataset
        / "rgb_zstd_v2"
    )
    configured_frame_cache_root = str(
        get_value(cfg, "fsq_frame_cache_root", "") or ""
    ).strip()
    fsq_frame_cache_root = (
        resolve_path(root, configured_frame_cache_root)
        if configured_frame_cache_root
        else str(default_frame_cache_root)
    )
    # The submit script supplies the fingerprinted, completed directory through
    # FSQ_FRAME_CACHE_DIR. It stays blank during the initial config resolution.
    fsq_frame_cache_dir = resolve_path(
        root,
        get_value(cfg, "fsq_frame_cache_dir", ""),
    )
    fsq_frame_cache_partition = str(
        get_value(cfg, "fsq_frame_cache_partition", "dell_cpu")
    ).strip()
    fsq_frame_cache_qos = str(
        get_value(cfg, "fsq_frame_cache_qos", "cpu_qos")
    ).strip()
    fsq_frame_cache_cpus_per_task = int(
        get_value(cfg, "fsq_frame_cache_cpus_per_task", 16)
    )
    fsq_frame_cache_workers = int(
        get_value(cfg, "fsq_frame_cache_workers", 16)
    )
    fsq_frame_cache_decoder_threads = int(
        get_value(cfg, "fsq_frame_cache_decoder_threads", 1)
    )
    if not fsq_frame_cache_partition or not fsq_frame_cache_qos:
        raise ValueError("FSQ frame-cache partition and qos must be non-empty.")
    if min(
        fsq_frame_cache_cpus_per_task,
        fsq_frame_cache_workers,
        fsq_frame_cache_decoder_threads,
    ) < 1:
        raise ValueError(
            "FSQ frame-cache cpus_per_task, workers, and decoder_threads must all be >= 1."
        )

    dp_n_obs_steps = int(get_value(cfg, "dp_n_obs_steps", 10))
    dp_horizon = int(get_value(cfg, "dp_horizon", 16))
    dp_relative = as_bool(get_value(cfg, "dp_relative", False))
    dp_eef_relative = as_bool(get_value(cfg, "dp_eef_relative", False))
    if dp_relative and dp_eef_relative:
        raise ValueError("dp_relative and dp_eef_relative are mutually exclusive.")
    dp_n_action_steps = int(
        get_value(
            cfg,
            "dp_n_action_steps",
            1 if dp_eef_relative else dp_horizon - dp_n_obs_steps + 1,
        )
    )
    if dp_eef_relative and dp_n_action_steps != 1:
        raise ValueError("dp_eef_relative requires dp_n_action_steps=1.")
    if dp_eef_relative:
        deprecated_scale_keys = [
            key
            for key in ("dp_eef_position_scale", "dp_eef_rotation_scale")
            if key in cfg or key.upper() in os.environ
        ]
        if deprecated_scale_keys:
            raise ValueError(
                "Remove manual EEF scale settings; they are loaded from the dataset contract: "
                f"{deprecated_scale_keys}"
            )
        dp_eef_position_scale, dp_eef_rotation_scale = resolve_dp_eef_scales(
            raw_dataset_dir
        )
    else:
        # Inert compatibility values; they are not passed to the policy unless EEF-relative mode is on.
        dp_eef_position_scale, dp_eef_rotation_scale = 0.05, 0.5
    dp_action_suffix = "_eefrel" if dp_eef_relative else ""
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
        dp_action_suffix=dp_action_suffix,
    )
    if dp_eef_relative and "{dp_action_suffix}" not in dp_policy_template:
        dp_policy += dp_action_suffix
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

    fsq_quantizer = str(get_value(cfg, "fsq_quantizer", "fsq")).strip().lower()
    if fsq_quantizer not in {"fsq", "bsq"}:
        raise ValueError(
            f"fsq_quantizer must be fsq|bsq, got {fsq_quantizer!r}."
        )
    bsq_code_dim = int(get_value(cfg, "bsq_code_dim", 5))
    if bsq_code_dim < 2:
        raise ValueError(f"bsq_code_dim must be >= 2, got {bsq_code_dim}.")
    configured_fsq_levels = as_levels(get_value(cfg, "fsq_levels", [5, 5, 5]))
    fsq_levels = (
        [2] * bsq_code_dim
        if fsq_quantizer == "bsq"
        else configured_fsq_levels
    )
    fsq_tag = (
        f"bsq{bsq_code_dim}"
        if fsq_quantizer == "bsq"
        else "fsq" + "".join(str(v) for v in fsq_levels)
    )
    # Main FSQ uses the compact preset interface below. FSQ-original and BSQ
    # intentionally keep their historical surfaces and launcher contracts.
    legacy_original_config = "fsq_orig_encoder_arch" in cfg
    fsq_exp = str(get_value(cfg, "fsq_exp", "")).strip()
    if legacy_original_config and not fsq_exp:
        raise ValueError("fsq_exp is required for legacy FSQ-original/BSQ runs.")
    if fsq_exp and (
        fsq_exp in {".", ".."}
        or Path(fsq_exp).name != fsq_exp
        or any(not (char.isalnum() or char in "._-") for char in fsq_exp)
    ):
        raise ValueError(
            "fsq_exp must be empty or one safe suffix using letters, numbers, "
            "'.', '_', or '-', "
            f"got {fsq_exp!r}."
        )

    # Compact mappings are source configuration, while the similarly named
    # FSQ_* environment variables are flattened outputs from a previous
    # resolution. Slurm inherits those outputs and resolves the snapshot a
    # second time, so allowing env precedence here would replace a mapping with
    # a scalar (for example {enabled: ...} -> "true").
    raw_calibration = cfg.get("fsq_init_calibration", False)
    if not legacy_original_config:
        exposed = sorted(
            {"fsq_init_calibration_gain", "fsq_init_calibration_samples"}.intersection(
                cfg
            )
        )
        if exposed or (
            "fsq_init_calibration" in cfg and not isinstance(raw_calibration, dict)
        ):
            details = exposed or ["fsq_init_calibration"]
            raise ValueError(
                "Main FSQ calibration uses one fsq_init_calibration mapping; "
                "remove split/scalar keys: " + ", ".join(details)
            )
    if isinstance(raw_calibration, dict):
        unknown = sorted(
            set(raw_calibration).difference({"enabled", "gain", "samples"})
        )
        if unknown:
            raise ValueError(
                "fsq_init_calibration supports enabled|gain|samples, got: "
                + ", ".join(unknown)
            )
        fsq_init_calibration = as_bool(raw_calibration.get("enabled", False))
        fsq_init_calibration_gain = float(raw_calibration.get("gain", 1.0))
        fsq_init_calibration_samples = int(raw_calibration.get("samples", 0))
    else:
        # Backward-compatible parsing for FSQ-original/BSQ and old configs.
        fsq_init_calibration = as_bool(raw_calibration)
        fsq_init_calibration_gain = float(
            get_value(cfg, "fsq_init_calibration_gain", 1.0)
        )
        fsq_init_calibration_samples = int(
            get_value(cfg, "fsq_init_calibration_samples", 0)
        )
    if fsq_init_calibration_gain <= 0:
        raise ValueError(
            "fsq_init_calibration_gain must be positive, "
            f"got {fsq_init_calibration_gain}."
        )
    if fsq_init_calibration_samples < 0:
        raise ValueError(
            "fsq_init_calibration_samples must be non-negative, "
            f"got {fsq_init_calibration_samples}."
        )
    fsq_samples_per_skill = int(get_value(cfg, "fsq_samples_per_skill", 2))
    fsq_lr_schedule = str(get_value(cfg, "fsq_lr_schedule", "cosine")).strip().lower()
    if fsq_lr_schedule not in {"cosine", "constant"}:
        raise ValueError(f"fsq_lr_schedule must be cosine|constant, got {fsq_lr_schedule!r}.")
    if legacy_original_config:
        fsq_encoder_lr = str(get_value(cfg, "fsq_encoder_lr", "3e-4"))
        fsq_terminator_lr = str(
            get_value(cfg, "fsq_terminator_lr", fsq_encoder_lr)
        )
        fsq_reconstructor_lr = str(
            get_value(cfg, "fsq_reconstructor_lr", fsq_encoder_lr)
        )
        fsq_lr = str(get_value(cfg, "fsq_lr", fsq_encoder_lr))
    else:
        split_lr_keys = {
            "fsq_encoder_lr",
            "fsq_reconstructor_lr",
            "fsq_terminator_lr",
        }
        exposed = sorted(split_lr_keys.intersection(cfg))
        if exposed:
            raise ValueError(
                "Main FSQ uses one fsq_lr for encoder, reconstructor, and "
                "terminator; remove split LR keys: " + ", ".join(exposed)
            )
        fsq_lr = str(get_value(cfg, "fsq_lr", "3e-4"))
        fsq_encoder_lr = fsq_lr
        fsq_reconstructor_lr = fsq_lr
        fsq_terminator_lr = fsq_lr
    try:
        parsed_fsq_lr = float(fsq_lr)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fsq_lr must be numeric, got {fsq_lr!r}.") from exc
    if parsed_fsq_lr <= 0:
        raise ValueError(f"fsq_lr must be positive, got {fsq_lr!r}.")
    if legacy_original_config:
        raw_terminator_arch = str(
            get_value(cfg, "fsq_terminator_arch", "default")
        ).strip().lower()
        # Legacy configs used fsq_terminator_arch for small/fusion as well as
        # for the later default/rnn model choice.
        if raw_terminator_arch in {"small", "fusion"}:
            fsq_terminator_model = "default"
            legacy_visual_arch = raw_terminator_arch
        elif raw_terminator_arch in {"default", "rnn"}:
            fsq_terminator_model = raw_terminator_arch
            legacy_visual_arch = "small"
        else:
            raise ValueError(
                "fsq_terminator_arch must be default|rnn (or legacy "
                f"small|fusion), got {raw_terminator_arch!r}."
            )
        fsq_terminator_default_arch = str(
            get_value(cfg, "fsq_terminator_default_arch", legacy_visual_arch)
        ).strip().lower()
        fsq_vision_backbone = str(
            get_value(cfg, "fsq_vision_backbone", "dino")
        ).strip().lower()
        fsq_freeze_vision_encoder = as_bool(
            get_value(cfg, "fsq_freeze_vision_encoder", True)
        )
        fsq_terminator_context = str(
            get_value(cfg, "fsq_terminator_context", "proprio")
        ).strip().lower()
    else:
        hidden_terminator_keys = {
            "fsq_decoder_terminator_progress",
            "fsq_decoder_terminator_termination",
            "fsq_terminator_arch",
            "fsq_terminator_input_space",
            "fsq_state_rnn_terminator",
            "fsq_terminator_default_arch",
            "fsq_terminator_layers",
            "fsq_terminator_heads",
            "fsq_vision_backbone",
            "fsq_freeze_vision_encoder",
            "fsq_dino_model_path",
        }
        exposed = sorted(hidden_terminator_keys.intersection(cfg))
        if exposed:
            raise ValueError(
                "Main FSQ terminator is configured only with the compact "
                "fsq_terminator mapping; remove hidden keys: "
                + ", ".join(exposed)
            )
        terminator = cfg.get("fsq_terminator", {})
        if terminator is None:
            terminator = {}
        if not isinstance(terminator, dict):
            raise ValueError("fsq_terminator must be an inline mapping.")
        unknown = sorted(
            set(terminator).difference(
                {
                    "termination",
                    "context",
                    "default_arch",
                    "vision_backbone",
                    "freeze_vision_encoder",
                }
            )
        )
        if unknown:
            raise ValueError(
                "fsq_terminator supports termination|context|default_arch|"
                "vision_backbone|freeze_vision_encoder, got: "
                + ", ".join(unknown)
            )
        # The cleaned trainer always uses the default multimodal terminator.
        fsq_terminator_model = "default"
        fsq_terminator_default_arch = str(
            terminator.get("default_arch", "small")
        ).strip().lower()
        fsq_vision_backbone = str(
            terminator.get("vision_backbone", "dino")
        ).strip().lower()
        fsq_freeze_vision_encoder = as_bool(
            terminator.get("freeze_vision_encoder", True)
        )
        # Missing means an immutable pre-migration snapshot: keep its original
        # absolute-proprio architecture on requeue. New configs state the fixed
        # prev_action contract explicitly in the compact mapping.
        fsq_terminator_context = str(
            terminator.get("context", "proprio")
        ).strip().lower()
    if fsq_terminator_context not in {"prev_action", "proprio"}:
        raise ValueError(
            "fsq_terminator context must be prev_action|proprio, "
            f"got {fsq_terminator_context!r}."
        )
    if fsq_terminator_default_arch not in {"small", "fusion"}:
        raise ValueError(
            "fsq_terminator_default_arch must be small|fusion, "
            f"got {fsq_terminator_default_arch!r}."
        )
    if legacy_original_config:
        fsq_terminator_layers = int(get_value(cfg, "fsq_terminator_layers", 2))
        fsq_terminator_heads = int(get_value(cfg, "fsq_terminator_heads", 4))
        if fsq_terminator_layers < 1 or fsq_terminator_heads < 1:
            raise ValueError(
                "fsq_terminator_layers and fsq_terminator_heads must both be >= 1."
            )
    else:
        fsq_terminator_layers = 3
        fsq_terminator_heads = 4
    if not legacy_original_config:
        fsq_decoder_reconstructor = as_bool(
            get_value(cfg, "fsq_decoder_reconstructor", True)
        )
        fsq_decoder_terminator_progress = False
        fsq_decoder_terminator_termination = as_bool(
            terminator.get("termination", True)
        )
    else:
        # Backward-compatible read of the old mutually-dependent booleans.
        legacy_reconstructor_only = as_bool(get_value(cfg, "fsq_reconstructor_only", False))
        legacy_terminator_only = as_bool(get_value(cfg, "fsq_terminator_only", False))
        legacy_termination_only = as_bool(
            get_value(cfg, "fsq_terminator_termination_only", False)
        )
        if legacy_reconstructor_only and legacy_terminator_only:
            raise ValueError("Legacy reconstructor_only and terminator_only cannot both be true.")
        fsq_decoder_reconstructor = not legacy_terminator_only
        terminator_enabled = not legacy_reconstructor_only
        fsq_decoder_terminator_progress = terminator_enabled and not legacy_termination_only
        fsq_decoder_terminator_termination = terminator_enabled
    terminator_enabled = (
        fsq_decoder_terminator_progress or fsq_decoder_terminator_termination
    )
    if not fsq_decoder_reconstructor and not terminator_enabled:
        raise ValueError("At least one FSQ decoder output must be enabled.")
    fsq_reconstructor_only = fsq_decoder_reconstructor and not terminator_enabled
    fsq_terminator_only = not fsq_decoder_reconstructor and terminator_enabled
    fsq_terminator_termination_only = (
        fsq_decoder_terminator_termination and not fsq_decoder_terminator_progress
    )
    if legacy_original_config:
        legacy_state_rnn = as_bool(
            get_value(cfg, "fsq_state_rnn_terminator", False)
        )
        fsq_terminator_input_space = str(
            get_value(
                cfg,
                "fsq_terminator_input_space",
                "state" if legacy_state_rnn else "both",
            )
        ).strip().lower()
        if fsq_terminator_input_space not in {"state", "image", "both"}:
            raise ValueError(
                "fsq_terminator_input_space must be state|image|both, "
                f"got {fsq_terminator_input_space!r}."
            )
        if legacy_state_rnn and "fsq_terminator_arch" not in cfg:
            fsq_terminator_model = "rnn"
        fsq_state_rnn_terminator = (
            terminator_enabled and fsq_terminator_model == "rnn"
        )
        if fsq_state_rnn_terminator and fsq_terminator_input_space != "state":
            raise ValueError(
                "The RNN terminator currently supports input_space=state only."
            )
    else:
        fsq_terminator_input_space = "both"
        fsq_state_rnn_terminator = False
    if fsq_vision_backbone not in {"dino", "siglip", "resnet"}:
        raise ValueError(
            "fsq_vision_backbone must be dino|siglip|resnet, "
            f"got {fsq_vision_backbone!r}."
        )
    # Main FSQ exposes one indivisible encoder/decoder preset. The older
    # FSQ-original/BSQ launchers still carry their own fsq_orig_* architecture
    # surface and are resolved below only for backward compatibility.
    if legacy_original_config:
        fsq_autoencoder_mode = "legacy_original"
        fsq_action_gripper_weight = 1.0
        fsq_start_state_conditioning = "none"
        fsq_encoder_input_mode = str(
            get_value(cfg, "fsq_encoder_input_mode", "zero_grounded")
        ).strip().lower()
        fsq_encoder_input_mode = (
            "raw_state" if fsq_encoder_input_mode == "raw" else fsq_encoder_input_mode
        )
        fsq_encoder_arch = "spline"
        fsq_reconstructor_arch = "oneshot"
        fsq_reconstructor_output_mode = (
            "raw_state"
            if fsq_encoder_input_mode == "raw_state"
            else "zero_grounded"
        )
        fsq_reconstructor_start_state = False
    else:
        hidden_architecture_keys = {
            "fsq_encoder_arch",
            "fsq_encoder_input_mode",
            "fsq_encoder_length_token",
            "fsq_reconstructor_arch",
            "fsq_reconstructor_output_mode",
            "fsq_reconstructor_start_state",
        }
        exposed = sorted(hidden_architecture_keys.intersection(cfg))
        if exposed:
            raise ValueError(
                "Main FSQ architecture is selected only with "
                "fsq_autoencoder={mode: raw|zero|action|norm_action, ...}; "
                "remove hidden keys: "
                + ", ".join(exposed)
            )
        raw_autoencoder = cfg.get("fsq_autoencoder")
        if raw_autoencoder is None:
            # Read-only compatibility for existing snapshots and tests. New
            # main-FSQ configs use the single compact mapping below.
            fsq_autoencoder_mode = str(
                get_value(cfg, "fsq_autoencoder_mode", "")
            ).strip().lower()
            fsq_action_gripper_weight = 1.0
        else:
            if "fsq_autoencoder_mode" in cfg:
                raise ValueError(
                    "Use fsq_autoencoder only; remove legacy fsq_autoencoder_mode."
                )
            if not isinstance(raw_autoencoder, dict):
                raise ValueError("fsq_autoencoder must be an inline mapping.")
            unknown = sorted(
                set(raw_autoencoder).difference({"mode", "gripper_weight"})
            )
            if unknown:
                raise ValueError(
                    "fsq_autoencoder supports mode|gripper_weight, got: "
                    + ", ".join(unknown)
                )
            fsq_autoencoder_mode = str(
                raw_autoencoder.get("mode", "")
            ).strip().lower()
            fsq_action_gripper_weight = float(
                raw_autoencoder.get("gripper_weight", 1.0)
            )
        if (
            not math.isfinite(fsq_action_gripper_weight)
            or not 0.0 < fsq_action_gripper_weight <= 1.0
        ):
            raise ValueError(
                "fsq_autoencoder.gripper_weight must be in (0, 1], got "
                f"{fsq_action_gripper_weight}."
            )
        presets = {
            "raw": {
                "encoder_input_mode": "raw_state",
                "encoder_arch": "spline",
                "reconstructor_arch": "oneshot",
                "reconstructor_output_mode": "raw_state",
            },
            "zero": {
                "encoder_input_mode": "zero_grounded",
                "encoder_arch": "spline",
                "reconstructor_arch": "oneshot",
                "reconstructor_output_mode": "zero_grounded",
            },
            "action": {
                # These coordinate flags are ignored by action-sequence modules;
                # retaining zero_grounded internally avoids expanding old APIs.
                "encoder_input_mode": "zero_grounded",
                "encoder_arch": "action_seq",
                "reconstructor_arch": "action_seq_transformer",
                "reconstructor_output_mode": "zero_grounded",
            },
            "norm_action": {
                # q01/q99-normalized controller actions, clipped to [-1, 1].
                # sqrt(gripper_weight) scales the final action axis in both
                # encoder input and target, yielding that exact MSE weight.
                "encoder_input_mode": "zero_grounded",
                "encoder_arch": "action_seq",
                "reconstructor_arch": "action_seq_transformer",
                "reconstructor_output_mode": "zero_grounded",
            },
        }
        if fsq_autoencoder_mode not in presets:
            raise ValueError(
                "fsq_autoencoder.mode must be raw|zero|action|norm_action, "
                f"got {fsq_autoencoder_mode!r}."
            )
        preset = presets[fsq_autoencoder_mode]
        fsq_encoder_input_mode = preset["encoder_input_mode"]
        fsq_encoder_arch = preset["encoder_arch"]
        fsq_reconstructor_arch = preset["reconstructor_arch"]
        fsq_reconstructor_output_mode = preset["reconstructor_output_mode"]
        fsq_start_state_conditioning = str(
            get_value(cfg, "fsq_start_state_conditioning", "none")
        ).strip().lower()
        if fsq_start_state_conditioning not in {"none", "adaln"}:
            raise ValueError(
                "fsq_start_state_conditioning must be none|adaln, "
                f"got {fsq_start_state_conditioning!r}."
            )
        fsq_reconstructor_start_state = fsq_start_state_conditioning == "adaln"

    if fsq_encoder_input_mode not in {
        "zero_grounded", "start_grounded", "raw_state", "optimal"
    }:
        raise ValueError(
            "fsq_encoder_input_mode must be raw|zero_grounded|start_grounded|optimal, "
            f"got {fsq_encoder_input_mode!r}."
        )
    fsq_encoder_grounding_convention = (
        "trajectory_start_se3_v1"
        if fsq_encoder_input_mode == "start_grounded"
        else "trajectory_mean_xyz_v1"
    )
    fsq_encoder_length_token = False
    fsq_entropy = as_bool(get_value(cfg, "fsq_entropy", False))
    if fsq_quantizer == "bsq" and fsq_entropy:
        raise ValueError(
            "BSQ in fsq_config uses the recon + pair-loss path only; "
            "set fsq_entropy=false."
        )
    fsq_entropy_conf_ceiling = float(
        get_value(cfg, "fsq_entropy_conf_ceiling", 0.0)
    )
    if not 0.0 <= fsq_entropy_conf_ceiling <= 1.0:
        raise ValueError(
            "fsq_entropy_conf_ceiling must be in [0, 1], "
            f"got {fsq_entropy_conf_ceiling}."
        )
    if not legacy_original_config:
        exposed = sorted(
            {
                "fsq_pair_weight",
                "fsq_pair_inv_temperature",
                "fsq_action_loss_weight",
                "fsq_delta_loss_weight",
                "fsq_end_loss_weight",
                "fsq_end_target_sigma",
            }.intersection(cfg)
        )
        if exposed:
            raise ValueError(
                "Main FSQ losses use the compact per-loss mappings; remove "
                "split keys: " + ", ".join(exposed)
            )
    raw_pair_loss = cfg.get("fsq_pair_loss", "none")
    if (
        not legacy_original_config
        and "fsq_pair_loss" in cfg
        and not isinstance(raw_pair_loss, dict)
    ):
        raise ValueError("Main FSQ fsq_pair_loss must be an inline mapping.")
    if isinstance(raw_pair_loss, dict):
        unknown = sorted(
            set(raw_pair_loss).difference({"type", "weight", "inv_temperature"})
        )
        if unknown:
            raise ValueError(
                "fsq_pair_loss supports type|weight|inv_temperature, got: "
                + ", ".join(unknown)
            )
        fsq_pair_loss = str(raw_pair_loss.get("type", "none")).strip().lower()
        fsq_pair_weight = float(raw_pair_loss.get("weight", 0.1))
        fsq_pair_inv_temperature = float(
            raw_pair_loss.get("inv_temperature", 5.0)
        )
    else:
        # Backward-compatible parsing for older experiment configs.
        fsq_pair_loss = str(raw_pair_loss).strip().lower()
        fsq_pair_weight = float(get_value(cfg, "fsq_pair_weight", 0.1))
        fsq_pair_inv_temperature = float(
            get_value(cfg, "fsq_pair_inv_temperature", 5.0)
        )
    if fsq_pair_loss not in {"none", "overlap", "js", "contrastive"}:
        raise ValueError(
            "fsq_pair_loss must be none|overlap|js|contrastive, "
            f"got {fsq_pair_loss!r}."
        )
    if fsq_pair_weight < 0:
        raise ValueError(
            f"fsq_pair_weight must be non-negative, got {fsq_pair_weight}."
        )
    if fsq_pair_inv_temperature <= 0:
        raise ValueError(
            "fsq_pair_inv_temperature must be positive, "
            f"got {fsq_pair_inv_temperature}."
        )
    if "fsq_route_loss" in cfg and "fsq_reconstruction_route_loss" in cfg:
        raise ValueError(
            "Use only fsq_route_loss; remove the legacy "
            "fsq_reconstruction_route_loss key."
        )
    # Keep immutable pre-rename YAML snapshots runnable on requeue.
    raw_route_loss = cfg.get(
        "fsq_route_loss", cfg.get("fsq_reconstruction_route_loss", False)
    )
    if isinstance(raw_route_loss, dict):
        unknown = sorted(set(raw_route_loss).difference({"enabled"}))
        if unknown:
            raise ValueError(
                "fsq_route_loss supports enabled, got: "
                + ", ".join(unknown)
            )
        fsq_route_loss = as_bool(
            raw_route_loss.get("enabled", False)
        )
    else:
        fsq_route_loss = as_bool(raw_route_loss)
    if fsq_route_loss and not fsq_decoder_reconstructor:
        raise ValueError(
            "fsq_route_loss requires "
            "fsq_decoder_reconstructor=true."
        )

    raw_action_loss = cfg.get("fsq_action_loss", {})
    if isinstance(raw_action_loss, dict):
        unknown = sorted(set(raw_action_loss).difference({"weight"}))
        if unknown:
            raise ValueError(
                "fsq_action_loss supports weight, got: " + ", ".join(unknown)
            )
        fsq_action_loss_weight = float(raw_action_loss.get("weight", 1.0))
    else:
        raise ValueError("fsq_action_loss must be an inline mapping.")
    raw_end_loss = cfg.get("fsq_end_loss", {})
    if isinstance(raw_end_loss, dict):
        unknown = sorted(set(raw_end_loss).difference({"weight", "target_sigma"}))
        if unknown:
            raise ValueError(
                "fsq_end_loss supports weight|target_sigma, got: "
                + ", ".join(unknown)
            )
        fsq_end_loss_weight = float(raw_end_loss.get("weight", 1.0))
        fsq_end_target_sigma = float(raw_end_loss.get("target_sigma", 0.0))
    else:
        raise ValueError("fsq_end_loss must be an inline mapping.")
    if fsq_action_loss_weight < 0 or fsq_end_loss_weight < 0:
        raise ValueError("FSQ action/end loss weights must be non-negative.")
    if fsq_end_target_sigma < 0:
        raise ValueError("fsq_end_loss.target_sigma must be non-negative.")
    raw_pair_warmup = cfg.get("fsq_pair_warmup", False)
    if not legacy_original_config:
        exposed = sorted(
            {"fsq_pair_warmup_epochs", "fsq_pair_ramp_epochs"}.intersection(cfg)
        )
        if exposed or (
            "fsq_pair_warmup" in cfg and not isinstance(raw_pair_warmup, dict)
        ):
            details = exposed or ["fsq_pair_warmup"]
            raise ValueError(
                "Main FSQ pair warm-up uses one fsq_pair_warmup mapping; "
                "remove split/scalar keys: " + ", ".join(details)
            )
    if isinstance(raw_pair_warmup, dict):
        unknown = sorted(
            set(raw_pair_warmup).difference({"enabled", "epochs", "ramp_epochs"})
        )
        if unknown:
            raise ValueError(
                "fsq_pair_warmup supports enabled|epochs|ramp_epochs, got: "
                + ", ".join(unknown)
            )
        fsq_pair_warmup = as_bool(raw_pair_warmup.get("enabled", False))
        fsq_pair_warmup_epochs = int(raw_pair_warmup.get("epochs", 0))
        fsq_pair_ramp_epochs = int(raw_pair_warmup.get("ramp_epochs", 0))
    else:
        # Backward-compatible parsing for older experiment configs.
        fsq_pair_warmup = as_bool(raw_pair_warmup)
        fsq_pair_warmup_epochs = int(
            get_value(cfg, "fsq_pair_warmup_epochs", 0)
        )
        fsq_pair_ramp_epochs = int(get_value(cfg, "fsq_pair_ramp_epochs", 0))
    if fsq_pair_warmup_epochs < 0 or fsq_pair_ramp_epochs < 0:
        raise ValueError(
            "fsq_pair_warmup_epochs and fsq_pair_ramp_epochs must be non-negative, "
            f"got {fsq_pair_warmup_epochs} and {fsq_pair_ramp_epochs}."
        )
    legacy_boundary_aug_pmax = int(get_value(cfg, "fsq_boundary_aug_pmax", 0))
    if legacy_boundary_aug_pmax < 0:
        raise ValueError(
            "fsq_boundary_aug_pmax must be non-negative, "
            f"got {legacy_boundary_aug_pmax}."
        )

    directional_boundary_aug_pmaxes: dict[str, int] = {}
    for direction in ("early_start", "late_start", "early_end", "late_end"):
        key = f"fsq_boundary_aug_{direction}_pmax"
        raw_value = get_value(cfg, key, None)
        value = legacy_boundary_aug_pmax if raw_value is None else int(raw_value)
        if value < 0:
            raise ValueError(f"{key} must be non-negative, got {value}.")
        directional_boundary_aug_pmaxes[direction] = value
    fsq_boundary_aug_pmax = max(directional_boundary_aug_pmaxes.values())
    if legacy_original_config:
        fsq_boundary_aug_distribution = str(
            get_value(cfg, "fsq_boundary_aug_distribution", "half_normal")
        ).strip().lower().replace("-", "_").replace(" ", "_")
        if fsq_boundary_aug_distribution not in {"half_normal", "uniform"}:
            raise ValueError(
                "fsq_boundary_aug_distribution must be half_normal|uniform, "
                f"got {fsq_boundary_aug_distribution!r}."
            )
    else:
        if "fsq_boundary_aug_distribution" in cfg:
            raise ValueError(
                "Main FSQ boundary augmentation distribution is fixed to "
                "half_normal; remove fsq_boundary_aug_distribution."
            )
        fsq_boundary_aug_distribution = "half_normal"
    if fsq_pair_loss != "none":
        if not any(directional_boundary_aug_pmaxes.values()):
            raise ValueError(
                "fsq_pair_loss requires at least one positive directional "
                "boundary augmentation pmax."
            )
    fsq_skill_cond_mode = str(get_value(cfg, "fsq_skill_cond_mode", "token")).strip().lower()
    if fsq_skill_cond_mode not in {"token", "broadcast"}:
        raise ValueError(
            "fsq_skill_cond_mode must be token|broadcast, "
            f"got {fsq_skill_cond_mode!r}."
        )
    if legacy_original_config:
        # FSQ-original/BSQ retain their historical explicit folder names.
        fsq_decoder_name = "legacy"
        fsq_loss_name = "legacy"
        fsq_run_name = fsq_exp
    else:
        vision_name = {
            "dino": "DINO",
            "resnet": "RES",
            "siglip": "SIGLIP",
        }[fsq_vision_backbone]
        if fsq_decoder_reconstructor and not terminator_enabled:
            fsq_decoder_name = "recon_only"
        elif not fsq_decoder_reconstructor and terminator_enabled:
            fsq_decoder_name = f"term{vision_name}_only"
        else:
            fsq_decoder_name = f"recon_term{vision_name}"

        pair_name = "pairOFF" if fsq_pair_loss == "none" else f"{fsq_pair_loss}ON"
        route_name = "routeON" if fsq_route_loss else "routeOFF"
        fsq_loss_name = f"{pair_name}_{route_name}_loss"
        fsq_autoencoder_name = (
            fsq_autoencoder_mode
            + _compact_decimal_tag(fsq_action_gripper_weight)
        )
        fsq_run_name = (
            f"{fsq_autoencoder_name}_{fsq_decoder_name}__{fsq_loss_name}"
        )
        if fsq_start_state_conditioning == "adaln":
            fsq_run_name += "__inital_proprio_conditioned"
        if fsq_exp:
            fsq_run_name += f"__{fsq_exp}"
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
        "fsq_frame_cache_enabled": fsq_frame_cache_enabled,
        "fsq_frame_cache_stage_local": fsq_frame_cache_stage_local,
        "fsq_frame_cache_local_root": fsq_frame_cache_local_root,
        "fsq_frame_cache_local_reserve_gb": fsq_frame_cache_local_reserve_gb,
        "fsq_frame_cache_root": fsq_frame_cache_root,
        "fsq_frame_cache_dir": fsq_frame_cache_dir,
        "fsq_frame_cache_partition": fsq_frame_cache_partition,
        "fsq_frame_cache_qos": fsq_frame_cache_qos,
        "fsq_frame_cache_cpus_per_task": fsq_frame_cache_cpus_per_task,
        "fsq_frame_cache_workers": fsq_frame_cache_workers,
        "fsq_frame_cache_decoder_threads": fsq_frame_cache_decoder_threads,
        "fsq_frame_cache_mem": str(
            get_value(cfg, "fsq_frame_cache_mem", "32G")
        ),
        "fsq_frame_cache_time": str(
            get_value(cfg, "fsq_frame_cache_time", "12:00:00")
        ),
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
        "dp_n_action_steps": dp_n_action_steps,
        "dp_horizon": dp_horizon,
        "dp_batch_size": int(get_value(cfg, "dp_batch_size", 64)),
        "dp_relative": dp_relative,
        "dp_eef_relative": dp_eef_relative,
        "dp_eef_position_scale": dp_eef_position_scale,
        "dp_eef_rotation_scale": dp_eef_rotation_scale,
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
        "fsq_quantizer": fsq_quantizer,
        "bsq_code_dim": bsq_code_dim,
        "fsq_tag": fsq_tag,
        "fsq_exp": fsq_exp,
        "fsq_skill_cond_mode": fsq_skill_cond_mode,
        "fsq_run_name": fsq_run_name,
        "fsq_decoder_name": fsq_decoder_name,
        "fsq_loss_name": fsq_loss_name,
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
        "fsq_encoder_lr": fsq_encoder_lr,
        "fsq_terminator_lr": fsq_terminator_lr,
        "fsq_reconstructor_lr": fsq_reconstructor_lr,
        "fsq_lr": fsq_lr,
        "fsq_lr_schedule": fsq_lr_schedule,
        "fsq_samples_per_skill": fsq_samples_per_skill,
        "fsq_autoencoder_mode": fsq_autoencoder_mode,
        "fsq_action_gripper_weight": str(fsq_action_gripper_weight),
        "fsq_start_state_conditioning": fsq_start_state_conditioning,
        "fsq_encoder_input_mode": fsq_encoder_input_mode,
        "fsq_encoder_grounding_convention": fsq_encoder_grounding_convention,
        "fsq_encoder_length_token": fsq_encoder_length_token,
        "fsq_encoder_arch": fsq_encoder_arch,
        "fsq_entropy": fsq_entropy,
        "fsq_entropy_conf_weight": str(get_value(cfg, "fsq_entropy_conf_weight", 0.1)),
        "fsq_entropy_conf_ceiling": str(fsq_entropy_conf_ceiling),
        "fsq_entropy_div_weight": str(get_value(cfg, "fsq_entropy_div_weight", 0.1)),
        "fsq_entropy_inv_temperature": str(get_value(cfg, "fsq_entropy_inv_temperature", 10.0)),
        "fsq_init_calibration": fsq_init_calibration,
        "fsq_init_calibration_gain": str(fsq_init_calibration_gain),
        "fsq_init_calibration_samples": fsq_init_calibration_samples,
        "fsq_pair_loss": fsq_pair_loss,
        "fsq_pair_weight": str(fsq_pair_weight),
        "fsq_pair_inv_temperature": str(fsq_pair_inv_temperature),
        "fsq_route_loss": fsq_route_loss,
        "fsq_pair_warmup": fsq_pair_warmup,
        "fsq_pair_warmup_epochs": fsq_pair_warmup_epochs,
        "fsq_pair_ramp_epochs": fsq_pair_ramp_epochs,
        "fsq_boundary_aug_pmax": fsq_boundary_aug_pmax,
        "fsq_boundary_aug_early_start_pmax": directional_boundary_aug_pmaxes["early_start"],
        "fsq_boundary_aug_late_start_pmax": directional_boundary_aug_pmaxes["late_start"],
        "fsq_boundary_aug_early_end_pmax": directional_boundary_aug_pmaxes["early_end"],
        "fsq_boundary_aug_late_end_pmax": directional_boundary_aug_pmaxes["late_end"],
        "fsq_boundary_aug_distribution": fsq_boundary_aug_distribution,
        "fsq_reconstructor_start_state": fsq_reconstructor_start_state,
        "fsq_reconstructor_arch": fsq_reconstructor_arch,
        "fsq_reconstructor_output_mode": fsq_reconstructor_output_mode,
        "fsq_decoder_reconstructor": fsq_decoder_reconstructor,
        "fsq_decoder_terminator_progress": fsq_decoder_terminator_progress,
        "fsq_decoder_terminator_termination": fsq_decoder_terminator_termination,
        "fsq_terminator_input_space": fsq_terminator_input_space,
        "fsq_terminator_context": fsq_terminator_context,
        "fsq_terminator_model": fsq_terminator_model,
        "fsq_terminator_arch": fsq_terminator_model,
        "fsq_terminator_default_arch": fsq_terminator_default_arch,
        "fsq_terminator_layers": fsq_terminator_layers,
        "fsq_terminator_heads": fsq_terminator_heads,
        "fsq_terminator_termination_only": fsq_terminator_termination_only,
        "fsq_reconstructor_only": fsq_reconstructor_only,
        "fsq_terminator_only": fsq_terminator_only,
        "fsq_state_rnn_terminator": fsq_state_rnn_terminator,
        "fsq_vision_backbone": fsq_vision_backbone,
        "fsq_freeze_vision_encoder": fsq_freeze_vision_encoder,
        "fsq_hidden_dim": int(get_value(cfg, "fsq_hidden_dim", 256)),
        "fsq_num_layers": int(get_value(cfg, "fsq_num_layers", 2)),
        "fsq_n_control": int(get_value(cfg, "fsq_n_control", 30)),
        "fsq_image_token_dim": int(get_value(cfg, "fsq_image_token_dim", 128)),
        "fsq_terminator_use_third": as_bool(get_value(cfg, "fsq_terminator_use_third", True)),
        "fsq_terminator_use_wrist": as_bool(get_value(cfg, "fsq_terminator_use_wrist", False)),
        # Main FSQ fixes DINO to the project-local v3-S/16 model. Legacy
        # FSQ-original/BSQ configs may still provide their historical override.
        "fsq_dino_model_path": resolve_path(
            root,
            (
                get_value(
                    cfg,
                    "fsq_dino_model_path",
                    f"models/{DINO_IMAGE_MODEL_DIR}",
                )
                if legacy_original_config
                else f"models/{DINO_IMAGE_MODEL_DIR}"
            ),
        ),
        "dino_image_model_path": resolve_path(
            root,
            (
                get_value(
                    cfg,
                    "fsq_dino_model_path",
                    f"models/{DINO_IMAGE_MODEL_DIR}",
                )
                if legacy_original_config
                else f"models/{DINO_IMAGE_MODEL_DIR}"
            ),
        ),
        "dino_feature_dim": int(get_value(cfg, "dino_feature_dim", get_value(cfg, "fsq_dino_feature_dim", 384))),
        "dino_image_size": int(get_value(cfg, "dino_image_size", get_value(cfg, "fsq_dino_image_size", 224))),
        "fsq_dino_image_size": int(get_value(cfg, "fsq_dino_image_size", 224)),
        "fsq_siglip_image_size": int(get_value(cfg, "fsq_siglip_image_size", 224)),
        "fsq_resnet_image_size": int(get_value(cfg, "fsq_resnet_image_size", 224)),
        "fsq_chunk_size": int(get_value(cfg, "fsq_chunk_size", 10)),
        "fsq_action_loss_weight": str(fsq_action_loss_weight),
        "fsq_delta_loss_weight": str(fsq_action_loss_weight),
        "fsq_progress_loss_weight": str(get_value(cfg, "fsq_progress_loss_weight", 1.0)),
        "fsq_end_loss_weight": str(fsq_end_loss_weight),
        # best-val SELECTION metric weights — empty → "" (sbatch omits the flag → selection follows loss)
        "fsq_val_select_action_weight": str(get_value(cfg, "fsq_val_select_action_weight", "") or ""),
        "fsq_val_select_delta_weight": str(
            get_value(cfg, "fsq_val_select_delta_weight", get_value(cfg, "fsq_val_select_action_weight", "")) or ""
        ),
        "fsq_val_select_progress_weight": str(get_value(cfg, "fsq_val_select_progress_weight", "") or ""),
        "fsq_val_select_end_weight": str(get_value(cfg, "fsq_val_select_end_weight", "") or ""),
        "fsq_end_target_sigma": str(fsq_end_target_sigma),
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
