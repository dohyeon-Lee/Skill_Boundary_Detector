#!/usr/bin/env python3
"""Config helpers for configs/train_skillVLA (SkillVLA data generation).

Resolves paths + run tags for the pipeline that turns trained DP + FSQ models
into SkillVLA training data, and emits them as shell exports (--shell).

Root/yaml helpers are reused from train_skills_config.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# reuse the train_skills yaml-load + shell-emit helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import (  # noqa: E402
    as_bool,
    as_list,
    exists_here,
    get_value,
    load_config,
    print_shell,
    resolve_path,
    stage1_run_dirs,
    skillset_probe_settings,
)
from sbd_build_contract import from_settings as segmentation_contract_from_settings

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "train_skillVLA_config.yaml"


def _levels(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    cleaned = str(value).replace("[", " ").replace("]", " ").replace(",", " ")
    return [int(v) for v in cleaned.split()]


def _scale_percent_tag(scale: float) -> str:
    return f"{scale * 100:g}".replace(".", "p") + "p"


def _skillvla_output_suffix(cfg: dict[str, Any]) -> str:
    """Return a validated optional suffix for the final SkillVLA run folder."""
    raw = str(get_value(cfg, "skillvla_output_suffix", "") or "").strip()
    if not raw:
        return ""
    tag = raw[1:] if raw.startswith("_") else raw
    if not tag or not all(char.isalnum() or char in "._-" for char in tag):
        raise ValueError(
            "skillvla_output_suffix may contain only letters, digits, '.', '_' and '-', "
            f"got {raw!r}."
        )
    return f"_{tag}"


def _proprio_grounding(cfg: dict[str, Any]) -> tuple[str, str]:
    """Return the canonical grounding contract and its dataset identity tag."""
    mode = str(get_value(cfg, "proprio_grounding", "none") or "none").strip().lower()
    aliases = {
        "none": "none",
        "off": "none",
        "false": "none",
        "episode_start_xyz": "episode_start_xyz",
        "episode-start-xyz": "episode_start_xyz",
    }
    if mode not in aliases:
        raise ValueError(
            "proprio_grounding must be none|episode_start_xyz, "
            f"got {mode!r}."
        )
    canonical = aliases[mode]
    return canonical, "" if canonical == "none" else "_grounded"


def _rotation_outlier_exclusion(cfg: dict[str, Any]) -> tuple[bool, str, float]:
    """Return the dataset-level rare-rotation exclusion contract.

    The threshold is deliberately fixed rather than exposed as another tuning
    parameter: LIBERO/LangGap actions live in [-1, 1], while the known LangGap
    episodes contain saturated +/-1 rotation commands and every ordinary
    episode stays below 0.5 in action[3:6].
    """
    enabled = as_bool(get_value(cfg, "exclude_rotation_outlier_episodes", False))
    return enabled, "_except_outlier" if enabled else "", 0.5 if enabled else 0.0


def _infer_libero_suite(source_dataset: str) -> str:
    """Infer the benchmark/original-HDF5 family from a generated dataset name."""
    candidates = (
        "libero_90",
        "libero_10",
        "libero_goal",
        "libero_object",
        "libero_spatial",
    )
    for suite in candidates:
        if source_dataset == suite or source_dataset.startswith(f"{suite}_"):
            return suite
    raise ValueError(
        "focus_uv.suite cannot be inferred from source_dataset "
        f"{source_dataset!r}. Set focus_uv.suite explicitly, or disable focus_uv "
        "for a dataset without recorded LIBERO camera calibration."
    )


def _focus_uv_settings(
    cfg: dict[str, Any],
    *,
    root: Path,
    skillvla_root: Path,
    source_dataset: str,
    run_dir: Path,
) -> dict[str, Any]:
    raw = get_value(cfg, "focus_uv", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            "focus_uv must be a mapping: {enabled: true, camera: agentview, ...}."
        )
    enabled = as_bool(raw.get("enabled", False))
    camera = str(raw.get("camera", "agentview") or "agentview").strip()
    if enabled and not camera:
        raise ValueError("focus_uv.camera cannot be blank when focus_uv is enabled.")

    suite = str(raw.get("suite", "") or "").strip()
    if enabled and not suite:
        suite = _infer_libero_suite(source_dataset)

    exact_raw = str(raw.get("eval_init_states_path", "") or "").strip()
    exact_path = (
        resolve_path(root, exact_raw)
        if exact_raw
        else skillvla_root / source_dataset / "eval_init_states.npz"
    )
    original_raw = str(raw.get("original_dataset_dir", "") or "").strip()
    original_dir = (
        resolve_path(root, original_raw)
        if original_raw
        else root / "libero_original_dataset" / suite
        if suite
        else Path("")
    )
    if enabled:
        if not original_dir.is_dir():
            raise FileNotFoundError(
                "focus_uv requires original LIBERO HDF5 demonstrations with "
                f"recorded camera XML: {original_dir}. Set "
                "focus_uv.original_dataset_dir."
            )
    return {
        "focus_uv_enabled": str(enabled).lower(),
        "focus_uv_camera": camera,
        "focus_uv_suite": suite,
        "focus_uv_eval_init_states_path": exact_path if enabled else "",
        "focus_uv_original_dataset_dir": original_dir if enabled else "",
        "focus_uv_path": run_dir / "skill_focus_uv.npz",
    }


def _fsq_semantic_suffix(fsq_meta: dict[str, Any]) -> str:
    """Build stable FSQ architecture/loss tags from metadata, never the run name."""
    tags: list[str] = []
    if as_bool(fsq_meta.get("decoder_terminator_termination", False)):
        backbone = str(fsq_meta.get("vision_backbone", "dino")).strip().lower()
        terminator_tags = {
            "resnet": "termRES",
            "dino": "termDINO",
            "siglip": "termSIGLIP",
        }
        if backbone not in terminator_tags:
            raise ValueError(
                "FSQ metadata vision_backbone must be resnet|dino|siglip when "
                f"termination is enabled, got {backbone!r}."
            )
        tags.append(terminator_tags[backbone])

    pair_loss = str(fsq_meta.get("pair_loss", "none") or "none").strip().lower()
    if pair_loss in {"js", "contrastive"}:
        tags.append(pair_loss)
    elif pair_loss not in {"", "none", "off", "false"}:
        raise ValueError(
            "FSQ metadata pair_loss must be none|js|contrastive, "
            f"got {pair_loss!r}."
        )
    return "".join(f"_{tag}" for tag in tags)


def _load_fsq_skillset_manifest(
    fsq_meta: dict[str, Any],
    *,
    dataset_root: Path,
    fsq_meta_path: Path,
) -> tuple[dict[str, Any], Path]:
    """Load the immutable skillset contract used to train the selected FSQ."""
    component_keys = (
        "fsq_dataset_root",
        "target_dataset",
        "fsq_inputs_name",
        "skillset_seg_name",
        "skillset_name",
    )
    missing = [key for key in component_keys if not str(fsq_meta.get(key, "")).strip()]
    if missing:
        raise ValueError(
            f"FSQ metadata is missing skillset path fields {missing}: {fsq_meta_path}"
        )
    components = [str(fsq_meta[key]).strip() for key in component_keys]
    invalid = [value for value in components if Path(value).name != value]
    if invalid:
        raise ValueError(
            f"FSQ skillset path fields must be folder names, got {invalid}: {fsq_meta_path}"
        )
    manifest_path = dataset_root.joinpath(*components) / "skillset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "FSQ training skillset manifest not found: "
            f"{manifest_path}. Keep dataset_root aligned with the selected FSQ output root."
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid FSQ training skillset manifest: {manifest_path}: {error}") from error
    return manifest, manifest_path


def _resolve_predictor_checkpoint(
    outputs_root: Path,
    model_name: str,
    checkpoint: str,
) -> tuple[Path, str]:
    """Resolve a standalone or Joint predictor from PT, FT, or historical outputs."""
    if not model_name or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_name) is None:
        raise ValueError(
            "Predictor model must be one Predictor or Joint run "
            f"folder name, got {model_name!r}."
        )
    checkpoint = str(checkpoint or "last").strip()
    if not checkpoint or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", checkpoint) is None:
        raise ValueError(
            "Predictor checkpoint must be a checkpoint folder or 'last', "
            f"got {checkpoint!r}."
        )
    if checkpoint.isdigit():
        checkpoint = checkpoint.zfill(6)
    run_dirs = stage1_run_dirs(outputs_root, model_name, "Predictor")
    for run_dir in run_dirs:
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoint.lower() == "last":
            candidates = sorted(
                (child for child in checkpoints_dir.glob("*") if child.name.isdigit() and child.is_dir()),
                key=lambda child: int(child.name),
                reverse=True,
            )
        else:
            candidates = [checkpoints_dir / checkpoint]
        for candidate in candidates:
            path = candidate / "pretrained_model"
            if all((path / name).is_file() for name in ("config.json", "model.safetensors")):
                return path, candidate.name
    searched = ", ".join(str(run_dir / "checkpoints") for run_dir in run_dirs)
    raise FileNotFoundError(
        f"No complete skill predictor checkpoint {checkpoint!r} found in: {searched}"
    )


def validate_predictor_code_space(source_info: dict, predictor_info: dict, run_dir: Path) -> str:
    """Accept different PT/FT run tags only with proof of a shared labeling FSQ.

    Match the NewTask FT contract: fsq_source.json identifies the actual FSQ
    run and checkpoint, whereas a dataset-derived code-space tag may differ.
    Return the dataset's own identity after validating the Predictor's FSQ.
    """
    source_id = str(source_info.get("skill_code_space_id", run_dir.name) or run_dir.name).strip()
    fsq_path = str(predictor_info.get("fsq_path", "") or "").strip()
    predictor_id = str(predictor_info.get("skill_code_space_id", "") or "").strip()
    if not predictor_id and fsq_path:
        predictor_id = Path(fsq_path).parent.name
    if predictor_id == source_id:
        return source_id

    def fsq_identity(folder: Path) -> tuple[str, str] | None:
        path = folder / "fsq_source.json"
        if not exists_here(path, "file"):
            return None
        data = json.loads(path.read_text())
        model = str(data.get("source_fsq_run_name", "") or "").strip()
        step = str(data.get("source_fsq_checkpoint", "") or "").strip()
        return (model, step) if model and step else None

    source_identity = fsq_identity(run_dir)
    if source_identity and fsq_path:
        recorded = Path(fsq_path).expanduser().parent
        candidates = [recorded]
        if len(recorded.parts) >= 2:
            # Checkpoints from another server keep a stable <dataset>/<run> tail.
            candidates.append(run_dir.parent.parent / recorded.parts[-2] / recorded.parts[-1])
        if any(fsq_identity(candidate) == source_identity for candidate in candidates):
            return source_id
    raise ValueError(
        "Predictor and dataset use different FSQ code spaces, or shared FSQ provenance "
        f"could not be verified: dataset={source_id!r}, predictor={predictor_id!r}. "
        "Different PT/FT run tags require matching fsq_source.json records."
    )


def build_settings(
    cfg: dict,
    dataset: str | None = None,
) -> dict:
    root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = root / str(get_value(cfg, "dataset_root", "libero_dataset"))
    outputs_root = root / str(get_value(cfg, "outputs_root", "outputs"))
    # Fixed per-stage subdirs (match train_skills layout).
    dp_outputs_root = outputs_root / "DP"
    fsq_outputs_root = outputs_root / "FSQ"
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))

    source_dataset = dataset or str(get_value(cfg, "source_dataset", env="SOURCE_DATA"))
    skillvla_data_mode = str(get_value(cfg, "skillvla_data_mode", "pt")).strip().lower()
    if skillvla_data_mode not in {"pt", "ft", "ft_own", "ft_epi"}:
        raise ValueError(
            "skillvla_data_mode must be pt|ft|ft_own|ft_epi, "
            f"got {skillvla_data_mode!r}."
        )
    # ── FSQ reference (declared like dp_policy_name: folder name + checkpoint) ──
    # The FSQ artifact is the source of truth for the DP/skillset taxonomy it
    # learned. Do not duplicate those fields in train_skillVLA_config.yaml.
    fsq_run_name = str(get_value(cfg, "fsq_run_name"))
    fsq_checkpoint = str(get_value(cfg, "fsq_checkpoint", "1000"))
    fsq_model_dir = fsq_outputs_root / fsq_run_name
    fsq_meta_path = fsq_model_dir / "fsq_meta.json"
    if not fsq_meta_path.is_file():
        raise FileNotFoundError(
            f"FSQ metadata not found: {fsq_meta_path}. "
            "Select a current FSQ output folder containing fsq_meta.json."
        )
    fsq_meta = json.loads(fsq_meta_path.read_text())

    required_meta = ("dp_run_name", "dp_checkpoint", "skillset_mode")
    missing_meta = [key for key in required_meta if fsq_meta.get(key) in (None, "")]
    if missing_meta:
        raise ValueError(
            f"FSQ metadata is missing required fields {missing_meta}: {fsq_meta_path}"
        )
    fsq_skillset_manifest, fsq_skillset_manifest_path = _load_fsq_skillset_manifest(
        fsq_meta,
        dataset_root=dataset_root,
        fsq_meta_path=fsq_meta_path,
    )
    action_contract = fsq_skillset_manifest.get("action") or {}
    detector_contract = fsq_skillset_manifest.get("detector") or {}
    required_action = ("mode", "gripper_mode", "gripper_indices")
    missing_action = [key for key in required_action if key not in action_contract]
    required_detector = ("nms_dist", "min_skill_len")
    missing_detector = [key for key in required_detector if key not in detector_contract]
    if missing_action or missing_detector:
        raise ValueError(
            "FSQ training skillset manifest is missing segmentation fields: "
            f"action={missing_action}, detector={missing_detector}: "
            f"{fsq_skillset_manifest_path}"
        )

    dp_policy_name = str(fsq_meta["dp_run_name"])
    dp_checkpoint = str(fsq_meta["dp_checkpoint"])
    if dp_checkpoint.isdigit():
        dp_checkpoint = dp_checkpoint.zfill(6)
    dp_policy_path = (
        dp_outputs_root / dp_policy_name / "checkpoints" / dp_checkpoint / "pretrained_model"
    )

    fsq_boundary_threshold_mode = str(
        fsq_meta.get("skillset_boundary_threshold_mode", "episode_mean")
    ).strip().lower()
    if fsq_boundary_threshold_mode not in {"episode_mean", "global_mean"}:
        raise ValueError(
            "fsq_meta skillset_boundary_threshold_mode must be episode_mean|global_mean, "
            f"got {fsq_boundary_threshold_mode!r}: {fsq_meta_path}"
        )
    # FT episode-mean ablation: preserve the FSQ/DP/code-space contract while
    # selecting each target episode's own mean SBD score as its threshold.
    # All other data modes continue to inherit the FSQ training threshold mode.
    boundary_threshold_mode = (
        "episode_mean"
        if skillvla_data_mode == "ft_epi"
        else fsq_boundary_threshold_mode
    )
    boundary_threshold_scale = float(
        fsq_meta.get("skillset_boundary_threshold_scale", 1.0)
    )
    if boundary_threshold_scale <= 0.0:
        raise ValueError(
            "fsq_meta skillset_boundary_threshold_scale must be positive, "
            f"got {boundary_threshold_scale}: {fsq_meta_path}"
        )
    threshold_percent_tag = _scale_percent_tag(boundary_threshold_scale)

    # Reuse the exact action convention and probe settings that produced the FSQ
    # training skillset; the conversion YAML should not duplicate this contract.
    probe_contract = fsq_skillset_manifest.get("probe") or {}
    detector_contract = fsq_skillset_manifest.get("detector") or {}
    if str(fsq_meta["skillset_mode"]).lower() == "gaussian_x0hat":
        required_probe = ("generation", "gaussian_sampling", "gaussian_include_mean", "gaussian_seed",
                          "denoise_steps", "denoise_output")
        required_score = ("score_metric", "gmm_covariance", "gmm_n_init", "gmm_max_iter",
                          "min_effective_samples", "gmm_weighted", "score_power", "terminal_mask_frames")
        missing = [f"probe.{k}" for k in required_probe if k not in probe_contract]
        missing += [f"detector.{k}" for k in required_score if k not in detector_contract]
        if missing:
            raise ValueError(f"FSQ Gaussian segmentation provenance is incomplete: {missing}: {fsq_skillset_manifest_path}")
    artifact_cfg = dict(cfg)
    artifact_cfg.update(
        {
            "skillset_mode": fsq_meta["skillset_mode"],
            "skillset_action_mode": action_contract["mode"],
            "skillset_relative_exclude_joints": action_contract.get(
                "relative_exclude_joints", ["gripper"]
            ),
            "skillset_gripper_mode": action_contract["gripper_mode"],
            "skillset_gripper_indices": action_contract["gripper_indices"],
            "skillset_gripper_values": action_contract.get("gripper_values", [-1.0, 1.0]),
            "skillset_gripper_threshold": action_contract.get("gripper_threshold", 0.0),
            "skillset_probe_count": probe_contract.get("count", 24),
            "skillset_probe_alpha": probe_contract.get("alpha", 0.1),
            "skillset_pca_variance": probe_contract.get("pca_variance", 0.95),
            "skillset_pca_stride": probe_contract.get("pca_stride", 3),
            "skillset_probe_generation": probe_contract.get(
                "generation", "pca_offset"
            ),
            "skillset_gaussian_sampling": probe_contract.get(
                "gaussian_sampling", "iid"
            ),
            "skillset_gaussian_include_mean": probe_contract.get(
                "gaussian_include_mean", True
            ),
            "skillset_gaussian_seed": probe_contract.get("gaussian_seed", 42),
            "skillset_denoise_steps": probe_contract.get("denoise_steps", 1),
            "skillset_denoise_output": probe_contract.get(
                "denoise_output", "prev_sample"
            ),
            "skillset_score_metric": detector_contract.get(
                "score_metric", "legacy_cosine"
            ),
            "skillset_gmm_covariance": detector_contract.get(
                "gmm_covariance", "full"
            ),
            "skillset_gmm_n_init": detector_contract.get("gmm_n_init", 10),
            "skillset_gmm_max_iter": detector_contract.get("gmm_max_iter", 300),
            "skillset_min_effective_samples": detector_contract.get(
                "min_effective_samples", 0.0
            ),
            "skillset_gmm_weighted": detector_contract.get("gmm_weighted", True),
            "skillset_score_power": detector_contract.get("score_power", 1.0),
            "skillset_terminal_mask_frames": detector_contract.get(
                "terminal_mask_frames", 0
            ),
        }
    )
    probe_settings = skillset_probe_settings(artifact_cfg)
    fsq_skillset_mode = str(fsq_meta["skillset_mode"]).strip().lower()
    if fsq_skillset_mode == "spherical":
        probe_type, pca_scale_mode, probe_exclude_indices = "spherical_xyz", "none", ""
    elif fsq_skillset_mode == "full":
        probe_type, pca_scale_mode, probe_exclude_indices = "pca_action", "none", ""
    elif fsq_skillset_mode == "without_gripper":
        probe_type, pca_scale_mode = "pca_action", "none"
        probe_exclude_indices = probe_settings["skillset_gripper_indices"]
    elif fsq_skillset_mode in {"std", "gaussian_x0hat"}:
        probe_type, pca_scale_mode, probe_exclude_indices = "pca_action", "std", ""
    else:
        raise ValueError(
            "fsq_meta skillset_mode must be "
            "spherical|full|without_gripper|std|gaussian_x0hat, "
            f"got {fsq_skillset_mode!r}: {fsq_meta_path}"
        )
    probe_settings.update(
        skillset_mode=fsq_skillset_mode,
        skillset_probe_type=probe_type,
        skillset_pca_scale_mode=pca_scale_mode,
        skillset_probe_exclude_indices=probe_exclude_indices,
        skillset_probe_suffix=f"_{fsq_skillset_mode}",
    )
    skillset_min_skills = int(detector_contract.get("min_skills", 1))
    if skillset_min_skills < 1:
        raise ValueError(
            f"fsq_meta skillset_min_skills must be >= 1, got {skillset_min_skills}."
        )
    skillset_min_skill_len = int(detector_contract["min_skill_len"])
    if skillset_min_skill_len < 1:
        raise ValueError(
            f"skillset_min_skill_len must be >= 1, got {skillset_min_skill_len}."
        )

    # New runs store model identity in metadata because their folder name is
    # exactly the user-owned fsq_exp and deliberately encodes no parameters.
    # Keep folder parsing only as a read-only fallback for historical runs.
    meta_levels = fsq_meta.get("fsq_levels")
    if meta_levels is not None:
        if isinstance(meta_levels, str):
            fsq_levels = [
                int(value)
                for value in meta_levels.replace(",", " ").split()
            ]
        else:
            fsq_levels = [int(value) for value in meta_levels]
        if not fsq_levels or any(level < 2 for level in fsq_levels):
            raise ValueError(
                f"Invalid fsq_levels in FSQ metadata: {meta_levels!r}"
            )
        fsq_exp = str(fsq_meta.get("fsq_exp") or fsq_run_name)
    else:
        lv_match = re.search(r"fsq(\d+)", fsq_run_name)
        if not lv_match:
            raise ValueError(
                "Historical FSQ metadata has no fsq_levels and its folder name "
                f"has no 'fsq<levels>' fallback tag: {fsq_run_name}"
            )
        fsq_levels = [int(digit) for digit in lv_match.group(1)]
        fsq_exp = fsq_run_name[lv_match.end() :].strip("_")
    fsq_digits = "".join(str(level) for level in fsq_levels)
    fsq_exp_suffix = f"_{fsq_exp}" if fsq_exp else ""
    fsq_semantic_suffix = _fsq_semantic_suffix(fsq_meta)

    jitter_distribution = str(
        get_value(cfg, "transition_jitter_distribution", "half_normal")
    ).strip().lower().replace("-", "_").replace(" ", "_")
    if jitter_distribution not in {"half_normal", "uniform"}:
        raise ValueError(
            "transition_jitter_distribution must be half_normal|uniform, "
            f"got {jitter_distribution!r}."
        )
    jitter_max_percent = float(
        get_value(cfg, "transition_jitter_max_percent", 0.0)
    )
    if not math.isfinite(jitter_max_percent) or not 0.0 <= jitter_max_percent <= 100.0:
        raise ValueError(
            "transition_jitter_max_percent must be finite and in [0, 100], "
            f"got {jitter_max_percent}."
        )
    jitter_precompute_gripper_validity = as_bool(
        get_value(cfg, "transition_jitter_precompute_gripper_validity", False)
    )
    # The latent file does not exist yet when submit_build_all.sh resolves this
    # config. Size the ISS against the declared FSQ maximum length, so every
    # later per-skill percentage up to max_percent is guaranteed to fit.
    configured_max_length = int(get_value(cfg, "max_length", 200))
    if configured_max_length <= 0:
        raise ValueError("max_length must be positive.")
    legacy_directional = {
        name: int(get_value(cfg, f"transition_jitter_{name}_pmax", -1))
        for name in ("early_start", "late_start", "early_end", "late_end")
    }
    has_legacy_directional = any(value >= 0 for value in legacy_directional.values())
    if has_legacy_directional and jitter_max_percent > 0.0:
        raise ValueError(
            "Use transition_jitter_max_percent or legacy directional pmax fields, not both."
        )
    if has_legacy_directional:
        fallback = int(get_value(cfg, "pmax", 10))
        legacy_directional = {
            name: fallback if value < 0 else value
            for name, value in legacy_directional.items()
        }
    skill_pmax = (
        max(1, int(math.floor(configured_max_length * jitter_max_percent / 100.0 + 0.5)))
        if jitter_max_percent > 0.0
        else (
            max(legacy_directional.values())
            if has_legacy_directional
            else int(get_value(cfg, "transition_jitter_pmax", get_value(cfg, "pmax", 0)))
        )
    )
    if skill_pmax < 0:
        raise ValueError("transition_jitter_pmax must be non-negative.")
    skillset_min_skills_suffix = (
        "" if skillset_min_skills == 1 else f"_ms{skillset_min_skills}"
    )
    # Detailed segmentation, threshold, snap, and jitter settings are kept in
    # metadata/config rather than the readable SkillVLA dataset folder name.
    # Keep only a non-default minimum-skill constraint visible because it changes
    # the basic segmentation cardinality contract.
    data_identity_suffix = skillset_min_skills_suffix
    output_suffix = _skillvla_output_suffix(cfg)
    proprio_grounding, proprio_grounding_suffix = _proprio_grounding(cfg)
    (
        exclude_rotation_outlier_episodes,
        rotation_outlier_suffix,
        rotation_outlier_threshold,
    ) = _rotation_outlier_exclusion(cfg)

    # ── FSQ (step 4) — model path from the parsed run name + checkpoint ──
    if fsq_checkpoint in ("0", "best"):
        fsq_model_path = fsq_model_dir / "FSQ.pt"
        ckpt_tag = "best"
    else:
        fsq_model_path = fsq_model_dir / f"FSQ_epoch{int(fsq_checkpoint):04d}.pt"
        ckpt_tag = str(fsq_checkpoint)

    # ── output layout ──
    #   {skillvla_root}/{source_dataset}/{run_tag}/   ← final outputs (FSQ.pt, skillvla/)
    #   {skillvla_root}/{source_dataset}/_work/        ← intermediates, keyed by dependency:
    #       seg_{dp}_ck{ckpt}/        (DP-dependent: skillset + skill_tokens; shared across FSQ)
    # fsq_exp remains exported as provenance, but is intentionally omitted from
    # the SkillVLA folder name. Architecture/loss tags above come from stable
    # structured metadata instead of the user-owned experiment label.
    base_run_tag = f"FSQ{fsq_digits}{fsq_semantic_suffix}_{ckpt_tag}"
    run_tag = f"{base_run_tag}_{skillvla_data_mode}"
    pt_run_tag = (
        f"{base_run_tag}_pt{data_identity_suffix}"
        f"{proprio_grounding_suffix}{rotation_outlier_suffix}{output_suffix}"
    )
    # Snap changes the generated latent assignments but is intentionally hidden
    # from the concise folder name. Use skillvla_output_suffix when multiple snap
    # variants of the same FSQ/checkpoint/data mode must coexist.
    fsq_snap = as_bool(get_value(cfg, "fsq_snap_to_supported", False))
    fsq_snap_reference = ""
    if fsq_snap:
        if skillvla_data_mode == "pt":
            # PT vocabulary pruning: the just-encoded raw distribution is the
            # reference, so no user-maintained path is needed.
            fsq_snap_reference = "self"
        else:
            # FT must use the PT vocabulary for this exact FSQ/checkpoint and
            # pruning threshold. Search source-dataset directories so the FT
            # config needs no duplicated PT dataset/path field.
            pt_refs = sorted(skillvla_root.glob(f"*/{pt_run_tag}/skill_latents.npz"))
            if len(pt_refs) != 1:
                found = "\n  ".join(str(p) for p in pt_refs) or "(none)"
                raise ValueError(
                    f"skillvla_data_mode={skillvla_data_mode} requires exactly one completed PT reference "
                    f"at */{pt_run_tag}/skill_latents.npz; found:\n  {found}\n"
                    "Build the matching PT data first (same FSQ checkpoint and "
                    "fsq_snap_min_code_freq)."
                )
            fsq_snap_reference = str(pt_refs[0])
    # Append only the remaining concise identity and the optional user label.
    run_tag += (
        f"{data_identity_suffix}{proprio_grounding_suffix}"
        f"{rotation_outlier_suffix}{output_suffix}"
    )
    source_out_dir = skillvla_root / source_dataset
    run_dir = source_out_dir / run_tag
    work_dir = source_out_dir / "_work"
    focus_uv_settings = _focus_uv_settings(
        cfg,
        root=root,
        skillvla_root=skillvla_root,
        source_dataset=source_dataset,
        run_dir=run_dir,
    )
    # Keep boundary modes in disjoint work directories. For global_mean, pt and
    # ft_own reduce this source's curves while ft reuses the matching PT value.
    # ft_epi forces episode_mean, which needs neither a reducer nor a
    # cross-dataset threshold reference.
    seg_base = (
        f"seg_{dp_policy_name}_ck{dp_checkpoint}"
        f"{probe_settings['skillset_probe_suffix']}"
        f"{skillset_min_skills_suffix}"
    )
    skillset_global_threshold_source = ""
    own_global_seg_suffix = f"_globalmean_{threshold_percent_tag}"
    if boundary_threshold_mode == "global_mean" and skillvla_data_mode == "ft":
        pt_thresholds = sorted(
            path for path in skillvla_root.glob(
                f"*/_work/{seg_base}{own_global_seg_suffix}/skillset/"
                "global_boundary_threshold.json"
            )
            # ft_own writes the same work-directory name. Only sources with a
            # matching completed PT encoding can supply the PT threshold.
            if (path.parents[3] / pt_run_tag / "skill_latents.npz").is_file()
        )
        if len(pt_thresholds) != 1:
            found = "\n  ".join(str(p) for p in pt_thresholds) or "(none)"
            raise ValueError(
                "skillvla_data_mode=ft requires exactly one completed PT global threshold "
                f"at */_work/{seg_base}{own_global_seg_suffix}/skillset/"
                "global_boundary_threshold.json; "
                f"found:\n  {found}\nBuild the matching PT data first."
            )
        skillset_global_threshold_source = str(pt_thresholds[0])
    if boundary_threshold_mode == "episode_mean":
        seg_suffix = f"_episodemean_{threshold_percent_tag}"
    else:
        if skillset_global_threshold_source:
            seg_suffix = f"_globalref_{threshold_percent_tag}"
        else:
            seg_suffix = own_global_seg_suffix
    seg_dir = work_dir / f"{seg_base}{seg_suffix}"
    skillset_dir = seg_dir / "skillset"

    parallelism = {
        "skillset_tasks_per_job": int(get_value(cfg, "skillset_tasks_per_job", 5)),
        "skillset_array_throttle": int(get_value(cfg, "skillset_array_throttle", 0)),
        "skillset_max_sweeps": int(get_value(cfg, "skillset_max_sweeps", 2)),
        "skillset_single_gpu_workers": int(get_value(cfg, "skillset_single_gpu_workers", 0)),
    }
    for name, value in parallelism.items():
        minimum = 1 if name == "skillset_tasks_per_job" else 0
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value}.")

    def slurm(prefix: str, *, cpus: int, mem: str, time: str) -> dict:
        # partition/qos/nodelist/exclude are canonical (global_config.yaml train_*); output keys
        # keep the per-job prefix so submit scripts read the same $<PREFIX>_* vars.
        part = ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug"
        excl = ",".join(as_list(get_value(cfg, "train_exclude_nodes", [])))
        return {
            f"{prefix}_partition": part,
            f"{prefix}_qos": str(get_value(cfg, "train_qos", "base_qos")),
            f"{prefix}_gres": str(get_value(cfg, f"{prefix}_gres", "gpu:1")),
            f"{prefix}_cpus_per_task": int(get_value(cfg, f"{prefix}_cpus_per_task", cpus)),
            f"{prefix}_mem": str(get_value(cfg, f"{prefix}_mem", mem)),
            f"{prefix}_time": str(get_value(cfg, f"{prefix}_time", time)),
            f"{prefix}_nodelist": str(get_value(cfg, "train_nodelist", "")),
            f"{prefix}_exclude_nodes": excl,
        }

    settings: dict = {
        # roots
        "project_root": root,
        "lerobot_root": root / "lerobot",
        "dataset_root": dataset_root,
        # source dataset
        "source_dataset": source_dataset,
        "skillvla_data_mode": skillvla_data_mode,
        "raw_dataset_dir": dataset_root / source_dataset,
        # (DINO precompute emit 은퇴 — DINO는 어디서도 precompute 안 함. DP=state/raw-frames,
        #  FSQ 학습·terminator=ONLINE. dino_root/required/generate/base 등 소비자 0.)
        # DP (step 3)
        "dp_policy_name": dp_policy_name,
        "dp_checkpoint": dp_checkpoint,
        "dp_policy_path": dp_policy_path,
        "skillset_dir": skillset_dir,
        "fsq_skillset_manifest_path": fsq_skillset_manifest_path,
        "skill_latents_path": run_dir / "skill_latents.npz",
        "skillset_dn_step": int(detector_contract.get("eval_at_step", 7)),
        "skillset_n_gmm": int(detector_contract.get("n_gmm_components", 5)),
        "skillset_smooth_window": int(detector_contract.get("smooth_window", 7)),
        "skillset_savgol_polyorder": int(detector_contract.get("savgol_polyorder", 4)),
        "skillset_replan_interval": int(detector_contract.get("replan_interval", 3)),
        "skillset_nms_dist": int(detector_contract["nms_dist"]),
        "skillset_min_skills": skillset_min_skills,
        "skillset_min_skill_len": skillset_min_skill_len,
        "skillset_seed": int(probe_contract.get("seed", 42)),
        "skillset_noise_scheduler_type": str(detector_contract.get("noise_scheduler_type", "DDIM")),
        "skillset_num_inference_steps": int(detector_contract.get("num_inference_steps", 10)),
        "skillset_peak_nms": as_bool(detector_contract.get("peak_nms", True)),
        **probe_settings,
        "skillset_boundary_threshold_mode": boundary_threshold_mode,
        "skillset_boundary_threshold_scale": boundary_threshold_scale,
        "skillset_global_threshold_source": skillset_global_threshold_source,
        "skillset_global_threshold_path": skillset_dir / "global_boundary_threshold.json",
        "skillset_dino_feature_dir": resolve_path(
            root, get_value(cfg, "skillset_dino_feature_dir", "")
        ),
        # Task chunks/retry policy are shared by GPU arrays and the one-GPU pool.
        **parallelism,
        # FSQ (step 4)
        "fsq_run_name": fsq_run_name,
        "fsq_exp": fsq_exp,
        "fsq_exp_suffix": fsq_exp_suffix,
        "fsq_semantic_suffix": fsq_semantic_suffix,
        "fsq_model_dir": fsq_model_dir,
        "fsq_model_path": fsq_model_path,
        "fsq_meta_path": fsq_meta_path,
        "fsq_checkpoint": fsq_checkpoint,
        # transfer 안전망(B): 인코딩 시 미지원(학습때 안 쓰인) 코드 → 최근접 지원 코드로 snap.
        # (snap=true인데 reference가 없/틀리면 아래에서 이미 제출 전에 raise — 런타임 좀비 체인 방지)
        "fsq_snap_to_supported": fsq_snap,
        "fsq_snap_min_code_freq": int(get_value(cfg, "fsq_snap_min_code_freq", 1)),
        "fsq_snap_reference": fsq_snap_reference,
        "fsq_levels_str": " ".join(str(v) for v in fsq_levels),
        # SkillVLA build (step 5)
        "max_order": int(get_value(cfg, "max_order", 0)),
        "max_length": configured_max_length,
        "skill_pmax": skill_pmax,   # Stage-2 transition randomization 반폭 (ISS window)
        "skill_jitter_max_percent": jitter_max_percent,
        "skill_jitter_precompute_gripper_validity": str(
            jitter_precompute_gripper_validity
        ).lower(),
        # Read-only compatibility exports for historical submit scripts/tests.
        "skill_early_start_pmax": legacy_directional["early_start"] if has_legacy_directional else skill_pmax,
        "skill_late_start_pmax": legacy_directional["late_start"] if has_legacy_directional else skill_pmax,
        "skill_early_end_pmax": legacy_directional["early_end"] if has_legacy_directional else skill_pmax,
        "skill_late_end_pmax": legacy_directional["late_end"] if has_legacy_directional else skill_pmax,
        "skill_jitter_distribution": jitter_distribution,
        "proprio_grounding": proprio_grounding,
        "exclude_rotation_outlier_episodes": str(
            exclude_rotation_outlier_episodes
        ).lower(),
        "rotation_outlier_threshold": rotation_outlier_threshold,
        **focus_uv_settings,
        "skill_decoder_state_indices": str(get_value(cfg, "skill_decoder_state_indices", "[0,1,2,3,4,5,6,7]")),
        "cleanup_intermediate": str(get_value(cfg, "cleanup_intermediate", True)).lower(),
        # output layout
        "run_tag": run_tag,
        "skillvla_output_suffix": output_suffix,
        "skillvla_run_dir": run_dir,
        "skillvla_work_dir": work_dir,
        "skillvla_seg_dir": seg_dir,   # DP-keyed intermediates (skillset + skill_tokens)
        "iss_npz_path": run_dir / "skill_initial_state.npz",   # Stage-2 skill-initial-state (ISS)
        "fsq_copy_path": run_dir / "FSQ.pt",
        "skillvla_dataset_dir": run_dir / "skillvla",
    }
    settings["skillset_contract_json"] = json.dumps(segmentation_contract_from_settings(settings), sort_keys=True)
    settings.update(slurm("skillvla", cpus=8, mem="64G", time="8:00:00"))
    return settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--dataset", default=None, help="Override source_dataset")
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    settings = build_settings(
        load_config(args.config),
        dataset=args.dataset,
    )
    if args.shell:
        print_shell(settings)
    else:
        for k, v in settings.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
