#!/usr/bin/env python3
"""Config helpers for configs/train_skillVLA (SkillVLA data generation).

Resolves paths + run tags for the pipeline that turns trained DP + FSQ models
into SkillVLA training data, and emits them as shell exports (--shell).

Root/yaml helpers are reused from train_skills_config.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# reuse the train_skills yaml-load + shell-emit helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import (  # noqa: E402
    as_bool,
    as_list,
    get_value,
    load_config,
    print_shell,
    resolve_path,
    skillset_probe_settings,
)

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
    """Resolve one auxiliary predictor run without server-specific paths."""
    if not model_name or Path(model_name).name != model_name:
        raise ValueError(
            "skill_relabel.predictor_model must be one skillVLA_terminator "
            f"folder name, got {model_name!r}."
        )
    checkpoint = str(checkpoint or "last").strip()
    if not checkpoint or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", checkpoint) is None:
        raise ValueError(
            "skill_relabel.checkpoint must be a checkpoint folder or 'last', "
            f"got {checkpoint!r}."
        )
    run_dir = outputs_root / "skillVLA_terminator" / model_name
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoint.lower() == "last":
        candidates = (
            sorted(
                (
                    (int(child.name), child.name)
                    for child in checkpoints_dir.iterdir()
                    if child.is_dir()
                    and child.name.isdigit()
                    and (child / "pretrained_model" / "config.json").is_file()
                ),
                key=lambda item: item[0],
            )
            if checkpoints_dir.is_dir()
            else []
        )
        if not candidates:
            raise FileNotFoundError(
                f"No numeric predictor checkpoints found under {checkpoints_dir}."
            )
        checkpoint = candidates[-1][1]
    path = checkpoints_dir / checkpoint / "pretrained_model"
    if not (path / "config.json").is_file() or not (
        path / "model.safetensors"
    ).is_file():
        raise FileNotFoundError(f"Incomplete skill predictor checkpoint: {path}")
    return path, checkpoint


def _compact_checkpoint_tag(checkpoint: str) -> str:
    """Format a resolved checkpoint folder for a concise dataset suffix."""
    if checkpoint.isdigit():
        step = int(checkpoint)
        if step >= 1000:
            return f"{step / 1000:g}".replace(".", "p") + "k"
        return str(step)
    return checkpoint.lower()


def _relabel_settings(
    cfg: dict[str, Any],
    *,
    root: Path,
    outputs_root: Path,
    source_dataset: str,
    run_dir: Path,
    run_tag: str,
) -> dict[str, Any]:
    raw = get_value(cfg, "skill_relabel", {})
    if not isinstance(raw, dict):
        raise ValueError(
            "skill_relabel must be an inline mapping: "
            "{predictor_model: ..., checkpoint: ...}."
        )
    model_name = str(raw.get("predictor_model", "") or "").strip()
    checkpoint_value = str(raw.get("checkpoint", "last") or "last").strip()
    predictor_path, checkpoint = _resolve_predictor_checkpoint(
        outputs_root, model_name, checkpoint_value
    )
    if not (run_dir / "skillvla" / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            "Source SkillVLA dataset must be built before relabeling: "
            f"{run_dir / 'skillvla'}"
        )
    if "_relabeled" in run_tag:
        raise ValueError("A relabeled SkillVLA dataset cannot be relabeled again.")

    source_info = json.loads(
        (run_dir / "skillvla" / "meta" / "info.json").read_text()
    )
    predictor_info = json.loads((predictor_path / "config.json").read_text())
    if not as_bool(predictor_info.get("train_skill_predictor", False)):
        raise ValueError(
            f"Selected checkpoint has no trained skill predictor: {predictor_path}"
        )
    source_levels = [int(value) for value in source_info.get("skill_fsq_levels", [])]
    predictor_levels = [
        int(value) for value in predictor_info.get("skill_fsq_levels", [])
    ]
    if not source_levels or source_levels != predictor_levels:
        raise ValueError(
            "Relabel predictor FSQ geometry does not match the source dataset: "
            f"dataset={source_levels}, predictor={predictor_levels}."
        )
    code_space_id = str(
        source_info.get("skill_code_space_id", run_tag) or run_tag
    ).strip()
    predictor_code_space_id = str(
        predictor_info.get("skill_code_space_id", "") or ""
    ).strip()
    if not predictor_code_space_id:
        fsq_path = str(predictor_info.get("fsq_path", "") or "").strip()
        predictor_code_space_id = Path(fsq_path).parent.name if fsq_path else ""
    if predictor_code_space_id != code_space_id:
        raise ValueError(
            "Relabel predictor and dataset use different FSQ code spaces: "
            f"dataset={code_space_id!r}, predictor={predictor_code_space_id!r}."
        )

    checkpoint_tag = _compact_checkpoint_tag(checkpoint)
    output_run_dir = run_dir.with_name(f"{run_tag}_relabeled_{checkpoint_tag}")
    tokenizer_path = root / "models" / "paligemma-3b-pt-224-tokenizer"
    required_tokenizer_files = ("config.json", "tokenizer.json")
    missing = [
        name
        for name in required_tokenizer_files
        if not (tokenizer_path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Local predictor tokenizer is incomplete at {tokenizer_path}: missing {missing}."
        )
    return {
        "relabel_source_dataset": source_dataset,
        "relabel_source_run_dir": run_dir,
        "relabel_output_run_dir": output_run_dir,
        "relabel_predictor_model": model_name,
        "relabel_predictor_checkpoint": checkpoint,
        "relabel_predictor_path": predictor_path,
        "relabel_tokenizer_path": tokenizer_path,
        "relabel_code_space_id": code_space_id,
        # Four is conservative on 24-GiB GPUs because the predictor retains all
        # PaliGemma layer states for its all-layer reader.
        "relabel_batch_size": 4,
    }


def _standalone_relabel_settings(
    cfg: dict[str, Any],
    *,
    root: Path,
    dataset_root: Path,
    outputs_root: Path,
    source_dataset: str,
    skillvla_root: Path,
    source_run: str,
) -> dict[str, Any]:
    """Resolve relabeling solely from an already-built SkillVLA run.

    Relabeling neither segments data nor runs FSQ, so requiring the current
    build YAML's fsq_run_name/fsq_meta.json makes an otherwise valid completed
    dataset depend on stale build settings.
    """
    if Path(source_run).name != source_run or "_relabeled_" in source_run:
        raise ValueError(
            "skill_relabel.source_run must be one original SkillVLA run folder "
            f"name, got {source_run!r}."
        )
    run_dir = skillvla_root / source_dataset / source_run
    settings: dict[str, Any] = {
        "project_root": root,
        "lerobot_root": root / "lerobot",
        "dataset_root": dataset_root,
        "source_dataset": source_dataset,
        "run_tag": source_run,
        "skillvla_run_dir": run_dir,
        "skillvla_dataset_dir": run_dir / "skillvla",
    }
    settings.update(
        _relabel_settings(
            cfg,
            root=root,
            outputs_root=outputs_root,
            source_dataset=source_dataset,
            run_dir=run_dir,
            run_tag=source_run,
        )
    )
    part = ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug"
    settings.update(
        {
            "skillvla_partition": part,
            "skillvla_qos": str(get_value(cfg, "train_qos", "base_qos")),
            "skillvla_gres": str(get_value(cfg, "skillvla_gres", "gpu:1")),
            "skillvla_cpus_per_task": int(
                get_value(cfg, "skillvla_cpus_per_task", 8)
            ),
            "skillvla_mem": str(get_value(cfg, "skillvla_mem", "64G")),
            "skillvla_time": str(get_value(cfg, "skillvla_time", "8:00:00")),
            "skillvla_nodelist": str(get_value(cfg, "train_nodelist", "")),
            "skillvla_exclude_nodes": ",".join(
                as_list(get_value(cfg, "train_exclude_nodes", []))
            ),
        }
    )
    return settings


def build_settings(
    cfg: dict,
    dataset: str | None = None,
    *,
    require_relabel: bool = False,
) -> dict:
    root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = root / str(get_value(cfg, "dataset_root", "libero_dataset"))
    outputs_root = root / str(get_value(cfg, "outputs_root", "outputs"))
    # Fixed per-stage subdirs (match train_skills layout).
    dp_outputs_root = outputs_root / "DP"
    fsq_outputs_root = outputs_root / "FSQ"
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))

    source_dataset = dataset or str(get_value(cfg, "source_dataset", env="SOURCE_DATA"))
    if require_relabel:
        raw_relabel = get_value(cfg, "skill_relabel", {})
        if not isinstance(raw_relabel, dict):
            raise ValueError(
                "skill_relabel must be an inline mapping containing source_run, "
                "predictor_model, and checkpoint."
            )
        source_run = str(raw_relabel.get("source_run", "") or "").strip()
        if source_run:
            return _standalone_relabel_settings(
                cfg,
                root=root,
                dataset_root=dataset_root,
                outputs_root=outputs_root,
                source_dataset=source_dataset,
                skillvla_root=skillvla_root,
                source_run=source_run,
            )
    skillvla_data_mode = str(get_value(cfg, "skillvla_data_mode", "pt")).strip().lower()
    if skillvla_data_mode not in {"pt", "ft", "ft_own"}:
        raise ValueError(
            "skillvla_data_mode must be pt|ft|ft_own, "
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

    boundary_threshold_mode = str(
        fsq_meta.get("skillset_boundary_threshold_mode", "episode_mean")
    ).strip().lower()
    if boundary_threshold_mode not in {"episode_mean", "global_mean"}:
        raise ValueError(
            "fsq_meta skillset_boundary_threshold_mode must be episode_mean|global_mean, "
            f"got {boundary_threshold_mode!r}: {fsq_meta_path}"
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
    elif fsq_skillset_mode == "std":
        probe_type, pca_scale_mode, probe_exclude_indices = "pca_action", "std", ""
    else:
        raise ValueError(
            "fsq_meta skillset_mode must be spherical|full|without_gripper|std, "
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
    # Keep the old scalar pmax as a read-only fallback for historical configs,
    # while new builds carry the same four directional windows as FSQ boundary
    # augmentation.  The ISS remains one symmetric storage window sized by the
    # largest direction; sampling uses the four values independently.
    legacy_pmax = int(get_value(cfg, "pmax", 10))
    directional_pmaxes = {
        name: int(get_value(cfg, f"transition_jitter_{name}_pmax", legacy_pmax))
        for name in ("early_start", "late_start", "early_end", "late_end")
    }
    invalid_pmaxes = {
        name: value for name, value in directional_pmaxes.items() if value < 0
    }
    if invalid_pmaxes:
        raise ValueError(
            "transition jitter directional pmax values must be >= 0, got "
            f"{invalid_pmaxes}."
        )
    skill_pmax = max(directional_pmaxes.values())
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
            pt_run_tag = (
                f"{base_run_tag}_pt{data_identity_suffix}"
                f"{proprio_grounding_suffix}{rotation_outlier_suffix}{output_suffix}"
            )
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
    # Keep boundary modes in disjoint work directories. For global_mean, pt and
    # ft_own reduce this source's curves while ft reuses the matching PT value.
    # episode_mean needs neither a reducer nor a cross-dataset reference.
    seg_base = (
        f"seg_{dp_policy_name}_ck{dp_checkpoint}"
        f"{probe_settings['skillset_probe_suffix']}"
        f"{skillset_min_skills_suffix}"
    )
    skillset_global_threshold_source = ""
    own_global_seg_suffix = f"_globalmean_{threshold_percent_tag}"
    if boundary_threshold_mode == "global_mean" and skillvla_data_mode == "ft":
        pt_thresholds = sorted(
            skillvla_root.glob(
                f"*/_work/{seg_base}{own_global_seg_suffix}/skillset/"
                "global_boundary_threshold.json"
            )
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
        **probe_settings,
        "skillset_boundary_threshold_mode": boundary_threshold_mode,
        "skillset_boundary_threshold_scale": boundary_threshold_scale,
        "skillset_global_threshold_source": skillset_global_threshold_source,
        "skillset_global_threshold_path": skillset_dir / "global_boundary_threshold.json",
        "skillset_dino_feature_dir": resolve_path(
            root, get_value(cfg, "skillset_dino_feature_dir", "")
        ),
        # parallelism: split tasks into shards of this size, one shard per Slurm array job (1 GPU each)
        "skillset_tasks_per_job": int(get_value(cfg, "skillset_tasks_per_job", 5)),
        "skillset_array_throttle": int(get_value(cfg, "skillset_array_throttle", 0)),
        # post-array verify: re-run tasks with missing episodes up to this many times
        "skillset_max_sweeps": int(get_value(cfg, "skillset_max_sweeps", 2)),
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
        "max_length": int(get_value(cfg, "max_length", 200)),
        "skill_pmax": skill_pmax,   # Stage-2 transition randomization 반폭 (ISS window)
        "skill_early_start_pmax": directional_pmaxes["early_start"],
        "skill_late_start_pmax": directional_pmaxes["late_start"],
        "skill_early_end_pmax": directional_pmaxes["early_end"],
        "skill_late_end_pmax": directional_pmaxes["late_end"],
        "skill_jitter_distribution": jitter_distribution,
        "proprio_grounding": proprio_grounding,
        "exclude_rotation_outlier_episodes": str(
            exclude_rotation_outlier_episodes
        ).lower(),
        "rotation_outlier_threshold": rotation_outlier_threshold,
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
        # eval outputs (build_data_eval runs off: raw video + skillvla/ + dino.npz + FSQ.pt)
        "eval_dir": run_dir / "eval",
        "eval_dino_dir": run_dir / "eval" / "dino",
        "eval_skillset_dir": run_dir / "eval" / "skillset",
        "eval_fsq_patch_dir": run_dir / "eval" / "fsq_patch",
        "eval_fsq_recon_dir": run_dir / "eval" / "fsq_recon",
    }
    settings.update(slurm("skillvla", cpus=8, mem="64G", time="8:00:00"))
    if require_relabel:
        settings.update(
            _relabel_settings(
                cfg,
                root=root,
                outputs_root=outputs_root,
                source_dataset=source_dataset,
                run_dir=run_dir,
                run_tag=run_tag,
            )
        )
    return settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--dataset", default=None, help="Override source_dataset")
    ap.add_argument(
        "--relabel",
        action="store_true",
        help="Resolve and validate the predictor-relabeled dataset stage.",
    )
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    settings = build_settings(
        load_config(args.config),
        dataset=args.dataset,
        require_relabel=args.relabel,
    )
    if args.shell:
        print_shell(settings)
    else:
        for k, v in settings.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
