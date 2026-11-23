"""Expand the small user-facing SBD config into the evaluator's full recipe."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path


DEFAULTS = {
    "standalone": True, "target_dataset": "abc_sim_all", "source_tasks": ["grab_clutter"],
    "task_id_space": "dataset", "resume": True, "seed": 42,
    "threshold_calibration": {"source": "selected"},
    "report_image_key": "observation.images.top", "split_report_by_task": True,
    "split_report_by_group": True, "overlay_experiment_curves": True,
    "report_compare_limit": 1, "report_compare_columns": 1,
    "report_consistency_metrics": True, "report_consistency_tolerance_frames": 8,
    "pca_scope": "source_tasks", "pca_variance": 0.95, "pca_stride": 3,
    "rollout": {"enabled": False}, "prediction_loss": {"enabled": False},
    "eval_device": "cuda", "eval_workers": 2, "eval_gres": "gpu:1",
    "eval_cpus_per_task": 8, "eval_mem": "32G", "eval_time": "03:00:00",
}
CONTROLS = {
    "sbd_prediction_horizon": 1, "gmm_components": 5, "replan_interval": 8,
    "nms_frames": 45, "smooth_window": 7, "polyorder": 4,
    "score_power": 1, "peak_curvature_scale": 0, "initial_mask": True,
}
RECIPE_KEYS = {"probe_variants", "cluster_variants", "boundary_variants", "experiments"}


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return number


def resolve_checkpoint(cfg):
    """Resolve a DP run/step using the inherited global outputs root."""
    root = Path(cfg["project_root"]).expanduser()
    legacy = str(cfg.get("policy_checkpoint", cfg.get("policy_checkpoint_override", "")) or "").strip()
    run = str(cfg.get("dp_run_name", "") or "").strip()
    if run:
        if Path(run).name != run or run in {".", ".."}:
            raise ValueError("dp_run_name must be the model folder name under outputs_root/DP")
        outputs = str(cfg.get("outputs_root", "") or "").strip()
        if not outputs:
            raise ValueError("outputs_root must be defined in global_config.yaml")
        outputs = Path(outputs).expanduser()
        if not outputs.is_absolute():
            outputs = root / outputs
        step = cfg.get("dp_checkpoint", "100000")
        step = "last" if step == "last" else str(_integer(step, "dp_checkpoint")).zfill(6)
        checkpoint = outputs / "DP" / run / "checkpoints" / step / "pretrained_model"
        if legacy:
            old_path = Path(legacy).expanduser()
            if not old_path.is_absolute():
                old_path = root / old_path
            if old_path.resolve() != checkpoint.resolve():
                raise ValueError("Use dp_run_name/dp_checkpoint or policy_checkpoint; the two paths disagree")
    elif legacy:
        # Keep older frozen configs and explicit imported checkpoint paths readable.
        checkpoint = Path(legacy).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
    else:
        raise ValueError("DP evaluation requires dp_run_name and dp_checkpoint")
    return checkpoint.resolve()


def checkpoint_model_prefix(policy):
    """Name the actual checkpoint architecture, independently of its folder label."""
    encoder = str(policy.get("history_encoder", "flat")).lower()
    if encoder == "gru":
        mode = policy.get("history_gru_training", "iid")
        if mode not in {"iid", "epi"}:
            raise ValueError(f"Unsupported checkpoint GRU training mode: {mode}")
        encoder_tag = "GRUepi" if mode == "epi" else "GRUiid"
    else:
        tags = {"transformer": "trans", "transformer_cls": "transCLS", "flat": "flat"}
        if encoder not in tags:
            raise ValueError(f"Unsupported checkpoint history encoder: {encoder}")
        encoder_tag = tags[encoder]
    dims = tuple(int(value) for value in policy.get("down_dims", [512, 1024, 2048]))
    size = {(128, 256, 512): "small", (512, 1024, 2048): "base"}.get(dims)
    if size is None:
        size = "unet" + "x".join(map(str, dims))
    return f"{encoder_tag}_{size}"


def resolve_config(raw, *, domain="abc"):
    """Keep old frozen variant lists intact; expand compact settings otherwise.

    Optional ``advanced`` entries override hidden defaults. Public controls
    stay at the top level so there is only one place to edit them.
    """
    advanced = raw.get("advanced", {})
    if not isinstance(advanced, dict):
        raise ValueError("advanced must be a mapping")
    if set(advanced) & (set(CONTROLS) | RECIPE_KEYS):
        raise ValueError("Keep SBD controls at the top level; do not put variant lists in advanced")
    if domain not in {"abc", "libero"}:
        raise ValueError("SBD domain must be abc|libero")
    domain_defaults = {} if domain == "abc" else {
        "target_dataset": "libero_90_full_full", "source_tasks": [],
        "target_task": "libero_90", "task_id_space": "suite",
        "report_image_key": "observation.images.image", "pca_scope": "selected",
        "report_consistency_tolerance_frames": 6,
    }
    cfg = {**deepcopy(DEFAULTS), **domain_defaults, **deepcopy(advanced), **deepcopy(raw)}
    cfg.pop("advanced", None)
    if not cfg.get("target_task"):
        cfg["target_task"] = str(cfg["source_tasks"][0])
    checkpoint = resolve_checkpoint(cfg)
    cfg["policy_checkpoint"] = str(checkpoint)
    policy = json.loads((checkpoint / "config.json").read_text())
    prefix = checkpoint_model_prefix(policy)
    base_name = str(cfg.get("output_name", "boundary_ablation"))
    # Resolving an already expanded/frozen recipe must not add the prefix twice.
    if base_name.startswith(prefix + "_"):
        base_name = base_name[len(prefix) + 1:]
    if domain == "libero" and cfg.get("pca_scope", "selected") == "dataset" and not base_name.endswith("_pca_dataset"):
        base_name += "_pca_dataset"
    scope_tag = None
    if domain == "abc":
        scope = str(cfg["pca_scope"])
        if scope not in {"selected", "source_tasks", "dataset"}:
            raise ValueError("ABC pca_scope must be selected|source_tasks|dataset")
        scope_tag = f"pca_{scope}"
        if "n_tasks" in cfg:
            base_name = base_name.removesuffix("_" + scope_tag)
            selection = cfg["n_tasks"]
            tag = "all" if isinstance(selection, str) and selection.strip().lower() == "all" else str(_integer(selection, "n_tasks"))
            task_tag = f"ntasks{tag}"
            if not base_name.endswith("_" + task_tag):
                base_name += "_" + task_tag
        if not base_name.endswith("_" + scope_tag):
            base_name += "_" + scope_tag
    cfg["output_name"] = f"{prefix}_{base_name}"
    history = _integer(policy["n_obs_steps"], "checkpoint n_obs_steps")
    maximum = _integer(
        policy["future_action_horizon"] if policy.get("action_sequence_mode") == "history_reconstruction"
        else policy["horizon"], "checkpoint prediction horizon",
    )
    model = {"n_obs_steps": history, "prediction_horizon": maximum,
             "model_prefix": prefix, "base_output_name": base_name}
    if scope_tag:
        model["pca_scope_tag"] = scope_tag
    center = str(cfg.get("gaussian_center", "previous_action")).strip().lower()
    allowed_centers = {"previous_action", "current_gt", "pure_noise"}
    if center not in allowed_centers:
        raise ValueError(f"{domain.upper()} gaussian_center must be current_gt|previous_action|pure_noise")
    initial_mask = cfg.get("initial_mask", True)
    if not isinstance(initial_mask, bool):
        raise ValueError("initial_mask must be true or false")
    initial_mask_frames = history if initial_mask else 0
    if RECIPE_KEYS & raw.keys():
        if not RECIPE_KEYS.issubset(raw) or set(CONTROLS) & raw.keys():
            raise ValueError("Use compact SBD controls or a complete legacy recipe, not both")
        for probe in cfg["probe_variants"]:
            probe_center = probe.get("gaussian_center", "current_gt" if domain == "libero" else None)
            if (probe.get("probe_generation") != "scheduler_gaussian"
                    or probe_center not in allowed_centers):
                raise ValueError(f"{domain.upper()} eval requires scheduler_gaussian probes with gaussian_center in {sorted(allowed_centers)}")
            if probe_center == "previous_action" and maximum > 1 and not probe["name"].endswith("_hold_future"):
                raise ValueError("Long previous_action recipes require a new _hold_future probe name; use compact controls")
        model["sbd_prediction_horizon"] = int(cfg["probe_variants"][0].get("descriptor_temporal_length", maximum))
        for boundary in cfg["boundary_variants"]:
            boundary["initial_mask_frames"] = initial_mask_frames
        return cfg, model

    domain_controls = {} if domain == "abc" else {"replan_interval": 5, "nms_frames": 15, "score_power": 2}
    values = {**CONTROLS, **domain_controls, **{key: cfg[key] for key in CONTROLS if key in cfg}}
    requested = values["sbd_prediction_horizon"]
    length = maximum if requested == "all" else _integer(requested, "sbd_prediction_horizon")
    if length > maximum:
        raise ValueError(f"sbd_prediction_horizon={length} exceeds checkpoint prediction horizon {maximum}")
    model["sbd_prediction_horizon"] = length
    k = _integer(values["gmm_components"], "gmm_components")
    replan = _integer(values["replan_interval"], "replan_interval")
    nms = _integer(values["nms_frames"], "nms_frames", 0)
    window = _integer(values["smooth_window"], "smooth_window")
    polyorder = _integer(values["polyorder"], "polyorder", 0)
    from lerobot.utils.sbd_smoothing import validate_savgol
    validate_savgol(window, polyorder)
    try:
        power = float(values["score_power"])
    except (TypeError, ValueError, OverflowError):
        raise ValueError("score_power must be finite and positive") from None
    if isinstance(values["score_power"], bool) or not math.isfinite(power) or power <= 0:
        raise ValueError("score_power must be finite and positive")
    # Preserve numeric precision while spelling integral powers as 1 / 2.
    power_tag = str(power).removesuffix(".0")
    try:
        curvature = float(values["peak_curvature_scale"])
    except (TypeError, ValueError, OverflowError):
        raise ValueError("peak_curvature_scale must be finite and non-negative") from None
    if isinstance(values["peak_curvature_scale"], bool) or not math.isfinite(curvature) or curvature < 0:
        raise ValueError("peak_curvature_scale must be finite and non-negative")
    curvature_tag = str(curvature).removesuffix(".0")
    # Public YAML order; curvature is the final CPU-only control.
    model["output_suffix"] = "_".join(map(str, (length, replan, k, nms, window, polyorder, power_tag, curvature_tag)))
    # The PCA basis can be shared, but reports from different probe
    # centers must never overwrite each other.
    model["output_suffix"] += f"_{center}"
    model["output_suffix"] += f"_imask{initial_mask_frames}"
    horizon_tag = f"true_h{length}" if length == maximum else f"future{maximum}_h{length}_descriptor"
    probe_name = f"iid96_r{replan}_{horizon_tag}_{center}"
    if maximum > 1 and center == "previous_action":
        # Earlier long-DP/H=1 probes changed only the first slot and left
        # later slots GT-centered. Never reuse those caches for held futures.
        probe_name += "_hold_future"
    if center == "pure_noise":
        probe_name += "_reverse10_x0step7_final10_v1"
    cluster_name = f"k{k}_unanchored"
    boundary_name = f"{domain}_mean_r{replan}_nms{nms}"
    cfg["probe_variants"] = [{
        "name": probe_name, "mode": "std", "probe_generation": "scheduler_gaussian",
        "gaussian_center": center, "gaussian_sampling": "iid", "gaussian_include_mean": False,
        "gaussian_seed": int(cfg["seed"]), "count": 96, "eval_at_step": 7, "denoise_steps": 1,
        "denoise_output": "pred_original_sample", "replan_interval": replan,
        "pca_variance": cfg["pca_variance"], "pca_stride": cfg["pca_stride"], "pca_temporal_length": length,
        "descriptor_mode": "std", "descriptor_pca_representation": "mean",
        "descriptor_pca_variance": cfg["pca_variance"], "descriptor_pca_stride": cfg["pca_stride"],
        "descriptor_pca_temporal_length": length, "descriptor_temporal_length": length,
        "descriptor_temporal_bins": 1, "descriptor_include_endpoint": False,
    }]
    cfg["cluster_variants"] = [{
        "name": cluster_name, "selection": "fixed", "k": k, "covariance": "diag", "n_init": 10,
        "max_iter": 300, "min_effective_samples": 0, "weighted": True, "reference_center": "none",
    }]
    cfg["boundary_variants"] = [{
        "name": boundary_name, "method": "peak", "score_power": power, "threshold": "mean",
        "threshold_scope": "dataset", "threshold_scale": 1.0, "smooth_window": window,
        "polyorder": polyorder, "nms_frames": nms, "prominence": 0.0, "terminal_mask_frames": 0,
        "terminal_mask_mode": "abc_terminal_release" if domain == "abc" else "fixed",
        "min_skill_len": 15 if domain == "abc" else 10,
        "initial_mask_frames": initial_mask_frames,
        "peak_curvature_scale": curvature,
    }]
    if domain == "libero":
        cfg["boundary_variants"][0]["terminal_mask_frames"] = 24
    cfg["experiments"] = [{
        "name": f"{horizon_tag}_{center}_unanchored", "probe": probe_name, "cluster": cluster_name,
        "metric": "cosine" if domain == "abc" else "covariance_gated_cosine", "boundary": boundary_name,
    }]
    return cfg, model
