#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402


_DENOISING_SUMMARY_METRICS = {
    "denoising_total_drift",
    "denoising_late_drift",
    "denoising_correction_delay",
    "denoising_correction_step",
    "denoising_knee_step",
    "denoising_threshold_step",
    "denoising_relative_threshold_step",
    "denoising_aligned_threshold_step",
    "denoising_aligned_relative_threshold_step",
    "denoising_aligned_total_drift",
    "denoising_aligned_correction_step",
}


def _is_denoising_metric(value: object) -> bool:
    metric = str(value)
    return metric in _DENOISING_SUMMARY_METRICS or bool(
        re.fullmatch(r"denoising(?:_aligned)?_drift_[0-9]+_[0-9]+", metric)
    )


def _experiment_probe_names(experiment: dict) -> list[str]:
    probes = experiment.get("probes")
    if probes is not None:
        return [str(value) for value in probes]
    return [str(experiment["probe"])]


def _named_list(cfg: dict, key: str, *, allow_empty: bool = False) -> list[dict]:
    value = get_value(cfg, key, None)
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{key} must be {qualifier}.")
    names: set[str] = set()
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{key}[{index}] must be a mapping.")
        name = str(item.get("name", "")).strip()
        if not name or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise ValueError(f"{key}[{index}].name is invalid: {name!r}")
        if name in names:
            raise ValueError(f"Duplicate {key} name: {name}")
        names.add(name)
        result.append(dict(item))
    return result


def build_settings(path: str) -> dict:
    cfg = load_config(path)
    root = Path(str(get_value(cfg, "project_root"))).expanduser().resolve()
    dataset_root = Path(str(get_value(cfg, "dataset_root", "dataset_filtered")))
    if not dataset_root.is_absolute():
        dataset_root = root / dataset_root
    components = {
        key: str(get_value(cfg, key, "")).strip()
        for key in ("target_dataset", "fsq_dataset_root", "fsq_inputs_name", "skillset_seg_name", "skillset_name")
    }
    for key, value in components.items():
        if not value or Path(value).name != value:
            raise ValueError(f"{key} must be a folder name, got {value!r}")
    requested_ablations = [
        str(value).strip().lower()
        for value in as_list(get_value(cfg, "ablations", []))
        if str(value).strip()
    ]
    if len(requested_ablations) != len(set(requested_ablations)):
        raise ValueError("ablations must not contain duplicates")
    skillset_dir = (
        dataset_root / components["fsq_dataset_root"] / components["target_dataset"]
        / components["fsq_inputs_name"] / components["skillset_seg_name"]
        / components["skillset_name"]
    )
    manifest = skillset_dir / "skillset_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Skillset manifest not found: {manifest}")
    output_name = str(get_value(cfg, "output_name", "boundary_ablation")).strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", output_name):
        raise ValueError(f"Invalid output_name: {output_name!r}")
    if requested_ablations:
        selection_tag = "all" if "all" in requested_ablations else "-".join(requested_ablations)
        output_name = f"{output_name}__{selection_tag}"
    output_parent_name = str(
        get_value(cfg, "output_parent_name", "") or ""
    ).strip()
    if output_parent_name and not re.fullmatch(
        r"[A-Za-z0-9._-]+", output_parent_name
    ):
        raise ValueError(f"Invalid output_parent_name: {output_parent_name!r}")
    cache_source_output_name = str(
        get_value(cfg, "cache_source_output_name", "") or ""
    ).strip()
    if cache_source_output_name and not re.fullmatch(
        r"[A-Za-z0-9._-]+", cache_source_output_name
    ):
        raise ValueError(
            f"Invalid cache_source_output_name: {cache_source_output_name!r}"
        )
    cache_source_output_parent_name = str(
        get_value(cfg, "cache_source_output_parent_name", "") or ""
    ).strip()
    if cache_source_output_parent_name and not re.fullmatch(
        r"[A-Za-z0-9._-]+", cache_source_output_parent_name
    ):
        raise ValueError(
            "Invalid cache_source_output_parent_name: "
            f"{cache_source_output_parent_name!r}"
        )

    raw_probes = _named_list(cfg, "probe_variants", allow_empty=True)
    # A dense sample-count sweep should not repeat the DP forward pass for
    # nested Monte-Carlo clouds.  ``derive_from`` makes a smaller probe the
    # exact prefix of an earlier, larger probe (GT is sample zero).  Expand the
    # inherited recipe here so downstream cache metadata stays self-contained.
    probes = []
    resolved_probes: dict[str, dict] = {}
    for item in raw_probes:
        source_name = str(item.get("derive_from", "") or "").strip()
        if source_name:
            unsupported_overrides = set(item) - {"name", "derive_from", "count"}
            if unsupported_overrides:
                raise ValueError(
                    f"Derived probe {item['name']} may override only count; remove "
                    f"{sorted(unsupported_overrides)}."
                )
            if source_name not in resolved_probes:
                raise ValueError(
                    f"Probe {item['name']}.derive_from must name an earlier probe; "
                    f"got {source_name!r}"
                )
            source = resolved_probes[source_name]
            if source.get("derive_from"):
                raise ValueError(
                    f"Probe {item['name']} cannot derive from another derived probe "
                    f"({source_name}); derive directly from the largest probe."
                )
            merged = dict(source)
            merged.update(item)
            merged["name"] = item["name"]
            merged["derive_from"] = source_name
            source_count = int(source.get("count", 24))
            count = int(merged.get("count", 24))
            if count < 1 or count >= source_count:
                raise ValueError(
                    f"Probe {item['name']}.count must be in [1, {source_count - 1}] "
                    f"when derived from {source_name}; got {count}."
                )
            probes.append(merged)
            resolved_probes[item["name"]] = merged
        else:
            count = int(item.get("count", 24))
            if count < 1:
                raise ValueError(f"Probe {item['name']}.count must be positive")
            probes.append(item)
            resolved_probes[item["name"]] = item
    for item in probes:
        representation = str(item.get("pca_representation", "mean"))
        if representation not in {"mean", "trajectory"}:
            raise ValueError(
                f"Probe {item['name']}.pca_representation must be mean|trajectory"
            )
        descriptor_representation = str(
            item.get("descriptor_pca_representation", representation)
        )
        if descriptor_representation not in {"mean", "trajectory"}:
            raise ValueError(
                f"Probe {item['name']}.descriptor_pca_representation must be "
                "mean|trajectory"
            )
        generation = str(item.get("probe_generation", "pca_offset"))
        if generation not in {"pca_offset", "pca_sigma", "scheduler_gaussian"}:
            raise ValueError(
                f"Probe {item['name']}.probe_generation must be "
                "pca_offset|pca_sigma|scheduler_gaussian"
            )
        sampling = str(item.get("direction_sampling", "uniform"))
        if sampling not in {"uniform", "variance_weighted", "axes_pairwise"}:
            raise ValueError(
                f"Probe {item['name']}.direction_sampling must be "
                "uniform|variance_weighted|axes_pairwise"
            )
        component_limit = item.get("pca_component_limit")
        if component_limit is not None and int(component_limit) < 1:
            raise ValueError(
                f"Probe {item['name']}.pca_component_limit must be positive"
            )
        if sampling == "axes_pairwise":
            if component_limit is None:
                raise ValueError(
                    f"Probe {item['name']} axes_pairwise needs pca_component_limit"
                )
            expected = 4 * int(component_limit)
            if int(item.get("count", 24)) != expected:
                raise ValueError(
                    f"Probe {item['name']} axes_pairwise count must be {expected}"
                )
        if generation == "pca_sigma":
            if component_limit is None:
                raise ValueError(
                    f"Probe {item['name']} pca_sigma needs pca_component_limit"
                )
            expected = 2 * int(component_limit)
            if int(item.get("count", 24)) != expected:
                raise ValueError(
                    f"Probe {item['name']} pca_sigma count must be {expected}"
                )
            sigma_scale = float(item.get("sigma_scale", 1.0))
            if not math.isfinite(sigma_scale) or sigma_scale <= 0.0:
                raise ValueError(
                    f"Probe {item['name']}.sigma_scale must be finite and positive"
                )
        if generation == "scheduler_gaussian":
            gaussian_sampling = str(item.get("gaussian_sampling", "antithetic"))
            if gaussian_sampling not in {"iid", "antithetic"}:
                raise ValueError(
                    f"Probe {item['name']}.gaussian_sampling must be iid|antithetic"
                )
            if (
                gaussian_sampling == "antithetic"
                and int(item.get("count", 24)) % 2
            ):
                raise ValueError(
                    f"Probe {item['name']} antithetic Gaussian count must be even"
                )
            if "gaussian_seed" in item:
                int(item["gaussian_seed"])
        if "replan_interval" in item and int(item["replan_interval"]) < 1:
            raise ValueError(
                f"Probe {item['name']}.replan_interval must be positive"
            )
        denoise_output = str(item.get("denoise_output", "prev_sample"))
        if denoise_output not in {"prev_sample", "pred_original_sample"}:
            raise ValueError(
                f"Probe {item['name']}.denoise_output must be "
                "prev_sample|pred_original_sample"
            )
    clusters = _named_list(cfg, "cluster_variants")
    for item in clusters:
        reference_center = str(item.get("reference_center", "none"))
        if reference_center not in {"none", "sample0"}:
            raise ValueError(
                f"Cluster {item['name']}.reference_center must be none|sample0"
            )
    boundaries = _named_list(cfg, "boundary_variants")
    experiments = _named_list(cfg, "experiments")
    rollout = dict(get_value(cfg, "rollout", {}) or {})
    rollout["enabled"] = as_bool(rollout.get("enabled", False))
    rollout["x0_temporal_alignment"] = as_bool(
        rollout.get("x0_temporal_alignment", False)
    )
    prediction_loss = dict(get_value(cfg, "prediction_loss", {}) or {})
    prediction_loss["enabled"] = as_bool(prediction_loss.get("enabled", False))
    prediction_loss["common_noise"] = as_bool(
        prediction_loss.get("common_noise", True)
    )
    probe_names = {x["name"] for x in probes}
    selectable_probe_names = set(probe_names)
    if rollout["enabled"]:
        selectable_probe_names.add("direct_rollout")
    if prediction_loss["enabled"]:
        selectable_probe_names.add("prediction_loss")
    cluster_names = {x["name"] for x in clusters}
    boundary_names = {x["name"] for x in boundaries}
    for item in boundaries:
        if "min_skill_len" in item and int(item["min_skill_len"]) < 1:
            raise ValueError(
                f"Boundary {item['name']}.min_skill_len must be positive"
            )
        score_power = float(item.get("score_power", 1.0))
        if not math.isfinite(score_power) or score_power <= 0.0:
            raise ValueError(
                f"Boundary {item['name']}.score_power must be finite and positive"
            )
        terminal_mask_frames = int(item.get("terminal_mask_frames", 0))
        if terminal_mask_frames < 0:
            raise ValueError(
                f"Boundary {item['name']}.terminal_mask_frames must be non-negative"
            )
        start_guard_frames = int(
            item.get("start_guard_frames", item.get("nms_frames", 20))
        )
        if start_guard_frames < 0:
            raise ValueError(
                f"Boundary {item['name']}.start_guard_frames must be non-negative"
            )
        nms_frames = int(item.get("nms_frames", 20))
        if nms_frames < 0:
            raise ValueError(
                f"Boundary {item['name']}.nms_frames must be non-negative"
            )
        if "denoising_convergence_threshold" in item:
            convergence_threshold = float(item["denoising_convergence_threshold"])
            if not math.isfinite(convergence_threshold) or convergence_threshold <= 0.0:
                raise ValueError(
                    f"Boundary {item['name']}.denoising_convergence_threshold "
                    "must be finite and positive"
                )
        if "denoising_relative_convergence_threshold" in item:
            relative_threshold = float(
                item["denoising_relative_convergence_threshold"]
            )
            if not math.isfinite(relative_threshold) or not 0.0 < relative_threshold < 1.0:
                raise ValueError(
                    f"Boundary {item['name']}.denoising_relative_convergence_threshold "
                    "must be finite and in (0, 1)"
                )
        sample_aggregation = str(
            item.get("denoising_sample_aggregation", "median")
        )
        if sample_aggregation not in {"mean", "median"}:
            raise ValueError(
                f"Boundary {item['name']}.denoising_sample_aggregation must be "
                "mean|median"
            )
        convergence_mode = str(
            item.get("denoising_convergence_threshold_mode", "fixed")
        )
        if convergence_mode not in {"fixed", "episode_geometric_mean"}:
            raise ValueError(
                f"Boundary {item['name']}.denoising_convergence_threshold_mode "
                "must be fixed|episode_geometric_mean"
            )
        if (
            convergence_mode == "episode_geometric_mean"
            and "denoising_convergence_threshold" in item
        ):
            raise ValueError(
                f"Boundary {item['name']} cannot combine episode_geometric_mean "
                "with denoising_convergence_threshold"
            )
        method = str(item.get("method", "peak"))
        if method not in {
            "peak",
            "rising",
            "all_peaks",
            "persistent_local",
            "sustained_plateau",
            "running_surprise",
        }:
            raise ValueError(
                f"Boundary {item['name']}.method must be "
                "peak|rising|all_peaks|persistent_local|sustained_plateau|"
                "running_surprise"
            )
        threshold_target = str(item.get("threshold_target", "score"))
        if threshold_target not in {"score", "prominence"}:
            raise ValueError(
                f"Boundary {item['name']}.threshold_target must be "
                "score|prominence"
            )
        if threshold_target == "prominence" and method != "peak":
            raise ValueError(
                f"Boundary {item['name']} prominence thresholding requires "
                "method=peak"
            )
        threshold_scope = str(item.get("threshold_scope", "episode"))
        if threshold_scope not in {"episode", "task", "dataset"}:
            raise ValueError(
                f"Boundary {item['name']}.threshold_scope must be "
                "episode|task|dataset"
            )
        if threshold_scope != "episode" and method not in {"peak", "rising"}:
            raise ValueError(
                f"Boundary {item['name']}.threshold_scope={threshold_scope} "
                "is supported only for peak|rising detectors"
            )
        if str(item.get("threshold", "mean")) not in {
            "fixed",
            "mean",
            "rms",
            "median_mad",
            "median_mad_floor_zero",
            "gap_mad",
        }:
            raise ValueError(
                f"Boundary {item['name']}.threshold must be "
                "fixed|mean|rms|median_mad|median_mad_floor_zero|gap_mad"
            )
        if method == "running_surprise" and int(
            item.get("running_surprise_min_history_points", 2)
        ) < 1:
            raise ValueError(
                f"Boundary {item['name']}.running_surprise_min_history_points "
                "must be positive"
            )
        if method == "persistent_local":
            windows = item.get("smooth_windows", [5, 7, 9])
            if (
                not isinstance(windows, list)
                or len(windows) < 3
                or any(int(window) < 3 or int(window) % 2 == 0 for window in windows)
            ):
                raise ValueError(
                    f"Boundary {item['name']}.smooth_windows must contain at "
                    "least three odd integers >= 3"
                )
        if method == "sustained_plateau":
            if int(item.get("plateau_min_active_points", 3)) < 1:
                raise ValueError(
                    f"Boundary {item['name']}.plateau_min_active_points must be positive"
                )
            if int(item.get("plateau_max_gap_points", 1)) < 0:
                raise ValueError(
                    f"Boundary {item['name']}.plateau_max_gap_points must be non-negative"
                )
    for item in clusters:
        magnitude_gate_tau = float(item.get("magnitude_gate_tau", 1.0))
        if magnitude_gate_tau < 0.0:
            raise ValueError(
                f"Cluster {item['name']}.magnitude_gate_tau must be non-negative"
            )
        if "min_effective_samples" in item and "min_weight" in item:
            raise ValueError(
                f"Cluster {item['name']} must specify only one of "
                "min_effective_samples|min_weight"
            )
        min_effective_samples = float(item.get("min_effective_samples", 0.0))
        if not math.isfinite(min_effective_samples) or min_effective_samples < 0.0:
            raise ValueError(
                f"Cluster {item['name']}.min_effective_samples must be finite "
                "and non-negative"
            )
        min_weight = float(item.get("min_weight", 0.0))
        if not math.isfinite(min_weight) or not 0.0 <= min_weight <= 1.0:
            raise ValueError(
                f"Cluster {item['name']}.min_weight must be finite and in [0, 1]"
            )
    for item in experiments:
        selected_probes = item.get("probes")
        if selected_probes is not None:
            if not isinstance(selected_probes, list) or not selected_probes:
                raise ValueError(f"Experiment {item['name']}.probes must be a non-empty list")
            unknown = [name for name in selected_probes if name not in selectable_probe_names]
            if unknown:
                raise ValueError(f"Experiment {item['name']} selects unknown probes {unknown!r}")
        elif item.get("probe") not in selectable_probe_names:
            raise ValueError(f"Experiment {item['name']} selects unknown probe {item.get('probe')!r}")
        if item.get("cluster") not in cluster_names:
            raise ValueError(f"Experiment {item['name']} selects unknown cluster {item.get('cluster')!r}")
        if item.get("boundary") not in boundary_names:
            raise ValueError(f"Experiment {item['name']} selects unknown boundary {item.get('boundary')!r}")
        if selected_probes is not None:
            normalization = str(item.get("consensus_normalization", "none"))
            if normalization not in {"none", "robust_z"}:
                raise ValueError(
                    f"Experiment {item['name']}.consensus_normalization must be none|robust_z"
                )
            aggregation = str(item.get("consensus_aggregation", "median"))
            if aggregation not in {"median", "mean"}:
                raise ValueError(
                    f"Experiment {item['name']}.consensus_aggregation must be median|mean"
                )
            support = int(item.get("consensus_peak_support", 0))
            if support < 0 or support > len(selected_probes):
                raise ValueError(
                    f"Experiment {item['name']}.consensus_peak_support must be in "
                    f"[0, {len(selected_probes)}]"
                )
            if int(item.get("consensus_peak_tolerance_frames", 0)) < 0:
                raise ValueError(
                    f"Experiment {item['name']}.consensus_peak_tolerance_frames must be non-negative"
                )
        if item.get("metric") not in {
            "cosine",
            "magnitude_gated_cosine",
            "covariance_gated_cosine",
            "covariance_gated_angular_chord",
            "max_covariance_gated_cosine",
            "l2",
            "within_normalized_l2",
            "hybrid",
            "delta_bic",
            "delta_bic_gain",
            "whitened_spread",
            "whitened_spread_gain",
            "bic_multi_probability",
            "sliced_wasserstein_shift",
            "mmd_shift",
            "paired_rollout_shift",
            "trajectory_xyz_cosine",
            "trajectory_xyz_l2",
            "endpoint_xyz_spread",
            "action_sequence_cosine",
            "action_sequence_l2",
            "action_mean_cosine",
            "prediction_loss_current",
            "prediction_loss_current_no_gripper",
            "prediction_loss_future_mean",
            "prediction_loss_future_mean_no_gripper",
            "prediction_loss_aligned",
            "prediction_loss_aligned_no_gripper",
        } and not _is_denoising_metric(item.get("metric")):
            raise ValueError(
                f"Experiment {item['name']} metric must be cosine|magnitude_gated_cosine|"
                "covariance_gated_cosine|"
                "covariance_gated_angular_chord|"
                "max_covariance_gated_cosine|"
                "l2|within_normalized_l2|hybrid|"
                "delta_bic|delta_bic_gain|whitened_spread|whitened_spread_gain|"
                "bic_multi_probability|"
                "sliced_wasserstein_shift|mmd_shift|paired_rollout_shift|"
                "trajectory_xyz_cosine|trajectory_xyz_l2|endpoint_xyz_spread|"
                "action_sequence_cosine|action_sequence_l2|action_mean_cosine|"
                "prediction_loss_current[_no_gripper]|"
                "prediction_loss_future_mean[_no_gripper]|"
                "prediction_loss_aligned[_no_gripper]"
                "|denoising_drift_<left>_<right>|denoising_total_drift|"
                "denoising_late_drift|denoising_correction_delay|"
                "denoising_correction_step|denoising_knee_step"
                "|denoising_threshold_step"
                "|denoising_relative_threshold_step"
                "|denoising_aligned_threshold_step"
                "|denoising_aligned_relative_threshold_step"
            )

    # The single maintained YAML contains the useful experiment families.  A
    # short ``ablations`` list selects which families are actually evaluated;
    # then prune their unused probe/GMM/boundary definitions so a one-family
    # run does not pay for every experiment in the catalog.  Configs without
    # the selector retain the historical behavior and run every listed row.
    if requested_ablations:
        available_ablations = {
            str(item.get("ablation", "")).strip().lower()
            for item in experiments
            if str(item.get("ablation", "")).strip()
        }
        unknown_ablations = set(requested_ablations) - available_ablations - {"all"}
        if unknown_ablations:
            raise ValueError(
                "Unknown ablations "
                f"{sorted(unknown_ablations)}; choose from "
                f"{sorted(available_ablations)} or all"
            )
        if "all" not in requested_ablations:
            experiments = [
                item
                for item in experiments
                if str(item.get("ablation", "")).strip().lower()
                in requested_ablations
            ]
        if not experiments:
            raise ValueError("ablations selected no experiments")

        required_probe_names = {
            name
            for item in experiments
            for name in _experiment_probe_names(item)
            if name not in {"direct_rollout", "prediction_loss"}
        }
        # Derived probes need their full-size source even when only the derived
        # condition is displayed.
        by_probe_name = {item["name"]: item for item in probes}
        pending = list(required_probe_names)
        while pending:
            name = pending.pop()
            source = str(by_probe_name[name].get("derive_from", "") or "").strip()
            if source and source not in required_probe_names:
                required_probe_names.add(source)
                pending.append(source)
        required_cluster_names = {str(item["cluster"]) for item in experiments}
        required_boundary_names = {str(item["boundary"]) for item in experiments}
        probes = [item for item in probes if item["name"] in required_probe_names]
        clusters = [
            item for item in clusters if item["name"] in required_cluster_names
        ]
        boundaries = [
            item for item in boundaries if item["name"] in required_boundary_names
        ]

    if rollout["enabled"]:
        if rollout.get("probe") not in probe_names:
            raise ValueError(f"rollout.probe selects unknown probe {rollout.get('probe')!r}")
        if rollout.get("cluster") not in cluster_names:
            raise ValueError(f"rollout.cluster selects unknown cluster {rollout.get('cluster')!r}")
        if rollout.get("boundary") not in boundary_names:
            raise ValueError(f"rollout.boundary selects unknown boundary {rollout.get('boundary')!r}")
        if rollout.get("metric") not in {
            "cosine",
            "magnitude_gated_cosine",
            "covariance_gated_cosine",
            "covariance_gated_angular_chord",
            "max_covariance_gated_cosine",
            "l2",
            "within_normalized_l2",
            "hybrid",
            "delta_bic",
            "bic_multi_probability",
            "sliced_wasserstein_shift",
            "mmd_shift",
            "paired_rollout_shift",
            "trajectory_xyz_cosine",
            "trajectory_xyz_l2",
            "endpoint_xyz_spread",
            "action_sequence_cosine",
            "action_sequence_l2",
            "action_mean_cosine",
        } and not _is_denoising_metric(rollout.get("metric")):
            raise ValueError(
                "rollout.metric must be cosine|magnitude_gated_cosine|"
                "covariance_gated_cosine|covariance_gated_angular_chord|"
                "max_covariance_gated_cosine|"
                "l2|within_normalized_l2|hybrid|"
                "delta_bic|bic_multi_probability|"
                "sliced_wasserstein_shift|mmd_shift|paired_rollout_shift|"
                "trajectory_xyz_cosine|trajectory_xyz_l2|endpoint_xyz_spread|"
                "action_sequence_cosine|action_sequence_l2|action_mean_cosine"
                "|denoising_*"
            )
        if int(rollout.get("samples", 32)) < 2:
            raise ValueError("rollout.samples must be at least 2")
        if int(rollout.get("batch_size", 16)) < 1:
            raise ValueError("rollout.batch_size must be positive")
        if int(rollout.get("replan_interval", 1)) < 1:
            raise ValueError("rollout.replan_interval must be positive")
        snapshot_steps = [int(step) for step in rollout.get("x0_snapshot_steps", [])]
        if snapshot_steps and (
            len(snapshot_steps) < 2
            or snapshot_steps != sorted(set(snapshot_steps))
            or snapshot_steps[0] < 0
        ):
            raise ValueError(
                "rollout.x0_snapshot_steps must contain at least two unique, "
                "strictly increasing non-negative integers."
            )
        if rollout["x0_temporal_alignment"]:
            rollout_probe = resolved_probes[str(rollout["probe"])]
            if str(rollout_probe.get("pca_representation", "mean")) != "mean":
                raise ValueError(
                    "rollout.x0_temporal_alignment requires a per-action "
                    "PCA probe (pca_representation=mean)."
                )
    if prediction_loss["enabled"]:
        if int(prediction_loss.get("replan_interval", 1)) < 1:
            raise ValueError("prediction_loss.replan_interval must be positive")
        if int(prediction_loss.get("noise_samples_per_timestep", 1)) < 1:
            raise ValueError(
                "prediction_loss.noise_samples_per_timestep must be positive"
            )
        if int(prediction_loss.get("batch_size", 64)) < 1:
            raise ValueError("prediction_loss.batch_size must be positive")
        loss_timesteps = [
            int(step) for step in prediction_loss.get("noise_timesteps", [])
        ]
        if (
            not loss_timesteps
            or loss_timesteps != sorted(set(loss_timesteps))
            or loss_timesteps[0] < 0
        ):
            raise ValueError(
                "prediction_loss.noise_timesteps must contain unique, strictly "
                "increasing non-negative integers"
            )
    denoising_experiments = [
        item["name"] for item in experiments if _is_denoising_metric(item.get("metric"))
    ]
    if denoising_experiments and (
        not rollout["enabled"] or len(rollout.get("x0_snapshot_steps", [])) < 2
    ):
        raise ValueError(
            "Denoising-convergence experiments require rollout.enabled=true and "
            "at least two rollout.x0_snapshot_steps; experiments="
            f"{denoising_experiments}."
        )
    aligned_experiments = [
        item["name"]
        for item in experiments
        if str(item.get("metric", "")).startswith("denoising_aligned_")
    ]
    if aligned_experiments and not rollout["x0_temporal_alignment"]:
        raise ValueError(
            "Time-aligned denoising experiments require "
            "rollout.x0_temporal_alignment=true; experiments="
            f"{aligned_experiments}."
        )
    output_root = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / components["target_dataset"]
    )
    output_dir = output_root / output_name
    if output_parent_name:
        output_dir = output_root / output_parent_name / "runs" / output_name
    cache_source_path: Path | None = None
    if cache_source_output_name:
        cache_candidates = []
        if cache_source_output_parent_name:
            cache_candidates.append(
                output_root
                / cache_source_output_parent_name
                / "runs"
                / cache_source_output_name
            )
        if output_parent_name:
            cache_candidates.append(
                output_root
                / output_parent_name
                / "runs"
                / cache_source_output_name
            )
        cache_candidates.extend(
            [
                output_root / cache_source_output_name,
                output_root / "PREV" / cache_source_output_name,
            ]
        )
        cache_source_path = next(
            (candidate for candidate in cache_candidates if candidate.exists()),
            cache_candidates[0],
        )
    cache_source_output_names = [
        str(value).strip()
        for value in as_list(get_value(cfg, "cache_source_output_names", []))
    ]
    for value in cache_source_output_names:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError(f"Invalid cache_source_output_names entry: {value!r}")
    cache_source_paths: list[Path] = []
    for source_name in cache_source_output_names:
        source_candidates = []
        if cache_source_output_parent_name:
            source_candidates.append(
                output_root
                / cache_source_output_parent_name
                / "runs"
                / source_name
            )
        if output_parent_name:
            source_candidates.append(output_root / output_parent_name / "runs" / source_name)
        source_candidates.extend(
            [output_root / source_name, output_root / "PREV" / source_name]
        )
        cache_source_paths.append(
            next(
                (candidate for candidate in source_candidates if candidate.exists()),
                source_candidates[0],
            )
        )
    cache_source_output_groups = get_value(cfg, "cache_source_output_groups", [])
    if not isinstance(cache_source_output_groups, list):
        raise ValueError("cache_source_output_groups must be a list.")
    for index, group in enumerate(cache_source_output_groups):
        if not isinstance(group, dict):
            raise ValueError(
                f"cache_source_output_groups[{index}] must be a mapping."
            )
        parent_name = str(group.get("parent", "")).strip()
        names = [str(value).strip() for value in as_list(group.get("names", []))]
        if not parent_name or not re.fullmatch(r"[A-Za-z0-9._-]+", parent_name):
            raise ValueError(
                f"Invalid cache_source_output_groups[{index}].parent: {parent_name!r}"
            )
        if not names:
            raise ValueError(
                f"cache_source_output_groups[{index}].names must not be empty."
            )
        for source_name in names:
            if not re.fullmatch(r"[A-Za-z0-9._-]+", source_name):
                raise ValueError(
                    f"Invalid cache source run name in group {index}: {source_name!r}"
                )
            source_candidates = [
                output_root / parent_name / "runs" / source_name,
                output_root / "PREV" / parent_name / "runs" / source_name,
            ]
            cache_source_paths.append(
                next(
                    (candidate for candidate in source_candidates if candidate.exists()),
                    source_candidates[0],
                )
            )
    if cache_source_path is not None:
        cache_source_paths.insert(0, cache_source_path)
    # Preserve order while avoiding repeated cache scans when both the legacy
    # scalar option and the new multi-source option name the same run.
    cache_source_paths = list(dict.fromkeys(cache_source_paths))
    settings = {
        "project_root": str(root),
        "skillset_dir": str(skillset_dir),
        "output_dir": str(output_dir),
        "cache_source_dir": (
            "" if cache_source_path is None else str(cache_source_path)
        ),
        "cache_source_dirs": [str(path) for path in cache_source_paths],
        "policy_checkpoint_override": str(
            get_value(cfg, "policy_checkpoint_override", "") or ""
        ).strip(),
        "task_id_space": str(get_value(cfg, "task_id_space", "suite")),
        "target_task": str(get_value(cfg, "target_task", "")),
        "task_ids": [int(x) for x in as_list(get_value(cfg, "task_ids", []))],
        "n_episodes": int(get_value(cfg, "n_episodes", 20)),
        "resume": as_bool(get_value(cfg, "resume", True)),
        "split_report_by_step": as_bool(
            get_value(cfg, "split_report_by_step", False)
        ),
        "split_report_by_group": as_bool(
            get_value(cfg, "split_report_by_group", False)
        ),
        "split_report_by_task": as_bool(
            get_value(cfg, "split_report_by_task", False)
        ),
        "overlay_experiment_curves": as_bool(
            get_value(cfg, "overlay_experiment_curves", False)
        ),
        "report_compare_limit": int(get_value(cfg, "report_compare_limit", 6)),
        "report_compare_columns": int(get_value(cfg, "report_compare_columns", 0)),
        "report_image_key": str(
            get_value(cfg, "report_image_key", "observation.images.image")
        ).strip(),
        "report_consistency_metrics": as_bool(
            get_value(cfg, "report_consistency_metrics", False)
        ),
        "report_consistency_tolerance_frames": int(
            get_value(cfg, "report_consistency_tolerance_frames", 6)
        ),
        "seed": int(get_value(cfg, "seed", 42)),
        "pca_variance": float(get_value(cfg, "pca_variance", 0.95)),
        "pca_stride": int(get_value(cfg, "pca_stride", 3)),
        "ablations": requested_ablations,
        "probe_variants": probes,
        "cluster_variants": clusters,
        "boundary_variants": boundaries,
        "experiments": experiments,
        "rollout": rollout,
        "prediction_loss": prediction_loss,
    }
    if settings["task_id_space"] not in {"suite", "dataset"}:
        raise ValueError("task_id_space must be suite|dataset")
    if not settings["task_ids"]:
        raise ValueError("task_ids must select at least one task")
    if settings["n_episodes"] < 1:
        raise ValueError("n_episodes must be positive")
    if settings["report_compare_limit"] < 1:
        raise ValueError("report_compare_limit must be positive")
    if not 0 <= settings["report_compare_columns"] <= settings["report_compare_limit"]:
        raise ValueError(
            "report_compare_columns must be zero or no greater than "
            "report_compare_limit"
        )
    if settings["report_consistency_tolerance_frames"] < 0:
        raise ValueError("report_consistency_tolerance_frames must be non-negative")
    return {
        "project_root": str(root),
        "dp_ablation_skillset_dir": str(skillset_dir),
        "dp_ablation_output_dir": settings["output_dir"],
        "dp_ablation_selection": ",".join(requested_ablations) or "custom",
        "dp_ablation_settings_json": json.dumps(settings, separators=(",", ":")),
        "dp_ablation_partition": ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug",
        "dp_ablation_qos": str(
            get_value(cfg, "eval_qos", get_value(cfg, "train_qos", "base_qos"))
        ),
        "dp_ablation_gres": str(get_value(cfg, "eval_gres", "gpu:1")),
        "dp_ablation_cpus": int(get_value(cfg, "eval_cpus_per_task", 4)),
        "dp_ablation_mem": str(get_value(cfg, "eval_mem", "48G")),
        "dp_ablation_time": str(get_value(cfg, "eval_time", "04:00:00")),
        "dp_ablation_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "dp_ablation_exclude_nodes": ",".join(as_list(get_value(cfg, "train_exclude_nodes", []))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(args.config)
    if args.shell:
        print_shell(settings)
    else:
        print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
