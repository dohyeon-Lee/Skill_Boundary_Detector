#!/usr/bin/env python3
"""One-model DP boundary ablation with resumable raw descriptor caches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import re
import shutil
from pathlib import Path

import numpy as np

LIBERO_EXAMPLES = Path(__file__).resolve().parents[5]
PROJECT_ROOT = LIBERO_EXAMPLES.parents[2]


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return result or "item"


def _portable_path(raw: str, marker: str) -> Path:
    path = Path(raw).expanduser()
    if path.exists():
        return path
    if marker in path.parts:
        candidate = PROJECT_ROOT.joinpath(*path.parts[path.parts.index(marker) :])
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Path from manifest is unavailable: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(skillset_dir: Path) -> dict:
    path = skillset_dir / "skillset_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _named(items: list[dict], kind: str) -> dict[str, dict]:
    result = {str(item["name"]): dict(item) for item in items}
    if len(result) != len(items):
        raise ValueError(f"Duplicate {kind} names")
    return result


def _experiment_probe_names(experiment: dict) -> list[str]:
    values = experiment.get("probes")
    if values is not None:
        return [str(value) for value in values]
    return [str(experiment["probe"])]


def _consensus_kwargs(experiment: dict) -> dict:
    """Resolve optional multi-noise-level aggregation controls."""
    positive = experiment.get("consensus_min_fraction")
    return {
        "min_positive_fraction": None if positive is None else float(positive),
        "normalization": str(experiment.get("consensus_normalization", "none")),
        "aggregation": str(experiment.get("consensus_aggregation", "median")),
        "min_peak_support": int(experiment.get("consensus_peak_support", 0)),
        "peak_tolerance_frames": int(
            experiment.get("consensus_peak_tolerance_frames", 0)
        ),
    }


def _antithetic_directions(
    pca,
    count: int,
    seed: int,
    *,
    component_limit: int | None = None,
    sampling: str = "uniform",
) -> np.ndarray:
    if count % 2:
        raise ValueError("Antithetic probe count must be even.")
    half = pca.sample_directions(
        count // 2,
        seed,
        component_limit=component_limit,
        sampling=sampling,
    )
    return np.concatenate([half, -half], axis=0)


def _antithetic_gaussian_noise(
    count: int, horizon: int, action_dim: int, seed: int
) -> np.ndarray:
    """Fixed standard-normal templates paired as ``epsilon`` and ``-epsilon``."""
    if count < 2 or count % 2:
        raise ValueError("Antithetic Gaussian probe count must be a positive even number.")
    rng = np.random.default_rng(seed)
    half = rng.standard_normal((count // 2, horizon, action_dim)).astype(np.float32)
    return np.concatenate([half, -half], axis=0)


def _iid_gaussian_noise(
    count: int, horizon: int, action_dim: int, seed: int
) -> np.ndarray:
    """Fixed IID standard-normal templates shared across episode anchors."""
    if count < 1:
        raise ValueError("IID Gaussian probe count must be positive.")
    return np.random.default_rng(seed).standard_normal(
        (count, horizon, action_dim)
    ).astype(np.float32)


def _cache_load(path: Path, metadata: dict) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["metadata_json"]) != json.dumps(metadata, sort_keys=True):
                return None
            payload = {
                "replan_ts": data["replan_ts"].copy(),
                "descriptors": data["descriptors"].copy(),
                "n_frames": np.asarray(data["n_frames"]).copy(),
            }
            for key in (
                "trajectory_xyz",
                "action_sequence",
                "x0_descriptor_snapshots",
                "x0_temporal_corrections",
                "denoising_snapshot_steps",
                "prediction_loss_per_offset",
                "prediction_loss_per_offset_no_gripper",
                "prediction_loss_by_timestep",
                "prediction_loss_timesteps",
            ):
                if key in data:
                    payload[key] = data[key].copy()
            return payload
    except (KeyError, OSError, ValueError):
        return None


def _cache_save(path: Path, payload: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    arrays = {
        "replan_ts": np.asarray(payload["replan_ts"], dtype=np.int64),
        "descriptors": np.asarray(payload["descriptors"], dtype=np.float32),
        "n_frames": np.asarray(payload["n_frames"], dtype=np.int64),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }
    if "trajectory_xyz" in payload:
        arrays["trajectory_xyz"] = np.asarray(payload["trajectory_xyz"], dtype=np.float32)
    if "action_sequence" in payload:
        arrays["action_sequence"] = np.asarray(payload["action_sequence"], dtype=np.float32)
    if "x0_descriptor_snapshots" in payload:
        arrays["x0_descriptor_snapshots"] = np.asarray(
            payload["x0_descriptor_snapshots"], dtype=np.float32
        )
    if "x0_temporal_corrections" in payload:
        arrays["x0_temporal_corrections"] = np.asarray(
            payload["x0_temporal_corrections"], dtype=np.float32
        )
    if "denoising_snapshot_steps" in payload:
        arrays["denoising_snapshot_steps"] = np.asarray(
            payload["denoising_snapshot_steps"], dtype=np.int16
        )
    if "prediction_loss_per_offset" in payload:
        arrays["prediction_loss_per_offset"] = np.asarray(
            payload["prediction_loss_per_offset"], dtype=np.float32
        )
    if "prediction_loss_per_offset_no_gripper" in payload:
        arrays["prediction_loss_per_offset_no_gripper"] = np.asarray(
            payload["prediction_loss_per_offset_no_gripper"], dtype=np.float32
        )
    if "prediction_loss_by_timestep" in payload:
        arrays["prediction_loss_by_timestep"] = np.asarray(
            payload["prediction_loss_by_timestep"], dtype=np.float32
        )
    if "prediction_loss_timesteps" in payload:
        arrays["prediction_loss_timesteps"] = np.asarray(
            payload["prediction_loss_timesteps"], dtype=np.int16
        )
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _prefix_probe_payload(
    payload: dict[str, np.ndarray], probe_count: int
) -> dict[str, np.ndarray]:
    """Take GT plus the first ``probe_count`` members of a larger probe cloud.

    ``ActionPCA.sample_directions`` is seeded identically for every count, so
    its first N rows are identical to a standalone N-direction draw.  Keeping
    sample zero (the demonstration) and slicing only the probe axis therefore
    gives exactly the same cloud as a separate smaller DP forward pass.
    """
    descriptors = np.asarray(payload["descriptors"])
    required = int(probe_count) + 1
    if descriptors.ndim < 2 or descriptors.shape[1] < required:
        raise ValueError(
            "Cannot derive prefix probe cloud: "
            f"need {required} GT+probe samples, got shape={descriptors.shape}."
        )
    return {
        "replan_ts": np.asarray(payload["replan_ts"]).copy(),
        "descriptors": descriptors[:, :required].copy(),
        "n_frames": np.asarray(payload["n_frames"]).copy(),
    }


_CLUSTER_SCORE_KEYS = (
    "cosine",
    "magnitude_gated_cosine",
    "covariance_gated_cosine",
    "max_covariance_gated_cosine",
    "l2",
    "hybrid",
    "selected_k",
    "center_norm",
    "bic_k1",
    "bic_best_multi",
    "delta_bic",
    "bic_multi_probability",
)
_OPTIONAL_CLUSTER_SCORE_KEYS = (
    "within_normalized_l2",
    "covariance_gated_angular_chord",
)
_ALL_CLUSTER_SCORE_KEYS = _CLUSTER_SCORE_KEYS + _OPTIONAL_CLUSTER_SCORE_KEYS


def _empty_cluster_scores(length: int) -> dict[str, np.ndarray]:
    """Neutral GMM fields for metrics that do not use sampled action clouds."""
    scores = {
        key: np.zeros(int(length), dtype=np.float32) for key in _ALL_CLUSTER_SCORE_KEYS
    }
    scores["selected_k"] = np.ones(int(length), dtype=np.float32)
    return scores


def _component_covariance_matrices(model, keep: np.ndarray) -> np.ndarray:
    """Return retained GMM component covariances as full matrices.

    sklearn stores ``covariances_`` in four different layouts depending on
    ``covariance_type``. Converting them here keeps the directional-confidence
    score independent of that implementation detail.
    """
    dimension = int(model.means_.shape[1])
    covariance_type = str(model.covariance_type)
    covariances = np.asarray(model.covariances_, dtype=np.float64)
    if covariance_type == "full":
        return covariances[keep]
    if covariance_type == "tied":
        return np.repeat(covariances[None], len(keep), axis=0)
    if covariance_type == "diag":
        return np.stack([np.diag(row) for row in covariances[keep]], axis=0)
    if covariance_type == "spherical":
        identity = np.eye(dimension, dtype=np.float64)
        return np.stack([identity * value for value in covariances[keep]], axis=0)
    raise ValueError(f"Unsupported GMM covariance type: {covariance_type}")


def _directional_reliability(mean: np.ndarray, covariance: np.ndarray) -> float:
    """Parameter-free confidence that a component has a stable direction.

    Signal is squared mean motion. Noise is covariance energy orthogonal to
    that mean; variance along the direction changes speed but does not make
    the direction itself ambiguous. Their ratio is dimensionless and remains
    unchanged under a common rescaling of the descriptor.
    """
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    signal = float(np.dot(mean, mean))
    if signal <= np.finfo(np.float64).eps:
        return 0.0
    direction = mean / np.sqrt(signal)
    total_variance = float(np.trace(covariance))
    parallel_variance = float(direction @ covariance @ direction)
    orthogonal_variance = max(0.0, total_variance - parallel_variance)
    denominator = signal + orthogonal_variance
    return float(signal / denominator) if denominator > 0.0 else 0.0


def _angular_chord_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Physical action-space separation caused only by direction change.

    The law of cosines decomposes squared Euclidean center distance into a
    radial term and an angular term::

        ||a - b||^2 = (||a|| - ||b||)^2
                      + 2 ||a|| ||b|| (1 - cos(theta)).

    Returning the square root of the second term keeps the directional signal
    used by cosine while preventing tiny actions from producing a large score
    solely because their angle is numerically unstable.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    norm_product = left_norm * right_norm
    if norm_product <= np.finfo(np.float64).eps:
        return 0.0
    cosine = float(np.dot(left, right) / norm_product)
    angular_distance = max(0.0, 1.0 - float(np.clip(cosine, -1.0, 1.0)))
    return float(np.sqrt(2.0 * norm_product * angular_distance))


def _retained_component_indices(
    weights: np.ndarray, n_samples: int, cfg: dict
) -> np.ndarray:
    """Select components by minimum effective support.

    For a fitted GMM, ``n_samples * weight[k]`` is component ``k``'s soft
    effective sample count (the sum of its posterior responsibilities).  This
    makes a support threshold retain the same meaning when the probe count
    changes.  ``min_weight`` remains as a legacy fallback for old configs.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    if "min_effective_samples" in cfg:
        minimum = float(cfg["min_effective_samples"])
        effective_support = float(n_samples) * weights
        return np.flatnonzero(effective_support + 1e-12 >= minimum)
    return np.flatnonzero(weights + 1e-12 >= float(cfg.get("min_weight", 0.0)))


def _cluster_cache_load(path: Path, metadata: dict) -> dict[str, np.ndarray] | None:
    """Load GMM-derived curves when both the raw cloud and recipe still match."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["metadata_json"]) != json.dumps(metadata, sort_keys=True):
                return None
            scores = {key: data[key].copy() for key in _CLUSTER_SCORE_KEYS}
            scores.update(
                {
                    key: data[key].copy()
                    for key in _OPTIONAL_CLUSTER_SCORE_KEYS
                    if key in data
                }
            )
            return scores
    except (KeyError, OSError, ValueError):
        return None


def _cluster_cache_save(path: Path, scores: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        **{
            key: np.asarray(scores[key])
            for key in _ALL_CLUSTER_SCORE_KEYS
            if key in scores
        },
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    tmp.replace(path)


def _resolve_pca(
    *,
    variant: dict,
    skillset_dir: Path,
    output_dir: Path,
    cache_source_dir: Path | None,
    dataset_dir: Path,
    policy,
    normalizer,
    manifest: dict,
    variance: float,
    stride: int,
):
    from action_manifold import (
        ACTION_MODE_DATASET,
        PCA_SCALE_NONE,
        PCA_SCALE_STD,
        ActionPCA,
        get_or_fit_action_pca,
        resolve_indices,
    )
    from build_skill_dataset import (
        _fit_dataset_action_pca,
        _fit_dataset_trajectory_pca,
    )

    mode = str(variant.get("mode", "manifest"))
    representation = str(variant.get("pca_representation", "mean"))
    if representation not in {"mean", "trajectory"}:
        raise ValueError(
            f"Unknown pca_representation {representation!r}; expected mean|trajectory."
        )
    action_dim = int(policy.config.action_feature.shape[0])
    manifest_gripper = tuple(int(v) for v in manifest.get("action", {}).get("gripper_indices", [-1]))
    gripper_indices = resolve_indices(manifest_gripper, action_dim)
    if mode == "manifest" and representation == "mean":
        path = skillset_dir / "action_probe_pca.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Manifest probe PCA is missing: {path}")
        pca = ActionPCA.load(path)
        excluded = resolve_indices(manifest.get("probe", {}).get("exclude_indices", []), action_dim)
        indices = tuple(i for i in range(action_dim) if i not in excluded)
        return pca, indices, path
    if mode not in {"manifest", "std_without_gripper", "without_gripper", "std"}:
        raise ValueError(f"Unknown probe mode: {mode}")
    if mode == "manifest":
        excluded = resolve_indices(
            manifest.get("probe", {}).get("exclude_indices", []), action_dim
        )
        indices = tuple(i for i in range(action_dim) if i not in excluded)
        scale_mode = str(variant.get("pca_scale_mode", PCA_SCALE_STD))
    else:
        indices = (
            tuple(range(action_dim))
            if mode == "std"
            else tuple(i for i in range(action_dim) if i not in gripper_indices)
        )
        scale_mode = PCA_SCALE_STD if mode in {"std", "std_without_gripper"} else PCA_SCALE_NONE
    if not indices:
        raise ValueError(f"Probe mode {mode} excludes all action dimensions.")
    action_mode = str(manifest.get("action", {}).get("mode", ACTION_MODE_DATASET))
    if action_mode != ACTION_MODE_DATASET:
        raise ValueError("The standalone ablation currently supports dataset action mode only.")
    delta_indices = tuple(int(v) for v in policy.config.action_delta_indices)
    if getattr(policy.config, "history_conditioning", "state") == "action":
        # The leading n_obs_steps entries are causal conditioning inputs, not
        # the future action trajectory represented by the probe PCA.
        delta_indices = delta_indices[int(policy.config.n_obs_steps) :]
    descriptor_start = int(policy.config.action_execution_start_index)
    descriptor_horizon = int(policy.config.action_prediction_horizon)
    local_variance = float(variant.get("pca_variance", variance))
    local_stride = int(variant.get("pca_stride", stride))
    metadata = {
        "schema": 2 if representation == "trajectory" else 1,
        "dataset": dataset_dir.name,
        "policy": str(Path(manifest["policy_path"]).name),
        "action_dim": action_dim,
        "action_indices": list(indices),
        "action_delta_indices": list(delta_indices),
        "descriptor_start": descriptor_start,
        "descriptor_horizon": descriptor_horizon,
        "variance": local_variance,
        "stride": local_stride,
        "scale_mode": scale_mode,
        "normalizer_mode": normalizer.mode,
    }
    if representation == "trajectory":
        metadata["representation"] = representation
    path = output_dir / "pca" / (
        f"trajectory_{_slug(mode)}.npz"
        if representation == "trajectory"
        else f"{_slug(mode)}.npz"
    )
    if not path.is_file() and cache_source_dir is not None:
        source_path = cache_source_dir / "pca" / path.name
        if source_path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, path)
    pca = get_or_fit_action_pca(
        path,
        metadata,
        lambda: (
            _fit_dataset_trajectory_pca
            if representation == "trajectory"
            else _fit_dataset_action_pca
        )(
            dataset_dir=dataset_dir,
            action_delta_indices=delta_indices,
            descriptor_start=descriptor_start,
            descriptor_horizon=descriptor_horizon,
            stride=local_stride,
            action_dim=action_dim,
            action_indices=indices,
            action_mode=action_mode,
            rel_mask=None,
            normalizer=normalizer,
            variance_threshold=local_variance,
            scale_mode=scale_mode,
            metadata=metadata,
        ),
    )
    return pca, indices, path


def _probe_episode(
    *,
    policy,
    preprocessor,
    ep_df,
    pca,
    directions,
    variant,
    manifest,
    normalizer,
    indices,
    descriptor_pca,
    descriptor_indices,
) -> dict[str, np.ndarray]:
    from action_manifold import PROBE_PCA_ACTION, resolve_indices
    from skill_divider import run_vf_analysis

    action_dim = int(policy.config.action_feature.shape[0])
    gripper = resolve_indices(manifest.get("action", {}).get("gripper_indices", [-1]), action_dim)
    replan_ts, descriptors, _gt, _cos, _l2, _means, _mse = run_vf_analysis(
        policy,
        preprocessor,
        ep_df,
        {},
        [],
        int(variant.get("eval_at_step", manifest["detector"]["eval_at_step"])),
        int(variant.get("replan_interval", manifest["detector"]["replan_interval"])),
        n_gmm_components=int(manifest["detector"]["n_gmm_components"]),
        probe_type=PROBE_PCA_ACTION,
        action_pca=pca,
        probe_directions=directions,
        probe_alpha=float(variant.get("alpha", 0.1)),
        action_normalizer=normalizer,
        action_mode=str(manifest["action"].get("mode", "dataset")),
        gripper_mode=str(manifest["action"].get("gripper_mode", "continuous")),
        gripper_indices=gripper,
        gripper_values=tuple(float(v) for v in manifest["action"].get("gripper_values", [-1, 1])),
        gripper_threshold=float(manifest["action"].get("gripper_threshold", 0.0)),
        probe_action_indices=indices,
        # Grounding belongs to the selected checkpoint, not to the skillset
        # manifest. This matters when policy_checkpoint_override swaps a
        # state-conditioned checkpoint for an action-history checkpoint.
        proprio_grounding=str(getattr(policy.config, "proprio_grounding", "none")),
        denoise_steps=int(variant.get("denoise_steps", 1)),
        denoise_output=str(variant.get("denoise_output", "prev_sample")),
        descriptor_temporal_bins=int(variant.get("descriptor_temporal_bins", 1)),
        descriptor_pca_components=(
            None
            if variant.get("descriptor_pca_components") is None
            else int(variant["descriptor_pca_components"])
        ),
        descriptor_include_endpoint=bool(variant.get("descriptor_include_endpoint", False)),
        probe_generation=str(variant.get("probe_generation", "pca_offset")),
        probe_directions_are_offsets=(
            str(variant.get("probe_generation", "pca_offset")) == "pca_sigma"
        ),
        scheduler_gaussian_include_mean=bool(
            variant.get("gaussian_include_mean", True)
        ),
        descriptor_pca=descriptor_pca,
        descriptor_action_indices=descriptor_indices,
    )
    return {
        "replan_ts": np.asarray(replan_ts, dtype=np.int64),
        "descriptors": np.asarray(descriptors, dtype=np.float32),
        "n_frames": np.array(len(ep_df), dtype=np.int64),
    }


def _rollout_episode(
    *,
    policy,
    preprocessor,
    ep_df,
    pca,
    normalizer,
    indices,
    metric_action_indices,
    samples: int,
    batch_size: int,
    seed: int,
    replan_interval: int,
    descriptor_temporal_bins: int = 1,
    descriptor_pca_components: int | None = None,
    descriptor_include_endpoint: bool = False,
    common_noise: bool = False,
    x0_snapshot_steps: tuple[int, ...] = (),
    x0_temporal_alignment: bool = False,
) -> dict[str, np.ndarray]:
    import torch
    from action_manifold import action_plan_descriptors
    from lerobot.datasets.proprio_grounding import ground_state_xyz, normalize_proprio_grounding
    from lerobot.policies.diffusion.modeling_diffusion import OBS_ACTION_HISTORY
    from lerobot.utils.constants import OBS_STATE
    from skill_divider import _valid_replan_anchors

    states = np.stack(ep_df["observation.state"].values).astype(np.float32)
    grounding = normalize_proprio_grounding(getattr(policy.config, "proprio_grounding", "none"))
    if grounding == "episode_start_xyz":
        states = ground_state_xyz(states, states[0, :3])
    processed = []
    for state in states:
        value = preprocessor({OBS_STATE: torch.from_numpy(state).float()})[OBS_STATE]
        processed.append(value[0] if value.ndim == 2 and value.shape[0] == 1 else value)
    state_tensor = torch.stack(processed)
    action_history_conditioning = (
        getattr(policy.config, "history_conditioning", "state") == "action"
    )
    actions = np.stack(ep_df["action"].values).astype(np.float32)
    normalized_actions = normalizer.normalize(actions).astype(np.float32)
    future_horizon = int(policy.config.action_prediction_horizon)
    anchors = _valid_replan_anchors(len(states), future_horizon, int(replan_interval))
    n_obs = int(policy.config.n_obs_steps)
    parameter = next(policy.parameters())
    rng = np.random.default_rng(seed)
    shared_noise = (
        rng.standard_normal(
            (samples, int(policy.config.horizon), int(policy.config.action_feature.shape[0])),
            dtype=np.float32,
        )
        if common_noise
        else None
    )
    clouds = []
    trajectory_xyz_clouds = []
    action_sequence_clouds = []
    x0_snapshot_clouds = []
    x0_temporal_corrections = []
    snapshot_steps = tuple(sorted(set(int(step) for step in x0_snapshot_steps)))
    inference_steps = int(policy.diffusion.num_inference_steps)
    if snapshot_steps and (
        snapshot_steps[0] < 0 or snapshot_steps[-1] > inference_steps
    ):
        raise ValueError(
            f"x0 snapshot steps must be in [0, {inference_steps}], got {snapshot_steps}."
        )
    with torch.inference_mode():
        for anchor in anchors:
            if action_history_conditioning:
                history_value = np.zeros((n_obs, actions.shape[-1]), dtype=np.float32)
                start_index = max(0, anchor - n_obs)
                valid = normalized_actions[start_index:anchor]
                if len(valid):
                    history_value[-len(valid) :] = valid
                condition = torch.from_numpy(history_value).unsqueeze(0).to(parameter.device)
                condition_key = OBS_ACTION_HISTORY
            else:
                history = list(range(max(0, anchor - n_obs + 1), anchor + 1))
                history = [history[0]] * (n_obs - len(history)) + history
                condition = state_tensor[history].unsqueeze(0).to(parameter.device)
                condition_key = OBS_STATE
            global_cond = policy.diffusion._prepare_global_conditioning(
                {condition_key: condition}
            )
            all_noise = (
                shared_noise
                if shared_noise is not None
                else rng.standard_normal(
                    (
                        samples,
                        int(policy.config.horizon),
                        int(policy.config.action_feature.shape[0]),
                    ),
                    dtype=np.float32,
                )
            )
            outputs = []
            snapshot_outputs = {step: [] for step in snapshot_steps}
            for start in range(0, samples, batch_size):
                noise = torch.from_numpy(all_noise[start : start + batch_size]).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                gc = global_cond.expand(len(noise), -1)
                if snapshot_steps:
                    sample = noise
                    scheduler = policy.diffusion.noise_scheduler
                    scheduler.set_timesteps(inference_steps)
                    for completed_steps, timestep in enumerate(scheduler.timesteps):
                        timestep_batch = torch.full(
                            sample.shape[:1],
                            timestep,
                            dtype=torch.long,
                            device=sample.device,
                        )
                        model_output = policy.diffusion.unet(
                            sample, timestep_batch, global_cond=gc
                        )
                        scheduler_output = scheduler.step(
                            model_output, timestep, sample
                        )
                        # x0_hat(k): clean estimate made from the noisy state
                        # after k completed DDIM updates. At k=0 this is the
                        # first model prediction from the initial noise, not
                        # the initial noise itself.
                        if completed_steps in snapshot_outputs:
                            snapshot_outputs[completed_steps].append(
                                scheduler_output.pred_original_sample.float().cpu().numpy()
                            )
                        sample = scheduler_output.prev_sample
                    # After every scheduled update, the final DDIM sample is
                    # itself the clean estimate at step N.
                    if inference_steps in snapshot_outputs:
                        snapshot_outputs[inference_steps].append(
                            sample.float().cpu().numpy()
                        )
                    outputs.append(sample.float().cpu().numpy())
                else:
                    outputs.append(
                        policy.diffusion.conditional_sample(
                            len(noise), global_cond=gc, noise=noise
                        )
                        .float().cpu().numpy()
                    )
            chunks = np.concatenate(outputs, axis=0)
            execution_start = int(policy.config.action_execution_start_index)
            execution_end = execution_start + future_horizon
            # DP predicts normalized delta actions. Convert them back to the
            # dataset action scale, then integrate XYZ deltas into a path
            # relative to the current end-effector position. This preserves
            # physical motion direction without a PCA/GMM coordinate system.
            physical_chunks = normalizer.denormalize(chunks)
            xyz_delta = physical_chunks[:, execution_start:execution_end, :3]
            trajectory_xyz_clouds.append(np.cumsum(xyz_delta, axis=1))
            # Representation-agnostic metric input: retain the policy's own
            # normalized action sequence and exclude only configured gripper
            # channels. Unlike cumulative XYZ this remains valid for Cartesian
            # pose, absolute/relative joints, or delta-joint action spaces.
            action_sequence_clouds.append(
                chunks[:, execution_start:execution_end, list(metric_action_indices)]
            )
            clouds.append(
                action_plan_descriptors(
                    chunks,
                    pca,
                    action_indices=indices,
                    temporal_start=int(policy.config.action_execution_start_index),
                    temporal_length=future_horizon,
                    temporal_bins=descriptor_temporal_bins,
                    pca_components=descriptor_pca_components,
                    include_endpoint=descriptor_include_endpoint,
                )
            )
            if snapshot_steps:
                snapshot_descriptors = []
                temporal_snapshot_descriptors = []
                for step in snapshot_steps:
                    snapshot_chunks = np.concatenate(snapshot_outputs[step], axis=0)
                    snapshot_descriptors.append(
                        action_plan_descriptors(
                            snapshot_chunks,
                            pca,
                            action_indices=indices,
                            temporal_start=execution_start,
                            temporal_length=future_horizon,
                            temporal_bins=descriptor_temporal_bins,
                            pca_components=descriptor_pca_components,
                            include_endpoint=descriptor_include_endpoint,
                        )
                    )
                    if x0_temporal_alignment:
                        selected = snapshot_chunks[
                            :, execution_start:execution_end, list(indices)
                        ]
                        if pca.action_dim != selected.shape[-1]:
                            raise ValueError(
                                "Time-aligned denoising requires a per-action PCA; "
                                f"PCA dim={pca.action_dim}, selected action dim="
                                f"{selected.shape[-1]}."
                            )
                        components = (
                            pca.n_components
                            if descriptor_pca_components is None
                            else int(descriptor_pca_components)
                        )
                        projected = pca.transform(
                            selected.reshape(-1, selected.shape[-1])
                        )[:, :components]
                        temporal_snapshot_descriptors.append(
                            projected.reshape(
                                len(selected), future_horizon, components
                            )
                        )
                # (samples, snapshots, descriptor_dim)
                x0_snapshot_clouds.append(
                    np.stack(snapshot_descriptors, axis=1).astype(np.float32)
                )
                if x0_temporal_alignment:
                    # Preserve where in the predicted plan each denoising
                    # correction occurred.  Storing only whitened correction
                    # magnitudes is much smaller than caching every x0_hat
                    # action chunk and is sufficient for diagonal time
                    # alignment during report generation.
                    temporal_values = np.stack(
                        temporal_snapshot_descriptors, axis=1
                    ).astype(np.float64)
                    temporal_scale = np.sqrt(
                        np.maximum(
                            np.asarray(
                                pca.explained_variance[:components],
                                dtype=np.float64,
                            ),
                            1e-12,
                        )
                    )
                    whitened = temporal_values / temporal_scale[None, None, None, :]
                    # (samples, future offsets, denoising intervals)
                    corrections = np.sqrt(
                        np.mean(np.square(np.diff(whitened, axis=1)), axis=-1)
                    ).transpose(0, 2, 1)
                    x0_temporal_corrections.append(corrections.astype(np.float32))
    result = {
        "replan_ts": np.asarray(anchors, dtype=np.int64),
        "descriptors": np.asarray(clouds, dtype=np.float32),
        "trajectory_xyz": np.asarray(trajectory_xyz_clouds, dtype=np.float32),
        "action_sequence": np.asarray(action_sequence_clouds, dtype=np.float32),
        "n_frames": np.array(len(ep_df), dtype=np.int64),
    }
    if snapshot_steps:
        result["x0_descriptor_snapshots"] = np.asarray(
            x0_snapshot_clouds, dtype=np.float32
        )
        result["denoising_snapshot_steps"] = np.asarray(
            snapshot_steps, dtype=np.int16
        )
        if x0_temporal_alignment:
            result["x0_temporal_corrections"] = np.asarray(
                x0_temporal_corrections, dtype=np.float32
            )
    return result


def _prediction_loss_episode(
    *,
    policy,
    preprocessor,
    ep_df,
    normalizer,
    metric_action_indices: tuple[int, ...],
    noise_timesteps: tuple[int, ...],
    noise_samples_per_timestep: int,
    batch_size: int,
    seed: int,
    replan_interval: int,
    common_noise: bool,
) -> dict[str, np.ndarray]:
    """Evaluate the DP training objective on demonstrated future actions.

    This is the continuous-action diffusion analogue of the paper's
    ``-log p(a_t | history)`` signal.  Instead of sampling one rollout and
    comparing it with the demonstration, it forward-noises the demonstrated
    action chunk at fixed training timesteps and measures the denoiser's
    residual.  Fixed timesteps and common random numbers make adjacent-anchor
    changes reflect the demonstrated trajectory rather than Monte-Carlo draw
    noise.
    """
    import torch
    from skill_divider import _aligned_action_chunk, _valid_replan_anchors

    from lerobot.datasets.proprio_grounding import ground_state_xyz, normalize_proprio_grounding
    from lerobot.policies.diffusion.modeling_diffusion import OBS_ACTION_HISTORY
    from lerobot.utils.constants import OBS_STATE

    if not noise_timesteps:
        raise ValueError("prediction_loss.noise_timesteps must not be empty.")
    if noise_samples_per_timestep < 1 or batch_size < 1:
        raise ValueError("Prediction-loss sample count and batch size must be positive.")

    num_train_timesteps = int(policy.diffusion.noise_scheduler.config.num_train_timesteps)
    if min(noise_timesteps) < 0 or max(noise_timesteps) >= num_train_timesteps:
        raise ValueError(
            "Prediction-loss noise timesteps must be within the training schedule "
            f"[0, {num_train_timesteps - 1}], got {noise_timesteps}."
        )

    states = np.stack(ep_df["observation.state"].values).astype(np.float32)
    actions = np.stack(ep_df["action"].values).astype(np.float32)
    grounding = normalize_proprio_grounding(
        getattr(policy.config, "proprio_grounding", "none")
    )
    if grounding == "episode_start_xyz":
        states = ground_state_xyz(states, states[0, :3])

    processed_states = []
    for state in states:
        processed = preprocessor({OBS_STATE: torch.from_numpy(state).float()})[OBS_STATE]
        if processed.ndim == 2 and processed.shape[0] == 1:
            processed = processed[0]
        processed_states.append(processed)
    state_tensor = torch.stack(processed_states)

    cfg = policy.config
    n_obs = int(cfg.n_obs_steps)
    horizon = int(cfg.horizon)
    future_start = int(cfg.action_execution_start_index)
    future_horizon = int(cfg.action_prediction_horizon)
    action_history_conditioning = getattr(cfg, "history_conditioning", "state") == "action"
    target_delta_indices = [int(value) for value in cfg.action_delta_indices]
    if action_history_conditioning:
        target_delta_indices = target_delta_indices[n_obs:]
    if len(target_delta_indices) != horizon:
        raise ValueError(
            "DP target-index count must equal its diffusion horizon: "
            f"indices={len(target_delta_indices)}, horizon={horizon}."
        )
    if future_start + future_horizon > horizon:
        raise ValueError(
            "DP future scoring slice exceeds its diffusion horizon: "
            f"start={future_start}, length={future_horizon}, horizon={horizon}."
        )

    anchors = _valid_replan_anchors(len(states), future_horizon, int(replan_interval))
    normalized_actions = normalizer.normalize(actions).astype(np.float32)
    parameter = next(policy.parameters())
    trial_timesteps = np.repeat(
        np.asarray(noise_timesteps, dtype=np.int64), int(noise_samples_per_timestep)
    )
    trial_count = len(trial_timesteps)
    rng = np.random.default_rng(int(seed))
    shared_eps = (
        rng.standard_normal(
            (trial_count, horizon, actions.shape[-1]), dtype=np.float32
        )
        if common_noise
        else None
    )
    no_gripper = np.asarray(metric_action_indices, dtype=np.int64)
    if np.any(no_gripper < 0) or np.any(no_gripper >= actions.shape[-1]):
        raise ValueError(
            f"Invalid non-gripper action indices {metric_action_indices} for dim {actions.shape[-1]}."
        )

    per_offset = []
    per_offset_no_gripper = []
    by_timestep = []
    with torch.inference_mode():
        for anchor in anchors:
            if action_history_conditioning:
                history = np.zeros((n_obs, actions.shape[-1]), dtype=np.float32)
                history_start = max(0, anchor - n_obs)
                valid = normalized_actions[history_start:anchor]
                if len(valid):
                    history[-len(valid) :] = valid
                condition = torch.from_numpy(history).unsqueeze(0).to(parameter.device)
                condition_key = OBS_ACTION_HISTORY
            else:
                history_indices = list(range(max(0, anchor - n_obs + 1), anchor + 1))
                history_indices = [history_indices[0]] * (n_obs - len(history_indices)) + history_indices
                condition = state_tensor[history_indices].unsqueeze(0).to(parameter.device)
                condition_key = OBS_STATE
            global_cond = policy.diffusion._prepare_global_conditioning(
                {condition_key: condition}
            )

            target = _aligned_action_chunk(
                normalized_actions, anchor, target_delta_indices
            ).astype(np.float32)
            eps_values = (
                shared_eps
                if shared_eps is not None
                else rng.standard_normal(
                    (trial_count, *target.shape), dtype=np.float32
                )
            )
            squared_error_batches = []
            for start in range(0, trial_count, batch_size):
                end = min(trial_count, start + batch_size)
                clean = torch.from_numpy(target).to(
                    device=parameter.device, dtype=parameter.dtype
                ).unsqueeze(0).expand(end - start, -1, -1)
                eps = torch.from_numpy(eps_values[start:end]).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                timesteps = torch.from_numpy(trial_timesteps[start:end]).to(
                    device=parameter.device, dtype=torch.long
                )
                noisy = policy.diffusion.noise_scheduler.add_noise(clean, eps, timesteps)
                prediction = policy.diffusion.unet(
                    noisy,
                    timesteps,
                    global_cond=global_cond.expand(end - start, -1),
                )
                if cfg.prediction_type == "epsilon":
                    expected = eps
                elif cfg.prediction_type == "sample":
                    expected = clean
                else:
                    raise ValueError(
                        f"Unsupported DP prediction type {cfg.prediction_type!r}."
                    )
                squared_error_batches.append(
                    torch.square(prediction - expected).float().cpu().numpy()
                )
            squared_error = np.concatenate(squared_error_batches, axis=0)
            future_error = squared_error[
                :, future_start : future_start + future_horizon
            ]
            per_offset.append(np.mean(future_error, axis=(0, 2)))
            per_offset_no_gripper.append(
                np.mean(future_error[..., no_gripper], axis=(0, 2))
            )
            by_timestep.append(
                future_error.reshape(
                    len(noise_timesteps), noise_samples_per_timestep, future_horizon, -1
                ).mean(axis=(1, 2, 3))
            )

    # Descriptors are placeholders required by the shared report/cache schema;
    # prediction-loss experiments bypass GMM fitting entirely.
    descriptors = np.zeros((len(anchors), 2, 1), dtype=np.float32)
    return {
        "replan_ts": np.asarray(anchors, dtype=np.int64),
        "descriptors": descriptors,
        "n_frames": np.array(len(ep_df), dtype=np.int64),
        "prediction_loss_per_offset": np.asarray(per_offset, dtype=np.float32),
        "prediction_loss_per_offset_no_gripper": np.asarray(
            per_offset_no_gripper, dtype=np.float32
        ),
        "prediction_loss_by_timestep": np.asarray(by_timestep, dtype=np.float32),
        "prediction_loss_timesteps": np.asarray(noise_timesteps, dtype=np.int16),
    }


def _aligned_prediction_loss(
    raw: dict,
    *,
    exclude_gripper: bool = False,
    aggregation: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each chunk offset to ``anchor + offset`` and aggregate overlaps."""
    key = (
        "prediction_loss_per_offset_no_gripper"
        if exclude_gripper
        else "prediction_loss_per_offset"
    )
    if key not in raw:
        raise ValueError(f"Prediction-loss cache is missing {key!r}.")
    losses = np.asarray(raw[key], dtype=np.float64)
    anchors = np.asarray(raw["replan_ts"], dtype=np.int64)
    n_frames = int(np.asarray(raw["n_frames"]).item())
    if losses.ndim != 2 or losses.shape[0] != len(anchors):
        raise ValueError(
            f"Expected {key} shaped (anchors, future), got {losses.shape}."
        )
    if aggregation not in {"mean", "median"}:
        raise ValueError(
            f"prediction_loss alignment aggregation must be mean|median, got {aggregation!r}."
        )
    contributions: list[list[float]] = [[] for _ in range(n_frames)]
    for anchor, row in zip(anchors, losses, strict=True):
        for offset, value in enumerate(row):
            target = int(anchor) + int(offset)
            if target < n_frames and np.isfinite(value):
                contributions[target].append(float(value))
    valid_ts = np.asarray(
        [index for index, values in enumerate(contributions) if values], dtype=np.int64
    )
    reducer = np.mean if aggregation == "mean" else np.median
    values = np.asarray(
        [reducer(contributions[index]) for index in valid_ts], dtype=np.float32
    )
    support = np.asarray(
        [len(contributions[index]) for index in valid_ts], dtype=np.int32
    )
    return valid_ts, values, support


def _input_probe_descriptors(
    pca,
    directions: np.ndarray,
    alpha: float,
    variant: dict,
) -> np.ndarray:
    """Exact descriptor geometry injected by ``make_pca_action_probes``.

    The probe implementation scales unit directions by ``sqrt(action_dim)``.
    Mirroring that factor here is necessary for a meaningful input/output BIC
    gain, especially once the descriptor retains several temporal bins.
    """
    components = variant.get("descriptor_pca_components")
    components = pca.n_components if components is None else int(components)
    center = np.asarray(pca.mean, dtype=np.float32)
    offsets = float(alpha) * np.sqrt(pca.action_dim) * np.asarray(directions, dtype=np.float32)
    projected_center = pca.transform(center[None])[:, :components]
    projected_probes = pca.transform(center[None] + offsets)[:, :components]
    if str(variant.get("pca_representation", "mean")) == "trajectory":
        return np.concatenate([projected_center, projected_probes], axis=0)
    bins = int(variant.get("descriptor_temporal_bins", 1))
    repeats = bins + int(bool(variant.get("descriptor_include_endpoint", False)))
    return np.concatenate(
        [
            np.tile(projected_center, (1, repeats)),
            np.tile(projected_probes, (1, repeats)),
        ],
        axis=0,
    )


def _descriptor_whitening_scale(pca, variant: dict) -> np.ndarray:
    """Per-coordinate standard deviation for a cached PCA descriptor.

    ``ActionPCA.transform`` rotates into principal coordinates but deliberately
    does not divide by their eigenvalue scale.  Tiling ``sqrt(eigenvalue)`` in
    the same order as ``action_plan_descriptors`` makes a distance comparable
    across high- and low-variance action directions and temporal bins.
    """
    components = variant.get("descriptor_pca_components")
    components = pca.n_components if components is None else int(components)
    if not 1 <= components <= pca.n_components:
        raise ValueError(
            "descriptor_pca_components must be in "
            f"[1, {pca.n_components}], got {components}."
        )
    base = np.sqrt(
        np.maximum(
            np.asarray(pca.explained_variance[:components], dtype=np.float64),
            1e-12,
        )
    )
    if str(
        variant.get(
            "descriptor_pca_representation",
            variant.get("pca_representation", "mean"),
        )
    ) == "trajectory":
        return base.astype(np.float32)
    repeats = int(variant.get("descriptor_temporal_bins", 1)) + int(
        bool(variant.get("descriptor_include_endpoint", False))
    )
    return np.tile(base, repeats).astype(np.float32)


def _whitened_descriptor_spread(
    clouds: np.ndarray, whitening_scale: np.ndarray
) -> np.ndarray:
    """RMS pairwise distance after data-covariance whitening.

    The result is normalized by descriptor dimensionality, so its scale does
    not grow merely because more PCA coordinates or temporal bins are kept.
    It uses every sample directly and therefore has no GMM/K dependency.
    """
    values = np.asarray(clouds, dtype=np.float64)
    single_cloud = values.ndim == 2
    if single_cloud:
        values = values[None]
    if values.ndim != 3:
        raise ValueError(f"Expected descriptor clouds (T,N,D), got {values.shape}.")
    scale = np.asarray(whitening_scale, dtype=np.float64).reshape(-1)
    if values.shape[-1] != len(scale):
        raise ValueError(
            "Whitening scale does not match descriptor dimension: "
            f"cloud={values.shape[-1]}, scale={len(scale)}."
        )
    sample_count = int(values.shape[1])
    if sample_count < 2:
        raise ValueError("Whitened spread requires at least two samples.")
    whitened = values / np.maximum(scale[None, None], 1e-12)
    centered = whitened - whitened.mean(axis=1, keepdims=True)
    # Mean over unordered i<j pairs without materializing an N x N matrix:
    # sum_{i<j} ||x_i-x_j||^2 = N * sum_i ||x_i-mean(x)||^2.
    pairwise_mean_sq = (
        2.0
        * sample_count
        / (sample_count - 1)
        * np.mean(np.sum(centered * centered, axis=-1), axis=1)
    )
    result = np.sqrt(np.maximum(pairwise_mean_sq / values.shape[-1], 0.0))
    result = result.astype(np.float32)
    return result[0] if single_cloud else result


def _denoising_convergence_scores(
    raw: dict,
    whitening_scale: np.ndarray,
    *,
    convergence_threshold: float | None = None,
    relative_convergence_threshold: float | None = None,
    sample_aggregation: str = "median",
    convergence_threshold_mode: str = "fixed",
    terminal_mask_frames: int = 0,
) -> dict[str, np.ndarray]:
    """Measure how late each sampled clean-action estimate stabilizes.

    Distances are computed between consecutive saved ``x0_hat`` descriptors,
    whitened by the dataset action PCA and normalized per descriptor dimension.
    They are then mean- or median-aggregated across initial-noise samples. This
    deliberately compares clean estimates rather than noisy scheduler states
    ``x_t``.
    """
    if "x0_descriptor_snapshots" not in raw or "denoising_snapshot_steps" not in raw:
        raise KeyError(
            "Denoising-convergence metrics require a direct-rollout cache with "
            "x0 snapshots. Enable rollout.x0_snapshot_steps and rerun once."
        )
    values = np.asarray(raw["x0_descriptor_snapshots"], dtype=np.float64)
    steps = np.asarray(raw["denoising_snapshot_steps"], dtype=np.int64).reshape(-1)
    if values.ndim != 4:
        raise ValueError(
            "Expected x0_descriptor_snapshots (T,N,S,D), "
            f"got {values.shape}."
        )
    if values.shape[2] != len(steps) or len(steps) < 2:
        raise ValueError(
            f"Snapshot tensor/step mismatch: shape={values.shape}, steps={steps.tolist()}."
        )
    if np.any(np.diff(steps) <= 0):
        raise ValueError(f"Snapshot steps must be strictly increasing: {steps.tolist()}.")
    scale = np.asarray(whitening_scale, dtype=np.float64).reshape(-1)
    if values.shape[-1] != len(scale):
        raise ValueError(
            "Whitening scale does not match x0 descriptor dimension: "
            f"snapshots={values.shape[-1]}, scale={len(scale)}."
        )

    aggregation = str(sample_aggregation)
    if aggregation not in {"mean", "median"}:
        raise ValueError(
            "sample_aggregation must be mean|median, "
            f"got {sample_aggregation!r}."
        )
    aggregate = np.mean if aggregation == "mean" else np.median

    whitened = values / np.maximum(scale[None, None, None, :], 1e-12)
    # (anchors, samples, intervals): per-dimension RMS x0_hat correction.
    per_sample = np.sqrt(np.mean(np.square(np.diff(whitened, axis=2)), axis=-1))
    intervals = aggregate(per_sample, axis=1)
    per_sample_total = per_sample.sum(axis=-1)
    late_start = per_sample.shape[-1] // 2
    per_sample_late = per_sample[..., late_start:].sum(axis=-1)

    midpoints = (steps[:-1] + steps[1:]) / 2.0
    span = float(steps[-1] - steps[0])
    positions = (
        np.zeros_like(midpoints, dtype=np.float64)
        if span <= 0.0
        else (midpoints - float(steps[0])) / span
    )
    denominator = np.maximum(per_sample_total, 1e-12)
    per_sample_delay = np.sum(per_sample * positions[None, None, :], axis=-1) / denominator
    per_sample_delay = np.where(per_sample_total > 1e-12, per_sample_delay, 0.0)
    per_sample_correction_step = (
        np.sum(per_sample * midpoints[None, None, :], axis=-1) / denominator
    )
    per_sample_correction_step = np.where(
        per_sample_total > 1e-12,
        per_sample_correction_step,
        float(steps[0]),
    )

    # Discrete, threshold-free knee diagnostic. If the interval correction
    # never decreases, the estimate has not visibly stabilized by the final
    # snapshot. Otherwise report the snapshot after which the largest absolute
    # decrease in correction magnitude occurs.
    interval_drop = intervals[:, :-1] - intervals[:, 1:]
    knee_index = np.argmax(interval_drop, axis=1)
    knee_step = steps[1:-1][knee_index].astype(np.float64)
    knee_step[np.max(interval_drop, axis=1) <= 0.0] = float(steps[-1])

    result = {
        "denoising_total_drift": aggregate(per_sample_total, axis=1).astype(np.float32),
        "denoising_late_drift": aggregate(per_sample_late, axis=1).astype(np.float32),
        "denoising_correction_delay": aggregate(per_sample_delay, axis=1).astype(np.float32),
        "denoising_correction_step": aggregate(per_sample_correction_step, axis=1).astype(
            np.float32
        ),
        "denoising_knee_step": knee_step.astype(np.float32),
    }
    for index, (left, right) in enumerate(zip(steps[:-1], steps[1:], strict=True)):
        result[f"denoising_drift_{int(left)}_{int(right)}"] = intervals[:, index].astype(
            np.float32
        )
    threshold_mode = str(convergence_threshold_mode)
    if threshold_mode not in {"fixed", "episode_geometric_mean"}:
        raise ValueError(
            "convergence_threshold_mode must be fixed|episode_geometric_mean, "
            f"got {convergence_threshold_mode!r}."
        )
    threshold: float | None = None
    if threshold_mode == "episode_geometric_mean":
        if convergence_threshold is not None:
            raise ValueError(
                "episode_geometric_mean must not also set a fixed convergence_threshold."
            )
        replan_ts = np.asarray(raw.get("replan_ts"), dtype=np.int64).reshape(-1)
        if len(replan_ts) != len(intervals):
            raise ValueError(
                "episode_geometric_mean requires replan_ts aligned with x0 snapshots."
            )
        terminal_mask_frames = int(terminal_mask_frames)
        if terminal_mask_frames < 0:
            raise ValueError("terminal_mask_frames must be non-negative.")
        cutoff = int(replan_ts[-1]) - terminal_mask_frames
        valid_count = int(np.searchsorted(replan_ts, cutoff, side="right"))
        if valid_count < 1:
            raise ValueError(
                f"terminal_mask_frames={terminal_mask_frames} leaves no values "
                "for the episode geometric mean."
            )
        threshold_values = intervals[:valid_count]
        positive = threshold_values[
            np.isfinite(threshold_values) & (threshold_values > 0.0)
        ]
        if not len(positive):
            raise ValueError(
                "episode_geometric_mean requires at least one positive finite drift."
            )
        threshold = float(np.exp(np.mean(np.log(positive))))
    elif convergence_threshold is not None:
        threshold = float(convergence_threshold)

    if threshold is not None:
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "denoising_convergence_threshold must be finite and positive, "
                f"got {threshold}."
            )
        below = intervals <= threshold
        first_below = np.argmax(below, axis=1)
        # argmax returns zero when no interval passes. Distinguish that case
        # and conservatively mark it as unconverged until the final step.
        threshold_step = steps[1:][first_below].astype(np.float64)
        threshold_step[~np.any(below, axis=1)] = float(steps[-1])
        result["denoising_threshold_step"] = threshold_step.astype(np.float32)
        result["denoising_convergence_threshold_value"] = np.array(
            threshold, dtype=np.float32
        )
    if relative_convergence_threshold is not None:
        ratio = float(relative_convergence_threshold)
        if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
            raise ValueError(
                "denoising_relative_convergence_threshold must be finite and "
                f"in (0, 1), got {relative_convergence_threshold}."
            )
        # Each anchor gets its own convergence bar, expressed as a fraction of
        # that anchor's initial 0->first-snapshot correction magnitude.
        relative_bar = intervals[:, :1] * ratio
        below = intervals <= relative_bar
        first_below = np.argmax(below, axis=1)
        threshold_step = steps[1:][first_below].astype(np.float64)
        threshold_step[~np.any(below, axis=1)] = float(steps[-1])
        result["denoising_relative_threshold_step"] = threshold_step.astype(
            np.float32
        )
    return result


def _aligned_denoising_convergence_scores(
    raw: dict,
    *,
    convergence_threshold: float | None = None,
    relative_convergence_threshold: float | None = None,
    sample_aggregation: str = "median",
    convergence_threshold_mode: str = "fixed",
    terminal_mask_frames: int = 0,
) -> dict[str, np.ndarray]:
    """Align per-offset denoising corrections to their predicted frame.

    A chunk-level score assigns difficulty anywhere in a future action plan to
    the anchor where that plan was sampled.  Consequently, one transition is
    repeated at every anchor whose horizon overlaps it.  The rollout cache
    stores a correction magnitude for each future offset instead.  Here,
    ``anchor + offset`` contributions are gathered onto the same absolute
    episode frame before measuring the convergence step.
    """
    if "x0_temporal_corrections" not in raw or "denoising_snapshot_steps" not in raw:
        raise KeyError(
            "Time-aligned denoising requires rollout caches with per-offset "
            "corrections. Enable rollout.x0_temporal_alignment and rerun once."
        )
    corrections = np.asarray(raw["x0_temporal_corrections"], dtype=np.float64)
    steps = np.asarray(raw["denoising_snapshot_steps"], dtype=np.int64).reshape(-1)
    anchors = np.asarray(raw.get("replan_ts"), dtype=np.int64).reshape(-1)
    if corrections.ndim != 4:
        raise ValueError(
            "Expected x0_temporal_corrections "
            f"(anchors,samples,horizon,intervals), got {corrections.shape}."
        )
    if corrections.shape[0] != len(anchors):
        raise ValueError(
            "Temporal corrections/replan anchor mismatch: "
            f"{corrections.shape[0]} != {len(anchors)}."
        )
    if corrections.shape[-1] != len(steps) - 1 or len(steps) < 2:
        raise ValueError(
            "Temporal correction intervals do not match snapshot steps: "
            f"shape={corrections.shape}, steps={steps.tolist()}."
        )
    if np.any(np.diff(steps) <= 0):
        raise ValueError(f"Snapshot steps must be strictly increasing: {steps.tolist()}.")
    aggregation = str(sample_aggregation)
    if aggregation not in {"mean", "median"}:
        raise ValueError(
            "sample_aggregation must be mean|median, "
            f"got {sample_aggregation!r}."
        )
    aggregate = np.mean if aggregation == "mean" else np.median

    horizon = int(corrections.shape[2])
    target_grid = anchors[:, None] + np.arange(horizon, dtype=np.int64)[None, :]
    n_frames = int(np.asarray(raw["n_frames"]).item())
    valid = (target_grid >= 0) & (target_grid < n_frames)
    target_ts = np.unique(target_grid[valid])
    aligned_intervals = []
    support = []
    for target in target_ts:
        anchor_indices, offsets = np.nonzero(target_grid == target)
        keep = valid[anchor_indices, offsets]
        anchor_indices = anchor_indices[keep]
        offsets = offsets[keep]
        # (overlapping origins, noise samples, denoising intervals). Treat
        # every independently sampled prediction of the same target frame as
        # evidence for that frame, then robustly aggregate them together.
        values = np.stack(
            [
                corrections[anchor_index, :, offset, :]
                for anchor_index, offset in zip(anchor_indices, offsets, strict=True)
            ],
            axis=0,
        )
        aligned_intervals.append(aggregate(values, axis=(0, 1)))
        support.append(len(anchor_indices))
    intervals = np.asarray(aligned_intervals, dtype=np.float64)
    support_array = np.asarray(support, dtype=np.int16)

    midpoints = (steps[:-1] + steps[1:]) / 2.0
    total = intervals.sum(axis=-1)
    delay = np.sum(intervals * midpoints[None, :], axis=-1) / np.maximum(total, 1e-12)
    delay = np.where(total > 1e-12, delay, float(steps[0]))
    result = {
        "denoising_aligned_ts": target_ts.astype(np.int64),
        "denoising_aligned_support": support_array,
        "denoising_aligned_total_drift": total.astype(np.float32),
        "denoising_aligned_correction_step": delay.astype(np.float32),
    }
    for index, (left, right) in enumerate(zip(steps[:-1], steps[1:], strict=True)):
        result[
            f"denoising_aligned_drift_{int(left)}_{int(right)}"
        ] = intervals[:, index].astype(np.float32)

    threshold_mode = str(convergence_threshold_mode)
    if threshold_mode not in {"fixed", "episode_geometric_mean"}:
        raise ValueError(
            "convergence_threshold_mode must be fixed|episode_geometric_mean, "
            f"got {convergence_threshold_mode!r}."
        )
    threshold: float | None = None
    if threshold_mode == "episode_geometric_mean":
        if convergence_threshold is not None:
            raise ValueError(
                "episode_geometric_mean must not also set a fixed convergence_threshold."
            )
        terminal_mask_frames = int(terminal_mask_frames)
        if terminal_mask_frames < 0:
            raise ValueError("terminal_mask_frames must be non-negative.")
        cutoff = int(target_ts[-1]) - terminal_mask_frames
        valid_count = int(np.searchsorted(target_ts, cutoff, side="right"))
        if valid_count < 1:
            raise ValueError(
                f"terminal_mask_frames={terminal_mask_frames} leaves no values "
                "for the episode geometric mean."
            )
        positive = intervals[:valid_count]
        positive = positive[np.isfinite(positive) & (positive > 0.0)]
        if not len(positive):
            raise ValueError(
                "episode_geometric_mean requires at least one positive finite drift."
            )
        threshold = float(np.exp(np.mean(np.log(positive))))
    elif convergence_threshold is not None:
        threshold = float(convergence_threshold)

    if threshold is not None:
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "denoising_convergence_threshold must be finite and positive, "
                f"got {threshold}."
            )
        below = intervals <= threshold
        first_below = np.argmax(below, axis=1)
        threshold_step = steps[1:][first_below].astype(np.float64)
        threshold_step[~np.any(below, axis=1)] = float(steps[-1])
        result["denoising_aligned_threshold_step"] = threshold_step.astype(np.float32)
        result["denoising_convergence_threshold_value"] = np.array(
            threshold, dtype=np.float32
        )
    if relative_convergence_threshold is not None:
        ratio = float(relative_convergence_threshold)
        if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
            raise ValueError(
                "denoising_relative_convergence_threshold must be finite and "
                f"in (0, 1), got {relative_convergence_threshold}."
            )
        below = intervals <= intervals[:, :1] * ratio
        first_below = np.argmax(below, axis=1)
        threshold_step = steps[1:][first_below].astype(np.float64)
        threshold_step[~np.any(below, axis=1)] = float(steps[-1])
        result["denoising_aligned_relative_threshold_step"] = threshold_step.astype(
            np.float32
        )
    return result


def _sliced_wasserstein_shift(
    clouds: np.ndarray,
    *,
    projections: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """Adjacent-anchor sliced-Wasserstein distance in descriptor space."""
    values = np.asarray(clouds, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected descriptor clouds (T,N,D), got {values.shape}.")
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((int(projections), values.shape[-1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    projected = values @ directions.T
    result = np.zeros(len(values), dtype=np.float64)
    for index in range(1, len(values)):
        left = np.sort(projected[index - 1], axis=0)
        right = np.sort(projected[index], axis=0)
        # Probe counts are constant within a cached rollout/probe variant.
        result[index] = float(np.mean(np.abs(left - right)))
    return result.astype(np.float32)


def _rbf_mmd_shift(clouds: np.ndarray) -> np.ndarray:
    """Adjacent-anchor RBF-MMD with a per-pair median-distance bandwidth."""
    values = np.asarray(clouds, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected descriptor clouds (T,N,D), got {values.shape}.")
    result = np.zeros(len(values), dtype=np.float64)
    for index in range(1, len(values)):
        left, right = values[index - 1], values[index]
        joined = np.concatenate([left, right], axis=0)
        squared = np.sum((joined[:, None] - joined[None, :]) ** 2, axis=-1)
        positive = squared[squared > 1e-12]
        bandwidth2 = float(np.median(positive)) if len(positive) else 1.0
        bandwidth2 = max(bandwidth2, 1e-8)

        def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            dist2 = np.sum((a[:, None] - b[None, :]) ** 2, axis=-1)
            return np.exp(-dist2 / (2.0 * bandwidth2))

        kxx, kyy, kxy = kernel(left, left), kernel(right, right), kernel(left, right)
        # Biased MMD is non-negative and stable at these modest cloud sizes.
        mmd2 = float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
        result[index] = np.sqrt(max(mmd2, 0.0))
    return result.astype(np.float32)


def _paired_rollout_shift(clouds: np.ndarray) -> np.ndarray:
    """Mean adjacent-anchor displacement under common-noise sample pairing."""
    values = np.asarray(clouds, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected descriptor clouds (T,N,D), got {values.shape}.")
    result = np.zeros(len(values), dtype=np.float64)
    if len(values) > 1:
        result[1:] = np.linalg.norm(values[1:] - values[:-1], axis=-1).mean(axis=1)
    return result.astype(np.float32)


def _trajectory_xyz(raw: dict) -> np.ndarray:
    """Validate cached cumulative XYZ paths as (anchors, samples, horizon, 3)."""
    if "trajectory_xyz" not in raw:
        raise KeyError(
            "This metric requires a direct-rollout cache containing trajectory_xyz. "
            "Rerun the GPU evaluation once to upgrade the old descriptor-only cache."
        )
    values = np.asarray(raw["trajectory_xyz"], dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] != 3 or values.shape[1] < 2:
        raise ValueError(f"Expected trajectory_xyz (T,N,H,3), got {values.shape}.")
    return values


def _action_sequence(raw: dict) -> np.ndarray:
    """Validate normalized non-gripper action plans as (T,N,H,D)."""
    if "action_sequence" not in raw:
        raise KeyError(
            "This metric requires a direct-rollout cache containing action_sequence. "
            "Rerun the GPU evaluation once to upgrade the old cache."
        )
    values = np.asarray(raw["action_sequence"], dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] < 1 or values.shape[1] < 2:
        raise ValueError(f"Expected action_sequence (T,N,H,D), got {values.shape}.")
    return values


def _pairwise_flat_cosine(values: np.ndarray) -> np.ndarray:
    """Mean pairwise angular divergence for an arbitrary sampled sequence cloud."""
    result = np.zeros(len(values), dtype=np.float64)
    for index, cloud in enumerate(values):
        flat = cloud.reshape(len(cloud), -1)
        norms = np.linalg.norm(flat, axis=1)
        valid = norms > 1e-8
        if np.count_nonzero(valid) < 2:
            continue
        unit = flat[valid] / norms[valid, None]
        cosine = np.clip(unit @ unit.T, -1.0, 1.0)
        result[index] = np.mean(1.0 - cosine[np.triu_indices(len(unit), k=1)])
    return result.astype(np.float32)


def _pairwise_flat_l2(values: np.ndarray) -> np.ndarray:
    """Mean pairwise per-coordinate RMS separation for sampled sequences."""
    result = np.zeros(len(values), dtype=np.float64)
    for index, cloud in enumerate(values):
        flat = cloud.reshape(len(cloud), -1)
        diff = flat[:, None, :] - flat[None, :, :]
        rms = np.sqrt(np.mean(np.square(diff), axis=-1))
        result[index] = np.mean(rms[np.triu_indices(len(cloud), k=1)])
    return result.astype(np.float32)


def _trajectory_xyz_pairwise_cosine(raw: dict) -> np.ndarray:
    """Cluster-free angular dispersion across all sampled cumulative XYZ paths."""
    return _pairwise_flat_cosine(_trajectory_xyz(raw))


def _trajectory_xyz_pairwise_l2(raw: dict) -> np.ndarray:
    """Mean pairwise RMS separation between complete cumulative XYZ paths."""
    return _pairwise_flat_l2(_trajectory_xyz(raw))


def _action_sequence_pairwise_cosine(raw: dict) -> np.ndarray:
    """Angular dispersion of all normalized non-gripper action dimensions."""
    return _pairwise_flat_cosine(_action_sequence(raw))


def _action_sequence_pairwise_l2(raw: dict) -> np.ndarray:
    """RMS spread of all normalized non-gripper action dimensions."""
    return _pairwise_flat_l2(_action_sequence(raw))


def _action_mean_pairwise_cosine(raw: dict) -> np.ndarray:
    """Angular dispersion after averaging each non-gripper action plan over time.

    Mean pooling suppresses per-step sampling noise while remaining agnostic to
    whether the policy action represents Cartesian deltas, absolute targets, or
    relative/absolute joint commands.
    """
    action = _action_sequence(raw)
    return _pairwise_flat_cosine(action.mean(axis=2, keepdims=True))


def _endpoint_xyz_pairwise_l2(raw: dict) -> np.ndarray:
    """Mean pairwise Euclidean spread of sampled final XYZ displacements."""
    endpoints = _trajectory_xyz(raw)[:, :, -1, :]
    result = np.zeros(len(endpoints), dtype=np.float64)
    for index, cloud in enumerate(endpoints):
        diff = cloud[:, None, :] - cloud[None, :, :]
        distance = np.linalg.norm(diff, axis=-1)
        result[index] = np.mean(distance[np.triu_indices(len(cloud), k=1)])
    return result.astype(np.float32)


def _fit_gmm(data: np.ndarray, k: int, cfg: dict):
    from sklearn.mixture import GaussianMixture

    for reg in (1e-6, 1e-4, 1e-2):
        try:
            model = GaussianMixture(
                n_components=k,
                covariance_type=str(cfg.get("covariance", "diag")),
                n_init=int(cfg.get("n_init", 10)),
                random_state=0,
                max_iter=int(cfg.get("max_iter", 300)),
                reg_covar=reg,
            ).fit(np.asarray(data, dtype=np.float64))
            return model
        except (ValueError, np.linalg.LinAlgError):
            continue
    return None


def _cluster_scores(clouds: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    # Import sklearn before entering threadpool_limits so threadpoolctl can see
    # and constrain every BLAS/OpenMP runtime loaded by sklearn/scipy.
    from sklearn.mixture import GaussianMixture as _GaussianMixture  # noqa: F401
    from threadpoolctl import threadpool_limits

    result = {key: [] for key in _ALL_CLUSTER_SCORE_KEYS}
    magnitude_gate_tau = float(cfg.get("magnitude_gate_tau", 1.0))
    if not np.isfinite(magnitude_gate_tau) or magnitude_gate_tau < 0.0:
        raise ValueError(
            "magnitude_gate_tau must be a finite non-negative value, got "
            f"{magnitude_gate_tau}."
        )
    # Each GMM sees only a few dozen low-dimensional points.  Letting OpenBLAS
    # spawn all allocated CPU threads for these tiny matrix operations creates
    # severe oversubscription (and was ~29x slower in the 25x6 probe benchmark).
    # This changes only thread scheduling: GMM settings, random seed, fitting
    # order, and all score formulas remain identical.
    with threadpool_limits(limits=1):
        for cloud in clouds:
            cloud = np.asarray(cloud, dtype=np.float64)
            reference_center = str(cfg.get("reference_center", "none"))
            if reference_center == "sample0":
                # Sample zero is the DP result initialized from the observed
                # demonstration action; the remaining samples are perturbed
                # probes. Expressing every fitted mode relative to that
                # per-anchor reference removes the descriptor's absolute
                # translation without forcing the fitted mode means to sum to
                # zero (as subtracting the whole cloud mean would do).
                if len(cloud) == 0:
                    raise ValueError("sample0 reference centering needs a non-empty cloud.")
                cloud = cloud - cloud[0:1]
            elif reference_center != "none":
                raise ValueError(
                    "reference_center must be none|sample0, got "
                    f"{reference_center!r}."
                )
            if str(cfg.get("selection", "fixed")) == "bic":
                models = {
                    k: model
                    for k in range(
                        int(cfg.get("min_k", 1)), int(cfg.get("max_k", 3)) + 1
                    )
                    if (model := _fit_gmm(cloud, k, cfg)) is not None
                }
                bics = {k: float(model.bic(cloud)) for k, model in models.items()}
                model = models[min(bics, key=bics.get)] if bics else None
                bic_k1 = bics.get(1, float("nan"))
                multi_bics = {k: value for k, value in bics.items() if k >= 2}
                bic_best_multi = (
                    min(multi_bics.values()) if multi_bics else float("nan")
                )
                delta_bic = bic_k1 - bic_best_multi
                if bics:
                    bic_values = np.asarray(list(bics.values()), dtype=np.float64)
                    weights = np.exp(-0.5 * (bic_values - bic_values.min()))
                    keys = list(bics)
                    bic_multi_probability = float(
                        weights[[k >= 2 for k in keys]].sum() / weights.sum()
                    )
                else:
                    bic_multi_probability = float("nan")
            else:
                model = _fit_gmm(cloud, int(cfg.get("k", 5)), cfg)
                bic_k1 = bic_best_multi = delta_bic = bic_multi_probability = float(
                    "nan"
                )
            if model is None:
                cosine = magnitude_gated_cosine = covariance_gated_cosine = 0.0
                max_covariance_gated_cosine = l2 = within_normalized_l2 = 0.0
                covariance_gated_angular_chord = 0.0
                selected_k = 0
            else:
                selected_k = int(model.n_components)
                keep = _retained_component_indices(
                    model.weights_, int(len(cloud)), cfg
                )
                means = model.means_[keep]
                weights = model.weights_[keep]
                if len(means) < 2 or selected_k == 1:
                    cosine = magnitude_gated_cosine = covariance_gated_cosine = 0.0
                    max_covariance_gated_cosine = l2 = within_normalized_l2 = 0.0
                    covariance_gated_angular_chord = 0.0
                else:
                    covariances = _component_covariance_matrices(model, keep)
                    direction_reliability = np.asarray(
                        [
                            _directional_reliability(mean, covariance)
                            for mean, covariance in zip(means, covariances, strict=True)
                        ],
                        dtype=np.float64,
                    )
                    pairs = [
                        (i, j)
                        for i in range(len(means))
                        for j in range(i + 1, len(means))
                    ]
                    pair_weights = np.asarray(
                        [weights[i] * weights[j] for i, j in pairs], dtype=np.float64
                    )
                    if not bool(cfg.get("weighted", True)):
                        pair_weights[:] = 1.0
                    pair_weights /= pair_weights.sum()
                    angular = []
                    magnitude_gated_angular = []
                    covariance_gated_angular = []
                    covariance_gated_chords = []
                    distances = []
                    within_normalized_distances = []
                    for i, j in pairs:
                        norm_product = float(
                            np.linalg.norm(means[i]) * np.linalg.norm(means[j])
                        )
                        angular_distance = 1.0 - float(
                            np.dot(means[i], means[j]) / (norm_product + 1e-8)
                        )
                        angular.append(angular_distance)
                        # Cosine direction is ill-conditioned when both motion
                        # descriptors are close to zero (most visibly near an
                        # episode's terminal settling region).  Gate each pair
                        # by its own magnitude rather than the cloud center:
                        # opposite, high-magnitude modes may have a near-zero
                        # center and must remain visible to the detector.
                        reliability = (
                            1.0
                            if magnitude_gate_tau == 0.0
                            else norm_product
                            / (norm_product + magnitude_gate_tau**2)
                        )
                        magnitude_gated_angular.append(
                            angular_distance * reliability
                        )
                        # No descriptor-space threshold is selected here. A
                        # pair is reliable only when both fitted modes have a
                        # mean direction stronger than their perpendicular
                        # within-mode covariance.
                        covariance_reliability = float(
                            np.sqrt(
                                direction_reliability[i]
                                * direction_reliability[j]
                            )
                        )
                        covariance_gated_angular.append(
                            angular_distance * covariance_reliability
                        )
                        covariance_gated_chords.append(
                            _angular_chord_distance(means[i], means[j])
                            * covariance_reliability
                        )
                        center_distance = float(np.linalg.norm(means[i] - means[j]))
                        distances.append(center_distance)
                        # Expected squared distance of samples from a Gaussian
                        # component mean is trace(covariance). Divide center
                        # separation by the pair's pooled within-component RMS
                        # radius: high values mean distinct compact modes,
                        # while a broadly scattered single solution stays low.
                        pooled_within_radius = float(
                            np.sqrt(
                                max(
                                    0.5
                                    * (
                                        np.trace(covariances[i])
                                        + np.trace(covariances[j])
                                    ),
                                    np.finfo(np.float64).eps,
                                )
                            )
                        )
                        within_normalized_distances.append(
                            center_distance / pooled_within_radius
                        )
                    cosine = float(
                        np.sqrt(np.sum(pair_weights * np.square(angular)))
                    )
                    magnitude_gated_cosine = float(
                        np.sqrt(
                            np.sum(
                                pair_weights
                                * np.square(magnitude_gated_angular)
                            )
                        )
                    )
                    covariance_gated_cosine = float(
                        np.sqrt(
                            np.sum(
                                pair_weights
                                * np.square(covariance_gated_angular)
                            )
                        )
                    )
                    covariance_gated_angular_chord = float(
                        np.sqrt(
                            np.sum(
                                pair_weights
                                * np.square(covariance_gated_chords)
                            )
                        )
                    )
                    # Farthest-pair counterpart of the mixture-weighted RMS
                    # above. Component-support filtering and covariance
                    # reliability are unchanged; only the aggregation over
                    # retained mode pairs changes from all pairs to max(pair).
                    max_covariance_gated_cosine = float(
                        np.max(covariance_gated_angular)
                    )
                    l2 = float(
                        np.sqrt(np.sum(pair_weights * np.square(distances)))
                    )
                    within_normalized_l2 = float(
                        np.sqrt(
                            np.sum(
                                pair_weights
                                * np.square(within_normalized_distances)
                            )
                        )
                    )
            result["cosine"].append(cosine)
            result["magnitude_gated_cosine"].append(magnitude_gated_cosine)
            result["covariance_gated_cosine"].append(covariance_gated_cosine)
            result["covariance_gated_angular_chord"].append(
                covariance_gated_angular_chord
            )
            result["max_covariance_gated_cosine"].append(
                max_covariance_gated_cosine
            )
            result["l2"].append(l2)
            result["within_normalized_l2"].append(within_normalized_l2)
            result["hybrid"].append(cosine * l2)
            result["selected_k"].append(selected_k)
            result["center_norm"].append(float(np.linalg.norm(np.mean(cloud, axis=0))))
            result["bic_k1"].append(bic_k1)
            result["bic_best_multi"].append(bic_best_multi)
            result["delta_bic"].append(delta_bic)
            result["bic_multi_probability"].append(bic_multi_probability)
    return {key: np.asarray(value, dtype=np.float32) for key, value in result.items()}


def _smooth(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    from scipy.signal import savgol_filter

    window = min(int(window), len(values))
    window -= 1 - window % 2
    if window < polyorder + 2 or window <= 1:
        return values.astype(np.float64)
    return savgol_filter(values, window_length=window, polyorder=int(polyorder))


def _threshold(values: np.ndarray, cfg: dict) -> float:
    mode = str(cfg.get("threshold", "mean"))
    scale = float(cfg.get("threshold_scale", 1.0))
    if mode == "fixed":
        return float(cfg.get("threshold_value", 0.0))
    if mode == "mean":
        return float(np.mean(values)) * scale
    if mode == "rms":
        # For non-negative detector scores, RMS is a parameter-free threshold
        # that is guaranteed to lie between the arithmetic mean and maximum.
        # Clamp tiny SG undershoots so they cannot contribute artificial
        # energy to the threshold.
        nonnegative = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
        return float(np.sqrt(np.mean(np.square(nonnegative)))) * scale
    if mode == "median_mad":
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median + scale * 1.4826 * mad
    if mode == "median_mad_floor_zero":
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return max(0.0, median + scale * 1.4826 * mad)
    if mode == "gap_mad":
        # A paper-style running surprise is already centered by its segment
        # mean, so its GAP needs only a robust scale, not a location term.
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return scale * 1.4826 * mad
    raise ValueError(f"Unknown threshold mode: {mode}")


def _score_power_transform(values: np.ndarray, power: float) -> np.ndarray:
    """Apply a non-negative power transform used by boundary selection.

    The boundary metrics used by this experiment are non-negative by
    construction.  Savitzky-Golay smoothing can nevertheless introduce tiny
    negative undershoots, so clamp those numerical artifacts before applying
    a non-integer power.  ``power=1`` remains the ordinary score curve apart
    from that non-negativity correction.
    """
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError(f"score_power must be finite and positive, got {power}.")
    array = np.asarray(values, dtype=np.float64)
    if power == 1.0:
        # Preserve the legacy p=1 baseline exactly.
        return array.copy()
    return np.power(np.maximum(array, 0.0), power)


def _eligible_peak_prominences(
    valid_smooth: np.ndarray,
    values: np.ndarray,
    ts: np.ndarray,
    valid_count: int,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eligible peak indices and their local prominences.

    The first terminal-masked anchor is used only as right-hand topology
    context, matching the endpoint handling in ``_boundaries``. Peaks inside
    the protected start region are excluded before a shared prominence
    threshold is estimated, so invalid early peaks cannot bias that statistic.
    """
    from scipy.signal import find_peaks, peak_prominences

    minimum = float(cfg.get("prominence", 0.0))
    peaks, properties = find_peaks(valid_smooth, prominence=minimum)
    prominence_by_index = {
        int(index): float(prominence)
        for index, prominence in zip(
            peaks, properties.get("prominences", np.empty(0)), strict=True
        )
    }
    terminal_mask_frames = int(cfg.get("terminal_mask_frames", 0))
    if terminal_mask_frames > 0 and valid_count < len(values) and valid_count >= 2:
        topology_base = _smooth(
            values[: valid_count + 1],
            int(cfg.get("smooth_window", 7)),
            int(cfg.get("polyorder", 4)),
        )
        topology_smooth = _score_power_transform(
            topology_base, float(cfg.get("score_power", 1.0))
        )
        topology_peaks, topology_properties = find_peaks(
            topology_smooth, prominence=minimum
        )
        edge_index = valid_count - 1
        if edge_index in topology_peaks:
            location = int(np.flatnonzero(topology_peaks == edge_index)[0])
            topology_prominences = topology_properties.get("prominences")
            if topology_prominences is None:
                topology_prominences = peak_prominences(
                    topology_smooth, topology_peaks
                )[0]
            prominence_by_index[edge_index] = float(topology_prominences[location])

    start_guard_frames = int(
        cfg.get("start_guard_frames", cfg.get("nms_frames", 20))
    )
    eligible = np.asarray(
        sorted(
            index
            for index in prominence_by_index
            if int(ts[index]) - int(ts[0]) > start_guard_frames
        ),
        dtype=np.int64,
    )
    prominences = np.asarray(
        [prominence_by_index[int(index)] for index in eligible],
        dtype=np.float64,
    )
    return eligible, prominences


def _nms(candidates: list[int], values: np.ndarray, ts: np.ndarray, distance: int) -> list[int]:
    kept = []
    for index in sorted(candidates, key=lambda i: float(values[i]), reverse=True):
        if all(abs(int(ts[index]) - int(ts[other])) > distance for other in kept):
            kept.append(index)
    return sorted(kept, key=lambda i: int(ts[i]))


def _enforce_global_min_skill_length(
    cuts: np.ndarray,
    ts: np.ndarray,
    scores: np.ndarray,
    *,
    n_frames: int,
    min_skill_len: int,
) -> np.ndarray:
    """Keep the strongest cuts subject to a minimum length for every segment.

    Endpoint checks protect the first and final segment.  Interior conflicts
    are resolved in descending peak-score order so that a weaker early bump
    cannot suppress a nearby, stronger boundary merely because it occurs
    first in time.
    """
    cuts = np.asarray(cuts, dtype=np.int64)
    if min_skill_len <= 0 or not len(cuts):
        return cuts
    ts = np.asarray(ts, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(ts) != len(scores):
        raise ValueError("Boundary timestamps and scores must have equal length.")
    score_by_cut = {int(frame): float(scores[index]) for index, frame in enumerate(ts)}
    eligible = [
        int(cut)
        for cut in cuts
        if int(cut) >= min_skill_len and n_frames - int(cut) >= min_skill_len
    ]
    kept: list[int] = []
    for cut in sorted(eligible, key=lambda frame: score_by_cut[frame], reverse=True):
        if all(abs(cut - other) >= min_skill_len for other in kept):
            kept.append(cut)
    return np.asarray(sorted(kept), dtype=np.int64)


def _persistent_local_peaks(
    values: np.ndarray,
    ts: np.ndarray,
    cfg: dict,
) -> tuple[list[int], np.ndarray, float]:
    """Find single-episode peaks that survive smoothing scale and local noise.

    This intentionally uses no episode-level height threshold.  A candidate
    must instead (1) be a local maximum at the reference scale, (2) recur at a
    majority of the configured smoothing scales within one evaluation stride,
    and (3) have local prominence greater than the robust frame-to-frame noise
    floor of the reference curve.  The majority vote and a one-stride match
    remove tunable vote/tolerance constants from this diagnostic.
    """
    from scipy.signal import find_peaks, peak_prominences

    windows = [int(window) for window in cfg.get("smooth_windows", [5, 7, 9])]
    polyorder = int(cfg.get("polyorder", 4))
    reference_window = int(cfg.get("smooth_window", windows[len(windows) // 2]))
    if reference_window not in windows:
        raise ValueError(
            f"persistent_local smooth_window={reference_window} must be one of "
            f"smooth_windows={windows}."
        )
    curves = [_smooth(values, window, min(polyorder, window - 1)) for window in windows]
    peak_sets = [find_peaks(curve)[0] for curve in curves]
    reference_index = windows.index(reference_window)
    reference_curve = curves[reference_index]
    reference_peaks = peak_sets[reference_index]
    if not len(reference_peaks):
        return [], reference_curve, float("nan")

    positive_steps = np.diff(ts)
    positive_steps = positive_steps[positive_steps > 0]
    match_tolerance = int(np.median(positive_steps)) if len(positive_steps) else 0
    required_votes = len(windows) // 2 + 1
    persistent = []
    for peak in reference_peaks:
        timestamp = int(ts[peak])
        votes = sum(
            bool(
                len(other_peaks)
                and np.min(np.abs(ts[other_peaks] - timestamp)) <= match_tolerance
            )
            for other_peaks in peak_sets
        )
        if votes >= required_votes:
            persistent.append(int(peak))

    if not persistent:
        return [], reference_curve, float("nan")
    persistent_array = np.asarray(persistent, dtype=np.int64)
    prominences = peak_prominences(reference_curve, persistent_array)[0]
    delta = np.diff(reference_curve)
    if len(delta):
        delta_center = float(np.median(delta))
        noise_floor = float(1.4826 * np.median(np.abs(delta - delta_center)))
    else:
        noise_floor = 0.0
    # A perfectly smooth synthetic curve has zero estimated noise.  In that
    # case retain every persistent peak with positive topographic prominence.
    keep = prominences > max(noise_floor, 0.0)
    return persistent_array[keep].astype(int).tolist(), reference_curve, noise_floor


def _sustained_plateau_centers(
    values: np.ndarray,
    threshold: float,
    *,
    min_active_points: int = 3,
    max_gap_points: int = 1,
    exclude_initial_group: bool = True,
) -> list[int]:
    """Collapse consecutive sustained high scores into center boundaries.

    Away from episode initialization, a chunk must contain at least
    ``min_active_points`` *consecutive* above-threshold anchors. Short runs are
    ignored and are never bridged across a dip, which makes chunk membership
    unambiguous.

    Initialization is the sole exception. If the first evaluated anchor is
    active and ``exclude_initial_group`` is enabled, its initial chunk may
    bridge at most ``max_gap_points`` single-anchor dips. The entire bridged
    initial chunk is then discarded as initialization uncertainty. Thus an
    initial 10,10,8,10,10,10 sequence is treated as one excluded chunk, while
    the same pattern later in the episode produces only the final consecutive
    three-point chunk.
    """
    values = np.asarray(values, dtype=np.float64)
    active = values > float(threshold)
    if not bool(np.any(active)):
        return []
    min_active_points = int(min_active_points)
    max_gap_points = int(max_gap_points)
    if min_active_points < 1 or max_gap_points < 0:
        raise ValueError(
            "Sustained plateau requires min_active_points>=1 and max_gap_points>=0."
        )

    scan_start = 0
    if exclude_initial_group and bool(active[0]):
        # Consume the initial active chunk, allowing its configured number of
        # isolated dips. Only initialization receives this exception.
        index = 0
        gaps_used = 0
        while index < len(active):
            if bool(active[index]):
                index += 1
                continue
            can_bridge = (
                gaps_used < max_gap_points
                and index + 1 < len(active)
                and bool(active[index + 1])
            )
            if not can_bridge:
                break
            gaps_used += 1
            index += 1
        scan_start = index

    centers: list[int] = []
    index = scan_start
    while index < len(active):
        if not bool(active[index]):
            index += 1
            continue
        start = index
        while index < len(active) and bool(active[index]):
            index += 1
        end = index - 1
        if end - start + 1 >= min_active_points:
            # For an even span, choose the later central evaluation anchor.
            centers.append((start + end + 1) // 2)
    return centers


def _running_surprise(
    values: np.ndarray,
    ts: np.ndarray,
    gap: float,
    *,
    min_history_points: int = 2,
    start_guard_frames: int = 0,
    nms_frames: int = 0,
) -> tuple[list[int], np.ndarray]:
    """Paper-style ``loss - segment mean > GAP`` detection with reset.

    The current loss is included in the running mean before comparison, exactly
    as in Algorithm 1 of the reference paper.  Once a boundary is accepted the
    history is cleared, so the next point starts a fresh skill-local baseline.
    """
    values = np.asarray(values, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.int64)
    min_history_points = int(min_history_points)
    if min_history_points < 1:
        raise ValueError("running_surprise_min_history_points must be positive.")
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError(f"running-surprise GAP must be finite and non-negative, got {gap}.")

    surprise = np.zeros(len(values), dtype=np.float64)
    candidates: list[int] = []
    history: list[float] = []
    for index, value in enumerate(values):
        history.append(float(value))
        surprise[index] = float(value) - float(np.mean(history))
        outside_start_guard = int(ts[index]) - int(ts[0]) > int(start_guard_frames)
        separated = not candidates or (
            int(ts[index]) - int(ts[candidates[-1]]) > int(nms_frames)
        )
        if (
            len(history) >= min_history_points
            and outside_start_guard
            and separated
            and surprise[index] > gap
        ):
            candidates.append(index)
            history = []
    return candidates, surprise


def _boundaries(
    values: np.ndarray,
    ts: np.ndarray,
    cfg: dict,
    *,
    shared_threshold: float | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    from scipy.signal import find_peaks

    values = np.asarray(values, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.int64)
    if values.ndim != 1 or ts.ndim != 1 or len(values) != len(ts) or not len(values):
        raise ValueError(
            f"Boundary values/timestamps must be matching non-empty vectors, got "
            f"{values.shape} and {ts.shape}."
        )
    if np.any(np.diff(ts) < 0):
        raise ValueError("Boundary timestamps must be monotonically increasing.")

    # Terminal settling can create a large but task-agnostic direction change.
    # Mask it *before* smoothing and threshold estimation; merely deleting a
    # terminal cut afterwards still lets that spike raise the episode mean and
    # suppress legitimate earlier peaks.  The guard is measured backwards
    # from the final evaluated anchor (not the raw episode end), so a 24-frame
    # guard removes the final 24 frames of the detector curve.
    terminal_mask_frames = int(cfg.get("terminal_mask_frames", 0))
    if terminal_mask_frames < 0:
        raise ValueError("terminal_mask_frames must be non-negative.")
    terminal_cutoff = int(ts[-1]) - terminal_mask_frames
    valid_count = int(np.searchsorted(ts, terminal_cutoff, side="right"))
    if valid_count < 1:
        raise ValueError(
            f"terminal_mask_frames={terminal_mask_frames} masks every boundary anchor."
        )
    valid_values = values[:valid_count]
    valid_ts = ts[:valid_count]
    method = str(cfg.get("method", "peak"))
    threshold_target = str(cfg.get("threshold_target", "score"))
    if threshold_target not in {"score", "prominence"}:
        raise ValueError(
            f"threshold_target must be score|prominence, got {threshold_target!r}."
        )
    if threshold_target == "prominence" and method != "peak":
        raise ValueError("Prominence thresholding is supported only for method=peak.")
    threshold_override = shared_threshold
    if method == "persistent_local":
        candidates, valid_smooth, _prominence_noise_floor = _persistent_local_peaks(
            valid_values, valid_ts, cfg
        )
    elif method == "sustained_plateau":
        # This detector is defined on the raw discrete convergence-step
        # sequence. Do not apply SG or any other temporal smoothing.
        valid_smooth = _score_power_transform(
            valid_values, float(cfg.get("score_power", 1.0))
        )
    elif method == "running_surprise":
        if shared_threshold is not None:
            raise ValueError("running_surprise does not support a shared threshold.")
        base_smooth = _smooth(
            valid_values,
            int(cfg.get("smooth_window", 1)),
            int(cfg.get("polyorder", 0)),
        )
        threshold_override = _threshold(base_smooth, cfg)
        candidates, valid_smooth = _running_surprise(
            base_smooth,
            valid_ts,
            threshold_override,
            min_history_points=int(
                cfg.get("running_surprise_min_history_points", 2)
            ),
            start_guard_frames=int(cfg.get("start_guard_frames", 0)),
            nms_frames=int(cfg.get("nms_frames", 0)),
        )
    else:
        base_smooth = _smooth(
            valid_values,
            int(cfg.get("smooth_window", 7)),
            int(cfg.get("polyorder", 4)),
        )
        # Apply the contrast transform after smoothing, then estimate the
        # episode-specific threshold in transformed space.  For a non-negative
        # score and power=2, mean(score**2) is equivalent to using the RMS of
        # the original smoothed curve as its threshold.
        valid_smooth = _score_power_transform(
            base_smooth, float(cfg.get("score_power", 1.0))
        )
    # Keep report arrays aligned with the raw curve. A flat masked tail makes
    # the ignored region visible without allowing terminal values to leak back
    # through Savitzky-Golay smoothing or threshold statistics.
    smooth = np.full(len(values), float(valid_smooth[-1]), dtype=np.float64)
    smooth[:valid_count] = valid_smooth
    threshold = (
        threshold_override
        if threshold_override is not None
        else (
            float("nan")
            if method in {"all_peaks", "persistent_local"}
            or threshold_target == "prominence"
            else _threshold(valid_smooth, cfg)
        )
    )
    nms_frames = int(cfg.get("nms_frames", 20))
    start_guard_frames = int(cfg.get("start_guard_frames", nms_frames))
    if method == "peak":
        peaks, peak_prominences = _eligible_peak_prominences(
            valid_smooth, values, ts, valid_count, cfg
        )
        if threshold_target == "prominence":
            if threshold_override is None:
                threshold = (
                    _threshold(peak_prominences, cfg)
                    if len(peak_prominences)
                    else float("nan")
                )
            candidates = [
                int(index)
                for index, prominence in zip(peaks, peak_prominences, strict=True)
                if prominence > threshold
            ]
            # Resolve close peaks by the same quantity used for selection.
            nms_values = np.zeros_like(valid_smooth)
            nms_values[peaks] = peak_prominences
            candidates = _nms(candidates, nms_values, valid_ts, nms_frames)
        else:
            candidates = [int(index) for index in peaks if valid_smooth[index] > threshold]
            candidates = _nms(candidates, valid_smooth, valid_ts, nms_frames)
    elif method == "all_peaks":
        peaks, _ = find_peaks(valid_smooth)
        candidates = [
            int(i)
            for i in peaks
            if int(valid_ts[i]) - int(valid_ts[0]) > start_guard_frames
        ]
        candidates = _nms(candidates, valid_smooth, valid_ts, nms_frames)
    elif method == "persistent_local":
        candidates = [
            int(i)
            for i in candidates
            if int(valid_ts[i]) - int(valid_ts[0]) > start_guard_frames
        ]
        candidates = _nms(candidates, valid_smooth, valid_ts, nms_frames)
    elif method == "rising":
        above = valid_smooth > threshold
        sustain = max(1, int(cfg.get("sustain", 2)))
        candidates = []
        index = 0
        while index < len(above):
            if not above[index]:
                index += 1
                continue
            end = index
            while end < len(above) and above[end]:
                end += 1
            if end - index >= sustain:
                candidates.append(index)
            index = end
        candidates = [
            i
            for i in candidates
            if int(valid_ts[i]) - int(valid_ts[0]) > start_guard_frames
        ]
        candidates = _nms(candidates, valid_smooth, valid_ts, nms_frames)
    elif method == "sustained_plateau":
        candidates = _sustained_plateau_centers(
            valid_smooth,
            threshold,
            min_active_points=int(cfg.get("plateau_min_active_points", 3)),
            max_gap_points=int(cfg.get("plateau_max_gap_points", 1)),
            exclude_initial_group=bool(cfg.get("plateau_exclude_initial_group", True)),
        )
    elif method == "running_surprise":
        # Candidates were produced sequentially above because every accepted
        # cut resets the segment-local loss baseline.
        pass
    else:
        raise ValueError(f"Unknown boundary method: {method}")
    return (
        valid_ts[candidates].astype(np.int64),
        threshold,
        np.asarray(smooth, dtype=np.float32),
    )


def _derive(
    raw: dict,
    cluster: dict,
    metric: str,
    boundary: dict,
    *,
    min_skill_len: int,
    scores: dict[str, np.ndarray] | None = None,
    input_scores: dict[str, np.ndarray] | None = None,
    descriptor_whitening_scale: np.ndarray | None = None,
    input_descriptor_spread: float | None = None,
    shared_threshold: float | None = None,
) -> dict:
    if scores is None:
        scores = _cluster_scores(raw["descriptors"], cluster)
    score_ts = np.asarray(raw["replan_ts"], dtype=np.int64)
    convergence_threshold_value = None
    if metric == "delta_bic_gain":
        if input_scores is None:
            raise ValueError("delta_bic_gain requires input probe BIC scores.")
        score = scores["delta_bic"] - float(input_scores["delta_bic"][0])
    elif metric in {"whitened_spread", "whitened_spread_gain"}:
        if descriptor_whitening_scale is None:
            raise ValueError(f"{metric} requires a descriptor whitening scale.")
        score = _whitened_descriptor_spread(
            raw["descriptors"], descriptor_whitening_scale
        )
        if metric == "whitened_spread_gain":
            if input_descriptor_spread is None or input_descriptor_spread <= 1e-12:
                raise ValueError(
                    "whitened_spread_gain requires a positive input probe spread."
                )
            score = score / float(input_descriptor_spread)
    elif metric.startswith("prediction_loss_"):
        exclude_gripper = metric.endswith("_no_gripper")
        base_metric = (
            metric[: -len("_no_gripper")] if exclude_gripper else metric
        )
        loss_key = (
            "prediction_loss_per_offset_no_gripper"
            if exclude_gripper
            else "prediction_loss_per_offset"
        )
        if loss_key not in raw:
            raise ValueError(
                f"{metric} requires a prediction-loss cache containing {loss_key}."
            )
        per_offset = np.asarray(raw[loss_key], dtype=np.float64)
        if base_metric == "prediction_loss_current":
            score = per_offset[:, 0]
        elif base_metric == "prediction_loss_future_mean":
            score = np.mean(per_offset, axis=1)
        elif base_metric == "prediction_loss_aligned":
            score_ts, score, _support = _aligned_prediction_loss(
                raw,
                exclude_gripper=exclude_gripper,
                aggregation=str(
                    boundary.get("prediction_loss_alignment_aggregation", "median")
                ),
            )
        else:
            raise ValueError(f"Unknown prediction-loss metric {metric!r}.")
    elif metric.startswith("denoising_"):
        convergence_kwargs = {
            "convergence_threshold": boundary.get(
                "denoising_convergence_threshold"
            ),
            "relative_convergence_threshold": boundary.get(
                "denoising_relative_convergence_threshold"
            ),
            "sample_aggregation": boundary.get(
                "denoising_sample_aggregation", "median"
            ),
            "convergence_threshold_mode": boundary.get(
                "denoising_convergence_threshold_mode", "fixed"
            ),
            "terminal_mask_frames": int(boundary.get("terminal_mask_frames", 0)),
        }
        if metric.startswith("denoising_aligned_"):
            convergence = _aligned_denoising_convergence_scores(
                raw, **convergence_kwargs
            )
            score_ts = np.asarray(
                convergence["denoising_aligned_ts"], dtype=np.int64
            )
        else:
            if descriptor_whitening_scale is None:
                raise ValueError(f"{metric} requires a descriptor whitening scale.")
            convergence = _denoising_convergence_scores(
                raw, descriptor_whitening_scale, **convergence_kwargs
            )
        if metric not in convergence:
            raise ValueError(
                f"Unknown or unavailable denoising metric {metric!r}; "
                f"available={sorted(convergence)}."
            )
        score = convergence[metric]
        convergence_threshold_value = convergence.get(
            "denoising_convergence_threshold_value"
        )
    elif metric == "sliced_wasserstein_shift":
        score = _sliced_wasserstein_shift(raw["descriptors"])
    elif metric == "mmd_shift":
        score = _rbf_mmd_shift(raw["descriptors"])
    elif metric == "paired_rollout_shift":
        score = _paired_rollout_shift(raw["descriptors"])
    elif metric == "trajectory_xyz_cosine":
        score = _trajectory_xyz_pairwise_cosine(raw)
    elif metric == "trajectory_xyz_l2":
        score = _trajectory_xyz_pairwise_l2(raw)
    elif metric == "endpoint_xyz_spread":
        score = _endpoint_xyz_pairwise_l2(raw)
    elif metric == "action_sequence_cosine":
        score = _action_sequence_pairwise_cosine(raw)
    elif metric == "action_sequence_l2":
        score = _action_sequence_pairwise_l2(raw)
    elif metric == "action_mean_cosine":
        score = _action_mean_pairwise_cosine(raw)
    else:
        score = scores[metric]
    cuts, threshold, smooth = _boundaries(
        score, score_ts, boundary, shared_threshold=shared_threshold
    )
    display_score = (
        np.asarray(smooth, dtype=np.float64)
        if str(boundary.get("method", "peak")) == "running_surprise"
        else _score_power_transform(
            score, float(boundary.get("score_power", 1.0))
        )
    )
    # Match the dataset builder: merge a too-short terminal segment into the
    # previous one. The list here contains internal cuts only.
    n_frames = int(np.asarray(raw["n_frames"]).item())
    if "min_skill_len" in boundary:
        cuts = _enforce_global_min_skill_length(
            cuts,
            score_ts,
            smooth,
            n_frames=n_frames,
            min_skill_len=int(boundary["min_skill_len"]),
        )
    else:
        # Legacy behavior: only merge a too-short terminal segment. Existing
        # ablations remain byte-for-byte comparable unless they opt into the
        # global minimum-segment constraint explicitly.
        cuts_list = cuts.astype(np.int64).tolist()
        while cuts_list and n_frames - cuts_list[-1] < int(min_skill_len):
            cuts_list.pop()
        cuts = np.asarray(cuts_list, dtype=np.int64)
    terminal_mask_frames = int(boundary.get("terminal_mask_frames", 0))
    terminal_mask_start_ts = float("nan")
    replan_ts = score_ts
    if terminal_mask_frames > 0 and len(replan_ts):
        cutoff = int(replan_ts[-1]) - terminal_mask_frames
        valid_count = int(np.searchsorted(replan_ts, cutoff, side="right"))
        if valid_count < len(replan_ts):
            terminal_mask_start_ts = float(replan_ts[valid_count])
    result = {
        **scores,
        "metric_name": metric,
        "score": np.asarray(score, dtype=np.float32),
        # Keep the report bars and smoothed curve in the same transformed
        # units without changing the raw score consumed by consensus methods.
        "plot_score": display_score.astype(np.float32),
        "smooth": smooth,
        "threshold": np.array(threshold, dtype=np.float32),
        "threshold_scope": str(boundary.get("threshold_scope", "episode")),
        "threshold_target": str(boundary.get("threshold_target", "score")),
        "boundaries": cuts,
        "replan_ts": score_ts,
        # GMM diagnostics remain defined on the original rollout anchors even
        # when the selected denoising metric is aligned to dense target time.
        "cluster_replan_ts": np.asarray(raw["replan_ts"], dtype=np.int64),
        "n_frames": raw["n_frames"],
        "terminal_mask_start_ts": np.array(
            terminal_mask_start_ts, dtype=np.float32
        ),
    }
    if metric.startswith("denoising_") and convergence_threshold_value is not None:
        result["denoising_convergence_threshold_value"] = np.asarray(
            convergence_threshold_value, dtype=np.float32
        )
    return result


def _derive_consensus(
    parts: list[dict],
    boundary: dict,
    *,
    min_skill_len: int,
    min_positive_fraction: float | None = None,
    normalization: str = "none",
    aggregation: str = "median",
    min_peak_support: int = 0,
    peak_tolerance_frames: int = 0,
    shared_threshold: float | None = None,
) -> dict:
    """Aggregate the same local probe across several diffusion noise levels."""
    if not parts:
        raise ValueError("A consensus experiment requires at least one derived part.")
    reference_ts = np.asarray(parts[0]["replan_ts"])
    for part in parts[1:]:
        if not np.array_equal(reference_ts, np.asarray(part["replan_ts"])):
            raise ValueError("Consensus probes must share identical evaluation anchors.")
    raw_score_matrix = np.stack(
        [np.asarray(part["score"], dtype=np.float64) for part in parts]
    )
    score_matrix = raw_score_matrix.copy()
    if normalization == "robust_z":
        # Cosine-divergence scale changes with diffusion noise level.  Normalize
        # each curve independently so that one high-amplitude level cannot
        # dominate the cross-level consensus.  Fall back to standard deviation
        # only when a curve has effectively zero MAD.
        center = np.median(score_matrix, axis=1, keepdims=True)
        mad_scale = 1.4826 * np.median(
            np.abs(score_matrix - center), axis=1, keepdims=True
        )
        std_scale = np.std(score_matrix, axis=1, keepdims=True)
        scale = np.where(mad_scale > 1e-8, mad_scale, std_scale)
        scale = np.where(scale > 1e-8, scale, 1.0)
        score_matrix = (score_matrix - center) / scale
    elif normalization != "none":
        raise ValueError(f"Unknown consensus normalization: {normalization}")

    if aggregation == "median":
        # With three levels this rejects a spike present at only one level.
        score = np.median(score_matrix, axis=0)
    elif aggregation == "mean":
        score = np.mean(score_matrix, axis=0)
    else:
        raise ValueError(f"Unknown consensus aggregation: {aggregation}")

    positive_fraction = np.mean(raw_score_matrix > 0.0, axis=0)
    if min_positive_fraction is not None and float(min_positive_fraction) > 0.0:
        score = score * (positive_fraction >= float(min_positive_fraction))
    cuts, threshold, smooth = _boundaries(
        score, reference_ts, boundary, shared_threshold=shared_threshold
    )

    support_counts = np.zeros(len(cuts), dtype=np.int16)
    required_support = int(min_peak_support)
    if required_support > 0:
        if required_support > len(parts):
            raise ValueError(
                f"min_peak_support={required_support} exceeds {len(parts)} probes."
            )
        tolerance = max(0, int(peak_tolerance_frames))
        part_cuts = [np.asarray(part["boundaries"], dtype=np.int64) for part in parts]
        for index, cut in enumerate(cuts):
            support_counts[index] = sum(
                bool(len(candidate) and np.min(np.abs(candidate - int(cut))) <= tolerance)
                for candidate in part_cuts
            )
        cuts = cuts[support_counts >= required_support]
        support_counts = support_counts[support_counts >= required_support]

    n_frames = int(np.asarray(parts[0]["n_frames"]).item())
    cuts_list = cuts.astype(np.int64).tolist()
    while cuts_list and n_frames - cuts_list[-1] < int(min_skill_len):
        cuts_list.pop()
        if len(support_counts):
            support_counts = support_counts[:-1]
    selected = np.stack([np.asarray(part["selected_k"]) for part in parts])
    delta = np.stack([np.asarray(part["delta_bic"]) for part in parts])
    return {
        **parts[0],
        "metric_name": f"noise-level {normalization} {aggregation} consensus",
        "score": np.asarray(score, dtype=np.float32),
        "plot_score": _score_power_transform(
            score, float(boundary.get("score_power", 1.0))
        ).astype(np.float32),
        "smooth": smooth,
        "threshold": np.asarray(threshold, dtype=np.float32),
        "boundaries": np.asarray(cuts_list, dtype=np.int64),
        "selected_k": np.max(selected, axis=0),
        "delta_bic": np.median(delta, axis=0).astype(np.float32),
        "consensus_positive_fraction": positive_fraction.astype(np.float32),
        "consensus_peak_support": support_counts,
    }


def _threshold_pool_values(derived: dict, boundary: dict) -> np.ndarray:
    """Return only values that were eligible for this detector's threshold.

    ``smooth`` contains a flat display-only tail after terminal masking.  That
    tail must not enter a shared task/dataset statistic, just as it does not
    enter the legacy episode statistic.
    """
    smooth = np.asarray(derived["smooth"], dtype=np.float64)
    ts = np.asarray(derived["replan_ts"], dtype=np.int64)
    terminal_mask_frames = int(boundary.get("terminal_mask_frames", 0))
    valid_count = len(ts)
    if terminal_mask_frames > 0 and len(ts):
        cutoff = int(ts[-1]) - terminal_mask_frames
        valid_count = int(np.searchsorted(ts, cutoff, side="right"))
    if str(boundary.get("threshold_target", "score")) == "prominence":
        _peaks, values = _eligible_peak_prominences(
            smooth[:valid_count],
            np.asarray(derived["score"], dtype=np.float64),
            ts,
            valid_count,
            boundary,
        )
    else:
        values = smooth[:valid_count]
    values = values[np.isfinite(values)]
    if not len(values):
        if str(boundary.get("threshold_target", "score")) == "prominence":
            return np.empty(0, dtype=np.float64)
        raise ValueError("No finite, unmasked values remain for shared thresholding.")
    return values


def _rethreshold_derived(
    derived: dict,
    boundary: dict,
    *,
    threshold: float,
    threshold_scope: str,
    threshold_group: str,
    threshold_pool_points: int,
    min_skill_len: int,
) -> dict:
    """Re-run only peak selection with a precomputed shared threshold."""
    cuts, _threshold_value, smooth = _boundaries(
        np.asarray(derived["score"], dtype=np.float64),
        np.asarray(derived["replan_ts"], dtype=np.int64),
        boundary,
        shared_threshold=float(threshold),
    )
    n_frames = int(np.asarray(derived["n_frames"]).item())
    if "min_skill_len" in boundary:
        cuts = _enforce_global_min_skill_length(
            cuts,
            np.asarray(derived["replan_ts"], dtype=np.int64),
            smooth,
            n_frames=n_frames,
            min_skill_len=int(boundary["min_skill_len"]),
        )
    else:
        cuts_list = cuts.astype(np.int64).tolist()
        while cuts_list and n_frames - cuts_list[-1] < int(min_skill_len):
            cuts_list.pop()
        cuts = np.asarray(cuts_list, dtype=np.int64)

    result = dict(derived)
    result.update(
        {
            "smooth": np.asarray(smooth, dtype=np.float32),
            "threshold": np.asarray(threshold, dtype=np.float32),
            "threshold_scope": threshold_scope,
            "threshold_group": threshold_group,
            "threshold_pool_points": int(threshold_pool_points),
            "boundaries": cuts,
        }
    )
    return result


def _apply_shared_threshold_scopes(
    episode_rows: dict[int, list[tuple[str, dict]]],
    experiments: list[dict],
    boundary_defs: dict[str, dict],
    task_groups: dict[int, list[int]],
    *,
    min_skill_len: int,
) -> dict:
    """Apply task- or selected-dataset-wide thresholds in a second pass."""
    row_index = {
        episode_id: {name: index for index, (name, _data) in enumerate(rows)}
        for episode_id, rows in episode_rows.items()
    }
    selected_episodes = sorted(episode_rows)
    statistics: dict[str, dict] = {}
    for experiment in experiments:
        name = str(experiment["name"])
        boundary = boundary_defs[str(experiment["boundary"])]
        scope = str(boundary.get("threshold_scope", "episode"))
        if scope == "episode":
            continue
        if scope == "dataset":
            groups = {"dataset": selected_episodes}
        elif scope == "task":
            groups = {
                f"task_{task_id}": [
                    episode_id
                    for episode_id in episode_ids
                    if episode_id in episode_rows
                ]
                for task_id, episode_ids in task_groups.items()
            }
        else:
            raise ValueError(f"Unknown threshold_scope: {scope}")

        group_stats = {}
        covered: set[int] = set()
        for group_name, episode_ids in groups.items():
            if not episode_ids:
                continue
            pool_parts = []
            for episode_id in episode_ids:
                index = row_index[episode_id][name]
                pool_parts.append(
                    _threshold_pool_values(episode_rows[episode_id][index][1], boundary)
                )
            pool = np.concatenate(pool_parts)
            if not len(pool):
                raise ValueError(
                    f"No eligible values remain for shared threshold {name!r} "
                    f"in group {group_name!r}."
                )
            threshold = _threshold(pool, boundary)
            for episode_id in episode_ids:
                index = row_index[episode_id][name]
                _row_name, derived = episode_rows[episode_id][index]
                episode_rows[episode_id][index] = (
                    name,
                    _rethreshold_derived(
                        derived,
                        boundary,
                        threshold=threshold,
                        threshold_scope=scope,
                        threshold_group=group_name,
                        threshold_pool_points=len(pool),
                        min_skill_len=min_skill_len,
                    ),
                )
                covered.add(episode_id)
            group_stats[group_name] = {
                "threshold": float(threshold),
                "episodes": len(episode_ids),
                "points": int(len(pool)),
            }
        missing = set(selected_episodes) - covered
        if missing:
            raise ValueError(
                f"Shared threshold {name!r} did not assign episodes {sorted(missing)}."
            )
        statistics[name] = {
            "scope": scope,
            "estimator": str(boundary.get("threshold", "mean")),
            "target": str(boundary.get("threshold_target", "score")),
            "scale": float(boundary.get("threshold_scale", 1.0)),
            "terminal_mask_frames": int(boundary.get("terminal_mask_frames", 0)),
            "groups": group_stats,
        }
    return statistics


def _gripper_event_frames(signal: np.ndarray | None, threshold: float) -> np.ndarray:
    """Return episode frames where any configured gripper channel changes state."""
    if signal is None:
        return np.empty(0, dtype=np.int64)
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) < 2 or values.shape[1] == 0:
        return np.empty(0, dtype=np.int64)
    valid = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    changed = ((values[1:] > threshold) != (values[:-1] > threshold)) & valid
    return (np.flatnonzero(np.any(changed, axis=1)) + 1).astype(np.int64)


def _plot_episode(
    episode_id: int,
    rows: list[tuple[str, dict]],
    previous_cuts: np.ndarray,
    gripper_events: np.ndarray,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 1, figsize=(13, max(2.6 * len(rows), 5)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (name, data) in zip(axes, rows, strict=True):
        ts = np.asarray(data["replan_ts"], dtype=np.float64)
        raw_score = np.asarray(data.get("plot_score", data["score"]), dtype=np.float64)
        positive_steps = np.diff(ts)
        positive_steps = positive_steps[positive_steps > 0]
        bar_width = float(np.median(positive_steps) * 0.8) if len(positive_steps) else 0.8
        axis.bar(
            ts,
            raw_score,
            width=bar_width,
            color="#8fa9c4",
            alpha=0.32,
            linewidth=0,
            label="raw score",
            zorder=1,
        )
        axis.plot(ts, data["smooth"], color="#d97706", linewidth=2, label="smoothed")
        axis.axhline(0.0, color="#111", linewidth=0.8, alpha=0.25)
        threshold = float(data["threshold"])
        if np.isfinite(threshold) and data.get("threshold_target", "score") == "score":
            axis.axhline(threshold, color="#666", linestyle="--", linewidth=1)
        for cut in previous_cuts:
            axis.axvline(int(cut), color="#64748b", linestyle=":", linewidth=1, alpha=0.45)
        for event in gripper_events:
            axis.axvline(int(event), color="#16a34a", linestyle="--", linewidth=1.2, alpha=0.8)
        for cut in data["boundaries"]:
            axis.axvline(int(cut), color="#dc2626", linewidth=1.7)
        unique, counts = np.unique(data["selected_k"].astype(int), return_counts=True)
        k_text = ", ".join(f"K{k}:{n}" for k, n in zip(unique, counts, strict=True))
        axis.set_title(f"{name} · {k_text}", loc="left", fontsize=9)
        axis.grid(alpha=0.18)
        axis.set_ylabel("score")
    axes[0].legend(loc="upper right", ncols=3, fontsize=8)
    axes[-1].set_xlabel(
        "episode frame (gray dotted=previous detector, green dashed=gripper event, red=new)"
    )
    fig.suptitle(f"Task-local boundary ablation · episode {episode_id}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _normalized_curve(data: dict, points: int = 201) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw/smoothed score on a common 0--100% evaluated-time grid."""
    ts = np.asarray(data["replan_ts"], dtype=np.float64)
    if len(ts) < 2 or ts[-1] <= ts[0]:
        x = np.linspace(0.0, 1.0, len(ts), dtype=np.float64)
    else:
        x = (ts - ts[0]) / (ts[-1] - ts[0])
    grid = np.linspace(0.0, 1.0, points, dtype=np.float64)
    raw = np.interp(
        grid,
        x,
        np.asarray(data.get("plot_score", data["score"]), dtype=np.float64),
    )
    smooth = np.interp(grid, x, np.asarray(data["smooth"], dtype=np.float64))
    return grid, raw, smooth


def _terminal_mask_start_percent(data: dict) -> float | None:
    """Return the first masked anchor on the normalized evaluated-time axis."""
    value = data.get("terminal_mask_start_ts")
    if value is None:
        return None
    start = float(np.asarray(value).item())
    if not np.isfinite(start):
        return None
    ts = np.asarray(data["replan_ts"], dtype=np.float64)
    if not len(ts):
        return None
    if len(ts) < 2 or ts[-1] <= ts[0]:
        index = int(np.searchsorted(ts, start, side="left"))
        return 100.0 * index / max(len(ts) - 1, 1)
    return float(
        np.clip((start - ts[0]) / (ts[-1] - ts[0]) * 100.0, 0.0, 100.0)
    )


def _plot_model_overview(
    name: str,
    entries: list[tuple[int, dict]],
    previous_cuts: dict[int, np.ndarray],
    gripper_events: dict[int, np.ndarray],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(0.0, 1.0, 201)
    curves = []
    delta_bic_curves = []
    boundary_positions = []
    previous_positions = []
    gripper_positions = []
    for episode_id, data in entries:
        _grid, _raw, smooth = _normalized_curve(data, len(grid))
        curves.append(smooth)
        cluster_ts = np.asarray(
            data.get("cluster_replan_ts", data["replan_ts"]), dtype=np.float64
        )
        if len(cluster_ts) < 2 or cluster_ts[-1] <= cluster_ts[0]:
            normalized_ts = np.linspace(0.0, 1.0, len(cluster_ts), dtype=np.float64)
        else:
            normalized_ts = (cluster_ts - cluster_ts[0]) / (
                cluster_ts[-1] - cluster_ts[0]
            )
        delta_bic_curves.append(
            np.interp(
                grid,
                normalized_ts,
                np.asarray(data["delta_bic"], dtype=np.float64),
            )
        )
        last_ts = max(float(np.asarray(data["replan_ts"])[-1]), 1.0)
        boundary_positions.append(np.asarray(data["boundaries"], dtype=float) / last_ts)
        previous_positions.append(
            np.asarray(previous_cuts.get(episode_id, np.empty(0)), dtype=float) / last_ts
        )
        gripper_positions.append(
            np.asarray(gripper_events.get(episode_id, np.empty(0)), dtype=float) / last_ts
        )
    matrix = np.asarray(curves, dtype=np.float64)
    delta_bic_matrix = np.asarray(delta_bic_curves, dtype=np.float64)
    median = np.median(matrix, axis=0)
    lower, upper = np.quantile(matrix, [0.25, 0.75], axis=0)
    row_mean = matrix.mean(axis=1, keepdims=True)
    row_std = matrix.std(axis=1, keepdims=True)
    shape_matrix = (matrix - row_mean) / np.maximum(row_std, 1e-8)

    metric_label = str(entries[0][1].get("metric_name", "score"))
    has_bic = not metric_label.startswith("prediction_loss_") and bool(
        np.any(np.isfinite(delta_bic_matrix))
    )
    if has_bic:
        figure, axes = plt.subplots(
            4,
            1,
            figsize=(14, 14),
            gridspec_kw={"height_ratios": [2.2, 2.2, 2.0, 1.8]},
            constrained_layout=True,
        )
        overlay, absolute_bic, heatmap, rug = axes
    else:
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(14, 10.5),
            gridspec_kw={"height_ratios": [2.2, 2.0, 1.8]},
            constrained_layout=True,
        )
        overlay, heatmap, rug = axes
        absolute_bic = None
    for curve in matrix:
        overlay.plot(grid * 100.0, curve, color="#6096ba", alpha=0.22, linewidth=1)
    overlay.fill_between(
        grid * 100.0, lower, upper, color="#f59e0b", alpha=0.2, label="25--75%"
    )
    overlay.plot(grid * 100.0, median, color="#d97706", linewidth=2.5, label="median")
    overlay.axhline(0.0, color="#111", linewidth=0.9, alpha=0.4)
    overlay.set_title(f"{name} · all episodes aligned by evaluated-time progress", loc="left")
    overlay.set_ylabel(metric_label)
    overlay.grid(alpha=0.18)
    overlay.legend(loc="upper right")

    if has_bic and absolute_bic is not None:
        multi_fraction = float(
            np.mean(
                np.concatenate(
                    [np.asarray(data["selected_k"]) > 1 for _episode_id, data in entries]
                )
            )
        )
        bic_median = np.nanmedian(delta_bic_matrix, axis=0)
        bic_lower, bic_upper = np.nanquantile(
            delta_bic_matrix, [0.25, 0.75], axis=0
        )
        for curve in delta_bic_matrix:
            absolute_bic.plot(
                grid * 100.0, curve, color="#7c3aed", alpha=0.18, linewidth=1
            )
        absolute_bic.fill_between(
            grid * 100.0,
            bic_lower,
            bic_upper,
            color="#8b5cf6",
            alpha=0.16,
            label="25--75%",
        )
        absolute_bic.plot(
            grid * 100.0,
            bic_median,
            color="#6d28d9",
            linewidth=2.5,
            label="median",
        )
        absolute_bic.axhline(0.0, color="#dc2626", linestyle="--", linewidth=1.2)
        absolute_bic.set_title(
            "Absolute output ΔBIC (above red zero line means K=2/3 beats K=1) "
            f"· observed K>1={multi_fraction:.1%}",
            loc="left",
        )
        absolute_bic.set_ylabel("output ΔBIC")
        absolute_bic.grid(alpha=0.18)
        absolute_bic.legend(loc="upper right")
    image = heatmap.imshow(
        shape_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
        extent=(0, 100, len(entries) - 0.5, -0.5),
    )
    heatmap.set_yticks(range(len(entries)))
    heatmap.set_yticklabels([str(episode_id) for episode_id, _data in entries], fontsize=7)
    heatmap.set_ylabel("episode")
    heatmap.set_title("Per-episode shape heatmap (each row z-normalized)", loc="left")
    figure.colorbar(image, ax=heatmap, label="within-episode z-score", pad=0.01)

    for row, (predicted, old, gripper) in enumerate(
        zip(boundary_positions, previous_positions, gripper_positions, strict=True)
    ):
        if len(old):
            rug.scatter(old * 100.0, np.full(len(old), row), color="#64748b", marker="x", s=24)
        if len(gripper):
            rug.scatter(
                gripper * 100.0,
                np.full(len(gripper), row),
                color="#16a34a",
                marker="D",
                s=22,
            )
        if len(predicted):
            rug.scatter(
                predicted * 100.0,
                np.full(len(predicted), row),
                color="#dc2626",
                marker="o",
                s=25,
            )
    rug.scatter([], [], color="#64748b", marker="x", label="previous detector cut (not GT)")
    rug.scatter([], [], color="#16a34a", marker="D", label="gripper event reference")
    rug.scatter([], [], color="#dc2626", marker="o", label="new cut")
    rug.set_xlim(0, 100)
    rug.set_ylim(len(entries) - 0.5, -0.5)
    rug.set_yticks(range(len(entries)))
    rug.set_yticklabels([str(episode_id) for episode_id, _data in entries], fontsize=7)
    rug.set_xlabel("normalized evaluated-time progress (%)")
    rug.set_ylabel("episode")
    rug.set_title("Boundary alignment across episodes", loc="left")
    rug.grid(alpha=0.18, axis="x")
    rug.legend(loc="upper right")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=145)
    plt.close(figure)


def _mean_pairwise_shape_corr(entries: list[tuple[int, dict]]) -> float:
    curves = np.asarray([_normalized_curve(data)[2] for _episode_id, data in entries])
    valid = curves.std(axis=1) > 1e-8
    curves = curves[valid]
    if len(curves) < 2:
        return float("nan")
    corr = np.corrcoef(curves)
    upper = corr[np.triu_indices(len(corr), k=1)]
    return float(np.nanmean(upper))


def _cut_reference_metrics(
    entries: list[tuple[int, dict]],
    references: dict[int, np.ndarray],
    *,
    tolerance_frames: int = 6,
) -> tuple[float, float, float]:
    """Return reference coverage, novel-cut rate, and nearest-reference distance.

    References are diagnostic markers, not ground truth.  Coverage answers how
    much of the reference segmentation was retained; novel rate answers how
    many predicted cuts have no nearby reference marker.
    """
    reference_hits: list[bool] = []
    novel_predictions: list[bool] = []
    nearest_distances: list[float] = []
    tolerance = max(0, int(tolerance_frames))
    for episode_id, data in entries:
        predicted = np.asarray(data["boundaries"], dtype=np.int64)
        reference = np.asarray(references.get(episode_id, np.empty(0)), dtype=np.int64)
        for value in reference:
            reference_hits.append(
                bool(len(predicted) and np.min(np.abs(predicted - value)) <= tolerance)
            )
            if len(predicted):
                nearest_distances.append(float(np.min(np.abs(predicted - value))))
            else:
                nearest_distances.append(float(np.asarray(data["n_frames"]).item()))
        for value in predicted:
            novel_predictions.append(
                not bool(len(reference) and np.min(np.abs(reference - value)) <= tolerance)
            )
    coverage = float(np.mean(reference_hits)) if reference_hits else float("nan")
    novel_rate = (
        float(np.mean(novel_predictions)) if novel_predictions else float("nan")
    )
    median_distance = (
        float(np.median(nearest_distances)) if nearest_distances else float("nan")
    )
    return coverage, novel_rate, median_distance


def _resample_rows(values: np.ndarray, points: int) -> np.ndarray:
    """Linearly resample a time-major feature matrix to a fixed length."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"Expected a non-empty (T,D) trajectory, got {values.shape}.")
    points = max(2, int(points))
    if len(values) == 1:
        return np.repeat(values, points, axis=0)
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target = np.linspace(0.0, 1.0, points, dtype=np.float64)
    return np.stack(
        [np.interp(target, source, values[:, dim]) for dim in range(values.shape[1])],
        axis=1,
    )


def _downsample_rows(
    values: np.ndarray, max_points: int = 192
) -> tuple[np.ndarray, np.ndarray]:
    """Keep an episode's true frame coordinates while limiting DTW cost."""
    values = np.asarray(values, dtype=np.float64)
    count = min(len(values), max(2, int(max_points)))
    indices = np.unique(
        np.rint(np.linspace(0, len(values) - 1, count)).astype(np.int64)
    )
    return values[indices], indices


def _dtw_source_to_reference(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map every source sample to a reference sample with classic DTW."""
    source = np.asarray(source, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    local = np.sum((source[:, None, :] - reference[None, :, :]) ** 2, axis=2)
    rows, columns = local.shape
    cumulative = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    cumulative[0, 0] = 0.0
    direction = np.zeros((rows, columns), dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            choices = (
                cumulative[row, column],
                cumulative[row, column + 1],
                cumulative[row + 1, column],
            )
            move = int(np.argmin(choices))
            cumulative[row + 1, column + 1] = local[row, column] + choices[move]
            direction[row, column] = move

    row, column = rows - 1, columns - 1
    path: list[tuple[int, int]] = []
    while True:
        path.append((row, column))
        if row == 0 and column == 0:
            break
        move = int(direction[row, column])
        if move == 0:
            row -= 1
            column -= 1
        elif move == 1:
            row -= 1
        else:
            column -= 1
        row = max(row, 0)
        column = max(column, 0)
    path.reverse()

    mapped: list[list[int]] = [[] for _ in range(rows)]
    for source_index, reference_index in path:
        mapped[source_index].append(reference_index)
    return np.asarray(
        [float(np.median(indices)) for indices in mapped], dtype=np.float64
    )


def _ordered_boundary_f1(
    first: np.ndarray, second: np.ndarray, *, tolerance_frames: int
) -> float:
    """One-to-one ordered boundary F1; two empty sets are perfect agreement."""
    first = np.sort(np.asarray(first, dtype=np.float64))
    second = np.sort(np.asarray(second, dtype=np.float64))
    if not len(first) and not len(second):
        return 1.0
    if not len(first) or not len(second):
        return 0.0
    tolerance = max(0.0, float(tolerance_frames))
    left = right = matches = 0
    while left < len(first) and right < len(second):
        delta = first[left] - second[right]
        if abs(delta) <= tolerance:
            matches += 1
            left += 1
            right += 1
        elif delta < -tolerance:
            left += 1
        else:
            right += 1
    return float(2 * matches / (len(first) + len(second)))


def _alignment_features(
    dataset_dir: Path,
    episode_ids: list[int],
    gripper_indices: list[int],
) -> dict[int, np.ndarray]:
    """Load relative EE proprio plus non-gripper actions for temporal alignment."""
    from skill_divider import load_data

    frame_table = load_data(dataset_dir, episode_ids)
    result: dict[int, np.ndarray] = {}
    for episode_id in episode_ids:
        episode = frame_table[frame_table["episode_index"] == episode_id].sort_values(
            "frame_index"
        )
        if episode.empty:
            continue
        states = np.stack(episode["observation.state"].values).astype(np.float64)
        actions = np.stack(episode["action"].values).astype(np.float64)
        # LIBERO's first six proprio channels describe the EE pose; omitting
        # the two gripper channels prevents grasp events from dominating DTW.
        proprio = states[:, : min(6, states.shape[1])].copy()
        proprio[:, : min(3, proprio.shape[1])] -= proprio[0, : min(3, proprio.shape[1])]
        resolved_gripper = {
            index if index >= 0 else actions.shape[1] + index
            for index in gripper_indices
        }
        action_columns = [
            index for index in range(actions.shape[1]) if index not in resolved_gripper
        ]
        action = actions[:, action_columns] if action_columns else actions[:, :0]
        result[int(episode_id)] = np.concatenate([proprio, action], axis=1)
    return result


def _task_alignment_maps(
    trajectories: dict[int, np.ndarray], episode_ids: list[int]
) -> tuple[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Choose a trajectory medoid and map every episode into its frame axis."""
    present = [episode_id for episode_id in episode_ids if episode_id in trajectories]
    if not present:
        raise ValueError("No trajectories are available for task alignment.")
    concatenated = np.concatenate([trajectories[episode_id] for episode_id in present])
    center = np.mean(concatenated, axis=0)
    scale = np.std(concatenated, axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = {
        episode_id: (trajectories[episode_id] - center) / scale
        for episode_id in present
    }
    signatures = np.stack(
        [_resample_rows(standardized[episode_id], 64).reshape(-1) for episode_id in present]
    )
    pairwise = np.sqrt(
        np.mean((signatures[:, None, :] - signatures[None, :, :]) ** 2, axis=2)
    )
    reference_id = present[int(np.argmin(np.mean(pairwise, axis=1)))]
    reference_values, reference_frames = _downsample_rows(standardized[reference_id])

    mappings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for episode_id in present:
        source_values, source_frames = _downsample_rows(standardized[episode_id])
        if episode_id == reference_id:
            mapped_frames = source_frames.astype(np.float64)
        else:
            mapped_indices = _dtw_source_to_reference(source_values, reference_values)
            mapped_frames = np.interp(
                mapped_indices,
                np.arange(len(reference_frames), dtype=np.float64),
                reference_frames.astype(np.float64),
            )
        mappings[episode_id] = (source_frames.astype(np.float64), mapped_frames)
    return reference_id, mappings


def _task_consistency_metrics(
    task_rows: dict[int, list[tuple[str, dict]]],
    mappings: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    tolerance_frames: int,
) -> dict[str, dict[str, float | int]]:
    """Compute count stability and count-conditioned boundary agreement."""
    names = [name for name, _data in next(iter(task_rows.values()))]
    result: dict[str, dict[str, float | int]] = {}
    for name in names:
        mapped_cuts: list[np.ndarray] = []
        counts: list[int] = []
        for episode_id, rows in task_rows.items():
            if episode_id not in mappings:
                continue
            cuts = np.asarray(dict(rows)[name]["boundaries"], dtype=np.float64)
            source_frames, reference_frames = mappings[episode_id]
            aligned = (
                np.interp(cuts, source_frames, reference_frames)
                if len(cuts)
                else np.empty(0, dtype=np.float64)
            )
            mapped_cuts.append(aligned)
            counts.append(len(cuts))
        median_count = float(np.median(counts)) if counts else float("nan")
        count_consistency = (
            float(np.mean(np.abs(np.asarray(counts) - median_count) <= 1.0))
            if counts
            else float("nan")
        )
        # Boundary-count repeatability is already measured separately above.
        # Compare positions only between episodes with the same boundary count,
        # so an occasional retry/failure mode does not get penalized a second
        # time merely for containing one additional boundary.
        f1_values: list[float] = []
        empty_pairs = 0
        nonempty_pairs = 0
        for left in range(len(mapped_cuts)):
            for right in range(left + 1, len(mapped_cuts)):
                if counts[left] != counts[right]:
                    continue
                if counts[left] == 0:
                    empty_pairs += 1
                else:
                    nonempty_pairs += 1
                f1_values.append(
                    _ordered_boundary_f1(
                        mapped_cuts[left],
                        mapped_cuts[right],
                        tolerance_frames=tolerance_frames,
                    )
                )
        finite_f1 = np.asarray(f1_values, dtype=np.float64)
        finite_f1 = finite_f1[np.isfinite(finite_f1)]
        result[name] = {
            "episodes": len(counts),
            "median_boundary_count": median_count,
            "count_consistency_pm1": count_consistency,
            "count_conditioned_aligned_boundary_f1": (
                float(np.mean(finite_f1)) if len(finite_f1) else float("nan")
            ),
            "count_conditioned_alignment_pairs": int(len(finite_f1)),
            "count_conditioned_empty_alignment_pairs": int(empty_pairs),
            "count_conditioned_nonempty_alignment_pairs": int(nonempty_pairs),
        }
    return result


def _merge_nearby_positions(values: np.ndarray, tolerance_frames: int) -> np.ndarray:
    """Merge duplicate peak locations contributed by equivalent score curves."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    if not len(values):
        return np.empty(0, dtype=np.float64)
    groups: list[list[float]] = [[float(values[0])]]
    for value in values[1:]:
        if float(value) - groups[-1][-1] <= float(tolerance_frames):
            groups[-1].append(float(value))
        else:
            groups.append([float(value)])
    return np.asarray([float(np.median(group)) for group in groups], dtype=np.float64)


def _persistent_curve_peak_frames(
    data: dict, *, tolerance_frames: int
) -> np.ndarray:
    """Extract threshold-free, multi-scale persistent peaks from one score curve."""
    ts = np.asarray(data["replan_ts"], dtype=np.int64)
    values = np.asarray(data["score"], dtype=np.float64)
    if len(ts) != len(values) or len(ts) < 3:
        return np.empty(0, dtype=np.float64)
    mask_start = float(np.asarray(data.get("terminal_mask_start_ts", np.nan)).item())
    valid_count = (
        int(np.searchsorted(ts, mask_start, side="left"))
        if np.isfinite(mask_start)
        else len(ts)
    )
    if valid_count < 3:
        return np.empty(0, dtype=np.float64)
    valid_ts = ts[:valid_count]
    peaks, reference_curve, _noise_floor = _persistent_local_peaks(
        values[:valid_count],
        valid_ts,
        {
            "smooth_windows": [5, 7, 9],
            "smooth_window": 7,
            "polyorder": 4,
        },
    )
    guard_frames = max(1, 2 * int(tolerance_frames))
    peaks = [
        int(index)
        for index in peaks
        if int(valid_ts[index]) - int(valid_ts[0]) > guard_frames
    ]
    peaks = _nms(
        peaks,
        np.asarray(reference_curve, dtype=np.float64),
        valid_ts,
        guard_frames,
    )
    return valid_ts[np.asarray(peaks, dtype=np.int64)].astype(np.float64)


def _majority_supported_positions(
    position_sets: list[np.ndarray],
    *,
    tolerance_frames: int,
    maximum_frame: int,
) -> np.ndarray:
    """Return temporal modes supported by a strict majority of episodes."""
    if not position_sets or maximum_frame < 0:
        return np.empty(0, dtype=np.float64)
    support = np.zeros(maximum_frame + 1, dtype=np.int64)
    tolerance = max(0, int(tolerance_frames))
    for positions in position_sets:
        episode_support = np.zeros(maximum_frame + 1, dtype=bool)
        for position in np.asarray(positions, dtype=np.float64):
            lower = max(0, int(np.ceil(float(position) - tolerance)))
            upper = min(maximum_frame, int(np.floor(float(position) + tolerance)))
            if lower <= upper:
                episode_support[lower : upper + 1] = True
        support += episode_support.astype(np.int64)
    required = len(position_sets) // 2 + 1
    active = support >= required
    consensus: list[float] = []
    index = 0
    while index < len(active):
        if not bool(active[index]):
            index += 1
            continue
        end = index + 1
        while end < len(active) and bool(active[end]):
            end += 1
        segment = support[index:end]
        maximum = int(np.max(segment))
        maxima = np.flatnonzero(segment == maximum) + index
        consensus.append(float(np.median(maxima)))
        index = end
    return np.asarray(consensus, dtype=np.float64)


def _ordered_boundary_match_counts(
    predictions: np.ndarray,
    references: np.ndarray,
    *,
    tolerance_frames: int,
) -> tuple[int, int, int]:
    """Return one-to-one ordered true-positive, false-positive, false-negative counts."""
    predictions = np.sort(np.asarray(predictions, dtype=np.float64))
    references = np.sort(np.asarray(references, dtype=np.float64))
    left = right = matches = 0
    tolerance = max(0.0, float(tolerance_frames))
    while left < len(predictions) and right < len(references):
        delta = predictions[left] - references[right]
        if abs(delta) <= tolerance:
            matches += 1
            left += 1
            right += 1
        elif delta < -tolerance:
            left += 1
        else:
            right += 1
    return matches, len(predictions) - matches, len(references) - matches


def _prf_from_counts(true_positive: int, false_positive: int, false_negative: int) -> tuple[float, float, float]:
    """Precision/recall/F1 with an empty-empty comparison treated as correct."""
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        float(true_positive / precision_denominator)
        if precision_denominator
        else 1.0
    )
    recall = (
        float(true_positive / recall_denominator) if recall_denominator else 1.0
    )
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _task_consensus_peak_metrics(
    task_rows: dict[int, list[tuple[str, dict]]],
    mappings: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    tolerance_frames: int,
    previous_boundaries: dict[int, np.ndarray] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Score detector cuts against leave-one-episode-out repeated score peaks."""
    names = [name for name, _data in next(iter(task_rows.values()))]
    if previous_boundaries is not None:
        names.append("previous_stage1_ft")
    maximum_frame = int(
        np.ceil(
            max(
                float(np.max(reference_frames))
                for _source_frames, reference_frames in mappings.values()
            )
        )
    )

    episode_candidates: dict[int, np.ndarray] = {}
    mapped_boundaries: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in names
    }
    for episode_id, rows in task_rows.items():
        if episode_id not in mappings:
            continue
        source_frames, reference_frames = mappings[episode_id]
        unique_curves: set[tuple[bytes, bytes]] = set()
        candidate_parts = []
        for name, data in rows:
            ts = np.asarray(data["replan_ts"], dtype=np.int64)
            values = np.asarray(data["score"], dtype=np.float32)
            curve_key = (ts.tobytes(), values.tobytes())
            if curve_key not in unique_curves:
                unique_curves.add(curve_key)
                peak_frames = _persistent_curve_peak_frames(
                    data, tolerance_frames=tolerance_frames
                )
                if len(peak_frames):
                    candidate_parts.append(
                        np.interp(peak_frames, source_frames, reference_frames)
                    )
            cuts = np.asarray(data["boundaries"], dtype=np.float64)
            mapped_boundaries[name][episode_id] = (
                np.interp(cuts, source_frames, reference_frames)
                if len(cuts)
                else np.empty(0, dtype=np.float64)
            )
        episode_candidates[episode_id] = _merge_nearby_positions(
            np.concatenate(candidate_parts) if candidate_parts else np.empty(0),
            tolerance_frames,
        )
        if previous_boundaries is not None:
            cuts = np.asarray(
                previous_boundaries.get(episode_id, np.empty(0)), dtype=np.float64
            )
            mapped_boundaries["previous_stage1_ft"][episode_id] = (
                np.interp(cuts, source_frames, reference_frames)
                if len(cuts)
                else np.empty(0, dtype=np.float64)
            )

    episode_ids = sorted(episode_candidates)
    loo_consensus: dict[int, np.ndarray] = {}
    for episode_id in episode_ids:
        loo_consensus[episode_id] = _majority_supported_positions(
            [
                episode_candidates[other]
                for other in episode_ids
                if other != episode_id
            ],
            tolerance_frames=tolerance_frames,
            maximum_frame=maximum_frame,
        )

    result: dict[str, dict[str, float | int]] = {}
    for name in names:
        true_positive = false_positive = false_negative = 0
        detected_count = consensus_count = 0
        exact_empty_agreements = 0
        for episode_id in episode_ids:
            predictions = mapped_boundaries[name][episode_id]
            references = loo_consensus[episode_id]
            matched, extra, missed = _ordered_boundary_match_counts(
                predictions,
                references,
                tolerance_frames=tolerance_frames,
            )
            true_positive += matched
            false_positive += extra
            false_negative += missed
            detected_count += len(predictions)
            consensus_count += len(references)
            exact_empty_agreements += int(not len(predictions) and not len(references))
        precision, recall, f1 = _prf_from_counts(
            true_positive, false_positive, false_negative
        )
        episode_count = len(episode_ids)
        result[name] = {
            "episodes": episode_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": int(true_positive),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "mean_detected_boundary_count": (
                float(detected_count / episode_count) if episode_count else float("nan")
            ),
            "mean_consensus_peak_count": (
                float(consensus_count / episode_count) if episode_count else float("nan")
            ),
            "empty_empty_episodes": int(exact_empty_agreements),
        }
    return result


def _plot_task_consistency_summary(
    *,
    output_dir: Path,
    dataset_dir: Path,
    episode_rows: dict[int, list[tuple[str, dict]]],
    task_groups: dict[int, list[int]],
    gripper_indices: list[int],
    tolerance_frames: int,
    previous_boundaries: dict[int, np.ndarray] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    """Save repeatability and cross-episode consensus-peak summaries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline_name = "previous_stage1_ft"
    metric_rows = {episode_id: list(rows) for episode_id, rows in episode_rows.items()}
    if previous_boundaries is not None:
        for episode_id, rows in metric_rows.items():
            rows.append(
                (
                    baseline_name,
                    {
                        "boundaries": np.asarray(
                            previous_boundaries.get(episode_id, np.empty(0)),
                            dtype=np.int64,
                        )
                    },
                )
            )
    names = [name for name, _data in next(iter(metric_rows.values()))]
    trajectories = _alignment_features(
        dataset_dir, sorted(metric_rows), gripper_indices
    )
    task_statistics: dict[str, dict] = {}
    consensus_task_statistics: dict[str, dict] = {}
    for task_id, episode_ids in task_groups.items():
        selected = {
            episode_id: metric_rows[episode_id]
            for episode_id in episode_ids
            if episode_id in metric_rows
        }
        selected_raw = {
            episode_id: episode_rows[episode_id]
            for episode_id in episode_ids
            if episode_id in episode_rows
        }
        reference_id, mappings = _task_alignment_maps(trajectories, list(selected))
        task_statistics[str(task_id)] = {
            "reference_episode": int(reference_id),
            "experiments": _task_consistency_metrics(
                selected, mappings, tolerance_frames=tolerance_frames
            ),
        }
        if previous_boundaries is not None:
            baseline = task_statistics[str(task_id)]["experiments"][baseline_name]
            for name, metrics in task_statistics[str(task_id)]["experiments"].items():
                metrics["delta_count_consistency_vs_previous"] = float(
                    metrics["count_consistency_pm1"]
                    - baseline["count_consistency_pm1"]
                )
                metrics["delta_count_conditioned_aligned_boundary_f1_vs_previous"] = float(
                    metrics["count_conditioned_aligned_boundary_f1"]
                    - baseline["count_conditioned_aligned_boundary_f1"]
                )

        consensus_task_statistics[str(task_id)] = {
            "reference_episode": int(reference_id),
            "experiments": _task_consensus_peak_metrics(
                selected_raw,
                mappings,
                tolerance_frames=tolerance_frames,
                previous_boundaries=(
                    {
                        episode_id: previous_boundaries.get(
                            episode_id, np.empty(0, dtype=np.int64)
                        )
                        for episode_id in selected_raw
                    }
                    if previous_boundaries is not None
                    else None
                ),
            ),
        }

    task_ids = list(task_groups)
    count_matrix = np.asarray(
        [
            [
                task_statistics[str(task_id)]["experiments"][name][
                    "count_consistency_pm1"
                ]
                for task_id in task_ids
            ]
            for name in names
        ],
        dtype=np.float64,
    )
    position_matrix = np.asarray(
        [
            [
                task_statistics[str(task_id)]["experiments"][name][
                    "count_conditioned_aligned_boundary_f1"
                ]
                for task_id in task_ids
            ]
            for name in names
        ],
        dtype=np.float64,
    )
    median_count_matrix = np.asarray(
        [
            [
                task_statistics[str(task_id)]["experiments"][name][
                    "median_boundary_count"
                ]
                for task_id in task_ids
            ]
            for name in names
        ],
        dtype=np.float64,
    )
    position_pair_matrix = np.asarray(
        [
            [
                task_statistics[str(task_id)]["experiments"][name][
                    "count_conditioned_alignment_pairs"
                ]
                for task_id in task_ids
            ]
            for name in names
        ],
        dtype=np.float64,
    )
    count_plot = np.column_stack([count_matrix, np.nanmean(count_matrix, axis=1)])
    position_plot = np.column_stack(
        [position_matrix, np.nanmean(position_matrix, axis=1)]
    )
    median_count_plot = np.column_stack(
        [median_count_matrix, np.nanmean(median_count_matrix, axis=1)]
    )
    for row, name in enumerate(names):
        count_value = float(count_plot[row, -1])
        position_value = float(position_plot[row, -1])
        pair_weights = position_pair_matrix[row]
        finite = np.isfinite(position_matrix[row]) & (pair_weights > 0)
        weighted_position_value = (
            float(np.average(position_matrix[row, finite], weights=pair_weights[finite]))
            if np.any(finite)
            else float("nan")
        )
        task_statistics.setdefault("overall", {})[name] = {
            "mean_count_consistency_pm1": count_value,
            "mean_count_conditioned_aligned_boundary_f1": position_value,
            "pair_weighted_count_conditioned_aligned_boundary_f1": (
                weighted_position_value
            ),
            "mean_task_median_boundary_count": float(median_count_plot[row, -1]),
            "total_count_conditioned_alignment_pairs": int(
                np.sum(pair_weights[finite])
            ),
        }
    if previous_boundaries is not None:
        baseline_overall = task_statistics["overall"][baseline_name]
        for metrics in task_statistics["overall"].values():
            metrics["delta_count_consistency_vs_previous"] = float(
                metrics["mean_count_consistency_pm1"]
                - baseline_overall["mean_count_consistency_pm1"]
            )
            metrics["delta_count_conditioned_aligned_boundary_f1_vs_previous"] = float(
                metrics["mean_count_conditioned_aligned_boundary_f1"]
                - baseline_overall["mean_count_conditioned_aligned_boundary_f1"]
            )
            metrics[
                "delta_pair_weighted_count_conditioned_aligned_boundary_f1_vs_previous"
            ] = float(
                metrics["pair_weighted_count_conditioned_aligned_boundary_f1"]
                - baseline_overall[
                    "pair_weighted_count_conditioned_aligned_boundary_f1"
                ]
            )
    json_path = output_dir / "consistency_metrics.json"
    json_path.write_text(
        json.dumps(
            {
                "definition": {
                    "count_consistency_pm1": (
                        "fraction of episodes within one boundary of the task median"
                    ),
                    "count_conditioned_aligned_boundary_f1": (
                        "mean pairwise one-to-one boundary F1 after DTW alignment of "
                        "relative EE proprio and non-gripper actions, comparing only "
                        "episodes with the same boundary count"
                    ),
                    "pair_weighted_count_conditioned_aligned_boundary_f1": (
                        "same position F1 pooled across tasks with each task weighted "
                        "by its number of eligible episode pairs"
                    ),
                    "tolerance_frames": int(tolerance_frames),
                    "empty_pair_policy": (
                        "two empty boundary sets receive F1=1 as consistent no-cut outputs"
                    ),
                    "baseline": (
                        baseline_name if previous_boundaries is not None else None
                    ),
                },
                "tasks": task_statistics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    consensus_names = list(
        consensus_task_statistics[str(task_ids[0])]["experiments"]
    )
    consensus_overall: dict[str, dict[str, float | int]] = {}
    for name in consensus_names:
        per_task = [
            consensus_task_statistics[str(task_id)]["experiments"][name]
            for task_id in task_ids
        ]
        true_positive = int(sum(int(item["true_positive"]) for item in per_task))
        false_positive = int(sum(int(item["false_positive"]) for item in per_task))
        false_negative = int(sum(int(item["false_negative"]) for item in per_task))
        micro_precision, micro_recall, micro_f1 = _prf_from_counts(
            true_positive, false_positive, false_negative
        )
        consensus_overall[name] = {
            "task_macro_precision": float(
                np.mean([float(item["precision"]) for item in per_task])
            ),
            "task_macro_recall": float(
                np.mean([float(item["recall"]) for item in per_task])
            ),
            "task_macro_f1": float(
                np.mean([float(item["f1"]) for item in per_task])
            ),
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "mean_detected_boundary_count": float(
                np.mean(
                    [float(item["mean_detected_boundary_count"]) for item in per_task]
                )
            ),
            "mean_consensus_peak_count": float(
                np.mean(
                    [float(item["mean_consensus_peak_count"]) for item in per_task]
                )
            ),
        }
    consensus_task_statistics["overall"] = consensus_overall
    consensus_json_path = output_dir / "consensus_peak_metrics.json"
    consensus_json_path.write_text(
        json.dumps(
            {
                "definition": {
                    "candidate_peaks": (
                        "threshold-free peaks that persist across SG windows 5/7/9 "
                        "and exceed the robust frame-difference noise floor"
                    ),
                    "episode_consensus": (
                        "leave-one-episode-out strict-majority support after DTW "
                        "alignment; unique score curves are unioned per episode"
                    ),
                    "precision": "fraction of detector cuts matching consensus peaks",
                    "recall": "fraction of consensus peaks recovered by detector cuts",
                    "f1": "harmonic mean of consensus precision and recall",
                    "tolerance_frames": int(tolerance_frames),
                    "empty_policy": (
                        "no detector cuts and no consensus peaks is perfect agreement"
                    ),
                    "baseline": (
                        baseline_name if previous_boundaries is not None else None
                    ),
                },
                "tasks": consensus_task_statistics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    columns = [f"Task {task_id}" for task_id in task_ids] + ["Mean"]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(32, max(8.0, 0.56 * len(names) + 3.0)),
        constrained_layout=True,
    )
    panels = (
        (
            axes[0],
            median_count_plot,
            "Median boundary count (collapse diagnostic)",
            None,
            "boundaries",
            ".1f",
        ),
        (
            axes[1],
            count_plot,
            "Count consistency (within task median ±1)",
            1.0,
            "higher is more consistent",
            ".2f",
        ),
        (
            axes[2],
            position_plot,
            "Count-conditioned DTW-aligned boundary consistency "
            f"(pairwise F1 @ ±{tolerance_frames}f)",
            1.0,
            "higher is more consistent",
            ".2f",
        ),
    )
    for axis, matrix, title, maximum, colorbar_label, number_format in panels:
        vmax = maximum if maximum is not None else max(1.0, float(np.nanmax(matrix)))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        axis.set_xticks(range(len(columns)), labels=columns, rotation=45, ha="right")
        axis.set_yticks(range(len(names)), labels=names, fontsize=8)
        if previous_boundaries is not None:
            axis.axhline(len(names) - 1.5, color="white", linewidth=3.0)
            axis.get_yticklabels()[-1].set_fontweight("bold")
        axis.set_title(title, loc="left")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                label = "—" if not np.isfinite(value) else format(value, number_format)
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=(
                        "white"
                        if np.isfinite(value) and value < 0.62 * vmax
                        else "black"
                    ),
                )
        figure.colorbar(image, ax=axis, shrink=0.72, label=colorbar_label)
    figure.suptitle(
        "Task-wise segmentation repeatability · previous_stage1_ft is the old split baseline "
        "(diagnostic consistency, not ground-truth accuracy)"
    )
    image_path = output_dir / "consistency_metrics.png"
    figure.savefig(image_path, dpi=145, bbox_inches="tight")
    plt.close(figure)

    overall = task_statistics["overall"]
    ranking = sorted(
        names,
        key=lambda name: float(
            overall[name]["pair_weighted_count_conditioned_aligned_boundary_f1"]
        ),
    )
    count_values = np.asarray(
        [overall[name]["mean_count_consistency_pm1"] for name in ranking],
        dtype=np.float64,
    )
    macro_position_values = np.asarray(
        [
            overall[name]["mean_count_conditioned_aligned_boundary_f1"]
            for name in ranking
        ],
        dtype=np.float64,
    )
    weighted_position_values = np.asarray(
        [
            overall[name]["pair_weighted_count_conditioned_aligned_boundary_f1"]
            for name in ranking
        ],
        dtype=np.float64,
    )
    ranking_figure, ranking_axis = plt.subplots(
        figsize=(12, max(6.0, 0.48 * len(ranking) + 2.0)),
        constrained_layout=True,
    )
    y_positions = np.arange(len(ranking), dtype=np.float64)
    bar_height = 0.24
    count_bars = ranking_axis.barh(
        y_positions + bar_height,
        count_values,
        height=bar_height,
        color="#2563eb",
        alpha=0.88,
        label="Count consistency",
    )
    macro_position_bars = ranking_axis.barh(
        y_positions,
        macro_position_values,
        height=bar_height,
        color="#f59e0b",
        alpha=0.9,
        label="Position F1 · task macro",
    )
    weighted_position_bars = ranking_axis.barh(
        y_positions - bar_height,
        weighted_position_values,
        height=bar_height,
        color="#16a34a",
        alpha=0.9,
        label="Position F1 · pair weighted",
    )
    ranking_axis.set_yticks(y_positions, ranking)
    if previous_boundaries is not None:
        ranking_axis.get_yticklabels()[ranking.index(baseline_name)].set_fontweight("bold")
    ranking_axis.set_xlim(0.0, 1.0)
    ranking_axis.set_xlabel("Consistency score (higher is better)")
    ranking_axis.set_title(
        "Model summary · sorted by pair-weighted aligned F1",
        loc="left",
    )
    ranking_axis.grid(axis="x", alpha=0.2)
    ranking_axis.legend(loc="lower right")
    for bars, values in (
        (count_bars, count_values),
        (macro_position_bars, macro_position_values),
        (weighted_position_bars, weighted_position_values),
    ):
        for bar, value in zip(bars, values, strict=True):
            ranking_axis.text(
                min(float(value) + 0.008, 0.985),
                bar.get_y() + bar.get_height() / 2.0,
                f"{float(value):.3f}",
                va="center",
                ha="left" if value < 0.94 else "right",
                fontsize=8,
            )
    ranking_path = output_dir / "model_metric_summary.png"
    ranking_figure.savefig(ranking_path, dpi=150, bbox_inches="tight")
    plt.close(ranking_figure)

    consensus_f1_matrix = np.asarray(
        [
            [
                consensus_task_statistics[str(task_id)]["experiments"][name]["f1"]
                for task_id in task_ids
            ]
            + [consensus_overall[name]["task_macro_f1"]]
            for name in consensus_names
        ],
        dtype=np.float64,
    )
    consensus_ranking = sorted(
        consensus_names,
        key=lambda name: float(consensus_overall[name]["task_macro_f1"]),
    )
    consensus_figure, consensus_axes = plt.subplots(
        1,
        2,
        figsize=(25, max(7.0, 0.58 * len(consensus_names) + 3.0)),
        constrained_layout=True,
    )
    consensus_image = consensus_axes[0].imshow(
        consensus_f1_matrix,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    consensus_axes[0].set_xticks(
        range(len(columns)), labels=columns, rotation=45, ha="right"
    )
    consensus_axes[0].set_yticks(
        range(len(consensus_names)), labels=consensus_names, fontsize=8
    )
    if previous_boundaries is not None:
        baseline_row = consensus_names.index(baseline_name)
        consensus_axes[0].axhline(baseline_row - 0.5, color="white", linewidth=3.0)
        consensus_axes[0].get_yticklabels()[baseline_row].set_fontweight("bold")
    consensus_axes[0].set_title(
        f"Leave-one-out consensus-peak F1 @ ±{tolerance_frames}f",
        loc="left",
    )
    for row in range(consensus_f1_matrix.shape[0]):
        for column in range(consensus_f1_matrix.shape[1]):
            value = float(consensus_f1_matrix[row, column])
            consensus_axes[0].text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value < 0.62 else "black",
            )
    consensus_figure.colorbar(
        consensus_image,
        ax=consensus_axes[0],
        shrink=0.72,
        label="higher captures repeated peaks more completely",
    )

    consensus_y = np.arange(len(consensus_ranking), dtype=np.float64)
    consensus_bar_height = 0.24
    consensus_series = (
        (
            "task_macro_precision",
            "Precision",
            "#2563eb",
            consensus_y + consensus_bar_height,
        ),
        ("task_macro_recall", "Recall", "#f59e0b", consensus_y),
        (
            "task_macro_f1",
            "F1",
            "#16a34a",
            consensus_y - consensus_bar_height,
        ),
    )
    for key, label, color, positions in consensus_series:
        values = np.asarray(
            [consensus_overall[name][key] for name in consensus_ranking],
            dtype=np.float64,
        )
        bars = consensus_axes[1].barh(
            positions,
            values,
            height=consensus_bar_height,
            color=color,
            alpha=0.9,
            label=label,
        )
        for bar, value in zip(bars, values, strict=True):
            consensus_axes[1].text(
                min(float(value) + 0.008, 0.985),
                bar.get_y() + bar.get_height() / 2.0,
                f"{float(value):.3f}",
                va="center",
                ha="left" if value < 0.94 else "right",
                fontsize=8,
            )
    consensus_axes[1].set_yticks(consensus_y, consensus_ranking)
    if previous_boundaries is not None:
        consensus_axes[1].get_yticklabels()[
            consensus_ranking.index(baseline_name)
        ].set_fontweight("bold")
    consensus_axes[1].set_xlim(0.0, 1.0)
    consensus_axes[1].set_xlabel("Consensus-peak score (higher is better)")
    consensus_axes[1].set_title(
        "Task-macro precision / recall / F1 · sorted by F1",
        loc="left",
    )
    consensus_axes[1].grid(axis="x", alpha=0.2)
    consensus_axes[1].legend(loc="lower right")
    consensus_figure.suptitle(
        "Cross-episode repeated-peak coverage · threshold-independent consensus target"
    )
    consensus_image_path = output_dir / "consensus_peak_metrics.png"
    consensus_figure.savefig(consensus_image_path, dpi=145, bbox_inches="tight")
    plt.close(consensus_figure)
    return (
        image_path,
        ranking_path,
        json_path,
        consensus_image_path,
        consensus_json_path,
    )


def _plot_model_episode_stack(
    name: str,
    entries: list[tuple[int, dict]],
    previous_cuts: dict[int, np.ndarray],
    gripper_events: dict[int, np.ndarray],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalized = []
    y_values = [0.0]
    for episode_id, data in entries:
        grid, raw, smooth = _normalized_curve(data)
        normalized.append((episode_id, data, grid, raw, smooth))
        y_values.extend(raw.tolist())
        y_values.extend(smooth.tolist())
        threshold = float(data["threshold"])
        if np.isfinite(threshold) and data.get("threshold_target", "score") == "score":
            y_values.append(threshold)
    y_lo, y_hi = np.quantile(np.asarray(y_values, dtype=float), [0.005, 0.995])
    padding = max((y_hi - y_lo) * 0.08, 1e-5)

    figure, axes = plt.subplots(
        len(entries), 1, figsize=(14, max(2.0 * len(entries), 8)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, (episode_id, data, grid, raw, smooth) in zip(axes, normalized, strict=True):
        original_ts = np.asarray(data["replan_ts"], dtype=np.float64)
        original_score = np.asarray(
            data.get("plot_score", data["score"]), dtype=np.float64
        )
        if len(original_ts) < 2 or original_ts[-1] <= original_ts[0]:
            original_x = np.linspace(0.0, 100.0, len(original_ts), dtype=np.float64)
            bar_width = 0.8
        else:
            original_x = (
                (original_ts - original_ts[0]) / (original_ts[-1] - original_ts[0])
            ) * 100.0
            positive_steps = np.diff(original_x)
            positive_steps = positive_steps[positive_steps > 0]
            bar_width = (
                float(np.median(positive_steps) * 0.8) if len(positive_steps) else 0.8
            )
        axis.bar(
            original_x,
            original_score,
            width=bar_width,
            color="#8fa9c4",
            alpha=0.30,
            linewidth=0,
            zorder=1,
        )
        mask_start = _terminal_mask_start_percent(data)
        smooth_for_plot = smooth.copy()
        if mask_start is not None:
            smooth_for_plot[grid * 100.0 >= mask_start] = np.nan
            axis.axvspan(
                mask_start,
                100.0,
                color="#64748b",
                alpha=0.14,
                linewidth=0,
                zorder=0,
            )
        axis.plot(grid * 100.0, smooth_for_plot, color="#d97706", linewidth=1.8)
        axis.axhline(0.0, color="#111", linewidth=0.7, alpha=0.3)
        threshold = float(data["threshold"])
        if np.isfinite(threshold) and data.get("threshold_target", "score") == "score":
            axis.axhline(threshold, color="#666", linestyle="--", linewidth=0.8)
        last_ts = max(float(np.asarray(data["replan_ts"])[-1]), 1.0)
        for cut in previous_cuts.get(episode_id, np.empty(0)):
            axis.axvline(
                float(cut) / last_ts * 100.0,
                color="#64748b",
                linestyle=":",
                alpha=0.45,
            )
        for event in gripper_events.get(episode_id, np.empty(0)):
            axis.axvline(
                float(event) / last_ts * 100.0,
                color="#16a34a",
                linestyle="--",
                alpha=0.75,
            )
        for cut in data["boundaries"]:
            axis.axvline(float(cut) / last_ts * 100.0, color="#dc2626", linewidth=1.4)
        axis.set_ylim(float(y_lo - padding), float(y_hi + padding))
        axis.set_ylabel(str(episode_id), rotation=0, labelpad=28, fontsize=8)
        axis.grid(alpha=0.14)
    axes[0].set_title(
        f"{name} · episode stack (gray shade=terminal mask, gray line=previous, "
        "green=gripper, red=new)",
        loc="left",
    )
    axes[-1].set_xlabel("normalized evaluated-time progress (%)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def _plot_experiment_curve_overlay(
    episode_rows: dict[int, list[tuple[str, dict]]],
    path: Path,
) -> None:
    """Overlay alternative smoothing curves and their own thresholds.

    This view is intended for recipes that share the same raw detector signal
    and differ only in boundary post-processing.  A solid line is the curve
    after smoothing and score-power transformation; the matching dashed line
    is the episodic threshold computed from that exact curve.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name for name, _data in next(iter(episode_rows.values()))]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, max(len(names), 2)))
    figure, axes = plt.subplots(
        len(episode_rows),
        1,
        figsize=(15, max(2.25 * len(episode_rows), 8)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    all_values = [0.0]
    normalized_rows: list[tuple[int, list[tuple[str, dict, np.ndarray, np.ndarray, np.ndarray]]]] = []
    for episode_id, rows in episode_rows.items():
        normalized = []
        for name, data in rows:
            grid, raw, smooth = _normalized_curve(data)
            normalized.append((name, data, grid, raw, smooth))
            all_values.extend(raw.tolist())
            all_values.extend(smooth.tolist())
            threshold = float(data["threshold"])
            if np.isfinite(threshold) and data.get("threshold_target", "score") == "score":
                all_values.append(threshold)
        normalized_rows.append((episode_id, normalized))
    y_lo, y_hi = np.quantile(np.asarray(all_values, dtype=np.float64), [0.005, 0.995])
    padding = max((y_hi - y_lo) * 0.08, 1e-5)

    for axis, (episode_id, rows) in zip(axes, normalized_rows, strict=True):
        first_data = rows[0][1]
        original_ts = np.asarray(first_data["replan_ts"], dtype=np.float64)
        original_score = np.asarray(
            first_data.get("plot_score", first_data["score"]), dtype=np.float64
        )
        if len(original_ts) < 2 or original_ts[-1] <= original_ts[0]:
            original_x = np.linspace(0.0, 100.0, len(original_ts), dtype=np.float64)
            bar_width = 0.8
        else:
            original_x = (
                (original_ts - original_ts[0]) / (original_ts[-1] - original_ts[0])
            ) * 100.0
            positive_steps = np.diff(original_x)
            positive_steps = positive_steps[positive_steps > 0]
            bar_width = (
                float(np.median(positive_steps) * 0.72) if len(positive_steps) else 0.8
            )
        axis.bar(
            original_x,
            original_score,
            width=bar_width,
            color="#94a3b8",
            alpha=0.22,
            linewidth=0,
            label="raw score" if axis is axes[0] else None,
            zorder=1,
        )
        for index, (name, data, grid, _raw, smooth) in enumerate(rows):
            color = colors[index]
            mask_start = _terminal_mask_start_percent(data)
            smooth_for_plot = smooth.copy()
            if mask_start is not None:
                smooth_for_plot[grid * 100.0 >= mask_start] = np.nan
            axis.plot(
                grid * 100.0,
                smooth_for_plot,
                color=color,
                linewidth=1.8,
                label=name if axis is axes[0] else None,
                zorder=3,
            )
            threshold = float(data["threshold"])
            if np.isfinite(threshold) and data.get("threshold_target", "score") == "score":
                axis.axhline(
                    threshold,
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.72,
                    label=f"{name} threshold" if axis is axes[0] else None,
                    zorder=2,
                )
        mask_starts = [
            value
            for _name, data, _grid, _raw, _smooth in rows
            if (value := _terminal_mask_start_percent(data)) is not None
        ]
        if mask_starts:
            axis.axvspan(
                min(mask_starts),
                100.0,
                color="#64748b",
                alpha=0.10,
                linewidth=0,
                zorder=0,
            )
        axis.axhline(0.0, color="#111827", linewidth=0.7, alpha=0.28)
        axis.set_ylim(float(y_lo - padding), float(y_hi + padding))
        axis.set_ylabel(str(episode_id), rotation=0, labelpad=28, fontsize=8)
        axis.grid(alpha=0.14)
    axes[0].set_title(
        "Detector overlay · solid=display curve, dashed=experiment threshold",
        loc="left",
    )
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("normalized evaluated-time progress (%)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(figure)


def _prepare_skill_frame_assets(
    *,
    dataset_dir: Path,
    output_dir: Path,
    episode_rows: dict[int, list[tuple[str, dict]]],
    image_key: str,
    thumb_width: int = 224,
) -> tuple[dict[int, dict[int, str]], str]:
    """Decode each episode once and cache frames used by any predicted segment.

    Every experiment can produce different cuts.  Saving the union of required
    frames once keeps the HTML compact while still letting every experiment
    show its own start/end pairs.
    """
    from PIL import Image
    from dp_skillset_eval import (
        _episode_row,
        _load_episodes_meta,
        _read_episode_clip,
        _resolve_image_key,
        _video_path,
    )

    try:
        episodes_meta = _load_episodes_meta(dataset_dir)
        resolved_key = _resolve_image_key(episodes_meta, image_key)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] skill-frame camera setup failed: {exc}", flush=True)
        return {}, image_key

    assets: dict[int, dict[int, str]] = {}
    root = output_dir / "skill_frames"
    for episode_id, rows in episode_rows.items():
        if not rows:
            continue
        n_frames = int(np.asarray(rows[0][1]["n_frames"]).item())
        requested = {0, max(0, n_frames - 1)}
        for _name, data in rows:
            cuts = np.asarray(data["boundaries"], dtype=np.int64)
            cuts = cuts[(cuts > 0) & (cuts < n_frames)]
            for cut in cuts.tolist():
                requested.add(int(cut))
                requested.add(int(cut) - 1)
        try:
            episode_dir = root / f"ep{episode_id:07d}"
            expected = {
                frame_index: episode_dir / f"frame{frame_index:05d}.jpg"
                for frame_index in sorted(requested)
            }
            if expected and all(path.is_file() for path in expected.values()):
                assets[episode_id] = {
                    frame_index: path.relative_to(output_dir).as_posix()
                    for frame_index, path in expected.items()
                }
                continue
            row = _episode_row(episodes_meta, episode_id)
            from_ts = float(row[f"videos/{resolved_key}/from_timestamp"])
            to_ts = float(row[f"videos/{resolved_key}/to_timestamp"])
            clip = _read_episode_clip(
                _video_path(dataset_dir, episodes_meta, episode_id, resolved_key),
                from_ts,
                to_ts,
                int(row["length"]),
            )
            if clip is None or len(clip) == 0:
                raise ValueError("decoded clip is empty")
            episode_assets: dict[int, str] = {}
            episode_dir.mkdir(parents=True, exist_ok=True)
            for frame_index in sorted(requested):
                source_index = int(np.clip(frame_index, 0, len(clip) - 1))
                path = episode_dir / f"frame{frame_index:05d}.jpg"
                if not path.is_file():
                    image = Image.fromarray(np.asarray(clip[source_index], dtype=np.uint8)[..., :3])
                    if image.width > thumb_width:
                        height = max(1, round(image.height * thumb_width / image.width))
                        image = image.resize((thumb_width, height), Image.Resampling.LANCZOS)
                    image.save(path, quality=88, optimize=True)
                episode_assets[frame_index] = path.relative_to(output_dir).as_posix()
            assets[episode_id] = episode_assets
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] skill-frame extraction failed for episode {episode_id}: {exc}",
                flush=True,
            )
    return assets, resolved_key


def _skill_frame_gallery(
    entries: list[tuple[int, dict]],
    frame_assets: dict[int, dict[int, str]],
    camera_key: str,
) -> str:
    episode_sections = []
    for episode_id, data in entries:
        assets = frame_assets.get(episode_id, {})
        if not assets:
            continue
        n_frames = int(np.asarray(data["n_frames"]).item())
        cuts = np.asarray(data["boundaries"], dtype=np.int64)
        cuts = np.unique(cuts[(cuts > 0) & (cuts < n_frames)])
        edges = [0, *cuts.astype(int).tolist(), n_frames]
        cards = []
        for skill_index, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            end_frame = max(start, end - 1)
            start_src = assets.get(start, "")
            end_src = assets.get(end_frame, "")
            if not start_src or not end_src:
                continue
            cards.append(
                "<div class='skill-pair'>"
                f"<div class='skill-pair-title'>skill {skill_index} · [{start}, {end})</div>"
                "<div class='skill-pair-images'>"
                f"<figure><img loading='lazy' src='{html.escape(start_src, quote=True)}'>"
                f"<figcaption>start · f{start}</figcaption></figure>"
                f"<span class='skill-arrow'>→</span>"
                f"<figure><img loading='lazy' src='{html.escape(end_src, quote=True)}'>"
                f"<figcaption>end · f{end_frame}</figcaption></figure>"
                "</div></div>"
            )
        episode_sections.append(
            f"<section class='skill-episode'><h3>episode {episode_id}</h3>"
            f"<div class='skill-pairs'>{''.join(cards)}</div></section>"
        )
    if not episode_sections:
        return "<p>No source frames were available.</p>"
    return (
        f"<p class='camera-label'>camera: {html.escape(camera_key)}</p>"
        + "".join(episode_sections)
    )


def _render_report(
    output_dir: Path,
    episode_rows: dict[int, list[tuple[str, dict]]],
    previous_cuts: dict[int, np.ndarray],
    gripper_events: dict[int, np.ndarray],
    recipes: list[dict],
    *,
    split_by_step: bool = False,
    split_by_group: bool = False,
    overlay_experiment_curves: bool = False,
    compare_limit: int = 6,
    compare_columns: int = 0,
    report_filename: str = "index.html",
    page_title: str = "One-model DP boundary mechanism ablation",
    frame_assets: dict[int, dict[int, str]] | None = None,
    frame_camera: str = "",
) -> Path:
    names = [name for name, _ in next(iter(episode_rows.values()))]
    if split_by_step or split_by_group:
        recipe_by_name = {str(item["name"]): item for item in recipes}
        grouped_names: dict[str, list[str]] = {}
        for name in names:
            if split_by_group:
                page = str(recipe_by_name[name].get("report_group", "other"))
            else:
                match = re.match(r"^(step\d+)_", name)
                page = match.group(1) if match else "other"
            grouped_names.setdefault(page, []).append(name)
        page_links = []
        for page, selected_names in grouped_names.items():
            selected = set(selected_names)
            selected_rows = {
                episode_id: [item for item in rows if item[0] in selected]
                for episode_id, rows in episode_rows.items()
            }
            selected_recipes = [recipe_by_name[name] for name in selected_names]
            filename = f"{_slug(page)}.html"
            _render_report(
                output_dir,
                selected_rows,
                previous_cuts,
                gripper_events,
                selected_recipes,
                split_by_step=False,
                split_by_group=False,
                overlay_experiment_curves=overlay_experiment_curves,
                compare_limit=compare_limit,
                compare_columns=compare_columns,
                report_filename=filename,
                page_title=f"DP boundary ablation · {page}",
                frame_assets=frame_assets,
                frame_camera=frame_camera,
            )
            page_links.append(
                "<a class='page-card' href='"
                f"{filename}'><strong>{html.escape(page)}</strong>"
                f"<span>{len(selected_names)} experiments</span></a>"
            )
        report = output_dir / report_filename
        report.write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>DP boundary ablation</title><style>"
            "body{font-family:Inter,system-ui,sans-serif;background:#f4f6fa;color:#172033;"
            "margin:0;padding:40px}main{max-width:900px;margin:auto}.pages{display:grid;"
            "grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px;margin-top:24px}"
            ".page-card{display:flex;flex-direction:column;gap:8px;background:white;"
            "border:1px solid #d7deea;border-radius:14px;padding:24px;color:#1d4ed8;"
            "text-decoration:none;box-shadow:0 4px 16px #17203310}.page-card strong{font-size:24px}"
            ".page-card span{color:#64748b}</style></head><body><main>"
            "<h1>DP boundary ablation</h1>"
            "<p>Experiments are separated into comparable diagnostic groups. "
            "Open a page to compare episode-aligned curves and detected cuts.</p>"
            f"<div class='pages'>{''.join(page_links)}</div></main></body></html>",
            encoding="utf-8",
        )
        return report
    show_bic_diagnostics = any(
        not str(data.get("metric_name", "")).startswith("prediction_loss_")
        and np.any(
            np.isfinite(np.asarray(data.get("delta_bic", []), dtype=np.float64))
        )
        for rows in episode_rows.values()
        for _name, data in rows
    )
    has_denoising_convergence = any(
        str(item.get("metric", "")).startswith("denoising_") for item in recipes
    )
    summaries = []
    for name in names:
        counts = [len(dict(rows)[name]["boundaries"]) for rows in episode_rows.values()]
        entries = [
            (episode_id, dict(rows)[name]) for episode_id, rows in episode_rows.items()
        ]
        previous_coverage, novel_rate, _previous_distance = _cut_reference_metrics(
            entries, previous_cuts, tolerance_frames=6
        )
        _gripper_coverage, _gripper_novel, gripper_distance = _cut_reference_metrics(
            entries, gripper_events, tolerance_frames=6
        )
        summaries.append(
            (
                name,
                float(np.mean(counts)),
                float(np.std(counts)),
                min(counts),
                max(counts),
                _mean_pairwise_shape_corr(entries),
                float(
                    np.mean(
                        np.concatenate(
                            [np.asarray(data["selected_k"]) > 1 for _episode_id, data in entries]
                        )
                    )
                ),
                previous_coverage,
                novel_rate,
                gripper_distance,
            )
        )
    table_rows = []
    for summary_index, (
        name,
        mean,
        std,
        lo,
        hi,
        corr,
        multi,
        previous_coverage,
        novel_rate,
        gripper_distance,
    ) in enumerate(summaries):
        coverage_text = "—" if not np.isfinite(previous_coverage) else f"{previous_coverage:.1%}"
        novel_text = "—" if not np.isfinite(novel_rate) else f"{novel_rate:.1%}"
        gripper_text = "—" if not np.isfinite(gripper_distance) else f"{gripper_distance:.1f}f"
        slug = _slug(name)
        multi_cell = f"<td>{multi:.1%}</td>" if show_bic_diagnostics else ""
        checked_attr = " checked" if summary_index < compare_limit else ""
        table_rows.append(
            "<tr><td><input class='compare-toggle' type='checkbox'"
            f"{checked_attr} "
            f"data-name='{html.escape(name, quote=True)}' "
            f"data-overview='model_{slug}_overview.png' "
            f"data-episodes='model_{slug}_episodes.png' "
            "aria-label='Add experiment to comparison'></td>"
            f"<td>{html.escape(name)}</td><td>{mean:.2f} ± {std:.2f}</td>"
            f"<td>{lo}</td><td>{hi}</td><td>{corr:.3f}</td>{multi_cell}"
            f"<td>{coverage_text}</td><td>{novel_text}</td><td>{gripper_text}</td></tr>"
        )
    table = "".join(table_rows)
    recipe_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['name']))}</td>"
        f"<td>{html.escape(', '.join(_experiment_probe_names(item)))}</td>"
        f"<td>{html.escape(str(item['cluster']))}</td>"
        f"<td>{html.escape(str(item['metric']))}</td>"
        f"<td>{html.escape(str(item['boundary']))}</td>"
        "</tr>"
        for item in recipes
    )
    model_sections = []
    overlay_panel = ""
    if overlay_experiment_curves and len(names) > 1:
        overlay_name = f"{Path(report_filename).stem}_curve_overlay.png"
        _plot_experiment_curve_overlay(episode_rows, output_dir / overlay_name)
        overlay_panel = (
            "<article><h2>Overlaid detector curves and thresholds</h2>"
            "<p>Solid lines are each detector's displayed score. Matching dashed "
            "lines are the thresholds used by that detector.</p>"
            f"<img loading='lazy' src='{overlay_name}'></article>"
        )
    for name in names:
        slug = _slug(name)
        entries = [
            (episode_id, dict(rows)[name]) for episode_id, rows in episode_rows.items()
        ]
        overview_name = f"model_{slug}_overview.png"
        stack_name = f"model_{slug}_episodes.png"
        _plot_model_overview(
            name, entries, previous_cuts, gripper_events, output_dir / overview_name
        )
        _plot_model_episode_stack(
            name, entries, previous_cuts, gripper_events, output_dir / stack_name
        )
        cuts = " · ".join(
            f"ep{episode_id}={data['boundaries'].astype(int).tolist()}"
            for episode_id, data in entries
        )
        convergence_values = [
            (episode_id, float(np.asarray(data["denoising_convergence_threshold_value"])))
            for episode_id, data in entries
            if "denoising_convergence_threshold_value" in data
        ]
        convergence_note = ""
        if convergence_values:
            values = np.asarray([value for _episode_id, value in convergence_values])
            per_episode = " · ".join(
                f"ep{episode_id}={value:.5f}"
                for episode_id, value in convergence_values
            )
            convergence_note = (
                "<p><strong>Convergence distance threshold:</strong> "
                f"{float(np.mean(values)):.5f} ± {float(np.std(values)):.5f} · "
                f"{per_episode}</p>"
            )
        model_sections.append(
            f"<article id='{slug}'><h2>{html.escape(name)}</h2>"
            f"{convergence_note}"
            f"<img loading='lazy' src='{overview_name}'>"
            "<details><summary>Show all episode curves with shared scales</summary>"
            f"<img loading='lazy' src='{stack_name}'><p>{cuts}</p></details>"
            "<details><summary>Show predicted skill start/end frames</summary>"
            f"{_skill_frame_gallery(entries, frame_assets or {}, frame_camera)}"
            "</details></article>"
        )
    navigation = " ".join(
        f"<a href='#{_slug(name)}'>{html.escape(name)}</a>" for name in names
    )
    report = output_dir / report_filename
    back_link = "" if report_filename == "index.html" else "<p><a href='index.html'>← Step selection</a></p>"
    bic_explanation = (
        " The second panel shows absolute output ΔBIC: only values above zero actually "
        "prefer K=2/3."
        if show_bic_diagnostics
        else ""
    )
    convergence_explanation = (
        " Denoising drift compares PCA-whitened clean-action estimates x0_hat, "
        "not noisy x_t states. correction_step is the continuous center of "
        "correction timing; knee_step is the discrete largest-drop point."
        if has_denoising_convergence
        else ""
    )
    bic_header = "<th>K&gt;1 anchors</th>" if show_bic_diagnostics else ""
    comparison_panel = (
        "<section class='compare-panel'><div class='compare-head'><div>"
        "<h2>Side-by-side comparison</h2>"
        f"<p>Select up to {compare_limit} experiments from the table below.</p></div>"
        "<div class='compare-actions'><button type='button' data-view='overview' "
        "class='view-button active'>Overview</button><button type='button' "
        "data-view='episodes' class='view-button'>Episode stacks</button>"
        "<button type='button' id='compare-clear'>Clear</button></div></div>"
        f"<p id='compare-status'>0 / {compare_limit} selected</p>"
        "<div id='compare-grid'></div></section>"
    )
    resolved_compare_columns = compare_columns or compare_limit
    comparison_script = """<script>
(() => {
  const maxCompare = __MAX_COMPARE__;
  const compareColumns = __COMPARE_COLUMNS__;
  const selected = [];
  let view = 'overview';
  const toggles = [...document.querySelectorAll('.compare-toggle')];
  const grid = document.getElementById('compare-grid');
  const status = document.getElementById('compare-status');
  toggles.filter(toggle => toggle.checked).slice(0, maxCompare).forEach(toggle =>
    selected.push({name: toggle.dataset.name, overview: toggle.dataset.overview,
                   episodes: toggle.dataset.episodes})
  );
  function render() {
    status.textContent = `${selected.length} / ${maxCompare} selected`;
    grid.style.setProperty(
      '--compare-columns', Math.max(Math.min(selected.length, compareColumns), 1)
    );
    grid.innerHTML = selected.map(item =>
      `<article class="compare-card"><h3>${item.name}</h3>` +
      `<a href="${item[view]}" target="_blank"><img src="${item[view]}" ` +
      `alt="${item.name} ${view}"></a></article>`
    ).join('');
  }
  toggles.forEach(toggle => toggle.addEventListener('change', () => {
    const index = selected.findIndex(item => item.name === toggle.dataset.name);
    if (toggle.checked && index < 0) {
      if (selected.length >= maxCompare) {
        toggle.checked = false;
        status.textContent = `Maximum ${maxCompare} experiments can be compared.`;
        return;
      }
      selected.push({name: toggle.dataset.name, overview: toggle.dataset.overview,
                     episodes: toggle.dataset.episodes});
    } else if (!toggle.checked && index >= 0) {
      selected.splice(index, 1);
    }
    render();
  }));
  document.querySelectorAll('[data-view]').forEach(button =>
    button.addEventListener('click', () => {
      view = button.dataset.view;
      document.querySelectorAll('[data-view]').forEach(item =>
        item.classList.toggle('active', item === button));
      render();
    }));
  document.getElementById('compare-clear').addEventListener('click', () => {
    selected.splice(0); toggles.forEach(toggle => { toggle.checked = false; }); render();
  });
  render();
})();
</script>""".replace("__MAX_COMPARE__", str(compare_limit)).replace(
        "__COMPARE_COLUMNS__", str(resolved_compare_columns)
    )
    report.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>DP boundary ablation</title>"
        "<style>body{font-family:Inter,system-ui,sans-serif;background:#f4f6fa;color:#172033;"
        "margin:0;padding:24px}main{max-width:1500px;margin:auto}article,table,nav{background:white;"
        "border:1px solid #d7deea;border-radius:12px;margin:16px 0;padding:14px}img{width:100%;height:auto}"
        "table{border-collapse:collapse;width:100%}th,td{padding:8px 12px;border-bottom:1px solid #e5e7eb;"
        "text-align:left}p{font:12px ui-monospace,monospace;overflow-wrap:anywhere}nav{position:sticky;"
        "top:8px;z-index:5;display:flex;gap:10px;flex-wrap:wrap}nav a{color:#1d4ed8;text-decoration:none}"
        "summary{cursor:pointer;font-weight:600;padding:10px 0}button{border:1px solid #cbd5e1;"
        "border-radius:8px;background:white;padding:8px 12px;cursor:pointer}.view-button.active{"
        "background:#1d4ed8;color:white;border-color:#1d4ed8}.compare-panel{background:white;"
        "border:1px solid #d7deea;border-radius:12px;margin:16px 0;padding:14px}.compare-head{"
        "display:flex;justify-content:space-between;gap:16px;align-items:center}.compare-head h2{"
        "margin-bottom:4px}.compare-actions{display:flex;gap:8px;flex-wrap:wrap}#compare-status{"
        "color:#64748b}#compare-grid{display:grid;grid-template-columns:repeat(var(--compare-columns),"
        "minmax(360px,1fr));gap:12px;overflow-x:auto}.compare-card{margin:0;padding:10px;"
        "min-width:0}.compare-card h3{font-size:14px;margin:0 0 8px;overflow-wrap:anywhere}"
        ".compare-card img{display:block}.camera-label{color:#64748b}.skill-episode{"
        "margin:12px 0 22px}.skill-episode h3{margin:0 0 8px}.skill-pairs{display:grid;"
        "grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:10px}.skill-pair{"
        "border:1px solid #dbe3ef;border-radius:10px;padding:8px;background:#f8fafc}"
        ".skill-pair-title{font:600 12px ui-monospace,monospace;margin-bottom:7px}"
        ".skill-pair-images{display:grid;grid-template-columns:1fr auto 1fr;gap:7px;"
        "align-items:center}.skill-pair figure{margin:0}.skill-pair figure img{width:100%;"
        "border-radius:6px;display:block}.skill-pair figcaption{font:11px ui-monospace,monospace;"
        "color:#64748b;margin-top:3px}.skill-arrow{font-size:20px;color:#64748b}</style>"
        "</head><body><main>"
        f"{back_link}<h1>{html.escape(page_title)}</h1>"
        "<p>Each model is grouped across episodes. Time is normalized by the last valid DP evaluation anchor. "
        "Gray marks are cuts from the previous detector and are not ground truth. Green marks are observed "
        "gripper-state changes used only as a reference; red marks are new cuts. The first panel is the selected "
        "experiment score (relative ΔBIC gain, cluster-center cosine divergence, or adjacent-distribution shift)."
        f"{bic_explanation}{convergence_explanation}</p>"
        f"<nav>{navigation}</nav>"
        f"{comparison_panel}{overlay_panel}"
        "<table><thead><tr><th>Compare</th><th>Experiment</th><th>boundaries/episode</th><th>min</th><th>max</th>"
        f"<th>episode shape corr</th>{bic_header}"
        "<th>previous coverage @6f</th><th>new-only cuts @6f</th>"
        "<th>gripper-ref median |Δ|</th>"
        f"</tr></thead><tbody>{table}</tbody></table>"
        "<table><thead><tr><th>Experiment</th><th>probe</th><th>clustering</th>"
        f"<th>metric</th><th>boundary rule</th></tr></thead><tbody>{recipe_rows}</tbody></table>"
        f"{''.join(model_sections)}{comparison_script}</main></body></html>",
        encoding="utf-8",
    )
    return report


def _selected_task_episode_groups(
    settings: dict,
    skillset_dir: Path,
    selected_episodes: list[int],
) -> dict[int, list[int]]:
    """Recover task-local episode groups in the user's requested ID space."""
    from dp_action_error_eval import _selected_episode_ids

    selected_set = set(int(value) for value in selected_episodes)
    groups: dict[int, list[int]] = {}
    for task_id in settings["task_ids"]:
        task_episodes = _selected_episode_ids(
            skillset_dir,
            task_ids=[int(task_id)],
            task_id_space=settings["task_id_space"],
            target_task=settings["target_task"],
            n_episodes=int(settings["n_episodes"]),
        )
        groups[int(task_id)] = [
            int(episode_id)
            for episode_id in task_episodes
            if int(episode_id) in selected_set
        ]
    return {task_id: values for task_id, values in groups.items() if values}


def _root_global_comparison_panel(
    experiments: list[dict], task_ids: list[int]
) -> str:
    """Build a landing-page viewer for the global-threshold experiments."""
    global_names = [
        str(experiment["name"])
        for experiment in experiments
        if "_global_" in str(experiment["name"])
    ]
    if not global_names or not task_ids:
        return ""
    task_options = "".join(
        f"<option value='{int(task_id)}'>Task {int(task_id)}</option>"
        for task_id in task_ids
    )
    toggles = "".join(
        "<label class='global-option'><input class='global-toggle' type='checkbox' "
        f"value='{html.escape(name, quote=True)}' checked>"
        f"<span>{html.escape(name)}</span></label>"
        for name in global_names
    )
    names_json = json.dumps(global_names, ensure_ascii=False)
    return f"""
<section class='global-compare'>
  <div class='global-head'>
    <div><h2>Global threshold comparison</h2>
    <p>Choose a task and any of the four global variants to inspect them together.</p></div>
    <div class='global-actions'>
      <label for='global-task'>Task</label>
      <select id='global-task'>{task_options}</select>
      <button type='button' data-global-view='overview' class='global-view-button active'>Overview</button>
      <button type='button' data-global-view='episodes' class='global-view-button'>Episode stacks</button>
    </div>
  </div>
  <div class='global-options'>{toggles}</div>
  <p id='global-status'></p>
  <div id='global-grid'></div>
</section>
<script>
(() => {{
  const root = document.querySelector('.global-compare');
  const experimentNames = {names_json};
  const taskSelect = root.querySelector('#global-task');
  const toggles = [...root.querySelectorAll('.global-toggle')];
  const grid = root.querySelector('#global-grid');
  const status = root.querySelector('#global-status');
  let view = 'overview';

  function render() {{
    const task = taskSelect.value;
    const selected = experimentNames.filter(name =>
      toggles.some(toggle => toggle.value === name && toggle.checked)
    );
    grid.style.setProperty('--global-columns', Math.max(selected.length, 1));
    status.textContent = `Task ${{task}} · ${{selected.length}} / ${{experimentNames.length}} selected · ${{view}}`;
    grid.innerHTML = selected.map(name => {{
      const source = `task_${{task}}/model_${{name}}_${{view}}.png`;
      return `<article class="global-card"><h3>${{name}}</h3>` +
        `<a href="${{source}}" target="_blank"><img src="${{source}}" ` +
        `alt="Task ${{task}} ${{name}} ${{view}}"></a></article>`;
    }}).join('');
    if (!selected.length) {{
      grid.innerHTML = '<p class="global-empty">Select at least one global variant.</p>';
    }}
  }}

  toggles.forEach(toggle => toggle.addEventListener('change', render));
  taskSelect.addEventListener('change', render);
  root.querySelectorAll('[data-global-view]').forEach(button =>
    button.addEventListener('click', () => {{
      view = button.dataset.globalView;
      root.querySelectorAll('[data-global-view]').forEach(item =>
        item.classList.toggle('active', item === button));
      render();
    }})
  );
  render();
}})();
</script>"""


def _render_task_split_reports(
    *,
    output_dir: Path,
    dataset_dir: Path,
    episode_rows: dict[int, list[tuple[str, dict]]],
    previous_cuts: dict[int, np.ndarray],
    gripper_events: dict[int, np.ndarray],
    experiments: list[dict],
    task_groups: dict[int, list[int]],
    image_key: str,
    split_by_step: bool = False,
    split_by_group: bool = False,
    overlay_experiment_curves: bool = False,
    compare_limit: int = 6,
    compare_columns: int = 0,
    consistency_metrics: bool = False,
    consistency_tolerance_frames: int = 6,
    gripper_indices: list[int] | None = None,
) -> Path:
    """Render one compact report per task plus a task-selection landing page."""
    links = []
    for task_id, episode_ids in task_groups.items():
        task_rows = {
            episode_id: episode_rows[episode_id]
            for episode_id in episode_ids
            if episode_id in episode_rows
        }
        if not task_rows:
            continue
        task_dir = output_dir / f"task_{task_id}"
        task_previous = {
            episode_id: previous_cuts[episode_id] for episode_id in task_rows
        }
        task_gripper = {
            episode_id: gripper_events[episode_id] for episode_id in task_rows
        }
        frame_assets, frame_camera = _prepare_skill_frame_assets(
            dataset_dir=dataset_dir,
            output_dir=task_dir,
            episode_rows=task_rows,
            image_key=image_key,
        )
        _render_report(
            task_dir,
            task_rows,
            task_previous,
            task_gripper,
            experiments,
            split_by_step=split_by_step,
            split_by_group=split_by_group,
            overlay_experiment_curves=overlay_experiment_curves,
            compare_limit=compare_limit,
            compare_columns=compare_columns,
            page_title=f"DP boundary ablation · Task {task_id}",
            frame_assets=frame_assets,
            frame_camera=frame_camera,
        )
        links.append(
            "<a class='task-card' href='"
            f"task_{task_id}/index.html'><strong>Task {task_id}</strong>"
            f"<span>{len(task_rows)} episodes</span></a>"
        )
    consistency_panel = ""
    if consistency_metrics:
        try:
            (
                image_path,
                ranking_path,
                json_path,
                consensus_image_path,
                consensus_json_path,
            ) = _plot_task_consistency_summary(
                output_dir=output_dir,
                dataset_dir=dataset_dir,
                episode_rows=episode_rows,
                task_groups=task_groups,
                gripper_indices=gripper_indices or [],
                tolerance_frames=consistency_tolerance_frames,
                previous_boundaries=previous_cuts,
            )
            consistency_panel = (
                "<section class='metrics'><h2>Task-wise segmentation consistency</h2>"
                "<p>Count consistency permits one extra/missing boundary so occasional "
                "retry behavior is not over-penalized. Position consistency compares "
                "only episode pairs with the same boundary count, using one-to-one "
                "boundary F1 after DTW alignment of relative end-effector proprio and "
                "non-gripper actions. Two empty boundary sets receive F1=1 because "
                "they agree on producing no cut. Both are repeatability diagnostics, "
                "not ground-truth segmentation accuracy. The bold previous_stage1_ft row is the "
                "old segmentation used to build the Stage1/FT data.</p>"
                "<h3>Model summary</h3>"
                "<p>Each model shows boundary-count consistency plus both task-macro "
                "and eligible-pair-weighted aligned position consistency. Rows are "
                "sorted by the pair-weighted position score. Per-task median boundary "
                "counts are shown below as a no-cut collapse diagnostic.</p>"
                f"<a href='{html.escape(ranking_path.name)}'><img src='"
                f"{html.escape(ranking_path.name)}' alt='Two-number model summary'></a>"
                "<h3>Per-task details</h3>"
                f"<a href='{html.escape(image_path.name)}'><img src='"
                f"{html.escape(image_path.name)}' alt='Task consistency metrics'></a>"
                f"<p><a href='{html.escape(json_path.name)}'>Raw metric values (JSON)</a></p>"
                "</section>"
                "<section class='metrics'><h2>Cross-episode consensus peaks</h2>"
                "<p>The target peaks are built without either mean or MAD threshold: "
                "local peaks must persist across smoothing scales and recur in a strict "
                "majority of the other DTW-aligned episodes. Precision penalizes "
                "non-repeated cuts; recall penalizes repeated peaks that a threshold "
                "misses. Each episode is evaluated against a leave-one-out consensus.</p>"
                f"<a href='{html.escape(consensus_image_path.name)}'><img src='"
                f"{html.escape(consensus_image_path.name)}' "
                "alt='Cross-episode consensus peak precision recall and F1'></a>"
                f"<p><a href='{html.escape(consensus_json_path.name)}'>"
                "Raw consensus metric values (JSON)</a></p></section>"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] consistency metric report failed: {exc}", flush=True)
            consistency_panel = (
                "<section class='metrics'><h2>Task-wise segmentation consistency</h2>"
                f"<p>Metric generation failed: {html.escape(str(exc))}</p></section>"
            )
    global_comparison_panel = _root_global_comparison_panel(
        experiments, list(task_groups)
    )
    report = output_dir / "index.html"
    report.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>DP boundary tasks</title><style>"
        "body{font-family:Inter,system-ui,sans-serif;background:#f4f6fa;color:#172033;"
        "margin:0;padding:40px}main{max-width:1500px;margin:auto}.tasks{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-top:24px}"
        ".task-card{display:flex;flex-direction:column;gap:7px;background:white;"
        "border:1px solid #d7deea;border-radius:14px;padding:22px;color:#1d4ed8;"
        "text-decoration:none;box-shadow:0 4px 16px #17203310}.task-card strong{font-size:22px}"
        ".task-card span{color:#64748b}.metrics{margin-top:28px;background:white;"
        "border:1px solid #d7deea;border-radius:14px;padding:20px}.metrics img{width:100%;"
        "height:auto}.metrics p{color:#475569;line-height:1.5}.global-compare{margin-top:28px;"
        "background:white;border:1px solid #d7deea;border-radius:14px;padding:20px}"
        ".global-head{display:flex;justify-content:space-between;align-items:center;gap:16px;"
        "flex-wrap:wrap}.global-head h2{margin-bottom:4px}.global-head p{margin-top:0;"
        "color:#475569}.global-actions,.global-options{display:flex;align-items:center;gap:10px;"
        "flex-wrap:wrap}.global-actions select,.global-actions button{border:1px solid #cbd5e1;"
        "border-radius:8px;background:white;padding:8px 12px}.global-view-button.active{"
        "background:#1d4ed8;color:white;border-color:#1d4ed8}.global-option{display:flex;"
        "align-items:center;gap:6px;background:#f8fafc;border:1px solid #dbe3ef;border-radius:8px;"
        "padding:7px 10px;font:12px ui-monospace,monospace}.global-option input{margin:0}"
        "#global-status{color:#64748b;font:12px ui-monospace,monospace}#global-grid{display:grid;"
        "grid-template-columns:repeat(var(--global-columns),minmax(320px,1fr));gap:12px;"
        "overflow-x:auto}.global-card{margin:0;padding:10px;min-width:0;border:1px solid #e2e8f0;"
        "border-radius:10px}.global-card h3{font:600 13px ui-monospace,monospace;margin:0 0 8px;"
        "overflow-wrap:anywhere}.global-card img{display:block;width:100%;height:auto}"
        ".global-empty{color:#64748b}</style></head><body><main>"
        "<h1>Task-wise DP boundary evaluation</h1>"
        "<p>Each page contains only episodes from one requested task.</p>"
        f"<div class='tasks'>{''.join(links)}</div>{global_comparison_panel}"
        f"{consistency_panel}</main></body></html>",
        encoding="utf-8",
    )
    return report


def render_cached_report(settings: dict) -> Path:
    """Rebuild plots/HTML from existing raw and clustering caches without a GPU."""
    from dp_action_error_eval import _selected_episode_ids
    from dp_skillset_eval import index_skillset, load_episode_skills

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_source_values = [
        str(value).strip()
        for value in settings.get("cache_source_dirs", [])
        if str(value).strip()
    ]
    legacy_cache_source = str(settings.get("cache_source_dir", "")).strip()
    if legacy_cache_source:
        cache_source_values.insert(0, legacy_cache_source)
    cache_source_dirs = [Path(value) for value in dict.fromkeys(cache_source_values)]
    cache_roots = [output_dir, *cache_source_dirs]

    saved_summary: dict[str, dict] = {}
    summary_paths = []
    for root in cache_roots:
        summary_path = root / "summary.json"
        if summary_path.is_file():
            summary_paths.append(summary_path)
            saved_summary.update(json.loads(summary_path.read_text(encoding="utf-8")))
    if not summary_paths:
        raise FileNotFoundError(
            "Cached summary is missing from output/cache roots: "
            + ", ".join(str(root) for root in cache_roots)
        )
    requested_episodes = _selected_episode_ids(
        Path(settings["skillset_dir"]),
        task_ids=settings["task_ids"],
        task_id_space=settings["task_id_space"],
        target_task=settings["target_task"],
        n_episodes=int(settings["n_episodes"]),
    )
    episode_ids = [
        int(episode_id)
        for episode_id in requested_episodes
        if str(int(episode_id)) in saved_summary
    ]
    missing_summaries = set(map(int, requested_episodes)) - set(episode_ids)
    if missing_summaries:
        raise FileNotFoundError(
            "Cached summaries are missing requested episodes: "
            f"{sorted(missing_summaries)}"
        )
    experiments = list(settings["experiments"])
    cluster_defs = _named(settings["cluster_variants"], "cluster")
    boundary_defs = _named(settings["boundary_variants"], "boundary")
    manifest = _load_manifest(Path(settings["skillset_dir"]))

    episode_rows: dict[int, list[tuple[str, dict]]] = {}
    for episode_id in episode_ids:
        rows = []
        for experiment in experiments:
            name = str(experiment["name"])
            cluster_name = str(experiment["cluster"])
            parts = []
            for probe_name in _experiment_probe_names(experiment):
                raw_relative = (
                    Path("raw") / _slug(probe_name) / f"ep{episode_id:07d}.npz"
                )
                cluster_relative = (
                    Path("clusters")
                    / _slug(probe_name)
                    / _slug(cluster_name)
                    / f"ep{episode_id:07d}.npz"
                )
                raw_path = next(
                    (root / raw_relative for root in cache_roots if (root / raw_relative).is_file()),
                    cache_roots[0] / raw_relative,
                )
                cluster_path = next(
                    (
                        root / cluster_relative
                        for root in cache_roots
                        if (root / cluster_relative).is_file()
                    ),
                    cache_roots[0] / cluster_relative,
                )
                if not raw_path.is_file():
                    raise FileNotFoundError(
                        f"Missing cached probe for episode={episode_id}, "
                        f"probe={probe_name}."
                    )
                with np.load(raw_path, allow_pickle=False) as raw_file:
                    raw = {
                        "replan_ts": raw_file["replan_ts"].copy(),
                        "descriptors": raw_file["descriptors"].copy(),
                        "n_frames": np.asarray(raw_file["n_frames"]).copy(),
                    }
                    for optional_key in (
                        "trajectory_xyz",
                        "action_sequence",
                        "x0_descriptor_snapshots",
                        "x0_temporal_corrections",
                        "denoising_snapshot_steps",
                        "prediction_loss_per_offset",
                        "prediction_loss_per_offset_no_gripper",
                        "prediction_loss_by_timestep",
                        "prediction_loss_timesteps",
                    ):
                        if optional_key in raw_file:
                            raw[optional_key] = raw_file[optional_key].copy()
                if cluster_path.is_file():
                    with np.load(cluster_path, allow_pickle=False) as cluster_file:
                        scores = {
                            key: cluster_file[key].copy()
                            for key in _ALL_CLUSTER_SCORE_KEYS
                            if key in cluster_file
                        }
                else:
                    # Report-only ablations can reuse expensive DP descriptor
                    # caches while introducing a new CPU-only clustering
                    # transform. Persist that derived cache in the new report
                    # folder; no policy loading or GPU inference is required.
                    scores = _cluster_scores(
                        raw["descriptors"], cluster_defs[cluster_name]
                    )
                    cluster_path = output_dir / cluster_relative
                    metadata = {
                        "schema": 2,
                        "raw_sha256": _sha256(raw_path),
                        "cluster": cluster_defs[cluster_name],
                    }
                    _cluster_cache_save(cluster_path, scores, metadata)
                    print(
                        f"[derive cached cluster] {probe_name} + {cluster_name} "
                        f"· ep{episode_id}",
                        flush=True,
                    )
                # A report-only rebuild may introduce a new derived metric that
                # was not present when summary.json was first written.  All
                # cluster scores are already cached, so only input-baseline
                # metadata (needed by delta_bic_gain) is optional here.
                summary_item = saved_summary.get(str(episode_id), {}).get(name, {})
                input_by_probe = summary_item.get("input_delta_bic_by_probe", {})
                input_delta = input_by_probe.get(
                    probe_name,
                    summary_item.get("input_delta_bic") if len(_experiment_probe_names(experiment)) == 1 else None,
                )
                input_scores = (
                    None
                    if input_delta is None
                    else {"delta_bic": np.asarray([input_delta], dtype=np.float32)}
                )
                whitening_by_probe = summary_item.get(
                    "descriptor_whitening_scale_by_probe", {}
                )
                whitening_scale = whitening_by_probe.get(probe_name)
                input_spread = summary_item.get(
                    "input_whitened_spread_by_probe", {}
                ).get(probe_name)
                # A new report-only recipe has no same-named summary entry.
                # Whitening is probe-specific rather than metric-specific, so
                # recover it from any prior experiment using this probe.
                if whitening_scale is None:
                    for prior_item in saved_summary.get(str(episode_id), {}).values():
                        candidate = prior_item.get(
                            "descriptor_whitening_scale_by_probe", {}
                        ).get(probe_name)
                        if candidate is not None:
                            whitening_scale = candidate
                            input_spread = prior_item.get(
                                "input_whitened_spread_by_probe", {}
                            ).get(probe_name, input_spread)
                            break
                parts.append(
                    _derive(
                        raw,
                        cluster_defs[cluster_name],
                        str(experiment["metric"]),
                        boundary_defs[str(experiment["boundary"])],
                        min_skill_len=int(manifest["detector"].get("min_skill_len", 10)),
                        scores=scores,
                        input_scores=input_scores,
                        descriptor_whitening_scale=(
                            None
                            if whitening_scale is None
                            else np.asarray(whitening_scale, dtype=np.float32)
                        ),
                        input_descriptor_spread=(
                            None if input_spread is None else float(input_spread)
                        ),
                    )
                )
            derived = (
                parts[0]
                if len(parts) == 1
                else _derive_consensus(
                    parts,
                    boundary_defs[str(experiment["boundary"])],
                    min_skill_len=int(manifest["detector"].get("min_skill_len", 10)),
                    **_consensus_kwargs(experiment),
                )
            )
            rows.append((name, derived))
        episode_rows[episode_id] = rows

    task_groups = _selected_task_episode_groups(
        settings, Path(settings["skillset_dir"]), episode_ids
    )
    threshold_statistics = _apply_shared_threshold_scopes(
        episode_rows,
        experiments,
        boundary_defs,
        task_groups,
        min_skill_len=int(manifest["detector"].get("min_skill_len", 10)),
    )
    (output_dir / "threshold_statistics.json").write_text(
        json.dumps(threshold_statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _episode_task, episode_files = index_skillset(Path(settings["skillset_dir"]) / "skills")
    previous_cuts = {}
    gripper_events = {}
    gripper_indices = manifest.get("action", {}).get("gripper_indices", [])
    gripper_threshold = float(manifest.get("action", {}).get("gripper_threshold", 0.0))
    for episode_id in episode_ids:
        skills, gripper_signal, _indices = load_episode_skills(
            episode_files[episode_id], gripper_indices
        )
        previous_cuts[episode_id] = np.asarray(
            [end for _start, end, _label in skills[:-1]], dtype=np.int64
        )
        gripper_events[episode_id] = _gripper_event_frames(
            gripper_signal, gripper_threshold
        )
    dataset_dir = _portable_path(str(manifest["dataset_dir"]), "dataset_filtered")
    if bool(settings.get("split_report_by_task", False)):
        report = _render_task_split_reports(
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            episode_rows=episode_rows,
            previous_cuts=previous_cuts,
            gripper_events=gripper_events,
            experiments=experiments,
            task_groups=task_groups,
            image_key=str(
                settings.get("report_image_key", "observation.images.image")
            ),
            split_by_step=bool(settings.get("split_report_by_step", False)),
            split_by_group=bool(settings.get("split_report_by_group", False)),
            overlay_experiment_curves=bool(
                settings.get("overlay_experiment_curves", False)
            ),
            compare_limit=int(settings.get("report_compare_limit", 6)),
            compare_columns=int(settings.get("report_compare_columns", 0)),
            consistency_metrics=bool(
                settings.get("report_consistency_metrics", False)
            ),
            consistency_tolerance_frames=int(
                settings.get("report_consistency_tolerance_frames", 6)
            ),
            gripper_indices=[int(value) for value in gripper_indices],
        )
    else:
        frame_assets, frame_camera = _prepare_skill_frame_assets(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            episode_rows=episode_rows,
            image_key=str(settings.get("report_image_key", "observation.images.image")),
        )
        report = _render_report(
            output_dir,
            episode_rows,
            previous_cuts,
            gripper_events,
            experiments,
            split_by_step=bool(settings.get("split_report_by_step", False)),
            split_by_group=bool(settings.get("split_report_by_group", False)),
            overlay_experiment_curves=bool(settings.get("overlay_experiment_curves", False)),
            compare_limit=int(settings.get("report_compare_limit", 6)),
            compare_columns=int(settings.get("report_compare_columns", 0)),
            frame_assets=frame_assets,
            frame_camera=frame_camera,
        )
    print(f"[dp boundary ablation] cached report rebuilt -> {report}")
    return report


def run(settings: dict) -> Path:
    import torch
    from action_manifold import NumpyActionNormalizer, resolve_indices
    from dp_action_error_eval import _selected_episode_ids
    from dp_skillset_eval import index_skillset, load_episode_skills
    from skill_divider import load_data, load_policy

    skillset_dir = Path(settings["skillset_dir"])
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_source_value = str(settings.get("cache_source_dir", "")).strip()
    cache_source_dir = Path(cache_source_value) if cache_source_value else None
    manifest = _load_manifest(skillset_dir)
    dataset_dir = _portable_path(str(manifest["dataset_dir"]), "dataset_filtered")
    policy_override = str(settings.get("policy_checkpoint_override", "") or "").strip()
    if policy_override:
        override_path = Path(policy_override).expanduser()
        if not override_path.is_absolute():
            override_path = Path(settings["project_root"]) / override_path
        policy_path = _portable_path(str(override_path), "outputs_filtered")
        manifest = dict(manifest)
        manifest["policy_path"] = str(policy_path)
    else:
        policy_path = _portable_path(str(manifest["policy_path"]), "outputs_filtered")
    episodes = _selected_episode_ids(
        skillset_dir,
        task_ids=settings["task_ids"],
        task_id_space=settings["task_id_space"],
        target_task=settings["target_task"],
        n_episodes=int(settings["n_episodes"]),
    )
    if not episodes:
        raise ValueError(
            f"No episodes selected for task_ids={settings['task_ids']} "
            f"in {settings['task_id_space']} id space."
        )
    probe_defs = _named(settings["probe_variants"], "probe")
    cluster_defs = _named(settings["cluster_variants"], "cluster")
    boundary_defs = _named(settings["boundary_variants"], "boundary")
    raw_by_probe: dict[str, dict[int, dict]] = {name: {} for name in probe_defs}
    raw_paths: dict[str, dict[int, Path]] = {name: {} for name in probe_defs}

    policy = preprocessor = normalizer = data = None
    rollout_cfg = dict(settings.get("rollout", {}))
    prediction_loss_cfg = dict(settings.get("prediction_loss", {}))

    # Policy is also needed to validate PCA/cache provenance. Loading weights is
    # cheap relative to inference and keeps stale caches from being accepted.
    policy, preprocessor = load_policy(
        str(policy_path),
        "cuda",
        str(manifest["detector"].get("noise_scheduler_type", "DDIM")),
        int(manifest["detector"].get("num_inference_steps", 10)),
    )
    if not bool(getattr(policy.config, "state_only", False)):
        raise ValueError("This first ablation implementation intentionally targets state-only DP.")
    normalizer = NumpyActionNormalizer.from_preprocessor(preprocessor)
    action_dim = int(policy.config.action_feature.shape[0])
    resolved_gripper_indices = resolve_indices(
        manifest.get("action", {}).get("gripper_indices", []), action_dim
    )
    metric_action_indices = tuple(
        index for index in range(action_dim) if index not in resolved_gripper_indices
    )
    if not metric_action_indices:
        raise ValueError("All policy action dimensions were configured as gripper channels.")
    data = load_data(dataset_dir, episode_ids=episodes)

    pca_info = {}
    for probe_index, (name, variant) in enumerate(probe_defs.items()):
        derive_from = str(variant.get("derive_from", "") or "").strip()
        if derive_from:
            if derive_from not in pca_info:
                raise ValueError(
                    f"Derived probe {name} requires earlier source probe {derive_from}."
                )
            (
                pca,
                indices,
                pca_path,
                source_directions,
                descriptor_pca,
                descriptor_indices,
                descriptor_pca_path,
            ) = pca_info[derive_from]
        else:
            pca, indices, pca_path = _resolve_pca(
                variant=variant,
                skillset_dir=skillset_dir,
                output_dir=output_dir,
                cache_source_dir=cache_source_dir,
                dataset_dir=dataset_dir,
                policy=policy,
                normalizer=normalizer,
                manifest=manifest,
                variance=float(settings["pca_variance"]),
                stride=int(settings["pca_stride"]),
            )
            descriptor_variant = dict(variant)
            descriptor_variant["mode"] = str(
                variant.get("descriptor_mode", variant.get("mode", "manifest"))
            )
            descriptor_variant["pca_representation"] = str(
                variant.get(
                    "descriptor_pca_representation",
                    variant.get("pca_representation", "mean"),
                )
            )
            if "descriptor_pca_scale_mode" in variant:
                descriptor_variant["pca_scale_mode"] = variant[
                    "descriptor_pca_scale_mode"
                ]
            if "descriptor_pca_variance" in variant:
                descriptor_variant["pca_variance"] = variant[
                    "descriptor_pca_variance"
                ]
            if "descriptor_pca_stride" in variant:
                descriptor_variant["pca_stride"] = variant["descriptor_pca_stride"]
            descriptor_pca, descriptor_indices, descriptor_pca_path = _resolve_pca(
                variant=descriptor_variant,
                skillset_dir=skillset_dir,
                output_dir=output_dir,
                cache_source_dir=cache_source_dir,
                dataset_dir=dataset_dir,
                policy=policy,
                normalizer=normalizer,
                manifest=manifest,
                variance=float(settings["pca_variance"]),
                stride=int(settings["pca_stride"]),
            )
        count = int(variant.get("count", 24))
        component_limit = (
            None
            if variant.get("pca_component_limit") is None
            else int(variant["pca_component_limit"])
        )
        direction_sampling = str(variant.get("direction_sampling", "uniform"))
        probe_generation = str(variant.get("probe_generation", "pca_offset"))
        if derive_from:
            directions = np.asarray(source_directions[:count]).copy()
        elif probe_generation == "scheduler_gaussian":
            gaussian_sampling = str(
                variant.get("gaussian_sampling", "antithetic")
            )
            gaussian_sampler = (
                _iid_gaussian_noise
                if gaussian_sampling == "iid"
                else _antithetic_gaussian_noise
            )
            directions = gaussian_sampler(
                count,
                int(policy.config.horizon),
                action_dim,
                int(variant.get("gaussian_seed", settings["seed"])),
            )
        elif probe_generation == "pca_sigma":
            directions = pca.principal_sigma_offsets(
                count,
                component_limit=component_limit,
                sigma_scale=float(variant.get("sigma_scale", 1.0)),
            )
        else:
            directions = (
                _antithetic_directions(
                    pca,
                    count,
                    int(settings["seed"]),
                    component_limit=component_limit,
                    sampling=direction_sampling,
                )
                if bool(variant.get("antithetic", False))
                else pca.sample_directions(
                    count,
                    int(settings["seed"]),
                    component_limit=component_limit,
                    sampling=direction_sampling,
                )
            )
        pca_info[name] = (
            pca,
            indices,
            pca_path,
            directions,
            descriptor_pca,
            descriptor_indices,
            descriptor_pca_path,
        )
        metadata = {
            "schema": 2,
            "policy_path": str(policy_path.resolve()),
            "pca_sha256": _sha256(pca_path),
            "descriptor_pca_sha256": _sha256(descriptor_pca_path),
            "variant": variant,
            "seed": int(settings["seed"]),
            "eval_at_step": int(
                variant.get("eval_at_step", manifest["detector"]["eval_at_step"])
            ),
            "replan_interval": int(
                variant.get(
                    "replan_interval", manifest["detector"]["replan_interval"]
                )
            ),
        }
        for episode_id in episodes:
            cache_path = output_dir / "raw" / _slug(name) / f"ep{episode_id:07d}.npz"
            cached = _cache_load(cache_path, metadata) if settings["resume"] else None
            reused_source = False
            if cached is None and settings["resume"] and cache_source_dir is not None:
                source_path = (
                    cache_source_dir / "raw" / _slug(name) / f"ep{episode_id:07d}.npz"
                )
                cached = _cache_load(source_path, metadata)
                if cached is not None:
                    _cache_save(cache_path, cached, metadata)
                    reused_source = True
            if cached is None:
                if derive_from:
                    source_payload = raw_by_probe.get(derive_from, {}).get(episode_id)
                    if source_payload is None:
                        raise RuntimeError(
                            f"Derived probe {name} cannot find {derive_from} episode "
                            f"{episode_id} in memory. Source probes must appear first."
                        )
                    print(
                        f"[derive probe {probe_index + 1}/{len(probe_defs)}] "
                        f"{name} <- {derive_from}[:{count}] · ep{episode_id}",
                        flush=True,
                    )
                    cached = _prefix_probe_payload(source_payload, count)
                else:
                    ep_df = data[data["episode_index"] == episode_id].reset_index(drop=True)
                    print(
                        f"[probe {probe_index + 1}/{len(probe_defs)}] "
                        f"{name} · ep{episode_id}",
                        flush=True,
                    )
                    cached = _probe_episode(
                        policy=policy,
                        preprocessor=preprocessor,
                        ep_df=ep_df,
                        pca=pca,
                        directions=directions,
                        variant=variant,
                        manifest=manifest,
                        normalizer=normalizer,
                        indices=indices,
                        descriptor_pca=descriptor_pca,
                        descriptor_indices=descriptor_indices,
                    )
                _cache_save(cache_path, cached, metadata)
            elif reused_source:
                print(f"[reuse source probe] {name} · ep{episode_id}", flush=True)
            else:
                print(f"[resume probe] {name} · ep{episode_id}", flush=True)
            raw_by_probe[name][episode_id] = cached
            raw_paths[name][episode_id] = cache_path

    if rollout_cfg.get("enabled"):
        probe_name = str(rollout_cfg["probe"])
        pca, indices, pca_path, _directions, *_descriptor = pca_info[probe_name]
        temporal_alignment = bool(
            rollout_cfg.get("x0_temporal_alignment", False)
        )
        metadata = {
            # Preserve compatibility with existing rollout caches when the
            # new per-offset diagnostic is not requested.
            "schema": 5 if temporal_alignment else 4,
            "policy_path": str(policy_path.resolve()),
            "pca_sha256": _sha256(pca_path),
            "samples": int(rollout_cfg.get("samples", 32)),
            "seed": int(settings["seed"]),
            "num_inference_steps": int(manifest["detector"].get("num_inference_steps", 10)),
            "descriptor_temporal_bins": int(rollout_cfg.get("descriptor_temporal_bins", 1)),
            "descriptor_pca_components": rollout_cfg.get("descriptor_pca_components"),
            "descriptor_include_endpoint": bool(
                rollout_cfg.get("descriptor_include_endpoint", False)
            ),
            "common_noise": bool(rollout_cfg.get("common_noise", False)),
            "replan_interval": int(
                rollout_cfg.get(
                    "replan_interval", manifest["detector"]["replan_interval"]
                )
            ),
            "x0_snapshot_steps": [
                int(step) for step in rollout_cfg.get("x0_snapshot_steps", [])
            ],
            "physical_xyz_trajectory": "cumulative_denormalized_action_xyz_v1",
            "action_sequence": "normalized_non_gripper_v1",
            "action_sequence_indices": list(metric_action_indices),
        }
        if temporal_alignment:
            metadata["x0_temporal_alignment"] = True
        raw_by_probe["direct_rollout"] = {}
        raw_paths["direct_rollout"] = {}
        for episode_id in episodes:
            cache_path = output_dir / "raw" / "direct_rollout" / f"ep{episode_id:07d}.npz"
            cached = _cache_load(cache_path, metadata) if settings["resume"] else None
            reused_source = False
            if cached is None and settings["resume"] and cache_source_dir is not None:
                source_path = (
                    cache_source_dir / "raw" / "direct_rollout" / f"ep{episode_id:07d}.npz"
                )
                cached = _cache_load(source_path, metadata)
                if cached is not None:
                    _cache_save(cache_path, cached, metadata)
                    reused_source = True
            if cached is None:
                ep_df = data[data["episode_index"] == episode_id].reset_index(drop=True)
                print(f"[direct rollout] ep{episode_id}", flush=True)
                cached = _rollout_episode(
                    policy=policy,
                    preprocessor=preprocessor,
                    ep_df=ep_df,
                    pca=pca,
                    normalizer=normalizer,
                    indices=indices,
                    metric_action_indices=metric_action_indices,
                    samples=int(rollout_cfg.get("samples", 32)),
                    batch_size=int(rollout_cfg.get("batch_size", 16)),
                    seed=(
                        int(settings["seed"])
                        if bool(rollout_cfg.get("common_noise", False))
                        else int(settings["seed"]) + episode_id * 1009
                    ),
                    replan_interval=int(
                        rollout_cfg.get(
                            "replan_interval",
                            manifest["detector"]["replan_interval"],
                        )
                    ),
                    descriptor_temporal_bins=int(
                        rollout_cfg.get("descriptor_temporal_bins", 1)
                    ),
                    descriptor_pca_components=(
                        None
                        if rollout_cfg.get("descriptor_pca_components") is None
                        else int(rollout_cfg["descriptor_pca_components"])
                    ),
                    descriptor_include_endpoint=bool(
                        rollout_cfg.get("descriptor_include_endpoint", False)
                    ),
                    common_noise=bool(rollout_cfg.get("common_noise", False)),
                    x0_snapshot_steps=tuple(
                        int(step)
                        for step in rollout_cfg.get("x0_snapshot_steps", [])
                    ),
                    x0_temporal_alignment=temporal_alignment,
                )
                _cache_save(cache_path, cached, metadata)
            elif reused_source:
                print(f"[reuse source rollout] ep{episode_id}", flush=True)
            raw_by_probe["direct_rollout"][episode_id] = cached
            raw_paths["direct_rollout"][episode_id] = cache_path

    if prediction_loss_cfg.get("enabled"):
        action_mode = str(manifest.get("action", {}).get("mode", "dataset"))
        if action_mode != "dataset":
            raise ValueError(
                "Prediction-loss evaluation currently requires dataset-space actions; "
                f"manifest action mode is {action_mode!r}."
            )
        loss_timesteps = tuple(
            int(step) for step in prediction_loss_cfg["noise_timesteps"]
        )
        loss_samples = int(
            prediction_loss_cfg.get("noise_samples_per_timestep", 1)
        )
        loss_interval = int(prediction_loss_cfg.get("replan_interval", 1))
        loss_common_noise = bool(prediction_loss_cfg.get("common_noise", True))
        metadata = {
            "schema": 1,
            "policy_path": str(policy_path.resolve()),
            "seed": int(settings["seed"]),
            "replan_interval": loss_interval,
            "noise_timesteps": list(loss_timesteps),
            "noise_samples_per_timestep": loss_samples,
            "common_noise": loss_common_noise,
            "score": "demonstration_denoising_residual_v1",
            "future_slice": "action_execution_start_index:action_prediction_horizon",
        }
        raw_by_probe["prediction_loss"] = {}
        raw_paths["prediction_loss"] = {}
        for episode_id in episodes:
            cache_path = (
                output_dir / "raw" / "prediction_loss" / f"ep{episode_id:07d}.npz"
            )
            cached = _cache_load(cache_path, metadata) if settings["resume"] else None
            reused_source = False
            if cached is None and settings["resume"] and cache_source_dir is not None:
                source_path = (
                    cache_source_dir
                    / "raw"
                    / "prediction_loss"
                    / f"ep{episode_id:07d}.npz"
                )
                cached = _cache_load(source_path, metadata)
                if cached is not None:
                    _cache_save(cache_path, cached, metadata)
                    reused_source = True
            if cached is None:
                ep_df = data[data["episode_index"] == episode_id].reset_index(drop=True)
                print(f"[prediction loss] ep{episode_id}", flush=True)
                cached = _prediction_loss_episode(
                    policy=policy,
                    preprocessor=preprocessor,
                    ep_df=ep_df,
                    normalizer=normalizer,
                    metric_action_indices=metric_action_indices,
                    noise_timesteps=loss_timesteps,
                    noise_samples_per_timestep=loss_samples,
                    batch_size=int(prediction_loss_cfg.get("batch_size", 64)),
                    seed=int(settings["seed"]),
                    replan_interval=loss_interval,
                    common_noise=loss_common_noise,
                )
                _cache_save(cache_path, cached, metadata)
            elif reused_source:
                print(f"[reuse source prediction loss] ep{episode_id}", flush=True)
            else:
                print(f"[resume prediction loss] ep{episode_id}", flush=True)
            raw_by_probe["prediction_loss"][episode_id] = cached
            raw_paths["prediction_loss"][episode_id] = cache_path

    del policy, preprocessor, normalizer, data
    gc.collect()
    torch.cuda.empty_cache()

    experiments = list(settings["experiments"])
    if rollout_cfg.get("enabled") and not any(
        "direct_rollout" in _experiment_probe_names(item) for item in experiments
    ):
        experiments.append(
            {
                "name": "direct_rollout",
                "probe": "direct_rollout",
                "cluster": rollout_cfg["cluster"],
                "metric": rollout_cfg["metric"],
                "boundary": rollout_cfg["boundary"],
            }
        )

    # GMM fitting dominates the CPU-only post-process. Several experiments use
    # the same probe cloud and clustering recipe but differ only in metric or
    # boundary rule. Fit each unique pair once, then persist it so later YAML
    # threshold/plot changes need no GMM work either.
    unique_pairs = list(
        dict.fromkeys(
            (probe_name, str(item["cluster"]))
            for item in experiments
            for probe_name in _experiment_probe_names(item)
        )
    )
    scores_by_pair: dict[tuple[str, str], dict[int, dict[str, np.ndarray]]] = {}
    input_scores_by_pair: dict[tuple[str, str], dict[str, np.ndarray] | None] = {}
    descriptor_whitening_scales: dict[str, np.ndarray] = {}
    input_descriptor_spreads: dict[str, float] = {}
    if rollout_cfg.get("enabled"):
        rollout_probe = str(rollout_cfg["probe"])
        rollout_pca = pca_info[rollout_probe][0]
        descriptor_whitening_scales["direct_rollout"] = (
            _descriptor_whitening_scale(rollout_pca, rollout_cfg)
        )
    total_cluster_jobs = len(unique_pairs) * len(episodes)
    cluster_job = 0
    for probe_name, cluster_name in unique_pairs:
        pair = (probe_name, cluster_name)
        scores_by_pair[pair] = {}
        cluster_cfg = cluster_defs[cluster_name]
        if probe_name in {"direct_rollout", "prediction_loss"}:
            input_scores_by_pair[pair] = None
        else:
            (
                pca,
                _indices,
                _pca_path,
                directions,
                descriptor_pca,
                _descriptor_indices,
                _descriptor_pca_path,
            ) = pca_info[probe_name]
            probe_variant = probe_defs[probe_name]
            descriptor_whitening_scales[probe_name] = _descriptor_whitening_scale(
                descriptor_pca, probe_variant
            )
            if str(probe_variant.get("probe_generation", "pca_offset")) != "pca_offset":
                # The legacy input-BIC diagnostic assumes one affine spherical
                # offset cloud represented in the same PCA coordinates.  That
                # assumption is false for trajectory-sigma probes and for
                # scheduler-noised x_t samples, so leave it intentionally
                # undefined instead of reporting a misleading number.
                input_scores_by_pair[pair] = None
                continue_to_episode_scores = True
            else:
                continue_to_episode_scores = False
            alpha = float(probe_defs[probe_name].get("alpha", 0.1))
            # Probe construction is an affine translation of this same cloud
            # at every episode anchor; BIC is translation invariant. Compute
            # the input-geometry baseline once per probe recipe.
            if not continue_to_episode_scores:
                input_cloud = _input_probe_descriptors(
                    pca,
                    directions,
                    alpha,
                    probe_defs[probe_name],
                )
                whitening_scale = descriptor_whitening_scales[probe_name]
                input_spread = float(
                    _whitened_descriptor_spread(input_cloud, whitening_scale)
                )
                input_descriptor_spreads[probe_name] = input_spread
                input_scores_by_pair[pair] = _cluster_scores(input_cloud[None], cluster_cfg)
                print(
                    f"[input BIC] {probe_name} + {cluster_name}: "
                    f"delta={float(input_scores_by_pair[pair]['delta_bic'][0]):.4f}, "
                    f"whitened_spread={input_spread:.4f}",
                    flush=True,
                )
        for episode_id in episodes:
            cluster_job += 1
            raw_path = raw_paths[probe_name][episode_id]
            cache_path = (
                output_dir / "clusters" / _slug(probe_name) / _slug(cluster_name)
                / f"ep{episode_id:07d}.npz"
            )
            metadata = {
                "schema": 2,
                "raw_sha256": _sha256(raw_path),
                "cluster": cluster_cfg,
            }
            cached_scores = (
                _cluster_cache_load(cache_path, metadata) if settings["resume"] else None
            )
            reused_cluster_source = False
            if (
                cached_scores is None
                and settings["resume"]
                and cache_source_dir is not None
            ):
                source_cluster_path = (
                    cache_source_dir
                    / "clusters"
                    / _slug(probe_name)
                    / _slug(cluster_name)
                    / f"ep{episode_id:07d}.npz"
                )
                cached_scores = _cluster_cache_load(source_cluster_path, metadata)
                if cached_scores is not None:
                    _cluster_cache_save(cache_path, cached_scores, metadata)
                    reused_cluster_source = True
            if cached_scores is None:
                if probe_name == "prediction_loss":
                    cached_scores = _empty_cluster_scores(
                        len(raw_by_probe[probe_name][episode_id]["replan_ts"])
                    )
                    print(
                        f"[skip cluster {cluster_job}/{total_cluster_jobs}] "
                        f"{probe_name} · ep{episode_id}",
                        flush=True,
                    )
                else:
                    print(
                        f"[cluster {cluster_job}/{total_cluster_jobs}] "
                        f"{probe_name} + {cluster_name} · ep{episode_id}",
                        flush=True,
                    )
                    cached_scores = _cluster_scores(
                        raw_by_probe[probe_name][episode_id]["descriptors"], cluster_cfg
                    )
                _cluster_cache_save(cache_path, cached_scores, metadata)
            elif reused_cluster_source:
                print(
                    f"[reuse source cluster {cluster_job}/{total_cluster_jobs}] "
                    f"{probe_name} + {cluster_name} · ep{episode_id}",
                    flush=True,
                )
            else:
                print(
                    f"[resume cluster {cluster_job}/{total_cluster_jobs}] "
                    f"{probe_name} + {cluster_name} · ep{episode_id}",
                    flush=True,
                )
            scores_by_pair[pair][episode_id] = cached_scores

    episode_rows = {}
    derived_summary = {}
    for episode_id in episodes:
        rows = []
        derived_summary[str(episode_id)] = {}
        for experiment in experiments:
            name = str(experiment["name"])
            probe_names = _experiment_probe_names(experiment)
            parts = [
                _derive(
                    raw_by_probe[probe_name][episode_id],
                    cluster_defs[str(experiment["cluster"])],
                    str(experiment["metric"]),
                    boundary_defs[str(experiment["boundary"])],
                    min_skill_len=int(manifest["detector"].get("min_skill_len", 10)),
                    scores=scores_by_pair[
                        (probe_name, str(experiment["cluster"]))
                    ][episode_id],
                    input_scores=input_scores_by_pair[
                        (probe_name, str(experiment["cluster"]))
                    ],
                    descriptor_whitening_scale=descriptor_whitening_scales.get(
                        probe_name
                    ),
                    input_descriptor_spread=input_descriptor_spreads.get(probe_name),
                )
                for probe_name in probe_names
            ]
            derived = (
                parts[0]
                if len(parts) == 1
                else _derive_consensus(
                    parts,
                    boundary_defs[str(experiment["boundary"])],
                    min_skill_len=int(manifest["detector"].get("min_skill_len", 10)),
                    **_consensus_kwargs(experiment),
                )
            )
            rows.append((name, derived))
            input_by_probe = {
                probe_name: (
                    None
                    if input_scores_by_pair[(probe_name, str(experiment["cluster"]))] is None
                    else float(
                        input_scores_by_pair[(probe_name, str(experiment["cluster"]))][
                            "delta_bic"
                        ][0]
                    )
                )
                for probe_name in probe_names
            }
            input_spread_by_probe = {
                probe_name: input_descriptor_spreads.get(probe_name)
                for probe_name in probe_names
            }
            whitening_scale_by_probe = {
                probe_name: descriptor_whitening_scales[probe_name].astype(float).tolist()
                for probe_name in probe_names
                if probe_name in descriptor_whitening_scales
            }
            finite_input = [value for value in input_by_probe.values() if value is not None]
            derived_summary[str(episode_id)][name] = {
                "boundaries": derived["boundaries"].astype(int).tolist(),
                "denoising_convergence_threshold_value": (
                    None
                    if "denoising_convergence_threshold_value" not in derived
                    else float(
                        np.asarray(
                            derived["denoising_convergence_threshold_value"]
                        ).item()
                    )
                ),
                "input_delta_bic": (
                    None if not finite_input else float(np.median(finite_input))
                ),
                "input_delta_bic_by_probe": input_by_probe,
                "input_whitened_spread_by_probe": input_spread_by_probe,
                "descriptor_whitening_scale_by_probe": whitening_scale_by_probe,
                "output_delta_bic_mean": float(np.mean(derived["delta_bic"])),
                "output_delta_bic_max": float(np.max(derived["delta_bic"])),
                "output_delta_bic_positive_fraction": float(
                    np.mean(derived["delta_bic"] > 0)
                ),
                "score_mean": float(np.mean(derived["score"])),
                "score_max": float(np.max(derived["score"])),
                "selected_k_histogram": {
                    str(int(k)): int(n)
                    for k, n in zip(*np.unique(derived["selected_k"].astype(int), return_counts=True), strict=True)
                },
            }
        episode_rows[episode_id] = rows

    task_groups = _selected_task_episode_groups(settings, skillset_dir, episodes)
    threshold_statistics = _apply_shared_threshold_scopes(
        episode_rows,
        experiments,
        boundary_defs,
        task_groups,
        min_skill_len=int(manifest["detector"].get("min_skill_len", 10)),
    )
    for episode_id, rows in episode_rows.items():
        for name, derived in rows:
            derived_summary[str(episode_id)][name]["boundaries"] = (
                np.asarray(derived["boundaries"]).astype(int).tolist()
            )
            derived_summary[str(episode_id)][name]["threshold"] = float(
                np.asarray(derived["threshold"]).item()
            )
            derived_summary[str(episode_id)][name]["threshold_scope"] = str(
                derived.get("threshold_scope", "episode")
            )
            if "threshold_group" in derived:
                derived_summary[str(episode_id)][name]["threshold_group"] = str(
                    derived["threshold_group"]
                )
                derived_summary[str(episode_id)][name]["threshold_pool_points"] = int(
                    derived["threshold_pool_points"]
                )
    (output_dir / "threshold_statistics.json").write_text(
        json.dumps(threshold_statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _ep_task, ep_files = index_skillset(skillset_dir / "skills")
    previous_cuts = {}
    gripper_events = {}
    gripper_indices = manifest.get("action", {}).get("gripper_indices", [])
    gripper_threshold = float(manifest.get("action", {}).get("gripper_threshold", 0.0))
    for episode_id in episodes:
        skills, gripper_signal, _indices = load_episode_skills(
            ep_files[episode_id], gripper_indices
        )
        previous_cuts[episode_id] = np.asarray(
            [end for _start, end, _label in skills[:-1]], dtype=np.int64
        )
        gripper_events[episode_id] = _gripper_event_frames(
            gripper_signal, gripper_threshold
        )
    if bool(settings.get("split_report_by_task", False)):
        report = _render_task_split_reports(
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            episode_rows=episode_rows,
            previous_cuts=previous_cuts,
            gripper_events=gripper_events,
            experiments=experiments,
            task_groups=task_groups,
            image_key=str(
                settings.get("report_image_key", "observation.images.image")
            ),
            split_by_step=bool(settings.get("split_report_by_step", False)),
            split_by_group=bool(settings.get("split_report_by_group", False)),
            overlay_experiment_curves=bool(
                settings.get("overlay_experiment_curves", False)
            ),
            compare_limit=int(settings.get("report_compare_limit", 6)),
            compare_columns=int(settings.get("report_compare_columns", 0)),
            consistency_metrics=bool(
                settings.get("report_consistency_metrics", False)
            ),
            consistency_tolerance_frames=int(
                settings.get("report_consistency_tolerance_frames", 6)
            ),
            gripper_indices=[int(value) for value in gripper_indices],
        )
    else:
        frame_assets, frame_camera = _prepare_skill_frame_assets(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            episode_rows=episode_rows,
            image_key=str(settings.get("report_image_key", "observation.images.image")),
        )
        report = _render_report(
            output_dir,
            episode_rows,
            previous_cuts,
            gripper_events,
            experiments,
            split_by_step=bool(settings.get("split_report_by_step", False)),
            split_by_group=bool(settings.get("split_report_by_group", False)),
            overlay_experiment_curves=bool(settings.get("overlay_experiment_curves", False)),
            compare_limit=int(settings.get("report_compare_limit", 6)),
            compare_columns=int(settings.get("report_compare_columns", 0)),
            frame_assets=frame_assets,
            frame_camera=frame_camera,
        )
    (output_dir / "summary.json").write_text(
        json.dumps(derived_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[dp boundary ablation] done -> {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings-json", required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.settings_json)
    if args.report_only:
        render_cached_report(settings)
    else:
        run(settings)


if __name__ == "__main__":
    main()
