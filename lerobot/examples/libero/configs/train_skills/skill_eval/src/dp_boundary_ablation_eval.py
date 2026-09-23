#!/usr/bin/env python3
"""One-model DP boundary ablation with resumable raw descriptor caches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np

LIBERO_EXAMPLES = Path(__file__).resolve().parents[4]
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


def _antithetic_directions(pca, count: int, seed: int) -> np.ndarray:
    if count % 2:
        raise ValueError("Antithetic probe count must be even.")
    half = pca.sample_directions(count // 2, seed)
    return np.concatenate([half, -half], axis=0)


def _cache_load(path: Path, metadata: dict) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["metadata_json"]) != json.dumps(metadata, sort_keys=True):
                return None
            return {
                "replan_ts": data["replan_ts"].copy(),
                "descriptors": data["descriptors"].copy(),
                "n_frames": np.asarray(data["n_frames"]).copy(),
            }
    except (KeyError, OSError, ValueError):
        return None


def _cache_save(path: Path, payload: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        replan_ts=np.asarray(payload["replan_ts"], dtype=np.int64),
        descriptors=np.asarray(payload["descriptors"], dtype=np.float32),
        n_frames=np.asarray(payload["n_frames"], dtype=np.int64),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    tmp.replace(path)


_CLUSTER_SCORE_KEYS = (
    "cosine",
    "l2",
    "hybrid",
    "selected_k",
    "center_norm",
    "bic_k1",
    "bic_best_multi",
    "delta_bic",
    "bic_multi_probability",
)


def _cluster_cache_load(path: Path, metadata: dict) -> dict[str, np.ndarray] | None:
    """Load GMM-derived curves when both the raw cloud and recipe still match."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["metadata_json"]) != json.dumps(metadata, sort_keys=True):
                return None
            return {key: data[key].copy() for key in _CLUSTER_SCORE_KEYS}
    except (KeyError, OSError, ValueError):
        return None


def _cluster_cache_save(path: Path, scores: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        **{key: np.asarray(scores[key]) for key in _CLUSTER_SCORE_KEYS},
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    tmp.replace(path)


def _resolve_pca(
    *,
    variant: dict,
    skillset_dir: Path,
    output_dir: Path,
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
    from build_skill_dataset import _fit_dataset_action_pca

    mode = str(variant.get("mode", "manifest"))
    action_dim = int(policy.config.action_feature.shape[0])
    manifest_gripper = tuple(int(v) for v in manifest.get("action", {}).get("gripper_indices", [-1]))
    gripper_indices = resolve_indices(manifest_gripper, action_dim)
    if mode == "manifest":
        path = skillset_dir / "action_probe_pca.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Manifest probe PCA is missing: {path}")
        pca = ActionPCA.load(path)
        excluded = resolve_indices(manifest.get("probe", {}).get("exclude_indices", []), action_dim)
        indices = tuple(i for i in range(action_dim) if i not in excluded)
        return pca, indices, path
    if mode not in {"std_without_gripper", "without_gripper", "std"}:
        raise ValueError(f"Unknown probe mode: {mode}")
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
    descriptor_start = int(policy.config.action_execution_start_index)
    descriptor_horizon = int(policy.config.action_prediction_horizon)
    metadata = {
        "schema": 1,
        "dataset": dataset_dir.name,
        "policy": str(Path(manifest["policy_path"]).name),
        "action_dim": action_dim,
        "action_indices": list(indices),
        "action_delta_indices": list(delta_indices),
        "descriptor_start": descriptor_start,
        "descriptor_horizon": descriptor_horizon,
        "variance": variance,
        "stride": stride,
        "scale_mode": scale_mode,
        "normalizer_mode": normalizer.mode,
    }
    path = output_dir / "pca" / f"{_slug(mode)}.npz"
    pca = get_or_fit_action_pca(
        path,
        metadata,
        lambda: _fit_dataset_action_pca(
            dataset_dir=dataset_dir,
            action_delta_indices=delta_indices,
            descriptor_start=descriptor_start,
            descriptor_horizon=descriptor_horizon,
            stride=stride,
            action_dim=action_dim,
            action_indices=indices,
            action_mode=action_mode,
            rel_mask=None,
            normalizer=normalizer,
            variance_threshold=variance,
            scale_mode=scale_mode,
            metadata=metadata,
        ),
    )
    return pca, indices, path


def _probe_episode(
    *, policy, preprocessor, ep_df, pca, directions, variant, manifest, normalizer, indices
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
        int(manifest["detector"]["eval_at_step"]),
        int(manifest["detector"]["replan_interval"]),
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
        proprio_grounding=str(manifest.get("proprio_grounding", "none")),
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
    indices,
    samples: int,
    batch_size: int,
    seed: int,
    replan_interval: int,
) -> dict[str, np.ndarray]:
    import torch
    from action_manifold import action_plan_descriptors
    from lerobot.datasets.proprio_grounding import ground_state_xyz, normalize_proprio_grounding
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
    future_horizon = int(policy.config.action_prediction_horizon)
    anchors = _valid_replan_anchors(len(states), future_horizon, int(replan_interval))
    n_obs = int(policy.config.n_obs_steps)
    parameter = next(policy.parameters())
    rng = np.random.default_rng(seed)
    clouds = []
    with torch.inference_mode():
        for anchor in anchors:
            history = list(range(max(0, anchor - n_obs + 1), anchor + 1))
            history = [history[0]] * (n_obs - len(history)) + history
            global_cond = policy.diffusion._prepare_global_conditioning(
                {OBS_STATE: state_tensor[history].unsqueeze(0).to(parameter.device)}
            )
            all_noise = rng.standard_normal(
                (samples, int(policy.config.horizon), int(policy.config.action_feature.shape[0])),
                dtype=np.float32,
            )
            outputs = []
            for start in range(0, samples, batch_size):
                noise = torch.from_numpy(all_noise[start : start + batch_size]).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                gc = global_cond.expand(len(noise), -1)
                outputs.append(
                    policy.diffusion.conditional_sample(len(noise), global_cond=gc, noise=noise)
                    .float().cpu().numpy()
                )
            chunks = np.concatenate(outputs, axis=0)
            clouds.append(
                action_plan_descriptors(
                    chunks,
                    pca,
                    action_indices=indices,
                    temporal_start=int(policy.config.action_execution_start_index),
                    temporal_length=future_horizon,
                )
            )
    return {
        "replan_ts": np.asarray(anchors, dtype=np.int64),
        "descriptors": np.asarray(clouds, dtype=np.float32),
        "n_frames": np.array(len(ep_df), dtype=np.int64),
    }


def _fit_gmm(data: np.ndarray, k: int, cfg: dict):
    from sklearn.mixture import GaussianMixture

    for reg in (1e-6, 1e-4, 1e-2):
        try:
            model = GaussianMixture(
                n_components=k,
                covariance_type=str(cfg.get("covariance", "diag")),
                n_init=int(cfg.get("n_init", 10)),
                random_state=0,
                max_iter=300,
                reg_covar=reg,
            ).fit(np.asarray(data, dtype=np.float64))
            return model
        except (ValueError, np.linalg.LinAlgError):
            continue
    return None


def _cluster_scores(clouds: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    result = {key: [] for key in _CLUSTER_SCORE_KEYS}
    for cloud in clouds:
        if str(cfg.get("selection", "fixed")) == "bic":
            models = {
                k: model
                for k in range(int(cfg.get("min_k", 1)), int(cfg.get("max_k", 3)) + 1)
                if (model := _fit_gmm(cloud, k, cfg)) is not None
            }
            bics = {k: float(model.bic(cloud)) for k, model in models.items()}
            model = models[min(bics, key=bics.get)] if bics else None
            bic_k1 = bics.get(1, float("nan"))
            multi_bics = {k: value for k, value in bics.items() if k >= 2}
            bic_best_multi = min(multi_bics.values()) if multi_bics else float("nan")
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
            bic_k1 = bic_best_multi = delta_bic = bic_multi_probability = float("nan")
        if model is None:
            cosine = l2 = 0.0
            selected_k = 0
        else:
            selected_k = int(model.n_components)
            keep = np.flatnonzero(model.weights_ >= float(cfg.get("min_weight", 0.0)))
            means = model.means_[keep]
            weights = model.weights_[keep]
            if len(means) < 2 or selected_k == 1:
                cosine = l2 = 0.0
            else:
                pairs = [(i, j) for i in range(len(means)) for j in range(i + 1, len(means))]
                pair_weights = np.asarray(
                    [weights[i] * weights[j] for i, j in pairs], dtype=np.float64
                )
                if not bool(cfg.get("weighted", True)):
                    pair_weights[:] = 1.0
                pair_weights /= pair_weights.sum()
                angular = []
                distances = []
                for i, j in pairs:
                    denom = np.linalg.norm(means[i]) * np.linalg.norm(means[j]) + 1e-8
                    angular.append(1.0 - float(np.dot(means[i], means[j]) / denom))
                    distances.append(float(np.linalg.norm(means[i] - means[j])))
                cosine = float(np.sqrt(np.sum(pair_weights * np.square(angular))))
                l2 = float(np.sqrt(np.sum(pair_weights * np.square(distances))))
        result["cosine"].append(cosine)
        result["l2"].append(l2)
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
    if mode == "mean":
        return float(np.mean(values)) * scale
    if mode == "median_mad":
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median + scale * 1.4826 * mad
    if mode == "median_mad_floor_zero":
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return max(0.0, median + scale * 1.4826 * mad)
    raise ValueError(f"Unknown threshold mode: {mode}")


def _nms(candidates: list[int], values: np.ndarray, ts: np.ndarray, distance: int) -> list[int]:
    kept = []
    for index in sorted(candidates, key=lambda i: float(values[i]), reverse=True):
        if all(abs(int(ts[index]) - int(ts[other])) > distance for other in kept):
            kept.append(index)
    return sorted(kept, key=lambda i: int(ts[i]))


def _boundaries(values: np.ndarray, ts: np.ndarray, cfg: dict) -> tuple[np.ndarray, float, np.ndarray]:
    from scipy.signal import find_peaks

    smooth = _smooth(values, int(cfg.get("smooth_window", 7)), int(cfg.get("polyorder", 4)))
    threshold = _threshold(smooth, cfg)
    method = str(cfg.get("method", "peak"))
    margin = int(cfg.get("nms_frames", 20))
    if method == "peak":
        peaks, _ = find_peaks(smooth, prominence=float(cfg.get("prominence", 0.0)))
        candidates = [int(i) for i in peaks if smooth[i] > threshold]
        # Match the dataset builder exactly: discard the protected start region
        # before NMS. Otherwise an invalid early peak can suppress the first
        # valid boundary and only then be removed itself.
        candidates = [i for i in candidates if int(ts[i]) - int(ts[0]) > margin]
        candidates = _nms(candidates, smooth, ts, margin)
    elif method == "rising":
        above = smooth > threshold
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
        candidates = [i for i in candidates if int(ts[i]) - int(ts[0]) > margin]
        candidates = _nms(candidates, smooth, ts, margin)
    else:
        raise ValueError(f"Unknown boundary method: {method}")
    return ts[candidates].astype(np.int64), threshold, np.asarray(smooth, dtype=np.float32)


def _derive(
    raw: dict,
    cluster: dict,
    metric: str,
    boundary: dict,
    *,
    min_skill_len: int,
    scores: dict[str, np.ndarray] | None = None,
    input_scores: dict[str, np.ndarray] | None = None,
) -> dict:
    if scores is None:
        scores = _cluster_scores(raw["descriptors"], cluster)
    if metric == "delta_bic_gain":
        if input_scores is None:
            raise ValueError("delta_bic_gain requires input probe BIC scores.")
        score = scores["delta_bic"] - float(input_scores["delta_bic"][0])
    else:
        score = scores[metric]
    cuts, threshold, smooth = _boundaries(score, raw["replan_ts"], boundary)
    # Match the dataset builder: merge a too-short terminal segment into the
    # previous one. The list here contains internal cuts only.
    n_frames = int(np.asarray(raw["n_frames"]).item())
    cuts_list = cuts.astype(np.int64).tolist()
    while cuts_list and n_frames - cuts_list[-1] < int(min_skill_len):
        cuts_list.pop()
    cuts = np.asarray(cuts_list, dtype=np.int64)
    return {
        **scores,
        "score": score,
        "smooth": smooth,
        "threshold": np.array(threshold, dtype=np.float32),
        "boundaries": cuts,
        "replan_ts": raw["replan_ts"],
        "n_frames": raw["n_frames"],
    }


def _plot_episode(episode_id: int, rows: list[tuple[str, dict]], existing: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 1, figsize=(13, max(2.6 * len(rows), 5)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (name, data) in zip(axes, rows, strict=True):
        ts = data["replan_ts"]
        axis.plot(ts, data["score"], color="#8fa9c4", alpha=0.45, linewidth=1, label="raw")
        axis.plot(ts, data["smooth"], color="#d97706", linewidth=2, label="smoothed")
        axis.axhline(0.0, color="#111", linewidth=0.8, alpha=0.25)
        axis.axhline(float(data["threshold"]), color="#666", linestyle="--", linewidth=1)
        for cut in existing:
            axis.axvline(int(cut), color="#111", linestyle=":", linewidth=1, alpha=0.55)
        for cut in data["boundaries"]:
            axis.axvline(int(cut), color="#dc2626", linewidth=1.7)
        unique, counts = np.unique(data["selected_k"].astype(int), return_counts=True)
        k_text = ", ".join(f"K{k}:{n}" for k, n in zip(unique, counts, strict=True))
        axis.set_title(f"{name} · {k_text}", loc="left", fontsize=9)
        axis.grid(alpha=0.18)
        axis.set_ylabel("score")
    axes[0].legend(loc="upper right", ncols=3, fontsize=8)
    axes[-1].set_xlabel("episode frame (black dotted=existing, red=new)")
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
    raw = np.interp(grid, x, np.asarray(data["score"], dtype=np.float64))
    smooth = np.interp(grid, x, np.asarray(data["smooth"], dtype=np.float64))
    return grid, raw, smooth


def _plot_model_overview(
    name: str,
    entries: list[tuple[int, dict]],
    existing: dict[int, np.ndarray],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(0.0, 1.0, 201)
    curves = []
    delta_bic_curves = []
    boundary_positions = []
    existing_positions = []
    for episode_id, data in entries:
        _grid, _raw, smooth = _normalized_curve(data, len(grid))
        curves.append(smooth)
        ts = np.asarray(data["replan_ts"], dtype=np.float64)
        if len(ts) < 2 or ts[-1] <= ts[0]:
            normalized_ts = np.linspace(0.0, 1.0, len(ts), dtype=np.float64)
        else:
            normalized_ts = (ts - ts[0]) / (ts[-1] - ts[0])
        delta_bic_curves.append(
            np.interp(
                grid,
                normalized_ts,
                np.asarray(data["delta_bic"], dtype=np.float64),
            )
        )
        last_ts = max(float(np.asarray(data["replan_ts"])[-1]), 1.0)
        boundary_positions.append(np.asarray(data["boundaries"], dtype=float) / last_ts)
        existing_positions.append(
            np.asarray(existing.get(episode_id, np.empty(0)), dtype=float) / last_ts
        )
    matrix = np.asarray(curves, dtype=np.float64)
    delta_bic_matrix = np.asarray(delta_bic_curves, dtype=np.float64)
    median = np.median(matrix, axis=0)
    lower, upper = np.quantile(matrix, [0.25, 0.75], axis=0)
    row_mean = matrix.mean(axis=1, keepdims=True)
    row_std = matrix.std(axis=1, keepdims=True)
    shape_matrix = (matrix - row_mean) / np.maximum(row_std, 1e-8)

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(14, 14),
        gridspec_kw={"height_ratios": [2.2, 2.2, 2.0, 1.8]},
        constrained_layout=True,
    )
    overlay, absolute_bic, heatmap, rug = axes
    for curve in matrix:
        overlay.plot(grid * 100.0, curve, color="#6096ba", alpha=0.22, linewidth=1)
    overlay.fill_between(
        grid * 100.0, lower, upper, color="#f59e0b", alpha=0.2, label="25--75%"
    )
    overlay.plot(grid * 100.0, median, color="#d97706", linewidth=2.5, label="median")
    overlay.axhline(0.0, color="#111", linewidth=0.9, alpha=0.4)
    overlay.set_title(f"{name} · all episodes aligned by evaluated-time progress", loc="left")
    overlay.set_ylabel("ΔBIC gain")
    overlay.grid(alpha=0.18)
    overlay.legend(loc="upper right")

    bic_median = np.median(delta_bic_matrix, axis=0)
    bic_lower, bic_upper = np.quantile(delta_bic_matrix, [0.25, 0.75], axis=0)
    for curve in delta_bic_matrix:
        absolute_bic.plot(grid * 100.0, curve, color="#7c3aed", alpha=0.18, linewidth=1)
    absolute_bic.fill_between(
        grid * 100.0,
        bic_lower,
        bic_upper,
        color="#8b5cf6",
        alpha=0.16,
        label="25--75%",
    )
    absolute_bic.plot(
        grid * 100.0, bic_median, color="#6d28d9", linewidth=2.5, label="median"
    )
    absolute_bic.axhline(0.0, color="#dc2626", linestyle="--", linewidth=1.2)
    multi_fraction = float(
        np.mean(
            np.concatenate(
                [np.asarray(data["selected_k"]) > 1 for _episode_id, data in entries]
            )
        )
    )
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

    for row, (predicted, old) in enumerate(zip(boundary_positions, existing_positions, strict=True)):
        if len(old):
            rug.scatter(old * 100.0, np.full(len(old), row), color="#111", marker="x", s=24)
        if len(predicted):
            rug.scatter(
                predicted * 100.0,
                np.full(len(predicted), row),
                color="#dc2626",
                marker="o",
                s=25,
            )
    rug.scatter([], [], color="#111", marker="x", label="existing cut")
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


def _plot_model_episode_stack(
    name: str,
    entries: list[tuple[int, dict]],
    existing: dict[int, np.ndarray],
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
        y_values.append(float(data["threshold"]))
    y_lo, y_hi = np.quantile(np.asarray(y_values, dtype=float), [0.005, 0.995])
    padding = max((y_hi - y_lo) * 0.08, 1e-5)

    figure, axes = plt.subplots(
        len(entries), 1, figsize=(14, max(2.0 * len(entries), 8)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, (episode_id, data, grid, raw, smooth) in zip(axes, normalized, strict=True):
        axis.plot(grid * 100.0, raw, color="#8fa9c4", alpha=0.4, linewidth=0.9)
        axis.plot(grid * 100.0, smooth, color="#d97706", linewidth=1.8)
        axis.axhline(0.0, color="#111", linewidth=0.7, alpha=0.3)
        axis.axhline(float(data["threshold"]), color="#666", linestyle="--", linewidth=0.8)
        last_ts = max(float(np.asarray(data["replan_ts"])[-1]), 1.0)
        for cut in existing.get(episode_id, np.empty(0)):
            axis.axvline(float(cut) / last_ts * 100.0, color="#111", linestyle=":", alpha=0.5)
        for cut in data["boundaries"]:
            axis.axvline(float(cut) / last_ts * 100.0, color="#dc2626", linewidth=1.4)
        axis.set_ylim(float(y_lo - padding), float(y_hi + padding))
        axis.set_ylabel(str(episode_id), rotation=0, labelpad=28, fontsize=8)
        axis.grid(alpha=0.14)
    axes[0].set_title(
        f"{name} · episode stack (shared x/y scales; black dotted=existing, red=new)",
        loc="left",
    )
    axes[-1].set_xlabel("normalized evaluated-time progress (%)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def _render_report(
    output_dir: Path,
    episode_rows: dict[int, list[tuple[str, dict]]],
    existing: dict[int, np.ndarray],
    recipes: list[dict],
) -> Path:
    names = [name for name, _ in next(iter(episode_rows.values()))]
    summaries = []
    for name in names:
        counts = [len(dict(rows)[name]["boundaries"]) for rows in episode_rows.values()]
        entries = [
            (episode_id, dict(rows)[name]) for episode_id, rows in episode_rows.items()
        ]
        summaries.append(
            (
                name,
                float(np.mean(counts)),
                float(np.std(counts)),
                min(counts),
                max(counts),
                _mean_pairwise_shape_corr(entries),
            )
        )
    table = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{mean:.2f} ± {std:.2f}</td>"
        f"<td>{lo}</td><td>{hi}</td><td>{corr:.3f}</td></tr>"
        for name, mean, std, lo, hi, corr in summaries
    )
    recipe_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['name']))}</td>"
        f"<td>{html.escape(str(item['probe']))}</td>"
        f"<td>{html.escape(str(item['cluster']))}</td>"
        f"<td>{html.escape(str(item['metric']))}</td>"
        f"<td>{html.escape(str(item['boundary']))}</td>"
        "</tr>"
        for item in recipes
    )
    model_sections = []
    for name in names:
        slug = _slug(name)
        entries = [
            (episode_id, dict(rows)[name]) for episode_id, rows in episode_rows.items()
        ]
        overview_name = f"model_{slug}_overview.png"
        stack_name = f"model_{slug}_episodes.png"
        _plot_model_overview(name, entries, existing, output_dir / overview_name)
        _plot_model_episode_stack(name, entries, existing, output_dir / stack_name)
        cuts = " · ".join(
            f"ep{episode_id}={data['boundaries'].astype(int).tolist()}"
            for episode_id, data in entries
        )
        model_sections.append(
            f"<article id='{slug}'><h2>{html.escape(name)}</h2>"
            f"<img loading='lazy' src='{overview_name}'>"
            "<details><summary>Show all 20 episode curves with shared scales</summary>"
            f"<img loading='lazy' src='{stack_name}'><p>{cuts}</p></details></article>"
        )
    navigation = " ".join(
        f"<a href='#{_slug(name)}'>{html.escape(name)}</a>" for name in names
    )
    report = output_dir / "index.html"
    report.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>DP boundary ablation</title>"
        "<style>body{font-family:Inter,system-ui,sans-serif;background:#f4f6fa;color:#172033;"
        "margin:0;padding:24px}main{max-width:1500px;margin:auto}article,table,nav{background:white;"
        "border:1px solid #d7deea;border-radius:12px;margin:16px 0;padding:14px}img{width:100%;height:auto}"
        "table{border-collapse:collapse;width:100%}th,td{padding:8px 12px;border-bottom:1px solid #e5e7eb;"
        "text-align:left}p{font:12px ui-monospace,monospace;overflow-wrap:anywhere}nav{position:sticky;"
        "top:8px;z-index:5;display:flex;gap:10px;flex-wrap:wrap}nav a{color:#1d4ed8;text-decoration:none}"
        "summary{cursor:pointer;font-weight:600;padding:10px 0}</style></head><body><main>"
        "<h1>One-model DP boundary mechanism ablation</h1>"
        "<p>Each model is grouped across episodes. Time is normalized by the last valid DP evaluation anchor. "
        "Black marks are existing skillset cuts; red marks are new cuts. The first panel is the relative "
        "ΔBIC gain from the input probe cloud; it can be positive even while K=1 remains best. The second "
        "panel shows absolute output ΔBIC: only values above zero actually prefer K=2/3.</p>"
        f"<nav>{navigation}</nav>"
        "<table><thead><tr><th>Experiment</th><th>boundaries/episode</th><th>min</th><th>max</th>"
        "<th>episode shape corr</th>"
        f"</tr></thead><tbody>{table}</tbody></table>"
        "<table><thead><tr><th>Experiment</th><th>probe</th><th>clustering</th>"
        f"<th>metric</th><th>boundary rule</th></tr></thead><tbody>{recipe_rows}</tbody></table>"
        f"{''.join(model_sections)}</main></body></html>",
        encoding="utf-8",
    )
    return report


def render_cached_report(settings: dict) -> Path:
    """Rebuild plots/HTML from existing raw and clustering caches without a GPU."""
    from dp_skillset_eval import index_skillset, load_episode_skills

    output_dir = Path(settings["output_dir"])
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Cached summary is missing: {summary_path}")
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episode_ids = sorted(int(value) for value in saved_summary)
    experiments = list(settings["experiments"])
    probe_defs = _named(settings["probe_variants"], "probe")
    cluster_defs = _named(settings["cluster_variants"], "cluster")
    boundary_defs = _named(settings["boundary_variants"], "boundary")
    manifest = _load_manifest(Path(settings["skillset_dir"]))

    episode_rows: dict[int, list[tuple[str, dict]]] = {}
    for episode_id in episode_ids:
        rows = []
        for experiment in experiments:
            name = str(experiment["name"])
            probe_name = str(experiment["probe"])
            cluster_name = str(experiment["cluster"])
            raw_path = output_dir / "raw" / _slug(probe_name) / f"ep{episode_id:07d}.npz"
            cluster_path = (
                output_dir
                / "clusters"
                / _slug(probe_name)
                / _slug(cluster_name)
                / f"ep{episode_id:07d}.npz"
            )
            with np.load(raw_path, allow_pickle=False) as raw_file:
                raw = {
                    "replan_ts": raw_file["replan_ts"].copy(),
                    "descriptors": raw_file["descriptors"].copy(),
                    "n_frames": np.asarray(raw_file["n_frames"]).copy(),
                }
            with np.load(cluster_path, allow_pickle=False) as cluster_file:
                scores = {key: cluster_file[key].copy() for key in _CLUSTER_SCORE_KEYS}
            input_delta = saved_summary[str(episode_id)][name].get("input_delta_bic")
            input_scores = (
                None
                if input_delta is None
                else {"delta_bic": np.asarray([input_delta], dtype=np.float32)}
            )
            rows.append(
                (
                    name,
                    _derive(
                        raw,
                        cluster_defs[cluster_name],
                        str(experiment["metric"]),
                        boundary_defs[str(experiment["boundary"])],
                        min_skill_len=int(manifest["detector"].get("min_skill_len", 10)),
                        scores=scores,
                        input_scores=input_scores,
                    ),
                )
            )
        episode_rows[episode_id] = rows

    _episode_task, episode_files = index_skillset(Path(settings["skillset_dir"]) / "skills")
    existing = {}
    for episode_id in episode_ids:
        skills, _gripper, _indices = load_episode_skills(episode_files[episode_id])
        existing[episode_id] = np.asarray(
            [end for _start, end, _label in skills[:-1]], dtype=np.int64
        )
    report = _render_report(output_dir, episode_rows, existing, experiments)
    print(f"[dp boundary ablation] cached report rebuilt -> {report}")
    return report


def run(settings: dict) -> Path:
    import torch
    from action_manifold import NumpyActionNormalizer
    from dp_action_error_eval import _selected_episode_ids
    from dp_skillset_eval import index_skillset, load_episode_skills
    from skill_divider import load_data, load_policy

    skillset_dir = Path(settings["skillset_dir"])
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(skillset_dir)
    dataset_dir = _portable_path(str(manifest["dataset_dir"]), "dataset_filtered")
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
    data = load_data(dataset_dir, episode_ids=episodes)

    pca_info = {}
    for probe_index, (name, variant) in enumerate(probe_defs.items()):
        pca, indices, pca_path = _resolve_pca(
            variant=variant,
            skillset_dir=skillset_dir,
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            policy=policy,
            normalizer=normalizer,
            manifest=manifest,
            variance=float(settings["pca_variance"]),
            stride=int(settings["pca_stride"]),
        )
        count = int(variant.get("count", 24))
        directions = (
            _antithetic_directions(pca, count, int(settings["seed"]))
            if bool(variant.get("antithetic", False))
            else pca.sample_directions(count, int(settings["seed"]))
        )
        pca_info[name] = (pca, indices, pca_path, directions)
        metadata = {
            "schema": 1,
            "policy_path": str(policy_path.resolve()),
            "pca_sha256": _sha256(pca_path),
            "variant": variant,
            "seed": int(settings["seed"]),
            "eval_at_step": int(manifest["detector"]["eval_at_step"]),
            "replan_interval": int(manifest["detector"]["replan_interval"]),
        }
        for episode_id in episodes:
            cache_path = output_dir / "raw" / _slug(name) / f"ep{episode_id:07d}.npz"
            cached = _cache_load(cache_path, metadata) if settings["resume"] else None
            if cached is None:
                ep_df = data[data["episode_index"] == episode_id].reset_index(drop=True)
                print(f"[probe {probe_index + 1}/{len(probe_defs)}] {name} · ep{episode_id}", flush=True)
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
                )
                _cache_save(cache_path, cached, metadata)
            else:
                print(f"[resume probe] {name} · ep{episode_id}", flush=True)
            raw_by_probe[name][episode_id] = cached
            raw_paths[name][episode_id] = cache_path

    if rollout_cfg.get("enabled"):
        probe_name = str(rollout_cfg["probe"])
        pca, indices, pca_path, _directions = pca_info[probe_name]
        metadata = {
            "schema": 1,
            "policy_path": str(policy_path.resolve()),
            "pca_sha256": _sha256(pca_path),
            "samples": int(rollout_cfg.get("samples", 32)),
            "seed": int(settings["seed"]),
            "num_inference_steps": int(manifest["detector"].get("num_inference_steps", 10)),
        }
        raw_by_probe["direct_rollout"] = {}
        raw_paths["direct_rollout"] = {}
        for episode_id in episodes:
            cache_path = output_dir / "raw" / "direct_rollout" / f"ep{episode_id:07d}.npz"
            cached = _cache_load(cache_path, metadata) if settings["resume"] else None
            if cached is None:
                ep_df = data[data["episode_index"] == episode_id].reset_index(drop=True)
                print(f"[direct rollout] ep{episode_id}", flush=True)
                cached = _rollout_episode(
                    policy=policy,
                    preprocessor=preprocessor,
                    ep_df=ep_df,
                    pca=pca,
                    indices=indices,
                    samples=int(rollout_cfg.get("samples", 32)),
                    batch_size=int(rollout_cfg.get("batch_size", 16)),
                    seed=int(settings["seed"]) + episode_id * 1009,
                    replan_interval=int(manifest["detector"]["replan_interval"]),
                )
                _cache_save(cache_path, cached, metadata)
            raw_by_probe["direct_rollout"][episode_id] = cached
            raw_paths["direct_rollout"][episode_id] = cache_path

    del policy, preprocessor, normalizer, data
    gc.collect()
    torch.cuda.empty_cache()

    experiments = list(settings["experiments"])
    if rollout_cfg.get("enabled"):
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
        dict.fromkeys((str(item["probe"]), str(item["cluster"])) for item in experiments)
    )
    scores_by_pair: dict[tuple[str, str], dict[int, dict[str, np.ndarray]]] = {}
    input_scores_by_pair: dict[tuple[str, str], dict[str, np.ndarray] | None] = {}
    total_cluster_jobs = len(unique_pairs) * len(episodes)
    cluster_job = 0
    for probe_name, cluster_name in unique_pairs:
        pair = (probe_name, cluster_name)
        scores_by_pair[pair] = {}
        cluster_cfg = cluster_defs[cluster_name]
        if probe_name == "direct_rollout":
            input_scores_by_pair[pair] = None
        else:
            pca, _indices, _pca_path, directions = pca_info[probe_name]
            alpha = float(probe_defs[probe_name].get("alpha", 0.1))
            # Probe construction is an affine translation of this same cloud
            # at every episode anchor; BIC is translation invariant. Compute
            # the input-geometry baseline once per probe recipe.
            center = np.asarray(pca.mean, dtype=np.float32)
            input_cloud = np.concatenate(
                [
                    pca.transform(center[None]),
                    pca.transform(center[None] + alpha * directions),
                ],
                axis=0,
            )
            input_scores_by_pair[pair] = _cluster_scores(input_cloud[None], cluster_cfg)
            print(
                f"[input BIC] {probe_name} + {cluster_name}: "
                f"delta={float(input_scores_by_pair[pair]['delta_bic'][0]):.4f}",
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
            if cached_scores is None:
                print(
                    f"[cluster {cluster_job}/{total_cluster_jobs}] "
                    f"{probe_name} + {cluster_name} · ep{episode_id}",
                    flush=True,
                )
                cached_scores = _cluster_scores(
                    raw_by_probe[probe_name][episode_id]["descriptors"], cluster_cfg
                )
                _cluster_cache_save(cache_path, cached_scores, metadata)
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
            derived = _derive(
                raw_by_probe[str(experiment["probe"])][episode_id],
                cluster_defs[str(experiment["cluster"])],
                str(experiment["metric"]),
                boundary_defs[str(experiment["boundary"])],
                min_skill_len=int(manifest["detector"].get("min_skill_len", 10)),
                scores=scores_by_pair[
                    (str(experiment["probe"]), str(experiment["cluster"]))
                ][episode_id],
                input_scores=input_scores_by_pair[
                    (str(experiment["probe"]), str(experiment["cluster"]))
                ],
            )
            rows.append((name, derived))
            derived_summary[str(episode_id)][name] = {
                "boundaries": derived["boundaries"].astype(int).tolist(),
                "input_delta_bic": (
                    None
                    if input_scores_by_pair[
                        (str(experiment["probe"]), str(experiment["cluster"]))
                    ] is None
                    else float(
                        input_scores_by_pair[
                            (str(experiment["probe"]), str(experiment["cluster"]))
                        ]["delta_bic"][0]
                    )
                ),
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

    _ep_task, ep_files = index_skillset(skillset_dir / "skills")
    existing = {}
    for episode_id in episodes:
        skills, _gripper, _indices = load_episode_skills(ep_files[episode_id])
        existing[episode_id] = np.asarray([end for _start, end, _label in skills[:-1]], dtype=np.int64)
    report = _render_report(output_dir, episode_rows, existing, experiments)
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
