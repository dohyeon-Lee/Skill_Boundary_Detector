"""Generic action-manifold probes for diffusion-policy skill segmentation.

The legacy SBD probe rotates the first three action dimensions as delta EEF XYZ.
This module instead works in a configurable subset of the policy's normalized
action space:

1. Fit PCA to the temporal mean of normalized demonstration action chunks.
2. Sample fixed unit directions in the retained PCA subspace.
3. Translate every step of a demo chunk by the same action-space offset.
4. Query one diffusion denoising step and score the temporal-mean outputs in
   the same PCA coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


PROBE_SPHERICAL_XYZ = "spherical_xyz"
PROBE_PCA_ACTION = "pca_action"
SUPPORTED_PROBE_TYPES = (PROBE_SPHERICAL_XYZ, PROBE_PCA_ACTION)

ACTION_MODE_DATASET = "dataset"
ACTION_MODE_ANCHOR_RELATIVE = "anchor_relative"
SUPPORTED_ACTION_MODES = (ACTION_MODE_DATASET, ACTION_MODE_ANCHOR_RELATIVE)

GRIPPER_CONTINUOUS = "continuous"
GRIPPER_DISCRETE = "discrete"
SUPPORTED_GRIPPER_MODES = (GRIPPER_CONTINUOUS, GRIPPER_DISCRETE)

PCA_SCALE_NONE = "none"
PCA_SCALE_STD = "std"
SUPPORTED_PCA_SCALE_MODES = (PCA_SCALE_NONE, PCA_SCALE_STD)


@dataclass(frozen=True)
class NumpyActionNormalizer:
    """NumPy mirror of the action normalization saved with a policy processor."""

    mode: str
    stats: dict[str, np.ndarray]
    eps: float = 1e-8

    @classmethod
    def from_preprocessor(cls, preprocessor) -> "NumpyActionNormalizer":
        from lerobot.configs.types import FeatureType
        from lerobot.processor.normalize_processor import NormalizerProcessorStep

        steps = [step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)]
        if len(steps) != 1:
            raise ValueError(f"Expected exactly one action normalizer in the policy preprocessor, found {len(steps)}")
        step = steps[0]
        mode = step.norm_map[FeatureType.ACTION].value
        if "action" not in step._tensor_stats:
            raise ValueError("Policy preprocessor has no saved normalization statistics for 'action'.")
        stats = {
            name: tensor.detach().float().cpu().numpy().copy()
            for name, tensor in step._tensor_stats["action"].items()
        }
        return cls(mode=mode, stats=stats, eps=float(step.eps))

    def normalize(self, action: np.ndarray) -> np.ndarray:
        x = np.asarray(action, dtype=np.float32)
        if self.mode == "IDENTITY":
            return x.copy()
        if self.mode == "MEAN_STD":
            return (x - self.stats["mean"]) / (self.stats["std"] + self.eps)
        if self.mode == "MIN_MAX":
            lo, hi = self.stats["min"], self.stats["max"]
        elif self.mode == "QUANTILES":
            lo, hi = self.stats["q01"], self.stats["q99"]
        elif self.mode == "QUANTILE10":
            lo, hi = self.stats["q10"], self.stats["q90"]
        else:
            raise ValueError(f"Unsupported action normalization mode: {self.mode}")
        denom = np.where(hi == lo, self.eps, hi - lo)
        return 2.0 * (x - lo) / denom - 1.0

    def denormalize(self, action: np.ndarray) -> np.ndarray:
        x = np.asarray(action, dtype=np.float32)
        if self.mode == "IDENTITY":
            return x.copy()
        if self.mode == "MEAN_STD":
            return x * self.stats["std"] + self.stats["mean"]
        if self.mode == "MIN_MAX":
            lo, hi = self.stats["min"], self.stats["max"]
        elif self.mode == "QUANTILES":
            lo, hi = self.stats["q01"], self.stats["q99"]
        elif self.mode == "QUANTILE10":
            lo, hi = self.stats["q10"], self.stats["q90"]
        else:
            raise ValueError(f"Unsupported action normalization mode: {self.mode}")
        denom = np.where(hi == lo, self.eps, hi - lo)
        return (x + 1.0) * denom / 2.0 + lo


def resolve_indices(indices: Iterable[int], dimension: int) -> tuple[int, ...]:
    resolved = []
    for index in indices:
        actual = int(index) + dimension if int(index) < 0 else int(index)
        if actual < 0 or actual >= dimension:
            raise ValueError(f"Action index {index} is out of range for dimension {dimension}.")
        if actual not in resolved:
            resolved.append(actual)
    return tuple(resolved)


def relative_action_mask(
    action_dim: int,
    action_names: list[str] | None,
    exclude_tokens: Iterable[str],
) -> np.ndarray:
    """Match RelativeActionsProcessorStep's name-based exclusion semantics."""
    tokens = [str(token).lower() for token in exclude_tokens if str(token)]
    if tokens and not action_names:
        raise ValueError("anchor_relative action mode needs action names when dimensions are excluded.")
    mask = np.ones(action_dim, dtype=bool)
    if not tokens:
        return mask
    for i, name in enumerate((action_names or [])[:action_dim]):
        lowered = str(name).lower()
        if any(token == lowered or token in lowered for token in tokens):
            mask[i] = False
    return mask


def to_model_action_chunk(
    action_chunk: np.ndarray,
    anchor_state: np.ndarray,
    action_mode: str,
    relative_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert stored dataset actions to the representation learned by the DP."""
    chunk = np.asarray(action_chunk, dtype=np.float32).copy()
    if action_mode == ACTION_MODE_DATASET:
        return chunk
    if action_mode != ACTION_MODE_ANCHOR_RELATIVE:
        raise ValueError(f"Unsupported action mode: {action_mode}")
    if relative_mask is None:
        raise ValueError("anchor_relative action mode requires a relative-action mask.")
    dims = len(relative_mask)
    if chunk.shape[-1] < dims or len(anchor_state) < dims:
        raise ValueError(
            f"Cannot anchor relative action: chunk dim={chunk.shape[-1]}, state dim={len(anchor_state)}, mask={dims}."
        )
    chunk[:, :dims] -= np.asarray(anchor_state[:dims], dtype=np.float32) * relative_mask.astype(np.float32)
    return chunk


class RunningCovariance:
    """Mergeable population covariance accumulator with constant memory."""

    def __init__(self, dimension: int):
        self.count = 0
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.m2 = np.zeros((dimension, dimension), dtype=np.float64)

    def update_batch(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim == 1:
            x = x[None]
        if len(x) == 0:
            return
        batch_count = len(x)
        batch_mean = x.mean(axis=0)
        centered = x - batch_mean
        batch_m2 = centered.T @ centered
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.m2 += batch_m2 + np.outer(delta, delta) * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.count = total

    @property
    def covariance(self) -> np.ndarray:
        if self.count < 2:
            raise ValueError("At least two action-plan samples are required to fit PCA.")
        return self.m2 / self.count


@dataclass(frozen=True)
class ActionPCA:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    sample_count: int
    metadata: dict

    @property
    def action_dim(self) -> int:
        return int(self.components.shape[1])

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @classmethod
    def from_covariance(
        cls,
        accumulator: RunningCovariance,
        variance_threshold: float,
        metadata: dict | None = None,
        scale_mode: str = PCA_SCALE_NONE,
    ) -> "ActionPCA":
        if not 0.0 < variance_threshold <= 1.0:
            raise ValueError(f"PCA variance threshold must be in (0, 1], got {variance_threshold}.")
        if scale_mode not in SUPPORTED_PCA_SCALE_MODES:
            raise ValueError(f"Unsupported PCA scale mode: {scale_mode}.")

        covariance = accumulator.covariance
        if scale_mode == PCA_SCALE_STD:
            empirical_std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            scale = np.where(empirical_std > 1e-6, empirical_std, 1.0)
            fit_covariance = covariance / np.outer(scale, scale)
        else:
            scale = np.ones_like(accumulator.mean)
            fit_covariance = covariance
        fit_covariance = (fit_covariance + fit_covariance.T) / 2.0

        eigenvalues, eigenvectors = np.linalg.eigh(fit_covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        total = float(eigenvalues.sum())
        if total <= 0:
            raise ValueError("Action-plan covariance has no positive variance.")
        ratios = eigenvalues / total
        n_components = min(
            len(ratios),
            int(np.searchsorted(np.cumsum(ratios), variance_threshold, side="left") + 1),
        )
        return cls(
            mean=accumulator.mean.astype(np.float32),
            scale=scale.astype(np.float32),
            components=eigenvectors[:, :n_components].T.astype(np.float32),
            explained_variance=eigenvalues[:n_components].astype(np.float32),
            explained_variance_ratio=ratios[:n_components].astype(np.float32),
            sample_count=accumulator.count,
            metadata=dict(metadata or {}),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float32)
        return ((x - self.mean) / self.scale) @ self.components.T

    def sample_directions(self, count: int, seed: int) -> np.ndarray:
        if count < 1:
            raise ValueError(f"Probe count must be positive, got {count}.")
        rng = np.random.default_rng(seed)
        coefficients = rng.standard_normal((count, self.n_components))
        coefficient_norms = np.linalg.norm(coefficients, axis=1, keepdims=True)
        coefficients /= np.maximum(coefficient_norms, 1e-12)
        standardized_directions = coefficients @ self.components
        standardized_directions /= np.maximum(
            np.linalg.norm(standardized_directions, axis=1, keepdims=True), 1e-12
        )
        return (standardized_directions * self.scale).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.metadata, sort_keys=True)
        tmp_path = path.with_name(path.name + ".tmp.npz")
        np.savez(
            tmp_path,
            mean=self.mean,
            scale=self.scale,
            components=self.components,
            explained_variance=self.explained_variance,
            explained_variance_ratio=self.explained_variance_ratio,
            sample_count=np.array(self.sample_count, dtype=np.int64),
            metadata=np.array(payload),
        )
        tmp_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ActionPCA":
        with np.load(path, allow_pickle=False) as data:
            components = data["components"]
            scale = (
                data["scale"]
                if "scale" in data.files
                else np.ones(components.shape[1], dtype=np.float32)
            )
            return cls(
                mean=data["mean"],
                scale=scale,
                components=components,
                explained_variance=data["explained_variance"],
                explained_variance_ratio=data["explained_variance_ratio"],
                sample_count=int(data["sample_count"]),
                metadata=json.loads(str(data["metadata"])),
            )


def get_or_fit_action_pca(
    path: Path,
    expected_metadata: dict,
    fit: Callable[[], ActionPCA],
) -> ActionPCA:
    """Load one shared PCA artifact, serializing concurrent Slurm shard fitting."""
    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            pca = ActionPCA.load(path)
            if pca.metadata != expected_metadata:
                raise ValueError(
                    f"Cached action PCA metadata does not match this run: {path}\n"
                    f"cached={pca.metadata}\nexpected={expected_metadata}"
                )
            return pca
        pca = fit()
        if pca.metadata != expected_metadata:
            raise ValueError("PCA fit callback returned unexpected metadata.")
        pca.save(path)
        return pca


def make_pca_action_probes(
    normalized_demo_chunk: np.ndarray,
    directions: np.ndarray,
    alpha: float,
    normalizer: NumpyActionNormalizer,
    gripper_mode: str = GRIPPER_CONTINUOUS,
    gripper_indices: Iterable[int] = (),
    gripper_values: tuple[float, float] = (-1.0, 1.0),
    gripper_threshold: float = 0.0,
    action_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Return GT + local PCA probes as `(1 + N, H, D)` normalized chunks."""
    demo = np.asarray(normalized_demo_chunk, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    if demo.ndim != 2 or directions.ndim != 2:
        raise ValueError(f"Probe shape mismatch: demo={demo.shape}, directions={directions.shape}.")
    if alpha <= 0:
        raise ValueError(f"Probe alpha must be positive, got {alpha}.")
    action_dim = demo.shape[1]
    selected = (
        tuple(range(action_dim))
        if action_indices is None
        else resolve_indices(action_indices, action_dim)
    )
    if not selected:
        raise ValueError("At least one action dimension is required for PCA probes.")
    if directions.shape[1] != len(selected):
        raise ValueError(
            f"Probe direction dim {directions.shape[1]} != selected action dims {len(selected)}."
        )
    selected_offsets = alpha * np.sqrt(len(selected)) * directions
    offsets = np.zeros((len(directions), action_dim), dtype=np.float32)
    offsets[:, list(selected)] = selected_offsets
    probes = demo[None] + offsets[:, None, :]

    if gripper_mode == GRIPPER_DISCRETE:
        indices = resolve_indices(gripper_indices, action_dim)
        if not indices:
            raise ValueError("Discrete gripper mode requires at least one gripper action index.")
        active_indices = tuple(index for index in indices if index in selected)
        if active_indices:
            low, high = sorted(float(v) for v in gripper_values)
            raw = normalizer.denormalize(probes)
            raw[..., list(active_indices)] = np.where(
                raw[..., list(active_indices)] < gripper_threshold, low, high
            )
            probes = normalizer.normalize(raw)
    elif gripper_mode != GRIPPER_CONTINUOUS:
        raise ValueError(f"Unsupported gripper mode: {gripper_mode}")

    return np.concatenate([demo[None], probes.astype(np.float32)], axis=0)


def action_plan_descriptors(
    denoised_chunks: np.ndarray,
    pca: ActionPCA,
    action_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Temporal-mean normalized action plans expressed in the fitted PCA coordinates."""
    chunks = np.asarray(denoised_chunks, dtype=np.float32)
    if chunks.ndim != 3:
        raise ValueError(f"Expected denoised chunks (N,H,D), got {chunks.shape}.")
    selected = (
        tuple(range(chunks.shape[-1]))
        if action_indices is None
        else resolve_indices(action_indices, chunks.shape[-1])
    )
    if len(selected) != pca.action_dim:
        raise ValueError(f"Selected action dims {len(selected)} != PCA dim {pca.action_dim}.")
    return pca.transform(chunks.mean(axis=1)[:, list(selected)])


def compute_action_divergence(descriptors: np.ndarray, n_components: int) -> tuple[float, float, np.ndarray]:
    """GMM cluster separation over generic PCA action-plan descriptors."""
    from sklearn.mixture import GaussianMixture

    data = np.asarray(descriptors, dtype=np.float64)
    if data.ndim != 2 or len(data) < 2:
        raise ValueError(f"Expected at least two 2-D descriptor rows, got {data.shape}.")
    n_components = min(int(n_components), len(data))
    if n_components < 1:
        raise ValueError(f"GMM component count must be positive, got {n_components}.")
    fitted = None
    for reg_covar in (1e-6, 1e-4, 1e-2):
        try:
            candidate = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=0,
                max_iter=200,
                reg_covar=reg_covar,
            )
            candidate.fit(data)
            fitted = candidate
            break
        except (ValueError, np.linalg.LinAlgError):
            continue
    if fitted is None:
        return 0.0, 0.0, np.zeros((n_components, data.shape[1]), dtype=np.float32)

    means = fitted.means_
    if n_components < 2:
        return 0.0, 0.0, means.astype(np.float32)

    cos_sq = 0.0
    l2_sq = 0.0
    count = 0
    for i in range(n_components):
        for j in range(i + 1, n_components):
            denom = np.linalg.norm(means[i]) * np.linalg.norm(means[j]) + 1e-8
            cos_sq += (1.0 - float(np.dot(means[i], means[j]) / denom)) ** 2
            l2_sq += float(np.linalg.norm(means[i] - means[j])) ** 2
            count += 1
    return float(np.sqrt(cos_sq / count)), float(np.sqrt(l2_sq / count)), means.astype(np.float32)
