"""Production multimodality primitives for DP skill-boundary detection.

This production module is independent of experimental ablation/report code,
so the build pipeline remains usable after those utilities are removed.
"""

from __future__ import annotations

import numpy as np


def iid_scheduler_noise(
    count: int,
    horizon: int,
    action_dim: int,
    seed: int,
) -> np.ndarray:
    """Return reproducible IID standard-Gaussian diffusion noise templates."""
    if count < 1 or horizon < 1 or action_dim < 1:
        raise ValueError(
            "count, horizon, and action_dim must be positive; got "
            f"{count}, {horizon}, {action_dim}."
        )
    return np.random.default_rng(int(seed)).standard_normal(
        (int(count), int(horizon), int(action_dim)), dtype=np.float32
    )


def _component_covariance_matrices(model, keep: np.ndarray) -> np.ndarray:
    covariance_type = str(model.covariance_type)
    covariances = np.asarray(model.covariances_, dtype=np.float64)
    dimension = int(model.means_.shape[1])
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


def directional_reliability(mean: np.ndarray, covariance: np.ndarray) -> float:
    """Confidence that a Gaussian component has a stable mean direction."""
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    signal = float(np.dot(mean, mean))
    if signal <= np.finfo(np.float64).eps:
        return 0.0
    direction = mean / np.sqrt(signal)
    total_variance = float(np.trace(covariance))
    parallel_variance = float(direction @ covariance @ direction)
    orthogonal_variance = max(0.0, total_variance - parallel_variance)
    return float(signal / (signal + orthogonal_variance))


def _fit_gmm(cloud: np.ndarray, *, components: int, covariance: str, n_init: int, max_iter: int):
    from sklearn.mixture import GaussianMixture

    for regularization in (1e-6, 1e-4, 1e-2):
        try:
            return GaussianMixture(
                n_components=int(components),
                covariance_type=str(covariance),
                n_init=int(n_init),
                random_state=0,
                max_iter=int(max_iter),
                reg_covar=float(regularization),
            ).fit(np.asarray(cloud, dtype=np.float64))
        except (ValueError, np.linalg.LinAlgError):
            continue
    return None


def covariance_gated_cosine_curve(
    descriptor_clouds: np.ndarray,
    *,
    components: int = 5,
    covariance: str = "diag",
    n_init: int = 10,
    max_iter: int = 300,
    min_effective_samples: float = 0.0,
    weighted: bool = True,
) -> np.ndarray:
    """Score stable directional separation between GMM modes at every anchor.

    ``descriptor_clouds`` has shape ``(anchors, samples, descriptor_dim)``.
    Pairwise cosine divergence is suppressed when a component's perpendicular
    covariance is large relative to its squared mean magnitude.
    """
    from sklearn.mixture import GaussianMixture as _GaussianMixture  # noqa: F401
    from threadpoolctl import threadpool_limits

    clouds = np.asarray(descriptor_clouds, dtype=np.float64)
    if clouds.ndim != 3:
        raise ValueError(
            "descriptor_clouds must have shape (anchors, samples, dimensions), "
            f"got {clouds.shape}."
        )
    if not 1 <= int(components) <= int(clouds.shape[1]):
        raise ValueError(
            f"components must be in [1, {clouds.shape[1]}], got {components}."
        )
    if float(min_effective_samples) < 0.0:
        raise ValueError("min_effective_samples must be non-negative.")

    scores = np.zeros(len(clouds), dtype=np.float64)
    with threadpool_limits(limits=1):
        for anchor_index, cloud in enumerate(clouds):
            model = _fit_gmm(
                cloud,
                components=int(components),
                covariance=str(covariance),
                n_init=int(n_init),
                max_iter=int(max_iter),
            )
            if model is None:
                continue
            effective_support = len(cloud) * np.asarray(model.weights_, dtype=np.float64)
            keep = np.flatnonzero(
                effective_support + 1e-12 >= float(min_effective_samples)
            )
            if len(keep) < 2:
                continue
            means = np.asarray(model.means_[keep], dtype=np.float64)
            mixture_weights = np.asarray(model.weights_[keep], dtype=np.float64)
            covariances = _component_covariance_matrices(model, keep)
            reliabilities = np.asarray(
                [
                    directional_reliability(mean, component_covariance)
                    for mean, component_covariance in zip(
                        means, covariances, strict=True
                    )
                ],
                dtype=np.float64,
            )
            pairs = [
                (left, right)
                for left in range(len(means))
                for right in range(left + 1, len(means))
            ]
            pair_weights = np.asarray(
                [mixture_weights[left] * mixture_weights[right] for left, right in pairs],
                dtype=np.float64,
            )
            if not bool(weighted):
                pair_weights[:] = 1.0
            weight_sum = float(pair_weights.sum())
            if weight_sum <= 0.0:
                continue
            pair_weights /= weight_sum

            gated_divergence = []
            for left, right in pairs:
                norm_product = float(
                    np.linalg.norm(means[left]) * np.linalg.norm(means[right])
                )
                cosine = float(
                    np.dot(means[left], means[right]) / (norm_product + 1e-8)
                )
                angular_divergence = 1.0 - cosine
                reliability = float(
                    np.sqrt(reliabilities[left] * reliabilities[right])
                )
                gated_divergence.append(angular_divergence * reliability)
            scores[anchor_index] = float(
                np.sqrt(
                    np.sum(
                        pair_weights * np.square(np.asarray(gated_divergence))
                    )
                )
            )
    return scores.astype(np.float32)
