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


def gaussian_bhattacharyya_separation(
    left_mean: np.ndarray,
    left_covariance: np.ndarray,
    right_mean: np.ndarray,
    right_covariance: np.ndarray,
) -> float:
    """Return bounded non-overlap of two Gaussian components.

    Zero means identical Gaussian modes and values approaching one mean that
    the modes have negligible overlap.  Unlike a center-only distance, the
    Bhattacharyya distance treats two broad, overlapping fitted components as
    one ambiguous region rather than two distinct action choices.
    """
    left_mean = np.asarray(left_mean, dtype=np.float64)
    right_mean = np.asarray(right_mean, dtype=np.float64)
    left_covariance = np.asarray(left_covariance, dtype=np.float64)
    right_covariance = np.asarray(right_covariance, dtype=np.float64)
    pooled = 0.5 * (left_covariance + right_covariance)
    difference = left_mean - right_mean
    try:
        solved = np.linalg.solve(pooled, difference)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(pooled) @ difference
    quadratic = 0.125 * float(difference @ solved)
    pooled_sign, pooled_logdet = np.linalg.slogdet(pooled)
    left_sign, left_logdet = np.linalg.slogdet(left_covariance)
    right_sign, right_logdet = np.linalg.slogdet(right_covariance)
    if min(pooled_sign, left_sign, right_sign) <= 0:
        return 0.0
    volume = 0.5 * (
        float(pooled_logdet) - 0.5 * (float(left_logdet) + float(right_logdet))
    )
    distance = max(0.0, quadratic + volume)
    return float(np.clip(-np.expm1(-distance), 0.0, 1.0))


def gmm_overlap_entropy_score(
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    reference_components: int,
) -> float:
    """Score balanced, genuinely distinct action modes on a bounded scale.

    The normalized mixture-weight entropy answers whether more than one mode
    has meaningful probability mass.  The weighted Gaussian non-overlap then
    verifies that those modes are distinct relative to their covariance.  The
    product is in ``[0, 1]`` and saturates once modes are clearly separated,
    making true multimodality peaks more comparable across time.
    """
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    means = np.asarray(means, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    if len(weights) < 2:
        return 0.0
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    probabilities = weights / total
    denominator = np.log(max(2, int(reference_components)))
    entropy = float(
        -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
        / denominator
    )
    pairs = [
        (left, right)
        for left in range(len(probabilities))
        for right in range(left + 1, len(probabilities))
    ]
    pair_weights = np.asarray(
        [probabilities[left] * probabilities[right] for left, right in pairs],
        dtype=np.float64,
    )
    pair_total = float(pair_weights.sum())
    if pair_total <= 0.0:
        return 0.0
    pair_weights /= pair_total
    separation = np.asarray(
        [
            gaussian_bhattacharyya_separation(
                means[left],
                covariances[left],
                means[right],
                covariances[right],
            )
            for left, right in pairs
        ],
        dtype=np.float64,
    )
    return float(np.clip(entropy * np.sum(pair_weights * separation), 0.0, 1.0))


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
    reliability_pair_power: float = 0.5,
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
    if not np.isfinite(reliability_pair_power) or float(reliability_pair_power) < 0.0:
        raise ValueError("reliability_pair_power must be finite and non-negative.")

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
                    np.power(
                        reliabilities[left] * reliabilities[right],
                        float(reliability_pair_power),
                    )
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


def strong_covariance_gated_cosine_curve(
    descriptor_clouds: np.ndarray,
    *,
    components: int = 5,
    covariance: str = "diag",
    n_init: int = 10,
    max_iter: int = 300,
    min_effective_samples: float = 0.0,
    weighted: bool = True,
) -> np.ndarray:
    """Cosine SBD with full pairwise directional-reliability suppression.

    The production covariance-gated cosine uses the geometric mean
    ``sqrt(r_i r_j)``.  This ablation uses ``r_i r_j`` instead, preserving
    high-confidence directional modes while more strongly suppressing pairs
    whose component means are small relative to their orthogonal covariance.
    """
    return covariance_gated_cosine_curve(
        descriptor_clouds,
        components=components,
        covariance=covariance,
        n_init=n_init,
        max_iter=max_iter,
        min_effective_samples=min_effective_samples,
        weighted=weighted,
        reliability_pair_power=1.0,
    )


def gmm_overlap_entropy_curve(
    descriptor_clouds: np.ndarray,
    *,
    components: int = 5,
    covariance: str = "diag",
    n_init: int = 10,
    max_iter: int = 300,
    min_effective_samples: float = 0.0,
    weighted: bool = True,
) -> np.ndarray:
    """Measure instantaneous multimodality by GMM overlap and weight entropy.

    ``weighted`` is accepted for the common scorer interface.  This metric is
    intrinsically mixture-weighted and therefore requires it to remain true.
    """
    from sklearn.mixture import GaussianMixture as _GaussianMixture  # noqa: F401
    from threadpoolctl import threadpool_limits

    if not bool(weighted):
        raise ValueError("gmm_overlap_entropy is intrinsically mixture-weighted")
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
            scores[anchor_index] = gmm_overlap_entropy_score(
                np.asarray(model.weights_[keep], dtype=np.float64),
                np.asarray(model.means_[keep], dtype=np.float64),
                _component_covariance_matrices(model, keep),
                reference_components=int(components),
            )
    return scores.astype(np.float32)


def covariance_gated_angular_chord_curve(
    descriptor_clouds: np.ndarray,
    *,
    components: int = 5,
    covariance: str = "diag",
    n_init: int = 10,
    max_iter: int = 300,
    min_effective_samples: float = 0.0,
    weighted: bool = True,
) -> np.ndarray:
    """Score reliable directional separation with magnitude-aware chords.

    For each retained GMM component pair, the angular chord is::

        sqrt(2 ||mu_i|| ||mu_j|| (1 - cos(theta_ij)))

    It preserves the dataset-PCA origin and directional signal used by the
    cosine detector, but makes an unstable angle near that origin contribute
    little. The existing covariance reliability gate and mixture-weighted RMS
    aggregation are applied unchanged.
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

            gated_chords = []
            for left, right in pairs:
                left_norm = float(np.linalg.norm(means[left]))
                right_norm = float(np.linalg.norm(means[right]))
                norm_product = left_norm * right_norm
                if norm_product <= np.finfo(np.float64).eps:
                    chord = 0.0
                else:
                    cosine = float(
                        np.dot(means[left], means[right]) / norm_product
                    )
                    angular_divergence = max(
                        0.0, 1.0 - float(np.clip(cosine, -1.0, 1.0))
                    )
                    chord = float(
                        np.sqrt(2.0 * norm_product * angular_divergence)
                    )
                reliability = float(
                    np.sqrt(reliabilities[left] * reliabilities[right])
                )
                gated_chords.append(chord * reliability)
            scores[anchor_index] = float(
                np.sqrt(
                    np.sum(pair_weights * np.square(np.asarray(gated_chords)))
                )
            )
    return scores.astype(np.float32)


def within_normalized_l2_curve(
    descriptor_clouds: np.ndarray,
    *,
    components: int = 5,
    covariance: str = "diag",
    n_init: int = 10,
    max_iter: int = 300,
    min_effective_samples: float = 0.0,
    weighted: bool = True,
) -> np.ndarray:
    """Score GMM mode separation relative to the modes' pooled RMS radius.

    Unlike cosine divergence, this score is invariant to translating the whole
    descriptor cloud. For every retained component pair it computes::

        ||mu_i - mu_j|| / sqrt((trace(Sigma_i) + trace(Sigma_j)) / 2)

    A forced GMM split of one broad cloud therefore stays small, while compact
    modes with well-separated centers receive a large score. Pair scores are
    aggregated with the same mixture-weighted RMS convention as the production
    cosine detector.
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

            pair_scores = []
            for left, right in pairs:
                center_distance = float(np.linalg.norm(means[left] - means[right]))
                pooled_radius = float(
                    np.sqrt(
                        max(
                            0.5
                            * (
                                np.trace(covariances[left])
                                + np.trace(covariances[right])
                            ),
                            np.finfo(np.float64).eps,
                        )
                    )
                )
                pair_scores.append(center_distance / pooled_radius)
            scores[anchor_index] = float(
                np.sqrt(
                    np.sum(pair_weights * np.square(np.asarray(pair_scores)))
                )
            )
    return scores.astype(np.float32)
