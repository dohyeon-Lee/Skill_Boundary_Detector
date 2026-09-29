import sys
from pathlib import Path

import numpy as np

LIBERO_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "libero"
sys.path.insert(0, str(LIBERO_EXAMPLES))

from action_manifold import (  # noqa: E402
    ACTION_MODE_ANCHOR_RELATIVE,
    ActionPCA,
    GRIPPER_DISCRETE,
    NumpyActionNormalizer,
    PCA_SCALE_NONE,
    PCA_SCALE_STD,
    RunningCovariance,
    action_plan_descriptors,
    compute_action_divergence,
    compute_delta_bic,
    compute_denoising_gain,
    make_pca_action_probes,
    relative_action_mask,
    to_model_action_chunk,
)


def _fit_pca(
    values: np.ndarray,
    threshold: float = 0.95,
    scale_mode: str = PCA_SCALE_NONE,
) -> ActionPCA:
    accumulator = RunningCovariance(values.shape[1])
    accumulator.update_batch(values)
    return ActionPCA.from_covariance(
        accumulator,
        threshold,
        metadata={"test": True},
        scale_mode=scale_mode,
    )


def test_pca_probe_has_fixed_per_dimension_rms_and_preserves_temporal_differences():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(1000, 4)).astype(np.float32)
    pca = _fit_pca(values, threshold=1.0)
    directions = pca.sample_directions(count=5, seed=7)
    demo = rng.normal(size=(8, 4)).astype(np.float32)
    normalizer = NumpyActionNormalizer(mode="IDENTITY", stats={})

    probes = make_pca_action_probes(demo, directions, alpha=0.2, normalizer=normalizer)

    assert probes.shape == (6, 8, 4)
    np.testing.assert_allclose(probes[0], demo)
    offsets = probes[1:] - demo[None]
    np.testing.assert_allclose(offsets[:, 1:], np.repeat(offsets[:, :1], 7, axis=1), atol=1e-6)
    np.testing.assert_allclose(np.sqrt(np.mean(offsets[:, 0] ** 2, axis=1)), 0.2, atol=1e-6)
    expected_diffs = np.repeat(np.diff(demo, axis=0)[None], len(directions), axis=0)
    np.testing.assert_allclose(np.diff(probes[1:], axis=1), expected_diffs, atol=1e-6)


def test_pca_direction_sampling_supports_subspaces_weighting_and_axes():
    rng = np.random.default_rng(21)
    values = rng.normal(size=(4000, 4)).astype(np.float32)
    values *= np.array([4.0, 2.0, 1.0, 0.25], dtype=np.float32)
    pca = _fit_pca(values, threshold=1.0)

    top2 = pca.sample_directions(24, 7, component_limit=2)
    coordinates = (top2 / pca.scale) @ pca.components.T
    np.testing.assert_allclose(coordinates[:, 2:], 0.0, atol=1e-5)

    uniform = pca.sample_directions(24, 7, component_limit=4, sampling="uniform")
    weighted = pca.sample_directions(
        24, 7, component_limit=4, sampling="variance_weighted"
    )
    assert not np.allclose(uniform, weighted)

    axes = pca.sample_directions(
        16, 7, component_limit=4, sampling="axes_pairwise"
    )
    standardized = axes / pca.scale
    np.testing.assert_allclose(np.linalg.norm(standardized, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(axes[:4], -axes[4:8], atol=1e-6)
    np.testing.assert_allclose(axes[8:12], -axes[12:16], atol=1e-6)


def test_pca_sigma_offsets_follow_empirical_axis_variance():
    rng = np.random.default_rng(31)
    values = rng.normal(size=(12000, 3)).astype(np.float32)
    values *= np.array([5.0, 2.0, 0.25], dtype=np.float32)
    pca = _fit_pca(values, threshold=1.0)

    offsets = pca.principal_sigma_offsets(
        4, component_limit=2, sigma_scale=1.0
    )

    assert offsets.shape == (4, 3)
    np.testing.assert_allclose(offsets[:2], -offsets[2:], atol=1e-6)
    coordinates = (offsets / pca.scale) @ pca.components.T
    np.testing.assert_allclose(coordinates[:2, 2], 0.0, atol=1e-5)
    np.testing.assert_allclose(
        np.abs(np.diag(coordinates[:2, :2])),
        np.sqrt(pca.explained_variance[:2]),
        rtol=1e-5,
    )


def test_absolute_pca_offsets_are_not_rescaled_by_alpha_or_dimension():
    demo = np.zeros((2, 2), dtype=np.float32)
    offsets = np.array([[2.0, -3.0], [-2.0, 3.0]], dtype=np.float32)

    probes = make_pca_action_probes(
        demo,
        offsets,
        alpha=1.0,
        normalizer=NumpyActionNormalizer(mode="IDENTITY", stats={}),
        directions_are_offsets=True,
    )

    np.testing.assert_allclose(probes[1:, 0], offsets)
    np.testing.assert_allclose(probes[1:, 1], offsets)


def test_full_trajectory_pca_probes_and_descriptors_keep_temporal_order():
    rng = np.random.default_rng(22)
    pca = _fit_pca(rng.normal(size=(2000, 6)).astype(np.float32), threshold=1.0)
    demo = rng.normal(size=(3, 2)).astype(np.float32)
    directions = pca.sample_directions(8, 4)

    probes = make_pca_action_probes(
        demo,
        directions,
        alpha=0.1,
        normalizer=NumpyActionNormalizer(mode="IDENTITY", stats={}),
    )
    offsets = probes[1:] - demo[None]
    assert not np.allclose(offsets[:, 0], offsets[:, 1])
    np.testing.assert_allclose(
        np.sqrt(np.mean(offsets.reshape(len(offsets), -1) ** 2, axis=1)),
        0.1,
        atol=1e-6,
    )

    descriptors = action_plan_descriptors(probes, pca)
    expected = pca.transform(probes.reshape(len(probes), -1))
    np.testing.assert_allclose(descriptors, expected)


def test_pca_probe_can_perturb_only_future_slice():
    demo = np.arange(18, dtype=np.float32).reshape(6, 3)
    directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    probes = make_pca_action_probes(
        demo,
        directions,
        alpha=0.2,
        normalizer=NumpyActionNormalizer(mode="IDENTITY", stats={}),
        temporal_start=2,
        temporal_length=4,
    )

    np.testing.assert_allclose(probes[1, :2], demo[:2])
    assert np.any(np.abs(probes[1, 2:] - demo[2:]) > 0)


def test_std_scaled_pca_equalizes_dimensions_and_maps_probe_back_to_action_space(tmp_path: Path):
    rng = np.random.default_rng(12)
    values = rng.normal(size=(5000, 3)).astype(np.float32)
    values = values * np.array([0.1, 2.0, 10.0], dtype=np.float32) + np.array(
        [1.0, -3.0, 8.0], dtype=np.float32
    )
    pca = _fit_pca(values, threshold=1.0, scale_mode=PCA_SCALE_STD)

    np.testing.assert_allclose(pca.scale, values.std(axis=0), rtol=1e-5)
    standardized = (values - pca.mean) / pca.scale
    np.testing.assert_allclose(standardized.std(axis=0), np.ones(3), rtol=1e-5)

    directions = pca.sample_directions(count=8, seed=3)
    standardized_directions = directions / pca.scale
    np.testing.assert_allclose(
        np.linalg.norm(standardized_directions, axis=1), np.ones(8), atol=1e-6
    )

    demo = np.zeros((5, 3), dtype=np.float32)
    probes = make_pca_action_probes(
        demo,
        directions,
        alpha=0.2,
        normalizer=NumpyActionNormalizer(mode="IDENTITY", stats={}),
    )
    standardized_offsets = (probes[1:] - demo[None]) / pca.scale
    np.testing.assert_allclose(
        np.sqrt(np.mean(standardized_offsets[:, 0] ** 2, axis=1)),
        np.full(8, 0.2),
        atol=1e-6,
    )

    path = tmp_path / "std_pca.npz"
    pca.save(path)
    loaded = ActionPCA.load(path)
    np.testing.assert_allclose(loaded.scale, pca.scale)
    np.testing.assert_allclose(loaded.transform(values[:10]), pca.transform(values[:10]))


def test_discrete_gripper_is_projected_after_correlated_full_action_probe():
    normalizer = NumpyActionNormalizer(
        mode="MIN_MAX",
        stats={"min": np.array([-2.0, -1.0]), "max": np.array([2.0, 1.0])},
    )
    raw_demo = np.array([[0.0, -1.0], [0.5, -1.0]], dtype=np.float32)
    demo = normalizer.normalize(raw_demo)
    directions = np.array([[0.6, 0.8], [-0.6, -0.8]], dtype=np.float32)

    probes = make_pca_action_probes(
        demo,
        directions,
        alpha=1.0,
        normalizer=normalizer,
        gripper_mode=GRIPPER_DISCRETE,
        gripper_indices=(-1,),
        gripper_values=(-1.0, 1.0),
        gripper_threshold=0.0,
    )
    raw_probes = normalizer.denormalize(probes[1:])

    assert set(np.unique(raw_probes[..., -1]).tolist()) <= {-1.0, 1.0}
    assert np.all(raw_probes[0, :, -1] == 1.0)
    assert np.all(raw_probes[1, :, -1] == -1.0)


def test_pose_only_probe_preserves_gripper_and_excludes_it_from_descriptors():
    rng = np.random.default_rng(4)
    pca = _fit_pca(rng.normal(size=(500, 2)).astype(np.float32), threshold=1.0)
    directions = pca.sample_directions(count=6, seed=9)
    demo = np.array(
        [
            [0.1, -0.2, -0.8],
            [0.2, -0.1, -0.2],
            [0.3, 0.0, 0.3],
            [0.4, 0.1, 0.9],
        ],
        dtype=np.float32,
    )
    normalizer = NumpyActionNormalizer(mode="IDENTITY", stats={})

    probes = make_pca_action_probes(
        demo,
        directions,
        alpha=0.1,
        normalizer=normalizer,
        gripper_mode=GRIPPER_DISCRETE,
        gripper_indices=(-1,),
        action_indices=(0, 1),
    )

    np.testing.assert_allclose(probes[..., -1], np.repeat(demo[None, :, -1], 7, axis=0))
    assert np.any(np.abs(probes[1:, :, :2] - demo[None, :, :2]) > 0)

    changed_gripper = probes.copy()
    changed_gripper[..., -1] *= 100.0
    descriptors = action_plan_descriptors(probes, pca, action_indices=(0, 1))
    changed_descriptors = action_plan_descriptors(changed_gripper, pca, action_indices=(0, 1))
    np.testing.assert_allclose(changed_descriptors, descriptors)


def test_action_plan_descriptor_uses_temporal_mean_in_pca_space(tmp_path: Path):
    pca = _fit_pca(np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]], dtype=np.float32), threshold=0.9)
    chunks = np.array(
        [
            [[-1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [3.0, 0.0]],
        ],
        dtype=np.float32,
    )

    descriptors = action_plan_descriptors(chunks, pca)

    expected = pca.transform(chunks.mean(axis=1))
    np.testing.assert_allclose(descriptors, expected)

    path = tmp_path / "pca.npz"
    pca.save(path)
    loaded = ActionPCA.load(path)
    np.testing.assert_allclose(loaded.components, pca.components)
    np.testing.assert_allclose(loaded.scale, pca.scale)
    assert loaded.metadata == pca.metadata


def test_action_plan_descriptor_can_ignore_reconstructed_history():
    pca = _fit_pca(
        np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        threshold=0.9,
    )
    chunks = np.array(
        [
            [[100.0, 0.0], [100.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
            [[-100.0, 0.0], [-100.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
        ],
        dtype=np.float32,
    )

    descriptors = action_plan_descriptors(
        chunks,
        pca,
        temporal_start=2,
        temporal_length=2,
    )

    np.testing.assert_allclose(descriptors[0], descriptors[1])


def test_action_plan_descriptor_preserves_temporal_bins_and_endpoint():
    pca = _fit_pca(
        np.array([[-2.0, -1.0], [0.0, 0.0], [2.0, 1.0]], dtype=np.float32),
        threshold=1.0,
    )
    chunks = np.array(
        [
            [[-2.0, -1.0], [-1.0, -0.5], [1.0, 0.5], [2.0, 1.0]],
            [[2.0, 1.0], [1.0, 0.5], [-1.0, -0.5], [-2.0, -1.0]],
        ],
        dtype=np.float32,
    )

    legacy = action_plan_descriptors(chunks, pca)
    temporal = action_plan_descriptors(
        chunks,
        pca,
        temporal_bins=2,
        pca_components=1,
        include_endpoint=True,
    )

    # The two plans have the same whole-horizon mean but opposite temporal order.
    np.testing.assert_allclose(legacy[0], legacy[1])
    assert temporal.shape == (2, 3)
    assert not np.allclose(temporal[0], temporal[1])


def test_anchor_relative_action_keeps_named_grippers_absolute():
    names = [
        *[f"left_joint_{i}" for i in range(6)],
        "left_gripper",
        *[f"right_joint_{i}" for i in range(6)],
        "right_gripper",
    ]
    mask = relative_action_mask(14, names, exclude_tokens=["gripper"])
    actions = np.full((3, 14), 2.0, dtype=np.float32)
    state = np.full(14, 0.5, dtype=np.float32)

    relative = to_model_action_chunk(actions, state, ACTION_MODE_ANCHOR_RELATIVE, mask)

    np.testing.assert_allclose(relative[:, mask], 1.5)
    np.testing.assert_allclose(relative[:, ~mask], 2.0)
    assert np.flatnonzero(~mask).tolist() == [6, 13]


def test_action_divergence_is_finite_for_separated_modes():
    descriptors = np.array(
        [[1.0, 0.0], [0.9, 0.1], [1.1, -0.1], [-1.0, 0.0], [-0.9, -0.1], [-1.1, 0.1]],
        dtype=np.float32,
    )

    cosine, l2, means = compute_action_divergence(descriptors, n_components=2)

    assert np.isfinite(cosine) and cosine > 1.5
    assert np.isfinite(l2) and l2 > 1.0
    assert means.shape == (2, 2)


def test_denoising_gain_measures_relative_pairwise_spread():
    input_descriptors = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    output_descriptors = input_descriptors * 0.5 + np.array([10.0, -3.0], dtype=np.float32)

    gain = compute_denoising_gain(input_descriptors, output_descriptors)

    assert np.isclose(gain, 0.5)


def test_delta_bic_is_positive_for_two_tight_clusters():
    rng = np.random.default_rng(21)
    descriptors = np.concatenate(
        [
            rng.normal(loc=(-4.0, 0.0), scale=0.08, size=(30, 2)),
            rng.normal(loc=(4.0, 0.0), scale=0.08, size=(30, 2)),
        ],
        axis=0,
    ).astype(np.float32)

    delta_bic, bic_k1, bic_best, best_k = compute_delta_bic(descriptors, max_components=4)

    assert delta_bic > 0
    assert bic_best < bic_k1
    assert best_k in {2, 3, 4}
