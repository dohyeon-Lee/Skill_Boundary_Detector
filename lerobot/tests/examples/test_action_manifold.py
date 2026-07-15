import sys
from pathlib import Path

import numpy as np

LIBERO_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "libero"
sys.path.insert(0, str(LIBERO_EXAMPLES))

from action_manifold import (  # noqa: E402
    ActionPCA,
    GRIPPER_DISCRETE,
    NumpyActionNormalizer,
    RunningCovariance,
    ACTION_MODE_ANCHOR_RELATIVE,
    action_plan_descriptors,
    compute_action_divergence,
    make_pca_action_probes,
    relative_action_mask,
    to_model_action_chunk,
)


def _fit_pca(values: np.ndarray, threshold: float = 0.95) -> ActionPCA:
    accumulator = RunningCovariance(values.shape[1])
    accumulator.update_batch(values)
    return ActionPCA.from_covariance(accumulator, threshold, metadata={"test": True})


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
    assert loaded.metadata == pca.metadata


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
