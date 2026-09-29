import sys
from pathlib import Path

import numpy as np
import pandas as pd


LIBERO_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "libero"
sys.path.insert(0, str(LIBERO_EXAMPLES))

import build_skill_dataset  # noqa: E402
from action_manifold import (  # noqa: E402
    ACTION_MODE_DATASET,
    NumpyActionNormalizer,
    PCA_SCALE_NONE,
)
from build_skill_dataset import (  # noqa: E402
    Args,
    _fit_dataset_action_pca,
    _load_episode_policy_inputs,
)
from skill_divider import _aligned_action_chunk, _valid_replan_anchors  # noqa: E402
from sbd_multimodality import (  # noqa: E402
    covariance_gated_cosine_curve,
    directional_reliability,
    iid_scheduler_noise,
)


def test_state_only_policy_skips_all_visual_loading():
    def fail():
        raise AssertionError("state-only policy must not load visual inputs")

    camera_frames, dino_tokens = _load_episode_policy_inputs(
        use_dino=False,
        state_only=True,
        load_dino=fail,
        load_cameras=fail,
    )

    assert camera_frames == {}
    assert dino_tokens is None


def test_visual_policy_loads_only_its_selected_input():
    camera_frames, dino_tokens = _load_episode_policy_inputs(
        use_dino=True,
        state_only=False,
        load_dino=lambda: "tokens",
        load_cameras=lambda: (_ for _ in ()).throw(
            AssertionError("DINO policy must not decode camera videos")
        ),
    )
    assert camera_frames == {}
    assert dino_tokens == "tokens"

    expected_frames = {"camera": "frames"}
    camera_frames, dino_tokens = _load_episode_policy_inputs(
        use_dino=False,
        state_only=False,
        load_dino=lambda: (_ for _ in ()).throw(
            AssertionError("raw-camera policy must not load DINO tokens")
        ),
        load_cameras=lambda: expected_frames,
    )
    assert camera_frames is expected_frames
    assert dino_tokens is None


def test_cached_curve_mode_does_not_load_policy(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "skillset"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "data").mkdir()
    output_dir.mkdir()
    pd.DataFrame({"task_index": [0], "task": ["demo"]}).to_parquet(
        dataset_dir / "meta" / "tasks.parquet"
    )
    (output_dir / "skillset_manifest.json").write_text("{}")

    monkeypatch.setattr(
        build_skill_dataset,
        "load_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached segmentation must not load the DP policy")
        ),
    )
    monkeypatch.setattr(
        build_skill_dataset,
        "load_episodes_meta",
        lambda _path: pd.DataFrame({"episode_index": pd.Series(dtype=np.int64)}),
    )

    build_skill_dataset.main(
        Args(
            dataset_dir=str(dataset_dir),
            policy_path=str(tmp_path / "missing-policy"),
            output_dir=str(output_dir),
            task_ids=[0],
            use_cached_curves=True,
            boundary_threshold_mode="episode_mean",
        )
    )


def test_aligned_action_chunk_includes_history_and_copy_pads_episode_start():
    actions = np.arange(6, dtype=np.float32)[:, None]

    middle = _aligned_action_chunk(actions, anchor=2, action_delta_indices=[-2, -1, 0, 1])
    start = _aligned_action_chunk(actions, anchor=0, action_delta_indices=[-2, -1, 0, 1])

    np.testing.assert_array_equal(middle[:, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(start[:, 0], [0, 0, 0, 1])


def test_replan_anchors_exclude_incomplete_future_tail():
    anchors = _valid_replan_anchors(
        n_frames=100,
        future_horizon=24,
        replan_interval=1,
    )

    assert anchors[0] == 0
    assert anchors[-1] == 76
    assert len(anchors) == 77
    assert set(range(77, 100)).isdisjoint(anchors)


def test_iid_scheduler_noise_is_seeded_and_has_requested_shape():
    first = iid_scheduler_noise(96, 24, 7, seed=42)
    second = iid_scheduler_noise(96, 24, 7, seed=42)

    assert first.shape == (96, 24, 7)
    np.testing.assert_array_equal(first, second)


def test_directional_reliability_ignores_parallel_but_penalizes_orthogonal_noise():
    mean = np.array([2.0, 0.0])

    np.testing.assert_allclose(
        directional_reliability(mean, np.diag([10.0, 0.0])), 1.0
    )
    assert directional_reliability(mean, np.diag([0.0, 4.0])) == 0.5


def test_covariance_gated_cosine_curve_is_finite_for_clear_modes():
    rng = np.random.default_rng(7)
    cloud = np.concatenate(
        [
            rng.normal([2.0, 0.0], 0.03, size=(24, 2)),
            rng.normal([-2.0, 0.0], 0.03, size=(24, 2)),
        ],
        axis=0,
    )

    score = covariance_gated_cosine_curve(
        cloud[None], components=2, covariance="diag"
    )

    assert score.shape == (1,)
    assert np.isfinite(score[0])
    assert score[0] > 1.9


def test_action_pca_fits_only_current_future_descriptor_slice(monkeypatch, tmp_path):
    actions = np.arange(5, dtype=np.float32)[:, None]
    states = np.zeros_like(actions)
    monkeypatch.setattr(
        build_skill_dataset,
        "_iter_state_action_episodes",
        lambda _: iter([(0, actions, states)]),
    )

    pca = _fit_dataset_action_pca(
        dataset_dir=tmp_path,
        action_delta_indices=(-1, 0, 1),
        descriptor_start=1,
        descriptor_horizon=2,
        stride=1,
        action_dim=1,
        action_indices=(0,),
        action_mode=ACTION_MODE_DATASET,
        rel_mask=None,
        normalizer=NumpyActionNormalizer(mode="IDENTITY", stats={}),
        variance_threshold=1.0,
        scale_mode=PCA_SCALE_NONE,
        metadata={"test": True},
    )

    # Valid anchors 0..3 summarize [t, t+1], yielding 0.5, 1.5, 2.5, 3.5.
    assert pca.sample_count == 4
    np.testing.assert_allclose(pca.mean, [2.0])
