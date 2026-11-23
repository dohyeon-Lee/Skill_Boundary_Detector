import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_BUILD_SRC = (
    Path(__file__).resolve().parents[2]
    / "skillVLA/libero/configs/train_skills/build_data/src"
)
sys.path.insert(0, str(_BUILD_SRC))

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
from skill_divider import (  # noqa: E402
    EpisodeTooShortError,
    _aligned_action_chunk,
    _valid_replan_anchors,
)
from sbd_multimodality import (  # noqa: E402
    covariance_gated_angular_chord_curve,
    covariance_gated_cosine_curve,
    directional_reliability,
    gmm_overlap_entropy_curve,
    iid_scheduler_noise,
    strong_covariance_gated_cosine_curve,
    within_normalized_l2_curve,
)


def test_builder_defaults_still_resolve_from_the_project_root():
    import skill_divider

    project_root = Path(__file__).resolve().parents[3]
    assert build_skill_dataset.PROJECT_ROOT == project_root
    assert skill_divider.PROJECT_ROOT == project_root
    assert Path(build_skill_dataset.Args().dataset_dir) == (
        project_root / "dataset_filtered/libero_90_full_full"
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


@pytest.mark.parametrize("initial_mask", [None, 0])
@pytest.mark.parametrize("sharded", [False, True])
def test_cached_curve_mode_does_not_load_policy(monkeypatch, tmp_path, initial_mask, sharded):
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "skillset"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "data").mkdir()
    output_dir.mkdir()
    pd.DataFrame({"task_index": [0], "task": ["demo"]}).to_parquet(
        dataset_dir / "meta" / "tasks.parquet"
    )
    import json
    mask_frames = 20 if initial_mask is None else initial_mask
    (output_dir / "skillset_manifest.json").write_text(
        json.dumps({"detector": {"initial_mask_frames": mask_frames}}))

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

    policy = tmp_path / "metadata-only-policy"
    policy.mkdir()
    (policy / "config.json").write_text('{"n_obs_steps": 20}')
    args = Args(
            dataset_dir=str(dataset_dir),
            policy_path=str(policy),
            output_dir=str(output_dir),
            task_ids=[0],
            use_cached_curves=True,
            boundary_threshold_mode="episode_mean",
            initial_mask_frames=initial_mask,
        )
    if sharded:
        chunk = dataset_dir / 'data/chunk-000'; chunk.mkdir()
        pd.DataFrame({'episode_index': [1, 2], 'task_index': [0, 0]}).to_parquet(chunk / 'file-000.parquet')
        monkeypatch.setattr(build_skill_dataset, 'load_episodes_meta', lambda _: pd.DataFrame({'episode_index': [1, 2]}))
        assigned = tmp_path / 'assigned.json'; assigned.write_text('{"episode_ids":[2]}')
        args.episode_ids_path = str(assigned)
        skills = output_dir / 'skills/task00'; skills.mkdir(parents=True)
        (skills / 'ep00002_task00.done').touch()
        calls = []
        def load_assigned(dataset_dir, episode_ids):
            calls.append(episode_ids)
            return pd.DataFrame()  # The already-completed episode is skipped before reading frames.
        monkeypatch.setattr(build_skill_dataset, 'load_data', load_assigned)
    build_skill_dataset.main(args)
    assert args.initial_mask_frames == mask_frames
    if sharded:
        assert calls == [[2]]


def test_aligned_action_chunk_includes_history_and_copy_pads_episode_start():
    actions = np.arange(6, dtype=np.float32)[:, None]

    middle = _aligned_action_chunk(actions, anchor=2, action_delta_indices=[-2, -1, 0, 1])
    start = _aligned_action_chunk(actions, anchor=0, action_delta_indices=[-2, -1, 0, 1])

    np.testing.assert_array_equal(middle[:, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(start[:, 0], [0, 0, 0, 1])


@pytest.mark.parametrize('last_anchor', [40, 30])
def test_overlapping_masks_keep_episode_without_fake_peaks_or_smoothing(tmp_path, monkeypatch, last_anchor):
    args = Args(initial_mask_frames=20, terminal_mask_frames=24, smooth_window=7,
                savgol_polyorder=4, score_power=1, boundary_threshold_mode='global_mean',
                curves_only=True)
    ts = np.arange(0, last_anchor + 1, 5)
    scores = np.arange(len(ts), dtype=float)
    def fail(*a, **k): raise AssertionError('No valid points must not be smoothed')
    monkeypatch.setattr(build_skill_dataset, '_boundary_savgol', fail)
    boundaries = build_skill_dataset._detect_boundaries(ts, scores, 66, args)
    assert boundaries == [0, 66]
    build_skill_dataset._save_boundary_curve(tmp_path, 3152, 61, ts, scores, boundaries, 66, args)
    with np.load(tmp_path / 'ep0003152.npz') as curve:
        assert bool(curve['no_valid_boundary_anchors'])
        assert np.isnan(curve['sg_vals']).all() and np.isnan(curve['threshold_val'])
        assert not len(curve['peak_ts'])
        np.testing.assert_array_equal(curve['boundaries'], [0, 66])
    assert build_skill_dataset._detect_boundaries(ts, scores, 66, args, global_threshold=2.) == [0, 66]


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


def test_replan_anchors_identify_episode_shorter_than_future_horizon():
    with pytest.raises(EpisodeTooShortError) as exc_info:
        _valid_replan_anchors(
            n_frames=6,
            future_horizon=36,
            replan_interval=8,
        )

    assert exc_info.value.n_frames == 6
    assert exc_info.value.future_horizon == 36


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


def test_strong_reliability_gate_suppresses_uncertain_directional_modes():
    rng = np.random.default_rng(9)
    cloud = np.concatenate(
        [
            rng.normal([0.4, 0.0], [0.01, 0.3], size=(48, 2)),
            rng.normal([-0.4, 0.0], [0.01, 0.3], size=(48, 2)),
        ],
        axis=0,
    )

    standard = covariance_gated_cosine_curve(
        cloud[None], components=2, covariance="diag", n_init=3
    )[0]
    strong = strong_covariance_gated_cosine_curve(
        cloud[None], components=2, covariance="diag", n_init=3
    )[0]

    assert np.isfinite([standard, strong]).all()
    assert 0.0 < strong < standard


def test_angular_chord_suppresses_near_origin_direction_instability():
    rng = np.random.default_rng(11)
    near = np.concatenate(
        [
            rng.normal([0.02, 0.0], 0.001, size=(48, 2)),
            rng.normal([-0.02, 0.0], 0.001, size=(48, 2)),
        ],
        axis=0,
    )
    far = near * 100.0

    scores = covariance_gated_angular_chord_curve(
        np.stack([near, far]), components=2, covariance="diag", n_init=3
    )

    assert np.isfinite(scores).all()
    assert scores[0] < 0.05
    # GMM's fixed covariance regularizer makes the reliability gate only
    # approximately scale-equivariant at the tiny near-origin scale.
    np.testing.assert_allclose(scores[1], 100.0 * scores[0], rtol=5e-3)


def test_angular_chord_ignores_same_ray_radial_separation():
    rng = np.random.default_rng(13)
    cloud = np.concatenate(
        [
            rng.normal([1.0, 0.0], [0.002, 0.0001], size=(48, 2)),
            rng.normal([3.0, 0.0], [0.002, 0.0001], size=(48, 2)),
        ],
        axis=0,
    )

    score = covariance_gated_angular_chord_curve(
        cloud[None], components=2, covariance="diag", n_init=3
    )[0]

    assert np.isfinite(score)
    assert score < 0.01


def test_within_normalized_l2_is_translation_and_scale_invariant():
    rng = np.random.default_rng(17)
    cloud = np.concatenate(
        [
            rng.normal([-1.0, 0.0], 0.08, size=(48, 2)),
            rng.normal([1.0, 0.0], 0.08, size=(48, 2)),
        ],
        axis=0,
    )

    base = within_normalized_l2_curve(
        cloud[None], components=2, covariance="diag", n_init=3
    )[0]
    translated = within_normalized_l2_curve(
        (cloud + np.asarray([100.0, -70.0]))[None],
        components=2,
        covariance="diag",
        n_init=3,
    )[0]
    scaled = within_normalized_l2_curve(
        (cloud * 4.0)[None], components=2, covariance="diag", n_init=3
    )[0]

    assert np.isfinite(base)
    np.testing.assert_allclose(translated, base, rtol=2e-4)
    np.testing.assert_allclose(scaled, base, rtol=2e-4)


def test_within_normalized_l2_prefers_compact_separated_modes():
    rng = np.random.default_rng(23)
    compact = np.concatenate(
        [
            rng.normal([-1.0, 0.0], 0.05, size=(96, 2)),
            rng.normal([1.0, 0.0], 0.05, size=(96, 2)),
        ],
        axis=0,
    )
    broad = np.concatenate(
        [
            rng.normal([-1.0, 0.0], 0.60, size=(96, 2)),
            rng.normal([1.0, 0.0], 0.60, size=(96, 2)),
        ],
        axis=0,
    )

    scores = within_normalized_l2_curve(
        np.stack([compact, broad]), components=2, covariance="diag", n_init=3
    )

    assert scores[0] > 5.0 * scores[1]


def test_gmm_overlap_entropy_is_bounded_and_rewards_distinct_balanced_modes():
    rng = np.random.default_rng(29)
    overlapping = np.concatenate(
        [
            rng.normal([-0.05, 0.0], 0.45, size=(48, 2)),
            rng.normal([0.05, 0.0], 0.45, size=(48, 2)),
        ]
    )
    separated = np.concatenate(
        [
            rng.normal([-1.0, 0.0], 0.05, size=(48, 2)),
            rng.normal([1.0, 0.0], 0.05, size=(48, 2)),
        ]
    )

    scores = gmm_overlap_entropy_curve(
        np.stack([overlapping, separated]),
        components=2,
        covariance="diag",
        n_init=5,
    )

    assert np.all((0.0 <= scores) & (scores <= 1.0))
    assert scores[1] > scores[0]
    assert scores[1] > 0.9


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
