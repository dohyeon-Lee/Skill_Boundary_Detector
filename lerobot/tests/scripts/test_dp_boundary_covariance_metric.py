import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skills/skill_eval/skill_boundary_detection/src/evaluate.py"
)
SPEC = importlib.util.spec_from_file_location("dp_boundary_ablation_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_directional_reliability_ignores_parallel_variance() -> None:
    mean = np.array([3.0, 4.0, 0.0])
    direction = mean / np.linalg.norm(mean)
    parallel_covariance = 10.0 * np.outer(direction, direction)

    np.testing.assert_allclose(
        MODULE._directional_reliability(mean, parallel_covariance), 1.0
    )


def test_directional_reliability_uses_orthogonal_signal_to_noise_ratio() -> None:
    mean = np.array([3.0, 4.0, 0.0])
    # Squared mean magnitude and perpendicular covariance energy are both 25.
    orthogonal_covariance = np.diag([0.0, 0.0, 25.0])

    reliability = MODULE._directional_reliability(mean, orthogonal_covariance)
    scaled_reliability = MODULE._directional_reliability(
        7.0 * mean, 49.0 * orthogonal_covariance
    )

    np.testing.assert_allclose(reliability, 0.5)
    np.testing.assert_allclose(scaled_reliability, reliability)


def test_directional_reliability_rejects_zero_mean() -> None:
    assert MODULE._directional_reliability(np.zeros(3), np.eye(3)) == 0.0


def test_component_filter_uses_effective_sample_support() -> None:
    # With 25 inputs, weights 0.04 and 0.08 represent one and two effective
    # samples respectively. The singleton is rejected and the pair retained.
    keep = MODULE._retained_component_indices(
        np.array([0.04, 0.08, 0.88]),
        25,
        {"min_effective_samples": 2},
    )
    np.testing.assert_array_equal(keep, np.array([1, 2]))


def test_component_filter_adapts_to_probe_count() -> None:
    # The same two-effective-sample rule becomes 2/50 rather than staying at
    # the old hard-coded 0.08 mixture weight.
    keep = MODULE._retained_component_indices(
        np.array([0.03, 0.04, 0.93]),
        50,
        {"min_effective_samples": 2},
    )
    np.testing.assert_array_equal(keep, np.array([1, 2]))


def test_whitened_descriptor_spread_removes_coordinate_scale() -> None:
    reference = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    stretched = reference * np.array([2.0, 7.0])

    np.testing.assert_allclose(
        MODULE._whitened_descriptor_spread(reference, np.ones(2)),
        MODULE._whitened_descriptor_spread(stretched, np.array([2.0, 7.0])),
    )


def test_whitened_descriptor_spread_is_dimension_normalized_pairwise_rms() -> None:
    # The sole pair differs by one whitened unit in each of two dimensions.
    # Its RMS distance per dimension is therefore exactly one.
    cloud = np.array([[0.0, 0.0], [2.0, 4.0]])
    spread = MODULE._whitened_descriptor_spread(cloud, np.array([2.0, 4.0]))

    np.testing.assert_allclose(spread, 1.0)


def test_denoising_convergence_reports_interval_and_timing_scores() -> None:
    raw = {
        # Both noise samples make their full correction during step 0->2 and
        # remain unchanged from step 2 onward.
        "x0_descriptor_snapshots": np.array(
            [[[[0.0], [2.0], [2.0]], [[1.0], [3.0], [3.0]]]]
        ),
        "denoising_snapshot_steps": np.array([0, 2, 4]),
    }

    scores = MODULE._denoising_convergence_scores(raw, np.ones(1))

    np.testing.assert_allclose(scores["denoising_drift_0_2"], [2.0])
    np.testing.assert_allclose(scores["denoising_drift_2_4"], [0.0])
    np.testing.assert_allclose(scores["denoising_total_drift"], [2.0])
    np.testing.assert_allclose(scores["denoising_late_drift"], [0.0])
    np.testing.assert_allclose(scores["denoising_correction_step"], [1.0])
    np.testing.assert_allclose(scores["denoising_knee_step"], [2.0])


def test_denoising_threshold_step_is_discrete_first_passing_interval() -> None:
    raw = {
        "x0_descriptor_snapshots": np.array(
            [[[[0.0], [0.5], [0.6]], [[1.0], [1.5], [1.6]]]]
        ),
        "denoising_snapshot_steps": np.array([0, 2, 4]),
    }

    scores = MODULE._denoising_convergence_scores(
        raw, np.ones(1), convergence_threshold=0.2
    )

    # Movement is 0.5 for 0->2 and 0.1 for 2->4, so convergence is step 4.
    np.testing.assert_array_equal(scores["denoising_threshold_step"], [4.0])


def test_denoising_relative_threshold_uses_each_anchors_initial_drift() -> None:
    raw = {
        "x0_descriptor_snapshots": np.array(
            [
                [[[0.0], [1.0], [1.4], [1.5]]],
                [[[0.0], [2.0], [2.4], [2.5]]],
            ]
        ),
        "denoising_snapshot_steps": np.array([0, 2, 4, 6]),
    }

    scores = MODULE._denoising_convergence_scores(
        raw, np.ones(1), relative_convergence_threshold=0.25
    )

    # Anchor 1: bar=0.25, corrections 1.0,0.4,0.1 -> step 6.
    # Anchor 2: bar=0.50, corrections 2.0,0.4,0.1 -> step 4.
    np.testing.assert_array_equal(
        scores["denoising_relative_threshold_step"], [6.0, 4.0]
    )


def test_denoising_mean_aggregation_and_episode_geometric_threshold() -> None:
    raw = {
        "x0_descriptor_snapshots": np.array(
            [
                [
                    [[0.0], [1.0], [1.1]],
                    [[0.0], [1.0], [1.1]],
                    [[0.0], [4.0], [4.4]],
                ],
                [
                    [[0.0], [1.0], [1.1]],
                    [[0.0], [1.0], [1.1]],
                    [[0.0], [1.0], [1.1]],
                ],
            ]
        ),
        "denoising_snapshot_steps": np.array([0, 2, 4]),
        "replan_ts": np.array([0, 5]),
    }

    scores = MODULE._denoising_convergence_scores(
        raw,
        np.ones(1),
        sample_aggregation="mean",
        convergence_threshold_mode="episode_geometric_mean",
    )

    # Mean sample drifts are [2.0, 0.2] and [1.0, 0.1]. Their geometric
    # episode mean is (2*0.2*1*0.1)^(1/4) = sqrt(0.2).
    np.testing.assert_allclose(scores["denoising_drift_0_2"], [2.0, 1.0])
    np.testing.assert_allclose(
        scores["denoising_convergence_threshold_value"], np.sqrt(0.2)
    )
    np.testing.assert_array_equal(scores["denoising_threshold_step"], [4.0, 4.0])


def test_aligned_denoising_maps_offsets_to_absolute_episode_time() -> None:
    raw = {
        "replan_ts": np.array([0, 2]),
        "n_frames": np.array(5),
        "denoising_snapshot_steps": np.array([0, 2, 4]),
        # At absolute frame 2, anchor 0 offset 2 contributes [3,.3] and
        # anchor 2 offset 0 contributes [5,.5]. Their median is [4,.4].
        "x0_temporal_corrections": np.array(
            [
                [[ [1.0, 0.1], [2.0, 0.2], [3.0, 0.3] ]],
                [[ [5.0, 0.5], [6.0, 0.6], [7.0, 0.7] ]],
            ]
        ),
    }

    scores = MODULE._aligned_denoising_convergence_scores(
        raw, convergence_threshold=0.75
    )

    np.testing.assert_array_equal(scores["denoising_aligned_ts"], np.arange(5))
    np.testing.assert_allclose(
        scores["denoising_aligned_drift_0_2"], [1.0, 2.0, 4.0, 6.0, 7.0]
    )
    np.testing.assert_array_equal(
        scores["denoising_aligned_support"], [1, 1, 2, 1, 1]
    )
    np.testing.assert_array_equal(
        scores["denoising_aligned_threshold_step"], np.full(5, 4.0)
    )


def test_aligned_denoising_derive_uses_dense_target_timestamps() -> None:
    raw = {
        "replan_ts": np.array([0, 2]),
        "n_frames": np.array(5),
        "descriptors": np.zeros((2, 2, 1), dtype=np.float32),
        "denoising_snapshot_steps": np.array([0, 2, 4]),
        "x0_temporal_corrections": np.array(
            [
                [[[1.0, 0.1], [2.0, 0.1], [3.0, 0.1]]],
                [[[3.0, 0.1], [2.0, 0.1], [1.0, 0.1]]],
            ]
        ),
    }
    zero_scores = {
        key: np.zeros(2, dtype=np.float32) for key in MODULE._CLUSTER_SCORE_KEYS
    }

    derived = MODULE._derive(
        raw,
        {},
        "denoising_aligned_threshold_step",
        {
            "method": "sustained_plateau",
            "denoising_convergence_threshold": 0.5,
            "threshold": "fixed",
            "threshold_value": 3.0,
            "plateau_min_active_points": 1,
            "plateau_exclude_initial_group": False,
            "terminal_mask_frames": 0,
            "min_skill_len": 1,
        },
        min_skill_len=1,
        scores=zero_scores,
    )

    np.testing.assert_array_equal(derived["replan_ts"], np.arange(5))
    assert len(derived["score"]) == 5


def test_prediction_loss_alignment_maps_chunk_offsets_to_episode_time() -> None:
    raw = {
        "replan_ts": np.array([0, 2]),
        "n_frames": np.array(5),
        "prediction_loss_per_offset": np.array(
            [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]], dtype=np.float32
        ),
    }

    timestamps, losses, support = MODULE._aligned_prediction_loss(raw)

    np.testing.assert_array_equal(timestamps, np.arange(5))
    np.testing.assert_allclose(losses, [1.0, 2.0, 4.0, 6.0, 7.0])
    np.testing.assert_array_equal(support, [1, 1, 2, 1, 1])


def test_running_surprise_resets_segment_loss_history_after_boundary() -> None:
    values = np.array([1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 6.0])
    timestamps = np.arange(len(values), dtype=np.int64)

    candidates, surprise = MODULE._running_surprise(
        values,
        timestamps,
        gap=2.0,
        min_history_points=3,
    )

    assert candidates == [3, 6]
    # Algorithm 1 includes the current loss in the segment mean before testing.
    np.testing.assert_allclose(surprise[3], 3.0)
    # The second spike is compared only with points after the first reset.
    np.testing.assert_allclose(surprise[6], 10.0 / 3.0)


def test_prediction_loss_current_metric_uses_only_zero_future_offset() -> None:
    raw = {
        "replan_ts": np.array([0, 1, 2]),
        "n_frames": np.array(30),
        "descriptors": np.zeros((3, 2, 1), dtype=np.float32),
        "prediction_loss_per_offset": np.array(
            [[1.0, 50.0], [5.0, 0.0], [1.0, 50.0]], dtype=np.float32
        ),
    }
    neutral_scores = MODULE._empty_cluster_scores(3)

    derived = MODULE._derive(
        raw,
        {},
        "prediction_loss_current",
        {
            "method": "peak",
            "threshold": "fixed",
            "threshold_value": 2.0,
            "smooth_window": 1,
            "polyorder": 0,
            "start_guard_frames": 0,
            "nms_frames": 0,
            "terminal_mask_frames": 0,
            "min_skill_len": 1,
        },
        min_skill_len=1,
        scores=neutral_scores,
    )

    np.testing.assert_allclose(derived["score"], [1.0, 5.0, 1.0])
    np.testing.assert_array_equal(derived["boundaries"], [1])


def test_prefix_probe_payload_keeps_gt_and_requested_probe_count() -> None:
    descriptors = np.arange(3 * 10 * 2, dtype=np.float32).reshape(3, 10, 2)
    payload = {
        "replan_ts": np.array([0, 5, 10]),
        "descriptors": descriptors,
        "n_frames": np.array(24),
    }

    prefix = MODULE._prefix_probe_payload(payload, probe_count=4)

    assert prefix["descriptors"].shape == (3, 5, 2)
    np.testing.assert_array_equal(prefix["descriptors"], descriptors[:, :5])
    np.testing.assert_array_equal(prefix["replan_ts"], payload["replan_ts"])
    assert int(prefix["n_frames"]) == 24


def test_terminal_mask_is_applied_before_threshold_estimation() -> None:
    values = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 100.0, 0.0])
    timestamps = np.arange(len(values), dtype=np.int64)
    common = {
        "method": "peak",
        "threshold": "mean",
        "threshold_scale": 1.0,
        "smooth_window": 1,
        "polyorder": 0,
        "nms_frames": 0,
        "prominence": 0.0,
    }

    unmasked_cuts, unmasked_threshold, _ = MODULE._boundaries(
        values, timestamps, common
    )
    masked_cuts, masked_threshold, masked_smooth = MODULE._boundaries(
        values, timestamps, {**common, "terminal_mask_frames": 2}
    )

    np.testing.assert_array_equal(unmasked_cuts, np.array([5]))
    np.testing.assert_array_equal(masked_cuts, np.array([1]))
    assert masked_threshold < unmasked_threshold
    # The ignored tail is flat and cannot leak back into the valid smoothing.
    np.testing.assert_allclose(masked_smooth[-2:], masked_smooth[4])


def test_start_guard_is_independent_from_inter_peak_nms() -> None:
    values = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0])
    timestamps = np.array([0, 3, 6, 15, 27, 30, 33], dtype=np.int64)
    common = {
        "method": "peak",
        "threshold": "fixed",
        "threshold_value": 0.0,
        "smooth_window": 1,
        "polyorder": 0,
        "nms_frames": 20,
    }

    legacy_cuts, _, _ = MODULE._boundaries(values, timestamps, common)
    unguarded_cuts, _, _ = MODULE._boundaries(
        values, timestamps, {**common, "start_guard_frames": 0}
    )

    np.testing.assert_array_equal(legacy_cuts, np.array([30]))
    np.testing.assert_array_equal(unguarded_cuts, np.array([3, 30]))


def test_score_power_raises_episode_mean_selectivity() -> None:
    # Both local maxima exceed the arithmetic mean, while only the dominant
    # maximum exceeds the RMS-equivalent threshold produced by power=2.
    values = np.array([0.0, 2.0, 0.8, 1.0, 0.8, 0.0])
    timestamps = np.arange(len(values), dtype=np.int64)
    common = {
        "method": "peak",
        "threshold": "mean",
        "threshold_scale": 1.0,
        "smooth_window": 1,
        "polyorder": 0,
        "start_guard_frames": 0,
        "nms_frames": 0,
        "terminal_mask_frames": 0,
    }

    linear_cuts, linear_threshold, _ = MODULE._boundaries(
        values, timestamps, {**common, "score_power": 1.0}
    )
    squared_cuts, squared_threshold, squared_curve = MODULE._boundaries(
        values, timestamps, {**common, "score_power": 2.0}
    )

    np.testing.assert_array_equal(linear_cuts, np.array([1, 3]))
    np.testing.assert_array_equal(squared_cuts, np.array([1]))
    np.testing.assert_allclose(squared_curve, np.square(values))
    assert np.sqrt(squared_threshold) > linear_threshold


def test_rms_threshold_is_bounded_and_matches_squared_mean_selection() -> None:
    values = np.array([0.0, 2.0, 0.8, 1.0, 0.8, 0.0])
    timestamps = np.arange(len(values), dtype=np.int64)
    common = {
        "method": "peak",
        "threshold_scale": 1.0,
        "smooth_window": 1,
        "polyorder": 0,
        "start_guard_frames": 0,
        "nms_frames": 0,
        "terminal_mask_frames": 0,
    }

    rms_cuts, rms_threshold, _ = MODULE._boundaries(
        values,
        timestamps,
        {**common, "threshold": "rms", "score_power": 1.0},
    )
    squared_cuts, squared_threshold, _ = MODULE._boundaries(
        values,
        timestamps,
        {**common, "threshold": "mean", "score_power": 2.0},
    )

    assert np.mean(values) <= rms_threshold <= np.max(values)
    np.testing.assert_array_equal(rms_cuts, squared_cuts)
    np.testing.assert_allclose(np.square(rms_threshold), squared_threshold)


def test_global_min_skill_length_keeps_stronger_nearby_peak() -> None:
    timestamps = np.array([0, 15, 66, 78, 120], dtype=np.int64)
    scores = np.array([0.0, 0.4, 0.5, 0.8, 0.0])

    cuts = MODULE._enforce_global_min_skill_length(
        np.array([15, 66, 78]),
        timestamps,
        scores,
        n_frames=160,
        min_skill_len=20,
    )

    # The short initial segment is rejected; among the 12-frame pair, retain
    # the stronger later peak rather than whichever appears first.
    np.testing.assert_array_equal(cuts, np.array([78]))


def test_persistent_local_uses_scale_persistence_and_robust_noise_floor() -> None:
    x = np.arange(31, dtype=np.float64)
    values = (
        np.exp(-0.5 * np.square((x - 7.0) / 2.0))
        + 2.0 * np.exp(-0.5 * np.square((x - 22.0) / 2.0))
        + np.asarray([0.02 * (-1) ** index for index in range(len(x))])
    )
    timestamps = np.arange(len(values), dtype=np.int64) * 3

    cuts, threshold, _ = MODULE._boundaries(
        values,
        timestamps,
        {
            "method": "persistent_local",
            "smooth_windows": [5, 7, 9],
            "smooth_window": 7,
            "polyorder": 4,
            "start_guard_frames": 0,
            "nms_frames": 0,
            "terminal_mask_frames": 0,
        },
    )

    np.testing.assert_array_equal(cuts, np.array([21, 66]))
    assert np.isnan(threshold)


def test_sustained_plateau_does_not_bridge_noninitial_gap() -> None:
    values = np.array([0.0, 10.0, 10.0, 8.0, 10.0, 0.0])
    timestamps = np.arange(len(values), dtype=np.int64) * 5

    cuts, threshold, raw_curve = MODULE._boundaries(
        values,
        timestamps,
        {
            "method": "sustained_plateau",
            "threshold": "fixed",
            "threshold_value": 9.0,
            "plateau_min_active_points": 3,
            "plateau_max_gap_points": 1,
            "plateau_exclude_initial_group": True,
            "terminal_mask_frames": 0,
        },
    )

    assert threshold == 9.0
    np.testing.assert_array_equal(raw_curve, values)
    # Neither consecutive active run reaches three points. Gap bridging is
    # reserved for the excluded initialization chunk, so no cut is emitted.
    assert not len(cuts)


def test_sustained_plateau_rejects_short_and_initial_groups() -> None:
    values = np.array(
        [10.0, 10.0, 8.0, 10.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0]
    )
    timestamps = np.arange(len(values), dtype=np.int64) * 5

    cuts, _, _ = MODULE._boundaries(
        values,
        timestamps,
        {
            "method": "sustained_plateau",
            "threshold": "fixed",
            "threshold_value": 9.0,
            "plateau_min_active_points": 3,
            "plateau_max_gap_points": 1,
            "plateau_exclude_initial_group": True,
            "terminal_mask_frames": 0,
        },
    )

    # The six-anchor initial group is dropped wholesale. The final two-point
    # group is too short, so neither produces a boundary.
    assert not len(cuts)


def test_sustained_plateau_allows_only_one_gap_in_entire_group() -> None:
    values = np.array([0.0, 10.0, 8.0, 10.0, 8.0, 10.0, 10.0, 10.0])
    timestamps = np.arange(len(values), dtype=np.int64) * 5

    cuts, _, _ = MODULE._boundaries(
        values,
        timestamps,
        {
            "method": "sustained_plateau",
            "threshold": "fixed",
            "threshold_value": 9.0,
            "plateau_min_active_points": 3,
            "plateau_max_gap_points": 1,
            "plateau_exclude_initial_group": True,
            "terminal_mask_frames": 0,
        },
    )

    # The second inactive point splits the short 1,3 group from the valid
    # consecutive 5,6,7 group. Only the latter emits its center (index 6).
    np.testing.assert_array_equal(cuts, np.array([30]))


def test_sustained_plateau_ignores_singleton_between_valid_chunks() -> None:
    values = np.array(
        [0.0, 10.0, 10.0, 10.0, 8.0, 10.0, 8.0, 10.0, 10.0, 10.0, 10.0]
    )
    timestamps = np.arange(len(values), dtype=np.int64) * 5

    cuts, _, _ = MODULE._boundaries(
        values,
        timestamps,
        {
            "method": "sustained_plateau",
            "threshold": "fixed",
            "threshold_value": 9.0,
            "plateau_min_active_points": 3,
            "plateau_max_gap_points": 1,
            "plateau_exclude_initial_group": True,
            "terminal_mask_frames": 0,
        },
    )

    # Runs 1..3 and 7..10 are independently valid. The isolated active point
    # at index 5 is ignored rather than greedily attached to the first chunk.
    np.testing.assert_array_equal(cuts, np.array([10, 45]))


def test_sustained_plateau_bridges_only_the_excluded_initial_chunk() -> None:
    values = np.array(
        [10.0, 10.0, 8.0, 10.0, 10.0, 10.0, 8.0, 8.0, 10.0, 10.0, 10.0]
    )
    timestamps = np.arange(len(values), dtype=np.int64) * 5

    cuts, _, _ = MODULE._boundaries(
        values,
        timestamps,
        {
            "method": "sustained_plateau",
            "threshold": "fixed",
            "threshold_value": 9.0,
            "plateau_min_active_points": 3,
            "plateau_max_gap_points": 1,
            "plateau_exclude_initial_group": True,
            "terminal_mask_frames": 0,
        },
    )

    # Initial 10,10,8,10,10,10 is one bridged chunk and is discarded. The
    # later consecutive run 8..10 remains and emits its center at index 9.
    np.testing.assert_array_equal(cuts, np.array([45]))


def test_ordered_boundary_f1_matches_each_cut_once() -> None:
    score = MODULE._ordered_boundary_f1(
        np.array([10, 20, 40]), np.array([12, 18, 70]), tolerance_frames=3
    )

    np.testing.assert_allclose(score, 2.0 / 3.0)
    np.testing.assert_allclose(
        MODULE._ordered_boundary_f1(
            np.empty(0), np.empty(0), tolerance_frames=6
        ),
        1.0,
    )


def test_prominence_threshold_is_invariant_to_episode_score_offset() -> None:
    timestamps = np.arange(0, 21, 3, dtype=np.int64)
    shape = np.array([0.0, 0.2, 0.5, 1.2, 0.4, 0.2, 0.0])
    config = {
        "method": "peak",
        "threshold": "fixed",
        "threshold_target": "prominence",
        "threshold_value": 0.5,
        "smooth_window": 1,
        "polyorder": 0,
        "start_guard_frames": 0,
        "nms_frames": 0,
        "terminal_mask_frames": 0,
    }

    low_cuts, low_threshold, _ = MODULE._boundaries(shape, timestamps, config)
    high_cuts, high_threshold, _ = MODULE._boundaries(
        shape + 10.0, timestamps, config
    )

    np.testing.assert_array_equal(low_cuts, np.array([9]))
    np.testing.assert_array_equal(high_cuts, low_cuts)
    assert low_threshold == high_threshold == 0.5


def test_prominence_threshold_pool_contains_peaks_not_curve_samples() -> None:
    timestamps = np.arange(0, 21, 3, dtype=np.int64)
    smooth = np.array([3.0, 3.2, 3.5, 4.2, 3.4, 3.2, 3.0])
    boundary = {
        "method": "peak",
        "threshold_target": "prominence",
        "smooth_window": 1,
        "polyorder": 0,
        "start_guard_frames": 0,
        "nms_frames": 0,
        "terminal_mask_frames": 0,
    }

    values = MODULE._threshold_pool_values(
        {"score": smooth, "smooth": smooth, "replan_ts": timestamps},
        boundary,
    )

    np.testing.assert_allclose(values, [1.2])


def test_sample0_reference_centered_cluster_score_is_translation_invariant() -> None:
    rng = np.random.default_rng(7)
    cloud = np.concatenate(
        [
            rng.normal([-1.0, 0.3], 0.04, size=(12, 2)),
            rng.normal([0.8, -0.5], 0.04, size=(13, 2)),
        ],
        axis=0,
    )[None]
    config = {
        "selection": "fixed",
        "k": 2,
        "covariance": "diag",
        "n_init": 3,
        "max_iter": 100,
        "min_effective_samples": 0,
        "weighted": True,
        "reference_center": "sample0",
    }

    original = MODULE._cluster_scores(cloud, config)
    translated = MODULE._cluster_scores(cloud + np.array([100.0, -40.0]), config)

    np.testing.assert_allclose(
        original["covariance_gated_cosine"],
        translated["covariance_gated_cosine"],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        original["center_norm"], translated["center_norm"], rtol=1e-5, atol=1e-6
    )


def test_within_normalized_l2_rewards_compact_separated_modes() -> None:
    rng = np.random.default_rng(19)
    compact = np.concatenate(
        [
            rng.normal([-1.0, 0.0], 0.03, size=(20, 2)),
            rng.normal([1.0, 0.0], 0.03, size=(20, 2)),
        ]
    )
    broad = np.concatenate(
        [
            rng.normal([-1.0, 0.0], 0.30, size=(20, 2)),
            rng.normal([1.0, 0.0], 0.30, size=(20, 2)),
        ]
    )
    config = {
        "selection": "fixed",
        "k": 2,
        "covariance": "diag",
        "n_init": 5,
        "max_iter": 100,
        "min_effective_samples": 0,
        "weighted": True,
    }

    scores = MODULE._cluster_scores(np.stack([compact, broad]), config)

    assert scores["within_normalized_l2"][0] > scores["within_normalized_l2"][1]
    np.testing.assert_allclose(scores["l2"][0], scores["l2"][1], rtol=0.15)


def test_angular_chord_preserves_direction_and_physical_scale() -> None:
    unit = MODULE._angular_chord_distance([1.0, 0.0], [0.0, 1.0])
    scaled = MODULE._angular_chord_distance([10.0, 0.0], [0.0, 10.0])

    np.testing.assert_allclose(unit, np.sqrt(2.0))
    np.testing.assert_allclose(scaled, 10.0 * unit)
    np.testing.assert_allclose(
        MODULE._angular_chord_distance([1.0, 0.0], [3.0, 0.0]), 0.0
    )
    np.testing.assert_allclose(
        MODULE._angular_chord_distance([0.0, 0.0], [1.0, 0.0]), 0.0
    )


def test_dtw_mapping_tracks_time_stretched_trajectory() -> None:
    reference = np.arange(4, dtype=np.float64)[:, None]
    source = np.array([0.0, 0.0, 1.0, 2.0, 3.0])[:, None]

    mapping = MODULE._dtw_source_to_reference(source, reference)

    assert np.all(np.diff(mapping) >= 0)
    np.testing.assert_allclose(mapping[[0, -1]], [0.0, 3.0])


def test_task_consistency_allows_one_count_difference() -> None:
    task_rows = {
        1: [("model", {"boundaries": np.array([10, 20])})],
        2: [("model", {"boundaries": np.array([11, 21, 40])})],
    }
    identity = np.arange(50, dtype=np.float64)
    metrics = MODULE._task_consistency_metrics(
        task_rows,
        {1: (identity, identity), 2: (identity, identity)},
        tolerance_frames=2,
    )["model"]

    assert metrics["count_consistency_pm1"] == 1.0
    assert np.isnan(metrics["count_conditioned_aligned_boundary_f1"])
    assert metrics["count_conditioned_alignment_pairs"] == 0


def test_task_consistency_compares_positions_only_with_equal_counts() -> None:
    task_rows = {
        1: [("model", {"boundaries": np.array([10, 20])})],
        2: [("model", {"boundaries": np.array([11, 21])})],
        3: [("model", {"boundaries": np.array([10, 20, 40])})],
        4: [("model", {"boundaries": np.array([11, 21, 70])})],
    }
    identity = np.arange(80, dtype=np.float64)
    metrics = MODULE._task_consistency_metrics(
        task_rows,
        {
            episode_id: (identity, identity)
            for episode_id in task_rows
        },
        tolerance_frames=2,
    )["model"]

    # The 2-boundary pair scores 1.0 and the 3-boundary pair scores 2/3.
    # Cross-count pairs are deliberately excluded.
    np.testing.assert_allclose(
        metrics["count_conditioned_aligned_boundary_f1"], 5.0 / 6.0
    )
    assert metrics["count_conditioned_alignment_pairs"] == 2


def test_task_consistency_counts_two_empty_outputs_as_agreement() -> None:
    task_rows = {
        1: [("model", {"boundaries": np.empty(0, dtype=np.int64)})],
        2: [("model", {"boundaries": np.empty(0, dtype=np.int64)})],
        3: [("model", {"boundaries": np.array([20])})],
    }
    identity = np.arange(50, dtype=np.float64)
    metrics = MODULE._task_consistency_metrics(
        task_rows,
        {
            episode_id: (identity, identity)
            for episode_id in task_rows
        },
        tolerance_frames=2,
    )["model"]

    np.testing.assert_allclose(
        metrics["count_conditioned_aligned_boundary_f1"], 1.0
    )
    assert metrics["count_conditioned_alignment_pairs"] == 1
    assert metrics["count_conditioned_empty_alignment_pairs"] == 1
    assert metrics["count_conditioned_nonempty_alignment_pairs"] == 0


def test_root_global_comparison_panel_includes_only_global_experiments() -> None:
    panel = MODULE._root_global_comparison_panel(
        [
            {"name": "r3_episode_mean"},
            {"name": "r3_global_mean"},
            {"name": "r5_global_mad35"},
        ],
        [0, 7],
    )

    assert "r3_episode_mean" not in panel
    assert "r3_global_mean" in panel
    assert "r5_global_mad35" in panel
    assert "<option value='0'>Task 0</option>" in panel
    assert "<option value='7'>Task 7</option>" in panel
    assert "model_${name}_${view}.png" in panel


def test_terminal_mask_uses_first_masked_anchor_to_confirm_edge_peak() -> None:
    timestamps = np.arange(0, 95, 5, dtype=np.int64)
    values = np.zeros(len(timestamps), dtype=np.float64)
    values[timestamps == 65] = 2.0
    values[timestamps == 70] = 1.0

    cuts, _, _ = MODULE._boundaries(
        values,
        timestamps,
        {
            "method": "peak",
            "threshold": "fixed",
            "threshold_value": 1.5,
            "smooth_window": 1,
            "polyorder": 0,
            "nms_frames": 15,
            "terminal_mask_frames": 24,
        },
    )

    np.testing.assert_array_equal(cuts, np.array([65]))


def test_terminal_mask_does_not_turn_rising_edge_into_peak() -> None:
    timestamps = np.arange(0, 95, 5, dtype=np.int64)
    values = np.zeros(len(timestamps), dtype=np.float64)
    values[timestamps == 65] = 2.0
    values[timestamps == 70] = 3.0

    cuts, _, _ = MODULE._boundaries(
        values,
        timestamps,
        {
            "method": "peak",
            "threshold": "fixed",
            "threshold_value": 1.5,
            "smooth_window": 1,
            "polyorder": 0,
            "nms_frames": 15,
            "terminal_mask_frames": 24,
        },
    )

    assert not len(cuts)


def test_majority_supported_positions_requires_strict_episode_majority() -> None:
    consensus = MODULE._majority_supported_positions(
        [
            np.array([10.0, 40.0]),
            np.array([11.0, 39.0]),
            np.array([9.0]),
            np.empty(0),
        ],
        tolerance_frames=2,
        maximum_frame=50,
    )

    # Three of four episodes support the first temporal mode, whereas only
    # two support the second. A strict majority therefore keeps only the first.
    np.testing.assert_allclose(consensus, np.array([10.0]))


def test_consensus_peak_metric_penalizes_a_missed_repeated_peak() -> None:
    timestamps = np.arange(60, dtype=np.int64)
    score = np.exp(-0.5 * ((timestamps - 15) / 1.5) ** 2)
    score += 0.8 * np.exp(-0.5 * ((timestamps - 40) / 1.5) ** 2)
    common = {
        "replan_ts": timestamps,
        "score": score,
        "terminal_mask_start_ts": np.asarray(np.nan),
    }
    task_rows = {
        episode_id: [
            ("complete", {**common, "boundaries": np.array([15, 40])}),
            ("missing", {**common, "boundaries": np.array([40])}),
        ]
        for episode_id in range(4)
    }
    identity = np.arange(60, dtype=np.float64)
    metrics = MODULE._task_consensus_peak_metrics(
        task_rows,
        {
            episode_id: (identity, identity)
            for episode_id in task_rows
        },
        tolerance_frames=2,
    )

    np.testing.assert_allclose(metrics["complete"]["f1"], 1.0)
    np.testing.assert_allclose(metrics["missing"]["precision"], 1.0)
    np.testing.assert_allclose(metrics["missing"]["recall"], 0.5)
    np.testing.assert_allclose(metrics["missing"]["f1"], 2.0 / 3.0)
