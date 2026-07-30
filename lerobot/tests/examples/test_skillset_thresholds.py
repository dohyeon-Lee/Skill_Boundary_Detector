import sys
from pathlib import Path

import numpy as np

LIBERO_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "libero"
sys.path.insert(0, str(LIBERO_EXAMPLES))

from build_skill_dataset import (  # noqa: E402
    Args,
    _detect_boundaries,
    _find_peaks_above_threshold,
    _merge_short_final_segment,
    _save_boundary_curve,
    _skillset_manifest,
    _write_skillset_manifest,
)


def test_peak_nms_keeps_peak_near_episode_end_but_not_start():
    ts = [0, 5, 10, 15, 20, 25, 30]

    end_peak_ts, _ = _find_peaks_above_threshold(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
        ts,
        threshold=1.0,
        min_distance=20,
        start_margin=20,
    )
    assert end_peak_ts == [25]

    start_peak_ts, _ = _find_peaks_above_threshold(
        np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ts,
        threshold=1.0,
        min_distance=20,
        start_margin=20,
    )
    assert start_peak_ts == []


def test_short_final_segment_merges_into_previous_skill():
    assert _merge_short_final_segment([0, 30, 55, 60], min_skill_len=10) == [0, 30, 60]
    assert _merge_short_final_segment([0, 30, 50, 60], min_skill_len=10) == [0, 30, 50, 60]


def test_boundary_detection_accepts_exact_minimum_final_length():
    args = Args(
        smooth_window=1,
        peak_nms=False,
        boundary_threshold_mode="episode_mean",
        min_skill_len=10,
    )

    too_short = _detect_boundaries(
        [0, 10, 20, 30, 35, 40],
        np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
        40,
        args,
    )
    assert too_short == [0, 40]

    exactly_ten = _detect_boundaries(
        [0, 10, 20, 30, 40],
        np.array([0.0, 0.0, 0.0, 2.0, 0.0]),
        40,
        args,
    )
    assert exactly_ten == [0, 30, 40]


def test_global_threshold_changes_boundaries_and_curve_metadata(tmp_path: Path):
    replan_ts = [0, 10, 20, 30, 40]
    divergence = np.array([0.0, 2.0, 0.0, 1.0, 0.0], dtype=np.float32)
    args = Args(
        probe_mode="std",
        probe_type="pca_action",
        pca_scale_mode="std",
        smooth_window=1,
        peak_nms=False,
        boundary_threshold_mode="episode_mean",
    )

    episode_boundaries = _detect_boundaries(replan_ts, divergence, 50, args)
    assert episode_boundaries == [0, 10, 30, 50]

    args.boundary_threshold_scale = 2.0
    scaled_boundaries = _detect_boundaries(replan_ts, divergence, 50, args)
    assert scaled_boundaries == [0, 10, 50]

    args.boundary_threshold_mode = "global_mean"
    args.boundary_threshold_scale = 1.0
    global_boundaries = _detect_boundaries(
        replan_ts, divergence, 50, args, global_threshold=1.5
    )
    assert global_boundaries == [0, 10, 50]

    _save_boundary_curve(
        tmp_path,
        ep_id=3,
        task_id=2,
        replan_ts=replan_ts,
        div_cos=divergence,
        boundaries=global_boundaries,
        n_frames=50,
        args=args,
        global_threshold=1.5,
    )
    with np.load(tmp_path / "ep0000003.npz", allow_pickle=False) as curve:
        assert str(curve["probe_mode"]) == "std"
        assert str(curve["threshold_mode"]) == "global_mean"
        assert float(curve["threshold_scale"]) == 1.0
        assert float(curve["mean_val"]) == 1.5
        assert curve["boundaries"].tolist() == [0, 10, 50]


def test_manifest_is_idempotent_and_rejects_mixed_configuration(tmp_path: Path):
    path = tmp_path / "skillset_manifest.json"
    payload = {"mode": "std", "detector": {"boundary_threshold_mode": "global_mean"}}

    _write_skillset_manifest(path, payload)
    _write_skillset_manifest(path, payload)

    with np.testing.assert_raises_regex(ValueError, "manifest mismatch"):
        _write_skillset_manifest(
            path,
            {"mode": "full", "detector": {"boundary_threshold_mode": "global_mean"}},
        )


def test_manifest_preserves_abc_primary_camera(tmp_path: Path):
    manifest = _skillset_manifest(
        args=Args(probe_mode="std", probe_type="pca_action", pca_scale_mode="std"),
        dataset_dir=tmp_path / "abc_toy",
        policy_path=str(tmp_path / "outputs" / "DP" / "abc_state" / "pretrained_model"),
        image_key="observation.images.top",
        mode="std",
        action_dim=14,
        action_pca=None,
    )

    assert manifest["dataset_name"] == "abc_toy"
    assert manifest["image_key"] == "observation.images.top"
    assert manifest["action"]["dim"] == 14
