from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EVAL_SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skills/skill_eval/src"
sys.path.insert(0, str(EVAL_SRC))

import dp_skillset_eval as dp_eval  # noqa: E402
from skillset_boundary_viz import (  # noqa: E402
    build_gmm_visualizations,
    plot_boundary_curve,
    render_skillset_card,
    save_gallery,
)


def test_episode_video_clip_is_h264_and_reused(monkeypatch, tmp_path: Path):
    source = tmp_path / "long-av1-chunk.mp4"
    source.write_bytes(b"source-video")
    cache_dir = tmp_path / "cache"
    commands = []

    def fake_run(command, check):
        assert check is True
        commands.append(command)
        Path(command[-1]).write_bytes(b"short-h264-clip")

    monkeypatch.setattr(dp_eval.subprocess, "run", fake_run)
    first = dp_eval._cached_episode_video_clip(
        source,
        episode_id=42,
        image_key="observation.images.image",
        episode_start=123.5,
        episode_end=130.0,
        cache_dir=cache_dir,
    )
    second = dp_eval._cached_episode_video_clip(
        source,
        episode_id=42,
        image_key="observation.images.image",
        episode_start=123.5,
        episode_end=130.0,
        cache_dir=cache_dir,
    )

    assert first == second
    assert first.read_bytes() == b"short-h264-clip"
    assert len(commands) == 1
    assert commands[0][commands[0].index("-c:v") + 1] == "libx264"
    assert commands[0][commands[0].index("-movflags") + 1] == "+faststart"
    assert json.loads(first.with_suffix(".json").read_text())["episode_start_sec"] == 123.5


def test_episode_player_contains_live_skill_overlay_and_timeline(tmp_path: Path):
    skills = [(0, 10, None), (10, 25, 7)]
    curve = {
        "replan_ts": np.asarray([0, 5, 10, 15, 20]),
        "div_cos": np.asarray([0.1, 0.2, 0.5, 0.2, 0.1]),
        "sg_vals": np.asarray([0.01, 0.04, 0.25, 0.04, 0.01]),
        "score_power": np.asarray(2.0),
        "score_power1_mean": np.asarray(0.2),
        "mean_val": np.asarray(0.05),
        "peak_ts": np.asarray([10]),
        "peak_vals": np.asarray([0.25]),
        "n_frames": np.asarray(25),
    }
    media = render_skillset_card(
        skills,
        raw=None,
        curve=curve,
        episode_video={
            "video_src": "../../raw/episode.mp4",
            "episode_start_sec": 0.0,
            "episode_end_sec": 1.0,
            "fps": 25.0,
            "aspect_ratio": 4.0 / 3.0,
            "object_position_y": 0.0,
        },
        show_frames=False,
    )
    save_gallery(tmp_path, "episode player", [(None, "episode 1", media)])
    page = (tmp_path / "index.html").read_text()

    assert "class='episode-video'" in page
    assert "class='skill-badge'" in page
    assert "class='skill-playhead'" in page
    assert "class='timeline-playhead'" in page
    assert "data-plot-left=" in page
    assert "timeline-synced" in page
    assert "requestAnimationFrame" in page
    assert "SKILL CHANGE" in page
    assert "영상 크게 보기" in page
    assert "requestFullscreen" in page
    assert "data-frame-start='10'" in page
    assert "tok7" in page
    assert "updateEpisodePlayer" in page
    assert "<div class='skill-grid'>" not in page


def test_episode_player_labels_within_normalized_gmm_metric():
    import matplotlib.pyplot as plt

    curve = {
        "replan_ts": np.asarray([0, 5, 10]),
        "div_cos": np.asarray([1.0, 3.0, 1.0]),
        "sg_vals": np.asarray([1.0, 9.0, 1.0]),
        "score_power": np.asarray(2.0),
        "score_metric": np.asarray("within_normalized_l2"),
        "mean_val": np.asarray(2.0),
        "peak_ts": np.asarray([5]),
        "peak_vals": np.asarray([9.0]),
        "n_frames": np.asarray(15),
    }
    figure, axis = plt.subplots()
    plot_boundary_curve(axis, curve, [(0, 5, None), (5, 15, None)])
    try:
        assert axis.get_ylabel() == "GMM separation ratio"
        assert "center distance / within-mode radius (raw^2)" in axis.get_legend_handles_labels()[1]
    finally:
        plt.close(figure)


def test_episode_player_labels_angular_chord_metric():
    import matplotlib.pyplot as plt

    curve = {
        "replan_ts": np.asarray([0, 5, 10]),
        "div_cos": np.asarray([0.2, 0.8, 0.2]),
        "sg_vals": np.asarray([0.04, 0.64, 0.04]),
        "score_power": np.asarray(2.0),
        "score_metric": np.asarray("covariance_gated_angular_chord"),
        "mean_val": np.asarray(0.3),
        "peak_ts": np.asarray([5]),
        "peak_vals": np.asarray([0.64]),
        "n_frames": np.asarray(15),
    }
    figure, axis = plt.subplots()
    plot_boundary_curve(axis, curve, [(0, 5, None), (5, 15, None)])
    try:
        assert axis.get_ylabel() == "angular chord score"
        assert "covariance-gated angular chord (raw^2)" in axis.get_legend_handles_labels()[1]
    finally:
        plt.close(figure)


def test_episode_player_labels_strong_reliability_gate():
    import matplotlib.pyplot as plt

    curve = {
        "replan_ts": np.asarray([0, 5, 10]),
        "div_cos": np.asarray([0.2, 0.8, 0.2]),
        "sg_vals": np.asarray([0.04, 0.64, 0.04]),
        "score_power": np.asarray(2.0),
        "score_metric": np.asarray("strong_covariance_gated_cosine"),
        "mean_val": np.asarray(0.3),
        "peak_ts": np.asarray([5]),
        "peak_vals": np.asarray([0.64]),
        "n_frames": np.asarray(15),
    }
    figure, axis = plt.subplots()
    plot_boundary_curve(axis, curve, [(0, 5, None), (5, 15, None)])
    try:
        assert axis.get_ylabel() == "SBD detector score"
        assert "strong reliability-gated cosine (raw^2)" in axis.get_legend_handles_labels()[1]
    finally:
        plt.close(figure)


def test_episode_player_shades_terminal_motion_mask_red():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    curve = {
        "replan_ts": np.asarray([0, 5, 10]),
        "div_cos": np.asarray([0.2, 0.8, 0.2]),
        "sg_vals": np.asarray([0.04, 0.64, 0.04]),
        "score_power": np.asarray(2.0),
        "mean_val": np.asarray(0.3),
        "peak_ts": np.asarray([5]),
        "peak_vals": np.asarray([0.64]),
        "terminal_mask_start_frame": np.asarray(11),
        "n_frames": np.asarray(15),
    }
    figure, axis = plt.subplots()
    plot_boundary_curve(axis, curve, [(0, 5, None), (5, 15, None)])
    try:
        labels = axis.get_legend_handles_labels()[1]
        assert "terminal motion excluded" in labels
        expected = np.asarray(to_rgba("#fca5a5", alpha=0.26))
        assert any(
            np.allclose(np.asarray(patch.get_facecolor()), expected)
            for patch in axis.patches
        )
    finally:
        plt.close(figure)


def test_episode_player_syncs_three_dimensional_gmm_cloud(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "iid96"
    raw_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    output = rng.normal(size=(3, 20, 6)).astype(np.float32)
    before = (output * 1.8 + 0.2).astype(np.float32)
    np.savez_compressed(
        raw_dir / "ep0000004.npz",
        replan_ts=np.asarray([0, 5, 10]),
        descriptors=output,
        input_descriptors=before,
        n_frames=np.asarray(15),
    )
    visualizations = build_gmm_visualizations(raw_dir, [4], components=5)
    assert visualizations[4]["descriptorDim"] == 6
    assert visualizations[4]["centering"] == "per_frame"
    assert len(visualizations[4]["labels"]) == 3
    assert len(visualizations[4]["centers"][0]) == 5
    output_3d = np.asarray(visualizations[4]["output"])
    input_3d = np.asarray(visualizations[4]["input"])
    assert np.allclose(output_3d.mean(axis=1), 0.0, atol=1e-4)
    assert np.allclose(input_3d.mean(axis=1), 0.0, atol=1e-4)

    media = render_skillset_card(
        [(0, 15, None)],
        raw=None,
        curve=None,
        episode_video={
            "video_src": "episode.mp4",
            "episode_start_sec": 0.0,
            "episode_end_sec": 0.5,
            "fps": 30.0,
        },
        gmm_visualization=visualizations[4],
        show_frames=False,
    )
    save_gallery(tmp_path, "gmm player", [(None, "episode 4", media)])
    page = (tmp_path / "index.html").read_text()
    assert "class='gmm-canvas'" in page
    assert "denoising 전·후" in page
    assert "updateGmmForFrame(card,frame)" in page
    assert "frame-centered PC1–PC3" in page
    assert "centered ${data.descriptorDim}D→3D" in page


def test_abc_stacked_video_metadata_crops_requested_camera(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "features": {
                    "observation.images.top": {"shape": [168, 224, 3]},
                    "observation.images.left_wrist": {"shape": [168, 224, 3]},
                },
            }
        )
    )
    video_path = tmp_path / "combined.mp4"
    video_path.touch()
    monkeypatch.setattr(dp_eval, "_video_path", lambda *_args, **_kwargs: video_path)
    layout = json.dumps(
        [
            {"source": "top", "camera": "top", "row_start": 0, "row_end": 168},
            {"source": "left", "camera": "left_wrist", "row_start": 168, "row_end": 336},
            {"source": "right", "camera": "right_wrist", "row_start": 336, "row_end": 504},
        ]
    )
    episodes = pd.DataFrame(
        [{"episode_index": 4, "length": 300, "video_path": "unused.mp4", "camera_layout_json": layout}]
    )

    top = dp_eval._episode_video_metadata(
        dataset_dir, episodes, "observation.images.top", 4, tmp_path / "report", 30.0
    )
    left = dp_eval._episode_video_metadata(
        dataset_dir, episodes, "observation.images.left_wrist", 4, tmp_path / "report", 30.0
    )

    assert top["aspect_ratio"] == 224 / 168
    assert top["object_position_y"] == 0.0
    assert left["object_position_y"] == 50.0
    assert top["episode_start_sec"] == 0.0
    assert top["episode_end_sec"] == 10.0


def test_global_curve_mean_uses_only_terminal_valid_anchors(tmp_path: Path):
    curves = tmp_path / "curves"
    curves.mkdir()
    np.savez(
        curves / "ep0000000.npz",
        denoising_gain_sg=np.asarray([1.0, 3.0, 100.0]),
        threshold_valid_count=np.asarray(2),
    )
    np.savez(
        curves / "ep0000001.npz",
        denoising_gain_sg=np.asarray([5.0, 7.0, 200.0]),
        threshold_valid_count=np.asarray(2),
    )

    mean, count = dp_eval.global_curve_mean(curves, "denoising_gain_sg")
    power2_mean, power2_count = dp_eval.global_curve_power_mean(
        curves,
        "denoising_gain_sg",
        source_power=1.0,
        target_power=2.0,
    )

    assert mean == 4.0
    assert count == 4
    assert power2_mean == 21.0
    assert power2_count == 4


def test_sbd_score_recut_accepts_all_peaks_and_keeps_spacing_rules():
    curve = {
        "replan_ts": np.arange(0, 80, 8),
        "sg_vals": np.asarray([0.1, 0.2, 0.8, 0.1, 0.9, 0.1, 0.7, 0.1, 0.2, 0.1]),
        "div_cos": np.asarray([0.1, 0.2, 0.8, 0.1, 0.9, 0.1, 0.7, 0.1, 0.2, 0.1]),
        "score_power": np.asarray(1.0),
        "threshold_valid_count": np.asarray(9),
        "n_frames": np.asarray(80),
    }

    skills, peak_ts, peak_vals = dp_eval.skills_from_sbd_score(
        curve,
        threshold=-np.inf,
        nms_dist=15,
        min_skill_len=15,
        smooth_window=1,
        savgol_polyorder=4,
    )

    assert skills == [
        (0, 16, None),
        (16, 32, None),
        (32, 48, None),
        (48, 64, None),
        (64, 80, None),
    ]
    assert peak_ts.tolist() == [16, 32, 48, 64]
    assert np.allclose(peak_vals, [0.8, 0.9, 0.7, 0.2])


def test_global_curve_power_mean_maps_power2_back_to_power1(tmp_path: Path):
    curves = tmp_path / "curves"
    curves.mkdir()
    np.savez(
        curves / "ep0000000.npz",
        sg_vals=np.asarray([1.0, 4.0, 10_000.0]),
        threshold_valid_count=np.asarray(2),
    )
    np.savez(
        curves / "ep0000001.npz",
        sg_vals=np.asarray([9.0, 16.0, 20_000.0]),
        threshold_valid_count=np.asarray(2),
    )

    mean, count = dp_eval.global_curve_power_mean(
        curves,
        "sg_vals",
        source_power=2.0,
        target_power=1.0,
    )
    half_mean, half_count = dp_eval.global_curve_power_mean(
        curves,
        "sg_vals",
        source_power=2.0,
        target_power=0.5,
    )

    assert mean == 2.5
    assert count == 4
    assert np.isclose(half_mean, np.mean([1.0, np.sqrt(2.0), np.sqrt(3.0), 2.0]))
    assert half_count == 4

    low, high = dp_eval.global_curve_power_range(
        curves,
        ("sg_vals",),
        source_power=2.0,
        target_power=1.0,
    )
    assert low == 1.0
    assert high == np.sqrt(20_000.0)


def test_initial_motion_start_requires_sustained_non_gripper_motion():
    states = np.zeros((20, 3), dtype=np.float32)
    states[2, 0] = 0.1  # one-frame arm impulse must not trigger the detector
    states[3, 0] = 0.0
    states[5:, 2] = np.arange(15, dtype=np.float32) * 0.01  # excluded gripper
    states[10:, 1] = np.arange(10, dtype=np.float32) * 0.01

    start = dp_eval.detect_initial_motion_start(
        states,
        excluded_indices=[2],
        velocity_threshold=0.005,
        sustain_frames=5,
    )

    assert start == 11


def test_initial_masked_power1_mean_excludes_prefix_and_terminal_tail(tmp_path: Path):
    curves = tmp_path / "curves"
    curves.mkdir()
    np.savez(
        curves / "ep0000007.npz",
        replan_ts=np.asarray([0, 8, 16, 24, 32, 40]),
        div_cos=np.asarray([100.0, 100.0, 1.0, 3.0, 5.0, 100.0]),
        threshold_valid_count=np.asarray(5),
    )

    mean, count = dp_eval.global_initial_masked_score_mean(
        curves,
        {7: 15},
        score_power=1.0,
        smooth_window=1,
        savgol_polyorder=4,
    )

    assert mean == 3.0
    assert count == 3


def test_reset_jump_mask_extends_through_three_stable_anchors(tmp_path: Path):
    curves = tmp_path / "curves"
    curves.mkdir()
    np.savez(
        curves / "ep0000007.npz",
        replan_ts=np.arange(0, 88, 8),
        # After the base mask at frame 15, values stay high through frame 32,
        # then frames 40/48/56 establish the first three-anchor stable run.
        div_cos=np.asarray([9.0, 9.0, 2.0, 2.0, 2.0, 0.4, 0.3, 0.2, 0.8, 0.2, 0.1]),
        threshold_valid_count=np.asarray(10),
    )

    masks, extensions = dp_eval.extend_reset_jump_masks(
        curves,
        {7: 15},
        {7: 1.2},
        provisional_threshold=0.5,
        reset_jump_velocity_threshold=0.5,
        stable_anchors=3,
        max_extra_frames=100,
        smooth_window=1,
        savgol_polyorder=4,
    )

    assert masks == {7: 56}
    assert extensions == {7: 56}


def test_sbd_recut_rejects_peaks_inside_dynamic_initial_mask():
    curve = {
        "replan_ts": np.arange(0, 80, 8),
        "sg_vals": np.asarray([0.1, 0.9, 0.1, 0.8, 0.1, 0.7, 0.1, 0.2, 0.1, 0.0]),
        "div_cos": np.asarray([0.1, 0.9, 0.1, 0.8, 0.1, 0.7, 0.1, 0.2, 0.1, 0.0]),
        "score_power": np.asarray(1.0),
        "threshold_valid_count": np.asarray(9),
        "n_frames": np.asarray(80),
    }

    skills, peak_ts, _peak_vals = dp_eval.skills_from_sbd_score(
        curve,
        threshold=0.15,
        nms_dist=15,
        min_skill_len=15,
        smooth_window=1,
        savgol_polyorder=4,
        initial_mask_until_frame=20,
    )

    # Frame 24 is the first eligible anchor and therefore has no unmasked
    # left-hand neighbour with which to establish local-maximum topology.
    assert peak_ts.tolist() == [40, 56]
    assert skills == [(0, 40, None), (40, 56, None), (56, 80, None)]


def test_padded_axis_range_includes_reference_line():
    low, high = dp_eval.padded_axis_range(0.2, 0.7, include=(1.0,))

    assert np.isclose(low, 0.16)
    assert np.isclose(high, 1.04)


def test_denoising_gain_boundaries_follow_mean_nms_and_tail_rules():
    curve = {
        "replan_ts": np.arange(0, 80, 8),
        "denoising_gain_sg": np.asarray(
            [0.1, 0.2, 0.8, 0.1, 0.9, 0.1, 0.7, 0.1, 0.2, 0.1]
        ),
        "threshold_valid_count": np.asarray(9),
        "n_frames": np.asarray(62),
    }

    skills, peak_ts, peak_vals = dp_eval.skills_from_denoising_gain(
        curve,
        threshold=0.5,
        nms_dist=15,
        min_skill_len=15,
    )

    # Peaks at 16, 32, and 48 pass the mean/NMS rules, but the final 14-frame
    # skill is merged back into the preceding skill.
    assert skills == [(0, 16, None), (16, 32, None), (32, 62, None)]
    assert peak_ts.tolist() == [16, 32]
    assert np.allclose(peak_vals, [0.8, 0.9])
