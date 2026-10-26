"""Simulator-free XYZ-head report tests; no checkpoint or GPU required."""

import sys
from pathlib import Path

import numpy as np
import pytest


SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/PT/eval/src"
)
sys.path.insert(0, str(SRC))
import xyz_head_eval as offline  # noqa: E402


def _record(task, episode, error):
    error = np.asarray(error, dtype=np.float64)
    return {
        "uid": f"t{task}e{episode}",
        "task_id": task,
        "episode_index": episode,
        "skill_index": 0,
        "frame_start": 0,
        "gt_skill": 3,
        "gt_xyz": [0.0, 0.0, 0.0],
        "pred_xyz": error.tolist(),
        "error_xyz_m": error.tolist(),
        "abs_error_xyz_m": np.abs(error).tolist(),
        "l2_error_m": float(np.linalg.norm(error)),
        "language": "test task",
        "image": None,
        "pixel_l2_error": 5.0,
    }


def test_summary_reports_task_axis_and_distribution_metrics():
    records = [_record(0, 10, [.01, 0, 0]), _record(0, 11, [0, .02, 0]),
               _record(1, 12, [0, 0, .03])]
    metrics = offline.summarize(records)
    assert metrics["overall"]["samples"] == 3
    assert metrics["overall"]["episodes"] == 3
    assert metrics["per_task"]["0"]["samples"] == 2
    np.testing.assert_allclose(
        metrics["overall"]["mae_xyz_m"], [0.01 / 3, 0.02 / 3, 0.03 / 3]
    )
    assert metrics["overall"]["l2_p90_m"] >= metrics["overall"]["l2_median_m"]
    assert metrics["overall"]["pixel_l2_mean"] == 5.0


def test_html_contains_model_task_metrics_and_offline_contract():
    records = [_record(0, 10, [.01, 0, 0])]
    page = offline.render_report(
        {"label": "diffaux4", "path": "/checkpoint"},
        records,
        offline.summarize(records),
    )
    assert "diffaux4" in page
    assert "no simulator" in page
    assert "Task 0" in page
    assert "10.0 mm" in page


def test_load_samples_joins_focus_by_episode_and_skill(tmp_path, monkeypatch):
    base = [
        {"episode_index": 2, "skill_index": 1, "uid": "a"},
        {"episode_index": 3, "skill_index": 0, "uid": "b"},
    ]
    monkeypatch.setattr(
        offline, "load_base_samples", lambda settings: ([dict(row) for row in base], {"x": 1})
    )
    focus = tmp_path / "focus.npz"
    np.savez(
        focus,
        episode_id=np.asarray([3, 2]),
        skill_index=np.asarray([0, 1]),
        focus_uv=np.asarray([[.3, .4], [.1, .2]], dtype=np.float32),
        focus_valid=np.asarray([False, True]),
    )
    samples, info = offline.load_samples({"focus_uv_path": str(focus)})
    assert info == {"x": 1}
    np.testing.assert_allclose(samples[0]["focus_uv"], [.1, .2])
    assert samples[0]["focus_valid"] is True
    assert samples[1]["focus_valid"] is False


def test_make_record_rejects_nonfinite_prediction():
    sample = {"gt_xyz": [0, 0, 0]}
    with pytest.raises(ValueError, match="non-finite"):
        offline.make_record(sample, [0, float("nan"), 0], "task")
