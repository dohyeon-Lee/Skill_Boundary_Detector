"""Legacy pi05 eval output -> Stage-1 layout (videos re-annotated with a stub, no torch)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

EVAL_DIR = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_pi05/pi05_eval"
SPEC = importlib.util.spec_from_file_location("pi05_convert_legacy", EVAL_DIR / "src/convert_legacy_output.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _legacy_panel(out: Path, label: str, results: dict[int, list[bool]]) -> Path:
    panel = out / "panels" / label
    for task_id, successes in results.items():
        videos = panel / "videos" / f"libero_90_{task_id}"
        videos.mkdir(parents=True)
        (videos / "language.txt").write_text(f"task {task_id}")
        (videos / "success.json").write_text(json.dumps(successes))
        for episode in range(2):
            frames = [np.full((16, 16, 3), 40 * episode, dtype=np.uint8) for _ in range(3)]
            imageio.mimsave(str(videos / f"eval_episode_{episode}.mp4"), frames, fps=10)
        (panel / f"eval_info_w{task_id:03d}_t{task_id}-{task_id}.json").write_text(json.dumps({
            "per_task": [{
                "task_group": "libero_90", "task_id": task_id,
                "metrics": {"successes": successes, "video_paths": [str(videos / "eval_episode_0.mp4")]},
            }],
            "overall": {"pc_success": 100.0 * sum(successes) / len(successes)},
        }))
    (panel / "eval_info.json").write_text(json.dumps({"overall": {"pc_success": -1}, "per_task": []}))
    (panel / ".merged_wandb_done").touch()
    return panel


def test_legacy_output_is_rebuilt_in_the_stage1_layout(tmp_path: Path) -> None:
    _legacy_panel(tmp_path, "batch16", {0: [True, False, True, True], 1: [False, False, True, True]})
    calls = []

    def annotate(frames, success, language):
        calls.append((success, language))
        bar = np.zeros((frames.shape[0], 16, frames.shape[2], 3), dtype=frames.dtype)
        return np.concatenate([bar, frames], axis=1)

    assert MODULE.convert(tmp_path, annotate=annotate, expected_tasks=2) == ["batch16"]

    # success flag k and the task language go to episode k's annotation
    assert sorted(calls) == sorted([(True, "task 0"), (False, "task 0"), (False, "task 1"), (False, "task 1")])
    video = tmp_path / "panels/00_batch16/videos/libero_90_0/eval_episode_0.mp4"
    assert np.asarray(imageio.mimread(str(video))[0]).shape[0] == 32
    assert not (video.parent / "success.json").exists() and not (video.parent / "language.txt").exists()

    chunk = json.loads((tmp_path / "metrics/eval_info_w000_t0-0.json").read_text())
    assert chunk["batch16"]["per_task"][0]["metrics"]["video_paths"] == [str(video)]
    assert not (tmp_path / "metrics/eval_info_legacy.json").exists()  # chunks win over the old merged file
    merged = json.loads((tmp_path / "metrics/eval_info_merged.json").read_text())["batch16"]
    assert merged["overall"]["pc_success"] == pytest.approx(62.5)
    assert merged["overall"]["n_tasks"] == 2
    assert (tmp_path / "task_success_rates.png").is_file()
    assert not (tmp_path / "side_by_side").exists()  # single panel

    assert not (tmp_path / "panels/batch16").exists()
    assert (tmp_path / ".legacy_format/panels/batch16/eval_info.json").is_file()
    assert MODULE.convert(tmp_path, annotate=annotate) == []  # idempotent


def test_merged_only_legacy_panel_becomes_one_chunk(tmp_path: Path) -> None:
    panel = tmp_path / "panels/run a"
    panel.mkdir(parents=True)
    (panel / "eval_info.json").write_text(json.dumps({"per_task": [
        {"task_group": "libero_10", "task_id": 3, "metrics": {"successes": [True, False]}}
    ]}))
    assert MODULE.convert(tmp_path, annotate=lambda frames, success, language: frames) == ["run a"]
    chunk = json.loads((tmp_path / "metrics/eval_info_legacy.json").read_text())
    assert chunk["run a"]["per_task"][0]["task_id"] == 3
    assert MODULE.panel_dir_name(0, "run a") == "00_run-a"
