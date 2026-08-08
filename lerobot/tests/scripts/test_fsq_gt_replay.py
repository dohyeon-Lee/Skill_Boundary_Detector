from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "examples/libero/configs/train_skills/skill_eval/src"
sys.path.insert(0, str(SRC))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONFIG = _load("fsq_gt_replay_config_test", "fsq_gt_replay_config.py")
REPORT = _load("fsq_gt_replay_report_test", "fsq_gt_replay_report.py")
RUNNER = _load("run_fsq_gt_replay_test", "run_fsq_gt_replay.py")


def test_output_name_accepts_nested_run_and_epoch() -> None:
    assert CONFIG._safe_relative_output("", default="run/epoch0500") == Path(
        "run/epoch0500"
    )


def test_checkpoint_list_accepts_multiple_noncontiguous_epochs() -> None:
    assert CONFIG._fsq_checkpoints(
        {"fsq_eval_checkpoint": [125, 175, 300]}
    ) == ["125", "175", "300"]


@pytest.mark.parametrize("value", ["../escape", "/absolute", "bad name"])
def test_output_name_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(ValueError, match="output_name"):
        CONFIG._safe_relative_output(value, default="unused")


@pytest.mark.parametrize("checkpoint", ["250", "last"])
def test_resolve_artifact_missing_ok_skips_untrained_checkpoint(
    tmp_path: Path, checkpoint: str
) -> None:
    (tmp_path / "outputs" / "FSQ" / "run").mkdir(parents=True)
    cfg = {"fsq_eval_run_name": "run"}
    assert (
        CONFIG._resolve_fsq_artifact(
            cfg,
            dataset_root=tmp_path / "dataset",
            outputs_root=tmp_path / "outputs",
            checkpoint=checkpoint,
            missing_ok=True,
        )
        is None
    )
    with pytest.raises(FileNotFoundError):
        CONFIG._resolve_fsq_artifact(
            cfg,
            dataset_root=tmp_path / "dataset",
            outputs_root=tmp_path / "outputs",
            checkpoint=checkpoint,
        )


def test_resolve_artifact_missing_ok_still_rejects_unknown_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run folder"):
        CONFIG._resolve_fsq_artifact(
            {"fsq_eval_run_name": "no_such_run"},
            dataset_root=tmp_path / "dataset",
            outputs_root=tmp_path / "outputs",
            checkpoint="100",
            missing_ok=True,
        )


def test_end_state_uses_next_frame_start_state() -> None:
    class _Occurrence:
        frame_start = 0
        frame_end = 2

    class _Aligned:
        original_action_indices = np.asarray([0, 1, 2])
        original_states = np.asarray([[0.0], [1.0], [2.0]])

        def state_at(self, filtered_frame: int) -> np.ndarray:
            return self.original_states[
                self.original_action_indices[filtered_frame]
            ].astype(np.float64)

        def original_frame_at(self, filtered_frame: int) -> int:
            return int(self.original_action_indices[filtered_frame])

    np.testing.assert_array_equal(
        RUNNER._end_state(_Aligned(), _Occurrence()), [2.0]
    )


def test_end_state_clamps_final_episode_skill() -> None:
    class _Occurrence:
        frame_start = 1
        frame_end = 3

    class _Aligned:
        original_action_indices = np.asarray([0, 1, 2])
        original_states = np.asarray([[0.0], [1.0], [2.0]])

        def original_frame_at(self, filtered_frame: int) -> int:
            return int(self.original_action_indices[filtered_frame])

    np.testing.assert_array_equal(
        RUNNER._end_state(_Aligned(), _Occurrence()), [2.0]
    )


def test_report_groups_occurrences_by_fsq_token() -> None:
    manifest = {
        "levels": [3, 3, 3],
        "run_name": "fsq333",
        "epoch_tag": "epoch0500",
        "signature": {
            "target_task": "libero_90",
            "selected_episodes": {"0": [2]},
        },
        "records": {
            "b": {
                "token": 4,
                "task_id": 0,
                "episode_id": 2,
                "frame_start": 20,
            },
            "a": {
                "token": 4,
                "task_id": 0,
                "episode_id": 2,
                "frame_start": 0,
            },
        },
    }

    payload = REPORT.report_payload(manifest)

    assert payload["occurrence_count"] == 2
    assert payload["skills"][0]["token"] == 4
    assert payload["skills"][0]["coord"] == [1, 1, 0]
    assert [row["frame_start"] for row in payload["skills"][0]["occurrences"]] == [
        0,
        20,
    ]


def test_write_image_preserves_rendered_frame(tmp_path: Path) -> None:
    frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    path = tmp_path / "frame.png"

    RUNNER._write_image(path, frame)

    np.testing.assert_array_equal(np.asarray(Image.open(path)), frame)


def test_report_shows_start_and_final_image_pair(tmp_path: Path) -> None:
    payload = {
        "levels": [3, 3, 3],
        "run_name": "fsq333",
        "epoch_tag": "epoch0500",
        "target_task": "libero_90",
        "task_ids": [0],
        "episode_count": 1,
        "occurrence_count": 1,
        "skills": [
            {
                "token": 0,
                "coord": [0, 0, 0],
                "occurrences": [
                    {
                        "task_id": 0,
                        "task_description": "test",
                        "episode_id": 0,
                        "skill_index": 0,
                        "frame_start": 0,
                        "frame_end": 2,
                        "length": 2,
                        "start_image_path": "images/test_start.png",
                        "final_image_path": "images/test_final.png",
                    }
                ],
            }
        ],
    }

    report = REPORT.write_html_report(tmp_path, payload)
    html = report.read_text(encoding="utf-8")

    assert "images/test_start.png" in html
    assert "images/test_final.png" in html
    assert "<video" not in html
    assert 'class="pair"' in html
    assert "GT start" in html
    assert "GT end" in html
    assert 'id="checkpoint"' in html
    assert 'id="tasks"' in html
    assert 'class="task-group"' in html
    assert 'class="occ-row"' in html
    assert 'id="positionMode"' in html


def test_collection_keeps_checkpoint_codebooks_and_prefixes_media() -> None:
    def manifest(epoch: int, token: int) -> dict:
        epoch_tag = f"epoch{epoch:04d}"
        return {
            "levels": [3, 3, 3],
            "run_name": "fsq333",
            "epoch_tag": epoch_tag,
            "signature": {
                "target_task": "libero_90",
                "selected_episodes": {"1": [11], "3": [33]},
            },
            "records": {
                f"{epoch_tag}-sample": {
                    "token": token,
                    "task_id": 3,
                    "episode_id": 33,
                    "frame_start": 0,
                    "start_image_path": "images/sample_start.png",
                    "final_image_path": "images/sample_final.png",
                }
            },
        }

    payload = REPORT.collection_payload([manifest(125, 2), manifest(300, 7)])

    assert [item["epoch_tag"] for item in payload["checkpoints"]] == [
        "epoch0125",
        "epoch0300",
    ]
    assert [item["skills"][0]["token"] for item in payload["checkpoints"]] == [
        2,
        7,
    ]
    occurrence = payload["checkpoints"][1]["skills"][0]["occurrences"][0]
    assert occurrence["start_image_path"] == (
        "checkpoints/epoch0300/images/sample_start.png"
    )
    assert occurrence["final_image_path"] == (
        "checkpoints/epoch0300/images/sample_final.png"
    )
