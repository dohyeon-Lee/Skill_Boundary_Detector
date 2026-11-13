from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml


ABC_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/abc/configs/generate_training_dataset/download_dataset/ABC_dataset/src"
)
ABC_TASK_GROUPS = (
    ABC_SRC.parent / "ABC_task_groups.yaml"
)
sys.path.insert(0, str(ABC_SRC))

from download_abc_subset import (  # noqa: E402
    COMPLETION_MARKER,
    _completion_status,
    _request_sha256,
    _write_completion_marker,
)
import abc_mcap_to_cache as cache_converter  # noqa: E402


def _request() -> dict:
    return {
        "repo_id": "XDOF/ABC-130k",
        "split": "train",
        "out_dir": "/tmp/abc_a",
        "group_subdirs": False,
        "include_meta": False,
        "max_workers": 4,
        "hf_transfer": False,
        "downloads": [{"task": "pick_up_the_apple", "episodes": 2}],
        "groups": {"pick_and_place": ["pick_up_the_apple"]},
    }


def _episode(subset: Path, name: str = "episode_a") -> Path:
    path = subset / "data" / "train" / "pick_up_the_apple" / name / "episode.mcap"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"mcap-data")
    return path


def test_request_hash_ignores_runtime_knobs_but_tracks_selection() -> None:
    request = _request()
    expected = _request_sha256(request)

    runtime_changed = {
        **request,
        "out_dir": "/another/machine/path",
        "max_workers": 32,
        "hf_transfer": True,
    }
    assert _request_sha256(runtime_changed) == expected

    selection_changed = {
        **request,
        "downloads": [{"task": "pick_up_the_apple", "episodes": 3}],
    }
    assert _request_sha256(selection_changed) != expected


def test_completion_is_recorded_per_subset(tmp_path: Path) -> None:
    request_hash = _request_sha256(_request())
    first = tmp_path / "first"
    second = tmp_path / "second"
    _episode(first)
    _episode(second)

    _write_completion_marker(first, "first", request_hash)

    assert (first / COMPLETION_MARKER).is_file()
    assert _completion_status(first, request_hash)[0] is True
    complete, reason = _completion_status(second, request_hash)
    assert complete is False
    assert reason == "completion marker missing"


def test_completion_invalidates_changed_partial_or_damaged_subset(tmp_path: Path) -> None:
    subset = tmp_path / "subset"
    episode = _episode(subset)
    request_hash = _request_sha256(_request())
    _write_completion_marker(subset, "subset", request_hash)

    changed = {**_request(), "split": "val"}
    assert _completion_status(subset, _request_sha256(changed))[0] is False

    pending = subset / "download.incomplete"
    pending.write_bytes(b"partial")
    complete, reason = _completion_status(subset, request_hash)
    assert complete is False
    assert "incomplete/lock" in reason
    pending.unlink()

    episode.write_bytes(b"changed-size")
    complete, reason = _completion_status(subset, request_hash)
    assert complete is False
    assert "size mismatch" in reason


def test_taxonomy_covers_current_train_task_snapshot_without_duplicates() -> None:
    config = yaml.safe_load(ABC_TASK_GROUPS.read_text())
    grouped_tasks = [task for tasks in config["groups"].values() for task in tasks]

    assert len(grouped_tasks) == 201
    assert len(set(grouped_tasks)) == 201
    assert {
        "assemble_paper_box",
        "insert_wireless_bluetooth_earbuds_into_charging_case",
        "pack_books_pencil_case_and_school_supplies_into_a_backpack",
        "unlock_a_padlock_with_a_key",
    } <= set(grouped_tasks)


def test_local_mcap_converter_resamples_and_writes_compatible_cache(
    tmp_path: Path, monkeypatch,
) -> None:
    tick = cache_converter.TICK_NS
    source_timestamps = np.array([0, 20_000_000, 50_000_000, 80_000_000], dtype=np.int64)
    camera_timestamps = np.array([0, 40_000_000, 70_000_000], dtype=np.int64)
    source = cache_converter.Episode(
        states=np.arange(4 * 14, dtype=np.float64).reshape(4, 14),
        actions=(1000 + np.arange(4 * 14, dtype=np.float64)).reshape(4, 14),
        timestamps=source_timestamps,
        cameras={
            name: cache_converter.CameraStream(
                frames=[b"frame0", b"frame1", b"frame2"],
                timestamps=camera_timestamps,
                width=0,
                height=0,
                codec="h264",
            )
            for name in ("top", "left_wrist")
        },
        meta=cache_converter.EpisodeMeta(
            task="test task",
            fps=0.0,
            cameras=["top", "left_wrist"],
            camera_resolutions={"top": (0, 0), "left_wrist": (0, 0)},
            camera_codecs={"top": "h264", "left_wrist": "h264"},
        ),
    )
    decoded = np.stack(
        [np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
    )
    encoded: dict[str, np.ndarray] = {}

    monkeypatch.setattr(cache_converter, "read_abc_mcap", lambda _: source)
    monkeypatch.setattr(
        cache_converter, "_decode_annexb", lambda chunks, codec, width, height: decoded
    )

    def fake_encode(frames: np.ndarray, output_path: str) -> None:
        encoded["frames"] = frames.copy()
        Path(output_path).write_bytes(b"fake-mp4")

    monkeypatch.setattr(cache_converter, "_encode_strict_h264", fake_encode)

    output = tmp_path / "cache_episode"
    cache_converter.mcap_to_abc_cache("unused.mcap", str(output), size=2)

    # ticks are 33.3 ms and 66.6 ms: source state floors to rows 1 and 2.
    values = np.fromfile(output / "states_actions.bin", dtype="<f8").reshape(2, 28)
    np.testing.assert_array_equal(values[:, :14], source.states[[1, 2]])
    np.testing.assert_array_equal(values[:, 14:], source.actions[[1, 2]])

    # Cameras floor to decoded frames [0, 1] and are vertically stacked.
    assert encoded["frames"].shape == (2, 4, 2, 3)
    np.testing.assert_array_equal(encoded["frames"][:, :2], decoded[[0, 1]])
    np.testing.assert_array_equal(encoded["frames"][:, 2:], decoded[[0, 1]])

    metadata = yaml.safe_load((output / "episode_metadata.json").read_text())
    assert metadata["num_steps"] == 2
    assert metadata["state_dim"] == metadata["action_dim"] == 14
    assert metadata["tick_ns"] == tick
    assert metadata["alignment"] == "fixed_clock_30hz_causal"
    assert metadata["camera_resolutions"] == {"top": [2, 2], "left_wrist": [2, 2]}
