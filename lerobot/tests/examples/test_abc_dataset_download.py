from __future__ import annotations

import sys
from pathlib import Path


ABC_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/generate_training_dataset/download_dataset/ABC_dataset/src"
)
sys.path.insert(0, str(ABC_SRC))

from download_abc_subset import (  # noqa: E402
    COMPLETION_MARKER,
    _completion_status,
    _request_sha256,
    _write_completion_marker,
)


def _request() -> dict:
    return {
        "repo_id": "XDOF/ABC-130k",
        "split": "train",
        "out_dir": "/tmp/abc_a",
        "group_subdirs": False,
        "include_meta": False,
        "max_workers": 4,
        "hf_transfer": False,
        "convert_to_abcdl": False,
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
