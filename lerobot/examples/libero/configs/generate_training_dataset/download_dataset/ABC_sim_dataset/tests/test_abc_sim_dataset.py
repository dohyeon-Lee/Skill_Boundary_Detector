from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import threading
import time

import av
import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ABC_sim_dataset_config import load_config, selected_subsets  # noqa: E402
from build_abc_sim_native import NATIVE_FORMAT, build_subset as build_native_subset  # noqa: E402
import download_abc_sim as download_module  # noqa: E402
from convert_abc_sim_dataset import (  # noqa: E402
    _build_v3_dataset,
    fit_square,
    prompt_at,
    prompt_timeline,
    select_episode_dirs,
)
from download_abc_sim import (  # noqa: E402
    SOURCE_MARKER,
    configured_tasks,
    install_task,
    install_tasks_parallel,
    task_installed,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str]):
        super().__init__(payload)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_resume_rejection_never_truncates_partial(tmp_path: Path, monkeypatch) -> None:
    partial = tmp_path / "task.tar.part"
    partial.write_bytes(b"abc")
    responses = iter(
        [
            _FakeResponse(b"abcdef", status=200, headers={}),
            _FakeResponse(
                b"def",
                status=206,
                headers={"Content-Range": "bytes 3-5/6"},
            ),
        ]
    )
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        assert timeout == 120
        assert request.get_header("Range") == "bytes=3-"
        if calls:
            assert partial.read_bytes() == b"abc"
        calls += 1
        return next(responses)

    monkeypatch.setattr(download_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(download_module.time, "sleep", lambda _delay: None)

    download_module._download_with_resume(
        "https://example.invalid/task.tar",
        partial,
        expected_bytes=6,
        retries=2,
        label="task",
    )

    assert calls == 2
    assert partial.read_bytes() == b"abcdef"


def test_default_config_selects_all_training_tasks() -> None:
    config = load_config()
    specs = selected_subsets(config)

    assert specs == {"abc_sim_all": {"split": "train", "tasks": "all"}}
    assert config["abc_sim_camera_rename"] == {"left": "left_wrist", "right": "right_wrist"}


def test_all_task_selector_expands_from_manifest_order() -> None:
    specs = {"all": {"split": "both", "tasks": "all"}}

    assert configured_tasks(specs, ["task_b", "task_a"]) == ["task_b", "task_a"]


def test_prompt_timeline_preserves_mid_episode_directives() -> None:
    timeline = prompt_timeline(
        {
            "task_name": "multi_drawer_search",
            "prompt_timeline": [
                {"frame": 0, "prompt": "find the red block"},
                {"frame": 3, "prompt": "find the blue block"},
            ],
        },
        6,
    )

    assert [prompt_at(timeline, index) for index in range(6)] == [
        "find the red block",
        "find the red block",
        "find the red block",
        "find the blue block",
        "find the blue block",
        "find the blue block",
    ]
    assert prompt_timeline({"task_name": "sim_pouring_beads"}, 2) == (
        (0, "sim pouring beads"),
    )


def test_fit_square_preserves_aspect_ratio_and_pads() -> None:
    image = np.full((6, 12, 3), 200, dtype=np.uint8)
    square = fit_square(image, 12)

    assert square.shape == (12, 12, 3)
    assert np.all(square[:3] == 0)
    assert np.all(square[3:9] == 200)
    assert np.all(square[9:] == 0)


def test_task_tar_is_hash_verified_extracted_and_reused(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    for split in ("train", "val"):
        episode = payload / f"{split}_sim" / "episode_000"
        episode.mkdir(parents=True)
        (episode / "states_actions.bin").write_bytes(b"rows")
    archive_path = tmp_path / "source.tar"
    with tarfile.open(archive_path, "w") as archive:
        # Official archives use train/ and val/; installation normalizes these
        # to the local train_sim/ and val_sim/ episode layout.
        archive.add(payload / "train_sim", arcname="train")
        archive.add(payload / "val_sim", arcname="val")
    task = {
        "uri": archive_path.as_uri(),
        "bytes": archive_path.stat().st_size,
        "sha256": _sha256(archive_path),
        "episodes": {"train": 1, "val": 1},
        "frames": 2,
        "hours": 0.0,
    }
    raw_root = tmp_path / "raw"
    downloads = tmp_path / "downloads"

    result = install_task(
        "sim_test",
        task,
        raw_root=raw_root,
        download_root=downloads,
        retries=1,
        keep_tar=False,
        force=False,
        manifest_generated="test",
    )

    assert result == "done"
    assert task_installed(raw_root / "sim_test", task)
    assert not (downloads / "sim_test.tar").exists()
    assert install_task(
        "sim_test",
        task,
        raw_root=raw_root,
        download_root=downloads,
        retries=1,
        keep_tar=False,
        force=False,
        manifest_generated="test",
    ) == "skip"


def test_two_task_downloads_are_parallel_but_extraction_is_serial(
    tmp_path: Path, monkeypatch
) -> None:
    tasks = {}
    for name in ("task_a", "task_b"):
        payload = tmp_path / f"payload_{name}"
        for split in ("train", "val"):
            episode = payload / split / "episode_000"
            episode.mkdir(parents=True)
            (episode / "states_actions.bin").write_bytes(b"rows")
        archive_path = tmp_path / f"source_{name}.tar"
        with tarfile.open(archive_path, "w") as archive:
            archive.add(payload / "train", arcname="train")
            archive.add(payload / "val", arcname="val")
        tasks[name] = {
            "uri": archive_path.as_uri(),
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
            "episodes": {"train": 1, "val": 1},
            "frames": 2,
            "hours": 0.0,
        }

    download_barrier = threading.Barrier(2)
    state_lock = threading.Lock()
    active_downloads = 0
    max_downloads = 0
    active_extracts = 0
    max_extracts = 0
    real_download = download_module.download_task
    real_extract = download_module.extract_task

    def tracked_download(*args, **kwargs):
        nonlocal active_downloads, max_downloads
        with state_lock:
            active_downloads += 1
            max_downloads = max(max_downloads, active_downloads)
        download_barrier.wait(timeout=5)
        try:
            time.sleep(0.05)
            return real_download(*args, **kwargs)
        finally:
            with state_lock:
                active_downloads -= 1

    def tracked_extract(*args, **kwargs):
        nonlocal active_extracts, max_extracts
        with state_lock:
            active_extracts += 1
            max_extracts = max(max_extracts, active_extracts)
        try:
            time.sleep(0.05)
            return real_extract(*args, **kwargs)
        finally:
            with state_lock:
                active_extracts -= 1

    monkeypatch.setattr(download_module, "download_task", tracked_download)
    monkeypatch.setattr(download_module, "extract_task", tracked_extract)
    raw_root = tmp_path / "raw"
    download_root = tmp_path / "downloads"
    raw_root.mkdir()
    download_root.mkdir()

    failures = install_tasks_parallel(
        list(tasks),
        tasks,
        raw_root=raw_root,
        download_root=download_root,
        retries=1,
        keep_tar=False,
        force=False,
        manifest_generated="test",
        download_workers=2,
    )

    assert failures == []
    assert max_downloads == 2
    assert max_extracts == 1
    assert all(task_installed(raw_root / name, task) for name, task in tasks.items())


def test_episode_selection_uses_shared_verified_raw_task(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    task_root = raw_root / "sim_test"
    for index in range(3):
        episode = task_root / "train_sim" / f"episode_{index:03d}"
        episode.mkdir(parents=True)
        (episode / "states_actions.bin").write_bytes(b"rows")
        metadata = {"task_name": "sim test"}
        if index == 0:
            metadata["prompt_source"] = {"status": "excluded"}
        (episode / "episode_metadata.json").write_text(json.dumps(metadata))
    task = {
        "sha256": "abc",
        "episodes": {"train": 3, "val": 0},
    }
    (task_root / SOURCE_MARKER).write_text(json.dumps({"sha256": "abc"}))
    manifest = {"tasks": {"sim_test": task}}
    subset = {"split": "train", "tasks": [{"task": "sim_test", "episodes": 2}]}

    selected = select_episode_dirs("toy", subset, raw_root, manifest)

    assert [path.name for path in selected] == ["episode_001", "episode_002"]


def _write_episode(root: Path) -> None:
    root.mkdir(parents=True)
    steps = 3
    table = np.arange(steps * 28, dtype=np.float64).reshape(steps, 28)
    table.astype("<f8").tofile(root / "states_actions.bin")
    metadata = {
        "num_steps": steps,
        "task_name": "multi_drawer_search",
        "cameras": ["top", "left", "right"],
        "camera_resolutions": {name: [12, 8] for name in ("top", "left", "right")},
        "prompt_timeline": [
            {"frame": 0, "prompt": "first directive"},
            {"frame": 2, "prompt": "second directive"},
        ],
    }
    (root / "episode_metadata.json").write_text(json.dumps(metadata))
    container = av.open(str(root / "combined_camera-images-rgb.mp4"), mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width = 12
    stream.height = 24
    stream.pix_fmt = "yuv420p"
    for frame_index in range(steps):
        stacked = np.zeros((24, 12, 3), dtype=np.uint8)
        stacked[:8] = 20 + frame_index
        stacked[8:16] = 80 + frame_index
        stacked[16:] = 140 + frame_index
        frame = av.VideoFrame.from_ndarray(stacked, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_synthetic_episode_builds_real_lerobot_v3_with_frame_prompts(tmp_path: Path) -> None:
    episode = tmp_path / "episode_000"
    _write_episode(episode)
    output = tmp_path / "lerobot"

    episodes, frames = _build_v3_dataset(
        "local/abc_sim_test",
        str(output),
        [str(episode)],
        fps=30,
        image_size=16,
        cameras=["top", "left_wrist", "right_wrist"],
        rename={"left": "left_wrist", "right": "right_wrist"},
        fallback_size=(12, 8),
        encoder_threads=1,
        prefix="[test] ",
    )

    info = json.loads((output / "meta" / "info.json").read_text())
    tasks = pd.read_parquet(output / "meta" / "tasks.parquet")
    assert (episodes, frames) == (1, 3)
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 3
    assert set(tasks.index.astype(str)) == {"first directive", "second directive"}
    assert set(info["features"]) >= {
        "observation.state",
        "action",
        "observation.images.top",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }


def test_synthetic_episode_builds_native_view_without_copying_frames(tmp_path: Path) -> None:
    raw_root = tmp_path / "dataset_ABC" / "_sim_raw"
    task_root = raw_root / "sim_test"
    episode = task_root / "train_sim" / "episode_000"
    _write_episode(episode)
    (task_root / SOURCE_MARKER).write_text(json.dumps({"sha256": "source-hash"}))
    output = tmp_path / "dataset_ABC" / "abc_sim_test"

    build_native_subset(
        "abc_sim_test",
        [episode],
        output,
        raw_root,
        fps=30,
        image_size=16,
        cameras=["top", "left_wrist", "right_wrist"],
        rename={"left": "left_wrist", "right": "right_wrist"},
        fallback_size=(12, 8),
        quantile_sample_cap=100,
        force=False,
    )

    marker = json.loads((output / "native_dataset.json").read_text())
    info = json.loads((output / "meta" / "info.json").read_text())
    stats = json.loads((output / "meta" / "stats.json").read_text())
    tasks = pd.read_parquet(output / "meta" / "tasks.parquet").reset_index()
    episodes = pd.read_parquet(output / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    assert marker["format"] == NATIVE_FORMAT
    assert marker["raw_root"] == "../_sim_raw"
    assert marker["total_frames"] == 3
    assert info["storage_format"] == NATIVE_FORMAT
    assert info["total_episodes"] == 1
    assert set(tasks["task"]) == {"first directive", "second directive"}
    assert episodes.loc[0, "dataset_from_index"] == 0
    assert episodes.loc[0, "dataset_to_index"] == 3
    assert episodes.loc[0, "source_path"] == "sim_test/train_sim/episode_000"
    assert len(json.loads(episodes.loc[0, "prompt_segments_json"])) == 2
    assert stats["observation.state"]["count"] == [3]
    assert stats["action"]["count"] == [3]
    assert not (output / "data").exists()
    assert not (output / "videos").exists()

    # An identical request is idempotent and keeps the same provenance.
    build_native_subset(
        "abc_sim_test",
        [episode],
        output,
        raw_root,
        fps=30,
        image_size=16,
        cameras=["top", "left_wrist", "right_wrist"],
        rename={"left": "left_wrist", "right": "right_wrist"},
        fallback_size=(12, 8),
        quantile_sample_cap=100,
        force=False,
    )
    assert json.loads((output / "native_dataset.json").read_text())["request_sha256"] == marker[
        "request_sha256"
    ]
