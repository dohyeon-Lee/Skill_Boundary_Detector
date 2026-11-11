from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile

import av
import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ABC_sim_dataset_config import load_config, selected_subsets  # noqa: E402
from convert_abc_sim_dataset import (  # noqa: E402
    _build_v3_dataset,
    fit_square,
    prompt_at,
    prompt_timeline,
    select_episode_dirs,
)
from download_abc_sim import SOURCE_MARKER, install_task, task_installed  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_config_selects_a_small_deterministic_subset() -> None:
    config = load_config()
    specs = selected_subsets(config, "abc_sim_toy")

    assert list(specs) == ["abc_sim_toy"]
    assert specs["abc_sim_toy"]["split"] == "train"
    assert specs["abc_sim_toy"]["tasks"][0]["episodes"] == 20
    assert config["abc_sim_camera_rename"] == {"left": "left_wrist", "right": "right_wrist"}


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
        archive.add(payload / "train_sim", arcname="train_sim")
        archive.add(payload / "val_sim", arcname="val_sim")
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
