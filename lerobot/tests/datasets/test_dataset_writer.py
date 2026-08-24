#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contract tests for DatasetWriter."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from lerobot.datasets.dataset_writer import _encode_video_worker
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
from tests.fixtures.constants import DEFAULT_FPS, DUMMY_REPO_ID

SIMPLE_FEATURES = {
    "state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
}

VIDEO_FEATURES = {
    **SIMPLE_FEATURES,
    "observation.images.camera": {
        "dtype": "video",
        "shape": (16, 16, 3),
        "names": ["height", "width", "channels"],
        # Keep this non-empty so the unit test does not need a real video just
        # to populate codec metadata. Actual video validity is covered by the
        # converter round-trip test.
        "info": {"video.fps": DEFAULT_FPS, "video.codec": "av1"},
    },
}


def _make_frame(features: dict, task: str = "Dummy task") -> dict:
    """Build a valid frame dict for the given features."""
    frame = {"task": task}
    for key, ft in features.items():
        if ft["dtype"] in ("image", "video"):
            frame[key] = np.random.randint(0, 256, size=ft["shape"], dtype=np.uint8)
        elif ft["dtype"] in ("float32", "float64"):
            frame[key] = torch.randn(ft["shape"])
        elif ft["dtype"] == "int64":
            frame[key] = torch.zeros(ft["shape"], dtype=torch.int64)
    return frame


# ── Existing encode_video_worker tests ───────────────────────────────


def test_encode_video_worker_forwards_vcodec(tmp_path):
    """_encode_video_worker correctly forwards the vcodec parameter."""
    video_key = "observation.images.laptop"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=0, frame_index=0)
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="red").save(img_dir / "frame-000000.png")

    captured_kwargs = {}

    def mock_encode(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.dataset_writer.encode_video_frames", side_effect=mock_encode):
        _encode_video_worker(video_key, 0, tmp_path, fps=30, vcodec="h264")

    assert captured_kwargs["vcodec"] == "h264"


def test_encode_video_worker_default_vcodec(tmp_path):
    """_encode_video_worker uses libsvtav1 as the default codec."""
    video_key = "observation.images.laptop"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=0, frame_index=0)
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="red").save(img_dir / "frame-000000.png")

    captured_kwargs = {}

    def mock_encode(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.dataset_writer.encode_video_frames", side_effect=mock_encode):
        _encode_video_worker(video_key, 0, tmp_path, fps=30)

    assert captured_kwargs["vcodec"] == "libsvtav1"


# ── add_frame contracts ──────────────────────────────────────────────


def test_add_frame_increments_buffer_size(tmp_path):
    """Each add_frame() call increases episode_buffer['size'] by 1."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    assert dataset.writer.episode_buffer["size"] == 0

    dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    assert dataset.writer.episode_buffer["size"] == 1

    dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    assert dataset.writer.episode_buffer["size"] == 2


def test_add_frame_rejects_missing_feature(tmp_path):
    """add_frame() raises ValueError when a required feature is missing."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    with pytest.raises(ValueError, match="Missing features"):
        dataset.add_frame({"task": "Dummy task", "state": torch.randn(6)})
        # missing 'action'


# ── save_episode contracts ───────────────────────────────────────────


def test_save_episode_writes_parquet(tmp_path):
    """After save_episode(), at least one .parquet file exists under data/."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    parquet_files = list((tmp_path / "ds" / "data").rglob("*.parquet"))
    assert len(parquet_files) > 0


def test_save_episode_updates_counters(tmp_path):
    """After save_episode(), metadata counters are updated."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(5):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 5


def test_save_episode_resets_buffer(tmp_path):
    """After save_episode(), the episode buffer is reset."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    assert dataset.writer.episode_buffer["size"] == 0


def test_save_multiple_episodes(tmp_path):
    """Recording 3 episodes results in correct total counts."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    total_frames = 0
    for ep in range(3):
        n_frames = ep + 2  # 2, 3, 4
        for _ in range(n_frames):
            dataset.add_frame(_make_frame(SIMPLE_FEATURES))
        dataset.save_episode()
        total_frames += n_frames

    assert dataset.meta.total_episodes == 3
    assert dataset.meta.total_frames == total_frames


def test_deferred_video_packing_remuxes_once_per_shard(tmp_path):
    """Deferred packing assigns timestamps immediately but remuxes once per shard."""
    root = tmp_path / "deferred"
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=DEFAULT_FPS,
        features=VIDEO_FEATURES,
        root=root,
        deferred_video_packing=True,
    )
    dataset.meta.update_chunk_settings(video_files_size_in_mb=1)
    writer = dataset.writer
    video_key = "observation.images.camera"

    temp_paths = []
    for episode_index in range(3):
        temp_dir = root / f"temp-{episode_index}"
        temp_dir.mkdir()
        temp_path = temp_dir / f"episode-{episode_index}.mp4"
        temp_path.write_bytes(bytes([episode_index]) * 400_000)
        temp_paths.append(temp_path)

    concat_calls = []

    def fake_concat(input_paths, output_path, overwrite=True):
        concat_calls.append(list(input_paths))
        output_path.write_bytes(b"".join(Path(path).read_bytes() for path in input_paths))

    with (
        patch("lerobot.datasets.dataset_writer.get_video_duration_in_s", return_value=1.0),
        patch("lerobot.datasets.dataset_writer.concatenate_video_files", side_effect=fake_concat),
    ):
        metadata = [
            writer._save_episode_video(video_key, episode_index, temp_path=temp_path)
            for episode_index, temp_path in enumerate(temp_paths)
        ]

        # Two 0.38 MiB episodes share file 0. The third rotates to file 1,
        # flushing file 0 once rather than once after each episode.
        assert len(concat_calls) == 1
        assert metadata[0][f"videos/{video_key}/file_index"] == 0
        assert metadata[1][f"videos/{video_key}/file_index"] == 0
        assert metadata[2][f"videos/{video_key}/file_index"] == 1
        assert metadata[0][f"videos/{video_key}/from_timestamp"] == 0.0
        assert metadata[1][f"videos/{video_key}/from_timestamp"] == 1.0
        assert metadata[2][f"videos/{video_key}/from_timestamp"] == 0.0

        dataset.finalize()

    packed_0 = root / dataset.meta.video_path.format(video_key=video_key, chunk_index=0, file_index=0)
    packed_1 = root / dataset.meta.video_path.format(video_key=video_key, chunk_index=0, file_index=1)
    assert packed_0.stat().st_size == 800_000
    assert packed_1.stat().st_size == 400_000
    assert not any(path.parent.exists() for path in temp_paths)


def test_streaming_encoding_can_defer_video_packing(tmp_path):
    """Streaming episode MP4s can use the same one-remux-per-shard packing path."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=DEFAULT_FPS,
        features=VIDEO_FEATURES,
        root=tmp_path / "streaming-deferred",
        vcodec="h264",
        streaming_encoding=True,
        deferred_video_packing=True,
    )

    assert dataset.writer._streaming_encoder is not None
    assert dataset.writer._deferred_video_packing is True
    dataset.writer.flush_pending_videos()
    assert dataset.writer._streaming_encoder._closed is True


# ── clear / lifecycle ────────────────────────────────────────────────


def test_clear_resets_buffer(tmp_path):
    """clear_episode_buffer() resets the buffer size to 0."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    assert dataset.writer.episode_buffer["size"] == 1

    dataset.clear_episode_buffer()
    assert dataset.writer.episode_buffer["size"] == 0


def test_finalize_is_idempotent(tmp_path):
    """Calling finalize() twice does not raise."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    dataset.finalize()
    dataset.finalize()  # second call should not raise


def test_finalize_then_read_roundtrip(tmp_path):
    """Write data, finalize, re-open, and verify data matches."""
    root = tmp_path / "roundtrip"
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    dataset = LeRobotDataset.create(repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=features, root=root)

    # Record known values
    known_states = []
    for i in range(5):
        state = torch.tensor([float(i), float(i * 10)])
        known_states.append(state)
        dataset.add_frame({"task": "Test task", "state": state})
    dataset.save_episode()
    dataset.finalize()

    # Read back
    for i in range(5):
        item = dataset[i]
        assert torch.allclose(item["state"], known_states[i], atol=1e-5)
