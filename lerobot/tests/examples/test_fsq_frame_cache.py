from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys

import av
import numpy as np
import pytest
import torch


LIBERO_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "libero"
sys.path.insert(0, str(LIBERO_EXAMPLES))

from fsq_frame_cache import (  # noqa: E402
    CACHE_FORMAT_NAME,
    LEGACY_CACHE_FORMAT_NAME,
    RGBFrameCache,
    build_frame_cache,
    cache_status,
    legacy_source_fingerprint,
)


def _write_test_video(path: Path, frame_count: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width = 16
        stream.height = 12
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            pixels = np.empty((12, 16, 3), dtype=np.uint8)
            pixels[..., 0] = index * 31
            pixels[..., 1] = np.arange(16, dtype=np.uint8)[None, :]
            pixels[..., 2] = np.arange(12, dtype=np.uint8)[:, None]
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 20)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _decode_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(float(frame.pts * frame.time_base))
    return np.stack(frames), np.asarray(timestamps, dtype=np.float64)


def test_rgb_frame_cache_matches_sequential_video_decode(tmp_path: Path) -> None:
    raw_dataset = tmp_path / "dataset"
    (raw_dataset / "meta").mkdir(parents=True)
    (raw_dataset / "meta" / "info.json").write_text('{"fps": 20}\n')
    video = (
        raw_dataset
        / "videos"
        / "observation.images.image"
        / "chunk-000"
        / "file-000.mp4"
    )
    _write_test_video(video)
    reference_frames, reference_pts = _decode_reference(video)

    cache_root = tmp_path / "cache"
    cache_dir = build_frame_cache(
        raw_dataset,
        cache_root,
        workers=1,
        decoder_threads=1,
    )

    status = cache_status(raw_dataset, cache_root)
    assert status["complete"] is True
    assert Path(status["cache_dir"]) == cache_dir
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert manifest["format"] == CACHE_FORMAT_NAME
    assert manifest["compression"] == "zstd"
    assert manifest["total_compressed_bytes"] < manifest["total_rgb_bytes"]
    assert list((cache_dir / "data").rglob("*.zstframes"))
    assert not (cache_dir / "frames").exists()
    reader = RGBFrameCache(cache_dir, raw_dataset)
    selected = reader.get_frames(
        video,
        [float(reference_pts[1]), float(reference_pts[4])],
        tolerance_s=1e-6,
    )
    expected = torch.from_numpy(reference_frames[[1, 4]].copy()).permute(0, 3, 1, 2)
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)
    assert selected.dtype == torch.uint8

    # Re-running preparation is an idempotent no-op against the published
    # fingerprint directory.
    assert build_frame_cache(raw_dataset, cache_root, workers=1) == cache_dir

    # A damaged published directory is not reused. Preparation preserves it in
    # quarantine and atomically publishes a repaired cache under the same key.
    next((cache_dir / "pts").rglob("*.npy")).unlink()
    assert cache_status(raw_dataset, cache_root)["complete"] is False
    assert build_frame_cache(raw_dataset, cache_root, workers=1) == cache_dir
    assert cache_status(raw_dataset, cache_root)["complete"] is True
    assert any((cache_root / ".invalid").iterdir())


def test_rgb_frame_cache_rejects_timestamp_outside_tolerance(tmp_path: Path) -> None:
    raw_dataset = tmp_path / "dataset"
    (raw_dataset / "meta").mkdir(parents=True)
    (raw_dataset / "meta" / "info.json").write_text("{}\n")
    video = raw_dataset / "videos" / "camera" / "chunk-000" / "file-000.mp4"
    _write_test_video(video, frame_count=2)
    cache_dir = build_frame_cache(raw_dataset, tmp_path / "cache", workers=1)

    reader = RGBFrameCache(cache_dir, raw_dataset)
    with pytest.raises(ValueError, match="exceed tolerance"):
        reader.get_frames(video, [10.0], tolerance_s=0.01)


def test_zstd_cache_parallel_builder_publishes_multiple_videos(tmp_path: Path) -> None:
    raw_dataset = tmp_path / "dataset"
    (raw_dataset / "meta").mkdir(parents=True)
    (raw_dataset / "meta" / "info.json").write_text("{}\n")
    videos = [
        raw_dataset / f"videos/camera-{camera}/chunk-000/file-000.mp4"
        for camera in ("top", "wrist")
    ]
    for video in videos:
        _write_test_video(video, frame_count=4)

    cache_dir = build_frame_cache(raw_dataset, tmp_path / "cache", workers=2)

    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert len(manifest["videos"]) == 2
    assert {record["source_kind"] for record in manifest["videos"].values()} == {
        "video_decode"
    }
    reader = RGBFrameCache(cache_dir, raw_dataset)
    for video in videos:
        frames, pts = _decode_reference(video)
        selected = reader.get_frames(video, [float(pts[2])], tolerance_s=1e-6)
        expected = torch.from_numpy(frames[[2]].copy()).permute(0, 3, 1, 2)
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)


def _write_legacy_raw_cache(
    raw_dataset: Path,
    legacy_root: Path,
    video: Path,
    frames: np.ndarray,
    pts: np.ndarray,
) -> Path:
    fingerprint = legacy_source_fingerprint(raw_dataset)
    cache_dir = legacy_root / fingerprint
    relative = video.relative_to(raw_dataset)
    stem = relative.relative_to("videos").with_suffix("")
    frames_path = cache_dir / "frames" / stem.with_suffix(".npy")
    pts_path = cache_dir / "pts" / stem.with_suffix(".npy")
    frames_path.parent.mkdir(parents=True)
    pts_path.parent.mkdir(parents=True)
    np.save(frames_path, frames)
    np.save(pts_path, pts)
    stat = video.stat()
    record = {
        "video": relative.as_posix(),
        "frames": frames_path.relative_to(cache_dir).as_posix(),
        "pts": pts_path.relative_to(cache_dir).as_posix(),
        "shape": list(frames.shape),
        "dtype": "uint8",
        "pts_dtype": "float64",
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }
    manifest = {
        "format_version": 1,
        "format": LEGACY_CACHE_FORMAT_NAME,
        "source_fingerprint": fingerprint,
        "videos": {relative.as_posix(): record},
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))
    (cache_dir / "_SUCCESS").write_text(fingerprint + "\n")
    return cache_dir


def test_zstd_cache_converts_legacy_raw_cache_and_reader_remains_compatible(
    tmp_path: Path,
) -> None:
    raw_dataset = tmp_path / "dataset"
    (raw_dataset / "meta").mkdir(parents=True)
    (raw_dataset / "meta" / "info.json").write_text('{"fps":20}\n')
    video = raw_dataset / "videos/camera/chunk-000/file-000.mp4"
    _write_test_video(video, frame_count=5)
    frames, pts = _decode_reference(video)
    legacy_root = tmp_path / "rgb_uint8_v1"
    legacy_dir = _write_legacy_raw_cache(
        raw_dataset, legacy_root, video, frames, pts
    )

    # Historical jobs can still open their already-resolved v1 directory.
    legacy_reader = RGBFrameCache(legacy_dir, raw_dataset)
    legacy_selected = legacy_reader.get_frames(
        video, [float(pts[0]), float(pts[-1])], tolerance_s=1e-6
    )
    expected = torch.from_numpy(frames[[0, -1]].copy()).permute(0, 3, 1, 2)
    torch.testing.assert_close(legacy_selected, expected, rtol=0, atol=0)

    zstd_dir = build_frame_cache(
        raw_dataset,
        tmp_path / "rgb_zstd_v2",
        workers=1,
        legacy_cache_root=legacy_root,
    )
    manifest = json.loads((zstd_dir / "manifest.json").read_text())
    record = next(iter(manifest["videos"].values()))
    assert record["source_kind"] == "legacy_raw_memmap"
    reader = RGBFrameCache(zstd_dir, raw_dataset)
    selected = reader.get_frames(
        video, [float(pts[0]), float(pts[-1])], tolerance_s=1e-6
    )
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)
