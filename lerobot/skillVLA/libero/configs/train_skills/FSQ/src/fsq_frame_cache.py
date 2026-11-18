#!/usr/bin/env python3
"""Exact, lossless, random-access RGB cache for FSQ visual terminators.

The source LeRobot datasets store long, inter-frame-compressed videos. FSQ
samples random timesteps from every skill, so seeking and decoding AV1 on every
epoch is much more expensive than the vision model itself. This module stores
each decoded uint8 RGB frame as one independent zstd frame plus a compact offset
index. Training can therefore read exactly the requested images without an AV1
seek, while a node-local copy remains roughly one third the size of raw RGB.

The cache is content-versioned by source-video metadata and atomically
published. The reader also accepts the historical raw-memmap v1 cache so jobs
submitted before the format migration remain valid while they are running.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import shlex
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CACHE_FORMAT_VERSION = 2
CACHE_FORMAT_NAME = "fsq_rgb_uint8_hwc_zstd_frame_v2"
CACHE_COMPRESSION = "zstd"
CACHE_COMPRESSION_LEVEL = 1
LEGACY_CACHE_FORMAT_VERSION = 1
LEGACY_CACHE_FORMAT_NAME = "fsq_rgb_uint8_hwc_memmap_v1"
SUCCESS_FILE = "_SUCCESS"
MANIFEST_FILE = "manifest.json"


def _video_files(raw_dataset_dir: str | Path) -> list[Path]:
    root = Path(raw_dataset_dir).resolve()
    videos = sorted((root / "videos").rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No MP4 videos found under {root / 'videos'}.")
    return videos


def _source_fingerprint_for_format(
    raw_dataset_dir: str | Path,
    format_name: str,
) -> str:
    """Stable cache key for one cache format and dataset/video revision."""
    root = Path(raw_dataset_dir).resolve()
    digest = hashlib.sha256()
    digest.update(f"{format_name}\0".encode())
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot dataset info.json not found: {info_path}")
    digest.update(b"meta/info.json\0")
    digest.update(info_path.read_bytes())
    for path in _video_files(root):
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode())
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def source_fingerprint(raw_dataset_dir: str | Path) -> str:
    """Stable cache key for the current lossless zstd format."""
    return _source_fingerprint_for_format(raw_dataset_dir, CACHE_FORMAT_NAME)


def legacy_source_fingerprint(raw_dataset_dir: str | Path) -> str:
    """Fingerprint used by the historical raw uint8 memmap cache."""
    return _source_fingerprint_for_format(raw_dataset_dir, LEGACY_CACHE_FORMAT_NAME)


def resolved_cache_dir(
    raw_dataset_dir: str | Path,
    cache_root: str | Path,
) -> Path:
    return Path(cache_root).resolve() / source_fingerprint(raw_dataset_dir)


def cache_job_file(cache_root: str | Path, fingerprint: str) -> Path:
    return Path(cache_root).resolve() / ".jobs" / f"{fingerprint}.job"


def _load_manifest(
    cache_dir: Path,
    *,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    manifest_path = cache_dir / MANIFEST_FILE
    success_path = cache_dir / SUCCESS_FILE
    if not manifest_path.is_file() or not success_path.is_file():
        raise FileNotFoundError(
            f"Frame cache is incomplete (missing {MANIFEST_FILE} or {SUCCESS_FILE}): "
            f"{cache_dir}"
        )
    manifest = json.loads(manifest_path.read_text())
    success_fingerprint = success_path.read_text().strip()
    cache_format = manifest.get("format")
    cache_version = int(manifest.get("format_version", -1))
    accepted = {(CACHE_FORMAT_NAME, CACHE_FORMAT_VERSION)}
    if allow_legacy:
        accepted.add((LEGACY_CACHE_FORMAT_NAME, LEGACY_CACHE_FORMAT_VERSION))
    if (cache_format, cache_version) not in accepted:
        raise ValueError(
            f"Unsupported frame-cache format in {manifest_path}: "
            f"format={cache_format!r}, version={cache_version!r}."
        )
    if success_fingerprint != manifest.get("source_fingerprint"):
        raise ValueError(
            f"Frame-cache completion marker does not match its manifest: {cache_dir}."
        )
    return manifest


def _safe_record_path(cache_dir: Path, value: Any) -> Path | None:
    relative = Path(str(value or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        return None
    path = cache_dir / relative
    return path if path.is_file() else None


def _record_files_are_complete(
    cache_dir: Path,
    record: dict[str, Any],
    cache_format: str,
) -> bool:
    try:
        shape = tuple(int(value) for value in record.get("shape", ()))
        if len(shape) != 4 or shape[-1] != 3 or min(shape) < 1:
            return False
        pts_path = _safe_record_path(cache_dir, record.get("pts"))
        if pts_path is None:
            return False
        if cache_format == LEGACY_CACHE_FORMAT_NAME:
            return _safe_record_path(cache_dir, record.get("frames")) is not None

        data_path = _safe_record_path(cache_dir, record.get("data"))
        offsets_path = _safe_record_path(cache_dir, record.get("offsets"))
        if data_path is None or offsets_path is None:
            return False
        compressed_bytes = int(record.get("compressed_bytes", -1))
        return (
            compressed_bytes == data_path.stat().st_size
            and compressed_bytes > 0
            and record.get("compression") == CACHE_COMPRESSION
            and int(record.get("compression_level", -1)) == CACHE_COMPRESSION_LEVEL
        )
    except (OSError, TypeError, ValueError):
        return False


def _manifest_matches_source(
    raw_root: Path,
    cache_dir: Path,
    manifest: dict[str, Any],
) -> bool:
    cache_format = str(manifest.get("format", ""))
    expected_fingerprint = _source_fingerprint_for_format(raw_root, cache_format)
    if manifest.get("source_fingerprint") != expected_fingerprint:
        return False
    records = manifest.get("videos")
    if not isinstance(records, dict):
        return False
    videos = _video_files(raw_root)
    expected_names = {path.relative_to(raw_root).as_posix() for path in videos}
    if set(records) != expected_names:
        return False
    for source_path in videos:
        relative = source_path.relative_to(raw_root).as_posix()
        record = records[relative]
        if not isinstance(record, dict):
            return False
        source_stat = source_path.stat()
        if (
            record.get("source_size") != source_stat.st_size
            or record.get("source_mtime_ns") != source_stat.st_mtime_ns
            or not _record_files_are_complete(cache_dir, record, cache_format)
        ):
            return False
    return True


def cache_is_complete(
    raw_dataset_dir: str | Path,
    cache_root: str | Path,
) -> bool:
    raw_root = Path(raw_dataset_dir).resolve()
    expected = source_fingerprint(raw_root)
    cache_dir = Path(cache_root).resolve() / expected
    try:
        manifest = _load_manifest(cache_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    return manifest.get("source_fingerprint") == expected and _manifest_matches_source(
        raw_root, cache_dir, manifest
    )


def cache_status(
    raw_dataset_dir: str | Path,
    cache_root: str | Path,
) -> dict[str, Any]:
    fingerprint = source_fingerprint(raw_dataset_dir)
    cache_dir = Path(cache_root).resolve() / fingerprint
    return {
        "fingerprint": fingerprint,
        "cache_dir": str(cache_dir),
        "complete": cache_is_complete(raw_dataset_dir, cache_root),
        "job_file": str(cache_job_file(cache_root, fingerprint)),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class _ZstdCodec:
    """Minimal dependency-free binding to the system libzstd."""

    def __init__(self) -> None:
        library = ctypes.util.find_library("zstd") or "libzstd.so.1"
        try:
            self.lib = ctypes.CDLL(library)
        except OSError as exc:
            raise RuntimeError(
                "lossless FSQ frame cache requires the system libzstd runtime"
            ) from exc
        self.lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_compressBound.restype = ctypes.c_size_t
        self.lib.ZSTD_compress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.ZSTD_compress.restype = ctypes.c_size_t
        self.lib.ZSTD_decompress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.ZSTD_decompress.restype = ctypes.c_size_t
        self.lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_isError.restype = ctypes.c_uint
        self.lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_getErrorName.restype = ctypes.c_char_p

    def _checked_size(self, value: int) -> int:
        if self.lib.ZSTD_isError(value):
            name = self.lib.ZSTD_getErrorName(value).decode("utf-8", "replace")
            raise RuntimeError(f"zstd frame-cache operation failed: {name}")
        return int(value)

    def compress(self, frame: Any, level: int = CACHE_COMPRESSION_LEVEL) -> bytes:
        import numpy as np

        source = np.ascontiguousarray(frame, dtype=np.uint8)
        bound = int(self.lib.ZSTD_compressBound(source.nbytes))
        destination = ctypes.create_string_buffer(bound)
        written = self._checked_size(
            self.lib.ZSTD_compress(
                destination,
                bound,
                ctypes.c_void_p(source.ctypes.data),
                source.nbytes,
                int(level),
            )
        )
        return destination.raw[:written]

    def decompress_into(self, payload: bytes, destination: Any) -> None:
        import numpy as np

        output = np.asarray(destination)
        if output.dtype != np.uint8 or not output.flags.c_contiguous:
            raise TypeError("zstd destination must be a contiguous uint8 array")
        source = ctypes.c_char_p(payload)
        written = self._checked_size(
            self.lib.ZSTD_decompress(
                ctypes.c_void_p(output.ctypes.data),
                output.nbytes,
                source,
                len(payload),
            )
        )
        if written != output.nbytes:
            raise RuntimeError(
                f"zstd frame decoded {written} bytes, expected {output.nbytes}"
            )


_ZSTD_CODEC: _ZstdCodec | None = None


def _zstd_codec() -> _ZstdCodec:
    global _ZSTD_CODEC
    if _ZSTD_CODEC is None:
        _ZSTD_CODEC = _ZstdCodec()
    return _ZSTD_CODEC


def _cache_paths(
    build_dir: Path,
    relative_video: Path,
) -> tuple[Path, Path, Path, Path]:
    relative_without_suffix = relative_video.relative_to("videos").with_suffix("")
    data_path = build_dir / "data" / relative_without_suffix.with_suffix(".zstframes")
    offsets_path = build_dir / "offsets" / relative_without_suffix.with_suffix(".npy")
    pts_path = build_dir / "pts" / relative_without_suffix.with_suffix(".npy")
    record_path = build_dir / "records" / relative_without_suffix.with_suffix(".json")
    return data_path, offsets_path, pts_path, record_path


def _record_is_reusable(
    record_path: Path,
    data_path: Path,
    offsets_path: Path,
    pts_path: Path,
    source_path: Path,
    stage_root: Path,
) -> dict[str, Any] | None:
    if not record_path.is_file():
        return None
    try:
        import numpy as np

        record = json.loads(record_path.read_text())
        source_stat = source_path.stat()
        if (
            record.get("source_size") != source_stat.st_size
            or record.get("source_mtime_ns") != source_stat.st_mtime_ns
            or not _record_files_are_complete(stage_root, record, CACHE_FORMAT_NAME)
        ):
            return None
        shape = tuple(int(value) for value in record["shape"])
        offsets = np.load(offsets_path, mmap_mode="r")
        pts = np.load(pts_path, mmap_mode="r")
        if (
            offsets.dtype != np.uint64
            or offsets.shape != (shape[0] + 1,)
            or int(offsets[0]) != 0
            or bool(np.any(offsets[1:] <= offsets[:-1]))
            or int(offsets[-1]) != data_path.stat().st_size
            or pts.dtype != np.float64
            or pts.shape != (shape[0],)
            or (len(pts) > 1 and bool(np.any(np.diff(pts) <= 0)))
        ):
            return None
        return record
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _validated_legacy_cache_dir(
    raw_root: Path,
    legacy_cache_root: Path | None,
) -> Path | None:
    if legacy_cache_root is None:
        return None
    cache_dir = legacy_cache_root / legacy_source_fingerprint(raw_root)
    try:
        manifest = _load_manifest(cache_dir, allow_legacy=True)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return None
    if manifest.get("format") != LEGACY_CACHE_FORMAT_NAME:
        return None
    return cache_dir if _manifest_matches_source(raw_root, cache_dir, manifest) else None


def _build_one_video(
    raw_dataset_dir: str,
    build_dir: str,
    relative_video_text: str,
    decoder_threads: int,
    legacy_cache_dir: str | None,
) -> dict[str, Any]:
    """Decode or convert one MP4 into independently compressed zstd frames."""
    import av
    import numpy as np

    raw_root = Path(raw_dataset_dir)
    stage_root = Path(build_dir)
    relative_video = Path(relative_video_text)
    source_path = raw_root / relative_video
    data_path, offsets_path, pts_path, record_path = _cache_paths(
        stage_root, relative_video
    )
    reusable = _record_is_reusable(
        record_path,
        data_path,
        offsets_path,
        pts_path,
        source_path,
        stage_root,
    )
    if reusable is not None:
        return reusable

    for path in (data_path, offsets_path, pts_path, record_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".partial.{os.getpid()}"
    data_tmp = data_path.with_name(data_path.name + suffix)
    offsets_tmp = offsets_path.with_name(offsets_path.name + suffix)
    pts_tmp = pts_path.with_name(pts_path.name + suffix)
    temporary_paths = (data_tmp, offsets_tmp, pts_tmp)
    for final_path in (data_path, offsets_path, pts_path):
        for stale in final_path.parent.glob(final_path.name + ".partial.*"):
            stale.unlink()
    for temporary in temporary_paths:
        if temporary.exists():
            temporary.unlink()

    legacy_frames = legacy_pts = None
    source_kind = "video_decode"
    if legacy_cache_dir:
        legacy_root = Path(legacy_cache_dir)
        legacy_manifest = _load_manifest(legacy_root, allow_legacy=True)
        legacy_record = legacy_manifest["videos"].get(relative_video.as_posix())
        if legacy_record is None:
            raise KeyError(f"Legacy cache has no record for {relative_video}.")
        legacy_frames = np.load(legacy_root / legacy_record["frames"], mmap_mode="r")
        legacy_pts = np.load(legacy_root / legacy_record["pts"], mmap_mode="r")
        expected_frames, height, width, channels = legacy_frames.shape
        if (
            legacy_frames.dtype != np.uint8
            or channels != 3
            or legacy_pts.dtype != np.float64
            or legacy_pts.shape != (expected_frames,)
        ):
            raise RuntimeError(f"Invalid legacy raw cache for {relative_video}.")
        source_kind = "legacy_raw_memmap"
    else:
        with av.open(str(source_path), mode="r") as probe:
            stream = probe.streams.video[0]
            expected_frames = int(stream.frames)
            height, width = int(stream.height), int(stream.width)
        if expected_frames <= 0:
            raise ValueError(
                f"Video has no indexed frame count and cannot be cached safely: {source_path}"
            )

    offsets = np.lib.format.open_memmap(
        offsets_tmp,
        mode="w+",
        dtype=np.uint64,
        shape=(expected_frames + 1,),
    )
    pts = np.lib.format.open_memmap(
        pts_tmp,
        mode="w+",
        dtype=np.float64,
        shape=(expected_frames,),
    )
    offsets[0] = 0
    codec = _zstd_codec()
    count = 0
    compressed_bytes = 0

    def append_frame(data_stream: Any, rgb: Any, timestamp: float) -> None:
        nonlocal count, compressed_bytes
        if count >= expected_frames:
            raise RuntimeError(
                f"Decoded more than the indexed {expected_frames} frames from {source_path}."
            )
        array = np.asarray(rgb)
        if array.shape != (height, width, 3) or array.dtype != np.uint8:
            raise RuntimeError(
                f"Unexpected decoded frame {array.shape}/{array.dtype} in {source_path}; "
                f"expected {(height, width, 3)}/uint8."
            )
        payload = codec.compress(array)
        data_stream.write(payload)
        compressed_bytes += len(payload)
        pts[count] = float(timestamp)
        count += 1
        offsets[count] = compressed_bytes

    try:
        with data_tmp.open("wb", buffering=4 * 1024 * 1024) as data_stream:
            if legacy_frames is not None:
                for index in range(expected_frames):
                    append_frame(data_stream, legacy_frames[index], float(legacy_pts[index]))
            else:
                with av.open(str(source_path), mode="r") as container:
                    stream = container.streams.video[0]
                    if decoder_threads > 0:
                        stream.thread_count = int(decoder_threads)
                    for frame in container.decode(stream):
                        if frame.pts is None or frame.time_base is None:
                            raise RuntimeError(
                                f"Video frame has no PTS: {source_path}, frame {count}."
                            )
                        append_frame(
                            data_stream,
                            frame.to_ndarray(format="rgb24"),
                            float(frame.pts * frame.time_base),
                        )
        if count != expected_frames:
            raise RuntimeError(
                f"Decoded {count} frames from {source_path}, indexed count is "
                f"{expected_frames}."
            )
        if count > 1 and bool(np.any(np.diff(pts) <= 0)):
            raise RuntimeError(f"Non-monotonic frame PTS in {source_path}.")
        first_pts = float(pts[0])
        last_pts = float(pts[-1])
        offsets.flush()
        pts.flush()
        del offsets, pts, legacy_frames, legacy_pts

        os.replace(data_tmp, data_path)
        os.replace(offsets_tmp, offsets_path)
        os.replace(pts_tmp, pts_path)
        source_stat = source_path.stat()
        raw_rgb_bytes = expected_frames * height * width * 3
        record = {
            "video": relative_video.as_posix(),
            "data": data_path.relative_to(stage_root).as_posix(),
            "offsets": offsets_path.relative_to(stage_root).as_posix(),
            "pts": pts_path.relative_to(stage_root).as_posix(),
            "shape": [expected_frames, height, width, 3],
            "dtype": "uint8",
            "pts_dtype": "float64",
            "compression": CACHE_COMPRESSION,
            "compression_level": CACHE_COMPRESSION_LEVEL,
            "compressed_bytes": compressed_bytes,
            "raw_rgb_bytes": raw_rgb_bytes,
            "compression_ratio": compressed_bytes / raw_rgb_bytes,
            "source_kind": source_kind,
            "first_pts": first_pts,
            "last_pts": last_pts,
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
        }
        _atomic_json(record_path, record)
        return record
    except BaseException:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()
        raise


def build_frame_cache(
    raw_dataset_dir: str | Path,
    cache_root: str | Path,
    *,
    expected_fingerprint: str | None = None,
    workers: int = 1,
    decoder_threads: int = 1,
    legacy_cache_root: str | Path | None = None,
) -> Path:
    raw_root = Path(raw_dataset_dir).resolve()
    cache_root = Path(cache_root).resolve()
    fingerprint = source_fingerprint(raw_root)
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise RuntimeError(
            "Source dataset changed between cache submission and execution: "
            f"expected {expected_fingerprint}, now {fingerprint}."
        )
    final_dir = cache_root / fingerprint
    if cache_is_complete(raw_root, cache_root):
        print(f"[FSQ frame cache] already complete: {final_dir}", flush=True)
        return final_dir
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}.")
    if decoder_threads < 1:
        raise ValueError(f"decoder_threads must be >= 1, got {decoder_threads}.")

    legacy_root = (
        Path(legacy_cache_root).resolve()
        if legacy_cache_root
        else (
            cache_root.parent / "rgb_uint8_v1"
            if cache_root.name == "rgb_zstd_v2"
            else None
        )
    )
    legacy_dir = _validated_legacy_cache_dir(raw_root, legacy_root)
    if legacy_dir is not None:
        print(
            f"[FSQ frame cache] converting exact RGB from legacy cache {legacy_dir}",
            flush=True,
        )
    else:
        print("[FSQ frame cache] no valid legacy cache; decoding source videos", flush=True)

    build_dir = cache_root / ".building" / fingerprint
    build_dir.mkdir(parents=True, exist_ok=True)
    videos = _video_files(raw_root)
    relative_videos = [path.relative_to(raw_root).as_posix() for path in videos]
    print(
        f"[FSQ frame cache] building {len(videos)} videos with {workers} workers "
        f"under {build_dir}",
        flush=True,
    )
    records: dict[str, dict[str, Any]] = {}

    def report(completed: int, relative: str, record: dict[str, Any]) -> None:
        raw_gib = int(record["raw_rgb_bytes"]) / 2**30
        compressed_gib = int(record["compressed_bytes"]) / 2**30
        print(
            f"[FSQ frame cache] {completed}/{len(videos)} {relative}: "
            f"{record['shape'][0]} frames, {raw_gib:.2f} -> "
            f"{compressed_gib:.2f} GiB "
            f"({100.0 * compressed_gib / raw_gib:.1f}%)",
            flush=True,
        )

    effective_workers = min(workers, len(videos))
    if effective_workers == 1:
        for completed, relative in enumerate(relative_videos, start=1):
            record = _build_one_video(
                str(raw_root),
                str(build_dir),
                relative,
                decoder_threads,
                str(legacy_dir) if legacy_dir is not None else None,
            )
            records[relative] = record
            report(completed, relative, record)
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures = {
                pool.submit(
                    _build_one_video,
                    str(raw_root),
                    str(build_dir),
                    relative,
                    decoder_threads,
                    str(legacy_dir) if legacy_dir is not None else None,
                ): relative
                for relative in relative_videos
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                relative = futures[future]
                record = future.result()
                records[relative] = record
                report(completed, relative, record)

    current_fingerprint = source_fingerprint(raw_root)
    if current_fingerprint != fingerprint:
        raise RuntimeError(
            "Source videos changed while their cache was being built: "
            f"{fingerprint} -> {current_fingerprint}."
        )
    ordered_records = {relative: records[relative] for relative in relative_videos}
    total_frames = sum(int(record["shape"][0]) for record in ordered_records.values())
    total_rgb_bytes = sum(
        int(record["raw_rgb_bytes"]) for record in ordered_records.values()
    )
    total_compressed_bytes = sum(
        int(record["compressed_bytes"]) for record in ordered_records.values()
    )
    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "format": CACHE_FORMAT_NAME,
        "source_fingerprint": fingerprint,
        "source_dataset": str(raw_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_videos": len(ordered_records),
        "total_frames": total_frames,
        "total_rgb_bytes": total_rgb_bytes,
        "total_compressed_bytes": total_compressed_bytes,
        "compression": CACHE_COMPRESSION,
        "compression_level": CACHE_COMPRESSION_LEVEL,
        "compression_ratio": total_compressed_bytes / total_rgb_bytes,
        "legacy_cache_source": str(legacy_dir) if legacy_dir is not None else None,
        "videos": ordered_records,
    }
    _atomic_json(build_dir / MANIFEST_FILE, manifest)
    success_tmp = build_dir / f".{SUCCESS_FILE}.tmp.{os.getpid()}"
    success_tmp.write_text(fingerprint + "\n")
    os.replace(success_tmp, build_dir / SUCCESS_FILE)

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        if cache_is_complete(raw_root, cache_root):
            print(f"[FSQ frame cache] another builder completed: {final_dir}", flush=True)
            return final_dir
        quarantine = cache_root / ".invalid" / (
            fingerprint
            + "."
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f".{os.getpid()}"
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        os.replace(final_dir, quarantine)
        print(
            f"[FSQ frame cache] moved incomplete published cache to {quarantine}",
            flush=True,
        )
    os.replace(build_dir, final_dir)
    _load_manifest(final_dir)
    print(
        f"[FSQ frame cache] complete: {final_dir} "
        f"({total_frames} frames, {total_rgb_bytes / 2**30:.2f} raw GiB -> "
        f"{total_compressed_bytes / 2**30:.2f} compressed GiB)",
        flush=True,
    )
    return final_dir


class RGBFrameCache:
    """Worker-local lazy reader for zstd v2 and historical raw v1 caches."""

    def __init__(
        self,
        cache_dir: str | Path,
        raw_dataset_dir: str | Path,
        *,
        verify_source: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.raw_dataset_dir = Path(raw_dataset_dir).resolve()
        self.manifest = _load_manifest(self.cache_dir, allow_legacy=True)
        self.cache_format = str(self.manifest["format"])
        if verify_source:
            expected = _source_fingerprint_for_format(
                self.raw_dataset_dir, self.cache_format
            )
            if self.manifest.get("source_fingerprint") != expected:
                raise ValueError(
                    f"Frame cache {self.cache_dir} belongs to source "
                    f"{self.manifest.get('source_fingerprint')}, current dataset is {expected}."
                )
        self._videos: dict[str, dict[str, Any]] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_videos"] = {}
        return state

    def close(self) -> None:
        for opened in self._videos.values():
            file_descriptor = opened.get("fd")
            if file_descriptor is not None:
                try:
                    os.close(int(file_descriptor))
                except OSError:
                    pass
        self._videos.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _relative_video(self, video_path: str | Path) -> str:
        path = Path(video_path).resolve()
        try:
            return path.relative_to(self.raw_dataset_dir).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Video {path} is outside cached dataset {self.raw_dataset_dir}."
            ) from exc

    def _open_video(self, relative_video: str):
        import numpy as np

        if relative_video in self._videos:
            return self._videos[relative_video]
        record = self.manifest.get("videos", {}).get(relative_video)
        if record is None:
            raise KeyError(
                f"Video {relative_video!r} is absent from frame cache {self.cache_dir}."
            )
        pts = np.load(self.cache_dir / record["pts"], mmap_mode="r")
        shape = tuple(int(value) for value in record["shape"])
        if pts.dtype != np.float64 or pts.shape != (shape[0],):
            raise RuntimeError(f"Corrupt frame-cache arrays for {relative_video}.")
        if self.cache_format == LEGACY_CACHE_FORMAT_NAME:
            frames = np.load(self.cache_dir / record["frames"], mmap_mode="r")
            if frames.dtype != np.uint8 or tuple(frames.shape) != shape:
                raise RuntimeError(f"Corrupt legacy frame cache for {relative_video}.")
            opened = {"record": record, "pts": pts, "frames": frames}
        else:
            offsets = np.load(self.cache_dir / record["offsets"], mmap_mode="r")
            data_path = self.cache_dir / record["data"]
            if (
                offsets.dtype != np.uint64
                or offsets.shape != (shape[0] + 1,)
                or int(offsets[0]) != 0
                or bool(np.any(offsets[1:] <= offsets[:-1]))
                or int(offsets[-1]) != data_path.stat().st_size
            ):
                raise RuntimeError(f"Corrupt zstd frame index for {relative_video}.")
            opened = {
                "record": record,
                "pts": pts,
                "offsets": offsets,
                "fd": os.open(data_path, os.O_RDONLY),
            }
        self._videos[relative_video] = opened
        return opened

    def get_frames(
        self,
        video_path: str | Path,
        timestamps: list[float],
        tolerance_s: float,
    ):
        """Return exact decoded uint8 frames as contiguous ``(M,C,H,W)`` tensors."""
        import numpy as np
        import torch

        if not timestamps:
            raise ValueError("At least one frame timestamp is required.")
        relative_video = self._relative_video(video_path)
        opened = self._open_video(relative_video)
        pts = opened["pts"]
        query = np.asarray(timestamps, dtype=np.float64)
        right = np.searchsorted(pts, query, side="left").clip(0, len(pts) - 1)
        left = (right - 1).clip(0, len(pts) - 1)
        left_distance = np.abs(pts[left] - query)
        right_distance = np.abs(pts[right] - query)
        indices = np.where(left_distance <= right_distance, left, right)
        distance = np.abs(pts[indices] - query)
        if bool(np.any(distance >= float(tolerance_s))):
            bad = distance >= float(tolerance_s)
            raise ValueError(
                f"Cached frame timestamps exceed tolerance {tolerance_s} for "
                f"{relative_video}: query={query[bad].tolist()}, "
                f"distance={distance[bad].tolist()}."
            )
        if self.cache_format == LEGACY_CACHE_FORMAT_NAME:
            # Advanced indexing materializes a writable contiguous copy rather
            # than exposing the historical read-only memmap directly to torch.
            selected = np.ascontiguousarray(opened["frames"][indices])
        else:
            shape = tuple(int(value) for value in opened["record"]["shape"])
            selected = np.empty((len(indices), *shape[1:]), dtype=np.uint8)
            offsets = opened["offsets"]
            file_descriptor = int(opened["fd"])
            codec = _zstd_codec()
            for frame_index in np.unique(indices):
                positions = np.flatnonzero(indices == frame_index)
                start = int(offsets[int(frame_index)])
                end = int(offsets[int(frame_index) + 1])
                payload = os.pread(file_descriptor, end - start, start)
                if len(payload) != end - start:
                    raise RuntimeError(
                        f"Short zstd frame read for {relative_video}, index "
                        f"{int(frame_index)}: {len(payload)} != {end - start}."
                    )
                first_position = int(positions[0])
                codec.decompress_into(payload, selected[first_position])
                if len(positions) > 1:
                    selected[positions[1:]] = selected[first_position]
        return torch.from_numpy(selected).permute(0, 3, 1, 2)


def _print_status_shell(status: dict[str, Any]) -> None:
    values = {
        "FSQ_FRAME_CACHE_FINGERPRINT": status["fingerprint"],
        "FSQ_FRAME_CACHE_DIR": status["cache_dir"],
        "FSQ_FRAME_CACHE_COMPLETE": "true" if status["complete"] else "false",
        "FSQ_FRAME_CACHE_JOB_FILE": status["job_file"],
    }
    for key, value in values.items():
        print(f"export {key}={shlex.quote(str(value))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--raw-dataset-dir", type=Path, required=True)
    status_parser.add_argument("--cache-root", type=Path, required=True)
    status_parser.add_argument("--shell", action="store_true")

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--raw-dataset-dir", type=Path, required=True)
    build_parser.add_argument("--cache-root", type=Path, required=True)
    build_parser.add_argument("--expected-fingerprint", default=None)
    build_parser.add_argument("--workers", type=int, default=1)
    build_parser.add_argument("--decoder-threads", type=int, default=1)
    build_parser.add_argument(
        "--legacy-cache-root",
        type=Path,
        default=None,
        help="Optional rgb_uint8_v1 root to convert instead of decoding MP4 again.",
    )

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--raw-dataset-dir", type=Path, required=True)
    validate_parser.add_argument("--cache-root", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "status":
        status = cache_status(args.raw_dataset_dir, args.cache_root)
        if args.shell:
            _print_status_shell(status)
        else:
            print(json.dumps(status, indent=2, sort_keys=True))
        return
    if args.command == "build":
        build_frame_cache(
            args.raw_dataset_dir,
            args.cache_root,
            expected_fingerprint=args.expected_fingerprint,
            workers=args.workers,
            decoder_threads=args.decoder_threads,
            legacy_cache_root=args.legacy_cache_root,
        )
        return
    status = cache_status(args.raw_dataset_dir, args.cache_root)
    if not status["complete"]:
        raise SystemExit(f"Frame cache is incomplete: {status['cache_dir']}")
    RGBFrameCache(status["cache_dir"], args.raw_dataset_dir, verify_source=True)
    print(f"[FSQ frame cache] valid: {status['cache_dir']}")


if __name__ == "__main__":
    main()
