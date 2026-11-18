#!/usr/bin/env python3
"""Build and read exact frozen-DINO tokens for FSQ term22 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CACHE_FORMAT_VERSION = 1
CACHE_FORMAT_NAME = "fsq_dino_top_tokens_bfloat16_v1"
SUCCESS_FILE = "_SUCCESS"
MANIFEST_FILE = "manifest.json"
TOP_CAMERA_PREFIX = "videos/observation.images.image/"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _model_fingerprint(model_path: str | Path) -> str:
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DINO model directory not found: {root}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name == "config.json"
            or path.suffix in {".safetensors", ".bin"}
        )
    )
    if not files:
        raise FileNotFoundError(f"No DINO config/weights found under {root}.")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()[:24]


def source_fingerprint(
    raw_dataset_dir: str | Path,
    model_path: str | Path,
    image_size: int,
) -> str:
    from fsq_frame_cache import source_fingerprint as rgb_source_fingerprint

    digest = hashlib.sha256()
    fields = (
        CACHE_FORMAT_NAME,
        rgb_source_fingerprint(raw_dataset_dir),
        _model_fingerprint(model_path),
        str(int(image_size)),
        "imagenet_mean_std",
        "cls_plus_patch_tokens_without_registers",
    )
    digest.update("\0".join(fields).encode())
    return digest.hexdigest()[:24]


def resolved_cache_dir(
    raw_dataset_dir: str | Path,
    model_path: str | Path,
    image_size: int,
    cache_root: str | Path,
) -> Path:
    return Path(cache_root).resolve() / source_fingerprint(
        raw_dataset_dir, model_path, image_size
    )


def cache_job_file(cache_root: str | Path, fingerprint: str) -> Path:
    return Path(cache_root).resolve() / ".jobs" / f"{fingerprint}.job"


def _load_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / MANIFEST_FILE
    success_path = cache_dir / SUCCESS_FILE
    if not manifest_path.is_file() or not success_path.is_file():
        raise FileNotFoundError(f"DINO feature cache is incomplete: {cache_dir}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("format") != CACHE_FORMAT_NAME
        or int(manifest.get("format_version", -1)) != CACHE_FORMAT_VERSION
    ):
        raise ValueError(f"Unsupported DINO feature cache: {manifest_path}")
    if success_path.read_text().strip() != manifest.get("source_fingerprint"):
        raise ValueError(f"DINO cache completion marker mismatch: {cache_dir}")
    return manifest


def _records_complete(cache_dir: Path, manifest: dict[str, Any]) -> bool:
    import numpy as np

    records = manifest.get("videos")
    if not isinstance(records, dict) or not records:
        return False
    try:
        for relative, record in records.items():
            if not relative.startswith(TOP_CAMERA_PREFIX):
                return False
            features = np.load(cache_dir / record["features"], mmap_mode="r")
            pts = np.load(cache_dir / record["pts"], mmap_mode="r")
            shape = tuple(int(value) for value in record["shape"])
            if (
                features.dtype != np.uint16
                or tuple(features.shape) != shape
                or len(shape) != 3
                or pts.dtype != np.float64
                or pts.shape != (shape[0],)
            ):
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def cache_is_complete(
    raw_dataset_dir: str | Path,
    model_path: str | Path,
    image_size: int,
    cache_root: str | Path,
) -> bool:
    expected = source_fingerprint(raw_dataset_dir, model_path, image_size)
    cache_dir = Path(cache_root).resolve() / expected
    try:
        manifest = _load_manifest(cache_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    return manifest.get("source_fingerprint") == expected and _records_complete(
        cache_dir, manifest
    )


def cache_status(
    raw_dataset_dir: str | Path,
    model_path: str | Path,
    image_size: int,
    cache_root: str | Path,
) -> dict[str, Any]:
    fingerprint = source_fingerprint(raw_dataset_dir, model_path, image_size)
    return {
        "fingerprint": fingerprint,
        "cache_dir": str(Path(cache_root).resolve() / fingerprint),
        "complete": cache_is_complete(
            raw_dataset_dir, model_path, image_size, cache_root
        ),
        "job_file": str(cache_job_file(cache_root, fingerprint)),
    }


def _cache_paths(build_dir: Path, relative_video: str) -> tuple[Path, Path, Path]:
    stem = Path(relative_video).relative_to("videos").with_suffix("")
    return (
        build_dir / "features" / stem.with_suffix(".npy"),
        build_dir / "pts" / stem.with_suffix(".npy"),
        build_dir / "records" / stem.with_suffix(".json"),
    )


def build_feature_cache(
    raw_dataset_dir: str | Path,
    rgb_cache_dir: str | Path,
    model_path: str | Path,
    image_size: int,
    cache_root: str | Path,
    *,
    expected_fingerprint: str | None = None,
    batch_size: int = 256,
) -> Path:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from FSQ import _load_dino_model
    from fsq_frame_cache import RGBFrameCache

    if not torch.cuda.is_available():
        raise RuntimeError("Building the DINO feature cache requires CUDA.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    raw_root = Path(raw_dataset_dir).resolve()
    cache_root = Path(cache_root).resolve()
    fingerprint = source_fingerprint(raw_root, model_path, image_size)
    if expected_fingerprint and expected_fingerprint != fingerprint:
        raise RuntimeError(
            "DINO cache inputs changed after submission: "
            f"{expected_fingerprint} -> {fingerprint}."
        )
    final_dir = cache_root / fingerprint
    if cache_is_complete(raw_root, model_path, image_size, cache_root):
        print(f"[FSQ DINO cache] already complete: {final_dir}", flush=True)
        return final_dir

    rgb = RGBFrameCache(rgb_cache_dir, raw_root, verify_source=True)
    rgb_records = {
        relative: record
        for relative, record in rgb.manifest["videos"].items()
        if relative.startswith(TOP_CAMERA_PREFIX)
    }
    if not rgb_records:
        raise ValueError("RGB cache contains no top-camera videos.")

    device = torch.device("cuda")
    model = _load_dino_model(str(Path(model_path).resolve())).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    n_register = int(getattr(model.config, "num_register_tokens", 0))
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    build_dir = cache_root / ".building" / fingerprint
    build_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    total_frames = 0
    total_bytes = 0
    try:
        for video_number, (relative, rgb_record) in enumerate(
            sorted(rgb_records.items()), start=1
        ):
            feature_path, pts_path, record_path = _cache_paths(build_dir, relative)
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            pts_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            source_pts = np.load(Path(rgb_cache_dir) / rgb_record["pts"], mmap_mode="r")
            frame_count = int(len(source_pts))
            feature_tmp = feature_path.with_name(feature_path.name + f".partial.{os.getpid()}")
            pts_tmp = pts_path.with_name(pts_path.name + f".partial.{os.getpid()}")
            for stale in (feature_tmp, pts_tmp):
                if stale.exists():
                    stale.unlink()

            output = None
            video_path = raw_root / relative
            for start in range(0, frame_count, batch_size):
                stop = min(start + batch_size, frame_count)
                timestamps = source_pts[start:stop].astype(float).tolist()
                frames = rgb.get_frames(video_path, timestamps, tolerance_s=1e-7)
                x = frames.to(device=device, non_blocking=True).float().div_(255.0)
                x = F.interpolate(
                    x,
                    size=(int(image_size), int(image_size)),
                    mode="bilinear",
                    align_corners=False,
                )
                x = (x - mean) / std
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    hidden = model(x).last_hidden_state
                    tokens = torch.cat(
                        [hidden[:, :1], hidden[:, 1 + n_register :]], dim=1
                    ).to(torch.bfloat16)
                bits = tokens.contiguous().view(torch.uint16).cpu().numpy()
                if output is None:
                    output = np.lib.format.open_memmap(
                        feature_tmp,
                        mode="w+",
                        dtype=np.uint16,
                        shape=(frame_count, bits.shape[1], bits.shape[2]),
                    )
                output[start:stop] = bits
            if output is None:
                raise RuntimeError(f"Top-camera video has no frames: {relative}")
            shape = tuple(int(value) for value in output.shape)
            output.flush()
            del output
            np.save(pts_tmp, np.asarray(source_pts, dtype=np.float64))
            # np.save appends .npy only when the provided path lacks that suffix.
            saved_pts_tmp = pts_tmp if pts_tmp.is_file() else Path(str(pts_tmp) + ".npy")
            os.replace(feature_tmp, feature_path)
            os.replace(saved_pts_tmp, pts_path)
            record = {
                "video": relative,
                "features": feature_path.relative_to(build_dir).as_posix(),
                "pts": pts_path.relative_to(build_dir).as_posix(),
                "shape": list(shape),
                "dtype": "bfloat16_bits_uint16",
                "bytes": feature_path.stat().st_size,
            }
            _atomic_json(record_path, record)
            records[relative] = record
            total_frames += frame_count
            total_bytes += int(record["bytes"])
            print(
                f"[FSQ DINO cache] {video_number}/{len(rgb_records)} {relative}: "
                f"{frame_count} frames",
                flush=True,
            )
    finally:
        rgb.close()

    current_fingerprint = source_fingerprint(raw_root, model_path, image_size)
    if current_fingerprint != fingerprint:
        raise RuntimeError(
            "Dataset or DINO model changed while the feature cache was built: "
            f"{fingerprint} -> {current_fingerprint}."
        )
    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "format": CACHE_FORMAT_NAME,
        "source_fingerprint": fingerprint,
        "source_dataset": str(raw_root),
        "rgb_cache_dir": str(Path(rgb_cache_dir).resolve()),
        "model_path": str(Path(model_path).resolve()),
        "model_fingerprint": _model_fingerprint(model_path),
        "image_size": int(image_size),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_videos": len(records),
        "total_frames": total_frames,
        "total_bytes": total_bytes,
        "videos": {key: records[key] for key in sorted(records)},
    }
    _atomic_json(build_dir / MANIFEST_FILE, manifest)
    marker = build_dir / f".{SUCCESS_FILE}.tmp.{os.getpid()}"
    marker.write_text(fingerprint + "\n")
    os.replace(marker, build_dir / SUCCESS_FILE)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        if cache_is_complete(raw_root, model_path, image_size, cache_root):
            print(
                f"[FSQ DINO cache] another builder completed: {final_dir}",
                flush=True,
            )
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
            f"[FSQ DINO cache] moved incomplete cache to {quarantine}",
            flush=True,
        )
    os.replace(build_dir, final_dir)
    print(
        f"[FSQ DINO cache] complete: {final_dir} "
        f"({total_frames} frames, {total_bytes / 2**30:.2f} GiB)",
        flush=True,
    )
    return final_dir


class DINOFeatureCache:
    """Worker-local, lazy mmap reader for frozen top-camera DINO tokens."""

    def __init__(
        self,
        cache_dir: str | Path,
        raw_dataset_dir: str | Path,
        *,
        model_path: str | Path,
        image_size: int,
        verify_source: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.raw_dataset_dir = Path(raw_dataset_dir).resolve()
        self.manifest = _load_manifest(self.cache_dir)
        if verify_source:
            expected = source_fingerprint(
                self.raw_dataset_dir, model_path, image_size
            )
            if self.manifest.get("source_fingerprint") != expected:
                raise ValueError(
                    f"DINO cache {self.cache_dir} belongs to "
                    f"{self.manifest.get('source_fingerprint')}, expected {expected}."
                )
        self._videos: dict[str, tuple[Any, Any]] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_videos"] = {}
        return state

    def _relative_video(self, video_path: str | Path) -> str:
        return Path(video_path).resolve().relative_to(self.raw_dataset_dir).as_posix()

    def _open_video(self, relative: str):
        import numpy as np

        if relative not in self._videos:
            record = self.manifest["videos"].get(relative)
            if record is None:
                raise KeyError(f"Top-camera video is absent from DINO cache: {relative}")
            features = np.load(self.cache_dir / record["features"], mmap_mode="r")
            pts = np.load(self.cache_dir / record["pts"], mmap_mode="r")
            self._videos[relative] = (features, pts)
        return self._videos[relative]

    def get_features(
        self,
        video_path: str | Path,
        timestamps: list[float],
        tolerance_s: float,
    ):
        import numpy as np
        import torch

        features, pts = self._open_video(self._relative_video(video_path))
        query = np.asarray(timestamps, dtype=np.float64)
        right = np.searchsorted(pts, query, side="left").clip(0, len(pts) - 1)
        left = (right - 1).clip(0, len(pts) - 1)
        indices = np.where(
            np.abs(pts[left] - query) <= np.abs(pts[right] - query), left, right
        )
        distance = np.abs(pts[indices] - query)
        if bool(np.any(distance >= float(tolerance_s))):
            raise ValueError(
                f"Cached DINO timestamps exceed tolerance {tolerance_s}: "
                f"{distance[distance >= float(tolerance_s)].tolist()}"
            )
        selected = np.ascontiguousarray(features[indices])
        return torch.from_numpy(selected).view(torch.bfloat16)


def _print_status_shell(status: dict[str, Any]) -> None:
    values = {
        "FSQ_DINO_FEATURE_CACHE_FINGERPRINT": status["fingerprint"],
        "FSQ_DINO_FEATURE_CACHE_DIR": status["cache_dir"],
        "FSQ_DINO_FEATURE_CACHE_COMPLETE": "true" if status["complete"] else "false",
        "FSQ_DINO_FEATURE_CACHE_JOB_FILE": status["job_file"],
    }
    for key, value in values.items():
        print(f"export {key}={shlex.quote(str(value))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "validate"):
        command = subparsers.add_parser(name)
        command.add_argument("--raw-dataset-dir", type=Path, required=True)
        command.add_argument("--model-path", type=Path, required=True)
        command.add_argument("--image-size", type=int, required=True)
        command.add_argument("--cache-root", type=Path, required=True)
        if name == "status":
            command.add_argument("--shell", action="store_true")
    build = subparsers.add_parser("build")
    build.add_argument("--raw-dataset-dir", type=Path, required=True)
    build.add_argument("--rgb-cache-dir", type=Path, required=True)
    build.add_argument("--model-path", type=Path, required=True)
    build.add_argument("--image-size", type=int, required=True)
    build.add_argument("--cache-root", type=Path, required=True)
    build.add_argument("--expected-fingerprint")
    build.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.command == "build":
        build_feature_cache(
            args.raw_dataset_dir,
            args.rgb_cache_dir,
            args.model_path,
            args.image_size,
            args.cache_root,
            expected_fingerprint=args.expected_fingerprint,
            batch_size=args.batch_size,
        )
        return
    status = cache_status(
        args.raw_dataset_dir, args.model_path, args.image_size, args.cache_root
    )
    if args.command == "status" and args.shell:
        _print_status_shell(status)
    elif args.command == "status":
        print(json.dumps(status, indent=2, sort_keys=True))
    elif not status["complete"]:
        raise SystemExit(f"DINO feature cache is incomplete: {status['cache_dir']}")
    else:
        DINOFeatureCache(
            status["cache_dir"],
            args.raw_dataset_dir,
            model_path=args.model_path,
            image_size=args.image_size,
        )
        print(f"[FSQ DINO cache] valid: {status['cache_dir']}")


if __name__ == "__main__":
    main()
