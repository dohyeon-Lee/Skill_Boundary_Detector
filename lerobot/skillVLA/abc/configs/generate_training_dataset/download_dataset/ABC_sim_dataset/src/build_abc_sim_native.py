#!/usr/bin/env python3
"""Build a lightweight, read-only ABC Sim dataset view over official episodes.

The output deliberately contains no copied frame table and no re-encoded video.
It stores LeRobot-shaped metadata plus native episode references; a later dataset
adapter resolves those references at training time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from ABC_sim_dataset_config import DEFAULT_CONFIG_PATH, dataset_root, load_config, selected_subsets
from convert_abc_sim_dataset import (
    ACTION_DIM,
    SOURCE_FPS,
    STATE_DIM,
    _camera_layout,
    _episode_payload,
    _joint_names,
    prompt_timeline,
    select_episode_dirs,
)
from download_abc_sim import MANIFEST_FORMAT, SOURCE_MARKER


NATIVE_FORMAT = "sbd_abc_sim_native/v1"
NATIVE_MARKER = "native_dataset.json"


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _split_and_task(path: Path, raw_root: Path) -> tuple[str, str]:
    relative = path.relative_to(raw_root)
    if len(relative.parts) != 3 or not relative.parts[1].endswith("_sim"):
        raise ValueError(f"unexpected ABC Sim episode path: {relative}")
    return relative.parts[0], relative.parts[1][:-4]


def _episode_steps(path: Path) -> int:
    metadata = json.loads((path / "episode_metadata.json").read_text(encoding="utf-8"))
    values = (path / "states_actions.bin").stat().st_size // np.dtype("<f8").itemsize
    width = STATE_DIM + ACTION_DIM
    if values % width:
        raise ValueError(f"{path}: states_actions.bin has {values} values, not divisible by {width}")
    inferred = values // width
    declared = int(metadata.get("num_steps", 0))
    if declared > 0 and declared != inferred:
        raise ValueError(f"{path}: metadata num_steps={declared}, binary has {inferred}")
    if inferred <= 0:
        raise ValueError(f"{path}: episode is empty")
    return int(inferred)


class _VectorStats:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.count = 0
        self.total = np.zeros(dimension, dtype=np.float64)
        self.total_sq = np.zeros(dimension, dtype=np.float64)
        self.minimum = np.full(dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(dimension, -np.inf, dtype=np.float64)
        self.samples: list[np.ndarray] = []

    def update(self, values: np.ndarray, sample: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"expected [N,{self.dimension}], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("state/action contains non-finite values")
        self.count += len(values)
        self.total += values.sum(axis=0)
        self.total_sq += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        if len(sample):
            self.samples.append(np.asarray(sample, dtype=np.float32))

    def finish(self) -> dict[str, list[float] | list[int]]:
        if self.count == 0:
            raise ValueError("cannot compute statistics for an empty dataset")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        sampled = np.concatenate(self.samples, axis=0)
        quantiles = np.quantile(sampled, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
        return {
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [self.count],
            "q01": quantiles[0].astype(np.float64).tolist(),
            "q10": quantiles[1].astype(np.float64).tolist(),
            "q50": quantiles[2].astype(np.float64).tolist(),
            "q90": quantiles[3].astype(np.float64).tolist(),
            "q99": quantiles[4].astype(np.float64).tolist(),
        }


def _request(
    name: str,
    episode_dirs: list[Path],
    raw_root: Path,
    *,
    fps: int,
    cameras: list[str],
    rename: dict[str, str],
    fallback_size: tuple[int, int],
) -> tuple[str, dict[str, Any]]:
    sources = []
    task_hashes: dict[str, str] = {}
    for path in episode_dirs:
        source_task, split = _split_and_task(path, raw_root)
        if source_task not in task_hashes:
            marker = json.loads((raw_root / source_task / SOURCE_MARKER).read_text(encoding="utf-8"))
            task_hashes[source_task] = str(marker["sha256"])
        sources.append({"path": path.relative_to(raw_root).as_posix(), "split": split})
    request = {
        "format": NATIVE_FORMAT,
        "dataset": name,
        "raw_root": "../_sim_raw",
        "fps": fps,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "cameras": cameras,
        "camera_rename": rename,
        "source_image_size": list(fallback_size),
        "image_preprocessing": "raw_camera_crop",
        "task_sha256": task_hashes,
        "sources": sources,
    }
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), request


def _migrate_legacy_square_metadata(
    final_dir: Path, existing: dict[str, Any], request: dict[str, Any], fingerprint: str
) -> bool:
    """Upgrade the old virtual 256-square contract without rescanning episodes or stats."""
    if not isinstance(existing.get("image_size"), int):
        return False
    shared_request = {
        key: value for key, value in request.items() if key != "image_preprocessing"
    }
    if any(existing.get(key) != value for key, value in shared_request.items()):
        return False

    info_path = final_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    source_width, source_height = [int(value) for value in request["source_image_size"]]
    for camera in request["cameras"]:
        feature = info["features"][f"observation.images.{camera}"]
        feature["shape"] = [source_height, source_width, 3]

    upgraded = dict(existing)
    upgraded.pop("image_size", None)
    upgraded["image_preprocessing"] = request["image_preprocessing"]
    upgraded["request_sha256"] = fingerprint
    _json_dump(info_path, info)
    _json_dump(final_dir / NATIVE_MARKER, upgraded)
    print(
        f"[{request['dataset']}] migrated native images to raw "
        f"{source_width}x{source_height} crops -> skip"
    )
    return True


def build_subset(
    name: str,
    episode_dirs: list[Path],
    final_dir: Path,
    raw_root: Path,
    *,
    fps: int,
    cameras: list[str],
    rename: dict[str, str],
    fallback_size: tuple[int, int],
    quantile_sample_cap: int,
    force: bool,
) -> None:
    fingerprint, request = _request(
        name,
        episode_dirs,
        raw_root,
        fps=fps,
        cameras=cameras,
        rename=rename,
        fallback_size=fallback_size,
    )
    marker_path = final_dir / NATIVE_MARKER
    if marker_path.is_file() and not force:
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") == fingerprint:
            print(f"[{name}] native view already matches request -> skip")
            return
        if _migrate_legacy_square_metadata(final_dir, existing, request, fingerprint):
            return
        raise RuntimeError(f"[{name}] existing native view has different provenance; rerun with FORCE=1")
    if final_dir.exists() and not force:
        raise RuntimeError(f"[{name}] non-native or incomplete output exists: {final_dir}; rerun with FORCE=1")

    lengths = [_episode_steps(path) for path in episode_dirs]
    total_frames = sum(lengths)
    if quantile_sample_cap <= 0:
        raise ValueError("abc_sim_stats_quantile_samples must be positive")
    sample_stride = max(1, math.ceil(total_frames / quantile_sample_cap))

    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.native-", dir=parent))
    try:
        state_stats = _VectorStats(STATE_DIM)
        action_stats = _VectorStats(ACTION_DIM)
        task_to_index: dict[str, int] = {}
        episode_rows: list[dict[str, Any]] = []
        global_offset = 0

        for episode_index, (episode_dir, num_steps) in enumerate(zip(episode_dirs, lengths, strict=True)):
            metadata, table, loaded_steps = _episode_payload(episode_dir)
            if loaded_steps != num_steps:
                raise RuntimeError(f"{episode_dir}: episode length changed during build")
            timeline = prompt_timeline(metadata, num_steps)
            layout = _camera_layout(metadata, rename, cameras, fallback_size)
            source_task, source_split = _split_and_task(episode_dir, raw_root)
            video_path = episode_dir / "combined_camera-images-rgb.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(video_path)

            segments = []
            episode_tasks: list[str] = []
            for frame, prompt in timeline:
                if prompt not in task_to_index:
                    task_to_index[prompt] = len(task_to_index)
                segments.append({"frame": int(frame), "task_index": task_to_index[prompt]})
                if prompt not in episode_tasks:
                    episode_tasks.append(prompt)

            first_sample = (-global_offset) % sample_stride
            sampled = table[first_sample::sample_stride]
            state_stats.update(table[:, :STATE_DIM], sampled[:, :STATE_DIM])
            action_stats.update(table[:, STATE_DIM:], sampled[:, STATE_DIM:])

            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "length": num_steps,
                    "dataset_from_index": global_offset,
                    "dataset_to_index": global_offset + num_steps,
                    "tasks": episode_tasks,
                    "task_indices": [task_to_index[prompt] for prompt in episode_tasks],
                    "source_task": source_task,
                    "source_split": source_split,
                    "source_episode": episode_dir.name,
                    "source_path": episode_dir.relative_to(raw_root).as_posix(),
                    "states_actions_path": (episode_dir / "states_actions.bin").relative_to(raw_root).as_posix(),
                    "video_path": video_path.relative_to(raw_root).as_posix(),
                    "prompt_segments_json": json.dumps(segments, separators=(",", ":")),
                    "camera_layout_json": json.dumps(
                        [
                            {"source": raw, "camera": canonical, "row_start": start, "row_end": end}
                            for raw, canonical, start, end in layout
                        ],
                        separators=(",", ":"),
                    ),
                }
            )
            global_offset += num_steps
            if (episode_index + 1) % 100 == 0 or episode_index + 1 == len(episode_dirs):
                print(
                    f"[{name}] indexed {episode_index + 1}/{len(episode_dirs)} episodes "
                    f"({global_offset} frames)",
                    flush=True,
                )

        (temporary / "meta").mkdir(parents=True, exist_ok=True)
        tasks = pd.DataFrame(
            {"task_index": list(task_to_index.values())}, index=pd.Index(task_to_index.keys(), name="task")
        )
        tasks.to_parquet(temporary / "meta" / "tasks.parquet")
        episodes_path = temporary / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(episode_rows).to_parquet(episodes_path, index=False)

        joint_names = _joint_names()
        features: dict[str, Any] = {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": joint_names},
            "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": joint_names},
        }
        for camera in cameras:
            features[f"observation.images.{camera}"] = {
                "dtype": "video",
                "shape": [fallback_size[1], fallback_size[0], 3],
                "names": ["height", "width", "channel"],
            }
        info = {
            "codebase_version": "v3.0",
            "storage_format": NATIVE_FORMAT,
            "robot_type": "yam",
            "total_episodes": len(episode_rows),
            "total_frames": total_frames,
            "total_tasks": len(task_to_index),
            "fps": fps,
            "splits": {"all": f"0:{len(episode_rows)}"},
            "native_split_counts": {
                split: sum(row["source_split"] == split for row in episode_rows)
                for split in sorted({row["source_split"] for row in episode_rows})
            },
            "features": features,
        }
        _json_dump(temporary / "meta" / "info.json", info)
        _json_dump(
            temporary / "meta" / "stats.json",
            {"observation.state": state_stats.finish(), "action": action_stats.finish()},
        )
        request.update(
            {
                "request_sha256": fingerprint,
                "total_episodes": len(episode_rows),
                "total_frames": total_frames,
                "total_tasks": len(task_to_index),
                "quantile_sample_stride": sample_stride,
                "quantile_sample_cap": quantile_sample_cap,
            }
        )
        _json_dump(temporary / NATIVE_MARKER, request)

        if final_dir.exists():
            shutil.rmtree(final_dir)
        temporary.replace(final_dir)
        print(
            f"[{name}] native view ready: {len(episode_rows)} episodes / "
            f"{total_frames} frames / {len(task_to_index)} prompts"
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--only", default=os.environ.get("ABC_SIM_ONLY", ""))
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE", "") not in {"", "0"})
    args = parser.parse_args()

    config = load_config(args.config)
    root = dataset_root(config)
    manifest_path = root / "_sim_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}; run download_ABC_sim.sh")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"unexpected ABC Sim manifest format: {manifest.get('format')!r}")
    fps = int(config.get("abc_sim_fps", SOURCE_FPS))
    if int(manifest.get("fps", SOURCE_FPS)) != fps:
        raise ValueError("configured fps differs from the official ABC Sim manifest")
    source_width, source_height = [int(value) for value in manifest.get("image_size", [224, 168])]
    specs = selected_subsets(config, args.only)
    if not specs:
        raise SystemExit("no abc_sim_subsets configured")
    raw_root = root / "_sim_raw"
    for name, subset in specs.items():
        episodes = select_episode_dirs(name, subset, raw_root, manifest)
        build_subset(
            name,
            episodes,
            root / name,
            raw_root,
            fps=fps,
            cameras=[str(value) for value in config.get("abc_sim_cameras", ["top", "left_wrist", "right_wrist"])],
            rename={str(key): str(value) for key, value in config.get("abc_sim_camera_rename", {}).items()},
            fallback_size=(source_width, source_height),
            quantile_sample_cap=int(config.get("abc_sim_stats_quantile_samples", 1_000_000)),
            force=args.force,
        )


if __name__ == "__main__":
    main()
