#!/usr/bin/env python3
"""Convert official ABC Sim episode tar contents into LeRobot v3 datasets."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from ABC_sim_dataset_config import (
    DEFAULT_CONFIG_PATH,
    dataset_root,
    load_config,
    project_root,
    selected_subsets,
)
from download_abc_sim import MANIFEST_FORMAT, SOURCE_MARKER, task_installed

STATE_DIM = 14
ACTION_DIM = 14
SOURCE_FPS = 30
PROVENANCE_FILE = "abc_sim_source.json"


def _setup_paths(config: dict[str, Any]) -> None:
    lerobot_src = project_root(config) / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(lerobot_src)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    )


def _joint_names() -> list[str]:
    names: list[str] = []
    for side in ("left", "right"):
        names.extend([f"{side}_joint_{index}" for index in range(6)])
        names.append(f"{side}_gripper")
    return names


def _task_entry(entry: str | dict[str, Any]) -> tuple[str, Any, str | None]:
    if isinstance(entry, str):
        return entry, "all", None
    if not isinstance(entry, dict) or not str(entry.get("task", "")).strip():
        raise ValueError(f"invalid task entry: {entry!r}")
    return str(entry["task"]), entry.get("episodes", "all"), entry.get("split")


def _splits(value: str) -> list[str]:
    value = str(value).strip().lower()
    if value == "both":
        return ["train", "val"]
    if value not in {"train", "val"}:
        raise ValueError(f"split must be train, val, or both; got {value!r}")
    return [value]


def _select_from_pool(pool: list[Path], selector: Any, *, label: str) -> list[Path]:
    if selector == "all" or selector is None:
        return pool
    if isinstance(selector, int):
        if selector < 0:
            raise ValueError(f"{label}: episodes must be non-negative")
        return pool[:selector]
    if not isinstance(selector, list):
        raise ValueError(f"{label}: episodes must be all, an integer, or a list")
    selected: list[Path] = []
    for value in selector:
        if isinstance(value, int):
            try:
                selected.append(pool[value])
            except IndexError as exc:
                raise ValueError(f"{label}: episode index {value} out of range ({len(pool)})") from exc
            continue
        matches = [path for path in pool if path.name == str(value)]
        if len(matches) != 1:
            raise ValueError(f"{label}: episode name {value!r} matched {len(matches)} entries")
        selected.append(matches[0])
    if len(set(selected)) != len(selected):
        raise ValueError(f"{label}: duplicate episode selection")
    return selected


def _prompt_eligible(path: Path) -> bool:
    metadata_path = path / "episode_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"episode metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (metadata.get("prompt_source") or {}).get("status") != "excluded"


def select_episode_dirs(
    subset_name: str,
    subset: dict[str, Any],
    raw_root: Path,
    manifest: dict[str, Any],
) -> list[Path]:
    result: list[Path] = []
    default_split = str(subset.get("split", "train"))
    entries = subset.get("tasks") or []
    if not entries:
        raise ValueError(f"{subset_name}: no tasks configured")
    for entry in entries:
        task_name, selector, split_override = _task_entry(entry)
        task = manifest.get("tasks", {}).get(task_name)
        if task is None:
            raise ValueError(f"{subset_name}: task absent from manifest: {task_name}")
        task_root = raw_root / task_name
        if not task_installed(task_root, task):
            raise FileNotFoundError(
                f"{subset_name}: raw task is missing or stale: {task_root}; run download_ABC_sim.sh"
            )
        for split in _splits(split_override or default_split):
            complete_pool = sorted(
                path.parent
                for path in (task_root / f"{split}_sim").glob("episode_*/states_actions.bin")
            )
            pool = [path for path in complete_pool if _prompt_eligible(path)]
            if len(pool) != len(complete_pool):
                print(
                    f"[{subset_name}] {task_name}/{split}: skipped "
                    f"{len(complete_pool) - len(pool)} episode(s) explicitly excluded by prompt metadata"
                )
            selected = _select_from_pool(
                pool, selector, label=f"{subset_name}/{task_name}/{split}"
            )
            result.extend(selected)
    if not result:
        raise ValueError(f"{subset_name}: selection produced no episodes")
    if len(set(result)) != len(result):
        raise ValueError(f"{subset_name}: the same source episode was selected more than once")
    return result


def _normal_prompt(value: str) -> str:
    return " ".join(str(value).replace("-", " ").replace("_", " ").split())


def prompt_timeline(metadata: dict[str, Any], num_steps: int) -> tuple[tuple[int, str], ...]:
    if (metadata.get("prompt_source") or {}).get("status") == "excluded":
        raise ValueError("episode is explicitly excluded from prompt reconstruction")
    task_name = str(metadata.get("task_name") or "").strip()
    default = _normal_prompt(task_name)
    instruction = metadata.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        if _normal_prompt(instruction) != default:
            default = instruction.strip()
    raw_timeline = metadata.get("prompt_timeline")
    if raw_timeline:
        timeline = [(int(item["frame"]), str(item["prompt"]).strip()) for item in raw_timeline]
        if any(not prompt for _, prompt in timeline):
            raise ValueError("prompt_timeline contains an empty prompt")
        if timeline != sorted(timeline) or len({frame for frame, _ in timeline}) != len(timeline):
            raise ValueError("prompt_timeline frames must be strictly increasing")
        if timeline[0][0] > 0:
            if not default:
                raise ValueError("prompt_timeline starts after frame 0 without a fallback prompt")
            timeline.insert(0, (0, default))
    else:
        if not default:
            raise ValueError("episode has neither task_name nor prompt_timeline")
        timeline = [(0, default)]
    if timeline[0][0] < 0 or timeline[-1][0] >= num_steps:
        raise ValueError(f"prompt_timeline is outside [0, {num_steps})")
    return tuple(timeline)


def prompt_at(timeline: tuple[tuple[int, str], ...], frame_index: int) -> str:
    frames = [frame for frame, _ in timeline]
    return timeline[max(0, bisect_right(frames, int(frame_index)) - 1)][1]


def _camera_layout(
    metadata: dict[str, Any],
    rename: dict[str, str],
    target_cameras: list[str],
    fallback_size: tuple[int, int],
) -> list[tuple[str, str, int, int]]:
    raw_cameras = list(metadata.get("cameras") or ["top", "left", "right"])
    resolutions = metadata.get("camera_resolutions") or {}
    result: list[tuple[str, str, int, int]] = []
    offset = 0
    for raw_name in raw_cameras:
        width, height = resolutions.get(raw_name, fallback_size)
        canonical = rename.get(raw_name, raw_name)
        result.append((raw_name, canonical, offset, offset + int(height)))
        offset += int(height)
    actual = [canonical for _, canonical, _, _ in result]
    if len(actual) != len(set(actual)) or set(actual) != set(target_cameras):
        raise ValueError(f"camera mapping {actual} does not match target {target_cameras}")
    return result


def fit_square(image: np.ndarray, size: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got {image.shape}")
    if image.shape[:2] == (size, size):
        return np.ascontiguousarray(image)
    import cv2

    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x = (size - resized_width) // 2
    y = (size - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _episode_payload(path: Path) -> tuple[dict[str, Any], np.ndarray, int]:
    metadata = json.loads((path / "episode_metadata.json").read_text(encoding="utf-8"))
    num_steps = int(metadata.get("num_steps", 0))
    raw = np.fromfile(path / "states_actions.bin", dtype="<f8")
    if num_steps <= 0:
        if raw.size % (STATE_DIM + ACTION_DIM):
            raise ValueError(f"{path}: invalid states_actions.bin size {raw.size}")
        num_steps = raw.size // (STATE_DIM + ACTION_DIM)
    expected = num_steps * (STATE_DIM + ACTION_DIM)
    if raw.size != expected:
        raise ValueError(f"{path}: states_actions.bin has {raw.size} values, expected {expected}")
    return metadata, raw.reshape(num_steps, STATE_DIM + ACTION_DIM), num_steps


def _build_v3_dataset(
    repo_id: str,
    output: str,
    episode_dirs: list[str],
    *,
    fps: int,
    image_size: int,
    cameras: list[str],
    rename: dict[str, str],
    fallback_size: tuple[int, int],
    encoder_threads: int,
    prefix: str,
) -> tuple[int, int]:
    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = _joint_names()
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": names},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": names},
    }
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        }
    dataset = LeRobotDataset.create(
        repo_id,
        fps=fps,
        features=features,
        root=output,
        vcodec="h264",
        streaming_encoding=True,
        encoder_threads=encoder_threads,
    )
    total_frames = 0
    for episode_number, episode_dir_value in enumerate(episode_dirs, 1):
        episode_dir = Path(episode_dir_value)
        metadata, table, num_steps = _episode_payload(episode_dir)
        timeline = prompt_timeline(metadata, num_steps)
        layout = _camera_layout(metadata, rename, cameras, fallback_size)
        frame_count = 0
        video_path = episode_dir / "combined_camera-images-rgb.mp4"
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                if frame_count >= num_steps:
                    raise ValueError(f"{episode_dir}: video contains more than {num_steps} frames")
                stacked = frame.to_ndarray(format="rgb24")
                item: dict[str, Any] = {
                    "observation.state": table[frame_count, :STATE_DIM].astype(np.float32),
                    "action": table[frame_count, STATE_DIM:].astype(np.float32),
                    "task": prompt_at(timeline, frame_count),
                }
                for _, canonical, row_start, row_end in layout:
                    item[f"observation.images.{canonical}"] = fit_square(
                        stacked[row_start:row_end], image_size
                    )
                dataset.add_frame(item)
                frame_count += 1
        if frame_count != num_steps:
            raise ValueError(f"{episode_dir}: decoded {frame_count} frames, expected {num_steps}")
        dataset.save_episode()
        total_frames += num_steps
        print(
            f"  {prefix}[{episode_number}/{len(episode_dirs)}] {episode_dir.name}: "
            f"{num_steps} frames, {len(timeline)} prompt segment(s)",
            flush=True,
        )
    dataset.finalize()
    return len(episode_dirs), total_frames


def _build_shard(
    index: int,
    episode_dirs: list[str],
    output: str,
    base_repo: str,
    settings: dict[str, Any],
) -> tuple[int, int, int]:
    shard = Path(output)
    if (shard / "meta" / "info.json").is_file():
        info = json.loads((shard / "meta" / "info.json").read_text())
        return index, int(info["total_episodes"]), int(info["total_frames"])
    if shard.exists():
        shutil.rmtree(shard)
    episodes, frames = _build_v3_dataset(
        f"{base_repo}_s{index:02d}",
        str(shard),
        episode_dirs,
        prefix=f"[shard {index:02d}] ",
        **settings,
    )
    return index, episodes, frames


def _request_fingerprint(
    name: str,
    episode_dirs: list[Path],
    raw_root: Path,
    settings: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    sources = []
    for path in episode_dirs:
        task_root = next(parent for parent in path.parents if parent.parent == raw_root)
        marker = json.loads((task_root / SOURCE_MARKER).read_text())
        sources.append({
            "path": path.relative_to(raw_root).as_posix(),
            "task_sha256": marker["sha256"],
        })
    request = {"format": "sbd_abc_sim_lerobot/v1", "dataset": name, "sources": sources, **settings}
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), request


def build_subset(
    name: str,
    episode_dirs: list[Path],
    final_dir: Path,
    raw_root: Path,
    *,
    fps: int,
    image_size: int,
    cameras: list[str],
    rename: dict[str, str],
    fallback_size: tuple[int, int],
    shards: int,
    encoder_threads: int,
    force: bool,
) -> None:
    settings = {
        "fps": fps,
        "image_size": image_size,
        "cameras": cameras,
        "rename": rename,
        "fallback_size": fallback_size,
        "encoder_threads": encoder_threads,
    }
    fingerprint, request = _request_fingerprint(name, episode_dirs, raw_root, settings)
    provenance_path = final_dir / "meta" / PROVENANCE_FILE
    if (final_dir / "meta" / "info.json").is_file() and not force:
        existing = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
        if existing.get("request_sha256") == fingerprint:
            print(f"[{name}] LeRobot v3 already matches request -> skip")
            return
        raise RuntimeError(f"[{name}] existing dataset has different provenance; rerun with FORCE=1")
    if final_dir.exists():
        if not force:
            raise RuntimeError(f"[{name}] incomplete output exists: {final_dir}; rerun with FORCE=1")
        shutil.rmtree(final_dir)

    shard_count = max(1, min(int(shards), len(episode_dirs)))
    bounds = np.linspace(0, len(episode_dirs), shard_count + 1, dtype=int)
    groups = [
        [str(path) for path in episode_dirs[bounds[index] : bounds[index + 1]]]
        for index in range(shard_count)
    ]
    shards_root = final_dir.parent / "_sim_v3_shards" / name
    request_path = shards_root / ".request.json"
    if request_path.is_file():
        previous = json.loads(request_path.read_text())
        if previous.get("request_sha256") != fingerprint:
            shutil.rmtree(shards_root)
    elif shards_root.exists():
        shutil.rmtree(shards_root)
    shards_root.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps({"request_sha256": fingerprint}, indent=2) + "\n")

    print(
        f"[{name}] {len(episode_dirs)} episodes -> {shard_count} shard(s), "
        f"{image_size}x{image_size}, cameras={cameras}"
    )
    results: dict[int, tuple[int, int]] = {}
    with ProcessPoolExecutor(max_workers=shard_count) as pool:
        futures = [
            pool.submit(
                _build_shard,
                index,
                groups[index],
                str(shards_root / f"shard_{index:02d}"),
                f"dohyeon/{name}",
                settings,
            )
            for index in range(shard_count)
        ]
        for future in as_completed(futures):
            index, episodes, frames = future.result()
            results[index] = (episodes, frames)
            print(f"[{name}] shard {index:02d}: {episodes} episodes / {frames} frames", flush=True)

    if sum(value[0] for value in results.values()) != len(episode_dirs):
        raise RuntimeError(f"[{name}] shard episode total mismatch")
    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(
        repo_ids=[f"dohyeon/{name}_s{index:02d}" for index in range(shard_count)],
        aggr_repo_id=f"dohyeon/{name}",
        roots=[shards_root / f"shard_{index:02d}" for index in range(shard_count)],
        aggr_root=final_dir,
    )
    info = json.loads((final_dir / "meta" / "info.json").read_text())
    expected_frames = sum(value[1] for value in results.values())
    if (int(info["total_episodes"]), int(info["total_frames"])) != (
        len(episode_dirs),
        expected_frames,
    ):
        raise RuntimeError(f"[{name}] aggregate totals do not match completed shards")
    request.update(
        {
            "request_sha256": fingerprint,
            "total_episodes": len(episode_dirs),
            "total_frames": expected_frames,
        }
    )
    provenance_path.write_text(json.dumps(request, indent=2) + "\n")
    if os.environ.get("KEEP_SHARDS", "") in {"", "0"}:
        shutil.rmtree(shards_root)
    print(f"[{name}] LeRobot v3 ready: {len(episode_dirs)} episodes / {expected_frames} frames")


def stage_stats(name: str, root: Path, config_path: Path, config: dict[str, Any]) -> None:
    download_dataset_dir = Path(__file__).resolve().parents[2]
    quantile_script = download_dataset_dir / "filtered_dataset" / "src" / "ensure_quantile_stats.py"
    subprocess.run(
        [sys.executable, str(quantile_script), "--config", str(config_path), "--dataset", name, "--root", str(root)],
        check=True,
    )
    relative_script = download_dataset_dir / "ABC_dataset" / "src" / "compute_relative_action_stats.py"
    excluded = ",".join(str(value) for value in config.get("relative_exclude_joints", []))
    subprocess.run(
        [
            sys.executable,
            str(relative_script),
            "--root",
            str(root),
            "--dataset",
            name,
            "--chunk-size",
            str(int(config.get("relative_chunk_size", 50))),
            "--exclude-joints",
            excluded,
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--only", default=os.environ.get("ABC_SIM_ONLY", ""))
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE", "") not in {"", "0"})
    parser.add_argument("--skip-stats", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    _setup_paths(config)
    root = dataset_root(config)
    manifest_path = root / "_sim_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}; run download_ABC_sim.sh")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"unexpected ABC Sim manifest format: {manifest.get('format')!r}")
    if int(manifest.get("fps", SOURCE_FPS)) != int(config.get("abc_sim_fps", SOURCE_FPS)):
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
            fps=int(config.get("abc_sim_fps", SOURCE_FPS)),
            image_size=int(config.get("abc_sim_image_size", 256)),
            cameras=[str(value) for value in config.get("abc_sim_cameras", ["top", "left_wrist", "right_wrist"])],
            rename={str(key): str(value) for key, value in config.get("abc_sim_camera_rename", {}).items()},
            fallback_size=(source_width, source_height),
            shards=int(config.get("v3_shards", 16)),
            encoder_threads=int(config.get("v3_encoder_threads", 2)),
            force=args.force,
        )
        if not args.skip_stats:
            stage_stats(name, root, args.config, config)


if __name__ == "__main__":
    main()
