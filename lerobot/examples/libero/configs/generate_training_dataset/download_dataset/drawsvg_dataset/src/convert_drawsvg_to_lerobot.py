#!/usr/bin/env python3
"""Convert completed DrawSVG episode NPZ files into canonical LeRobot v3."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from drawsvg_dataset_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    conversion_settings,
    load_config,
)


EPISODE_RE = re.compile(r"^episode(\d+)$")
REQUIRED_ARRAYS = {
    "rgb",
    "wrist_rgb",
    "tcp_position",
    "joint_position",
    "action",
    "timestamp",
}


@dataclass(frozen=True)
class EpisodeSource:
    group: str
    path: Path
    source_episode_index: int
    task: str
    frame_count: int
    job_id: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--include-group",
        action="append",
        dest="include_groups",
        default=None,
        help="Override YAML group selection; repeat for multiple groups.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Debug limit after validation.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate all selected inputs without writing.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing required metadata: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Metadata must be a JSON object: {path}")
    return value


def _episode_sort_key(path: Path) -> int:
    match = EPISODE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not an episode directory: {path}")
    return int(match.group(1))


def _validate_metadata(group: str, episode_dir: Path) -> EpisodeSource:
    index = _episode_sort_key(episode_dir)
    meta_path = episode_dir / "vla_episode.json"
    npz_path = episode_dir / "vla_episode.npz"
    metadata = _load_json(meta_path)

    trajectory_metadata_path = episode_dir / "trajectory.raw.json"
    trajectory_metadata = _load_json(trajectory_metadata_path)
    try:
        fit_to_content = trajectory_metadata["coordinate_transform"]["fit_to_content"]
    except (KeyError, TypeError):
        raise ValueError(
            f"Missing coordinate_transform.fit_to_content: {trajectory_metadata_path}"
        ) from None
    if fit_to_content is not False:
        raise ValueError(
            f"Episode still uses automatic content centering/scaling: {trajectory_metadata_path}"
        )

    if int(metadata.get("schema_version", 0)) < 2:
        raise ValueError(f"Wrist-ready schema_version >= 2 required: {meta_path}")
    if int(metadata.get("control_frequency_hz", 0)) != 20:
        raise ValueError(f"Expected 20 Hz source episode: {meta_path}")
    if metadata.get("episode_success") is not True:
        raise ValueError(f"Only completed successful episodes can be converted: {meta_path}")

    frame_count = int(metadata.get("frame_count", 0))
    if frame_count < 1:
        raise ValueError(f"frame_count must be positive: {meta_path}")
    features = metadata.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Missing features metadata: {meta_path}")
    expected_shapes = {
        "rgb": [256, 256, 3],
        "wrist_rgb": [256, 256, 3],
        "tcp_position": [3],
        "joint_position": [7],
        "action": [3],
        "timestamp": [],
    }
    for key, expected_shape in expected_shapes.items():
        spec = features.get(key)
        if not isinstance(spec, dict) or list(spec.get("shape", [])) != expected_shape:
            raise ValueError(
                f"Feature {key!r} must have per-frame shape {expected_shape}: {meta_path}"
            )

    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing VLA episode data: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as arrays:
        missing = REQUIRED_ARRAYS - set(arrays.files)
        if missing:
            raise ValueError(f"Missing arrays {sorted(missing)}: {npz_path}")
        # These arrays are small and make a cheap, strong time-axis preflight.
        tcp = arrays["tcp_position"]
        joints = arrays["joint_position"]
        action = arrays["action"]
        timestamp = arrays["timestamp"]
        expected = {
            "tcp_position": (frame_count, 3),
            "joint_position": (frame_count, 7),
            "action": (frame_count, 3),
            "timestamp": (frame_count,),
        }
        actual = {
            "tcp_position": tcp.shape,
            "joint_position": joints.shape,
            "action": action.shape,
            "timestamp": timestamp.shape,
        }
        if actual != expected:
            raise ValueError(f"Numeric array shapes disagree with metadata in {npz_path}: {actual}")
        if not all(np.issubdtype(value.dtype, np.number) for value in (tcp, joints, action, timestamp)):
            raise ValueError(f"Numeric arrays have non-numeric dtype: {npz_path}")
        if not all(np.isfinite(value).all() for value in (tcp, joints, action, timestamp)):
            raise ValueError(f"Numeric arrays contain NaN/Inf: {npz_path}")
        canonical_timestamps = np.arange(frame_count, dtype=np.float64) / 20.0
        if not np.allclose(timestamp, canonical_timestamps, rtol=0.0, atol=1e-6):
            raise ValueError(f"timestamp is not aligned to 20 Hz ticks: {npz_path}")

    task = str(metadata.get("task_instruction", "")).strip()
    if not task:
        raise ValueError(f"task_instruction is empty: {meta_path}")
    job_path = episode_dir / "job.meta.json"
    job_id = None
    if job_path.is_file():
        job_id = str(_load_json(job_path).get("job_id") or "") or None
    return EpisodeSource(group, episode_dir, index, task, frame_count, job_id)


def discover_episodes(source_root: Path, groups: list[str]) -> list[EpisodeSource]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"DrawSVG generated root not found: {source_root}")
    selected: list[EpisodeSource] = []
    errors: list[str] = []
    for group in groups:
        group_dir = source_root / group
        if not group_dir.is_dir():
            errors.append(f"group folder not found: {group_dir}")
            continue
        episode_dirs = sorted(
            (path for path in group_dir.iterdir() if path.is_dir() and EPISODE_RE.fullmatch(path.name)),
            key=_episode_sort_key,
        )
        if not episode_dirs:
            errors.append(f"group has no episodeNNN folders: {group_dir}")
            continue
        group_episodes: list[EpisodeSource] = []
        for episode_dir in episode_dirs:
            try:
                group_episodes.append(_validate_metadata(group, episode_dir))
            except (FileNotFoundError, ValueError) as error:
                errors.append(str(error))
        task_names = {episode.task for episode in group_episodes}
        if len(task_names) > 1:
            errors.append(f"group {group!r} contains multiple task instructions: {sorted(task_names)}")
        selected.extend(group_episodes)

    if errors:
        preview = "\n".join(f"  - {message}" for message in errors[:30])
        suffix = "" if len(errors) <= 30 else f"\n  ... and {len(errors) - 30} more"
        raise ValueError(
            f"DrawSVG preflight failed with {len(errors)} incomplete/invalid episode(s):\n"
            f"{preview}{suffix}"
        )
    if not selected:
        raise ValueError("Selected DrawSVG groups contain no convertible episodes")
    return selected


def features() -> dict[str, dict[str, Any]]:
    return {
        "observation.images.image": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (10,),
            "names": {
                "motors": ["tcp_x", "tcp_y", "tcp_z", *[f"joint_{i}" for i in range(7)]]
            },
        },
        "observation.states.tcp_position": {
            "dtype": "float32",
            "shape": (3,),
            "names": {"motors": ["x", "y", "z"]},
        },
        "observation.states.joint_state": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": [f"joint_{i}" for i in range(7)]},
        },
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": {"motors": ["delta_x", "delta_y", "delta_z"]},
        },
    }


def load_episode_arrays(episode: EpisodeSource) -> dict[str, np.ndarray]:
    path = episode.path / "vla_episode.npz"
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: np.ascontiguousarray(source[key]) for key in REQUIRED_ARRAYS}
    expected_image_shape = (episode.frame_count, 256, 256, 3)
    for key in ("rgb", "wrist_rgb"):
        if arrays[key].shape != expected_image_shape or arrays[key].dtype != np.uint8:
            raise ValueError(
                f"{key} must be uint8 {expected_image_shape}, got "
                f"{arrays[key].dtype} {arrays[key].shape}: {path}"
            )
    for key, width in (("tcp_position", 3), ("joint_position", 7), ("action", 3)):
        arrays[key] = np.ascontiguousarray(arrays[key], dtype=np.float32)
        if arrays[key].shape != (episode.frame_count, width):
            raise ValueError(f"{key} shape changed after preflight: {path}")
    return arrays


def validate_output(output_dir: Path, expected_episodes: int, expected_frames: int) -> None:
    info_path = output_dir / "meta" / "info.json"
    stats_path = output_dir / "meta" / "stats.json"
    tasks_path = output_dir / "meta" / "tasks.parquet"
    episode_files = list((output_dir / "meta" / "episodes").glob("**/*.parquet"))
    data_files = list((output_dir / "data").glob("**/*.parquet"))
    if not all(path.is_file() for path in (info_path, stats_path, tasks_path)):
        raise RuntimeError(f"Converted dataset metadata is incomplete: {output_dir}")
    if not episode_files or not data_files:
        raise RuntimeError(f"Converted dataset parquet files are incomplete: {output_dir}")
    info = _load_json(info_path)
    if info.get("codebase_version") != "v3.0" or int(info.get("fps", 0)) != 20:
        raise RuntimeError(f"Converted dataset is not canonical LeRobot v3 at 20 Hz: {info_path}")
    if int(info.get("total_episodes", -1)) != expected_episodes:
        raise RuntimeError(f"Episode count mismatch in {info_path}")
    if int(info.get("total_frames", -1)) != expected_frames:
        raise RuntimeError(f"Frame count mismatch in {info_path}")
    expected_features = features()
    actual_features = info.get("features", {})
    for key, spec in expected_features.items():
        actual = actual_features.get(key)
        if not isinstance(actual, dict):
            raise RuntimeError(f"Output is missing feature {key!r}: {info_path}")
        if actual.get("dtype") != spec["dtype"] or tuple(actual.get("shape", ())) != spec["shape"]:
            raise RuntimeError(f"Output feature contract mismatch for {key!r}: {actual}")
    for key in ("observation.images.image", "observation.images.wrist_image"):
        if not list((output_dir / "videos" / key).glob("**/*.mp4")):
            raise RuntimeError(f"Output has no encoded videos for {key!r}")


def write_manifest(output_dir: Path, episodes: list[EpisodeSource], source_root: Path) -> None:
    group_counts: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for output_index, episode in enumerate(episodes):
        group_counts[episode.group] = group_counts.get(episode.group, 0) + 1
        records.append(
            {
                "lerobot_episode_index": output_index,
                "group": episode.group,
                "source_episode": episode.path.name,
                "source_episode_index": episode.source_episode_index,
                "task": episode.task,
                "frames": episode.frame_count,
                "job_id": episode.job_id,
            }
        )
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "included_groups": list(group_counts),
        "group_episode_counts": group_counts,
        "total_episodes": len(episodes),
        "total_frames": sum(episode.frame_count for episode in episodes),
        "fps": 20,
        "alignment": "frame t is the pre-action observation paired with action[t]",
        "action_contract": {
            "shape": [3],
            "names": ["delta_x", "delta_y", "delta_z"],
            "meaning": "normalized pd_ee_target_delta_pos; 1.0 equals 0.1 m per axis",
        },
        "state_contract": {
            "shape": [10],
            "layout": ["tcp_position[3]", "joint_position[7]"],
        },
        "episodes": records,
    }
    path = output_dir / "meta" / "drawsvg" / "conversion_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    settings = conversion_settings(load_config(args.config))
    source_root = (args.source_root or settings["drawsvg_source_root"]).expanduser().resolve()
    output_root = (args.output_root or settings["drawsvg_output_root"]).expanduser().resolve()
    output_name = args.output_name or str(settings["drawsvg_output_name"])
    groups = args.include_groups or list(settings["drawsvg_include_groups"])

    if Path(output_name).name != output_name or output_name in {"", ".", ".."}:
        raise ValueError(f"output_name must be one folder name: {output_name!r}")
    episodes = discover_episodes(source_root, groups)
    total_available = len(episodes)
    if args.max_episodes is not None:
        if args.max_episodes < 1:
            raise ValueError("--max-episodes must be positive")
        episodes = episodes[: args.max_episodes]

    print("Convert DrawSVG to LeRobot v3", flush=True)
    print(f"  source       : {source_root}", flush=True)
    print(f"  groups       : {groups}", flush=True)
    print(f"  valid input  : {total_available} episodes", flush=True)
    print(f"  selected     : {len(episodes)} episodes / {sum(e.frame_count for e in episodes)} frames", flush=True)
    print(f"  output       : {output_root / output_name}", flush=True)
    if args.dry_run:
        print("DRY RUN OK: every selected group is complete and wrist-ready.", flush=True)
        return

    project_dir = Path(settings["project_root"])
    sys.path.insert(0, str(project_dir / "lerobot" / "src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import resolve_vcodec

    output_dir = output_root / output_name
    if source_root == output_dir or source_root in output_dir.parents or output_dir in source_root.parents:
        raise ValueError("DrawSVG source and converted output must not contain one another")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=f"dohyeon/{output_name}",
        fps=20,
        root=output_dir,
        robot_type="franka_drawsvg",
        features=features(),
        image_writer_threads=int(settings["drawsvg_image_writer_threads"]),
        image_writer_processes=int(settings["drawsvg_image_writer_processes"]),
        vcodec=resolve_vcodec(str(settings["drawsvg_vcodec"])),
        batch_encoding_size=1,
        streaming_encoding=False,
    )

    total_frames = 0
    for output_index, episode in enumerate(episodes):
        arrays = load_episode_arrays(episode)
        for frame_index in range(episode.frame_count):
            tcp = arrays["tcp_position"][frame_index]
            joints = arrays["joint_position"][frame_index]
            dataset.add_frame(
                {
                    "observation.images.image": arrays["rgb"][frame_index],
                    "observation.images.wrist_image": arrays["wrist_rgb"][frame_index],
                    "observation.state": np.concatenate((tcp, joints)).astype(np.float32, copy=False),
                    "observation.states.tcp_position": tcp,
                    "observation.states.joint_state": joints,
                    "action": arrays["action"][frame_index],
                    "task": episode.task,
                }
            )
        dataset.save_episode()
        total_frames += episode.frame_count
        print(
            f"  episode {output_index + 1:3d}/{len(episodes)}: "
            f"frames={episode.frame_count:4d} group={episode.group} source={episode.path.name}",
            flush=True,
        )

    dataset.finalize()
    write_manifest(output_dir, episodes, source_root)
    validate_output(output_dir, len(episodes), total_frames)
    print("DONE", flush=True)
    print(f"  episodes : {len(episodes)}", flush=True)
    print(f"  frames   : {total_frames}", flush=True)
    print(f"  output   : {output_dir}", flush=True)


if __name__ == "__main__":
    main()
