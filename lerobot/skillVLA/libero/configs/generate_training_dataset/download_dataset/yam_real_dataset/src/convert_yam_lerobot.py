#!/usr/bin/env python3
"""Convert an i2rt YAM recorder LeRobot v3 dataset to canonical Pi0.5 v3.

The source dataset is never modified. The conversion performs only the explicit
YAM canonicalization steps:

* 42D recorder state -> 14D follower joint positions
* physical camera keys -> top/left_wrist/right_wrist
* optional outcome/episode filtering
* exact global state/action quantile statistics

Homing frames are intentionally preserved as part of each demonstration.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from yam_real_dataset_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    dataset_sets,
    load_config,
    output_root,
    project_root,
    raw_root,
)

QUANTILES = (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99))
DEFAULT_CAMERA_MAP = {
    "agentview": "top",
    "wrist_left": "left_wrist",
    "wrist_right": "right_wrist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--set", dest="set_name", required=True, help="Entry under yaml yam_sets")
    parser.add_argument("--source", type=Path, default=None, help="Override source dataset directory")
    parser.add_argument("--output", type=Path, default=None, help="Override output dataset directory")
    parser.add_argument("--repo-id", default=None, help="Output LeRobot repo id")
    parser.add_argument("--max-episodes", type=int, default=None, help="Debug conversion limit")
    parser.add_argument("--include-outcome", action="append", default=None, help="Override outcome filter")
    parser.add_argument("--overwrite", action="store_true", help="Replace only the resolved output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print selected episodes only")
    return parser.parse_args()


def setting(config: dict[str, Any], entry: dict[str, Any], key: str, default: Any = None) -> Any:
    return entry[key] if key in entry else config.get(key, default)


def canonical_camera_map(value: object) -> dict[str, str]:
    raw = dict(value or DEFAULT_CAMERA_MAP)
    result: dict[str, str] = {}
    for source, target in raw.items():
        source_key = str(source)
        target_key = str(target)
        if not source_key.startswith("observation.images."):
            source_key = f"observation.images.{source_key}"
        if not target_key.startswith("observation.images."):
            target_key = f"observation.images.{target_key}"
        result[source_key] = target_key
    if len(result) != 3 or len(set(result.values())) != 3:
        raise ValueError(f"camera_map must define three unique source/target cameras, got {result}")
    return result


def yam_joint_names() -> list[str]:
    names: list[str] = []
    for side in ("left", "right"):
        names.extend([f"{side}_joint_{index}" for index in range(6)])
        names.append(f"{side}_gripper")
    return names


def recorder_position_indices(state_feature: dict[str, Any]) -> list[int]:
    """Resolve [left.pos.0..6, right.pos.0..6], with a guarded 42D fallback."""

    shape = tuple(int(value) for value in state_feature.get("shape", ()))
    names = state_feature.get("names")
    expected = [f"{arm}.pos.{index}" for arm in ("left", "right") for index in range(7)]
    if isinstance(names, list) and all(name in names for name in expected):
        indices = [names.index(name) for name in expected]
        if len(set(indices)) != 14:
            raise ValueError(f"Recorder state position names are not unique: {indices}")
        return indices
    if shape == (42,):
        print("[schema] WARNING: state names missing; using recorder v1 42D positional fallback [0:7,21:28]")
        return [*range(0, 7), *range(21, 28)]
    raise ValueError(
        "Cannot identify YAM joint positions. Expected named recorder 42D state "
        f"or shape (42,), got shape={shape}, names={names!r}"
    )


def validate_source_schema(
    info: dict[str, Any], state_key: str, action_key: str, camera_map: dict[str, str]
) -> tuple[list[int], int]:
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Source must be LeRobot v3.0, got {info.get('codebase_version')!r}")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("Source meta/info.json has no features mapping")
    missing = [key for key in (state_key, action_key, *camera_map) if key not in features]
    if missing:
        raise ValueError(f"Source dataset is missing required recorder features: {missing}")
    if tuple(features[action_key].get("shape", ())) != (14,):
        raise ValueError(f"Source action must be 14D, got {features[action_key].get('shape')}")
    for key in camera_map:
        feature = features[key]
        if feature.get("dtype") != "video":
            raise ValueError(f"Source camera {key!r} must be video-backed, got {feature.get('dtype')!r}")
        shape = tuple(feature.get("shape", ()))
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"Source camera {key!r} must be HWC RGB, got {shape}")
    fps = int(info.get("fps", 0))
    if fps <= 0:
        raise ValueError(f"Source fps must be positive, got {fps}")
    return recorder_position_indices(features[state_key]), fps


def load_outcomes(source: Path) -> dict[int, dict[str, Any]]:
    path = source / "outcomes.jsonl"
    if not path.exists():
        return {}
    result: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            index = int(entry["episode"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid outcomes.jsonl line {line_number}: {line!r}") from error
        if index in result:
            raise ValueError(f"Duplicate outcome entry for source episode {index}")
        result[index] = entry
    return result


def select_episode_rows(
    rows: list[dict[str, Any]],
    outcomes: dict[int, dict[str, Any]],
    *,
    include_outcomes: set[str],
    require_outcomes: bool,
    min_frames: int,
    include_indices: set[int],
    exclude_indices: set[int],
    max_episodes: int | None,
) -> tuple[list[dict[str, Any]], Counter]:
    selected: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for row in sorted(rows, key=lambda item: int(item["episode_index"])):
        episode = int(row["episode_index"])
        length = int(row.get("length", 0))
        if include_indices and episode not in include_indices:
            skipped["not_in_include_indices"] += 1
            continue
        if episode in exclude_indices:
            skipped["excluded_index"] += 1
            continue
        if length < min_frames:
            skipped["too_short"] += 1
            continue
        outcome_entry = outcomes.get(episode)
        if require_outcomes and outcome_entry is None:
            raise ValueError(f"Source episode {episode} has no outcomes.jsonl entry")
        outcome = None if outcome_entry is None else outcome_entry.get("outcome")
        if include_outcomes and str(outcome) not in include_outcomes:
            skipped[f"outcome={outcome}"] += 1
            continue
        selected.append(row)
        if max_episodes is not None and len(selected) >= max_episodes:
            break
    return selected, skipped


def read_parquet_frames(source: Path, episode_indices: list[int], columns: list[str]):
    import pyarrow.dataset as pyarrow_dataset

    files = sorted((source / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No source data parquet files under {source / 'data'}")
    dataset = pyarrow_dataset.dataset([str(path) for path in files], format="parquet")
    table = dataset.to_table(
        columns=columns,
        filter=pyarrow_dataset.field("episode_index").isin(episode_indices),
    )
    return table.to_pandas().sort_values(["episode_index", "frame_index"])


def task_index_map(source: Path) -> dict[int, str]:
    import pandas as pd

    path = source / "meta" / "tasks.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Source tasks metadata is missing: {path}")
    frame = pd.read_parquet(path)
    result: dict[int, str] = {}
    for index_value, row in frame.iterrows():
        if "task_index" not in row:
            raise ValueError(f"Source tasks metadata has no task_index column: {path}")
        task = row.get("task", index_value)
        result[int(row["task_index"])] = str(task)
    return result


def read_episode_rows(source: Path) -> list[dict[str, Any]]:
    import pandas as pd

    files = sorted((source / "meta" / "episodes").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet files under {source / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True).to_dict("records")


def resize_with_pad(image: np.ndarray, image_size: int | None) -> np.ndarray:
    value = np.asarray(image)
    if value.dtype != np.uint8:
        value = np.clip(value * 255.0 if np.issubdtype(value.dtype, np.floating) else value, 0, 255).astype(
            np.uint8
        )
    if image_size is None or value.shape[:2] == (image_size, image_size):
        return np.ascontiguousarray(value)
    height, width = value.shape[:2]
    ratio = min(image_size / width, image_size / height)
    resized_width = max(1, round(width * ratio))
    resized_height = max(1, round(height * ratio))
    resized = np.asarray(
        Image.fromarray(value).resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    )
    output = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output


def output_features(
    source_info: dict[str, Any], camera_map: dict[str, str], image_size: int | None
) -> dict[str, dict[str, Any]]:
    names = yam_joint_names()
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": names},
        "action": {"dtype": "float32", "shape": (14,), "names": names},
    }
    source_features = source_info["features"]
    sizes: set[tuple[int, int]] = set()
    for source_key, target_key in camera_map.items():
        source_shape = tuple(int(value) for value in source_features[source_key]["shape"])
        shape = (image_size, image_size, 3) if image_size is not None else source_shape
        sizes.add((shape[0], shape[1]))
        features[target_key] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }
    if len(sizes) != 1:
        raise ValueError(f"Canonical Pi0.5 cameras must share one HxW shape, got {sizes}")
    return features


def decode_episode_videos(
    source: Path,
    source_info: dict[str, Any],
    row: dict[str, Any],
    camera_keys: tuple[str, ...],
    source_fps: int,
) -> dict[str, np.ndarray]:
    import torch

    from lerobot.datasets.video_utils import decode_episode_video_frames

    length = int(row["length"])
    template = source_info.get("video_path")
    if not isinstance(template, str):
        raise ValueError("Source info.json has no video_path template")
    videos: dict[str, np.ndarray] = {}
    for key in camera_keys:
        chunk = int(row[f"videos/{key}/chunk_index"])
        file_index = int(row[f"videos/{key}/file_index"])
        from_timestamp = float(row[f"videos/{key}/from_timestamp"])
        to_timestamp = float(row[f"videos/{key}/to_timestamp"])
        path = source / template.format(video_key=key, chunk_index=chunk, file_index=file_index)
        frames = decode_episode_video_frames(
            path,
            from_timestamp,
            to_timestamp,
            length,
            source_fps,
            backend="pyav",
            decoder_num_threads=None,
        )
        videos[key] = (frames.permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return videos


def exact_feature_stats(dataset_dir: Path, feature_keys: tuple[str, ...]) -> None:
    import pandas as pd

    files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    frame = pd.concat(
        [pd.read_parquet(path, columns=list(feature_keys)) for path in files], ignore_index=True
    )
    stats_path = dataset_dir / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for key in feature_keys:
        values = np.stack(frame[key].to_numpy()).astype(np.float64)
        block: dict[str, object] = {
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
            "count": np.asarray([values.shape[0]], dtype=np.int64),
        }
        for name, quantile in QUANTILES:
            block[name] = np.quantile(values, quantile, axis=0)
        stats[key] = {
            name: [int(value.reshape(-1)[0])] if name == "count" else value.reshape(-1).astype(float).tolist()
            for name, raw in block.items()
            for value in [np.asarray(raw)]
        }
    stats_path.write_text(json.dumps(stats, indent=4))


def validate_output(
    output: Path,
    *,
    fps: int,
    episodes: int,
    frames: int,
    expected_features: dict[str, dict[str, Any]],
) -> None:
    info_path = output / "meta" / "info.json"
    stats_path = output / "meta" / "stats.json"
    if not info_path.exists() or not stats_path.exists():
        raise RuntimeError(f"Canonical output is missing LeRobot v3 metadata: {output}")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0" or int(info.get("fps", 0)) != fps:
        raise RuntimeError(f"Invalid output version/fps: {info.get('codebase_version')}, {info.get('fps')}")
    if int(info.get("total_episodes", -1)) != episodes or int(info.get("total_frames", -1)) != frames:
        raise RuntimeError(
            f"Output totals mismatch: got {info.get('total_episodes')} eps/{info.get('total_frames')} frames, "
            f"expected {episodes}/{frames}"
        )
    for key, expected in expected_features.items():
        actual = info.get("features", {}).get(key)
        if actual is None:
            raise RuntimeError(f"Output is missing feature {key!r}")
        if actual.get("dtype") != expected["dtype"] or tuple(actual.get("shape", ())) != tuple(
            expected["shape"]
        ):
            raise RuntimeError(f"Output feature mismatch for {key}: actual={actual}, expected={expected}")
    stats = json.loads(stats_path.read_text())
    for key in ("observation.state", "action"):
        missing = [name for name, _ in QUANTILES if name not in stats.get(key, {})]
        if missing:
            raise RuntimeError(f"Output stats for {key!r} are missing quantiles {missing}")


def resolve_dataset_paths(
    args: argparse.Namespace, config: dict[str, Any], entry: dict[str, Any]
) -> tuple[Path, Path, str]:
    source_value = args.source if args.source is not None else entry.get("source")
    output_value = args.output if args.output is not None else entry.get("output", args.set_name)
    if source_value is None:
        raise ValueError(f"yam_sets.{args.set_name}.source is required")
    source = (
        Path(source_value).expanduser().resolve()
        if args.source is not None or Path(str(source_value)).is_absolute()
        else (raw_root(config) / str(source_value)).resolve()
    )
    output = (
        Path(output_value).expanduser().resolve()
        if args.output is not None or Path(str(output_value)).is_absolute()
        else (output_root(config) / str(output_value)).resolve()
    )
    repo_id = args.repo_id or f"{config.get('yam_repo_prefix', 'local')}/{output.name}"
    return source, output, repo_id


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sets = dataset_sets(config)
    if args.set_name not in sets:
        raise KeyError(f"Unknown --set {args.set_name!r}; choices: {sorted(sets)}")
    entry = sets[args.set_name]
    source, output, repo_id = resolve_dataset_paths(args, config, entry)
    if source == output:
        raise ValueError("Source and output dataset directories must be different")

    info_path = source / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Raw YAM LeRobot dataset not found: {info_path}")
    source_info = json.loads(info_path.read_text())
    state_key = str(setting(config, entry, "source_state_key", "observation.state"))
    action_key = str(setting(config, entry, "source_action_key", "action"))
    camera_map = canonical_camera_map(setting(config, entry, "camera_map", DEFAULT_CAMERA_MAP))
    position_indices, source_fps = validate_source_schema(source_info, state_key, action_key, camera_map)
    expected_fps = setting(config, entry, "expected_fps")
    if expected_fps is not None and int(expected_fps) != source_fps:
        raise ValueError(f"Source fps={source_fps}, configured expected_fps={expected_fps}")

    outcomes = load_outcomes(source)
    include_outcomes_value = args.include_outcome
    if include_outcomes_value is None:
        include_outcomes_value = setting(config, entry, "include_outcomes", []) or []
    include_outcomes = {str(value) for value in include_outcomes_value}
    require_outcomes = bool(setting(config, entry, "require_outcomes", bool(include_outcomes)))
    if require_outcomes and not outcomes:
        raise FileNotFoundError(f"Outcome filtering requires {source / 'outcomes.jsonl'}")

    rows = read_episode_rows(source)
    selected, skipped = select_episode_rows(
        rows,
        outcomes,
        include_outcomes=include_outcomes,
        require_outcomes=require_outcomes,
        min_frames=int(setting(config, entry, "min_episode_frames", 1)),
        include_indices={int(value) for value in setting(config, entry, "include_episode_indices", []) or []},
        exclude_indices={int(value) for value in setting(config, entry, "exclude_episode_indices", []) or []},
        max_episodes=args.max_episodes,
    )
    if not selected:
        raise ValueError(f"No episodes selected; skipped={dict(skipped)}")

    image_size_value = setting(config, entry, "image_size", 256)
    image_size = None if image_size_value is None else int(image_size_value)
    features = output_features(source_info, camera_map, image_size)
    total_input_frames = sum(int(row["length"]) for row in selected)
    print("Convert i2rt YAM recorder dataset -> Pi0.5 canonical LeRobot v3")
    print(f"  source        : {source}")
    print(f"  output        : {output}")
    print(f"  repo id       : {repo_id}")
    print(f"  fps           : {source_fps} (preserved)")
    print(f"  episodes      : {len(selected)}/{len(rows)} selected; skipped={dict(skipped)}")
    print(f"  frames        : {total_input_frames}")
    print(f"  outcomes      : {sorted(include_outcomes) if include_outcomes else 'ALL'}")
    print(f"  state         : {state_key} 42D -> observation.state 14D indices={position_indices}")
    print(f"  action        : {action_key} 14D absolute -> action 14D")
    print(f"  cameras       : {camera_map}")
    print(f"  image size    : {image_size or 'source'}")
    print("  homing frames : preserved (no trimming)")
    if args.dry_run:
        print("DRY RUN complete; no output written")
        return

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
        output_root_path = output_root(config)
        paths_overlap = source in output.parents or output in source.parents
        if output in (Path("/"), project_root(config), output_root_path) or paths_overlap:
            raise ValueError(f"Refusing unsafe output deletion: {output}")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    lerobot_src = project_root(config) / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import resolve_vcodec

    resolved_vcodec = resolve_vcodec(str(setting(config, entry, "vcodec", "h264")))
    streaming = bool(setting(config, entry, "streaming_encoding", True))
    create_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "fps": source_fps,
        "root": output,
        "robot_type": "yam_bimanual",
        "features": features,
        "vcodec": resolved_vcodec,
        "streaming_encoding": streaming,
        "encoder_threads": int(setting(config, entry, "encoder_threads", 2)),
    }
    if not streaming:
        create_kwargs.update(
            image_writer_threads=int(setting(config, entry, "image_writer_threads", 6)),
            image_writer_processes=int(setting(config, entry, "image_writer_processes", 0)),
        )
    dataset = LeRobotDataset.create(**create_kwargs)

    selected_indices = [int(row["episode_index"]) for row in selected]
    frame_data = read_parquet_frames(
        source,
        selected_indices,
        ["episode_index", "frame_index", "task_index", state_key, action_key],
    )
    frames_by_episode = {int(index): group for index, group in frame_data.groupby("episode_index")}
    tasks = task_index_map(source)
    camera_keys = tuple(camera_map)
    prefetch = max(1, int(setting(config, entry, "decode_prefetch", 2)))
    manifest: list[dict[str, Any]] = []
    total_frames = 0

    with ThreadPoolExecutor(max_workers=prefetch) as pool:
        futures: dict[int, Future] = {}
        for position in range(min(prefetch, len(selected))):
            futures[position] = pool.submit(
                decode_episode_videos, source, source_info, selected[position], camera_keys, source_fps
            )
        for position, row in enumerate(selected):
            videos = futures.pop(position).result()
            next_position = position + prefetch
            if next_position < len(selected):
                futures[next_position] = pool.submit(
                    decode_episode_videos,
                    source,
                    source_info,
                    selected[next_position],
                    camera_keys,
                    source_fps,
                )

            source_episode = int(row["episode_index"])
            group = frames_by_episode.get(source_episode)
            length = int(row["length"])
            if group is None or len(group) != length:
                actual = 0 if group is None else len(group)
                raise ValueError(
                    f"Source episode {source_episode}: parquet rows {actual} != metadata length {length}"
                )
            raw_states = np.stack(group[state_key].to_numpy()).astype(np.float32)
            states = raw_states[:, position_indices]
            actions = np.stack(group[action_key].to_numpy()).astype(np.float32)
            if states.shape != (length, 14) or actions.shape != (length, 14):
                raise ValueError(
                    f"Source episode {source_episode}: transformed shapes state={states.shape}, action={actions.shape}"
                )
            if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
                raise ValueError(f"Source episode {source_episode} contains NaN or infinity in state/action")
            task_indices = {int(value) for value in group["task_index"].to_numpy()}
            if len(task_indices) != 1:
                raise ValueError(f"Source episode {source_episode} has multiple task indices: {task_indices}")
            task_index = next(iter(task_indices))
            if task_index not in tasks:
                raise ValueError(
                    f"Source episode {source_episode} references unknown task_index={task_index}"
                )
            task = tasks[task_index]
            for source_key, images in videos.items():
                if images.shape[0] != length:
                    raise ValueError(
                        f"Source episode {source_episode} camera {source_key} length mismatch: {images.shape}"
                    )

            for frame_index in range(length):
                frame: dict[str, Any] = {
                    "observation.state": states[frame_index],
                    "action": actions[frame_index],
                    "task": task,
                }
                for source_key, target_key in camera_map.items():
                    frame[target_key] = resize_with_pad(videos[source_key][frame_index], image_size)
                dataset.add_frame(frame)
            dataset.save_episode()
            outcome_entry = outcomes.get(source_episode, {})
            manifest.append(
                {
                    "episode": position,
                    "source_episode": source_episode,
                    "outcome": outcome_entry.get("outcome"),
                    "task": task,
                    "frames": length,
                }
            )
            total_frames += length
            print(
                f"  [{position + 1}/{len(selected)}] source_ep={source_episode} -> output_ep={position}: "
                f"{length} frames | {task}",
                flush=True,
            )

    dataset.finalize()
    if bool(setting(config, entry, "exact_quantile_stats", True)):
        exact_feature_stats(output, ("observation.state", "action"))
    (output / "meta" / "yam_conversion.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_repo_id": source_info.get("repo_id"),
                "state_position_indices": position_indices,
                "camera_map": camera_map,
                "include_outcomes": sorted(include_outcomes),
                "homing_trimmed": False,
                "episodes": manifest,
            },
            indent=2,
        )
    )
    with open(output / "outcomes.jsonl", "w", encoding="utf-8") as stream:
        for item in manifest:
            stream.write(json.dumps(item) + "\n")
    validate_output(
        output,
        fps=source_fps,
        episodes=len(selected),
        frames=total_frames,
        expected_features=features,
    )
    print("DONE")
    print(f"  canonical v3 : {output}")
    print(f"  episodes/frames: {len(selected)}/{total_frames}")
    print("  state/action quantiles: verified")


if __name__ == "__main__":
    main()
