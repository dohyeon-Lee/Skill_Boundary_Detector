#!/usr/bin/env python3
"""Convert raw CALVIN into annotated-task or continuous-play LeRobot v3 datasets."""

from __future__ import annotations

import argparse
import bisect
import errno
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from calvin_dataset_config import DEFAULT_CONFIG_PATH, conversion_settings, load_config


REQUIRED_TIMESTEP_KEYS = {
    "actions",
    "rel_actions",
    "robot_obs",
    "scene_obs",
    "rgb_static",
    "rgb_gripper",
    "rgb_tactile",
    "depth_static",
    "depth_gripper",
    "depth_tactile",
}

ROBOT_OBS_NAMES = [
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_roll",
    "tcp_pitch",
    "tcp_yaw",
    "gripper_opening_width_m",
    *[f"arm_joint_{index}_rad" for index in range(7)],
    "gripper_command",
]
ACTION_RELATIVE_NAMES = [
    "delta_x_scaled",
    "delta_y_scaled",
    "delta_z_scaled",
    "delta_roll_scaled",
    "delta_pitch_scaled",
    "delta_yaw_scaled",
    "gripper",
]
ACTION_ABSOLUTE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
SCENE_OBS_NAMES = [
    "slider_joint",
    "drawer_joint",
    "button_joint",
    "switch_joint",
    "lightbulb_state",
    "led_state",
    *[f"{color}_block_{axis}" for color in ("red", "blue", "pink") for axis in ("x", "y", "z", "roll", "pitch", "yaw")],
]
STATE_PRESETS = {
    "robot_obs": (tuple(range(15)), ROBOT_OBS_NAMES),
    "tcp_pose_gripper": (
        (0, 1, 2, 3, 4, 5, 6, 14),
        ROBOT_OBS_NAMES[:7] + [ROBOT_OBS_NAMES[14]],
    ),
    "joint_gripper": (
        tuple(range(6, 15)),
        ROBOT_OBS_NAMES[6:15],
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_numpy_object(path: Path) -> Any:
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value


def load_annotations(source_dir: Path, annotation_folder: str) -> dict[str, Any]:
    path = source_dir / annotation_folder / "auto_lang_ann.npy"
    if not path.is_file():
        raise FileNotFoundError(f"CALVIN language annotation file not found: {path}")
    data = _load_numpy_object(path)
    if not isinstance(data, dict):
        raise ValueError(f"CALVIN annotation must contain a dictionary: {path}")
    try:
        language = data["language"]
        info = data["info"]
        annotations = np.asarray(language["ann"])
        task_ids = np.asarray(language["task"])
        embeddings = np.asarray(language["emb"])
        intervals = np.asarray(info["indx"], dtype=np.int64)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"CALVIN annotation has an unexpected structure: {path}") from exc
    count = len(annotations)
    if not (len(task_ids) == len(embeddings) == len(intervals) == count):
        raise ValueError(
            "CALVIN annotation arrays disagree in length: "
            f"ann={count}, task={len(task_ids)}, emb={len(embeddings)}, indx={len(intervals)}"
        )
    if intervals.shape != (count, 2):
        raise ValueError(f"CALVIN info.indx must have shape ({count}, 2), got {intervals.shape}")
    if count == 0:
        raise ValueError(f"CALVIN annotation contains no segments: {path}")
    return {
        "path": path,
        "raw": data,
        "annotations": annotations,
        "task_ids": task_ids,
        "embeddings": embeddings,
        "intervals": intervals,
    }


def load_play_recordings(source_dir: Path) -> list[tuple[int, int]]:
    """Load CALVIN's inclusive teleoperation recording boundaries.

    Official CALVIN archives do not store these rows in temporal order. Keep
    their original row order, but validate non-overlap on a sorted copy.
    """
    path = source_dir / "ep_start_end_ids.npy"
    if not path.is_file():
        raise FileNotFoundError(f"CALVIN play boundary file not found: {path}")
    intervals = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)
    if intervals.shape == (2,):
        intervals = intervals.reshape(1, 2)
    if intervals.ndim != 2 or intervals.shape[1] != 2 or len(intervals) == 0:
        raise ValueError(f"CALVIN ep_start_end_ids.npy must have shape (N, 2), got {intervals.shape}")

    recordings: list[tuple[int, int]] = []
    for index, pair in enumerate(intervals):
        start, end = map(int, pair)
        if end < start:
            raise ValueError(f"play recording {index} has descending interval [{start}, {end}]")
        recordings.append((start, end))

    ordered = sorted(enumerate(recordings), key=lambda item: item[1][0])
    for (previous_index, (_, previous_end)), (index, (start, _)) in zip(
        ordered, ordered[1:], strict=False
    ):
        if start <= previous_end:
            raise ValueError(
                "CALVIN play recording intervals overlap: "
                f"recording {index} starts at {start} before recording "
                f"{previous_index} ends at {previous_end}"
            )
    return recordings


def _merged_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError(f"cannot merge descending interval [{start}, {end}]")
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def subtract_intervals(
    recordings: list[tuple[int, int]],
    removed: list[tuple[int, int]],
) -> list[tuple[int, int, int, int, int]]:
    """Subtract inclusive intervals without joining across a removed gap.

    Returns ``(recording_index, segment_start, segment_end,
    recording_start, recording_end)`` for every remaining contiguous segment.
    """
    removed_merged = _merged_intervals(removed)
    segments: list[tuple[int, int, int, int, int]] = []
    for recording_index, (recording_start, recording_end) in enumerate(recordings):
        cursor = recording_start
        for removed_start, removed_end in removed_merged:
            if removed_end < cursor:
                continue
            if removed_start > recording_end:
                break
            if removed_start > cursor:
                segments.append(
                    (
                        recording_index,
                        cursor,
                        min(removed_start - 1, recording_end),
                        recording_start,
                        recording_end,
                    )
                )
            cursor = max(cursor, removed_end + 1)
            if cursor > recording_end:
                break
        if cursor <= recording_end:
            segments.append(
                (
                    recording_index,
                    cursor,
                    recording_end,
                    recording_start,
                    recording_end,
                )
            )
    return segments


def conversion_units(
    annotation: dict[str, Any],
    recordings: list[tuple[int, int]],
    mode: str,
    task_split: str,
    heldout_tasks: list[str],
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Select annotated episodes or construct contiguous language-free play episodes."""
    task_ids = [str(value) for value in annotation["task_ids"]]
    available_tasks = set(task_ids)
    heldout = set(heldout_tasks)
    unknown = sorted(heldout - available_tasks)
    if unknown:
        raise ValueError(
            f"calvin_heldout_tasks contains task IDs absent from this source split: {unknown}; "
            f"available={sorted(available_tasks)}"
        )

    heldout_intervals = [
        tuple(map(int, annotation["intervals"][index]))
        for index, task_id in enumerate(task_ids)
        if task_id in heldout
    ]
    if mode == "annotated":
        units: list[dict[str, Any]] = []
        for annotation_index, task_id in enumerate(task_ids):
            selected = (
                task_split == "all"
                or (task_split == "pretrain" and task_id not in heldout)
                or (task_split == "heldout" and task_id in heldout)
            )
            if not selected:
                continue
            start, end = map(int, annotation["intervals"][annotation_index])
            units.append(
                {
                    "kind": "annotation",
                    "source_unit_index": annotation_index,
                    "start": start,
                    "end": end,
                    "recording_start": None,
                    "recording_end": None,
                    "task_id": task_id,
                    "language": str(annotation["annotations"][annotation_index]),
                    "embedding": annotation["embeddings"][annotation_index],
                }
            )
        return units, _merged_intervals(heldout_intervals)

    removed = heldout_intervals if task_split == "pretrain" else []
    play_segments = subtract_intervals(recordings, removed)
    units = [
        {
            "kind": "play",
            "source_unit_index": recording_index,
            "start": start,
            "end": end,
            "recording_start": recording_start,
            "recording_end": recording_end,
            "task_id": "play",
            "language": "",
            "embedding": None,
        }
        for recording_index, start, end, recording_start, recording_end in play_segments
    ]
    return units, _merged_intervals(removed)


def policy_state(robot_obs: np.ndarray, preset: str) -> np.ndarray:
    array = np.asarray(robot_obs)
    if array.shape != (15,):
        raise ValueError(f"CALVIN robot_obs must have shape (15,), got {array.shape}")
    indices, _ = STATE_PRESETS[preset]
    return np.ascontiguousarray(array[list(indices)], dtype=np.float32)


def policy_action(timestep: dict[str, np.ndarray], representation: str) -> np.ndarray:
    source_key = "rel_actions" if representation == "relative" else "actions"
    array = np.asarray(timestep[source_key])
    if array.shape != (7,):
        raise ValueError(f"CALVIN {source_key} must have shape (7,), got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _resize_rgb(array: np.ndarray, image_size: int | str) -> np.ndarray:
    image = np.asarray(array)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected uint8 HWC RGB image, got dtype={image.dtype}, shape={image.shape}")
    if image_size == "native":
        return np.ascontiguousarray(image)
    size = int(image_size)
    if image.shape[:2] == (size, size):
        return np.ascontiguousarray(image)
    resized = Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def _read_timestep(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"CALVIN timestep file not found: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        missing = sorted(REQUIRED_TIMESTEP_KEYS - set(bundle.files))
        if missing:
            raise ValueError(f"CALVIN timestep {path} is missing fields: {missing}")
        return {key: np.array(bundle[key], copy=True) for key in bundle.files}


def _raw_inventory(timestep: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        key: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for key, value in sorted(timestep.items())
    }


def _scene_ranges(source_dir: Path) -> dict[str, tuple[int, int]]:
    path = source_dir / "scene_info.npy"
    data = _load_numpy_object(path)
    if not isinstance(data, dict):
        raise ValueError(f"CALVIN scene_info.npy must contain a dictionary: {path}")
    result: dict[str, tuple[int, int]] = {}
    for name, interval in data.items():
        pair = np.asarray(interval, dtype=np.int64).reshape(-1)
        if pair.shape != (2,):
            raise ValueError(f"invalid scene range for {name!r}: {interval!r}")
        result[str(name)] = (int(pair[0]), int(pair[1]))
    return result


def _environment_for_frame(frame_index: int, ranges: dict[str, tuple[int, int]]) -> str:
    matches = [name for name, (start, end) in ranges.items() if start <= frame_index <= end]
    if len(matches) != 1:
        raise ValueError(f"source frame {frame_index} belongs to {len(matches)} CALVIN scenes: {matches}")
    return matches[0]


def _find_values(node: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(node, dict):
        for child_key, child in node.items():
            if child_key == key:
                values.append(child)
            values.extend(_find_values(child, key))
    elif isinstance(node, list):
        for child in node:
            values.extend(_find_values(child, key))
    return values


def _validate_fps(source_dir: Path, expected_fps: int) -> None:
    merged = source_dir / ".hydra" / "merged_config.yaml"
    if not merged.is_file():
        raise FileNotFoundError(f"CALVIN collection config not found: {merged}")
    config = yaml.safe_load(merged.read_text(encoding="utf-8")) or {}
    recorded = [float(value) for value in _find_values(config, "record_fps")]
    if not recorded:
        recorded = [float(value) for value in _find_values(config, "control_freq")]
    if not recorded:
        raise ValueError(f"cannot determine CALVIN FPS from {merged}")
    if any(abs(value - expected_fps) > 1e-6 for value in recorded):
        raise ValueError(f"configured FPS {expected_fps} disagrees with source values {recorded}")


def _feature(dtype: str, shape: tuple[int, ...], names: Any = None) -> dict[str, Any]:
    return {"dtype": dtype, "shape": shape, "names": names}


def make_features(
    first_timestep: dict[str, np.ndarray],
    image_size: int | str,
    action_representation: str,
    state_preset: str,
    embedding_dim: int,
) -> dict[str, dict[str, Any]]:
    if image_size == "native":
        static_shape = tuple(first_timestep["rgb_static"].shape)
        wrist_shape = tuple(first_timestep["rgb_gripper"].shape)
    else:
        static_shape = wrist_shape = (int(image_size), int(image_size), 3)
    state_indices, state_names = STATE_PRESETS[state_preset]
    action_names = (
        ACTION_RELATIVE_NAMES if action_representation == "relative" else ACTION_ABSOLUTE_NAMES
    )
    return {
        "observation.images.image": _feature(
            "video", static_shape, ["height", "width", "channel"]
        ),
        "observation.images.wrist_image": _feature(
            "video", wrist_shape, ["height", "width", "channel"]
        ),
        "observation.state": _feature(
            "float32", (len(state_indices),), {"motors": list(state_names)}
        ),
        "action": _feature("float32", (7,), {"motors": action_names}),
        # Convenience copies retain the native float64 source values. Keys under
        # calvin.* are ignored by LeRobot's automatic policy-feature inference.
        "calvin.actions": _feature(
            str(first_timestep["actions"].dtype), (7,), {"motors": ACTION_ABSOLUTE_NAMES}
        ),
        "calvin.rel_actions": _feature(
            str(first_timestep["rel_actions"].dtype),
            (7,),
            {"motors": ACTION_RELATIVE_NAMES},
        ),
        "calvin.robot_obs": _feature(
            str(first_timestep["robot_obs"].dtype), (15,), {"motors": ROBOT_OBS_NAMES}
        ),
        "calvin.scene_obs": _feature(
            str(first_timestep["scene_obs"].dtype), (24,), {"state": SCENE_OBS_NAMES}
        ),
        "calvin.source_indices": _feature(
            "int64",
            (4,),
            ["source_frame_index", "source_unit_index", "segment_start", "segment_end"],
        ),
        "calvin.language_embedding": _feature("float32", (embedding_dim,), None),
        "calvin.task_id": _feature("string", (1,), None),
        "calvin.language_annotation": _feature("string", (1,), None),
        "calvin.source_split": _feature("string", (1,), None),
        "calvin.environment": _feature("string", (1,), None),
    }


def _copy_source_tree(
    source_dir: Path,
    output_dir: Path,
    mode: str,
    included_timestep_intervals: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    if mode == "none":
        return {"mode": "none", "destination": None, "files": 0, "linked": 0, "copied": 0}
    destination = output_dir / "calvin_source" / source_dir.name
    if destination.exists():
        raise FileExistsError(f"preserved CALVIN source destination already exists: {destination}")
    counters = {
        "files": 0,
        "linked": 0,
        "copied": 0,
        "copied_bytes": 0,
        "excluded_timestep_files": 0,
    }
    included = (
        None
        if included_timestep_intervals is None
        else _merged_intervals(included_timestep_intervals)
    )
    included_starts = [] if included is None else [start for start, _ in included]

    def include_timestep(frame_index: int) -> bool:
        if included is None:
            return True
        position = bisect.bisect_right(included_starts, frame_index) - 1
        return position >= 0 and frame_index <= included[position][1]

    def ignore(directory: str, names: list[str]) -> set[str]:
        if Path(directory).resolve() != source_dir.resolve() or included is None:
            return set()
        excluded: set[str] = set()
        for name in names:
            if not (name.startswith("episode_") and name.endswith(".npz")):
                continue
            frame_text = name[len("episode_") : -len(".npz")]
            if frame_text.isdigit() and not include_timestep(int(frame_text)):
                excluded.add(name)
        counters["excluded_timestep_files"] += len(excluded)
        return excluded

    def copy_function(source: str, target: str) -> str:
        source_path = Path(source)
        target_path = Path(target)
        counters["files"] += 1
        if mode == "hardlink":
            try:
                os.link(source_path, target_path)
                counters["linked"] += 1
                return str(target_path)
            except OSError as exc:
                if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK}:
                    raise
        shutil.copy2(source_path, target_path)
        counters["copied"] += 1
        counters["copied_bytes"] += source_path.stat().st_size
        return str(target_path)

    shutil.copytree(
        source_dir,
        destination,
        copy_function=copy_function,
        ignore=ignore,
    )
    return {
        "mode": mode,
        "scope": "full_source" if included is None else "converted_unit_frames_only",
        "included_timestep_intervals": (
            None if included is None else [list(interval) for interval in included]
        ),
        "destination": str(destination.relative_to(output_dir)),
        **counters,
    }


def _action_contract(action_representation: str) -> dict[str, Any]:
    if action_representation == "relative":
        return {
            "canonical_feature": "action",
            "source_key": "rel_actions",
            "representation": "relative_cartesian_delta",
            "coordinate_frame": "world",
            "orientation": "euler_xyz_delta",
            "position_scale": 50.0,
            "orientation_scale": 20.0,
            "clipped_range": [-1.0, 1.0],
            "gripper": {"close": -1.0, "open": 1.0},
        }
    return {
        "canonical_feature": "action",
        "source_key": "actions",
        "representation": "absolute_tcp_pose",
        "coordinate_frame": "world",
        "orientation": "euler_xyz",
        "gripper": {"close": -1.0, "open": 1.0},
    }


def _validate_output(output_dir: Path, repo_id: str, expected_episodes: int) -> None:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.feature_utils import dataset_to_policy_features

    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=output_dir)
    if metadata.total_episodes != expected_episodes:
        raise RuntimeError(
            f"written episode count mismatch: {metadata.total_episodes} != {expected_episodes}"
        )
    policy_features = dataset_to_policy_features(metadata.features)
    required = {
        "observation.images.image",
        "observation.images.wrist_image",
        "observation.state",
        "action",
    }
    missing = required - set(policy_features)
    if missing:
        raise RuntimeError(f"converted dataset lacks policy features: {sorted(missing)}")
    leaked = sorted(key for key in policy_features if key.startswith("calvin."))
    if leaked:
        raise RuntimeError(f"CALVIN preservation features leaked into policy inputs: {leaked}")


def _load_unit_plan(
    path: Path,
    settings: dict[str, Any],
    annotation: dict[str, Any],
    source_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[int, int]], dict[str, Any]]:
    """Load a provenance-rich custom unit plan produced by the CALVIN splitter."""
    plan_path = Path(path).expanduser().resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported CALVIN unit-plan schema: {payload.get('schema_version')!r}")
    planned_source = Path(str(payload.get("source_dir", ""))).expanduser().resolve()
    if planned_source != source_dir:
        raise ValueError(
            f"unit plan source_dir {planned_source} does not match converter source {source_dir}"
        )
    expected_sha = str(payload.get("annotation_sha256", ""))
    actual_sha = _sha256(Path(annotation["path"]))
    if expected_sha != actual_sha:
        raise ValueError(
            "unit plan was created from a different CALVIN annotation file: "
            f"expected {expected_sha}, actual {actual_sha}"
        )
    output_name = str(payload.get("output_name", "")).strip()
    if Path(output_name).name != output_name or not output_name.endswith("_full_full"):
        raise ValueError("unit plan output_name must be one *_full_full directory name")
    conversion_mode = str(payload.get("conversion_mode", "")).strip()
    if conversion_mode not in {"play", "annotated", "long_horizon"}:
        raise ValueError(
            "unit plan conversion_mode must be play, annotated, or long_horizon"
        )
    role = str(payload.get("dataset_role", "")).strip()
    if role not in {"play_pretrain", "language_pretrain", "heldout"}:
        raise ValueError(f"unsupported unit plan dataset_role: {role!r}")

    raw_units = payload.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ValueError("unit plan must contain at least one unit")
    units: list[dict[str, Any]] = []
    for plan_index, raw in enumerate(raw_units):
        if not isinstance(raw, dict):
            raise TypeError(f"unit plan units[{plan_index}] must be a mapping")
        unit = dict(raw)
        for key in ("kind", "source_unit_index", "start", "end", "task_id", "language"):
            if key not in unit:
                raise ValueError(f"unit plan units[{plan_index}] is missing {key!r}")
        unit["source_unit_index"] = int(unit["source_unit_index"])
        unit["start"] = int(unit["start"])
        unit["end"] = int(unit["end"])
        if unit["end"] < unit["start"]:
            raise ValueError(
                f"unit plan units[{plan_index}] has descending interval "
                f"[{unit['start']}, {unit['end']}]"
            )
        unit.setdefault("recording_start", None)
        unit.setdefault("recording_end", None)
        embedding_index = unit.pop("embedding_annotation_index", None)
        if embedding_index is None:
            unit["embedding"] = None
        else:
            embedding_index = int(embedding_index)
            if not 0 <= embedding_index < len(annotation["embeddings"]):
                raise ValueError(
                    f"unit plan units[{plan_index}] has invalid embedding annotation "
                    f"index {embedding_index}"
                )
            unit["embedding"] = annotation["embeddings"][embedding_index]
        units.append(unit)

    removed = [tuple(map(int, pair)) for pair in payload.get("removed_intervals", [])]
    resolved = dict(settings)
    output_root = Path(resolved["calvin_convert_output_root"]).resolve()
    resolved.update(
        {
            "calvin_convert_output_name": output_name,
            "calvin_convert_output_dir": output_root / output_name,
            "calvin_convert_repo_id": f"dohyeon/{output_name}",
            "calvin_convert_mode": conversion_mode,
            "calvin_task_split": role,
            "calvin_heldout_tasks": list(payload.get("selected_candidate_keys", [])),
            "calvin_convert_overwrite": bool(payload.get("overwrite", False)),
            # A task-disjoint full_full split must never inherit debug truncation.
            "calvin_convert_max_episodes": None,
            "calvin_convert_max_frames_per_episode": None,
        }
    )
    return resolved, units, removed, payload


def convert(config_path: Path, unit_plan_path: Path | None = None) -> Path:
    config = load_config(config_path)
    settings = conversion_settings(config)
    project_root = Path(str(config["project_root"])).expanduser().resolve()
    lerobot_src = project_root / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import resolve_vcodec

    source_root = Path(settings["calvin_convert_source_root"]).resolve()
    source_dir = Path(settings["calvin_convert_source_dir"]).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"CALVIN source split not found: {source_dir}\nRun ./download_calvin.sh first."
        )
    annotation = load_annotations(source_dir, str(settings["calvin_convert_annotation_folder"]))
    intervals = annotation["intervals"]
    unit_plan: dict[str, Any] | None = None
    if unit_plan_path is not None:
        settings, units, removed_intervals, unit_plan = _load_unit_plan(
            unit_plan_path, settings, annotation, source_dir
        )
        recordings = load_play_recordings(source_dir)
    else:
        recordings = (
            load_play_recordings(source_dir)
            if settings["calvin_convert_mode"] == "play"
            else []
        )
        units, removed_intervals = conversion_units(
            annotation,
            recordings,
            str(settings["calvin_convert_mode"]),
            str(settings["calvin_task_split"]),
            list(settings["calvin_heldout_tasks"]),
        )
    output_root = Path(settings["calvin_convert_output_root"]).resolve()
    output_dir = Path(settings["calvin_convert_output_dir"]).resolve()
    repo_id = str(settings["calvin_convert_repo_id"])
    if source_dir == output_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise ValueError("CALVIN source and converted output must not contain one another")
    if output_dir.exists():
        if not settings["calvin_convert_overwrite"]:
            raise FileExistsError(
                f"converted output already exists: {output_dir}; set calvin_convert_overwrite: true"
            )
        shutil.rmtree(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    max_episodes = settings["calvin_convert_max_episodes"]
    if max_episodes is not None:
        units = units[: int(max_episodes)]
    if not units:
        raise ValueError(
            "CALVIN conversion selected zero episodes; check calvin_convert_mode, "
            "calvin_task_split, and calvin_heldout_tasks"
        )
    first_index = int(units[0]["start"])
    first_timestep = _read_timestep(source_dir / f"episode_{first_index:07d}.npz")
    inventory = _raw_inventory(first_timestep)
    embedding_dim = int(np.asarray(annotation["embeddings"][0]).size)
    if embedding_dim < 1:
        raise ValueError("CALVIN language embedding is empty")
    _validate_fps(source_dir, int(settings["calvin_convert_fps"]))
    scene_ranges = _scene_ranges(source_dir)

    features = make_features(
        first_timestep,
        settings["calvin_convert_image_size"],
        str(settings["calvin_policy_action"]),
        str(settings["calvin_policy_state"]),
        embedding_dim,
    )
    vcodec = resolve_vcodec(str(settings["calvin_convert_vcodec"]))
    encoder_threads_raw = str(settings["calvin_convert_encoder_threads"])
    encoder_threads = int(encoder_threads_raw) if encoder_threads_raw else None

    print("Convert CALVIN to LeRobot v3", flush=True)
    print(f"  variant       : {settings['calvin_convert_variant']}", flush=True)
    print(f"  source split  : {source_dir}", flush=True)
    print(f"  mode          : {settings['calvin_convert_mode']}", flush=True)
    print(f"  task split    : {settings['calvin_task_split']}", flush=True)
    print(f"  held-out      : {list(settings['calvin_heldout_tasks'])}", flush=True)
    print(f"  episodes      : {len(units)}", flush=True)
    print(f"  output        : {output_dir}", flush=True)
    print(f"  action        : {settings['calvin_policy_action']}", flush=True)
    print(f"  state         : {settings['calvin_policy_state']}", flush=True)
    print(f"  preserve raw  : {settings['calvin_preserve_raw_mode']}", flush=True)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(settings["calvin_convert_fps"]),
        root=output_dir,
        robot_type="calvin_franka",
        features=features,
        image_writer_threads=int(settings["calvin_convert_image_writer_threads"]),
        image_writer_processes=int(settings["calvin_convert_image_writer_processes"]),
        vcodec=vcodec,
        encoder_threads=encoder_threads,
        batch_encoding_size=int(settings["calvin_convert_batch_encoding_size"]),
        streaming_encoding=bool(settings["calvin_convert_streaming_encoding"]),
        encoder_queue_maxsize=int(settings["calvin_convert_encoder_queue_maxsize"]),
    )

    records: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    total_frames = 0
    max_frames = settings["calvin_convert_max_frames_per_episode"]
    for episode_index, unit in enumerate(units):
        original_start = int(unit["start"])
        original_end = int(unit["end"])
        if original_end < original_start:
            raise ValueError(
                f"selected episode {episode_index} has descending interval "
                f"[{original_start}, {original_end}]"
            )
        effective_end = original_end
        if max_frames is not None:
            effective_end = min(effective_end, original_start + int(max_frames) - 1)
        language = str(unit["language"])
        task_id = str(unit["task_id"])
        if unit["embedding"] is None:
            embedding = np.zeros((embedding_dim,), dtype=np.float32)
        else:
            embedding = np.ascontiguousarray(
                np.asarray(unit["embedding"]).reshape(-1), dtype=np.float32
            )
        if embedding.shape != (embedding_dim,):
            raise ValueError(
                f"episode {episode_index} embedding shape changed to {embedding.shape}"
            )

        environment_names: set[str] = set()
        for source_index in range(original_start, effective_end + 1):
            source_path = source_dir / f"episode_{source_index:07d}.npz"
            timestep = _read_timestep(source_path)
            current_inventory = _raw_inventory(timestep)
            if current_inventory != inventory:
                raise ValueError(
                    f"CALVIN timestep schema changed at {source_path}: "
                    f"expected={inventory}, actual={current_inventory}"
                )
            environment = _environment_for_frame(source_index, scene_ranges)
            environment_names.add(environment)
            frame = {
                "observation.images.image": _resize_rgb(
                    timestep["rgb_static"], settings["calvin_convert_image_size"]
                ),
                "observation.images.wrist_image": _resize_rgb(
                    timestep["rgb_gripper"], settings["calvin_convert_image_size"]
                ),
                "observation.state": policy_state(
                    timestep["robot_obs"], str(settings["calvin_policy_state"])
                ),
                "action": policy_action(timestep, str(settings["calvin_policy_action"])),
                "calvin.actions": np.ascontiguousarray(timestep["actions"]),
                "calvin.rel_actions": np.ascontiguousarray(timestep["rel_actions"]),
                "calvin.robot_obs": np.ascontiguousarray(timestep["robot_obs"]),
                "calvin.scene_obs": np.ascontiguousarray(timestep["scene_obs"]),
                "calvin.source_indices": np.asarray(
                    [
                        source_index,
                        int(unit["source_unit_index"]),
                        original_start,
                        original_end,
                    ],
                    dtype=np.int64,
                ),
                "calvin.language_embedding": embedding,
                "calvin.task_id": task_id,
                "calvin.language_annotation": language,
                "calvin.source_split": str(settings["calvin_convert_split"]),
                "calvin.environment": environment,
                "task": language if language else "play",
            }
            dataset.add_frame(frame)
        dataset.save_episode()
        length = effective_end - original_start + 1
        records.append(
            {
                "lerobot_episode_index": episode_index,
                "source_kind": str(unit["kind"]),
                "source_unit_index": int(unit["source_unit_index"]),
                "annotation_index": (
                    int(unit["source_unit_index"])
                    if unit["kind"] == "annotation"
                    else None
                ),
                "recording_index": (
                    int(unit["source_unit_index"])
                    if unit["kind"] == "play"
                    else None
                ),
                "candidate_key": unit.get("candidate_key"),
                "candidate_occurrence_index": unit.get("candidate_occurrence_index"),
                "task_id": task_id,
                "language": language,
                "source_start": original_start,
                "source_end": original_end,
                "source_recording_start": unit["recording_start"],
                "source_recording_end": unit["recording_end"],
                "converted_end": effective_end,
                "converted_length": length,
                "environments": sorted(environment_names),
            }
        )
        task_counts[task_id] += 1
        total_frames += length
        print(
            f"  episode {episode_index + 1:3d}/{len(units)}: "
            f"frames={length:3d} task={task_id} | {language}",
            flush=True,
        )

    dataset.finalize()
    print("Preserving exact raw CALVIN source split...", flush=True)
    retained_timestep_intervals = (
        None
        if unit_plan is None
        else _merged_intervals(
            [(int(unit["start"]), int(unit["end"])) for unit in units]
        )
    )
    raw_retention = _copy_source_tree(
        source_dir,
        output_dir,
        str(settings["calvin_preserve_raw_mode"]),
        included_timestep_intervals=retained_timestep_intervals,
    )

    calvin_meta = output_dir / "meta" / "calvin"
    contract = {
        "schema_version": 1,
        "action": _action_contract(str(settings["calvin_policy_action"])),
        "state": {
            "canonical_feature": "observation.state",
            "source_key": "robot_obs",
            "preset": str(settings["calvin_policy_state"]),
            "source_indices": list(STATE_PRESETS[str(settings["calvin_policy_state"])][0]),
            "names": list(STATE_PRESETS[str(settings["calvin_policy_state"])][1]),
        },
        "images": {
            "observation.images.image": "rgb_static",
            "observation.images.wrist_image": "rgb_gripper",
            "output_size": settings["calvin_convert_image_size"],
            "orientation_transform": "none",
        },
        "fps": int(settings["calvin_convert_fps"]),
        "auxiliary_policy_visibility": "calvin.* features are intentionally ignored by policy inference",
    }
    _json_dump(calvin_meta / "action_state_contract.json", contract)
    _json_dump(calvin_meta / "raw_field_inventory.json", inventory)
    _json_dump(calvin_meta / "episodes.json", records)
    if settings["calvin_convert_mode"] != "play":
        _json_dump(calvin_meta / "annotations.json", records)
    if unit_plan is not None:
        _json_dump(calvin_meta / "unit_plan.json", unit_plan)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_variant": str(settings["calvin_convert_variant"]),
        "source_root": str(source_root),
        "source_split": str(settings["calvin_convert_split"]),
        "conversion_mode": str(settings["calvin_convert_mode"]),
        "task_split": str(settings["calvin_task_split"]),
        "heldout_tasks": list(settings["calvin_heldout_tasks"]),
        "annotation_file": str(annotation["path"]),
        "annotation_sha256": _sha256(annotation["path"]),
        "annotation_folder": str(settings["calvin_convert_annotation_folder"]),
        "annotation_boundaries": "inclusive [start, end] with margin=0",
        "converted_episodes": len(units),
        "converted_annotations": (
            len(units) if settings["calvin_convert_mode"] != "play" else 0
        ),
        "source_annotations": len(intervals),
        "source_play_recordings": (
            len(recordings) if settings["calvin_convert_mode"] == "play" else None
        ),
        "removed_heldout_intervals": [list(interval) for interval in removed_intervals],
        "removed_interval_margin": 0,
        "converted_frames": total_frames,
        "task_segment_counts": dict(sorted(task_counts.items())),
        "raw_source_retention": raw_retention,
        "raw_field_inventory": "raw_field_inventory.json",
        "action_state_contract": "action_state_contract.json",
    }
    if unit_plan is not None:
        manifest["custom_unit_plan"] = "unit_plan.json"
        manifest["custom_unit_plan_sha256"] = _sha256(Path(unit_plan_path).resolve())
        manifest["selected_candidate_keys"] = list(
            unit_plan.get("selected_candidate_keys", [])
        )
        manifest["candidate_report"] = unit_plan.get("candidate_report")
        manifest["candidate_report_sha256"] = unit_plan.get("candidate_report_sha256")
    completion_marker = source_root / ".calvin_download_complete.json"
    if completion_marker.is_file():
        shutil.copy2(completion_marker, calvin_meta / "source_download_complete.json")
        manifest["source_download_marker"] = "source_download_complete.json"
    _json_dump(calvin_meta / "conversion_manifest.json", manifest)
    _validate_output(output_dir, repo_id, len(units))

    print("DONE", flush=True)
    print(f"  LeRobot       : v3.0", flush=True)
    print(f"  episodes      : {len(units)}", flush=True)
    print(f"  frames        : {total_frames}", flush=True)
    print(f"  raw files     : {raw_retention['files']} ({raw_retention['mode']})", flush=True)
    print(f"  output        : {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--unit-plan", type=Path)
    args = parser.parse_args()
    try:
        convert(args.config, args.unit_plan)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
