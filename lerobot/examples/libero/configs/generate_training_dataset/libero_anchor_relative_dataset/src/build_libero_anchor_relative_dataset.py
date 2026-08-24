#!/usr/bin/env python3
"""Build a LeRobot-v3 LIBERO dataset with absolute commanded EEF targets.

The canonical LIBERO HDF5 replay stores ``row t = (observation after action[t],
action[t])``.  A correctly aligned behavior-cloning row is therefore formed as:

    output observation[t] = source observation[t]
    output action[t]      = absolute_target(source action[t+1], observation[t])

The final source row has no next action and is dropped.  Images and proprioception
are otherwise preserved.  No simulator replay is required.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from compute_relative_action_stats import compute_and_write  # noqa: E402
from libero_anchor_relative_dataset_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    dataset_root,
    dataset_specs,
    load_config,
    project_root,
)

ACTION_NAMES = {
    "motors": [
        "target_x",
        "target_y",
        "target_z",
        "target_axis_angle1",
        "target_axis_angle2",
        "target_axis_angle3",
        "gripper",
    ]
}


def _normalize_feature(spec: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in spec.items() if key in {"dtype", "shape", "names"}}
    if isinstance(out.get("shape"), list):
        out["shape"] = tuple(out["shape"])
    return out


def output_features(source_info: dict[str, Any]) -> dict[str, Any]:
    source = source_info.get("features") or {}
    required = {"observation.state", "action"}
    missing = required - set(source)
    if missing:
        raise ValueError(f"Source dataset is missing required features: {sorted(missing)}")
    if tuple(source["observation.state"].get("shape", ())) != (8,):
        raise ValueError("LIBERO anchor-relative conversion requires 8D EEF+gripper observation.state")
    if tuple(source["action"].get("shape", ())) != (7,):
        raise ValueError("LIBERO anchor-relative conversion requires the canonical 7D OSC action")
    emitted: dict[str, Any] = {}
    for key, spec in source.items():
        if (
            spec.get("dtype") == "video"
            or key == "observation.state"
            or key.startswith("observation.states.")
        ):
            emitted[key] = _normalize_feature(spec)
    emitted["action"] = {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES}
    return emitted


def derive_aligned_absolute_targets(
    states: np.ndarray,
    raw_actions: np.ndarray,
    *,
    position_scale: float,
    rotation_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(kept_states, absolute_targets)`` for one contiguous episode."""
    states = np.asarray(states, dtype=np.float32)
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 8:
        raise ValueError(f"Expected states (T,8), got {states.shape}")
    if raw_actions.ndim != 2 or raw_actions.shape != (len(states), 7):
        raise ValueError(f"Expected actions (T,7) aligned with states, got {raw_actions.shape}")
    if len(states) < 2:
        raise ValueError("An episode needs at least two rows for next-action temporal alignment")
    if not np.isfinite(states).all() or not np.isfinite(raw_actions).all():
        raise ValueError("State/action contains NaN or Inf")
    gripper_values = np.unique(np.round(raw_actions[:, 6], decimals=5))
    if not set(gripper_values.tolist()).issubset({-1.0, 1.0}):
        raise ValueError(
            f"Expected canonical LIBERO gripper commands in {{-1,+1}}, got {gripper_values.tolist()}"
        )

    # row t observation is the pre-action state for source action[t+1].
    current = torch.from_numpy(states[:-1]).to(torch.float64)
    next_actions = torch.from_numpy(raw_actions[1:]).to(torch.float64)
    from lerobot.processor.eef_relative_action_processor import osc_actions_to_absolute_eef

    absolute = osc_actions_to_absolute_eef(
        next_actions,
        current,
        position_scale=position_scale,
        rotation_scale=rotation_scale,
        clip=True,
    )
    return states[:-1].copy(), absolute.to(torch.float32).numpy()


def _read_all_parquet(directory: Path) -> pd.DataFrame:
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {directory}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _task_lookup(source: Path) -> dict[int, str]:
    tasks = pd.read_parquet(source / "meta" / "tasks.parquet")
    return {int(row.task_index): str(name) for name, row in tasks.iterrows()}


def safe_output_path(root: Path, output_name: str) -> Path:
    """Resolve one direct child of root; overwrite must never reach a source/parent path."""
    if not output_name or Path(output_name).name != output_name or output_name in {".", ".."}:
        raise ValueError(f"Output dataset name must be one folder name, got {output_name!r}")
    output = root / output_name
    if output.resolve().parent != root.resolve():
        raise ValueError(f"Output dataset escapes configured root: {output}")
    return output


def balanced_contiguous_shard_bounds(lengths: list[int], num_shards: int) -> list[tuple[int, int]]:
    """Split ordered episodes into contiguous shards with roughly equal frame counts."""
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if num_shards > len(lengths):
        raise ValueError(f"Cannot split {len(lengths)} episodes into {num_shards} non-empty shards")
    if any(length <= 1 for length in lengths):
        raise ValueError("Every source episode must contain at least two rows")

    converted_lengths = np.asarray(lengths, dtype=np.int64) - 1
    cumulative = np.cumsum(converted_lengths)
    total = int(cumulative[-1])
    boundaries = [0]
    for shard in range(1, num_shards):
        boundary = int(np.searchsorted(cumulative, total * shard / num_shards, side="left")) + 1
        boundary = max(boundary, boundaries[-1] + 1)
        boundary = min(boundary, len(lengths) - (num_shards - shard))
        boundaries.append(boundary)
    boundaries.append(len(lengths))
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def probe_h264_nvenc() -> tuple[bool, str]:
    """Exercise an actual NVENC session; codec registration alone is not sufficient."""
    try:
        import av

        sink = io.BytesIO()
        with av.open(sink, mode="w", format="mp4") as container:
            stream = container.add_stream("h264_nvenc", rate=20)
            # 256x256 matches LIBERO and avoids minimum-dimension restrictions on
            # some NVENC generations that reject otherwise valid 64x64 sessions.
            stream.width = 256
            stream.height = 256
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame.from_ndarray(
                np.zeros((256, 256, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        if not sink.getvalue():
            return False, "encoder produced no MP4 bytes"
        return True, "one-frame H.264 NVENC probe succeeded"
    except Exception as exc:  # Hardware/session failures vary by driver and PyAV version.
        return False, f"{type(exc).__name__}: {exc}"


def resolve_builder_vcodec(requested: str) -> tuple[str, str]:
    """Resolve the portable builder policy to NVENC or its software H.264 fallback."""
    from lerobot.datasets.video_utils import resolve_vcodec

    if requested != "portable_h264":
        resolved = resolve_vcodec(requested)
        return resolved, f"explicit codec {requested!r} resolved by LeRobot"
    available, detail = probe_h264_nvenc()
    return ("h264_nvenc" if available else "h264"), detail


def shard_output_path(root: Path, output_name: str, shard_index: int) -> Path:
    """Return a controlled private location used only for intermediate datasets."""
    safe_output_path(root, output_name)
    if shard_index < 0:
        raise ValueError(f"shard_index must be non-negative, got {shard_index}")
    parent = root / "_libero_anchor_relative_shards" / output_name
    output = parent / f"shard-{shard_index:03d}"
    if output.resolve().parent != parent.resolve():
        raise ValueError(f"Shard output escapes its configured parent: {output}")
    return output


def checkpoint_output_path(
    root: Path,
    output_name: str,
    array_shard_index: int,
    checkpoint_index: int,
) -> Path:
    """Return the private path for one restart-safe episode checkpoint dataset."""
    safe_output_path(root, output_name)
    if array_shard_index < 0 or checkpoint_index < 0:
        raise ValueError("Checkpoint indices must be non-negative")
    parent = (
        root
        / "_libero_anchor_relative_checkpoints"
        / output_name
        / f"array-{array_shard_index:03d}"
    )
    output = parent / f"checkpoint-{checkpoint_index:04d}"
    if output.resolve().parent != parent.resolve():
        raise ValueError(f"Checkpoint output escapes its configured parent: {output}")
    return output


def _validate_written_video_lengths(
    output: Path,
    expected_lengths: list[int],
    video_keys: list[str],
    fps: int,
) -> None:
    """Catch streaming queue drops by comparing stored episode durations to row counts."""
    written = _read_all_parquet(output / "meta" / "episodes").sort_values("episode_index")
    actual_lengths = written["length"].astype(int).tolist()
    if actual_lengths != expected_lengths:
        raise RuntimeError(f"Written episode lengths {actual_lengths} != expected {expected_lengths}")
    for row in written.to_dict("records"):
        expected = int(row["length"])
        for key in video_keys:
            duration = float(row[f"videos/{key}/to_timestamp"]) - float(
                row[f"videos/{key}/from_timestamp"]
            )
            encoded_frames = duration * fps
            if abs(encoded_frames - expected) > 0.51:
                raise RuntimeError(
                    f"episode {int(row['episode_index'])} {key}: video duration implies "
                    f"{encoded_frames:.3f} frames, expected {expected}; streaming frames may have dropped"
                )


def build_one(
    *,
    source: Path,
    output: Path,
    output_name: str,
    cfg: dict[str, Any],
    spec: dict[str, Any],
    overwrite: bool,
    max_episodes: int | None,
    shard_index: int | None = None,
    num_shards: int | None = None,
    logical_output_name: str | None = None,
    episode_position_range: tuple[int, int] | None = None,
    completion_metadata: dict[str, Any] | None = None,
    preloaded: dict[str, Any] | None = None,
    codec_resolution: tuple[str, str] | None = None,
) -> bool:
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if (
        source_resolved == output_resolved
        or source_resolved.is_relative_to(output_resolved)
        or output_resolved.is_relative_to(source_resolved)
    ):
        raise ValueError("Source and output dataset paths must be separate, non-nested directories")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError(f"max_episodes must be positive, got {max_episodes}")
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Source LeRobot dataset not found: {source}")
    source_info = (
        preloaded["source_info"]
        if preloaded is not None
        else json.loads((source / "meta" / "info.json").read_text())
    )
    fps = int(source_info["fps"])
    features = output_features(source_info)
    video_keys = [key for key, value in features.items() if value.get("dtype") == "video"]
    position_scale = float(spec.get("osc_position_scale", cfg.get("osc_position_scale", 0.05)))
    rotation_scale = float(spec.get("osc_rotation_scale", cfg.get("osc_rotation_scale", 0.5)))
    requested_vcodec = str(spec.get("vcodec", cfg.get("convert_vcodec", "libsvtav1")))
    streaming_encoding = bool(cfg.get("convert_streaming_encoding", False))

    lerobot_src = project_root(cfg) / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import decode_episode_video_frames

    episodes = (
        preloaded["episodes"]
        if preloaded is not None
        else _read_all_parquet(source / "meta" / "episodes").sort_values("episode_index")
    )
    episode_rows = episodes.to_dict("records")
    if max_episodes is not None:
        episode_rows = episode_rows[:max_episodes]
    if episode_position_range is not None:
        start, end = episode_position_range
        if not 0 <= start < end <= len(episode_rows):
            raise ValueError(
                f"episode_position_range {(start, end)} is invalid for {len(episode_rows)} episodes"
            )
        episode_rows = episode_rows[start:end]
    elif shard_index is not None or num_shards is not None:
        if shard_index is None or num_shards is None:
            raise ValueError("shard_index and num_shards must be specified together")
        if not 0 <= shard_index < num_shards:
            raise ValueError(f"shard_index {shard_index} is outside [0, {num_shards})")
        bounds = balanced_contiguous_shard_bounds(
            [int(row["length"]) for row in episode_rows], num_shards
        )
        start, end = bounds[shard_index]
        episode_rows = episode_rows[start:end]
    else:
        start, end = 0, len(episode_rows)
    if not episode_rows:
        raise ValueError("No source episodes selected")

    if output.exists():
        if not overwrite:
            if completion_metadata is not None or (
                shard_index is not None and num_shards is not None
            ):
                manifest_name = (
                    "checkpoint_manifest.json"
                    if completion_metadata is not None
                    else "shard_manifest.json"
                )
                manifest_path = output / "meta" / manifest_name
                if not manifest_path.is_file():
                    if completion_metadata is not None:
                        print(f"[{output_name}] remove interrupted checkpoint: {output}")
                        shutil.rmtree(output)
                    else:
                        raise FileExistsError(
                            f"Incomplete shard output exists: {output}. Re-run this shard with FORCE=1."
                        )
                else:
                    manifest = json.loads(manifest_path.read_text())
                    expected_manifest = {
                        "logical_output_name": logical_output_name or output_name,
                        "source_dataset": str(source_resolved),
                        "source_episode_position_start": start,
                        "source_episode_position_end": end,
                        "source_episode_indices": [int(row["episode_index"]) for row in episode_rows],
                        "requested_vcodec": requested_vcodec,
                        "streaming_encoding": streaming_encoding,
                        "osc_position_scale": position_scale,
                        "osc_rotation_scale": rotation_scale,
                    }
                    if completion_metadata is not None:
                        expected_manifest.update(completion_metadata)
                    else:
                        expected_manifest.update(
                            {"shard_index": shard_index, "num_shards": num_shards}
                        )
                    mismatches = {
                        key: (manifest.get(key), value)
                        for key, value in expected_manifest.items()
                        if manifest.get(key) != value
                    }
                    if not mismatches:
                        print(f"[{output_name}] matching completed unit already exists, skip: {output}")
                        return False
                    if completion_metadata is not None:
                        print(f"[{output_name}] replace stale checkpoint ({mismatches}): {output}")
                        shutil.rmtree(output)
                    else:
                        raise FileExistsError(
                            f"Existing shard was built with different inputs/settings: {mismatches}. "
                            "Re-run this shard with FORCE=1."
                        )
            else:
                print(f"[{output_name}] already exists, skip: {output}")
                return False
        if output.exists():
            shutil.rmtree(output)

    resolved_vcodec, codec_detail = codec_resolution or resolve_builder_vcodec(requested_vcodec)
    configured_queue = int(cfg.get("convert_encoder_queue_maxsize", 30))
    encoder_queue_maxsize = max(configured_queue, max(int(row["length"]) for row in episode_rows) + 1)
    configured_encoder_threads = int(cfg.get("convert_encoder_threads", 0))
    encoder_threads = configured_encoder_threads or None
    dataset = LeRobotDataset.create(
        repo_id=f"dohyeon/{output_name}",
        fps=fps,
        root=output,
        robot_type=source_info.get("robot_type", "franka"),
        features=features,
        vcodec=resolved_vcodec,
        deferred_video_packing=bool(cfg.get("convert_deferred_video_packing", True)),
        streaming_encoding=streaming_encoding,
        encoder_queue_maxsize=encoder_queue_maxsize,
        encoder_threads=encoder_threads,
        image_writer_threads=0
        if streaming_encoding
        else int(cfg.get("convert_image_writer_threads", 10)),
        image_writer_processes=0
        if streaming_encoding
        else int(cfg.get("convert_image_writer_processes", 5)),
    )
    if preloaded is not None:
        task_by_index = preloaded["task_by_index"]
        frames_by_episode = preloaded["frames_by_episode"]
    else:
        data = _read_all_parquet(source / "data").sort_values(["episode_index", "frame_index"])
        task_by_index = _task_lookup(source)
        frames_by_episode = {int(index): group for index, group in data.groupby("episode_index")}

    def decode_episode(row: dict[str, Any]) -> dict[str, np.ndarray]:
        length = int(row["length"])
        decoded: dict[str, np.ndarray] = {}
        for key in video_keys:
            chunk = int(row[f"videos/{key}/chunk_index"])
            file_index = int(row[f"videos/{key}/file_index"])
            path = source / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
            frames = decode_episode_video_frames(
                path,
                float(row[f"videos/{key}/from_timestamp"]),
                float(row[f"videos/{key}/to_timestamp"]),
                length,
                fps,
                backend="pyav",
                decoder_num_threads=None,
            )
            decoded[key] = (frames.permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
        return decoded

    print(f"[{output_name}] source={source} output={output}")
    print("  alignment     : observation[t] + action[t+1] (drop final row per episode)")
    print(f"  OSC scales    : position={position_scale} rotation={rotation_scale}")
    print(f"  codec policy  : {requested_vcodec}")
    print(f"  codec selected: {resolved_vcodec} ({codec_detail})")
    print(
        f"  video path    : {'streaming' if streaming_encoding else 'PNG staging'}"
        f" (queue={encoder_queue_maxsize}, encoder_threads={encoder_threads})"
    )
    total_frames = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(decode_episode, row) for row in episode_rows[:2]]
        for position, row in enumerate(episode_rows):
            videos = futures[position].result()
            if position + 2 < len(episode_rows):
                futures.append(pool.submit(decode_episode, episode_rows[position + 2]))

            episode_index = int(row["episode_index"])
            group = frames_by_episode[episode_index]
            length = int(row["length"])
            if len(group) != length:
                raise ValueError(f"episode {episode_index}: parquet rows {len(group)} != metadata {length}")
            states = np.stack(group["observation.state"].to_numpy()).astype(np.float32)
            raw_actions = np.stack(group["action"].to_numpy()).astype(np.float32)
            kept_states, absolute_targets = derive_aligned_absolute_targets(
                states,
                raw_actions,
                position_scale=position_scale,
                rotation_scale=rotation_scale,
            )
            task_index = int(group["task_index"].iloc[0])
            task = task_by_index[task_index]

            auxiliary = {
                key: np.stack(group[key].to_numpy()).astype(np.float32)[:-1]
                for key in features
                if key.startswith("observation.states.")
            }
            for frame_index in range(length - 1):
                item: dict[str, Any] = {
                    "observation.state": kept_states[frame_index],
                    "action": absolute_targets[frame_index],
                    "task": task,
                }
                for key in video_keys:
                    item[key] = videos[key][frame_index].copy()
                for key, values in auxiliary.items():
                    item[key] = values[frame_index]
                dataset.add_frame(item)
            dataset.save_episode()
            total_frames += length - 1
            if (position + 1) % 50 == 0 or position + 1 == len(episode_rows):
                print(f"  episodes={position + 1}/{len(episode_rows)} frames={total_frames}")

    dataset.finalize()
    expected_lengths = [int(row["length"]) - 1 for row in episode_rows]
    _validate_written_video_lengths(output, expected_lengths, video_keys, fps)
    contract = {
        "schema_version": 1,
        "storage_representation": "absolute_eef_command",
        "model_representation": "eef_anchor_relative_so3",
        "rotation_representation": "axis_angle_rotation_vector",
        "rotation_composition": "left_world",
        "temporal_alignment": "output observation[t] + source action[t+1]",
        "source_dataset": str(source),
        "dropped_rows_per_episode": {"first_source_action": 1, "last_source_observation": 1},
        "osc_position_scale": position_scale,
        "osc_rotation_scale": rotation_scale,
        "gripper": "absolute_-1_or_+1",
        "video_encoding": {
            "requested": requested_vcodec,
            "resolved": resolved_vcodec,
            "probe": codec_detail,
            "streaming": streaming_encoding,
        },
    }
    if shard_index is not None and num_shards is not None:
        contract["shard"] = {
            "logical_output_name": logical_output_name or output_name,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "source_episode_position_start": start,
            "source_episode_position_end": end,
        }
    if completion_metadata is not None:
        contract["checkpoint"] = completion_metadata
    (output / "meta" / "action_contract.json").write_text(json.dumps(contract, indent=2))
    final_info = json.loads((output / "meta" / "info.json").read_text())
    if int(final_info["total_frames"]) != total_frames:
        raise RuntimeError(f"Written frame count {final_info['total_frames']} != expected {total_frames}")
    if completion_metadata is not None or (shard_index is not None and num_shards is not None):
        manifest = {
            "schema_version": 1,
            "logical_output_name": logical_output_name or output_name,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "source_dataset": str(source_resolved),
            "source_episode_position_start": start,
            "source_episode_position_end": end,
            "source_episode_indices": [int(row["episode_index"]) for row in episode_rows],
            "total_episodes": len(episode_rows),
            "total_frames": total_frames,
            "requested_vcodec": requested_vcodec,
            "resolved_vcodec": resolved_vcodec,
            "streaming_encoding": streaming_encoding,
            "osc_position_scale": position_scale,
            "osc_rotation_scale": rotation_scale,
        }
        if completion_metadata is not None:
            manifest.update(completion_metadata)
            manifest_name = "checkpoint_manifest.json"
        else:
            manifest.update({"shard_index": shard_index, "num_shards": num_shards})
            manifest_name = "shard_manifest.json"
        (output / "meta" / manifest_name).write_text(json.dumps(manifest, indent=2))
    print(f"[{output_name}] done: {len(episode_rows)} episodes / {total_frames} frames")
    return True


def build_resumable_shard(
    *,
    source: Path,
    root: Path,
    output_name: str,
    cfg: dict[str, Any],
    spec: dict[str, Any],
    overwrite: bool,
    max_episodes: int | None,
    shard_index: int,
    num_shards: int,
) -> bool:
    """Build one array shard from restart-safe episode checkpoint datasets."""
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} is outside [0, {num_shards})")
    checkpoint_episodes = int(cfg.get("convert_checkpoint_episodes", 20))
    if checkpoint_episodes <= 0:
        raise ValueError(f"convert_checkpoint_episodes must be positive, got {checkpoint_episodes}")

    source_resolved = source.resolve()
    source_info = json.loads((source / "meta" / "info.json").read_text())
    episodes = _read_all_parquet(source / "meta" / "episodes").sort_values("episode_index")
    all_rows = episodes.to_dict("records")
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError(f"max_episodes must be positive, got {max_episodes}")
        all_rows = all_rows[:max_episodes]
    bounds = balanced_contiguous_shard_bounds(
        [int(row["length"]) for row in all_rows], num_shards
    )
    shard_start, shard_end = bounds[shard_index]
    shard_rows = all_rows[shard_start:shard_end]
    checkpoint_ranges = [
        (start, min(start + checkpoint_episodes, shard_end))
        for start in range(shard_start, shard_end, checkpoint_episodes)
    ]
    source_indices = [int(row["episode_index"]) for row in shard_rows]
    requested_vcodec = str(spec.get("vcodec", cfg.get("convert_vcodec", "libsvtav1")))
    position_scale = float(spec.get("osc_position_scale", cfg.get("osc_position_scale", 0.05)))
    rotation_scale = float(spec.get("osc_rotation_scale", cfg.get("osc_rotation_scale", 0.5)))
    streaming_encoding = bool(cfg.get("convert_streaming_encoding", False))
    output = shard_output_path(root, output_name, shard_index)
    checkpoint_parent = (
        root
        / "_libero_anchor_relative_checkpoints"
        / output_name
        / f"array-{shard_index:03d}"
    )

    expected_shard = {
        "logical_output_name": output_name,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "source_dataset": str(source_resolved),
        "source_episode_position_start": shard_start,
        "source_episode_position_end": shard_end,
        "source_episode_indices": source_indices,
        "requested_vcodec": requested_vcodec,
        "streaming_encoding": streaming_encoding,
        "osc_position_scale": position_scale,
        "osc_rotation_scale": rotation_scale,
        "checkpoint_episodes": checkpoint_episodes,
        "num_checkpoints": len(checkpoint_ranges),
    }
    if overwrite:
        if output.exists():
            shutil.rmtree(output)
        if checkpoint_parent.exists():
            shutil.rmtree(checkpoint_parent)
    elif output.exists():
        manifest_path = output / "meta" / "shard_manifest.json"
        mismatches: dict[str, tuple[Any, Any]] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            mismatches = {
                key: (manifest.get(key), value)
                for key, value in expected_shard.items()
                if manifest.get(key) != value
            }
            info_path = output / "meta" / "info.json"
            if not info_path.is_file():
                mismatches["info.json"] = (False, True)
            else:
                info = json.loads(info_path.read_text())
                expected_frames = sum(int(row["length"]) - 1 for row in shard_rows)
                if int(info["total_episodes"]) != len(shard_rows):
                    mismatches["total_episodes"] = (info["total_episodes"], len(shard_rows))
                if int(info["total_frames"]) != expected_frames:
                    mismatches["total_frames"] = (info["total_frames"], expected_frames)
        else:
            mismatches["shard_manifest.json"] = (False, True)
        if not mismatches:
            if checkpoint_parent.exists() and not bool(cfg.get("convert_keep_checkpoints", False)):
                shutil.rmtree(checkpoint_parent)
            print(f"[{output_name} shard {shard_index}] completed shard already exists, resume skip")
            return False
        print(f"[{output_name} shard {shard_index}] replace interrupted/stale aggregate: {mismatches}")
        shutil.rmtree(output)

    checkpoint_paths = [
        checkpoint_output_path(root, output_name, shard_index, index)
        for index in range(len(checkpoint_ranges))
    ]
    completion_metadata = [
        {
            "array_shard_index": shard_index,
            "num_array_shards": num_shards,
            "checkpoint_index": index,
            "num_checkpoints": len(checkpoint_ranges),
            "checkpoint_episodes": checkpoint_episodes,
        }
        for index in range(len(checkpoint_ranges))
    ]

    incomplete: list[int] = []
    for index, (path, (start, end), metadata) in enumerate(
        zip(checkpoint_paths, checkpoint_ranges, completion_metadata, strict=True)
    ):
        marker = path / "meta" / "checkpoint_manifest.json"
        if not marker.is_file():
            incomplete.append(index)
            if path.exists():
                print(f"[{output_name} shard {shard_index}] remove interrupted checkpoint: {path}")
                shutil.rmtree(path)
            continue
        manifest = json.loads(marker.read_text())
        rows = all_rows[start:end]
        expected = {
            **metadata,
            "logical_output_name": output_name,
            "source_dataset": str(source_resolved),
            "source_episode_position_start": start,
            "source_episode_position_end": end,
            "source_episode_indices": [int(row["episode_index"]) for row in rows],
            "requested_vcodec": requested_vcodec,
            "streaming_encoding": streaming_encoding,
            "osc_position_scale": position_scale,
            "osc_rotation_scale": rotation_scale,
        }
        valid = not any(manifest.get(key) != value for key, value in expected.items())
        info_path = path / "meta" / "info.json"
        if valid and info_path.is_file():
            checkpoint_info = json.loads(info_path.read_text())
            valid = (
                int(checkpoint_info["total_episodes"]) == int(manifest["total_episodes"])
                and int(checkpoint_info["total_frames"]) == int(manifest["total_frames"])
            )
        else:
            valid = False
        if not valid:
            incomplete.append(index)
            print(f"[{output_name} shard {shard_index}] remove invalid checkpoint: {path}")
            shutil.rmtree(path)

    if incomplete:
        print(
            f"[{output_name} shard {shard_index}] resume: "
            f"{len(checkpoint_ranges) - len(incomplete)}/{len(checkpoint_ranges)} checkpoints complete"
        )
        data = _read_all_parquet(source / "data").sort_values(["episode_index", "frame_index"])
        preloaded = {
            "source_info": source_info,
            "episodes": episodes,
            "task_by_index": _task_lookup(source),
            "frames_by_episode": {
                int(episode): group for episode, group in data.groupby("episode_index")
            },
        }
        codec_resolution = resolve_builder_vcodec(requested_vcodec)
        for index in incomplete:
            start, end = checkpoint_ranges[index]
            build_one(
                source=source,
                output=checkpoint_paths[index],
                output_name=(
                    f"{output_name}_array_{shard_index:03d}_checkpoint_{index:04d}"
                ),
                cfg=cfg,
                spec=spec,
                overwrite=False,
                max_episodes=max_episodes,
                logical_output_name=output_name,
                episode_position_range=(start, end),
                completion_metadata=completion_metadata[index],
                preloaded=preloaded,
                codec_resolution=codec_resolution,
            )
    else:
        print(
            f"[{output_name} shard {shard_index}] resume: all "
            f"{len(checkpoint_ranges)} checkpoints already complete"
        )

    checkpoint_manifests = [
        json.loads((path / "meta" / "checkpoint_manifest.json").read_text())
        for path in checkpoint_paths
    ]
    lerobot_src = project_root(cfg) / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.datasets.aggregate import aggregate_datasets

    repo_ids = [
        f"dohyeon/{output_name}_array_{shard_index:03d}_checkpoint_{index:04d}"
        for index in range(len(checkpoint_paths))
    ]
    print(
        f"[{output_name} shard {shard_index}] packing "
        f"{len(checkpoint_paths)} checkpoints into {output}"
    )
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=f"dohyeon/{output_name}_shard_{shard_index:03d}",
        roots=checkpoint_paths,
        aggr_root=output,
    )

    expected_lengths = [int(row["length"]) - 1 for row in shard_rows]
    expected_frames = sum(expected_lengths)
    info = json.loads((output / "meta" / "info.json").read_text())
    if int(info["total_episodes"]) != len(shard_rows) or int(info["total_frames"]) != expected_frames:
        raise RuntimeError(
            f"Packed shard counts {(info['total_episodes'], info['total_frames'])} != "
            f"expected {(len(shard_rows), expected_frames)}"
        )
    video_keys = [key for key, value in info["features"].items() if value.get("dtype") == "video"]
    _validate_written_video_lengths(output, expected_lengths, video_keys, int(info["fps"]))

    contract = json.loads((checkpoint_paths[0] / "meta" / "action_contract.json").read_text())
    contract.pop("checkpoint", None)
    contract["shard"] = {
        "logical_output_name": output_name,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "source_episode_position_start": shard_start,
        "source_episode_position_end": shard_end,
        "checkpoint_episodes": checkpoint_episodes,
    }
    contract["video_encoding"] = {
        "requested": requested_vcodec,
        "resolved_per_checkpoint": [
            manifest["resolved_vcodec"] for manifest in checkpoint_manifests
        ],
        "streaming": streaming_encoding,
    }
    (output / "meta" / "action_contract.json").write_text(json.dumps(contract, indent=2))
    resolved = sorted({manifest["resolved_vcodec"] for manifest in checkpoint_manifests})
    shard_manifest = {
        "schema_version": 1,
        **expected_shard,
        "total_episodes": len(shard_rows),
        "total_frames": expected_frames,
        "resolved_vcodec": resolved[0] if len(resolved) == 1 else f"mixed:{','.join(resolved)}",
    }
    (output / "meta" / "shard_manifest.json").write_text(json.dumps(shard_manifest, indent=2))
    print(
        f"[{output_name} shard {shard_index}] checkpoint aggregate complete: "
        f"{len(shard_rows)} episodes / {expected_frames} frames"
    )
    if not bool(cfg.get("convert_keep_checkpoints", False)):
        shutil.rmtree(checkpoint_parent)
        print(f"[{output_name} shard {shard_index}] removed packed checkpoints: {checkpoint_parent}")
    return True


def _read_shard_manifests(root: Path, output_name: str, num_shards: int) -> tuple[list[Path], list[dict]]:
    shard_paths = [shard_output_path(root, output_name, index) for index in range(num_shards)]
    manifests: list[dict[str, Any]] = []
    expected_start = 0
    for index, shard in enumerate(shard_paths):
        manifest_path = shard / "meta" / "shard_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Shard {index}/{num_shards} is not complete (missing {manifest_path})"
            )
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "logical_output_name": output_name,
            "shard_index": index,
            "num_shards": num_shards,
            "source_episode_position_start": expected_start,
        }
        mismatches = {
            key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Invalid shard manifest {manifest_path}: {mismatches}")
        info = json.loads((shard / "meta" / "info.json").read_text())
        if int(info["total_episodes"]) != int(manifest["total_episodes"]):
            raise ValueError(f"Shard {index} episode count does not match its manifest")
        if int(info["total_frames"]) != int(manifest["total_frames"]):
            raise ValueError(f"Shard {index} frame count does not match its manifest")
        selected_count = int(manifest["source_episode_position_end"]) - int(
            manifest["source_episode_position_start"]
        )
        if selected_count != int(manifest["total_episodes"]):
            raise ValueError(f"Shard {index} source range does not match its episode count")
        if len(manifest.get("source_episode_indices", [])) != int(manifest["total_episodes"]):
            raise ValueError(f"Shard {index} source episode list does not match its episode count")
        expected_start = int(manifest["source_episode_position_end"])
        manifests.append(manifest)
    return shard_paths, manifests


def aggregate_shards(
    *,
    root: Path,
    output_name: str,
    cfg: dict[str, Any],
    spec: dict[str, Any],
    num_shards: int,
    overwrite: bool,
    skip_stats: bool,
) -> bool:
    """Atomically gate aggregation on complete shard manifests, then compute stats once."""
    output = safe_output_path(root, output_name)
    completion_path = output / "meta" / "aggregation_manifest.json"
    chunk_size = int(spec.get("relative_chunk_size", cfg.get("relative_chunk_size", 50)))
    if output.exists() and not overwrite and not completion_path.is_file():
        if not bool(cfg.get("convert_replace_incomplete_output", False)):
            raise FileExistsError(
                f"Incomplete aggregate output exists: {output}. Re-run aggregation with FORCE=1."
            )
        print(
            f"[{output_name}] incomplete previous output detected; it will be replaced "
            "only after every new shard validates"
        )
    elif output.exists() and not overwrite:
        completion = json.loads(completion_path.read_text())
        if int(completion.get("num_shards", -1)) != num_shards:
            raise FileExistsError(
                f"Existing aggregate used {completion.get('num_shards')} shards, requested {num_shards}. "
                "Re-run with FORCE=1."
            )
        contract = json.loads((output / "meta" / "action_contract.json").read_text())
        expected_contract = {
            "osc_position_scale": float(
                spec.get("osc_position_scale", cfg.get("osc_position_scale", 0.05))
            ),
            "osc_rotation_scale": float(
                spec.get("osc_rotation_scale", cfg.get("osc_rotation_scale", 0.5))
            ),
        }
        mismatches = {
            key: (contract.get(key), value)
            for key, value in expected_contract.items()
            if contract.get(key) != value
        }
        if mismatches:
            raise FileExistsError(
                f"Existing aggregate used different action settings: {mismatches}. Re-run with FORCE=1."
            )
        stats_need_update = (
            not completion.get("relative_stats_generated", False)
            or int(completion.get("relative_chunk_size", -1)) != chunk_size
            or not (output / "meta" / "relative_action_stats.json").is_file()
        )
        if not skip_stats and stats_need_update:
            compute_and_write(output, chunk_size=chunk_size, overwrite=False)
            completion["relative_stats_generated"] = True
            completion["relative_chunk_size"] = chunk_size
            completion_path.write_text(json.dumps(completion, indent=2))
        print(f"[{output_name}] completed aggregate already exists, skip: {output}")
        return False

    shard_paths, manifests = _read_shard_manifests(root, output_name, num_shards)
    sources = {manifest["source_dataset"] for manifest in manifests}
    if len(sources) != 1:
        raise ValueError(f"Shards refer to different source datasets: {sorted(sources)}")
    if output.exists():
        shutil.rmtree(output)

    lerobot_src = project_root(cfg) / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.datasets.aggregate import aggregate_datasets

    repo_ids = [f"dohyeon/{output_name}_shard_{index:03d}" for index in range(num_shards)]
    print(f"[{output_name}] aggregating {num_shards} completed shards into {output}")
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=f"dohyeon/{output_name}",
        roots=shard_paths,
        aggr_root=output,
        # portable_h264 is resolved independently on every GPU shard. Software H.264 and
        # NVENC can use different B-frame delays, so packet-level concatenation can make
        # DTS move backwards at a shard boundary. Keep each source MP4 separate instead.
        concatenate_videos=False,
    )

    expected_episodes = sum(int(manifest["total_episodes"]) for manifest in manifests)
    expected_frames = sum(int(manifest["total_frames"]) for manifest in manifests)
    final_info = json.loads((output / "meta" / "info.json").read_text())
    if int(final_info["total_episodes"]) != expected_episodes:
        raise RuntimeError(
            f"Aggregated episodes {final_info['total_episodes']} != expected {expected_episodes}"
        )
    if int(final_info["total_frames"]) != expected_frames:
        raise RuntimeError(f"Aggregated frames {final_info['total_frames']} != expected {expected_frames}")

    expected_lengths: list[int] = []
    for shard in shard_paths:
        shard_episodes = _read_all_parquet(shard / "meta" / "episodes").sort_values("episode_index")
        expected_lengths.extend(shard_episodes["length"].astype(int).tolist())
    video_keys = [
        key for key, value in final_info["features"].items() if value.get("dtype") == "video"
    ]
    _validate_written_video_lengths(output, expected_lengths, video_keys, int(final_info["fps"]))

    shard_contract = json.loads((shard_paths[0] / "meta" / "action_contract.json").read_text())
    shard_contract.pop("shard", None)
    shard_contract["source_dataset"] = next(iter(sources))
    shard_contract["video_encoding"] = {
        "requested": sorted({manifest["requested_vcodec"] for manifest in manifests}),
        "resolved_per_shard": [manifest["resolved_vcodec"] for manifest in manifests],
        "streaming": bool(cfg.get("convert_streaming_encoding", False)),
    }
    (output / "meta" / "action_contract.json").write_text(json.dumps(shard_contract, indent=2))

    if not skip_stats:
        compute_and_write(output, chunk_size=chunk_size, overwrite=True)
    completion = {
        "schema_version": 1,
        "num_shards": num_shards,
        "total_episodes": expected_episodes,
        "total_frames": expected_frames,
        "relative_stats_generated": not skip_stats,
        "relative_chunk_size": chunk_size if not skip_stats else None,
        "resolved_vcodecs": [manifest["resolved_vcodec"] for manifest in manifests],
    }
    completion_path.write_text(json.dumps(completion, indent=2))
    print(f"[{output_name}] aggregate complete: {expected_episodes} episodes / {expected_frames} frames")

    if not bool(cfg.get("convert_keep_shards", False)):
        shard_parent = root / "_libero_anchor_relative_shards" / output_name
        shutil.rmtree(shard_parent)
        print(f"[{output_name}] removed verified intermediate shards: {shard_parent}")
    if not bool(cfg.get("convert_keep_checkpoints", False)):
        checkpoint_parent = root / "_libero_anchor_relative_checkpoints" / output_name
        if checkpoint_parent.exists():
            shutil.rmtree(checkpoint_parent)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--only", default="", help="Comma/space-separated output dataset names")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = dataset_root(cfg)
    specs = dataset_specs(cfg)
    selected = {name for name in args.only.replace(",", " ").split() if name}
    if selected:
        unknown = selected - set(specs)
        if unknown:
            raise ValueError(f"Unknown output datasets: {sorted(unknown)}")
        specs = {name: spec for name, spec in specs.items() if name in selected}
    if not specs:
        raise ValueError("No anchor_relative_datasets configured")
    if args.aggregate_only or args.shard_index is not None or args.num_shards is not None:
        if len(specs) != 1:
            raise ValueError("Shard and aggregate modes require --only to select exactly one dataset")
        if args.num_shards is None:
            raise ValueError("Shard and aggregate modes require --num-shards")
        if args.aggregate_only and args.shard_index is not None:
            raise ValueError("--aggregate-only cannot be combined with --shard-index")
        if not args.aggregate_only and args.shard_index is None:
            raise ValueError("Shard mode requires --shard-index")

    for output_name, spec in specs.items():
        source_raw = Path(str(spec["source_dataset"])).expanduser()
        source = source_raw if source_raw.is_absolute() else root / source_raw
        if args.aggregate_only:
            aggregate_shards(
                root=root,
                output_name=output_name,
                cfg=cfg,
                spec=spec,
                num_shards=args.num_shards,
                overwrite=args.overwrite or bool(cfg.get("convert_overwrite", False)),
                skip_stats=args.skip_stats,
            )
            continue
        if args.shard_index is not None:
            build_resumable_shard(
                source=source,
                root=root,
                output_name=output_name,
                cfg=cfg,
                spec=spec,
                overwrite=args.overwrite or bool(cfg.get("convert_overwrite", False)),
                max_episodes=args.max_episodes,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            continue
        output = safe_output_path(root, output_name)
        build_one(
            source=source,
            output=output,
            output_name=output_name,
            cfg=cfg,
            spec=spec,
            overwrite=args.overwrite or bool(cfg.get("convert_overwrite", False)),
            max_episodes=args.max_episodes,
        )
        if not args.skip_stats:
            compute_and_write(
                output,
                chunk_size=int(spec.get("relative_chunk_size", cfg.get("relative_chunk_size", 50))),
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
