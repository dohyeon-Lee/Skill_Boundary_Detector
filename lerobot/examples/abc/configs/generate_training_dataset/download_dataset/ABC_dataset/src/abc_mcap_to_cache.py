#!/usr/bin/env python3
"""One-way ABC-130k MCAP -> local training-cache conversion.

This module intentionally lives with the ABC dataset pipeline.  It contains only
the format code required by ``convert_abc_dataset.py`` and has no dependency on
the standalone ``abcdl`` package.

The cache layout remains byte/layout-compatible with the previous intermediate
format so an already-created cache can still be reused::

    combined_camera-images-rgb.mp4
    states_actions.bin
    episode_metadata.json
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import warnings
from dataclasses import dataclass

import numpy as np


FPS = 30
TICK_NS = 33_333_333
TIMESCALE = 15_360
DEFAULT_SIZE = 224

_CAM_TOPICS = {
    "top": "/top-camera",
    "top_left": "/top-left-camera",
    "top_right": "/top-right-camera",
    "left_wrist": "/left-wrist-camera",
    "right_wrist": "/right-wrist-camera",
}
_TOPIC_TO_CAM = {topic: name for name, topic in _CAM_TOPICS.items()}


@dataclass
class CameraStream:
    frames: list[bytes] | np.ndarray
    timestamps: np.ndarray
    width: int
    height: int
    codec: str


@dataclass
class EpisodeMeta:
    task: str
    fps: float
    cameras: list[str]
    camera_resolutions: dict[str, tuple[int, int]]
    camera_codecs: dict[str, str]
    operator_id: str | None = None
    alignment: str = "native"
    t0_ns: int | None = None
    tick_ns: int | None = None
    session_id: str | None = None


@dataclass
class Episode:
    states: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    cameras: dict[str, CameraStream]
    meta: EpisodeMeta

    @property
    def num_steps(self) -> int:
        return int(self.states.shape[0])

    def validate(self) -> None:
        steps = self.num_steps
        if self.actions.shape[0] != steps:
            raise ValueError(f"actions length {self.actions.shape[0]} != states {steps}")
        if self.timestamps.shape[0] != steps:
            raise ValueError(f"timestamps length {self.timestamps.shape[0]} != states {steps}")
        for name in self.meta.cameras:
            if name not in self.cameras:
                raise ValueError(f"metadata lists missing camera {name!r}")


def _floor_idx(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Index the latest source timestamp at-or-before each target timestamp."""
    return np.clip(np.searchsorted(source, target, side="right") - 1, 0, len(source) - 1)


def _channel_time(message) -> int:
    return int(message.timestamp.seconds) * 1_000_000_000 + int(message.timestamp.nanos)


def _read_metadata(reader) -> tuple[str | None, str | None, str | None]:
    """Read both metadata naming variants seen in released ABC episodes."""
    for metadata in reader.iter_metadata():
        values = dict(metadata.metadata)
        if metadata.name in ("episode-metadata", "session-metadata"):
            return (
                values.get("session_id") or values.get("session-uuid"),
                values.get("operator_id") or values.get("operator-id"),
                values.get("task_name") or values.get("instruction"),
            )
    return None, None, None


def read_abc_mcap(path: str) -> Episode:
    """Decode one ABC-130k MCAP episode into the minimal in-memory representation."""
    # Optional for cache-only rebuilds; required only when stage ② actually reads MCAP.
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    with open(path, "rb") as stream:
        session_id, operator_id, metadata_task = _read_metadata(make_reader(stream))

    arms: dict[tuple[str, str], list[tuple[int, list[float]]]] = {}
    grippers: dict[tuple[str, str], list[tuple[int, float]]] = {}
    cameras: dict[str, list[tuple[int, bytes, str]]] = {}
    task: str | None = None

    with open(path, "rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, _, decoded in reader.iter_decoded_messages():
            topic = channel.topic
            if topic == "/instruction":
                task = decoded.data
            elif topic.endswith("-arm-state") or topic.endswith("-arm-action"):
                side = "left" if "left" in topic else "right"
                kind = "state" if topic.endswith("state") else "action"
                arms.setdefault((side, kind), []).append(
                    (_channel_time(decoded), list(decoded.position))
                )
            elif topic.endswith("-ee-state") or topic.endswith("-ee-action"):
                side = "left" if "left" in topic else "right"
                kind = "state" if topic.endswith("state") else "action"
                grippers.setdefault((side, kind), []).append(
                    (_channel_time(decoded), float(decoded.position[0]))
                )
            elif topic in _TOPIC_TO_CAM:
                name = _TOPIC_TO_CAM[topic]
                cameras.setdefault(name, []).append(
                    (_channel_time(decoded), bytes(decoded.data), decoded.format)
                )

    def stack(side: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
        if (side, kind) not in arms:
            raise ValueError(f"MCAP missing required topic /{side}-arm-{kind}")
        if (side, kind) not in grippers:
            raise ValueError(f"MCAP missing required topic /{side}-ee-{kind}")
        arm_samples = sorted(arms[(side, kind)])
        grip_samples = sorted(grippers[(side, kind)])
        arm_ts = np.asarray([sample[0] for sample in arm_samples], dtype=np.int64)
        arm_values = np.asarray([sample[1] for sample in arm_samples], dtype=np.float64)
        grip_ts = np.asarray([sample[0] for sample in grip_samples], dtype=np.int64)
        grip_values = np.asarray([sample[1] for sample in grip_samples], dtype=np.float64)[:, None]
        return arm_ts, np.concatenate([arm_values, grip_values[_floor_idx(grip_ts, arm_ts)]], axis=1)

    left_ts, left_state = stack("left", "state")
    left_action_ts, left_action_values = stack("left", "action")
    right_state_ts, right_state_values = stack("right", "state")
    right_action_ts, right_action_values = stack("right", "action")

    states = np.concatenate(
        [left_state, right_state_values[_floor_idx(right_state_ts, left_ts)]], axis=1
    )
    actions = np.concatenate(
        [
            left_action_values[_floor_idx(left_action_ts, left_ts)],
            right_action_values[_floor_idx(right_action_ts, left_ts)],
        ],
        axis=1,
    )

    known = [name for name in _CAM_TOPICS if name in cameras]
    unknown = sorted(name for name in cameras if name not in _CAM_TOPICS)
    camera_names = known + unknown
    streams: dict[str, CameraStream] = {}
    resolutions: dict[str, tuple[int, int]] = {}
    codecs: dict[str, str] = {}
    for name in camera_names:
        samples = sorted(cameras[name], key=lambda sample: sample[0])
        codec = samples[0][2]
        streams[name] = CameraStream(
            frames=[sample[1] for sample in samples],
            timestamps=np.asarray([sample[0] for sample in samples], dtype=np.int64),
            width=0,
            height=0,
            codec=codec,
        )
        resolutions[name] = (0, 0)
        codecs[name] = codec

    return Episode(
        states=states,
        actions=actions,
        timestamps=left_ts,
        cameras=streams,
        meta=EpisodeMeta(
            task=task or metadata_task or "",
            fps=0.0,
            cameras=camera_names,
            camera_resolutions=resolutions,
            camera_codecs=codecs,
            operator_id=operator_id,
            session_id=session_id,
        ),
    )


def _decode_annexb(
    chunks: list[bytes], codec: str, output_width: int, output_height: int
) -> np.ndarray:
    """Decode, aspect-preserving scale and pad an encoded camera stream."""
    fd, raw_path = tempfile.mkstemp(suffix=f".{codec}")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(b"".join(chunks))
        video_filter = (
            f"scale={output_width}:{output_height}:force_original_aspect_ratio=decrease:flags=bicubic,"
            f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2"
        )
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                raw_path,
                "-vf",
                video_filter,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-v",
                "error",
                "pipe:1",
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg decode failed: {result.stderr.decode(errors='replace')[-500:]}"
            )
        buffer = np.frombuffer(result.stdout, dtype=np.uint8)
        frame_size = output_height * output_width * 3
        count = buffer.size // frame_size
        return buffer[: count * frame_size].reshape(count, output_height, output_width, 3)
    finally:
        os.unlink(raw_path)


_X264_PARAMS = (
    f"keyint={FPS}:min-keyint={FPS}:scenecut=0:"
    f"fps={FPS}/1:timebase=1/{TIMESCALE}:force-cfr=1"
)


def _probe_frame_count(path: str) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()[-500:]}")
    return int(result.stdout.strip())


def _encode_strict_h264(frames: np.ndarray, output_path: str) -> None:
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError("frames must be (N,H,W,3) uint8")
    count, height, width, _ = frames.shape
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-vsync",
            "0",
            "-enc_time_base",
            f"1/{TIMESCALE}",
            "-video_track_timescale",
            str(TIMESCALE),
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-x264-params",
            _X264_PARAMS,
            "-threads",
            "1",
            output_path,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, stderr = process.communicate(input=np.ascontiguousarray(frames).tobytes())
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {stderr.decode(errors='replace')[-500:]}")
    actual = _probe_frame_count(output_path)
    if actual != count:
        raise RuntimeError(f"encoded {actual} frames, expected {count}")


def write_abc_cache(episode: Episode, output_dir: str) -> None:
    """Write the compact cache consumed by the LeRobot-v3 build stage."""
    episode.validate()
    os.makedirs(output_dir, exist_ok=True)
    steps = episode.num_steps

    state_action = np.concatenate(
        [np.asarray(episode.states, np.float64), np.asarray(episode.actions, np.float64)], axis=1
    )
    state_action.astype("<f8").tofile(os.path.join(output_dir, "states_actions.bin"))

    camera_stacks: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for name in episode.meta.cameras:
        frames = np.asarray(episode.cameras[name].frames, dtype=np.uint8)
        if frames.shape[0] != steps:
            raise ValueError(f"camera {name} has {frames.shape[0]} frames, expected {steps}")
        current_shape = (int(frames.shape[1]), int(frames.shape[2]))
        if shape is None:
            shape = current_shape
        elif current_shape != shape:
            raise ValueError("all cameras must share H,W for vertical stacking")
        camera_stacks.append(frames)
    combined = np.concatenate(camera_stacks, axis=1)
    _encode_strict_h264(
        combined, os.path.join(output_dir, "combined_camera-images-rgb.mp4")
    )

    tick_ns = int(episode.meta.tick_ns or TICK_NS)
    metadata = {
        "task_name": episode.meta.task,
        "cameras": list(episode.meta.cameras),
        "camera_resolutions": {
            key: list(value) for key, value in episode.meta.camera_resolutions.items()
        },
        "alignment": episode.meta.alignment,
        "t0_ns": int(episode.meta.t0_ns or 0),
        "tick_ns": tick_ns,
        "fps": float(episode.meta.fps),
        "num_steps": steps,
        "state_dim": int(episode.states.shape[1]),
        "action_dim": int(episode.actions.shape[1]),
        "operator_id": episode.meta.operator_id,
        "session_id": episode.meta.session_id,
        "frame_feature_keys": [],
        "frame_features": {},
    }
    with open(os.path.join(output_dir, "episode_metadata.json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)


def mcap_to_abc_cache(src_mcap: str, output_dir: str, size: int = DEFAULT_SIZE) -> None:
    """Convert one ABC MCAP to a 30 Hz, square-camera local cache episode."""
    source = read_abc_mcap(src_mcap)
    start = int(source.timestamps[0])
    end = int(source.timestamps[-1])
    ticks = np.arange(start + TICK_NS, end + 1, TICK_NS, dtype=np.int64)
    if not len(ticks):
        raise ValueError(f"episode is shorter than one {FPS} Hz tick: {src_mcap}")

    state_indices = _floor_idx(source.timestamps, ticks)
    streams: dict[str, CameraStream] = {}
    resolutions: dict[str, tuple[int, int]] = {}
    codecs: dict[str, str] = {}
    for name in source.meta.cameras:
        camera = source.cameras[name]
        decoded = _decode_annexb(camera.frames, camera.codec, size, size)
        if not len(decoded):
            raise ValueError(f"camera {name!r} decoded zero frames: {src_mcap}")
        if len(decoded) == len(camera.timestamps):
            camera_timestamps = camera.timestamps
        else:
            warnings.warn(
                f"camera {name!r}: decoded {len(decoded)} frames but "
                f"{len(camera.timestamps)} timestamps; respacing",
                stacklevel=2,
            )
            camera_timestamps = np.linspace(
                camera.timestamps[0], camera.timestamps[-1], len(decoded), dtype=np.int64
            )
        selected = decoded[_floor_idx(np.asarray(camera_timestamps, np.int64), ticks)]
        streams[name] = CameraStream(selected, ticks.copy(), size, size, "raw")
        resolutions[name] = (size, size)
        codecs[name] = "h264"

    converted = Episode(
        states=source.states[state_indices],
        actions=source.actions[state_indices],
        timestamps=ticks,
        cameras=streams,
        meta=EpisodeMeta(
            task=source.meta.task,
            fps=float(FPS),
            cameras=list(source.meta.cameras),
            camera_resolutions=resolutions,
            camera_codecs=codecs,
            operator_id=source.meta.operator_id,
            alignment="fixed_clock_30hz_causal",
            t0_ns=start,
            tick_ns=TICK_NS,
            session_id=source.meta.session_id,
        ),
    )
    write_abc_cache(converted, output_dir)
