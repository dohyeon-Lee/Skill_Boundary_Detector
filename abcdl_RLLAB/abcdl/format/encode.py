"""Strict H.264 encoding for the abcdl format — enables analytic frame indexing."""

from __future__ import annotations

import subprocess

import numpy as np

from abcdl.constants import FPS, TIMESCALE

_X264_PARAMS = (
    f"keyint={FPS}:min-keyint={FPS}:scenecut=0:"
    f"fps={FPS}/1:timebase=1/{TIMESCALE}:force-cfr=1"
)
_FFMPEG_ARGS = [
    "-vsync", "0",
    "-enc_time_base", f"1/{TIMESCALE}",
    "-video_track_timescale", str(TIMESCALE),
    "-bf", "0",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-x264-params", _X264_PARAMS,
    "-threads", "1",
]


def probe_frame_count(path: str) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()[-500:]}")
    return int(result.stdout.strip())


def encode_strict_h264(rgb_frames: np.ndarray, out_path: str) -> None:
    if rgb_frames.dtype != np.uint8 or rgb_frames.ndim != 4 or rgb_frames.shape[3] != 3:
        raise ValueError("rgb_frames must be (N,H,W,3) uint8")
    n, h, w, _ = rgb_frames.shape
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", *_FFMPEG_ARGS, out_path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _, stderr = proc.communicate(input=np.ascontiguousarray(rgb_frames).tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {stderr.decode(errors='replace')[-500:]}")
    got = probe_frame_count(out_path)
    if got != n:
        raise RuntimeError(f"encoded {got} frames, expected {n}")
