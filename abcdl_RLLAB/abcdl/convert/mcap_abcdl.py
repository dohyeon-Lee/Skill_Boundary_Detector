"""Convert between the ABC-130k MCAP format and the abcdl MP4+binary format.

Fidelity caveat (abcdl_to_mcap single-frame re-encode):
  Each decoded frame is re-encoded by encode_strict_h264 into a tiny MP4 container
  file, whose raw bytes are then used as the "chunk" written to MCAP. This yields an
  MP4-container blob, not a pure Annex-B elementary stream. The abcdl reader/writer
  handle this correctly for round-trip purposes, but the resulting MCAP is NOT
  byte-identical to the original ABC-130k per-frame Annex-B representation.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import warnings

import numpy as np

from abcdl.constants import TICK_NS
from abcdl.episode import CameraStream, Episode, EpisodeMeta
from abcdl.format.reader import read_abcdl
from abcdl.format.writer import write_abcdl
from abcdl.mcap.reader import read_mcap
from abcdl.mcap.writer import write_mcap

DEFAULT_SIZE = 224  # default square output resolution for the abcdl training cache


def abcdl_format_name(size: int = DEFAULT_SIZE) -> str:
    """Format/branch name for an abcdl training cache at *size* px, e.g. ``abcdl_224``.

    The HF layout names the source branch ``mcap`` and each training-cache branch
    ``abcdl_<size>``, so multiple resolutions can coexist in one dataset repo:
    ``hf.push(repo_id, dir, fmt=abcdl_format_name(224), version="v1")``.
    """
    return f"abcdl_{int(size)}"


def _floor_idx(src_ts: np.ndarray, tgt_ts: np.ndarray) -> np.ndarray:
    """For each element of tgt_ts, return the index of the latest src_ts at-or-before it."""
    return np.clip(np.searchsorted(src_ts, tgt_ts, side="right") - 1, 0, len(src_ts) - 1)


def _decode_annexb(chunks: list[bytes], codec: str, out_w: int, out_h: int) -> np.ndarray:
    """Decode a list of Annex-B frame chunks to (N, out_h, out_w, 3) uint8 via ffmpeg.

    Scale each frame preserving aspect ratio, then pad to exactly out_w x out_h.
    """
    fd, raw_name = tempfile.mkstemp(suffix=f".{codec}")
    try:
        with os.fdopen(fd, "wb") as raw:
            raw.write(b"".join(chunks))

        # scale preserving AR, then pad to exact size; force_original_aspect_ratio=decrease
        # ensures the image fits within out_w x out_h before padding.
        vf = (
            f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=bicubic,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
        )
        proc = subprocess.run(
            [
                "ffmpeg", "-i", raw_name,
                "-vf", vf,
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-v", "error",
                "pipe:1",
            ],
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg decode failed: {proc.stderr.decode(errors='replace')[-500:]}"
            )
        buf = np.frombuffer(proc.stdout, np.uint8)
        # Exact frame count: integer division avoids residual byte issues.
        n = buf.size // (out_h * out_w * 3)
        return buf[: n * out_h * out_w * 3].reshape(n, out_h, out_w, 3)
    finally:
        os.unlink(raw_name)


def mcap_to_abcdl(src_mcap: str, out_dir: str, size: int = DEFAULT_SIZE) -> None:
    """Read *src_mcap* (ABC-130k), resample to a fixed 30 Hz causal clock, write abcdl.

    *size* is the square output resolution (pixels) every camera is scaled/padded to;
    it is also the resolution encoded in the format/branch name convention
    ``abcdl_<size>`` (e.g. ``abcdl_224``) — see :func:`abcdl_format_name`.

    Resampling strategy:
      - ``ticks = np.arange(t0 + TICK_NS, t_end + 1, TICK_NS)``  (causal: first tick
        is one period after the first state sample so all ticks have a valid floor sample).
      - States and actions are floor-sampled (latest sample at-or-before each tick).
      - Each camera's Annex-B stream is decoded via ffmpeg, scaled/padded to size x size,
        then floor-sampled onto the same tick grid.
      - The resulting Episode has alignment == "fixed_clock_30hz_causal" and fps == 30.
    """
    ep = read_mcap(src_mcap)
    t0 = int(ep.timestamps[0])
    t_end = int(ep.timestamps[-1])
    # Causal fixed-clock grid: ticks start one period after t0 so every tick has
    # a valid floor sample in the source stream.
    ticks = np.arange(t0 + TICK_NS, t_end + 1, TICK_NS, dtype=np.int64)
    T = len(ticks)

    # --- resample states / actions ---
    si = _floor_idx(ep.timestamps, ticks)
    states = ep.states[si]
    actions = ep.actions[si]

    # --- decode and resample cameras ---
    cams: dict[str, CameraStream] = {}
    res: dict[str, tuple[int, int]] = {}
    codecs: dict[str, str] = {}
    for name in ep.meta.cameras:
        c = ep.cameras[name]
        decoded = _decode_annexb(c.frames, c.codec, size, size)  # (Nframes, H, W, 3)
        nf = decoded.shape[0]
        if nf == len(c.timestamps):
            cam_ts = c.timestamps
        else:
            warnings.warn(
                f"camera {name!r}: decoded {nf} frames but {len(c.timestamps)} timestamps; respacing",
                stacklevel=2,
            )
            cam_ts = np.linspace(c.timestamps[0], c.timestamps[-1], nf, dtype=np.int64)
        cam_ts = np.asarray(cam_ts, np.int64)
        fi = _floor_idx(cam_ts, ticks)
        sel = decoded[fi]  # (T, size, size, 3)
        cams[name] = CameraStream(
            frames=sel, timestamps=ticks.copy(), width=size, height=size, codec="raw"
        )
        res[name] = (size, size)
        codecs[name] = "h264"

    meta = EpisodeMeta(
        task=ep.meta.task,
        fps=30.0,
        cameras=list(ep.meta.cameras),
        camera_resolutions=res,
        camera_codecs=codecs,
        operator_id=ep.meta.operator_id,
        alignment="fixed_clock_30hz_causal",
        t0_ns=t0,
        tick_ns=TICK_NS,
        session_id=ep.meta.session_id,
    )
    out_ep = Episode(
        states=states,
        actions=actions,
        timestamps=ticks,
        cameras=cams,
        meta=meta,
    )
    write_abcdl(out_ep, out_dir)


def abcdl_to_mcap(in_dir: str, out_mcap: str) -> None:
    """Read *in_dir* (abcdl), re-encode frames as per-frame H.264 chunks, write MCAP.

    Each decoded frame is encoded to a single-frame MP4 container via
    ``encode_strict_h264``. The raw file bytes are used as the chunk payload.
    See module-level docstring for the fidelity caveat.
    """
    from abcdl.format.encode import encode_strict_h264  # reuse strict encoder per-frame

    ep = read_abcdl(in_dir)
    for name in ep.meta.cameras:
        c = ep.cameras[name]
        frames_arr = np.asarray(c.frames, np.uint8)  # (T, H, W, 3)
        chunks: list[bytes] = []
        for fr in frames_arr:
            fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            try:
                encode_strict_h264(fr[None], tmp_name)
                with open(tmp_name, "rb") as fh:
                    chunks.append(fh.read())
            finally:
                os.unlink(tmp_name)
        # Mutate in-place: write_mcap's _encoded_frames checks isinstance(x, bytes).
        c.frames = chunks
        c.codec = "h264"
    write_mcap(ep, out_mcap)
