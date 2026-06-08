#!/usr/bin/env python3
"""One-off migration: rotate 180 the videos of an already-converted LIBERO LeRobot dataset.

The original LIBERO HDF5 stores agentview/eye-in-hand frames rotated 180 (mujoco renders
bottom-to-top). Datasets converted before convert_original_libero_to_lerobot.py applied the
flip are upside-down; this re-encodes their videos in place to the upright convention used by
libero_dataset and eval's LiberoProcessorStep.

Only the video files are touched:
  - parquet/meta are unchanged (frame order, count, fps, timestamps all preserved)
  - stats.json is unchanged (per-pixel stats are invariant under a 180 rotation)

Encoding params mirror LeRobotDataset's writer exactly (libsvtav1, yuv420p, g=2, crf=30,
preset=12) so seek timestamps keep resolving.

Usage:
  python rotate180_postprocess.py --root /path/to/dataset/libero_90_full_full [--workers 6]
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import av.logging
import numpy as np

from lerobot.datasets.video_utils import _get_codec_options

VCODEC = "libsvtav1"
PIX_FMT = "yuv420p"
G = 2
CRF = 30
PRESET = None  # -> "12" in _get_codec_options
FPS = 20


def reencode_rotated(src: str) -> tuple[str, int]:
    """Decode src, rotate every frame 180, re-encode in place. Streams frame-by-frame."""
    src_path = Path(src)
    tmp_path = src_path.with_suffix(".rot.tmp.mp4")
    av.logging.set_level(av.logging.ERROR)

    options = _get_codec_options(VCODEC, G, CRF, PRESET)

    in_cont = av.open(str(src_path))
    in_stream = in_cont.streams.video[0]
    out_cont = av.open(str(tmp_path), "w")
    out_stream = None
    n = 0
    try:
        for frame in in_cont.decode(in_stream):
            arr = frame.to_ndarray(format="rgb24")[::-1, ::-1]
            arr = np.ascontiguousarray(arr)
            if out_stream is None:
                out_stream = out_cont.add_stream(VCODEC, FPS, options=options)
                out_stream.pix_fmt = PIX_FMT
                out_stream.height = arr.shape[0]
                out_stream.width = arr.shape[1]
            vframe = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for pkt in out_stream.encode(vframe):
                out_cont.mux(pkt)
            n += 1
        if out_stream is not None:
            for pkt in out_stream.encode():
                out_cont.mux(pkt)
    finally:
        out_cont.close()
        in_cont.close()

    os.replace(tmp_path, src_path)
    return src, n


def main() -> None:
    logging.getLogger("libav").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Dataset root (contains videos/).")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    vids = sorted(glob.glob(str(args.root / "videos" / "**" / "*.mp4"), recursive=True))
    if not vids:
        raise SystemExit(f"No .mp4 found under {args.root}/videos")
    print(f"Rotating 180 in place: {len(vids)} video files under {args.root}")
    print(f"  encode: {VCODEC} {PIX_FMT} g={G} crf={CRF} preset=12 fps={FPS} | workers={args.workers}")

    total = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(reencode_rotated, v): v for v in vids}
        for fut in as_completed(futs):
            src, n = fut.result()
            total += n
            done += 1
            rel = Path(src).relative_to(args.root)
            print(f"  [{done}/{len(vids)}] {rel}  frames={n}", flush=True)

    print(f"DONE  files={len(vids)} total_frames={total}")


if __name__ == "__main__":
    main()
