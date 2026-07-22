#!/usr/bin/env python3
"""Stitch the multi-scale panel videos into ONE labelled grid clip per task/episode.

Input layout (written by eval.sbatch's multi-model loop, one panel per model):
    {panels_root}/{label}/videos/{task}/eval_episode_{ep}.mp4
For every task/episode present in ALL panels, the rollouts are glued into a grid — ``per_row``
labelled panels per row (0 = all in ONE row; e.g. 6 models @ per_row=3 → 2×3; a short last row is
right-padded black) → {out_dir}/{task}/eval_episode_{ep}.mp4. Panel order = the labels_json order
(= the yaml `models` order). Reuses stage1_eval's video_compare helpers. Never fails the eval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "train_skillVLA" / "stage1_eval" / "video_compare"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels_root", type=Path, required=True, help=".../outputs/<run>/panels")
    ap.add_argument("--out_dir", type=Path, required=True, help=".../outputs/<run>/side_by_side")
    ap.add_argument("--labels_json", required=True, help='panel order, e.g. \'["tttf","ttff"]\'')
    ap.add_argument("--per_row", type=int, default=0, help="panels per row (0 = one row)")
    ap.add_argument("--height", type=int, default=256)
    args = ap.parse_args()

    try:
        from compare_videos import even, label_bar, load_font, make_panel, read_video  # noqa: PLC0415
        import imageio.v2 as imageio  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 — stitching is a convenience; never fail the eval over it
        print(f"stitch skipped (video libs unavailable): {exc}")
        return

    labels = [l for l in json.loads(args.labels_json)
              if (args.panels_root / l / "videos").is_dir()]
    if len(labels) < 2:
        print(f"stitch skipped: <2 panels with videos under {args.panels_root}")
        return
    panels = [(args.panels_root / l / "videos", l) for l in labels]

    H, bar_h = even(args.height), even(max(20, args.height // 9))
    font = load_font(int(bar_h * 0.62))
    ncols = args.per_row if args.per_row and args.per_row > 0 else len(panels)
    n = 0
    dir0 = panels[0][0]
    for taskdir0 in sorted(p for p in dir0.glob("*") if p.is_dir()):
        for mp4_0 in sorted(taskdir0.glob("eval_episode_*.mp4")):
            mp4s = [d / taskdir0.name / mp4_0.name for d, _ in panels]
            if not all(p.exists() for p in mp4s):
                continue          # this task/episode not finished in every panel yet — a later job stitches it
            out_mp4 = args.out_dir / taskdir0.name / mp4_0.name
            if out_mp4.exists() and out_mp4.stat().st_mtime > max(p.stat().st_mtime for p in mp4s):
                continue          # already stitched from these inputs (idempotent re-runs skip)
            try:
                reads = [read_video(p) for p in mp4s]
            except Exception as exc:  # noqa: BLE001
                # A panel's mp4 EXISTS but is still being written by another (per-task, cross-job) call —
                # write_video is not atomic, so an in-flight file has no moov atom yet. Skip for now;
                # a later stitch (progressive per-task, or the end-of-job one) picks it up once complete.
                print(f"stitch: skip {taskdir0.name}/{mp4_0.name} (a panel mp4 not readable yet: {exc})")
                continue
            frames_list, fps = [r[0] for r in reads], reads[0][1]
            if any(not fr for fr in frames_list):
                continue
            bars = []
            for (_, lbl), fr in zip(panels, frames_list):
                h, w = fr[0].shape[:2]
                bars.append(label_bar(even(max(2, round(w * H / h))), bar_h, lbl, font))
            out_mp4.parent.mkdir(parents=True, exist_ok=True)
            # ATOMIC write (parallel per-model jobs may stitch concurrently): tmp → os.replace
            tmp_mp4 = out_mp4.with_name(out_mp4.name + f".tmp{os.getpid()}.mp4")
            writer = imageio.get_writer(str(tmp_mp4), fps=fps,
                                        codec="libx264", quality=8, macro_block_size=None)
            for i in range(max(len(fr) for fr in frames_list)):
                tiles = [make_panel(fr[min(i, len(fr) - 1)], H, bar)
                         for fr, bar in zip(frames_list, bars)]
                rows = [np.hstack(tiles[r : r + ncols]) for r in range(0, len(tiles), ncols)]
                w_max = max(r.shape[1] for r in rows)                 # short last row → right-pad black
                rows = [r if r.shape[1] == w_max else
                        np.pad(r, ((0, 0), (0, w_max - r.shape[1]), (0, 0))) for r in rows]
                frame = np.vstack(rows)
                # libx264 needs even dims (crop a pixel if odd — width AND height once gridded)
                frame = frame[: frame.shape[0] - frame.shape[0] % 2, : frame.shape[1] - frame.shape[1] % 2]
                writer.append_data(frame)
            writer.close()
            os.replace(tmp_mp4, out_mp4)
            n += 1
    print(f"stitch: wrote {n} grid clips ({len(panels)} panels, {ncols}/row) → {args.out_dir}")


if __name__ == "__main__":
    main()
