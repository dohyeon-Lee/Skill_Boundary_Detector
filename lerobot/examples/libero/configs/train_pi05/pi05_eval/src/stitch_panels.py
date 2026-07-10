#!/usr/bin/env python3
"""Stitch the multi-model panel videos into ONE labelled grid clip per task/episode (pi05 eval).

Input layout (written by eval.sbatch's multi-model loop — lerobot_eval writes videos under
videos/{task_group}_{task_id}/eval_episode_{ep}.mp4, plus a success.json per task dir):
    {panels_root}/{label}/videos/{task}/eval_episode_{ep}.mp4
    {panels_root}/{label}/videos/{task}/success.json   ([bool per episode, in video order])
For every task/episode present in ALL panels, the rollouts are glued into a grid — ``per_row`` labelled
panels per row (0 = all in ONE row; short last row right-padded black) → {out_dir}/{task}/eval_episode_{ep}.mp4.
Each panel's title bar is tinted GREEN (success) / RED (failure) from that panel's success.json.
Panel order = labels_json order (= yaml `models` order). Reuses stage1_eval's video_compare helpers.
Never fails the eval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
# .../configs/train_pi05/pi05_eval/src/stitch_panels.py → parents[3] = .../configs
sys.path.insert(0, str(_HERE.parents[3] / "train_skillVLA" / "stage1_eval" / "video_compare"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels_root", type=Path, required=True, help=".../outputs/<run>/panels")
    ap.add_argument("--out_dir", type=Path, required=True, help=".../outputs/<run>/side_by_side")
    ap.add_argument("--labels_json", required=True, help='panel order, e.g. \'["ff_ff","tt_ff"]\'')
    ap.add_argument("--per_row", type=int, default=0, help="panels per row (0 = one row)")
    ap.add_argument("--height", type=int, default=256)
    args = ap.parse_args()

    try:
        from compare_videos import even, load_font, make_panel, read_video  # noqa: PLC0415
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
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

    # Title bar tinted by this panel/episode's success (green ok, red fail, gray unknown).
    def colored_bar(width: int, text: str, success) -> np.ndarray:
        bg = (20, 20, 20) if success is None else ((30, 120, 45) if success else (155, 40, 40))
        img = Image.new("RGB", (width, bar_h), bg)
        d = ImageDraw.Draw(img)
        f = font
        while d.textlength(text, font=f) > width - 8 and f.size > 8:
            f = ImageFont.truetype(f.path, f.size - 1) if hasattr(f, "path") else f
            if not hasattr(f, "path"):
                break
        tw = d.textlength(text, font=f)
        d.text(((width - tw) / 2, max(0, (bar_h - f.size) / 2 - 1)), text, fill=(245, 245, 245), font=f)
        return np.asarray(img)

    succ_cache: dict = {}
    def _success(panel_videos_dir: Path, task: str, ep: int):
        key = (str(panel_videos_dir), task)
        if key not in succ_cache:
            p = panel_videos_dir / task / "success.json"
            try:
                succ_cache[key] = json.loads(p.read_text()) if p.exists() else []
            except Exception:  # noqa: BLE001
                succ_cache[key] = []
        lst = succ_cache[key]
        return lst[ep] if isinstance(lst, list) and 0 <= ep < len(lst) else None

    def _language(task: str) -> str:
        # per-task LIBERO instruction (any panel has it — use panel 0's sidecar)
        p = panels[0][0] / task / "language.txt"
        try:
            return p.read_text().strip() if p.exists() else ""
        except Exception:  # noqa: BLE001
            return ""

    def caption_bar(width: int, text: str) -> np.ndarray:
        img = Image.new("RGB", (width, bar_h), (15, 15, 15))
        d = ImageDraw.Draw(img)
        f = font
        while d.textlength(text, font=f) > width - 8 and f.size > 8:
            f = ImageFont.truetype(f.path, f.size - 1) if hasattr(f, "path") else f
            if not hasattr(f, "path"):
                break
        tw = d.textlength(text, font=f)
        d.text(((width - tw) / 2, max(0, (bar_h - f.size) / 2 - 1)), text, fill=(230, 230, 230), font=f)
        return np.asarray(img)

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
            try:
                ep = int(mp4_0.stem.split("_")[-1])
            except ValueError:
                ep = -1
            bars = []
            for (pdir, lbl), fr in zip(panels, frames_list):
                h, w = fr[0].shape[:2]
                s = _success(pdir, taskdir0.name, ep)
                bars.append(colored_bar(even(max(2, round(w * H / h))), lbl, s))
            language = _language(taskdir0.name)
            out_mp4.parent.mkdir(parents=True, exist_ok=True)
            # ATOMIC write (parallel per-model jobs may stitch concurrently): tmp → os.replace
            tmp_mp4 = out_mp4.with_name(out_mp4.name + f".tmp{os.getpid()}.mp4")
            writer = imageio.get_writer(str(tmp_mp4), fps=fps,
                                        codec="libx264", quality=8, macro_block_size=None)
            caption = None                                            # built once (grid width is constant)
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
                if language:                                         # task prompt caption at the BOTTOM
                    if caption is None:
                        caption = caption_bar(frame.shape[1], language)
                    frame = np.vstack([frame, caption])
                writer.append_data(frame)
            writer.close()
            os.replace(tmp_mp4, out_mp4)
            n += 1
    print(f"stitch: wrote {n} grid clips ({len(panels)} panels, {ncols}/row) → {args.out_dir}")


if __name__ == "__main__":
    main()
