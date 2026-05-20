"""Visual sanity check for skill boundary splits.

For each selected episode, this script draws all skills in temporal order.
Each skill contributes two images side by side:

  skill start frame | skill end frame

Skills are laid out horizontally with thick vertical separators. Multiple
episodes are stacked vertically.

Example:
    python eval.py --task_id 25 --num_episodes 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

CONFIG_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONFIG_DIR))

from pipeline_config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--homedir", default=cfg.homedir)
    p.add_argument("--projdir", default=cfg.projdir)
    p.add_argument("--dataset", default=cfg.data)
    p.add_argument("--dataset_root", default=cfg.datadir)
    p.add_argument("--image_key", default=cfg.image_key)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=4)
    p.add_argument("--episode_offset", type=int, default=0)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--separator_width", type=int, default=6)
    p.add_argument("--output_dir", default="")
    return p.parse_args()


def load_episodes_meta(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files under {dataset_dir / 'meta' / 'episodes'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if "tasks" in df.columns and tasks_path.exists():
        tasks_df = pd.read_parquet(tasks_path)
        task_to_idx = {str(task): int(row["task_index"]) for task, row in tasks_df.iterrows()}

        def resolve_task_index(tasks) -> int:
            if isinstance(tasks, (list, tuple, np.ndarray)) and len(tasks) > 0:
                return task_to_idx.get(str(tasks[0]), -1)
            return task_to_idx.get(str(tasks), -1)

        df["__task_index"] = df["tasks"].apply(resolve_task_index)
    return df


def dataset_fps(dataset_dir: Path) -> float:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return 20.0
    with open(info_path) as f:
        return float(json.load(f).get("fps", 20.0))


def video_path(dataset_dir: Path, episodes_meta: pd.DataFrame, episode_id: int, image_key: str) -> Path:
    row = episodes_meta[episodes_meta["episode_index"] == episode_id].iloc[0]
    chunk_idx = int(row[f"videos/{image_key}/chunk_index"])
    file_idx = int(row[f"videos/{image_key}/file_index"])
    return dataset_dir / "videos" / image_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def read_video_frames(path: Path, start_ts: float | None = None, end_ts: float | None = None) -> np.ndarray:
    try:
        from torchvision.io import read_video

        kwargs = {}
        if start_ts is not None:
            kwargs["start_pts"] = float(start_ts)
        if end_ts is not None:
            kwargs["end_pts"] = float(end_ts)
        frames, _, _ = read_video(str(path), output_format="THWC", pts_unit="sec", **kwargs)
        return frames.numpy().astype(np.uint8)[..., :3]
    except Exception:
        import imageio

        reader = imageio.get_reader(str(path))
        try:
            frames = [frame[..., :3] for frame in reader]
        finally:
            reader.close()
        if not frames:
            raise ValueError(f"No frames in {path}")
        return np.stack(frames).astype(np.uint8)


def task_episode_ids(meta: pd.DataFrame, task_id: int, num_episodes: int, offset: int) -> list[int]:
    if "__task_index" not in meta.columns:
        raise ValueError("No task index metadata found")
    ids = sorted(meta[meta["__task_index"].astype(int) == task_id]["episode_index"].astype(int).tolist())
    if not ids:
        raise ValueError(f"No episodes found for task_id={task_id}")
    return ids[offset : offset + num_episodes]


def load_episode_skills(skillset_dir: Path, task_id: int, episode_id: int) -> list[dict]:
    task_dir = skillset_dir / f"task{task_id:02d}"
    files = sorted(task_dir.glob(f"ep{episode_id:05d}_task{task_id:02d}_skill*.npz"))
    skills = []
    for path in files:
        d = np.load(str(path))
        skills.append(
            {
                "skill_index": int(d["skill_index"]),
                "frame_start": int(d["frame_start"]),
                "frame_end": int(d["frame_end"]),
                "path": path,
            }
        )
    skills.sort(key=lambda x: (x["frame_start"], x["skill_index"]))
    return skills


def resize_frame(frame: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(frame).resize((size, size), Image.BILINEAR)


def make_episode_row(
    frames: np.ndarray,
    skills: list[dict],
    *,
    image_size: int,
    separator_width: int,
    episode_id: int,
) -> Image.Image:
    T = len(frames)
    if not skills:
        row = Image.new("RGB", (image_size * 2, image_size), (40, 40, 40))
        ImageDraw.Draw(row).text((6, 6), f"ep {episode_id}: no skills", fill=(255, 255, 255))
        return row

    width_per_skill = image_size * 2 + separator_width
    canvas_w = image_size + separator_width + width_per_skill * len(skills) + image_size
    canvas = Image.new("RGB", (canvas_w, image_size), "white")
    draw = ImageDraw.Draw(canvas)

    canvas.paste(resize_frame(frames[0], image_size), (0, 0))
    draw.rectangle([image_size, 0, image_size + separator_width - 1, image_size], fill=(10, 10, 10))
    draw.text((4, 4), f"ep{episode_id} start", fill=(255, 255, 255))

    for i, skill in enumerate(skills):
        x = image_size + separator_width + i * width_per_skill
        fs = max(0, min(int(skill["frame_start"]), T - 1))
        fe = max(0, min(int(skill["frame_end"]) - 1, T - 1))
        canvas.paste(resize_frame(frames[fs], image_size), (x, 0))
        canvas.paste(resize_frame(frames[fe], image_size), (x + image_size, 0))

        draw.rectangle([x, 0, x + separator_width - 1, image_size], fill=(10, 10, 10))
        draw.text((x + separator_width + 4, 4), f"ep{episode_id} s{skill['skill_index']}", fill=(255, 255, 255))
        draw.text((x + separator_width + 4, 18), f"{fs}->{fe}", fill=(255, 255, 255))

    end_x = image_size + separator_width + width_per_skill * len(skills)
    canvas.paste(resize_frame(frames[-1], image_size), (end_x, 0))
    draw.text((end_x + 4, 4), f"ep{episode_id} end", fill=(255, 255, 255))

    return canvas


def main() -> None:
    args = parse_args()
    root = Path(args.homedir + args.projdir)
    dataset_dir = root / args.dataset_root / args.dataset
    skillset_dir = root / args.dataset_root / f"{args.dataset}_data" / f"{args.dataset}_for_FSQ" / f"{args.dataset}_skillset" / "skills"
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "image"
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_episodes_meta(dataset_dir)
    episode_ids = task_episode_ids(meta, args.task_id, args.num_episodes, args.episode_offset)

    rows = []
    for ep_id in episode_ids:
        row_meta = meta[meta["episode_index"] == ep_id].iloc[0]
        from_ts = float(row_meta[f"videos/{args.image_key}/from_timestamp"])
        to_ts = float(row_meta[f"videos/{args.image_key}/to_timestamp"])
        ep_len = int(row_meta["length"])
        frames = read_video_frames(video_path(dataset_dir, meta, ep_id, args.image_key), from_ts, to_ts - 0.001)
        if len(frames) > ep_len:
            frames = frames[-ep_len:]
        skills = load_episode_skills(skillset_dir, args.task_id, ep_id)
        rows.append(
            make_episode_row(
                frames,
                skills,
                image_size=args.image_size,
                separator_width=args.separator_width,
                episode_id=ep_id,
            )
        )
        print(
            f"[skillset eval] ep={ep_id}, from_ts={from_ts:.3f}, to_ts={to_ts:.3f}, "
            f"len={len(frames)}, expected_len={ep_len}, skills={len(skills)}"
        )

    max_w = max(row.width for row in rows)
    out = Image.new("RGB", (max_w, args.image_size * len(rows)), "white")
    for i, row in enumerate(rows):
        out.paste(row, (0, i * args.image_size))

    out_path = output_dir / f"task{args.task_id:02d}_episodes{episode_ids[0]:05d}-{episode_ids[-1]:05d}.png"
    out.save(out_path)
    print(f"[skillset eval] saved -> {out_path}")


if __name__ == "__main__":
    main()
