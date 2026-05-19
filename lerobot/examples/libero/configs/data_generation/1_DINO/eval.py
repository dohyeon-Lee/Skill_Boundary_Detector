"""Visual sanity check for precomputed frame-level DINO patch tokens.

The script reads raw episode frames and the matching precomputed DINO episode
npz, then saves one image with N temporal columns:

  row 0: raw frames over time
  row 1: PCA RGB of 8x8 patch tokens over time

Examples:
    python eval.py --task_id 1 --episode_ordinal 0 --n_frames 20
    python eval.py --episode_id 123 --n_frames 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--homedir", default="/data2/dohyeon")
    p.add_argument("--projdir", default="/SBD")
    p.add_argument("--dataset", default="libero_90")
    p.add_argument("--dataset_root", default="libero_dataset")
    p.add_argument("--visual_backbone", default="dinov3_vits16")
    p.add_argument("--patch_grid", type=int, default=8)
    p.add_argument("--image_key", default="observation.images.image")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--episode_id", type=int, default=-1)
    p.add_argument("--episode_ordinal", type=int, default=0)
    p.add_argument("--n_frames", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--output_dir", default="")
    return p.parse_args()


def safe_key(image_key: str) -> str:
    return image_key.replace("/", "_").replace(".", "_")


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


def task_column(df: pd.DataFrame) -> str | None:
    for col in ("__task_index", "task_index", "task_id", "tasks/task_index", "tasks/task_id"):
        if col in df.columns:
            return col
    return None


def resolve_episode(df: pd.DataFrame, task_id: int, episode_id: int, episode_ordinal: int) -> int:
    if episode_id >= 0:
        if episode_id not in set(df["episode_index"].astype(int)):
            raise ValueError(f"episode_id={episode_id} not found in metadata")
        col = task_column(df)
        if task_id >= 0 and col is not None:
            row = df[df["episode_index"] == episode_id].iloc[0]
            got_task = int(row[col])
            if got_task != task_id:
                print(f"[warn] episode {episode_id} belongs to task {got_task}, not requested task {task_id}")
        return int(episode_id)

    if task_id < 0:
        raise ValueError("Set either --episode_id or both --task_id and --episode_ordinal")

    col = task_column(df)
    if col is None:
        raise ValueError("No task column found in episode metadata; use --episode_id instead")

    episodes = sorted(df[df[col].astype(int) == task_id]["episode_index"].astype(int).tolist())
    if not episodes:
        raise ValueError(f"No episodes found for task_id={task_id}")
    if not (0 <= episode_ordinal < len(episodes)):
        raise ValueError(
            f"episode_ordinal={episode_ordinal} out of range for task {task_id} "
            f"(num episodes={len(episodes)})"
        )
    return int(episodes[episode_ordinal])


def video_path(dataset_dir: Path, episodes_meta: pd.DataFrame, episode_id: int, image_key: str) -> Path:
    row = episodes_meta[episodes_meta["episode_index"] == episode_id].iloc[0]
    chunk_idx = int(row[f"videos/{image_key}/chunk_index"])
    file_idx = int(row[f"videos/{image_key}/file_index"])
    return dataset_dir / "videos" / image_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def dataset_fps(dataset_dir: Path) -> float:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return 20.0
    with open(info_path) as f:
        return float(json.load(f).get("fps", 20.0))


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


def pca_rgb(patches: np.ndarray, grid: int) -> np.ndarray:
    """patches: (N_frames, grid*grid, dim) -> uint8 RGB maps."""
    flat = patches.reshape(-1, patches.shape[-1]).astype(np.float32)
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    coords = flat @ vt[:3].T
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    lo = np.percentile(coords, 1, axis=0, keepdims=True)
    hi = np.percentile(coords, 99, axis=0, keepdims=True)
    rgb = np.clip((coords - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return (rgb.reshape(patches.shape[0], grid, grid, 3) * 255).astype(np.uint8)


def resize(arr: np.ndarray, size: int, resample: int) -> Image.Image:
    return Image.fromarray(arr).resize((size, size), resample)


def main() -> None:
    args = parse_args()
    root = Path(args.homedir + args.projdir)
    dataset_dir = root / args.dataset_root / args.dataset
    dino_dir = (
        root
        / args.dataset_root
        / f"{args.dataset}_data"
        / f"{args.dataset}_DINO"
        / f"{args.visual_backbone}_pg{args.patch_grid}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "image"
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_episodes_meta(dataset_dir)
    episode_id = resolve_episode(meta, args.task_id, args.episode_id, args.episode_ordinal)

    ep_npz = dino_dir / safe_key(args.image_key) / f"episode_{episode_id:07d}.npz"
    if not ep_npz.exists():
        raise FileNotFoundError(f"DINO episode file not found: {ep_npz}")

    row = meta[meta["episode_index"] == episode_id].iloc[0]
    from_ts = float(row[f"videos/{args.image_key}/from_timestamp"])
    to_ts = float(row[f"videos/{args.image_key}/to_timestamp"])
    expected_len = int(row["length"])
    frames = read_video_frames(video_path(dataset_dir, meta, episode_id, args.image_key), from_ts, to_ts - 0.001)
    if len(frames) > expected_len:
        frames = frames[-expected_len:]
    features = np.load(str(ep_npz))["features"].astype(np.float32)
    if len(frames) != expected_len:
        print(
            f"[warn] raw video slice length mismatch: got {len(frames)}, "
            f"expected {expected_len}, from_ts={from_ts}, to_ts={to_ts}"
        )
    if len(features) != expected_len:
        print(f"[warn] DINO feature length mismatch: got {len(features)}, expected {expected_len}")
    n_tokens = int(features.shape[1])
    grid = int(round((n_tokens - 1) ** 0.5))
    if 1 + grid * grid != n_tokens:
        raise ValueError(f"Expected CLS + square patch grid, got n_tokens={n_tokens}")

    T = min(len(frames), len(features))
    n = min(args.n_frames, T)
    rng = np.random.default_rng(args.seed)
    frame_ids = np.sort(rng.choice(np.arange(T), size=n, replace=False))

    pca_maps = pca_rgb(features[frame_ids, 1:, :], grid)
    canvas = Image.new("RGB", (args.image_size * n, args.image_size * 2), "white")
    draw = ImageDraw.Draw(canvas)

    for col, frame_idx in enumerate(frame_ids):
        x = col * args.image_size
        raw = resize(frames[frame_idx], args.image_size, Image.BILINEAR)
        pca = resize(pca_maps[col], args.image_size, Image.NEAREST)
        canvas.paste(raw, (x, 0))
        canvas.paste(pca, (x, args.image_size))
        draw.text((x + 4, 4), f"t={int(frame_idx)}", fill=(255, 255, 255))
        draw.text((x + 4, args.image_size + 4), f"{grid}x{grid} PCA", fill=(255, 255, 255))

    task_tag = f"task{args.task_id:02d}" if args.task_id >= 0 else "taskNA"
    out_path = output_dir / f"{task_tag}_ep{episode_id:07d}_{safe_key(args.image_key)}.png"
    canvas.save(out_path)
    print(
        f"[DINO eval] episode={episode_id}, from_ts={from_ts:.3f}, to_ts={to_ts:.3f}, "
        f"episode_len={expected_len}, feature_len={len(features)}, local_frames={frame_ids.tolist()}"
    )
    print(f"[DINO eval] saved -> {out_path}")


if __name__ == "__main__":
    main()
