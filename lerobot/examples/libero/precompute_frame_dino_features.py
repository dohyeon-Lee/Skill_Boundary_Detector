"""Precompute frame-level DINO/DINOv3 features for raw LeRobot datasets.

This script is meant for policies such as Diffusion Policy that consume
timestep-level observations. It reads raw dataset videos and writes one shard
per episode:

  output_dir/
    observation.images.image/
      episode_0000000.npz
      episode_0000001.npz
      ...

Each shard stores:
  features: (T, 1 + patch_grid**2, feat_dim) float16

Token 0 is the CLS token. The remaining tokens are spatial patch tokens pooled
to patch_grid x patch_grid, e.g. 8x8 => 64 patch tokens.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from tqdm import tqdm


def _read_video_frames(path: Path) -> np.ndarray:
    try:
        from torchvision.io import read_video

        frames, _, _ = read_video(str(path), output_format="THWC", pts_unit="sec")
        return frames.numpy().astype(np.uint8)[..., :3]
    except Exception:
        import imageio

        reader = imageio.get_reader(str(path))
        try:
            frames = []
            for frame in reader:
                if frame.ndim == 2:
                    frame = np.repeat(frame[..., None], 3, axis=-1)
                frames.append(frame[..., :3])
        finally:
            reader.close()
        if not frames:
            raise ValueError(f"No frames in {path}")
        return np.stack(frames).astype(np.uint8)


def _load_episodes_meta(dataset_dir: Path):
    import pandas as pd

    files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files under {dataset_dir / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _available_video_keys(episodes_meta) -> list[str]:
    cols = [c for c in episodes_meta.columns if c.startswith("videos/") and c.endswith("/chunk_index")]
    return sorted(c.split("/")[1] for c in cols)


def _parse_image_keys(image_keys: str, episodes_meta) -> list[str]:
    available = _available_video_keys(episodes_meta)
    if image_keys.strip().lower() in {"all", "*"}:
        return available
    requested = [k.strip() for k in image_keys.split(",") if k.strip()]
    missing = [k for k in requested if k not in available]
    if missing:
        raise ValueError(f"Missing image keys {missing}. Available video keys: {available}")
    return requested


def _video_path(dataset_dir: Path, episodes_meta, episode_id: int, image_key: str) -> Path:
    row = episodes_meta[episodes_meta["episode_index"] == episode_id].iloc[0]
    chunk_idx = int(row[f"videos/{image_key}/chunk_index"])
    file_idx = int(row[f"videos/{image_key}/file_index"])
    return dataset_dir / "videos" / image_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def _safe_key(image_key: str) -> str:
    return image_key.replace("/", "_").replace(".", "_")


@dataclass
class Args:
    dataset_dir: str
    """Raw LeRobot dataset directory containing videos/ and meta/."""

    output_dir: str
    """Directory where per-episode feature shards will be written."""

    image_keys: str = "observation.images.image,observation.images.wrist_image"
    """Comma-separated video keys, or 'all'."""

    image_model_name: str = "/data2/dohyeon/SBD/models/dinov3-vits16"
    """Local path or Hugging Face model id for the frozen visual backbone."""

    image_size: int = 224
    patch_grid: int = 8
    n_patch_raw: int = 196
    """Raw patch token count from the backbone. DINOv3 ViT-S/16 at 224 uses 14x14=196."""

    batch_size: int = 256
    device: str = "cuda"

    num_workers: int = 1
    worker_id: int = 0
    """Shard episodes by episode-list index modulo num_workers."""

    skip_existing: bool = True
    compile_model: bool = False
    dtype: str = "float16"
    wandb_project: str = ""


def load_dino_model(model_name: str, device: str):
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(model_name)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().to(device)

    try:
        proc = AutoImageProcessor.from_pretrained(model_name)
        mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
        std = getattr(proc, "image_std", [0.229, 0.224, 0.225])
    except Exception:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return model, mean_t, std_t


@torch.no_grad()
def encode_frames(
    model,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    frames: np.ndarray,
    *,
    batch_size: int,
    image_size: int,
    device: str,
    device_type: str,
    patch_grid: int,
    n_patch_raw: int,
) -> np.ndarray:
    pool = torch.nn.AdaptiveAvgPool2d((patch_grid, patch_grid))
    sqrt_raw = int(n_patch_raw**0.5)
    if sqrt_raw * sqrt_raw != n_patch_raw:
        raise ValueError(f"n_patch_raw must be a square number, got {n_patch_raw}")

    chunks: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        x = torch.from_numpy(batch.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
        x = (x - mean_t) / std_t

        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == "cuda")):
            out = model(pixel_values=x)

        hidden = out.last_hidden_state.float()
        bsz, _, feat_dim = hidden.shape
        cls_tok = hidden[:, :1, :]
        patch_tok = hidden[:, -n_patch_raw:, :]

        patch_map = patch_tok.reshape(bsz, sqrt_raw, sqrt_raw, feat_dim).permute(0, 3, 1, 2)
        patch_pooled = pool(patch_map).permute(0, 2, 3, 1).reshape(
            bsz, patch_grid * patch_grid, feat_dim
        )
        tokens = torch.cat([cls_tok, patch_pooled], dim=1)
        chunks.append(tokens.cpu().numpy())

    return np.concatenate(chunks, axis=0)


def _episode_out_path(output_dir: Path, image_key: str, episode_id: int) -> Path:
    return output_dir / _safe_key(image_key) / f"episode_{episode_id:07d}.npz"


def _all_outputs_exist(output_dir: Path, image_keys: list[str], episode_id: int) -> bool:
    return all(_episode_out_path(output_dir, key, episode_id).exists() for key in image_keys)


def _save_episode_features(
    path: Path,
    *,
    features: np.ndarray,
    episode_id: int,
    image_key: str,
    image_model_name: str,
    patch_grid: int,
    n_patch_raw: int,
    dtype: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dtype == "float16":
        features = features.astype(np.float16)
    elif dtype == "float32":
        features = features.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    tmp_path = path.with_suffix(".tmp.npz")
    np.savez(
        str(tmp_path),
        features=features,
        episode_id=np.array(episode_id, dtype=np.int64),
        image_key=np.array(image_key),
        image_model_name=np.array(image_model_name),
        n_tokens=np.array(features.shape[1], dtype=np.int64),
        feat_dim=np.array(features.shape[2], dtype=np.int64),
        patch_grid=np.array(patch_grid, dtype=np.int64),
        n_patch_raw=np.array(n_patch_raw, dtype=np.int64),
        dtype=np.array(str(features.dtype)),
    )
    os.replace(tmp_path, path)


def _write_manifest(
    output_dir: Path,
    *,
    dataset_dir: Path,
    image_keys: list[str],
    image_model_name: str,
    image_size: int,
    patch_grid: int,
    n_patch_raw: int,
    dtype: str,
) -> None:
    manifest = {
        "dataset_dir": str(dataset_dir),
        "image_keys": image_keys,
        "image_model_name": image_model_name,
        "image_size": image_size,
        "patch_grid": patch_grid,
        "n_patch_raw": n_patch_raw,
        "tokens_per_frame": 1 + patch_grid * patch_grid,
        "dtype": dtype,
        "format": "per_episode_npz",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(args: Args) -> None:
    from collections import defaultdict

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes_meta = _load_episodes_meta(dataset_dir)
    image_keys = _parse_image_keys(args.image_keys, episodes_meta)
    episode_ids = sorted(int(x) for x in episodes_meta["episode_index"].tolist())

    # Read FPS from meta/info.json
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            dataset_info = json.load(f)
        fps = float(dataset_info.get("fps", 20.0))
    else:
        fps = 20.0

    device = args.device if torch.cuda.is_available() else "cpu"
    device_type = device.split(":")[0]
    n_tokens = 1 + args.patch_grid * args.patch_grid

    # Build O(1) lookup
    ep_index = episodes_meta.set_index("episode_index")

    # Each LeRobot v2 mp4 file bundles hundreds of episodes sequentially.
    # Group episodes by unique (key, chunk_idx, file_idx) so each mp4 is read once.
    # Shard by unique video-file index instead of by episode to avoid redundant reads.
    #
    # file_map[(key, chunk_idx, file_idx)] = [(ep_id, frame_start, length), ...]
    file_map: dict[tuple, list] = defaultdict(list)
    for ep_id in episode_ids:
        row = ep_index.loc[ep_id]
        length = int(row["length"])
        for key in image_keys:
            chunk_idx = int(row[f"videos/{key}/chunk_index"])
            file_idx = int(row[f"videos/{key}/file_index"])
            from_ts = float(row[f"videos/{key}/from_timestamp"])
            frame_start = round(from_ts * fps)
            file_map[(key, chunk_idx, file_idx)].append((ep_id, frame_start, length))

    all_work_items = sorted(file_map.items())  # [(key, ci, fi), [(ep_id, fs, len), ...]]
    my_work_items = [
        (file_key, eps)
        for i, (file_key, eps) in enumerate(all_work_items)
        if i % args.num_workers == args.worker_id
    ]

    # Filter episodes whose output already exists (per-key check)
    def _filter_existing(file_key, eps):
        key = file_key[0]
        remaining = [(ep_id, fs, ln) for (ep_id, fs, ln) in eps
                     if not (args.skip_existing and _episode_out_path(output_dir, key, ep_id).exists())]
        return remaining

    work_items = []
    skipped = 0
    for file_key, eps in my_work_items:
        remaining_eps = _filter_existing(file_key, eps)
        skipped += len(eps) - len(remaining_eps)
        if remaining_eps:
            work_items.append((file_key, remaining_eps))

    total_ep_count = sum(len(eps) for _, eps in work_items)
    total_files = len(work_items)

    print(f"[FrameDINO] dataset={dataset_dir}  fps={fps}")
    print(f"[FrameDINO] output={output_dir}")
    print(f"[FrameDINO] image_keys={image_keys}")
    print(f"[FrameDINO] model={args.image_model_name}")
    print(f"[FrameDINO] worker={args.worker_id}/{args.num_workers}  unique_files={total_files}  episodes={total_ep_count}  skipped={skipped}")
    print(f"[FrameDINO] tokens/frame={n_tokens}  device={device}  batch_size={args.batch_size}")

    _write_manifest(
        output_dir,
        dataset_dir=dataset_dir,
        image_keys=image_keys,
        image_model_name=args.image_model_name,
        image_size=args.image_size,
        patch_grid=args.patch_grid,
        n_patch_raw=args.n_patch_raw,
        dtype=args.dtype,
    )

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb

            group = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
            wandb_run = wandb.init(
                project=args.wandb_project,
                group=group,
                name=f"frame_dino_w{args.worker_id}",
                config=vars(args),
                resume="allow",
            )
        except Exception as exc:
            print(f"[FrameDINO] wandb init failed: {exc}")

    if not work_items:
        print("[FrameDINO] nothing to do")
        if wandb_run:
            wandb_run.finish()
        return

    model, mean_t, std_t = load_dino_model(args.image_model_name, device)
    if args.compile_model and device_type == "cuda" and hasattr(torch, "compile"):
        print("[FrameDINO] torch.compile model")
        model = torch.compile(model)

    total_frames = 0
    episodes_done = 0
    t0 = time.perf_counter()

    def _load_video_file(key: str, chunk_idx: int, file_idx: int) -> np.ndarray:
        path = dataset_dir / "videos" / key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        return _read_video_frames(path)

    progress = tqdm(total=total_files, desc=f"[w{args.worker_id}] frame DINO (files)")

    # Prefetch next mp4 file in background while GPU encodes current file.
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_key, first_ci, first_fi = work_items[0][0]
        prefetch = executor.submit(_load_video_file, first_key, first_ci, first_fi)

        for i, ((key, chunk_idx, file_idx), eps_in_file) in enumerate(work_items):
            all_frames = prefetch.result()

            # Prefetch next file while GPU encodes current
            if i + 1 < len(work_items):
                next_key, next_ci, next_fi = work_items[i + 1][0]
                prefetch = executor.submit(_load_video_file, next_key, next_ci, next_fi)

            # Encode all frames in this mp4 file in one pass (correct: each file has ~400 episodes)
            all_features = encode_frames(
                model,
                mean_t,
                std_t,
                all_frames,
                batch_size=args.batch_size,
                image_size=args.image_size,
                device=device,
                device_type=device_type,
                patch_grid=args.patch_grid,
                n_patch_raw=args.n_patch_raw,
            )
            total_frames += len(all_frames)
            del all_frames  # free raw frame memory before saving

            # Slice per-episode by from_timestamp and save
            for ep_id, frame_start, length in eps_in_file:
                ep_features = all_features[frame_start : frame_start + length]
                if len(ep_features) != length:
                    print(
                        f"[WARN] ep {ep_id}: expected {length} frames, got {len(ep_features)} "
                        f"(file={chunk_idx}/{file_idx}, total={len(all_features)}, start={frame_start})"
                    )
                _save_episode_features(
                    _episode_out_path(output_dir, key, ep_id),
                    features=ep_features,
                    episode_id=ep_id,
                    image_key=key,
                    image_model_name=args.image_model_name,
                    patch_grid=args.patch_grid,
                    n_patch_raw=args.n_patch_raw,
                    dtype=args.dtype,
                )
                episodes_done += 1

            elapsed = max(time.perf_counter() - t0, 1e-6)
            vid_fps = total_frames / elapsed
            progress.update(1)
            progress.set_postfix({"ep_done": episodes_done, "frames": total_frames, "fps": f"{vid_fps:.1f}"})
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "files_done": i + 1,
                        "episodes_done": episodes_done + skipped,
                        "progress_pct": 100.0 * (episodes_done + skipped) / max(1, total_ep_count + skipped),
                        "frames_done": total_frames,
                        "fps": vid_fps,
                    }
                )

    elapsed = max(time.perf_counter() - t0, 1e-6)
    print(f"[FrameDINO] done worker={args.worker_id}/{args.num_workers}")
    print(f"[FrameDINO] encoded frames={total_frames}  fps={total_frames / elapsed:.2f}")
    print(f"[FrameDINO] episodes saved={episodes_done}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
