"""
Precompute frozen DINO/DINOv3 token features for FSQ-AE skill decoder training.

Output format per frame: CLS token + 8×8 pooled patch tokens = 65 tokens × feat_dim.
  features: (N_total_frames, 65, feat_dim)  float16

The 14×14 patch tokens from DINOv3-vits16 are spatially average-pooled to 8×8
at precompute time, so training needs no additional pooling ops.

Output npz keys:
  features      : (N_total, n_tokens, feat_dim) float16
  offsets       : (N_skills + 1,)  int64  — features[offsets[i]:offsets[i+1]] = skill i
  episode_id, task_id, skill_index, frame_start, frame_end, length: (N_skills,)
"""

from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).parent))


# ── video / dataset helpers ───────────────────────────────────────────────────

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


def _resolve_image_key(episodes_meta, image_key: str) -> str:
    video_cols = [c for c in episodes_meta.columns if c.startswith("videos/") and c.endswith("/chunk_index")]
    keys = [c.split("/")[1] for c in video_cols]
    if image_key and image_key in keys:
        return image_key
    if not keys:
        raise ValueError("No video camera keys found in episode metadata.")
    return keys[0]


def _video_path(dataset_dir: Path, episodes_meta, episode_id: int, image_key: str) -> Path:
    row = episodes_meta[episodes_meta["episode_index"] == episode_id].iloc[0]
    chunk_idx = int(row[f"videos/{image_key}/chunk_index"])
    file_idx  = int(row[f"videos/{image_key}/file_index"])
    return dataset_dir / "videos" / image_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def _episode_language_map(episodes_meta) -> dict[int, str]:
    """episode_index → task language string (from the episodes 'tasks' column)."""
    lang: dict[int, str] = {}
    if "tasks" not in episodes_meta.columns:
        return lang
    for _, row in episodes_meta.iterrows():
        ts = row["tasks"]
        if isinstance(ts, (list, tuple, np.ndarray)):
            ts = ts[0] if len(ts) else ""
        lang[int(row["episode_index"])] = str(ts)
    return lang


def _task_language_map(dataset_dir: Path) -> dict[int, str]:
    """task_index → task language string (fallback when episode 'tasks' is absent)."""
    import pandas as pd  # noqa: PLC0415
    p = dataset_dir / "meta" / "tasks.parquet"
    if not p.exists():
        return {}
    tm = pd.read_parquet(p).reset_index()
    return {int(r["task_index"]): str(r["task"]) for _, r in tm.iterrows()}


def _read_file_start_frames(path: Path, targets: list[tuple[int, int]], fps: float):
    """Decode one shared mp4 ONCE, returning only the requested absolute frames.

    LIBERO packs many episodes per mp4; the absolute frame for a skill is
    round(from_timestamp*fps) + episode-relative frame_start. targets: (skill_idx,
    absolute_frame). Picks frames by presentation time (round(frame.time * fps)) so it
    is robust to B-frame decode reordering, and stops once every target is collected.
    Returns [(skill_idx, rgb_uint8 (H,W,3))].
    """
    import av  # noqa: PLC0415
    from collections import defaultdict  # noqa: PLC0415

    want: dict[int, list[int]] = defaultdict(list)
    for si, af in targets:
        want[af].append(si)
    max_t = (max(want) + 2) / fps
    out: list[tuple[int, np.ndarray]] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            if frame.time > max_t:
                break
            fidx = int(round(frame.time * fps))
            if fidx in want:
                img = frame.to_ndarray(format="rgb24")
                for si in want.pop(fidx):
                    out.append((si, img))
                if not want:
                    break
    return out


# ── args ──────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    skills_dir: str
    """Directory containing skill .npz files."""
    dataset_dir: str
    """LeRobot dataset directory (videos/ + meta/)."""
    output_path: str
    """Output npz path. In parallel mode, partial files are written alongside."""
    image_key: str = "observation.images.image"
    image_model_name: str = "/data2/dohyeon/SBD/models/dinov3-vits16"
    """Local path or HF model ID for the frozen visual backbone."""
    image_size: int = 224
    patch_grid: int = 8
    """Pool DINOv3 14×14 patches to patch_grid × patch_grid (default 8→64 patch tokens + 1 CLS = 65)."""
    n_patch_raw: int = 196
    """Number of raw patch tokens from the backbone (14×14=196 for vits16)."""
    batch_size: int = 256
    device: str = "cuda"
    checkpoint_every: int = 50
    """Save per-episode checkpoint every N episodes (0 = disable)."""
    num_workers: int = 1
    worker_id: int = 0
    wandb_project: str = "dino-precompute"


# ── skill metadata ────────────────────────────────────────────────────────────

def load_skill_metadata(skills_dir: Path) -> list[dict]:
    npz_files = sorted(skills_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {skills_dir}")
    meta = []
    for f in npz_files:
        d = np.load(str(f))
        meta.append({
            "file":        str(f),
            "episode_id":  int(d["episode_id"]),
            "task_id":     int(d["task_id"]) if "task_id" in d else -1,
            "skill_index": int(d["skill_index"]),
            "frame_start": int(d["frame_start"]),
            "frame_end":   int(d["frame_end"]),
            "length":      len(d["actions"]),
        })
    return meta


# ── DINO model loading ────────────────────────────────────────────────────────

def load_dino_model(model_name: str, image_size: int, device: str):
    """Returns (model, mean, std) ready for inference."""
    from transformers import AutoImageProcessor, AutoModel
    model = AutoModel.from_pretrained(model_name)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().to(device)

    try:
        proc = AutoImageProcessor.from_pretrained(model_name)
        mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
        std  = getattr(proc, "image_std",  [0.229, 0.224, 0.225])
    except Exception:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return model, mean_t, std_t


# ── feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def encode_frames(
    model,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    frames: np.ndarray,       # (T, H, W, 3) uint8
    batch_size: int,
    image_size: int,
    device: str,
    device_type: str,
    patch_grid: int,          # pool to patch_grid × patch_grid
    n_patch_raw: int,         # raw patch count from backbone (e.g. 196)
) -> np.ndarray:
    """Returns (T, 1 + patch_grid^2, feat_dim) float32.

    CLS token is at index 0; pooled patch tokens follow at indices 1..patch_grid^2.
    """
    pool = torch.nn.AdaptiveAvgPool2d((patch_grid, patch_grid))
    sqrt_raw = int(n_patch_raw ** 0.5)   # 14 for 196
    all_tokens: list[np.ndarray] = []

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

        last_hidden = out.last_hidden_state.float()  # (B, n_total_tokens, feat_dim)
        B, _, feat_dim = last_hidden.shape

        cls_tok     = last_hidden[:, :1, :]               # (B, 1, feat_dim)
        patch_tok   = last_hidden[:, -n_patch_raw:, :]    # (B, 196, feat_dim)

        # spatial pool: (B, 196, F) → (B, F, 14, 14) → pool → (B, F, 8, 8) → (B, 64, F)
        patch_spatial = patch_tok.reshape(B, sqrt_raw, sqrt_raw, feat_dim).permute(0, 3, 1, 2)
        patch_pooled  = pool(patch_spatial).permute(0, 2, 3, 1).reshape(B, patch_grid * patch_grid, feat_dim)

        tokens = torch.cat([cls_tok, patch_pooled], dim=1)  # (B, 1+patch_grid^2, feat_dim)
        all_tokens.append(tokens.cpu().numpy().astype(np.float32))

    return np.concatenate(all_tokens, axis=0)  # (T, n_tokens, feat_dim)


def fit_feature_length(features: np.ndarray, length: int) -> np.ndarray:
    """Trim or pad-by-repeat the temporal axis to `length`."""
    if len(features) == length:
        return features
    if len(features) > length:
        return features[:length]
    pad = np.repeat(features[-1:], length - len(features), axis=0)
    return np.concatenate([features, pad], axis=0)


# ── checkpointing ─────────────────────────────────────────────────────────────

def _ckpt_path(ckpt_dir: Path, ep_id: int) -> Path:
    return ckpt_dir / f"ep_{ep_id:07d}.npz"


def _save_checkpoint(ckpt_dir: Path, ep_id: int, skill_ids: list[int], chunks: list) -> None:
    data = {f"feat_{i}": chunks[i] for i in skill_ids if chunks[i] is not None}
    np.savez(str(_ckpt_path(ckpt_dir, ep_id)), **data)


def _load_checkpoints(ckpt_dir: Path, by_episode: dict, chunks: list) -> set[int]:
    done: set[int] = set()
    for ep_id, skill_ids in by_episode.items():
        cp = _ckpt_path(ckpt_dir, ep_id)
        if not cp.exists():
            continue
        try:
            data = np.load(str(cp))
            for i in skill_ids:
                key = f"feat_{i}"
                if key in data:
                    chunks[i] = data[key]
            done.add(ep_id)
        except Exception:
            pass
    return done


# ── partial-file helpers (parallel mode) ─────────────────────────────────────

def _partial_path(output_path: Path, worker_id: int, num_workers: int) -> Path:
    return output_path.parent / f"{output_path.stem}_part{worker_id:02d}of{num_workers:02d}.npz"


def _save_partial(path: Path, chunks: list, metadata: list) -> None:
    skill_indices = [i for i, f in enumerate(chunks) if f is not None]
    if not skill_indices:
        return
    all_feats = np.concatenate([chunks[i] for i in skill_indices], axis=0).astype(np.float16)
    lengths   = np.array([len(chunks[i]) for i in skill_indices], dtype=np.int64)
    offsets   = np.concatenate([[0], np.cumsum(lengths)])
    np.savez(str(path),
             skill_indices=np.array(skill_indices, dtype=np.int64),
             features=all_feats,
             offsets=offsets)
    print(f"[DINO] Partial save: {len(skill_indices)} skills → {path}  shape={all_feats.shape}")


def _save_npz(output_path: Path, features: np.ndarray, offsets: list, metadata: list,
              image_key: str, model_name: str, patch_grid: int) -> None:
    np.savez(
        str(output_path),
        features=features.astype(np.float16),
        offsets=np.array(offsets, dtype=np.int64),
        episode_id  =np.array([m["episode_id"]  for m in metadata], dtype=np.int64),
        task_id     =np.array([m["task_id"]     for m in metadata], dtype=np.int64),
        skill_index =np.array([m["skill_index"] for m in metadata], dtype=np.int64),
        frame_start =np.array([m["frame_start"] for m in metadata], dtype=np.int64),
        frame_end   =np.array([m["frame_end"]   for m in metadata], dtype=np.int64),
        length      =np.array([m["length"]      for m in metadata], dtype=np.int64),
        image_key        =np.array(image_key),
        image_model_name =np.array(model_name),
        n_tokens         =np.array(1 + patch_grid * patch_grid, dtype=np.int64),
        feat_dim         =np.array(features.shape[-1], dtype=np.int64),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    skills_dir  = Path(args.skills_dir)
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parallel = args.num_workers > 1
    if parallel:
        print(f"[DINO] Parallel mode: worker {args.worker_id}/{args.num_workers}")

    ckpt_dir     = output_path.parent / f"{output_path.stem}_ckpt_w{args.worker_id}"
    use_ckpt     = args.checkpoint_every > 0
    if use_ckpt:
        ckpt_dir.mkdir(exist_ok=True)

    metadata      = load_skill_metadata(skills_dir)
    episodes_meta = _load_episodes_meta(dataset_dir)
    image_key     = _resolve_image_key(episodes_meta, args.image_key)
    device        = args.device if torch.cuda.is_available() else "cpu"
    device_type   = device.split(":")[0]

    n_tokens = 1 + args.patch_grid ** 2  # CLS + pooled patches
    print(f"[DINO] skills={len(metadata)}  model={args.image_model_name}")
    print(f"[DINO] output tokens per frame: {n_tokens} (1 CLS + {args.patch_grid}×{args.patch_grid} pooled patches)")
    print(f"[DINO] device={device}  batch_size={args.batch_size}")

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            group = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
            wandb_run = wandb.init(project=args.wandb_project, group=group,
                                   name=f"worker_{args.worker_id}", config=vars(args), resume="allow")
        except Exception as e:
            print(f"[DINO] wandb init failed: {e}")

    model, mean_t, std_t = load_dino_model(args.image_model_name, args.image_size, device)
    if device_type == "cuda" and hasattr(torch, "compile"):
        print("[DINO] torch.compile ...")
        model = torch.compile(model)

    by_episode: dict[int, list[int]] = {}
    for idx, m in enumerate(metadata):
        by_episode.setdefault(int(m["episode_id"]), []).append(idx)

    all_episode_items = list(by_episode.items())
    my_items = [item for i, item in enumerate(all_episode_items) if i % args.num_workers == args.worker_id]

    print(f"[DINO] total episodes={len(by_episode)}, this worker={len(my_items)}")

    chunks: list[np.ndarray | None] = [None] * len(metadata)
    done_episodes: set[int] = set()
    if use_ckpt:
        done_episodes = _load_checkpoints(ckpt_dir, by_episode, chunks)
        if done_episodes:
            print(f"[DINO] Resuming: {len(done_episodes)} episodes already done")

    remaining = [(ep, sids) for ep, sids in my_items if ep not in done_episodes]

    total_vframes = total_sframes = 0
    t0 = time.perf_counter()

    def load_frames(ep_id: int) -> np.ndarray:
        return _read_video_frames(_video_path(dataset_dir, episodes_meta, ep_id, image_key))

    progress = tqdm(remaining, desc=f"[w{args.worker_id}] DINO tokens",
                    total=len(my_items), initial=len(done_episodes))

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(load_frames, remaining[0][0]) if remaining else None

        for i, (ep, skill_ids) in enumerate(remaining):
            frames = future.result()
            if i + 1 < len(remaining):
                future = executor.submit(load_frames, remaining[i + 1][0])

            ep_tokens = encode_frames(
                model, mean_t, std_t, frames,
                args.batch_size, args.image_size, device, device_type,
                args.patch_grid, args.n_patch_raw,
            )
            total_vframes += len(frames)

            for idx in skill_ids:
                m = metadata[idx]
                fs, fe = int(m["frame_start"]), int(m["frame_end"])
                clip = ep_tokens[fs : min(fe, len(ep_tokens))]
                chunks[idx] = fit_feature_length(clip, int(m["length"]))
                total_sframes += int(m["length"])

            if use_ckpt and (i + 1) % args.checkpoint_every == 0:
                _save_checkpoint(ckpt_dir, ep, skill_ids, chunks)

            elapsed    = max(time.perf_counter() - t0, 1e-6)
            done_count = len(done_episodes) + i + 1
            eta_h      = (len(my_items) - done_count) * (elapsed / (i + 1)) / 3600
            progress.set_postfix({"fps": f"{total_vframes/elapsed:.0f}", "eta": f"{eta_h:.1f}h"})
            progress.update(1)

            if wandb_run is not None:
                wandb_run.log({
                    "episodes_done": done_count,
                    "progress_pct": 100.0 * done_count / len(my_items),
                    "video_fps": total_vframes / elapsed,
                    "eta_hours": eta_h,
                })

    if use_ckpt:
        for ep, skill_ids in remaining:
            if not _ckpt_path(ckpt_dir, ep).exists():
                _save_checkpoint(ckpt_dir, ep, skill_ids, chunks)

    if parallel:
        partial_path = _partial_path(output_path, args.worker_id, args.num_workers)
        _save_partial(partial_path, chunks, metadata)
        elapsed = max(time.perf_counter() - t0, 1e-6)
        print(f"[DINO] Worker {args.worker_id} done. video={total_vframes} ({total_vframes/elapsed:.1f} fps)")
        if wandb_run is not None:
            wandb_run.finish()
        return

    # single-worker: build full output
    ordered: list[np.ndarray] = []
    offsets = [0]
    for idx, feat in enumerate(chunks):
        if feat is None:
            raise RuntimeError(f"Missing features for skill index {idx}")
        ordered.append(feat)
        offsets.append(offsets[-1] + len(feat))

    features = np.concatenate(ordered, axis=0)
    _save_npz(output_path, features, offsets, metadata, image_key, args.image_model_name, args.patch_grid)

    if use_ckpt and ckpt_dir.exists():
        for cp in ckpt_dir.glob("ep_*.npz"):
            cp.unlink()
        try:
            ckpt_dir.rmdir()
        except OSError:
            pass

    elapsed = max(time.perf_counter() - t0, 1e-6)
    print(f"[DINO] video frames={total_vframes} ({total_vframes/elapsed:.2f} fps)")
    print(f"[DINO] Saved features {features.shape} (float16) → {output_path}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
