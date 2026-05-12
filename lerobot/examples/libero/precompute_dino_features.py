"""
Precompute frozen DINO/DINOv3 features for VQ-VAE skill decoder training.

The output npz is ordered exactly like sorted(skills_dir.rglob("*.npz")) and
contains flattened per-frame CLS features plus offsets for variable-length
skills. These features are a train-time cache: VQ-VAE/VLA training can read the
cached visual feature directly, while eval/sim can still pass raw camera images
through the same frozen visual encoder inside the model.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import os
import torch
import tyro
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from spline_vqae import FrozenDINOv2Encoder
from train_vae import _load_episodes_meta, _resolve_image_key, _video_path


def _read_video_frames(path: Path) -> np.ndarray:
    """Read all frames via torchvision (PyAV backend), fallback to imageio."""
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
            raise ValueError(f"No frames decoded from {path}")
        return np.stack(frames).astype(np.uint8)


@dataclass
class Args:
    skills_dir: str
    """Directory containing skill .npz files."""
    dataset_dir: str
    """Original LeRobot dataset directory containing videos and metadata."""
    output_path: str
    """Where to save the feature npz (or partial npz when num_workers > 1)."""
    image_key: str = "observation.images.image"
    image_model_name: str = "/data2/dohyeon/SBD/models/dinov2-small"
    """Local path or Hugging Face id for the frozen visual backbone."""
    image_size: int = 224
    batch_size: int = 512
    device: str = "cuda"
    checkpoint_every: int = 50
    """Save checkpoint every N episodes (0 = disable)."""
    num_workers: int = 1
    """Total number of parallel workers (for multi-job mode)."""
    worker_id: int = 0
    """Index of this worker, 0-indexed (used with num_workers > 1)."""
    wandb_project: str = "dino-precompute"
    """wandb project name. Set to empty string to disable wandb."""


def load_skill_metadata(skills_dir: Path) -> list[dict]:
    npz_files = sorted(skills_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {skills_dir}")

    metadata = []
    for f in npz_files:
        d = np.load(str(f))
        metadata.append({
            "file": str(f),
            "episode_id": int(d["episode_id"]),
            "task_id": int(d["task_id"]) if "task_id" in d else -1,
            "skill_index": int(d["skill_index"]),
            "frame_start": int(d["frame_start"]),
            "frame_end": int(d["frame_end"]),
            "length": len(d["actions"]),
        })
    return metadata


@torch.no_grad()
def encode_frames(
    model: FrozenDINOv2Encoder,
    frames: np.ndarray,
    batch_size: int,
    device: str,
    device_type: str,
) -> np.ndarray:
    feats = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size].astype(np.float32) / 255.0
        batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).unsqueeze(0)
        if device_type == "cuda":
            batch_t = batch_t.pin_memory().to(device, non_blocking=True)
        else:
            batch_t = batch_t.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == "cuda")):
            feat = model(batch_t).squeeze(0)
        feats.append(feat.float().cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


def fit_feature_length(features: np.ndarray, length: int) -> np.ndarray:
    if len(features) == length:
        return features
    if len(features) > length:
        return features[:length]
    if len(features) == 0:
        raise ValueError("Cannot pad an empty DINO feature clip")
    pad = np.repeat(features[-1:], length - len(features), axis=0)
    return np.concatenate([features, pad], axis=0)


def _checkpoint_path(checkpoint_dir: Path, ep_id: int) -> Path:
    return checkpoint_dir / f"ep_{ep_id:07d}.npz"


def _save_checkpoint(checkpoint_dir: Path, ep_id: int, skill_ids: list[int], feature_chunks: list) -> None:
    data = {f"feat_{idx}": feature_chunks[idx] for idx in skill_ids if feature_chunks[idx] is not None}
    np.savez(str(_checkpoint_path(checkpoint_dir, ep_id)), **data)


def _load_checkpoints(checkpoint_dir: Path, by_episode: dict, feature_chunks: list) -> set[int]:
    done = set()
    for ep_id, skill_ids in by_episode.items():
        cp = _checkpoint_path(checkpoint_dir, ep_id)
        if not cp.exists():
            continue
        try:
            data = np.load(str(cp))
            for idx in skill_ids:
                key = f"feat_{idx}"
                if key in data:
                    feature_chunks[idx] = data[key]
            done.add(ep_id)
        except Exception:
            pass
    return done


def _partial_output_path(output_path: Path, worker_id: int, num_workers: int) -> Path:
    return output_path.parent / f"{output_path.stem}_part{worker_id:02d}of{num_workers:02d}.npz"


def _save_partial(path: Path, feature_chunks: list, metadata: list) -> None:
    """Save this worker's features indexed by global skill index."""
    skill_indices = [i for i, f in enumerate(feature_chunks) if f is not None]
    if not skill_indices:
        return
    all_feats = np.concatenate([feature_chunks[i] for i in skill_indices], axis=0)
    lengths = np.array([len(feature_chunks[i]) for i in skill_indices], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    np.savez(
        str(path),
        skill_indices=np.array(skill_indices, dtype=np.int64),
        features=all_feats.astype(np.float32),
        offsets=offsets,
    )
    print(f"[DINO] Partial save: {len(skill_indices)} skills → {path}")


def main(args: Args) -> None:
    skills_dir = Path(args.skills_dir)
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parallel = args.num_workers > 1
    if parallel:
        print(f"[DINO] Parallel mode: worker {args.worker_id}/{args.num_workers}")

    checkpoint_dir = output_path.parent / f"{output_path.stem}_ckpt_w{args.worker_id}"
    use_checkpoint = args.checkpoint_every > 0
    if use_checkpoint:
        checkpoint_dir.mkdir(exist_ok=True)

    metadata = load_skill_metadata(skills_dir)
    episodes_meta = _load_episodes_meta(dataset_dir)
    image_key = _resolve_image_key(episodes_meta, args.image_key)
    device = args.device if torch.cuda.is_available() else "cpu"
    device_type = device.split(":")[0]

    # wandb — group all workers under the same SLURM array job ID
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            group = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
            wandb_run = wandb.init(
                project=args.wandb_project,
                group=group,
                name=f"worker_{args.worker_id}",
                config=vars(args),
                resume="allow",
            )
            print(f"[DINO] wandb: project={args.wandb_project} group={group} run={wandb_run.name}")
        except Exception as e:
            print(f"[DINO] wandb init failed (continuing without it): {e}")

    print(f"[DINO] skills={len(metadata)}")
    print(f"[DINO] dataset={dataset_dir}")
    print(f"[DINO] camera={image_key}")
    print(f"[DINO] model={args.image_model_name}")
    print(f"[DINO] device={device}")
    print(f"[DINO] batch_size={args.batch_size}")

    model = FrozenDINOv2Encoder(args.image_model_name, args.image_size).to(device).eval()
    if device_type == "cuda" and hasattr(torch, "compile"):
        print("[DINO] Compiling model with torch.compile ...")
        model = torch.compile(model)

    by_episode: dict[int, list[int]] = {}
    for idx, m in enumerate(metadata):
        by_episode.setdefault(int(m["episode_id"]), []).append(idx)

    # Assign episodes round-robin for load balancing
    all_episode_items = list(by_episode.items())
    my_episode_items = [item for i, item in enumerate(all_episode_items) if i % args.num_workers == args.worker_id]

    print(f"[DINO] total episodes={len(by_episode)}, this worker={len(my_episode_items)}")
    print("[DINO] encoding each episode video once, then slicing per-skill features")

    feature_chunks: list[np.ndarray | None] = [None] * len(metadata)

    done_episodes: set[int] = set()
    if use_checkpoint:
        done_episodes = _load_checkpoints(checkpoint_dir, by_episode, feature_chunks)
        if done_episodes:
            print(f"[DINO] Resuming: {len(done_episodes)} episodes already done")

    remaining = [(ep, sids) for ep, sids in my_episode_items if ep not in done_episodes]

    total_video_frames = 0
    total_skill_frames = 0
    start_time = time.perf_counter()

    def load_frames(ep_id: int) -> np.ndarray:
        return _read_video_frames(_video_path(dataset_dir, episodes_meta, ep_id, image_key))

    progress = tqdm(
        remaining,
        desc=f"[worker {args.worker_id}] DINO features",
        total=len(my_episode_items),
        initial=len(done_episodes),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(load_frames, remaining[0][0]) if remaining else None

        for i, (ep, skill_ids) in enumerate(remaining):
            frames = future.result()

            if i + 1 < len(remaining):
                future = executor.submit(load_frames, remaining[i + 1][0])

            episode_features = encode_frames(model, frames, args.batch_size, device, device_type)
            total_video_frames += len(frames)

            for idx in skill_ids:
                m = metadata[idx]
                fs, fe = int(m["frame_start"]), int(m["frame_end"])
                clip_feat = episode_features[fs : min(fe, len(episode_features))]
                feature_chunks[idx] = fit_feature_length(clip_feat, int(m["length"]))
                total_skill_frames += int(m["length"])

            if use_checkpoint and (i + 1) % args.checkpoint_every == 0:
                _save_checkpoint(checkpoint_dir, ep, skill_ids, feature_chunks)

            elapsed = max(time.perf_counter() - start_time, 1e-6)
            episodes_done = len(done_episodes) + i + 1
            video_fps = total_video_frames / elapsed
            skill_fps = total_skill_frames / elapsed
            sec_per_ep = elapsed / (i + 1)
            remaining_eps = len(my_episode_items) - episodes_done
            eta_h = remaining_eps * sec_per_ep / 3600

            progress.set_postfix({
                "video_fps": f"{video_fps:.0f}",
                "skill_fps": f"{skill_fps:.1f}",
                "eta": f"{eta_h:.1f}h",
            })
            progress.update(1)

            if wandb_run is not None:
                wandb_run.log({
                    "episodes_done": episodes_done,
                    "episodes_total": len(my_episode_items),
                    "progress_pct": 100.0 * episodes_done / len(my_episode_items),
                    "video_fps": video_fps,
                    "skill_fps": skill_fps,
                    "sec_per_episode": sec_per_ep,
                    "eta_hours": eta_h,
                })

    if use_checkpoint:
        for ep, skill_ids in remaining:
            if not _checkpoint_path(checkpoint_dir, ep).exists():
                _save_checkpoint(checkpoint_dir, ep, skill_ids, feature_chunks)

    if parallel:
        partial_path = _partial_output_path(output_path, args.worker_id, args.num_workers)
        _save_partial(partial_path, feature_chunks, metadata)
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        print(f"[DINO] Worker {args.worker_id} done. video={total_video_frames} ({total_video_frames/elapsed:.1f} Hz)")
        if wandb_run is not None:
            wandb_run.finish()
        return

    # Single-worker: build and save full output
    ordered_chunks: list[np.ndarray] = []
    offsets = [0]
    for idx, feat in enumerate(feature_chunks):
        if feat is None:
            raise RuntimeError(f"Missing DINO features for skill index {idx}")
        ordered_chunks.append(feat)
        offsets.append(offsets[-1] + len(feat))

    features = np.concatenate(ordered_chunks, axis=0).astype(np.float32)
    _save_npz(output_path, features, offsets, metadata, image_key, args.image_model_name)

    if use_checkpoint and checkpoint_dir.exists():
        for cp in checkpoint_dir.glob("ep_*.npz"):
            cp.unlink()
        try:
            checkpoint_dir.rmdir()
        except OSError:
            pass

    elapsed = max(time.perf_counter() - start_time, 1e-6)
    print(f"[DINO] Encoded video frames={total_video_frames} ({total_video_frames / elapsed:.2f} Hz)")
    print(f"[DINO] Saved skill frames={features.shape[0]} ({features.shape[0] / elapsed:.2f} Hz)")
    print(f"[DINO] Saved {features.shape} -> {output_path}")
    if wandb_run is not None:
        wandb_run.finish()


def _save_npz(output_path: Path, features: np.ndarray, offsets: list, metadata: list, image_key: str, model_name: str) -> None:
    np.savez(
        str(output_path),
        features=features,
        offsets=np.array(offsets, dtype=np.int64),
        episode_id=np.array([m["episode_id"] for m in metadata], dtype=np.int64),
        task_id=np.array([m["task_id"] for m in metadata], dtype=np.int64),
        skill_index=np.array([m["skill_index"] for m in metadata], dtype=np.int64),
        frame_start=np.array([m["frame_start"] for m in metadata], dtype=np.int64),
        frame_end=np.array([m["frame_end"] for m in metadata], dtype=np.int64),
        length=np.array([m["length"] for m in metadata], dtype=np.int64),
        image_key=np.array(image_key),
        image_model_name=np.array(model_name),
        feature_dim=np.array(features.shape[-1], dtype=np.int64),
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
