"""
Train a spline-encoded VQ-VAE skill tokenizer with an observation-conditioned RNN decoder.

Usage:
    python examples/libero/train_vae.py \
      --skills_dir /path/to/skills \
      --dataset_dir /path/to/original_lerobot_dataset \
      --output_dir /path/to/output \
      --latent_dim 64 --num_embeddings 512 --epochs 5000
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).parent))


# ── Args ───────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    skills_dir: str = ""
    """Directory containing .npz skill files."""
    output_dir: str = ""
    """Where to save model and latents. Defaults to skills_dir/.."""

    # Model
    latent_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    n_control: int = 30
    spline_degree: int = 3
    num_embeddings: int = 512
    commitment_cost: float = 0.25
    encoder_type: str = "mlp"
    """Skill encoder type: 'mlp', 'gru', or 'bilstm'."""
    decoder_rnn_type: str = "mlp"
    """Decoder type: 'mlp' per-step decoder or vanilla 'rnn'."""
    delta_loss_weight: float = 1.0
    """Weight for the reconstruction loss."""
    end_loss_weight: float = 10.0
    """Weight for the end-signal BCE loss."""
    end_pos_weight: float = 0.0
    """Positive class weight for end-signal BCE. <=0 computes negative/positive ratio from skill lengths."""
    end_threshold: float = 0.5
    """Threshold used only for end-signal metrics/eval decisions, not for the BCE loss."""
    max_length: float = 200.0 # 500
    eef_dims: list[int] | None = None
    """State dimensions used as absolute EEF pose. Defaults to [0,1,2,3,4,5]."""
    gripper_action_dim: int = -1
    """Action dimension used as gripper signal."""
    zero_start_eef: bool = True
    """If true, subtract the first EEF pose from encoder trajectories (gripper unchanged)."""
    use_images: bool = True
    """Use visual conditioning in the decoder."""
    dataset_dir: str = ""
    """Original LeRobot dataset directory used to fetch image frames by episode/frame metadata."""
    image_key: str = "observation.images.image"
    """Camera key under dataset_dir/videos to use for visual decoder conditioning."""
    image_features_path: str = ""
    """Optional precomputed visual feature npz. If set, train_vae does not load raw videos."""
    image_model_name: str = "facebook/dinov2-small"
    image_size: int = 224
    image_feature_dim: int = 384

    # Training
    epochs: int = 5000
    lr: float = 3e-4
    batch_size: int = 64
    grad_clip: float = 1.0
    val_split: float = 0.1
    log_every: int = 10
    seed: int = 42
    device: str = "cuda"
    checkpoint_every: int = 0
    resume_from: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str = "spline_vqae"


# ── Data loading ───────────────────────────────────────────────────────────────

def _compute_skill_orders(metadata: list[dict]) -> list[float]:
    """Return BOS-aware normalized skill order.

    BOS occupies order 0.0.  For N real skills, ranks are mapped to
    (rank + 1) / N, so the first real skill is > 0 and the last is 1.0.
    """
    by_ep: dict[int, list[int]] = {}
    for i, m in enumerate(metadata):
        by_ep.setdefault(int(m["episode_id"]), []).append(i)

    orders = [0.0] * len(metadata)
    for ids in by_ep.values():
        ids.sort(key=lambda i: (metadata[i]["frame_start"], metadata[i]["frame_end"], metadata[i]["skill_index"]))
        denom = max(1, len(ids))
        for rank, idx in enumerate(ids):
            orders[idx] = float((rank + 1) / denom)
    return orders


def _read_video_frames(path: Path) -> np.ndarray:
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


def _load_episodes_meta(dataset_dir: Path):
    import pandas as pd

    files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {dataset_dir / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _resolve_image_key(episodes_meta, image_key: str) -> str:
    video_cols = [c for c in episodes_meta.columns if c.startswith("videos/") and c.endswith("/chunk_index")]
    keys = [c.split("/")[1] for c in video_cols]
    if image_key:
        if image_key not in keys:
            raise ValueError(f"image_key={image_key!r} not found. Available camera keys: {keys}")
        return image_key
    if not keys:
        raise ValueError("No video camera keys found in episode metadata.")
    return keys[0]


def _video_path(dataset_dir: Path, episodes_meta, episode_id: int, image_key: str) -> Path:
    row = episodes_meta[episodes_meta["episode_index"] == episode_id].iloc[0]
    chunk_idx = int(row[f"videos/{image_key}/chunk_index"])
    file_idx = int(row[f"videos/{image_key}/file_index"])
    return dataset_dir / "videos" / image_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def _fit_image_length(clip: np.ndarray, length: int) -> np.ndarray:
    if len(clip) == length:
        return clip
    if len(clip) <= 0:
        raise ValueError("Cannot fit an empty image clip.")
    if len(clip) > length:
        return clip[:length]
    pad = np.repeat(clip[-1:], length - len(clip), axis=0)
    return np.concatenate([clip, pad], axis=0)


def load_skill_images(dataset_dir: Path, metadata: list[dict], image_key: str) -> list[np.ndarray]:
    episodes_meta = _load_episodes_meta(dataset_dir)
    image_key = _resolve_image_key(episodes_meta, image_key)
    cache: dict[tuple[int, str], np.ndarray] = {}
    images: list[np.ndarray] = []
    print(f"[VQAE] Loading image clips from {dataset_dir} camera={image_key}")
    for m in metadata:
        ep = int(m["episode_id"])
        key = (ep, image_key)
        if key not in cache:
            cache[key] = _read_video_frames(_video_path(dataset_dir, episodes_meta, ep, image_key))
        frames = cache[key]
        fs, fe = int(m["frame_start"]), int(m["frame_end"])
        clip = frames[fs:min(fe, len(frames))]
        images.append(_fit_image_length(clip, int(m["length"])))
    return images


def load_skill_image_features(features_path: Path, metadata: list[dict]) -> list[np.ndarray]:
    d = np.load(str(features_path), allow_pickle=False)
    required = {"features", "offsets", "episode_id", "frame_start", "frame_end", "length"}
    missing = required - set(d.files)
    if missing:
        raise ValueError(f"{features_path} is missing keys: {sorted(missing)}")

    features = d["features"].astype(np.float32)
    offsets = d["offsets"].astype(np.int64)
    if len(offsets) != len(metadata) + 1:
        raise ValueError(f"Feature count mismatch: {len(offsets) - 1} features vs {len(metadata)} skills")

    for i, m in enumerate(metadata):
        got = (
            int(d["episode_id"][i]),
            int(d["frame_start"][i]),
            int(d["frame_end"][i]),
            int(d["length"][i]),
        )
        expected = (
            int(m["episode_id"]),
            int(m["frame_start"]),
            int(m["frame_end"]),
            int(m["length"]),
        )
        if got != expected:
            raise ValueError(f"Feature metadata mismatch at index {i}: got {got}, expected {expected}")

    clips = [features[offsets[i]:offsets[i + 1]] for i in range(len(metadata))]
    print(f"[VQAE] Loaded precomputed DINO features from {features_path} ({features.shape[0]} frames)")
    return clips


def _make_target_delta(states: np.ndarray, actions: np.ndarray, eef_dims: list[int], gripper_idx: int) -> np.ndarray:
    pose = states[:, eef_dims].astype(np.float32)
    delta = np.zeros_like(pose, dtype=np.float32)
    if len(pose) > 1:
        delta[:-1] = pose[1:] - pose[:-1]
    gripper = actions[:, gripper_idx:gripper_idx + 1].astype(np.float32)
    return np.concatenate([delta, gripper], axis=-1)


def _make_encoder_traj(
    states: np.ndarray,
    actions: np.ndarray,
    eef_dims: list[int],
    gripper_idx: int,
    zero_start: bool,
) -> np.ndarray:
    pose = states[:, eef_dims].astype(np.float32)
    if zero_start:
        pose = pose - pose[:1]
    gripper = actions[:, gripper_idx:gripper_idx + 1].astype(np.float32)
    return np.concatenate([pose, gripper], axis=-1)


def load_skill_files(
    skills_dir: Path,
    eef_dims: list[int],
    gripper_action_dim: int,
    zero_start_eef: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict], list[float]]:
    npz_files = sorted(skills_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {skills_dir}")

    segments, decoder_states, decoder_targets, metadata = [], [], [], []
    for f in npz_files:
        d = np.load(str(f))
        actions = d["actions"].astype(np.float32)
        states = d["states"].astype(np.float32)
        gripper_idx = (actions.shape[-1] + gripper_action_dim) % actions.shape[-1]
        segments.append(_make_encoder_traj(states, actions, eef_dims, gripper_idx, zero_start_eef))
        decoder_states.append(states[:, eef_dims].astype(np.float32))
        decoder_targets.append(_make_target_delta(states, actions, eef_dims, gripper_idx))
        metadata.append({
            "file": str(f.name),
            "episode_id": int(d["episode_id"]),
            "task_id": int(d["task_id"]) if "task_id" in d else -1,
            "skill_index": int(d["skill_index"]),
            "frame_start": int(d["frame_start"]),
            "frame_end": int(d["frame_end"]),
            "length": len(d["actions"]),
        })

    skill_orders = _compute_skill_orders(metadata)
    print(f"[VQAE] Loaded {len(segments)} skill segments from {skills_dir}")
    lengths = [m["length"] for m in metadata]
    print(f"[VQAE] Skill lengths — min: {min(lengths)}, max: {max(lengths)}, mean: {np.mean(lengths):.1f}")
    print(f"[VQAE] Encoder trajectory: {'zero-start' if zero_start_eef else 'absolute'} EEF + gripper")
    return segments, decoder_states, decoder_targets, metadata, skill_orders


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    from spline_vqae import SplineVQAEConfig, encode_skill_vectors, encode_skills, train_spline_vqae

    skills_dir = Path(args.skills_dir)
    output_dir = Path(args.output_dir) if args.output_dir else skills_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    eef_dims = args.eef_dims if args.eef_dims is not None else [0, 1, 2, 3, 4, 5]
    segments, decoder_states, decoder_targets, metadata, skill_orders = load_skill_files(
        skills_dir,
        eef_dims=eef_dims,
        gripper_action_dim=args.gripper_action_dim,
        zero_start_eef=args.zero_start_eef,
    )

    images = None
    if args.use_images:
        if args.image_features_path:
            images = load_skill_image_features(Path(args.image_features_path), metadata)
        else:
            if not args.dataset_dir:
                raise ValueError("--dataset_dir is required when --use_images is true and --image_features_path is empty.")
            images = load_skill_images(Path(args.dataset_dir), metadata, args.image_key)

    all_encoder = np.concatenate(segments, axis=0)
    action_min = all_encoder.min(axis=0)
    action_max = all_encoder.max(axis=0)
    all_targets = np.concatenate(decoder_targets, axis=0)
    delta_min = all_targets.min(axis=0)
    delta_max = all_targets.max(axis=0)
    np.savez(
        str(output_dir / "action_stats.npz"),
        action_min=action_min,
        action_max=action_max,
        delta_min=delta_min,
        delta_max=delta_max,
        eef_dims=np.array(eef_dims),
        zero_start_eef=np.array(args.zero_start_eef),
    )
    print(f"[VQAE] encoder_min: {np.round(action_min, 4)}")
    print(f"[VQAE] encoder_max: {np.round(action_max, 4)}")
    print(f"[VQAE] delta_min:   {np.round(delta_min, 4)}")
    print(f"[VQAE] delta_max:   {np.round(delta_max, 4)}")

    device = args.device if torch.cuda.is_available() else "cpu"
    action_dim = segments[0].shape[-1]
    state_dim = decoder_states[0].shape[-1]

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **vars(args),
                "action_dim": action_dim,
                "state_dim": state_dim,
                "n_segments": len(segments),
                "eef_dims": eef_dims,
            },
        )

    cfg = SplineVQAEConfig(
        action_dim=action_dim, state_dim=state_dim,
        n_control=args.n_control, spline_degree=args.spline_degree,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        num_layers=args.num_layers, dropout=args.dropout,
        encoder_type=args.encoder_type,
        decoder_rnn_type=args.decoder_rnn_type,
        commitment_cost=args.commitment_cost,
        delta_loss_weight=args.delta_loss_weight,
        end_loss_weight=args.end_loss_weight,
        end_pos_weight=args.end_pos_weight,
        end_threshold=args.end_threshold,
        max_length=args.max_length,
        lr=args.lr, batch_size=args.batch_size, grad_clip=args.grad_clip,
        epochs=args.epochs, val_split=args.val_split, log_every=args.log_every,
        device=device,
        save_path=str(output_dir / "spline_vqae.pt"),
        checkpoint_every=args.checkpoint_every,
        action_min=action_min, action_max=action_max,
        delta_min=delta_min, delta_max=delta_max,
        image_model_name=args.image_model_name,
        image_size=args.image_size,
        image_feature_dim=args.image_feature_dim,
        use_images=args.use_images,
    )
    model = train_spline_vqae(
        segments,
        cfg,
        wandb_run=wandb_run,
        metadata=metadata,
        resume_from=args.resume_from,
        decoder_states=decoder_states,
        decoder_targets=decoder_targets,
        images=images,
        skill_orders=skill_orders,
    )

    latent_codes = encode_skill_vectors(model, segments, device="cpu", skill_orders=skill_orders)
    latent_tokens = encode_skills(model, segments, device="cpu", skill_orders=skill_orders)

    latents_path = output_dir / "skill_latents.npz"
    save_dict = {
        "latents": latent_codes,
        "tokens":  latent_tokens,
        "skill_order": np.array(skill_orders, dtype=np.float32),
    }
    for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"):
        save_dict[key] = np.array([m[key] for m in metadata])
    np.savez(str(latents_path), **save_dict)
    print(f"[VQAE] Saved latents → {latents_path}")
    print(f"[VQAE] Saved model   → {output_dir / 'spline_vqae.pt'}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
