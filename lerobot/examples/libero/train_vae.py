"""
Skill VAE Training Script.

Loads skill segment files (.npz) produced by replay_demo.py --save_skills
and trains a BiLSTM VAE (defined in skill_vae.py).

Each .npz file contains:
    actions  : (T, action_dim)  float32
    episode_id, skill_index, frame_start, frame_end  (scalar metadata)

After training, saves:
    <output_dir>/skill_vae.pt        — model weights + config
    <output_dir>/skill_latents.npz   — latent codes for every segment

Usage:
    python examples/libero/train_vae.py \\
      --skills_dir /path/to/replay_output/skills \\
      --output_dir /path/to/vae_output \\
      --latent_dim 64 --hidden_dim 256 --epochs 200 --beta 1.0
    
    python examples/libero/train_vae.py \
    --skills_dir /scratch/mdorazi/Skill_Boundary_Detector/outputs/replay_libero10 \
    --output_dir /scratch/mdorazi/Skill_Boundary_Detector/outputs/vae \
    --latent_dim 64 --hidden_dim 256 --epochs 200 --beta 1.0
    
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).parent))
from skill_vae import VAEConfig, encode_skills, train_skill_vae


# ── Args ───────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    skills_dir: str = ""
    """Directory containing .npz skill files output by replay_demo.py."""
    output_dir: str = ""
    """Where to save skill_vae.pt and skill_latents.npz. Defaults to skills_dir/.."""

    # Model
    latent_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1

    # Training
    epochs: int = 200
    beta: float = 1.0
    """KL weight in the ELBO loss."""
    lr: float = 3e-4
    batch_size: int = 32
    grad_clip: float = 1.0
    val_split: float = 0.1
    log_every: int = 10
    seed: int = 42
    device: str = "cuda"
    min_skill_len: int = 20
    """Filter out skill segments shorter than this (frames)."""
    checkpoint_every: int = 0
    """Save a checkpoint every N epochs. 0 = disabled."""
    wandb_project: str | None = None
    """If set, log training loss to this wandb project."""
    wandb_run_name: str = "skill_vae"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_skill_files(
    skills_dir: Path,
) -> tuple[list[np.ndarray], list[dict]]:
    """Load all .npz skill files. Returns (segments, metadata_list).

    segments     : list of action numpy arrays
    metadata_list: list of dicts with episode_id, skill_index, frame_start, frame_end
    """
    npz_files = sorted(skills_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {skills_dir}")

    segments = []
    metadata = []
    for f in npz_files:
        d = np.load(str(f))
        segments.append(d["actions"])
        metadata.append({
            "file": str(f.name),
            "episode_id": int(d["episode_id"]),
            "skill_index": int(d["skill_index"]),
            "frame_start": int(d["frame_start"]),
            "frame_end": int(d["frame_end"]),
            "length": len(d["actions"]),
        })

    print(f"[VAE] Loaded {len(segments)} skill segments from {skills_dir}")
    lengths = [m["length"] for m in metadata]
    print(f"[VAE] Skill lengths — min: {min(lengths)}, max: {max(lengths)}, mean: {np.mean(lengths):.1f}")
    return segments, metadata


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    skills_dir = Path(args.skills_dir)
    output_dir = Path(args.output_dir) if args.output_dir else skills_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    segments, metadata = load_skill_files(skills_dir)

    if args.min_skill_len > 0:
        before = len(segments)
        segments, metadata = zip(*[
            (s, m) for s, m in zip(segments, metadata) if m["length"] >= args.min_skill_len
        ])
        segments, metadata = list(segments), list(metadata)
        print(f"[VAE] Filtered short skills: {before} → {len(segments)} (min_len={args.min_skill_len})")

    example_a = segments[0]
    action_dim = example_a.shape[-1]

    cfg = VAEConfig(
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        beta=args.beta,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        epochs=args.epochs,
        val_split=args.val_split,
        log_every=args.log_every,
        device=args.device if torch.cuda.is_available() else "cpu",
        save_path=str(output_dir / "skill_vae.pt"),
        checkpoint_every=args.checkpoint_every,
    )

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "action_dim": action_dim,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "beta": args.beta,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "n_segments": len(segments),
            },
        )

    model = train_skill_vae(segments, cfg, wandb_run=wandb_run, metadata=metadata)

    # Encode all segments with the trained model
    print("[VAE] Encoding all skill segments...")
    latent_codes = encode_skills(model, segments, device="cpu")  # (N, latent_dim)
    print(f"[VAE] Latent codes: {latent_codes.shape}")

    # Save latent codes alongside metadata
    latents_path = output_dir / "skill_latents.npz"
    save_dict = {"latents": latent_codes}
    for key in ("episode_id", "skill_index", "frame_start", "frame_end", "length"):
        save_dict[key] = np.array([m[key] for m in metadata])
    np.savez(str(latents_path), **save_dict)
    print(f"[VAE] Saved latents + metadata → {latents_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
