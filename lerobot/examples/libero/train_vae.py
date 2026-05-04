"""
Spline VQAE Training Script.

Trains a deterministic MLP AE with a VQ codebook over spline-encoded skill trajectories.

Usage:
    python examples/libero/train_vae.py \
      --skills_dir /path/to/skills \
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
    length_weight: float = 10.0
    max_length: float = 500.0

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

def load_skill_files(skills_dir: Path) -> tuple[list, list[np.ndarray], list[dict]]:
    npz_files = sorted(skills_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {skills_dir}")

    segments, init_states, metadata = [], [], []
    for f in npz_files:
        d = np.load(str(f))
        segments.append(d["actions"])
        init_states.append(d["states"][0])
        metadata.append({
            "file": str(f.name),
            "episode_id": int(d["episode_id"]),
            "task_id": int(d["task_id"]) if "task_id" in d else -1,
            "skill_index": int(d["skill_index"]),
            "frame_start": int(d["frame_start"]),
            "frame_end": int(d["frame_end"]),
            "length": len(d["actions"]),
        })

    print(f"[VQAE] Loaded {len(segments)} skill segments from {skills_dir}")
    lengths = [m["length"] for m in metadata]
    print(f"[VQAE] Skill lengths — min: {min(lengths)}, max: {max(lengths)}, mean: {np.mean(lengths):.1f}")
    return segments, init_states, metadata


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    from spline_vqae import SplineVQAEConfig, encode_skill_vectors, encode_skills, train_spline_vqae

    skills_dir = Path(args.skills_dir)
    output_dir = Path(args.output_dir) if args.output_dir else skills_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    segments, init_states, metadata = load_skill_files(skills_dir)

    all_actions = np.concatenate(segments, axis=0)
    action_min  = all_actions.min(axis=0)
    action_max  = all_actions.max(axis=0)
    np.savez(str(output_dir / "action_stats.npz"), action_min=action_min, action_max=action_max)
    print(f"[VQAE] action_min: {np.round(action_min, 4)}")
    print(f"[VQAE] action_max: {np.round(action_max, 4)}")

    device     = args.device if torch.cuda.is_available() else "cpu"
    action_dim = segments[0].shape[-1]
    state_dim  = init_states[0].shape[-1]

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), "action_dim": action_dim, "state_dim": state_dim, "n_segments": len(segments)},
        )

    cfg = SplineVQAEConfig(
        action_dim=action_dim, state_dim=state_dim,
        n_control=args.n_control, spline_degree=args.spline_degree,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        num_layers=args.num_layers, dropout=args.dropout,
        commitment_cost=args.commitment_cost,
        length_weight=args.length_weight, max_length=args.max_length,
        lr=args.lr, batch_size=args.batch_size, grad_clip=args.grad_clip,
        epochs=args.epochs, val_split=args.val_split, log_every=args.log_every,
        device=device,
        save_path=str(output_dir / "spline_vqae.pt"),
        checkpoint_every=args.checkpoint_every,
        action_min=action_min, action_max=action_max,
    )
    model = train_spline_vqae(segments, init_states, cfg, wandb_run=wandb_run,
                              metadata=metadata, resume_from=args.resume_from)

    latent_codes  = encode_skill_vectors(model, segments, device="cpu")  # (N, latent_dim) float
    latent_tokens = encode_skills(model, segments, device="cpu")          # (N,) int32

    latents_path = output_dir / "skill_latents.npz"
    save_dict = {
        "latents": latent_codes,
        "tokens":  latent_tokens,
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
