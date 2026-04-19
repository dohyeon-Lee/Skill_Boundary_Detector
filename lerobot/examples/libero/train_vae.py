"""
Skill VAE Training Script.

Loads skill segment files (.npz) produced by replay_demo.py --save_skills
and trains a BiLSTM VAE with stop-token decoder.

Each .npz file contains:
    actions  : (T, action_dim)  float32
    episode_id, skill_index, frame_start, frame_end  (scalar metadata)

After training, saves:
    <output_dir>/skill_vae.pt        — model weights + config
    <output_dir>/skill_latents.npz   — latent codes for every segment

Usage:
    python examples/libero/train_vae.py \
      --skills_dir /path/to/skills \
      --output_dir /path/to/vae_output \
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
    stop_threshold: float = 0.5
    max_decode_steps: int = 500

    # Training
    epochs: int = 200
    beta: float = 1.0
    """KL weight in the ELBO loss."""
    stop_weight: float = 1.0
    """Stop loss weight."""
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
    resume_from: str | None = None
    """Path to a checkpoint .pt file to resume training from."""
    wandb_project: str | None = None
    """If set, log training loss to this wandb project."""
    wandb_run_name: str = "skill_vae"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_skill_files(skills_dir: Path) -> tuple[list, list[np.ndarray], list[dict]]:
    npz_files = sorted(skills_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {skills_dir}")

    segments    = []
    init_states = []
    metadata    = []
    for f in npz_files:
        d = np.load(str(f))
        segments.append(d["actions"])
        init_states.append(d["states"][0])  # first frame eef state
        metadata.append({
            "file": str(f.name),
            "episode_id": int(d["episode_id"]),
            "task_id": int(d["task_id"]) if "task_id" in d else -1,
            "skill_index": int(d["skill_index"]),
            "frame_start": int(d["frame_start"]),
            "frame_end": int(d["frame_end"]),
            "length": len(d["actions"]),
        })

    print(f"[VAE] Loaded {len(segments)} skill segments from {skills_dir}")
    lengths = [m["length"] for m in metadata]
    print(f"[VAE] Skill lengths — min: {min(lengths)}, max: {max(lengths)}, mean: {np.mean(lengths):.1f}")
    return segments, init_states, metadata


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    skills_dir = Path(args.skills_dir)
    output_dir = Path(args.output_dir) if args.output_dir else skills_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    segments, init_states, metadata = load_skill_files(skills_dir)

    if args.min_skill_len > 0:
        before = len(segments)
        segments, init_states, metadata = zip(*[
            (s, st, m) for s, st, m in zip(segments, init_states, metadata) if m["length"] >= args.min_skill_len
        ])
        segments, init_states, metadata = list(segments), list(init_states), list(metadata)
        print(f"[VAE] Filtered short skills: {before} → {len(segments)} (min_len={args.min_skill_len})")

    all_actions = np.concatenate(segments, axis=0)
    action_min = all_actions.min(axis=0)
    action_max = all_actions.max(axis=0)
    np.savez(str(output_dir / "action_stats.npz"), action_min=action_min, action_max=action_max)
    print(f"[VAE] Action stats saved → {output_dir / 'action_stats.npz'}")
    print(f"[VAE] action_min: {np.round(action_min, 4)}")
    print(f"[VAE] action_max: {np.round(action_max, 4)}")

    device = args.device if torch.cuda.is_available() else "cpu"

    from skill_vae import VAEConfig, encode_skills, train_skill_vae

    action_dim = segments[0].shape[-1]
    state_dim  = init_states[0].shape[-1]
    cfg = VAEConfig(
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        beta=args.beta,
        stop_weight=args.stop_weight,
        stop_threshold=args.stop_threshold,
        max_decode_steps=args.max_decode_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        epochs=args.epochs,
        val_split=args.val_split,
        log_every=args.log_every,
        device=device,
        save_path=str(output_dir / "skill_vae.pt"),
        checkpoint_every=args.checkpoint_every,
        action_min=action_min,
        action_max=action_max,
    )

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "action_dim": action_dim,
                "state_dim": state_dim,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "beta": args.beta,
                "stop_weight": args.stop_weight,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "n_segments": len(segments),
            },
        )

    model = train_skill_vae(segments, init_states, cfg, wandb_run=wandb_run, metadata=metadata,
                            resume_from=args.resume_from)

    print("[VAE] Encoding all skill segments...")
    latent_codes = encode_skills(model, segments, device="cpu")
    print(f"[VAE] Latent codes: {latent_codes.shape}")

    latents_path = output_dir / "skill_latents.npz"
    save_dict = {"latents": latent_codes}
    for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"):
        save_dict[key] = np.array([m[key] for m in metadata])
    np.savez(str(latents_path), **save_dict)
    print(f"[VAE] Saved latents + metadata → {latents_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
