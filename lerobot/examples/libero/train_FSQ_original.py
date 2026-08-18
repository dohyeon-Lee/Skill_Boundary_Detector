"""Train the one-shot (original-style) FSQ trajectory autoencoder.

The model reconstructs the encoder-input representation itself — normalized
spline control points + normalized length — in one shot from z_q. No images,
no terminator, no raw-dataset access: only the per-skill .npz files are read.

Usage:
    python examples/libero/train_FSQ_original.py \
      --skills_dir /path/to/skillset/skills \
      --output_dir /path/to/output
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


@dataclass
class Args:
    # ── data
    skills_dir: str = ""
    """Directory (recursively searched) for per-skill .npz files."""
    output_dir: str = ""
    """Output directory. Defaults to parent of skills_dir."""

    # ── model
    hidden_dim: int = 256
    fsq_levels: list[int] = field(default_factory=lambda: [3, 3, 3])
    quantizer: str = "fsq"
    """fsq | bsq. bsq ignores fsq_levels; the codebook is 2^bsq_code_dim corners."""
    bsq_code_dim: int = 5
    bsq_inv_temperature: float = 10.0
    bsq_entropy_conf_weight: float = 0.1
    """Per-sample entropy MINIMIZATION weight (bit confidence). 0 disables."""
    bsq_entropy_div_weight: float = 0.1
    """Batch entropy MAXIMIZATION weight (bit-usage balance). 0 = confidence-only."""
    bsq_entropy_joint: bool = True
    """Exact dataset entropy over all codes (project standard). The factorized
    fallback exists only for code_dim > 14."""
    fsq_entropy: bool = False
    """FSQ quantizer only: apply BSQ-style entropy terms to the FSQ grid,
    reusing the bsq_inv_temperature / bsq_entropy_* knobs (attribution ablation)."""
    bsq_entropy_cov_weight: float = 0.0
    """Coverage (dead-code revival) loss weight — pressure only on codes below
    cov_floor, zero once all codes are used. 0 = off."""
    bsq_entropy_cov_floor: float = 0.0
    """Soft batch-mass floor for coverage; 0 = auto (1/batch_size)."""
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    n_control: int = 30
    spline_degree: int = 3
    encoder_input_mode: str = "zero_grounded"
    """zero_grounded | raw_state | optimal (zero-grounded + one absolute start-EEF token).
    The start pose conditions the encoder only; it is never reconstructed."""
    encoder_length_token: bool = True
    """False: drop the encoder's length token — duration reaches z only through
    motion shape, becoming a learned per-code property."""
    encoder_arch: str = "spline"
    """spline: fixed control-point tokens. action_seq: variable-length ACTION
    sequence transformer (no spline codec, no grounding/length-token choices);
    requires decoder_arch='rnn'."""
    decoder_layers: int = 3
    """One-shot decoder MLP depth; rnn arch: GRU layer count."""
    decoder_arch: str = "oneshot"
    """oneshot: full control-point grid in one MLP pass. rnn: z-only GRU unroll
    emitting one normalized action + one termination logit per step (length is
    implicit in the termination signal)."""
    reconstruct_length: bool = True
    """oneshot only. False: reconstruct ONLY the control points — no length head,
    no length loss. Decoding then requires an explicit target length."""

    # ── loss
    ctrl_loss_weight: float = 1.0
    length_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    """rnn arch: masked per-step action MSE weight."""
    term_loss_weight: float = 1.0
    """rnn arch: masked per-step termination BCE weight."""
    term_pos_weight: float = 1.0
    """rnn arch: BCE positive-class weight for the termination head."""
    term_sigma: float = 1.0
    """rnn arch: Gaussian termination-target std in frames; 0 = hard spike."""

    # ── training
    epochs: int = 300
    encoder_lr: float = 3e-4
    decoder_lr: float = 3e-4
    lr_schedule: str = "cosine"
    """cosine: decay to 1% of each configured LR; constant: keep each LR fixed."""
    batch_size: int = 64
    num_workers: int = 0
    grad_clip: float = 1.0
    val_split: float = 0.1
    val_every: int = 1
    """Run validation every N epochs; 0 disables it."""
    save_best_model: bool = True
    """Write FSQ.pt whenever the validation selection metric improves."""
    val_select_ctrl_weight: float | None = None
    val_select_length_weight: float | None = None
    log_every: int = 10
    seed: int = 42
    device: str = "cuda"
    checkpoint_every: int = 0
    resume_from: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str = "fsq_original"


def main(args: Args) -> None:
    from FSQ import encoder_start_eef_pose, prepare_encoder_trajectory
    from FSQ_original import FSQOriginalConfig, train_fsq_original
    from skills_bundle import load_skills

    skills_dir = Path(args.skills_dir)
    output_dir = Path(args.output_dir) if args.output_dir else skills_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.encoder_input_mode not in {"zero_grounded", "raw_state", "optimal"}:
        raise ValueError(
            "--encoder_input_mode must be zero_grounded|raw_state|optimal, "
            f"got {args.encoder_input_mode!r}."
        )
    if args.decoder_arch not in {"oneshot", "rnn"}:
        raise ValueError(f"--decoder_arch must be oneshot|rnn, got {args.decoder_arch!r}.")
    if args.quantizer not in {"fsq", "bsq"}:
        raise ValueError(f"--quantizer must be fsq|bsq, got {args.quantizer!r}.")
    if args.encoder_arch not in {"spline", "action_seq"}:
        raise ValueError(f"--encoder_arch must be spline|action_seq, got {args.encoder_arch!r}.")
    if args.encoder_arch == "action_seq" and args.decoder_arch != "rnn":
        raise ValueError("--encoder_arch action_seq requires --decoder_arch rnn.")

    # Bundle-first loading: identical content/order to the per-file layout, but
    # one large read instead of 11k+ small-file Lustre round trips.
    segments, actions, metadata = load_skills(skills_dir)
    skill_lengths = [m["length"] for m in metadata]
    print(
        f"[FSQ-orig] Skill lengths — min:{min(skill_lengths)} max:{max(skill_lengths)} "
        f"mean:{np.mean(skill_lengths):.1f}"
    )

    # Encoder normalization stats follow the exact checkpointed input convention
    # applied before spline fitting; length stats are data-driven min/max.
    encoder_trajectories = np.concatenate([
        prepare_encoder_trajectory(s, args.encoder_input_mode) for s in segments
    ])
    encoder_min, encoder_max = encoder_trajectories.min(0), encoder_trajectories.max(0)
    encoder_start_min = encoder_start_max = None
    if args.encoder_input_mode == "optimal":
        start_poses = np.stack([encoder_start_eef_pose(s) for s in segments])
        encoder_start_min, encoder_start_max = start_poses.min(0), start_poses.max(0)
    lengths = [int(m["length"]) for m in metadata]
    length_min, length_max = float(min(lengths)), float(max(lengths))
    # rnn arch: q01/q99 action normalization computed from the skill data itself
    # (self-contained; robust to outliers, same convention as v3's dataset stats).
    action_q01 = action_q99 = None
    action_dim = actions[0].shape[-1]
    if args.decoder_arch == "rnn" or args.encoder_arch == "action_seq":
        all_actions = np.concatenate(actions)
        action_q01 = np.quantile(all_actions, 0.01, axis=0).astype(np.float32)
        action_q99 = np.quantile(all_actions, 0.99, axis=0).astype(np.float32)

    stats = dict(
        encoder_min=encoder_min, encoder_max=encoder_max,
        encoder_input_mode=np.asarray(args.encoder_input_mode),
        length_min=np.float32(length_min), length_max=np.float32(length_max),
    )
    if encoder_start_min is not None:
        stats.update(encoder_start_min=encoder_start_min, encoder_start_max=encoder_start_max)
    if action_q01 is not None:
        stats.update(action_q01=action_q01, action_q99=action_q99)
    np.savez(str(output_dir / "encoder_stats.npz"), **stats)
    print(f"[FSQ-orig] encoder input mode: {args.encoder_input_mode}")
    print(f"[FSQ-orig] encoder_min: {np.round(encoder_min, 4)}")
    print(f"[FSQ-orig] encoder_max: {np.round(encoder_max, 4)}")
    if encoder_start_min is not None:
        print(
            f"[FSQ-orig] optimal start-EEF min/max: {np.round(encoder_start_min, 4)} / "
            f"{np.round(encoder_start_max, 4)}"
        )
    print(f"[FSQ-orig] length_min/max: {length_min:.0f} / {length_max:.0f}")
    print(f"[FSQ-orig] encoder arch: {args.encoder_arch}, decoder arch: {args.decoder_arch}")
    if args.quantizer == "fsq" and args.fsq_entropy:
        print(
            f"[FSQ-orig] fsq entropy terms: on tau={args.bsq_inv_temperature} "
            f"conf/div={args.bsq_entropy_conf_weight}/{args.bsq_entropy_div_weight} "
            f"dataset_entropy={'joint' if args.bsq_entropy_joint else 'factorized'}"
        )
    if args.quantizer == "bsq":
        print(
            f"[FSQ-orig] quantizer: bsq code_dim={args.bsq_code_dim} "
            f"(codebook {2**args.bsq_code_dim}) tau={args.bsq_inv_temperature} "
            f"entropy conf/div={args.bsq_entropy_conf_weight}/{args.bsq_entropy_div_weight} "
            f"dataset_entropy={'joint' if args.bsq_entropy_joint else 'factorized'}"
        )
    if action_q01 is not None:
        print(f"[FSQ-orig] action q01/q99: {np.round(action_q01, 4)} / {np.round(action_q99, 4)}")

    device = args.device if torch.cuda.is_available() else "cpu"
    enc_dim = segments[0].shape[-1]

    wandb_run = None
    if args.wandb_project:
        import os

        import wandb

        # Pin wandb's system-metrics GPU monitor to THIS job's GPU(s) (telemetry
        # scoping only; same fix as train_FSQ.py).
        gpu_settings = None
        try:
            ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
            if ids:
                gpu_settings = wandb.Settings(x_stats_gpu_device_ids=ids)
        except Exception:  # noqa: BLE001 — cosmetic monitor scoping must never break training
            gpu_settings = None
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), "enc_dim": enc_dim, "n_segments": len(segments)},
            settings=gpu_settings,
        )

    cfg = FSQOriginalConfig(
        enc_dim=enc_dim,
        n_control=args.n_control,
        spline_degree=args.spline_degree,
        encoder_input_mode=args.encoder_input_mode,
        encoder_length_token=args.encoder_length_token,
        encoder_arch=args.encoder_arch,
        hidden_dim=args.hidden_dim,
        fsq_levels=args.fsq_levels,
        quantizer=args.quantizer,
        bsq_code_dim=args.bsq_code_dim,
        bsq_inv_temperature=args.bsq_inv_temperature,
        bsq_entropy_conf_weight=args.bsq_entropy_conf_weight,
        bsq_entropy_div_weight=args.bsq_entropy_div_weight,
        bsq_entropy_joint=args.bsq_entropy_joint,
        fsq_entropy=args.fsq_entropy,
        bsq_entropy_cov_weight=args.bsq_entropy_cov_weight,
        bsq_entropy_cov_floor=args.bsq_entropy_cov_floor,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        decoder_layers=args.decoder_layers,
        decoder_arch=args.decoder_arch,
        reconstruct_length=args.reconstruct_length,
        length_min=length_min,
        length_max=length_max,
        action_dim=action_dim,
        ctrl_loss_weight=args.ctrl_loss_weight,
        length_loss_weight=args.length_loss_weight,
        action_loss_weight=args.action_loss_weight,
        term_loss_weight=args.term_loss_weight,
        term_pos_weight=args.term_pos_weight,
        term_sigma=args.term_sigma,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        lr_schedule=args.lr_schedule,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        val_split=args.val_split,
        val_every=args.val_every,
        save_best_model=args.save_best_model,
        val_select_ctrl_weight=args.val_select_ctrl_weight,
        val_select_length_weight=args.val_select_length_weight,
        log_every=args.log_every,
        save_path=str(output_dir / "FSQ.pt"),
        checkpoint_every=args.checkpoint_every,
        device=device,
        encoder_min=encoder_min,
        encoder_max=encoder_max,
        encoder_start_min=encoder_start_min,
        encoder_start_max=encoder_start_max,
        action_q01=action_q01,
        action_q99=action_q99,
    )

    train_fsq_original(
        segments=segments,
        metadata=metadata,
        cfg=cfg,
        actions=actions if args.decoder_arch == "rnn" else None,
        wandb_run=wandb_run,
        resume_from=args.resume_from,
    )

    if args.save_best_model:
        print(f"[FSQ-orig] Saved best model → {output_dir / 'FSQ.pt'}")
    else:
        print(f"[FSQ-orig] Saved periodic checkpoints → {output_dir / 'FSQ_epoch*.pt'}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
