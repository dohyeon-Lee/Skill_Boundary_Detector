"""Train the v3 FSQ tokenizer, VSA flow expert, and query terminator jointly.

The terminator reads live third-person and wrist frames only at the M timesteps
sampled for each trajectory. Its DINO/SigLIP frontend is the same unpooled visual
contract used by Stage-1 cond. There is no DINO precompute, warm pass, token cache,
or camera-selection mode in the FSQ path.

Usage:
    python examples/libero/train_FSQ.py \
      --skills_dir      /path/to/skillset/skills \
      --raw_dataset_dir /path/to/lerobot_dataset   (videos/ + meta/ 포함) \
      --output_dir      /path/to/output

Data requirements:
  skills_dir/*.npz  — per-skill npz with keys: actions, states, episode_id, skill_index,
                       frame_start, frame_end, task_id (optional)
  raw_dataset_dir   — LeRobot v3 dataset (videos/{image_key}/chunk-*/file-*.mp4 + meta/)
"""

from __future__ import annotations

import sys
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# ── Args ───────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    # ── data
    skills_dir: str = ""
    """Directory (recursively searched) for per-skill .npz files."""
    raw_dataset_dir: str = ""
    """Raw LeRobot dataset. Selected M timesteps load both camera frames live."""
    output_dir: str = ""
    """Output directory. Defaults to parent of skills_dir."""

    # ── model
    hidden_dim: int = 256
    fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    num_layers: int = 3
    dropout: float = 0.1
    n_control: int = 30
    spline_degree: int = 3
    encoder_input_mode: str = "zero_grounded"
    """zero_grounded | raw_state | optimal (zero-grounded + one absolute start-EEF token)."""
    terminator_arch: str = "small"
    """small: lightweight query transformer; cond: Stage-1-cond-compatible Gemma."""
    vision_backbone: str = "dino"
    """dino or siglip; shared by third-person and wrist images."""
    freeze_vision_encoder: bool = True
    dino_model_path: str = "../models/dinov3-vits16"
    dino_image_size: int = 224
    siglip_image_size: int = 224
    cond_encoder_variant: str = "gemma_300m"
    image_encoder_layers: int = 1
    image_encoder_heads: int = 4
    chunk_size: int = 10
    """Steps per motion action chunk."""
    samples_per_skill: int = 2
    """M: random decoder timesteps sampled from each of the B skill trajectories per update."""
    action_expert_variant: str = "gemma_300m"
    state_cond_mode: str = "state"
    """state: z_q prefix; state_skill: z_q joins state/time in expert AdaRMS."""
    max_state_dim: int = 32
    max_action_dim: int = 32
    pi_base: str = "../models/pi05_base"
    """PI05 checkpoint used only to initialize Gemma/action/time tensors of the FSQ expert."""
    expert_dtype: str = "bfloat16"

    # ── loss
    action_loss_weight: float = 1.0
    progress_loss_weight: float = 0.1
    end_loss_weight: float = 0.1
    end_pos_weight: float = 1.0
    """BCE positive-class weight for the termination head."""
    end_threshold: float = 0.5
    end_target_sigma: float = 0.0
    """Soft termination target std in frames (Gaussian bump at the skill end). 0 = hard 1-frame
    spike. σ≈2-3 curbs the val overfit a sharp spike causes and adds ±tolerance to recall/precision."""
    weighted_loss: bool = False
    """End-weight the per-sampled-timestep flow loss."""
    weighted_loss_end_weight: float = 2.0
    """Flow-loss weight at progress=1; progress=0 always has weight 1."""

    # ── training
    epochs: int = 300
    encoder_lr: float = 3e-4
    terminator_lr: float = 3e-4
    expert_lr: float = 2.5e-5
    batch_size: int = 64
    num_workers: int = 8
    grad_clip: float = 1.0
    val_split: float = 0.1
    # Best-val SELECTION metric weights. Unset (None) → follow the actual loss (total val). Set to
    # override: FSQ.pt = argmin over epochs of wa*action + wp*progress + we*end on val.
    val_select_action_weight: float | None = None
    val_select_progress_weight: float | None = None
    val_select_end_weight: float | None = None
    log_every: int = 10
    seed: int = 42
    device: str = "cuda"
    checkpoint_every: int = 0
    resume_from: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str = "spline_fsqae"


# ── Data loading ───────────────────────────────────────────────────────────────

def _compute_skill_orders(metadata: list[dict]) -> list[float]:
    by_ep: dict[int, list[int]] = {}
    for i, m in enumerate(metadata):
        by_ep.setdefault(int(m["episode_id"]), []).append(i)
    orders = [0.0] * len(metadata)
    for ids in by_ep.values():
        ids.sort(key=lambda i: (metadata[i]["frame_start"], metadata[i]["skill_index"]))
        denom = max(1, len(ids))
        for rank, idx in enumerate(ids):
            orders[idx] = float((rank + 1) / denom)
    return orders


def _make_encoder_traj(states):
    """Encoder trajectory = full observation.state (EEF pose + gripper STATE), same source as the
    decoder state. Pose dims are zero-grounded downstream; the trailing gripper-STATE dims stay
    absolute (see N_GRIPPER_DIMS in FSQ.py). For LIBERO this is 8D: ee_state(6) + gripper_state(2)."""
    return states.astype(np.float32)


def _make_decoder_state(states):
    """Decoder/terminator proprioception = full observation.state (EEF pose + gripper STATE).

    Gripper here is the observed gripper STATE (part of observation.state), not the
    action command, so there is no target leak — the current-step value is used as-is.
    For LIBERO this is 8D: ee_state(6) + gripper_state(2).
    """
    return states.astype(np.float32)


def _make_target_action(actions):
    """Decoder target is the real dataset action_t."""
    return actions.astype(np.float32)


def load_skill_files(
    skills_dir: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict]]:
    npz_files = sorted(skills_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {skills_dir}")

    segments, dec_states, dec_targets, metadata = [], [], [], []
    for f in npz_files:
        d = np.load(str(f))
        actions = d["actions"].astype(np.float32)
        states  = d["states"].astype(np.float32)
        segments.append(_make_encoder_traj(states))
        dec_states.append(_make_decoder_state(states))
        dec_targets.append(_make_target_action(actions))
        metadata.append({
            "file":        str(f.name),
            "episode_id":  int(d["episode_id"]),
            "task_id":     int(d["task_id"]) if "task_id" in d else -1,
            "skill_index": int(d["skill_index"]),
            "frame_start": int(d["frame_start"]),
            "frame_end":   int(d["frame_end"]),
            "length":      len(actions),
        })

    print(f"[FSQ] Loaded {len(segments)} skills from {skills_dir}")
    lengths = [m["length"] for m in metadata]
    print(f"[FSQ] Skill lengths — min:{min(lengths)} max:{max(lengths)} mean:{np.mean(lengths):.1f}")
    return segments, dec_states, dec_targets, metadata


def attach_episode_offsets(raw_dataset_dir: str, metadata: list[dict]) -> None:
    """Attach the absolute dataset index corresponding to each skill's episode."""
    from lerobot.datasets.io_utils import load_episodes

    episodes = load_episodes(Path(raw_dataset_dir))
    for item in metadata:
        episode = episodes[int(item["episode_id"])]
        item["dataset_from_index"] = int(episode["dataset_from_index"])
        episode_length = int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
        if int(item["frame_start"]) + int(item["length"]) > episode_length:
            raise ValueError(
                f"Skill {item['file']} exceeds episode {item['episode_id']}: "
                f"start={item['frame_start']} length={item['length']} episode_length={episode_length}"
            )


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    from FSQ import SplineFSQAEConfig, prepare_encoder_trajectory, train_spline_fsqae

    skills_dir = Path(args.skills_dir)
    output_dir = Path(args.output_dir) if args.output_dir else skills_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    segments, dec_states, dec_targets, metadata = load_skill_files(skills_dir)

    if not args.raw_dataset_dir:
        raise ValueError("--raw_dataset_dir is required (LeRobot videos + metadata).")
    attach_episode_offsets(args.raw_dataset_dir, metadata)

    if args.encoder_input_mode not in {"zero_grounded", "raw_state", "optimal"}:
        raise ValueError(
            "--encoder_input_mode must be zero_grounded|raw_state|optimal, "
            f"got {args.encoder_input_mode!r}."
        )

    # Encoder normalization stats must follow the exact checkpointed input convention used before
    # spline fitting. Length stats are data-driven min/max over skill lengths.
    encoder_trajectories = np.concatenate([
        prepare_encoder_trajectory(s, args.encoder_input_mode)
        for s in segments
    ])
    encoder_min, encoder_max = encoder_trajectories.min(0), encoder_trajectories.max(0)
    encoder_start_min = encoder_start_max = None
    if args.encoder_input_mode == "optimal":
        from FSQ import encoder_start_eef_pose

        start_poses = np.stack([
            encoder_start_eef_pose(s) for s in segments
        ])
        encoder_start_min, encoder_start_max = start_poses.min(0), start_poses.max(0)
    all_state = np.concatenate(dec_states)   # decoder/terminator proprioception (raw, all timesteps)
    state_min,  state_max  = all_state.min(0), all_state.max(0)
    _lens = [int(m["length"]) for m in metadata]
    length_min, length_max = float(min(_lens)), float(max(_lens))
    stats_path = Path(args.raw_dataset_dir) / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Dataset normalization stats are required for Stage-1-compatible FSQ training: {stats_path}"
        )
    raw_stats = json.loads(stats_path.read_text())
    try:
        state_q01 = np.asarray(raw_stats["observation.state"]["q01"], dtype=np.float32)
        state_q99 = np.asarray(raw_stats["observation.state"]["q99"], dtype=np.float32)
        action_q01 = np.asarray(raw_stats["action"]["q01"], dtype=np.float32)
        action_q99 = np.asarray(raw_stats["action"]["q99"], dtype=np.float32)
    except KeyError as exc:
        raise KeyError(f"{stats_path} lacks the q01/q99 stats required by Stage-1: {exc}") from exc
    state_dim, action_dim = dec_states[0].shape[-1], dec_targets[0].shape[-1]
    if len(state_q01) != state_dim or len(action_q01) != action_dim:
        raise ValueError(
            f"Dataset stats dimensionality mismatch: state {len(state_q01)}!={state_dim}, "
            f"action {len(action_q01)}!={action_dim}."
        )
    stats = dict(
        encoder_min=encoder_min, encoder_max=encoder_max,
        encoder_input_mode=np.asarray(args.encoder_input_mode),
        state_q01=state_q01, state_q99=state_q99,
        action_q01=action_q01, action_q99=action_q99,
        state_min=state_min, state_max=state_max,
        length_min=np.float32(length_min), length_max=np.float32(length_max),
    )
    if encoder_start_min is not None:
        stats.update(encoder_start_min=encoder_start_min, encoder_start_max=encoder_start_max)
    np.savez(str(output_dir / "action_stats.npz"), **stats)
    print(f"[FSQ] encoder input mode: {args.encoder_input_mode}")
    print(f"[FSQ] encoder_min: {np.round(encoder_min, 4)}")
    print(f"[FSQ] encoder_max: {np.round(encoder_max, 4)}")
    if encoder_start_min is not None:
        print(f"[FSQ] optimal start-EEF min/max: {np.round(encoder_start_min, 4)} / "
              f"{np.round(encoder_start_max, 4)}")
    print(f"[FSQ] state q01/q99: {np.round(state_q01, 4)} / {np.round(state_q99, 4)}")
    print(f"[FSQ] action q01/q99: {np.round(action_q01, 4)} / {np.round(action_q99, 4)}")
    print(f"[FSQ] state_min:  {np.round(state_min,  4)}")
    print(f"[FSQ] state_max:  {np.round(state_max,  4)}")
    print(f"[FSQ] length_min/max: {length_min:.0f} / {length_max:.0f}")

    device = args.device if torch.cuda.is_available() else "cpu"
    enc_dim    = segments[0].shape[-1]

    wandb_run = None
    if args.wandb_project:
        import os

        import wandb

        # Pin wandb's system-metrics GPU monitor to THIS job's GPU(s) so the panel shows only our device
        # instead of every GPU on the shared node (same fix as lerobot's WandBLogger — READ-only use of
        # the CUDA_VISIBLE_DEVICES Slurm already set; telemetry scoping only, never affects allocation).
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
            config={**vars(args), "action_dim": action_dim, "enc_dim": enc_dim,
                    "state_dim": state_dim, "n_segments": len(segments)},
            settings=gpu_settings,
        )

    cfg = SplineFSQAEConfig(
        action_dim=action_dim,
        enc_dim=enc_dim,
        state_dim=state_dim,
        n_control=args.n_control,
        spline_degree=args.spline_degree,
        encoder_input_mode=args.encoder_input_mode,
        hidden_dim=args.hidden_dim,
        fsq_levels=args.fsq_levels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        action_expert_variant=args.action_expert_variant,
        state_cond_mode=args.state_cond_mode,
        max_state_dim=args.max_state_dim,
        max_action_dim=args.max_action_dim,
        pi_base=args.pi_base,
        expert_dtype=args.expert_dtype,
        terminator_arch=args.terminator_arch,
        vision_backbone=args.vision_backbone,
        freeze_vision_encoder=args.freeze_vision_encoder,
        dino_model_path=args.dino_model_path,
        dino_image_size=args.dino_image_size,
        siglip_image_size=args.siglip_image_size,
        cond_encoder_variant=args.cond_encoder_variant,
        image_encoder_layers=args.image_encoder_layers,
        image_encoder_heads=args.image_encoder_heads,
        chunk_size=args.chunk_size,
        samples_per_skill=args.samples_per_skill,
        length_min=length_min,
        length_max=length_max,
        action_loss_weight=args.action_loss_weight,
        progress_loss_weight=args.progress_loss_weight,
        end_loss_weight=args.end_loss_weight,
        end_pos_weight=args.end_pos_weight,
        end_threshold=args.end_threshold,
        end_target_sigma=args.end_target_sigma,
        weighted_loss=args.weighted_loss,
        weighted_loss_end_weight=args.weighted_loss_end_weight,
        encoder_lr=args.encoder_lr,
        terminator_lr=args.terminator_lr,
        expert_lr=args.expert_lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        epochs=args.epochs,
        val_split=args.val_split,
        val_select_action_weight=args.val_select_action_weight,
        val_select_progress_weight=args.val_select_progress_weight,
        val_select_end_weight=args.val_select_end_weight,
        log_every=args.log_every,
        device=device,
        save_path=str(output_dir / "FSQ.pt"),
        checkpoint_every=args.checkpoint_every,
        encoder_min=encoder_min,
        encoder_max=encoder_max,
        encoder_start_min=encoder_start_min,
        encoder_start_max=encoder_start_max,
        state_min=state_min,
        state_max=state_max,
        state_q01=state_q01,
        state_q99=state_q99,
        action_q01=action_q01,
        action_q99=action_q99,
    )

    train_spline_fsqae(
        segments=segments,
        decoder_states=dec_states,
        decoder_targets=dec_targets,
        raw_dataset_dir=args.raw_dataset_dir,
        cfg=cfg,
        wandb_run=wandb_run,
        metadata=metadata,
        resume_from=args.resume_from,
    )

    # Skill latents (skill_latents.npz) are produced separately by encode_FSQ_skills.py
    # (build_data/encode_skills.sbatch) — the canonical, consumer-facing producer. We do
    # not re-emit them here to avoid a stale/duplicate copy under the training output_dir.
    print(f"[FSQ] Saved model   → {output_dir / 'FSQ.pt'}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
