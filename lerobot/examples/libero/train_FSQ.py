"""
Train SplineFSQAE: Finite Scalar Quantization skill tokenizer with DINO-token-conditioned decoder.

Usage:
    python examples/libero/train_FSQ.py \
      --skills_dir     /path/to/skillset/skills \
      --dino_features  /path/to/dino_features.npz \
      --sam2_masks_dir /path/to/sam2_masks_or_merged_flags.npz \
      --output_dir     /path/to/output

Data requirements:
  skills_dir/*.npz  — per-skill npz with keys: actions, states, episode_id, skill_index,
                       frame_start, frame_end, task_id (optional)
  dino_features.npz — precomputed DINO tokens produced by precompute_dino_features.py,
                       keys: features (N_total, n_tokens, feat_dim), offsets, episode_id,
                       frame_start, frame_end, length
  sam2_masks_dir/   — either a merged temporal patch_flags.npz produced by
                       merge_sam2_patch_flags.py, or the per-skill directory
                       produced by precompute_sam2_masks.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).parent))


# ── Args ───────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    # ── data
    skills_dir: str = ""
    """Directory (recursively searched) for per-skill .npz files."""
    dino_features: str = ""
    """Path to precomputed DINO token npz (features shape: N_total × n_tokens × feat_dim)."""
    sam2_masks_dir: str = ""
    """Merged SAM2 patch_flags.npz or directory with per-skill SAM2 mask npz files."""
    output_dir: str = ""
    """Output directory. Defaults to parent of skills_dir."""
    eef_dims: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    """State indices used as EEF pose for the encoder trajectory and decoder state."""
    gripper_action_dim: int = -1
    """Action dim index for gripper signal (used in encoder trajectory and decoder state)."""

    # ── model
    hidden_dim: int = 256
    fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    num_layers: int = 3
    dropout: float = 0.1
    n_control: int = 30
    spline_degree: int = 3
    feat_dim: int = 384
    """DINO feature dimension per token."""
    n_tokens: int = 65
    """DINO tokens per frame: 1 CLS + 64 patch."""
    image_model_name: str = "/data2/dohyeon/SBD/models/dinov3-vits16"
    """Local path or HF model ID used for raw-image FSQ decoder inference."""
    image_size: int = 224
    patch_grid: int = 8
    n_patch_raw: int = 196
    reconstructor_mode: str = "flags"
    """'flags' | 'only' (siglip added in stage 2). Controls the reconstructor's flag token."""
    image_token_dim: int = 128
    """Internal per-patch width N in the image encoders (must be divisible by heads, >2)."""
    image_encoder_layers: int = 1
    image_encoder_heads: int = 4
    chunk_size: int = 10
    """Steps per motion action chunk."""
    max_length: float = 200.0

    # ── loss
    delta_loss_weight: float = 1.0
    progress_loss_weight: float = 1.0
    end_loss_weight: float = 10.0
    end_pos_weight: float = 1.0
    """BCE positive-class weight for the termination head."""
    end_threshold: float = 0.5

    # ── training
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


def _make_encoder_traj(states, actions, eef_dims, gripper_idx):
    """Absolute EEF pose (6D) + gripper command (1D) per step."""
    pose = states[:, eef_dims].astype(np.float32)
    gripper = actions[:, gripper_idx:gripper_idx + 1].astype(np.float32)
    return np.concatenate([pose, gripper], axis=-1)


def _make_decoder_state(states, actions, eef_dims, gripper_idx):
    """Decoder/terminator proprioception = full observation.state (EEF pose + gripper STATE).

    Gripper here is the observed gripper STATE (part of observation.state), not the
    action command, so there is no target leak — the current-step value is used as-is.
    For LIBERO this is 8D: ee_state(6) + gripper_state(2).
    """
    return states.astype(np.float32)


def _make_target_action(actions):
    """Decoder target is the real dataset action_t."""
    return actions.astype(np.float32)


def _make_episode_future_targets(current_targets: list[np.ndarray], metadata: list[dict]) -> list[np.ndarray]:
    """Return per-skill action targets from this skill start to the episode end.

    Decoder inputs remain limited to the current skill. In chunk mode, however,
    target slots after the current skill boundary can be supervised by the next
    skill's actions as long as the chunk stays inside the same episode.
    """
    by_episode: dict[int, list[int]] = {}
    for i, m in enumerate(metadata):
        by_episode.setdefault(int(m["episode_id"]), []).append(i)

    future_targets: list[np.ndarray | None] = [None] * len(current_targets)
    for ids in by_episode.values():
        ids.sort(key=lambda i: (metadata[i]["frame_start"], metadata[i]["skill_index"]))
        for a, b in zip(ids, ids[1:]):
            if int(metadata[a]["frame_end"]) != int(metadata[b]["frame_start"]):
                raise ValueError(
                    "Non-contiguous skill boundary in episode "
                    f"{metadata[a]['episode_id']}: skill {metadata[a]['skill_index']} "
                    f"ends at {metadata[a]['frame_end']}, next skill starts at {metadata[b]['frame_start']}."
                )
        episode_actions = np.concatenate([current_targets[i] for i in ids], axis=0)
        offsets = np.cumsum([0] + [len(current_targets[i]) for i in ids])
        for local_i, idx in enumerate(ids):
            future_targets[idx] = episode_actions[offsets[local_i]:]

    return [x if x is not None else current_targets[i] for i, x in enumerate(future_targets)]


def load_skill_files(
    skills_dir: Path,
    eef_dims: list[int],
    gripper_action_dim: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict]]:
    npz_files = sorted(skills_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {skills_dir}")

    segments, dec_states, dec_targets, metadata = [], [], [], []
    for f in npz_files:
        d = np.load(str(f))
        actions = d["actions"].astype(np.float32)
        states  = d["states"].astype(np.float32)
        gripper_idx = (actions.shape[-1] + gripper_action_dim) % actions.shape[-1]
        segments.append(_make_encoder_traj(states, actions, eef_dims, gripper_idx))
        dec_states.append(_make_decoder_state(states, actions, eef_dims, gripper_idx))
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

    dec_targets = _make_episode_future_targets(dec_targets, metadata)

    print(f"[FSQ] Loaded {len(segments)} skills from {skills_dir}")
    lengths = [m["length"] for m in metadata]
    print(f"[FSQ] Skill lengths — min:{min(lengths)} max:{max(lengths)} mean:{np.mean(lengths):.1f}")
    return segments, dec_states, dec_targets, metadata


def load_dino_tokens(
    features_path: Path,
    metadata: list[dict],
) -> list[np.ndarray]:
    """Load per-skill DINO token sequences (T, n_tokens, feat_dim) from precomputed npz.

    The npz must contain:
      features  : (N_total, n_tokens, feat_dim) float32
      offsets   : (N_skills + 1,) int64
      episode_id, frame_start, frame_end, length : (N_skills,)
    """
    d = np.load(str(features_path), allow_pickle=False)
    # Keep the native dtype (usually float16); downstream consumers convert per
    # clip, so a full float32 copy here would just waste memory/time (the array is
    # tens of GB) and risk swapping.
    features = d["features"]                         # (N_total, n_tokens, F)
    offsets  = d["offsets"].astype(np.int64)        # (N+1,)

    if features.ndim == 2:
        # Old format: (N_total, F) — CLS only. Expand with dummy patches so
        # the model sees (N_total, 1, F). n_tokens must be set to 1.
        features = features[:, None, :]

    if len(offsets) != len(metadata) + 1:
        raise ValueError(f"Offset count {len(offsets)-1} != skill count {len(metadata)}")

    clips = []
    for i, m in enumerate(metadata):
        expected = (int(m["episode_id"]), int(m["frame_start"]), int(m["frame_end"]), int(m["length"]))
        got = (int(d["episode_id"][i]), int(d["frame_start"][i]), int(d["frame_end"][i]), int(d["length"][i]))
        if got != expected:
            raise ValueError(f"DINO metadata mismatch at index {i}: got {got}, expected {expected}")
        clips.append(features[offsets[i]:offsets[i + 1]])

    print(f"[FSQ] Loaded DINO tokens from {features_path} — shape per clip: {clips[0].shape}")
    return clips


def load_patch_flags(
    sam2_masks_dir: Path,
    metadata: list[dict],
    n_patches: int = 64,
) -> list[np.ndarray]:
    """Load per-skill temporal [is_changed, is_green] flags (T, n_patches, 2).

    Prefer a merged patch_flags.npz for fast training. Directory loading is kept
    as a fallback and preserves every timestep from each per-skill mask npz.
    """
    if sam2_masks_dir.is_file():
        d = np.load(str(sam2_masks_dir))
        flags = d["patch_flags"].astype(np.float32)
        if "offsets" not in d:
            raise ValueError("Merged SAM2 patch_flags.npz must contain temporal offsets.")
        offsets = d["offsets"].astype(np.int64)
        for i, m in enumerate(metadata):
            expected = (
                int(m["episode_id"]),
                int(m["skill_index"]),
                int(m["frame_start"]),
                int(m["frame_end"]),
                int(m["length"]),
            )
            got = (
                int(d["episode_id"][i]),
                int(d["skill_index"][i]),
                int(d["frame_start"][i]),
                int(d["frame_end"][i]),
                int(d["length"][i]),
            )
            if got != expected:
                raise ValueError(f"Merged SAM2 metadata mismatch at index {i}: got {got}, expected {expected}")

        if len(offsets) != len(metadata) + 1:
            raise ValueError(f"Merged SAM2 offsets count {len(offsets) - 1} != skill count {len(metadata)}")
        expected_total = sum(int(m["length"]) for m in metadata)
        expected_shape = (expected_total, n_patches, 2)
        if flags.shape != expected_shape:
            raise ValueError(f"Merged SAM2 patch_flags shape mismatch: got {flags.shape}, expected {expected_shape}")
        out = []
        for i, m in enumerate(metadata):
            clip = flags[offsets[i]:offsets[i + 1]]
            if len(clip) != int(m["length"]):
                raise ValueError(
                    f"Merged SAM2 length mismatch at index {i}: got {len(clip)}, expected {int(m['length'])}"
                )
            out.append(clip)

        n_missing = int(d["missing"].sum()) if "missing" in d else 0
        if n_missing:
            print(f"[FSQ] Warning: merged SAM2 has {n_missing}/{len(metadata)} missing skills — using zero flags")
        else:
            print(f"[FSQ] Loaded merged temporal SAM2 patch flags for all {len(metadata)} skills")
        return out

    def mask_path(m: dict) -> Path:
        return (
            sam2_masks_dir
            / f"task{int(m['task_id'])}"
            / f"ep{int(m['episode_id']):05d}_skill{int(m['skill_index']):03d}.npz"
        )

    flags_list = []
    n_missing = 0
    for m in metadata:
        expected_len = int(m["length"])
        f = mask_path(m)
        if not f.exists():
            flags_list.append(np.zeros((expected_len, n_patches, 2), dtype=np.float32))
            n_missing += 1
            continue
        msk = np.load(str(f))
        pm = msk["patch_masks"].astype(np.float32)  # (T_samp, H_p, W_p, 2)
        if len(pm) == 0:
            flags_list.append(np.zeros((expected_len, n_patches, 2), dtype=np.float32))
            n_missing += 1
            continue
        temporal = pm.reshape(len(pm), -1, 2)
        if temporal.shape[1] != n_patches:
            raise ValueError(f"{f} has {temporal.shape[1]} patches after flattening, expected {n_patches}")
        if len(temporal) != expected_len:
            raise ValueError(f"{f} length mismatch: patch_masks has {len(temporal)}, expected {expected_len}")
        flags_list.append(temporal)

    if n_missing:
        print(f"[FSQ] Warning: {n_missing}/{len(metadata)} skills missing SAM2 masks — using zero flags")
    else:
        print(f"[FSQ] Loaded temporal SAM2 patch flags for all {len(metadata)} skills")
    return flags_list


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    from FSQ import SplineFSQAEConfig, train_spline_fsqae

    skills_dir = Path(args.skills_dir)
    output_dir = Path(args.output_dir) if args.output_dir else skills_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    segments, dec_states, dec_targets, metadata = load_skill_files(
        skills_dir,
        eef_dims=args.eef_dims,
        gripper_action_dim=args.gripper_action_dim,
    )

    dec_tokens = load_dino_tokens(Path(args.dino_features), metadata)
    n_patches = (args.n_tokens - 1)  # subtract CLS token
    patch_flags = (
        load_patch_flags(Path(args.sam2_masks_dir), metadata, n_patches=n_patches)
        if args.sam2_masks_dir
        else [np.zeros((m["length"], n_patches, 2), dtype=np.float32) for m in metadata]
    )

    all_enc = np.concatenate(segments)
    all_tgt = np.concatenate([target[: int(m["length"])] for target, m in zip(dec_targets, metadata, strict=True)])
    action_min, action_max = all_enc.min(0), all_enc.max(0)
    delta_min,  delta_max  = all_tgt.min(0), all_tgt.max(0)
    np.savez(str(output_dir / "action_stats.npz"),
             action_min=action_min, action_max=action_max,
             delta_min=delta_min,   delta_max=delta_max,
             eef_dims=np.array(args.eef_dims))
    print(f"[FSQ] action_min: {np.round(action_min, 4)}")
    print(f"[FSQ] action_max: {np.round(action_max, 4)}")
    print(f"[FSQ] delta_min:  {np.round(delta_min,  4)}")
    print(f"[FSQ] delta_max:  {np.round(delta_max,  4)}")

    device = args.device if torch.cuda.is_available() else "cpu"
    action_dim = segments[0].shape[-1]
    state_dim  = dec_states[0].shape[-1]

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), "action_dim": action_dim, "state_dim": state_dim,
                    "n_segments": len(segments)},
        )

    cfg = SplineFSQAEConfig(
        action_dim=action_dim,
        state_dim=state_dim,
        n_control=args.n_control,
        spline_degree=args.spline_degree,
        hidden_dim=args.hidden_dim,
        fsq_levels=args.fsq_levels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        feat_dim=args.feat_dim,
        n_tokens=args.n_tokens,
        image_model_name=args.image_model_name,
        image_size=args.image_size,
        patch_grid=args.patch_grid,
        n_patch_raw=args.n_patch_raw,
        reconstructor_mode=args.reconstructor_mode,
        image_token_dim=args.image_token_dim,
        image_encoder_layers=args.image_encoder_layers,
        image_encoder_heads=args.image_encoder_heads,
        chunk_size=args.chunk_size,
        max_length=args.max_length,
        delta_loss_weight=args.delta_loss_weight,
        progress_loss_weight=args.progress_loss_weight,
        end_loss_weight=args.end_loss_weight,
        end_pos_weight=args.end_pos_weight,
        end_threshold=args.end_threshold,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        epochs=args.epochs,
        val_split=args.val_split,
        log_every=args.log_every,
        device=device,
        save_path=str(output_dir / "FSQ.pt"),
        checkpoint_every=args.checkpoint_every,
        action_min=action_min,
        action_max=action_max,
        delta_min=delta_min,
        delta_max=delta_max,
    )

    skill_orders = _compute_skill_orders(metadata)
    model = train_spline_fsqae(
        segments=segments,
        dec_tokens=dec_tokens,
        patch_flags=patch_flags,
        decoder_states=dec_states,
        decoder_targets=dec_targets,
        cfg=cfg,
        wandb_run=wandb_run,
        metadata=metadata,
        resume_from=args.resume_from,
    )

    # save latent vectors and indices for all skills
    latent_codes = np.stack([
        model.encode_numpy(seg, dec_tokens[i][0], dec_tokens[i][min(len(dec_tokens[i]) - 1, m["length"] - 1)])
        for i, (seg, m) in enumerate(zip(segments, metadata))
    ])
    latent_tokens = np.array([
        model.encode_index(seg, dec_tokens[i][0], dec_tokens[i][min(len(dec_tokens[i]) - 1, m["length"] - 1)])
        for i, (seg, m) in enumerate(zip(segments, metadata))
    ], dtype=np.int32)

    latents_path = output_dir / "skill_latents.npz"
    save_dict: dict = {
        "latents":     latent_codes,
        "tokens":      latent_tokens,
        "skill_order": np.array(skill_orders, dtype=np.float32),
    }
    for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"):
        save_dict[key] = np.array([m[key] for m in metadata])
    np.savez(str(latents_path), **save_dict)
    print(f"[FSQ] Saved latents → {latents_path}")
    print(f"[FSQ] Saved model   → {output_dir / 'FSQ.pt'}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
