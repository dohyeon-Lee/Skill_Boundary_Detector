"""
Train SplineFSQAE: Finite Scalar Quantization skill tokenizer with DINO-token-conditioned decoder.

The terminator reads BOTH cameras (3rd-person + wrist) via separate DINO-token encoders, so
two skill-token npz files are required (one per camera, same skill order).

Usage:
    python examples/libero/train_FSQ.py \
      --skills_dir          /path/to/skillset/skills \
      --dino_features       /path/to/dino_tokens.npz \
      --dino_features_wrist /path/to/dino_tokens_wrist.npz \
      --output_dir          /path/to/output

Data requirements:
  skills_dir/*.npz       — per-skill npz with keys: actions, states, episode_id, skill_index,
                            frame_start, frame_end, task_id (optional)
  dino_features.npz       — precomputed 3rd-person DINO tokens (extract_skill_dino_tokens.py),
                            keys: features (N_total, n_tokens, feat_dim), offsets, episode_id,
                            frame_start, frame_end, length
  dino_features_wrist.npz — the same, extracted from the wrist camera (same skill order).
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).parent))

from precompute_dino_features import fit_feature_length  # noqa: E402  (lazy per-skill DINO slicing)


# ── Args ───────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    # ── data
    skills_dir: str = ""
    """Directory (recursively searched) for per-skill .npz files."""
    dino_features: str = ""
    """3rd-person DINO source. EITHER a per-frame DINO dir (precompute_frame_dino_features.py output)
    → tokens sliced lazily on the fly (no materialized file, saves ~27GB/camera/skillset); OR a
    precomputed token npz (extract_skill_dino_tokens.py). Both give identical per-skill clips."""
    dino_features_wrist: str = ""
    """Wrist DINO token npz (same skill order). Required iff terminator_use_wrist AND dino_features is a
    materialized npz. In lazy mode (dino_features is a dir) wrist is read from the SAME dir."""
    image_key: str = "observation.images.image"
    """Lazy mode only: primary-camera subdir of the per-frame DINO dir. For wrist-only FSQ set this to
    observation.images.wrist_image. Ignored when dino_features is a materialized npz."""
    image_key_wrist: str = "observation.images.wrist_image"
    """Lazy mode only: wrist-camera subdir for the dual-camera terminator. Ignored otherwise."""
    terminator_use_third: bool = True
    """Terminator 3rd-person camera. With terminator_use_wrist → 3 modes: 3rd-only / both / wrist-only."""
    terminator_use_wrist: bool = True
    """Terminator cameras: True = 3rd-person + wrist (two DINO encoders); False = 3rd-person only
    (original single-camera FSQ; no wrist tokens needed, old checkpoints stay loadable)."""
    output_dir: str = ""
    """Output directory. Defaults to parent of skills_dir."""

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
    image_token_dim: int = 128
    """Internal per-patch width N in the image encoders (must be divisible by heads, >2)."""
    image_encoder_layers: int = 1
    image_encoder_heads: int = 4
    chunk_size: int = 10
    """Steps per motion action chunk."""

    # ── loss
    delta_loss_weight: float = 1.0
    progress_loss_weight: float = 1.0
    end_loss_weight: float = 10.0
    end_pos_weight: float = 1.0
    """BCE positive-class weight for the termination head."""
    end_threshold: float = 0.5
    end_target_sigma: float = 0.0
    """Soft termination target std in frames (Gaussian bump at the skill end). 0 = hard 1-frame
    spike. σ≈2-3 curbs the val overfit a sharp spike causes and adds ±tolerance to recall/precision."""
    weighted_loss: bool = False
    """Per-frame end-weight the reconstructor delta loss (w=1+progress; skill END ~2× the start).
    Steers the FSQ latent toward the skill's latter/handoff portion. progress/termination unchanged."""

    # ── training
    epochs: int = 5000
    lr: float = 3e-4
    batch_size: int = 64
    num_workers: int = 8
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

    dec_targets = _make_episode_future_targets(dec_targets, metadata)

    print(f"[FSQ] Loaded {len(segments)} skills from {skills_dir}")
    lengths = [m["length"] for m in metadata]
    print(f"[FSQ] Skill lengths — min:{min(lengths)} max:{max(lengths)} mean:{np.mean(lengths):.1f}")
    return segments, dec_states, dec_targets, metadata


def _image_key_norm(image_key: str) -> str:
    """Per-frame DINO camera subdir name (matches extract_skill_dino_tokens.py / precompute)."""
    return image_key.replace(".", "_").replace("/", "_")


class _LazyDinoClips:
    """Per-skill DINO token sequences sliced ON-DEMAND from per-frame DINO npz — no materialized token
    file (saves ~27GB/camera/skillset + the extract step). Indexable like list[np.ndarray]; clip i ==
    fit_feature_length(episode_features[frame_start:frame_end], length), byte-identical to
    extract_skill_dino_tokens.py. A small LRU of episode memmaps keeps grouped/sequential access to one
    open file per episode; mmap means only the sliced frames are paged in.

    Consumed in the MAIN process (SplineFSQDataset copies it into a plain list before any DataLoader
    fork), so the open memmap handles never need to be pickled to workers.
    """

    _CACHE_MAX = 8

    def __init__(self, frame_dino_dir: Path, metadata: list[dict], image_key: str):
        frame_dino_dir = Path(frame_dino_dir)
        self._ep_dir = frame_dino_dir / _image_key_norm(image_key)
        if not self._ep_dir.is_dir():
            avail = [p.name for p in frame_dino_dir.iterdir() if p.is_dir()] if frame_dino_dir.is_dir() else []
            raise FileNotFoundError(
                f"Per-frame DINO camera dir not found: {self._ep_dir}\n"
                f"  frame_dino_dir={frame_dino_dir}  image_key={image_key}\n  available cameras: {avail}")
        self._meta = metadata
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._meta)

    def _episode_features(self, ep_id: int) -> np.ndarray:
        c = self._cache
        feats = c.get(ep_id)
        if feats is not None:
            c.move_to_end(ep_id)
            return feats
        f = self._ep_dir / f"episode_{ep_id:07d}.npz"
        if not f.exists():
            raise FileNotFoundError(f"Per-frame DINO episode file missing: {f}")
        feats = np.load(str(f), mmap_mode="r")["features"]   # (T, n_tokens, F) memmap
        c[ep_id] = feats
        if len(c) > self._CACHE_MAX:
            c.popitem(last=False)
        return feats

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self._meta)
        m = self._meta[i]
        feats = self._episode_features(int(m["episode_id"]))
        fs, fe = int(m["frame_start"]), int(m["frame_end"])
        clip = np.ascontiguousarray(feats[fs:min(fe, len(feats))])   # copy the small slice into RAM
        if clip.ndim == 2:                       # legacy CLS-only (T, F) → (T, 1, F)
            clip = clip[:, None, :]
        return fit_feature_length(clip, int(m["length"]))

    def __iter__(self):
        for i in range(len(self._meta)):
            yield self[i]


def load_dino_tokens(
    source: Path,
    metadata: list[dict],
    image_key: str = "observation.images.image",
) -> list[np.ndarray]:
    """Per-skill DINO token sequences (T, n_tokens, feat_dim).

    `source` is EITHER a per-frame DINO **directory** (sliced lazily on access — no materialized file,
    saves ~27GB/camera/skillset; `image_key` selects the camera subdir) OR a materialized token
    **npz** (extract_skill_dino_tokens.py; `image_key` ignored). Both yield identical clips.

    A materialized npz must contain:
      features  : (N_total, n_tokens, feat_dim) float16/float32
      offsets   : (N_skills + 1,) int64
      episode_id, frame_start, frame_end, length : (N_skills,)
    """
    source = Path(source)
    if source.is_dir():
        clips = _LazyDinoClips(source, metadata, image_key)
        print(f"[FSQ] Lazy DINO tokens ← {source} (key={image_key}, {len(clips)} skills) — clip0 {clips[0].shape}")
        return clips

    # mmap_mode='r': file is memory-mapped — initial open is instant and only accessed
    # pages are read from disk. Critical for 30+ GB npz files where a full upfront
    # np.load would block for >1h on a cold page cache.
    d = np.load(str(source), allow_pickle=False, mmap_mode='r')
    # Keep the native dtype (usually float16); downstream consumers convert per
    # clip, so a full float32 copy here would just waste memory/time and risk swapping.
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

    print(f"[FSQ] Loaded DINO tokens from {source} — shape per clip: {clips[0].shape}")
    return clips


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    from FSQ import SplineFSQAEConfig, train_spline_fsqae, zero_ground_trajectory

    skills_dir = Path(args.skills_dir)
    output_dir = Path(args.output_dir) if args.output_dir else skills_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    segments, dec_states, dec_targets, metadata = load_skill_files(skills_dir)

    dec_tokens = load_dino_tokens(Path(args.dino_features), metadata, image_key=args.image_key)
    if args.terminator_use_wrist:
        # Lazy mode: wrist comes from the SAME per-frame DINO dir via its wrist subdir (image_key_wrist).
        # Materialized mode: wrist is a separate token npz (dino_features_wrist).
        if Path(args.dino_features).is_dir():
            dec_tokens_wrist = load_dino_tokens(Path(args.dino_features), metadata, image_key=args.image_key_wrist)
        else:
            if not args.dino_features_wrist:
                raise ValueError("--dino_features_wrist is required when terminator_use_wrist=True (materialized mode).")
            dec_tokens_wrist = load_dino_tokens(Path(args.dino_features_wrist), metadata)
    else:
        dec_tokens_wrist = None  # single-camera terminator

    # Encoder stats on ZERO-GROUNDED trajectories — must match what the encoder normalizes
    # (grounded control points). Length stats are data-driven min/max over skill lengths, used for
    # the [-1,1] length-token normalization (same scheme as control points).
    grounded = np.concatenate([zero_ground_trajectory(s) for s in segments])
    all_tgt = np.concatenate([target[: int(m["length"])] for target, m in zip(dec_targets, metadata, strict=True)])
    action_min, action_max = grounded.min(0), grounded.max(0)
    delta_min,  delta_max  = all_tgt.min(0), all_tgt.max(0)
    all_state = np.concatenate(dec_states)   # decoder/terminator proprioception (raw, all timesteps)
    state_min,  state_max  = all_state.min(0), all_state.max(0)
    _lens = [int(m["length"]) for m in metadata]
    length_min, length_max = float(min(_lens)), float(max(_lens))
    np.savez(str(output_dir / "action_stats.npz"),
             action_min=action_min, action_max=action_max,
             delta_min=delta_min,   delta_max=delta_max,
             state_min=state_min,   state_max=state_max,
             length_min=np.float32(length_min), length_max=np.float32(length_max))
    print(f"[FSQ] action_min: {np.round(action_min, 4)}")
    print(f"[FSQ] action_max: {np.round(action_max, 4)}")
    print(f"[FSQ] delta_min:  {np.round(delta_min,  4)}")
    print(f"[FSQ] delta_max:  {np.round(delta_max,  4)}")
    print(f"[FSQ] state_min:  {np.round(state_min,  4)}")
    print(f"[FSQ] state_max:  {np.round(state_max,  4)}")
    print(f"[FSQ] length_min/max: {length_min:.0f} / {length_max:.0f}")

    device = args.device if torch.cuda.is_available() else "cpu"
    enc_dim    = segments[0].shape[-1]
    action_dim = dec_targets[0].shape[-1]
    state_dim  = dec_states[0].shape[-1]

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), "action_dim": action_dim, "enc_dim": enc_dim,
                    "state_dim": state_dim, "n_segments": len(segments)},
        )

    cfg = SplineFSQAEConfig(
        action_dim=action_dim,
        enc_dim=enc_dim,
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
        terminator_use_third=args.terminator_use_third,
        terminator_use_wrist=args.terminator_use_wrist,
        image_token_dim=args.image_token_dim,
        image_encoder_layers=args.image_encoder_layers,
        image_encoder_heads=args.image_encoder_heads,
        chunk_size=args.chunk_size,
        length_min=length_min,
        length_max=length_max,
        delta_loss_weight=args.delta_loss_weight,
        progress_loss_weight=args.progress_loss_weight,
        end_loss_weight=args.end_loss_weight,
        end_pos_weight=args.end_pos_weight,
        end_threshold=args.end_threshold,
        end_target_sigma=args.end_target_sigma,
        weighted_loss=args.weighted_loss,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
        state_min=state_min,
        state_max=state_max,
    )

    train_spline_fsqae(
        segments=segments,
        dec_tokens=dec_tokens,
        dec_tokens_wrist=dec_tokens_wrist,
        decoder_states=dec_states,
        decoder_targets=dec_targets,
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
