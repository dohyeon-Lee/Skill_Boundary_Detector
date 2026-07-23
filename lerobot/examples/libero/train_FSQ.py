"""
Train SplineFSQAE: Finite Scalar Quantization skill tokenizer with DINO-token-conditioned decoder.

DINO 토큰은 ONLINE으로 계산한다 (디스크 precompute 완전 제거): 학습 시작 시 raw 데이터셋의 mp4를
파일당 1회 순차 디코드하며 frozen DINO(OnlineDino — FSQ 인퍼런스의 raw-이미지 경로와 동일 구현)로
warm-pass 인코딩해 per-skill 토큰을 RAM에 올린다. FSQ 학습은 스킬 전체 프레임을 수백 에폭 반복
소비하므로 per-batch 라이브는 같은 프레임을 수백 번 재계산하는 낭비 — warm-pass(시작 ~10-20분,
fp16 ~27GB/카메라 RAM)가 기존 precompute와 학습속도 동일 + 빌드단계/디스크 산출물 0 을 준다.
terminator가 dual(3rd+wrist)이면 두 카메라 모두 인코딩한다.

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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # lerobot (OnlineDino)

from lerobot.utils.online_dino import fit_feature_length  # noqa: E402  (per-skill 길이 맞춤)
from lerobot.utils.online_dino import OnlineDino, encode_episode_dino  # noqa: E402


# ── Args ───────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    # ── data
    skills_dir: str = ""
    """Directory (recursively searched) for per-skill .npz files."""
    raw_dataset_dir: str = ""
    """Raw LeRobot dataset (videos/ + meta/) — DINO 토큰의 원천. 학습 시작 시 online warm-pass로
    mp4를 디코드해 frozen DINO로 인코딩한다 (디스크 precompute 없음)."""
    image_key: str = "observation.images.image"
    """Primary 카메라 video key. wrist-only(wot) FSQ면 observation.images.wrist_image로 설정."""
    image_key_wrist: str = "observation.images.wrist_image"
    """Dual(both) terminator의 wrist 카메라 video key (terminator_use_wrist=True일 때만 사용)."""
    dino_batch_size: int = 256
    """Warm-pass DINO 인코딩의 GPU 프레임 배치 크기."""
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
    skill_cond_mode: str = "token"
    """token: z is a decoder token; broadcast: z conditions each decoder/terminator pool block input."""
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
    # Best-val SELECTION metric weights. Unset (None) → follow the actual loss (total val). Set to
    # override: FSQ.pt = argmin over epochs of wd*delta + wp*progress + we*end on val.
    val_select_delta_weight: float | None = None
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


def load_dino_tokens_online(raw_dataset_dir: str, metadata: list[dict], image_key: str,
                            dino: OnlineDino, batch_size: int) -> list[np.ndarray]:
    """Per-skill DINO token clips (T, n_tokens, feat_dim) — ONLINE warm-pass (디스크 precompute 없음).

    encode_episode_dino가 mp4를 파일당 1회 디코드해 에피소드별 (T,65,384) fp16을 RAM에 만들고,
    여기서 skill의 frame_start/frame_end로 슬라이스한다. 슬라이스는 axis0 연속이라 **뷰(no-copy)** —
    SplineFSQDataset의 ascontiguousarray도 이미 연속이면 복사하지 않으므로 총 메모리는 에피소드
    배열(~27GB/카메라, fp16)이 전부다. 결과 클립은 기존 precompute 경로(_LazyDinoClips /
    extract_skill_dino_tokens)와 수치 동일 (verify_online_dino.py: max|Δ|≈3e-3 = fp16 저장 반올림)."""
    ep_ids = sorted({int(m["episode_id"]) for m in metadata})
    ep_tok = encode_episode_dino(raw_dataset_dir, ep_ids, image_key, dino,
                                 batch_size=batch_size, log_prefix="[FSQ][OnlineDino]")
    clips = []
    for m in metadata:
        feats = ep_tok[int(m["episode_id"])]
        fs, fe = int(m["frame_start"]), int(m["frame_end"])
        clip = feats[fs:min(fe, len(feats))]          # 뷰 (에피소드 배열이 실소유)
        if clip.ndim == 2:                            # 방어: CLS-only (T,F) → (T,1,F)
            clip = clip[:, None, :]
        clips.append(fit_feature_length(clip, int(m["length"])))
    print(f"[FSQ] Online DINO tokens ← {raw_dataset_dir} (key={image_key}, {len(clips)} skills) "
          f"— clip0 {clips[0].shape}")
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

    # ── DINO warm-pass (ONLINE — 디스크 precompute 완전 제거) ──
    # mp4 → frozen DINO(OnlineDino; FSQ 인퍼런스 raw-이미지 경로와 동일 구현) → per-skill 토큰 (RAM).
    # dual(both) terminator면 wrist 카메라도 같은 방식으로 인코딩. warm-pass 후 DINO는 GPU에서 내림.
    if not args.raw_dataset_dir:
        raise ValueError("--raw_dataset_dir is required (LeRobot dataset with videos/ + meta/) — "
                         "DINO tokens are computed ONLINE from the raw videos now.")
    _dino_dev = "cuda" if torch.cuda.is_available() else "cpu"
    dino = OnlineDino(args.image_model_name, image_size=args.image_size,
                      patch_grid=args.patch_grid, n_patch_raw=args.n_patch_raw).to(_dino_dev)
    dec_tokens = load_dino_tokens_online(args.raw_dataset_dir, metadata, args.image_key,
                                         dino, args.dino_batch_size)
    if args.terminator_use_wrist:
        # wrist-only(wot) 모드는 primary=wrist라 두 키가 같음 — 같은 카메라를 두 번 인코딩하지 않고
        # 클립 리스트를 공유 (기존 lazy 모드의 mmap 공유와 동일한 의미; RAM/GPU 2배 낭비 방지).
        dec_tokens_wrist = (dec_tokens if args.image_key_wrist == args.image_key
                            else load_dino_tokens_online(args.raw_dataset_dir, metadata,
                                                         args.image_key_wrist, dino, args.dino_batch_size))
    else:
        dec_tokens_wrist = None                   # single-camera terminator
    del dino
    if _dino_dev == "cuda":
        torch.cuda.empty_cache()                  # FSQ 학습이 쓸 GPU 메모리 반환

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
        skill_cond_mode=args.skill_cond_mode,
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
        val_select_delta_weight=args.val_select_delta_weight,
        val_select_progress_weight=args.val_select_progress_weight,
        val_select_end_weight=args.val_select_end_weight,
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
