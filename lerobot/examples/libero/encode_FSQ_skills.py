"""Encode skillset trajectories with a trained FSQ checkpoint.

This produces the skill_latents*.npz file consumed by codebook_visualizer.py
and decoder_eval.py. The FSQ encoder uses spline control points, skill length,
and start/end DINO tokens (encoding uses only the pure-DINO encoder).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from FSQ import SplineFSQAE, SplineFSQAEConfig
from train_FSQ import _compute_skill_orders, load_skill_files


@dataclass
class Args:
    model_path: str
    """FSQ checkpoint path: FSQ.pt or FSQ_epochXXXX.pt."""

    skills_dir: str
    """Directory containing per-skill npz files."""

    output_path: str
    """Output skill_latents npz path."""

    device: str = "cuda"

    # ── 전이 안전망 (B): 새 데이터셋 인코딩 시 미지원 코드 → 최근접 지원 코드로 snap ──
    snap_to_supported: bool = False
    """raw FSQ code가 참조(training) 데이터셋에서 미지원(빈/희귀)이면 가장 가까운 지원 코드로 옮김.
    다운스트림 VLA가 학습 못 한 코드로 스킬이 배정되는 것을 방지 (OOD graceful degradation)."""
    supported_freq_path: str = ""
    """지원 코드 참조 npz. training 빌드의 skill_code_freq.npz(motion_counts) 또는 skill_latents.npz(tokens)."""
    min_code_freq: int = 1
    """이 값 미만 빈도의 코드는 '미지원'으로 간주 → snap 대상. 1 = 완전 빈칸만; ↑ 하면 희귀코드도."""
    snap_metric: str = "l1"
    """격자 거리 (l1|l2)."""


def load_model(model_path: Path, device: str) -> SplineFSQAE:
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    cfg: SplineFSQAEConfig = ckpt["cfg"]
    model = SplineFSQAE(
        action_dim=cfg.action_dim,
        enc_dim=getattr(cfg, "enc_dim", 8),
        state_dim=cfg.state_dim,
        n_control=cfg.n_control,
        spline_degree=cfg.spline_degree,
        hidden_dim=cfg.hidden_dim,
        fsq_levels=cfg.fsq_levels,
        num_layers=cfg.num_layers,
        dropout=0.0,
        feat_dim=cfg.feat_dim,
        n_tokens=cfg.n_tokens,
        image_model_name=getattr(cfg, "image_model_name", "/data2/dohyeon/SBD/models/dinov3-vits16"),
        image_size=getattr(cfg, "image_size", 224),
        patch_grid=getattr(cfg, "patch_grid", 8),
        n_patch_raw=getattr(cfg, "n_patch_raw", 196),
        image_token_dim=getattr(cfg, "image_token_dim", 128),
        image_encoder_layers=getattr(cfg, "image_encoder_layers", 1),
        image_encoder_heads=getattr(cfg, "image_encoder_heads", 4),
        chunk_size=cfg.chunk_size,
        length_min=getattr(cfg, "length_min", 0.0),
        length_max=getattr(cfg, "length_max", 200.0),
        action_min=cfg.action_min,
        action_max=cfg.action_max,
        delta_min=cfg.delta_min,
        delta_max=cfg.delta_max,
        state_min=getattr(cfg, "state_min", None),
        state_max=getattr(cfg, "state_max", None),
        terminator_use_third=getattr(cfg, "terminator_use_third", True),
        terminator_use_wrist=getattr(cfg, "terminator_use_wrist", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def _grid_coords(model: SplineFSQAE) -> np.ndarray:
    """All C codes → their integer cell coords (C, D), matching FSQ.forward's index convention:
    level_d = (code // stride_d) % L_d ; coord_d = level_d - half_width_d.  (== encode_numpy's z_q space)"""
    fsq = model.fsq
    L = np.array([int(round(2 * h + 1)) for h in fsq.levels_half.cpu().tolist()])   # levels_half=(L-1)/2
    strides = fsq.strides.cpu().numpy()
    half = fsq.half_width.cpu().numpy()
    codes = np.arange(int(np.prod(L)))
    lvl = (codes[:, None] // strides[None, :]) % L[None, :]                          # (C, D) level 0..L-1
    return (lvl - half[None, :]).astype(np.float32)                                  # (C, D) integer coord


def _supported_from_freq(freq: np.ndarray, n_codes: int, min_freq: int) -> np.ndarray:
    return np.where(np.asarray(freq).ravel()[:n_codes] >= min_freq)[0]


def _load_supported(ref_path: Path, n_codes: int, min_freq: int) -> np.ndarray:
    """Reference npz → indices of supported codes (freq >= min_freq). Accepts skill_code_freq.npz
    (motion_counts) or skill_latents.npz (tokens → bincount)."""
    raw = np.load(str(ref_path))
    if "motion_counts" in raw:
        freq = raw["motion_counts"]
    elif "tokens" in raw:
        freq = np.bincount(raw["tokens"].astype(np.int64).ravel(), minlength=n_codes)
    else:
        raise KeyError(f"{ref_path}: need 'motion_counts' or 'tokens' key for supported-code reference.")
    return _supported_from_freq(freq, n_codes, min_freq)


def _snap_to_supported(latents: np.ndarray, tokens: np.ndarray, model: SplineFSQAE, args: Args):
    """Remap each skill whose raw code is unsupported → nearest supported code (grid distance in the
    integer cell-coord space, which is exactly what `latents` holds). Returns (latents, tokens)."""
    coords = _grid_coords(model)                                    # (C, D)
    if str(args.supported_freq_path).strip().lower() == "self":
        # self-pruning: 방금 인코딩한 RAW 토큰 분포가 곧 기준표 (외부 파일 불필요 — 1-pass 자기완결).
        # un-snap 빌드의 skill_code_freq.npz(motion_counts)를 참조하는 것과 수치적으로 동일.
        freq = np.bincount(tokens.astype(np.int64).ravel(), minlength=len(coords))
        supported = _supported_from_freq(freq, len(coords), args.min_code_freq)
    else:
        supported = _load_supported(Path(args.supported_freq_path), len(coords), args.min_code_freq)
    if len(supported) == 0:
        raise ValueError("no supported codes at this min_code_freq — lower --min_code_freq.")
    sup_coords = coords[supported]                                  # (S, D)
    is_sup = np.zeros(len(coords), dtype=bool)
    is_sup[supported] = True

    lat, tok = latents.copy(), tokens.copy()
    dead = ~is_sup[tokens]
    dists = []
    for i in np.where(dead)[0]:
        diff = sup_coords - latents[i][None, :]
        d = np.abs(diff).sum(1) if args.snap_metric == "l1" else (diff ** 2).sum(1)
        j = int(np.argmin(d))
        tok[i] = supported[j]
        lat[i] = sup_coords[j]
        dists.append(float(d[j]))
    n = len(dead)
    print(f"[FSQ encode] snap: {int(dead.sum())}/{n} skills ({dead.mean()*100:.1f}%) landed on "
          f"unsupported codes (freq<{args.min_code_freq}) → snapped to nearest of {len(supported)} "
          f"supported codes | snap dist({args.snap_metric}) mean={np.mean(dists) if dists else 0:.2f} "
          f"max={np.max(dists) if dists else 0:.0f}")
    return lat, tok


def main(args: Args) -> None:
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    skills_dir = Path(args.skills_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments, _, _, metadata = load_skill_files(skills_dir)

    model = load_model(Path(args.model_path), device)
    print(f"[FSQ encode] model={args.model_path}")
    print(f"[FSQ encode] device={device} skills={len(segments)}")

    latents = []
    tokens = []
    for seg in tqdm(segments, desc="Encoding FSQ skills"):  # action-only encoder: no images needed
        latents.append(model.encode_numpy(seg, device=device))
        tokens.append(model.encode_index(seg, device=device))

    latents_arr = np.stack(latents).astype(np.float32)
    tokens_arr = np.array(tokens, dtype=np.int32)

    if args.snap_to_supported:
        if not args.supported_freq_path:
            raise ValueError("--snap_to_supported requires --supported_freq_path (training code freq).")
        latents_arr, tokens_arr = _snap_to_supported(latents_arr, tokens_arr, model, args)

    save_dict: dict[str, np.ndarray] = {
        "latents": latents_arr,
        "tokens": tokens_arr.astype(np.int32),
        "skill_order": np.array(_compute_skill_orders(metadata), dtype=np.float32),
    }
    for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"):
        save_dict[key] = np.array([m[key] for m in metadata])
    np.savez(str(output_path), **save_dict)
    print(f"[FSQ encode] saved -> {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
