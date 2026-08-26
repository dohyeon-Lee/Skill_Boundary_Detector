"""Encode skillset trajectories with a trained FSQ checkpoint.

This produces the skill_latents*.npz consumed by the SkillVLA data builders.
Only ``encoder.*`` is instantiated and loaded; the 300M action expert and image
terminator are deliberately absent from this build step.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from FSQ import SplineFSQEncoder, load_fsq_encoder
from train_FSQ import _compute_skill_orders, load_skill_files


@dataclass
class Args:
    skills_dir: str
    """Directory containing per-skill npz files."""

    model_path: str = ""
    """FSQ checkpoint path: FSQ.pt or FSQ_epochXXXX.pt. Omit when using --plan-path."""

    output_path: str = ""
    """Output skill_latents npz path. Omit when using --plan-path."""

    plan_path: str = ""
    """JSON file holding [{"model_path": ..., "output_path": ...}, ...].

    Reading the skillset dominates a single-checkpoint run (11k npz files versus
    ~40s of encoding), so a plan encodes many checkpoints of one run in one
    process and pays that cost once instead of once per checkpoint."""

    overwrite: bool = False
    """Re-encode a plan entry whose output already exists (default: skip it)."""

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


def load_model(model_path: Path, device: str) -> SplineFSQEncoder:
    try:
        model, _ = load_fsq_encoder(model_path, device)
    except Exception as v3_error:
        # FSQ-original (one-shot) checkpoints carry an FSQOriginalConfig `cfg`;
        # their model wraps the same SplineFSQEncoder and exposes the identical
        # encode_numpy / encode_index interface.
        try:
            from FSQ_original import load_fsq_original_model

            model, _ = load_fsq_original_model(model_path, device)
        except Exception:
            raise v3_error
        print(f"[FSQ encode] FSQ-original (one-shot) checkpoint: {model_path}")
    return model


def _grid_coords(model: SplineFSQEncoder) -> np.ndarray:
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


def _snap_to_supported(latents: np.ndarray, tokens: np.ndarray, model: SplineFSQEncoder, args: Args):
    """Remap each skill whose raw code is unsupported → nearest supported code (grid distance in the
    integer cell-coord space, which is exactly what `latents` holds). Returns (latents, tokens)."""
    if type(model).__name__ == "SplineFSQOriginalAE":
        model = model.encoder
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


def _encode_plan(args: Args) -> list[tuple[Path, Path]]:
    """(checkpoint, output) pairs to encode, from --plan-path or the single-run flags."""
    if args.plan_path:
        if args.model_path or args.output_path:
            raise ValueError("--plan-path replaces --model-path/--output-path; pass one form.")
        entries = json.loads(Path(args.plan_path).read_text())
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{args.plan_path} must hold a non-empty list of entries.")
        return [(Path(e["model_path"]), Path(e["output_path"])) for e in entries]
    if not args.model_path or not args.output_path:
        raise ValueError("Pass --model-path and --output-path, or --plan-path.")
    return [(Path(args.model_path), Path(args.output_path))]


def _encode_one(
    args: Args,
    *,
    model_path: Path,
    output_path: Path,
    device: str,
    segments,
    skill_actions,
    metadata,
) -> None:
    model = load_model(model_path, device)
    print(f"[FSQ encode] model={model_path}")
    print(f"[FSQ encode] device={device} skills={len(segments)}")

    latents = []
    tokens = []
    # action_seq checkpoints encode ACTION sequences; every other variant
    # encodes the state trajectory. No images either way.
    action_seq_encoder = getattr(getattr(model, "cfg", None), "encoder_arch", "spline") == "action_seq"
    source = skill_actions if action_seq_encoder else segments
    for item in tqdm(source, desc="Encoding FSQ skills"):
        if action_seq_encoder:
            latents.append(model.encode_actions_numpy(item, device=device))
            tokens.append(model.encode_actions_index(item, device=device))
        else:
            latents.append(model.encode_numpy(item, device=device))
            tokens.append(model.encode_index(item, device=device))

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
    # Write through a temp file so a killed job never leaves a half-written npz
    # that later runs would treat as an already-encoded checkpoint. The name must
    # keep the .npz suffix: np.savez appends one when it is missing, and the
    # rename would then target a file that was never written.
    temporary = output_path.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez(str(temporary), **save_dict)
    temporary.replace(output_path)
    print(f"[FSQ encode] saved -> {output_path}")


def main(args: Args) -> None:
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    plan = _encode_plan(args)
    for _, output_path in plan:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the skillset ONCE: it costs far more than a checkpoint's encoding.
    segments, _, skill_actions, metadata = load_skill_files(Path(args.skills_dir))

    for index, (model_path, output_path) in enumerate(plan, start=1):
        if output_path.is_file() and not args.overwrite:
            print(f"[FSQ encode] ({index}/{len(plan)}) exists, skipping -> {output_path}")
            continue
        print(f"[FSQ encode] ({index}/{len(plan)}) {model_path.name}")
        _encode_one(
            args,
            model_path=model_path,
            output_path=output_path,
            device=device,
            segments=segments,
            skill_actions=skill_actions,
            metadata=metadata,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
