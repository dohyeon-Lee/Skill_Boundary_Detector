#!/usr/bin/env python
"""OnlineDino ≟ precompute 동치 검증 (DINO precompute 제거 작업의 관문).

기존 per-episode precompute 산출물(_work/dino/pg8/{cam}/episode_*.npz — ground truth)과 완전히
같은 프레임(같은 mp4, 같은 슬라이스)을 온라인으로 다시 인코딩해 수치 대조한다. 이게 통과해야
FSQ/terminator의 토큰 공급원을 OnlineDino로 교체해도 기존 체크포인트(FSQ.pt, stage1 terminator)와
호환된다는 보장이 선다.

기대치: GT는 fp16으로 저장 + CUDA fp16 autocast로 계산됨 →
  - CUDA에서 실행 시: max|Δ| ~1e-3 (fp16 반올림 수준), cos ≥ 0.9999
  - CPU에서 실행 시: autocast가 꺼져 fp32 forward → ~1e-2 수준까지 허용 (구조 검증용)

사용:
  PYTHONPATH=$PROJECT/lerobot/src:$PROJECT/lerobot/examples/libero \
    python verify_online_dino.py \
      --dataset_dir  $PROJECT/dataset_filtered/libero_90_full_full \
      --dino_dir     $PROJECT/dataset_filtered/skillvla_dataset/libero_90_full_full/_work/dino/pg8 \
      --episodes 0,7,42 --n_frames 8 --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _load_episodes_meta(dataset_dir: Path):
    import pandas as pd
    files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files under {dataset_dir / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _read_video_frames(path: Path) -> np.ndarray:
    """precompute_frame_dino_features._read_video_frames와 동일 (torchvision 우선, pyav 폴백)."""
    try:
        from torchvision.io import read_video
        frames, _, _ = read_video(str(path), output_format="THWC", pts_unit="sec")
        return frames.numpy().astype(np.uint8)[..., :3]
    except Exception:  # noqa: BLE001
        import av
        frames = []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24")[..., :3])
        if not frames:
            raise ValueError(f"No frames in {path}")
        return np.stack(frames).astype(np.uint8)


def _episode_frames(dataset_dir: Path, meta, ep_id: int, image_key: str, fps: float) -> np.ndarray:
    """precompute의 file_map 규약 그대로: mp4 파일 전체 디코드 → from_timestamp 슬라이스."""
    row = meta[meta["episode_index"] == ep_id].iloc[0]
    chunk_idx = int(row[f"videos/{image_key}/chunk_index"])
    file_idx = int(row[f"videos/{image_key}/file_index"])
    from_ts = float(row[f"videos/{image_key}/from_timestamp"])
    length = int(row["length"])
    frame_start = round(from_ts * fps)
    path = dataset_dir / "videos" / image_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
    frames = _read_video_frames(path)
    if frame_start + length > len(frames):
        raise ValueError(f"ep{ep_id}: slice [{frame_start}:{frame_start+length}] > file frames {len(frames)}")
    return frames[frame_start: frame_start + length]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--dino_dir", required=True, help="_work/dino/pg8 (per-episode GT npz의 루트)")
    ap.add_argument("--image_key", default="observation.images.image")
    ap.add_argument("--model_path", default="")
    ap.add_argument("--episodes", default="0,7,42")
    ap.add_argument("--n_frames", type=int, default=8, help="에피소드당 균등 샘플 프레임 수")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from lerobot.utils.online_dino import OnlineDino

    dataset_dir, dino_dir = Path(args.dataset_dir), Path(args.dino_dir)
    key_dir = dino_dir / args.image_key.replace("/", "_").replace(".", "_")
    manifest = json.loads((dino_dir / "manifest.json").read_text())
    model_path = args.model_path or manifest.get("image_model_name", "")
    print(f"[verify] GT manifest: model={model_path} grid={manifest.get('patch_grid')} "
          f"size={manifest.get('image_size')}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type != "cuda":
        print("[verify] WARNING: CPU 실행 — GT는 CUDA fp16 autocast로 계산됨 → 오차 상한이 커짐 (~1e-2)")
    dino = OnlineDino(model_path,
                      image_size=int(manifest.get("image_size", 224)),
                      patch_grid=int(manifest.get("patch_grid", 8)),
                      n_patch_raw=int(manifest.get("n_patch_raw", 196))).to(device)

    meta = _load_episodes_meta(dataset_dir)
    fps = float(json.loads((dataset_dir / "meta" / "info.json").read_text()).get("fps", 20.0))

    worst = {"max": 0.0, "mean": 0.0, "cos": 1.0}
    print(f"\n{'ep':>5} {'frames':>18} {'max|Δ|':>10} {'mean|Δ|':>10} {'min cos':>9}")
    for ep_id in [int(e) for e in args.episodes.split(",")]:
        gt = np.load(key_dir / f"episode_{ep_id:07d}.npz")
        gt_feat = gt["features"]                                   # (T, 65, 384) fp16
        frames = _episode_frames(dataset_dir, meta, ep_id, args.image_key, fps)
        if len(frames) != len(gt_feat):
            print(f"  ep{ep_id}: frame count mismatch video={len(frames)} gt={len(gt_feat)} — skip")
            continue
        idx = np.linspace(0, len(frames) - 1, args.n_frames).round().astype(int)
        x = torch.from_numpy(frames[idx].copy())                   # uint8 (N,H,W,3)
        with torch.no_grad():
            ours = dino(x.to(device)).cpu().numpy()                # (N, 65, 384) fp32
        ref = gt_feat[idx].astype(np.float32)
        d = np.abs(ours - ref)
        cos = np.array([
            float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            for a, b in zip(ours.reshape(len(idx), -1), ref.reshape(len(idx), -1))])
        print(f"{ep_id:>5} {str(idx.tolist()):>18} {d.max():>10.4f} {d.mean():>10.5f} {cos.min():>9.5f}")
        worst["max"] = max(worst["max"], float(d.max()))
        worst["mean"] = max(worst["mean"], float(d.mean()))
        worst["cos"] = min(worst["cos"], float(cos.min()))

    print(f"\nworst: max|Δ|={worst['max']:.4f} mean|Δ|={worst['mean']:.5f} min cos={worst['cos']:.5f}")
    tol = 5e-3 if device.type == "cuda" else 5e-2
    ok = worst["max"] <= tol and worst["cos"] >= 0.999
    print("✅ ONLINE == PRECOMPUTE (계약 재현 성공)" if ok else
          "⚠️ 허용오차 초과 — 전처리/풀링/모델 어딘가 계약과 다름. 교체 진행 금지.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
