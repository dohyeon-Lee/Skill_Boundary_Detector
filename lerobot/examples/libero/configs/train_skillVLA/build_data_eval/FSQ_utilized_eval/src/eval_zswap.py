#!/usr/bin/env python3
"""분석군 ② z-swap — FSQ 디코더의 z 의존도 (run_dir만으로 동작; FSQ_inputs 불필요).

디코더 입력 [z, 시작 state, 시작 이미지, GT progress]에서 z만 다른 세그먼트 것으로 교체:
  mse_true / mse_swap : 자기 z vs 남의 z 재구성 MSE — z의 조건부 정보량 (비율↑ = z 의존↑)
  out_delta           : z 교체 시 출력 chunk 변화량 (GT action 스케일 대비) — z 민감도
  win_rate            : P(mse_true < mse_swap) — 0.5 = z 무시, →1 = z가 궤적 결정
                        (stage1 teacher-forced win_rate의 사실상 상한 = '천장')

소스: GT action/state = skillvla parquet에서 세그먼트 재구성, z = skill_latents.npz,
이 v2 실험의 action expert는 이미지를 받지 않으므로 state+z만 교체한다. fsq_ckpt override 시 encoder(z)-decoder
epoch 불일치 경고.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsq_utilized_eval_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH, load_config, load_fsq, load_targets, merge_metrics, section_cached,
)

SECTION = "zswap"


def build_segments(run_dir: Path, K: int) -> tuple[list[dict], dict]:
    """skillvla parquet → 세그먼트 목록 (GT future actions, states, 시작 프레임 global index)."""
    root = run_dir / "skillvla"
    files = sorted((root / "data").glob("**/*.parquet"))
    cols = ["episode_index", "frame_index", "index", "task_index", "skill_index",
            "skill_sequence", "skill_initial_frame", "observation.state", "action"]
    df = pd.concat([pd.read_parquet(p, columns=cols) for p in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"])

    segments = []
    for ep, g in df.groupby("episode_index"):
        acts = np.stack(g["action"].to_numpy()).astype(np.float32)
        states = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        gidx = g["index"].to_numpy()
        seq = [int(x) for x in np.asarray(g.iloc[0]["skill_sequence"]).reshape(-1)]
        ifs = [int(x) for x in np.asarray(g.iloc[0]["skill_initial_frame"]).reshape(-1)]
        si = g["skill_index"].to_numpy()
        for k in np.unique(si):
            if k >= len(seq) or seq[k] >= K:
                continue
            m = si == k
            t0 = int(np.argmax(m))                     # 세그먼트 시작 행
            T = int(m.sum())
            segments.append({
                "episode_id": int(ep), "frame_start": int(ifs[k]) if k < len(ifs) else t0,
                "states": states[m], "future": acts[m],
                "start_global_index": int(gidx[t0]),
            })
    return segments, {"root": root}


@torch.no_grad()
def evaluate(name: str, spec: dict, cfg: dict, n_segments: int, n_swap: int, seed: int) -> dict:
    fsq, fsq_cfg = load_fsq(spec["run_dir"], spec["fsq_ckpt"], cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fsq = fsq.to(device)
    if device.type == "cuda" and str(fsq.cfg.expert_dtype) == "bfloat16":
        fsq.action_expert.to(dtype=torch.bfloat16)
    elif device.type == "cuda" and str(fsq.cfg.expert_dtype) == "float16":
        fsq.action_expert.to(dtype=torch.float16)
    if spec["fsq_ckpt"] is not None:
        print(f"[warn] {name}: fsq_ckpt override — skill_latents.npz의 z(인코더)와 다른 epoch 디코더의 교차 실험입니다.")

    K_chunk, A = int(fsq.chunk_size), int(fsq.action_dim)
    levels = [int(x) for x in fsq_cfg["fsq_levels"]]
    segments, _ = build_segments(spec["run_dir"], int(np.prod(levels)))

    lat = np.load(spec["run_dir"] / "skill_latents.npz")
    zmap = {(int(e), int(s)): lat["latents"][i]
            for i, (e, s) in enumerate(zip(lat["episode_id"], lat["frame_start"]))}

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(segments), size=min(n_segments, len(segments)), replace=False)
    res = {"mse_true": [], "mse_swap": [], "out_delta": []}
    skipped = 0
    for count, i in enumerate(idxs):
        s = segments[i]
        key = (s["episode_id"], s["frame_start"])
        if key not in zmap:
            skipped += 1
            continue
        T = len(s["states"])
        states = torch.from_numpy(s["states"][:T]).unsqueeze(0).to(device)
        fut = s["future"]
        gt = np.zeros((T, K_chunk, A), dtype=np.float32)
        mask = np.zeros((T, K_chunk), dtype=bool)
        for t in range(T):
            n = min(K_chunk, len(fut) - t)
            gt[t, :n], mask[t, :n] = fut[t:t + n], True
        gt_t = torch.from_numpy(gt).to(device)
        mask_t = torch.from_numpy(mask)[..., None].float().to(device)
        noise = torch.randn(
            T, K_chunk, fsq.cfg.max_action_dim, device=device,
            generator=torch.Generator(device=device).manual_seed(seed + int(i)),
        )

        def recon(zvec):
            z = torch.as_tensor(np.asarray(zvec, dtype=np.float32), device=device).unsqueeze(0)
            return fsq.sample_action_chunks(z, states, noise=noise)[0]

        pred_true = recon(zmap[key])
        res["mse_true"].append(float((((pred_true - gt_t) ** 2) * mask_t).sum() / mask_t.sum() / A))
        for _ in range(n_swap):
            j = segments[int(rng.integers(len(segments)))]
            kj = (j["episode_id"], j["frame_start"])
            if kj not in zmap or kj == key:
                continue
            pred_s = recon(zmap[kj])
            res["mse_swap"].append(float((((pred_s - gt_t) ** 2) * mask_t).sum() / mask_t.sum() / A))
            res["out_delta"].append(float((pred_s - pred_true).abs().mean()))
        if (count + 1) % 50 == 0:
            print(f"  {name}: {count + 1}/{len(idxs)} segments")

    mt, ms, od = (np.array(res[k]) for k in ("mse_true", "mse_swap", "out_delta"))
    wins, k = [], 0
    for v in mt:
        c = min(n_swap, len(ms) - k)
        wins.extend(v < ms[k:k + c])
        k += c
    gt_scale = float(np.mean([np.abs(s["future"][:len(s["states"])]).mean() for s in segments[:500]]))
    return {
        "n_segments": int(len(mt)), "n_swaps": int(len(ms)), "skipped": skipped,
        "fsq_ckpt": str(spec["fsq_ckpt"] or (spec["run_dir"] / "FSQ.pt")),
        "mse_true": float(mt.mean()), "mse_swap": float(ms.mean()),
        "ratio": float(ms.mean() / max(mt.mean(), 1e-12)),
        "out_delta": float(od.mean()), "gt_action_scale": gt_scale,
        "out_delta_pct_scale": float(od.mean() / max(gt_scale, 1e-12)),
        "win_rate": float(np.mean(wins)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not bool(cfg.get("run_zswap", True)):
        print("[skip] run_zswap: false")
        return
    n_seg = int(cfg.get("zswap_segments", 300))
    n_swap = int(cfg.get("zswap_swaps", 4))
    for name, spec in load_targets(cfg).items():
        if section_cached(name, SECTION) and not args.force:
            print(f"[skip] {name}: {SECTION} cached")
            continue
        res = evaluate(name, spec, cfg, n_seg, n_swap, int(cfg.get("seed", 0)))
        merge_metrics(name, SECTION, res)
        print(f"[done] {name}: ratio {res['ratio']:.1f}x | win {res['win_rate']:.3f}")


if __name__ == "__main__":
    main()
