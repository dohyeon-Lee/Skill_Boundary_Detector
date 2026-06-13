#!/usr/bin/env python3
"""분석군 ③ 라벨 품질 (입력: {run_dir}/skillvla parquet).

  first_code_agreement : task별 "첫 코드 최빈 비율"의 평균 — 같은 task가 같은 코드로 시작? (이상 1)
  full_seq_agreement   : 시퀀스 전체 일치판 (DP×FSQ 합성; oracle eval 주입 시퀀스의 안정성) (이상 1)
  n_skills_std         : task 내 스킬 개수 std 평균 — DP 경계 '개수' 일관성 (이상 0)
  first_len_cv         : 첫 세그먼트 길이 변동계수 평균 — DP 경계 '위치' 일관성 (이상 0)
  motion_r2            : 모션특징(EE 6D 변위+그리퍼 평균) 분산 중 코드 그룹이 설명하는 비율 (이상 1,
                         특징이 거칠어 절대값은 낮음 — 상대 비교용)
  centroid_acc         : 코드별 모션 centroid 분류 정확도 (chance = 1/codes_used)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsq_utilized_eval_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH, fsq_levels_of, load_config, load_targets, merge_metrics, section_cached,
)

SECTION = "label_quality"


def evaluate(run_dir: Path, levels: list[int]) -> dict:
    K = int(np.prod(levels))
    root = run_dir / "skillvla"
    files = sorted((root / "data").glob("**/*.parquet"))
    cols = ["episode_index", "frame_index", "task_index", "skill_index",
            "skill_sequence", "skill_length_sequence", "observation.state"]
    df = pd.concat([pd.read_parquet(p, columns=cols) for p in files], ignore_index=True)

    # ── 시퀀스 단위 (에피소드 첫 행) ──
    first = df[df.frame_index == 0].sort_values("episode_index")
    task_seqs, task_lens = defaultdict(list), defaultdict(list)
    for _, r in first.iterrows():
        seq = np.asarray(r["skill_sequence"]).reshape(-1)
        lens = np.asarray(r["skill_length_sequence"]).reshape(-1)[: len(seq)]
        real = seq < K
        if real.sum() == 0:
            continue
        t = int(r["task_index"])
        task_seqs[t].append(tuple(int(x) for x in seq[real]))
        task_lens[t].append(int(lens[real][0]))

    a1, af, nstd, lcv, per_task = [], [], [], [], {}
    for t, seqs in task_seqs.items():
        c1 = Counter(s[0] for s in seqs).most_common(1)[0][1] / len(seqs)
        cf = Counter(seqs).most_common(1)[0][1] / len(seqs)
        ns = float(np.std([len(s) for s in seqs]))
        l1 = np.array(task_lens[t], dtype=float)
        cv = float(l1.std() / max(l1.mean(), 1))
        a1.append(c1); af.append(cf); nstd.append(ns); lcv.append(cv)
        per_task[t] = {"n": len(seqs), "first_agree": c1, "seq_agree": cf,
                       "n_skills_std": ns, "first_len_cv": cv}

    # ── 세그먼트 단위 모션특징 → 코드 설명력 ──
    df = df.sort_values(["episode_index", "frame_index"])
    seg = defaultdict(list)
    for _, g in df.groupby("episode_index"):
        st = np.stack(g["observation.state"].to_numpy())
        si = g["skill_index"].to_numpy()
        seq = [int(x) for x in np.asarray(g.iloc[0]["skill_sequence"]).reshape(-1)]
        for k in np.unique(si):
            m = si == k
            if k >= len(seq) or seq[k] >= K or m.sum() < 2:
                continue
            seg[seq[k]].append(np.concatenate([st[m][-1][:6] - st[m][0][:6], [st[m][:, 6:8].mean()]]))
    codes = [c for c, v in seg.items() if len(v) >= 10]
    X = np.array([f for c in codes for f in seg[c]])
    y = np.array([c for c in codes for _ in seg[c]])
    ss_tot = ((X - X.mean(0)) ** 2).sum()
    ss_w = sum(((X[y == c] - X[y == c].mean(0)) ** 2).sum() for c in codes)
    cent = np.stack([X[y == c].mean(0) for c in codes])
    pred = np.array(codes)[np.argmin(((X[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)]

    return {
        "K": K, "n_tasks": len(task_seqs), "n_episodes": int(len(first)),
        "first_code_agreement": float(np.mean(a1)),
        "full_seq_agreement": float(np.mean(af)),
        "n_skills_std": float(np.mean(nstd)),
        "first_len_cv": float(np.mean(lcv)),
        "motion_r2": float(1 - ss_w / ss_tot),
        "centroid_acc": float((pred == y).mean()),
        "centroid_chance": 1.0 / len(codes),
        "per_task": per_task,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for name, spec in load_targets(cfg).items():
        if section_cached(name, SECTION) and not args.force:
            print(f"[skip] {name}: {SECTION} cached")
            continue
        levels = fsq_levels_of(spec["run_dir"], cfg)
        res = evaluate(spec["run_dir"], levels)
        merge_metrics(name, SECTION, res)
        print(f"[done] {name}: 1st-agree {res['first_code_agreement']:.2f} | R² {res['motion_r2']:.3f}")


if __name__ == "__main__":
    main()
