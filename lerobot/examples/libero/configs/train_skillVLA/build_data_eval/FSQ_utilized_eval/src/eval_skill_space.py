#!/usr/bin/env python3
"""분석군 ① skill-space 구성 품질 (입력: {run_dir}/skill_latents.npz).

"같은 행동" 프록시 = 같은 task의 첫 세그먼트(skill_index==0) 그룹. 메트릭:
  codes_used / K        : 사용된 고유 코드 수 / 코드북 크기
  entropy_norm          : 코드 사용 분포 엔트로피 / log2(K)  (1 = 완전 균등; collapse 검출용)
  top1_share            : 최빈 코드 점유율
  z_std_per_dim         : 그룹별 z_q 차원별 std(셀=1)의 task 평균 — 클러스터의 절대 폭 (이상 0)
  within_over_random    : 그룹 내 쌍 평균 L1 ÷ (그룹 vs 전체) 평균 L1 — K-중립 응집도 (이상 0, 1=구조없음)
  same_cell_rate        : 그룹 쌍이 같은 코드(L1=0)일 확률 — 이산 어휘 일관성 (이상 1)
  le1_cell_rate         : L1<=1 완화판 (이상 1)
  purity                : 코드별 최빈 (task,skill_index) 비율 평균, 출현 5회+ 코드만 (이상 1)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsq_utilized_eval_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH, fsq_levels_of, load_config, load_targets, merge_metrics, section_cached,
)

SECTION = "skill_space"


def evaluate(run_dir: Path, levels: list[int], seed: int) -> dict:
    d = np.load(run_dir / "skill_latents.npz")
    z, tok, task, sidx = d["latents"], d["tokens"], d["task_id"], d["skill_index"]
    K = int(np.prod(levels))
    rng = np.random.default_rng(seed)

    cnt = Counter(tok.tolist())
    probs = np.array(list(cnt.values())) / len(tok)
    entropy = float(-(probs * np.log2(probs)).sum())

    spread, within, rand, same, le1, per_task = [], [], [], [], [], {}
    for t in np.unique(task):
        m = (task == t) & (sidx == 0)
        if m.sum() < 10:
            continue
        zz = z[m]
        spread.append(zz.std(0))
        i = rng.integers(0, len(zz), 300)
        j = rng.integers(0, len(zz), 300)
        dd = np.abs(zz[i] - zz[j]).sum(1)
        k = rng.integers(0, len(z), 300)
        rd = np.abs(zz[i] - z[k]).sum(1)
        within.append(dd.mean()); rand.append(rd.mean())
        same.append((dd == 0).mean()); le1.append((dd <= 1).mean())
        per_task[int(t)] = {"n": int(m.sum()), "same_cell": float((dd == 0).mean()),
                            "z_std": [float(x) for x in zz.std(0)]}

    key = task * 100 + np.minimum(sidx, 99)
    purities = [Counter(key[tok == c].tolist()).most_common(1)[0][1] / (tok == c).sum()
                for c in np.unique(tok) if (tok == c).sum() >= 5]

    return {
        "K": K, "levels": levels, "n_segments": int(len(tok)),
        "codes_used": len(cnt),
        "entropy_norm": entropy / float(np.log2(K)),
        "top1_share": float(max(probs)),
        "z_std_per_dim": [float(x) for x in np.mean(spread, 0)],
        "within_over_random": float(np.mean(within) / np.mean(rand)),
        "same_cell_rate": float(np.mean(same)),
        "le1_cell_rate": float(np.mean(le1)),
        "purity": float(np.mean(purities)),
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
        res = evaluate(spec["run_dir"], levels, int(cfg.get("seed", 0)))
        merge_metrics(name, SECTION, res)
        print(f"[done] {name}: same_cell {res['same_cell_rate']:.1%} | w/r {res['within_over_random']:.2f} "
              f"| purity {res['purity']:.2f}")


if __name__ == "__main__":
    main()
