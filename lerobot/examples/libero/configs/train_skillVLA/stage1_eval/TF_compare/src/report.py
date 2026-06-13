#!/usr/bin/env python3
"""Render the Stage-1 teacher-forced comparison table (outputs/summary.md) from cached
per-target metrics.json. Targets with no metrics yet are skipped (run run_targets.py first)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tf_compare_config as C

LEGEND = """\
## 지표 해석

teacher-forced: GT 데이터 프레임 + GT 스킬코드를 넣고, 그 코드를 틀린 코드로 바꿨을 때(swap)
액션 예측이 얼마나 달라지는지로 "stage1이 스킬을 실제로 쓰는가"를 잰다.

| 지표 | 의미 | 이상값 |
|---|---|---|
| mse_true | 맞는 스킬코드일 때 액션청크 예측 오차 (낮을수록 학습 잘됨; 높으면 언더트레이닝) | 낮을수록 |
| mse_swap | 틀린 스킬코드로 바꿨을 때 오차 | — |
| swapΔ | mse_swap − mse_true (스킬 바꿔서 나빠진 양) | 클수록 스킬 영향↑ |
| %scale | swapΔ ÷ GT 액션 스텝 스케일 (스킬 바꿔도 행동이 몇 % 변하나) | 클수록 |
| win | P(mse_true < mse_swap). 0.5=스킬 무시 · →1.0=스킬이 행동을 결정 | **→1.0** |

기준점: FSQ 디코더 z-swap 천장 ≈0.94 (FSQ_utilized_eval의 ② win) · id-embedding 구버전 0.51~0.54(무시).
win이 천장(≈0.94)에 가까울수록 stage1이 FSQ 스킬 스페이스를 제대로 활용하는 것.
mse_true가 높으면(=언더트레이닝) win이 낮아도 판단 보류 — 더 학습 후 재평가.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=C.DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = C.load(args.config)

    rows = []
    for name, _spec in C.iter_targets(cfg):
        mp = C.OUTPUTS_DIR / name / "metrics.json"
        if mp.exists():
            rows.append(json.loads(mp.read_text()))
    if not rows:
        raise SystemExit(f"No metrics.json under {C.OUTPUTS_DIR}; run run_targets.py first.")

    lines = ["# Stage-1 teacher-forced comparison", "", "```",
             f"{'target':<34s} {'ckpt':>8s} {'mse_true':>9s} {'mse_swap':>9s} "
             f"{'swapΔ':>8s} {'%scale':>7s} {'win':>6s}"]
    for m in rows:
        pct = 100.0 * m["skill_swap_delta"] / max(m["gt_action_step_norm"], 1e-9)
        lines.append(
            f"{m['name']:<34s} {m['checkpoint']:>8s} {m['chunk_mse_true']:9.4f} "
            f"{m['chunk_mse_swapped']:9.4f} {m['skill_swap_delta']:8.4f} {pct:6.2f}% "
            f"{m['true_code_win_rate']:6.3f}")
    lines += ["```", "", LEGEND]

    summary = C.OUTPUTS_DIR / "summary.md"
    summary.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {summary}")


if __name__ == "__main__":
    main()
