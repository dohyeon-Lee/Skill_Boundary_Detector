#!/usr/bin/env python3
"""outputs/*/metrics.json → 비교표 (stdout + outputs/summary.md).

해석 기준:
  z-std→0, w/r→0(1=구조없음), same셀/≤1셀→1, purity→1, agreement→1,
  n_skills_std/len_CV→0, R²→1(상대비교용), z-swap win→1 (0.5=z무시; 기존 FSQ 기준 0.94)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

LEGEND = """
## 지표 해석

"같은 행동" 프록시 = 같은 task의 첫 세그먼트(skill_index=0) 모음 — 같은 task의 demo들은
거의 같은 모션으로 시작하므로, 이들이 z-공간/코드에서 모이는지를 잰다.

### ① skill-space (skill_latents.npz)
| 지표 | 의미 | 이상값 |
|---|---|---|
| K / used | 코드북 크기 / 실제 사용된 코드 수 | — (죽은 코드 비율 참고) |
| ent | 코드 사용 엔트로피 ÷ log2(K). 1=완전 균등 | — (낮으면 collapse 의심, top1 확인) |
| z-std | 같은 행동 그룹의 z_q 차원별 표준편차 (셀=1) — 클러스터의 절대 폭 | **0** (1↑이면 한 셀에 안 담김) |
| w/r | 그룹 내 쌍 평균 L1거리 ÷ 무작위쌍 평균. K가 달라도 비교 가능한 응집도 | **0** (1=구조 없음) |
| same셀 | 같은 행동 쌍이 같은 코드를 받을 확률 — 이산 어휘의 재현성, stage2 VLM 타겟 안정성 | **1** |
| ≤1셀 | 인접 셀(L1≤1)까지 허용한 완화판 — z_q 벡터 입력에서의 실효 일관성 | **1** |
| purity | 한 코드에 모인 세그먼트들이 같은 (task,단계)인 비율 — "한 코드 = 한 의미" | **1** |

주의: same셀↑인데 purity↓이면 격자를 거칠게 해 얻은 가짜 일관성 (예: FSQ88).
둘 다 높아야 좋은 어휘.

### ③ label quality (skillvla parquet)
| 지표 | 의미 | 이상값 |
|---|---|---|
| 1st-agr | task별 "첫 코드 최빈 비율" 평균 — 같은 task가 같은 코드로 시작? | **1** |
| seq-agr | 코드 시퀀스 전체 일치판 (DP×FSQ 합성; oracle eval 주입 시퀀스의 안정성) | **1** |
| nskl-std | task 내 스킬 개수 std — DP 경계 '개수' 일관성 (순수 DP 지표) | **0** |
| len-CV | 첫 세그먼트 길이 변동계수 — DP 경계 '위치' 일관성 (순수 DP; 0.1~0.2는 자연 변동) | **0** |
| R² | 모션특징(EE 변위+그리퍼) 분산 중 코드가 설명하는 비율 — 코드↔실제 모션 결부 | **1** (특징이 거칠어 절대값 낮음 — 상대 비교용) |
| cent-acc | 코드별 모션 centroid 분류 정확도 (chance = 1/used) | **1** |

### ② z-swap (FSQ.pt — 디코더의 z 의존도)
디코더 입력 [z, 시작 state, 시작 이미지, GT progress]에서 z만 다른 세그먼트 것으로 교체.
| 지표 | 의미 | 이상값 |
|---|---|---|
| mse_true / mse_swap / ratio | 자기 z vs 남의 z 재구성 오차 — z의 조건부 정보량 | ratio **↑** |
| outΔ% | z 교체 시 출력 변화량 (GT action 스케일 대비) — z 민감도(반응 크기) | ↑ |
| win | P(자기 z가 남의 z보다 잘 복원) — 반응이 '옳은 방향'인가 | **1** (0.5 = z 무시) |

win은 stage1 teacher-forced win_rate의 사실상 **천장**: 기존 FSQ 디코더 기준 ≈0.94,
stage1 id-embedding 시절 ≈0.51. (teacher-forced 측정은 stage1_eval의 sweep 참조)
"""


def fmt_table(rows: list[dict]) -> str:
    lines = []
    h1 = (f"{'target':16s} {'K':>4s} {'used':>5s} {'ent':>5s} {'z-std':>15s} {'w/r':>5s} "
          f"{'same셀':>7s} {'≤1셀':>6s} {'purity':>6s}")
    lines.append("① skill-space")
    lines.append(h1)
    for r in rows:
        s = r.get("skill_space")
        if not s:
            continue
        zstd = "[" + " ".join(f"{x:.2f}" for x in s["z_std_per_dim"]) + "]"
        lines.append(f"{r['name']:16s} {s['K']:4d} {s['codes_used']:5d} {s['entropy_norm']:5.2f} "
                     f"{zstd:>15s} {s['within_over_random']:5.2f} {s['same_cell_rate']:6.1%} "
                     f"{s['le1_cell_rate']:5.1%} {s['purity']:6.2f}")
    lines.append("")
    lines.append("③ label quality")
    lines.append(f"{'target':16s} {'1st-agr':>7s} {'seq-agr':>7s} {'nskl-std':>8s} {'len-CV':>6s} "
                 f"{'R²':>6s} {'cent-acc':>8s}")
    for r in rows:
        s = r.get("label_quality")
        if not s:
            continue
        lines.append(f"{r['name']:16s} {s['first_code_agreement']:7.2f} {s['full_seq_agreement']:7.2f} "
                     f"{s['n_skills_std']:8.2f} {s['first_len_cv']:6.2f} {s['motion_r2']:6.3f} "
                     f"{s['centroid_acc']:8.3f}")
    lines.append("")
    lines.append("② z-swap (디코더의 z 의존도 — stage1 win_rate의 천장)")
    lines.append(f"{'target':16s} {'mse_true':>9s} {'mse_swap':>9s} {'ratio':>6s} {'outΔ%':>6s} {'win':>6s}")
    for r in rows:
        s = r.get("zswap")
        if not s:
            continue
        lines.append(f"{r['name']:16s} {s['mse_true']:9.4f} {s['mse_swap']:9.4f} {s['ratio']:5.1f}x "
                     f"{s['out_delta_pct_scale']:6.1%} {s['win_rate']:6.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", type=Path, default=OUTPUTS_DIR)
    args = ap.parse_args()
    rows = []
    for f in sorted(args.outputs.glob("*/metrics.json")):
        rows.append(json.loads(f.read_text()))
    if not rows:
        raise SystemExit(f"No metrics.json under {args.outputs}")
    table = fmt_table(rows)
    print(table)
    md = args.outputs / "summary.md"
    md.write_text("# FSQ utilized eval — comparison\n\n```\n" + table + "\n```\n" + LEGEND)
    print(f"\nsaved → {md}")


if __name__ == "__main__":
    main()
