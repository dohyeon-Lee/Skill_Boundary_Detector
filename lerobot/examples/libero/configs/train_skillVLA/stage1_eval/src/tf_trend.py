#!/usr/bin/env python3
"""Aggregate teacher-forced results across checkpoints of one Stage-1 run → trend table.

Reads every stage1_eval/outputs/{model_dir}_{ckpt}_adv-*/teacher_forced/teacher_forced.json
and prints skill-usage metrics per training step. win_rate rising from 0.5 over training is
THE signal that the policy started using the skill (z_q) condition; flat ≈0.5 means the
current recipe is insufficient. Reference points: FSQ decoder z-swap = 0.94 (uses z),
id-embedding Stage-1 runs = 0.51-0.54 (ignores skill).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", required=True, help="Stage-1 run folder name (without checkpoint)")
    ap.add_argument("--outputs_dir", default=str(Path(__file__).resolve().parent.parent / "outputs"))
    args = ap.parse_args()

    rows = []
    for f in Path(args.outputs_dir).glob(f"{args.model_dir}_*_adv-*/teacher_forced/teacher_forced.json"):
        m = re.match(rf"{re.escape(args.model_dir)}_(\w+)_adv-", f.parent.parent.name)
        if not m:
            continue
        s = json.load(open(f))["summary"]
        ckpt = m.group(1)
        step = int(ckpt) if ckpt.isdigit() else float("inf")  # "last" sorts last
        rows.append((step, ckpt, s))

    if not rows:
        raise SystemExit(f"No teacher_forced.json found for {args.model_dir} under {args.outputs_dir}")

    print(f"\n== teacher-forced trend: {args.model_dir} ==")
    print(f"{'ckpt':>8s} {'mse_true':>9s} {'mse_swap':>9s} {'swap_delta':>10s} {'%scale':>7s} {'win_rate':>8s}")
    for _, ckpt, s in sorted(rows):
        pct = 100.0 * s["skill_swap_delta"] / max(s["gt_action_step_norm"], 1e-9)
        print(f"{ckpt:>8s} {s['chunk_mse_true']:9.4f} {s['chunk_mse_swapped']:9.4f} "
              f"{s['skill_swap_delta']:10.4f} {pct:6.2f}% {s['true_code_win_rate']:8.3f}")
    print("(win_rate: 0.5 = skill ignored · →1.0 = skill drives the action · FSQ decoder ref 0.94)")


if __name__ == "__main__":
    main()
