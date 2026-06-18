#!/usr/bin/env python3
"""Render a readable summary.md from a state_probe.json (pi05 vs cond state-influence comparison).

Usage:
  report.py --output_dir <run dir with state_probe.json>   # writes <run dir>/summary.md
  report.py                                                 # render every outputs/*/state_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve()
OUTPUTS_DIR = _HERE.parents[1] / "outputs"


def _state_rows(table: dict) -> list[str]:
    return sorted([n for n in table if n.startswith("state_")])


def render(results: dict) -> str:
    labels = list(results.keys())
    perts = sorted({n for r in results.values() for n in r["perturbations"]},
                   key=lambda n: (not n.startswith("state_"), n))  # state_* first

    L = ["# State-influence probe — pi05 vs SkillVLA-cond", "",
         "How much the predicted **action chunk** moves when ONE input is perturbed (before preprocessing, "
         "so each model's own state path runs), as a fraction of the GT action step norm. "
         "Larger ⇒ the model relies on that input more.", "",
         "- `state_shuffle` — swap in another frame's state (in-distribution; **the fair state metric**)",
         "- `state_gauss@σ` — add σ·std noise to the state",
         "- `image_zero` — blank the images (vision-dominance reference)",
         "- `skill_swap` — swap the GT FSQ code (cond/skill models only)", ""]

    # Main table: Δ/gt_norm per perturbation × model.
    L.append("## Relative influence (Δ ÷ gt_action_step_norm)")
    L.append("")
    L.append("| perturbation | " + " | ".join(labels) + " |")
    L.append("|" + "---|" * (len(labels) + 1))
    for n in perts:
        cells = []
        for l in labels:
            m = results[l]["perturbations"].get(n)
            cells.append(f"{m['rel_to_gt_norm']:.3f}" if m else "—")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L.append("| _gt_action_step_norm_ | " + " | ".join(f"{results[l]['gt_action_step_norm']:.4f}" for l in labels) + " |")
    L.append("")

    # Verdict: state_shuffle across models + state-vs-image ratio per model.
    L.append("## Reading")
    L.append("")
    for l in labels:
        t = results[l]["perturbations"]
        sh = t.get("state_shuffle", {}).get("rel_to_gt_norm")
        iz = t.get("image_zero", {}).get("rel_to_gt_norm")
        line = f"- **{l}**: state_shuffle Δ/gt = `{sh:.3f}`" if sh is not None else f"- **{l}**: (no state_shuffle)"
        if sh is not None and iz:
            line += f"  (= {sh / iz * 100:.0f}% of image_zero `{iz:.3f}`)"
        L.append(line)
    # Cross-model ratio on the fair metric.
    shuffles = {l: results[l]["perturbations"].get("state_shuffle", {}).get("rel_to_gt_norm") for l in labels}
    if "pi05" in shuffles and "cond" in shuffles and shuffles["pi05"] and shuffles["cond"]:
        ratio = shuffles["pi05"] / shuffles["cond"]
        L.append("")
        L.append(f"**pi05 / cond state_shuffle ratio ≈ {ratio:.1f}×** — "
                 + ("pi05 uses state far more; the cond 1-token state is effectively ignored (dilution)."
                    if ratio >= 3 else
                    "comparable — the cond state is used similarly to pi05's (no strong dilution)."))
    L.append("")
    L.append("> state_* Δ ≈ 0 **and** << image_zero / skill_swap ⇒ the state input is effectively a dead "
             "input for that model. Fixes: per-dim state tokens, or cond_attend_language.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, default=None, help="run dir with state_probe.json (default: all of outputs/)")
    args = ap.parse_args()

    if args.output_dir:
        dirs = [args.output_dir]
    else:
        dirs = sorted(p.parent for p in OUTPUTS_DIR.glob("*/state_probe.json")) if OUTPUTS_DIR.is_dir() else []
    if not dirs:
        print("No state_probe.json found.")
        return
    for d in dirs:
        jp = d / "state_probe.json"
        if not jp.is_file():
            print(f"[skip] no state_probe.json in {d}")
            continue
        md = render(json.loads(jp.read_text()))
        (d / "summary.md").write_text(md)
        print(f"[report] {d / 'summary.md'}")


if __name__ == "__main__":
    main()
