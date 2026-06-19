#!/usr/bin/env python3
"""Render a readable summary.md from an input_probe.json (state/skill/progress influence comparison).

Usage:
  report.py --output_dir <run dir with input_probe.json>   # writes <run dir>/summary.md
  report.py                                                 # render every outputs/*/input_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve()
OUTPUTS_DIR = _HERE.parents[1] / "outputs"

_GROUPS = [("state", "state_"), ("skill", "skill_"), ("progress", "progress_")]


def _order(n: str):
    """Sort key: state_ first, then skill_, progress_, then the rest (image_zero) last."""
    for i, (_, pref) in enumerate(_GROUPS):
        if n.startswith(pref):
            return (i, n)
    return (len(_GROUPS), n)


def render(results: dict) -> str:
    labels = list(results.keys())
    perts = sorted({n for r in results.values() for n in r["perturbations"]}, key=_order)

    L = ["# Input-influence probe — state / skill / progress", "",
         "How much the predicted **action chunk** moves when ONE input is perturbed (before preprocessing, "
         "so each model's own encoder path runs), as a fraction of the GT action step norm. "
         "Larger ⇒ the model relies on that input more. SKILL/PROGRESS exist only for skill models "
         "(pi05 → —).", "",
         "- `*_shuffle` — swap in another frame's value (in-distribution; **the fair metric**)",
         "- `state_gauss@σ` / `progress_gauss@σ` — add σ·std noise (skill is discrete → shuffle only)",
         "- `image_zero` — blank the images (vision-dominance reference)", ""]

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
    L.append("| _gt_action_step_norm_ | "
             + " | ".join(f"{results[l]['gt_action_step_norm']:.4f}" for l in labels) + " |")
    L.append("")

    # TF_compare-style table vs GT: mse_true / mse_swap / swapΔ / %scale / win (per model × perturbation).
    L.append("## TF-style metrics vs GT (mse_true / mse_swap / swapΔ / %scale / win)")
    L.append("")
    L.append("`mse_true` = GT error with REAL inputs (per model; ↓ = better fit). `mse_swap` = GT error with the "
             "perturbed input. `swapΔ` = output L2 move (= the influence Δ above). `%scale` = swapΔ ÷ gt_step_norm. "
             "`win` = P(real input closer to GT than perturbed): **0.5 = ignored, →1.0 = decisive**.")
    L.append("")
    L.append("| model | perturbation | mse_true | mse_swap | swapΔ | %scale | win |")
    L.append("|---|---|---|---|---|---|---|")
    for l in labels:
        mt = results[l].get("chunk_mse_true")
        mt_s = f"{mt:.4f}" if mt is not None else "—"
        for n in perts:
            m = results[l]["perturbations"].get(n)
            if not m:
                continue
            sw, w = m.get("mse_swap"), m.get("win")
            L.append(f"| {l} | {n} | {mt_s} | "
                     + (f"{sw:.4f}" if sw is not None else "—") + " | "
                     + f"{m['chunk_delta']:.4f} | {m['rel_to_gt_norm'] * 100:.1f}% | "
                     + (f"{w:.3f}" if w is not None else "—") + " |")
    L.append("")

    # Reading: per input group, the fair *_shuffle metric per model (+ % of that model's image_zero).
    L.append("## Reading")
    L.append("")
    for gname, gpref in _GROUPS:
        sh = gpref + "shuffle"
        if not any(sh in results[l]["perturbations"] for l in labels):
            continue
        L.append(f"**{gname}** (`{sh}`):")
        for l in labels:
            t = results[l]["perturbations"]
            v = t.get(sh, {}).get("rel_to_gt_norm")
            iz = t.get("image_zero", {}).get("rel_to_gt_norm")
            if v is None:
                L.append(f"- {l}: —")
            else:
                w = t.get(sh, {}).get("win")
                line = f"- {l}: win = `{w:.3f}`, " if w is not None else f"- {l}: "
                line += f"Δ/gt = `{v:.3f}`"
                if iz:
                    line += f"  (= {v / iz * 100:.0f}% of image_zero `{iz:.3f}`)"
                L.append(line)
        L.append("")
    L.append("> An input whose Δ ≈ 0 **and** << that model's image_zero is effectively a DEAD input "
             "for the model.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, default=None,
                    help="run dir with input_probe.json (default: all of outputs/)")
    args = ap.parse_args()

    if args.output_dir:
        dirs = [args.output_dir]
    else:
        dirs = sorted(p.parent for p in OUTPUTS_DIR.glob("*/input_probe.json")) if OUTPUTS_DIR.is_dir() else []
    if not dirs:
        print("No input_probe.json found.")
        return
    for d in dirs:
        jp = d / "input_probe.json"
        if not jp.is_file():
            print(f"[skip] no input_probe.json in {d}")
            continue
        md = render(json.loads(jp.read_text()))
        (d / "summary.md").write_text(md)
        print(f"[report] {d / 'summary.md'}")


if __name__ == "__main__":
    main()
