#!/usr/bin/env python3
"""Render summary.md (+ summary.csv) from an input_probe.json (state/skill/progress influence).

Tables are emitted as fixed-width monospace blocks (```fences```) so columns ALIGN regardless of the
viewer (plain markdown pipe tables shift with text length). A summary.csv is also written for
spreadsheet use.

Usage:
  report.py --output_dir <run dir with input_probe.json>   # writes summary.md + summary.csv there
  report.py                                                 # render every outputs/*/input_probe.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_HERE = Path(__file__).resolve()
OUTPUTS_DIR = _HERE.parents[1] / "outputs"

_GROUPS = [("state", "state_"), ("skill", "skill_"), ("progress", "progress_")]
# Focused summary rows (the intuitive *_shuffle metrics + per-camera image), in display order.
_FOCUS_ORDER = ["state_shuffle", "skill_shuffle", "progress_shuffle",
                "image_shuffle@3rd", "image_shuffle@wrist", "image_shuffle",
                "image_zero@3rd", "image_zero@wrist", "image_zero"]


def _order(n: str):
    """Sort key: state_ first, then skill_, progress_, then image_* last."""
    for i, (_, pref) in enumerate(_GROUPS):
        if n.startswith(pref):
            return (i, n)
    return (len(_GROUPS), n)


def _mono(headers: list[str], rows: list[list]) -> str:
    """Fixed-width monospace table in a code fence — first column left-aligned, the rest right-aligned."""
    if not rows:
        return "```\n(no data)\n```"
    cols = list(zip(*([headers] + rows)))
    w = [max(len(str(c)) for c in col) for col in cols]
    def line(r):
        return "  ".join((str(v).ljust(w[i]) if i == 0 else str(v).rjust(w[i])) for i, v in enumerate(r))
    body = [line(headers), "  ".join("-" * x for x in w)] + [line(r) for r in rows]
    return "```\n" + "\n".join(body) + "\n```"


def _cell(m: dict | None, key: str, scale: float = 1.0, pct: bool = False) -> str:
    if not m or m.get(key) is None:
        return "—"
    v = m[key] * scale
    return f"{v:.1f}%" if pct else f"{v:.3f}"


def _mark(cells: list[str], minimize: bool = False) -> list[str]:
    """Prefix the best cell(s) in a row with '*' (max by default, min when minimize=True). Non-numeric
    cells ('—') are skipped. _mono right-justifies, so the '*' sits in the leading pad → numbers stay aligned."""
    vals = []
    for c in cells:
        try:
            vals.append(float(c))
        except ValueError:
            vals.append(None)
    nums = [v for v in vals if v is not None]
    if not nums:
        return cells
    best = min(nums) if minimize else max(nums)
    return [f"*{c}" if (v is not None and v == best) else c for c, v in zip(cells, vals)]


def render(results: dict) -> str:
    labels = list(results.keys())
    cks = [results[l].get("checkpoint") for l in labels]
    uniform_ck = cks[0] if (cks and cks[0] and len(set(cks)) == 1) else None  # all same → omit from headers
    disp = {l: (l if (uniform_ck or not results[l].get("checkpoint")) else f"{l}({results[l]['checkpoint']})")
            for l in labels}
    perts = sorted({n for r in results.values() for n in r["perturbations"]}, key=_order)

    L = ["# Input-influence probe — state / skill / progress", ""]
    if uniform_ck:
        L += [f"_(all models @ checkpoint **{uniform_ck}**)_", ""]
    L += ["Perturb ONE input, measure how the predicted action chunk changes. **`win`** = P(real input "
         "closer to GT than the perturbed one): **0.5 = ignored, → 1.0 = decisive** (most intuitive). "
         "`Δ/gt` = output L2 move ÷ GT step. SKILL/PROGRESS exist only for skill models (pi05 → —). "
         "`image_shuffle*` swaps in another frame's image (in-distribution; the fair image metric); "
         "`image_zero*` blanks the camera (OOD reference). `@3rd/@wrist` = one camera, no suffix = both.", ""]

    # ── ⭐ Focused summary: the *_shuffle metrics + per-camera image, by win ──
    focus = [p for p in _FOCUS_ORDER if any(p in results[l]["perturbations"] for l in labels)]
    rows = [[p] + _mark([_cell(results[l]["perturbations"].get(p), "win") for l in labels]) for p in focus]
    mt = [(f"{results[l]['chunk_mse_true']:.3f}" if results[l].get("chunk_mse_true") is not None else "—") for l in labels]
    rows.append(["mse_true(↓fit)"] + _mark(mt, minimize=True))
    L += ["## ⭐ Summary — win (↑: 0.5=ignored → 1.0=decisive) per input · last row = mse_true (↓: GT fit) · "
          "`*` = best in row", "", _mono(["input", *[disp[l] for l in labels]], rows), ""]

    # ── Full influence (Δ ÷ gt) cross-tab ──
    rows = [[p] + _mark([_cell(results[l]["perturbations"].get(p), "rel_to_gt_norm") for l in labels]) for p in perts]
    rows.append(["gt_step_norm", *[f"{results[l]['gt_action_step_norm']:.4f}" for l in labels]])
    L += ["## Relative influence (Δ ÷ gt_action_step_norm) · `*` = highest in row", "",
          _mono(["perturbation", *[disp[l] for l in labels]], rows), ""]

    # ── TF-style metrics per (model × perturbation) ──
    tf = []
    for l in labels:
        mt = results[l].get("chunk_mse_true")
        mts = f"{mt:.4f}" if mt is not None else "—"
        for p in perts:
            m = results[l]["perturbations"].get(p)
            if not m:
                continue
            tf.append([disp[l], p, mts, _cell(m, "mse_swap"), _cell(m, "chunk_delta"),
                       _cell(m, "rel_to_gt_norm", 100, pct=True), _cell(m, "win")])
    L += ["## TF-style metrics vs GT  (mse_true / mse_swap / swapΔ / %scale / win)", "",
          _mono(["model", "perturbation", "mse_true", "mse_swap", "swapΔ", "%scale", "win"], tf), "",
          "> win ≈ 0.5 (and Δ ≈ 0) ⇒ the input is effectively DEAD for that model. Full numbers in summary.csv."]
    return "\n".join(L) + "\n"


def write_csv(results: dict, path: Path) -> None:
    labels = list(results.keys())
    perts = sorted({n for r in results.values() for n in r["perturbations"]}, key=_order)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "checkpoint", "perturbation", "mse_true", "mse_swap", "swapDelta",
                    "pct_scale", "win", "rel_to_gt"])
        for l in labels:
            mt = results[l].get("chunk_mse_true")
            ck = results[l].get("checkpoint")
            for p in perts:
                m = results[l]["perturbations"].get(p)
                if not m:
                    continue
                rg = m.get("rel_to_gt_norm")
                w.writerow([l, ck, p, mt, m.get("mse_swap"), m.get("chunk_delta"),
                            (rg * 100 if rg is not None else None), m.get("win"), rg])


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
        results = json.loads(jp.read_text())
        (d / "summary.md").write_text(render(results))
        write_csv(results, d / "summary.csv")
        print(f"[report] {d / 'summary.md'}  (+ summary.csv)")


if __name__ == "__main__":
    main()
