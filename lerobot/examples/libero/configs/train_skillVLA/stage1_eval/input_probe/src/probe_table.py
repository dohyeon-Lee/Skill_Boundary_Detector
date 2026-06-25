#!/usr/bin/env python3
"""Render input-probe heatmap tables from summary.csv.

This is the reusable version of the per-output make_probe_table.py helper:
after report.py writes summary.csv, this script writes
probe_table_win.{html,png} and probe_table_delta.{html,png} into the same run
directory.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

_HERE = Path(__file__).resolve()
OUTPUTS_DIR = _HERE.parents[1] / "outputs"

WIN_ROWS = [
    ("state_shuffle", "state_shuffle", "inputs", False, True),
    ("skill_shuffle", "skill_shuffle", "inputs", False, True),
    ("image_shuffle@3rd", "image_shuffle@3rd", "image shuffle", False, True),
    ("image_shuffle@wrist", "image_shuffle@wrist", "image shuffle", False, True),
    ("image_shuffle", "image_shuffle", "image shuffle", False, True),
    ("mse_true", "mse_true(&darr;fit)", "fit", True, True),
]

DELTA_ROWS = [
    ("state_shuffle", "state_shuffle", "state / skill", False, True),
    ("skill_shuffle", "skill_shuffle", "state / skill", False, True),
    ("image_shuffle", "image_shuffle", "image shuffle", False, True),
    ("image_shuffle@3rd", "image_shuffle@3rd", "image shuffle", False, True),
    ("image_shuffle@wrist", "image_shuffle@wrist", "image shuffle", False, True),
]


def load(run_dir: Path):
    win, rel, mse, checkpoints = {}, {}, {}, {}
    models: list[str] = []
    with open(run_dir / "summary.csv", newline="") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if model not in models:
                models.append(model)
            perturbation = row["perturbation"]
            win.setdefault(model, {})[perturbation] = float(row["win"])
            rel.setdefault(model, {})[perturbation] = float(row["rel_to_gt"])
            mse[model] = float(row["mse_true"])
            checkpoints[model] = row.get("checkpoint") or ""
    return models, win, rel, mse, checkpoints


def value_for(metric: str, model: str, key: str, win: dict, rel: dict, mse: dict):
    if key == "mse_true":
        return mse.get(model)
    data = win if metric == "win" else rel
    return data.get(model, {}).get(key)


def shade(v, lo: float, hi: float, lower_better: bool) -> str:
    if v is None or hi == lo:
        return "#ffffff"
    t = (v - lo) / (hi - lo)
    if lower_better:
        t = 1 - t
    r = int(255 + t * (31 - 255))
    g = int(255 + t * (122 - 255))
    b = int(255 + t * (140 - 255))
    return f"rgb({r},{g},{b})"


def fmt(v) -> str:
    return "&mdash;" if v is None else f"{v:.3f}"


def best_values(values: list, lower_better: bool):
    present = [v for v in values if v is not None]
    if not present:
        return set(), 0.0, 1.0
    target = min(present) if lower_better else max(present)
    return {v for v in present if abs(v - target) < 1e-12}, min(present), max(present)


def build_rows(metric: str, row_defs: list[tuple], models: list[str], win: dict, rel: dict, mse: dict):
    rows = []
    for key, label, group, lower_better, mark_best in row_defs:
        values = [value_for(metric, model, key, win, rel, mse) for model in models]
        if all(v is None for v in values):
            continue
        best, lo, hi = best_values(values, lower_better)
        rows.append((key, label, group, lower_better, mark_best, values, best, lo, hi))
    return rows


def checkpoint_tag(checkpoints: dict[str, str]) -> str:
    values = [v for v in checkpoints.values() if v]
    if values and len(set(values)) == 1:
        return f" <span style=\"font-weight:400;font-size:14px;color:#888\">(ckpt {values[0]})</span>"
    return ""


def emit(run_dir: Path, metric: str, models: list[str], win: dict, rel: dict, mse: dict, checkpoints: dict):
    row_defs = WIN_ROWS if metric == "win" else DELTA_ROWS
    table = build_rows(metric, row_defs, models, win, rel, mse)

    if metric == "win":
        title = "Input-influence probe &mdash; win"
        caption = (
            "<b>win</b> = P(real input closer to GT than the perturbed one); "
            "0.5 = ignored, 1.0 = decisive. Last row is <b>mse_true</b> "
            "(lower is better)."
        )
        note = (
            "Read by row: compare how much each model relies on the same input. "
            "Models without a skill input leave skill cells blank."
        )
    else:
        title = "Input-influence probe &mdash; &Delta; &divide; gt_action_step_norm"
        caption = (
            "Relative influence is the predicted action-chunk move divided by the GT action step norm. "
            "Higher values mean the output moved more when that input was perturbed."
        )
        note = "Read by row: compare how much each model's output moves under the same perturbation."

    header = "".join(f"<th>{model}</th>" for model in models)
    body, prev_group = "", None
    for key, label, group, lower_better, mark_best, values, best, lo, hi in table:
        if group != prev_group:
            body += f'<tr class="grp"><td colspan="{len(models) + 1}">{group}</td></tr>'
            prev_group = group
        body += f'<tr><td class="rowlab">{label}</td>'
        for v in values:
            is_best = mark_best and v is not None and v in best
            body += (
                f'<td class="cell{" best" if is_best else ""}" '
                f'style="background:{shade(v, lo, hi, lower_better)}">{fmt(v)}</td>'
            )
        body += "</tr>"

    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
      body{{margin:0;background:#fff;font-family:"Helvetica Neue",Arial,sans-serif;color:#222}}
      .wrap{{display:inline-block;padding:22px 26px}}
      h2{{font-size:18px;margin:0 0 3px}}
      .cap{{font-size:12px;color:#777;margin:0 0 14px;font-style:italic;max-width:760px;line-height:1.5}}
      table{{border-collapse:separate;border-spacing:0;font-size:13px}}
      th{{font-size:12px;font-weight:700;padding:6px 14px;text-align:center;border-bottom:2px solid #cfd6dd;white-space:nowrap}}
      th:first-child{{text-align:left}}
      td.rowlab{{padding:5px 16px 5px 14px;white-space:nowrap;color:#333}}
      td.cell{{width:96px;text-align:center;padding:6px 0;border-bottom:1px solid #eef1f4;font-variant-numeric:tabular-nums}}
      td.best{{font-weight:800}}
      tr.grp td{{font-size:11px;font-weight:700;color:#5a6b78;letter-spacing:.04em;text-transform:uppercase;
                 padding:11px 12px 4px;background:#fff;border:none}}
      .key{{font-size:11.5px;color:#666;margin-top:12px;max-width:760px;line-height:1.5}}
      .key b{{color:#1f7a8c}}
    </style></head><body><div class="wrap">
      <h2>{title}{checkpoint_tag(checkpoints)}</h2>
      <p class="cap">{caption} Cell shade is row-normalized; <b>bold</b> = best in row.</p>
      <table><thead><tr><th>input</th>{header}</tr></thead><tbody>{body}</tbody></table>
      <div class="key">{note}</div>
    </div></body></html>"""

    out = run_dir / f"probe_table_{'delta' if metric == 'delta' else 'win'}.html"
    out.write_text(html)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(device_scale_factor=2)
        page.goto(out.resolve().as_uri())
        page.wait_for_timeout(120)
        page.query_selector(".wrap").screenshot(path=str(out.with_suffix(".png")))
        browser.close()
    print(f"[probe_table] {out}  (+ {out.with_suffix('.png').name})")


def render_dir(run_dir: Path) -> None:
    if not (run_dir / "summary.csv").is_file():
        print(f"[skip] no summary.csv in {run_dir}")
        return
    models, win, rel, mse, checkpoints = load(run_dir)
    emit(run_dir, "win", models, win, rel, mse, checkpoints)
    emit(run_dir, "delta", models, win, rel, mse, checkpoints)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, default=None,
                    help="run dir with summary.csv (default: all outputs/*/summary.csv)")
    args = ap.parse_args()

    if args.output_dir:
        dirs = [args.output_dir]
    else:
        dirs = sorted(p.parent for p in OUTPUTS_DIR.glob("*/summary.csv")) if OUTPUTS_DIR.is_dir() else []
    if not dirs:
        print("No summary.csv found.")
        return
    for d in dirs:
        render_dir(d)


if __name__ == "__main__":
    main()
