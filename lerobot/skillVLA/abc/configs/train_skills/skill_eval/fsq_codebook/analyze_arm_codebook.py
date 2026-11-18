#!/usr/bin/env python3
"""Measure how an ABC bimanual FSQ codebook organizes left/right arm usage.

The encoded ``skill_latents_epochXXXX.npz`` owns the FSQ assignment.  The
matching SBD skill files own the untouched 14-D absolute YAM action sequences.
This script joins them exactly by (task, episode, skill index), computes motion
statistics from the six arm joints on each side, and writes a compact report.
Gripper dimensions 6 and 13 are reported separately and never contribute to the
left/right dominance label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np


LEFT_JOINTS = slice(0, 6)
LEFT_GRIPPER = 6
RIGHT_JOINTS = slice(7, 13)
RIGHT_GRIPPER = 13
LABELS = ("left", "bimanual", "right")


def _coord_from_token(token: int) -> tuple[int, int, int]:
    return tuple((int(token) // (3**axis)) % 3 - 1 for axis in range(3))


def _nmi(xs: list, ys: list) -> float:
    n = len(xs)
    x_counts, y_counts = Counter(xs), Counter(ys)
    joint = Counter(zip(xs, ys, strict=True))
    mi = sum(
        (count / n)
        * math.log(
            (count / n) / ((x_counts[x] / n) * (y_counts[y] / n))
        )
        for (x, y), count in joint.items()
    )
    hx = -sum((count / n) * math.log(count / n) for count in x_counts.values())
    hy = -sum((count / n) * math.log(count / n) for count in y_counts.values())
    return mi / math.sqrt(hx * hy) if hx > 0.0 and hy > 0.0 else 0.0


def _eta_squared(values: list[float], groups: list[int]) -> float:
    values_np = np.asarray(values, dtype=np.float64)
    groups_np = np.asarray(groups, dtype=np.int64)
    grand_mean = float(values_np.mean())
    total = float(np.square(values_np - grand_mean).sum())
    if total <= 0.0:
        return 0.0
    between = 0.0
    for group in np.unique(groups_np):
        selected = values_np[groups_np == group]
        between += len(selected) * (float(selected.mean()) - grand_mean) ** 2
    return between / total


def _load_assignments(latents_path: Path) -> dict[tuple[int, int, int], dict]:
    data = np.load(latents_path, allow_pickle=False)
    required = {
        "latents",
        "tokens",
        "task_id",
        "episode_id",
        "skill_index",
        "skill_order",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"{latents_path} is missing {missing}")
    if tuple(data["latents"].shape[1:]) != (3,):
        raise ValueError("This analysis expects a 3x3x3 FSQ codebook.")

    assignments = {}
    for index in range(len(data["tokens"])):
        key = (
            int(data["task_id"][index]),
            int(data["episode_id"][index]),
            int(data["skill_index"][index]),
        )
        token = int(data["tokens"][index])
        coord = tuple(int(value) for value in data["latents"][index])
        if coord != _coord_from_token(token):
            raise ValueError(
                f"Token/coordinate mismatch for {key}: token={token}, coord={coord}"
            )
        assignments[key] = {
            "token": token,
            "coord": coord,
            "skill_order": float(data["skill_order"][index]),
        }
    return assignments


def _dominance(left_path: float, right_path: float, ratio: float) -> str:
    if left_path >= ratio * right_path:
        return "left"
    if right_path >= ratio * left_path:
        return "right"
    return "bimanual"


def _load_metrics(
    skill_dir: Path,
    assignments: dict[tuple[int, int, int], dict],
    dominance_ratio: float,
) -> list[dict]:
    rows = []
    seen = set()
    for path in sorted(skill_dir.glob("task*/*.npz")):
        data = np.load(path, allow_pickle=False)
        key = (
            int(data["task_id"]),
            int(data["episode_id"]),
            int(data["skill_index"]),
        )
        assignment = assignments.get(key)
        if assignment is None:
            raise KeyError(f"No epoch assignment for {path}: {key}")
        seen.add(key)
        actions = np.asarray(data["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < 2:
            raise ValueError(f"Expected T x 14 actions with T >= 2: {path}")
        delta = np.diff(actions, axis=0)
        left_speed = np.linalg.norm(delta[:, LEFT_JOINTS], axis=1)
        right_speed = np.linalg.norm(delta[:, RIGHT_JOINTS], axis=1)
        left_path = float(left_speed.sum())
        right_path = float(right_speed.sum())
        total_path = left_path + right_path
        balance = (left_path - right_path) / max(total_path, 1e-12)
        coactivity = float(
            np.minimum(left_speed, right_speed).sum()
            / max(float(np.maximum(left_speed, right_speed).sum()), 1e-12)
        )
        if np.std(left_speed) > 1e-10 and np.std(right_speed) > 1e-10:
            speed_correlation = float(np.corrcoef(left_speed, right_speed)[0, 1])
        else:
            speed_correlation = 0.0
        rows.append(
            {
                **assignment,
                "task_id": key[0],
                "episode_id": key[1],
                "skill_index": key[2],
                "frames": int(len(actions)),
                "left_path": left_path,
                "right_path": right_path,
                "arm_balance": balance,
                "bilateral_balance": 1.0 - abs(balance),
                "coactivity": coactivity,
                "speed_correlation": speed_correlation,
                "motion_per_frame": total_path / len(delta),
                "left_displacement": float(
                    np.linalg.norm(actions[-1, LEFT_JOINTS] - actions[0, LEFT_JOINTS])
                ),
                "right_displacement": float(
                    np.linalg.norm(actions[-1, RIGHT_JOINTS] - actions[0, RIGHT_JOINTS])
                ),
                "left_gripper_start": float(actions[0, LEFT_GRIPPER]),
                "left_gripper_end": float(actions[-1, LEFT_GRIPPER]),
                "left_gripper_delta": float(
                    actions[-1, LEFT_GRIPPER] - actions[0, LEFT_GRIPPER]
                ),
                "right_gripper_start": float(actions[0, RIGHT_GRIPPER]),
                "right_gripper_end": float(actions[-1, RIGHT_GRIPPER]),
                "right_gripper_delta": float(
                    actions[-1, RIGHT_GRIPPER] - actions[0, RIGHT_GRIPPER]
                ),
                "arm_class": _dominance(left_path, right_path, dominance_ratio),
            }
        )
    missing = sorted(set(assignments).difference(seen))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} encoded skill files; first={missing[0]}")
    return rows


def _summarize(rows: list[dict], dominance_ratio: float) -> dict:
    class_counts = Counter(row["arm_class"] for row in rows)
    token_rows = []
    for token in range(27):
        selected = [row for row in rows if row["token"] == token]
        counts = Counter(row["arm_class"] for row in selected)
        majority, majority_count = counts.most_common(1)[0]
        token_rows.append(
            {
                "token": token,
                "coord": list(_coord_from_token(token)),
                "count": len(selected),
                "left": counts["left"],
                "bimanual": counts["bimanual"],
                "right": counts["right"],
                "majority": majority,
                "purity": majority_count / len(selected),
                "mean_arm_balance": float(
                    np.mean([row["arm_balance"] for row in selected])
                ),
                "mean_coactivity": float(
                    np.mean([row["coactivity"] for row in selected])
                ),
                "mean_frames": float(np.mean([row["frames"] for row in selected])),
            }
        )

    tokens = [row["token"] for row in rows]
    classes = [row["arm_class"] for row in rows]
    majority_by_token = {
        item["token"]: item["majority"] for item in token_rows
    }
    recall = {}
    for label in LABELS:
        selected = [row for row in rows if row["arm_class"] == label]
        recall[label] = sum(
            majority_by_token[row["token"]] == label for row in selected
        ) / len(selected)

    feature_names = (
        "arm_balance",
        "bilateral_balance",
        "coactivity",
        "speed_correlation",
        "motion_per_frame",
        "frames",
        "left_displacement",
        "right_displacement",
        "left_gripper_delta",
        "right_gripper_delta",
        "skill_order",
    )
    axis_rows = []
    axis_effects = []
    for axis in range(3):
        groups = [row["coord"][axis] for row in rows]
        effects = {
            feature: _eta_squared([row[feature] for row in rows], groups)
            for feature in feature_names
        }
        axis_effects.append(
            {
                "axis": axis + 1,
                "arm_class_nmi": _nmi(groups, classes),
                "eta_squared": effects,
            }
        )
        for level in (-1, 0, 1):
            selected = [row for row in rows if row["coord"][axis] == level]
            counts = Counter(row["arm_class"] for row in selected)
            axis_rows.append(
                {
                    "axis": axis + 1,
                    "level": level,
                    "count": len(selected),
                    **{label: counts[label] for label in LABELS},
                    **{
                        feature: float(np.mean([row[feature] for row in selected]))
                        for feature in feature_names
                    },
                    "left_gripper_start": float(
                        np.mean([row["left_gripper_start"] for row in selected])
                    ),
                    "left_gripper_end": float(
                        np.mean([row["left_gripper_end"] for row in selected])
                    ),
                    "right_gripper_start": float(
                        np.mean([row["right_gripper_start"] for row in selected])
                    ),
                    "right_gripper_end": float(
                        np.mean([row["right_gripper_end"] for row in selected])
                    ),
                }
            )

    probabilities = np.asarray([item["count"] for item in token_rows], dtype=float)
    probabilities /= probabilities.sum()
    effective_codes = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return {
        "dominance_ratio": dominance_ratio,
        "skill_count": len(rows),
        "class_counts": dict(class_counts),
        "token_class_nmi": _nmi(tokens, classes),
        "token_majority_accuracy": sum(
            majority_by_token[row["token"]] == row["arm_class"] for row in rows
        )
        / len(rows),
        "global_majority_baseline": max(class_counts.values()) / len(rows),
        "token_majority_macro_recall": float(np.mean(list(recall.values()))),
        "class_recall": recall,
        "used_codes": sum(item["count"] > 0 for item in token_rows),
        "effective_codes": effective_codes,
        "tokens": token_rows,
        "axis_levels": axis_rows,
        "axis_effects": axis_effects,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: int, total: int) -> str:
    return f"{100.0 * value / total:.0f}%"


def _render_html(summary: dict, epoch: str) -> str:
    tokens = {item["token"]: item for item in summary["tokens"]}
    class_names = {"left": "Left", "bimanual": "Both", "right": "Right"}
    slices = []
    for axis3 in (-1, 0, 1):
        cells = []
        for axis2 in (1, 0, -1):
            for axis1 in (-1, 0, 1):
                token = (axis1 + 1) + 3 * (axis2 + 1) + 9 * (axis3 + 1)
                row = tokens[token]
                total = row["count"]
                cells.append(
                    f"""
                    <div class="token {row['majority']}">
                      <div class="token-head"><b>code {token}</b><span>n={total}</span></div>
                      <div class="coord">({axis1:+d}, {axis2:+d}, {axis3:+d})</div>
                      <div class="bar" aria-label="Left {_pct(row['left'], total)}, both {_pct(row['bimanual'], total)}, right {_pct(row['right'], total)}">
                        <i class="left" style="width:{100*row['left']/total:.2f}%"></i><i class="bimanual" style="width:{100*row['bimanual']/total:.2f}%"></i><i class="right" style="width:{100*row['right']/total:.2f}%"></i>
                      </div>
                      <div class="mix"><span>L {_pct(row['left'], total)}</span><span>B {_pct(row['bimanual'], total)}</span><span>R {_pct(row['right'], total)}</span></div>
                    </div>"""
                )
        slices.append(
            f"<section><h2>axis 3 = {axis3:+d}</h2><div class='grid'>{''.join(cells)}</div></section>"
        )

    axes = []
    for axis in range(1, 4):
        levels = [row for row in summary["axis_levels"] if row["axis"] == axis]
        effect = next(row for row in summary["axis_effects"] if row["axis"] == axis)
        body = []
        for row in levels:
            n = row["count"]
            body.append(
                "<tr>"
                f"<td>{row['level']:+d}</td><td>{n}</td>"
                f"<td>{_pct(row['left'], n)} / {_pct(row['bimanual'], n)} / {_pct(row['right'], n)}</td>"
                f"<td>{row['arm_balance']:+.3f}</td><td>{row['coactivity']:.3f}</td>"
                f"<td>{row['frames']:.1f}</td><td>{row['skill_order']:.3f}</td>"
                f"<td>{row['left_gripper_start']:.3f}→{row['left_gripper_end']:.3f}</td>"
                f"<td>{row['right_gripper_start']:.3f}→{row['right_gripper_end']:.3f}</td>"
                "</tr>"
            )
        strongest = sorted(
            effect["eta_squared"].items(), key=lambda item: item[1], reverse=True
        )[:3]
        axes.append(
            f"<section><h2>Axis {axis}</h2>"
            f"<p>arm-class NMI {effect['arm_class_nmi']:.3f}; strongest marginal effects: "
            + ", ".join(f"{escape(name)} η²={value:.3f}" for name, value in strongest)
            + "</p><div class='table-wrap'><table><thead><tr><th>level</th><th>n</th><th>L / B / R</th><th>arm balance</th><th>coactivity</th><th>frames</th><th>skill order</th><th>left grip</th><th>right grip</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div></section>"
        )

    counts = summary["class_counts"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(epoch)} FSQ arm-use analysis</title>
<style>
:root{{--bg:#f5f7fb;--panel:#fff;--text:#18202b;--muted:#657080;--line:#dce2ea;--left:#3274d9;--both:#8b5fc7;--right:#e27635}}
@media(prefers-color-scheme:dark){{:root{{--bg:#11151b;--panel:#1a2029;--text:#edf1f7;--muted:#a9b3c1;--line:#343d49;--left:#65a1ff;--both:#b58ae6;--right:#ff9b5d}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}} main{{max-width:1500px;margin:auto;padding:24px}} h1{{margin:0 0 4px;font-size:24px}} h2{{font-size:17px;margin:0 0 10px}} p{{margin:4px 0 14px;color:var(--muted)}} .summary{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px;margin:18px 0 24px}} .stat,.token,section{{background:var(--panel);border:1px solid var(--line);border-radius:10px}} .stat{{padding:14px}} .stat b{{display:block;font-size:21px}} .stat span{{color:var(--muted)}} .legend{{display:flex;gap:18px;margin:12px 0}} .legend i{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}} .left{{background:var(--left)}} .bimanual{{background:var(--both)}} .right{{background:var(--right)}} .slices{{display:grid;grid-template-columns:repeat(3,minmax(300px,1fr));gap:14px}} section{{padding:14px;margin-bottom:14px}} .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}} .token{{padding:9px;min-width:0}} .token-head,.mix{{display:flex;justify-content:space-between;gap:4px}} .token-head span,.coord,.mix{{color:var(--muted);font-size:12px}} .bar{{display:flex;height:9px;overflow:hidden;border-radius:5px;background:var(--line);margin:8px 0 5px}} .bar i{{display:block;height:100%}} .table-wrap{{overflow-x:auto}} table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} .axis-block{{margin-top:22px}} @media(max-width:1000px){{.slices{{grid-template-columns:1fr}}}} @media(max-width:650px){{main{{padding:12px}}.summary{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>{escape(epoch)} · FSQ 3×3×3 arm-use analysis</h1>
<p>All {summary['skill_count']:,} encoded SBD skills. Dominant means one arm's six-joint path is at least {summary['dominance_ratio']:.1f}× the other; grippers are excluded from this label.</p>
<div class="summary">
 <div class="stat"><b>{summary['used_codes']} / 27</b><span>codes used · effective {summary['effective_codes']:.2f}</span></div>
 <div class="stat"><b>{summary['token_class_nmi']:.3f}</b><span>token ↔ arm-class NMI</span></div>
 <div class="stat"><b>{100*summary['token_majority_accuracy']:.1f}%</b><span>token-majority accuracy · baseline {100*summary['global_majority_baseline']:.1f}%</span></div>
 <div class="stat"><b>{100*summary['token_majority_macro_recall']:.1f}%</b><span>macro recall across L / B / R</span></div>
</div>
<div class="legend"><span><i class="left"></i>Left {counts['left']:,}</span><span><i class="bimanual"></i>Both {counts['bimanual']:,}</span><span><i class="right"></i>Right {counts['right']:,}</span></div>
<p>Columns are axis 1 (−1, 0, +1); rows are axis 2 (+1, 0, −1); the three panels slice axis 3.</p>
<div class="slices">{''.join(slices)}</div>
<div class="axis-block">{''.join(axes)}</div>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--skill-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch", default="epoch0800")
    parser.add_argument("--dominance-ratio", type=float, default=1.5)
    args = parser.parse_args()
    if args.dominance_ratio <= 1.0:
        parser.error("--dominance-ratio must be greater than 1")

    assignments = _load_assignments(args.latents)
    rows = _load_metrics(args.skill_dir, assignments, args.dominance_ratio)
    summary = _summarize(rows, args.dominance_ratio)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(args.output_dir / "tokens.csv", summary["tokens"])
    _write_csv(args.output_dir / "axis_levels.csv", summary["axis_levels"])
    (args.output_dir / "index.html").write_text(_render_html(summary, args.epoch))
    print(f"wrote {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
