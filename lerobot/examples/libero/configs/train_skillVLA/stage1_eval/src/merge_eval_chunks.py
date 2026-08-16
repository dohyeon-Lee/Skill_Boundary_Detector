#!/usr/bin/env python3
"""Merge chunked skill-eval metrics and draw one success-rate chart.

The shared skill-conditioned evaluator writes one ``metrics/eval_info[_tA-B].json``
per Slurm array chunk, keyed by panel label. This tool merges every chunk found
so far and regenerates:

* ``metrics/eval_info_merged.json`` — per-label union of per-task metrics with a
  recomputed overall success rate.
* ``task_success_rates.png`` — grouped horizontal bars, one group per task and
  one bar per panel, so Stage-2 and its frozen prior sit next to each other.

Idempotent and safe to run from every chunk job; whichever job finishes last
leaves the complete chart behind.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Okabe-Ito subset validated for CVD separation on a light surface; identity is
# never color-alone (legend + per-bar value labels).
_PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
_INK = "#333333"
_MUTED_INK = "#666666"


def _chunk_files(out_dir: Path) -> list[Path]:
    files = sorted((out_dir / "metrics").glob("eval_info*.json"))
    files = [path for path in files if path.name != "eval_info_merged.json"]
    # Chunks written before eval JSONs moved under metrics/.
    files += sorted(out_dir.glob("eval_info*.json"))
    return files


def _task_success(task: dict) -> float | None:
    metrics = task.get("metrics", task)
    successes = metrics.get("successes")
    if isinstance(successes, list) and successes:
        return float(sum(bool(value) for value in successes)) / len(successes)
    rate = metrics.get("pc_success", metrics.get("success_rate"))
    if rate is None:
        return None
    rate = float(rate)
    return rate / 100.0 if rate > 1.0 else rate


def merge(out_dir: Path) -> tuple[dict, list[str]]:
    labels: list[str] = []
    merged: dict[str, dict] = {}
    for path in _chunk_files(out_dir):
        chunks = json.loads(path.read_text())
        if not isinstance(chunks, dict):
            continue
        for label, info in chunks.items():
            if not isinstance(info, dict):
                continue
            if label not in merged:
                merged[label] = {"per_task": {}}
                labels.append(label)
            for task in info.get("per_task", []):
                key = (str(task.get("task_group", "")), int(task.get("task_id", -1)))
                merged[label]["per_task"][key] = task

    result = {}
    for label in labels:
        tasks = [
            task
            for _, task in sorted(merged[label]["per_task"].items())
        ]
        rates = [rate for task in tasks if (rate := _task_success(task)) is not None]
        result[label] = {
            "overall": {
                "pc_success": 100.0 * sum(rates) / len(rates) if rates else 0.0,
                "n_tasks": len(tasks),
            },
            "per_task": tasks,
        }
    return result, labels


def _atomic_write(path: Path, write) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    temporary.replace(path)


def draw_chart(
    merged: dict,
    labels: list[str],
    chart_path: Path,
    *,
    expected_tasks: int,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # noqa: BLE001
        print(f"merge: chart skipped (matplotlib unavailable: {error})")
        return False
    if not labels:
        return False
    if len(labels) > len(_PALETTE):
        print(
            f"merge: chart skipped ({len(labels)} panels exceed the fixed "
            f"{len(_PALETTE)}-color palette; split the comparison instead)"
        )
        return False

    task_keys: list[tuple[str, int]] = sorted(
        {
            (str(task.get("task_group", "")), int(task.get("task_id", -1)))
            for info in merged.values()
            for task in info["per_task"]
        }
    )
    by_label = {
        label: {
            (str(task.get("task_group", "")), int(task.get("task_id", -1))): task
            for task in merged[label]["per_task"]
        }
        for label in labels
    }

    rows: list[tuple[str, tuple[str, int] | None]] = [("overall", None)]
    rows += [
        (f"{group}_{task_id}", (group, task_id)) for group, task_id in task_keys
    ]
    n_labels = len(labels)
    bar_height = 0.72 / n_labels
    figure_height = max(2.4, 0.34 * len(rows) * n_labels + 1.2)
    figure, axis = plt.subplots(figsize=(8.0, figure_height))

    for label_index, label in enumerate(labels):
        positions, values = [], []
        for row_index, (_, key) in enumerate(rows):
            if key is None:
                rate = merged[label]["overall"]["pc_success"] / 100.0
            else:
                task = by_label[label].get(key)
                if task is None:
                    continue
                rate = _task_success(task)
                if rate is None:
                    continue
            positions.append(
                row_index + (label_index - (n_labels - 1) / 2.0) * bar_height
            )
            values.append(rate)
        bars = axis.barh(
            positions,
            values,
            height=bar_height * 0.92,
            color=_PALETTE[label_index],
            label=label,
        )
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                min(value + 0.015, 1.02),
                bar.get_y() + bar.get_height() / 2.0,
                f"{100.0 * value:.0f}%",
                va="center",
                ha="left",
                fontsize=8,
                color=_INK,
            )

    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(
        [name for name, _ in rows],
        fontsize=9,
        color=_INK,
    )
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.12)
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=_MUTED_INK)
    axis.set_xlabel("success rate", fontsize=9, color=_INK)
    axis.xaxis.grid(True, linestyle=":", linewidth=0.6, color="#cccccc")
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    completeness = (
        f"{len(task_keys)}/{expected_tasks} tasks"
        if expected_tasks and len(task_keys) < expected_tasks
        else f"{len(task_keys)} tasks"
    )
    axis.set_title(f"Task success rates ({completeness})", fontsize=10, color=_INK)
    if len(labels) > 1:
        axis.legend(loc="lower right", fontsize=8, frameon=False)
    figure.tight_layout()
    _atomic_write(
        chart_path, lambda path: figure.savefig(path, dpi=150, format="png")
    )
    plt.close(figure)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--expected_tasks",
        type=int,
        default=0,
        help="Total task count across all chunks; 0 skips the partial annotation.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated panel order; defaults to MODELS_JSON order.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir

    merged, found_labels = merge(out_dir)
    if not merged:
        print(f"merge: no eval_info chunks under {out_dir}; nothing to do.")
        return

    ordered = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not ordered:
        try:
            ordered = [
                spec["label"]
                for spec in json.loads(os.environ.get("MODELS_JSON", "") or "[]")
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            ordered = []
    labels = [label for label in ordered if label in merged]
    labels += [label for label in found_labels if label not in labels]

    merged_path = out_dir / "metrics" / "eval_info_merged.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        merged_path,
        lambda path: path.write_text(json.dumps(merged, indent=2)),
    )
    for label in labels:
        overall = merged[label]["overall"]
        print(
            f"merge: {label}: pc_success={overall['pc_success']:.1f} "
            f"over {overall['n_tasks']} task(s)"
        )
    if draw_chart(
        merged,
        labels,
        out_dir / "task_success_rates.png",
        expected_tasks=args.expected_tasks,
    ):
        print(f"merge: wrote {out_dir / 'task_success_rates.png'}")
    print(f"merge: wrote {merged_path}")


if __name__ == "__main__":
    main()
