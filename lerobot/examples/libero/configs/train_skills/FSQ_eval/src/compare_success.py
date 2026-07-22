#!/usr/bin/env python3
"""Generate one grouped task-success chart after all FSQ scale panels complete."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def _task_rates(info: dict) -> dict[int, float]:
    rates = {}
    for task in info.get("per_task", []):
        values = task.get("metrics", {}).get("successes", [])
        if values:
            rates[int(task.get("task_id", 0))] = float(np.mean([float(bool(v)) for v in values]))
    return rates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels_json", required=True)
    args = parser.parse_args()
    labels = json.loads(args.labels_json)

    panel_rates: list[tuple[str, dict[int, float]]] = []
    for label in labels:
        info_path = args.panels_root / label / "eval_info.json"
        if not info_path.is_file():
            print(f"compare: waiting for {info_path}")
            return
        panel_rates.append((label, _task_rates(json.loads(info_path.read_text()))))
    tasks = sorted(set.intersection(*(set(rates) for _, rates in panel_rates)))
    if not tasks:
        print("compare: no common completed tasks")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = min(0.8 / max(len(panel_rates), 1), 0.24)
    x = np.arange(len(tasks), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(max(9.0, 0.72 * len(tasks)), 5.0))
    center = (len(panel_rates) - 1) / 2
    for index, (label, rates) in enumerate(panel_rates):
        values = [rates[task] for task in tasks]
        ax.bar(x + (index - center) * width, values, width=width, label=label)
    ax.set_xticks(x, [f"task{task:02d}" for task in tasks], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("success rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.out.parent, suffix=".png", delete=False) as handle:
        tmp = Path(handle.name)
    fig.savefig(tmp, dpi=170)
    plt.close(fig)
    os.replace(tmp, args.out)
    print(f"compare: wrote {args.out} ({len(panel_rates)} panels, {len(tasks)} tasks)")


if __name__ == "__main__":
    main()
