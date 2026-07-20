#!/usr/bin/env python
"""Aggregate lerobot-eval LIBERO results by cycle-training GROUP.

Joins eval_info.json (per-task successes, env task ids) with the PT run's groups.json
(task language → group id) via the LIBERO benchmark's task-id → language mapping, so the
closed-loop success numbers line up with the probe/forget curves per group.

Tasks in the eval suite that were NOT in the PT dataset (filtered out) land in group -1
("not_in_pt") — for libero_90 that's the 90−73=17 excluded tasks.

Usage (inside the eval environment, LIBERO_CONFIG_PATH set):
  python aggregate_group_success.py --eval_info .../eval_info.json \
      --groups_json .../groups.json --suite libero_90 --out_dir .../outdir
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower().rstrip("."))


def task_names_from_benchmark(suite: str) -> dict[int, str]:
    """env task_id → language instruction, via the LIBERO benchmark."""
    from libero.libero import benchmark

    bm = benchmark.get_benchmark_dict()[suite]()
    return {i: bm.get_task(i).language for i in range(bm.get_num_tasks())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_info", type=Path, required=True)
    ap.add_argument("--groups_json", type=Path, required=True)
    ap.add_argument("--suite", default="libero_90")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    info = json.loads(args.eval_info.read_text())
    per_task = info.get("per_task", [])
    if not per_task:
        raise SystemExit("eval_info.json has no per_task entries")

    groups = json.loads(args.groups_json.read_text())
    group_of_task = {norm(t): g["group_id"] for g in groups for t in g["tasks"]}
    id2name = task_names_from_benchmark(args.suite)

    rows = []
    for t in per_task:
        tid = int(t.get("task_id", -1))
        name = id2name.get(tid, "")
        gid = group_of_task.get(norm(name), -1)  # -1 = not in PT dataset
        succs = t.get("metrics", {}).get("successes", [])
        rows.append({"task_id": tid, "task": name, "group": gid,
                     "n_episodes": len(succs), "success_rate": float(np.mean(succs)) if succs else 0.0})

    # per-group aggregation (episode-weighted)
    summary = {}
    for gid in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == gid]
        n_ep = sum(r["n_episodes"] for r in rs)
        rate = (sum(r["success_rate"] * r["n_episodes"] for r in rs) / n_ep) if n_ep else 0.0
        key = "not_in_pt" if gid == -1 else f"g{gid}"
        summary[key] = {"n_tasks": len(rs), "n_episodes": n_ep, "success_rate": round(rate, 4)}

    in_pt = [r for r in rows if r["group"] >= 0]
    overall = (sum(r["success_rate"] * r["n_episodes"] for r in in_pt)
               / max(1, sum(r["n_episodes"] for r in in_pt)))
    summary["overall_in_pt"] = {"n_tasks": len(in_pt),
                                "n_episodes": sum(r["n_episodes"] for r in in_pt),
                                "success_rate": round(overall, 4)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "group_success.json").write_text(
        json.dumps({"summary": summary, "per_task": rows}, indent=2))

    print(f"\n{'group':>10s} {'tasks':>6s} {'eps':>5s} {'success':>8s}")
    for key, v in summary.items():
        print(f"{key:>10s} {v['n_tasks']:>6d} {v['n_episodes']:>5d} {v['success_rate']:>8.3f}")

    # chart: per-group bars + per-task bars colored by group
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gkeys = [k for k in summary if k.startswith("g")]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3.0, 0.28 * len(rows))),
                             gridspec_kw={"width_ratios": [1, 2.2]})
    axes[0].barh(gkeys, [summary[k]["success_rate"] for k in gkeys], color="#4C78A8")
    axes[0].axvline(summary["overall_in_pt"]["success_rate"], color="k", ls="--", lw=1,
                    label=f"overall {summary['overall_in_pt']['success_rate']:.2f}")
    axes[0].set_xlim(0, 1); axes[0].invert_yaxis(); axes[0].legend(fontsize=8)
    axes[0].set_title("success by group"); axes[0].grid(axis="x", alpha=0.25)

    rows_sorted = sorted(rows, key=lambda r: (r["group"], -r["success_rate"]))
    cmap = plt.get_cmap("tab10")
    colors = ["#bbbbbb" if r["group"] < 0 else cmap(r["group"] % 10) for r in rows_sorted]
    labels = [f"[{'x' if r['group'] < 0 else 'g' + str(r['group'])}] t{r['task_id']:02d}" for r in rows_sorted]
    axes[1].barh(range(len(rows_sorted)), [r["success_rate"] for r in rows_sorted], color=colors)
    axes[1].set_yticks(range(len(rows_sorted))); axes[1].set_yticklabels(labels, fontsize=6)
    axes[1].invert_yaxis(); axes[1].set_xlim(0, 1)
    axes[1].set_title("success by task (grey = not in PT)"); axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "group_success.png", dpi=160)
    print(f"Saved -> {args.out_dir / 'group_success.json'} , group_success.png")


if __name__ == "__main__":
    main()
