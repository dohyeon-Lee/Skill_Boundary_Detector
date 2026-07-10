#!/usr/bin/env python3
"""Merge task-split eval chunks (submit_eval.sh eval_num_gpus>1) into ONE summary + chart (pi05 eval).

Each chunk job writes eval_info_t{a}-{b}.json into the SHARED out dir (lerobot_eval suffixes by the
TASK_TAG env when task-split); this script (run at the END of every chunk job — idempotent, last
finisher wins with the complete set) merges them into:
  * eval_info.json          — combined per_task list + recomputed overall pc_success
  * task_success_rates.png  — regenerated over ALL merged tasks (same look as the single-job chart)
Earlier finishers produce partial merges that later finishers overwrite (tmp+mv atomic). Safe to
re-run any time; no-op if no chunk files exist. (Same as stage2_eval's merge, minus index.html —
pi05's lerobot_eval doesn't emit one.)
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--job_name", default="", help="merged-wandb run name")
    ap.add_argument("--expected_tasks", type=int, default=0,
                    help="total task count across ALL chunks; when the merge reaches it, log ONE merged wandb run")
    ap.add_argument("--wandb_project", default="", help="wandb project for the merged run ('' = skip)")
    args = ap.parse_args()
    out = args.out_dir

    chunks = sorted(out.glob("eval_info_t*.json"))
    if not chunks:
        # Non-chunked (single job for this panel/model) → still (re)generate the summary + chart from
        # the plain eval_info.json so every model gets its own success-rate graph.
        single = out / "eval_info.json"
        if single.exists():
            chunks = [single]
        else:
            print("merge: no eval_info found — nothing to do")
            return

    # Merge per_task across chunks (dedupe on (task_group, task_id); newer file wins).
    per_task: dict[tuple, dict] = {}
    for cj in chunks:  # sorted → later files override on collision; collisions shouldn't happen
        try:
            info = json.loads(cj.read_text())
        except Exception as exc:  # noqa: BLE001 — a chunk still mid-write must not kill the others' merge
            print(f"merge: skipping unreadable {cj.name}: {exc}")
            continue
        for t in info.get("per_task", []):
            per_task[(str(t.get("task_group", "")), int(t.get("task_id", 0)))] = t
    tasks = [per_task[k] for k in sorted(per_task)]

    def _pc(t: dict):
        m = t.get("metrics", {})
        pc = m.get("pc_success")
        if pc is None:
            s = m.get("successes", [])
            pc = (sum(1 for x in s if x) / len(s) * 100) if s else None
        return pc

    pcs = [p for t in tasks if (p := _pc(t)) is not None]
    merged = {
        "overall": {"pc_success": (sum(pcs) / len(pcs)) if pcs else 0.0, "n_tasks": len(tasks),
                    "merged_from": [c.name for c in chunks]},
        "per_task": tasks,
    }
    # atomic write (concurrent chunk jobs may merge simultaneously — last complete merge wins)
    with tempfile.NamedTemporaryFile("w", dir=out, suffix=".json.tmp", delete=False) as f:
        json.dump(merged, f, indent=2)
        tmp = f.name
    os.replace(tmp, out / "eval_info.json")
    print(f"merge: eval_info.json ← {len(chunks)} chunks, {len(tasks)} tasks, "
          f"overall pc_success={merged['overall']['pc_success']:.1f}")

    # Regenerate the FULL task_success_rates.png over the merged set (chunk jobs write suffixed partials;
    # this canonical one always reflects everything merged so far — the last finisher leaves all tasks).
    chart_path = None
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        labels, rates = [], []
        for t in tasks:
            s = [float(bool(x)) for x in t.get("metrics", {}).get("successes", [])]
            labels.append(f"task{int(t.get('task_id', 0)):02d}")
            rates.append(float(np.mean(s)) if s else 0.0)
        if labels:
            height = max(2.8, 0.38 * len(labels) + 0.9)
            fig, ax = plt.subplots(figsize=(8.0, height))
            y = np.arange(len(labels))
            ax.barh(y, rates, color="#4C78A8")
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("success rate")
            ax.grid(axis="x", alpha=0.25)
            for yi, v in zip(y, rates):
                ax.text(min(v + 0.02, 0.98), yi, f"{v:.2f}", va="center", fontsize=8)
            fig.tight_layout()
            chart_path = out / "task_success_rates.png"
            fig.savefig(chart_path, dpi=160)
            plt.close(fig)
            print(f"merge: task_success_rates.png regenerated ({len(labels)} tasks)")
    except Exception as exc:  # noqa: BLE001
        print(f"merge: chart regeneration skipped: {exc}")

    # ONE merged wandb run once ALL chunks are in (sentinel = exactly-once across racing finishers).
    if args.wandb_project and args.expected_tasks > 0 and len(tasks) >= args.expected_tasks and chart_path:
        try:
            (out / ".merged_wandb_done").touch(exist_ok=False)      # O_EXCL: only the first complete merge logs
        except FileExistsError:
            print("merge: merged wandb run already logged — skipping")
            return
        try:
            import wandb  # noqa: PLC0415

            wandb.init(project=args.wandb_project, name=args.job_name or "merged_eval",
                       config={"merged_from": [c.name for c in chunks], "n_tasks": len(tasks)})
            wandb.log({"charts/task_success_rate": wandb.Image(str(chart_path)),
                       "overall/pc_success": merged["overall"]["pc_success"]})
            wandb.finish()
            print(f"merge: merged wandb run '{args.job_name}' logged ({len(tasks)} tasks)")
        except Exception as exc:  # noqa: BLE001
            print(f"merge: merged wandb logging failed: {exc}")


if __name__ == "__main__":
    main()
