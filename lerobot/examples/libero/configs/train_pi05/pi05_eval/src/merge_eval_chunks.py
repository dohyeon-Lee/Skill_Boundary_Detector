#!/usr/bin/env python3
"""Merge a pi05 evaluation with the Stage-1 evaluator's merge, then log one W&B run.

Workers write ``<out>/metrics/eval_info_<tag>.json`` keyed by panel label (the Stage-1 chunk
format), so this reuses stage1_eval/src/merge_eval_chunks.py verbatim:

* ``metrics/eval_info_merged.json``  per-panel union of per-task metrics + overall success rate
* ``task_success_rates.png``          grouped bars incl. an "overall" row (<= 6 panels)
* ``checkpoint_success_rates.png``    success vs checkpoint when panels form a checkpoint sweep

Idempotent; every Slurm element runs it and the last one leaves the complete result. Once all
tasks are in, a single W&B run gets the charts and each panel's overall success rate.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_STAGE1_MERGE = (
    Path(__file__).resolve().parents[3] / "train_skillVLA" / "stage1_eval" / "src" / "merge_eval_chunks.py"
)


def _stage1_merge():
    spec = importlib.util.spec_from_file_location("stage1_merge_eval_chunks", _STAGE1_MERGE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _log_wandb_once(out_dir: Path, merged: dict, labels: list[str], project: str, run_name: str) -> None:
    sentinel = out_dir / "metrics" / ".wandb_logged"
    try:
        sentinel.touch(exist_ok=False)  # O_EXCL: exactly one racing element logs
    except FileExistsError:
        print("merge: W&B run already logged; skipping")
        return
    try:
        import wandb  # noqa: PLC0415

        wandb.init(project=project, name=run_name, config={"panels": labels})
        payload = {f"overall/{label}/pc_success": merged[label]["overall"]["pc_success"] for label in labels}
        for name in ("task_success_rates", "checkpoint_success_rates"):
            chart = out_dir / f"{name}.png"
            if chart.is_file():
                payload[f"charts/{name}"] = wandb.Image(str(chart))
        wandb.log(payload)
        wandb.finish()
        print(f"merge: W&B run {run_name!r} logged")
    except Exception as error:  # noqa: BLE001 - never fail an evaluation over logging
        sentinel.unlink(missing_ok=True)
        print(f"merge: W&B logging failed: {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--expected_tasks", type=int, default=0)
    parser.add_argument("--labels", default="", help="Comma-separated panel order (default: MODELS_JSON)")
    parser.add_argument("--wandb_project", default="", help="'' skips W&B")
    parser.add_argument("--job_name", default="")
    args = parser.parse_args()

    merged, labels = _stage1_merge().run_merge(
        args.out_dir,
        expected_tasks=args.expected_tasks,
        labels=[label.strip() for label in args.labels.split(",") if label.strip()],
    )
    complete = bool(labels) and all(
        merged[label]["overall"]["n_tasks"] >= args.expected_tasks for label in labels
    )
    if args.wandb_project and args.expected_tasks > 0 and complete:
        _log_wandb_once(args.out_dir, merged, labels, args.wandb_project, args.job_name or args.out_dir.name)


if __name__ == "__main__":
    main()
