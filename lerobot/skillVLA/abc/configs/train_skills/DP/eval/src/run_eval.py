#!/usr/bin/env python3
"""Run a standalone ABC SBD subset, or recut its cached curves on CPU."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import build_settings
from dataset_source import prepare_source
from run_parallel_ablation import configure_cpu_threads, load_evaluator
from cache_state import cache_status
from video_report import prepare_report_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    resolved = build_settings(str(args.config))
    settings = json.loads(resolved["dp_ablation_settings_json"])
    configure_cpu_threads(settings)
    if "source_manifest" not in settings:
        raise ValueError("DP subset evaluation requires standalone: true")
    ready, reason = cache_status(settings, require_clusters=False)
    if args.report_only and not ready:
        raise ValueError(f"Report-only cache is unavailable: {reason}")
    if ready:
        # Hide CUDA even for direct Python execution inside a GPU allocation.
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        settings["device"] = "cpu"
    prepare_source(settings)
    prepare_report_layout(settings)
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Selected episodes: {resolved['dp_ablation_episode_count']}", flush=True)
    print(f"Partial build: {settings['artifact_dir']}", flush=True)
    print(f"Report: {Path(settings.get('report_root', output)) / 'index.html'}", flush=True)
    if args.prepare_only:
        return
    if args.report_only or ready:
        print("Reusing DP/PCA caches; GMM and report generation run on CPU", flush=True)
        load_evaluator().render_cached_report(settings)
        return
    work = Path(settings.get("work_dir", Path(settings["artifact_dir"]) / "workers"))
    work.mkdir(parents=True, exist_ok=True)
    settings_file = work / "settings.json"
    settings_file.write_text(json.dumps(settings, indent=2) + "\n")
    # Replace this launcher so Slurm signals reach the worker-pool supervisor.
    import os
    os.execv(sys.executable, [sys.executable, "-u", str(Path(__file__).with_name("run_parallel_ablation.py")),
                             "--settings-file", str(settings_file), "--work-dir", str(work)])


if __name__ == "__main__":
    main()
