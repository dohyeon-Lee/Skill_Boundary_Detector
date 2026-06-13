#!/usr/bin/env python3
"""Run offline teacher-forced eval for every target in tf_compare_config.yaml, caching each
model's skill-usage metrics to outputs/{name}/metrics.json (skipped if present; --force to redo).

Reuses stage1_eval/src/teacher_forced.py unchanged (one model per call). report.py then renders
the comparison table from the cached metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import tf_compare_config as C

# stage1_eval/src/teacher_forced.py — the per-model metric computer (shared with the single-model sweep).
_TF_SCRIPT = Path(__file__).resolve().parents[2] / "src" / "teacher_forced.py"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=C.DEFAULT_CONFIG_PATH)
    ap.add_argument("--force", action="store_true", help="recompute even if metrics.json exists")
    args = ap.parse_args()

    cfg = C.load(args.config)
    project_root = Path(str(C.get_value(cfg, "project_root"))).expanduser()
    lerobot_root = project_root / "lerobot"
    venv_py = project_root / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.exists() else sys.executable

    n_frames = int(C.get_value(cfg, "n_frames", 512))
    batch_size = int(C.get_value(cfg, "batch_size", 32))
    n_swap = int(C.get_value(cfg, "n_swap", 4))
    seed = int(C.get_value(cfg, "seed", 1000))

    env = dict(os.environ)
    env["PYTHONPATH"] = str(lerobot_root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("LD_LIBRARY_PATH", None)  # avoid system cuDNN shadowing the bundled one (matches the sbatch)

    for name, spec in C.iter_targets(cfg):
        model_dir = str(spec["model_dir"])
        checkpoint = str(spec.get("checkpoint", "last"))
        out_dir = C.OUTPUTS_DIR / name
        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"[skip] {name} (cached → {metrics_path})")
            continue

        policy_path, dataset_dir = C.resolve_paths(cfg, spec)
        if not policy_path.exists():
            print(f"[warn] {name}: policy checkpoint not found, skipping → {policy_path}")
            continue
        if not (dataset_dir / "meta").exists():
            print(f"[warn] {name}: skillvla dataset not found, skipping → {dataset_dir}")
            continue
        if not any((dataset_dir / "videos").rglob("*.mp4")):
            print(f"[warn] {name}: dataset has no video files (purged/incomplete?), skipping → "
                  f"{dataset_dir}/videos  (teacher_forced needs obs frames; re-transfer the videos)")
            continue

        tf_out = out_dir / "teacher_forced"
        cmd = [
            py, "-u", str(_TF_SCRIPT),
            "--policy_path", str(policy_path),
            "--dataset_dir", str(dataset_dir),
            "--output_dir", str(tf_out),
            "--n_frames", str(n_frames),
            "--batch_size", str(batch_size),
            "--n_swap", str(n_swap),
            "--seed", str(seed),
        ]
        print(f"[run ] {name}: {model_dir} @ {checkpoint}")
        subprocess.run(cmd, check=True, cwd=str(lerobot_root), env=env)

        summary = json.loads((tf_out / "teacher_forced.json").read_text())["summary"]
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(
            {"name": name, "model_dir": model_dir, "checkpoint": checkpoint, **summary}, indent=2))
        print(f"[done] {name}: win_rate={summary['true_code_win_rate']:.3f} → {metrics_path}")


if __name__ == "__main__":
    main()
