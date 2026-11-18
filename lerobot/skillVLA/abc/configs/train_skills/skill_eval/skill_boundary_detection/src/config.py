#!/usr/bin/env python3
"""ABC wrapper around the maintained shared SBD evaluator configuration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def _shared_module():
    path = next(
        parent
        / "libero/configs/train_skills/skill_eval/skill_boundary_detection/src/config.py"
        for parent in Path(__file__).resolve().parents
        if (
            parent
            / "libero/configs/train_skills/skill_eval/skill_boundary_detection/src/config.py"
        ).is_file()
    )
    spec = importlib.util.spec_from_file_location("sbd_shared_boundary_eval_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared SBD evaluator config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_settings(config_path: str) -> dict:
    shared = _shared_module()
    settings = shared.build_settings(config_path, analysis_domain="abc")
    shard_tag = os.environ.get("ABC_SBD_SHARD_TAG", "").strip()
    task_ids_text = os.environ.get("ABC_SBD_TASK_IDS", "").strip()
    if bool(shard_tag) != bool(task_ids_text):
        raise ValueError(
            "ABC_SBD_SHARD_TAG and ABC_SBD_TASK_IDS must be set together."
        )
    if not shard_tag:
        return settings

    task_ids = [int(value.strip()) for value in task_ids_text.split(",") if value.strip()]
    if not task_ids:
        raise ValueError("ABC_SBD_TASK_IDS must select at least one task.")
    base_output_dir = Path(settings["dp_ablation_output_dir"])
    shard_output_dir = base_output_dir / "shards" / shard_tag
    payload = json.loads(settings["dp_ablation_settings_json"])
    payload["task_ids"] = task_ids
    payload["output_dir"] = str(shard_output_dir)
    payload["cache_source_dir"] = str(base_output_dir)
    payload["cache_source_dirs"] = [
        str(base_output_dir),
        *[
            value
            for value in payload.get("cache_source_dirs", [])
            if value != str(base_output_dir)
        ],
    ]
    calibration = dict(payload.get("threshold_calibration", {}))
    calibration["source"] = "selected"
    calibration["expected_episodes"] = len(task_ids) * int(payload["n_episodes"])
    payload["threshold_calibration"] = calibration

    settings["dp_ablation_output_dir"] = str(shard_output_dir)
    settings["dp_ablation_settings_json"] = json.dumps(
        payload, separators=(",", ":")
    )
    settings["dp_built_report_task_ids"] = ",".join(str(value) for value in task_ids)
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    shared = _shared_module()
    settings = build_settings(args.config)
    if args.shell:
        shared.print_shell(settings)
    else:
        print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
