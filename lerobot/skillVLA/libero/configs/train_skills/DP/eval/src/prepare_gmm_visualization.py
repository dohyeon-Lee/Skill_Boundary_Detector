#!/usr/bin/env python3
"""Reduce a normal SBD evaluation recipe to one cached GMM episode per task."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def prepare(
    settings: dict,
    output_name: str,
    probe_name: str = "iid96",
    task_ids: list[int] | None = None,
    n_episodes: int = 1,
    cache_source_dir: str = "",
) -> dict:
    if int(n_episodes) < 1:
        raise ValueError("n_episodes must be positive")
    result = dict(settings)
    probes = [dict(item) for item in result["probe_variants"] if item["name"] == probe_name]
    if len(probes) != 1:
        raise ValueError(f"Expected exactly one probe named {probe_name!r}")
    probes[0]["cache_input_descriptors"] = True

    experiments = [
        dict(item)
        for item in result["experiments"]
        if item.get("probe") == probe_name and not item.get("probes")
    ]
    if not experiments:
        raise ValueError(f"No single-probe experiment uses {probe_name!r}")
    experiment = experiments[0]
    clusters = [
        dict(item)
        for item in result["cluster_variants"]
        if item["name"] == experiment["cluster"]
    ]
    boundaries = [
        dict(item)
        for item in result["boundary_variants"]
        if item["name"] == experiment["boundary"]
    ]
    if len(clusters) != 1 or len(boundaries) != 1:
        raise ValueError("The selected experiment has an unresolved cluster or boundary")

    output_parent = Path(result["output_dir"]).parent
    result.update(
        output_dir=str(output_parent / output_name),
        n_episodes=int(n_episodes),
        threshold_calibration={"source": "selected", "expected_episodes": None},
        probe_variants=probes,
        cluster_variants=clusters,
        boundary_variants=boundaries,
        experiments=[experiment],
        split_report_by_step=False,
        split_report_by_group=False,
        split_report_by_task=False,
        overlay_experiment_curves=False,
        report_consistency_metrics=False,
        rollout={"enabled": False, "x0_temporal_alignment": False},
        prediction_loss={"enabled": False, "common_noise": True},
    )
    if task_ids is not None:
        result["task_ids"] = [int(value) for value in task_ids]
    if str(cache_source_dir).strip():
        result["cache_source_dir"] = str(Path(cache_source_dir).expanduser().resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings-json", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--probe", default="iid96")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--cache-source-dir", default="")
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = prepare(
        json.loads(args.settings_json),
        args.output_name,
        args.probe,
        args.task_ids,
        args.n_episodes,
        args.cache_source_dir,
    )
    encoded = json.dumps(settings, separators=(",", ":"))
    if args.shell:
        print(f"SBD_GMM_SETTINGS_JSON={shlex.quote(encoded)}")
        print(f"SBD_GMM_OUTPUT_DIR={shlex.quote(settings['output_dir'])}")
    else:
        print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
