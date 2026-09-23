#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402


def _named_list(cfg: dict, key: str) -> list[dict]:
    value = get_value(cfg, key, None)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list.")
    names: set[str] = set()
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{key}[{index}] must be a mapping.")
        name = str(item.get("name", "")).strip()
        if not name or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise ValueError(f"{key}[{index}].name is invalid: {name!r}")
        if name in names:
            raise ValueError(f"Duplicate {key} name: {name}")
        names.add(name)
        result.append(dict(item))
    return result


def build_settings(path: str) -> dict:
    cfg = load_config(path)
    root = Path(str(get_value(cfg, "project_root"))).expanduser().resolve()
    dataset_root = Path(str(get_value(cfg, "dataset_root", "dataset_filtered")))
    if not dataset_root.is_absolute():
        dataset_root = root / dataset_root
    components = {
        key: str(get_value(cfg, key, "")).strip()
        for key in ("target_dataset", "fsq_dataset_root", "fsq_inputs_name", "skillset_seg_name", "skillset_name")
    }
    for key, value in components.items():
        if not value or Path(value).name != value:
            raise ValueError(f"{key} must be a folder name, got {value!r}")
    skillset_dir = (
        dataset_root / components["fsq_dataset_root"] / components["target_dataset"]
        / components["fsq_inputs_name"] / components["skillset_seg_name"]
        / components["skillset_name"]
    )
    manifest = skillset_dir / "skillset_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Skillset manifest not found: {manifest}")
    output_name = str(get_value(cfg, "output_name", "boundary_ablation")).strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", output_name):
        raise ValueError(f"Invalid output_name: {output_name!r}")

    probes = _named_list(cfg, "probe_variants")
    clusters = _named_list(cfg, "cluster_variants")
    boundaries = _named_list(cfg, "boundary_variants")
    experiments = _named_list(cfg, "experiments")
    probe_names = {x["name"] for x in probes}
    cluster_names = {x["name"] for x in clusters}
    boundary_names = {x["name"] for x in boundaries}
    for item in experiments:
        if item.get("probe") not in probe_names:
            raise ValueError(f"Experiment {item['name']} selects unknown probe {item.get('probe')!r}")
        if item.get("cluster") not in cluster_names:
            raise ValueError(f"Experiment {item['name']} selects unknown cluster {item.get('cluster')!r}")
        if item.get("boundary") not in boundary_names:
            raise ValueError(f"Experiment {item['name']} selects unknown boundary {item.get('boundary')!r}")
        if item.get("metric") not in {
            "cosine",
            "l2",
            "hybrid",
            "delta_bic",
            "delta_bic_gain",
            "bic_multi_probability",
        }:
            raise ValueError(
                f"Experiment {item['name']} metric must be cosine|l2|hybrid|"
                "delta_bic|delta_bic_gain|bic_multi_probability"
            )

    rollout = dict(get_value(cfg, "rollout", {}) or {})
    rollout["enabled"] = as_bool(rollout.get("enabled", False))
    if rollout["enabled"]:
        if rollout.get("probe") not in probe_names:
            raise ValueError(f"rollout.probe selects unknown probe {rollout.get('probe')!r}")
        if rollout.get("cluster") not in cluster_names:
            raise ValueError(f"rollout.cluster selects unknown cluster {rollout.get('cluster')!r}")
        if rollout.get("boundary") not in boundary_names:
            raise ValueError(f"rollout.boundary selects unknown boundary {rollout.get('boundary')!r}")
        if rollout.get("metric") not in {
            "cosine",
            "l2",
            "hybrid",
            "delta_bic",
            "bic_multi_probability",
        }:
            raise ValueError(
                "rollout.metric must be cosine|l2|hybrid|delta_bic|bic_multi_probability"
            )
        if int(rollout.get("samples", 32)) < 2:
            raise ValueError("rollout.samples must be at least 2")
        if int(rollout.get("batch_size", 16)) < 1:
            raise ValueError("rollout.batch_size must be positive")
    settings = {
        "project_root": str(root),
        "skillset_dir": str(skillset_dir),
        "output_dir": str(
            Path(__file__).resolve().parent.parent / "outputs" / "dp_boundary_ablation"
            / components["target_dataset"] / output_name
        ),
        "task_id_space": str(get_value(cfg, "task_id_space", "suite")),
        "target_task": str(get_value(cfg, "target_task", "")),
        "task_ids": [int(x) for x in as_list(get_value(cfg, "task_ids", []))],
        "n_episodes": int(get_value(cfg, "n_episodes", 20)),
        "resume": as_bool(get_value(cfg, "resume", True)),
        "seed": int(get_value(cfg, "seed", 42)),
        "pca_variance": float(get_value(cfg, "pca_variance", 0.95)),
        "pca_stride": int(get_value(cfg, "pca_stride", 3)),
        "probe_variants": probes,
        "cluster_variants": clusters,
        "boundary_variants": boundaries,
        "experiments": experiments,
        "rollout": rollout,
    }
    if settings["task_id_space"] not in {"suite", "dataset"}:
        raise ValueError("task_id_space must be suite|dataset")
    if not settings["task_ids"]:
        raise ValueError("task_ids must select at least one task")
    if settings["n_episodes"] < 1:
        raise ValueError("n_episodes must be positive")
    return {
        "project_root": str(root),
        "dp_ablation_skillset_dir": str(skillset_dir),
        "dp_ablation_output_dir": settings["output_dir"],
        "dp_ablation_settings_json": json.dumps(settings, separators=(",", ":")),
        "dp_ablation_partition": ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug",
        "dp_ablation_qos": str(get_value(cfg, "train_qos", "base_qos")),
        "dp_ablation_gres": str(get_value(cfg, "eval_gres", "gpu:1")),
        "dp_ablation_cpus": int(get_value(cfg, "eval_cpus_per_task", 4)),
        "dp_ablation_mem": str(get_value(cfg, "eval_mem", "48G")),
        "dp_ablation_time": str(get_value(cfg, "eval_time", "04:00:00")),
        "dp_ablation_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "dp_ablation_exclude_nodes": ",".join(as_list(get_value(cfg, "train_exclude_nodes", []))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(args.config)
    if args.shell:
        print_shell(settings)
    else:
        print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
