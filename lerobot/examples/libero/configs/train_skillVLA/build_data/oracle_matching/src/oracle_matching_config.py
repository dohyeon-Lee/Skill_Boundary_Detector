#!/usr/bin/env python3
"""Resolve oracle_matching YAML into validated shell settings."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
ORACLE_MATCHING_DIR = SRC_DIR.parent
CONFIGS_ROOT = ORACLE_MATCHING_DIR.parents[2]
sys.path.insert(0, str(CONFIGS_ROOT / "train_skills" / "src"))

from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = ORACLE_MATCHING_DIR / "oracle_matching_config.yaml"


def _path(project_root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else project_root / path


def _dataset_root(project_root: Path, raw: Any, source_dataset: str) -> Path:
    value = "" if raw is None else str(raw).strip()
    if value and value.lower() != "auto":
        root = _path(project_root, value)
        if not (root / source_dataset).is_dir():
            raise FileNotFoundError(
                f"Source dataset not found: {root / source_dataset}. "
                "Set dataset_root to the root that actually contains it, or use auto."
            )
        return root

    conventional = [project_root / "dataset", project_root / "dataset_filtered"]
    discovered = sorted(
        path for path in project_root.glob("dataset*") if path.is_dir()
    )
    candidates: list[Path] = []
    for root in [*conventional, *discovered]:
        if root not in candidates and (root / source_dataset).is_dir():
            candidates.append(root)
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates) or "none"
        raise ValueError(
            f"dataset_root=auto requires exactly one root containing {source_dataset!r}; "
            f"found {rendered}. Set dataset_root explicitly."
        )
    return candidates[0]


def _task_ids(raw: Any) -> str:
    if raw is None or str(raw).strip().lower() == "all":
        return ""
    values = as_list(raw)
    try:
        ids = sorted({int(value) for value in values})
    except ValueError as exc:
        raise ValueError(f"task_ids must be all or integer IDs, got {raw!r}.") from exc
    if any(value < 0 for value in ids):
        raise ValueError(f"task_ids must be non-negative, got {ids}.")
    return ",".join(str(value) for value in ids)


def _optional_positive_int(value: Any, field: str) -> str:
    if value in (None, "", "null", "none"):
        return ""
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field} must be positive or null, got {resolved}.")
    return str(resolved)


def _slurm(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = get_value(cfg, "slurm", {})
    if not isinstance(raw, dict):
        raise ValueError(
            "slurm must be an inline mapping: "
            '{gres: "gpu:1", cpus: 8, memory: 64G, time: "12:00:00"}.'
        )
    gres = str(raw.get("gres", "gpu:1")).strip()
    cpus = int(raw.get("cpus", 8))
    memory = str(raw.get("memory", "64G")).strip()
    time = str(raw.get("time", "12:00:00")).strip()
    if not gres or cpus <= 0 or not memory or not time:
        raise ValueError("slurm gres/memory/time must be non-empty and cpus must be positive.")
    return {
        "oracle_partition": ",".join(as_list(get_value(cfg, "train_partition", []))),
        "oracle_qos": str(get_value(cfg, "train_qos", "")).strip(),
        "oracle_nodelist": str(get_value(cfg, "train_nodelist", "")).strip(),
        "oracle_exclude_nodes": ",".join(
            value for value in as_list(get_value(cfg, "train_exclude_nodes", [])) if value
        ),
        "oracle_gres": gres,
        "oracle_cpus_per_task": cpus,
        "oracle_mem": memory,
        "oracle_time": time,
    }


def build_settings(cfg: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    project_root = _path(
        ORACLE_MATCHING_DIR.parents[6],
        get_value(
            cfg,
            "project_root",
            str(ORACLE_MATCHING_DIR.parents[6]),
        ),
    ).resolve()
    source_dataset = str(
        get_value(cfg, "source_dataset", "", env="ORACLE_SOURCE_DATASET")
    ).strip()
    if not source_dataset or source_dataset in {".", ".."} or "/" in source_dataset:
        raise ValueError(
            f"source_dataset must be one dataset folder name, got {source_dataset!r}."
        )

    dataset_root = _dataset_root(
        project_root, get_value(cfg, "dataset_root", "auto"), source_dataset
    )
    output_root_raw = get_value(cfg, "output_dataset_root", "")
    output_root = (
        dataset_root
        if output_root_raw in (None, "")
        else _path(project_root, output_root_raw).resolve()
    )
    output_filename = str(get_value(cfg, "output_filename", "eval_init_states.npz")).strip()
    if Path(output_filename).name != output_filename or not output_filename.endswith(".npz"):
        raise ValueError(f"output_filename must be one .npz filename, got {output_filename!r}.")

    mode = str(get_value(cfg, "mode", "auto")).strip().lower()
    if mode == "auto":
        mode = "langgap" if source_dataset.startswith("langgap_") else "libero"
    if mode not in {"langgap", "libero"}:
        raise ValueError(f"mode must be auto|langgap|libero, got {mode!r}.")

    source_path = (dataset_root / source_dataset).resolve()
    output_path = (
        output_root / "skillvla_dataset" / source_dataset / output_filename
    ).resolve()

    original_raw = str(get_value(cfg, "original_dataset", "")).strip()
    original_dataset = ""
    if mode == "libero":
        if original_raw:
            original_dataset = str(_path(project_root, original_raw).resolve())
        else:
            match = re.match(
                r"^(libero_(?:90|10|goal|object|spatial))(?:_|$)", source_dataset
            )
            if match is None:
                raise ValueError(
                    "Cannot derive the original LIBERO suite from source_dataset "
                    f"{source_dataset!r}; set original_dataset explicitly."
                )
            original_dataset = str(
                (project_root / "libero_original_dataset" / match.group(1)).resolve()
            )
        if not Path(original_dataset).is_dir():
            raise FileNotFoundError(f"Original LIBERO dataset not found: {original_dataset}")

    signature_size = int(get_value(cfg, "signature_size", 64))
    num_steps_wait = int(get_value(cfg, "num_steps_wait", 10))
    if signature_size <= 0 or num_steps_wait < 0:
        raise ValueError("signature_size must be positive and num_steps_wait non-negative.")
    cache_raw = str(get_value(cfg, "cache_dir", "")).strip()
    cache_dir = str(_path(project_root, cache_raw).resolve()) if cache_raw else ""

    scores = {
        "state_weight": float(get_value(cfg, "state_weight", 1.0)),
        "image_weight": float(get_value(cfg, "image_weight", 4.0)),
        "wrist_weight": float(get_value(cfg, "wrist_weight", 0.0)),
        "max_state_score": float(get_value(cfg, "max_state_score", 1.0)),
        "max_image_mae": float(get_value(cfg, "max_image_mae", 0.18)),
        "min_score_margin": float(get_value(cfg, "min_score_margin", 1e-6)),
    }
    if any(value < 0 for value in scores.values()):
        raise ValueError(f"Matching weights/thresholds must be non-negative, got {scores}.")
    if not any(scores[key] > 0 for key in ("state_weight", "image_weight", "wrist_weight")):
        raise ValueError("At least one of state_weight/image_weight/wrist_weight must be positive.")

    settings: dict[str, Any] = {
        "project_root": project_root,
        "oracle_matching_dir": ORACLE_MATCHING_DIR,
        "oracle_matching_config": config_path.resolve(),
        "oracle_mode": mode,
        "oracle_source_dataset": source_dataset,
        "oracle_dataset_root": dataset_root.resolve(),
        "oracle_lerobot_dataset": source_path,
        "oracle_output_dataset_root": output_root.resolve(),
        "oracle_output_path": output_path,
        "oracle_original_dataset": original_dataset,
        "oracle_overwrite": as_bool(get_value(cfg, "overwrite", False)),
        "oracle_task_ids": _task_ids(get_value(cfg, "task_ids", "all")),
        "oracle_max_episodes": _optional_positive_int(
            get_value(cfg, "max_episodes", None), "max_episodes"
        ),
        "oracle_signature_size": signature_size,
        "oracle_num_steps_wait": num_steps_wait,
        "oracle_cache_dir": cache_dir,
        "oracle_state_weight": scores["state_weight"],
        "oracle_image_weight": scores["image_weight"],
        "oracle_wrist_weight": scores["wrist_weight"],
        "oracle_max_state_score": scores["max_state_score"],
        "oracle_max_image_mae": scores["max_image_mae"],
        "oracle_min_score_margin": scores["min_score_margin"],
        "oracle_accept_ambiguous": as_bool(get_value(cfg, "accept_ambiguous", False)),
    }
    settings.update(_slurm(cfg))
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config), config_path=args.config)
    if args.shell:
        print_shell(settings)
        return
    for key, value in settings.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
