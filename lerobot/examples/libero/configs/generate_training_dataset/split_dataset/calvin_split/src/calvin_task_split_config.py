#!/usr/bin/env python3
"""Validated settings for the CALVIN long-horizon three-way split."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SRC_DIR = Path(__file__).resolve().parent
CALVIN_SPLIT_DIR = SRC_DIR.parent
GENERATE_DATASET_DIR = CALVIN_SPLIT_DIR.parent.parent
CALVIN_DOWNLOAD_DIR = GENERATE_DATASET_DIR / "download_dataset" / "calvin_dataset"
CALVIN_DOWNLOAD_SRC = CALVIN_DOWNLOAD_DIR / "src"
DEFAULT_CONFIG_PATH = CALVIN_SPLIT_DIR / "calvin_task_split_config.yaml"
DEFAULT_CONVERSION_CONFIG = CALVIN_DOWNLOAD_DIR / "calvin_dataset_config.yaml"

if str(CALVIN_DOWNLOAD_SRC) not in sys.path:
    sys.path.insert(0, str(CALVIN_DOWNLOAD_SRC))

from calvin_dataset_config import conversion_settings, load_config as load_conversion_config  # noqa: E402


@dataclass(frozen=True)
class SelectedCandidate:
    candidate_key: str
    language: str


@dataclass(frozen=True)
class TaskSplitSettings:
    config_path: Path
    conversion_config_path: Path
    candidate_report_path: Path
    plan_dir: Path
    selected_candidates: tuple[SelectedCandidate, ...]
    plan_only: bool
    overwrite: bool
    conversion: dict[str, Any]


def _bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    raise TypeError(f"{key} must be true or false, got {value!r}")


def _split_path(value: Any, key: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{key} must not be blank")
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (CALVIN_SPLIT_DIR / path).resolve()


def _selected_candidates(value: Any) -> tuple[SelectedCandidate, ...]:
    rows = value if value is not None else []
    if not isinstance(rows, list):
        raise TypeError("selected_candidates must be a list of mappings")
    selected: list[SelectedCandidate] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"selected_candidates[{index}] must be a mapping")
        key = str(row.get("candidate_key", "")).strip()
        language = str(row.get("language", "")).strip()
        if not key:
            raise ValueError(f"selected_candidates[{index}].candidate_key must not be blank")
        if not language:
            raise ValueError(f"selected_candidates[{index}].language must not be blank")
        selected.append(SelectedCandidate(candidate_key=key, language=language))
    keys = [row.candidate_key for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("selected_candidates contains duplicate candidate_key values")
    return tuple(selected)


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> TaskSplitSettings:
    config_path = Path(path).expanduser().resolve()
    local = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(local, dict):
        raise TypeError("CALVIN task split config must be a mapping")

    override = os.environ.get("CALVIN_CONVERSION_CONFIG", "").strip()
    conversion_value = override or local.get("conversion_config", "")
    conversion_path = (
        _split_path(conversion_value, "conversion_config")
        if str(conversion_value).strip()
        else DEFAULT_CONVERSION_CONFIG.resolve()
    )
    conversion = conversion_settings(load_conversion_config(conversion_path))
    return TaskSplitSettings(
        config_path=config_path,
        conversion_config_path=conversion_path,
        candidate_report_path=_split_path(
            local.get(
                "candidate_report",
                "outputs/calvin_D_training_long_horizon/candidates.json",
            ),
            "candidate_report",
        ),
        plan_dir=_split_path(
            local.get("plan_dir", "plans/calvin_D_long_horizon"), "plan_dir"
        ),
        selected_candidates=_selected_candidates(local.get("selected_candidates", [])),
        plan_only=_bool(local.get("plan_only", False), "plan_only"),
        overwrite=_bool(local.get("overwrite", False), "overwrite"),
        conversion=conversion,
    )


def output_names(settings: TaskSplitSettings) -> dict[str, str]:
    variant = str(settings.conversion["calvin_convert_variant"])
    split = str(settings.conversion["calvin_convert_split"])
    prefix = f"calvin_{variant}"
    if split != "training":
        prefix += f"_{split}"
    return {
        "play_pretrain": f"{prefix}_play_pretrain_full_full",
        "language_pretrain": f"{prefix}_pretrain_full_full",
        "heldout": f"{prefix}_heldout_full_full",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    try:
        resolved = load_settings(args.config)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if not args.shell:
        print(resolved)
        return
    values = {
        "conversion_config": resolved.conversion_config_path,
        "source_dir": resolved.conversion["calvin_convert_source_dir"],
        "output_root": resolved.conversion["calvin_convert_output_root"],
        "partition": resolved.conversion["calvin_convert_partition"],
        "qos": resolved.conversion["calvin_convert_qos"],
        "gres": resolved.conversion["calvin_convert_gres"],
        "cpus_per_task": resolved.conversion["calvin_convert_cpus_per_task"],
        "mem": resolved.conversion["calvin_convert_mem"],
        "time": resolved.conversion["calvin_convert_time"],
        "nodelist": resolved.conversion["calvin_convert_nodelist"],
        "exclude_nodes": resolved.conversion["calvin_convert_exclude_nodes"],
    }
    for key, value in values.items():
        print(f"export CALVIN_SPLIT_{key.upper()}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
