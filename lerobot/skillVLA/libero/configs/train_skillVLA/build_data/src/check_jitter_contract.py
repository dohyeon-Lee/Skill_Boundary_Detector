#!/usr/bin/env python3
"""Validate the hidden SkillVLA transition-jitter metadata contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return normalized == "true"


def contract_matches(
    dataset_dir: Path,
    *,
    storage_pmax: int,
    max_percent: float | None = None,
    gripper_validity: bool | None = None,
    distribution: str,
    early_start_pmax: int | None = None,
    late_start_pmax: int | None = None,
    early_end_pmax: int | None = None,
    late_end_pmax: int | None = None,
) -> tuple[bool, str]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        return False, f"missing {info_path}"
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid {info_path}: {error}"

    stored_scalar = int(info.get("skill_pmax", -1))
    if max_percent is None:
        legacy = {
            "skill_jitter_early_start_pmax": early_start_pmax,
            "skill_jitter_late_start_pmax": late_start_pmax,
            "skill_jitter_early_end_pmax": early_end_pmax,
            "skill_jitter_late_end_pmax": late_end_pmax,
        }
        expected = {
            "skill_pmax": int(storage_pmax),
            **{key: int(value) for key, value in legacy.items()},
            "skill_jitter_distribution": str(distribution),
        }
        actual = {
            "skill_pmax": stored_scalar,
            **{
                key: int(info.get(key, stored_scalar))
                for key in legacy
            },
            "skill_jitter_distribution": str(
                info.get("skill_jitter_distribution", "half_normal")
            ).replace("-", "_"),
        }
    else:
        expected = {
            "skill_pmax": int(storage_pmax),
            "skill_jitter_max_percent": float(max_percent),
            "skill_jitter_gripper_validity": bool(gripper_validity),
            "skill_jitter_distribution": str(distribution),
        }
        actual = {
            "skill_pmax": stored_scalar,
            "skill_jitter_max_percent": float(
                info.get("skill_jitter_max_percent", 0.0)
            ),
            "skill_jitter_gripper_validity": bool(
                info.get("skill_jitter_gripper_validity", False)
            ),
            "skill_jitter_distribution": str(
                info.get("skill_jitter_distribution", "half_normal")
            ).replace("-", "_"),
        }
    mismatches = [
        f"{key}: dataset={actual[key]!r}, requested={value!r}"
        for key, value in expected.items()
        if actual[key] != value
    ]
    return not mismatches, "; ".join(mismatches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--storage-pmax", type=int, required=True)
    parser.add_argument("--max-percent", type=float, required=True)
    parser.add_argument("--gripper-validity", type=_bool, required=True)
    parser.add_argument("--distribution", required=True)
    args = parser.parse_args()
    matched, detail = contract_matches(
        args.dataset_dir,
        storage_pmax=args.storage_pmax,
        max_percent=args.max_percent,
        gripper_validity=args.gripper_validity,
        distribution=args.distribution,
    )
    if not matched:
        print(f"SkillVLA jitter contract mismatch: {detail}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
