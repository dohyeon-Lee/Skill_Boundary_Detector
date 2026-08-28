#!/usr/bin/env python3
"""Validate the hidden SkillVLA transition-jitter metadata contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def contract_matches(
    dataset_dir: Path,
    *,
    storage_pmax: int,
    early_start_pmax: int,
    late_start_pmax: int,
    early_end_pmax: int,
    late_end_pmax: int,
    distribution: str,
) -> tuple[bool, str]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        return False, f"missing {info_path}"
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid {info_path}: {error}"

    stored_scalar = int(info.get("skill_pmax", -1))
    expected = {
        "skill_pmax": int(storage_pmax),
        "skill_jitter_early_start_pmax": int(early_start_pmax),
        "skill_jitter_late_start_pmax": int(late_start_pmax),
        "skill_jitter_early_end_pmax": int(early_end_pmax),
        "skill_jitter_late_end_pmax": int(late_end_pmax),
        "skill_jitter_distribution": str(distribution),
    }
    actual = {
        "skill_pmax": stored_scalar,
        # Historical scalar datasets mean the same pmax in all directions.
        "skill_jitter_early_start_pmax": int(
            info.get("skill_jitter_early_start_pmax", stored_scalar)
        ),
        "skill_jitter_late_start_pmax": int(
            info.get("skill_jitter_late_start_pmax", stored_scalar)
        ),
        "skill_jitter_early_end_pmax": int(
            info.get("skill_jitter_early_end_pmax", stored_scalar)
        ),
        "skill_jitter_late_end_pmax": int(
            info.get("skill_jitter_late_end_pmax", stored_scalar)
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
    parser.add_argument("--early-start-pmax", type=int, required=True)
    parser.add_argument("--late-start-pmax", type=int, required=True)
    parser.add_argument("--early-end-pmax", type=int, required=True)
    parser.add_argument("--late-end-pmax", type=int, required=True)
    parser.add_argument("--distribution", required=True)
    args = parser.parse_args()
    matched, detail = contract_matches(
        args.dataset_dir,
        storage_pmax=args.storage_pmax,
        early_start_pmax=args.early_start_pmax,
        late_start_pmax=args.late_start_pmax,
        early_end_pmax=args.early_end_pmax,
        late_end_pmax=args.late_end_pmax,
        distribution=args.distribution,
    )
    if not matched:
        print(f"SkillVLA jitter contract mismatch: {detail}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
