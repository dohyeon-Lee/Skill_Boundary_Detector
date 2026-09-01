#!/usr/bin/env python3
"""Plan and build task-disjoint CALVIN long-horizon LeRobot datasets."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from calvin_long_horizon import candidate_key
from calvin_task_split_config import (
    CALVIN_DOWNLOAD_DIR,
    TaskSplitSettings,
    output_names,
)

from convert_calvin_to_lerobot import (  # type: ignore[import-not-found]
    _merged_intervals,
    load_annotations,
    load_play_recordings,
    subtract_intervals,
)


DATASET_ROLES = ("play_pretrain", "language_pretrain", "heldout")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def load_candidate_report(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("candidates")
    if not isinstance(rows, list):
        raise ValueError(f"candidate report has no candidates list: {path}")
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"candidate report candidates[{index}] must be a mapping")
        task_ids = tuple(str(value) for value in row.get("task_ids", []))
        stable_key = candidate_key(task_ids)
        reported_key = str(row.get("candidate_key", stable_key))
        if reported_key != stable_key:
            raise ValueError(
                f"candidate report key mismatch: {reported_key!r} != {stable_key!r}"
            )
        if stable_key in indexed:
            raise ValueError(f"duplicate candidate key in report: {stable_key}")
        indexed[stable_key] = row
    return report, indexed


def _selected_occurrences(
    indexed: dict[str, dict[str, Any]],
    selections: tuple[Any, ...],
    recordings: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    occurrences: list[dict[str, Any]] = []
    selected_metadata: list[dict[str, str]] = []
    unknown = [row.candidate_key for row in selections if row.candidate_key not in indexed]
    if unknown:
        raise ValueError(
            f"selected candidate keys are absent from the report: {unknown}; "
            "regenerate the report after changing search settings"
        )
    for selection in selections:
        candidate = indexed[selection.candidate_key]
        rows = candidate.get("occurrences")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"candidate {selection.candidate_key} has no occurrences")
        selected_metadata.append(
            {"candidate_key": selection.candidate_key, "language": selection.language}
        )
        for occurrence_index, row in enumerate(rows):
            recording_index = int(row["recording_index"])
            if not 0 <= recording_index < len(recordings):
                raise ValueError(
                    f"candidate {selection.candidate_key} has invalid recording index "
                    f"{recording_index}"
                )
            start, end = int(row["source_start"]), int(row["source_end"])
            recording_start, recording_end = recordings[recording_index]
            if not (recording_start <= start <= end <= recording_end):
                raise ValueError(
                    f"candidate {selection.candidate_key} occurrence [{start}, {end}] "
                    f"is outside recording [{recording_start}, {recording_end}]"
                )
            occurrences.append(
                {
                    "candidate_key": selection.candidate_key,
                    "language": selection.language,
                    "candidate_occurrence_index": occurrence_index,
                    "recording_index": recording_index,
                    "recording_start": recording_start,
                    "recording_end": recording_end,
                    "start": start,
                    "end": end,
                }
            )
    return occurrences, selected_metadata


def make_split_units(
    annotation: dict[str, Any],
    recordings: list[tuple[int, int]],
    selected_occurrences: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[int, int]], list[int]]:
    heldout_intervals = _merged_intervals(
        [(int(row["start"]), int(row["end"])) for row in selected_occurrences]
    )
    play_units = [
        {
            "kind": "play",
            "source_unit_index": recording_index,
            "start": start,
            "end": end,
            "recording_start": recording_start,
            "recording_end": recording_end,
            "task_id": "play",
            "language": "",
            "embedding_annotation_index": None,
        }
        for recording_index, start, end, recording_start, recording_end in subtract_intervals(
            recordings, heldout_intervals
        )
    ]

    removed_annotation_indices: list[int] = []
    language_units: list[dict[str, Any]] = []
    for annotation_index, interval_raw in enumerate(np.asarray(annotation["intervals"])):
        interval = tuple(map(int, interval_raw))
        if any(_overlaps(interval, heldout) for heldout in heldout_intervals):
            removed_annotation_indices.append(annotation_index)
            continue
        language_units.append(
            {
                "kind": "annotation",
                "source_unit_index": annotation_index,
                "start": interval[0],
                "end": interval[1],
                "recording_start": None,
                "recording_end": None,
                "task_id": str(annotation["task_ids"][annotation_index]),
                "language": str(annotation["annotations"][annotation_index]),
                "embedding_annotation_index": annotation_index,
            }
        )

    heldout_units = [
        {
            "kind": "long_horizon",
            "source_unit_index": unit_index,
            "start": int(row["start"]),
            "end": int(row["end"]),
            "recording_start": int(row["recording_start"]),
            "recording_end": int(row["recording_end"]),
            "task_id": str(row["candidate_key"]),
            "language": str(row["language"]),
            "embedding_annotation_index": None,
            "candidate_key": str(row["candidate_key"]),
            "candidate_occurrence_index": int(row["candidate_occurrence_index"]),
        }
        for unit_index, row in enumerate(selected_occurrences)
    ]
    return (
        {
            "play_pretrain": play_units,
            "language_pretrain": language_units,
            "heldout": heldout_units,
        },
        heldout_intervals,
        removed_annotation_indices,
    )


def create_unit_plans(settings: TaskSplitSettings) -> dict[str, Path]:
    if not settings.selected_candidates:
        raise ValueError(
            "selected_candidates is empty; copy candidate_key values from the HTML and "
            "assign one language instruction to each"
        )
    report_path = settings.candidate_report_path
    if not report_path.is_file():
        raise FileNotFoundError(
            f"candidate report not found: {report_path}; run find_long_horizon_candidates.py first"
        )
    source_dir = Path(settings.conversion["calvin_convert_source_dir"]).resolve()
    report, indexed = load_candidate_report(report_path)
    report_source = Path(str(report.get("source_dir", ""))).expanduser().resolve()
    if report_source != source_dir:
        raise ValueError(
            f"candidate report source {report_source} does not match conversion source {source_dir}"
        )
    annotation = load_annotations(
        source_dir, str(settings.conversion["calvin_convert_annotation_folder"])
    )
    recordings = load_play_recordings(source_dir)
    occurrences, selected_metadata = _selected_occurrences(
        indexed, settings.selected_candidates, recordings
    )
    units_by_role, heldout_intervals, removed_annotations = make_split_units(
        annotation, recordings, occurrences
    )
    if any(not units_by_role[role] for role in DATASET_ROLES):
        empty = [role for role in DATASET_ROLES if not units_by_role[role]]
        raise ValueError(f"task split produced empty dataset roles: {empty}")

    names = output_names(settings)
    report_sha = _sha256(report_path)
    annotation_sha = _sha256(Path(annotation["path"]))
    settings.plan_dir.mkdir(parents=True, exist_ok=True)
    plans: dict[str, Path] = {}
    for role in DATASET_ROLES:
        conversion_mode = {
            "play_pretrain": "play",
            "language_pretrain": "annotated",
            "heldout": "long_horizon",
        }[role]
        payload = {
            "schema_version": 1,
            "dataset_role": role,
            "conversion_mode": conversion_mode,
            "output_name": names[role],
            "overwrite": settings.overwrite,
            "source_dir": str(source_dir),
            "annotation_file": str(annotation["path"]),
            "annotation_sha256": annotation_sha,
            "candidate_report": str(report_path),
            "candidate_report_sha256": report_sha,
            "selected_candidate_keys": [
                row.candidate_key for row in settings.selected_candidates
            ],
            "selected_candidates": selected_metadata,
            "selected_occurrence_count": len(occurrences),
            "removed_intervals": [list(interval) for interval in heldout_intervals],
            "removed_annotation_indices": removed_annotations,
            "interval_policy": (
                "remove the union of exact selected candidate [start, end] spans; margin=0"
            ),
            "language_pretrain_overlap_policy": (
                "remove any original language annotation overlapping a selected span"
            ),
            "units": units_by_role[role],
        }
        plan_path = settings.plan_dir / f"{role}.json"
        plan_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        plans[role] = plan_path
    return plans


def preflight_outputs(settings: TaskSplitSettings) -> dict[str, Path]:
    output_root = Path(settings.conversion["calvin_convert_output_root"]).resolve()
    outputs = {role: output_root / name for role, name in output_names(settings).items()}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not settings.overwrite:
        raise FileExistsError(
            "split outputs already exist and overwrite=false: "
            + ", ".join(str(path) for path in existing)
        )
    return outputs


def build(settings: TaskSplitSettings) -> dict[str, Path]:
    outputs = preflight_outputs(settings)
    plans = create_unit_plans(settings)
    print("CALVIN long-horizon task split", flush=True)
    print(
        "  selected : "
        + ", ".join(row.candidate_key for row in settings.selected_candidates),
        flush=True,
    )
    for role in DATASET_ROLES:
        with plans[role].open(encoding="utf-8") as stream:
            plan = json.load(stream)
        print(
            f"  {role:17s}: {len(plan['units'])} episodes -> {outputs[role]}",
            flush=True,
        )
    if settings.plan_only:
        print(f"PLAN ONLY: unit plans written to {settings.plan_dir}", flush=True)
        return outputs

    converter = CALVIN_DOWNLOAD_DIR / "src" / "convert_calvin_to_lerobot.py"
    for role in DATASET_ROLES:
        print(f"\n== Build {role} ==", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(converter),
                "--config",
                str(settings.conversion_config_path),
                "--unit-plan",
                str(plans[role]),
            ],
            check=True,
        )
    print("DONE: all three CALVIN task-disjoint datasets were built", flush=True)
    return outputs
