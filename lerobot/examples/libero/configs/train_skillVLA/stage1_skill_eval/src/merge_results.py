#!/usr/bin/env python3
"""Race-free aggregation of multi-policy skill-eval worker manifests."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from html_report import write_html_report  # noqa: E402
from review_common import review_id_for_signature  # noqa: E402
from skill_data import token_to_coord  # noqa: E402


def _model_skill_success(
    manifest: dict,
    *,
    records: list[dict] | None = None,
) -> list[dict]:
    """Count green ID and OOD policy branches for each policy model."""
    branch_groups = {
        "id": {"policy", "policy_alt_noise"},
        "ood": {"policy_early", "policy_late"},
    }
    policies = manifest["signature"].get("policies", [])
    stats = [
        {
            "model_index": index,
            "label": str(policy.get("label", f"model_{index:02d}")),
            **{
                group: {"success_count": 0, "total_count": 0}
                for group in branch_groups
            },
        }
        for index, policy in enumerate(policies)
    ]
    source_records = manifest["records"].values() if records is None else records
    for record in source_records:
        model_index = int(record.get("model_index", 0))
        if not 0 <= model_index < len(stats):
            raise ValueError(f"Record has unknown model_index={model_index}.")
        for branch in record.get("branches", []):
            branch_name = branch.get("name")
            group = next(
                (
                    group_name
                    for group_name, names in branch_groups.items()
                    if branch_name in names
                ),
                None,
            )
            # GT is not in either group. Invalid early/late shifts were not
            # evaluated, so they are excluded from the corresponding denominator.
            if group is None or branch.get("unavailable_reason") is not None:
                continue
            stats[model_index][group]["total_count"] += 1
            stats[model_index][group]["success_count"] += int(
                bool(branch.get("green_tint", False))
            )
    for stat in stats:
        for group in branch_groups:
            total = int(stat[group]["total_count"])
            stat[group]["success_rate"] = (
                float(stat[group]["success_count"]) / total if total else 0.0
            )
    # Rank ID and OOD independently.  Dense ranking makes tied rates share the
    # same color while the next distinct rate remains rank 2.  A metric with no
    # evaluated branches is deliberately unranked.
    for group in branch_groups:
        rates = sorted(
            {
                float(stat[group]["success_rate"])
                for stat in stats
                if int(stat[group]["total_count"]) > 0
            },
            reverse=True,
        )
        ranks = {rate: index + 1 for index, rate in enumerate(rates[:2])}
        for stat in stats:
            rate = float(stat[group]["success_rate"])
            stat[group]["rank"] = (
                ranks.get(rate) if int(stat[group]["total_count"]) > 0 else None
            )
    return stats


def report_payload(manifest: dict, *, levels: list[int]) -> dict:
    by_token: dict[int, list[dict]] = {}
    for record in manifest["records"].values():
        by_token.setdefault(int(record["token"]), []).append(record)
    skills = []
    for token, records in sorted(by_token.items()):
        records.sort(
            key=lambda value: (
                value["task_id"],
                value["episode_id"],
                value["frame_start"],
                value.get("model_index", 0),
            )
        )
        skills.append(
            {
                "token": token,
                "coord": token_to_coord(token, levels),
                "model_skill_success": _model_skill_success(
                    manifest,
                    records=records,
                ),
                "occurrences": records,
            }
        )
    signature = manifest["signature"]
    review_id = review_id_for_signature(signature)
    return {
        "review_id": review_id,
        "levels": levels,
        "model_label": manifest["model_label"],
        "models": signature.get("policies", manifest.get("models", [])),
        "main_terminator": signature.get("main_terminator", {}),
        "target_task": signature["target_task"],
        "task_ids": sorted(int(value) for value in signature["selected_episodes"]),
        "selected_episode_count": sum(
            len(value) for value in signature["selected_episodes"].values()
        ),
        "occurrence_count": len(
            {
                record.get("occurrence_uid", record["uid"])
                for record in manifest["records"].values()
            }
        ),
        "evaluation_count": len(manifest["records"]),
        "model_skill_success": _model_skill_success(manifest),
        "time_shift_offset": signature["time_shift_offset"],
        "terminator_models": signature.get("terminator_models", []),
        "skills": skills,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def maybe_merge_chunks(
    output_dir: str | Path,
    *,
    expected_chunks: int,
) -> Path | None:
    """Merge only when every expected chunk exists; safe for all workers to call."""
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    chunks_dir = metrics_dir / "chunks"
    lock_path = metrics_dir / "merge.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        paths = [chunks_dir / f"chunk_{index:03d}.json" for index in range(expected_chunks)]
        if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
            return None
        chunks = [json.loads(path.read_text()) for path in paths]
        # Workers create their chunk manifest before rollout so resume can
        # recover partial work.  Presence alone therefore does not mean that a
        # chunk is ready for aggregation.
        if not all(chunk.get("completed", False) for chunk in chunks):
            return None
        signature = chunks[0].get("signature")
        levels = chunks[0].get("levels")
        for index, chunk in enumerate(chunks):
            if chunk.get("signature") != signature:
                raise ValueError(f"Chunk {index} signature does not match chunk 0.")
            if chunk.get("levels") != levels:
                raise ValueError(f"Chunk {index} FSQ levels do not match chunk 0.")
            if int(chunk.get("chunk_index", -1)) != index:
                raise ValueError(
                    f"Expected chunk_index={index}, got {chunk.get('chunk_index')}."
                )
            if int(chunk.get("chunk_count", -1)) != expected_chunks:
                raise ValueError(
                    f"Chunk {index} expected chunk_count={expected_chunks}, "
                    f"got {chunk.get('chunk_count')}."
                )

        manifest_path = metrics_dir / "manifest.json"
        report_path = output_dir / "index.html"
        latest_chunk_mtime = max(path.stat().st_mtime_ns for path in paths)
        if (
            manifest_path.is_file()
            and report_path.is_file()
            and manifest_path.stat().st_mtime_ns >= latest_chunk_mtime
            and report_path.stat().st_mtime_ns >= latest_chunk_mtime
        ):
            return report_path

        records = {}
        for chunk in chunks:
            overlap = sorted(set(records) & set(chunk.get("records", {})))
            if overlap:
                raise ValueError(f"Worker chunks contain duplicate occurrences: {overlap[:5]}")
            records.update(chunk.get("records", {}))
        merged = {
            "signature": signature,
            "model_label": chunks[0]["model_label"],
            "models": chunks[0].get("models", []),
            "architecture_label": chunks[0].get("architecture_label", ""),
            "levels": levels,
            "chunk_count": expected_chunks,
            "records": records,
        }
        _atomic_json(manifest_path, merged)
        return write_html_report(
            output_dir,
            report_payload(merged, levels=[int(value) for value in levels]),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-chunks", type=int, required=True)
    args = parser.parse_args()
    report = maybe_merge_chunks(args.output_dir, expected_chunks=args.expected_chunks)
    if report is None:
        raise SystemExit("Not all worker chunks are present yet.")
    print(report)


if __name__ == "__main__":
    main()
