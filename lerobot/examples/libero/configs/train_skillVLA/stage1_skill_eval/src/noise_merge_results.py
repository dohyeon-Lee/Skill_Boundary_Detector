#!/usr/bin/env python3
"""Merge repeated-noise worker chunks and build the linked-codebook report."""

from __future__ import annotations

import argparse
import fcntl
import json
from copy import deepcopy
from pathlib import Path

from noise_html_report import write_noise_html_report
from skill_data import token_to_coord


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _merge_record(target: dict, incoming: dict) -> None:
    target_meta = {key: value for key, value in target.items() if key != "rollouts"}
    incoming_meta = {key: value for key, value in incoming.items() if key != "rollouts"}
    if target_meta != incoming_meta:
        raise ValueError(f"Worker metadata disagrees for {target.get('uid')}.")
    original_token = int(target["token"])
    by_route = {
        (
            str(item.get("rollout_path", "main")),
            int(item.get("eval_token", original_token)),
            int(item["noise_index"]),
        ): item
        for item in target.get("rollouts", [])
    }
    for rollout in incoming.get("rollouts", []):
        rollout_path = str(rollout.get("rollout_path", "main"))
        eval_token = int(rollout.get("eval_token", original_token))
        noise_index = int(rollout["noise_index"])
        key = (rollout_path, eval_token, noise_index)
        previous = by_route.get(key)
        if previous is not None and previous != rollout:
            raise ValueError(
                f"Conflicting code/noise rollout {key} for {target.get('uid')}."
            )
        by_route[key] = rollout
    target["rollouts"] = [by_route[key] for key in sorted(by_route)]


def report_payload(manifest: dict, *, rollout_path: str = "main") -> dict:
    rollout_path = str(rollout_path).strip().lower()
    if rollout_path not in {"main", "skill_only"}:
        raise ValueError(f"Unknown rollout path: {rollout_path!r}.")
    signature = manifest["signature"]
    code_probe_mode = signature.get("code_probe_mode")
    if code_probe_mode is None:
        code_probe_mode = (
            "neighbor" if signature.get("neighbor_code_probe", False) else "off"
        )
    records = sorted(
        (
            {
                **record,
                "rollouts": [
                    rollout
                    for rollout in record.get("rollouts", [])
                    if str(rollout.get("rollout_path", "main")) == rollout_path
                ],
            }
            for record in manifest["records"].values()
        ),
        key=lambda item: (
            int(item["task_id"]),
            int(item["episode_id"]),
            int(item["frame_start"]),
            int(item["model_index"]),
        ),
    )
    model_levels = manifest["model_levels"]
    skill_spaces = []
    for model_index, policy in enumerate(signature["policies"]):
        by_token: dict[int, set[str]] = {}
        for record in records:
            if int(record["model_index"]) != model_index:
                continue
            by_token.setdefault(int(record["token"]), set()).add(
                str(record["occurrence_uid"])
            )
        skill_spaces.append(
            {
                "model_index": model_index,
                "label": str(policy["label"]),
                "levels": model_levels[model_index],
                "skills": [
                    {
                        "token": token,
                        "coord": token_to_coord(token, model_levels[model_index]),
                        "member_ids": sorted(member_ids),
                    }
                    for token, member_ids in sorted(by_token.items())
                ],
            }
        )
    selected = signature["selected_episodes"]
    return {
        "models": signature["policies"],
        "target_task": signature["target_task"],
        "task_ids": sorted(int(task_id) for task_id in selected),
        "env_count": sum(len(episode_ids) for episode_ids in selected.values()),
        "noise_rollouts_per_env": int(signature["noise_rollouts_per_env"]),
        "rollout_randomization": str(
            signature.get("rollout_randomization", "noise")
        ),
        "code_probe_mode": str(code_probe_mode),
        "rollout_path": rollout_path,
        "rollout_view_label": (
            "Main action path" if rollout_path == "main" else "Skill-only path"
        ),
        "skill_only_rollout_probe": bool(
            signature.get("skill_only_rollout_probe", False)
        ),
        "occurrence_count": len(
            {str(record["occurrence_uid"]) for record in records}
        ),
        "skill_spaces": skill_spaces,
        "occurrences": records,
    }


def write_noise_html_reports(output_dir: str | Path, manifest: dict) -> Path:
    """Write the main report and, when requested, its skill-only companion."""
    output_dir = Path(output_dir)
    main_report = write_noise_html_report(
        output_dir,
        report_payload(manifest, rollout_path="main"),
    )
    if bool(manifest["signature"].get("skill_only_rollout_probe", False)):
        write_noise_html_report(
            output_dir,
            report_payload(manifest, rollout_path="skill_only"),
        )
    return main_report


def maybe_merge_noise_chunks(
    output_dir: str | Path,
    *,
    expected_chunks: int,
) -> Path | None:
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    chunks_dir = metrics_dir / "chunks"
    lock_path = metrics_dir / "noise_merge.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        paths = [
            chunks_dir / f"chunk_{index:03d}.json"
            for index in range(expected_chunks)
        ]
        if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
            return None
        chunks = [json.loads(path.read_text()) for path in paths]
        if not all(chunk.get("completed", False) for chunk in chunks):
            return None

        signature = chunks[0]["signature"]
        model_levels = chunks[0]["model_levels"]
        for index, chunk in enumerate(chunks):
            if chunk.get("signature") != signature:
                raise ValueError(f"Chunk {index} signature differs from chunk 0.")
            if chunk.get("model_levels") != model_levels:
                raise ValueError(f"Chunk {index} model levels differ from chunk 0.")
            if int(chunk.get("chunk_index", -1)) != index:
                raise ValueError(
                    f"Expected chunk_index={index}, got {chunk.get('chunk_index')}."
                )
            if int(chunk.get("chunk_count", -1)) != expected_chunks:
                raise ValueError(
                    f"Chunk {index} has chunk_count={chunk.get('chunk_count')}, "
                    f"expected {expected_chunks}."
                )

        manifest_path = metrics_dir / "manifest.json"
        report_paths = [output_dir / "index.html"]
        if bool(signature.get("skill_only_rollout_probe", False)):
            report_paths.append(output_dir / "skill_only.html")
        latest_chunk = max(path.stat().st_mtime_ns for path in paths)
        if (
            manifest_path.is_file()
            and all(path.is_file() for path in report_paths)
            and manifest_path.stat().st_mtime_ns >= latest_chunk
            and all(path.stat().st_mtime_ns >= latest_chunk for path in report_paths)
        ):
            return report_paths[0]

        records: dict[str, dict] = {}
        for chunk in chunks:
            for uid, incoming in chunk.get("records", {}).items():
                if uid not in records:
                    records[uid] = deepcopy(incoming)
                else:
                    _merge_record(records[uid], incoming)
        expected_rollouts = int(signature["noise_rollouts_per_env"])
        expected_paths = ["main"]
        if bool(signature.get("skill_only_rollout_probe", False)):
            expected_paths.append("skill_only")
        incomplete = {}
        for uid, record in records.items():
            original_token = int(record["token"])
            expected_tokens = [
                int(value)
                for value in record.get("evaluated_tokens", [original_token])
            ]
            counts = {
                (rollout_path, token): 0
                for rollout_path in expected_paths
                for token in expected_tokens
            }
            for rollout in record.get("rollouts", []):
                rollout_path = str(rollout.get("rollout_path", "main"))
                eval_token = int(rollout.get("eval_token", original_token))
                key = (rollout_path, eval_token)
                if key in counts:
                    counts[key] += 1
            bad = {
                f"{rollout_path}:{token}": count
                for (rollout_path, token), count in counts.items()
                if count != expected_rollouts
            }
            if bad:
                incomplete[uid] = bad
        if incomplete:
            preview = list(incomplete.items())[:5]
            raise ValueError(
                "Merged records do not contain every configured code/noise rollout: "
                f"{preview}."
            )
        merged = {
            "signature": signature,
            "model_levels": model_levels,
            "chunk_count": expected_chunks,
            "records": records,
        }
        _atomic_json(manifest_path, merged)
        return write_noise_html_reports(output_dir, merged)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-chunks", type=int, required=True)
    args = parser.parse_args()
    report = maybe_merge_noise_chunks(
        args.output_dir,
        expected_chunks=args.expected_chunks,
    )
    if report is None:
        raise SystemExit("Not all worker chunks are complete yet.")
    print(report)


if __name__ == "__main__":
    main()
