"""Read explicit trainer exit records independently of a Slurm allocation.

Never infer success from a quiet log, a disappeared PID, or a checkpoint count.
Old persistent-shell launches are imported without restarting their trainers.
"""
from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .slurm import CONCRETE_JOB_ID, output_paths, read_log, small_json, training_progress
else:
    from slurm import CONCRETE_JOB_ID, output_paths, read_log, small_json, training_progress


def timestamp(value):
    if not value:
        return ""
    parsed = datetime.fromisoformat(value)
    # Legacy compute-node records used naive UTC; new records include the offset.
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).isoformat()


def collect(project_root, allocations, known=(), history_days=3):
    root = Path(project_root)
    paths = list((root / ".cache/training_events").glob("*.json"))
    for folder in (".cache/srun_sessions", ".cache/maintenance"):
        paths.extend((root / folder).glob("*/training_status.json"))
    latest = {}
    for allocation in allocations:
        previous = latest.get(allocation["job_id"], {})
        if allocation.get("submitted_at", "") >= previous.get("submitted_at", ""):
            latest[allocation["job_id"]] = allocation
    rows = {}
    cutoff = time.time() - history_days * 86400
    for path in sorted(paths):
        try:
            if path.is_symlink() or path.stat().st_uid != os.getuid():
                continue
            record = small_json(path)
            if not isinstance(record, dict):
                continue
            identifier, started = str(record.get("job_id", "")), record.get("started_at")
            if not CONCRETE_JOB_ID.fullmatch(identifier) or not isinstance(started, str) or not started:
                continue
            manifest = path.parent / "launch.json"
            identity = record.get("run_id") or hashlib.sha256(
                (str(manifest) + "@" + started).encode()).hexdigest()[:32]
            key = "training:" + identity
            if key not in known and path.stat().st_mtime < cutoff:
                continue
            launch = small_json(manifest)
            if not isinstance(launch, dict):
                launch = {}
            logs = {value: read_log(value) for name in ("output", "error")
                    if isinstance((value := record.get(name)), str) and value}
            outputs = record.get("outputs") or list(dict.fromkeys(
                output for text in logs.values() for output in output_paths(text)))
            state = record.get("state")
            if state not in {"RUNNING", "COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"}:
                continue
            allocation = latest.get(identifier, {})
            resource = small_json(record.get("resource_status_path", ""))
            if not isinstance(resource, dict):
                resource = {}
            reason = resource.get("reason", "")
            if state == "RUNNING" and resource.get("state") == "WAITING_FOR_MEMORY":
                state = "WAITING_FOR_MEMORY"
            if state == "FAILED" and resource.get("state") == "CANCELLED":
                state = "CANCELLED"
            ended = timestamp(record.get("ended_at", ""))
            if state in {"RUNNING", "WAITING_FOR_MEMORY"} and allocation.get("terminal"):
                # Even COMPLETED only proves the allocation ended, not its trainers.
                state, ended = "INTERRUPTED", allocation.get("ended_at", "")
                reason = "학습 종료 기록 없이 GPU 할당 종료: " + allocation["state"]
            row = {"kind": "training", "run_id": identity, "job_id": identifier,
                   "job_name": record.get("name") or launch.get("name") or path.parent.name,
                   "state": state, "terminal": state not in {"RUNNING", "WAITING_FOR_MEMORY"},
                   "submitted_at": timestamp(started), "started_at": timestamp(started), "ended_at": ended,
                   "exit_code": record.get("exit_code"), "outputs": outputs,
                   "work_dir": record.get("work_dir") or launch.get("cwd", ""),
                   "log_paths": list(logs), "output_source": "개별 학습 실행 기록",
                   "reason": reason, "status_path": str(path),
                   "allocation_state": allocation.get("state", "UNKNOWN"),
                   "resource_status_path": record.get("resource_status_path", ""),
                   "parent_run_id": record.get("parent_run_id", "")}
            row["training_progress"] = training_progress(row, logs)
            # The journal and latest status file describe the same attempt.
            if key not in rows or not rows[key]["terminal"]:
                rows[key] = row
        except (OSError, ValueError, TypeError):
            continue
    # Group worker events are individual models; omit their launcher container.
    parents = {row["parent_run_id"] for row in rows.values() if row["parent_run_id"]}
    for row in rows.values():
        row["container"] = row["run_id"] in parents
    return rows
