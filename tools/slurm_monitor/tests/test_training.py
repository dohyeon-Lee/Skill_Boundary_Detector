from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_monitor import monitor, only


def record(m, run="one", state="RUNNING", **extra):
    directory = Path(m.config["project_root"]) / ".cache/training_events"
    directory.mkdir(parents=True, exist_ok=True)
    value = {"run_id": run, "name": run, "job_id": "123", "state": state,
             "started_at": "2026-11-09T00:00:00+00:00", "outputs": ["/repo/" + run], **extra}
    (directory / (run + ".json")).write_text(json.dumps(value))
    return "training:" + run


def test_one_training_finishes_and_sends_while_gpu_and_other_training_continue(monitor):
    allocation_key = only(monitor)
    monitor.update_job(allocation_key, {"prompt": "keep allocation instruction", "armed": True})
    first = record(monitor)
    second = record(monitor, "two")
    monitor.poll()
    monitor.update_job(first, {"prompt": "transfer this model", "armed": True})
    record(monitor, state="COMPLETED", exit_code=0)
    monitor.poll()
    assert monitor.deliver_one()
    assert '"kind": "training"' in monitor.sent[0][1]
    assert '"allocation_state": "RUNNING"' in monitor.sent[0][1]
    assert monitor.jobs[allocation_key]["state"] == "RUNNING"
    assert monitor.jobs[allocation_key]["prompt"] == "keep allocation instruction"
    assert monitor.records[second]["state"] == "RUNNING"
    monitor.poll()
    assert not monitor.deliver_one()
    assert len(monitor.sent) == 1
    row = next(r for r in monitor.snapshot()["jobs"] if r["key"] == first)
    assert row["group"] == "training-ended" and not row["can_cancel"]
    with pytest.raises(ValueError):
        monitor.cancel_job(second)
    assert not monitor.slurm.cancelled


def test_timeout_does_not_claim_unfinished_training_completed(monitor):
    key = record(monitor)
    monitor.poll()
    monitor.update_job(key, {"prompt": "resume", "armed": True})
    monitor.slurm.rows[0].update(state="TIMEOUT", terminal=True, ended_at="2026-11-09T03:36:33")
    monitor.poll()
    assert monitor.records[key]["state"] == "INTERRUPTED"
    assert monitor.deliver_one()
    assert "TIMEOUT" in monitor.sent[0][1]


def test_failed_trainer_and_retries_are_independent_and_survive_restart(monitor):
    first = record(monitor)
    monitor.poll()
    monitor.update_job(first, {"prompt": "inspect failure", "armed": True})
    record(monitor, state="FAILED", exit_code=137)
    monitor.poll()
    assert monitor.deliver_one()
    second = record(monitor, "retry")
    monitor.poll()
    monitor.poll()
    assert monitor.records[first]["delivery"]["status"] == "sent"
    assert monitor.records[second]["delivery"]["status"] == "none"
    assert not monitor.deliver_one()
    saved = json.loads(monitor.store.path.read_text())["trainings"]
    assert saved[first]["events"] and second in saved


def test_training_poll_survives_scheduler_query_failure(monitor):
    key = record(monitor)
    monitor.poll()
    monitor.update_job(key, {"prompt": "report", "armed": True})
    monitor.slurm.error = "sacct temporarily down"
    record(monitor, state="COMPLETED", exit_code=0)
    monitor.poll()
    assert monitor.deliver_one()
    assert "sacct" in monitor.poll_error


def test_import_legacy_live_status_and_never_autosend_historical_record(monitor):
    folder = Path(monitor.config["project_root"]) / ".cache/maintenance/live"
    folder.mkdir(parents=True)
    path = folder / "training_status.json"
    log = folder / "train.out"
    log.write_text("output    : /repo/legacy\nschedule  : steps=50000\n")
    path.write_text(json.dumps({"job_id": "123", "state": "RUNNING",
                              "started_at": "2026-11-08T13:08:23", "output": str(log)}))
    monitor.poll()
    legacy = next(r for r in monitor.snapshot()["jobs"] if r.get("kind") == "training")
    assert legacy["outputs"] == ["/repo/legacy"]
    assert legacy["started_at"].endswith("+00:00")
    key = record(monitor, "history", state="COMPLETED", exit_code=0)
    monitor.poll()
    monitor.update_job(key, {"prompt": "manual only", "armed": True})
    monitor.poll()
    assert not monitor.deliver_one()
    monitor.request_send(key)
    assert monitor.deliver_one()


def test_group_only_displays_independent_workers(monitor):
    record(monitor, "parent")
    record(monitor, "worker1", parent_run_id="parent")
    record(monitor, "worker2", parent_run_id="parent")
    monitor.poll()
    names = {r["run_id"] for r in monitor.snapshot()["jobs"] if r.get("kind") == "training"}
    assert names == {"worker1", "worker2"}
