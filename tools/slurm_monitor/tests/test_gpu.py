import json
import os
import time
from types import SimpleNamespace

import pytest

from tools.slurm_monitor.gpu import GPUs, allocation, parse_gpu, sample_local, sample_processes, process_run_id, attach_processes
from tools.slurm_monitor import gpu as module


DETAIL = ("JobId=123 UserId=test(123) JobState=RUNNING SubmitTime=2026-11-09T01:00:00 "
          "Nodes=node200 CPU_IDs=0-15 Mem=0 GRES=gpu:B200:1(IDX:6)")
JOB = dict(job_id="123", state="RUNNING", submitted_at="2026-11-09T01:00:00")
CSV = "0, GPU-other, NVIDIA B200, 500, 1500, 2048\n6, GPU-mine, NVIDIA B200, 1024, 1024, 2048"


def test_physical_allocation_and_identity():
    assert allocation(DETAIL, "test", JOB["submitted_at"]) == {"node200": {6}}
    multi = DETAIL + " Nodes=node100 CPU_IDs=0-15 Mem=0 GRES=gpu:L40S:3(IDX:0-1,3)"
    assert allocation(multi, "test", JOB["submitted_at"])["node100"] == {0, 1, 3}
    for user, submit in [("other", JOB["submitted_at"]), ("test", "different")]:
        with pytest.raises(ValueError):
            allocation(DETAIL, user, submit)
    assert allocation(DETAIL.replace("RUNNING", "COMPLETED"), "test", JOB["submitted_at"]) == {}


def test_memory_parser_rejects_unavailable_values():
    assert parse_gpu(CSV)[1]["used_gib"] == 1
    for value in ["", "N/A", "nan", "inf", "-1"]:
        with pytest.raises(ValueError):
            parse_gpu(f"6, GPU-id, B200, {value}, 1024, 2048")


def test_allocation_cgroup_remapped_index():
    devices = sample_local({"SLURM_JOB_GPUS": "6"}, lambda _: "0, GPU-mine, B200, 1024, 1024, 2048")
    assert devices[0]["index"] == 6 and devices[0]["visible_index"] == 0
    assert devices[0]["uuid"] == "GPU-mine"


def test_collector_uses_original_allocation_without_ssh(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "TELEMETRY", tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.setattr(module.socket, "gethostname", lambda: "node200")
    monkeypatch.setattr(module.pwd, "getpwuid", lambda _: SimpleNamespace(pw_name="test"))
    calls = []
    def runner(command):
        calls.append(command)
        if command[0] == "scontrol":
            return DETAIL
        assert command[0] == "nvidia-smi"
        return "0, GPU-mine, NVIDIA B200, 1024, 1024, 2048"
    monkeypatch.setattr(module, "run", runner)
    def stop(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(module.time, "sleep", stop)
    with pytest.raises(KeyboardInterrupt):
        module.collect()
    result = json.loads((tmp_path / "123/node200.json").read_text())
    assert result["job_id"] == "123" and result["uid"] == os.getuid()
    assert result["devices"][0]["index"] == 6
    assert all(command[0] not in {"ssh", "srun"} for command in calls)


def test_only_own_allocated_gpu_and_stale_fallback(tmp_path):
    calls = []
    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if command[:3] == ["scontrol", "show", "job"]:
            return DETAIL
        if command[:3] == ["scontrol", "show", "hostnames"]:
            return "node200\n"
        raise AssertionError("Telemetry must never open SSH or an extra srun step")
    folder = tmp_path / "123"
    folder.mkdir()
    path = folder / "node200.json"
    sample = dict(job_id="123", node="node200", uid=os.getuid(), sampled_at=time.time(), devices=parse_gpu(CSV))
    path.write_text(json.dumps(sample))
    gpu = GPUs("test", runner, telemetry=tmp_path)
    gpu.poll([JOB, {**JOB, "state": "PENDING", "job_id": "999"}])
    rows = gpu.snapshot()["devices"]
    assert len(rows) == 1 and rows[0]["index"] == 6 and rows[0]["job_ids"] == ["123"]
    assert rows[0]["used_gib"] == rows[0]["free_gib"] == 1
    sample["sampled_at"] -= 60
    path.write_text(json.dumps(sample))
    gpu.poll([JOB])
    assert gpu.snapshot()["devices"][0]["stale"]
    assert gpu.snapshot()["devices"][0]["updated_at"] == rows[0]["updated_at"]
    assert gpu.snapshot()["errors"]
    gpu.poll([])
    assert gpu.snapshot()["devices"] == []
    assert not any(command[0] in {"ssh", "srun", "scancel"} for command, _ in calls)


def test_other_owner_never_queried():
    def runner(command, **kwargs):
        assert command[0] == "scontrol"
        return DETAIL.replace("test(123)", "other(456)")
    gpu = GPUs("test", runner)
    gpu.poll([JOB])
    assert gpu.snapshot()["devices"] == []
    assert gpu.snapshot()["errors"]


def test_gpu_processes_keep_explicit_run_identity_and_deduplicate():
    run_a, run_b = "a" * 32, "b" * 32
    def identity(pid):
        if pid == 999:
            raise PermissionError("not ours")
        return {101: run_a, 102: run_a, 201: run_b}.get(pid)
    text = "101, GPU-mine, 1024\n102, GPU-mine, 512\n201, GPU-mine, 2048\n101, GPU-mine, 1024\n999, GPU-mine, 5000\n"
    samples = sample_processes(lambda _: text, identity)["GPU-mine"]
    assert len(samples) == 3
    assert sum(row["used_gib"] for row in samples if row["run_id"] == run_a) == 1.5
    assert sum(row["used_gib"] for row in samples if row["run_id"] == run_b) == 2


def test_process_marker_reads_only_run_id_and_rejects_missing_or_invalid(tmp_path):
    process = tmp_path / "101"
    process.mkdir()
    (process / "stat").write_text("101 (trainer) " + " ".join(["S"] + ["0"] * 18 + ["12345"]))
    (process / "environ").write_bytes(b"HF_TOKEN=secret\0SBD_TRAINING_PARENT_ID=" + b"a" * 32 + b"\0")
    assert process_run_id(101, tmp_path) == "a" * 32
    (process / "environ").write_bytes(b"HF_TOKEN=secret\0")
    assert process_run_id(101, tmp_path) is None
    (process / "environ").write_bytes(b"SBD_TRAINING_PARENT_ID=not-a-run\0")
    assert process_run_id(101, tmp_path) is None


def test_failed_process_query_keeps_stale_process_reading():
    previous = [{"uuid": "GPU-mine", "processes": [{"pid": 101, "run_id": "a" * 32, "used_gib": 1.5}], "process_sampled_at": 100}]
    def failed(_):
        raise RuntimeError("process query unavailable")
    current = attach_processes([{"uuid": "GPU-mine", "used_gib": 3}], failed, previous)[0]
    assert current["used_gib"] == 3
    assert current["processes"] == previous[0]["processes"]
    assert current["processes_stale"] and current["process_sampled_at"] == 100
