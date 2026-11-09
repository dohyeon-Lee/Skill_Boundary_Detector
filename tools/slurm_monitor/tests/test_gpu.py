import subprocess

import pytest

from tools.slurm_monitor.gpu import BEGIN, END, GPUs, allocation, parse_gpu


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


def test_only_own_allocated_gpu_and_stale_fallback():
    calls = []
    failed = False
    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if command[:3] == ["scontrol", "show", "job"]:
            return DETAIL
        if command[:3] == ["scontrol", "show", "hostnames"]:
            return "node200\n"
        assert command[0] == "ssh" and command[-1] == "node200"
        assert "nvidia-smi --query-gpu" in kwargs["input"]
        if failed:
            raise subprocess.TimeoutExpired(command, 12)
        return f"Login banner\n{BEGIN}\n{CSV}\n{END}\n"
    gpu = GPUs("test", runner)
    gpu.poll([JOB, {**JOB, "state": "PENDING", "job_id": "999"}])
    rows = gpu.snapshot()["devices"]
    assert len(rows) == 1 and rows[0]["index"] == 6 and rows[0]["job_ids"] == ["123"]
    assert rows[0]["used_gib"] == rows[0]["free_gib"] == 1
    failed = True
    gpu.poll([JOB])
    assert gpu.snapshot()["devices"][0]["stale"]
    assert gpu.snapshot()["devices"][0]["updated_at"] == rows[0]["updated_at"]
    assert gpu.snapshot()["errors"]
    gpu.poll([])
    assert gpu.snapshot()["devices"] == []
    assert not any(command[0] in {"srun", "scancel"} for command, _ in calls)


def test_other_owner_never_queried():
    def runner(command, **kwargs):
        assert command[0] == "scontrol"
        return DETAIL.replace("test(123)", "other(456)")
    gpu = GPUs("test", runner)
    gpu.poll([JOB])
    assert gpu.snapshot()["devices"] == []
    assert gpu.snapshot()["errors"]
