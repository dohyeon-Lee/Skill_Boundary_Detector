import subprocess

import pytest

from tools.slurm_monitor.cluster_queue import ClusterQueue, priority_estimate, read


CONFIG = """PriorityType = priority/multifactor
PriorityWeightAge = 1000
PriorityWeightFairShare = 100000
PriorityWeightJobSize = 0
PriorityWeightPartition = 0
PriorityWeightQOS = 0
PriorityWeightTRES = (null)
PrioritySiteFactorPlugin = (null)
"""
SHARES = "root||\n default||\n default|test|0.230769\n"
NODES = "\n".join(
    f"NodeName={name} Gres=gpu:{kind}:8 State=MIXED Partitions=debug CfgTRES=cpu=256,gres/gpu=8 AllocTRES=cpu=16,gres/gpu={used}"
    for name, kind, used in [("node200", "B200", 8), ("node01", "L40S", 7), ("node100", "L40S", 8)])


def job(identifier, priority, *, user="other", gpu="B200", requested="node200", excluded="(null)", state="PENDING"):
    gres = "gres:gpu:1" if gpu is None else f"gres:gpu:{gpu}:1"
    return (f"JobId={identifier} JobName=model UserId={user}(123) Priority={priority} Account=default QOS=normal "
            f"JobState={state} Reason=Resources Partition=debug SubmitTime=2026-11-09T12:00:00 "
            f"StartTime=Unknown ReqNodeList={requested} ExcNodeList={excluded} NumCPUs=16 "
            f"TRES=cpu=16,gres/gpu=1 TresPerNode={gres}")


class Runner:
    def __init__(self, jobs):
        self.jobs, self.calls, self.fail = jobs, [], False

    def __call__(self, command):
        self.calls.append(command)
        assert command[0] in {"scontrol", "sshare"}
        if command[0] == "scontrol":
            assert command[1] == "show"
            if self.fail:
                raise subprocess.TimeoutExpired(command, 12)
            if command[2] == "node":
                return NODES
            if command[2] == "job":
                return self.jobs
            if command[2] == "config":
                return CONFIG
            if command[2] == "hostnames":
                if command[-1] == "node[01,100]":
                    return "node01\nnode100\n"
                return command[-1].replace(",", "\n")
        return SHARES


def test_typed_generic_and_excluded_nodes_and_ties():
    runner = Runner("\n".join([
        job("10", 40000), job("11", 40000, user="test"),
        job("12", 50000, gpu="L40S", requested="(null)"),
        job("13", 23077, gpu=None, requested="(null)"),
        job("14", 20000, gpu=None, requested="(null)", excluded="node200"),
        job("15", 60000, gpu="B200", requested="node100"),
        job("16", 90000, state="RUNNING"),
    ]))
    queue = ClusterQueue("test", runner)
    queue.poll()
    state = queue.snapshot()
    assert not state["error"]
    b200, l40s = state["groups"]
    assert [r["job_id"] for r in b200["pending"]] == ["10", "11", "13"]
    assert b200["pending"][1]["mine"]
    assert b200["pending"][0]["rank_first"] == 1 and b200["pending"][0]["rank_last"] == 2
    assert (b200["estimate"]["first"], b200["estimate"]["last"]) == (3, 4)
    assert [r["job_id"] for r in l40s["pending"]] == ["12", "13", "14"]
    assert l40s["nodes"][0]["unallocated"] == 1
    assert all(command[0] not in {"ssh", "srun", "sbatch", "scancel"} for command in runner.calls)


def test_failed_refresh_retains_last_successful_queue_and_timestamp():
    runner = Runner(job("10", 40000))
    queue = ClusterQueue("test", runner)
    queue.poll()
    before = queue.snapshot()
    runner.fail = True
    queue.poll()
    after = queue.snapshot()
    assert after["error"] and after["updated_at"] == before["updated_at"]
    assert after["groups"] == before["groups"]


def test_estimate_unavailable_with_unknown_factors_or_multiple_accounts():
    assert priority_estimate(CONFIG, SHARES, "test")["priority"] == 23077
    assert not priority_estimate(CONFIG.replace("PriorityWeightQOS = 0", "PriorityWeightQOS = 100"), SHARES, "test")["available"]
    assert not priority_estimate(CONFIG, SHARES + " another|test|0.5\n", "test")["available"]
    assert not priority_estimate(CONFIG.replace("priority/multifactor", "priority/basic"), SHARES, "test")["available"]


def test_empty_queue_does_not_invent_jobs():
    runner = Runner("")
    queue = ClusterQueue("test", runner)
    queue.poll()
    for group in queue.snapshot()["groups"]:
        assert group["pending"] == []
        assert group["estimate"]["first"] == group["estimate"]["last"] == 1
    assert all(command[0] != "sbatch" for command in runner.calls)


@pytest.mark.parametrize("command", [["ssh", "node200"], ["scancel", "123"], ["srun", "true"],
    ["sbatch", "--test-only", "file.sh"], ["scontrol", "update", "JobId=123"]])
def test_reader_rejects_non_readonly_operations(command):
    with pytest.raises(ValueError, match="read-only"):
        read(command)
