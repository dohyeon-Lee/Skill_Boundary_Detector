"""Read-only cluster queue and explicitly approximate new-job priority."""
from __future__ import annotations

import math
import re
import subprocess
import threading
from datetime import datetime, timezone

if __package__:
    from .slurm import parse_details
else:
    from slurm import parse_details


GROUPS = (("b200", "B200", {"node200"}), ("l40s", "L40S", {"node01", "node100"}))
EMPTY = {"", "(null)", "N/A", "None"}


def read(command):
    # This component cannot submit, cancel or modify jobs, or open SSH sessions.
    if not (command[:2] == ["scontrol", "show"] or command[0] == "sshare"):
        raise ValueError("Cluster queue accepts read-only Slurm queries only.")
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=12)
    if result.returncode:
        raise RuntimeError((result.stderr.strip() or "Slurm 대기열 조회 실패")[-500:])
    return result.stdout


def records(text):
    return [parse_details(line) for line in text.splitlines() if line.strip()]


def gpu_count(tres):
    match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres or "")
    return int(match[1]) if match else 0


def rank(rows, priority):
    higher = sum(row["priority"] > priority for row in rows)
    tied = sum(row["priority"] == priority for row in rows)
    return {"first": higher + 1, "last": higher + tied + 1, "ahead": higher, "tied": tied}


def priority_estimate(config_text, shares_text, user):
    config = dict(re.findall(r"^\s*(\w+)\s*=\s*(.*?)\s*$", config_text, re.MULTILINE))
    if config.get("PriorityType") != "priority/multifactor":
        return {"available": False, "reason": "현재 priority 방식은 신규 잡 추정을 지원하지 않습니다."}
    for name, value in config.items():
        if name.startswith("PriorityWeight") and name not in {"PriorityWeightFairShare", "PriorityWeightAge"}:
            if value not in {"0", "(null)", ""}:
                return {"available": False, "reason": "자원·QoS 등 추가 priority 요인이 있어 제출 전 순위를 추정하지 않습니다."}
    if config.get("PrioritySiteFactorPlugin", "(null)") not in EMPTY:
        return {"available": False, "reason": "별도 site priority 정책으로 신규 잡 순위를 추정할 수 없습니다."}
    matches = []
    for line in shares_text.splitlines():
        fields = [part.strip() for part in line.split("|")]
        if len(fields) >= 3 and fields[1] == user:
            value = float(fields[2])
            if math.isfinite(value) and 0 <= value <= 1:
                matches.append((fields[0], value))
    if len(matches) != 1:
        return {"available": False, "reason": "계정별 fair-share를 하나로 특정할 수 없어 신규 잡 순위를 추정하지 않습니다."}
    weight = int(config.get("PriorityWeightFairShare", "0"))
    account, share = matches[0]
    return {"available": True, "priority": max(1, round(share * weight)),
            "account": account, "fair_share": share,
            "basis": "현재 fair-share · 새 잡 대기시간 0 · Nice 0 기준 (반올림 오차 약 ±1)."}


class ClusterQueue:
    def __init__(self, user, runner=read):
        self.user, self.run = user, runner
        self.lock = threading.Lock()
        self.value = {"groups": [], "updated_at": None, "error": "", "interval": 30}

    def snapshot(self):
        with self.lock:
            # Poll replaces the complete immutable tree, never mutates it.
            return dict(self.value)

    def poll(self):
        try:
            nodes = {row["NodeName"]: row for row in records(self.run(["scontrol", "show", "node", "-o"]))}
            jobs = records(self.run(["scontrol", "show", "job", "-d", "-o"]))
            warnings = []
            try:
                estimate = priority_estimate(self.run(["scontrol", "show", "config"]),
                    self.run(["sshare", "-n", "-P", "-u", self.user, "--format=Account,User,FairShare"]), self.user)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                estimate = {"available": False, "reason": "신규 잡 priority 추정 불가: " + str(error)}
            expanded = {}
            def hosts(expression):
                if expression in EMPTY:
                    return set()
                if expression not in expanded:
                    expanded[expression] = set(self.run(["scontrol", "show", "hostnames", expression]).split())
                return expanded[expression]
            pending = []
            for job in jobs:
                if job.get("JobState") not in {"PENDING", "REQUEUED", "REQUEUE_HOLD", "RESV_DEL_HOLD"}:
                    continue
                gpu_request = ",".join(job.get(key, "") for key in ("TresPerNode", "TresPerJob", "TresPerTask", "TresPerSocket", "Gres") if job.get(key))
                if not gpu_count(job.get("TRES")) and "gpu" not in gpu_request:
                    continue
                try:
                    requested, excluded = hosts(job.get("ReqNodeList", "")), hosts(job.get("ExcNodeList", ""))
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    warnings.append("#" + job.get("JobId", "?") + " 노드 조건 조회 실패: " + str(error))
                    continue
                partitions = set(job.get("Partition", "").split(","))
                types = {value.casefold() for value in re.findall(r"(?:gres[:/])?gpu:([A-Za-z][A-Za-z0-9_-]*):\d+", gpu_request)}
                eligible = set()
                for node, detail in nodes.items():
                    if requested and node not in requested or node in excluded:
                        continue
                    if partitions and not partitions.intersection(detail.get("Partitions", "").split(",")):
                        continue
                    node_types = {value.casefold() for value in re.findall(r"gpu:([A-Za-z][A-Za-z0-9_-]*):", detail.get("Gres", ""))}
                    if types and not types.intersection(node_types):
                        continue
                    eligible.add(node)
                user = job.get("UserId", "").split("(")[0]
                pending.append({"job_id": job.get("JobId", ""), "name": job.get("JobName", ""),
                    "user": user, "mine": user == self.user, "account": job.get("Account", ""),
                    "priority": int(job.get("Priority", "0")), "submitted_at": job.get("SubmitTime", ""),
                    "reason": job.get("Reason", ""), "gpu_request": gpu_request,
                    "cpus": job.get("NumCPUs", ""), "qos": job.get("QOS", ""),
                    "expected_start": job.get("StartTime", "Unknown"),
                    "eligible_nodes": sorted(eligible), "features": job.get("Features", ""),
                    "dependency": job.get("Dependency", ""), "reservation": job.get("Reservation", "")})
            groups = []
            for identifier, label, target in GROUPS:
                target = {node for node in target if node in nodes and re.search(
                    r"gpu:" + re.escape(label) + r":", nodes[node].get("Gres", ""), re.IGNORECASE)}
                visible = sorted((row for row in pending if target.intersection(row["eligible_nodes"])),
                    key=lambda row: (-row["priority"], row["submitted_at"], row["job_id"]))
                rows = []
                for row in visible:
                    position = rank(visible, row["priority"])
                    rows.append({**row, "rank_first": position["first"], "rank_last": position["last"] - 1})
                devices = []
                for node in sorted(target.intersection(nodes)):
                    detail = nodes[node]
                    total, allocated = gpu_count(detail.get("CfgTRES")), gpu_count(detail.get("AllocTRES"))
                    devices.append({"name": node, "total": total, "allocated": allocated,
                        "unallocated": max(0, total - allocated), "state": detail.get("State", "UNKNOWN")})
                hypothetical = dict(estimate)
                if not target:
                    hypothetical = {"available": False, "reason": "이 클러스터에서 해당 rllab GPU 노드를 확인할 수 없습니다."}
                if hypothetical["available"]:
                    hypothetical.update(rank(visible, hypothetical["priority"]))
                groups.append({"id": identifier, "label": label, "nodes": devices, "pending": rows,
                               "estimate": hypothetical})
            value = {"groups": groups, "updated_at": datetime.now(timezone.utc).isoformat(),
                     "error": "\n".join(warnings), "interval": 30}
            with self.lock:
                self.value = value
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            with self.lock:
                self.value = {**self.value, "error": str(error)}
