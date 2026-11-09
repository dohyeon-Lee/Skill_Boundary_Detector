"""Read-only GPU telemetry for the current user's running allocations."""
from __future__ import annotations

import csv
import fcntl
import io
import json
import math
import os
import pwd
import re
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

QUERY = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,memory.total",
         "--format=csv,noheader,nounits"]
TELEMETRY = Path(__file__).resolve().parents[2] / ".cache/gpu_telemetry"


def sample_local(environment, runner=None):
    rows = (runner or run)(QUERY)
    devices = parse_gpu(rows)
    assigned = environment.get("SLURM_JOB_GPUS", "")
    # Device cgroups can expose physical GPU 6 as index 0 to nvidia-smi.
    # A single visible GPU + a single Slurm physical ID is unambiguous.
    if len(devices) == 1 and re.fullmatch(r"\d+", assigned):
        devices[0]["visible_index"] = devices[0]["index"]
        devices[0]["index"] = int(assigned)
    return devices


def collect():
    """Run inside an existing allocation, never through a compute-node SSH login."""
    identifier = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"\d+(?:_\d+)?(?:\+\d+)?", identifier):
        raise ValueError("GPU collector must run inside a Slurm allocation.")
    node = socket.gethostname().split(".")[0]
    sample_environment = dict(os.environ)
    if not sample_environment.get("SLURM_JOB_GPUS"):
        try:
            detail = run(["scontrol", "show", "job", "-d", "-o", identifier])
            submit = re.search(r"\bSubmitTime=(\S+)", detail)
            assigned = allocation(detail, pwd.getpwuid(os.getuid()).pw_name, submit[1]).get(node, set()) if submit else set()
            if len(assigned) == 1:
                sample_environment["SLURM_JOB_GPUS"] = str(next(iter(assigned)))
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
            pass
    folder = TELEMETRY / identifier
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (folder / (node + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        target = folder / (node + ".json")
        while True:
            try:
                data = {"job_id": identifier, "node": node, "uid": os.getuid(),
                        "sampled_at": time.time(), "pid": os.getpid(), "devices": sample_local(sample_environment)}
                temporary = target.with_suffix(".tmp")
                temporary.write_text(json.dumps(data))
                temporary.replace(target)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
                # Keep the previous timestamp; readers flag it as stale.
                pass
            time.sleep(15)


def run(command, *, input=None):
    result = subprocess.run(command, input=input, stdin=subprocess.DEVNULL if input is None else None,
                            capture_output=True, text=True, timeout=12)
    if result.returncode:
        raise RuntimeError((result.stderr.strip() or "GPU 조회 실패")[-400:])
    return result.stdout


def indices(value):
    result = set()
    for part in value.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            raise ValueError("할당된 GPU 번호를 확인할 수 없습니다.")
        low, _, high = part.partition("-")
        low, high = int(low), int(high or low)
        if not 0 <= low <= high <= 1024:
            raise ValueError("GPU 번호 범위가 올바르지 않습니다.")
        result.update(range(low, high + 1))
    return result


def allocation(text, user, submitted_at):
    owner = re.search(r"\bUserId=([^\s(]+)", text)
    submit = re.search(r"\bSubmitTime=(\S+)", text)
    if not owner or owner[1] != user or not submit or submit[1] != submitted_at:
        raise ValueError("GPU 할당 소유자 또는 제출 시각이 변경되었습니다.")
    if not re.search(r"\bJobState=RUNNING\b", text):
        return {}
    nodes = {}
    for match in re.finditer(r"\bNodes=(\S+).*?\bGRES=(\S+)", text):
        selected = re.findall(r"gpu[^\s()]*\(IDX:([\d,-]+)\)", match[2])
        if selected:
            nodes[match[1]] = set().union(*(indices(value) for value in selected))
    if not nodes:
        raise ValueError("Slurm에서 할당된 GPU 번호를 제공하지 않습니다.")
    return nodes


def parse_gpu(text):
    rows = []
    for cells in csv.reader(io.StringIO(text), skipinitialspace=True):
        if len(cells) != 6:
            raise ValueError("GPU 메모리 응답 형식이 올바르지 않습니다.")
        index, uuid, name, used, free, total = (cell.strip() for cell in cells)
        values = [float(used), float(free), float(total)]
        if not all(math.isfinite(value) and value >= 0 for value in values) or values[2] <= 0:
            raise ValueError("GPU 메모리 값을 확인할 수 없습니다.")
        rows.append(dict(index=int(index), uuid=uuid, name=name,
                         used_gib=values[0] / 1024, free_gib=values[1] / 1024,
                         total_gib=values[2] / 1024))
    if not rows:
        raise ValueError("GPU 조회 결과가 없습니다.")
    return rows


class GPUs:
    def __init__(self, user, runner=run, telemetry=TELEMETRY):
        self.user, self.run = user, runner
        self.telemetry = Path(telemetry)
        self.lock = threading.Lock()
        self.value = {"devices": [], "errors": [], "updated_at": None, "interval": 15}

    def snapshot(self):
        with self.lock:
            return {**self.value, "devices": [dict(row) for row in self.value["devices"]]}

    def read_node(self, node, job_ids):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", node):
            raise ValueError("노드 이름이 올바르지 않습니다.")
        if node in {socket.gethostname(), socket.gethostname().split(".")[0]}:
            return parse_gpu(self.run(QUERY))
        for identifier in job_ids:
            try:
                path = self.telemetry / identifier / (node + ".json")
                data = json.loads(path.read_text())
                if (path.stat().st_uid == os.getuid() and data["uid"] == os.getuid()
                        and data["job_id"] == identifier and data["node"] == node
                        and 0 <= time.time() - data["sampled_at"] < 45):
                    return [{**row, "sampled_at": data["sampled_at"]} for row in data["devices"]]
            except (OSError, ValueError, KeyError, TypeError):
                continue
        raise ValueError("할당 내부 GPU 수집기 응답 대기 중 (SSH 원격 조회는 사용하지 않음)")

    def poll(self, jobs):
        targets, errors = {}, []
        for job in jobs:
            if job.get("terminal") or job.get("state") != "RUNNING":
                continue
            identifier = job["job_id"]
            if not re.fullmatch(r"\d+(?:_\d+)?(?:\+\d+)?", identifier):
                continue
            try:
                detail = self.run(["scontrol", "show", "job", "-d", "-o", identifier])
                for expression, selected in allocation(detail, self.user, job.get("submitted_at")).items():
                    nodes = self.run(["scontrol", "show", "hostnames", expression]).split()
                    for node in nodes:
                        for index in selected:
                            targets.setdefault(node, {}).setdefault(index, []).append(identifier)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                errors.append(f"#{identifier}: {error}")
        devices = []
        previous = {(row["node"], row["index"]): row for row in self.snapshot()["devices"]}
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = {node: pool.submit(self.read_node, node,
                sorted({identifier for ids in targets[node].values() for identifier in ids})) for node in targets}
            for node, future in pending.items():
                try:
                    rows = {row["index"]: row for row in future.result()}
                    selected_rows = []
                    for index, job_ids in targets[node].items():
                        if index not in rows:
                            raise ValueError(f"할당된 GPU {index}를 조회할 수 없습니다.")
                        selected_rows.append({**rows[index], "node": node, "job_ids": job_ids,
                            "updated_at": datetime.fromtimestamp(rows[index].get("sampled_at", time.time()), timezone.utc).isoformat(), "stale": False})
                    devices.extend(selected_rows)
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                    errors.append(f"{node}: {error}")
                    for index, job_ids in targets[node].items():
                        if (node, index) in previous:
                            devices.append({**previous[node, index], "job_ids": job_ids, "stale": True})
        with self.lock:
            self.value = {"devices": sorted(devices, key=lambda row: (row["node"], row["index"])),
                          "errors": errors, "updated_at": datetime.now(timezone.utc).isoformat(), "interval": 15}


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["--collect"]:
        raise SystemExit("usage: gpu.py --collect (inside a Slurm allocation)")
    collect()
