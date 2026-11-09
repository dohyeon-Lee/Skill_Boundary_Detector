"""Read-only GPU telemetry for the current user's running allocations."""
from __future__ import annotations

import csv
import io
import math
import re
import socket
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

QUERY = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,memory.total",
         "--format=csv,noheader,nounits"]
BEGIN, END = "__SBD_GPU_BEGIN__", "__SBD_GPU_END__"


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
    def __init__(self, user, runner=run):
        self.user, self.run = user, runner
        self.lock = threading.Lock()
        self.value = {"devices": [], "errors": [], "updated_at": None, "interval": 15}

    def snapshot(self):
        with self.lock:
            return {**self.value, "devices": [dict(row) for row in self.value["devices"]]}

    def read_node(self, node):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", node):
            raise ValueError("노드 이름이 올바르지 않습니다.")
        if node in {socket.gethostname(), socket.gethostname().split(".")[0]}:
            return parse_gpu(self.run(QUERY))
        # Some compute nodes start a login shell regardless of the SSH command.
        # Feed a fixed query to that shell; never create a Slurm step.
        script = f"printf '{BEGIN}\\n'\n{' '.join(QUERY)}\nprintf '{END}\\n'\nexit\n"
        text = self.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
                         "-o", "ConnectionAttempts=1", "-o", "ClearAllForwardings=yes", node], input=script)
        if BEGIN not in text or END not in text:
            raise ValueError("GPU 응답을 확인할 수 없습니다. SSH 접근 권한을 확인하세요.")
        return parse_gpu(text.split(BEGIN, 1)[1].split(END, 1)[0].strip())

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
            pending = {node: pool.submit(self.read_node, node) for node in targets}
            for node, future in pending.items():
                try:
                    rows = {row["index"]: row for row in future.result()}
                    selected_rows = []
                    for index, job_ids in targets[node].items():
                        if index not in rows:
                            raise ValueError(f"할당된 GPU {index}를 조회할 수 없습니다.")
                        selected_rows.append({**rows[index], "node": node, "job_ids": job_ids,
                            "updated_at": datetime.now(timezone.utc).isoformat(), "stale": False})
                    devices.extend(selected_rows)
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                    errors.append(f"{node}: {error}")
                    for index, job_ids in targets[node].items():
                        if (node, index) in previous:
                            devices.append({**previous[node, index], "job_ids": job_ids, "stale": True})
        with self.lock:
            self.value = {"devices": sorted(devices, key=lambda row: (row["node"], row["index"])),
                          "errors": errors, "updated_at": datetime.now(timezone.utc).isoformat(), "interval": 15}
