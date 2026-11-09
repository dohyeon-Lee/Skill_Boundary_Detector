"""RunPod Desk: account Pods, SSH job snapshots, durable eval charts and explicit stop."""
from __future__ import annotations

import base64
import copy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import update_ssh_config as api

HERE = Path(__file__).resolve().parent
BASE = "https://api.runpod.io/v2"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (ValueError, TypeError):
        return None


def timestamp(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None


class RunPodAPI:
    def pods(self):
        return api.fetch_pods(api.load_key(api.KEY_FILE))

    def pod(self, identifier):
        return api.fetch_pod(api.load_key(api.KEY_FILE), identifier)

    def billing(self, pod):
        created = timestamp(pod.get("created_at"))
        start = (created or datetime.now(timezone.utc) - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        query = urllib.parse.urlencode({"podId": pod["id"], "bucketSize": "day",
            "startTime": start.isoformat(), "endTime": end.isoformat()})
        data = api._api_json(api.load_key(api.KEY_FILE), BASE + "/billing/pods?" + query)
        records = data.get("records")
        if not isinstance(records, list) or any(row.get("podId") != pod["id"] for row in records):
            raise ValueError("Pod별 비용 응답 형식이 예상과 다릅니다.")
        totals = {}
        for field in ("gpuAmount", "cpuAmount", "diskAmount", "totalAmount"):
            values = [number(row.get(field)) for row in records]
            if any(value is None for value in values):
                raise ValueError("Pod별 비용 값이 올바르지 않습니다.")
            totals[field] = sum(values)
        return {**totals, "updated_at": now(), "since": start.isoformat(), "lifetime": created is not None,
                "currency": "USD"}

    def stop(self, identifier):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Pod ID가 올바르지 않습니다.")
        request = urllib.request.Request(BASE + "/pods/" + identifier + "/action",
            data=b'{"action":"stop"}', method="POST", headers={"Content-Type": "application/json",
            "Authorization": "Bearer " + api.load_key(api.KEY_FILE)})
        try:
            with api.api_opener().open(request, timeout=30) as response:
                response.read(65536)
        except urllib.error.HTTPError as error:
            raise ValueError(f"Pod 중단 요청 실패 (HTTP {error.code}). RunPod 권한·Pod 상태를 확인하세요.") from None
        except OSError:
            raise ValueError("Pod 중단 요청의 응답을 확인하지 못했습니다. 새로고침으로 실제 상태를 확인하세요.") from None


def ssh_snapshot(pod, config, known, folder):
    host, port = api.endpoint(pod)
    encoded = base64.b64encode(json.dumps({"root": config["project_root"], "known": known}).encode()).decode()
    identity = str(Path(config["identity"]).expanduser())
    command = ["ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
        "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={folder / 'known_hosts'}",
        "-i", identity, "-p", str(port), "root@" + host, "python3 - " + shlex.quote(encoded)]
    # Raw stderr can contain remote shell startup output; never expose it or
    # environment/commands through the browser. Only the bounded probe JSON is used.
    with (HERE / "desk_probe.py").open("rb") as source:
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(command, stdin=source, stdout=output, stderr=subprocess.DEVNULL, timeout=35)
            if result.returncode:
                raise ValueError("SSH 조회 실패: Pod의 22/tcp·공개키·프로젝트 경로를 확인하세요.")
            if output.tell() > 32 * 1024 * 1024:
                raise ValueError("Pod 조회 결과가 너무 큽니다.")
            output.seek(0)
            data = json.load(output)
    if not isinstance(data, dict) or data.get("error"):
        raise ValueError("Pod 프로젝트 조회에 실패했습니다. 프로젝트 경로를 확인하세요.")
    return data


class Monitor:
    def __init__(self, project, state_dir=None, *, client=None, probe=None):
        self.project = Path(project)
        self.folder = Path(state_dir or self.project / ".cache/runpod_desk")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.owner = (self.folder / "owner.lock").open("a")
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise ValueError("다른 RunPod Desk가 같은 기록을 사용 중입니다.") from None
        self.client, self.probe = client or RunPodAPI(), probe or ssh_snapshot
        self.config = {"interval": 30, "project_root": "/workspace/Skill_Boundary_Detector",
                       "identity": str(Path.home() / ".ssh/id_ed25519")}
        self.pods, self.results, self.updated_at, self.error = {}, {}, None, ""
        try:
            data = json.loads((self.folder / "state.json").read_text())
            self.config.update(data.get("config", {}))
            self.pods, self.results = data.get("pods", {}), data.get("results", {})
            for pod in self.pods.values():
                pod["stale"] = True
        except (OSError, ValueError):
            pass
        self.lock, self.poll_lock = threading.RLock(), threading.Lock()
        self.refresh, self.stopping = threading.Event(), threading.Event()
        self.thread = None

    def save(self):
        api.atomic_write(self.folder / "state.json", json.dumps({"config": self.config,
            "pods": self.pods, "results": self.results}, ensure_ascii=False, allow_nan=False).encode())

    def snapshot(self):
        with self.lock:
            return copy.deepcopy({"pods": list(self.pods.values()), "results": list(self.results.values()),
                "config": self.config, "updated_at": self.updated_at, "error": self.error,
                "polling": self.thread is not None and self.thread.is_alive()})

    def configure(self, values):
        if not isinstance(values, dict) or set(values) - set(self.config):
            raise ValueError("설정 항목이 올바르지 않습니다.")
        candidate = {**self.config, **values}
        if type(candidate["interval"]) is not int or not 10 <= candidate["interval"] <= 3600:
            raise ValueError("조회 주기는 10~3600초입니다.")
        for field in ("project_root", "identity"):
            value = candidate[field]
            if not isinstance(value, str) or not value or any(c in value for c in "\r\n\0"):
                raise ValueError("프로젝트·SSH 키 경로를 확인하세요.")
        if not candidate["project_root"].startswith("/"):
            raise ValueError("Pod 프로젝트 경로는 절대 경로로 지정하세요.")
        with self.lock:
            self.config = candidate
            self.save()
        self.refresh.set()

    def collect(self, raw, row):
        identifier = row["id"]
        with self.lock:
            known = {result["remote_id"]: result["fingerprint"] for result in self.results.values()
                     if result["pod_id"] == identifier}
            config = dict(self.config)
        data = self.probe(raw, config, known, self.folder)
        jobs = data.get("jobs", [])
        if not isinstance(jobs, list) or not isinstance(data.get("results", []), list):
            raise ValueError("Pod 조회 결과 형식이 올바르지 않습니다.")
        with self.lock:
            # Prefix remote IDs with the immutable Pod ID to avoid cross-Pod collisions.
            row.update(jobs=jobs, gpus=data.get("gpus", []), ssh_error="",
                       warnings=data.get("warnings", []), collected_at=now())
            for job in jobs:
                if job.get("result_id"):
                    job["result_key"] = identifier + ":" + job["result_id"]
            for result in data.get("results", []):
                remote_id = result["id"]
                if not re.fullmatch(r"[0-9a-f]{24}", remote_id):
                    raise ValueError("결과 ID가 올바르지 않습니다.")
                key = identifier + ":" + remote_id
                chart = result.pop("chart", None)
                image_id = hashlib.sha256(key.encode()).hexdigest()
                image_path = self.folder / "charts" / (image_id + ".png")
                if chart:
                    payload = base64.b64decode(chart, validate=True)
                    if len(payload) > 6 * 1024 * 1024 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise ValueError("평가 그래프 형식이 올바르지 않습니다.")
                    api.atomic_write(image_path, payload)
                elif image_path.exists():
                    image_path.unlink()
                self.results[key] = {**result, "key": key, "remote_id": remote_id, "pod_id": identifier,
                    "pod_name": row["name"], "collected_at": now(),
                    "chart_url": "/runpod/api/chart/" + image_id if chart else None}

    def poll(self):
        with self.poll_lock:
            try:
                raw_pods = self.client.pods()
                seen = set()
                for raw in raw_pods:
                    if self.stopping.is_set():
                        break
                    identifier = raw.get("id", "")
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
                        continue
                    seen.add(identifier)
                    with self.lock:
                        row = self.pods.setdefault(identifier, {"id": identifier, "jobs": [], "first_seen": now()})
                    status = raw.get("status", "UNKNOWN")
                    if status == "RUNNING":
                        # The list endpoint can omit direct SSH/runtime details.
                        try:
                            raw = self.client.pod(identifier)
                        except (ValueError, OSError):
                            pass
                    if self.stopping.is_set():
                        return
                    gpu = raw.get("gpu") or {}
                    rate = number(raw.get("cost"))
                    uptime = number((raw.get("runtime") or {}).get("uptime"))
                    with self.lock:
                        row.update(name=raw.get("name", identifier), status=raw.get("status", status), present=True,
                            stale=False, gpu_name=gpu.get("name", gpu.get("id", "GPU 정보 대기")),
                            gpu_count=gpu.get("count", 0), hourly_cost=rate,
                            created_at=raw.get("createdAt") or row.get("created_at"),
                            session_estimate=(rate * uptime / 3600) if rate is not None and uptime is not None else None)
                        requested = timestamp(row.get("stop_requested_at"))
                        if row["status"] == "RUNNING" and requested and (datetime.now(timezone.utc) - requested).total_seconds() < 300:
                            row["status"] = "STOP_REQUESTED"
                    if status == "RUNNING":
                        try:
                            self.collect(raw, row)
                        except (ValueError, OSError, subprocess.SubprocessError, KeyError) as error:
                            with self.lock:
                                row["ssh_error"] = str(error) if isinstance(error, ValueError) else "SSH 조회에 실패했습니다. 이전 기록을 표시합니다."
                    self.update_billing(row)
                if self.stopping.is_set():
                    return
                with self.lock:
                    for identifier, row in self.pods.items():
                        if identifier not in seen:
                            row.update(present=False, status="DELETED", stale=False, hourly_cost=0)
                        if row.get("status") in {"EXITED", "TERMINATED", "DELETED"}:
                            for job in row.get("jobs", []):
                                if job.get("state") in {"running", "pending"}:
                                    job["state"] = "pod_stopped"
                for identifier, row in list(self.pods.items()):
                    if identifier not in seen:
                        self.update_billing(row)
                with self.lock:
                    self.error, self.updated_at = "", now()
                    self.save()
            except (ValueError, OSError, KeyError, TypeError) as error:
                with self.lock:
                    self.error = str(error) if isinstance(error, ValueError) else "RunPod 조회에 실패했습니다. 이전 기록을 표시합니다."
                    for row in self.pods.values():
                        row["stale"] = True

    def update_billing(self, row):
        if self.stopping.is_set() or time.time() - row.get("billing_checked", 0) < 300:
            return
        try:
            billing = self.client.billing(row)
            with self.lock:
                row.update(billing=billing, billing_error="", billing_checked=time.time())
        except (ValueError, OSError):
            with self.lock:
                row.update(billing_error="비용 조회 실패: API 키의 billing 권한을 확인하세요.", billing_checked=time.time())

    def stop_pod(self, identifier, confirmation):
        if not isinstance(identifier, str) or confirmation != identifier or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("중단할 Pod 확인이 필요합니다.")
        with self.poll_lock:
            with self.lock:
                row = self.pods.get(identifier)
                if not row or not row.get("present"):
                    raise ValueError("현재 조회된 Pod만 중단할 수 있습니다. 새로고침하세요.")
            raw = self.client.pod(identifier)
            if raw.get("status") != "RUNNING":
                raise ValueError("Pod가 실행 중이 아닙니다. 새로고침하세요.")
            # One last best-effort backup before stopping. Collection failure
            # must not prevent the user from stopping compute charges.
            warning = ""
            try:
                self.collect(raw, row)
                with self.lock:
                    self.save()
            except (ValueError, OSError, subprocess.SubprocessError, KeyError):
                warning = "중단 직전 결과 수집에 실패했습니다. 이전에 보관된 결과만 유지됩니다."
            self.client.stop(identifier)
            with self.lock:
                row.update(stop_requested_at=now(), status="STOP_REQUESTED")
                self.save()
        self.refresh.set()
        return {"ok": True, "warning": warning, "message": "중단을 요청했습니다. 실제 종료 상태는 새로고침으로 확인합니다."}

    def start(self):
        if self.thread is not None:
            return
        def loop():
            try:
                while not self.stopping.is_set():
                    self.refresh.clear()
                    self.poll()
                    self.refresh.wait(self.config["interval"])
            finally:
                if self.stopping.is_set():
                    self.owner.close()
        self.thread = threading.Thread(target=loop, name="runpod-desk", daemon=True)
        self.thread.start()

    def close(self):
        self.stopping.set()
        self.refresh.set()
        if self.thread:
            self.thread.join(timeout=40)
        if not self.thread or not self.thread.is_alive():
            self.owner.close()
