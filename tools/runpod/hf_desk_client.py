"""Enqueue eval publications in the existing Hugging Face Desk, never bypass it."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def local_url(value):
    parsed = urllib.parse.urlsplit(value.rstrip("/"))
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/hf"}):
        raise ValueError("Desk 주소는 http://127.0.0.1:<포트> 또는 그 뒤 /hf 만 지원합니다.")
    return value.rstrip("/")


class HFDesk:
    def __init__(self, url, project, repo, *, opener=None):
        self.url = local_url(url)
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.config = self.request("/api/config")
        endpoints = {item["id"]: item for item in self.config.get("endpoints", [])}
        if Path(endpoints["local-checkpoints"]["base"]).resolve() != Path(project).resolve():
            raise ValueError("이 Desk는 다른 프로젝트에서 실행 중입니다.")
        expected = "https://huggingface.co/datasets/" + repo if repo else None
        if expected and endpoints["hub-datasets"]["base"].rstrip("/") != expected:
            raise ValueError("Desk의 비공개 데이터 저장소가 현재 평가 설정과 다릅니다.")
        self.base = Path(endpoints["local-datasets"]["base"])
        self.checkpoint_repo = endpoints.get("hub-checkpoints", {}).get("base", "").removeprefix("https://huggingface.co/")
        self.token = self.config["token"]

    def request(self, path, body=None):
        headers = {}
        if body is not None:
            headers = {"Content-Type": "application/json", "X-Sync-Token": self.token}
        req = urllib.request.Request(self.url + path, headers=headers,
            data=None if body is None else json.dumps(body).encode())
        with self.opener.open(req, timeout=30) as response:
            return json.load(response)

    def jobs(self):
        return self.request("/api/jobs")["jobs"]

    def publish(self, path, *, checkpoint=False, eval_results=False, runpod_id=None):
        if checkpoint and eval_results:
            raise ValueError("체크포인트와 평가 결과 업로드를 동시에 지정할 수 없습니다.")
        if runpod_id and not eval_results:
            raise ValueError("Pod ID는 평가 결과 업로드에서만 지정할 수 있습니다.")
        if eval_results and not self.config.get("eval_results_upload"):
            raise ValueError("평가 결과 업로드를 지원하는 최신 HF Desk가 필요합니다.")
        if checkpoint and not self.config.get("checkpoint_pretrained_only"):
            raise ValueError("HF Desk를 최신 코드로 재시작하세요. 이 버전은 평가 모델만 업로드하는 옵션을 지원하지 않습니다.")
        route = "eval-push" if eval_results else "checkpoints-push" if checkpoint else "dataset-push"
        active = {"queued", "running", "cancelling"}
        matches = [job for job in self.jobs() if job.get("route") == route
                   and job.get("path") == path and job.get("status") in active
                   and (not runpod_id or job.get("options", {}).get("runpod_id") == runpod_id)]
        if matches:
            identifiers = [matches[0]["id"]]
        else:
            body = {"source": "local-eval-results" if eval_results else "local-checkpoints" if checkpoint else "local-datasets",
                    "destination": "hub-eval-results" if eval_results else "hub-checkpoints" if checkpoint else "hub-datasets", "paths": [path],
                    "operation": route, "dry_run": False}
            if checkpoint:
                body["options"] = {"pretrained_only": True}
            if runpod_id:
                body["options"] = {"runpod_id": runpod_id}
            try:
                identifiers = self.request("/api/transfers", body)["ids"]
            except (urllib.error.URLError, TimeoutError):
                # A response can be lost after the Desk queued the upload. Do not duplicate it.
                matches = [job for job in self.jobs() if job.get("route") == route
                           and job.get("path") == path and job.get("status") in active | {"completed"}
                           and (not runpod_id or job.get("options", {}).get("runpod_id") == runpod_id)]
                if not matches:
                    raise ValueError("Desk 작업 접수를 확인하지 못했습니다. Desk 작업 목록을 확인하세요.") from None
                identifiers = [matches[0]["id"]]
        return self.wait(identifiers)

    def wait_pending(self, prefix):
        identifiers = [job["id"] for job in self.jobs()
                       if job.get("route") == "dataset-push"
                       and job.get("path", "").startswith(prefix + "/")
                       and job.get("status") in {"queued", "running", "cancelling"}]
        if identifiers:
            print("이전에 접수된 평가 업로드가 끝나기를 기다립니다.", flush=True)
            self.wait(identifiers)

    def wait(self, identifiers):
        print(f"HF Desk 작업: {', '.join(identifiers)} · {self.url}/", flush=True)
        seen = {}
        heartbeat = time.monotonic()
        while True:
            jobs = {job["id"]: job for job in self.jobs()}
            if any(identifier not in jobs for identifier in identifiers):
                raise ValueError("Desk 작업 이력이 없어졌습니다. Desk를 확인하고 다시 실행하세요.")
            for identifier in identifiers:
                job = jobs[identifier]
                state = (job["status"], job.get("progress"))
                if seen.get(identifier) != state:
                    print(f"  {job['status']} · {job.get('progress', 0)}% · {job.get('path', '')}", flush=True)
                    seen[identifier] = state
                if job["status"] in {"failed", "cancelled"}:
                    raise ValueError("HF Desk 업로드가 완료되지 않았습니다. Desk의 상세 로그를 확인하세요.")
            if all(jobs[identifier]["status"] == "completed" for identifier in identifiers):
                return identifiers
            if time.monotonic() - heartbeat > 30:
                print("  HF Desk 업로드 진행 중...", flush=True)
                heartbeat = time.monotonic()
            time.sleep(2)


def connect(project, repo, url=None):
    project = Path(project)
    cache = project / ".cache/eval_hf"
    cached = cache / "desk.json"
    candidates = [url] if url else ["http://127.0.0.1:8765/hf", "http://127.0.0.1:8766"]
    if not url and cached.exists():
        try:
            candidates.insert(0, json.loads(cached.read_text())["url"])
        except (ValueError, KeyError):
            pass
    for candidate in dict.fromkeys(candidates):
        try:
            return HFDesk(candidate, project, repo)
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            if url:
                raise ValueError("지정한 HF Desk에 접속하지 못했습니다. 주소·프로젝트·저장소를 확인하세요.") from None
    cache.mkdir(parents=True, exist_ok=True)
    for port in range(8766, 8776):
        base = f"http://127.0.0.1:{port}"
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(base + "/api/config", timeout=1):
                continue
        except urllib.error.HTTPError:
            continue
        except OSError:
            pass
        # A standalone HF Desk avoids restarting an existing UI or Slurm monitor.
        log = (cache / f"desk-{port}.log").open("ab")
        os.chmod(log.name, 0o600)
        env = os.environ.copy()
        env.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        process = subprocess.Popen([sys.executable, str(project / "tools/server_sync/hf_server.py"),
            "--port", str(port), "--repo", repo], cwd=project, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        for _ in range(60):
            if process.poll() is not None:
                break
            try:
                desk = HFDesk(base, project, repo)
                cached.write_text(json.dumps({"url": base, "pid": process.pid}) + "\n")
                print(f"HF Desk 시작: {base}/ · VS Code에서 {port} 포트를 전달하면 작업을 볼 수 있습니다.", flush=True)
                return desk
            except (OSError, ValueError, KeyError):
                time.sleep(0.25)
        if process.poll() is None:
            process.terminate()
    raise ValueError("HF Desk를 시작하지 못했습니다. bash sync_server_ui.sh 실행 후 다시 시도하세요.")
