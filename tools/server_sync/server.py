#!/usr/bin/env python3
"""Loopback-only, dependency-free UI for the existing SSH/rsync sync workflow."""

from __future__ import annotations

import argparse
import codecs
import hmac
import json
import os
import queue
import re
import secrets
import shlex
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fs_helper import probe, relative_path
from transport import Endpoint, SSH_OPTIONS, configured_endpoints, rsync_options

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ACTIVE = {"queued", "running", "cancelling"}


def overlaps(first, second):
    return first == second or first.startswith(second.rstrip("/") + "/") or second.startswith(first.rstrip("/") + "/")


class Backend:
    def __init__(self, endpoints, streams=8):
        self.endpoints = {item.id: item for item in endpoints}
        self.streams = streams
        self.helper = (HERE / "fs_helper.py").read_text(encoding="utf-8")

    def endpoint(self, identifier):
        if not isinstance(identifier, str) or identifier not in self.endpoints:
            raise ValueError("Unknown endpoint")
        return self.endpoints[identifier]

    def inspect(self, endpoint, path, operation="stat", offset=0, search=""):
        if not endpoint.host:
            return probe(endpoint.base, path, operation, offset, 200, search)
        args = json.dumps([endpoint.base, path, operation, offset, 200, search], ensure_ascii=False)
        command = "python3 -c " + shlex.quote(self.helper) + " " + shlex.quote(args)
        try:
            result = subprocess.run(["ssh", "-T", *SSH_OPTIONS, endpoint.host, command],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired as error:
            raise ValueError("SSH 응답 시간 초과. 외부 서버에서 연결 상태를 확인하세요.") from error
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        if result.returncode or "error" in payload:
            detail = payload.get("error") or result.stderr.strip()[-1500:] or "SSH probe failed"
            if "Permission denied" in detail or "Host key verification failed" in detail:
                detail += "\n터미널에서 ssh " + endpoint.host + " 로 연결/호스트 키를 확인하고 SSH 키 인증을 설정하세요."
            raise ValueError(detail)
        if not payload:
            raise ValueError(result.stderr.strip()[-1500:] or "SSH 응답을 읽을 수 없습니다. 원격 python3 설치/로그인 메시지를 확인하세요.")
        return payload

    def browse(self, identifier, path="", offset=0, search=""):
        return self.inspect(self.endpoint(identifier), path, "list", offset, search)

    def plan(self, source_id, destination_id, paths):
        source, destination = self.endpoint(source_id), self.endpoint(destination_id)
        if source.id == destination.id:
            raise ValueError("출발과 도착 위치가 같습니다.")
        if source.host and destination.host and source.host != destination.host:
            raise ValueError("서로 다른 원격 호스트 간 전송은 지원하지 않습니다.")
        if not isinstance(paths, list) or not 1 <= len(paths) <= 200:
            raise ValueError("1~200개 항목을 선택하세요.")
        normalized = sorted(set(relative_path(path) for path in paths), key=lambda path: (path.count("/"), path))
        unique = []
        for path in normalized:
            if not any(path.startswith(parent + "/") for parent in unique):
                unique.append(path)
        plans = []
        for path in unique:
            origin = self.inspect(source, path)
            target = self.inspect(destination, path)
            if not origin["exists"]:
                raise ValueError("출발 항목이 없습니다: " + path)
            if origin["kind"] not in ("file", "directory"):
                raise ValueError("파일/폴더만 전송할 수 있습니다: " + path)
            if target["exists"] and target["kind"] != origin["kind"]:
                raise ValueError("목적지 파일/폴더 유형이 다릅니다: " + path)
            if source.host == destination.host and overlaps(origin["real_path"], target["real_path"]):
                raise ValueError("출발/도착 경로가 겹칩니다: " + path)
            plans.append({"source_id": source_id, "destination_id": destination_id,
                "path": path, "kind": origin["kind"], "source": origin["real_path"],
                "destination": target["real_path"], "source_name": source.name,
                "destination_name": destination.name, "destination_host": destination.host})
        return {"items": plans, "collapsed": len(paths) - len(plans)}

    def commands(self, plan):
        source, destination = self.endpoint(plan["source_id"]), self.endpoint(plan["destination_id"])
        path = plan["path"]
        # Revalidate at execution, not just at preview: trainers/files can change
        # while a transfer waits. Only this explicit transfer creates folders.
        origin = self.inspect(source, path)
        target = self.inspect(destination, path)
        if not origin["exists"] or origin["kind"] != plan["kind"]:
            raise ValueError("출발 항목이 변경/삭제되었습니다: " + path)
        if target["exists"] and target["kind"] != origin["kind"]:
            raise ValueError("목적지 유형이 변경되었습니다: " + path)
        if source.host == destination.host and overlaps(origin["real_path"], target["real_path"]):
            raise ValueError("전송 경로가 겹칩니다.")
        parent = path if plan["kind"] == "directory" else path.rsplit("/", 1)[0]
        self.inspect(destination, parent, "mkdir")
        src, dst = origin["real_path"], target["real_path"]
        if not source.host and not destination.host and plan["kind"] == "directory":
            # Reuse the original eight-stream local copier, not a second implementation.
            return ["bash", "-c", 'source "$1"; parallel_local_copy "$2" "$3" "$4"',
                    "sync-ui", str(REPO / "sync_server.sh"), src, dst, str(self.streams)]
        options = rsync_options()
        if plan["kind"] == "directory":
            src, dst = src + "/", dst + "/"
        if source.host and destination.host:
            # A PTY makes disconnect/cancel hang up the remote rsync process.
            command = "exec " + shlex.join(options + ["--", src, dst])
            return ["ssh", "-tt", *SSH_OPTIONS, source.host, command]
        if source.host or destination.host:
            options += ["-z", "-e", shlex.join(["ssh", "-T", *SSH_OPTIONS])]
            if source.host:
                src = source.host + ":" + src
            if destination.host:
                dst = destination.host + ":" + dst
        return options + ["--", src, dst]


class Transfers:
    def __init__(self, backend, concurrency=2):
        self.backend, self.concurrency = backend, concurrency
        self.lock = threading.RLock()
        self.pending = queue.Queue()
        self.jobs = {}
        self.closed = False
        self.workers = [threading.Thread(target=self._worker, daemon=True) for _ in range(concurrency)]
        for thread in self.workers:
            thread.start()

    def snapshot(self):
        with self.lock:
            return [{key: list(value) if key == "logs" else value for key, value in job.items()
                     if key not in ("process", "cancel")} for job in reversed(list(self.jobs.values()))]

    def submit(self, items):
        with self.lock:
            if self.closed:
                raise ValueError("전송 서버가 종료 중입니다.")
            if sum(job["status"] in ACTIVE for job in self.jobs.values()) + len(items) > 200:
                raise ValueError("대기/실행 항목은 최대 200개입니다.")
            for item in items:
                for job in self.jobs.values():
                    if job["status"] in ACTIVE and job["destination_host"] == item["destination_host"] and overlaps(
                        job["destination"], item["destination"]
                    ):
                        raise ValueError("같은 목적지에 이미 대기/전송 중인 항목이 있습니다: " + item["path"])
            ids = []
            for item in items:
                identifier = uuid.uuid4().hex
                self.jobs[identifier] = {**item, "id": identifier, "status": "queued", "progress": 0,
                    "detail": "전송 대기", "logs": deque(maxlen=120), "created": time.time(),
                    "started": None, "ended": None, "cancel": False, "process": None}
                ids.append(identifier)
                self.pending.put(identifier)
            # Bound in-memory history; do not prune active jobs.
            for identifier in list(self.jobs):
                if len(self.jobs) <= 500:
                    break
                if self.jobs[identifier]["status"] not in ACTIVE:
                    self.jobs.pop(identifier)
            return ids

    @staticmethod
    def _terminate(process):
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def cancel(self, identifier):
        with self.lock:
            if not isinstance(identifier, str) or identifier not in self.jobs:
                raise ValueError("전송 항목을 찾을 수 없습니다.")
            job = self.jobs[identifier]
            if job["status"] not in ACTIVE:
                return
            job["cancel"] = True
            job["status"] = "cancelled" if job["status"] == "queued" else "cancelling"
            if job["status"] == "cancelled":
                job["ended"] = time.time()
            process = job["process"]
        self._terminate(process)

    def _log(self, job, line):
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip()
        if not line:
            return
        with self.lock:
            job["logs"].append(line[:2000])
            job["detail"] = line[:220]
            match = re.search(r"(\d{1,3})%", line)
            if match:
                job["progress"] = min(99, int(match.group(1)))

    def _execute(self, job):
        command = self.backend.commands(job)
        with self.lock:
            if job["cancel"]:
                return
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, "LC_ALL": "C.UTF-8"})
            job["process"] = process
        decoder, buffer = codecs.getincrementaldecoder("utf-8")("replace"), ""
        try:
            while True:
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break
                buffer += decoder.decode(chunk)
                lines = re.split(r"[\r\n]", buffer)
                buffer = lines.pop()[-4000:]
                for line in lines:
                    self._log(job, line)
            self._log(job, buffer + decoder.decode(b"", final=True))
            code = process.wait()
            if code and not job["cancel"]:
                raise ValueError("rsync/SSH 종료 코드 " + str(code) + ". 상세 로그를 확인하세요.")
        finally:
            self._terminate(process)
            process.stdout.close()

    def _worker(self):
        while True:
            identifier = self.pending.get()
            try:
                if identifier is None:
                    return
                with self.lock:
                    job = self.jobs.get(identifier)
                    if job is None or job["cancel"]:
                        continue
                    job.update(status="running", started=time.time(), detail="경로 확인 및 전송 준비")
                try:
                    self._execute(job)
                    with self.lock:
                        job["status"] = "cancelled" if job["cancel"] else "completed"
                        if not job["cancel"]:
                            job.update(progress=100, detail="완료 · 기존 목적지의 추가 파일은 삭제하지 않았습니다.")
                except Exception as error:
                    with self.lock:
                        job["status"] = "cancelled" if job["cancel"] else "failed"
                        job["detail"] = str(error)
                        job["logs"].append(str(error))
                finally:
                    with self.lock:
                        job.update(ended=time.time(), process=None)
            finally:
                self.pending.task_done()

    def close(self):
        with self.lock:
            self.closed = True
            active = [key for key, job in self.jobs.items() if job["status"] in ACTIVE]
        for identifier in active:
            self.cancel(identifier)
        for _ in self.workers:
            self.pending.put(None)
        for thread in self.workers:
            thread.join(timeout=35)


class SyncHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, backend, manager, mode):
        self.backend, self.manager, self.mode = backend, manager, mode
        self.token = secrets.token_urlsafe(32)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Paths/selection/token are not printed into access logs.

    def _guard(self, mutation=False):
        host = self.headers.get("Host", "")
        if urlsplit("http://" + host).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise PermissionError("Loopback Host header required")
        origin = self.headers.get("Origin")
        if origin and (urlsplit(origin).scheme != "http" or urlsplit(origin).netloc != host):
            raise PermissionError("Cross-origin request blocked")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise PermissionError("Cross-site request blocked")
        if mutation and not hmac.compare_digest(self.headers.get("X-Sync-Token", "").encode("utf-8"), self.server.token.encode("utf-8")):
            raise PermissionError("Invalid session token; reload the page")

    def _reply(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, error):
        self._reply(403 if isinstance(error, PermissionError) else 400, {"error": str(error)})

    def do_GET(self):
        try:
            self._guard()
            request = urlsplit(self.path)
            params = parse_qs(request.query)
            if request.path == "/api/config":
                self._reply(200, {"mode": self.server.mode, "token": self.server.token,
                    "concurrency": self.server.manager.concurrency, "streams": self.server.backend.streams,
                    "endpoints": [item.__dict__ for item in self.server.backend.endpoints.values()]})
            elif request.path == "/api/browse":
                self._reply(200, self.server.backend.browse(params.get("endpoint", [""])[0],
                    params.get("path", [""])[0], int(params.get("offset", [0])[0]), params.get("search", [""])[0]))
            elif request.path == "/api/jobs":
                self._reply(200, {"jobs": self.server.manager.snapshot()})
            elif request.path in ("/", "/app.js", "/style.css"):
                file = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[request.path]
                content_type = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[file]
                self._reply(200, (HERE / "static" / file).read_bytes(), content_type + "; charset=utf-8")
            else:
                self._reply(404, {"error": "Not found"})
        except (ValueError, OSError, PermissionError) as error:
            self._error(error)

    def do_POST(self):
        try:
            self._guard(mutation=True)
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 65536:
                raise ValueError("Request body must be 1~65536 bytes")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("JSON object required")
            path = urlsplit(self.path).path
            if path in ("/api/plan", "/api/transfers"):
                plan = self.server.backend.plan(body.get("source"), body.get("destination"), body.get("paths"))
                if path == "/api/plan":
                    self._reply(200, plan)
                else:
                    self._reply(202, {"ids": self.server.manager.submit(plan["items"])})
            elif path == "/api/cancel":
                self.server.manager.cancel(body.get("id"))
                self._reply(200, {"ok": True})
            else:
                self._reply(404, {"error": "Not found"})
        except (ValueError, OSError, PermissionError) as error:
            self._error(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("SYNC_UI_PORT", "8765")))
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=int(os.environ.get("SYNC_UI_CONCURRENCY", "2")))
    args = parser.parse_args()
    streams = int(os.environ.get("SYNC_JOBS", "8"))
    if not 1 <= streams <= 32 or not 1 <= args.concurrency <= 4 or not 1 <= args.port <= 65535:
        parser.error("streams=1~32, concurrency=1~4, port=1~65535 required")
    mode, endpoints = configured_endpoints()
    backend = Backend(endpoints, streams)
    manager = Transfers(backend, args.concurrency)
    try:
        server = SyncHTTPServer(("127.0.0.1", args.port), backend, manager, mode)
    except OSError:
        manager.close()
        raise
    print("SBD Transfer Desk · " + mode, flush=True)
    print("Local URL: http://127.0.0.1:" + str(args.port), flush=True)
    print("외부 서버에서 실행 중이면 VSCode Ports에서 " + str(args.port) + "를 전달하세요.", flush=True)
    print("종료: Ctrl+C (실행/대기 전송도 취소됩니다). 브라우저를 닫는 것은 전송을 중단하지 않습니다.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close()


if __name__ == "__main__":
    main()
