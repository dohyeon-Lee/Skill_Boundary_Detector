#!/usr/bin/env python3
"""One loopback UI for server transfers and Hugging Face operations."""
from __future__ import annotations

import argparse
import errno
import json
import os
import secrets
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from hf_server import HFBackend, HFHandler, HFTransfers, SYNC
from server import Backend, Handler, HERE, REPO, Transfers, configured_endpoints
from storage import storage_snapshot


class DeskHandler(HFHandler):
    def _reply(self, status, body, content_type="application/json; charset=utf-8"):
        if content_type.startswith("text/html") and isinstance(body, bytes):
            html = body.decode("utf-8")
            hf = getattr(self, "_hf_request", False)
            if hf:
                html = html.replace('src="/app.js"', 'src="/hf/app.js"')
                html = html.replace('href="/hf.css"', 'href="/hf/hf.css"')
            tabs = '<nav class="desk-tabs" aria-label="동기화 기능">'
            for path, label, selected in (("/", "서버 간 전송", not hf), ("/hf/", "Hugging Face", hf)):
                current = ' aria-current="page"' if selected else ""
                tabs += f'<a href="{path}"{current}>{label}</a>'
            tabs += '</nav>'
            storage = ('<section class="storage-panel" aria-label="실행 서버 디스크 용량">'
                       '<div class="storage-heading"><strong id="storage-host">서버 용량</strong>'
                       '<button type="button" id="storage-refresh" class="button quiet">새로고침</button></div>'
                       '<div id="storage-volumes" class="storage-volumes" aria-live="polite">용량 확인 중…</div></section>')
            html = html.replace('</head>', '<link rel="stylesheet" href="/desk.css">'
                                '<script src="/storage.js" defer></script></head>')
            body = html.replace('</header>', '</header>' + tabs + storage, 1).encode("utf-8")
        if isinstance(body, dict) and not getattr(self, "_hf_request", False) and "endpoints" in body and "token" in body:
            body = {**body, "ui": "unified", "hf_url": "/hf/"}
        super()._reply(status, body, content_type)

    def _hf(self, method):
        original_server, original_path = self.server, self.path
        self._hf_request = True
        try:
            self.server = original_server.hf_service()
            self.path = original_path[3:] or "/"
            getattr(HFHandler, method)(self)
        finally:
            self.server, self.path = original_server, original_path
            self._hf_request = False

    def _proxy(self, method):
        """Keep an already running server's transfers alive during UI migration."""
        base = f"http://127.0.0.1:{self.server.reuse_server_port}"
        headers = {}
        data = None
        if method == "POST":
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 65536:
                raise ValueError("Request body must be 1~65536 bytes")
            data = self.rfile.read(length)
            with urlopen(base + "/api/config", timeout=10) as response:
                headers["X-Sync-Token"] = json.load(response)["token"]
            headers["Content-Type"] = "application/json"
        try:
            response = urlopen(Request(base + self.path, data=data, headers=headers, method=method), timeout=45)
        except HTTPError as error:
            response = error
        with response:
            payload = json.load(response)
            if method == "GET" and urlsplit(self.path).path == "/api/config":
                payload["token"] = self.server.token
            self._reply(response.code, payload)

    def do_GET(self):
        try:
            self._guard()
            path = urlsplit(self.path).path
            if path == "/hf" or path.startswith("/hf/"):
                return self._hf("do_GET")
            if path == "/desk.css":
                return self._reply(200, (HERE / "static/desk.css").read_bytes(), "text/css; charset=utf-8")
            if path == "/storage.js":
                return self._reply(200, (HERE / "static/storage.js").read_bytes(), "text/javascript; charset=utf-8")
            if path == "/api/storage":
                locations = [(e.name, e.base) for e in self.server.backend.endpoints.values() if not e.host]
                locations += [("프로젝트", REPO), ("시스템", "/")]
                volume = SYNC["load_config"]().get("storage_volume")
                if volume:
                    locations.insert(0, ("데이터 볼륨", volume))
                return self._reply(200, storage_snapshot(locations))
            if self.server.reuse_server_port and path in {"/api/config", "/api/browse", "/api/jobs"}:
                return self._proxy("GET")
            return Handler.do_GET(self)
        except (ValueError, OSError) as error:
            self._error(error)

    def do_POST(self):
        try:
            self._guard(mutation=True)
            path = urlsplit(self.path).path
            if path.startswith("/hf/"):
                return self._hf("do_POST")
            if self.server.reuse_server_port and path in {"/api/plan", "/api/transfers", "/api/cancel"}:
                return self._proxy("POST")
            return Handler.do_POST(self)
        except (ValueError, OSError) as error:
            self._error(error)


class DeskHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, backend, manager, mode, *, hf_factory, concurrency=2, reuse_server_port=None):
        self.backend, self.manager, self.mode = backend, manager, mode
        self.token = secrets.token_urlsafe(32)
        self.reuse_server_port = reuse_server_port
        self._hf_factory, self._hf_lock, self._hf_service = hf_factory, threading.Lock(), None
        self._hf_concurrency = concurrency
        super().__init__(address, DeskHandler)

    def hf_service(self):
        with self._hf_lock:
            if self._hf_service is None:
                backend = self._hf_factory()
                self._hf_service = SimpleNamespace(backend=backend, token=self.token,
                    manager=HFTransfers(backend, self._hf_concurrency), mode=self.mode)
            return self._hf_service

    def close_managers(self):
        if self.manager is not None:
            self.manager.close()
        if self._hf_service is not None:
            self._hf_service.manager.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("SYNC_UI_PORT", "8765")))
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=int(os.environ.get("SYNC_UI_CONCURRENCY", "2")))
    parser.add_argument("--server", default=None, help="HF server profile override")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--checkpoint-repo", default=None)
    parser.add_argument("--reuse-server-port", type=int, help="Reuse an existing loopback transfer service without interrupting its jobs")
    args = parser.parse_args()
    streams = int(os.environ.get("SYNC_JOBS", "8"))
    if not 1 <= args.port <= 65535 or not 1 <= streams <= 32 or not 1 <= args.concurrency <= 4:
        parser.error("port=1~65535, streams=1~32, concurrency=1~4 required")
    if args.reuse_server_port is not None and (not 1 <= args.reuse_server_port <= 65535 or args.reuse_server_port == args.port):
        parser.error("--reuse-server-port must select a different valid loopback port")
    mode, endpoints = configured_endpoints()
    backend = Backend(endpoints, streams)
    manager = None if args.reuse_server_port else Transfers(backend, args.concurrency)
    def factory():
        return HFBackend(SYNC["load_config"](args.server), server=args.server,
                         repo=args.repo, checkpoint_repo=args.checkpoint_repo)
    try:
        server = DeskHTTPServer(("127.0.0.1", args.port), backend, manager, mode,
            hf_factory=factory, concurrency=args.concurrency, reuse_server_port=args.reuse_server_port)
    except OSError as error:
        if manager is not None:
            manager.close()
        if error.errno == errno.EADDRINUSE:
            parser.exit(2, f"포트 {args.port}가 사용 중입니다. 실행 중인 UI를 열거나 --port로 다른 포트를 지정하세요.\n")
        raise
    print(f"SBD Transfer Desk · 서버 전송 + Hugging Face · {mode}", flush=True)
    print(f"서버 전송: http://127.0.0.1:{args.port}/", flush=True)
    print(f"Hugging Face: http://127.0.0.1:{args.port}/hf/", flush=True)
    if args.reuse_server_port:
        print(f"기존 전송 서비스 {args.reuse_server_port}의 작업을 유지합니다.", flush=True)
    print("VSCode Ports에서 위 포트 하나를 전달하세요. Ctrl+C: 이 UI의 실행/대기 작업도 취소됩니다.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.close_managers()


if __name__ == "__main__":
    main()
