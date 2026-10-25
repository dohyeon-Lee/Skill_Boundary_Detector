#!/usr/bin/env python3
"""A loopback web UI for Slurm jobs and completion prompts to Codex."""
from __future__ import annotations

import argparse
import errno
import hmac
import json
import os
import secrets
import signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from monitor import Monitor

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


class HTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, monitor):
        self.monitor = monitor
        self.token = secrets.token_urlsafe(32)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def guard(self, mutation=False):
        host = self.headers.get("Host", "")
        if urlsplit("http://" + host).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise PermissionError("localhost 주소로 접근하세요.")
        origin = self.headers.get("Origin")
        if origin and (urlsplit(origin).scheme != "http" or urlsplit(origin).netloc != host):
            raise PermissionError("다른 사이트의 요청은 허용하지 않습니다.")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise PermissionError("다른 사이트의 요청은 허용하지 않습니다.")
        if mutation and not hmac.compare_digest(self.headers.get("X-Monitor-Token", "").encode(), self.server.token.encode()):
            raise PermissionError("세션이 변경되었습니다. 페이지를 새로고침하세요.")

    def reply(self, code, value, content_type="application/json; charset=utf-8"):
        data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            self.guard()
            path = urlsplit(self.path).path
            if path == "/api/state":
                self.reply(200, {**self.server.monitor.snapshot(), "token": self.server.token})
            elif path in {"/", "/app.js", "/style.css"}:
                file, mime = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                              "/style.css": ("style.css", "text/css")}[path]
                self.reply(200, (HERE / "static" / file).read_bytes(), mime + "; charset=utf-8")
            else:
                self.reply(404, {"error": "Not found"})
        except PermissionError as error:
            self.reply(403, {"error": str(error)})
        except (ValueError, OSError) as error:
            self.reply(400, {"error": str(error)})

    def do_POST(self):
        try:
            self.guard(mutation=True)
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 160000:
                raise ValueError("요청 크기가 올바르지 않습니다.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("JSON 객체가 필요합니다.")
            path = urlsplit(self.path).path
            monitor = self.server.monitor
            if path == "/api/settings":
                monitor.configure(body)
            elif path == "/api/job":
                monitor.update_job(body.get("key"), body.get("values"))
            elif path == "/api/send":
                monitor.request_send(body.get("key"))
            elif path == "/api/cancel":
                monitor.cancel_job(body.get("key"))
            elif path == "/api/refresh":
                monitor.refresh.set()
            else:
                self.reply(404, {"error": "Not found"})
                return
            self.reply(200, {"ok": True})
        except PermissionError as error:
            self.reply(403, {"error": str(error)})
        except (ValueError, OSError, TypeError) as error:
            self.reply(400, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("SLURM_MONITOR_PORT", "8767")))
    parser.add_argument("--state-dir", type=Path, default=REPO / ".slurm_monitor")
    parser.add_argument("--thread-id", help="대상 Codex 세션 UUID. 최초 실행 시 CODEX_THREAD_ID도 자동 사용")
    parser.add_argument("--codex-home", help="대상 세션을 관리하는 CODEX_HOME")
    parser.add_argument("--codex-command", help="Codex CLI 실행 파일 경로")
    parser.add_argument("--interval", type=int, help="Slurm 확인 주기(10~3600초)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or (args.interval is not None and not 10 <= args.interval <= 3600):
        parser.error("port=1~65535, interval=10~3600초")
    defaults = {key: value for key, value in {"thread_id": args.thread_id,
                "codex_home": args.codex_home, "codex_command": args.codex_command,
                "interval": args.interval}.items() if value is not None}
    try:
        monitor = Monitor(args.state_dir, REPO, defaults=defaults)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, str(error) + "\n")
    try:
        server = HTTPServer(("127.0.0.1", args.port), monitor)
    except OSError as error:
        monitor.close()
        if error.errno == errno.EADDRINUSE:
            parser.exit(1, f"포트 {args.port}가 이미 사용 중입니다. 기존 UI를 열거나 --port로 바꾸세요.\n")
        raise
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    monitor.start()
    print(f"Slurm Monitor · http://127.0.0.1:{args.port}", flush=True)
    print(f"사용자: {monitor.config['user']} · 상태 저장: {monitor.store.path}", flush=True)
    print("VSCode Ports / SSH 터널로 포트를 전달하세요. 브라우저를 닫아도 감시는 계속됩니다.", flush=True)
    print("job별 지시를 저장하고 '종료 시 자동 전송'을 켠 job만 자동 전송합니다.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        monitor.close()


if __name__ == "__main__":
    main()
