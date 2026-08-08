#!/usr/bin/env python3
"""Serve one skill-eval report and persist human corrections beside it."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from datetime import UTC, datetime
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from review_common import REVIEW_SCHEMA, review_id_for_signature

MAX_REQUEST_BYTES = 10 * 1024 * 1024


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ReviewStore:
    """Validate and atomically persist overrides for exactly one report."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.index_path = self.output_dir / "index.html"
        self.manifest_path = self.output_dir / "metrics" / "manifest.json"
        self.corrections_path = (
            self.output_dir / "metrics" / "human_corrections.json"
        )
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Report HTML not found: {self.index_path}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Merged report manifest not found: {self.manifest_path}"
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.report_id = review_id_for_signature(manifest["signature"])
        self._allowed = self._allowed_corrections(manifest)
        self._lock = threading.Lock()
        # Fail at startup instead of discovering a corrupt or foreign sidecar
        # only after the reviewer has already begun working.
        self.load()

    @staticmethod
    def _allowed_corrections(manifest: dict) -> dict[str, bool]:
        allowed: dict[str, bool] = {}
        for record in manifest.get("records", {}).values():
            uid = str(record["uid"])
            for branch in record.get("branches", []):
                name = str(branch.get("name", ""))
                if (
                    not name
                    or name == "gt"
                    or branch.get("unavailable_reason") is not None
                ):
                    continue
                allowed[f"{uid}::{name}"] = bool(branch.get("green_tint", False))
        return allowed

    def empty(self) -> dict:
        return {
            "schema": REVIEW_SCHEMA,
            "report_id": self.report_id,
            "updated_at": None,
            "corrections": {},
        }

    def _normalize(self, payload: Any) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Corrections payload must be a JSON object.")
        if payload.get("schema") != REVIEW_SCHEMA:
            raise ValueError(f"Unsupported corrections schema: {payload.get('schema')!r}.")
        if str(payload.get("report_id", "")) != self.report_id:
            raise ValueError("Corrections belong to a different evaluation report.")
        raw = payload.get("corrections")
        if not isinstance(raw, dict):
            raise ValueError("corrections must be a JSON object.")
        if len(raw) > len(self._allowed):
            raise ValueError("Corrections contain more entries than this report allows.")

        payload_updated_at = str(payload.get("updated_at") or _now())
        normalized: dict[str, dict[str, object]] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or key not in self._allowed:
                raise ValueError(f"Unknown or non-reviewable branch key: {key!r}.")
            success = value if isinstance(value, bool) else (
                value.get("success") if isinstance(value, dict) else None
            )
            if not isinstance(success, bool):
                raise ValueError(f"Correction {key!r} must contain a boolean success.")
            # Returning to the model's original decision removes the override.
            if success == self._allowed[key]:
                continue
            updated_at = value.get("updated_at") if isinstance(value, dict) else None
            normalized[key] = {
                "success": success,
                "updated_at": str(updated_at or payload_updated_at),
            }
        return {
            "schema": REVIEW_SCHEMA,
            "report_id": self.report_id,
            "updated_at": payload_updated_at,
            "corrections": normalized,
        }

    def load(self) -> dict:
        with self._lock:
            if not self.corrections_path.is_file():
                return self.empty()
            payload = json.loads(self.corrections_path.read_text(encoding="utf-8"))
            return self._normalize(payload)

    def save(self, payload: Any) -> dict:
        normalized = self._normalize(payload)
        with self._lock:
            self.corrections_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.corrections_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(normalized, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self.corrections_path)
        return normalized


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    server: "ReviewHTTPServer"

    def _is_corrections_api(self) -> bool:
        return urlsplit(self.path).path.rstrip("/") == "/api/corrections"

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if not self._is_corrections_api():
            super().do_GET()
            return
        try:
            self._send_json(HTTPStatus.OK, self.server.store.load())
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": str(error)},
            )

    def do_POST(self) -> None:  # noqa: N802
        if not self._is_corrections_api():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown API endpoint."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_REQUEST_BYTES:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "Invalid or oversized request body."},
            )
            return
        try:
            payload = json.loads(self.rfile.read(length))
            saved = self.server.store.save(payload)
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except OSError as error:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": str(error)},
            )
            return
        self._send_json(HTTPStatus.OK, saved)


class ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler,
        store: ReviewStore,
    ):
        self.store = store
        super().__init__(server_address, handler)


def make_server(
    output_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ReviewHTTPServer:
    store = ReviewStore(output_dir)
    handler = partial(ReviewRequestHandler, directory=str(store.output_dir))
    return ReviewHTTPServer((host, int(port)), handler, store)


def refresh_html(output_dir: str | Path) -> Path:
    """Rebuild only index.html from an existing merged manifest."""
    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / "metrics" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Merged report manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    levels = [int(value) for value in manifest["levels"]]
    # These heavier evaluator imports are intentionally lazy: serving an
    # already-current report itself remains Python-standard-library only.
    from html_report import write_html_report
    from merge_results import report_payload

    return write_html_report(output_dir, report_payload(manifest, levels=levels))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a Stage-1 skill-eval HTML report and autosave human review "
            "overrides into its metrics directory."
        )
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Evaluation output containing index.html",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--refresh-html",
        action="store_true",
        help="Rebuild index.html from metrics/manifest.json before serving",
    )
    args = parser.parse_args()

    if args.refresh_html:
        print(f"Refreshed HTML : {refresh_html(args.output)}", flush=True)
    server = make_server(args.output, host=args.host, port=args.port)
    host, port = server.server_address[:2]
    print(f"Review report : http://{host}:{port}/", flush=True)
    print(f"Autosave file : {server.store.corrections_path}", flush=True)
    print(
        "Forward this port in VS Code, then open the forwarded URL. "
        "Ctrl-C stops the server.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review server.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
