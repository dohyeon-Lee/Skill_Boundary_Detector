from __future__ import annotations

import io
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import download_abc_sim as download  # noqa: E402


class Response(io.BytesIO):
    def __init__(self, payload, status=206, content_range=""):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Range": content_range}


def run_download(partial, retries=1):
    download._download_with_resume(
        "https://example.invalid/task.tar", partial, 12, retries, label="task"
    )


@pytest.fixture(autouse=True)
def small_ranges(monkeypatch):
    monkeypatch.setattr(download, "DOWNLOAD_RANGE_SIZE", 4)
    monkeypatch.setattr(download.time, "sleep", lambda _delay: None)


def test_resumes_partial_in_bounded_windows(tmp_path, monkeypatch):
    partial = tmp_path / "task.part"
    partial.write_bytes(b"abc")
    requests = []

    def urlopen(request, timeout):
        requested = request.get_header("Range")
        requests.append(requested)
        start, end = map(int, requested.removeprefix("bytes=").split("-"))
        return Response(b"abcdefghijkl"[start:end + 1], content_range=f"bytes {start}-{end}/12")

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    run_download(partial)
    assert requests == ["bytes=3-6", "bytes=7-10", "bytes=11-11"]
    assert partial.read_bytes() == b"abcdefghijkl"


@pytest.mark.parametrize("status,content_range", [
    (200, ""),
    (206, "bytes 0-3/12"),
    (206, "bytes 3-6/99"),
    (206, "bytes 3-12/12"),
    (206, "invalid"),
])
def test_bad_resume_response_preserves_partial(tmp_path, monkeypatch, status, content_range):
    partial = tmp_path / "task.part"
    partial.write_bytes(b"abc")
    monkeypatch.setattr(
        download.urllib.request, "urlopen",
        lambda *_args, **_kwargs: Response(b"wrong", status, content_range),
    )
    with pytest.raises(RuntimeError, match="download failed"):
        run_download(partial)
    assert partial.read_bytes() == b"abc"


def test_ignored_initial_range_is_capped_and_then_resumed(tmp_path, monkeypatch):
    partial = tmp_path / "task.part"
    responses = iter([
        Response(b"abcdefghijkl", 200),
        Response(b"efghijkl", content_range="bytes 4-11/12"),
        Response(b"ijkl", content_range="bytes 8-11/12"),
    ])
    requests = []

    def urlopen(request, timeout):
        requests.append(request.get_header("Range"))
        return next(responses)

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    run_download(partial)
    assert requests == ["bytes=0-3", "bytes=4-7", "bytes=8-11"]
    assert partial.read_bytes() == b"abcdefghijkl"


def test_timeout_keeps_small_fragments_and_completed_window_resets_retries(tmp_path, monkeypatch):
    partial = tmp_path / "task.part"

    class TimeoutResponse(Response):
        def read1(self, size):
            if self.tell():
                raise TimeoutError("stalled connection")
            return super().read1(1)

    responses = iter([
        TimeoutResponse(b"a", content_range="bytes 0-3/12"),
        Response(b"bcde", content_range="bytes 1-4/12"),
        TimeoutResponse(b"f", content_range="bytes 5-8/12"),
        Response(b"ghij", content_range="bytes 6-9/12"),
        Response(b"kl", content_range="bytes 10-11/12"),
    ])
    requests = []

    def urlopen(request, timeout):
        requests.append(request.get_header("Range"))
        return next(responses)

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    run_download(partial, retries=1)
    assert requests == ["bytes=0-3", "bytes=1-4", "bytes=5-8", "bytes=6-9", "bytes=10-11"]
    assert partial.read_bytes() == b"abcdefghijkl"


def test_real_http_disconnect_resumes_from_saved_bytes(tmp_path):
    payload = b"abcdefghijkl"
    requested = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested.append(self.headers["Range"])
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            # First connection closes after only two of four advertised bytes.
            self.wfile.write(payload[start:start + 2] if len(requested) == 1 else payload[start:end + 1])

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    partial = tmp_path / "task.part"
    try:
        download._download_with_resume(
            f"http://127.0.0.1:{server.server_port}/task.tar",
            partial, len(payload), 2, label="task",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert requested == ["bytes=0-3", "bytes=2-5", "bytes=6-9", "bytes=10-11"]
    assert partial.read_bytes() == payload
