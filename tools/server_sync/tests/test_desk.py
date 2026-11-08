"""Unified UI routes, isolated queues, and live-service reuse; no Hub writes."""
import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desk import DeskHTTPServer
from server import Backend, SyncHTTPServer, Transfers
from transport import Endpoint


class FakeHF:
    def public_config(self):
        return {"mode": "test", "authenticated": False, "endpoints": [], "routes": []}

    def browse(self, *args):
        return {"path": "hub", "entries": [], "total": 0, "offset": 0, "limit": 200}

    def plan(self, source, destination, paths, options=None, dry_run=False, *, operation=None):
        return {"items": [{"path": p, "source": source, "destination": "hub/" + p,
                           "destination_host": "huggingface", "route": "dataset-push",
                           "source_name": source, "destination_name": destination} for p in paths]}

    def commands(self, job):
        return [sys.executable, "-c", "print('mock HF transfer')"]


class DeskTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / "src").mkdir()
        (root / "dst").mkdir()
        (root / "src/dataset_filtered/checkpoint").mkdir(parents=True)
        (root / "src/dataset_filtered/checkpoint/weights").write_bytes(b"test")
        self.backend = Backend([Endpoint("a", "source", str(root / "src")),
                                Endpoint("b", "destination", str(root / "dst"))])
        self.manager = Transfers(self.backend)
        self.factory_calls = 0

        def factory():
            self.factory_calls += 1
            return FakeHF()

        self.server = DeskHTTPServer(("127.0.0.1", 0), self.backend, self.manager,
                                     "test", hf_factory=factory)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.server.close_managers()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        connection.request(method, path, body=None if body is None else json.dumps(body), headers=headers or {})
        response = connection.getresponse()
        status, data = response.status, response.read()
        connection.close()
        return status, data

    def test_tabs_assets_and_one_token_for_both_namespaces(self):
        for page, active in (("/", "/"), ("/hf/", "/hf/")):
            status, data = self.request("GET", page)
            self.assertEqual(status, 200)
            self.assertIn(b'aria-label="', data)
            self.assertIn(f'href="{active}" aria-current="page"'.encode(), data)
            self.assertIn(b'href="/desk.css"', data)
        for asset in ("/app.js", "/style.css", "/desk.css", "/storage.js", "/hf/app.js", "/hf/hf.css"):
            self.assertEqual(self.request("GET", asset)[0], 200, asset)
        self.assertIn(b'src="/hf/app.js"', self.request("GET", "/hf/")[1])
        config = json.loads(self.request("GET", "/api/config")[1])
        hf = json.loads(self.request("GET", "/hf/api/config")[1])
        self.assertEqual(config["token"], hf["token"])
        self.assertEqual(config["ui"], "unified")
        self.assertEqual(self.factory_calls, 1)

    def test_storage_panel_and_api_are_shared_by_both_tabs(self):
        for page in ('/', '/hf/'):
            status, html = self.request('GET', page)
            self.assertEqual(status, 200)
            self.assertIn(b'id="storage-volumes"', html)
            self.assertIn(b'src="/storage.js"', html)
        status, body = self.request('GET', '/api/storage')
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertTrue(data['hostname'])
        self.assertTrue(data['volumes'])
        self.assertIn('free', data['volumes'][0])
        self.assertEqual(self.request('GET', '/api/storage', headers={'Host': 'evil.example'})[0], 403)

    def test_security_and_request_local_backend_switch(self):
        for path in ("/api/config", "/hf/api/config"):
            self.assertEqual(self.request("GET", path, headers={"Host": "evil.example"})[0], 403)
            self.assertEqual(self.request("GET", path, headers={"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.request("POST", "/hf/api/transfers", {})[0], 403)
        paths = ["/api/config", "/hf/api/config"] * 6
        with ThreadPoolExecutor(max_workers=4) as pool:
            payloads = list(pool.map(lambda p: json.loads(self.request("GET", p)[1]), paths))
        for path, payload in zip(paths, payloads):
            self.assertEqual("routes" in payload, path.startswith("/hf/"))
        self.assertEqual(self.factory_calls, 1)
        self.assertIs(self.server.backend, self.backend)

    def test_hf_submissions_do_not_enter_server_queue(self):
        token = json.loads(self.request("GET", "/hf/api/config")[1])["token"]
        status, data = self.request("POST", "/hf/api/transfers",
            {"source": "local", "destination": "hub", "paths": ["dataset"]}, {"X-Sync-Token": token})
        self.assertEqual(status, 202, data)
        for _ in range(100):
            jobs = json.loads(self.request("GET", "/hf/api/jobs")[1])["jobs"]
            if jobs[0]["status"] == "completed":
                break
            time.sleep(.01)
        self.assertEqual(jobs[0]["status"], "completed")
        self.assertEqual(json.loads(self.request("GET", "/api/jobs")[1])["jobs"], [])

    def test_reuse_existing_queue_and_translate_tokens(self):
        old = SyncHTTPServer(("127.0.0.1", 0), self.backend, self.manager, "old")
        thread = threading.Thread(target=old.serve_forever, daemon=True)
        thread.start()
        try:
            self.server.reuse_server_port = old.server_port
            token = json.loads(self.request("GET", "/api/config")[1])["token"]
            self.assertEqual(token, self.server.token)
            self.assertNotEqual(token, old.token)
            status, data = self.request("POST", "/api/plan",
                {"source": "a", "destination": "b", "paths": ["dataset_filtered/checkpoint"]}, {"X-Sync-Token": token})
            self.assertEqual(status, 200, data)
            self.assertEqual(json.loads(data)["items"][0]["path"], "dataset_filtered/checkpoint")
            self.assertEqual(self.request("GET", "/api/jobs")[0], 200)
            self.assertEqual(self.request("GET", "/hf/api/config")[0], 200)
        finally:
            old.shutdown()
            old.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
