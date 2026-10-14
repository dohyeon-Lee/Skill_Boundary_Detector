"""Read-only remote mocks and real copies ONLY inside fresh temporary directories."""

import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fs_helper import probe, relative_path
from server import Backend, Endpoint, SyncHTTPServer, Transfers, configured_endpoints


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sync-ui-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source, self.destination = self.base / "source", self.base / "destination"
        self.source.mkdir()
        self.destination.mkdir()
        self.endpoints = [Endpoint("a", "Source", str(self.source)), Endpoint("b", "Destination", str(self.destination))]
        self.backend = Backend(self.endpoints, streams=2)

    def checkpoint(self, name="residual_1", step="050000"):
        relative = "outputs_filtered/skillVLA_stage1/VSA/" + name + "/checkpoints/" + step
        path = self.source / relative
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_bytes(b"test model\x00data")
        (path / "config.json").write_text("{}")
        return relative

    def wait_done(self, manager, timeout=12):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            jobs = manager.snapshot()
            if jobs and all(job["status"] not in {"running", "queued", "cancelling"} for job in jobs):
                return jobs
            time.sleep(0.03)
        self.fail("Transfer did not finish")

    def test_endpoint_defaults_and_internal_mode(self):
        mode, endpoints = configured_endpoints(self.source, {}, "node100")
        self.assertEqual(mode, "rllab")
        self.assertEqual(len(endpoints), 4)
        self.assertEqual(endpoints[0].base, str(self.source))
        self.assertEqual(endpoints[-1].host, "mdorazi@165.132.142.208")
        mode, endpoints = configured_endpoints(self.source, {}, "yonsei-ai-login4")
        self.assertEqual(mode, "yonsei")
        self.assertEqual(len(endpoints), 2)
        self.assertFalse(any(endpoint.host for endpoint in endpoints))
        with self.assertRaises(ValueError):
            configured_endpoints(self.source, {"SYNC_MODE": "wrong"}, "node100")

    def test_browse_roots_search_pagination_and_files(self):
        for variant in range(1, 7):
            self.checkpoint("residual_" + str(variant))
        (self.source / ".ssh").mkdir()
        (self.source / "README.md").write_text("private")
        self.assertEqual([item["name"] for item in self.backend.browse("a")["entries"]], ["outputs_filtered"])
        listing = self.backend.browse("a", "outputs_filtered/skillVLA_stage1/VSA", search="residual_2")
        self.assertEqual(listing["total"], 1)
        self.assertEqual(listing["entries"][0]["name"], "residual_2")
        listing = probe(str(self.source), "outputs_filtered/skillVLA_stage1/VSA", "list", 3, 2)
        self.assertEqual([item["name"] for item in listing["entries"]], ["residual_4", "residual_5"])

    def test_plan_multiple_50k_and_parent_collapse(self):
        paths = [self.checkpoint("residual_" + str(i)) for i in range(1, 7)]
        plan = self.backend.plan("a", "b", paths + [paths[0], paths[0] + "/config.json"])
        self.assertEqual(len(plan["items"]), 6)
        self.assertEqual(plan["collapsed"], 2)
        for item in plan["items"]:
            self.assertEqual(item["destination"], str(self.destination / item["path"]))
        self.assertFalse((self.destination / "outputs_filtered").exists())

    def test_reject_unsafe_paths_endpoints_and_mismatched_types(self):
        for path in ("", "/tmp/file", "../outputs", "outputs/../secret", "outputs//child", ".ssh", "models/./file", "models/file\x00"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                relative_path(path)
        with self.assertRaises(ValueError):
            Endpoint("remote", "remote", "/tmp/project", "-oProxyCommand=bad")
        path = self.checkpoint()
        for source, destination in (("a", "a"), ("wrong", "b"), ([], "b")):
            with self.assertRaises(ValueError):
                self.backend.plan(source, destination, [path])
        target = self.destination / path
        target.parent.mkdir(parents=True)
        target.write_text("not a folder")
        with self.assertRaises(ValueError):
            self.backend.plan("a", "b", [path])

    def test_source_and_destination_symlink_escapes_rejected(self):
        (self.source / "models").symlink_to(self.base)
        with self.assertRaises(ValueError):
            self.backend.browse("a", "models")
        path = self.checkpoint()
        (self.destination / "outputs_filtered").symlink_to(self.base)
        with self.assertRaises(ValueError):
            self.backend.plan("a", "b", [path])

    def test_overlapping_physical_endpoints_rejected(self):
        path = self.checkpoint()
        backend = Backend([self.endpoints[0], Endpoint("alias", "alias", str(self.source))])
        with self.assertRaises(ValueError):
            backend.plan("a", "alias", [path])
        inside = self.source / path / "nested"
        backend = Backend([self.endpoints[0], Endpoint("inside", "inside", str(inside))])
        with self.assertRaises(ValueError):
            backend.plan("a", "inside", [path])

    def test_remote_probe_uses_quoted_json_no_local_shell(self):
        remote = Endpoint("remote", "remote", "/scratch2/test user's/SBD", "user@host")
        with patch("server.subprocess.run", return_value=subprocess.CompletedProcess([], 0, '{"entries":[],"total":0}', "")) as run:
            result = self.backend.inspect(remote, "outputs_filtered/a'; echo bad", "list")
        self.assertEqual(result["total"], 0)
        args = run.call_args.args[0]
        self.assertEqual(args[0], "ssh")
        self.assertIn("BatchMode=yes", args)
        self.assertEqual(args[-2], "user@host")
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertIn("python3 -c", args[-1])

    def test_remote_to_remote_command_and_unsupported_host_pair(self):
        first = Endpoint("r1", "first", "/tmp/source", "user@host")
        second = Endpoint("r2", "second", "/tmp/destination", "user@host")
        third = Endpoint("r3", "third", "/tmp/destination", "user@other")
        backend = Backend([first, second, third])
        with self.assertRaises(ValueError):
            backend.plan("r1", "r3", ["models/checkpoint"])
        def inspect(endpoint, relative, operation="stat"):
            return {"exists":endpoint.id=="r1", "kind":"directory", "real_path":endpoint.base+"/"+relative}
        with patch.object(backend, "inspect", side_effect=inspect):
            plan = backend.plan("r1", "r2", ["models/checkpoint"])["items"][0]
            command = backend.commands(plan)
        self.assertEqual(command[:2], ["ssh", "-tt"])
        self.assertTrue(command[-1].startswith("exec rsync "))
        self.assertNotIn("--delete", command[-1])

    def test_remote_auth_error_includes_terminal_guidance(self):
        remote = Endpoint("remote", "remote", "/scratch2/project", "user@host")
        with patch("server.subprocess.run", return_value=subprocess.CompletedProcess([], 255, "", "Permission denied (publickey).")):
            with self.assertRaisesRegex(ValueError, "터미널에서 ssh user@host"):
                self.backend.inspect(remote, "models", "list")

    @unittest.skipUnless(shutil.which("rsync"), "rsync not installed")
    def test_real_multi_directory_and_file_copy_preserves_paths_and_extra_files(self):
        paths = [self.checkpoint("residual one's [1]"), self.checkpoint("residual_2")]
        paths.append(paths[1] + "/config.json")  # collapsed under its selected checkpoint
        (self.source / "models").mkdir()
        (self.source / "models" / "weight with spaces.pt").write_bytes(b"weights")
        paths.append("models/weight with spaces.pt")
        extra = self.destination / paths[0] / "extra.txt"
        extra.parent.mkdir(parents=True)
        extra.write_text("keep this")
        manager = Transfers(self.backend, concurrency=2)
        self.addCleanup(manager.close)
        identifiers = manager.submit(self.backend.plan("a", "b", paths)["items"])
        self.assertEqual(len(identifiers), 3)
        jobs = self.wait_done(manager)
        self.assertTrue(all(job["status"]=="completed" for job in jobs), jobs)
        for path in paths[:2]:
            self.assertEqual((self.source/path/"model.safetensors").read_bytes(), (self.destination/path/"model.safetensors").read_bytes())
        self.assertEqual((self.destination/"models/weight with spaces.pt").read_bytes(), b"weights")
        self.assertEqual(extra.read_text(), "keep this")

    def test_cancel_running_and_queued_and_duplicate_destination(self):
        paths = [self.checkpoint("residual_1"), self.checkpoint("residual_2")]
        manager = Transfers(self.backend, concurrency=1)
        self.addCleanup(manager.close)
        with patch.object(self.backend, "commands", return_value=[sys.executable, "-c", "import time; time.sleep(30)"]):
            items = self.backend.plan("a", "b", paths)["items"]
            identifiers = manager.submit(items)
            until = time.monotonic()+5
            while manager.jobs[identifiers[0]]["process"] is None and time.monotonic()<until:
                time.sleep(0.02)
            with self.assertRaises(ValueError):
                manager.submit(items)
            manager.cancel(identifiers[1])
            manager.cancel(identifiers[0])
            self.assertTrue(all(job["status"]=="cancelled" for job in self.wait_done(manager)))

    def test_error_is_failed_not_completed_and_readout_bounded(self):
        path = self.checkpoint()
        manager = Transfers(self.backend)
        self.addCleanup(manager.close)
        with patch.object(self.backend, "commands", return_value=[sys.executable,"-c","import sys; print('failed'); sys.exit(23)"]):
            manager.submit(self.backend.plan("a", "b", [path])["items"])
            job = self.wait_done(manager)[0]
        self.assertEqual(job["status"], "failed")
        self.assertIn("23", job["detail"])
        for i in range(200):
            manager._log(manager.jobs[job["id"]], "100% line " + str(i))
        snapshot = manager.snapshot()[0]
        self.assertEqual(len(snapshot["logs"]),120)
        self.assertLess(snapshot["progress"],100)


class HTTPTest(WorkspaceTest):
    def setUp(self):
        super().setUp()
        self.manager = Transfers(self.backend)
        self.server = SyncHTTPServer(("127.0.0.1",0),self.backend,self.manager,"test")
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.manager.close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=5)
        connection.request(method,path,body=json.dumps(body) if body is not None else None,headers=headers or {})
        response = connection.getresponse()
        data=response.read()
        status=response.status
        connection.close()
        return status,data

    def test_http_assets_security_plan_and_job_snapshot(self):
        status,data=self.request("GET","/")
        self.assertEqual(status,200)
        self.assertIn(b"Transfer Desk",data)
        for asset in ("/app.js","/style.css"):
            self.assertEqual(self.request("GET",asset)[0],200)
        self.assertEqual(self.request("GET","/../../.ssh/id_ed25519")[0],404)
        self.assertEqual(self.request("GET","/api/config",headers={"Host":"evil.example"})[0],403)
        self.assertEqual(self.request("GET","/api/config",headers={"Origin":"https://evil.example"})[0],403)
        self.assertEqual(self.request("POST","/api/plan",{})[0],403)
        status,data=self.request("GET","/api/config")
        token=json.loads(data)["token"]
        path=self.checkpoint()
        status,data=self.request("POST","/api/plan",{"source":"a","destination":"b","paths":[path]},headers={"X-Sync-Token":token})
        self.assertEqual(status,200,data)
        self.assertFalse((self.destination/"outputs_filtered").exists())
        self.assertEqual(self.request("GET","/api/jobs")[0],200)
        status,data=self.request("POST","/api/plan",["bad"],headers={"X-Sync-Token":token})
        self.assertEqual(status,400)


if __name__ == "__main__":
    unittest.main()
