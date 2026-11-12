"""RunPod Desk contracts, with offline Pods/SSH and no chargeable operations."""
import base64
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.runpod.desk import Monitor, RunPodAPI
from tools.runpod.desk_probe import snapshot, percentage

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jA6sAAAAASUVORK5CYII=")


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "lerobot/examples/libero/configs/train_skillVLA/FT/eval/outputs/run"
        (self.out / "metrics").mkdir(parents=True)
        self.metrics = {"model": {"overall": {"pc_success": 50}, "per_task": [
            {"task_group": "libero_10", "task_id": 0, "metrics": {"pc_success": 100}},
            {"task_group": "libero_10", "task_id": 1, "metrics": {"successes": [False, False]}}]}}
        (self.out / "metrics/eval_info_merged.json").write_text(json.dumps(self.metrics))
        (self.out / "task_success_rates.png").write_bytes(PNG)
        self.jobs = self.root / ".cache/local_jobs"
        self.jobs.mkdir(parents=True)
        self.record("100", "done")

    def record(self, identifier, state):
        base = self.jobs / identifier
        base.with_suffix(".job").write_text(f"name=FTeval\nscript={self.out.parents[1]}/src/eval.sbatch\n"
            f"eval_output={self.out}\nstarted={identifier}\ngpus=0\nHF_TOKEN=never-export-this\n")
        if state == "done":
            base.with_suffix(".exit").write_text("0")
        elif state == "failed":
            base.with_suffix(".exit").write_text("1")
        else:
            base.with_suffix(".pid").write_text(str(os.getpid()))
        return base

    def collect(self, known=None):
        with patch.dict(os.environ, {"SBD_LOCAL_JOBS_DIR": str(self.jobs)}), patch("tools.runpod.desk_probe.subprocess.check_output", side_effect=OSError):
            return snapshot(self.root, known)

    def test_collects_final_chart_job_name_and_total_and_per_task_rates(self):
        data = self.collect()
        result = data["results"][0]
        self.assertEqual(result["phase"], "final")
        self.assertEqual(result["models"][0]["overall_pct"], 50)
        self.assertEqual([task["success_pct"] for task in result["models"][0]["tasks"]], [100, 0])
        self.assertEqual(base64.b64decode(result["chart"]), PNG)
        self.assertEqual(data["jobs"][0]["name"], "FTeval")
        self.assertNotIn("never-export-this", json.dumps(data))

    def test_completed_job_duration_uses_exit_time_and_stays_fixed(self):
        from datetime import datetime
        base = self.jobs / "100"
        path = base.with_suffix(".job")
        path.write_text(path.read_text().replace("started=100", "started=2026-11-10T00:00:00+00:00"))
        ended = datetime.fromisoformat("2026-11-10T01:01:30+00:00").timestamp()
        os.utime(base.with_suffix(".exit"), (ended, ended))
        for _ in range(2):
            job = self.collect()["jobs"][0]
            self.assertEqual(job["elapsed_seconds"], 3690)
            self.assertEqual(job["finished"], "2026-11-10T01:01:30+00:00")

    def test_stopped_job_without_exit_time_has_unknown_duration(self):
        base = self.jobs / "100"
        base.with_suffix(".exit").unlink()
        job = self.collect()["jobs"][0]
        self.assertIsNone(job["finished"])
        self.assertIsNone(job["elapsed_seconds"])

    def test_unchanged_chart_is_not_transferred_again(self):
        result = self.collect()["results"][0]
        warm = self.collect({result["id"]: result["fingerprint"]})
        self.assertEqual(warm["results"], [])
        self.assertEqual(warm["jobs"][0]["result_id"], result["id"])

    def test_boundary_result_keeps_thresholds_under_its_job(self):
        self.out = self.root / "lerobot/examples/libero/configs/train_skills/skill_eval/skill_boundary_detection/outputs/comparison"
        self.out.mkdir(parents=True)
        thresholds = {"without_gripper": {"groups": {"dataset": {"threshold": 0.2, "episodes": 3921, "points": 77782}},
                        "calibration": {"episodes": 3921, "evaluation_episodes": 200}}}
        (self.out / "evaluation_complete.json").write_text(json.dumps({"kind": "skill_boundary_detection", "thresholds": thresholds}))
        self.record("300", "done")
        data = self.collect()
        result = next(r for r in data["results"] if r.get("kind") == "skill_boundary_detection")
        self.assertEqual(result["thresholds"], thresholds)
        self.assertEqual(result["models"], [])
        self.assertEqual(result["phase"], "final")
        self.assertEqual(next(j for j in data["jobs"] if j["id"] == "300")["result_id"], result["id"])

    def test_running_result_is_partial_and_retry_supersedes_old_failure(self):
        base = self.jobs / "100"
        base.with_suffix(".exit").unlink()
        base.with_suffix(".pid").write_text(str(os.getpid()))
        self.assertEqual(self.collect()["results"][0]["phase"], "partial")
        base.with_suffix(".exit").write_text("1")
        self.record("200", "done")
        self.assertEqual(self.collect()["results"][0]["phase"], "final")

    def test_percent_fields_do_not_turn_one_percent_into_one_hundred(self):
        self.assertEqual(percentage({"pc_success": 1}), 1)
        self.assertEqual(percentage({"success_rate": .01}), 1)
        self.assertIsNone(percentage({"pc_success": float("nan")}))

    def test_missing_chart_keeps_numeric_result_and_corrupt_json_is_reported(self):
        (self.out / "task_success_rates.png").unlink()
        self.assertIsNone(self.collect()["results"][0]["chart"])
        (self.out / "metrics/eval_info_merged.json").write_text("{")
        result = self.collect()
        self.assertEqual(result["results"], [])
        self.assertTrue(result["warnings"])

    def test_old_records_extract_only_output_path_from_log(self):
        record = self.jobs / "100.job"
        log = self.out / "stdout.log"
        log.write_text(f"secret-key-is-not-exported\noutput   : {self.out}\n")
        record.write_text(f"name=old-eval\nscript={self.out.parent}/src/eval.sbatch\noutput={log}\n")
        data = self.collect()
        self.assertEqual(data["results"][0]["phase"], "final")
        self.assertNotIn("secret-key-is-not-exported", json.dumps(data))

    def test_output_symlink_cannot_export_files_outside_project(self):
        with tempfile.TemporaryDirectory() as outside:
            (self.out / "metrics/eval_info_merged.json").unlink()
            secret = Path(outside) / "secret.json"
            secret.write_text(json.dumps(self.metrics))
            (self.out / "metrics/eval_info_merged.json").symlink_to(secret)
            self.assertEqual(self.collect()["results"], [])


class Client:
    def __init__(self):
        self.values = [{"id": "pod-one", "name": "FT GPU", "status": "RUNNING", "cost": .6,
            "createdAt": "2026-11-09T00:00:00Z", "gpu": {"name": "A40", "count": 1},
            "runtime": {"uptime": 3600}, "env": {"HF_TOKEN": "do-not-persist"}}]
        self.stops, self.fail = [], False
        self.balance_calls, self.balance_fail = 0, False

    def balance(self):
        self.balance_calls += 1
        if self.balance_fail:
            raise ValueError("private-account-error-do-not-export")
        return {"balance": 125.26, "currency": "USD", "updated_at": "2026-11-12T00:00:00Z"}

    def pods(self):
        if self.fail:
            raise ValueError("offline")
        return copy.deepcopy(self.values)

    def pod(self, identifier):
        return copy.deepcopy(next(value for value in self.values if value["id"] == identifier))

    def billing(self, pod):
        return {"totalAmount": .61, "gpuAmount": .6, "cpuAmount": 0, "diskAmount": .01,
                "lifetime": True, "updated_at": "2026-11-09T01:00:00Z"}

    def stop(self, identifier):
        self.stops.append(identifier)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.client = Client()
        self.data = {"jobs": [{"id": "100", "name": "FTeval", "state": "done", "result_id": "a" * 24}],
            "gpus": [{"name": "A40", "memory_total_mib": 49152}], "results": [{"id": "a" * 24,
                "fingerprint": "b" * 64, "chart": base64.b64encode(PNG).decode(), "phase": "final",
                "name": "FT-result", "models": [], "mtime": 100, "job_ids": ["100"]}]}
        self.probe = lambda *args: copy.deepcopy(self.data)
        self.monitor = Monitor(self.folder, client=self.client, probe=self.probe)
        self.addCleanup(self.monitor.close)

    def test_account_balance_works_without_pods_and_has_a_one_minute_cache(self):
        self.client.values = []
        self.monitor.poll()
        self.monitor.poll()
        self.assertEqual(self.client.balance_calls, 1)
        account = self.monitor.snapshot()["account"]
        self.assertEqual(account["balance"], 125.26)
        self.assertFalse(account["stale"])
        with patch("tools.runpod.desk.time.monotonic", return_value=self.monitor.balance_checked + 61):
            self.monitor.poll()
        self.assertEqual(self.client.balance_calls, 2)
        self.monitor.poll(force_balance=True)
        self.assertEqual(self.client.balance_calls, 3)

    def test_balance_failure_retains_last_value_and_does_not_block_pod_polling(self):
        self.monitor.poll()
        self.client.balance_fail = True
        self.monitor.poll(force_balance=True)
        state = self.monitor.snapshot()
        self.assertEqual(state["account"]["balance"], 125.26)
        self.assertTrue(state["account"]["stale"])
        self.assertTrue(state["account"]["error"])
        self.assertEqual(state["pods"][0]["status"], "RUNNING")
        self.assertNotIn("private-account-error", json.dumps(state))
        self.client.balance_fail = False
        self.monitor.poll(force_balance=True)
        self.assertFalse(self.monitor.snapshot()["account"]["stale"])

    def test_pod_api_failure_does_not_hide_account_balance_and_cache_survives_restart(self):
        self.client.fail = True
        self.monitor.poll()
        self.assertEqual(self.monitor.snapshot()["account"]["balance"], 125.26)
        self.assertTrue(self.monitor.snapshot()["error"])
        self.monitor.close()
        restored = Monitor(self.folder, client=self.client, probe=self.probe)
        try:
            self.assertEqual(restored.snapshot()["account"]["balance"], 125.26)
            self.assertTrue(restored.snapshot()["account"]["stale"])
        finally:
            restored.close()

    def test_state_and_chart_survive_deletion_and_monitor_restart(self):
        self.monitor.poll()
        state = self.monitor.snapshot()
        self.assertEqual(state["pods"][0]["session_estimate"], .6)
        self.assertEqual(state["pods"][0]["billing"]["totalAmount"], .61)
        self.assertNotIn("do-not-persist", (self.monitor.folder / "state.json").read_text())
        chart = self.monitor.folder / "charts" / (state["results"][0]["chart_url"].rsplit("/", 1)[-1] + ".png")
        self.assertEqual(chart.read_bytes(), PNG)
        self.client.values = []
        self.monitor.poll()
        self.assertEqual(self.monitor.snapshot()["pods"][0]["status"], "DELETED")
        self.assertEqual(len(self.monitor.snapshot()["results"]), 1)
        self.monitor.close()
        restored = Monitor(self.folder, client=self.client, probe=self.probe)
        try:
            self.assertEqual(restored.snapshot()["results"][0]["name"], "FT-result")
            self.assertEqual(chart.read_bytes(), PNG)
        finally:
            restored.close()

    def test_api_failure_keeps_previous_state_and_marks_it_stale(self):
        self.monitor.poll()
        self.client.fail = True
        self.monitor.poll()
        state = self.monitor.snapshot()
        self.assertTrue(state["error"])
        self.assertTrue(state["pods"][0]["stale"])
        self.assertEqual(state["pods"][0]["status"], "RUNNING")

    def test_completed_pod_duration_survives_later_polls_and_restart(self):
        with patch("tools.runpod.desk.now", return_value="2026-11-09T01:00:00+00:00"):
            self.monitor.poll()
        self.assertEqual(self.monitor.snapshot()["pods"][0]["elapsed_seconds"], 3600)
        self.client.values[0]["status"] = "EXITED"
        with patch("tools.runpod.desk.now", return_value="2026-11-09T01:10:00+00:00"):
            self.monitor.poll()
        with patch("tools.runpod.desk.now", return_value="2026-11-09T02:00:00+00:00"):
            self.monitor.poll()
        self.assertEqual(self.monitor.snapshot()["pods"][0]["elapsed_seconds"], 4200)
        self.monitor.close()
        restored = Monitor(self.folder, client=self.client, probe=self.probe)
        try:
            self.assertEqual(restored.snapshot()["pods"][0]["elapsed_seconds"], 4200)
            self.assertEqual(restored.snapshot()["pods"][0]["finished_at"], "2026-11-09T01:10:00+00:00")
        finally:
            restored.close()

    def test_historical_stopped_pod_does_not_invent_an_end_time(self):
        self.client.values[0]["status"] = "EXITED"
        self.monitor.poll()
        self.assertIsNone(self.monitor.snapshot()["pods"][0]["elapsed_seconds"])
        self.assertNotIn("finished_at", self.monitor.snapshot()["pods"][0])

    def test_stop_requires_confirmation_and_does_not_run_automatically(self):
        self.monitor.poll()
        self.assertEqual(self.client.stops, [])
        for identifier, confirmation in (("pod-one", "wrong"), ("unknown", "unknown"), ("../x", "../x")):
            with self.assertRaises(ValueError):
                self.monitor.stop_pod(identifier, confirmation)
        self.assertEqual(self.client.stops, [])
        result = self.monitor.stop_pod("pod-one", "pod-one")
        self.assertTrue(result["ok"])
        self.assertEqual(self.client.stops, ["pod-one"])
        self.assertEqual(self.monitor.snapshot()["pods"][0]["status"], "STOP_REQUESTED")
        # A delayed RUNNING response must not claim the stop has already happened.
        self.monitor.poll()
        self.assertEqual(self.monitor.snapshot()["pods"][0]["status"], "STOP_REQUESTED")

    def test_backup_failure_warns_but_does_not_prevent_stopping_compute(self):
        self.monitor.poll()
        self.monitor.probe = lambda *args: (_ for _ in ()).throw(ValueError("SSH offline"))
        result = self.monitor.stop_pod("pod-one", "pod-one")
        self.assertTrue(result["warning"])
        self.assertEqual(self.client.stops, ["pod-one"])

    def test_stop_rechecks_live_status_and_propagates_failed_action(self):
        self.monitor.poll()
        self.client.values[0]["status"] = "EXITED"
        with self.assertRaisesRegex(ValueError, "실행 중"):
            self.monitor.stop_pod("pod-one", "pod-one")
        self.assertEqual(self.client.stops, [])
        self.client.values[0]["status"] = "RUNNING"
        with patch.object(self.client, "stop", side_effect=ValueError("denied")):
            with self.assertRaisesRegex(ValueError, "denied"):
                self.monitor.stop_pod("pod-one", "pod-one")
        self.assertEqual(self.monitor.snapshot()["pods"][0]["status"], "RUNNING")

    def test_only_one_state_owner_and_settings_validate(self):
        with self.assertRaises(ValueError):
            Monitor(self.folder, client=self.client)
        for values in ({"interval": 1}, {"project_root": "relative"}, {"identity": "a\nkey"}, {"api_key": "not-in-browser"}):
            with self.assertRaises(ValueError):
                self.monitor.configure(values)
        self.monitor.configure({"interval": 10, "project_root": "/workspace/another repo"})
        self.assertEqual(self.monitor.config["interval"], 10)


class APIContractTests(unittest.TestCase):
    def balance_reply(self, response):
        with patch("tools.runpod.desk.api.load_key", return_value="never-print-this-key"), patch("tools.runpod.desk.api.api_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value = SimpleNamespace(
                read=lambda size: json.dumps(response).encode())
            value = RunPodAPI().balance()
        return value, opener.return_value.open.call_args.args[0]

    def test_balance_query_reads_only_account_credit_without_leaking_credentials(self):
        value, request = self.balance_reply({"data": {"myself": {"clientBalance": 125.2603347002,
                                                               "email": "not-exported"}}})
        self.assertEqual(value["balance"], 125.2603347002)
        self.assertEqual(value["currency"], "USD")
        self.assertNotIn("not-exported", json.dumps(value))
        self.assertEqual(request.full_url, "https://api.runpod.io/graphql")
        self.assertEqual(request.method, "POST")
        self.assertEqual(json.loads(request.data), {"query": "query DeskBalance { myself { clientBalance } }"})
        self.assertNotIn("never-print-this-key", request.full_url)

    def test_zero_and_negative_balances_are_valid_but_missing_or_invalid_values_are_unknown(self):
        for balance in (0, -.15):
            with self.subTest(balance=balance):
                self.assertEqual(self.balance_reply({"data": {"myself": {"clientBalance": balance}}})[0]["balance"], balance)
        for balance in (None, True, "25", float("nan"), float("inf")):
            with self.subTest(balance=balance), self.assertRaises(ValueError):
                self.balance_reply({"data": {"myself": {"clientBalance": balance}}})
        for response in ({"errors": [{"message": "secret"}]}, {"data": {"myself": None}},
                         {"data": {"myself": {"clientBalance": 0}}, "errors": ["denied"]}):
            with self.assertRaises(ValueError) as caught:
                self.balance_reply(response)
            self.assertNotIn("secret", str(caught.exception))

    def test_billing_is_filtered_and_uses_record_amounts_not_hourly_estimate(self):
        response = {"records": [{"podId": "pod-one", "gpuAmount": .6, "cpuAmount": 0, "diskAmount": .01, "totalAmount": .61}]}
        with patch("tools.runpod.desk.api.load_key", return_value="not-printed"), patch("tools.runpod.desk.api._api_json", return_value=response) as call:
            value = RunPodAPI().billing({"id": "pod-one", "created_at": "2026-11-09T02:00:00Z"})
        self.assertEqual(value["totalAmount"], .61)
        self.assertTrue(value["lifetime"])
        query = parse_qs(urlsplit(call.call_args.args[1]).query)
        self.assertEqual(query["podId"], ["pod-one"])
        self.assertEqual(query["startTime"], ["2026-11-09T00:00:00+00:00"])

    def test_stop_uses_v2_action_post_without_delete_or_create(self):
        with patch("tools.runpod.desk.api.load_key", return_value="not-printed"), patch("tools.runpod.desk.api.api_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value = SimpleNamespace(read=lambda size: b"{}")
            RunPodAPI().stop("pod-one")
        request = opener.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.runpod.io/v2/pods/pod-one/action")
        self.assertEqual(request.method, "POST")
        self.assertEqual(json.loads(request.data), {"action": "stop"})


if __name__ == "__main__":
    unittest.main()
