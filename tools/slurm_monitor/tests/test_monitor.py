from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import codex
import slurm
from monitor import Monitor
from server import HTTPServer

THREAD = "01a13faa-4bbb-7ca3-8d35-57c2d849bce5"


def job(state="RUNNING", identifier="123", submit="2026-10-24T12:00:00"):
    return {"job_id": identifier, "job_name": "stage1", "state": state,
            "terminal": state in slurm.TERMINAL, "submitted_at": submit,
            "exit_code": "0:0", "elapsed": "01:00:00", "outputs": ["/repo/outputs/model"]}


class FakeSlurm:
    def __init__(self):
        self.rows = [job()]
        self.error = None
        self.cancelled = []

    def collect(self, tracked):
        if self.error:
            raise RuntimeError(self.error)
        return {slurm.job_key(row): row for row in self.rows}

    def enrich(self, row):
        return row

    def cancel(self, row):
        self.cancelled.append(row["job_id"])


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    monkeypatch.setattr(codex, "connection_status", lambda config: {"configured": True, "session_found": True})
    fake = FakeSlurm()
    sent = []
    def sender(config, message):
        sent.append((dict(config), message))
        return {"status": "sent", "receipt": "accepted"}
    m = Monitor(tmp_path / "state", tmp_path, slurm=fake, sender=sender, defaults={"thread_id": THREAD})
    m.sent = sent
    m.poll()
    yield m
    m.close()


def only(m):
    return next(iter(m.jobs))


def test_no_prompt_means_no_message_even_after_completion(monitor):
    monitor.slurm.rows = [job("COMPLETED")]
    monitor.poll()
    assert not monitor.deliver_one()
    assert not monitor.sent
    assert monitor.jobs[only(monitor)]["delivery"]["status"] == "none"


def test_armed_prompt_sends_once_and_preserves_user_text(monitor):
    key = only(monitor)
    prompt = '노션 SR 갱신\n"quotes" `literal` $(touch /tmp/do-not-run)'
    monitor.update_job(key, {"prompt": prompt, "armed": True})
    monitor.poll()
    assert not monitor.deliver_one()
    monitor.slurm.rows = [job("COMPLETED")]
    monitor.poll()
    assert monitor.deliver_one()
    assert monitor.sent[0][1].startswith(prompt + "\n\n---")
    assert '"job_id": "123"' in monitor.sent[0][1]
    assert monitor.sent[0][0]["thread_id"] == THREAD
    for _ in range(3):
        monitor.poll()
        assert not monitor.deliver_one()
    assert len(monitor.sent) == 1
    assert monitor.jobs[key]["events"][0]["prompt"] == prompt


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"])
def test_failure_states_are_reported_without_claiming_success(monitor, state):
    monitor.update_job(only(monitor), {"prompt": "실패면 원인을 알려줘", "armed": True})
    monitor.slurm.rows = [job(state)]
    monitor.poll()
    monitor.deliver_one()
    assert f'"state": "{state}"' in monitor.sent[0][1]


def test_history_prompt_save_does_not_send_without_explicit_action(monitor):
    monitor.slurm.rows = [job("COMPLETED")]
    monitor.poll()
    key = only(monitor)
    monitor.update_job(key, {"prompt": "과거 결과 정리", "armed": True})
    monitor.poll()
    assert not monitor.deliver_one()
    monitor.request_send(key)
    assert monitor.deliver_one()
    with pytest.raises(ValueError, match="이미"):
        monitor.request_send(key)


def test_disabled_prompt_never_queues(monitor):
    monitor.update_job(only(monitor), {"prompt": "실행하지 않음", "armed": False})
    monitor.slurm.rows = [job("COMPLETED")]
    monitor.poll()
    assert not monitor.deliver_one()


def test_queue_failure_and_missing_entry_never_imply_completion(monitor):
    monitor.update_job(only(monitor), {"prompt": "종료 후 실행", "armed": True})
    monitor.slurm.rows = []
    monitor.poll()
    assert monitor.jobs[only(monitor)]["state"] == "RUNNING"
    monitor.slurm.error = "sacct offline"
    monitor.poll()
    assert monitor.poll_error == "sacct offline"
    assert not monitor.deliver_one()


def test_requeue_keeps_prompt_without_notifying(monitor):
    monitor.update_job(only(monitor), {"prompt": "유지할 지시", "armed": True})
    monitor.slurm.rows = [job("PENDING", submit="2026-10-24T13:00:00")]
    monitor.poll()
    assert len(monitor.jobs) == 1
    row = monitor.jobs[only(monitor)]
    assert row["prompt"] == "유지할 지시" and row["armed"]
    assert not monitor.deliver_one()


def test_restart_retains_sent_receipt_and_never_sends_twice(monitor):
    monitor.update_job(only(monitor), {"prompt": "한 번", "armed": True})
    monitor.slurm.rows = [job("COMPLETED")]
    monitor.poll()
    monitor.deliver_one()
    directory = monitor.store.directory
    monitor.close()
    new = Monitor(directory, monitor.config["project_root"], slurm=monitor.slurm, sender=monitor.sender)
    try:
        new.poll()
        assert not new.deliver_one()
        assert new.jobs[only(new)]["delivery"]["status"] == "sent"
    finally:
        new.close()


def test_restart_during_delivery_requires_manual_reconciliation(monitor):
    row = monitor.jobs[only(monitor)]
    row.update(prompt="지시", armed=True)
    row["delivery"] = {"status": "sending"}
    monitor.save()
    directory = monitor.store.directory
    monitor.close()
    new = Monitor(directory, monitor.config["project_root"], slurm=monitor.slurm, sender=monitor.sender)
    try:
        new.poll()
        assert new.jobs[only(new)]["delivery"]["status"] == "uncertain"
        assert not new.deliver_one()
    finally:
        new.close()


def test_failure_does_not_retry_in_background(monitor):
    calls = []
    def failing(*args):
        calls.append(args)
        return {"status": "failed", "error": "no daemon"}
    monitor.sender = failing
    monitor.update_job(only(monitor), {"prompt": "지시", "armed": True})
    monitor.slurm.rows = [job("COMPLETED")]
    monitor.poll()
    monitor.deliver_one()
    monitor.poll()
    assert not monitor.deliver_one()
    assert len(calls) == 1
    monitor.request_send(only(monitor))
    assert monitor.deliver_one()
    assert len(calls) == 2


def test_save_during_send_cannot_change_payload(monitor):
    key = only(monitor)
    monitor.update_job(key, {"prompt": "원문", "armed": True})
    monitor.slurm.rows = [job("COMPLETED")]
    monitor.poll()
    def sending(config, message):
        with pytest.raises(ValueError, match="전송 중"):
            monitor.update_job(key, {"prompt": "다른 지시"})
        with pytest.raises(ValueError, match="전송 중"):
            monitor.configure({"thread_id": THREAD})
        return {"status": "sent"}
    monitor.sender = sending
    monitor.deliver_one()
    assert monitor.jobs[key]["delivery"]["prompt"] == "원문"


@pytest.mark.parametrize("values", [{"interval": 1}, {"thread_id": "bad"}, {"thread_id": None}, {"codex_remote": "http://bad"}])
def test_invalid_connection_settings_rejected(monitor, values):
    with pytest.raises(ValueError):
        monitor.configure(values)


def test_prompt_and_labels_survive_polling_and_state_is_private(monitor):
    key = only(monitor)
    monitor.update_job(key, {"prompt": "저장됨", "output_label": "직접 보정한 폴더"})
    monitor.poll()
    assert monitor.snapshot()["jobs"][0]["display_name"] == "직접 보정한 폴더"
    assert json.loads(monitor.store.path.read_text())["jobs"][key]["prompt"] == "저장됨"
    assert monitor.store.path.stat().st_mode & 0o077 == 0
    with pytest.raises(RuntimeError, match="이미 실행"):
        Monitor(monitor.store.directory, monitor.config["project_root"])


def test_sender_uses_argv_with_exact_text_and_explicit_codex_home(tmp_path):
    seen = []
    def runner(args, **kwargs):
        seen.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="Queued message", stderr="")
    config = {"codex_command": "codex", "thread_id": THREAD, "codex_home": str(tmp_path),
              "project_root": str(tmp_path), "codex_remote": "unix:///tmp/test.sock"}
    message = '한글\n`literal` $(literal) "quotes"'
    assert codex.send(config, message, runner)["status"] == "sent"
    args, options = seen[0]
    assert args == ["codex", "--remote", "unix:///tmp/test.sock", "queue", "--thread", THREAD, "--message", message]
    assert options["env"]["CODEX_HOME"] == str(tmp_path)
    assert "shell" not in options


def test_sender_timeout_is_ambiguous_not_safe_to_automatically_retry(tmp_path):
    def runner(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 45)
    result = codex.send({"thread_id": THREAD, "project_root": str(tmp_path)}, "test", runner)
    assert result["status"] == "uncertain"


def test_slurm_parsers_keep_user_arrays_and_actual_failure():
    text = ("123_2|S1eval|CANCELLED by 1025|0:15|1:00|6:00|2026-10-24T12:00:00|start|end|/repo|me\n"
            "124|other|COMPLETED|0:0|1:00|6:00|submit|start|end|/repo|other\n"
            "123_2.batch|batch|COMPLETED|0:0|1:00|6:00|submit|start|end|/repo|me\n")
    rows = slurm.parse_accounting(text, "me")
    assert len(rows) == 1
    row = next(iter(rows.values()))
    assert row["state"] == "CANCELLED" and row["terminal"]
    queue = slurm.parse_queue("123_2|RUNNING|S1eval|1:00|6:00|node100|submit|start\n")
    assert not queue[0]["terminal"]
    assert slurm.expand_log_path("logs/%x_%A_%a.out", row) == "logs/S1eval_123_2.out"


def test_slurm21_query_uses_supported_fields_and_queue_overrides_stale_sacct():
    commands = []
    def runner(args):
        commands.append(args)
        if args[0] == "squeue":
            return "123|PENDING|stage1|0:00|6:00|Resources|2026-10-24T12:00:00|N/A\n"
        return "123|stage1|COMPLETED|0:0|1:00|6:00|2026-10-24T12:00:00|start|end|/repo|me\n"
    rows = slurm.Slurm("me", runner=runner).collect()
    row = next(iter(rows.values()))
    assert row["state"] == "PENDING" and not row["terminal"]
    assert all("StdOut" not in " ".join(command) for command in commands)


def test_grouped_job_uses_all_child_output_folders():
    text = "output : /repo/base\nSTART modelA pid=1 output=/repo/modelA logs=/x\nSTART modelB pid=2 output=/repo/modelB logs=/y\n"
    assert slurm.output_paths(text) == ["/repo/modelA", "/repo/modelB"]
    assert slurm.output_paths("output  : /repo/model\nDONE -> /repo/model\n") == ["/repo/model"]


def test_pending_folder_uses_unique_submission_snapshot_not_current_yaml(tmp_path):
    directory = tmp_path / ".config_snapshots/stage1_eval_config.yaml.20261024-120000_1_1"
    directory.mkdir(parents=True)
    (directory / "stage1_eval_config.yaml").write_text('output_name: "original_model"\n')
    (tmp_path / "stage1_eval_config.yaml").write_text('output_name: "new_model"\n')
    result = slurm.snapshot_output(tmp_path, "2026-10-24T12:00:01")
    assert Path(result[0]).name == "original_model"
    duplicate = tmp_path / ".config_snapshots/stage1_eval_config.yaml.20261024-120000_2_2"
    duplicate.mkdir()
    (duplicate / "stage1_eval_config.yaml").write_text('output_name: "ambiguous"\n')
    assert slurm.snapshot_output(tmp_path, "2026-10-24T12:00:01") is None


def test_http_requires_origin_and_session_token(monitor):
    server = HTTPServer(("127.0.0.1", 0), monitor)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port)
        conn.request("GET", "/api/state")
        response = conn.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200 and payload["jobs"]
        body = json.dumps({"key": only(monitor), "values": {"prompt": "UI에서 저장"}})
        conn.request("POST", "/api/job", body, {"Content-Type": "application/json"})
        response = conn.getresponse(); response.read()
        assert response.status == 403
        headers = {"Content-Type": "application/json", "X-Monitor-Token": payload["token"], "Origin": "https://outside.invalid"}
        conn.request("POST", "/api/job", body, headers)
        response = conn.getresponse(); response.read()
        assert response.status == 403
        headers.pop("Origin")
        conn.request("POST", "/api/cancel", json.dumps({"key": only(monitor)}))
        response = conn.getresponse(); response.read()
        assert response.status == 403
        assert not monitor.slurm.cancelled
        conn.request("POST", "/api/cancel", json.dumps({"key": only(monitor)}), headers)
        response = conn.getresponse(); response.read()
        assert response.status == 200
        assert monitor.slurm.cancelled == ["123"]
        conn.request("POST", "/api/job", body, headers)
        response = conn.getresponse(); response.read()
        assert response.status == 200
        assert monitor.jobs[only(monitor)]["prompt"] == "UI에서 저장"
        conn.request("GET", "/", headers={"Host": "outside.invalid"})
        response = conn.getresponse(); response.read()
        assert response.status == 403
        conn.close()
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_cancel_preserves_instructions_and_waits_for_actual_terminal_state(monitor):
    key = only(monitor)
    monitor.update_job(key, {"prompt": "원문 그대로\n취소면 원인만 확인", "armed": True, "output_label": "내 폴더"})
    before = {field: monitor.jobs[key][field] for field in ("prompt", "armed", "armed_for_completion", "output_label", "delivery", "events")}
    monitor.cancel_job(key)
    assert monitor.slurm.cancelled == ["123"]
    assert {field: monitor.jobs[key][field] for field in before} == before
    assert monitor.jobs[key]["state"] == "RUNNING"
    assert not monitor.deliver_one()
    monitor.poll()
    assert monitor.jobs[key]["cancel"]["status"] == "requested"
    with pytest.raises(ValueError, match="이미 취소"):
        monitor.cancel_job(key)
    monitor.slurm.rows = [job("CANCELLED")]
    monitor.poll()
    assert monitor.deliver_one()
    assert monitor.sent[0][1].startswith(before["prompt"] + "\n\n---")
    assert '"state": "CANCELLED"' in monitor.sent[0][1]


def test_cancel_failure_does_not_change_prompt_or_mark_terminal(monitor):
    monitor.update_job(only(monitor), {"prompt": "저장됨", "armed": True})
    def fail(row):
        raise RuntimeError("응답 시간 초과")
    monitor.slurm.cancel = fail
    with pytest.raises(ValueError, match="시간 초과"):
        monitor.cancel_job(only(monitor))
    row = monitor.jobs[only(monitor)]
    assert row["cancel"]["status"] == "uncertain"
    assert not row["terminal"] and row["armed"] and row["prompt"] == "저장됨"
    assert not monitor.deliver_one()


def test_terminal_job_cannot_be_cancelled(monitor):
    monitor.slurm.rows = [job("COMPLETED")]
    monitor.poll()
    with pytest.raises(ValueError, match="실행 중"):
        monitor.cancel_job(only(monitor))
    assert not monitor.slurm.cancelled


@pytest.mark.parametrize("identifier", ["123", "123_2", "123+1"])
def test_scancel_only_runs_for_verified_owner_and_submission(identifier):
    commands = []
    def runner(args):
        commands.append(args)
        return identifier + "|me|RUNNING|2026-10-24T12:00:00\n" if args[0] == "squeue" else ""
    slurm.Slurm("me", runner=runner).cancel(job(identifier=identifier))
    assert commands[-1] == ["scancel", "--user", "me", identifier]


@pytest.mark.parametrize("reply", ["", "123|other|RUNNING|2026-10-24T12:00:00\n",
    "123|me|RUNNING|2026-10-24T13:00:00\n", "123|me|COMPLETED|2026-10-24T12:00:00\n"])
def test_scancel_rejects_missing_foreign_reused_or_ended_job(reply):
    commands = []
    def runner(args):
        commands.append(args)
        return reply
    with pytest.raises(ValueError):
        slurm.Slurm("me", runner=runner).cancel(job())
    assert all(args[0] != "scancel" for args in commands)


@pytest.mark.parametrize("identifier", ["--user=other", "123;touch /tmp/x", "123_[1-5]"])
def test_scancel_rejects_options_shell_text_and_collapsed_arrays(identifier):
    with pytest.raises(ValueError):
        slurm.Slurm("me", runner=lambda args: pytest.fail("must not run")).cancel(job(identifier=identifier))


def test_armed_instructions_and_receipts_survive_upgrade_restart(monitor):
    key = only(monitor)
    monitor.update_job(key, {"prompt": "기존\n종료 후 지시", "armed": True, "output_label": "이름 유지"})
    before = json.loads(monitor.store.path.read_text())
    monitor.close()
    new = Monitor(monitor.store.directory, monitor.config["project_root"], slurm=monitor.slurm, sender=monitor.sender)
    try:
        new.poll()
        for field in ("prompt", "armed", "armed_for_completion", "output_label", "delivery", "events"):
            assert new.jobs[key][field] == before["jobs"][key][field]
        assert new.config == before["config"]
        assert not new.deliver_one()
    finally:
        new.close()


def test_training_metrics_keep_abbreviations_and_exact_checkpoint_steps():
    assert slurm.training_steps("schedule : steps=100000\nINFO step:63.2K smpl:1M loss:0.1\n") == (63200, 100000, True)
    assert slurm.training_steps("INFO step:63,200 smpl:1M\n") == (63200, None, False)
    assert slurm.training_steps("INFO step:60K smpl:1M\nCheckpoint policy after step 60000\n") == (60000, None, False)
    assert slurm.training_steps("eval task step:100\n") == (None, None, False)


def test_training_checkpoint_fallback_and_rounded_live_log(tmp_path):
    checkpoint = tmp_path / "model/checkpoints/last"
    (checkpoint / "pretrained_model").mkdir(parents=True)
    (checkpoint / "training_state").mkdir()
    (checkpoint / "pretrained_model/train_config.json").write_text('{"steps": 100000}')
    (checkpoint / "training_state/training_step.json").write_text('{"step": 50000}')
    row = {**job(), "work_dir": "/repo/PT/VSA", "outputs": [str(tmp_path / "model")]}
    progress = slurm.training_progress(row, {})[0]
    assert (progress["step"], progress["total"], progress["approximate"]) == (50000, 100000, False)
    progress = slurm.training_progress(row, {"/log": "step:55K smpl:880K"})[0]
    assert (progress["step"], progress["approximate"]) == (55000, True)
    (checkpoint / "training_state/training_step.json").write_text('{incomplete')
    assert slurm.training_progress(row, {})[0]["step"] is None
    assert slurm.training_progress({**row, "job_name": "S1eval"}, {}) == []
    assert slurm.training_progress({**row, "work_dir": "/repo/PT/eval"}, {}) == []


def test_grouped_training_models_use_their_own_logs(tmp_path):
    a, b = str(tmp_path / "A"), str(tmp_path / "B")
    (tmp_path / "a.err").write_text("step:12K smpl:192K\n")
    (tmp_path / "b.err").write_text("step:9K smpl:144K\n")
    text = (f"schedule: steps=100000\nSTART a output={a} logs={tmp_path}/a.{{out,err}}\n"
            f"START b output={b} logs={tmp_path}/b.{{out,err}}\n")
    rows = slurm.training_progress({**job(), "outputs": [a, b]}, {"parent.out": text})
    assert [row["step"] for row in rows] == [12000, 9000]
    assert all(row["total"] == 100000 for row in rows)
