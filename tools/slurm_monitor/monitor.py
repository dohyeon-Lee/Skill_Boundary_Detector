"""Persistent Slurm monitor and a single, explicit completion-message outbox."""
from __future__ import annotations

import fcntl
import getpass
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

if __package__:
    from . import codex, training
    from .slurm import CONCRETE_JOB_ID, Slurm, job_key
else:
    import codex
    import training
    from slurm import CONCRETE_JOB_ID, Slurm, job_key


DEFAULT_COMPLETION_PROMPT = (
    "실패 시 원인 분석 후 해결한 뒤 이어학습 혹은 새로 학습할 수 있도록 재제출 "
    "(생성물이 방해되면 생성물도 삭제후 재제출). 저장공간부족으로 인한실패일 경우 재제출 금지."
)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class State:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_file = (self.directory / "monitor.lock").open("a+")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_file.close()
            raise RuntimeError("같은 상태 폴더의 모니터가 이미 실행 중입니다.") from None
        self.path = self.directory / "state.json"

    def load(self):
        if not self.path.exists():
            return {"version": 1, "config": {}, "jobs": {}}
        data = json.loads(self.path.read_text())
        if data.get("version") != 1 or not isinstance(data.get("jobs"), dict):
            raise ValueError("모니터 상태 파일을 읽을 수 없습니다. 원본을 보존하고 상태 폴더를 확인하세요.")
        return data

    def save(self, data):
        temporary = self.path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)

    def close(self):
        self.lock_file.close()


class Monitor:
    def __init__(self, directory, project_root, *, slurm=None, sender=codex.send, defaults=None):
        self.store = State(directory)
        try:
            self.data = self.store.load()
        except Exception:
            self.store.close()
            raise
        default = {"project_root": str(Path(project_root).resolve()), "user": getpass.getuser(),
                   "interval": 30, "history_days": 3, "thread_id": os.environ.get("CODEX_THREAD_ID", ""),
                   "codex_home": os.environ.get("CODEX_HOME", str(Path.home() / ".codex")),
                   "codex_command": "codex", "codex_remote": "",
                   "completion_prompt": DEFAULT_COMPLETION_PROMPT, "completion_armed": True}
        self.data["config"] = {**default, **self.data.get("config", {}), **(defaults or {})}
        self.data.setdefault("trainings", {})
        self.slurm = slurm or Slurm(self.data["config"]["user"], self.data["config"]["history_days"])
        self.sender = sender
        self.lock = threading.RLock()
        self.refresh = threading.Event()
        self.stopping = threading.Event()
        self.dispatch = threading.Event()
        self.threads = []
        self.poll_error = ""
        self.updated_at = None
        self.polling = False
        self.connection = codex.connection_status(self.config)
        for job in self.records.values():
            if job.get("delivery", {}).get("status") == "sending":
                job["delivery"].update(status="uncertain", error="전송 중 모니터가 재시작됨. 중복 방지를 위해 접수 여부 확인 후 수동 재전송하세요.")
            if job.get("cancel", {}).get("status") == "requesting":
                job["cancel"].update(status="uncertain", error="취소 요청 중 재시작됨. Slurm 상태를 먼저 확인하세요.")
        self.save()

    @property
    def config(self):
        return self.data["config"]

    @property
    def jobs(self):
        return self.data["jobs"]

    @property
    def records(self):
        return {**self.jobs, **self.data["trainings"]}

    def save(self):
        self.store.save(self.data)

    def completion_defaults(self, row):
        row.setdefault("prompt", self.config["completion_prompt"])
        row.setdefault("armed", self.config["completion_armed"])
        # Newly discovered history is shown with the preset, but only an
        # observed active attempt is eligible for automatic completion delivery.
        row.setdefault("armed_for_completion", bool(row["armed"] and not row["terminal"]))
        row.setdefault("events", [])
        row.setdefault("delivery", {"status": "none"})

    def snapshot(self):
        with self.lock:
            jobs = []
            for key, job in self.records.items():
                if job.get("container"):
                    continue
                row = {**job, "key": key}
                outputs = row.get("outputs", [])
                row["display_name"] = row.get("output_label") or (
                    Path(outputs[0]).name + (f" 외 {len(outputs) - 1}개" if len(outputs) > 1 else "")
                    if outputs else row["job_name"] + " · 출력 폴더 확인 중")
                waiting = row["state"] in {"PENDING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD"}
                row["group"] = "ended" if row.get("terminal") else "pending" if waiting else "running"
                if row.get("kind") == "training":
                    row["group"] = "training-ended" if row["terminal"] else "training-running"
                row["can_cancel"] = bool(row.get("kind") != "training" and not row.get("terminal") and CONCRETE_JOB_ID.fullmatch(row["job_id"]))
                jobs.append(row)
            jobs.sort(key=lambda row: (row.get("ended_at", "") if row["terminal"] else row.get("submitted_at", ""), row["job_id"]), reverse=True)
            return json.loads(json.dumps({"jobs": jobs, "updated_at": self.updated_at,
                "polling": self.polling, "error": self.poll_error, "config": self.config,
                "connection": self.connection}, ensure_ascii=False))

    def poll(self):
        with self.lock:
            if self.polling:
                return
            self.polling = True
            # Status tracking is independent of whether completion instructions
            # are armed. Otherwise old unarmed jobs stay RUNNING forever.
            tracked = [dict(job) for job in self.jobs.values() if not job.get("terminal")]
        try:
            found = self.slurm.collect(tracked)
            # Enrich only active jobs and recent history. Previously discovered
            # metadata stays in state after Slurm purges the completed record.
            ordered = sorted(found.items(), key=lambda pair: (not pair[1]["terminal"], pair[1].get("submitted_at", "")), reverse=True)
            enriched = {}
            for key, job in ordered[:300]:
                if self.stopping.is_set():
                    return
                with self.lock:
                    previous = self.jobs.get(key, {})
                enriched[key] = self.slurm.enrich({**previous, **job})
            with self.lock:
                for key, job in enriched.items():
                    if key not in self.jobs:
                        # Slurm can reset SubmitTime on requeue. Carry an armed
                        # still-active allocation forward instead of losing its prompt.
                        old_key = next((k for k, value in self.jobs.items()
                            if value["job_id"] == job["job_id"] and not value.get("terminal")), None)
                        previous = self.jobs.pop(old_key) if old_key else {}
                        # A cancellation request belongs to that submission,
                        # while completion instructions survive a requeue.
                        if old_key:
                            previous.pop("cancel", None)
                    else:
                        previous = self.jobs[key]
                    owned = {k: previous[k] for k in ("prompt", "armed", "armed_for_completion", "output_label", "delivery", "events", "cancel") if k in previous}
                    row = {**job, **owned}
                    self.completion_defaults(row)
                    if row["terminal"] and row["armed"] and row.get("armed_for_completion") and row["prompt"].strip() and row["delivery"]["status"] == "none":
                        row["delivery"] = {"status": "waiting", "queued_at": now()}
                    self.jobs[key] = row
                self.poll_error = ""
                self.updated_at = now()
                self.save()
                self.dispatch.set()
        except Exception as error:
            with self.lock:
                self.poll_error = str(error)
        finally:
            with self.lock:
                try:
                    found_training = training.collect(self.config["project_root"], self.jobs.values(),
                        self.data["trainings"], self.config["history_days"])
                    for key, row in found_training.items():
                        previous = self.data["trainings"].get(key, {})
                        owned = {k: previous[k] for k in ("prompt", "armed", "armed_for_completion", "output_label", "delivery", "events") if k in previous}
                        row.update(owned)
                        self.completion_defaults(row)
                        if row["terminal"] and row["armed"] and row.get("armed_for_completion") and row["prompt"].strip() and row["delivery"]["status"] == "none":
                            row["delivery"] = {"status": "waiting", "queued_at": now()}
                        self.data["trainings"][key] = row
                    self.save()
                    self.dispatch.set()
                except Exception as error:
                    self.poll_error = (self.poll_error + "\n" if self.poll_error else "") + "학습 상태 조회 실패: " + str(error)
                self.polling = False

    def configure(self, values):
        allowed = {"interval", "thread_id", "codex_home", "codex_command", "codex_remote"}
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError("지원하지 않는 설정입니다.")
        candidate = {**self.config, **values}
        if type(candidate["interval"]) is not int or not 10 <= candidate["interval"] <= 3600:
            raise ValueError("확인 주기는 10~3600초입니다.")
        if not isinstance(candidate["thread_id"], str) or (candidate["thread_id"] and not re.fullmatch(r"[0-9a-fA-F-]{36}", candidate["thread_id"])):
            raise ValueError("대상 Codex 세션 UUID를 입력하세요.")
        for key in ("codex_home", "codex_command", "codex_remote"):
            if not isinstance(candidate[key], str) or len(candidate[key]) > 4096 or "\x00" in candidate[key]:
                raise ValueError("설정 값이 올바르지 않습니다: " + key)
        remote = candidate["codex_remote"]
        if remote and not remote.startswith(("unix://", "ws://", "wss://")):
            raise ValueError("Codex 서버 주소는 unix://, ws://, wss:// 형식입니다.")
        with self.lock:
            if any(job.get("delivery", {}).get("status") == "sending" for job in self.records.values()):
                raise ValueError("전송 중에는 대상 연결을 바꿀 수 없습니다.")
            self.data["config"] = candidate
            self.connection = codex.connection_status(candidate)
            self.save()
            self.refresh.set()
            self.dispatch.set()

    def update_job(self, key, values):
        if not isinstance(values, dict) or set(values) - {"prompt", "armed", "output_label"}:
            raise ValueError("job 설정이 올바르지 않습니다.")
        if "prompt" in values and (not isinstance(values["prompt"], str) or len(values["prompt"]) > 32000 or "\x00" in values["prompt"]):
            raise ValueError("지시는 32,000자 이하 텍스트여야 합니다.")
        if "armed" in values and type(values["armed"]) is not bool:
            raise ValueError("자동 전송은 true/false여야 합니다.")
        if "output_label" in values and (not isinstance(values["output_label"], str) or len(values["output_label"]) > 512):
            raise ValueError("표시 이름은 512자 이하입니다.")
        with self.lock:
            if key not in self.records:
                raise ValueError("job을 찾을 수 없습니다.")
            job = self.records[key]
            if job["delivery"]["status"] == "sending":
                raise ValueError("전송 중입니다. 접수 결과를 확인한 뒤 수정하세요.")
            if values.get("armed") and not values.get("prompt", job["prompt"]).strip():
                raise ValueError("종료 후 실행할 지시를 먼저 입력하세요.")
            # Saving a historical job's prompt never silently sends it. Use its
            # explicit Send button instead. Active jobs can be armed for later.
            job.update(values)
            if "armed" in values:
                job["armed_for_completion"] = bool(job["armed"] and not job["terminal"])
            if not job["armed"] and job["delivery"]["status"] == "waiting":
                job["delivery"] = {"status": "none"}
            self.save()

    def request_send(self, key):
        with self.lock:
            if key not in self.records:
                raise ValueError("job을 찾을 수 없습니다.")
            job = self.records[key]
            if not job.get("terminal"):
                raise ValueError("종료 상태가 확인된 job만 지금 전송할 수 있습니다.")
            if not job["prompt"].strip():
                raise ValueError("지시를 저장하세요.")
            if job["delivery"]["status"] in {"sent", "sending", "waiting"}:
                raise ValueError("이미 접수됐거나 전송 중인 job입니다.")
            if not self.connection["configured"]:
                raise ValueError("Codex CLI와 대상 세션을 먼저 설정하세요.")
            job["delivery"] = {"status": "waiting", "queued_at": now()}
            self.save()
            self.dispatch.set()

    def cancel_job(self, key):
        with self.lock:
            if key not in self.jobs or self.jobs[key].get("terminal"):
                raise ValueError("실행 중이거나 대기 중인 job만 취소할 수 있습니다.")
            job = self.jobs[key]
            if not CONCRETE_JOB_ID.fullmatch(job["job_id"]):
                raise ValueError("개별 job ID만 취소할 수 있습니다.")
            if job.get("cancel", {}).get("status") in {"requesting", "requested"}:
                raise ValueError("이미 취소를 요청했습니다. Slurm 상태 확인을 기다리세요.")
            target = dict(job)
            job["cancel"] = {"status": "requesting", "requested_at": now()}
            self.save()
        try:
            self.slurm.cancel(target)
        except Exception as error:
            with self.lock:
                if key in self.jobs:
                    self.jobs[key]["cancel"].update(status="uncertain", error=str(error))
                    self.save()
                self.refresh.set()
            raise ValueError(str(error)) from error
        with self.lock:
            if key in self.jobs:
                self.jobs[key]["cancel"].update(status="requested", accepted_at=now())
                self.save()
            # A successful scancel request is not terminal-state evidence.
            self.refresh.set()

    def deliver_one(self):
        with self.lock:
            if not self.connection["configured"] or self.stopping.is_set():
                return False
            candidate = next(((key, job) for key, job in self.records.items()
                if not job.get("container") and job.get("terminal") and job["delivery"]["status"] == "waiting"), None)
            if not candidate:
                return False
            key, job = candidate
            identifier = str(uuid.uuid4())
            message = codex.notification_text(job, job["prompt"])
            config = dict(self.config)
            job["delivery"].update(status="sending", attempt_id=identifier, started_at=now(),
                thread_id=config["thread_id"], prompt=job["prompt"])
            self.save()  # Persist BEFORE calling the external queue command.
        try:
            result = self.sender(config, message)
        except Exception as error:
            result = {"status": "uncertain", "error": str(error)}
        with self.lock:
            job = self.records[key]
            job["delivery"].update(result, finished_at=now())
            job["events"].append(dict(job["delivery"]))
            job["events"] = job["events"][-20:]
            self.save()
        return True

    def start(self):
        def polling():
            while not self.stopping.is_set():
                self.poll()
                self.refresh.wait(self.config["interval"])
                self.refresh.clear()

        def sending():
            while not self.stopping.is_set():
                self.dispatch.wait(1)
                self.dispatch.clear()
                while self.deliver_one():
                    pass

        for function in (polling, sending):
            thread = threading.Thread(target=function, daemon=True)
            thread.start()
            self.threads.append(thread)

    def close(self):
        self.stopping.set()
        self.refresh.set()
        self.dispatch.set()
        deadline = time.monotonic() + 50
        for thread in self.threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        self.store.close()
