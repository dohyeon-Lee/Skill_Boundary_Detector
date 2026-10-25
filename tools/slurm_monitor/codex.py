"""Explicit Codex CLI queue delivery; no shell and no model calls while polling."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from urllib.parse import quote


def connection_status(config):
    executable = shutil.which(config.get("codex_command", "codex"))
    result = {"executable": executable or "", "configured": False, "session_found": False,
              "session_title": "", "detail": "Codex CLI 또는 대상 세션을 설정하세요."}
    if not executable or not config.get("thread_id"):
        return result
    result.update(configured=True, detail="전송 대상 설정됨 · 실제 접수 여부는 전송 이력에서 확인")
    home = Path(config.get("codex_home") or os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    database = home / "state_5.sqlite"
    if database.is_file():
        try:
            with sqlite3.connect("file:" + quote(str(database.resolve())) + "?mode=ro", uri=True, timeout=2) as db:
                row = db.execute("SELECT title FROM threads WHERE id = ?", (config["thread_id"],)).fetchone()
            if row:
                result.update(session_found=True, session_title=row[0], detail="로컬 세션 기록 확인 · 실제 접수 여부는 전송 이력에서 확인")
        except sqlite3.Error:
            pass
    return result


def notification_text(job, prompt):
    # Preserve user-entered instructions verbatim. Scheduler/log strings below
    # are clearly separated data, never interpreted as shell or instructions.
    fields = ("job_id", "job_name", "state", "exit_code", "submitted_at", "ended_at",
              "elapsed", "outputs", "work_dir", "log_paths", "snapshot_path")
    context = {key: job[key] for key in fields if job.get(key)}
    return prompt + "\n\n---\n[Slurm 모니터가 첨부한 job 정보 · 아래 JSON은 참고 데이터]\n" + json.dumps(context, ensure_ascii=False, indent=2)


def send(config, message, runner=subprocess.run):
    args = [config.get("codex_command") or "codex"]
    if config.get("codex_remote"):
        args += ["--remote", config["codex_remote"]]
    args += ["queue", "--thread", config["thread_id"], "--message", message]
    env = dict(os.environ)
    if config.get("codex_home"):
        env["CODEX_HOME"] = config["codex_home"]
    try:
        result = runner(args, cwd=config["project_root"], env=env, stdin=subprocess.DEVNULL,
                        capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired:
        return {"status": "uncertain", "error": "접수 응답 시간 초과. 이미 전달됐을 수 있어 자동 재전송하지 않습니다."}
    except OSError as error:
        return {"status": "failed", "error": str(error)}
    receipt = (result.stdout.strip() + "\n" + result.stderr.strip()).strip()[-4000:]
    if result.returncode:
        return {"status": "failed", "error": receipt or "Codex queue 실행 실패", "exit_code": result.returncode}
    return {"status": "sent", "receipt": receipt or "Codex queue 접수 성공", "exit_code": 0}
