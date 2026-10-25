"""Read the current user's Slurm allocations and discover their output folders."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}
FIELDS = ["job_id", "job_name", "state", "exit_code", "elapsed", "time_limit",
          "submitted_at", "started_at", "ended_at", "work_dir", "user"]
CONCRETE_JOB_ID = re.compile(r"\d+(?:_\d+)?(?:\+\d+)?")


def state_name(value):
    return value.split()[0].rstrip("+") if value else "UNKNOWN"


def job_key(job):
    return job["job_id"] + "@" + job.get("submitted_at", "")


def run(args, timeout=15, **kwargs):
    try:
        result = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(args[0] + " 응답 시간 초과") from error
    if result.returncode:
        raise RuntimeError((result.stderr.strip() or args[0] + " 실행 실패")[-1500:])
    return result.stdout


def parse_accounting(text, user):
    jobs = {}
    for line in text.splitlines():
        cells = line.split("|")
        if len(cells) != len(FIELDS):
            continue
        job = dict(zip(FIELDS, cells))
        if job["user"] != user or not re.fullmatch(r"\d+(?:_[\d\[\],:%-]+)?(?:\+\d+)?", job["job_id"]):
            continue
        job["state_detail"] = job["state"]
        job["state"] = state_name(job["state"])
        job["terminal"] = job["state"] in TERMINAL
        jobs[job_key(job)] = job
    return jobs


def parse_queue(text):
    jobs = []
    for line in text.splitlines():
        cells = line.split("|", 7)
        if len(cells) != 8:
            continue
        identifier, state, name, elapsed, limit, reason, submit, start = cells
        if not re.fullmatch(r"\d+(?:_\d+)?(?:\+\d+)?", identifier):
            continue
        jobs.append({"job_id": identifier, "state": state_name(state), "state_detail": state,
                     "job_name": name, "elapsed": elapsed, "time_limit": limit,
                     "reason": reason, "submitted_at": submit, "started_at": start,
                     "terminal": False})
    return jobs


def parse_details(text):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:]*)=(.*?)(?=\s+[A-Za-z][A-Za-z0-9_:]*=|$)", text.strip()))


def read_log(path, limit=131072):
    try:
        with Path(path).open("rb") as stream:
            head = stream.read(limit)
            stream.seek(0, 2)
            size = stream.tell()
            if size > limit:
                stream.seek(max(limit, size - limit))
                head += b"\n" + stream.read(limit)
        return head.decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def output_paths(text):
    # Grouped jobs log an output directory for each independently trained model.
    grouped = re.findall(r"^START\s+[^\n]*?\boutput=(/\S+)", text, re.MULTILINE)
    plain = re.findall(r"^(?:output\s*:\s*|DONE\s*->\s*)(/[^\r\n]+)", text, re.MULTILINE)
    return list(dict.fromkeys(path.strip() for path in (grouped or plain)))


def expand_log_path(path, job):
    identifier = job["job_id"]
    parent, _, index = identifier.partition("_")
    for key, value in {"%j": identifier, "%A": parent, "%a": index,
                       "%x": job["job_name"], "%u": job.get("user", "")}.items():
        path = path.replace(key, value)
    return path


def snapshot_output(work_dir, submitted_at):
    """Use a unique nearby frozen evaluation config, never today's edited YAML."""
    try:
        submitted = datetime.fromisoformat(submitted_at)
        root = Path(work_dir) / ".config_snapshots"
        candidates = []
        for bundle in root.glob("*.yaml.*"):
            match = re.search(r"\.yaml\.(\d{8}-\d{6})_", bundle.name)
            if not match:
                continue
            difference = (submitted - datetime.strptime(match[1], "%Y%m%d-%H%M%S")).total_seconds()
            if not 0 <= difference <= 5:
                continue
            for config in bundle.glob("*.yaml"):
                if config.name == "global_config.yaml":
                    continue
                text = read_log(config, 65536)
                found = re.search(r"^output_name:\s*['\"]?([^'\"\n#]+)", text, re.MULTILINE)
                if found and found[1].strip():
                    candidates.append((difference, found[1].strip(), str(config)))
        if not candidates:
            return None
        closest = min(row[0] for row in candidates)
        matches = [row for row in candidates if row[0] == closest]
        if len({row[1] for row in matches}) != 1:
            return None
        _, name, config = matches[0]
        return str(Path(work_dir) / "outputs" / name), config
    except (OSError, ValueError, TypeError):
        return None


def training_steps(text):
    """Read training metrics, preserving the precision of abbreviated K/M logs."""
    target = re.findall(r"^schedule\s*:\s*steps\s*=\s*(\d+)", text, re.MULTILINE)
    total = int(target[-1]) if target else None
    steps = list(re.finditer(
        r"\bstep:\s*([\d,]+(?:\.\d+)?)([KMB]?)\s+smpl:"
        r"|Checkpoint policy after step\s+(\d+)\b", text))
    if not steps:
        return None, total, False
    match = steps[-1]
    if match[3]:
        return int(match[3]), total, False
    scale = {"": 1, "K": 1000, "M": 1000000, "B": 1000000000}[match[2]]
    return int(float(match[1].replace(",", "")) * scale), total, bool(match[2])


def small_json(path):
    try:
        with Path(path).open("rb") as stream:
            return json.loads(stream.read(1048576))
    except (OSError, ValueError):
        return {}


def training_progress(job, logs):
    """Only read logs and tiny checkpoint metadata; never load model tensors."""
    if "eval" in job.get("job_name", "").lower() or any(
            part.lower() == "eval" for part in Path(job.get("work_dir", "")).parts):
        return []
    combined = "\n".join(logs.values())
    detected = (re.search(r"^schedule\s*:\s*steps=|\bstep:\s*[\d.,]+[KMB]?\s+smpl:", combined, re.MULTILINE)
                or any(part in {"VSA", "Predictor", "Terminator"} for part in Path(job.get("work_dir", "")).parts))
    if not detected:
        return []
    parent_total = training_steps(combined)[1]
    children = {}
    for output, pattern in re.findall(r"^START\s+[^\n]*?\boutput=(/\S+)\s+logs=(/\S+)", combined, re.MULTILINE):
        children[output] = ([pattern.replace("{out,err}", suffix) for suffix in ("out", "err")]
                            if "{out,err}" in pattern else [pattern])
    progress = []
    outputs = job.get("outputs") or [""]
    for output in outputs[:20]:
        # Grouped jobs have separate child logs, and must not share one step count.
        selected = ({path: read_log(path) for path in children.get(output, [])}
                    if len(outputs) > 1 else logs)
        candidates, total = [], None
        for path, text in selected.items():
            step, goal, approximate = training_steps(text)
            total = goal or total
            if step is not None:
                candidates.append((step, approximate, path))
        checkpoint = Path(output) / "checkpoints" / "last" if output else None
        if checkpoint:
            config = small_json(checkpoint / "pretrained_model" / "train_config.json")
            goal = config.get("steps") if isinstance(config, dict) else None
            if type(goal) is int and goal > 0:
                total = goal
            step_path = checkpoint / "training_state" / "training_step.json"
            record = small_json(step_path)
            step = record.get("step") if isinstance(record, dict) else None
            if type(step) is int and step >= 0:
                candidates.append((step, False, str(step_path)))
        total = total or parent_total
        latest = max(candidates, key=lambda row: (row[0], row[1])) if candidates else (None, False, "")
        step, approximate, source = latest
        try:
            updated = datetime.fromtimestamp(Path(source).stat().st_mtime, timezone.utc).isoformat(timespec="seconds") if source else None
        except OSError:
            updated = None
        progress.append({"output": output, "step": step, "total": total,
                         "approximate": approximate, "source": source, "updated_at": updated})
    return progress


class Slurm:
    def __init__(self, user, days=3, runner=run):
        self.user, self.days, self.run = user, days, runner
        self.details = {}

    def collect(self, tracked=()):
        # Both calls must succeed. Losing a queue entry is never evidence of completion.
        queue = parse_queue(self.run(["squeue", "-u", self.user, "--array", "--noheader",
            "--format=%i|%T|%j|%M|%l|%R|%V|%S"]))
        start = (datetime.now() - timedelta(days=self.days)).strftime("%Y-%m-%dT%H:%M:%S")
        args = ["sacct", "-u", self.user, "-X", "-S", start, "--parsable2", "--noheader",
                "--format=JobID%64,JobName%160,State%50,ExitCode,Elapsed,Timelimit,Submit,Start,End,WorkDir%512,User"]
        jobs = parse_accounting(self.run(args), self.user)
        # An old armed job can outlive the rolling history window. Query it explicitly.
        missing = [job["job_id"] for job in tracked if not job.get("terminal") and job_key(job) not in jobs]
        if missing:
            extra = ["sacct", "-u", self.user, "-X", "-S", "2020-01-01", "-j",
                     ",".join(missing[:200]), "--parsable2", "--noheader", args[-1]]
            jobs.update(parse_accounting(self.run(extra), self.user))
        for current in queue:
            key = job_key(current)
            jobs[key] = {**jobs.get(key, {}), **current, "user": self.user}
        return jobs

    def cancel(self, job):
        identifier = job["job_id"]
        if not CONCRETE_JOB_ID.fullmatch(identifier):
            raise ValueError("개별 job ID만 취소할 수 있습니다.")
        # Recheck owner, submission identity and live status immediately before
        # scancel, rather than trusting a browser value or an old saved record.
        text = self.run(["squeue", "--jobs", identifier, "--array", "--noheader", "--format=%i|%u|%T|%V"])
        row = next((line.split("|") for line in text.splitlines()
                    if len(line.split("|")) == 4 and line.split("|")[0] == identifier), None)
        if not row or row[1] != self.user or row[3] != job.get("submitted_at"):
            raise ValueError("현재 내 job과 일치하지 않습니다. 새로고침 후 확인하세요.")
        if state_name(row[2]) in TERMINAL:
            raise ValueError("이미 종료된 job입니다.")
        self.run(["scancel", "--user", self.user, identifier])

    def enrich(self, job):
        job = dict(job)
        key = job_key(job)
        detail = self.details.get(key, {})
        if not detail and not job["terminal"] and "[" not in job["job_id"]:
            try:
                detail = parse_details(self.run(["scontrol", "show", "job", "-o", job["job_id"]]))
                if detail:
                    self.details[key] = detail
            except (RuntimeError, OSError):
                pass
        work_dir = detail.get("WorkDir") or job.get("work_dir", "")
        job["work_dir"] = work_dir
        paths = []
        for field in ("StdOut", "StdErr"):
            if detail.get(field):
                paths.append(expand_log_path(detail[field], job))
        if work_dir:
            parent, _, index = job["job_id"].partition("_")
            names = {job["job_name"] + "_" + job["job_id"], job["job_name"] + "_" + parent + "_" + index}
            paths += [str(Path(work_dir) / "logs" / (name + suffix))
                      for name in names for suffix in (".out", ".err")]
        job["log_paths"] = list(dict.fromkeys(paths))
        logs = {path: read_log(path) for path in job["log_paths"]}
        outputs = []
        for text in logs.values():
            outputs += output_paths(text)
        if outputs:
            job["outputs"] = list(dict.fromkeys(outputs))
            job["output_source"] = "로그"
        elif work_dir and not job.get("outputs"):
            candidate = snapshot_output(work_dir, job.get("submitted_at", ""))
            if candidate:
                path, config = candidate
                job.update(outputs=[path], output_source="제출 시각과 일치하는 설정 스냅샷", snapshot_path=config)
        job["training_progress"] = training_progress(job, logs)
        return job
