"""Read-only, stdlib-only probe sent to a Pod over SSH (nothing is installed)."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

MAX_JSON = 8 * 1024 * 1024
MAX_CHART = 6 * 1024 * 1024


def bounded(path, limit=MAX_JSON):
    if path.stat().st_size > limit:
        raise ValueError("file too large")
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("file too large")
    return value


def percentage(metrics):
    successes = metrics.get("successes")
    if isinstance(successes, list) and successes:
        value = 100 * sum(bool(item) for item in successes) / len(successes)
    elif metrics.get("pc_success") is not None:
        value = float(metrics["pc_success"])  # pc_success is already a percentage.
    elif metrics.get("success_rate") is not None:
        value = 100 * float(metrics["success_rate"])
    else:
        return None
    return value if math.isfinite(value) and 0 <= value <= 100 else None


def summarize(value):
    models = []
    # SkillVLA merged output is keyed by model/panel; standard eval is unkeyed.
    if "per_task" in value:
        value = {"eval": value}
    for label, info in value.items():
        if not isinstance(info, dict) or not isinstance(info.get("per_task"), list):
            continue
        tasks = []
        for task in info["per_task"]:
            metrics = task.get("metrics", task)
            rate = percentage(metrics)
            tasks.append({"name": str(task.get("task_name") or
                f"{task.get('task_group', 'task')}_{task.get('task_id', '?')}"),
                "success_pct": rate})
        overall = percentage(info.get("overall", {}))
        known = [task["success_pct"] for task in tasks if task["success_pct"] is not None]
        models.append({"label": str(label), "overall_pct": overall,
                       "tasks": tasks, "task_count": len(tasks),
                       "task_mean_pct": sum(known) / len(known) if known else None})
    return models


def job_state(base):
    try:
        code = base.with_suffix(".exit").read_text().strip()
        return {"0": "done", "dependency": "cancelled", "143": "stopped"}.get(code, "failed"), code
    except FileNotFoundError:
        pass
    try:
        pid = int(base.with_suffix(".pid").read_text())
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        marker = base.with_suffix(".pidstart")
        if stat[0] != "Z" and (not marker.exists() or marker.read_text().strip() == stat[19]):
            return ("pending" if base.with_suffix(".pending").exists() else "running"), None
    except (OSError, ValueError, IndexError):
        pass
    return "stopped", None


def job_timing(base, started, state):
    """Freeze completed duration at the exit marker, including missed polls."""
    finished = None
    try:
        end = datetime.fromtimestamp(base.with_suffix(".exit").stat().st_mtime, timezone.utc)
        finished = end.isoformat(timespec="seconds")
    except FileNotFoundError:
        end = datetime.now(timezone.utc) if state in {"running", "pending"} else None
    try:
        start = datetime.fromisoformat(started.replace("Z", "+00:00"))
        if start.tzinfo is None or end is None:
            raise ValueError("Unknown job timing")
        elapsed = (end - start).total_seconds()
        elapsed = elapsed if elapsed >= 0 else None
    except (ValueError, TypeError, AttributeError):
        elapsed = None
    return {"finished": finished, "elapsed_seconds": elapsed}


def batch_progress(text, episodes, batch_size):
    """Read only the outer episode-batch bar, never the rollout's step SR.

    tqdm writes a new postfix before advancing its counter. Keep the first
    observation at each advancing counter; a later same-counter postfix may
    already describe the following episode. Repeated 100% lines count once.
    """
    runs, previous = [], None
    expected_batches = math.ceil(episodes / batch_size)
    for match in re.finditer(r"Stepping through eval batches:[^\r\n]*?\|\s*(\d+)/(\d+)[^\r\n]*", text):
        n, total = int(match[1]), int(match[2])
        if total != expected_batches or n > total:
            continue
        if previous is None or n < previous:
            if n != 0:  # A truncated log cannot establish earlier task identity.
                raise ValueError("Incomplete evaluation progress log")
            runs.append({"completed": 0, "successes": None})
        if previous is None or n != previous:
            count = min(n * batch_size, episodes)
            rate = re.search(r"running_success_rate=([\d.]+)%", match[0])
            successes = None
            if rate and count:
                pct = float(rate[1])
                # Recover an exact integer only if the one-decimal rounded SR
                # permits one answer. Otherwise leave outcomes unknown.
                low = max(0, math.ceil((pct - .050001) * count / 100))
                high = min(count, math.floor((pct + .050001) * count / 100))
                if 0 <= pct <= 100 and low == high:
                    successes = low
            runs[-1] = {"completed": count, "successes": successes}
        previous = n
    return runs


def evaluation_progress(root, base, output):
    """Read a safe plan plus bounded worker logs; no raw logs/env leave Pod."""
    plan_path = base.with_suffix(".eval.json")
    if not plan_path.is_file():
        return None
    plan_path.resolve().relative_to(root)
    plan = json.loads(bounded(plan_path, 512 * 1024))
    if plan.get("schema") != 1 or str(plan.get("job_id")) != base.name:
        raise ValueError("Invalid evaluation progress plan")
    if Path(plan["output"]).resolve() != Path(output).resolve():
        raise ValueError("Evaluation progress output differs from job")
    episodes, batch = int(plan["episodes_per_task"]), int(plan["batch_size"])
    if not 1 <= episodes <= 100000 or not 1 <= batch <= 1024:
        raise ValueError("Invalid episode counts")
    panels = plan["panels"]
    if not panels or len(panels) > 100 or len(plan["workers"]) > 100:
        raise ValueError("Invalid progress plan size")
    tasks, updated, incomplete, budget = {}, 0, False, 64 * 1024 * 1024
    for worker in plan["workers"]:
        units = [(int(panel), int(task)) for panel in worker["panels"] for task in worker["tasks"]]
        if len(units) > 10000 or any(unit in tasks for unit in units) or len(set(units)) != len(units):
            raise ValueError("Duplicate or excessive task/panel work")
        runs = []
        log = Path(worker["log"]).resolve()
        log.relative_to(root)
        if log.is_file():
            size = log.stat().st_size
            if size > min(32 * 1024 * 1024, budget):
                incomplete = True
            else:
                budget -= size
                runs = batch_progress(bounded(log, 32 * 1024 * 1024).decode(errors="replace"), episodes, batch)
                updated = max(updated, log.stat().st_mtime)
        if len(runs) > len(units):
            raise ValueError("Progress log contains another evaluation")
        for index, unit in enumerate(units):
            run = runs[index] if index < len(runs) else {"completed": 0, "successes": None}
            tasks[unit] = {"name": f"{plan['suite']}_{unit[1]}", "task_id": unit[1],
                "completed_episodes": run["completed"], "total_episodes": episodes,
                "successes": run["successes"],
                "success_pct": 100 * run["successes"] / run["completed"]
                if run["successes"] is not None and run["completed"] else None,
                "state": "done" if run["completed"] == episodes else "running" if index < len(runs) else "pending"}

    def aggregate(rows):
        complete = sum(row["completed_episodes"] for row in rows)
        total = sum(row["total_episodes"] for row in rows)
        observed = sum(row["completed_episodes"] for row in rows if row["successes"] is not None)
        successes = sum(row["successes"] for row in rows if row["successes"] is not None)
        return {"completed_episodes": complete, "total_episodes": total,
            "percent": 100 * complete / total if total and not incomplete else None,
            "success_pct": 100 * successes / observed if observed else None,
            "successes": successes, "observed_episodes": observed,
            "completed_tasks": sum(row["state"] == "done" for row in rows), "total_tasks": len(rows)}

    models = []
    for panel in panels:
        rows = [row for (index, _), row in tasks.items() if index == int(panel["index"])]
        models.append({"label": str(panel["label"]), **aggregate(rows), "tasks": rows})
    return {**aggregate(list(tasks.values())), "models": models, "source": "worker_logs",
            "updated_at": updated, "incomplete": incomplete}


def training_progress(root, job, budget):
    """Observe existing trainers without modifying/restarting their processes.

    LeRobot's tqdm counter is relative to the resume step: its denominator is
    target minus resume step. Never treat that counter as an absolute step.
    Only numeric progress and validated model paths leave this probe.
    """
    def allowed(value):
        path = Path(value)
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        path.relative_to(root)
        return path

    def read_log(value):
        path = allowed(value)
        size = path.stat().st_size
        amount = min(size, 256 * 1024)
        if amount > budget[0]:
            raise ValueError("Training log budget exceeded")
        budget[0] -= amount
        with path.open("rb") as stream:
            head = stream.read(min(size, 64 * 1024))
            stream.seek(max(len(head), size - 192 * 1024))
            tail = stream.read(192 * 1024)
        clean = lambda data: re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", data.decode(errors="replace"))
        return clean(head), clean(tail), path.stat().st_mtime

    def logs(values):
        rows = []
        for value in dict.fromkeys(v for v in values if v):
            try:
                head, tail, updated = read_log(value)
                rows.append((head, tail, updated))
            except FileNotFoundError:
                continue
        return rows

    parent = logs([job.get("output"), job.get("error")])
    # Group supervisors log the exact child log stems. Read each independent
    # trainer, rather than interpreting interleaved counters as one model.
    stems = list(dict.fromkeys(match[1] for head, tail, _ in parent
        for match in re.finditer(r"^START \w+ pid=\d+[^\r\n]* logs=([^\r\n]+)\.\{out,err\}\s*$",
                                 head + "\n" + tail, re.M)))
    if len(stems) > 100:
        raise ValueError("Too many training workers")
    groups = [logs([stem + ".out", stem + ".err"]) for stem in stems] if stems else [parent]
    runs = []
    for group in groups:
        text = "\n".join(head + "\n" + tail for head, tail, _ in group)
        targets = re.findall(r"\bcfg\.steps\s*=\s*(\d+)|^\s*schedule\s*:\s*steps=(\d+)\b", text, re.M)
        bars = list(re.finditer(r"\bTraining:[^\r\n]*?\|\s*(\d+)/(\d+)\b", text))
        if not targets and not bars:
            continue
        outputs = re.findall(r"(?:\bOutput dir:|^\s*output\s*:)\s*([^\r\n]+)", text, re.M)
        output = allowed(outputs[-1].strip()) if outputs else None
        checkpoint, checkpoint_target = None, None
        if output:
            checkpoints = output / "checkpoints"
            last = checkpoints / "last"
            if last.exists():
                saved = allowed(last)
                # `last` is updated only after save_checkpoint returns. Numeric
                # folders by themselves may still be in the middle of a save.
                if saved.parent != checkpoints.resolve() or not saved.name.isdigit():
                    raise ValueError("Invalid last checkpoint")
                marker = allowed(saved / "training_state/training_step.json")
                checkpoint = json.loads(bounded(marker, 4096)).get("step")
                if type(checkpoint) is not int or checkpoint != int(saved.name):
                    raise ValueError("Invalid checkpoint step")
                config = allowed(saved / "pretrained_model/train_config.json")
                if config.is_file():
                    checkpoint_target = json.loads(bounded(config, 512 * 1024)).get("steps")
        # A checkpoint's config can be newer than the bounded log header when
        # an appended/resumed log has grown beyond the tail window.
        tail_targets = re.findall(r"\bcfg\.steps\s*=\s*(\d+)", "\n".join(t for _, t, _ in group))
        total = int(tail_targets[-1]) if tail_targets else checkpoint_target or (int(next(v for v in targets[-1] if v)) if targets else None)
        if total is not None and (type(total) is not int or not 1 <= total <= 10**12):
            raise ValueError("Invalid training target")
        current = None
        if bars and total:
            done, remaining = map(int, bars[-1].groups())
            if 0 < remaining <= total and 0 <= done <= remaining:
                current = total - remaining + done
                if checkpoint is not None and current < checkpoint:
                    current = None  # An old header cannot identify this resumed session.
        if checkpoint is not None and total and checkpoint > total:
            raise ValueError("Checkpoint exceeds training target")
        runs.append({"label": output.name if output else job["name"],
            "completed_steps": current, "total_steps": total,
            "percent": 100 * current / total if current is not None and total else None,
            "latest_checkpoint": checkpoint, "source": "training_log",
            "updated_at": max((updated for _, _, updated in group), default=0)})
    if not runs:
        return None
    known = all(run["completed_steps"] is not None and run["total_steps"] for run in runs)
    complete = sum(run["completed_steps"] for run in runs) if known else None
    total = sum(run["total_steps"] for run in runs) if all(run["total_steps"] for run in runs) else None
    return {"completed_steps": complete, "total_steps": total,
        "percent": 100 * complete / total if known else None,
        "runs": runs, "updated_at": max(run["updated_at"] for run in runs)}


def snapshot(root, known=None):
    root = Path(root).expanduser().resolve()
    if not (root / "lerobot/examples/libero/configs").is_dir():
        raise ValueError("프로젝트 폴더가 없습니다. RunPod Desk의 Pod 프로젝트 경로를 확인하세요.")
    known = known or {}
    jobs, outputs, warnings = [], {}, []
    training_budget = [16 * 1024 * 1024]

    def allowed(path):
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("result outside project")
        return path

    folder = Path(os.environ.get("SBD_LOCAL_JOBS_DIR", root / ".cache/local_jobs"))
    for record in sorted(folder.glob("*.job"), key=lambda path: path.name, reverse=True)[:500]:
        try:
            fields = dict(line.split("=", 1) for line in bounded(record, 65536).decode().splitlines() if "=" in line)
            base = record.with_suffix("")
            state, code = job_state(base)
            job = {"id": base.name, "name": fields.get("name", base.name), "state": state,
                   "exit_code": code, "gpus": fields.get("gpus", ""), "started": fields.get("started", ""),
                   "cwd": fields.get("cwd", ""), "script": fields.get("script", ""),
                   "eval_output": fields.get("eval_output", ""), "output": fields.get("output", ""),
                   "error": fields.get("error", "")}
            job.update(job_timing(base, job["started"], state))
            jobs.append(job)
            candidates = [job["eval_output"]] if job["eval_output"] else []
            # Older records lack eval_output. Only extract output directory
            # markers from bounded log tails; raw logs/commands never leave Pod.
            if not candidates and "/eval/" in job["script"]:
                log = Path(job["output"])
                logs = [log]
                if log.name.endswith(".controller"):
                    logs += list(log.parent.glob(f"*{base.name}*"))[:100]
                for path in logs:
                    try:
                        with allowed(path).open("rb") as stream:
                            stream.seek(max(0, path.stat().st_size - 262144))
                            text = stream.read(262144).decode(errors="replace")
                        candidates += re.findall(r"^(?:output\s*:\s*|DONE ->\s*)([^\r\n]+)", text, re.M)
                    except (OSError, ValueError):
                        continue
            for value in candidates:
                out = allowed(value.strip())
                job["eval_output"] = str(out)
                outputs.setdefault(out, []).append(job)
            if job["eval_output"]:
                try:
                    progress = evaluation_progress(root, base, job["eval_output"])
                    if progress:
                        job["progress"] = progress
                except (OSError, ValueError, KeyError, TypeError):
                    warnings.append(f"평가 진행률 읽기 실패: {base.name}")
            elif "/eval/" not in job["script"]:
                try:
                    progress = training_progress(root, job, training_budget)
                    if progress:
                        job["training_progress"] = progress
                except (OSError, ValueError, KeyError, TypeError):
                    warnings.append(f"학습 진행률 읽기 실패: {base.name}")
        except (OSError, ValueError):
            warnings.append(f"잡 기록 읽기 실패: {record.name}")
    # Discover already completed runs even if the monitor first connected late.
    configs = root / "lerobot/examples/libero/configs"
    for work in configs.glob("train_*/**/eval"):
        for name in ("outputs", "attention_outputs"):
            base = work / name
            if base.is_dir():
                for metric in base.rglob("eval_info_merged.json"):
                    if metric.parent.name == "metrics":
                        try:
                            outputs.setdefault(allowed(metric.parent.parent), [])
                        except ValueError:
                            continue
    results, transfer_bytes = [], 0
    boundary_outputs = configs / "train_skills/skill_eval/skill_boundary_detection/outputs"
    if boundary_outputs.is_dir():
        for marker in boundary_outputs.rglob("evaluation_complete.json"):
            try:
                outputs.setdefault(allowed(marker.parent), [])
            except ValueError:
                continue
    for out, associated in sorted(outputs.items(), key=lambda item: str(item[0])):
        if len(results) >= 200:
            warnings.append("한 번에 최대 200개 평가 결과를 수집합니다.")
            break
        metric = out / "metrics/eval_info_merged.json"
        if not metric.is_file():
            metric = out / "metrics/eval_info.json"
        if not metric.is_file():
            metric = out / "evaluation_complete.json"
            if not metric.is_file():
                continue
        try:
            metric = allowed(metric)
            chart = allowed(out / "task_success_rates.png")
            raw = bounded(metric)
            # A retry can reuse the output directory of a failed earlier job.
            latest = max(associated, key=lambda job: (job["started"], job["id"])) if associated else None
            states = [latest["state"]] if latest else []
            phase = "final" if states and all(state == "done" for state in states) else (
                "partial" if states else "unverified")
            chart_stamp = [chart.stat().st_size, chart.stat().st_mtime_ns] if chart.is_file() else None
            signature = hashlib.sha256(raw + json.dumps([chart_stamp, phase]).encode()).hexdigest()
            identifier = hashlib.sha256(str(out).encode()).hexdigest()[:24]
            for job in associated:
                job["result_id"] = identifier
            if known.get(identifier) == signature:
                continue
            payload = json.loads(raw)
            boundary = payload.get("kind") == "skill_boundary_detection"
            models = [] if boundary else summarize(payload)
            if not models and not boundary:
                continue
            result = {"id": identifier, "fingerprint": signature, "output": str(out), "name": out.name,
                "phase": phase, "models": models, "job_ids": [job["id"] for job in associated],
                "mtime": metric.stat().st_mtime, "chart": None}
            if boundary:
                result.update(kind="skill_boundary_detection", thresholds=payload.get("thresholds", {}))
            if chart.is_file() and chart.stat().st_size <= MAX_CHART:
                payload = bounded(chart, MAX_CHART)
                if payload.startswith(b"\x89PNG\r\n\x1a\n"):
                    result["chart"] = base64.b64encode(payload).decode("ascii")
            size = len(json.dumps(result).encode())
            if results and transfer_bytes + size > 24 * 1024 * 1024:
                # The next poll sends the remaining changes, using known hashes.
                break
            transfer_bytes += size
            results.append(result)
        except (OSError, ValueError, TypeError, KeyError):
            warnings.append(f"평가 결과 읽기 실패: {out.name}")
    gpus = []
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], timeout=5, text=True, stderr=subprocess.DEVNULL)
        for row in text.strip().splitlines():
            index, name, total, used, util = [part.strip() for part in row.split(",")]
            gpus.append({"index": index, "name": name, "memory_total_mib": float(total),
                         "memory_used_mib": float(used), "util_pct": float(util)})
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {"jobs": jobs, "results": results, "gpus": gpus, "warnings": warnings[:20]}


if __name__ == "__main__":
    try:
        options = json.loads(base64.b64decode(sys.argv[1]))
        print(json.dumps(snapshot(options["root"], options.get("known")), allow_nan=False))
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"error": str(error)}))
        raise SystemExit(1)
