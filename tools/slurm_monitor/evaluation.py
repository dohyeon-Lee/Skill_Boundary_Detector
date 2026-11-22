"""Read-only evaluation progress; share RunPod's episode-log parser unchanged."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

if __package__:
    from .sbd_evaluation import progress as sbd_progress
    from .slurm import read_log
else:
    from sbd_evaluation import progress as sbd_progress
    from slurm import read_log


def _load_probe():
    path = Path(__file__).resolve().parents[1] / "runpod/desk_probe.py"
    spec = importlib.util.spec_from_file_location("slurm_eval_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


def aggregate(tasks):
    completed = sum(task["completed_episodes"] for task in tasks)
    total = sum(task["total_episodes"] for task in tasks)
    observed = sum(task["completed_episodes"] for task in tasks if task["successes"] is not None)
    successes = sum(task["successes"] for task in tasks if task["successes"] is not None)
    return {"completed_episodes": completed, "total_episodes": total,
            "percent": 100 * completed / total if total else None,
            "successes": successes, "observed_episodes": observed,
            "success_pct": 100 * successes / observed if observed else None,
            "completed_tasks": sum(task["state"] == "done" for task in tasks), "total_tasks": len(tasks)}


def metrics_progress(value, plan=None):
    if "per_task" in value:
        value = {"eval": value}
    expected = {}
    if plan:
        for worker in plan["workers"]:
            for panel in worker["panels"]:
                for task in worker["tasks"]:
                    unit = (int(panel), int(task))
                    if unit in expected:
                        raise ValueError("Duplicate evaluation work")
                    expected[unit] = int(plan["episodes_per_task"])
    models = []
    panels = plan["panels"] if plan else [{"index": i, "label": label} for i, label in enumerate(value)]
    for panel in panels:
        info = value.get(panel["label"], {})
        rows = info.get("per_task", [])
        actual = {int(row["task_id"]): row for row in rows}
        if len(actual) != len(rows):
            raise ValueError("Duplicate task metrics")
        task_ids = sorted(task for index, task in expected if index == int(panel["index"])) if plan else sorted(actual)
        tasks = []
        for task_id in task_ids:
            row = actual.get(task_id, {})
            metrics = row.get("metrics", row)
            outcomes = metrics.get("successes", [])
            if not isinstance(outcomes, list) or not all(type(item) is bool for item in outcomes):
                raise ValueError("Invalid episode outcomes")
            total = expected.get((int(panel["index"]), task_id), len(outcomes))
            if total < 1 or len(outcomes) > total:
                raise ValueError("Invalid episode count")
            tasks.append({"name": f"{row.get('task_group', (plan or {}).get('suite', 'task'))}_{task_id}",
                          "task_id": task_id, "completed_episodes": len(outcomes), "total_episodes": total,
                          "successes": sum(outcomes) if outcomes else None,
                          "success_pct": 100 * sum(outcomes) / len(outcomes) if outcomes else None,
                          "state": "done" if len(outcomes) == total else "running" if outcomes else "pending"})
        if tasks:
            models.append({"label": str(panel["label"]), **aggregate(tasks), "tasks": tasks})
    if not models:
        return None
    result = {**aggregate([task for model in models for task in model["tasks"]]), "models": models,
              "source": "metrics", "incomplete": False}
    result["phase"] = "final" if result["completed_episodes"] == result["total_episodes"] else "partial"
    return result


class Evaluations:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = {}

    def safe(self, path):
        path = Path(path).resolve()
        path.relative_to(self.root)
        return path

    def read_json(self, path, limit=probe.MAX_JSON):
        path = self.safe(path)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self.cache.get(str(path))
        if cached and cached[0] == signature:
            return cached[1]
        value = json.loads(probe.bounded(path, limit))
        if len(self.cache) >= 256:
            self.cache.pop(next(iter(self.cache)))
        self.cache[str(path)] = (signature, value)
        return value

    def enrich(self, row):
        if row.get("allocation_only"):
            return {}
        result = {}
        if row.get("sbd_eval_launch_path"):
            try:
                progress = sbd_progress(self, row, read_log)
                if progress:
                    return {"evaluation_progress": progress}
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                return {"evaluation_error": "DP 부분빌드 진행률을 확인할 수 없습니다. 다음 조회에서 다시 확인합니다."}
        identifier = str(row.get("run_id") or str(row.get("job_id", "")).split("_")[0])
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
            return result
        base = self.root / ".cache/eval_progress" / identifier
        plan_path = base.with_suffix(".eval.json")
        if not plan_path.is_file() and "/eval" not in row.get("work_dir", "") and not any(
                "/eval/" in str(path) for path in row.get("outputs", [])):
            return result
        try:
            plan = self.read_json(plan_path, 512 * 1024) if plan_path.is_file() else None
            if plan and (plan.get("schema") != 1 or plan.get("job_id") != identifier):
                raise ValueError("Invalid evaluation plan")
            outputs = [self.safe(plan["output"])] if plan else [self.safe(path) for path in row.get("outputs", [])]
            if not plan and "/eval" not in row.get("work_dir", "") and not any("/eval/" in str(path) for path in outputs):
                return result
            for output in outputs:
                for name in ("eval_info_merged.json", "eval_info.json"):
                    metric = output / "metrics" / name
                    if not metric.is_file():
                        continue
                    progress = metrics_progress(self.read_json(metric), plan)
                    if progress:
                        progress["updated_at"] = metric.stat().st_mtime
                        result["evaluation_progress"] = progress
                        break
                if "evaluation_progress" in result:
                    break
            if plan and "evaluation_progress" not in result:
                progress = probe.evaluation_progress(self.root, base, plan["output"])
                if progress:
                    result["evaluation_progress"] = self.artifacts(progress, plan)
            if plan and not row.get("outputs"):
                result.update(outputs=[str(outputs[0])], output_source="평가 실행 계획")
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            result["evaluation_error"] = "평가 진행률을 확인할 수 없습니다. 다음 조회에서 다시 확인합니다."
        return result

    def artifacts(self, progress, plan):
        """Older Slurm workers disable tqdm; their task HTML still records outcomes."""
        updated = progress["updated_at"]
        log_observed = progress["completed_episodes"] > 0
        for panel, model in zip(plan["panels"], progress["models"], strict=True):
            label = str(panel["label"])
            safe_label = "".join(char if char.isalnum() or char in "._-" else "-" for char in label).strip("-_") or "model"
            folder = self.safe(Path(plan["output"]) / "panels" / f"{int(panel['index']):02d}_{safe_label}")
            for task in model["tasks"]:
                record = folder / "videos" / f"{plan['suite']}_{task['task_id']}" / "eval_progress.json"
                html = folder / "skill_html" / f"{plan['suite']}_task{task['task_id']:02d}" / "skill_trace.html"
                outcomes = None
                source = None
                if record.is_file():
                    value = self.read_json(record, 2 * 1024 * 1024)
                    if value.get("schema") != 1 or value.get("total_episodes") != task["total_episodes"]:
                        raise ValueError("Episode progress belongs to another evaluation")
                    outcomes, source = value["successes"], record
                elif html.is_file():
                    value = self.html_payload(html)
                    if int(value["task_id"]) != task["task_id"] or value["task_group"] != plan["suite"]:
                        raise ValueError("Task HTML belongs to another evaluation")
                    episodes = value["episodes"]
                    if [ep["episode_index"] for ep in episodes] != list(range(len(episodes))):
                        raise ValueError("Ambiguous HTML episode identities")
                    outcomes, source = [ep["success"] for ep in episodes], html
                if outcomes is None:
                    continue
                if not isinstance(outcomes, list) or not all(type(item) is bool for item in outcomes) or len(outcomes) > task["total_episodes"]:
                    raise ValueError("Invalid completed episode outcomes")
                if len(outcomes) < task["completed_episodes"]:
                    continue
                updated = max(updated, source.stat().st_mtime)
                task.update(completed_episodes=len(outcomes), successes=sum(outcomes) if outcomes else None,
                            success_pct=100 * sum(outcomes) / len(outcomes) if outcomes else None,
                            state="done" if len(outcomes) == task["total_episodes"] else "running")
            model.update(aggregate(model["tasks"]))
        progress.update(aggregate([task for model in progress["models"] for task in model["tasks"]]))
        progress.update(phase="partial", updated_at=updated,
                        at_least=not plan.get("episode_records", False) and not log_observed)
        if progress["incomplete"]:
            progress["percent"] = None
        return progress

    def html_payload(self, path):
        path = self.safe(path)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        key = "html:" + str(path)
        cached = self.cache.get(key)
        if cached and cached[0] == signature:
            return cached[1]
        text = probe.bounded(path).decode("utf-8")
        marker = "const DATA = "
        offset = text.index(marker) + len(marker)
        value, _ = json.JSONDecoder().raw_decode(text[offset:])
        if len(self.cache) >= 256:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = (signature, value)
        return value
