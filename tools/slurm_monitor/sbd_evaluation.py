"""Read DP subset-build progress without importing torch or changing its jobs."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re


def _slug(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-") or "item"


def _stage(name, label, completed, total, state="pending"):
    return {"name": name, "label": label, "completed": completed, "total": total,
            "percent": 100 * completed / total if total else None, "state": state}


def progress(reader, row, read_log):
    """Use the attempt's frozen settings, never today's edited eval YAML."""
    launch_path = row.get("sbd_eval_launch_path")
    if not launch_path:
        return None
    launch = reader.read_json(launch_path, 4 * 1024 * 1024)
    import json
    environment = launch.get("environment", {})
    settings_file = environment.get("DP_ABLATION_SETTINGS_FILE")
    raw_settings = environment.get("DP_ABLATION_SETTINGS_JSON")
    if settings_file:
        settings = reader.read_json(reader.safe(settings_file), 4 * 1024 * 1024)
    elif raw_settings:
        settings = json.loads(raw_settings)
    else:
        return None
    source = settings.get("source_manifest")
    if not source or not settings.get("artifact_dir"):
        return None
    artifact = reader.safe(settings["artifact_dir"])
    output = reader.safe(settings["output_dir"])
    report = reader.safe(settings.get("report_root", output))
    episodes = {int(ep["episode_index"]) for ep in source["evaluation_episodes"]}
    if not episodes:
        return None
    manifest = reader.safe(Path(settings["skillset_dir"]) / "skillset_manifest.json")
    # Before prepare_source has run there may be no manifest yet. A different
    # model under the same folder must never borrow that folder's counters.
    compatible = not manifest.exists() or reader.read_json(manifest, 2 * 1024 * 1024) == source
    pairs = set()
    for experiment in settings["experiments"]:
        for probe in experiment.get("probes", [experiment.get("probe")]):
            if probe is not None:
                pairs.add((_slug(probe), _slug(experiment["cluster"])))
    probes = {probe for probe, _cluster in pairs}
    if not pairs:
        return None
    updated = 0.0
    ended = row.get("ended_at")
    if ended:
        ended = datetime.fromisoformat(ended)
        ended = (ended if ended.tzinfo else ended.replace(tzinfo=timezone.utc)).timestamp()

    def count(directory):
        nonlocal updated
        directory = reader.safe(directory)
        count = 0
        if not manifest.is_file() or not compatible or not directory.is_dir():
            return count
        for path in directory.iterdir():
            match = re.fullmatch(r"ep(\d+)\.npz", path.name)
            if match and int(match[1]) in episodes and not path.is_symlink() and path.is_file():
                modified = path.stat().st_mtime
                if ended and modified > ended:
                    continue
                count += 1
                updated = max(updated, modified)
        return count

    raw_total, cluster_total = len(episodes) * len(probes), len(episodes) * len(pairs)
    if row.get("state") == "COMPLETED" and compatible:
        raw_done, cluster_done = raw_total, cluster_total
    else:
        raw_done = sum(count(artifact / "raw" / probe) for probe in probes)
        cluster_done = sum(count(artifact / "clusters" / probe / cluster) for probe, cluster in pairs)
    report_total = len(episodes)
    logs = "\n".join(read_log(reader.safe(path)) for path in row.get("log_paths", []))
    phase = "probe" if raw_done < raw_total else "cluster" if cluster_done < cluster_total else "calibration"
    stats = reader.safe(output / "threshold_statistics.json")
    started = datetime.fromisoformat(row["started_at"])
    started = (started if started.tzinfo else started.replace(tzinfo=timezone.utc)).timestamp()
    if raw_done == raw_total and cluster_done == cluster_total:
        if (stats.exists() and started <= stats.stat().st_mtime <= (ended or float("inf"))) or "[SBD phase] plots" in logs:
            phase = "plots"
        if "[SBD phase] video" in logs or "[video report] task " in logs or "[SBD video]" in logs:
            phase = "video"
    counts = {}
    for name in ("plots", "video"):
        matches = re.findall(r"\[SBD " + name + r"\] episodes=(\d+)/(\d+)", logs)
        counts[name] = max((int(n) for n, total in matches if int(total) == report_total), default=0)
    if not counts["video"]:
        # Older reports log one completed task at a time. Do not double-count
        # head/tail overlap in the bounded log reader.
        tasks = {int(task): int(n) for task, n in re.findall(r"\[video report\] task (\d+): (\d+) episodes", logs)}
        counts["video"] = min(report_total, sum(tasks.values()))
    # Only the explicit attempt exit proves completion, never old HTML files
    # or full cache counts while a new CPU recut is still rendering.
    if row.get("state") == "COMPLETED" and compatible:
        phase = "final"
    order = ["probe", "cluster", "calibration", "plots", "video", "final"]
    index = order.index(phase)
    stages = [
        _stage("probe", "DP 부분빌드", raw_done, raw_total),
        _stage("cluster", "GMM 클러스터링", cluster_done, cluster_total),
        _stage("calibration", "평균 임계값·곡률 계산", int(index > 2), 1),
        _stage("plots", "통계·그래프 생성", report_total if index > 3 else counts["plots"], report_total),
        _stage("video", "영상 HTML 생성", report_total if index > 4 else counts["video"], report_total),
    ]
    for i, stage in enumerate(stages):
        if phase == "final":
            stage.update(completed=stage["total"], percent=100.0, state="done")
        else:
            stage["state"] = "done" if i < index or stage["completed"] == stage["total"] else "running" if i == index else "pending"
    active = stages[min(index, len(stages) - 1)]
    return {"kind": "sbd", "phase": phase, "phase_label": "완료" if phase == "final" else active["label"],
            "percent": 100.0 if phase == "final" else active["percent"],
            "completed_episodes": active["completed"], "total_episodes": active["total"],
            "episodes": report_total, "stages": stages, "updated_at": updated or None,
            "cache_conflict": not compatible, "report": str(report / "index.html")}
