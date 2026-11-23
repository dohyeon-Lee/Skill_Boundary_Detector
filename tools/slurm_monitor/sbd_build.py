"""Read ABC/LIBERO full-build counters from the attempt's frozen paths."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import zipfile


def progress(reader, row):
    launch = reader.read_json(row["sbd_build_launch_path"], 4 * 1024 * 1024)
    env = launch["environment"]
    pipeline = launch.get("kind") == "skillvla_build"
    libero = env.get("LIBERO_SINGLE_GPU_BUILD") == "true"
    total = int(env["LIBERO_EXPECTED_EPISODES" if libero else "ABC_EXPECTED_EPISODES"])
    if total < 1:
        raise ValueError("Invalid build episode count")
    root = reader.safe(env["SKILLSET_DIR"])
    manifest_path = root / "skillset_manifest.json"
    policy = reader.safe(env["DP_POLICY_PATH"])
    tasks = [] if libero else env["ABC_SOURCE_TASKS_CSV"].split(",")
    task_ids = set(map(int, json.loads(env["LIBERO_TASK_INDICES_JSON" if libero else "ABC_TASK_INDICES_JSON"])))
    selection = None
    if env.get("ABC_EPISODE_SELECTION_MANIFEST"):
        path = Path(env["ABC_EPISODE_SELECTION_MANIFEST"])
        if not path.is_absolute():
            path = reader.safe(env["RAW_DATASET_DIR"]) / path
        selection = set(map(int, reader.read_json(path, 4 * 1024 * 1024)["episodes"]))
        if len(selection) != total:
            raise ValueError("Build selection and episode count differ")
    compatible = True
    if manifest_path.is_file():
        manifest = reader.read_json(manifest_path, 2 * 1024 * 1024)
        history = int(reader.read_json(policy / "config.json")["n_obs_steps"])
        mask = 0 if env.get("SKILLSET_INITIAL_MASK", "true") == "false" else history
        compatible = (
            reader.safe(manifest["policy_path"]) == policy
            and manifest.get("source_tasks", []) == tasks
            and manifest.get("detector", {}).get("initial_mask_frames") == mask
            and (manifest.get("episode_selection") or {}).get("fingerprint", "")
            == env.get("ABC_EPISODE_SELECTION_FINGERPRINT", "")
        )
        detector = manifest.get("detector", {})
        for field, key, cast in (
            ("replan_interval", "SKILLSET_REPLAN_INTERVAL", int),
            ("n_gmm_components", "SKILLSET_N_GMM", int),
            ("nms_dist", "SKILLSET_NMS_DIST", int),
            ("smooth_window", "SKILLSET_SMOOTH_WINDOW", int),
            ("savgol_polyorder", "SKILLSET_SAVGOL_POLYORDER", int),
            ("score_power", "SKILLSET_SCORE_POWER", float),
        ):
            if key in env and detector.get(field) != cast(env[key]):
                compatible = False
    ended = row.get("ended_at")
    if ended:
        ended = datetime.fromisoformat(ended)
        ended = (ended if ended.tzinfo else ended.replace(tzinfo=timezone.utc)).timestamp()
    updated = 0.0

    def eligible(path):
        nonlocal updated
        if path.is_symlink() or not path.is_file():
            return False
        modified = path.stat().st_mtime
        if ended and modified > ended:
            return False
        updated = max(updated, modified)
        return True

    def count(folder, pattern, recursive=False):
        ids = set()
        if not compatible or not manifest_path.is_file():
            return ids
        for path in (folder.rglob("*") if recursive else folder.glob("*")):
            match = re.fullmatch(pattern, path.name)
            if not match or not eligible(path):
                continue
            episode = int(match[1])
            if selection is not None and episode not in selection:
                continue
            if match.lastindex == 2 and int(match[2]) not in task_ids:
                continue
            ids.add(episode)
        return ids

    curves = count(root / "curves", r"ep(\d+)\.npz")
    skipped = count(root / "curves", r"ep(\d+)\.too_short\.json")
    segments = count(root / "skills", r"ep(\d+)_task(\d+)\.done", recursive=True)
    curves_done = min(total, len(curves | skipped))
    segments_done = min(total, len(segments))
    threshold_done = False
    reference = env.get("SKILLSET_GLOBAL_THRESHOLD_SOURCE", "")
    threshold = reader.safe(reference or env["SKILLSET_GLOBAL_THRESHOLD_PATH"])
    if compatible and curves_done == total and eligible(threshold):
        value = reader.read_json(threshold)
        threshold_done = bool(reference) or int(value.get("n_episodes", 0)) + int(value.get("n_skipped_too_short", 0)) == total
    if compatible and curves_done == total and env.get("SKILLSET_BOUNDARY_THRESHOLD_MODE") == "episode_mean":
        threshold_done = True
    verified = compatible and eligible(reader.safe(env["SKILLSET_DONE_PATH"]))
    sbd_done = verified and threshold_done and segments_done == total
    final = row.get("state") == "COMPLETED" and sbd_done
    fsq_done = dataset_done = False
    if pipeline:
        output = reader.safe(launch["output_dir"])
        latent = reader.safe(launch["fsq_latents_path"])
        fsq_done = eligible(latent) and zipfile.is_zipfile(latent)
        fsq_done = fsq_done or (eligible(output / "skill_latents.npz") and zipfile.is_zipfile(output / "skill_latents.npz"))
        dataset_done = eligible(output / ".complete")
        final = final and fsq_done and dataset_done
    stages = []
    for name, label, done, target in (
        ("curves", "DP 곡선·GMM 생성", curves_done, total),
        ("threshold", "전역 평균 임계값 계산", int(threshold_done), 1),
        ("segment", "스킬 분할 저장", segments_done, total),
        ("verify", "SBD 전체 검증" if pipeline else "전체 빌드 최종 검증", int(sbd_done if pipeline else final), 1),
    ):
        stages.append({"name": name, "label": label, "completed": done, "total": target,
                       "percent": 100 * done / target, "state": "done" if done == target else "pending"})
    if pipeline:
        for name, label, done in [("fsq", "FSQ 스킬 인코딩", fsq_done), ("dataset", "SkillVLA·ISS·UV 생성", dataset_done)]:
            stages.append({"name": name, "label": label, "completed": int(done), "total": 1,
                           "percent": 100 * int(done), "state": "done" if done else "pending"})
    active = next((stage for stage in stages if stage["state"] != "done"), stages[-1])
    if not final:
        active["state"] = "running" if not row.get("terminal") else "pending"
    return {"kind": "sbd_build", "pipeline": pipeline, "phase": "final" if final else active["name"],
            "phase_label": "전체 빌드 완료" if final else active["label"],
            "percent": 100 if final else active["percent"], "stages": stages,
            "episodes": total, "workers": int(env.get("SKILLSET_SINGLE_GPU_WORKERS", 1)),
            "source_tasks": tasks, "cache_conflict": not compatible, "updated_at": updated,
            "output_dir": str(output if pipeline else root)}
