"""DP eval counters use immutable attempts and only the requested cache names."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import Evaluations
import training


def attempt(root, run="one", probe="current_gt", state="RUNNING"):
    session = root / ".cache/srun_sessions" / run
    session.mkdir(parents=True)
    artifact = root / "eval/partial_build/data"
    source = {"policy_path": str(root / "models/10k"),
              "evaluation_episodes": [{"episode_index": 1}, {"episode_index": 2}]}
    output = root / "eval/outputs" / probe / "src"
    output.mkdir(parents=True, exist_ok=True)
    settings = {"artifact_dir": str(artifact), "output_dir": str(output),
                "report_root": str(output.parent), "skillset_dir": str(artifact / "source"),
                "source_manifest": source, "experiments": [{"probe": probe, "cluster": "k5"}]}
    manifest = artifact / "source/skillset_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(source))
    (session / "launch.json").write_text(json.dumps({"environment": {
        "DP_ABLATION_SETTINGS_JSON": json.dumps(settings)}}))
    log = session / "eval.out"
    log.write_text("")
    record = {"run_id": run, "job_id": "73428", "state": state, "name": "LIBERO_DP_SBD_eval",
              "work_dir": str(root / "eval"), "started_at": datetime.now(timezone.utc).isoformat(),
              "output": str(log)}
    (session / "training_status.json").write_text(json.dumps(record))
    row = training.collect(root, [])["training:" + run]
    return row, settings, log


def cache(settings, kind, probe, episode):
    folder = Path(settings["artifact_dir"]) / kind / probe
    if kind == "clusters":
        folder /= "k5"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"ep{episode:07d}.npz").touch()


@pytest.mark.parametrize("file_transport", [False, True])
def test_counts_only_selected_episodes_and_probe_even_in_shared_allocation(tmp_path, file_transport):
    row, settings, _ = attempt(tmp_path)
    if file_transport:
        launch_path = tmp_path / ".cache/srun_sessions/one/launch.json"
        settings_path = tmp_path / ".cache/settings.json"
        settings_path.write_text(json.dumps(settings))
        launch_path.write_text(json.dumps({"environment": {
            "DP_ABLATION_SETTINGS_FILE": str(settings_path)}}))
        row = training.collect(tmp_path, [])["training:one"]
    cache(settings, "raw", "current_gt", 1)
    cache(settings, "raw", "current_gt", 999)
    cache(settings, "raw", "previous_action", 2)
    (Path(settings["artifact_dir"]) / "raw/current_gt/ep0000002.npz.tmp.npz").touch()
    cache(settings, "clusters", "current_gt", 1)
    progress = Evaluations(tmp_path).enrich(row)["evaluation_progress"]
    assert progress["kind"] == "sbd" and progress["phase"] == "probe"
    assert progress["stages"][0]["completed"] == 1 and progress["percent"] == 50
    other, _, _ = attempt(tmp_path, "two", "previous_action")
    assert Evaluations(tmp_path).enrich(other)["evaluation_progress"]["stages"][1]["completed"] == 0


def test_full_caches_and_old_html_do_not_finish_new_cpu_recut(tmp_path):
    row, settings, log = attempt(tmp_path)
    for kind in ["raw", "clusters"]:
        for episode in [1, 2]:
            cache(settings, kind, "current_gt", episode)
    (Path(settings["report_root"]) / "index.html").write_text("old report")
    reader = Evaluations(tmp_path)
    assert reader.enrich(row)["evaluation_progress"]["phase"] == "calibration"
    (Path(settings["output_dir"]) / "threshold_statistics.json").write_text("{}")
    assert reader.enrich(row)["evaluation_progress"]["phase"] == "plots"
    log.write_text("[SBD plots] episodes=1/2\n[SBD phase] video\n[SBD video] episodes=1/2\n")
    progress = reader.enrich(row)["evaluation_progress"]
    assert progress["phase"] == "video" and progress["percent"] == 50
    assert progress["stages"][3]["completed"] == 2
    row["state"] = "COMPLETED"
    progress = reader.enrich(row)["evaluation_progress"]
    assert progress["phase"] == "final" and all(s["percent"] == 100 for s in progress["stages"])


def test_conflicting_source_does_not_borrow_cache_progress(tmp_path):
    row, settings, _ = attempt(tmp_path)
    cache(settings, "raw", "current_gt", 1)
    manifest = Path(settings["skillset_dir"]) / "skillset_manifest.json"
    manifest.write_text(json.dumps({**settings["source_manifest"], "policy_path": "other-model"}))
    progress = Evaluations(tmp_path).enrich(row)["evaluation_progress"]
    assert progress["cache_conflict"] and progress["stages"][0]["completed"] == 0


def test_duplicate_status_event_keeps_frozen_launch_reference(tmp_path):
    row, _settings, _ = attempt(tmp_path)
    status = Path(row["status_path"])
    events = tmp_path / ".cache/training_events"
    events.mkdir()
    (events / "one.json").write_bytes(status.read_bytes())
    result = training.collect(tmp_path, [])["training:one"]
    assert Path(result["sbd_eval_launch_path"]).name == "launch.json"
    assert Evaluations(tmp_path).enrich(result)["evaluation_progress"]["phase"] == "probe"


def test_legacy_video_task_logs_are_not_double_counted(tmp_path):
    row, settings, log = attempt(tmp_path)
    for kind in ["raw", "clusters"]:
        for episode in [1, 2]:
            cache(settings, kind, "current_gt", episode)
    log.write_text("[video report] task 0: 1 episodes\n[video report] task 0: 1 episodes\n")
    progress = Evaluations(tmp_path).enrich(row)["evaluation_progress"]
    assert progress["phase"] == "video" and progress["stages"][-1]["completed"] == 1


def test_snapshot_paths_cannot_escape_project(tmp_path):
    row, _settings, _ = attempt(tmp_path)
    row["sbd_eval_launch_path"] = str(tmp_path.parent / "other/launch.json")
    assert "evaluation_error" in Evaluations(tmp_path).enrich(row)


def test_failed_attempt_does_not_pick_up_files_from_later_retry(tmp_path):
    row, settings, _ = attempt(tmp_path, state="FAILED")
    row["ended_at"] = datetime.now(timezone.utc).isoformat()
    for kind in ["raw", "clusters"]:
        cache(settings, kind, "current_gt", 1)
        folder = Path(settings["artifact_dir"]) / kind / "current_gt"
        if kind == "clusters":
            folder /= "k5"
        os.utime(folder / "ep0000001.npz", (4102444800, 4102444800))
    progress = Evaluations(tmp_path).enrich(row)["evaluation_progress"]
    assert progress["phase"] == "probe" and progress["stages"][0]["completed"] == 0
