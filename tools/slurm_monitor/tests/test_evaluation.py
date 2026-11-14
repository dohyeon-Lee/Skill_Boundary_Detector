"""Evaluation counters are read-only and isolated by attempt, not allocation."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import Evaluations
from test_monitor import monitor
from test_training import record


def bar(n, rate=None):
    return f"\rStepping through eval batches: 0%| | {n}/10" + (
        f" [running_success_rate={rate}%]" if rate is not None else "")


def setup_plan(root, run="attempt", output="result"):
    folder = root / ".cache/eval_progress"
    folder.mkdir(parents=True, exist_ok=True)
    out = root / "eval/outputs" / output
    logs = root / "eval/logs"
    logs.mkdir(parents=True, exist_ok=True)
    value = {"schema": 1, "job_id": run, "output": str(out), "suite": "libero_10",
             "episodes_per_task": 10, "batch_size": 1,
             "panels": [{"index": 0, "label": "30k"}, {"index": 1, "label": "40k"}],
             "workers": [{"panels": [0], "tasks": [0], "log": str(logs / (run + "_0.err"))},
                         {"panels": [1], "tasks": [0], "log": str(logs / (run + "_1.err"))}]}
    path = folder / (run + ".eval.json")
    path.write_text(json.dumps(value))
    row = {"run_id": run, "job_id": "123", "work_dir": str(root / "eval"), "outputs": [], "state": "RUNNING"}
    return row, value, path


def test_live_progress_updates_from_same_runpod_parser_and_ignores_rollout_sr(tmp_path):
    row, plan, _ = setup_plan(tmp_path)
    log = Path(plan["workers"][0]["log"])
    log.write_text(bar(0) + bar(0, 100) + bar(1, 100) + bar(1, 50) + bar(2, 50) +
                   "\rRunning rollout with at most 520 steps: 50%| | 260/520 [running_success_rate=100%]")
    reader = Evaluations(tmp_path)
    first = reader.enrich(row)
    assert first["outputs"] == [plan["output"]]
    live = first["evaluation_progress"]
    assert (live["completed_episodes"], live["total_episodes"], live["successes"]) == (2, 20, 1)
    assert live["percent"] == 10 and live["success_pct"] == 50
    assert live["models"][1]["tasks"][0]["success_pct"] is None
    with log.open("a") as stream:
        stream.write(bar(3, 66.7))
    assert reader.enrich(row)["evaluation_progress"]["completed_episodes"] == 3
    assert "running_success_rate" not in json.dumps(live)


def test_attempts_in_one_allocation_do_not_share_progress(tmp_path):
    first, plan, _ = setup_plan(tmp_path, "first", "first")
    second, _, _ = setup_plan(tmp_path, "second", "second")
    Path(plan["workers"][0]["log"]).write_text(bar(0) + bar(10, 60))
    reader = Evaluations(tmp_path)
    assert reader.enrich(first)["evaluation_progress"]["completed_episodes"] == 10
    assert reader.enrich(second)["evaluation_progress"]["completed_episodes"] == 0


def test_final_metrics_override_reused_or_oversized_logs_and_cache_refreshes(tmp_path):
    row, plan, _ = setup_plan(tmp_path)
    out = Path(plan["output"]) / "metrics"
    out.mkdir(parents=True)
    def data(wins):
        return {label: {"per_task": [{"task_group": "libero_10", "task_id": 0,
                "metrics": {"successes": [True] * wins + [False] * (10 - wins)}}]}
                for label in ("30k", "40k")}
    metric = out / "eval_info_merged.json"
    metric.write_text(json.dumps(data(6)))
    Path(plan["workers"][0]["log"]).write_text("another attempt, no episode identity")
    reader = Evaluations(tmp_path)
    value = reader.enrich(row)["evaluation_progress"]
    assert value["phase"] == "final" and value["percent"] == 100
    assert value["successes"] == 12 and value["success_pct"] == 60
    metric.write_text(json.dumps(data(7)))
    assert reader.enrich(row)["evaluation_progress"]["success_pct"] == 70


def test_partial_metrics_do_not_claim_missing_panels_completed(tmp_path):
    row, plan, _ = setup_plan(tmp_path)
    path = Path(plan["output"]) / "metrics/eval_info_merged.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"30k": {"per_task": [{"task_id": 0, "metrics": {"successes": [True] * 10}}]}}))
    value = Evaluations(tmp_path).enrich(row)["evaluation_progress"]
    assert value["phase"] == "partial" and value["percent"] == 50
    assert value["observed_episodes"] == 10 and value["success_pct"] == 100


def test_completed_legacy_evaluation_without_plan(tmp_path):
    out = tmp_path / "eval/outputs/legacy"
    (out / "metrics").mkdir(parents=True)
    (out / "metrics/eval_info_merged.json").write_text(json.dumps({"model": {"per_task": [
        {"task_id": 0, "metrics": {"successes": [False, True]}}]}}))
    row = {"job_id": "123", "outputs": [str(out)], "work_dir": str(tmp_path / "eval")}
    value = Evaluations(tmp_path).enrich(row)["evaluation_progress"]
    assert value["phase"] == "final" and value["success_pct"] == 50


def test_current_slurm_without_tqdm_reads_task_html_without_executing_scripts(tmp_path):
    row, plan, _ = setup_plan(tmp_path)
    path = Path(plan["output"]) / "panels/00_30k/skill_html/libero_10_task00/skill_trace.html"
    path.parent.mkdir(parents=True)
    data = {"task_group": "libero_10", "task_id": 0, "episodes": [
        {"episode_index": i, "success": i < 6} for i in range(10)]}
    path.write_text("<script>const DATA = " + json.dumps(data) + "; throw 'never executed';</script>")
    value = Evaluations(tmp_path).enrich(row)["evaluation_progress"]
    assert value["at_least"] and value["completed_episodes"] == 10
    assert value["successes"] == 6 and value["success_pct"] == 60
    assert value["total_episodes"] == 20 and value["percent"] == 50


def test_structured_episode_records_update_before_task_completion(tmp_path):
    row, plan, file = setup_plan(tmp_path)
    plan["episode_records"] = True
    file.write_text(json.dumps(plan))
    path = Path(plan["output"]) / "panels/00_30k/videos/libero_10_0/eval_progress.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": 1, "successes": [True, False, True], "total_episodes": 10}))
    reader = Evaluations(tmp_path)
    value = reader.enrich(row)["evaluation_progress"]
    assert not value["at_least"] and value["completed_episodes"] == 3
    assert value["observed_episodes"] == 3 and value["successes"] == 2
    path.write_text(json.dumps({"schema": 1, "successes": [True, False, True, False], "total_episodes": 10}))
    assert reader.enrich(row)["evaluation_progress"]["success_pct"] == 50


@pytest.mark.parametrize("target", ["log", "output"])
def test_symlink_paths_outside_workspace_are_never_read(tmp_path, target):
    root = tmp_path / "repo"
    row, plan, path = setup_plan(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    if target == "log":
        secret = outside / "secret"
        secret.write_text(bar(0) + bar(10, 100))
        Path(plan["workers"][0]["log"]).symlink_to(secret)
    else:
        out = Path(plan["output"])
        out.parent.mkdir(parents=True)
        out.symlink_to(outside, target_is_directory=True)
    result = Evaluations(root).enrich(row)
    assert "evaluation_progress" not in result and result["evaluation_error"]


def test_monitor_reports_eval_without_changing_status_or_sending_messages(monitor):
    root = Path(monitor.config["project_root"])
    row, plan, _ = setup_plan(root, "eval-run")
    key = record(monitor, "eval-run", work_dir=row["work_dir"], outputs=[plan["output"]])
    Path(plan["workers"][0]["log"]).write_text(bar(0) + bar(5, 40))
    monitor.poll()
    actual = next(job for job in monitor.snapshot()["jobs"] if job["key"] == key)
    assert actual["state"] == "RUNNING" and actual["evaluation_progress"]["completed_episodes"] == 5
    assert not monitor.slurm.cancelled and not monitor.sent


def test_slurm_publisher_uses_run_id_and_unique_worker_log_paths(tmp_path):
    file = Path(__file__).resolve().parents[3] / "lerobot/src/lerobot/utils/eval_progress.py"
    spec = importlib.util.spec_from_file_location("eval_progress_plan_test", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = {"PROJECT_ROOT": str(tmp_path), "SLURM_JOB_ID": "123", "SBD_TRAINING_PARENT_ID": "abcd",
           "MODELS_JSON": '[{"label":"20k"}]',
           "EVAL_GPU_GROUPS_JSON": '[[{"tag":"w0","items":[0],"panels":[0]}]]',
           "STAGE1_EVAL_WORK_DIR": str(tmp_path / "eval"), "EVAL_OUT_DIR": str(tmp_path / "eval/outputs/new"),
           "TARGET_TASK": "libero_10", "N_EPISODES": "10", "EVAL_BATCH_SIZE": "1", "HF_TOKEN": "secret"}
    module.publish(env, "123_abcd")
    plan = json.loads((tmp_path / ".cache/eval_progress/abcd.eval.json").read_text())
    assert plan["job_id"] == "abcd" and plan["workers"][0]["log"].endswith("123_abcd_0_w0.err")
    assert "secret" not in json.dumps(plan)


def test_episode_publisher_keeps_last_good_outcomes_and_never_changes_rng(tmp_path):
    import random
    file = Path(__file__).resolve().parents[3] / "lerobot/src/lerobot/utils/eval_progress.py"
    spec = importlib.util.spec_from_file_location("episode_progress_test", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = random.getstate()
    folder = tmp_path / "task"
    module.record_task(folder, [True, False], 10)
    assert random.getstate() == state
    assert json.loads((folder / "eval_progress.json").read_text())["successes"] == [True, False]
    with pytest.raises(ValueError):
        module.record_task(folder, [True] * 11, 10)
    assert json.loads((folder / "eval_progress.json").read_text())["successes"] == [True, False]
    assert not list(folder.glob(".episode-progress-*"))
