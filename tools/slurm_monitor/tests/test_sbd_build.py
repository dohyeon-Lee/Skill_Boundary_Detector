"""Full-build progress follows frozen paths and counts completed episode files."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import Evaluations
import training


def attempt(root):
    folder = root / ".cache/srun_sessions/build"
    folder.mkdir(parents=True)
    output = root / "dataset/skillset"
    (output / "curves").mkdir(parents=True)
    policy = root / "policy"
    policy.mkdir()
    (policy / "config.json").write_text('{"n_obs_steps":60}')
    (output / "skillset_manifest.json").write_text(json.dumps({
        "policy_path": str(policy), "source_tasks": ["put_relative"],
        "detector": {"initial_mask_frames": 0}}))
    env = {"ABC_SINGLE_GPU_BUILD": "true", "ABC_EXPECTED_EPISODES": "3",
           "SKILLSET_DIR": str(output), "DP_POLICY_PATH": str(policy),
           "ABC_SOURCE_TASKS_CSV": "put_relative", "ABC_TASK_INDICES_JSON": "[112]",
           "SKILLSET_INITIAL_MASK": "false", "SKILLSET_SINGLE_GPU_WORKERS": "16",
           "SKILLSET_GLOBAL_THRESHOLD_PATH": str(output / "global_boundary_threshold.json"),
           "SKILLSET_DONE_PATH": str(output / ".complete")}
    launch = folder / "launch.json"
    launch.write_text(json.dumps({"environment": env}))
    record = {"run_id": "build", "job_id": "74692", "state": "RUNNING",
              "name": "ABC_DP_skill", "work_dir": str(root / "abc/configs/train_skills/build_data"),
              "started_at": datetime.now(timezone.utc).isoformat()}
    (folder / "training_status.json").write_text(json.dumps(record))
    row = training.collect(root, [])["training:build"]
    return row, output, launch


def test_curves_include_short_skips_but_not_temporary_files(tmp_path):
    row, output, _ = attempt(tmp_path)
    (output / "curves/ep0000001.npz").touch()
    (output / "curves/ep0000002.too_short.json").write_text("{}")
    (output / "curves/ep0000003.tmp.npz").touch()
    progress = Evaluations(tmp_path).enrich(row)["evaluation_progress"]
    assert progress["kind"] == "sbd_build" and progress["phase"] == "curves"
    assert progress["workers"] == 16 and progress["episodes"] == 3
    assert progress["stages"][0]["completed"] == 2


def test_final_requires_completed_attempt_and_verification_marker(tmp_path):
    row, output, _ = attempt(tmp_path)
    (output / "skills/task112").mkdir(parents=True)
    for episode in range(3):
        (output / "curves" / f"ep{episode:07d}.npz").touch()
        (output / "skills/task112" / f"ep{episode:05d}_task112.done").touch()
    (output / "global_boundary_threshold.json").write_text('{"n_episodes":3}')
    reader = Evaluations(tmp_path)
    assert reader.enrich(row)["evaluation_progress"]["phase"] == "verify"
    (output / ".complete").touch()
    assert reader.enrich(row)["evaluation_progress"]["phase"] == "verify"
    row.update(state="COMPLETED", terminal=True)
    progress = reader.enrich(row)["evaluation_progress"]
    assert progress["phase"] == "final"
    assert all(stage["state"] == "done" for stage in progress["stages"])


def test_conflicting_history_mask_does_not_borrow_old_counts(tmp_path):
    row, output, _ = attempt(tmp_path)
    (output / "curves/ep0000001.npz").touch()
    path = output / "skillset_manifest.json"
    data = json.loads(path.read_text())
    data["detector"]["initial_mask_frames"] = 60
    path.write_text(json.dumps(data))
    progress = Evaluations(tmp_path).enrich(row)["evaluation_progress"]
    assert progress["cache_conflict"] and progress["stages"][0]["completed"] == 0


def test_duplicate_training_event_preserves_frozen_build_reference(tmp_path):
    row, _, launch = attempt(tmp_path)
    events = tmp_path / ".cache/training_events"
    events.mkdir()
    (events / "build.json").write_text(launch.with_name("training_status.json").read_text())
    collected = training.collect(tmp_path, [])["training:build"]
    assert collected["sbd_build_launch_path"] == row["sbd_build_launch_path"]
    assert Evaluations(tmp_path).enrich(collected)["evaluation_progress"]["episodes"] == 3


def test_libero_build_uses_its_frozen_episode_count_and_history_mask(tmp_path):
    row, output, launch = attempt(tmp_path)
    data = json.loads(launch.read_text())
    env = data['environment']
    for key in list(env):
        if key.startswith('ABC_'): del env[key]
    env.update(LIBERO_SINGLE_GPU_BUILD='true', LIBERO_EXPECTED_EPISODES='3',
               LIBERO_TASK_INDICES_JSON='[0]', SKILLSET_INITIAL_MASK='true')
    launch.write_text(json.dumps(data))
    (output / 'skillset_manifest.json').write_text(json.dumps({
        'policy_path': env['DP_POLICY_PATH'], 'detector': {'initial_mask_frames': 60}}))
    (output / 'curves/ep0000001.npz').touch()
    row = training.collect(tmp_path, [])['training:build']
    progress = Evaluations(tmp_path).enrich(row)['evaluation_progress']
    assert not progress['cache_conflict'] and progress['episodes'] == 3
    assert progress['stages'][0]['completed'] == 1
