"""Exercise shared-GPU concurrency, pipeline barriers and allocation routing on CPU."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/build_data/src"
sys.path.insert(0, str(SRC))
from run_single_gpu_build import SingleGPUBuilder
from test_train_skillvla_build_config import _config, build_settings


def builder(tmp_path, **overrides):
    src = tmp_path / "build/src"; src.mkdir(parents=True)
    raw = tmp_path / "raw"; (raw / "meta/episodes/chunk-000").mkdir(parents=True)
    pd.DataFrame({"task_index": [0, 2, 3, 5, 8, 9, 12]}).to_parquet(raw / "meta/tasks.parquet")
    pd.DataFrame({"episode_index": range(7)}).to_parquet(raw / "meta/episodes/chunk-000/file-000.parquet")
    env = dict(os.environ, SKILLSET_SINGLE_GPU_WORKERS="2", SKILLSET_TASKS_PER_JOB="2",
               SKILLSET_MAX_SWEEPS="2", SKILLVLA_CPUS_PER_TASK="8", RAW_DATASET_DIR=str(raw),
               SKILLSET_BOUNDARY_THRESHOLD_MODE="global_mean", SKILLSET_GLOBAL_THRESHOLD_SOURCE="",
               SKILL_LATENTS_PATH=str(tmp_path / "run/skill_latents.npz"),
               SKILLSET_DIR=str(tmp_path / "skillset"), SKILLVLA_RUN_DIR=str(tmp_path / "run"),
               SKILLSET_CONTRACT_JSON="{}", CUDA_VISIBLE_DEVICES="6")
    env.update(overrides)
    return SingleGPUBuilder(env, src=src, log_dir=tmp_path / "logs")


def test_pool_bounds_gpu_processes_and_assigns_each_task_once(tmp_path):
    b = builder(tmp_path)
    state = tmp_path / "state.json"
    worker = tmp_path / "worker.py"
    worker.write_text('''
import fcntl,json,os,time
from pathlib import Path
p=Path(os.environ['TEST_STATE'])
def update(delta):
 with p.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  d=json.loads(p.read_text()) if p.exists() else {'active':0,'peak':0,'seen':[]}
  d['active']+=delta;d['peak']=max(d['peak'],d['active'])
  if delta>0:
   assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
   assert os.environ['CURVES_ONLY']=='true' and os.environ['USE_CACHED_CURVES']=='false'
   assert os.environ['SBD_SRUN_SESSION']=='1'
   assert os.environ['SLURM_ARRAY_TASK_ID']=='0'
   d['seen']+=list(map(int,os.environ['ALL_TASK_IDS'].split()))
  p.write_text(json.dumps(d))
update(1)
try:time.sleep(.15)
finally:update(-1)
''')
    (b.src / "build_skillset.sbatch").write_text(
        f"#!/bin/bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(worker))}\n")
    b.env["TEST_STATE"] = str(state)
    ids = [0, 2, 3, 5, 8, 9, 12]
    assert b._pool(ids, phase="curves") == [0, 0, 0, 0]
    data = json.loads(state.read_text())
    assert data["peak"] == 2 and data["active"] == 0
    assert sorted(data["seen"]) == ids


@pytest.mark.parametrize("mode,reference,expected", [
    ("global_mean", "", ["curves", "compute_global_boundary_threshold.sbatch", "cached_segment", "verify_skillset.sbatch", "encode_skills.sbatch", "build_skillvla.sbatch"]),
    ("global_mean", "pt_threshold.json", ["curves", "cached_segment", "verify_skillset.sbatch", "encode_skills.sbatch", "build_skillvla.sbatch"]),
    ("episode_mean", "", ["segment", "verify_skillset.sbatch", "encode_skills.sbatch", "build_skillvla.sbatch"]),
])
def test_pipeline_waits_for_all_curves_and_reduces_once(tmp_path, monkeypatch, mode, reference, expected):
    b = builder(tmp_path, SKILLSET_BOUNDARY_THRESHOLD_MODE=mode, SKILLSET_GLOBAL_THRESHOLD_SOURCE=reference)
    seen = []
    monkeypatch.setattr(b, "_complete_skillset", lambda: False)
    monkeypatch.setattr(b, "_encoded", lambda: False)
    def pool(ids, *, phase):
        assert ids == [0, 2, 3, 5, 8, 9, 12]
        seen.append(phase)
        return [0] * 4
    def stage(name, *, env=None):
        seen.append(name)
        if name == "compute_global_boundary_threshold.sbatch":
            assert env["EXPECTED_EPISODES"] == "7"
        assert b.env["SKILLSET_MAX_SWEEPS"] == "2"
    monkeypatch.setattr(b, "_pool", pool)
    monkeypatch.setattr(b, "_stage", stage)
    b.run()
    assert seen == expected


def test_completed_skillset_and_verified_latents_are_reused(tmp_path, monkeypatch):
    b = builder(tmp_path)
    monkeypatch.setattr(b, "_complete_skillset", lambda: True)
    monkeypatch.setattr(b, "_encoded", lambda: True)
    monkeypatch.setattr(b, "_pool", lambda *a, **k: pytest.fail("already complete"))
    seen = []
    monkeypatch.setattr(b, "_stage", seen.append)
    b.run()
    assert seen == ["build_skillvla.sbatch"]


def test_failed_curves_block_threshold_and_dataset_build(tmp_path, monkeypatch):
    b = builder(tmp_path)
    monkeypatch.setattr(b, "_complete_skillset", lambda: False)
    monkeypatch.setattr(b, "_pool", lambda *a, **k: [0, 1])
    monkeypatch.setattr(b, "_stage", lambda *a, **k: pytest.fail("must stop"))
    with pytest.raises(RuntimeError, match="Curve collection failed"):
        b.run()


def test_storage_failure_stops_without_retry_or_allocation_commands(tmp_path):
    b = builder(tmp_path)
    with pytest.raises(RuntimeError, match="Storage full"):
        b._command([sys.executable, "-c", "import sys;sys.stderr.write('No space left on device');sys.exit(1)"], "full")
    assert b.stopping
    with pytest.raises(InterruptedError):
        b._command([sys.executable, "-c", "raise RuntimeError('must not start')"], "later")


def test_gpu_failure_stops_only_own_workers(tmp_path):
    b = builder(tmp_path)
    with pytest.raises(RuntimeError, match="GPU failure"):
        b._command([sys.executable, "-c", "import sys;sys.stderr.write('CUDA out of memory');sys.exit(1)"], "oom")
    assert b.stopping
    assert not b.processes


def test_stop_does_not_terminate_unrelated_processes(tmp_path):
    b = builder(tmp_path)
    ready = tmp_path / "ready"
    errors = []
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    def owned_worker():
        try:
            b._command([sys.executable, "-c",
                        f"from pathlib import Path;import time;Path({str(ready)!r}).touch();time.sleep(30)"], "owned")
        except InterruptedError as error:
            errors.append(error)
    thread = threading.Thread(target=owned_worker)
    try:
        thread.start()
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists()
        b.stop()
        thread.join(timeout=7)
        assert not thread.is_alive() and errors
        assert unrelated.poll() is None
    finally:
        b.stop()
        thread.join(timeout=7)
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.parametrize("workers,held,mode", [(0, False, "srun"), (2, False, "srun"),
                                               (2, True, "srun"), (2, False, "srun_existing")])
def test_submitter_preserves_arrays_or_uses_exactly_one_allocation(tmp_path, workers, held, mode):
    settings = build_settings(_config(tmp_path, "episode_mean"))
    settings["skillset_single_gpu_workers"] = workers
    settings["skillset_array_throttle"] = 1
    src = tmp_path / "lerobot/examples/libero/configs/train_skillVLA/build_data/src"
    src.mkdir(parents=True)
    root = src.parent
    shutil.copy2(SRC.parent / "submit_build_all.sh", root / "submit_build_all.sh")
    shutil.copy2(SRC / "skillset_args.sh", src / "skillset_args.sh")
    (tmp_path / ".venv/bin").mkdir(parents=True)
    python = tmp_path / ".venv/bin/python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    raw = Path(settings["raw_dataset_dir"]); (raw / "meta/episodes/chunk-000").mkdir(parents=True)
    pd.DataFrame({"task_index": range(7)}).to_parquet(raw / "meta/tasks.parquet")
    pd.DataFrame({"episode_index": range(7)}).to_parquet(raw / "meta/episodes/chunk-000/file-000.parquet")
    Path(settings["dp_policy_path"]).mkdir(parents=True)
    Path(settings["fsq_model_path"]).touch()
    config_data = tmp_path / "settings.json"
    config_data.write_text(json.dumps(settings, default=str))
    (src / "train_skillVLA_config.py").write_text(
        "import json,os,shlex\n"
        "for k,v in json.load(open(os.environ['TEST_SETTINGS'])).items():\n"
        " print('export '+k.upper()+'='+shlex.quote(str(v)))\n")
    (src / "verify_skillset.py").write_text("raise SystemExit(1)\n")
    contract = tmp_path / "lerobot/examples/libero/configs/train_skills/src/sbd_build_contract.py"
    contract.parent.mkdir(parents=True); contract.write_text("raise SystemExit(0)\n")
    capture = tmp_path / "capture.py"
    capture.write_text('''import json,os,sys
from pathlib import Path
with Path(os.environ['TEST_CAPTURE']).open('a') as stream:
 stream.write(json.dumps({'mode':sys.argv[1],'args':sys.argv[2:],
  'allocation':os.environ.get('SLURM_JOB_ID'), 'submit_dir':os.environ.get('SLURM_SUBMIT_DIR'),
  'snapshot':os.environ.get('SKILLVLA_ENV_SNAPSHOT')})+'\\n')
print('123')
''')
    submit_helper = src.parents[2] / "src/submit_job.sh"
    submit_helper.parent.mkdir(parents=True)
    submit_helper.write_text(
        '_SUBMIT_JOB_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'submit_job_scheduler() { echo slurm; }\n'
        'submit_job_slurm_mode() { echo "$TEST_SLURM_MODE"; }\n'
        'submit_job() { "$TEST_PYTHON" "$TEST_RECORDER" submit "$@"; }\n'
        'if [ "${BASH_SOURCE[0]}" = "$0" ]; then shift; "$TEST_PYTHON" "$TEST_RECORDER" batch "$@"; fi\n')
    (src / "build_single_gpu.sbatch").write_text(
        '#!/bin/bash\n"$TEST_PYTHON" "$TEST_RECORDER" inline\n')
    env = {k:v for k,v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(TEST_SETTINGS=str(config_data), TEST_CAPTURE=str(tmp_path / "calls.jsonl"),
               TEST_PYTHON=sys.executable, TEST_RECORDER=str(capture), TEST_SLURM_MODE=mode)
    if held:
        env.update(SLURM_JOB_ID="999", SLURM_SUBMIT_DIR="/previous/training/directory")
    result = subprocess.run(["bash", str(root / "submit_build_all.sh")], env=env,
                            text=True, capture_output=True, timeout=20)
    calls_path = tmp_path / "calls.jsonl"
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
    if mode == "srun_existing":
        assert result.returncode == 2 and not calls
        return
    assert result.returncode == 0, result.stdout + result.stderr
    if workers == 0:
        assert len(calls) == 4 and all(call["mode"] == "batch" for call in calls)
        assert "--array=0-1%1" in calls[0]["args"]
    else:
        assert len(calls) == 1
        call = calls[0]
        assert call["mode"] == ("inline" if held else "submit")
        assert Path(call["snapshot"]).is_file()
        if held:
            assert call["allocation"] == "999" and call["submit_dir"] == str(root)
        else:
            assert "--gres=gpu:1" in call["args"]
            assert not any(arg.startswith(("--array", "--dependency")) for arg in call["args"])
