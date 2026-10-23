from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/PT/Terminator/src/run_predictor_group.py"
)
SPEC = importlib.util.spec_from_file_location("predictor_group", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _runs(tmp_path):
    return [
        {"mode": mode, "run_name": f"test_{mode}", "output_dir": str(tmp_path / mode)}
        for mode in ("mode3_preskill", "mode3_preskill_proprioX")
    ]


def _wait_for_files(paths, timeout=8):
    deadline = time.monotonic() + timeout
    while not all(path.is_file() for path in paths):
        if time.monotonic() > deadline:
            raise AssertionError(f"Timed out waiting for {paths}")
        time.sleep(0.02)


def test_workers_overlap_on_same_gpu_with_independent_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-shared-test")
    monkeypatch.setenv("WANDB_RUN_ID", "must-not-share")
    monkeypatch.setenv("WANDB_NAME", "must-not-share")
    worker = '''
import json, os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
mode = os.environ["SKILL_AUX_PREDICTOR_MODE_OVERRIDE"]
(root / (mode + ".ready")).write_text(json.dumps(dict(os.environ)))
deadline = time.monotonic() + 5
while len(list(root.glob("*.ready"))) < 2:
    if time.monotonic() > deadline:
        sys.exit(9)  # sequential launch cannot pass this rendezvous
    time.sleep(.01)
print("stdout " + mode)
print("stderr " + mode, file=sys.stderr)
'''
    logs = tmp_path / "logs"
    assert MODULE.run_group(
        _runs(tmp_path), [sys.executable, "-c", worker, str(tmp_path)], logs, poll_interval=0.01
    ) == 0
    for run in _runs(tmp_path):
        env = json.loads((tmp_path / f"{run['mode']}.ready").read_text())
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-shared-test"
        assert env["SKILL_AUX_GROUP_WORKER"] == "1"
        assert "WANDB_RUN_ID" not in env and "WANDB_NAME" not in env
        assert (logs / f"{run['mode']}.out").read_text().strip() == f"stdout {run['mode']}"
        assert (logs / f"{run['mode']}.err").read_text().strip() == f"stderr {run['mode']}"
        assert not Path(run["output_dir"]).exists()  # don't break trainer's fresh-output check
    assert set(json.loads((logs / "exit_codes.json").read_text()).values()) == {0}


def test_failed_worker_does_not_cancel_healthy_worker(tmp_path):
    worker = '''
import os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
if os.environ["SKILL_AUX_PREDICTOR_MODE_OVERRIDE"] == "mode3_preskill":
    (root / "failed").touch()
    sys.exit(7)
while not (root / "failed").exists():
    time.sleep(.01)
time.sleep(.1)
(root / "healthy_finished").touch()
'''
    logs = tmp_path / "logs"
    assert MODULE.run_group(
        _runs(tmp_path), [sys.executable, "-c", worker, str(tmp_path)], logs, poll_interval=.01
    ) == 1
    assert (tmp_path / "healthy_finished").exists()
    assert json.loads((logs / "exit_codes.json").read_text()) == {
        "mode3_preskill": 7, "mode3_preskill_proprioX": 0,
    }


def test_cancel_stops_every_worker_and_records_failure(tmp_path):
    worker = '''
import os, sys, time
from pathlib import Path
mode = os.environ["SKILL_AUX_PREDICTOR_MODE_OVERRIDE"]
Path(sys.argv[1], mode + ".pid").write_text(str(os.getpid()))
time.sleep(60)
'''
    logs = tmp_path / "logs"
    runs = _runs(tmp_path)
    process = subprocess.Popen([
        sys.executable, str(SCRIPT), "--runs", json.dumps(runs), "--log-dir", str(logs),
        "--", sys.executable, "-c", worker, str(tmp_path),
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _wait_for_files([tmp_path / f"{run['mode']}.pid" for run in runs])
        pids = [int((tmp_path / f"{run['mode']}.pid").read_text()) for run in runs]
        process.send_signal(signal.SIGTERM)
        out, err = process.communicate(timeout=12)
        assert process.returncode == 128 + signal.SIGTERM, (out, err)
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        assert all(code != 0 for code in json.loads((logs / "exit_codes.json").read_text()).values())
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=12)


def test_startup_error_cleans_up_already_started_workers(tmp_path, monkeypatch):
    original = subprocess.Popen
    started = []

    def start(*args, **kwargs):
        if started:
            raise OSError("simulated worker startup error")
        process = original(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(MODULE.subprocess, "Popen", start)
    with pytest.raises(OSError, match="simulated"):
        MODULE.run_group(
            _runs(tmp_path), [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path / "logs", stop_timeout=.1,
        )
    assert started[0].poll() is not None


@pytest.mark.parametrize("modes", [["mode1", "mode1"], ["../unsafe", "mode2"]])
def test_group_rejects_duplicate_or_unsafe_mode_names(tmp_path, modes):
    runs = _runs(tmp_path)
    for run, mode in zip(runs, modes, strict=True):
        run["mode"] = mode
    with pytest.raises(ValueError):
        MODULE.run_group(runs, [sys.executable], tmp_path / "logs")


def _terminator_runs(tmp_path):
    return [{"architecture": name, "run_name": name, "output_dir": str(tmp_path / name)}
            for name in ("term_token_3_both", "term_token_4_both")]


def test_terminator_workers_overlap_with_separate_wandb_and_selector_environments(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-shared-term")
    monkeypatch.setenv("SKILL_AUX_PREDICTOR_MODE_OVERRIDE", "must-not-inherit")
    monkeypatch.setenv("WANDB_RUN_ID", "must-not-share")
    monkeypatch.setenv("WANDB_NAME", "must-not-share")
    worker = '''
import json, os, sys, time
from pathlib import Path
root=Path(sys.argv[1])
name=os.environ["SKILL_AUX_TERMINATOR_ARCHITECTURE_OVERRIDE"]
(root/(name+".ready")).write_text(json.dumps(dict(os.environ)))
deadline=time.monotonic()+5
while len(list(root.glob("*.ready")))<2:
    if time.monotonic()>deadline: sys.exit(9)
    time.sleep(.01)
print(name)
'''
    logs = tmp_path / "logs"
    runs = _terminator_runs(tmp_path)
    assert MODULE.run_group(runs, [sys.executable, "-c", worker, str(tmp_path)], logs,
                            component="terminator", poll_interval=.01) == 0
    for run in runs:
        name = run["architecture"]
        env = json.loads((tmp_path / f"{name}.ready").read_text())
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-shared-term"
        assert env["SKILL_AUX_GROUP_WORKER"] == "1"
        assert "SKILL_AUX_PREDICTOR_MODE_OVERRIDE" not in env
        assert "WANDB_RUN_ID" not in env and "WANDB_NAME" not in env
        assert (logs / f"{name}.out").read_text().strip() == name
        assert not Path(run["output_dir"]).exists()


def test_terminator_failure_keeps_healthy_model_running(tmp_path):
    worker = '''
import os, sys, time
from pathlib import Path
root=Path(sys.argv[1])
if os.environ["SKILL_AUX_TERMINATOR_ARCHITECTURE_OVERRIDE"]=="term_token_3_both":
    (root/"failed").touch()
    sys.exit(7)
while not (root/"failed").exists(): time.sleep(.01)
time.sleep(.1)
(root/"healthy_finished").touch()
'''
    logs = tmp_path / "logs"
    assert MODULE.run_group(_terminator_runs(tmp_path), [sys.executable, "-c", worker, str(tmp_path)], logs,
                            component="terminator", poll_interval=.01) == 1
    assert (tmp_path / "healthy_finished").is_file()
    assert json.loads((logs / "exit_codes.json").read_text()) == {"term_token_3_both": 7, "term_token_4_both": 0}


def test_cancel_stops_all_terminator_workers(tmp_path):
    worker = '''
import os, sys, time
from pathlib import Path
name=os.environ["SKILL_AUX_TERMINATOR_ARCHITECTURE_OVERRIDE"]
Path(sys.argv[1],name+".pid").write_text(str(os.getpid()))
time.sleep(60)
'''
    logs = tmp_path / "logs"
    runs = _terminator_runs(tmp_path)
    process = subprocess.Popen([
        sys.executable, str(SCRIPT), "--component", "terminator", "--runs", json.dumps(runs),
        "--log-dir", str(logs), "--", sys.executable, "-c", worker, str(tmp_path),
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        paths = [tmp_path / f"{run['architecture']}.pid" for run in runs]
        _wait_for_files(paths)
        pids = [int(path.read_text()) for path in paths]
        process.send_signal(signal.SIGTERM)
        out, err = process.communicate(timeout=12)
        assert process.returncode == 128 + signal.SIGTERM, (out, err)
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        assert all(code != 0 for code in json.loads((logs / "exit_codes.json").read_text()).values())
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=12)
