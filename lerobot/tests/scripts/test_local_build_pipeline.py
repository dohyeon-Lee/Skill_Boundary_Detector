"""Run the real pipeline launcher against CPU workers / a fake Slurm boundary."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd
import pytest


CONFIGS = Path(__file__).resolve().parents[2] / "examples/libero/configs"


@pytest.mark.parametrize("scheduler", ["local", "slurm"])
@pytest.mark.parametrize("threshold", ["episode_mean", "global_mean"])
def test_skillvla_pipeline_keeps_array_dependencies_and_frozen_env(tmp_path, scheduler, threshold):
    root = tmp_path / "repo"
    configs = root / "lerobot/examples/libero/configs"
    build = configs / "train_skillVLA/build_data"
    src = build / "src"
    src.mkdir(parents=True)
    shared = configs / "src"
    shared.mkdir()
    shutil.copy2(CONFIGS / "train_skillVLA/build_data/submit_build_all.sh", build)
    for name in ("submit_job.sh", "local_jobs.py"):
        shutil.copy2(CONFIGS / "src" / name, shared)
    (root / ".venv").symlink_to(sys.prefix, target_is_directory=True)
    raw = root / "dataset"
    (raw / "meta/episodes").mkdir(parents=True)
    pd.DataFrame({"task_index": [0, 1, 2]}).to_parquet(raw / "meta/tasks.parquet")
    pd.DataFrame({"episode_index": [0, 1, 2]}).to_parquet(raw / "meta/episodes/file-000.parquet")
    (root / "dp").mkdir()
    (root / "fsq.pt").touch()
    output = root / "result"
    output.mkdir()
    settings = dict(
        ISS_NPZ_PATH=output / "iss.npz", FSQ_COPY_PATH=output / "FSQ.pt", FOCUS_UV_ENABLED="false",
        SKILLVLA_DATASET_DIR=output / "skillvla", RAW_DATASET_DIR=raw, DP_POLICY_PATH=root / "dp",
        FSQ_MODEL_PATH=root / "fsq.pt", SKILLSET_DIR=output / "skillset", SKILL_LATENTS_PATH=output / "latents.npz",
        SKILLVLA_PARTITION="original_partition", SKILLVLA_QOS="original_qos", SKILLVLA_GRES="gpu:1",
        SKILLVLA_CPUS_PER_TASK=1, SKILLVLA_MEM="12G", SKILLVLA_TIME="00:10:00",
        SKILLVLA_NODELIST="", SKILLVLA_EXCLUDE_NODES="", SKILLSET_TASKS_PER_JOB=1,
        SKILLSET_ARRAY_THROTTLE=2, SKILLSET_BOUNDARY_THRESHOLD_MODE=threshold,
        SKILLSET_MAX_SWEEPS=2, SKILLVLA_RUN_DIR=output,
    )
    (src / "train_skillVLA_config.py").write_text(
        f"import shlex\nsettings={dict((key, str(value)) for key, value in settings.items())!r}\n"
        "for key,value in settings.items(): print('export '+key+'='+shlex.quote(value))\n"
    )
    (src / "verify_skillset.py").write_text("raise SystemExit(1)\n")
    (src / "build_skillset.sbatch").write_text(
        '#!/bin/bash\nset -eu\ntest -f "$SKILLVLA_ENV_SNAPSHOT"\n'
        f'if [ "${{CURVES_ONLY:-false}}" = true ]; then echo "$SLURM_ARRAY_TASK_ID" >> "{output}/curves"; '
        f'else echo "$SLURM_ARRAY_TASK_ID" >> "{output}/segments"; fi\n'
    )
    commands = {
        "compute_global_boundary_threshold": f'test "$(wc -l < "{output}/curves")" = 3; test "$EXPECTED_EPISODES" = 3; touch "{output}/threshold"',
        "verify_skillset": f'test "$(wc -l < "{output}/segments")" = 3; touch "{output}/verified"',
        "encode_skills": f'test -f "{output}/verified"; touch "{output}/latents.npz"',
        "build_skillvla": f'test -f "{output}/latents.npz"; touch "{output}/finished"',
    }
    for name, command in commands.items():
        (src / f"{name}.sbatch").write_text('#!/bin/bash\nset -eu\ntest -f "$SKILLVLA_ENV_SNAPSHOT"\n' + command + "\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    submissions = tmp_path / "slurm.jsonl"
    sbatch = binaries / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport os,json,sys\nfrom pathlib import Path\n"
        f"p=Path({str(submissions)!r})\ncount=len(p.read_text().splitlines()) if p.exists() else 0\n"
        "assert Path(os.environ['SKILLVLA_ENV_SNAPSHOT']).is_file()\n"
        "with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\nprint(1000+count)\n"
    )
    sbatch.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SLURM_", "SBD_", "SKILLVLA_", "TRAIN_SKILLVLA_")) and key != "SOURCE_DATA"}
    env.update(SBD_SCHEDULER=scheduler, SBD_SLURM_SUBMIT_MODE="srun", SBD_GPUS="0",
               SBD_LOCAL_JOBS_DIR=str(tmp_path / "jobs"), PATH=str(binaries) + os.pathsep + env["PATH"])
    result = subprocess.run(["bash", str(build / "submit_build_all.sh")], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    expected_jobs = 4 if threshold == "episode_mean" else 6
    if scheduler == "slurm":
        args = [json.loads(line) for line in submissions.read_text().splitlines()]
        assert len(args) == expected_jobs
        assert "--array=0-2%2" in args[0]
        assert any("--dependency=afterany:" in arg for row in args for arg in row)
        assert all("--partition=original_partition" in row and "--qos=original_qos" in row for row in args)
        assert not (tmp_path / "jobs").exists()
    else:
        assert not submissions.exists()
        deadline = time.monotonic() + 20
        while len(list((tmp_path / "jobs").glob("*.exit"))) < expected_jobs:
            assert time.monotonic() < deadline, "local pipeline did not finish"
            time.sleep(0.05)
        for completion in (tmp_path / "jobs").glob("*.exit"):
            assert completion.read_text().strip() == "0", result.stdout + result.stderr
        assert (output / "finished").exists()
        assert sorted((output / "segments").read_text().splitlines()) == ["0", "1", "2"]
        if threshold == "global_mean":
            assert (output / "threshold").exists()
