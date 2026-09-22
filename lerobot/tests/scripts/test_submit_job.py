"""src/submit_job.sh: sbatch on Slurm servers, detached local runs where the server has no Slurm."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[2] / "examples/libero/configs"
SUBMIT_JOB = CONFIGS / "src/submit_job.sh"
STAGE = CONFIGS / "src/stage_skillvla_dataset.sh"


def _env(tmp_path: Path, scheduler: str, **extra: str) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "sbatch").write_text('#!/bin/bash\necho "sbatch $* FOO=${FOO:-unset}"\n')
    (bin_dir / "nvidia-smi").write_text("#!/bin/bash\nprintf '0\\n1\\n'\n")       # a 2-GPU machine
    for tool in ("sbatch", "nvidia-smi"):
        (bin_dir / tool).chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBD_", "CUDA_VISIBLE"))}
    env.update(PATH=f"{bin_dir}:{env['PATH']}", SBD_SCHEDULER=scheduler,
               SBD_LOCAL_JOBS_DIR=str(tmp_path / "jobs"), **extra)
    return env


def _submit(tmp_path: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    command = f'set -euo pipefail; source "{SUBMIT_JOB}"; FOO=bar submit_job "$@"'
    return subprocess.run(["bash", "-c", command, "_", *args], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=60)


def _cli(tmp_path: Path, env: dict, *args: str) -> str:
    return subprocess.run(["bash", str(SUBMIT_JOB), *args], env=env, capture_output=True, text=True,
                          check=True, timeout=60).stdout


def _job_script(tmp_path: Path, body: str, gres: str = "gpu:1") -> Path:
    script = tmp_path / "job.sbatch"
    script.write_text(
        "#!/usr/bin/env bash\n#SBATCH --job-name=demo\n"
        f"#SBATCH --gres={gres}\n#SBATCH --output=logs/%x_%j.out\n#SBATCH --error=logs/%x_%j.err\n{body}\n"
    )
    return script


def _wait(predicate, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, "timed out"
        time.sleep(0.1)


def test_slurm_servers_still_call_sbatch_unchanged(tmp_path: Path) -> None:
    script = _job_script(tmp_path, "exit 0")
    result = _submit(tmp_path, _env(tmp_path, "slurm"), "--partition=a6000", "--qos=base_qos", str(script))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"sbatch --partition=a6000 --qos=base_qos {script} FOO=bar"
    assert not (tmp_path / "jobs").exists()


def test_local_job_runs_detached_with_env_logs_and_a_gpu(tmp_path: Path) -> None:
    env = _env(tmp_path, "local")
    script = _job_script(tmp_path, 'echo "gpu=${CUDA_VISIBLE_DEVICES} foo=${FOO} cwd=${PWD} arg=$1"; echo oops >&2; exit 3')
    result = _submit(tmp_path, env, "--partition=ignored", "--mem=128G", "--parsable", str(script), "A1")
    assert result.returncode == 0, result.stderr
    job_id = result.stdout.strip()
    assert job_id.isdigit()                                  # FSQ parses job ids as numbers
    _wait(lambda: (tmp_path / "jobs" / f"{job_id}.exit").exists())
    assert (tmp_path / f"logs/demo_{job_id}.out").read_text().strip() == f"gpu=0 foo=bar cwd={tmp_path} arg=A1"
    assert (tmp_path / f"logs/demo_{job_id}.err").read_text().strip() == "oops"
    assert "failed(3)" in _cli(tmp_path, env, "list")


def test_gpus_are_shared_out_and_released_on_stop(tmp_path: Path) -> None:
    env = _env(tmp_path, "local")
    script = _job_script(tmp_path, "sleep 60")
    ids = []
    try:
        for _ in range(2):
            result = _submit(tmp_path, env, "--parsable", str(script))
            assert result.returncode == 0, result.stderr
            ids.append(result.stdout.strip())
        records = [(tmp_path / "jobs" / f"{i}.job").read_text() for i in ids]
        assert ["gpus=0" in records[0], "gpus=1" in records[1]] == [True, True]

        full = _submit(tmp_path, env, str(script))
        assert full.returncode != 0 and "only 0 are free" in full.stderr

        assert "Sent SIGTERM" in _cli(tmp_path, env, "stop", ids[0])
        _wait(lambda: "stopped" in _cli(tmp_path, env, "list"))
        again = _submit(tmp_path, env, "--parsable", str(script))
        assert again.returncode == 0, again.stderr
        ids.append(again.stdout.strip())
        assert "gpus=0" in (tmp_path / "jobs" / f"{ids[-1]}.job").read_text()
        assert _cli(tmp_path, env, "list").count("running") == 2
    finally:
        for job_id in ids:
            _cli(tmp_path, env, "stop", job_id)


def test_slurm_only_features_are_refused_locally(tmp_path: Path) -> None:
    script = _job_script(tmp_path, "exit 0")
    env = _env(tmp_path, "local")
    result = _submit(tmp_path, env, "--array=0-3", str(script))
    assert result.returncode == 2 and "needs Slurm" in result.stderr
    result = _submit(tmp_path, env, "--dependency=afterok:1", str(script))
    assert result.returncode == 2 and "not a local job" in result.stderr


def test_dependencies_wait_and_a_failed_afterok_cancels(tmp_path: Path) -> None:
    env = _env(tmp_path, "local")
    first = _job_script(tmp_path, 'sleep 1; touch first.done; exit "${RC}"', gres="")

    def submit(*args: str, rc: str = "0") -> str:
        result = _submit(tmp_path, {**env, "RC": rc}, "--parsable", *args)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    ok = submit(str(first))
    after = tmp_path / "after.sbatch"
    after.write_text("#!/usr/bin/env bash\n#SBATCH --job-name=after\n#SBATCH --output=logs/%x_%j.out\n"
                     '[ -f first.done ] && echo "saw first"\n')
    waiting = submit("--dependency=afterok:" + ok, "--kill-on-invalid-dep=yes", str(after))
    assert "pending" in _cli(tmp_path, env, "list")
    _wait(lambda: (tmp_path / "jobs" / f"{waiting}.exit").exists())
    assert (tmp_path / f"logs/after_{waiting}.out").read_text().strip() == "saw first"

    (tmp_path / "first.done").unlink()
    bad = submit(str(first), rc="1")
    cancelled = submit("--dependency=afterok:" + bad, str(after))
    anyway = submit("--dependency=afterany:" + bad, str(after))
    for job_id in (cancelled, anyway):
        _wait(lambda: (tmp_path / "jobs" / f"{job_id}.exit").exists())
    assert (tmp_path / "jobs" / f"{cancelled}.exit").read_text().strip() == "dependency"
    assert "did not finish successfully" in (tmp_path / f"logs/after_{cancelled}.out").read_text()  # never ran
    assert (tmp_path / f"logs/after_{anyway}.out").read_text().strip() == "saw first"
    assert "cancelled(dep)" in _cli(tmp_path, env, "list")


def test_jobs_do_not_inherit_the_submitters_locks(tmp_path: Path) -> None:
    env = _env(tmp_path, "local", SBD_GPUS="0")
    script = _job_script(tmp_path, "sleep 60")
    lock = tmp_path / "submit.lock"
    command = (f'source "{SUBMIT_JOB}"; exec 9>"{lock}"; flock 9; '      # like the FSQ submit script
               f'id="$(submit_job --parsable "{script}")"; flock -u 9; exec 9>&-; '
               f'flock -n "{lock}" true && echo "free $id"')
    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, env=env, capture_output=True, text=True,
                            timeout=60)
    job_id = result.stdout.split()[-1]
    try:
        assert result.stdout.startswith("free "), result.stderr
    finally:
        _cli(tmp_path, env, "stop", job_id)


def test_without_slurm_the_dataset_is_staged_once_and_reused(tmp_path: Path) -> None:
    run = tmp_path / "dataset/run"
    (run / "skillvla/data").mkdir(parents=True)
    (run / "skillvla/data/a.parquet").write_text("a")
    (run / "skill_focus_uv.npz").write_text("uv")
    command = (f'source "{STAGE}"; stage_skillvla_dataset_on_node "{run}/skillvla" test >/dev/null 2>&1; '
               'echo "${STAGED_SKILLVLA_DATASET_DIR}"')

    def stage(**extra: str) -> Path:
        env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
        env.update(SKILLVLA_LOCAL_STAGE_ROOT=str(tmp_path / "stage"), **extra)
        out = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, check=True)
        return Path(out.stdout.strip())

    first, second = stage(), stage()
    assert first == second and first.parent.name.split("_skillvla_")[1].startswith("shared_")
    assert (first / "data/a.parquet").read_text() == "a" and (first.parent / "skill_focus_uv.npz").exists()
    assert stage(SLURM_JOB_ID="123").parent.name.endswith("_skillvla_123")      # Slurm: per job, as before
