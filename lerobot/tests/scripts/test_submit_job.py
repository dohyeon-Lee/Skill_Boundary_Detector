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
               SBD_SLURM_SUBMIT_MODE="sbatch", SBD_LOCAL_JOBS_DIR=str(tmp_path / "jobs"), SBD_LOCAL_DEP_POLL="1")
    env.update(extra)
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


def _wait(predicate, timeout: float = 60.0) -> None:
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

        full = _submit(tmp_path, env, "--parsable", str(script))
        assert full.returncode == 0, full.stderr
        queued = full.stdout.strip()
        ids.append(queued)
        assert "pending" in _cli(tmp_path, env, "list")

        assert "Sent SIGTERM" in _cli(tmp_path, env, "stop", ids[0])
        _wait(lambda: "gpus=0" in (tmp_path / "jobs" / f"{queued}.job").read_text())
        assert _cli(tmp_path, env, "list").count("running") == 2
    finally:
        for job_id in ids:
            _cli(tmp_path, env, "stop", job_id)


def test_unknown_local_dependency_is_rejected(tmp_path: Path) -> None:
    script = _job_script(tmp_path, "exit 0")
    env = _env(tmp_path, "local")
    result = _submit(tmp_path, env, "--dependency=afterok:1", str(script))
    assert result.returncode == 2 and "not a local job" in result.stderr


def test_local_array_waits_for_one_gpu_and_keeps_every_index_and_log(tmp_path):
    env = _env(tmp_path, "local", SBD_GPUS="0")
    script = _job_script(tmp_path, '''set -e
test -z "${SLURM_JOB_ID:-}"
mkdir gpu.lock
trap 'rmdir gpu.lock' EXIT
echo "${SLURM_ARRAY_TASK_ID}:gpu=${CUDA_VISIBLE_DEVICES}:count=${SLURM_ARRAY_TASK_COUNT}"
echo "${SLURM_ARRAY_TASK_ID}" >> visited
sleep 0.1''')
    result = _submit(tmp_path, env, "--parsable", "--array=1,3-7:2%4",
                     "--output=logs/%x_%A_%a.out", str(script))
    assert result.returncode == 0, result.stderr
    identifier = result.stdout.strip()
    completion = tmp_path / "jobs" / f"{identifier}.exit"
    _wait(completion.exists)
    assert completion.read_text().strip() == "0"
    assert (tmp_path / "visited").read_text().splitlines() == ["1", "3", "5", "7"]
    for index in (1, 3, 5, 7):
        assert (tmp_path / f"logs/demo_{identifier}_{index}.out").read_text().strip() == f"{index}:gpu=0:count=4"


def test_gpu_dependency_chain_waits_without_reserving_the_producers_gpu(tmp_path):
    env = _env(tmp_path, "local", SBD_GPUS="0")
    producer = _job_script(tmp_path, "sleep 0.3; touch ready")
    first = _submit(tmp_path, env, "--parsable", str(producer)).stdout.strip()
    follower = tmp_path / "follower.sbatch"
    follower.write_text('#!/bin/bash\n#SBATCH --gres=gpu:1\ntest -f ready && echo ok > followed\n')
    second = _submit(tmp_path, env, "--parsable", f"--dependency=afterok:{first}", str(follower))
    assert second.returncode == 0, second.stderr
    completion = tmp_path / "jobs" / f"{second.stdout.strip()}.exit"
    _wait(completion.exists)
    assert completion.read_text().strip() == "0"
    assert (tmp_path / "followed").read_text().strip() == "ok"


def test_failed_array_blocks_afterok_but_afterany_runs_after_all_tasks(tmp_path):
    env = _env(tmp_path, "local", SBD_GPUS="0")
    script = _job_script(tmp_path, 'echo "$SLURM_ARRAY_TASK_ID" >> finished; test "$SLURM_ARRAY_TASK_ID" != 1')
    result = _submit(tmp_path, env, "--parsable", "--array=0-2", str(script))
    assert result.returncode == 0, result.stderr
    first = result.stdout.strip()
    successors = []
    for mode, command in (("afterok", "touch should-not-run"), ("afterany", "test $(wc -l < finished) = 3 && touch repaired")):
        result = _submit(tmp_path, env, "--parsable", f"--dependency={mode}:{first}", f"--wrap={command}")
        assert result.returncode == 0, result.stderr
        successors.append(tmp_path / "jobs" / f"{result.stdout.strip()}.exit")
    for completion in successors:
        _wait(completion.exists)
    assert (tmp_path / "jobs" / f"{first}.exit").read_text().strip() != "0"
    assert successors[0].read_text().strip() == "dependency"
    assert successors[1].read_text().strip() == "0"
    assert not (tmp_path / "should-not-run").exists()
    assert (tmp_path / "repaired").exists()


def test_local_wrap_keeps_quoted_arguments_env_and_workdir(tmp_path):
    env = _env(tmp_path, "local")
    result = _submit(tmp_path, env, "--parsable", '--wrap=printf "%s|%s" "$FOO" "$PWD" > "result with spaces"')
    assert result.returncode == 0, result.stderr
    _wait((tmp_path / "jobs" / f"{result.stdout.strip()}.exit").exists)
    assert (tmp_path / "result with spaces").read_text() == f"bar|{tmp_path}"


def test_raw_batch_adapter_preserves_slurm_arguments_even_with_srun_server_mode(tmp_path):
    env = _env(tmp_path, "slurm", SBD_SLURM_SUBMIT_MODE="srun", FOO="pipeline")
    script = _job_script(tmp_path, "exit 0")
    args = ["--parsable", "--array=0-7%2", "--dependency=afterany:123", "--mem=48G", str(script)]
    output = _cli(tmp_path, env, "batch", *args)
    assert output.strip() == "sbatch " + " ".join(args) + " FOO=pipeline"
    assert not (tmp_path / "jobs").exists()


def test_stop_local_array_stops_its_children_and_does_not_start_waiting_tasks(tmp_path):
    env = _env(tmp_path, "local", SBD_GPUS="0")
    script = _job_script(tmp_path, 'echo "$BASHPID" > child.pid; echo "$SLURM_ARRAY_TASK_ID" >> visited; sleep 60')
    result = _submit(tmp_path, env, "--parsable", "--array=0-3", str(script))
    assert result.returncode == 0, result.stderr
    identifier = result.stdout.strip()
    try:
        _wait(lambda: (tmp_path / "child.pid").exists() and (tmp_path / "visited").exists())
        child = int((tmp_path / "child.pid").read_text())
        assert "Sent SIGTERM" in _cli(tmp_path, env, "stop", identifier)
        _wait((tmp_path / "jobs" / f"{identifier}.exit").exists)
        assert (tmp_path / "jobs" / f"{identifier}.exit").read_text().strip() == "143"
        assert (tmp_path / "visited").read_text().splitlines() == ["0"]
        proc_stat = Path(f"/proc/{child}/stat")
        assert not proc_stat.exists() or proc_stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    finally:
        _cli(tmp_path, env, "stop", identifier)


def test_local_array_job_environment_is_not_saved_to_metadata(tmp_path):
    env = _env(tmp_path, "local", HF_TOKEN="test-secret-not-on-disk")
    result = _submit(tmp_path, env, "--parsable", "--wrap=test -n \"$HF_TOKEN\"")
    assert result.returncode == 0, result.stderr
    identifier = result.stdout.strip()
    _wait((tmp_path / "jobs" / f"{identifier}.exit").exists)
    assert (tmp_path / "jobs" / f"{identifier}.exit").read_text().strip() == "0"
    for record in (tmp_path / "jobs").iterdir():
        assert "test-secret-not-on-disk" not in record.read_text()


def test_local_export_option_passes_frozen_eval_config_without_saving_secret_values(tmp_path):
    env = _env(tmp_path, "local")
    result = _submit(tmp_path, env, "--parsable",
                     "--export=ALL,DP_ABLATION_DIR=path with spaces,DP_ABLATION_CONFIG=config.yaml,PRIVATE_VALUE=not-in-records",
                     '--wrap=test "$FOO" = bar && test -n "$PRIVATE_VALUE" && printf "%s|%s" "$DP_ABLATION_DIR" "$DP_ABLATION_CONFIG" > exported')
    assert result.returncode == 0, result.stderr
    identifier = result.stdout.strip()
    _wait((tmp_path / "jobs" / f"{identifier}.exit").exists)
    assert (tmp_path / "exported").read_text() == "path with spaces|config.yaml"
    assert "not-in-records" not in (tmp_path / "jobs" / f"{identifier}.json").read_text()


def test_checkpoint_stop_watcher_can_stop_a_completed_local_training_checkpoint(tmp_path):
    env = _env(tmp_path, "local", SBD_GPUS="0", POLL_SECONDS="0.05")
    run = tmp_path / "model"
    checkpoint = run / "checkpoints/100000"
    (checkpoint / "pretrained_model").mkdir(parents=True)
    (checkpoint / "pretrained_model/model.safetensors").write_bytes(b"complete")
    (run / "checkpoints/last").symlink_to("100000")
    script = _job_script(tmp_path, f'echo "output : {run}"; sleep 60')
    submitted = _submit(tmp_path, env, "--parsable", str(script))
    assert submitted.returncode == 0, submitted.stderr
    identifier = submitted.stdout.strip()
    try:
        result = subprocess.run(["bash", str(CONFIGS / "src/stop_at_checkpoint.sh"), "100000", identifier],
                                cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "checkpoint 100000 complete" in result.stdout
        _wait((tmp_path / "jobs" / f"{identifier}.exit").exists)
        assert (tmp_path / "jobs" / f"{identifier}.exit").read_text().strip() == "143"
    finally:
        _cli(tmp_path, env, "stop", identifier)


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


def test_additional_steps_do_not_replace_another_steps_dataset(tmp_path):
    datasets=[]
    for index in (1,2):
        run=tmp_path/f'run{index}'
        (run/'skillvla/data').mkdir(parents=True)
        (run/'skillvla/data/a.parquet').write_text(str(index))
        env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
        env.update(SLURM_JOB_ID='123',SLURM_STEP_ID=str(index),SBD_SRUN_EXISTING_STEP='1',
                   SKILLVLA_LOCAL_STAGE_ROOT=str(tmp_path/'stage'))
        command='source "$1"; stage_skillvla_dataset_on_node "$2" test; echo "$STAGED_SKILLVLA_DATASET_DIR"'
        result=subprocess.run(['bash','-c',command,'_',str(STAGE),str(run/'skillvla')],env=env,
                              capture_output=True,text=True,check=True,timeout=10)
        path=Path(result.stdout.strip())
        assert path.parent.name.endswith(f'_123_step{index}')
        datasets.append(path)
    assert datasets[0]!=datasets[1]
    assert (datasets[0]/'data/a.parquet').read_text()=='1'
    assert (datasets[1]/'data/a.parquet').read_text()=='2'
