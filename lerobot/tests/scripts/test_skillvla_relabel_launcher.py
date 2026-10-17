"""Exercise relabel submission in an isolated checkout, without real jobs or GPUs."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


CONFIGS = Path(__file__).resolve().parents[2] / "examples/libero/configs"


def _checkout(tmp_path, *, invalid_config=False):
    root = tmp_path / "repo"
    configs = root / "lerobot/examples/libero/configs"
    build = configs / "train_skillVLA/build_data"
    (build / "src").mkdir(parents=True)
    (configs / "src").mkdir()
    shutil.copy2(CONFIGS / "train_skillVLA/build_data/submit_relabel.sh", build / "submit_relabel.sh")
    shutil.copy2(CONFIGS / "src/submit_job.sh", configs / "src/submit_job.sh")
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    settings = {
        "PROJECT_ROOT": str(root), "LEROBOT_ROOT": str(root / "lerobot"),
        "SKILLVLA_PARTITION": "local", "SKILLVLA_QOS": "base_qos", "SKILLVLA_GRES": "gpu:1",
        "SKILLVLA_CPUS_PER_TASK": "8", "SKILLVLA_MEM": "64G", "SKILLVLA_TIME": "1:00:00",
        "SKILLVLA_NODELIST": "", "SKILLVLA_EXCLUDE_NODES": "",
        "RELABEL_SOURCE_RUN_DIR": "/dataset/original", "RELABEL_OUTPUT_RUN_DIR": "/dataset/relabeled",
        "RELABEL_PREDICTOR_MODEL": "joint", "RELABEL_PREDICTOR_CHECKPOINT": "003000",
    }
    emitter = "raise SystemExit(2)\n" if invalid_config else (
        f"import shlex\nsettings = {settings!r}\n"
        "for key, value in settings.items(): print(f'export {key}={shlex.quote(value)}')\n"
    )
    (build / "src/train_skillVLA_config.py").write_text(emitter)
    (build / "src/relabel_skillvla.sbatch").write_text(
        "#!/bin/bash\n#SBATCH --job-name=skillVLA_relabel\n"
        "#SBATCH --output=logs/%x_%j.out\n#SBATCH --error=logs/%x_%j.err\n"
        'test -f "$SKILLVLA_ENV_SNAPSHOT" || exit 9\n'
        'echo "source=$SKILLVLA_RELABEL_SRC_DIR"\n'
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text(
        '#!/bin/bash\ntest -f "$SKILLVLA_ENV_SNAPSHOT" || exit 9\n'
        'echo "slurm source=$SKILLVLA_RELABEL_SRC_DIR"\n'
    )
    (bin_dir / "nvidia-smi").write_text("#!/bin/bash\nprintf '0\\n1\\n'\n")
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SBD_", "SLURM_", "CUDA_VISIBLE"))}
    for key in ("TRAIN_SKILLVLA_CONFIG", "SOURCE_DATA", "SKILLVLA_ENV_SNAPSHOT", "SKILLVLA_RELABEL_SRC_DIR"):
        env.pop(key, None)
    env.update(PATH=f"{bin_dir}:{env['PATH']}", SBD_LOCAL_JOBS_DIR=str(tmp_path / "jobs"))
    return build, env


@pytest.mark.parametrize("scheduler", ["slurm", "local"])
def test_relabel_uses_the_shared_scheduler_and_frozen_settings(tmp_path, scheduler):
    build, env = _checkout(tmp_path)
    result = subprocess.run(["bash", str(build / "submit_relabel.sh")],
                            env={**env, "SBD_SCHEDULER": scheduler}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    if scheduler == "slurm":
        assert f"slurm source={build / 'src'}" in result.stdout
        assert not (tmp_path / "jobs").exists()
    else:
        assert "Submitted local job" in result.stdout
        deadline = time.monotonic() + 10
        while not list((tmp_path / "jobs").glob("*.exit")):
            assert time.monotonic() < deadline, "local smoke job did not finish"
            time.sleep(0.05)
        exit_file = next((tmp_path / "jobs").glob("*.exit"))
        assert exit_file.read_text().strip() == "0"
        output = next((build / "logs").glob("skillVLA_relabel_*.out"))
        assert f"source={build / 'src'}" in output.read_text()


def test_invalid_relabel_config_never_submits_a_job(tmp_path):
    build, env = _checkout(tmp_path, invalid_config=True)
    result = subprocess.run(["bash", str(build / "submit_relabel.sh")],
                            env={**env, "SBD_SCHEDULER": "slurm"}, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "no job was submitted" in result.stderr
    assert "slurm source=" not in result.stdout
