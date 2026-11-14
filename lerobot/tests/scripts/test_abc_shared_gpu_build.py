from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
ABC_BUILD_SRC = ROOT / "lerobot/examples/abc/configs/train_skills/build_data/src"
ABC_CONFIG_SRC = ROOT / "lerobot/examples/abc/configs/train_skills/src/train_skills_config.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load(ABC_BUILD_SRC / "run_single_gpu_workers.py", "abc_single_gpu_workers")
CONFIG = _load(ABC_CONFIG_SRC, "abc_train_skills_config")


@pytest.mark.parametrize("workers", [0, 1, 6])
def test_abc_build_parallelism_accepts_array_or_shared_gpu(workers: int) -> None:
    settings = CONFIG.resolve_build_parallelism(
        {"skillset_tasks_per_job": 4, "skillset_single_gpu_workers": workers}
    )
    assert settings == {
        "skillset_tasks_per_job": 4,
        "skillset_single_gpu_workers": workers,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [("skillset_tasks_per_job", 0), ("skillset_single_gpu_workers", -1)],
)
def test_abc_build_parallelism_rejects_invalid_values(key: str, value: int) -> None:
    config = {"skillset_tasks_per_job": 4, "skillset_single_gpu_workers": 0}
    config[key] = value
    with pytest.raises(ValueError, match=key):
        CONFIG.resolve_build_parallelism(config)


def test_shared_gpu_pool_bounds_workers_and_covers_every_shard(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    state = tmp_path / "state.json"

    # The fake shard process updates a file under an advisory lock, allowing
    # the real subprocess pool to prove its peak concurrency and coverage.
    worker = src / "fake_worker.py"
    worker.write_text(
        """
import fcntl, json, os, time
from pathlib import Path
p = Path(os.environ['TEST_STATE'])
with p.with_suffix('.lock').open('a') as stream:
    fcntl.flock(stream, fcntl.LOCK_EX)
    data = json.loads(p.read_text()) if p.exists() else {'active': 0, 'peak': 0, 'seen': []}
    data['active'] += 1
    data['peak'] = max(data['peak'], data['active'])
    data['seen'].append(int(os.environ['ABC_WORKER_SHARD_INDEX']))
    p.write_text(json.dumps(data))
time.sleep(0.12)
with p.with_suffix('.lock').open('a') as stream:
    fcntl.flock(stream, fcntl.LOCK_EX)
    data = json.loads(p.read_text())
    data['active'] -= 1
    p.write_text(json.dumps(data))
""",
        encoding="utf-8",
    )
    (src / "build_skillset.sbatch").write_text(
        f"#!/usr/bin/env bash\nexec {sys.executable} {worker}\n", encoding="utf-8"
    )
    env = dict(
        os.environ,
        SKILLSET_SINGLE_GPU_WORKERS="2",
        SKILLSET_TASKS_PER_JOB="2",
        ABC_TASK_INDICES_JSON=json.dumps([59, 60, 61, 62, 63, 64, 65]),
        TEST_STATE=str(state),
        CUDA_VISIBLE_DEVICES="0",
        CURVES_ONLY="true",
        SLURM_CPUS_PER_TASK="8",
    )
    builder = RUNNER.ABCSharedGPUBuilder(env, src=src, log_dir=tmp_path / "logs")
    builder._pool(phase="curves")

    result = json.loads(state.read_text())
    assert result["peak"] == 2
    assert result["active"] == 0
    assert sorted(result["seen"]) == [0, 1, 2, 3]
