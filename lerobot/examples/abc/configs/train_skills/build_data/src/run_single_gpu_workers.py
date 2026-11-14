#!/usr/bin/env python3
"""Run ABC curve shards concurrently inside one shared GPU allocation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import threading


class ABCSharedGPUBuilder:
    """Bounded process pool whose children all inherit one visible CUDA device."""

    def __init__(self, env=None, *, src: Path | None = None, log_dir: Path | None = None):
        self.env = dict(os.environ if env is None else env)
        self.src = Path(src) if src is not None else Path(__file__).resolve().parent
        self.workers = int(self.env["SKILLSET_SINGLE_GPU_WORKERS"])
        self.tasks_per_job = int(self.env["SKILLSET_TASKS_PER_JOB"])
        self.task_ids = [int(value) for value in json.loads(self.env["ABC_TASK_INDICES_JSON"])]
        if self.workers < 1:
            raise ValueError("SKILLSET_SINGLE_GPU_WORKERS must be positive in shared-GPU mode.")
        if self.tasks_per_job < 1 or not self.task_ids:
            raise ValueError("ABC task sharding must contain tasks and use a positive shard size.")

        job_id = self.env.get("SLURM_JOB_ID", "local")
        self.log_dir = (
            Path(log_dir)
            if log_dir is not None
            else self.src.parent / "logs" / f"shared_gpu_{job_id}"
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.processes: set[subprocess.Popen] = set()
        self.stopping = False

    @property
    def shard_count(self) -> int:
        return (len(self.task_ids) + self.tasks_per_job - 1) // self.tasks_per_job

    def stop(self) -> None:
        with self.lock:
            self.stopping = True
            for process in self.processes:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def _run_shard(self, shard_index: int, *, phase: str) -> int:
        if phase not in {"curves", "cached_segment"}:
            raise ValueError(f"Unsupported ABC shared-GPU phase: {phase}")
        env = dict(
            self.env,
            ABC_WORKER_SHARD_INDEX=str(shard_index),
            CURVES_ONLY=str(phase == "curves").lower(),
            USE_CACHED_CURVES=str(phase == "cached_segment").lower(),
        )
        # Divide the allocation's CPU threads across concurrent GPU workers.
        total_cpus = int(env.get("SLURM_CPUS_PER_TASK", "1") or 1)
        threads = str(max(1, total_cpus // min(self.workers, self.shard_count)))
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "ARROW_NUM_THREADS",
        ):
            env[name] = threads

        stdout = self.log_dir / f"{phase}_shard_{shard_index:03d}.out"
        stderr = self.log_dir / f"{phase}_shard_{shard_index:03d}.err"
        print(
            f"[shared GPU:{phase}] shard {shard_index}/{self.shard_count - 1}: {stdout}",
            flush=True,
        )
        with stdout.open("wb") as out, stderr.open("wb") as err, self.lock:
            if self.stopping:
                return 130
            process = subprocess.Popen(
                ["bash", str(self.src / "build_skillset.sbatch")],
                cwd=self.src.parent,
                env=env,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            self.processes.add(process)
        try:
            code = process.wait()
        finally:
            with self.lock:
                self.processes.discard(process)
        if code:
            print(
                f"[shared GPU:{phase}] shard {shard_index} failed ({code}); see {stderr}",
                flush=True,
            )
        return code

    def _pool(self, *, phase: str) -> None:
        print(
            f"ABC shared-GPU {phase}: {self.shard_count} shards, "
            f"{self.workers} concurrent workers",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=min(self.workers, self.shard_count)) as executor:
            futures = [
                executor.submit(self._run_shard, shard_index, phase=phase)
                for shard_index in range(self.shard_count)
            ]
            codes = [future.result() for future in futures]
        failures = [index for index, code in enumerate(codes) if code]
        if failures:
            raise RuntimeError(
                f"ABC shared-GPU {phase} shards failed: {failures}; see {self.log_dir}"
            )

    def _stage(self, script_name: str) -> None:
        label = Path(script_name).stem
        stdout = self.log_dir / f"{label}.out"
        stderr = self.log_dir / f"{label}.err"
        print(f"[shared GPU:{label}] logs: {stdout}", flush=True)
        with stdout.open("wb") as out, stderr.open("wb") as err:
            result = subprocess.run(
                ["bash", str(self.src / script_name)],
                cwd=self.src.parent,
                env=self.env,
                stdout=out,
                stderr=err,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"ABC stage {script_name} failed; see {stderr}")

    def run(self) -> None:
        print(
            f"ABC one-allocation build: {self.shard_count} shards, "
            f"{self.workers} shared-GPU workers, logs={self.log_dir}",
            flush=True,
        )
        self._pool(phase="curves")
        self._stage("compute_global_boundary_threshold.sbatch")
        self._pool(phase="cached_segment")
        self._stage("mark_skillset_complete.sbatch")
        print("ABC shared-GPU build complete.", flush=True)


def main() -> None:
    builder = ABCSharedGPUBuilder()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: builder.stop())
    try:
        builder.run()
    finally:
        if builder.stopping:
            builder.stop()


if __name__ == "__main__":
    main()
