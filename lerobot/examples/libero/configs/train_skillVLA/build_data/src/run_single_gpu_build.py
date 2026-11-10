"""Run the existing SkillVLA stages in one allocation with a bounded task pool."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC.parents[2] / "train_skills/src"))
from sbd_build_contract import check_run  # noqa: E402


class SingleGPUBuilder:
    def __init__(self, env=None, *, src=SRC, log_dir=None):
        self.env = dict(os.environ if env is None else env)
        self.src = Path(src)
        self.cwd = self.src.parent
        self.workers = int(self.env["SKILLSET_SINGLE_GPU_WORKERS"])
        self.chunk_size = int(self.env["SKILLSET_TASKS_PER_JOB"])
        if self.workers < 1 or self.chunk_size < 1:
            raise ValueError("Single GPU workers and tasks per chunk must be positive.")
        self.env.update(SLURM_SUBMIT_DIR=str(self.cwd), SBD_SRUN_SESSION="1")
        for key in ("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_JOB_ID", "ALL_TASK_IDS", "CURVES_ONLY", "USE_CACHED_CURVES"):
            self.env.pop(key, None)
        self.lock = threading.RLock()
        self.processes = set()
        self.stopping = False
        self.stop_time = None
        self.stop_reason = None
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.log_dir = Path(log_dir) if log_dir else self.cwd / "logs" / f"single_gpu_{suffix}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def stop(self, reason=None):
        """Terminate this build's child processes only; never signal Slurm."""
        with self.lock:
            self.stopping = True
            self.stop_time = self.stop_time or time.monotonic()
            self.stop_reason = self.stop_reason or reason
            for process in self.processes:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def _command(self, command, label, *, env=None, capture=False):
        stdout = self.log_dir / f"{label}.out"
        stderr = self.log_dir / f"{label}.err"
        print(f"[{label}] logs: {stdout}", flush=True)
        with stdout.open("ab") as out, stderr.open("ab") as err:
            with self.lock:
                if self.stopping:
                    raise InterruptedError(self.stop_reason or "Build stopped; no additional worker was started.")
                process = subprocess.Popen(command, cwd=self.cwd, env=env or self.env,
                                           stdout=out, stderr=err, start_new_session=True)
                self.processes.add(process)
            try:
                while True:
                    try:
                        code = process.wait(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        if self.stopping and time.monotonic() - self.stop_time > 5:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
            finally:
                with self.lock:
                    self.processes.discard(process)
        if self.stopping:
            raise InterruptedError(self.stop_reason or "Build stopped; the GPU allocation was left unchanged.")
        if code:
            tails = []
            for path in (stdout, stderr):
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 65536))
                    tails.append(stream.read().decode("utf-8", errors="replace").lower())
            text = "\n".join(tails)
            if any(term in text for term in ("no space left on device", "disk quota exceeded", "errno 28", "errno 122")):
                reason = f"Storage full in {label}; stopped without retry. See {stderr}"
                self.stop(reason)
                raise RuntimeError(reason)
            if "out of memory" in text or "cuda error" in text:
                reason = f"GPU failure in {label}; stopped this build's workers. See {stderr}"
                self.stop(reason)
                raise RuntimeError(reason)
        return code, stdout.read_text() if capture else ""

    def _stage(self, name, *, env=None, allow_failure=False):
        code, _ = self._command(["bash", str(self.src / name)], Path(name).stem, env=env)
        if code and not allow_failure:
            raise RuntimeError(f"{name} failed ({code}); see {self.log_dir}")
        return code

    def _complete_skillset(self):
        code, _ = self._command(
            [sys.executable, str(self.src / "verify_skillset.py"),
             "--dataset_dir", self.env["RAW_DATASET_DIR"],
             "--skillset_dir", self.env["SKILLSET_DIR"], "--check"], "check_skillset")
        if code not in (0, 1):
            raise RuntimeError(f"Skillset verification failed ({code}); see {self.log_dir}")
        return code == 0

    def _encoded(self):
        if not Path(self.env["SKILL_LATENTS_PATH"]).is_file():
            return False
        check_run(Path(self.env["SKILLSET_DIR"]), Path(self.env["SKILLVLA_RUN_DIR"]),
                  json.loads(self.env["SKILLSET_CONTRACT_JSON"]), "encoded")
        return True

    def _pool(self, task_ids, *, phase):
        chunks = [task_ids[i:i + self.chunk_size] for i in range(0, len(task_ids), self.chunk_size)]
        cpus = int(self.env.get("SKILLVLA_CPUS_PER_TASK", "8"))
        if self.env.get("SLURM_CPUS_PER_TASK", "").isdigit():
            cpus = min(cpus, int(self.env["SLURM_CPUS_PER_TASK"]))
        if hasattr(os, "sched_getaffinity"):
            cpus = min(cpus, len(os.sched_getaffinity(0)))
        threads = str(max(1, cpus // min(self.workers, len(chunks))))

        def work(item):
            index, ids = item
            env = dict(self.env, ALL_TASK_IDS=" ".join(map(str, ids)), SLURM_ARRAY_TASK_ID="0",
                       CURVES_ONLY=str(phase == "curves").lower(),
                       USE_CACHED_CURVES=str(phase == "cached_segment").lower())
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
                env[key] = threads
            print(f"[{phase}] chunk {index}: tasks {ids}", flush=True)
            return self._command(["bash", str(self.src / "build_skillset.sbatch")],
                                 f"{phase}_{index:03d}", env=env)[0]

        with ThreadPoolExecutor(max_workers=min(self.workers, len(chunks))) as executor:
            codes = list(executor.map(work, enumerate(chunks)))
        return codes

    def run(self):
        import pandas as pd

        dataset = Path(self.env["RAW_DATASET_DIR"])
        tasks = pd.read_parquet(dataset / "meta/tasks.parquet").reset_index()
        task_ids = sorted(int(value) for value in tasks["task_index"].unique())
        if not task_ids:
            raise ValueError("No dataset tasks to build.")
        print(f"Single GPU: {len(task_ids)} tasks, {self.workers} workers; logs={self.log_dir}", flush=True)
        if not self._complete_skillset():
            if self.env["SKILLSET_BOUNDARY_THRESHOLD_MODE"] == "global_mean":
                codes = self._pool(task_ids, phase="curves")
                if any(codes):
                    raise RuntimeError(f"Curve collection failed; no threshold/encoding submitted. See {self.log_dir}")
                if not self.env.get("SKILLSET_GLOBAL_THRESHOLD_SOURCE"):
                    parts = sorted((dataset / "meta/episodes").rglob("file-*.parquet"))
                    expected = sum(len(pd.read_parquet(path)) for path in parts)
                    if expected < 1:
                        raise ValueError("Source dataset has no episode metadata.")
                    self._stage("compute_global_boundary_threshold.sbatch",
                                env=dict(self.env, EXPECTED_EPISODES=str(expected)))
                self._pool(task_ids, phase="cached_segment")
            else:
                self._pool(task_ids, phase="segment")
            # Reuse the existing missing-episode retry policy/max_sweeps.
            self._stage("verify_skillset.sbatch")
        if not self._encoded():
            self._stage("encode_skills.sbatch")
        else:
            print("Verified skill latents already exist → skip encoding", flush=True)
        self._stage("build_skillvla.sbatch")
        print(f"Build complete → {self.env['SKILLVLA_RUN_DIR']}", flush=True)


def main():
    builder = SingleGPUBuilder()
    def stop(signum, frame):
        builder.stop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    try:
        builder.run()
    except Exception:
        builder.stop()
        raise


if __name__ == "__main__":
    main()
