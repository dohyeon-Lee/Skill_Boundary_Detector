"""LIBERO SBD stages in one allocation, with balanced episode workers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import heapq
import importlib.util
import json
import os
from pathlib import Path
import re
import signal


SRC = Path(__file__).resolve().parent
# Reuse process ownership, cancellation and storage/GPU failure handling from
# the existing SkillVLA builder; this runner never encodes FSQ or builds VSA data.
_spec = importlib.util.spec_from_file_location(
    "libero_sbd_process_runner",
    SRC.parents[2] / "train_skillVLA/build_data/src/run_single_gpu_build.py",
)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)


def balance_episodes(episodes, workers):
    if workers < 1 or not episodes:
        raise ValueError("Episode sharding requires positive workers and nonempty episodes.")
    shards = [[] for _ in range(min(workers, len(episodes)))]
    loads = [(0, 0, i) for i in range(len(shards))]
    heapq.heapify(loads)
    for episode, frames in sorted(episodes, key=lambda item: (-item[1], item[0])):
        total, count, index = heapq.heappop(loads)
        shards[index].append(episode)
        heapq.heappush(loads, (total + frames, count + 1, index))
    return [sorted(shard) for shard in shards]


class LIBEROSharedGPUBuilder(_shared.SingleGPUBuilder):
    def __init__(self, env=None, *, src=SRC, log_dir=None):
        initial = dict(os.environ if env is None else env)
        self.curves_only = initial.get("CURVES_ONLY", "false") == "true"
        super().__init__(initial, src=src, log_dir=log_dir)
        import pandas as pd

        self.dataset = Path(self.env["RAW_DATASET_DIR"])
        parts = sorted((self.dataset / "meta/episodes").rglob("file-*.parquet"))
        rows = pd.concat([pd.read_parquet(path, columns=["episode_index", "length"])
                          for path in parts], ignore_index=True)
        self.episodes = [(int(row.episode_index), int(row.length)) for row in rows.itertuples()]
        if len({ep for ep, _ in self.episodes}) != len(self.episodes):
            raise ValueError("Dataset episode IDs are duplicated.")
        if self.env.get('DROID_SELECTION_PATH'):
            selection = json.loads(Path(self.env['DROID_SELECTION_PATH']).read_text())
            allowed = set(selection['episode_ids'])
            if len(allowed) != len(selection['episode_ids']) or not allowed.issubset(dict(self.episodes)):
                raise ValueError('DROID group has duplicate or missing episodes')
            self.episodes = [(ep, n) for ep, n in self.episodes if ep in allowed]
            import runpy
            project = Path(self.env['PROJECT_ROOT'])
            groups = runpy.run_path(str(project / 'lerobot/skillVLA/droid/configs/src/dataset_groups.py'))
            groups['write_selection'](Path(self.env['SKILLSET_DIR']) / 'dataset_selection.json', selection)
        expected = int(self.env["LIBERO_EXPECTED_EPISODES"])
        if len(self.episodes) != expected or expected < 1:
            raise ValueError("Episode count changed since submission.")
        self.env["EXPECTED_EPISODES"] = str(expected)
        self.env["LIBERO_BUILD_DIR"] = str(self.cwd)
        cpus = int(self.env["SKILLSET_CPUS_PER_TASK"])
        if self.env.get("SLURM_CPUS_PER_TASK", "").isdigit():
            cpus = min(cpus, int(self.env["SLURM_CPUS_PER_TASK"]))
        if hasattr(os, "sched_getaffinity"):
            cpus = min(cpus, len(os.sched_getaffinity(0)))
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
            self.env[key] = str(max(1, cpus))

    def _pool(self, *, phase):
        if phase not in {"curves", "cached_segment", "segment"}:
            raise ValueError(f"Unknown SBD phase: {phase}")
        output = Path(self.env["SKILLSET_DIR"])
        cached = ((output / "curves").glob("ep*") if phase == "curves"
                  else (output / "skills").rglob("*.done"))
        finished = set()
        for path in cached:
            match = re.fullmatch(r"ep(\d+)(?:\.npz|\.too_short\.json|_task\d+\.done)", path.name)
            if match:
                finished.add(int(match[1]))
        # Keep cached episodes assigned: each worker still validates them on
        # resume. Zero remaining weight balances only the unfinished work.
        shards = balance_episodes([(ep, 0 if ep in finished else frames)
                                   for ep, frames in self.episodes], self.workers)
        folder = self.log_dir / "episode_shards"
        folder.mkdir(exist_ok=True)
        cpus = int(self.env["SKILLSET_CPUS_PER_TASK"])
        if self.env.get("SLURM_CPUS_PER_TASK", "").isdigit():
            cpus = min(cpus, int(self.env["SLURM_CPUS_PER_TASK"]))
        if hasattr(os, "sched_getaffinity"):
            cpus = min(cpus, len(os.sched_getaffinity(0)))
        threads = str(max(1, cpus // len(shards)))
        print(f"[SBD {phase}] {len(self.episodes)} episodes; {len(shards)} workers, "
              f"{threads} CPU thread(s)/worker, {len(finished)} cached", flush=True)

        def work(item):
            index, ids = item
            path = folder / f"{phase}_{index:03d}.json"
            path.write_text(json.dumps({"episode_ids": ids}) + "\n")
            env = dict(self.env, LIBERO_EPISODE_SHARD_PATH=str(path),
                       CURVES_ONLY=str(phase == "curves").lower(),
                       USE_CACHED_CURVES=str(phase == "cached_segment").lower())
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
                env[key] = threads
            code, _ = self._command(["bash", str(self.src / "build_skillset.sbatch")],
                                    f"{phase}_{index:03d}", env=env)
            if code:
                self.stop(f"SBD {phase} worker {index} failed; see {self.log_dir}")
                raise RuntimeError(self.stop_reason)

        with ThreadPoolExecutor(max_workers=len(shards)) as executor:
            list(executor.map(work, enumerate(shards)))

    def run(self):
        print(f"LIBERO one-GPU SBD build: {len(self.episodes)} episodes; logs={self.log_dir}", flush=True)
        if self.curves_only:
            self._pool(phase="curves")
            return
        if self.env["SKILLSET_BOUNDARY_THRESHOLD_MODE"] == "global_mean":
            self._pool(phase="curves")
            if not self.env.get("SKILLSET_GLOBAL_THRESHOLD_SOURCE"):
                self._stage("compute_global_boundary_threshold.sbatch")
            self._pool(phase="cached_segment")
        else:
            self._pool(phase="segment")
        self._stage("mark_skillset_complete.sbatch")
        print(f"LIBERO SBD build complete → {self.env['SKILLSET_DIR']}", flush=True)


def main():
    builder = LIBEROSharedGPUBuilder()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: builder.stop())
    try:
        builder.run()
    finally:
        builder.stop()


if __name__ == "__main__":
    main()
