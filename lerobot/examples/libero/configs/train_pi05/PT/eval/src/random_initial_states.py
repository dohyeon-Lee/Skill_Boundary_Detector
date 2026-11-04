#!/usr/bin/env python3
"""Persistent, model-independent initial scene banks for pi05 random_range eval.

Only XY placement samplers for movable workspace objects are scaled. Fixtures,
conditional placements (inside drawers/on other objects), rotation and goals stay
unchanged. Banks contain PRE-settle simulator states; normal eval performs its
usual ten no-op settling steps after restoring them.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys


def range_scale(value=1.0) -> float:
    if isinstance(value, bool):
        raise ValueError("random_range must be a finite number >= 1.0, not a boolean")
    try:
        scale = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("random_range must be a finite number >= 1.0") from error
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("random_range must be a finite number >= 1.0")
    return scale


def range_tag(scale: float) -> str:
    return format(range_scale(scale), ".12g").replace(".", "p").replace("+", "")


def scale_intervals(intervals, scale):
    result = []
    for low, high in intervals:
        center, half = (low + high) / 2, (high - low) / 2 * scale
        if high < low:
            raise ValueError("Inverted placement interval")
        result.append([center - half, center + half])
    return result


def scale_workspace_samplers(domain, scale):
    """Modify samplers, NOT source BDDL or the shared LIBERO installation."""
    changed = {}
    for name, sampler in domain.placement_initializer.samplers.items():
        objects = getattr(sampler, "mujoco_objects", ())
        if not objects or not all(obj.name in domain.objects_dict for obj in objects):
            continue  # Never move fixtures or change conditional/site placement regions.
        if not hasattr(sampler, "x_ranges") or not hasattr(sampler, "y_ranges"):
            continue
        before = {axis: [list(pair) for pair in getattr(sampler, axis)] for axis in ("x_ranges", "y_ranges")}
        after = {axis: scale_intervals(intervals, scale) for axis, intervals in before.items()}
        for axis, intervals in after.items():
            setattr(sampler, axis, intervals)
        changed[name] = {"original": before, "scaled": after}
    if not changed:
        raise ValueError("This task has no supported movable workspace placement samplers")
    return changed


def _fixture_poses(domain):
    result = {}
    for obj in domain.fixtures_dict.values():
        body_id = domain.sim.model.body_name2id(obj.root_body)
        result[obj.root_body] = {
            "pos": domain.sim.model.body_pos[body_id].tolist(),
            "quat": domain.sim.model.body_quat[body_id].tolist(),
        }
    return result


def prepare_bank(*, suite_name, task_ids, episodes, scale, bank_root, seed=1000):
    """Cache a task set atomically; independent of model, worker packing or renderer.

    Requires a private LIBERO_CONFIG_PATH configured by the caller. Generating
    states uses NO cameras, OpenGL or GPU. All panels load this same NPZ through
    lerobot-eval's exact-init interface (no demo GT is involved).
    """
    import numpy as np
    import mujoco
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from robosuite.utils.errors import RandomizationError

    scale = range_scale(scale)
    task_ids = sorted(int(task_id) for task_id in task_ids)
    if episodes < 1 or not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Need positive episodes and unique task IDs")
    suite = benchmark.get_benchmark_dict()[suite_name]()
    tasks = [(task_id, suite.get_task(task_id)) for task_id in task_ids]
    sources = {
        task_id: Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        for task_id, task in tasks
    }
    source_hashes = {str(task_id): hashlib.sha256(path.read_bytes()).hexdigest() for task_id, path in sources.items()}
    metadata = {
        "version": 1, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "suite": suite_name, "task_ids": task_ids, "episodes": episodes, "random_range": scale,
        "seed": seed, "bddl_sha256": source_hashes, "numpy": np.__version__, "mujoco": mujoco.__version__,
        "state_timing": "before ten settling steps", "scope": "movable workspace samplers XY only",
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(bank_root) / f"{suite_name}_range{range_tag(scale)}_{digest}"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "eval_init_states.npz"
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if archive.is_file():
            with np.load(archive, allow_pickle=True) as existing:
                if json.loads(str(existing["metadata"].item())) != metadata:
                    raise ValueError(f"Initial-state bank metadata mismatch: {archive}")
            return archive
        states, scene_files, indices, seeds, model_poses, details = [], [], [], [], [], {}
        numpy_rng, python_rng = np.random.get_state(), random.getstate()
        try:
            for task_id, task in tasks:
                # No cameras and soft reset: retain scaled samplers between episodes.
                np.random.seed(seed)
                random.seed(seed)
                env = ControlEnv(
                    bddl_file_name=str(sources[task_id]), use_camera_obs=False,
                    has_offscreen_renderer=False, has_renderer=False, hard_reset=False,
                )
                try:
                    details[str(task_id)] = {"task_name": task.name, "ranges": scale_workspace_samplers(env.env, scale), "attempts": []}
                    for episode in range(episodes):
                        for attempt in range(20):
                            sample_seed = seed + episode + attempt * 100000
                            env.seed(sample_seed)
                            random.seed(sample_seed)
                            try:
                                # ControlEnv.reset retries forever; call the domain directly so
                                # impossible expanded placements produce a bounded, useful error.
                                env.env.reset()
                            except RandomizationError:
                                continue
                            state = np.asarray(env.get_sim_state()).copy()
                            poses = _fixture_poses(env.env)
                            initial_heights = {
                                name: float(env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body), 2])
                                for name, obj in env.env.objects_dict.items()
                            }
                            for _ in range(10):
                                env.step([0., 0., 0., 0., 0., 0., -1.])
                            fallen = any(
                                float(env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body), 2]) < initial_heights[name] - .10
                                for name, obj in env.env.objects_dict.items()
                            )
                            if not np.isfinite(env.get_sim_state()).all() or fallen or env.check_success():
                                continue
                            break
                        else:
                            raise RuntimeError(
                                f"Task {task_id}, episode {episode}: no valid placement after 20 attempts; "
                                "reduce random_range. No partial bank was published."
                            )
                        states.append(state)
                        scene_files.append(f"{task.name}_demo.hdf5")
                        indices.append(episode)
                        seeds.append(sample_seed)
                        model_poses.append(json.dumps(poses, sort_keys=True))
                        details[str(task_id)]["attempts"].append(attempt + 1)
                        print(f"[random_range={scale:g}] task {task_id}: {episode + 1}/{episodes}", file=sys.stderr, flush=True)
                finally:
                    env.close()
        finally:
            np.random.set_state(numpy_rng)
            random.setstate(python_rng)
        # State dimensions differ across tasks; retain one flattened vector per row.
        packed_states = np.empty(len(states), dtype=object)
        packed_states[:] = states
        temporary = directory / f".states.{os.getpid()}.npz"
        np.savez_compressed(
            temporary, episode_index=np.asarray(indices), init_states=packed_states,
            scene_file=np.asarray(scene_files), sampling_seed=np.asarray(seeds),
            model_body_poses=np.asarray(model_poses), metadata=json.dumps(metadata, sort_keys=True),
        )
        (directory / "manifest.json").write_text(json.dumps({**metadata, "tasks": details}, indent=2), encoding="utf-8")
        os.replace(temporary, archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-ids", required=True, help="JSON list")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    args = parser.parse_args()
    # LIBERO sometimes prints during import/construction; stdout is the path only.
    with contextlib.redirect_stdout(sys.stderr):
        path = prepare_bank(suite_name=args.suite, task_ids=json.loads(args.task_ids),
                            episodes=args.episodes, scale=args.scale, bank_root=args.bank_root)
    print(path, flush=True)


if __name__ == "__main__":
    main()
