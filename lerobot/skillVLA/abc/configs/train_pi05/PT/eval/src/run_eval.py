#!/usr/bin/env python3
"""Closed-loop ABC Sim evaluation runner with reusable policy adapters."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
from pathlib import Path
import re
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np

from eval_config import DEFAULT_CONFIG_PATH, EvalConfig, load_eval_config
from report import build_result, finish_video, panel_name, render_report, video_paths


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_dataset_tasks(config: EvalConfig) -> list[str]:
    if config.tasks != ("all",):
        return list(config.tasks)
    marker = config.dataset_dir / "native_dataset.json"
    if not marker.is_file():
        raise FileNotFoundError(
            f"tasks: all needs the ABC native manifest: {marker}; build {config.dataset_name} first"
        )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("format") != "sbd_abc_sim_native/v1":
        raise ValueError(f"unexpected ABC native manifest format in {marker}")
    tasks = list((payload.get("task_sha256") or {}).keys())
    if not tasks:
        raise ValueError(f"no source tasks recorded in {marker}")
    return tasks


class HoldPolicy:
    """Return the observed absolute 14D state as a fixed-length action chunk."""

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def reset(self) -> None:
        pass

    def describe(self) -> dict[str, Any]:
        return {
            "type": "hold",
            "execution_chunk_size": self.chunk_size,
            "action_space": "absolute_joint_position_14d",
        }

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape == (14,):
            return np.repeat(state[None, :], self.chunk_size, axis=0)
        if state.ndim == 2 and state.shape[1] == 14:
            return np.repeat(state[:, None, :], self.chunk_size, axis=1)
        raise ValueError(f"expected state shape (14,) or (B,14), got {state.shape}")


def make_policy(config: EvalConfig):
    if config.policy_type == "hold":
        return HoldPolicy(config.action_chunk_size)
    if config.policy_type == "pi05":
        from pi05_adapter import Pi05PolicyAdapter

        return Pi05PolicyAdapter(config)
    if config.policy_type == "skillvla":
        from skillvla_adapter import SkillVLAPolicyAdapter

        return SkillVLAPolicyAdapter(config)
    raise ValueError(f"unsupported policy type: {config.policy_type}")


def with_policy_prompt(observation: dict[str, Any], prefix: str) -> dict[str, Any]:
    result = dict(observation)
    prompt = observation.get("prompt", "")
    if isinstance(prompt, list):
        result["prompt"] = [prefix + str(item) for item in prompt]
    else:
        result["prompt"] = prefix + str(prompt)
    return result


def numpy_image(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.uint8)


def video_frame(
    images: dict[str, Any], cameras: tuple[str, ...], world: int | None = None
) -> np.ndarray:
    frames = []
    for camera in cameras:
        image = numpy_image(images[camera])
        if world is not None:
            image = image[world]
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"camera {camera} expected CHW image, got {image.shape}")
        frames.append(np.ascontiguousarray(np.moveaxis(image, 0, -1)))
    return np.concatenate(frames, axis=1)


def rollout_over(task_eval: dict[str, Any]) -> bool:
    if "ever_failed" in task_eval:
        return bool(task_eval["ever_failed"])
    return bool(task_eval.get("ever_success", task_eval.get("success", False)))


def scalar_eval(result: Any) -> dict[str, Any]:
    if result is None:
        return {"reward": 0.0, "success": False, "ever_success": False}
    info = result.to_info(squeeze=True)
    info.setdefault("ever_success", bool(info.get("success", False)))
    return jsonable(info)


def batched_eval(result: Any, index: int, per_world_keys: set[str]) -> dict[str, Any]:
    if result is None:
        return {"reward": 0.0, "success": False, "ever_success": False}
    info: dict[str, Any] = {
        "reward": float(result.reward[index]),
        "success": bool(result.success[index]),
    }
    for key, value in result.metrics.items():
        if key in per_world_keys:
            value = value[index]
        info[key] = jsonable(value)
    info.setdefault("ever_success", info["success"])
    return info


def task_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "task"


def _video_writer(path: Path, config: EvalConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=config.video_fps, macro_block_size=1)


def commit_world(config, panel, task, task_dir, world):
    """Persist the outcome before optional CPU video processing can fail."""
    path = task_dir / f"world_{world['world_index']:03d}.json"
    if config.save_video:
        raw, target = video_paths(getattr(config, "artifact_dir", config.output_dir), panel, task, world["world_index"])
        world.update(raw_video_path=str(raw), video_annotation_pending=True)
        write_json(path, world)
        finish_video(raw, target, world, config.video_fps)
        world.pop("video_annotation_pending", None)
    write_json(path, world)


def stack_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state": np.stack([obs["state"] for obs in observations]),
        "prompt": [obs["prompt"] for obs in observations],
        "images": {
            camera: np.stack([obs["images"][camera] for obs in observations])
            for camera in observations[0]["images"]
        },
    }


def run_sequential_task(
    config: EvalConfig,
    requested_task: str,
    spec: Any,
    policy: Any,
    task_dir: Path,
    on_progress=None,
) -> dict[str, Any]:
    """Independent MuJoCo models with one shared, batched policy.

    Physics steps run per environment on CPU. Only inference is batched on the
    GPU; scene topology, object count and mesh variants may differ freely.
    """
    import abc_sim

    task_dir.mkdir(parents=True, exist_ok=True)
    panel = panel_name(config.policy_type, config.policy_checkpoint)
    worlds = {}
    for index in range(config.num_worlds):
        path = task_dir / f"world_{index:03d}.json"
        if path.exists():
            world = json.loads(path.read_text())
            if world["seed"] != config.seed + index or world["world_index"] != index:
                raise ValueError(f"cached episode identity mismatch: {path}")
            if world.get("video_annotation_pending"):
                commit_world(config, panel, spec.name, task_dir, world)
            worlds[index] = world
    pending = [index for index in range(config.num_worlds) if index not in worlds]
    print(f"  cached episodes={len(worlds)}/{config.num_worlds}, remaining={len(pending)}", flush=True)
    options = None if config.randomization is None else {"randomization": config.randomization}
    for start in range(0, len(pending), config.parallel_worlds):
        indices = pending[start : start + config.parallel_worlds]
        environments, writers, observations, evaluations = [], [], [], []
        overlay_writers, overlay_paths = [], []
        finished = set()
        steps = [0] * len(indices)
        max_rewards = []
        started = time.perf_counter()
        policy.reset()
        try:
            for index in indices:
                env = abc_sim.make_env(
                    task=spec.name, render_cameras=True, camera_backend="mujoco",
                    camera_height=config.camera_height, camera_width=config.camera_width,
                    **(policy.world_env_kwargs(spec.name, index) if hasattr(policy, "world_env_kwargs") else {}),
                )
                environments.append(env)
                if hasattr(policy, "reset_world"):
                    observation = policy.reset_world(env, index, len(environments) - 1)
                else:
                    observation, _ = env.reset(seed=config.seed + index, options=options, randomize=True)
                observations.append(with_policy_prompt(observation, config.prompt_prefix))
                evaluation = scalar_eval(env.evaluate_task())
                evaluations.append(evaluation)
                max_rewards.append(float(evaluation["reward"]))
                raw, _ = video_paths(config.artifact_dir, panel, spec.name, index)
                writer = _video_writer(raw, config) if config.save_video else None
                writers.append(writer)
                overlay_path = (raw.parent.parent.with_name("videos_camera_overlay") / raw.parent.name / raw.name
                                if writer is not None and hasattr(policy, "video_frame") else None)
                overlay_paths.append(overlay_path)
                overlay_writer = _video_writer(overlay_path, config) if overlay_path is not None else None
                overlay_writers.append(overlay_writer)
                if writer is not None:
                    frame = video_frame(observation["images"], config.camera_names)
                    writer.append_data(frame)
                    if overlay_writer is not None:
                        overlay_writer.append_data(frame)

            while len(finished) < len(indices):
                active = [i for i in range(len(indices)) if i not in finished]
                observation_batch = stack_observations([observations[i] for i in active])
                if hasattr(policy, "infer_for_worlds"):
                    actions = policy.infer_for_worlds(observation_batch, active, [environments[i] for i in active])
                else:
                    actions = policy.infer(observation_batch)
                if actions.ndim != 3 or actions.shape[0] != len(active) or actions.shape[2] != 14 or actions.shape[1] == 0:
                    raise ValueError(f"invalid batched actions: {actions.shape}")
                for batch_index, local in enumerate(active):
                    env = environments[local]
                    for action in actions[batch_index]:
                        env.step(np.asarray(action, dtype=np.float32), render_obs=False)
                        steps[local] += 1
                        evaluation = scalar_eval(env.evaluate_task())
                        evaluations[local] = evaluation
                        max_rewards[local] = max(max_rewards[local], float(evaluation["reward"]))
                        step_observation = None
                        skill_ended = False
                        if hasattr(policy, "observe_step"):
                            step_observation = with_policy_prompt(env.get_obs(), config.prompt_prefix)
                            skill_ended = policy.observe_step(local, step_observation)
                        if writers[local] is not None and steps[local] % config.video_every_n_steps == 0:
                            images = (step_observation or env.get_obs())["images"]
                            writers[local].append_data(video_frame(images, config.camera_names))
                            if overlay_writers[local] is not None:
                                overlay_writers[local].append_data(policy.video_frame(local, env, images, config.camera_names))
                        done = (steps[local] >= config.max_steps or (config.stop_on_success and rollout_over(evaluation))
                                or (hasattr(policy, "world_done") and policy.world_done(local)))
                        if done:
                            index = indices[local]
                            if writers[local] is not None:
                                writers[local].close()
                                writers[local] = None
                            if overlay_writers[local] is not None:
                                overlay_writers[local].close()
                                overlay_writers[local] = None
                            world = {
                                "world_index": index, "seed": config.seed + index,
                                "prompt": observations[local]["prompt"],
                                "success": bool(evaluation.get("ever_success", evaluation["success"])),
                                "final_success": bool(evaluation["success"]),
                                "reward": float(evaluation["reward"]), "max_reward": max_rewards[local],
                                "steps": steps[local], "wall_s": time.perf_counter() - started,
                                "randomization": env._last_randomization,
                                "final_task_eval": evaluation,
                                "video_path": None,
                                "video_every_n_steps": config.video_every_n_steps,
                            }
                            if overlay_paths[local] is not None:
                                world["camera_overlay_path"] = str(overlay_paths[local])
                            if hasattr(policy, "finish_world"):
                                world.update(policy.finish_world(local, task_dir / f"world_{index:03d}_trace"))
                            commit_world(config, panel, spec.name, task_dir, world)
                            worlds[index] = world
                            finished.add(local)
                            partial = task_summary(requested_task, spec, "mujoco", sorted(worlds.values(), key=lambda w: w["world_index"]))
                            write_json(task_dir / "partial_summary.json", partial)
                            if on_progress is not None:
                                on_progress(partial)
                            print(f"task={spec.name} world={index:03d} success={world['success']} steps={steps[local]} completed={len(worlds)}/{config.num_worlds}", flush=True)
                            break
                        if skill_ended:
                            break  # Discard the old skill's remaining actions immediately.
                    if local not in finished:
                        observations[local] = step_observation or with_policy_prompt(env.get_obs(), config.prompt_prefix)
        finally:
            for writer in [*writers, *overlay_writers]:
                if writer is not None:
                    writer.close()
            for env in environments:
                env.close()
    return task_summary(requested_task, spec, "mujoco", [worlds[i] for i in range(config.num_worlds)])


def run_batched_task(
    config: EvalConfig,
    requested_task: str,
    spec: Any,
    policy: Any,
    task_dir: Path,
    on_progress=None,
) -> dict[str, Any]:
    import abc_sim

    panel = panel_name(config.policy_type, config.policy_checkpoint)
    worlds = []
    options = None if config.randomization is None else {"randomization": config.randomization}
    batch_size = config.parallel_worlds
    for batch_start in range(0, config.num_worlds, batch_size):
        policy.reset()
        started = time.perf_counter()
        env = abc_sim.make_batched_env(
            spec.name,
            num_worlds=batch_size,
            camera_height=config.camera_height,
            camera_width=config.camera_width,
        )
        writers = []
        try:
            seeds = [config.seed + index for index in range(batch_start, batch_start + batch_size)]
            observation = with_policy_prompt(env.reset(seeds, options), config.prompt_prefix)
            if config.save_video:
                writers = [
                    _video_writer(video_paths(config.output_dir, panel, spec.name, index)[0], config)
                    for index in range(batch_start, batch_start + batch_size)
                ]
                for local_index, writer in enumerate(writers):
                    writer.append_data(
                        video_frame(observation["images"], config.camera_names, local_index)
                    )
            result = env.evaluate()
            keys = set(env.per_world_metric_keys)
            final_evals = [batched_eval(result, index, keys) for index in range(batch_size)]
            max_rewards = np.asarray(
                [float(item["reward"]) for item in final_evals], dtype=np.float32
            )
            active = np.ones(batch_size, dtype=bool)
            steps = np.zeros(batch_size, dtype=np.int64)

            while int(steps.max(initial=0)) < config.max_steps and active.any():
                actions = policy.infer(observation)
                for action_index in range(actions.shape[1]):
                    if not active.any() or int(steps.max(initial=0)) >= config.max_steps:
                        break
                    selected = np.asarray(actions[:, action_index], dtype=np.float32)
                    selected[~active] = np.asarray(observation["state"], dtype=np.float32)[~active]
                    was_active = active.copy()
                    result = env.step(selected)
                    steps[was_active] += 1
                    final_evals = [batched_eval(result, index, keys) for index in range(batch_size)]
                    max_rewards = np.maximum(
                        max_rewards,
                        np.asarray([float(item["reward"]) for item in final_evals]),
                    )
                    if config.stop_on_success:
                        for index in np.flatnonzero(active):
                            if rollout_over(final_evals[index]):
                                active[index] = False
                    if writers and int(steps.max()) % config.video_every_n_steps == 0:
                        images = env.render_cameras()
                        for local_index, writer in enumerate(writers):
                            writer.append_data(video_frame(images, config.camera_names, local_index))
                if active.any():
                    observation = with_policy_prompt(env.get_obs(), config.prompt_prefix)

            for writer in writers:
                writer.close()
            writers = []
            batch_wall_s = time.perf_counter() - started
            for local_index, world_index in enumerate(
                range(batch_start, batch_start + batch_size)
            ):
                final_eval = final_evals[local_index]
                raw, video_path = video_paths(config.output_dir, panel, spec.name, world_index)
                world = {
                    "world_index": world_index,
                    "seed": env.seeds[local_index],
                    "prompt": observation["prompt"][local_index],
                    "success": bool(final_eval.get("ever_success", final_eval["success"])),
                    "final_success": bool(final_eval["success"]),
                    "reward": float(final_eval["reward"]),
                    "max_reward": float(max_rewards[local_index]),
                    "steps": int(steps[local_index]),
                    "wall_s": batch_wall_s,
                    "randomization": env.randomization[local_index],
                    "final_task_eval": final_eval,
                    "video_path": None,
                }
                commit_world(config, panel, spec.name, task_dir, world)
                worlds.append(world)
                if on_progress is not None:
                    on_progress(task_summary(requested_task, spec, "mjwarp", worlds))
                print(
                    f"task={spec.name} world={world_index:03d} backend=mjwarp "
                    f"success={world['success']} reward={world['reward']:.3f} "
                    f"steps={world['steps']}",
                    flush=True,
                )
        finally:
            for writer in writers:
                writer.close()
            env.close()
    return task_summary(requested_task, spec, "mjwarp", worlds)


def task_summary(
    requested_task: str, spec: Any, backend: str, worlds: list[dict[str, Any]]
) -> dict[str, Any]:
    successes = [bool(world["success"]) for world in worlds]
    return {
        "requested_task": requested_task,
        "task": spec.name,
        "prompt": spec.prompt,
        "evaluator": spec.evaluator_name,
        "backend": backend,
        "num_worlds": len(worlds),
        "successes": int(sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else None,
        "mean_reward": float(np.mean([world["reward"] for world in worlds])) if worlds else None,
        "worlds": worlds,
    }


def run(config: EvalConfig) -> dict[str, Any]:
    if config.policy_type != "skillvla":
        return _run(config)
    import fcntl
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    with (config.artifact_dir / "evaluation.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"This evaluation output is already running: {config.output_dir}") from error
        return _run(config)


def _run(config: EvalConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = config.artifact_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = metrics_dir / "resolved_config.json"
    if not resolved_path.exists() and (config.output_dir / "resolved_config.json").exists():
        raise ValueError("Legacy ABC result layout: organize the idle run with src/report.py before resuming")
    if resolved_path.exists():
        previous = json.loads(resolved_path.read_text())
        current = config.json_dict()
        # Worker reductions and scheduler/profile paths do not change episodes.
        ignored = {"config_path", "parallel_worlds", "output_dir"}
        contract = lambda value: {k: v for k, v in value.items() if k not in ignored and not k.startswith("slurm_")
                                  and not (k == "skillvla" and v is None)}
        if contract(previous) != contract(current):
            raise ValueError("Output contains a different evaluation contract; choose a new output.run_name")
    write_json(resolved_path, config.json_dict())
    import abc_sim

    policy = make_policy(config)

    requested_tasks = resolve_dataset_tasks(config)
    resolved: list[tuple[str, Any]] = []
    seen = set()
    for requested in requested_tasks:
        spec = abc_sim.get_task_spec(requested)
        if spec.name not in seen:
            resolved.append((requested, spec))
            seen.add(spec.name)

    summaries = []
    started = time.perf_counter()
    print(
        f"ABC eval: policy={config.policy_type} backend={config.backend} "
        f"tasks={len(resolved)} worlds/task={config.num_worlds} max_steps={config.max_steps}",
        flush=True,
    )
    def update_progress(task):
        result = build_result(policy.describe(), config.dataset_name, [*summaries, task],
                              expected=config.num_worlds * len(resolved), wall_s=time.perf_counter() - started)
        write_json(metrics_dir / "partial_summary.json", result)
        render_report(config.output_dir, result)

    for task_index, (requested, spec) in enumerate(resolved, start=1):
        task_dir = metrics_dir / "tasks" / task_slug(spec.name)
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{task_index}/{len(resolved)}] {requested} -> {spec.name}", flush=True)
        if config.backend == "mujoco":
            summary = run_sequential_task(config, requested, spec, policy, task_dir, update_progress)
        else:
            try:
                summary = run_batched_task(config, requested, spec, policy, task_dir, update_progress)
            except (NotImplementedError, ValueError) as exc:
                if config.backend != "auto" or (
                    isinstance(exc, ValueError) and "worlds in a batch must compile to the same model" not in str(exc)
                ):
                    raise
                print(f"  MJWarp unsupported ({exc}); falling back to independent MuJoCo scenes", flush=True)
                summary = run_sequential_task(config, requested, spec, policy, task_dir, update_progress)
        write_json(task_dir / "summary.json", summary)
        summaries.append(summary)

    result = build_result(policy.describe(), config.dataset_name, summaries,
                          expected=config.num_worlds * len(resolved), wall_s=time.perf_counter() - started)
    write_json(metrics_dir / "summary.json", result)
    render_report(config.output_dir, result)
    successes, total_worlds = result["successes"], result["num_worlds"]
    print(
        f"ABC eval complete: success={successes}/{total_worlds} "
        f"({100.0 * result['success_rate']:.1f}%) output={config.output_dir}",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tasks", help="comma-separated task aliases, or all")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-worlds", type=int)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--run-name", help="separate output folder for an integration smoke test")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_eval_config(args.config)
    if args.tasks:
        config = replace(
            config,
            tasks=tuple(item.strip() for item in args.tasks.split(",") if item.strip()),
        )
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        config = replace(config, max_steps=args.max_steps)
    if args.num_worlds is not None:
        if args.num_worlds <= 0 or (config.backend in {"mjwarp", "auto"} and args.num_worlds % config.parallel_worlds):
            raise ValueError("--num-worlds must be positive and divisible by parallel_worlds")
        config = replace(config, num_worlds=args.num_worlds)
    if args.run_name:
        if args.run_name in {".", ".."} or Path(args.run_name).name != args.run_name:
            raise ValueError("--run-name must be one folder name")
        config = replace(config, output_dir=config.output_dir.parent / args.run_name)
    if args.no_video:
        config = replace(config, save_video=False)
    if args.dry_run:
        print(json.dumps(config.json_dict(), indent=2))
        print("resolved_tasks=", json.dumps(resolve_dataset_tasks(config)))
        return
    from memory_guard import supervise_evaluation

    code = supervise_evaluation(config)
    if code is not None:
        raise SystemExit(code)
    run(config)


if __name__ == "__main__":
    main()
