#!/usr/bin/env python3
"""Closed-loop ABC Sim evaluation runner with reusable policy adapters."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np

from eval_config import DEFAULT_CONFIG_PATH, EvalConfig, load_eval_config


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


def run_sequential_task(
    config: EvalConfig,
    requested_task: str,
    spec: Any,
    policy: Any,
    task_dir: Path,
) -> dict[str, Any]:
    import abc_sim

    worlds = []
    options = None if config.randomization is None else {"randomization": config.randomization}
    for world_index in range(config.num_worlds):
        policy.reset()
        seed = config.seed + world_index
        started = time.perf_counter()
        env = abc_sim.make_env(
            task=spec.name,
            render_cameras=True,
            camera_backend="mujoco",
            camera_height=config.camera_height,
            camera_width=config.camera_width,
        )
        video_path = task_dir / f"world_{world_index:03d}.mp4" if config.save_video else None
        writer = _video_writer(video_path, config) if video_path is not None else None
        try:
            observation, reset_info = env.reset(seed=seed, options=options, randomize=True)
            observation = with_policy_prompt(observation, config.prompt_prefix)
            if writer is not None:
                writer.append_data(video_frame(observation["images"], config.camera_names))
            final_eval = scalar_eval(env.evaluate_task())
            max_reward = float(final_eval["reward"])
            steps = 0
            while steps < config.max_steps:
                actions = policy.infer(observation)
                for action in actions:
                    if steps >= config.max_steps:
                        break
                    env.step(np.asarray(action, dtype=np.float32), render_obs=False)
                    steps += 1
                    final_eval = scalar_eval(env.evaluate_task())
                    max_reward = max(max_reward, float(final_eval["reward"]))
                    if writer is not None and steps % config.video_every_n_steps == 0:
                        writer.append_data(video_frame(env.get_obs()["images"], config.camera_names))
                    if config.stop_on_success and rollout_over(final_eval):
                        break
                if config.stop_on_success and rollout_over(final_eval):
                    break
                observation = with_policy_prompt(env.get_obs(), config.prompt_prefix)
            world = {
                "world_index": world_index,
                "seed": seed,
                "prompt": observation["prompt"],
                "success": bool(final_eval.get("ever_success", final_eval["success"])),
                "final_success": bool(final_eval["success"]),
                "reward": float(final_eval["reward"]),
                "max_reward": max_reward,
                "steps": steps,
                "wall_s": time.perf_counter() - started,
                "randomization": reset_info.get("randomization"),
                "final_task_eval": final_eval,
                "video_path": video_path,
            }
            worlds.append(world)
            print(
                f"task={spec.name} world={world_index:03d} backend=mujoco "
                f"success={world['success']} reward={world['reward']:.3f} steps={steps}",
                flush=True,
            )
        finally:
            if writer is not None:
                writer.close()
            env.close()
    return task_summary(requested_task, spec, "mujoco", worlds)


def run_batched_task(
    config: EvalConfig,
    requested_task: str,
    spec: Any,
    policy: Any,
    task_dir: Path,
) -> dict[str, Any]:
    import abc_sim

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
                    _video_writer(task_dir / f"world_{index:03d}.mp4", config)
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

            batch_wall_s = time.perf_counter() - started
            for local_index, world_index in enumerate(
                range(batch_start, batch_start + batch_size)
            ):
                final_eval = final_evals[local_index]
                video_path = task_dir / f"world_{world_index:03d}.mp4" if config.save_video else None
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
                    "video_path": video_path,
                }
                worlds.append(world)
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
    import abc_sim

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "resolved_config.json", config.json_dict())
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
    for task_index, (requested, spec) in enumerate(resolved, start=1):
        task_dir = config.output_dir / task_slug(spec.name)
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{task_index}/{len(resolved)}] {requested} -> {spec.name}", flush=True)
        if config.backend == "mujoco":
            summary = run_sequential_task(config, requested, spec, policy, task_dir)
        else:
            try:
                summary = run_batched_task(config, requested, spec, policy, task_dir)
            except NotImplementedError as exc:
                if config.backend != "auto":
                    raise
                print(f"  MJWarp unsupported ({exc}); falling back to MuJoCo", flush=True)
                summary = run_sequential_task(config, requested, spec, policy, task_dir)
        write_json(task_dir / "summary.json", summary)
        summaries.append(summary)

    successes = sum(task["successes"] for task in summaries)
    total_worlds = sum(task["num_worlds"] for task in summaries)
    result = {
        "format": "sbd_abc_sim_eval/v1",
        "policy": policy.describe(),
        "dataset": config.dataset_name,
        "num_tasks": len(summaries),
        "num_worlds": total_worlds,
        "successes": successes,
        "success_rate": successes / total_worlds if total_worlds else None,
        "wall_s": time.perf_counter() - started,
        "tasks": summaries,
    }
    write_json(config.output_dir / "summary.json", result)
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
        if args.num_worlds <= 0 or args.num_worlds % config.parallel_worlds:
            raise ValueError("--num-worlds must be positive and divisible by parallel_worlds")
        config = replace(config, num_worlds=args.num_worlds)
    if args.no_video:
        config = replace(config, save_video=False)
    if args.dry_run:
        print(json.dumps(config.json_dict(), indent=2))
        print("resolved_tasks=", json.dumps(resolve_dataset_tasks(config)))
        return
    run(config)


if __name__ == "__main__":
    main()
