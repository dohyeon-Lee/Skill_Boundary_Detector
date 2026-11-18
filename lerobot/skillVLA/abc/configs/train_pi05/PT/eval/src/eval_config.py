#!/usr/bin/env python3
"""Resolve ABC Sim eval YAML together with the selected server config."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import sys
from typing import Any


CONFIGS_DIR = Path(__file__).resolve().parents[4]
GLOBAL_SRC = CONFIGS_DIR / "src"
if str(GLOBAL_SRC) not in sys.path:
    sys.path.insert(0, str(GLOBAL_SRC))

from global_config_loader import load_global_config, read_yaml  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "abc_sim_eval_config.yaml"


def _global_path(config_path: Path) -> Path:
    for directory in (config_path.resolve().parent, *config_path.resolve().parents):
        candidate = directory / "global_config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"global_config.yaml not found above {config_path}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(f"{name} must be a string or list of strings")
    result = tuple(str(item).strip() for item in values if str(item).strip())
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _server_list(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    values = value if isinstance(value, list) else [value]
    return tuple(str(item).strip() for item in values if str(item).strip())


@dataclass(frozen=True)
class EvalConfig:
    config_path: Path
    project_root: Path
    dataset_name: str
    dataset_dir: Path
    tasks: tuple[str, ...]
    policy_type: str
    policy_checkpoint: Path | None
    policy_tokenizer_path: Path | None
    policy_device: str
    policy_dtype: str
    policy_model_chunk_size: int
    policy_num_inference_steps: int
    action_chunk_size: int
    prompt_prefix: str
    backend: str
    num_worlds: int
    parallel_worlds: int
    seed: int
    max_steps: int
    stop_on_success: bool
    randomization: dict[str, Any] | None
    camera_names: tuple[str, ...]
    camera_height: int
    camera_width: int
    save_video: bool
    video_fps: int
    video_every_n_steps: int
    output_dir: Path
    slurm_partition: tuple[str, ...]
    slurm_qos: str
    slurm_nodelist: tuple[str, ...]
    slurm_exclude_nodes: tuple[str, ...]
    slurm_gres: str
    slurm_cpus: int
    slurm_memory: str
    slurm_time: str

    def json_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _project_path(project_root: Path, value: Any, name: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required")
    path = Path(text).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def load_eval_config(path: Path | str = DEFAULT_CONFIG_PATH) -> EvalConfig:
    path = Path(path).resolve()
    local = read_yaml(path)
    global_config = load_global_config(_global_path(path))

    project_root = Path(str(global_config["project_root"])).resolve()
    dataset_name = str(local.get("dataset", "abc_sim_all")).strip()
    if not dataset_name:
        raise ValueError("dataset cannot be empty")
    dataset_root = Path(str(global_config.get("abc_sim_dataset_root", "dataset_ABC")))
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    dataset_dir = dataset_root / dataset_name

    tasks = _string_list(local.get("tasks", ["sim_spell_abc"]), "tasks")
    if "all" in tasks and tasks != ("all",):
        raise ValueError("tasks: all cannot be combined with explicit tasks")

    policy = _mapping(local.get("policy"), "policy")
    rollout = _mapping(local.get("rollout"), "rollout")
    camera = _mapping(local.get("camera"), "camera")
    video = _mapping(local.get("video"), "video")
    output = _mapping(local.get("output"), "output")
    slurm = _mapping(local.get("slurm"), "slurm")

    policy_type = str(policy.get("type", "hold")).strip().lower()
    if policy_type not in {"hold", "pi05"}:
        raise ValueError(f"unsupported policy.type {policy_type!r}; available: hold, pi05")
    action_chunk_size = _positive_int(
        policy.get("action_chunk_size", 15), "policy.action_chunk_size"
    )
    model_chunk_size = _positive_int(
        policy.get("model_chunk_size", 30), "policy.model_chunk_size"
    )
    if action_chunk_size > model_chunk_size:
        raise ValueError("policy.action_chunk_size cannot exceed policy.model_chunk_size")
    policy_dtype = str(policy.get("dtype", "bfloat16")).strip().lower()
    if policy_dtype not in {"bfloat16", "float32"}:
        raise ValueError("policy.dtype must be bfloat16 or float32")
    checkpoint = tokenizer = None
    if policy_type == "pi05":
        checkpoint = _project_path(project_root, policy.get("checkpoint"), "policy.checkpoint")
        tokenizer = _project_path(
            project_root, policy.get("tokenizer_path"), "policy.tokenizer_path"
        )
        if not (checkpoint / "config.json").is_file() or not (
            checkpoint / "model.safetensors"
        ).is_file():
            raise FileNotFoundError(f"PI0.5 checkpoint is incomplete: {checkpoint}")
        if not tokenizer.is_dir():
            raise FileNotFoundError(f"PI0.5 tokenizer directory not found: {tokenizer}")
    backend = str(rollout.get("backend", "mjwarp")).strip().lower()
    if backend not in {"mjwarp", "mujoco", "auto"}:
        raise ValueError("rollout.backend must be mjwarp, mujoco, or auto")

    num_worlds = _positive_int(rollout.get("num_worlds", 2), "rollout.num_worlds")
    parallel_worlds = _positive_int(
        rollout.get("parallel_worlds", num_worlds), "rollout.parallel_worlds"
    )
    if backend in {"mjwarp", "auto"} and num_worlds % parallel_worlds:
        raise ValueError("rollout.num_worlds must be divisible by rollout.parallel_worlds")
    if backend == "mujoco" and parallel_worlds != 1:
        raise ValueError("rollout.parallel_worlds must be 1 for the mujoco backend")

    randomization = rollout.get("randomization")
    if randomization is not None and not isinstance(randomization, dict):
        raise ValueError("rollout.randomization must be null or a mapping")

    output_root = Path(str(output.get("root", global_config.get("outputs_root", "outputs_ABC"))))
    if not output_root.is_absolute():
        output_root = project_root / output_root
    run_name = str(output.get("run_name", "hold_smoke")).strip()
    if not run_name or run_name in {".", ".."} or "/" in run_name:
        raise ValueError("output.run_name must be one nonempty folder name")

    partitions = _server_list(global_config.get("train_partition"))
    if not partitions:
        raise ValueError("global train_partition is empty")

    return EvalConfig(
        config_path=path,
        project_root=project_root,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir.resolve(),
        tasks=tasks,
        policy_type=policy_type,
        policy_checkpoint=checkpoint,
        policy_tokenizer_path=tokenizer,
        policy_device=str(policy.get("device", "cuda")).strip(),
        policy_dtype=policy_dtype,
        policy_model_chunk_size=model_chunk_size,
        policy_num_inference_steps=_positive_int(
            policy.get("num_inference_steps", 10), "policy.num_inference_steps"
        ),
        action_chunk_size=action_chunk_size,
        prompt_prefix=str(policy.get("prompt_prefix", "sim ")),
        backend=backend,
        num_worlds=num_worlds,
        parallel_worlds=parallel_worlds,
        seed=int(rollout.get("seed", 0)),
        max_steps=_positive_int(rollout.get("max_steps", 30), "rollout.max_steps"),
        stop_on_success=bool(rollout.get("stop_on_success", True)),
        randomization=randomization,
        camera_names=_string_list(camera.get("names", ["top", "left", "right"]), "camera.names"),
        camera_height=_positive_int(camera.get("height", 168), "camera.height"),
        camera_width=_positive_int(camera.get("width", 224), "camera.width"),
        save_video=bool(video.get("enable", True)),
        video_fps=_positive_int(video.get("fps", 30), "video.fps"),
        video_every_n_steps=_positive_int(
            video.get("every_n_steps", 1), "video.every_n_steps"
        ),
        output_dir=(output_root / run_name).resolve(),
        slurm_partition=partitions,
        slurm_qos=str(global_config.get("train_qos", "")).strip(),
        slurm_nodelist=_server_list(global_config.get("train_nodelist")),
        slurm_exclude_nodes=_server_list(global_config.get("train_exclude_nodes")),
        slurm_gres=str(slurm.get("gres", "gpu:1")),
        slurm_cpus=_positive_int(slurm.get("cpus", 8), "slurm.cpus"),
        slurm_memory=str(slurm.get("memory", "32G")),
        slurm_time=str(slurm.get("time", "00:15:00")),
    )


def shell_exports(config: EvalConfig) -> str:
    values = {
        "ABC_EVAL_PROJECT_ROOT": config.project_root,
        "ABC_EVAL_CONFIG": config.config_path,
        "ABC_EVAL_OUTPUT_DIR": config.output_dir,
        "ABC_EVAL_TASKS_JSON": json.dumps(config.tasks),
        "ABC_EVAL_PARTITION": ",".join(config.slurm_partition),
        "ABC_EVAL_QOS": config.slurm_qos,
        "ABC_EVAL_NODELIST": ",".join(config.slurm_nodelist),
        "ABC_EVAL_EXCLUDE_NODES": ",".join(config.slurm_exclude_nodes),
        "ABC_EVAL_GRES": config.slurm_gres,
        "ABC_EVAL_CPUS": config.slurm_cpus,
        "ABC_EVAL_MEMORY": config.slurm_memory,
        "ABC_EVAL_TIME": config.slurm_time,
    }
    return "\n".join(
        f"export {name}={shlex.quote(str(value))}" for name, value in values.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    config = load_eval_config(args.config)
    print(shell_exports(config) if args.shell else json.dumps(config.json_dict(), indent=2))


if __name__ == "__main__":
    main()
