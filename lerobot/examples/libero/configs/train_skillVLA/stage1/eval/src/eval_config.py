#!/usr/bin/env python3
"""Resolve the minimal Stage-1 alignment-map config from shared metadata."""

from __future__ import annotations

import argparse
import json
import runpy
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvalConfig:
    config_path: Path
    project_root: Path
    checkpoint: Path
    dataset_dir: Path
    repo_id: str
    dino_model_path: Path
    video_backend: str
    device: str
    task_suite: str
    task_ids: tuple[int, ...]
    dataset_task_ids: tuple[int, ...]
    task_names: tuple[str, ...]
    task_episode_ids: tuple[tuple[int, ...], ...]
    episode_ids: tuple[int, ...]
    episodes_per_task: int
    skills_per_episode: int | None
    frames_per_skill: int
    max_samples: int | None
    top_n: int
    action_maps: bool
    action_probe_time: float
    overlay_alpha: float
    save_all_query_arrays: bool
    output_dir: Path
    slurm: dict[str, object]
    architecture_label: str
    visual_bottleneck_tokens: int


def _find_above(start: Path, filename: str) -> Path:
    for directory in (start.resolve(), *start.resolve().parents):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} above {start}.")


def _load_global(path: Path) -> dict[str, Any]:
    loader = next(
        directory / "src" / "global_config_loader.py"
        for directory in Path(__file__).resolve().parents
        if (directory / "src" / "global_config_loader.py").is_file()
    )
    return runpy.run_path(str(loader))["load_global_config"](path)


def _path(project_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _as_list(value: object) -> list[str]:
    if value in (None, ""):
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _resolve_tasks(
    dataset_dir: Path, specs: object, suite_name: str
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...], tuple[tuple[int, ...], ...]]:
    """Map benchmark IDs to exact dataset episodes through scene provenance."""
    requested = _as_list(specs)
    if not requested:
        return (), (), (), ()
    import numpy as np
    import pyarrow.parquet as pq
    from libero.libero import benchmark

    rows = pq.read_table(dataset_dir / "meta" / "tasks.parquet").to_pylist()
    dataset_by_name = {str(row["task"]): int(row["task_index"]) for row in rows}
    suite = benchmark.get_benchmark_dict()[suite_name]()
    suite_by_id = {index: str(task.language) for index, task in enumerate(suite.tasks)}
    suite_task_names = {index: str(task.name) for index, task in enumerate(suite.tasks)}
    suite_by_name: dict[str, list[int]] = {}
    for task_id, name in suite_by_id.items():
        suite_by_name.setdefault(name, []).append(task_id)
    exact_path = dataset_dir.parents[1] / "eval_init_states.npz"
    if not exact_path.is_file():
        raise FileNotFoundError(
            f"Exact task selection requires stage1_eval provenance: {exact_path}"
        )
    exact = np.load(exact_path, allow_pickle=True)
    episode_by_scene: dict[str, list[int]] = {}
    for episode, scene_file in zip(
        exact["episode_index"], exact["scene_file"], strict=True
    ):
        episode_by_scene.setdefault(str(scene_file), []).append(int(episode))
    benchmark_ids: list[int] = []
    dataset_ids: list[int] = []
    names: list[str] = []
    episode_ids: list[tuple[int, ...]] = []
    for value in requested:
        if value.lstrip("-").isdigit():
            task_id = int(value)
            if task_id not in suite_by_id:
                raise ValueError(
                    f"Unknown {suite_name} task ID {task_id}; valid range is "
                    f"0..{max(suite_by_id)}."
                )
        else:
            matches = suite_by_name.get(value, [])
            if not matches:
                raise ValueError(
                    f"Task language must match {suite_name} exactly, got {value!r}."
                )
            if len(matches) != 1:
                raise ValueError(
                    f"Task language {value!r} maps to benchmark IDs {matches}; "
                    "use the numeric task ID for exact scene matching."
                )
            task_id = matches[0]
        language = suite_by_id[task_id]
        if language not in dataset_by_name:
            raise ValueError(
                f"{suite_name} task {task_id} ({language!r}) is absent from the "
                "selected SkillVLA dataset."
            )
        if task_id not in benchmark_ids:
            scene_file = f"{suite_task_names[task_id]}_demo.hdf5"
            exact_episodes = tuple(sorted(episode_by_scene.get(scene_file, [])))
            if not exact_episodes:
                raise ValueError(
                    f"No exact dataset episodes found for {suite_name} task {task_id}: "
                    f"{scene_file}."
                )
            benchmark_ids.append(task_id)
            dataset_ids.append(dataset_by_name[language])
            names.append(language)
            episode_ids.append(exact_episodes)
    return (
        tuple(benchmark_ids), tuple(dataset_ids), tuple(names), tuple(episode_ids)
    )


def _resolve_checkpoint(
    project_root: Path,
    outputs_root: Path,
    spec: object,
) -> tuple[Path, str]:
    if not isinstance(spec, dict):
        raise ValueError("checkpoint must contain architecture and step.")
    direct = str(spec.get("path", "") or "").strip()
    if direct:
        return _path(project_root, direct), str(spec.get("step", "custom"))
    architecture = str(spec.get("architecture", "")).strip()
    step = str(spec.get("step", "last")).strip()
    if not architecture:
        raise ValueError("checkpoint.architecture cannot be empty.")
    base = outputs_root / "skillVLA_stage1" / "VSA"
    run = str(spec.get("run", "") or "").strip()
    configs = (
        [base / run / "checkpoints" / step / "pretrained_model" / "config.json"]
        if run
        else sorted(base.glob(f"*/checkpoints/{step}/pretrained_model/config.json"))
    )
    matches = []
    for config_path in configs:
        if not config_path.is_file():
            continue
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved.get("architecture_label") == architecture:
            matches.append(config_path.parent)
    if len(matches) != 1:
        found = "\n  ".join(str(path) for path in matches) or "(none)"
        raise ValueError(
            f"Expected one {architecture!r} checkpoint at step {step}, found {len(matches)}:\n  "
            f"{found}\nIf several runs match, add checkpoint.run to the YAML."
        )
    return matches[0], step


def _resolve_dataset(
    project_root: Path,
    dataset_root: Path,
    stage1_common: dict,
    policy_config: dict,
    override: object,
) -> Path:
    direct = str(override or "").strip()
    if direct:
        return _path(project_root, direct)
    skill_space = str(policy_config.get("skill_code_space_id", "")).strip()
    if not skill_space:
        raise ValueError("Checkpoint has no skill_code_space_id for dataset discovery.")
    shared = stage1_common.get("dataset") or {}
    skillvla_root = str(shared.get("skillvla_root", "skillvla_dataset"))
    source = str(shared.get("source", "")).strip()
    preferred = dataset_root / skillvla_root / source / skill_space / "skillvla"
    if source and (preferred / "meta" / "info.json").is_file():
        return preferred.resolve()
    matches = sorted(
        path
        for path in (dataset_root / skillvla_root).glob(f"*/{skill_space}/skillvla")
        if (path / "meta" / "info.json").is_file()
    )
    if len(matches) != 1:
        found = "\n  ".join(str(path) for path in matches) or "(none)"
        raise ValueError(
            f"Expected one dataset for skill_code_space_id={skill_space!r}, found {len(matches)}:\n  "
            f"{found}\nIf several datasets match, add dataset_dir to the YAML."
        )
    return matches[0].resolve()


def _resolve_dino(project_root: Path, policy_config: dict, override: object) -> Path:
    direct = str(override or "").strip()
    if direct:
        return _path(project_root, direct)
    recorded = Path(str(policy_config.get("dino_model_path", ""))).expanduser()
    if "models" in recorded.parts:
        candidate = project_root.joinpath(*recorded.parts[recorded.parts.index("models") :])
        if candidate.is_dir():
            return candidate.resolve()
    try:
        if recorded.is_dir():
            return recorded.resolve()
    except OSError:
        pass
    raise FileNotFoundError(
        f"Cannot remap checkpoint DINO path {recorded}; expected it under project_root/models."
    )


def load_eval_config(path: str | Path) -> EvalConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    global_config = _load_global(_find_above(config_path.parent, "global_config.yaml"))
    stage1_common_path = _find_above(config_path.parent, "stage1_common_config.yaml")
    stage1_common = yaml.safe_load(stage1_common_path.read_text(encoding="utf-8")) or {}
    project_root = Path(str(global_config["project_root"])).expanduser().resolve()
    outputs_root = _path(project_root, global_config.get("outputs_root", "outputs"))
    dataset_root = _path(project_root, global_config.get("dataset_root", "dataset"))

    checkpoint, checkpoint_step = _resolve_checkpoint(
        project_root, outputs_root, raw.get("checkpoint")
    )
    for name, required in (
        ("checkpoint config", checkpoint / "config.json"),
        ("checkpoint weights", checkpoint / "model.safetensors"),
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Missing {name}: {required}")
    policy_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    label = str(policy_config.get("architecture_label", ""))
    # Do not whitelist architecture names here. New architectures can expose
    # the same diagnostics without sharing a naming prefix (for example
    # wristonly_2). The loaded model is capability-checked in _load_policy via
    # wrist_patch_alignment_diagnostics and the optional action-map methods.

    dataset_dir = _resolve_dataset(
        project_root,
        dataset_root,
        stage1_common,
        policy_config,
        raw.get("dataset_dir"),
    )
    for name, required in (
        ("dataset metadata", dataset_dir / "meta" / "info.json"),
        ("dataset statistics", dataset_dir / "meta" / "stats.json"),
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Missing {name}: {required}")
    dataset_info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    repo_id = str(dataset_info.get("repo_id", "")).strip()
    if not repo_id:
        raise ValueError(f"Dataset metadata has no repo_id: {dataset_dir / 'meta/info.json'}")
    dino_model_path = _resolve_dino(
        project_root, policy_config, raw.get("dino_model_path")
    )
    if not (dino_model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing DINO model: {dino_model_path}")

    samples = raw.get("samples") or {}
    source = str((stage1_common.get("dataset") or {}).get("source", ""))
    task_suite = str(samples.get("task_suite", "") or "").strip()
    if not task_suite:
        task_suite = source.split("_full", 1)[0]
    if not task_suite:
        raise ValueError("samples.task_suite is required when it cannot be inferred from the dataset source.")
    task_ids, dataset_task_ids, task_names, task_episode_ids = _resolve_tasks(
        dataset_dir, samples.get("tasks", []), task_suite
    )
    episode_ids = tuple(int(value) for value in samples.get("episodes", []))
    episodes_per_task = int(samples.get("episodes_per_task", 1))
    skills_spec = samples.get("skills_per_episode", "all")
    skills = (
        None
        if str(skills_spec).strip().lower() == "all"
        else int(skills_spec)
    )
    frames = int(samples.get("frames_per_skill", 4))
    top_n = int(samples.get("top_queries", 8))
    action_options = raw.get("action_maps", {})
    if isinstance(action_options, bool):
        action_maps = action_options
        action_probe_time = 0.5
    elif isinstance(action_options, dict):
        action_maps = bool(action_options.get("enabled", True))
        action_probe_time = float(action_options.get("probe_time", 0.5))
    else:
        raise ValueError("action_maps must be a boolean or a mapping.")
    selected_episode_count = len(episode_ids) or max(len(task_ids), 1) * episodes_per_task
    max_samples_spec = samples.get("max_samples")
    max_samples = (
        int(max_samples_spec)
        if max_samples_spec is not None
        else (
            selected_episode_count * skills * frames
            if skills is not None
            else None
        )
    )
    positive_counts = [episodes_per_task, frames, top_n]
    if skills is not None:
        positive_counts.append(skills)
    if max_samples is not None:
        positive_counts.append(max_samples)
    if min(positive_counts) <= 0:
        raise ValueError("Sample counts and top_queries must be positive, or skills_per_episode: all.")
    if not 0.0 <= action_probe_time <= 1.0:
        raise ValueError("action_maps.probe_time must be between 0 and 1.")
    tokens = int(policy_config.get("visual_bottleneck_tokens", 0))
    if top_n > tokens:
        raise ValueError(f"top_queries={top_n} exceeds bottleneck tokens={tokens}.")

    output_override = str(raw.get("output_dir", "") or "").strip()
    output_name = str(raw.get("output_name", "") or "").strip()
    if output_name and (
        Path(output_name).name != output_name or output_name in {".", ".."}
    ):
        raise ValueError(
            "output_name must be one folder name without '/' or '..'."
        )
    default_output_root = (
        project_root
        / "lerobot/examples/libero/configs/train_skillVLA/stage1/eval/outputs"
    )
    output_dir = (
        _path(project_root, output_override)
        if output_override
        else default_output_root / (output_name or f"{label}_{checkpoint_step}")
    )
    resources = raw.get("resources") or {}
    slurm = {
        "partition": ",".join(_as_list(global_config.get("train_partition"))),
        "qos": str(global_config.get("train_qos", "")),
        "gres": str(resources.get("gres", "gpu:1")),
        "cpus_per_task": int(resources.get("cpus", 8)),
        "mem": str(resources.get("memory", "64G")),
        "time": str(resources.get("time", "02:00:00")),
        "nodelist": str(global_config.get("train_nodelist", "")),
        "exclude_nodes": ",".join(_as_list(global_config.get("train_exclude_nodes"))),
    }
    return EvalConfig(
        config_path=config_path,
        project_root=project_root,
        checkpoint=checkpoint,
        dataset_dir=dataset_dir,
        repo_id=repo_id,
        dino_model_path=dino_model_path,
        video_backend="pyav",
        device="cuda",
        task_suite=task_suite,
        task_ids=task_ids,
        dataset_task_ids=dataset_task_ids,
        task_names=task_names,
        task_episode_ids=task_episode_ids,
        episode_ids=episode_ids,
        episodes_per_task=episodes_per_task,
        skills_per_episode=skills,
        frames_per_skill=frames,
        max_samples=max_samples,
        top_n=top_n,
        action_maps=action_maps,
        action_probe_time=action_probe_time,
        overlay_alpha=0.5,
        save_all_query_arrays=True,
        output_dir=output_dir,
        slurm=slurm,
        architecture_label=label,
        visual_bottleneck_tokens=tokens,
    )


def shell_exports(config: EvalConfig) -> str:
    slurm = config.slurm
    values = {
        "PROJECT_ROOT": config.project_root,
        "LEROBOT_ROOT": config.project_root / "lerobot",
        "ATTENTION_EVAL_CONFIG": config.config_path,
        "EVAL_OUTPUT_DIR": config.output_dir,
        "EVAL_PARTITION": slurm["partition"],
        "EVAL_QOS": slurm["qos"],
        "EVAL_GRES": slurm["gres"],
        "EVAL_CPUS_PER_TASK": slurm["cpus_per_task"],
        "EVAL_MEM": slurm["mem"],
        "EVAL_TIME": slurm["time"],
        "EVAL_NODELIST": slurm["nodelist"],
        "EVAL_EXCLUDE_NODES": slurm["exclude_nodes"],
    }
    return "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    config = load_eval_config(args.config)
    print(shell_exports(config) if args.shell else json.dumps(config.__dict__, default=str, indent=2))


if __name__ == "__main__":
    main()
