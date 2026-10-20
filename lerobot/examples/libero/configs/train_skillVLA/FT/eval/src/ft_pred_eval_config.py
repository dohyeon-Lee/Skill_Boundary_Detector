"""Resolve offline Predictor-only evaluation without importing torch or a simulator."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONFIGS / "src"))
from eval_gpu_packing import balanced_chunks  # noqa: E402

sys.path.insert(0, str(CONFIGS / "train_skillVLA/build_data/src"))
from train_skillVLA_config import (  # noqa: E402
    _resolve_predictor_checkpoint,
    load_config,
    stage1_run_dirs,
    validate_relabel_code_space,
)

DEFAULT_CONFIG_PATH = EVAL_DIR / "ft_pred_eval_config.yaml"


def safe_name(value, field: str) -> str:
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{field} must be a single safe folder name, got {value!r}.")
    return value


def _path(root: Path, value) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _is_ft_checkpoint(path: Path) -> bool:
    return len(path.parents) > 4 and path.parents[4].name in {"skillVLA_FT", "skillVLA_NewTask_FT"}


def _resolve_eval_predictor_checkpoint(outputs_root: Path, name: str, requested: str):
    """FT step zero means the original PT weights AND their saved preprocessing.

    Saved FT step-zero weights have no optimizer updates, but their processors
    may already contain target-dataset statistics. Never use that as a silent
    substitute for PT zero-shot. Other checkpoint consumers remain unchanged.
    """
    requested = str(requested).strip()
    zero = requested.isdigit() and int(requested) == 0
    try:
        path, step = _resolve_predictor_checkpoint(outputs_root, name, requested)
    except FileNotFoundError:
        if not zero:
            raise
        # Retention may have pruned FT step zero. Later checkpoints still
        # retain the source in their policy config, so no FT weights are needed.
        candidates = [
            checkpoint / "pretrained_model"
            for run in stage1_run_dirs(outputs_root, name, "Predictor")
            for checkpoint in sorted((run / "checkpoints").glob("*"))
            if checkpoint.name.isdigit()
            and _is_ft_checkpoint(checkpoint / "pretrained_model")
            and (checkpoint / "pretrained_model/config.json").is_file()
        ]
        if not candidates:
            raise
        path, step = candidates[0], "000000"
    if not zero or not _is_ft_checkpoint(path):
        return path, step, {}

    run = path.parents[2]
    metadata_paths = [path, *sorted((run / "checkpoints").glob("*/pretrained_model"))]
    for metadata in dict.fromkeys(metadata_paths):
        for filename in ("config.json", "train_config.json"):
            config_path = metadata / filename
            if not config_path.is_file():
                continue
            policy = json.loads(config_path.read_text())
            if filename == "train_config.json":
                policy = policy.get("policy") or {}
            recorded = Path(str(policy.get("skill_predictor_checkpoint_path") or ""))
            if (
                recorded.name != "pretrained_model"
                or len(recorded.parents) < 3
                or recorded.parents[1].name != "checkpoints"
                or recorded.parents[2].name == name
            ):
                continue  # A resumed run can point to its own earlier FT step.
            source_name, source_step = recorded.parents[2].name, recorded.parent.name
            try:
                source, source_step = _resolve_predictor_checkpoint(outputs_root, source_name, source_step)
            except FileNotFoundError as error:
                if all((recorded / f).is_file() for f in ("config.json", "model.safetensors")):
                    source = recorded
                else:
                    raise FileNotFoundError(
                        f"Predictor 000000 requires the original PT checkpoint, including its "
                        f"preprocessor: {recorded}. Download that checkpoint; saved FT-zero "
                        "statistics will not be substituted."
                    ) from error
            if _is_ft_checkpoint(source):
                raise ValueError(f"Predictor 000000 source is another FT checkpoint, not PT: {source}")
            for artifact in (
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "train_config.json",
            ):
                if not (source / artifact).is_file():
                    raise FileNotFoundError(f"Incomplete PT zero-shot Predictor: {source / artifact}")
            return (
                source,
                "000000",
                {
                    "zero_shot": True,
                    "source_checkpoint": source_step,
                    "ft_reference_path": str(metadata),
                },
            )
    raise ValueError(
        f"Cannot determine the original PT Predictor for {name}: no external "
        "skill_predictor_checkpoint_path in its saved configs. Refusing to use FT-zero statistics."
    )


def build_settings(config: dict, *, validate: bool = True) -> dict:
    root = Path(config["project_root"]).resolve()
    dataset_root = _path(root, config.get("dataset_root", "dataset_filtered"))
    outputs_root = _path(root, config.get("outputs_root", "outputs_filtered"))
    data = config.get("dataset") or {}
    source = safe_name(data.get("source", ""), "dataset.source")
    run = safe_name(data.get("run", ""), "dataset.run")
    skill_root = dataset_root / "skillvla_dataset"
    run_dir = skill_root / source / run
    exclude_sources = data.get("exclude_sources", [])
    if not isinstance(exclude_sources, list):
        raise ValueError("dataset.exclude_sources must be a list of dataset names.")
    exclusions = [
        skill_root / safe_name(s, "dataset.exclude_sources") / "eval_init_states.npz" for s in exclude_sources
    ]
    training_sources = data.get("training_sources", [])
    if not isinstance(training_sources, list):
        raise ValueError("dataset.training_sources must be a list of dataset names.")
    training_maps = [
        skill_root / safe_name(s, "dataset.training_sources") / "eval_init_states.npz"
        for s in training_sources
    ]
    task_ids = data.get("task_ids", [])
    if not isinstance(task_ids, list) or any(type(v) is not int or v < 0 for v in task_ids):
        raise ValueError("dataset.task_ids must be a list of nonnegative integers.")
    episodes = int(data.get("episodes_per_task", 10))
    batch_size = int(config.get("batch_size", 1))
    threads = int(config.get("cpu_threads", 8))
    device = str(config.get("device", "cuda"))
    requested_gpus = config.get("eval_num_gpus", 1)
    if type(requested_gpus) is not int or requested_gpus < 1:
        raise ValueError("eval_num_gpus must be a positive integer.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu|cuda (no implicit CPU/GPU fallback).")
    if episodes < 0 or batch_size < 1 or threads < 1:
        raise ValueError("episodes_per_task must be >= 0; batch_size and cpu_threads must be positive.")
    visual = config.get("visualization") or {}
    max_images = int(visual.get("max_samples", 0))
    if max_images < 0:
        raise ValueError("visualization.max_samples must be >= 0.")
    output_name = safe_name(config.get("output_name", "predictor_offline"), "output_name")
    models = config.get("models", [])
    if not isinstance(models, list) or not models:
        raise ValueError("models must contain at least one Predictor.")
    if validate and not (run_dir / "skillvla/meta/info.json").is_file():
        raise FileNotFoundError(
            f"Completed SkillVLA dataset not found: {run_dir / 'skillvla/meta/info.json'}. "
            "eval_init_states.npz alone is not enough; finish build_data/submit_build_all.sh first."
        )
    info = json.loads((run_dir / "skillvla/meta/info.json").read_text()) if validate else {}
    panels = []
    for index, model in enumerate(models):
        label = str(model.get("label", f"predictor_{index}"))
        name = safe_name(model.get("predictor_dir", ""), "models[].predictor_dir")
        steps = model.get("predictor_checkpoint", "last")
        if not isinstance(steps, list):
            steps = [steps]
        if not steps or len({str(s) for s in steps}) != len(steps):
            raise ValueError("predictor_checkpoint must contain unique, nonempty checkpoints.")
        for step in steps:
            path, step, provenance = _resolve_eval_predictor_checkpoint(outputs_root, name, str(step))
            policy = json.loads((path / "config.json").read_text())
            if policy.get("type") not in {"skill_aux", "skill_expert"} or not (
                policy.get("train_skill_predictor") or policy.get("newtask_joint_enabled")
            ):
                raise ValueError(f"Not a trained Predictor checkpoint: {path}")
            if policy.get("skill_predictor_end_state_mode", "off") == "off":
                raise ValueError(f"Predictor has no XYZ/end-state head: {path}")
            levels = policy.get("skill_fsq_levels", [])
            if validate:
                for artifact in ("policy_preprocessor.json", "train_config.json"):
                    if not (path / artifact).is_file():
                        raise FileNotFoundError(f"Incomplete Predictor checkpoint: {path / artifact}")
                if levels != info.get("skill_fsq_levels") or math.prod(levels) < 1:
                    raise ValueError(f"Predictor/dataset FSQ geometry mismatch: {path}")
                validate_relabel_code_space(info, policy, run_dir)
                # Auxiliary configs historically did not save proprio_grounding;
                # dataset.root can be an ephemeral /tmp staging path. fsq_path
                # retains the stable training dataset/run tail across servers.
                grounding = policy.get("proprio_grounding")
                if grounding is None:
                    fsq_dir = Path(str(policy.get("fsq_path", ""))).parent
                    candidates = [fsq_dir / "skillvla/meta/info.json"]
                    if len(fsq_dir.parts) >= 2:
                        candidates.append(
                            skill_root / fsq_dir.parts[-2] / fsq_dir.name / "skillvla/meta/info.json"
                        )
                    training_info = next((p for p in candidates if p.is_file()), None)
                    if training_info is None:
                        raise FileNotFoundError(
                            f"Cannot verify Predictor grounding from its fsq_path: {path}"
                        )
                    grounding = json.loads(training_info.read_text()).get("proprio_grounding", "none")
                if grounding != info.get("proprio_grounding", "none"):
                    raise ValueError("Predictor/evaluation dataset proprio grounding mismatch.")
            panels.append(
                {
                    "label": f"{label} | pred {step}" + (" (PT zero-shot)" if provenance else ""),
                    "predictor_dir": name,
                    "checkpoint": step,
                    "path": str(path),
                    "folder": f"{index:02d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', label)}_{step}",
                    **provenance,
                }
            )
    if len({panel["folder"] for panel in panels}) != len(panels):
        raise ValueError("Predictor panels must have unique output folders; check duplicate checkpoints.")
    worker_count = min(requested_gpus, len(panels)) if device == "cuda" else 1
    required = [
        run_dir / "skill_latents.npz",
        skill_root / source / "eval_init_states.npz",
        *exclusions,
        *training_maps,
    ]
    if validate:
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Offline Predictor evaluation requires {path}")
    slurm = config.get("slurm") or {}
    cpu = config.get("slurm_cpu") or {}
    partitions = config.get("train_partition", []) if device == "cuda" else cpu.get("partition", "")
    if isinstance(partitions, list):
        partitions = ",".join(partitions)
    tokenizer = _path(root, config.get("tokenizer_path", "models/paligemma-3b-pt-224-tokenizer"))
    return {
        "project_root": str(root),
        "run_dir": str(run_dir),
        "source": source,
        "init_map": str(skill_root / source / "eval_init_states.npz"),
        "exclude_maps": [str(p) for p in exclusions],
        "training_maps": [str(p) for p in training_maps],
        "task_ids": task_ids,
        "episodes_per_task": episodes,
        "seed": int(data.get("seed", 1000)),
        "device": device,
        "batch_size": batch_size,
        "cpu_threads": threads,
        "eval_num_gpus": requested_gpus,
        "worker_count": worker_count,
        "panel_groups": balanced_chunks(list(range(len(panels))), worker_count),
        "visualization": {"enabled": bool(visual.get("enabled", True)), "max_samples": max_images},
        "resume": bool(config.get("resume", True)),
        "models": panels,
        "tokenizer_path": str(tokenizer),
        "original_dataset_dir": str(root / "libero_original_dataset/libero_10"),
        "output_dir": str(EVAL_DIR / "outputs_predictor" / output_name),
        "slurm": {
            "partition": str(partitions),
            "qos": str(config.get("train_qos", "") if device == "cuda" else cpu.get("qos", "")),
            "cpus": int(slurm.get("cpus", 8)),
            "memory": str(slurm.get("memory", "64G")),
            "time": str(slurm.get("time", "4:00:00")),
            "nodelist": str(config.get("train_nodelist", "") or ""),
            "exclude": ",".join(config.get("train_exclude_nodes", [])),
        },
    }


def shell_settings(settings: dict) -> str:
    slurm = settings["slurm"]
    values = {
        "DEVICE": settings["device"],
        "PARTITION": slurm["partition"],
        "QOS": slurm["qos"],
        "CPUS": slurm["cpus"],
        "MEM": slurm["memory"],
        "TIME": slurm["time"],
        "NODELIST": slurm["nodelist"],
        "EXCLUDE": slurm["exclude"],
        "WORKER_COUNT": settings["worker_count"],
        "OUTPUT_DIR": settings["output_dir"],
    }
    return "\n".join(f"export FT_PRED_{k}={shlex.quote(str(v))}" for k, v in values.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config))
    print(shell_settings(settings) if args.shell else json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
