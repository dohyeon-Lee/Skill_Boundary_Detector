"""Resolve simulator-free diffaux3/4 XYZ-head evaluation settings."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

from lerobot.configs.uv_fullskill import UV_BSPLINE_GT_XYZ_LABELS

HERE = Path(__file__).resolve()
EVAL_DIR = HERE.parent.parent
CONFIGS = HERE.parents[4]

sys.path.insert(0, str(CONFIGS / "train_skills/src"))
from train_skills_config import (  # noqa: E402
    as_list,
    get_value,
    load_config,
    stage1_run_dir,
)

sys.path.insert(0, str(HERE.parent))
from stage1_eval_config import (  # noqa: E402
    _checkpoint_contract,
    _resolve_vsa_eval_checkpoint,
)

DEFAULT_CONFIG_PATH = EVAL_DIR / "xyz_head_eval_config.yaml"


def _safe_name(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", text):
        raise ValueError(f"{field} must be a safe folder name, got {text!r}.")
    return text


def _path(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _checkpoint_name(value: object) -> str:
    text = str(value or "last").strip()
    return f"{int(text):06d}" if text.isdigit() else text


def build_settings(config: dict, *, validate: bool = True) -> dict:
    project_root = Path(str(get_value(config, "project_root"))).expanduser().resolve()
    outputs_root = _path(project_root, get_value(config, "outputs_root", "outputs_filtered"))
    rows = config.get("models") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("models must contain at least one diffaux3/4 VSA checkpoint.")

    panels = []
    dataset_contract = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"models[{index}] must be a mapping.")
        model_dir = _safe_name(row.get("model_dir"), field="models[].model_dir")
        requested = _checkpoint_name(row.get("checkpoint", "last"))
        run_dir = stage1_run_dir(outputs_root, model_dir, "VSA")
        policy_path, ft_reference = _resolve_vsa_eval_checkpoint(
            project_root, outputs_root, run_dir, requested
        )
        if ft_reference is not None:
            raise ValueError(
                "XYZ-head PT evaluation does not accept FT step-zero aliases; "
                f"select the original PT checkpoint directly: {policy_path}"
            )
        contract = _checkpoint_contract(policy_path, project_root)
        label = contract["architecture_label"]
        if label not in UV_BSPLINE_GT_XYZ_LABELS:
            raise ValueError(
                f"Checkpoint has no deploy-time start-image XYZ head: {label!r} at {policy_path}"
            )
        current_dataset = {
            "skill_dataset_dir": str(contract["skill_dataset_dir"]),
            "eval_init_states_path": str(contract["eval_init_states_path"]),
            "focus_uv_path": str(contract["focus_uv_path"]),
            "raw_dataset_dir": str(contract["raw_dataset_dir"]),
            "proprio_grounding": contract["proprio_grounding"],
            "skill_fsq_levels": contract["policy"].get("skill_fsq_levels"),
        }
        if dataset_contract is None:
            dataset_contract = current_dataset
        elif current_dataset != dataset_contract:
            raise ValueError(
                "All XYZ-head panels must use the same SkillVLA dataset, FSQ geometry, "
                "focus artifact and proprio grounding."
            )
        display = str(row.get("label") or label).strip()
        if not display:
            raise ValueError("models[].label cannot be blank.")
        folder_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", display).strip("_")
        panels.append(
            {
                "label": display,
                "model_dir": model_dir,
                "checkpoint": policy_path.parent.name,
                "path": str(policy_path),
                "folder": f"{index:02d}_{folder_label}_{policy_path.parent.name}",
                "architecture_label": label,
                "architecture_revision": contract["architecture_revision"],
                "dino_model_path": str(contract["dino_model_path"]),
                "tokenizer_path": str(contract["tokenizer_path"]),
            }
        )

    assert dataset_contract is not None
    if len({panel["folder"] for panel in panels}) != len(panels):
        raise ValueError("XYZ-head panel output folders must be unique.")

    dataset = config.get("dataset") or {}
    task_ids = dataset.get("task_ids", list(range(10)))
    if not isinstance(task_ids, list) or not task_ids or any(
        type(value) is not int or value < 0 for value in task_ids
    ):
        raise ValueError("dataset.task_ids must be a non-empty list of nonnegative integers.")
    episodes_per_task = int(dataset.get("episodes_per_task", 2))
    if episodes_per_task <= 0:
        raise ValueError("dataset.episodes_per_task must be positive.")
    seed = int(dataset.get("seed", 1000))

    device = str(config.get("device", "cuda")).strip().lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu|cuda.")
    batch_size = int(config.get("batch_size", 1))
    cpu_threads = int(config.get("cpu_threads", 8))
    requested_gpus = int(config.get("eval_num_gpus", len(panels)))
    if batch_size <= 0 or cpu_threads <= 0 or requested_gpus <= 0:
        raise ValueError("batch_size, cpu_threads and eval_num_gpus must be positive.")
    worker_count = min(requested_gpus, len(panels)) if device == "cuda" else 1

    skill_dataset_dir = Path(dataset_contract["skill_dataset_dir"])
    run_dir = skill_dataset_dir.parent
    focus_path = Path(dataset_contract["focus_uv_path"])
    original_name = str(dataset.get("original_suite", "libero_90") or "libero_90")
    original_dataset_dir = project_root / "libero_original_dataset" / _safe_name(
        original_name, field="dataset.original_suite"
    )
    required = [
        skill_dataset_dir / "meta/info.json",
        run_dir / "skill_latents.npz",
        Path(dataset_contract["eval_init_states_path"]),
        focus_path,
        original_dataset_dir,
    ]
    if validate:
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"XYZ-head evaluation requires {path}")

    output_name = _safe_name(
        config.get("output_name", "diffaux_xyz_head_offline"), field="output_name"
    )
    visualization = config.get("visualization") or {}
    max_samples = int(visualization.get("max_samples", 0))
    if max_samples < 0:
        raise ValueError("visualization.max_samples must be >= 0.")

    slurm = config.get("slurm") or {}
    cpu_slurm = config.get("slurm_cpu") or {}
    partitions = (
        as_list(get_value(config, "train_partition", []))
        if device == "cuda"
        else as_list(cpu_slurm.get("partition", ""))
    )
    return {
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "init_map": dataset_contract["eval_init_states_path"],
        "focus_uv_path": str(focus_path),
        "original_dataset_dir": str(original_dataset_dir),
        "task_ids": task_ids,
        "episodes_per_task": episodes_per_task,
        "seed": seed,
        "device": device,
        "batch_size": batch_size,
        "cpu_threads": cpu_threads,
        "eval_num_gpus": requested_gpus,
        "worker_count": worker_count,
        "models": panels,
        "visualization": {
            "enabled": bool(visualization.get("enabled", True)),
            "max_samples": max_samples,
        },
        "resume": bool(config.get("resume", True)),
        "output_dir": str(EVAL_DIR / "outputs_xyz_head" / output_name),
        "slurm": {
            "partition": ",".join(partitions),
            "qos": str(
                get_value(config, "train_qos", "")
                if device == "cuda"
                else cpu_slurm.get("qos", "")
            ),
            "cpus": int(slurm.get("cpus", 8)),
            "memory": str(slurm.get("memory", "96G")),
            "time": str(slurm.get("time", "4:00:00")),
            "nodelist": str(get_value(config, "train_nodelist", "") or ""),
            "exclude": ",".join(as_list(get_value(config, "train_exclude_nodes", []))),
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
    return "\n".join(
        f"export XYZ_HEAD_{key}={shlex.quote(str(value))}"
        for key, value in values.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config))
    print(shell_settings(settings) if args.shell else json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
