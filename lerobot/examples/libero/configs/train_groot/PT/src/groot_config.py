#!/usr/bin/env python3
"""Resolve our PT/FT YAML for the pinned Hugging Face LeRobot Groot port.

This module deliberately imports neither torch nor our in-tree LeRobot package;
submission and command previews work while model shards are still downloading.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any

CONFIGS_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CONFIGS_ROOT / "src"))
from global_config_loader import load_global_config, read_yaml


def _mapping(value: Any, name: str, allowed: set[str]) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unsupported {name} settings: {sorted(unknown)}")
    return value


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {value!r}.")
    return value


def _component(value: Any, name: str, empty: bool = False) -> str:
    value = str(value)
    if empty and not value:
        return value
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
        raise ValueError(f"{name} must be a single folder name, got {value!r}.")
    return value


def _path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    global_path = next((parent / "global_config.yaml" for parent in path.parents
                        if (parent / "global_config.yaml").is_file()), None)
    if global_path is None:
        raise FileNotFoundError(f"Cannot find global_config.yaml above {path}.")
    result = {**load_global_config(global_path), **read_yaml(path), "_config_path": str(path)}
    frozen = path.parent / "groot_resolved.json"
    if ".config_snapshots" in path.parts and frozen.is_file():
        result["_initial_checkpoint"] = json.loads(frozen.read_text())["initial_checkpoint"]
    return result


def checkpoint_step(value: Any) -> str:
    text = str(value)
    if text == "last":
        return text
    match = re.fullmatch(r"(?:step)?(\d+)", text)
    if not match or int(match[1]) < 1:
        raise ValueError("warm_start.checkpoint must be 'last', a positive step, or 'step<step>'.")
    return f"{int(match[1]):06d}"


def checkpoint_grounding(path: Path) -> str:
    """The coordinate convention travels with each transferable pretrained_model."""
    config_path = path / "pretrained_model/sbd_groot.json"
    if not config_path.is_file():
        if (path / "pretrained_model/config.json").exists():
            raise ValueError(f"Missing grounding provenance: {config_path}")
        return "none"  # command preview for a PT checkpoint that does not exist yet
    mode = str(json.loads(config_path.read_text())["proprio_grounding"])
    if mode not in {"none", "episode_start_xyz"}:
        raise ValueError(f"Unsupported checkpoint proprio_grounding={mode!r} in {config_path}.")
    return mode


def build_settings(config: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    stage = str(config.get("stage", ""))
    if stage not in {"pt", "ft"}:
        raise ValueError("stage must be pt or ft.")
    project = Path(config["project_root"]).expanduser().resolve()
    runtime = _mapping(config.get("runtime"), "runtime", {"venv", "source"})
    source = _path(project, runtime.get("source", "third_party/lerobot_groot"))
    python = _path(project, runtime.get("venv", ".venv_groot_lerobot")) / "bin/python"
    dataset = _mapping(config.get("dataset"), "dataset", {"name", "action_horizon"})
    name = _component(dataset.get("name", "libero_90_full_full" if stage == "pt" else "libero_10_full_1"), "dataset.name")
    horizon = _integer(dataset.get("action_horizon", 10), "dataset.action_horizon")
    if horizon > 40:
        raise ValueError("GR00T N1.7 supports action_horizon <= 40.")
    root = _path(project, config.get("dataset_root", "dataset_filtered"))
    outputs = _path(project, config.get("outputs_root", "outputs_filtered"))
    run = _mapping(config.get("run"), "run", {"suffix"})
    suffix = _component(run.get("suffix", ""), "run.suffix", empty=True)
    training = _mapping(config.get("training"), "training", {
        "dataloader", "tuning", "gradient_checkpointing",
        "optimizer", "schedule", "keep_checkpoints", "protect_checkpoints",
    })
    dl = _mapping(training.get("dataloader"), "training.dataloader", {"batch_size", "micro_batch_size", "workers", "gpus"})
    batch = _integer(dl.get("batch_size", 16), "batch_size")
    micro = _integer(dl.get("micro_batch_size", 1), "micro_batch_size")
    workers = _integer(dl.get("workers", 4), "workers", minimum=0)
    gpus = _integer(dl.get("gpus", 1), "gpus")
    if batch % (micro * gpus):
        raise ValueError("Global batch_size must be divisible by micro_batch_size * gpus.")
    tuning = training.get("tuning", "default")
    if tuning not in {"default", "full"}:
        raise ValueError("training.tuning must be default or full.")
    checkpointing = _boolean(training.get("gradient_checkpointing", True), "gradient_checkpointing")
    optimizer = _mapping(training.get("optimizer"), "training.optimizer", {"base_lr"})
    lr = optimizer.get("base_lr", 1e-4)
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not 0 < lr < 1:
        raise ValueError("training.optimizer.base_lr must be a positive learning rate < 1.")
    schedule = _mapping(training.get("schedule"), "training.schedule", {
        "steps", "lr_mode", "warmup_steps", "log_every", "save_every",
    })
    lr_mode = schedule.get("lr_mode", "cosine_decay")
    if lr_mode not in {"warmup_constant", "cosine_decay"}:
        raise ValueError("training.schedule.lr_mode must be warmup_constant or cosine_decay.")
    steps = _integer(schedule.get("steps", 100000 if stage == "pt" else 50000), "steps")
    warmup = _integer(schedule.get("warmup_steps", 200), "warmup_steps", minimum=0)
    log_every = _integer(schedule.get("log_every", 20), "log_every")
    save_every = _integer(schedule.get("save_every", 10000 if stage == "pt" else 5000), "save_every")
    keep = training.get("keep_checkpoints", -1)
    if isinstance(keep, bool) or not isinstance(keep, int) or keep == 0 or keep < -1:
        raise ValueError("keep_checkpoints must be -1 (keep all) or a positive integer.")
    protected = training.get("protect_checkpoints", [])
    if not isinstance(protected, list):
        raise ValueError("training.protect_checkpoints must be a list of positive integer steps.")
    protected = sorted({_integer(step, "training.protect_checkpoints entry") for step in protected})
    for step in protected:
        # This is a retention policy, not a new save scheduler. Native regular or
        # final saving must actually produce every requested protected checkpoint.
        if step % save_every != 0 and step != steps:
            raise ValueError(f"Protected step {step} must be a multiple of save_every={save_every} or the final step={steps}.")
    model = _mapping(config.get("model"), "model", {"checkpoint"})
    if stage == "pt":
        initial = _path(project, model.get("checkpoint", "models/groot_n1_7_base"))
        base = initial
        run_name = f"bs{batch}_{name}"
        grounding = "episode_start_xyz" if _boolean(config.get("pt_proprio_grounding", False), "pt_proprio_grounding") else "none"
    else:
        if "pt_proprio_grounding" in config:
            raise ValueError("FT inherits proprio grounding from its PT checkpoint; remove pt_proprio_grounding from the FT YAML.")
        warm_start = _mapping(config.get("warm_start"), "warm_start", {"run_name", "checkpoint"})
        pt_run = _component(warm_start.get("run_name", ""), "warm_start.run_name")
        step = checkpoint_step(warm_start.get("checkpoint", "last"))
        pt_output = outputs / "groot_PT" / pt_run
        if config.get("_initial_checkpoint"):
            initial = Path(config["_initial_checkpoint"])
        elif step == "last":
            initial = latest_checkpoint(pt_output) or pt_output / "checkpoints/last"
        else:
            initial = pt_output / "checkpoints" / step
        # Resolve 'last' into a concrete step so a later PT save cannot change an FT run's identity.
        run_name = f"{pt_run}_{initial.name}_{name}_ft_bs{batch}"
        grounding = checkpoint_grounding(initial)
        # LeRobot reconstructs the architecture from the raw base before loading
        # our PT weights. A copied checkpoint must not retain another server's path.
        saved_config = initial / "pretrained_model/config.json"
        saved_base = json.loads(saved_config.read_text()).get("base_model_path") if saved_config.is_file() else None
        base = _path(project, model.get("checkpoint") or saved_base or "models/groot_n1_7_base")
        if not base.is_dir() and "checkpoint" not in model:
            base = project / "models/groot_n1_7_base"
    if suffix:
        run_name += f"_{suffix}"
    run_name += "_n1_7_lerobot"
    logging = _mapping(config.get("logging"), "logging", {"wandb"})
    wandb = _mapping(logging.get("wandb"), "logging.wandb", {"enable", "project", "entity"})
    wandb_enable = _boolean(wandb.get("enable", True), "logging.wandb.enable")
    slurm = _mapping(config.get("slurm"), "slurm", {"gres", "cpus", "memory", "time"})
    cpus = _integer(slurm.get("cpus", 16), "slurm.cpus")
    if workers * gpus > cpus:
        raise ValueError("workers * gpus exceeds slurm.cpus; workers is per GPU.")
    gres = str(slurm.get("gres", f"gpu:{gpus}"))
    match = re.fullmatch(r"gpu(?::[^:]+)?:(\d+)", gres)
    if not match or int(match[1]) != gpus:
        raise ValueError("slurm.gres GPU count must match training.dataloader.gpus.")
    if smoke:
        run_name += "_smoke"
        batch, micro, workers, gpus = 1, 1, 0, 1
        steps, warmup, log_every, save_every, keep = 20, 2, 1, 20, 1
        protected = []
        wandb_enable, gres = False, "gpu:1"
    return {
        "stage": stage, "project_root": str(project), "python": str(python), "source": str(source),
        "dataset_root": str(root), "dataset_dir": str(root / name), "dataset": name, "horizon": horizon,
        "initial_checkpoint": str(initial), "base_checkpoint": str(base), "run_name": run_name,
        "output_dir": str(outputs / f"groot_{stage.upper()}" / run_name),
        "batch_size": batch, "micro_batch_size": micro, "workers": workers, "gpus": gpus,
        "accumulation_steps": batch // (micro * gpus), "backend": "hf_lerobot",
        "tuning": tuning, "checkpointing": checkpointing,
        "lr": float(lr), "steps": steps, "warmup_steps": warmup,
        "lr_mode": lr_mode, "proprio_grounding": grounding,
        "log_every": log_every, "save_every": save_every, "keep_checkpoints": keep,
        "protect_checkpoints": protected,
        "wandb_enable": wandb_enable, "wandb_project": str(wandb.get("project", "VLA_stage1" if stage == "pt" else "VLA_NewTask_FT")),
        "wandb_entity": str(wandb.get("entity", "") or ""),
        "partition": _csv(config.get("train_partition", "")), "qos": str(config.get("train_qos", "") or ""),
        "nodelist": _csv(config.get("train_nodelist", "")), "exclude": _csv(config.get("train_exclude_nodes", "")),
        "gres": gres, "cpus": cpus, "memory": str(slurm.get("memory", "128G")),
        "time": str(slurm.get("time", "24:00:00")), "smoke": smoke,
        "config_path": str(config.get("_config_path", "")),
    }


def _csv(value: Any) -> str:
    return ",".join(map(str, value)) if isinstance(value, list) else str(value or "")


def is_training_checkpoint(path: Path) -> bool:
    required = ("pretrained_model/config.json", "pretrained_model/train_config.json",
                "pretrained_model/model.safetensors", "pretrained_model/sbd_groot.json",
                "pretrained_model/policy_preprocessor.json", "pretrained_model/policy_postprocessor.json",
                "training_state/training_step.json", "training_state/scheduler_state.json",
                "training_state/rng_state.safetensors")
    optimizer = (all((path / "training_state" / name).is_file() for name in
                     ("optimizer_state.safetensors", "optimizer_param_groups.json"))
                 or (path / "training_state/optimizer_0/.metadata").is_file())
    return all((path / name).is_file() for name in required) and optimizer


def latest_checkpoint(output: Path) -> Path | None:
    paths = [p for p in (output / "checkpoints").glob("*") if p.name.isdigit()]
    if not paths:
        return None
    latest = max(paths, key=lambda p: int(p.name))
    if not is_training_checkpoint(latest):
        raise ValueError(f"Latest checkpoint is incomplete: {latest}. Resolve this before resuming.")
    return latest


def validate_inputs(settings: dict[str, Any], model: bool = True) -> None:
    for item in (settings["python"], str(Path(settings["source"]) / "src/lerobot/scripts/lerobot_train.py")):
        if not Path(item).is_file():
            raise FileNotFoundError(f"Required file missing: {item}")
    info_path = Path(settings["dataset_dir"]) / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata missing: {info_path}")
    info = json.loads(info_path.read_text())
    features = info["features"]
    expected = {"observation.images.image": [256, 256, 3], "observation.images.wrist_image": [256, 256, 3],
                "observation.state": [8], "action": [7]}
    for key, shape in expected.items():
        if features.get(key, {}).get("shape") != shape:
            raise ValueError(f"Expected LIBERO feature {key} shape {shape}, got {features.get(key)}")
    if not model:
        return
    checkpoint = Path(settings["initial_checkpoint"])
    validate_base_checkpoint(Path(settings["base_checkpoint"]))
    if settings["stage"] == "ft" and not is_training_checkpoint(checkpoint):
        raise FileNotFoundError(f"Complete Groot checkpoint not found: {checkpoint}")
    output = Path(settings["output_dir"])
    latest = latest_checkpoint(output)
    if latest:
        if checkpoint_grounding(latest) != settings["proprio_grounding"]:
            raise ValueError(f"Cannot change proprio grounding while resuming {latest}; use another run.suffix.")
    elif output.exists():
        raise ValueError(f"{output} exists but has no resumable checkpoint. Use a new run.suffix.")


def validate_base_checkpoint(checkpoint: Path) -> None:
    config = checkpoint / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"GR00T N1.7 base model missing: {checkpoint}. Download model weights separately first.")
    data = json.loads(config.read_text())
    markers = [data.get("model_version", ""), data.get("model_type", ""), *(data.get("architectures") or [])]
    if not any(str(value).lower().replace("-", "_") in
               {"n1.7", "n1_7", "gr00tn1d7", "gr00t_n1d7", "gr00t_n1_7"} for value in markers):
        raise ValueError(f"Expected a raw GR00T N1.7 checkpoint, not another model/version: {config}")
    index = checkpoint / "model.safetensors.index.json"
    shards = (set(json.loads(index.read_text())["weight_map"].values()) if index.is_file()
              else {p.name for p in checkpoint.glob("*.safetensors")})
    if any(Path(name).name != name for name in shards):
        raise ValueError(f"Invalid model shard paths in {index}")
    missing = [name for name in sorted(shards) if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0]
    if not shards or missing:
        raise FileNotFoundError(f"Model download is incomplete at {checkpoint}; missing shards: {missing or 'all'}")


def trainer_args(s: dict[str, Any]) -> list[str]:
    """Translate optimizer-update units into upstream's micro-batch clock."""
    accum = s["accumulation_steps"]
    values = {
        "dataset.repo_id": s["dataset"], "dataset.root": s["dataset_dir"],
        "dataset.video_backend": "pyav", "dataset.use_imagenet_stats": False,
        "batch_size": s["micro_batch_size"], "num_workers": s["workers"],
        "persistent_workers": s["workers"] > 0, "prefetch_factor": 2,
        "accelerator.gradient_accumulation.steps": accum,
        "accelerator.mixed_precision": "no",  # upstream policy owns BF16 autocast
        "steps": s["steps"] * accum, "log_freq": s["log_every"] * accum,
        "save_freq": s["save_every"] * accum, "env_eval_freq": 0,
        "output_dir": s["output_dir"], "job_name": s["run_name"],
        "policy.push_to_hub": False, "policy.chunk_size": s["horizon"],
        "policy.n_action_steps": s["horizon"], "policy.device": "cuda",
        "policy.dtype": "float32", "policy.use_bf16": True,
        "policy.base_model_path": s["base_checkpoint"],  # this pinned port supports only N1.7
        "policy.embodiment_tag": "libero_sim", "policy.use_relative_actions": False,
        "policy.tune_llm": s["tuning"] == "full", "policy.tune_visual": s["tuning"] == "full",
        "policy.tune_projector": True, "policy.tune_diffusion_model": True, "policy.tune_vlln": True,
        "policy.use_flash_attention": False, "policy.action_decode_transform": "libero",
        "policy.max_steps": s["steps"] * accum,
        "wandb.enable": s["wandb_enable"], "wandb.project": s["wandb_project"],
    }
    if s["wandb_entity"]:
        values["wandb.entity"] = s["wandb_entity"]
    values["policy.optimizer_lr"] = s["lr"]
    # Native GR00T owns 256->230 spatial crop. Do not apply Molmo's second crop/rotation.
    values.update({
        "dataset.image_transforms.enable": True,
        "dataset.image_transforms.max_num_transforms": 4,
        "dataset.image_transforms.random_order": False,
        "dataset.image_transforms.tfs": {
            "brightness": {"weight": 1., "type": "ColorJitter", "kwargs": {"brightness": [0.7, 1.3]}},
            "contrast": {"weight": 1., "type": "ColorJitter", "kwargs": {"contrast": [0.6, 1.4]}},
            "saturation": {"weight": 1., "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
            "hue": {"weight": 1., "type": "ColorJitter", "kwargs": {"hue": [-0.08, 0.08]}},
        },
    })
    resume = latest_checkpoint(Path(s["output_dir"]))
    prefix = []
    if resume:
        prefix = [f"--config_path={resume / 'pretrained_model/train_config.json'}", "--resume=true"]
    elif s["stage"] == "ft":
        prefix = [f"--policy.path={Path(s['initial_checkpoint']) / 'pretrained_model'}"]
    else:
        prefix = ["--policy.type=groot"]
    return prefix + [f"--{key}={json.dumps(value)}" for key, value in values.items()]


def command(s: dict[str, Any]) -> list[str]:
    argv = [s["python"], "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc-per-node={s['gpus']}", str(Path(__file__).with_name("train.py")),
            "--config", s["config_path"]]
    return argv + (["--smoke"] if s["smoke"] else [])


def shell_exports(s: dict[str, Any]) -> str:
    exports = {"PROJECT_ROOT": s["project_root"], "GROOT_PYTHON": s["python"],
               "GROOT_OUTPUT_DIR": s["output_dir"], "GROOT_DATASET_DIR": s["dataset_dir"],
               "GROOT_RUN_NAME": s["run_name"], "GROOT_NUM_GPUS": s["gpus"],
               "GROOT_STAGE": s["stage"], "GROOT_PARTITION": s["partition"],
               "GROOT_QOS": s["qos"], "GROOT_NODELIST": s["nodelist"],
               "GROOT_EXCLUDE": s["exclude"], "GROOT_GRES": s["gres"],
               "GROOT_CPUS": s["cpus"], "GROOT_MEM": s["memory"], "GROOT_TIME": s["time"]}
    return "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in exports.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--check", action="store_true", help="validate local dataset and complete model shards")
    parser.add_argument("--freeze", action="store_true", help="pin 'last' in an existing per-submit snapshot")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config), smoke=args.smoke)
    if args.check:
        validate_inputs(settings)
    if args.freeze:
        if ".config_snapshots" not in args.config.resolve().parts:
            raise ValueError("--freeze requires a per-submit .config_snapshots bundle.")
        frozen = args.config.resolve().parent / "groot_resolved.json"
        with frozen.open("x") as stream:
            json.dump({"initial_checkpoint": settings["initial_checkpoint"]}, stream)
    if args.shell:
        print(shell_exports(settings))
    else:
        print(json.dumps(settings, indent=2))
        print("\nLauncher:\n" + shlex.join(command(settings)))
        print("\nHF LeRobot trainer arguments:\n" + shlex.join(trainer_args(settings)))


if __name__ == "__main__":
    main()
