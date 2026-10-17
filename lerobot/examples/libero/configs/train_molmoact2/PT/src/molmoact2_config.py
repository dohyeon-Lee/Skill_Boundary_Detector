#!/usr/bin/env python3
"""Resolve our PT/FT YAML into arguments for the unmodified MolmoAct2 trainer.

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
    frozen = path.parent / "molmoact2_resolved.json"
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
    return f"step{int(match[1])}"


def checkpoint_grounding(path: Path) -> str:
    """The mode travels inside every native checkpoint, alongside its normalization stats."""
    config_path = path / "config.yaml"
    if not config_path.is_file():
        return "none"  # preview of a pending PT run, or an older ungrounded checkpoint
    model = read_yaml(config_path).get("model") or {}
    processor = model.get("robot_processor") or {}
    tags = processor.get("metadata_by_tag") or {}
    mode = str((tags.get("libero") or {}).get("proprio_grounding", "none"))
    if mode not in {"none", "episode_start_xyz"}:
        raise ValueError(f"Unsupported checkpoint proprio_grounding={mode!r} in {config_path}.")
    return mode


def build_settings(config: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    stage = str(config.get("stage", ""))
    if stage not in {"pt", "ft"}:
        raise ValueError("stage must be pt or ft.")
    project = Path(config["project_root"]).expanduser().resolve()
    runtime = _mapping(config.get("runtime"), "runtime", {"venv", "source"})
    source = _path(project, runtime.get("source", "third_party/molmoact2/experiments"))
    python = _path(project, runtime.get("venv", ".venv_molmoact2")) / "bin/python"
    dataset = _mapping(config.get("dataset"), "dataset", {"name", "action_horizon"})
    name = _component(dataset.get("name", "libero_90_full_full" if stage == "pt" else "libero_10_full_1"), "dataset.name")
    horizon = _integer(dataset.get("action_horizon", 10), "dataset.action_horizon")
    root = _path(project, config.get("dataset_root", "dataset_filtered"))
    outputs = _path(project, config.get("outputs_root", "outputs_filtered"))
    run = _mapping(config.get("run"), "run", {"suffix"})
    suffix = _component(run.get("suffix", ""), "run.suffix", empty=True)
    training = _mapping(config.get("training"), "training", {
        "dataloader", "tuning", "gradient_checkpointing", "action_format",
        "num_flow_timesteps", "optimizer", "schedule", "keep_checkpoints", "protect_checkpoints",
    })
    dl = _mapping(training.get("dataloader"), "training.dataloader", {"batch_size", "micro_batch_size", "workers", "gpus"})
    batch = _integer(dl.get("batch_size", 16), "batch_size")
    micro = _integer(dl.get("micro_batch_size", 1), "micro_batch_size")
    workers = _integer(dl.get("workers", 4), "workers", minimum=0)
    gpus = _integer(dl.get("gpus", 1), "gpus")
    if batch % (micro * gpus):
        raise ValueError("Global batch_size must be divisible by micro_batch_size * gpus.")
    tuning = training.get("tuning", "full")
    if tuning not in {"full", "action_expert"}:
        raise ValueError("training.tuning must be full or action_expert.")
    # Our PT and FT are both downstream fine-tunes of released MolmoAct2. Follow
    # the paper's final FT recipe (both / 8), not the CLI's fallback (continuous / 1).
    action_format = training.get("action_format", "both")
    if action_format not in {"both", "continuous"}:
        raise ValueError("training.action_format must be both or continuous.")
    if tuning == "action_expert" and action_format != "continuous":
        raise ValueError("AE-only tuning requires action_format: continuous (the VLM/CE head are frozen).")
    checkpointing = _boolean(training.get("gradient_checkpointing", True), "gradient_checkpointing")
    num_flow = _integer(training.get("num_flow_timesteps", 8), "num_flow_timesteps")
    optimizer = _mapping(training.get("optimizer"), "training.optimizer", {
        "base_lr", "llm_lr", "vit_lr", "connector_lr", "action_expert_lr",
    })
    lrs = {"llm_lr": 1e-5, "vit_lr": 5e-6, "connector_lr": 5e-6, "action_expert_lr": 5e-5}
    for key in lrs:
        value = optimizer.get(key, optimizer.get("base_lr", lrs[key]))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ValueError(f"training.optimizer.{key} must be a positive learning rate < 1.")
        lrs[key] = float(value)
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
    if stage == "pt":
        model = _mapping(config.get("model"), "model", {"checkpoint"})
        initial = _path(project, model.get("checkpoint", "models/molmoact2_base"))
        run_name = f"bs{batch}_{name}"
        grounding = "episode_start_xyz" if _boolean(config.get("pt_proprio_grounding", False), "pt_proprio_grounding") else "none"
    else:
        if "pt_proprio_grounding" in config:
            raise ValueError("FT inherits proprio grounding from its PT checkpoint; remove pt_proprio_grounding from the FT YAML.")
        warm_start = _mapping(config.get("warm_start"), "warm_start", {"run_name", "checkpoint"})
        pt_run = _component(warm_start.get("run_name", ""), "warm_start.run_name")
        step = checkpoint_step(warm_start.get("checkpoint", "last"))
        pt_output = outputs / "molmoact2_PT" / pt_run
        if config.get("_initial_checkpoint"):
            initial = Path(config["_initial_checkpoint"])
        elif step == "last":
            candidates = [(int(p.name[4:]), p) for p in pt_output.glob("step*")
                          if re.fullmatch(r"step\d+", p.name) and is_training_checkpoint(p)]
            initial = max(candidates, key=lambda item: item[0])[1] if candidates else pt_output / "last"
        else:
            initial = pt_output / step
        # Resolve 'last' into a concrete step so a later PT save cannot change an FT run's identity.
        run_name = f"{pt_run}_{initial.name}_{name}_ft_bs{batch}"
        grounding = checkpoint_grounding(initial)
    if suffix:
        run_name += f"_{suffix}"
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
        "initial_checkpoint": str(initial), "run_name": run_name,
        "output_dir": str(outputs / f"molmoact2_{stage.upper()}" / run_name),
        "batch_size": batch, "micro_batch_size": micro, "workers": workers, "gpus": gpus,
        "tuning": tuning, "checkpointing": checkpointing, "action_format": action_format,
        "num_flow_timesteps": num_flow, "optimizer": lrs, "steps": steps, "warmup_steps": warmup,
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
    return ((path / "config.yaml").is_file() and (path / "model_and_optim/.metadata").is_file()
            and (path / "train/rank0.pt").is_file())


def validate_inputs(settings: dict[str, Any], model: bool = True) -> None:
    for item in (settings["python"], str(Path(settings["source"]) / "launch_scripts/train_lerobot.py")):
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
    if (checkpoint / "config.json").is_file():
        index = checkpoint / "model.safetensors.index.json"
        shards = (set(json.loads(index.read_text())["weight_map"].values()) if index.is_file()
                  else {p.name for p in checkpoint.glob("*.safetensors")})
        missing = [name for name in sorted(shards) if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0]
        if not shards or missing:
            raise FileNotFoundError(f"Model download is incomplete at {checkpoint}; missing shards: {missing or 'all'}")
    elif not is_training_checkpoint(checkpoint):
        raise FileNotFoundError(f"Complete MolmoAct2 checkpoint not found: {checkpoint}")
    output = Path(settings["output_dir"])
    native_steps = [p for p in output.glob("step*") if re.fullmatch(r"step\d+", p.name)]
    if native_steps:
        latest = max(native_steps, key=lambda p: int(p.name[4:]))
        if not is_training_checkpoint(latest):
            raise ValueError(f"Latest checkpoint is incomplete: {latest}. Resolve this before resuming.")
        if checkpoint_grounding(latest) != settings["proprio_grounding"]:
            raise ValueError(f"Cannot change proprio grounding while resuming {latest}; use another run.suffix.")
    elif (output / "config.yaml").exists():
        raise ValueError(f"{output} has a trainer config but no resumable checkpoint. Use a new run.suffix.")


def trainer_args(s: dict[str, Any]) -> list[str]:
    full = s["tuning"] == "full"
    values: dict[str, Any] = {
        "frame_loading_backend": "av", "packing": False, "dynamic_seq_len": True,
        "device_batch_size": s["micro_batch_size"], "global_batch_size": s["batch_size"],
        "num_workers": s["workers"], "pin_memory": s["workers"] > 0, "prefetch_factor": 2,
        "data.persistent_workers": s["workers"] > 0, "data.timeout": 900 if s["workers"] else 0,
        "max_duration": s["steps"], "stop_at": s["steps"], "save_folder": s["output_dir"],
        "run_name": s["run_name"], "log_interval": s["log_every"],
        "save_interval": s["save_every"], "save_num_checkpoints_to_keep": s["keep_checkpoints"],
        "save_final_optim": True, "save_final_unsharded_checkpoint": True, "allow_resume": True,
        "activation_checkpointing": s["checkpointing"], "compile_loss": False,
        "ft_vlm": full, "ft_action_expert": True, "ft_embedding": "lm_head" if full else "none",
        "action_expert_detach_vlm": False, "lora_enable": False,
        "action_format": s["action_format"], "state_format": "discrete", "norm_mode": "q01_q99",
        "max_action_dim": 32, "max_action_horizon": s["horizon"], "n_obs_steps": 1,
        "num_flow_timesteps": s["num_flow_timesteps"], "add_depth_tokens": False,
        "enable_depth_reasoning": False, "img_aug": "full", "random_camera_order": "none",
    }
    if s["action_format"] == "both":
        values["discrete_action_tokenizer"] = "allenai/MolmoAct2-FAST-Tokenizer"
    for key, value in s["optimizer"].items():
        values[key.replace("_lr", "_learning_rate")] = value
    for component in ("llm", "vit", "connector", "action_expert"):
        # Molmo evaluates LR at optimizer step 1, pi05's LambdaLR at step 0.
        # W+1 and a zero starting LR reproduce pi05's (step+1)/(W+1) exactly.
        values[f"scheduler.{component}_t_warmup"] = s["warmup_steps"] + (s["lr_mode"] == "warmup_constant")
    values["scheduler.warmup_min_lr"] = 0.0
    # The original cosine scheduler becomes constant after warmup when eta_min == peak LR.
    values["scheduler.alpha_f"] = 1.0 if s["lr_mode"] == "warmup_constant" else 0.1
    if s["wandb_enable"]:
        values.update({"wandb.project": s["wandb_project"], "wandb.entity": s["wandb_entity"] or None,
                       "wandb.name": s["run_name"]})
    else:
        values["wandb"] = None
    def scalar(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return str(value).lower()
        # OmegaConf dotlist parses strings too; JSON quotes keep numeric names as strings.
        return json.dumps(value) if isinstance(value, str) else str(value)
    # argparse-handled arguments must not contain JSON quotes. Unknown dotlist arguments need them.
    argparse_keys = {"frame_loading_backend", "packing", "dynamic_seq_len", "device_batch_size", "global_batch_size",
                     "num_workers", "pin_memory", "prefetch_factor", "log_interval", "ft_vlm", "ft_action_expert",
                     "ft_embedding", "action_expert_detach_vlm", "lora_enable", "action_format", "state_format",
                     "norm_mode", "max_action_dim", "max_action_horizon", "n_obs_steps", "num_flow_timesteps",
                     "add_depth_tokens", "enable_depth_reasoning", "img_aug", "random_camera_order",
                     "discrete_action_tokenizer", "llm_learning_rate", "vit_learning_rate",
                     "connector_learning_rate", "action_expert_learning_rate"}
    return [s["initial_checkpoint"], "sbd_libero"] + [
        f"--{key}={str(value).lower() if isinstance(value, bool) else value}" if key in argparse_keys
        else f"--{key}={scalar(value)}" for key, value in values.items()
    ]


def command(s: dict[str, Any]) -> list[str]:
    argv = [s["python"], "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc-per-node={s['gpus']}", str(Path(__file__).with_name("train.py")),
            "--config", s["config_path"]]
    return argv + (["--smoke"] if s["smoke"] else [])


def shell_exports(s: dict[str, Any]) -> str:
    exports = {"PROJECT_ROOT": s["project_root"], "MOLMOACT2_PYTHON": s["python"],
               "MOLMOACT2_OUTPUT_DIR": s["output_dir"], "MOLMOACT2_DATASET_DIR": s["dataset_dir"],
               "MOLMOACT2_RUN_NAME": s["run_name"], "MOLMOACT2_NUM_GPUS": s["gpus"],
               "MOLMOACT2_STAGE": s["stage"], "MOLMOACT2_PARTITION": s["partition"],
               "MOLMOACT2_QOS": s["qos"], "MOLMOACT2_NODELIST": s["nodelist"],
               "MOLMOACT2_EXCLUDE": s["exclude"], "MOLMOACT2_GRES": s["gres"],
               "MOLMOACT2_CPUS": s["cpus"], "MOLMOACT2_MEM": s["memory"], "MOLMOACT2_TIME": s["time"]}
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
        frozen = args.config.resolve().parent / "molmoact2_resolved.json"
        with frozen.open("x") as stream:
            json.dump({"initial_checkpoint": settings["initial_checkpoint"]}, stream)
    if args.shell:
        print(shell_exports(settings))
    else:
        print(json.dumps(settings, indent=2))
        print("\nLauncher:\n" + shlex.join(command(settings)))
        print("\nOriginal trainer arguments:\n" + shlex.join(trainer_args(settings)))


if __name__ == "__main__":
    main()
