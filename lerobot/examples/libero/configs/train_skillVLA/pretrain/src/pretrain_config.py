#!/usr/bin/env python3
"""Resolve the compact SkillVLA FAST-pretraining YAML into shell exports."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, load_config, print_shell, resolve_path  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "pretrain_config.yaml"


def _at(cfg: dict, *path: str, default=None):
    value = cfg
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _local(project_root: Path, value) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else Path(resolve_path(project_root, path))


def build_settings(cfg: dict) -> dict:
    project_root = Path(str(cfg["project_root"])).expanduser()
    dataset_root = project_root / str(cfg.get("dataset_root", "dataset"))
    outputs_root = project_root / str(cfg.get("outputs_root", "outputs"))
    skillvla_root = dataset_root / str(_at(cfg, "dataset", "skillvla_root", default="skillvla_dataset"))
    source = str(_at(cfg, "dataset", "source"))
    dataset_run = str(_at(cfg, "dataset", "run"))
    run_dir = skillvla_root / source / dataset_run
    skillvla_dir = run_dir / "skillvla"
    fsq_path = run_dir / "FSQ.pt"
    if not skillvla_dir.is_dir():
        raise FileNotFoundError(f"SkillVLA pretrain dataset not found: {skillvla_dir}")
    if not fsq_path.is_file():
        raise FileNotFoundError(f"SkillVLA pretrain FSQ checkpoint not found: {fsq_path}")

    sys.path.insert(0, str(project_root / "lerobot" / "examples" / "libero"))
    from FSQ import load_fsq_action_expert_state  # noqa: PLC0415

    _, fsq_cfg = load_fsq_action_expert_state(fsq_path)
    levels = [int(level) for level in fsq_cfg.fsq_levels]
    match = re.search(r"FSQ(\d+)", dataset_run)
    if match and [int(value) for value in match.group(1)] != levels:
        raise ValueError(
            f"Dataset run says FSQ{match.group(1)}, but its FSQ.pt contains levels={levels}."
        )

    mode = str(_at(cfg, "model", "mode", default="lora")).strip().lower()
    if mode not in {"full", "lora"}:
        raise ValueError(f"model.mode must be full|lora, got {mode!r}.")
    rank = int(_at(cfg, "lora", "rank", default=16))
    alpha_raw = _at(cfg, "lora", "alpha", default="auto")
    alpha = float(2 * rank if str(alpha_raw).strip().lower() in {"", "auto", "none"} else alpha_raw)
    targets = ",".join(as_list(_at(cfg, "lora", "targets", default=["q", "k", "v", "o"])))
    full_lr = float(_at(cfg, "training", "optimizer", "full_lr", default=5e-6))
    lora_lr = float(_at(cfg, "training", "optimizer", "lora_lr", default=1e-4))
    lr = full_lr if mode == "full" else lora_lr

    fast_vocab = int(_at(cfg, "tokenizers", "fast_vocab_size", default=1024))
    fast_tokenizer = _local(project_root, _at(cfg, "tokenizers", "fast"))
    transition_pack = run_dir / "transitions.npz"
    pretrain_targets = run_dir / f"pretrain_targets_{fast_tokenizer.name}.npz"
    transition_randomization = as_bool(
        _at(cfg, "dataset", "transition_randomization", default=True)
    )
    max_fast_tokens = int(_at(cfg, "tokenizers", "max_fast_tokens", default=384))
    if max_fast_tokens <= 0:
        raise ValueError("tokenizers.max_fast_tokens must be positive.")
    tokenizer_dims = str(_at(cfg, "tokenizers", "train", "encoded_dims", default="0:7"))
    tokenizer_meta_path = fast_tokenizer / "metadata.json"
    if tokenizer_meta_path.is_file():
        tokenizer_meta = json.loads(tokenizer_meta_path.read_text())
        expected = {
            "vocab_size": fast_vocab,
            "encoded_dims": tokenizer_dims,
            "variable_horizon": True,
        }
        mismatches = {
            key: (tokenizer_meta.get(key), value)
            for key, value in expected.items()
            if tokenizer_meta.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"FAST tokenizer metadata does not match pretrain_config.yaml: {mismatches}. "
                "Use a different tokenizers.fast path or retrain it."
            )
    if pretrain_targets.is_file():
        import numpy as np  # noqa: PLC0415

        with np.load(pretrain_targets) as target_data:
            offsets = np.asarray(target_data["fast_token_offsets"], dtype=np.int64)
            target_max = int(np.diff(offsets).max(initial=0))
            target_vocab = int(target_data["vocab_size"])
            target_tokenizer = str(target_data["tokenizer_name"])
        if target_max > max_fast_tokens:
            raise ValueError(
                f"Pretrain targets need {target_max} FAST tokens, but max_fast_tokens="
                f"{max_fast_tokens}. Targets are never truncated."
            )
        if target_vocab != fast_vocab or target_tokenizer != fast_tokenizer.name:
            raise ValueError(
                "Pretrain target/tokenizer mismatch: "
                f"target vocab/name=({target_vocab},{target_tokenizer}), "
                f"configured=({fast_vocab},{fast_tokenizer.name})."
            )
    manual_name = str(_at(cfg, "run", "name", default="") or "").strip()
    suffix = str(_at(cfg, "run", "suffix", default="") or "").strip().strip("_")
    transition_tag = "rand" if transition_randomization else "gt"
    run_name = manual_name or f"{dataset_run}_pretrain_{mode}_{transition_tag}"
    if suffix:
        run_name = f"{run_name}_{suffix}"

    gpus = int(_at(cfg, "training", "dataloader", "gpus", default=1))
    # Cluster-wide scheduling follows global_config.yaml's canonical train_* keys. Resource size
    # remains module-local because it depends on this pretraining architecture.
    partition = ",".join(as_list(cfg.get("train_partition", ["debug"]))) or "debug"
    exclude = ",".join(as_list(cfg.get("train_exclude_nodes", [])))
    settings = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "source_dataset": source,
        "dataset_run": dataset_run,
        "skillvla_dataset_dir": skillvla_dir,
        "repo_id": f"dohyeon/{source}",
        "fsq_path": fsq_path,
        "pi_base": _local(project_root, _at(cfg, "warm_start", "pi_base", default="models/pi05_base")),
        "text_tokenizer": _local(
            project_root,
            _at(cfg, "tokenizers", "text", default="models/paligemma-3b-pt-224-tokenizer"),
        ),
        "fast_tokenizer": fast_tokenizer,
        "transition_pack": transition_pack,
        "pretrain_targets": pretrain_targets,
        "transition_randomization": transition_randomization,
        "fast_vocab_size": fast_vocab,
        "fast_skip_tokens": int(_at(cfg, "tokenizers", "fast_skip_tokens", default=128)),
        "max_fast_tokens": max_fast_tokens,
        "tokenizer_scale": float(_at(cfg, "tokenizers", "train", "scale", default=10.0)),
        "tokenizer_encoded_dims": tokenizer_dims,
        "training_mode": mode,
        "skill_fsq_levels": "[" + ",".join(str(level) for level in levels) + "]",
        "max_state_dim": int(fsq_cfg.max_state_dim),
        "max_action_dim": int(fsq_cfg.max_action_dim),
        "lora_targets": targets,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_dropout": float(_at(cfg, "lora", "dropout", default=0.0)),
        "skill_loss_weight": float(_at(cfg, "loss", "skill", default=1.0)),
        "fast_loss_weight": float(_at(cfg, "loss", "fast", default=1.0)),
        "structure_loss_weight": float(_at(cfg, "loss", "structure", default=0.1)),
        "batch_size": int(_at(cfg, "training", "dataloader", "batch_size", default=8)),
        "num_workers": int(_at(cfg, "training", "dataloader", "workers", default=4)),
        "num_gpus": gpus,
        "lr": lr * gpus,
        "steps": int(_at(cfg, "training", "schedule", "steps", default=30000)),
        "warmup_steps": int(_at(cfg, "training", "schedule", "warmup", default=1000)),
        "save_freq": int(_at(cfg, "training", "schedule", "save_every", default=5000)),
        "log_freq": int(_at(cfg, "training", "schedule", "log_every", default=100)),
        "pretrain_outputs_root": outputs_root / "skillVLA_pretrain",
        "pretrain_run_name": run_name,
        "pretrain_output_dir": outputs_root / "skillVLA_pretrain" / run_name,
        "wandb_enable": as_bool(_at(cfg, "logging", "wandb", "enable", default=True)),
        "wandb_project": str(_at(cfg, "logging", "wandb", "project", default="VLA_pretrain")),
        "train_partition": partition,
        "train_qos": str(cfg.get("train_qos", "base_qos")),
        "train_gres": str(_at(cfg, "slurm", "gres", default="gpu:1")),
        "train_cpus_per_task": int(_at(cfg, "slurm", "cpus", default=16)),
        "train_mem": str(_at(cfg, "slurm", "memory", default="192G")),
        "train_time": str(_at(cfg, "slurm", "time", default="48:00:00")),
        "train_nodelist": str(cfg.get("train_nodelist", "")),
        "train_exclude_nodes": exclude,
    }
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
