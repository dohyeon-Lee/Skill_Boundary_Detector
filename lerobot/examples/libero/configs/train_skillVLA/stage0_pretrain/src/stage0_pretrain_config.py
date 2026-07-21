#!/usr/bin/env python3
"""Resolve Stage0-pretrain YAML on top of the direct Stage-0 motor contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent / "stage0" / "src"))
from stage0_train_config import build_settings as build_stage0_settings  # noqa: E402

sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage0_pretrain_config.yaml"


def _at(cfg: dict, *path: str, default=None):
    value = cfg
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _local_artifact(recorded: str, local_dir: Path) -> Path:
    path = Path(str(recorded))
    if path.is_file():
        return path
    candidate = local_dir / path.name
    return candidate


def build_settings(cfg: dict) -> dict:
    # The shared resolver owns FSQ geometry, A/B matrices, optimizer, and Slurm behavior.
    compatible = dict(cfg)
    compatible["loss"] = {
        "A": dict(_at(cfg, "loss", "flow", "A", default={})),
        "B": dict(_at(cfg, "loss", "flow", "B", default={})),
    }
    settings = build_stage0_settings(compatible)
    project_root = Path(str(settings["project_root"]))
    outputs_root = project_root / str(cfg.get("outputs_root", "outputs"))
    dataset_run_dir = Path(settings["fsq_ckpt"]).parent

    pretrain_run = str(_at(cfg, "warm_start", "pretrain", "run", default="") or "").strip()
    pretrain_checkpoint = str(
        _at(cfg, "warm_start", "pretrain", "checkpoint", default="last") or "last"
    ).strip()
    if not pretrain_run:
        raise ValueError("warm_start.pretrain.run must name a skillVLA_pretrain output folder.")
    pretrain_path = (
        outputs_root / "skillVLA_pretrain" / pretrain_run / "checkpoints"
        / pretrain_checkpoint / "pretrained_model"
    )
    config_json = pretrain_path / "config.json"
    if not config_json.is_file():
        raise FileNotFoundError(f"Pretrained VLM config not found: {config_json}")
    pretrain = json.loads(config_json.read_text())
    if pretrain.get("model_type") != "skill_vla_pretrain":
        raise ValueError(
            f"Expected a skill_vla_pretrain checkpoint, got {pretrain.get('model_type')!r}."
        )
    source_levels = [int(level) for level in pretrain.get("skill_fsq_levels", [])]
    resolved_levels = json.loads(str(settings["skill_fsq_levels"]))
    if source_levels != resolved_levels:
        raise ValueError(
            f"Pretrain/Stage0 FSQ levels differ: {source_levels} != {resolved_levels}."
        )
    source_tokenizer = Path(str(pretrain.get("text_tokenizer_name") or ""))
    if source_tokenizer.name and source_tokenizer.name != Path(settings["tokenizer_path"]).name:
        raise ValueError(
            "Pretrain and Stage0 text tokenizer directories differ: "
            f"{source_tokenizer.name} != {Path(settings['tokenizer_path']).name}."
        )

    transition_pack = _local_artifact(
        str(pretrain.get("transition_packs") or "transitions.npz"), dataset_run_dir
    )
    target_pack = _local_artifact(
        str(pretrain.get("pretrain_target_packs") or ""), dataset_run_dir
    )
    for name, path in (("transition pack", transition_pack), ("FAST target pack", target_pack)):
        if not path.is_file():
            raise FileNotFoundError(f"Stage0-pretrain {name} not found: {path}")

    ar_skill_weight = float(
        _at(cfg, "loss", "autoregressive", "skill_weight", default=1.0)
    )
    if ar_skill_weight <= 0.0:
        raise ValueError("loss.autoregressive.skill_weight must be > 0.")
    settings.update(
        {
            "pretrain_checkpoint_path": pretrain_path,
            "pretrain_training_mode": str(pretrain.get("training_mode", "full")),
            "pretrain_lora_targets": str(pretrain.get("pretrain_lora_targets", "q,k,v,o")),
            "pretrain_lora_rank": int(pretrain.get("pretrain_lora_rank", 16)),
            "pretrain_lora_alpha": float(pretrain.get("pretrain_lora_alpha", 32.0)),
            "pretrain_lora_dropout": float(pretrain.get("pretrain_lora_dropout", 0.0)),
            "skill_unused_start": int(pretrain.get("skill_unused_start", 0)),
            "fast_vocab_size": int(pretrain.get("fast_vocab_size", 1024)),
            "fast_skip_tokens": int(pretrain.get("fast_skip_tokens", 128)),
            "max_action_tokens": int(pretrain.get("max_action_tokens", 384)),
            "transition_pack": transition_pack,
            "pretrain_target_pack": target_pack,
            "transition_randomization": as_bool(
                _at(cfg, "dataset", "transition_randomization", default=True)
            ),
            "attend_skill": as_bool(_at(cfg, "token_access", "skill", default=True)),
            "attend_fast": as_bool(_at(cfg, "token_access", "fast", default=False)),
            "ar_fast_loss": as_bool(
                _at(cfg, "loss", "autoregressive", "fast", default=False)
            ),
            "ar_batch_size": int(
                _at(cfg, "loss", "autoregressive", "batch_size", default=2)
            ),
            "ar_skill_loss_weight": ar_skill_weight,
            "ar_fast_loss_weight": float(
                _at(cfg, "loss", "autoregressive", "fast_weight", default=1.0)
            ),
            "ar_structure_loss_weight": float(
                _at(cfg, "loss", "autoregressive", "structure_weight", default=0.1)
            ),
            "warmup_steps": int(_at(cfg, "training", "schedule", "warmup", default=1000)),
            "log_freq": int(_at(cfg, "training", "schedule", "log_every", default=100)),
            "dataloader_timeout_s": float(
                _at(cfg, "training", "dataloader", "timeout_s", default=300)
            ),
            "pt_output_dir": outputs_root / "skillVLA_stage0_pretrain" / settings["pt_run_name"],
        }
    )
    if not 1 <= settings["ar_batch_size"] <= settings["batch_size"]:
        raise ValueError(
            "loss.autoregressive.batch_size must be between 1 and training batch_size."
        )
    if settings["dataloader_timeout_s"] < 0:
        raise ValueError("training.dataloader.timeout_s must be non-negative.")
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
