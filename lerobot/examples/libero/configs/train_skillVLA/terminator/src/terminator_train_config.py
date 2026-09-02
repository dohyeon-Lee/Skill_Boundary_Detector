#!/usr/bin/env python3
"""Resolve the unified PT/FT predictor + FSQ terminator config."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import (  # noqa: E402
    as_bool,
    as_list,
    load_config,
    print_shell,
    resolve_path,
    resolve_skillvla_dataset_run,
)

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "auxiliary_train_config.yaml"


def _at(config: dict, *path: str, default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _local_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return Path(resolve_path(project_root, path))
    if path.exists() or "models" not in path.parts:
        return path
    return project_root.joinpath(*path.parts[path.parts.index("models") :])


def _dataset_contract(dataset_dir: Path, run_tag: str) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Auxiliary dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    levels = [int(level) for level in info.get("skill_fsq_levels", [])]
    if not levels or any(level <= 1 for level in levels):
        raise ValueError(f"Invalid skill_fsq_levels in {info_path}: {levels}")
    match = re.search(r"FSQ(\d+)", run_tag)
    if match and [int(digit) for digit in match.group(1)] != levels:
        raise ValueError(
            f"Dataset run says FSQ{match.group(1)}, but metadata says levels={levels}."
        )
    features = info.get("features", {})
    return {
        "levels": levels,
        "skill_code_space_id": str(
            info.get("skill_code_space_id", run_tag) or run_tag
        ).strip(),
        "state_dim": int(features["observation.state"]["shape"][0]),
        "action_dim": int(features["action"]["shape"][0]),
    }


def _predictor_contract(config: dict) -> dict:
    lora_enabled = as_bool(_at(config, "skill_predictor", "lora", "enabled", default=True))
    return {
        "skill_predictor_vlm_variant": "gemma_2b",
        "skill_predictor_image_size": 224,
        "skill_predictor_reader_tokens": int(
            _at(config, "skill_predictor", "reader", "tokens", default=4)
        ),
        "skill_predictor_reader_depth": int(
            _at(config, "skill_predictor", "reader", "depth", default=2)
        ),
        "skill_predictor_reader_heads": int(
            _at(config, "skill_predictor", "reader", "heads", default=8)
        ),
        "skill_predictor_all_layers": as_bool(
            _at(config, "skill_predictor", "all_layers", default=True)
        ),
        "skill_predictor_detach_vlm": not lora_enabled,
        "skill_predictor_lora": lora_enabled,
        "skill_predictor_lora_targets": str(
            _at(config, "skill_predictor", "lora", "targets", default="q,k,v,o")
        ),
        "skill_predictor_lora_rank": int(
            _at(config, "skill_predictor", "lora", "rank", default=8)
        ),
        "skill_predictor_lora_alpha": float(
            _at(config, "skill_predictor", "lora", "alpha", default=16.0)
        ),
        "skill_predictor_lora_dropout": float(
            _at(config, "skill_predictor", "lora", "dropout", default=0.0)
        ),
        "skill_predictor_deadzone_frac": float(
            _at(config, "skill_predictor", "reader", "deadzone_frac", default=0.8)
        ),
        "skill_predictor_attend_image": as_bool(
            _at(config, "skill_predictor", "token_access", "image", default=True)
        ),
        "skill_predictor_attend_language": as_bool(
            _at(config, "skill_predictor", "token_access", "language", default=True)
        ),
        "tokenizer_max_length": 200,
    }


def _terminator_contract(config: dict) -> dict:
    raw = config.get("fsq_terminator", {})
    if not isinstance(raw, dict):
        raise ValueError("fsq_terminator must be an inline mapping.")
    allowed = {
        "termination",
        "context",
        "default_arch",
        "vision_backbone",
        "freeze_vision_encoder",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unsupported fsq_terminator keys: {unknown}")
    contract = {
        "train_terminator": as_bool(raw.get("termination", False)),
        "terminator_context": str(raw.get("context", "prev_action")).strip().lower(),
        "terminator_arch": str(raw.get("default_arch", "fusion")).strip().lower(),
        "terminator_vision_backbone": str(
            raw.get("vision_backbone", "resnet")
        ).strip().lower(),
        "terminator_freeze_vision_encoder": as_bool(
            raw.get("freeze_vision_encoder", True)
        ),
        # The simplified FSQ contract trains termination only; progress is gone.
        "terminator_termination_only": True,
    }
    if contract["terminator_context"] not in {"prev_action", "proprio"}:
        raise ValueError("fsq_terminator.context must be prev_action or proprio.")
    if contract["terminator_arch"] not in {"small", "fusion"}:
        raise ValueError("fsq_terminator.default_arch must be small or fusion.")
    if contract["terminator_vision_backbone"] not in {"dino", "siglip", "resnet"}:
        raise ValueError(
            "fsq_terminator.vision_backbone must be dino, siglip, or resnet."
        )
    return contract


def _load_auxiliary_checkpoint(checkpoint: Path, component: str) -> dict:
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Incomplete auxiliary checkpoint: {checkpoint}")
    source = json.loads(config_path.read_text())
    if source.get("type") != "skill_aux":
        raise ValueError(
            f"FT {component}_checkpoint must be a skill_aux PT checkpoint, "
            f"got {source.get('type')!r}."
        )
    flag = {
        "predictor": "train_skill_predictor",
        "terminator": "train_terminator",
    }[component]
    if not source.get(flag, False):
        raise ValueError(f"FT {component} checkpoint has no trained {component}.")
    return source


def _checkpoint_code_space_id(source: dict, checkpoint: Path) -> str:
    explicit = str(source.get("skill_code_space_id", "") or "").strip()
    if explicit:
        return explicit
    fsq_path = str(source.get("fsq_path", "") or "").strip()
    if fsq_path:
        # Backward-compatible identity for checkpoints saved before the explicit
        # field existed. SkillVLA datasets use the FSQ run name as their run dir.
        return Path(fsq_path).parent.name
    raise ValueError(
        f"FT checkpoint does not identify its FSQ code space: {checkpoint}"
    )


def _checkpoint_training_lineage(
    source: dict, checkpoint: Path
) -> tuple[int, list[str], list[str]]:
    batch_size = int(source.get("training_batch_size", 0) or 0)
    raw_lineage = source.get("dataset_source_lineage", [])
    lineage = (
        [str(value).strip() for value in raw_lineage if str(value).strip()]
        if isinstance(raw_lineage, list)
        else []
    )
    raw_suffixes = source.get("run_suffix_lineage", [])
    suffixes = (
        [str(value).strip() for value in raw_suffixes if str(value).strip()]
        if isinstance(raw_suffixes, list)
        else []
    )
    if batch_size <= 0 or not lineage:
        raise ValueError(
            "FT checkpoint is missing training_batch_size or "
            f"dataset_source_lineage: {checkpoint}. Recreate it with the unified PT trainer."
        )
    return batch_size, lineage, suffixes


def _merge_lineages(*lineages: list[str]) -> list[str]:
    merged = []
    for lineage in lineages:
        for source in lineage:
            if source not in merged:
                merged.append(source)
    return merged


def _validate_checkpoint_code_space(
    source: dict,
    checkpoint: Path,
    *,
    levels: list[int],
    code_space_id: str,
) -> None:
    expected = {
        "skill_fsq_levels": levels,
        "skill_vocab_size": math.prod(levels),
    }
    mismatches = [
        f"{key}: checkpoint={source.get(key)!r}, dataset={value!r}"
        for key, value in expected.items()
        if source.get(key) != value
    ]
    source_id = _checkpoint_code_space_id(source, checkpoint)
    if source_id != code_space_id:
        mismatches.append(
            f"skill_code_space_id: checkpoint={source_id!r}, dataset={code_space_id!r}"
        )
    if mismatches:
        raise ValueError(
            "Auxiliary checkpoint code-space mismatch: " + "; ".join(mismatches)
        )


def _checkpoint_predictor_contract(source: dict, checkpoint: Path) -> dict:
    contract = _predictor_contract({})
    missing = [key for key in contract if key not in source]
    if missing:
        raise ValueError(
            f"FT predictor checkpoint is missing contract fields {missing}: {checkpoint}"
        )
    return {key: source[key] for key in contract}


def _checkpoint_terminator_contract(source: dict, checkpoint: Path) -> dict:
    source_fields = {
        "terminator_context": "terminator_context",
        "terminator_arch": "terminator_arch",
        "terminator_vision_backbone": "terminator_vision_backbone",
        "terminator_freeze_vision_encoder": "terminator_freeze_vision_encoder",
        "terminator_termination_only": "terminator_termination_only",
    }
    missing = [source_key for source_key in source_fields.values() if source_key not in source]
    if missing:
        raise ValueError(
            f"FT terminator checkpoint is missing contract fields {missing}: {checkpoint}"
        )
    return {
        "train_terminator": True,
        **{
            target_key: source[source_key]
            for target_key, source_key in source_fields.items()
        },
    }


def build_settings(config: dict) -> dict:
    removed = {
        "terminator",
        "image_only_terminator",
        "wrist_only_terminator",
        "state_only_terminator",
        "state_rnn_terminator",
    } & set(config)
    if removed:
        raise ValueError(
            "Legacy terminator sections were removed; use fsq_terminator only: "
            f"{sorted(removed)}"
        )

    initialization_mode = str(config.get("mode", "pt")).strip().lower()
    if initialization_mode not in {"pt", "ft"}:
        raise ValueError("mode must be pt or ft.")

    project_root = Path(str(config["project_root"])).expanduser()
    dataset_root = project_root / str(config.get("dataset_root", "dataset"))
    outputs_root = project_root / str(config.get("outputs_root", "outputs"))
    source = str(_at(config, "dataset", "source"))
    base_run_tag = str(_at(config, "dataset", "run"))
    # Predictor targets must remain the canonical labels.  This trainer has one
    # shared DataLoader, so a predictor-only or predictor+terminator job ignores
    # dataset.relabeled as a whole.  Terminator-only jobs may consume relabeled
    # skill codes safely.
    predictor_requested = (
        as_bool(_at(config, "skill_predictor", "train", default=False))
        if initialization_mode == "pt"
        else bool(
            str(
                _at(config, "warm_start", "predictor_checkpoint", default="")
                or ""
            ).strip()
        )
    )
    requested_relabel = _at(config, "dataset", "relabeled", default="")
    selected_relabel = "" if predictor_requested else requested_relabel
    skillvla_root = (
        dataset_root
        / str(_at(config, "dataset", "skillvla_root", default="skillvla_dataset"))
    )
    run_tag, dataset_relabeled = resolve_skillvla_dataset_run(
        skillvla_root / source,
        base_run_tag,
        selected_relabel,
    )
    dataset_dir = (
        skillvla_root
        / source
        / run_tag
        / "skillvla"
    )
    dataset = _dataset_contract(dataset_dir, run_tag)
    pi_base = _local_path(
        project_root, str(_at(config, "warm_start", "pi_base", default="models/pi05_base"))
    )
    tokenizer = _local_path(
        project_root,
        str(
            _at(
                config,
                "warm_start",
                "tokenizer",
                default="models/paligemma-3b-pt-224-tokenizer",
            )
        ),
    )
    fsq_value = str(_at(config, "warm_start", "fsq", default="") or "").strip()
    if fsq_value:
        raise ValueError(
            "warm_start.fsq override was removed; the FSQ checkpoint is always "
            "the FSQ.pt beside dataset.source/run."
        )
    fsq_path = dataset_dir.parent / "FSQ.pt"
    legacy_auxiliary = str(
        _at(config, "warm_start", "auxiliary_checkpoint", default="") or ""
    ).strip()
    if legacy_auxiliary:
        raise ValueError(
            "warm_start.auxiliary_checkpoint was split into predictor_checkpoint "
            "and terminator_checkpoint."
        )
    predictor_value = str(
        _at(config, "warm_start", "predictor_checkpoint", default="") or ""
    ).strip()
    terminator_value = str(
        _at(config, "warm_start", "terminator_checkpoint", default="") or ""
    ).strip()
    predictor_checkpoint = (
        _local_path(project_root, predictor_value) if predictor_value else None
    )
    terminator_checkpoint = (
        _local_path(project_root, terminator_value) if terminator_value else None
    )
    requested_batch_size = int(
        _at(config, "training", "dataloader", "batch_size", default=16)
    )
    requested_suffix = str(
        _at(config, "run", "suffix", default="") or ""
    ).strip().strip("_")
    if requested_suffix and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", requested_suffix
    ) is None:
        raise ValueError("run.suffix contains unsupported characters.")

    if initialization_mode == "pt":
        if predictor_checkpoint is not None or terminator_checkpoint is not None:
            raise ValueError(
                "mode=pt must leave warm_start predictor/terminator checkpoints empty."
            )
        predictor_contract = _predictor_contract(config)
        terminator_contract = _terminator_contract(config)
        train_predictor = as_bool(
            _at(config, "skill_predictor", "train", default=False)
        )
        train_terminator = terminator_contract["train_terminator"]
        batch_size = requested_batch_size
        dataset_source_lineage = [source]
        run_suffix_lineage = [requested_suffix] if requested_suffix else []
        if not (train_predictor or train_terminator):
            raise ValueError(
                "Enable fsq_terminator.termination and/or skill_predictor.train."
            )
        if train_terminator and not fsq_path.is_file():
            raise FileNotFoundError(f"FSQ checkpoint not found: {fsq_path}")
        if train_predictor and not (pi_base / "model.safetensors").is_file():
            raise FileNotFoundError(f"pi0.5 predictor base not found: {pi_base}")
        termination_sigma = float(
            _at(config, "termination_loss", "target_sigma", default=2.0)
        )
        termination_positive_weight = float(
            _at(config, "termination_loss", "positive_weight", default=1.0)
        )
    else:
        # FT targets are inferred exclusively from the component checkpoint
        # paths. The PT-only model sections in this YAML are intentionally ignored.
        train_predictor = predictor_checkpoint is not None
        train_terminator = terminator_checkpoint is not None
        if not (train_predictor or train_terminator):
            raise ValueError(
                "mode=ft requires warm_start.predictor_checkpoint and/or "
                "warm_start.terminator_checkpoint."
            )
        predictor_contract = _predictor_contract({})
        terminator_contract = _terminator_contract({})
        predictor_source = None
        terminator_source = None
        checkpoint_batches = []
        checkpoint_lineages = []
        checkpoint_suffixes = []
        if predictor_checkpoint is not None:
            predictor_source = _load_auxiliary_checkpoint(
                predictor_checkpoint, "predictor"
            )
            _validate_checkpoint_code_space(
                predictor_source,
                predictor_checkpoint,
                levels=dataset["levels"],
                code_space_id=dataset["skill_code_space_id"],
            )
            predictor_contract = _checkpoint_predictor_contract(
                predictor_source, predictor_checkpoint
            )
            (
                predictor_batch,
                predictor_lineage,
                predictor_suffixes,
            ) = _checkpoint_training_lineage(predictor_source, predictor_checkpoint)
            checkpoint_batches.append(predictor_batch)
            checkpoint_lineages.append(predictor_lineage)
            checkpoint_suffixes.append(predictor_suffixes)
            tokenizer_value = str(
                predictor_source.get("tokenizer_path", "") or ""
            ).strip()
            if not tokenizer_value:
                raise ValueError(
                    f"FT predictor checkpoint has no tokenizer_path: {predictor_checkpoint}"
                )
            tokenizer = _local_path(project_root, tokenizer_value)
        if terminator_checkpoint is not None:
            terminator_source = _load_auxiliary_checkpoint(
                terminator_checkpoint, "terminator"
            )
            _validate_checkpoint_code_space(
                terminator_source,
                terminator_checkpoint,
                levels=dataset["levels"],
                code_space_id=dataset["skill_code_space_id"],
            )
            terminator_contract = _checkpoint_terminator_contract(
                terminator_source, terminator_checkpoint
            )
            (
                terminator_batch,
                terminator_lineage,
                terminator_suffixes,
            ) = _checkpoint_training_lineage(terminator_source, terminator_checkpoint)
            checkpoint_batches.append(terminator_batch)
            checkpoint_lineages.append(terminator_lineage)
            checkpoint_suffixes.append(terminator_suffixes)
        if predictor_source is not None and terminator_source is not None:
            predictor_space = _checkpoint_code_space_id(
                predictor_source, predictor_checkpoint
            )
            terminator_space = _checkpoint_code_space_id(
                terminator_source, terminator_checkpoint
            )
            if predictor_space != terminator_space:
                raise ValueError(
                    "FT predictor and terminator checkpoints use different FSQ "
                    f"code spaces: {predictor_space!r} != {terminator_space!r}."
                )
        if len(set(checkpoint_batches)) != 1:
            raise ValueError(
                "FT predictor and terminator checkpoints must have the same PT "
                f"batch size, got {checkpoint_batches}."
            )
        batch_size = checkpoint_batches[0]
        dataset_source_lineage = _merge_lineages(*checkpoint_lineages)
        # Unlike component-source merging, the current FT dataset is always
        # appended so repeated fine-tuning remains visible in the lineage.
        dataset_source_lineage.append(source)
        run_suffix_lineage = _merge_lineages(*checkpoint_suffixes)
        if requested_suffix and requested_suffix not in run_suffix_lineage:
            run_suffix_lineage.append(requested_suffix)
        if train_terminator and not fsq_path.is_file():
            raise FileNotFoundError(f"FSQ checkpoint not found: {fsq_path}")
        termination_sigma = float(
            terminator_source.get("terminator_end_target_sigma", 2.0)
            if terminator_source is not None
            else 2.0
        )
        termination_positive_weight = float(
            terminator_source.get("terminator_end_pos_weight", 1.0)
            if terminator_source is not None
            else 1.0
        )

    if train_predictor:
        required_tokenizer = ("config.json", "tokenizer_config.json", "tokenizer.json")
        missing = [name for name in required_tokenizer if not (tokenizer / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Tokenizer is incomplete at {tokenizer}: missing={missing}")

    num_gpus = int(_at(config, "training", "dataloader", "gpus", default=1))
    steps = int(_at(config, "training", "schedule", "steps", default=100000))
    warmup_steps = int(
        _at(config, "training", "schedule", "warmup_steps", default=1000)
    )
    decay_steps = int(
        _at(config, "training", "schedule", "lr_decay_steps", default=30000)
    )
    scheduler_mode = str(
        _at(config, "training", "schedule", "lr_mode", default="warmup_constant")
    ).strip().lower()
    if min(batch_size, num_gpus, steps) <= 0:
        raise ValueError("Batch size, GPU count, and steps must be positive.")
    if scheduler_mode not in {"warmup_constant", "cosine_decay"}:
        raise ValueError("training.schedule.lr_mode must be warmup_constant or cosine_decay.")
    if warmup_steps < 0 or decay_steps <= 0:
        raise ValueError("Invalid scheduler step counts.")

    target_names = []
    if train_predictor:
        target_names.append("predictor")
    if train_terminator:
        target_names.append("terminator")
    target_mode = "_".join(target_names)
    lineage_name = "_".join(dataset_source_lineage)
    run_name = f"bs{batch_size}_{run_tag}_{lineage_name}_{target_mode}"
    if run_suffix_lineage:
        run_name += "_" + "_".join(run_suffix_lineage)

    base_lr = float(_at(config, "training", "optimizer", "base_lr", default=2.5e-5))
    settings = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "skillvla_dataset_dir": dataset_dir,
        "dataset_relabeled": dataset_relabeled,
        "dataset_relabel_ignored_for_predictor": bool(
            predictor_requested and requested_relabel not in (None, "", False)
        ),
        "repo_id": f"dohyeon/{source}",
        "initialization_mode": initialization_mode,
        "training_mode": target_mode,
        "pi_base": pi_base,
        "tokenizer_path": tokenizer,
        "fsq_path": fsq_path,
        "predictor_checkpoint_path": predictor_checkpoint or "",
        "terminator_checkpoint_path": terminator_checkpoint or "",
        # Retained empty for old saved SkillAux configs; new jobs use the two
        # component-specific paths above.
        "auxiliary_checkpoint_path": "",
        "train_terminator": train_terminator,
        "train_skill_predictor": train_predictor,
        "skill_code_space_id": dataset["skill_code_space_id"],
        "training_batch_size": batch_size,
        "dataset_source_lineage": json.dumps(dataset_source_lineage),
        "run_suffix_lineage": json.dumps(run_suffix_lineage),
        "skill_fsq_levels": "[" + ",".join(str(level) for level in dataset["levels"]) + "]",
        "skill_vocab_size": math.prod(dataset["levels"]),
        "max_state_dim": dataset["state_dim"],
        "max_action_dim": dataset["action_dim"],
        **terminator_contract,
        "terminator_end_target_sigma": termination_sigma,
        "terminator_end_pos_weight": termination_positive_weight,
        **predictor_contract,
        "terminator_lr_scale": float(
            _at(config, "training", "optimizer", "terminator_lr_scale", default=1.0)
        ),
        "skill_predictor_lr_scale": float(
            _at(config, "training", "optimizer", "predictor_lr_scale", default=1.0)
        ),
        "skill_predictor_lora_lr_scale": float(
            _at(config, "training", "optimizer", "predictor_lora_lr_scale", default=10.0)
        ),
        "optimizer_grad_clip_norm": float(
            _at(config, "training", "optimizer", "grad_clip_norm", default=1.0)
        ),
        "gradient_checkpointing": as_bool(
            _at(config, "training", "gradient_checkpointing", default=False)
        ),
        "lr": base_lr * num_gpus,
        "batch_size": batch_size,
        "num_workers": int(_at(config, "training", "dataloader", "workers", default=4)),
        "num_gpus": num_gpus,
        "steps": steps,
        "scheduler_mode": scheduler_mode,
        "scheduler_warmup_steps": warmup_steps,
        "scheduler_decay_steps": decay_steps,
        "log_freq": int(_at(config, "training", "schedule", "log_every", default=100)),
        "save_freq": int(_at(config, "training", "schedule", "save_every", default=5000)),
        "run_name": run_name,
        "output_dir": outputs_root / "skillVLA_terminator" / run_name,
        "wandb_enable": as_bool(_at(config, "logging", "wandb", "enable", default=True)),
        "wandb_project": str(
            _at(config, "logging", "wandb", "project", default="VLA_auxiliary")
        ),
        "train_partition": ",".join(as_list(config.get("train_partition", ["big"]))) or "big",
        "train_qos": str(config.get("train_qos", "big_qos")),
        "train_gres": str(_at(config, "slurm", "gres", default="gpu:1")),
        "train_cpus_per_task": int(_at(config, "slurm", "cpus", default=10)),
        "train_mem": str(_at(config, "slurm", "memory", default="128G")),
        "train_time": str(_at(config, "slurm", "time", default="48:00:00")),
        "train_nodelist": str(config.get("train_nodelist", "")),
        "train_exclude_nodes": ",".join(as_list(config.get("train_exclude_nodes", []))),
    }
    positive = (
        "terminator_lr_scale",
        "skill_predictor_lr_scale",
        "skill_predictor_lora_lr_scale",
        "terminator_end_pos_weight",
    )
    invalid = [key for key in positive if settings[key] <= 0]
    if invalid:
        raise ValueError(f"Auxiliary settings must be positive: {invalid}.")
    if settings["terminator_end_target_sigma"] < 0:
        raise ValueError("termination_loss.target_sigma must be non-negative.")
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
