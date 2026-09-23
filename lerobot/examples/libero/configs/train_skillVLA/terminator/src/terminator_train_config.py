#!/usr/bin/env python3
"""Resolve the unified PT/FT predictor-or-terminator config."""

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
    load_stage1_component_config,
    print_shell,
    resolve_path,
    resolve_run_checkpoint,
    resolve_skillvla_dataset_run,
    stage1_run_dirs,
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
        "focus_uv_path": str(info.get("skill_focus_uv_path", "") or "").strip(),
    }


def _predictor_contract(config: dict, *, state_dim: int = 8) -> dict:
    freeze_vlm = as_bool(
        _at(config, "skill_predictor", "freeze_vlm", default=True)
    )
    requested_lora = as_bool(
        _at(config, "skill_predictor", "lora", "enabled", default=True)
    )
    # Full VLM co-training and LoRA are deliberately mutually exclusive.  The
    # YAML resolver makes this deterministic instead of requiring users to keep
    # two switches synchronized by hand.
    lora_enabled = bool(requested_lora and freeze_vlm)
    spatial_target = _at(config, "skill_predictor", "spatial_target", default=None)
    if spatial_target is None:
        spatial_target = (
            "uv" if as_bool(_at(config, "skill_predictor", "focus_uv", "enabled", default=False))
            else "off"
        )
    spatial_target = str(spatial_target).strip().lower()
    if spatial_target not in {"off", "uv", "xyz", "full_state"}:
        raise ValueError("skill_predictor.spatial_target must be off|uv|xyz|full_state.")
    spatial_weight = _at(config, "skill_predictor", "spatial_loss_weight", default=None)
    if spatial_weight is None:
        spatial_weight = _at(config, "skill_predictor", "focus_uv", "loss_weight", default=0.25)
    spatial_weight = float(spatial_weight)
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
        "skill_predictor_freeze_vlm": freeze_vlm,
        "skill_predictor_detach_vlm": bool(freeze_vlm and not lora_enabled),
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
        "skill_predictor_focus_uv_enabled": spatial_target == "uv",
        "skill_predictor_focus_uv_loss_weight": spatial_weight if spatial_target == "uv" else 0.25,
        "skill_predictor_end_state_mode": spatial_target if spatial_target in {"xyz", "full_state"} else "off",
        "skill_predictor_end_state_dim": int(state_dim),
        "skill_predictor_end_state_loss_weight": spatial_weight if spatial_target in {"xyz", "full_state"} else 1.0,
        "skill_predictor_sampling_mode": str(
            _at(config, "skill_predictor", "sampling", "mode", default="mode1")
        ).strip().lower(),
        "skill_predictor_boundary_fraction": float(
            _at(config, "skill_predictor", "sampling", "boundary_fraction", default=0.7)
        ),
        "skill_predictor_boundary_window": int(
            _at(config, "skill_predictor", "sampling", "boundary_window", default=10)
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
        "cameras",
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
        "terminator_cameras": str(raw.get("cameras", "both")).strip().lower(),
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
    if contract["terminator_context"] not in {"prev_action", "proprio", "none"}:
        raise ValueError("fsq_terminator.context must be prev_action, proprio, or none.")
    if contract["terminator_cameras"] not in {"both", "top", "wrist"}:
        raise ValueError("fsq_terminator.cameras must be both, top, or wrist.")
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


def _fsq_source_identity(run_dir: Path) -> tuple[str, str] | None:
    path = run_dir / "fsq_source.json"
    if not path.is_file():
        return None
    source = json.loads(path.read_text())
    run_name = str(source.get("source_fsq_run_name", "") or "").strip()
    checkpoint = str(source.get("source_fsq_checkpoint", "") or "").strip()
    return (run_name, checkpoint) if run_name and checkpoint else None


def _rebased_fsq_run_dir(fsq_path: str, dataset_run_dir: Path) -> Path | None:
    """Locate the checkpoint's original SkillVLA run on this machine.

    Checkpoints trained on another cluster record an absolute ``fsq_path``
    under that machine's project root. The ``<source>/<run>`` tail is stable,
    so re-anchor it beside the new dataset's SkillVLA root.
    """
    if not fsq_path:
        return None
    recorded = Path(fsq_path).parent
    if recorded.is_dir():
        return recorded
    skillvla_root = dataset_run_dir.parent.parent
    if len(recorded.parts) < 2:
        return None
    rebased = skillvla_root / recorded.parts[-2] / recorded.parts[-1]
    return rebased if rebased.is_dir() else None


def newtask_ft_code_space_id(
    source: dict,
    checkpoint: Path,
    dataset_run_dir: Path,
    *,
    verify_fsq_source: bool = True,
) -> str:
    """Return the checkpoint code-space id after proving the FSQ model is shared.

    A new-task SkillVLA run has its own run tag, so the tag-derived identity
    differs even though both datasets were labeled by one FSQ model. The
    ``fsq_source.json`` written by build_data names that model unambiguously.
    """
    checkpoint_space = _checkpoint_code_space_id(source, checkpoint)
    if not verify_fsq_source:
        return checkpoint_space
    checkpoint_fsq = str(source.get("fsq_path", "") or "").strip()
    checkpoint_run_dir = _rebased_fsq_run_dir(checkpoint_fsq, dataset_run_dir)
    checkpoint_identity = (
        _fsq_source_identity(checkpoint_run_dir) if checkpoint_run_dir is not None else None
    )
    dataset_identity = _fsq_source_identity(dataset_run_dir)
    if checkpoint_identity is None or dataset_identity is None:
        raise FileNotFoundError(
            "NewTask FT could not read fsq_source.json for both the checkpoint's "
            f"original dataset ({checkpoint_fsq or '<unrecorded>'}) and the new "
            f"dataset ({dataset_run_dir}). Set fsq.verify_source: false only if "
            "you know both were labeled by the same FSQ model."
        )
    if checkpoint_identity != dataset_identity:
        raise ValueError(
            "NewTask FT dataset was labeled by a different FSQ model: "
            f"checkpoint={checkpoint_identity}, dataset={dataset_identity}."
        )
    return checkpoint_space


def _relocate_run_checkpoint(checkpoint: Path, outputs_root: Path, component: str) -> Path:
    """Find ``<run>/checkpoints/<step>/pretrained_model`` under the other known run locations.

    Stage-1 auxiliaries live in ``skillVLA_stage1/<component>``, the legacy
    ``skillVLA_terminator``, or ``skillVLA_NewTask_FT/<component>``. A path that names
    the right run and step under the wrong group is relocated instead of rejected.
    """
    if (checkpoint / "config.json").is_file():
        return checkpoint
    parts = checkpoint.parts
    if len(parts) < 4 or parts[-1] != "pretrained_model" or parts[-3] != "checkpoints":
        return checkpoint
    run_name, step = parts[-4], parts[-2]
    for run_dir in stage1_run_dirs(outputs_root, run_name, component):
        candidate = run_dir / "checkpoints" / step / "pretrained_model"
        if (candidate / "config.json").is_file():
            return candidate
    return checkpoint


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
    # Checkpoints predating focus-UV/full-VLM probes use the frozen, UV-off
    # defaults.  Keep them valid for FT while preserving any new source values.
    backward_defaults = {
        "skill_predictor_freeze_vlm": True,
        "skill_predictor_focus_uv_enabled": False,
        "skill_predictor_focus_uv_loss_weight": 0.25,
        "skill_predictor_end_state_mode": "off",
        "skill_predictor_end_state_dim": 8,
        "skill_predictor_end_state_loss_weight": 1.0,
        "skill_predictor_sampling_mode": "mode1",
        "skill_predictor_boundary_fraction": 0.7,
        "skill_predictor_boundary_window": 10,
    }
    missing = [
        key for key in contract if key not in source and key not in backward_defaults
    ]
    if missing:
        raise ValueError(
            f"FT predictor checkpoint is missing contract fields {missing}: {checkpoint}"
        )
    return {
        key: source.get(key, backward_defaults.get(key)) for key in contract
    }


def _ft_predictor_vlm_override(config: dict, contract: dict) -> dict:
    """How an FT run may re-choose the way the inherited VLM adapts -- and nothing else.

    Everything else about the predictor (reader shape, spatial target, sampling) stays inherited,
    because changing it would leave the warm-started weights meaningless. Whether the VLM trains,
    is frozen, or is frozen behind LoRA is a different question: the same predictor can be adapted
    to a new task either way, so the FT YAML gets to decide. Keys absent from the YAML keep the
    checkpoint's value, so an FT run that says nothing behaves exactly as before.
    """
    # Its own block, never the PT "skill_predictor" one: an FT YAML may still carry PT model
    # sections, which mode=ft deliberately ignores, and reading them here would silently re-adapt
    # every existing FT run.
    requested_freeze = _at(config, "predictor_ft", "freeze_vlm", default=None)
    requested_lora = _at(config, "predictor_ft", "lora", "enabled", default=None)
    if requested_freeze is None and requested_lora is None:
        return {}
    freeze_vlm = (
        as_bool(requested_freeze) if requested_freeze is not None
        else bool(contract["skill_predictor_freeze_vlm"])
    )
    lora_enabled = (
        as_bool(requested_lora) if requested_lora is not None
        else bool(contract["skill_predictor_lora"])
    )
    if lora_enabled and not freeze_vlm:
        raise ValueError(
            "FT cannot co-train the complete predictor VLM and LoRA at once: set "
            "predictor_ft.freeze_vlm=true to adapt through LoRA, or predictor_ft.lora.enabled="
            "false to train the whole VLM."
        )
    override = {
        "skill_predictor_freeze_vlm": freeze_vlm,
        "skill_predictor_lora": lora_enabled,
        "skill_predictor_detach_vlm": bool(freeze_vlm and not lora_enabled),
    }
    adds_lora = lora_enabled and not bool(contract["skill_predictor_lora"])
    for key, leaf, cast, default in (
        ("skill_predictor_lora_targets", "targets", str, "q,k,v,o"),
        ("skill_predictor_lora_rank", "rank", int, 8),
        ("skill_predictor_lora_alpha", "alpha", float, 16.0),
        ("skill_predictor_lora_dropout", "dropout", float, 0.0),
    ):
        value = _at(config, "predictor_ft", "lora", leaf, default=None)
        if value is not None:
            override[key] = cast(value)
        elif adds_lora:
            # The checkpoint carries no adapter, so its inherited shape means nothing.
            override[key] = default
    return override


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
        "terminator_cameras": source.get("terminator_cameras", "both"),
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
    newtask_ft = as_bool(config.get("newtask_ft", False))
    if newtask_ft:
        component_name = config.get("stage1_component")
        if initialization_mode != "ft" or component_name not in {"Predictor", "Terminator"}:
            raise ValueError(
                "NewTask FT auxiliary configs require mode: ft and "
                "stage1_component: Predictor|Terminator."
            )
        # The shared NewTask_FT YAML lists every source checkpoint; each
        # component job consumes only its own.
        unused = "terminator_checkpoint" if component_name == "Predictor" else "predictor_checkpoint"
        used = "predictor_checkpoint" if component_name == "Predictor" else "terminator_checkpoint"
        warm_start = {**config.get("warm_start", {}), unused: ""}
        # NewTask FT names its source as {run, checkpoint}; the run is located automatically.
        if isinstance(warm_start.get(used), dict):
            located = resolve_run_checkpoint(
                Path(str(config["project_root"])).expanduser() / str(config.get("outputs_root", "outputs")),
                component_name,
                warm_start[used],
                field=f"warm_start.{used}",
            )
            warm_start[used] = str(located) if located is not None else ""
        config = {**config, "warm_start": warm_start}
    verify_fsq_source = as_bool(_at(config, "fsq", "verify_source", default=True))

    project_root = Path(str(config["project_root"])).expanduser()
    dataset_root = project_root / str(config.get("dataset_root", "dataset"))
    outputs_root = project_root / str(config.get("outputs_root", "outputs"))
    source = str(_at(config, "dataset", "source"))
    base_run_tag = str(_at(config, "dataset", "run"))
    # Predictor targets must remain the canonical labels. Predictor jobs ignore
    # dataset.relabeled; terminator-only jobs may consume relabeled skill codes.
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
    if newtask_ft:
        if predictor_checkpoint is not None:
            predictor_checkpoint = _relocate_run_checkpoint(predictor_checkpoint, outputs_root, "Predictor")
        if terminator_checkpoint is not None:
            terminator_checkpoint = _relocate_run_checkpoint(terminator_checkpoint, outputs_root, "Terminator")
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

    ft_vlm_override: dict = {}
    if initialization_mode == "pt":
        if predictor_checkpoint is not None or terminator_checkpoint is not None:
            raise ValueError(
                "mode=pt must leave warm_start predictor/terminator checkpoints empty."
            )
        predictor_contract = _predictor_contract(config, state_dim=dataset["state_dim"])
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
        if train_predictor and train_terminator:
            raise ValueError(
                "Train skill predictor and terminator in separate jobs: predictor "
                "training uses transition-occurrence sampling, while terminator "
                "training uses frame-level sampling."
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
        if train_predictor and train_terminator:
            raise ValueError(
                "FT skill predictor and terminator checkpoints must be run in "
                "separate jobs because their DataLoader sampling contracts differ."
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
            if newtask_ft:
                dataset["skill_code_space_id"] = newtask_ft_code_space_id(
                    predictor_source,
                    predictor_checkpoint,
                    dataset_dir.parent,
                    verify_fsq_source=verify_fsq_source,
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
            ft_vlm_override = _ft_predictor_vlm_override(config, predictor_contract)
            predictor_contract.update(ft_vlm_override)
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
            if newtask_ft:
                dataset["skill_code_space_id"] = newtask_ft_code_space_id(
                    terminator_source,
                    terminator_checkpoint,
                    dataset_dir.parent,
                    verify_fsq_source=verify_fsq_source,
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

    component = config.get("stage1_component")
    if component == "Predictor" and not (train_predictor and not train_terminator):
        raise ValueError("Stage-1 Predictor config must train only the predictor.")
    if component == "Terminator" and not (train_terminator and not train_predictor):
        raise ValueError("Stage-1 Terminator config must train only the terminator.")
    if component not in (None, "Predictor", "Terminator"):
        raise ValueError(f"Unknown auxiliary Stage-1 component: {component!r}.")

    if train_predictor and (
        predictor_contract["skill_predictor_focus_uv_enabled"]
        or predictor_contract["skill_predictor_end_state_mode"] != "off"
    ):
        if not dataset["focus_uv_path"]:
            raise FileNotFoundError(
                "Predictor spatial supervision requires a SkillVLA run "
                "built with focus_uv.enabled=true (missing skill_focus_uv_path in info.json)."
            )
    if train_predictor and predictor_contract["skill_predictor_end_state_mode"] == "full_state":
        if predictor_contract["skill_predictor_end_state_dim"] != dataset["state_dim"]:
            raise ValueError("Full-state predictor dimension does not match the SkillVLA dataset.")

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
        sampling_mode = predictor_contract["skill_predictor_sampling_mode"]
        if sampling_mode not in {"mode1", "mode2"}:
            raise ValueError("skill_predictor.sampling.mode must be mode1 or mode2.")
        if not 0 <= predictor_contract["skill_predictor_boundary_fraction"] <= 1:
            raise ValueError("skill_predictor.sampling.boundary_fraction must be in [0, 1].")
        if predictor_contract["skill_predictor_boundary_window"] < 1:
            raise ValueError("skill_predictor.sampling.boundary_window must be positive.")
        predictor_name = "predictor"
        if predictor_contract["skill_predictor_focus_uv_enabled"]:
            predictor_name += "_uv"
        elif predictor_contract["skill_predictor_end_state_mode"] == "xyz":
            predictor_name += "_xyz"
        elif predictor_contract["skill_predictor_end_state_mode"] == "full_state":
            predictor_name += "_state"
        if not predictor_contract["skill_predictor_freeze_vlm"]:
            predictor_name += "_fullvlm"
        elif ft_vlm_override:
            predictor_name += "_lora" if predictor_contract["skill_predictor_lora"] else "_frozenvlm"
        if sampling_mode == "mode2":
            predictor_name += "_mode2"
        target_names.append(predictor_name)
    if train_terminator:
        context_tag = {
            "prev_action": "prev",
            "proprio": "prop",
            "none": "none",
        }[terminator_contract["terminator_context"]]
        camera_tag = terminator_contract["terminator_cameras"]
        target_names.append(f"terminator_{context_tag}_{camera_tag}")
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
        "skill_predictor_vlm_lr_scale": float(
            _at(config, "training", "optimizer", "predictor_vlm_lr_scale", default=1.0)
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
        "output_dir": (
            outputs_root
            / ("skillVLA_NewTask_FT" if newtask_ft else "skillVLA_stage1")
            / config["stage1_component"]
            / run_name
            if config.get("stage1_component") in {"Predictor", "Terminator"}
            else outputs_root / "skillVLA_terminator" / run_name
        ),
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
        "skill_predictor_vlm_lr_scale",
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
    settings = build_settings(load_stage1_component_config(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
