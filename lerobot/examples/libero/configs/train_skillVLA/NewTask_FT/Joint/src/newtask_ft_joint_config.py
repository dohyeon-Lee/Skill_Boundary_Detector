#!/usr/bin/env python3
"""Resolve NewTask Joint Predictor/VSA co-training from two Stage-1 checkpoints."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TRAIN_SKILLVLA = _HERE.parents[3]
sys.path.insert(0, str(_TRAIN_SKILLVLA.parent / "train_skills" / "src"))
from train_skills_config import (  # noqa: E402
    as_bool,
    load_stage1_component_config,
    print_shell,
    resolve_run_checkpoint,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_VSA = _load_module(
    "newtask_ft_vsa_config",
    _HERE.parents[2] / "VSA" / "src" / "newtask_ft_vsa_config.py",
)
_AUX = _VSA._AUX
_at = _VSA._at
DEFAULT_CONFIG_PATH = _HERE.parent.parent / "joint_ft_config.yaml"

_PREDICTOR_SHAPE_FIELDS = (
    "skill_predictor_vlm_variant",
    "skill_predictor_image_size",
    "skill_predictor_reader_tokens",
    "skill_predictor_reader_depth",
    "skill_predictor_reader_heads",
    "skill_predictor_all_layers",
    "skill_predictor_deadzone_frac",
    "skill_predictor_attend_image",
    "skill_predictor_attend_language",
    "skill_predictor_focus_uv_enabled",
    "skill_predictor_end_state_mode",
    "skill_predictor_end_state_dim",
    "tokenizer_max_length",
)


def _predictor_checkpoint(outputs_root: Path, selector) -> Path:
    checkpoint = resolve_run_checkpoint(
        outputs_root,
        "Predictor",
        selector,
        field="warm_start.predictor_checkpoint",
    )
    if checkpoint is None:
        raise ValueError("warm_start.predictor_checkpoint is required for Joint FT.")
    for name in ("config.json", "model.safetensors"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(
                f"Incomplete Predictor checkpoint, missing {name}: {checkpoint}"
            )
    return checkpoint


def build_settings(config: dict) -> dict:
    if config.get("stage1_component") != "Joint" or not as_bool(
        config.get("newtask_ft", False)
    ):
        raise ValueError(
            "NewTask Joint requires stage1_component: Joint and newtask_ft: true."
        )

    # Reuse the VSA resolver for dataset provenance, FSQ identity, architecture,
    # normalization, and frozen-route validation.
    vsa_config = copy.deepcopy(config)
    vsa_config["stage1_component"] = "VSA"
    vsa_config["adaptation"] = {
        "train_dino": as_bool(_at(config, "adaptation", "train_dino", default=False)),
        "dino_lr_scale": float(
            _at(config, "adaptation", "dino_lr_scale", default=1.0)
        ),
        "unfreeze_action_head": False,
        "full_unfreeze": False,
        "skill_flow_loss": False,
    }
    settings = _VSA.build_settings(vsa_config)
    project_root = settings["project_root"]
    outputs_root = project_root / str(config.get("outputs_root", "outputs"))
    predictor_checkpoint = _predictor_checkpoint(
        outputs_root,
        _at(config, "warm_start", "predictor_checkpoint", default={}),
    )
    predictor_source = json.loads(
        (predictor_checkpoint / "config.json").read_text()
    )
    if predictor_source.get("type") not in {"skill_aux", "skill_expert"} or not predictor_source.get(
        "train_skill_predictor", False
    ):
        raise ValueError(
            "Joint warm start requires a checkpoint with train_skill_predictor=true."
        )
    if predictor_source.get("skill_predictor_end_state_mode") != "xyz":
        raise ValueError("Joint Predictor checkpoint must contain an XYZ head.")
    if predictor_source.get("skill_predictor_lora", False):
        raise ValueError(
            "Joint currently requires a non-LoRA Predictor checkpoint; the VLM is frozen."
        )

    vsa_source = json.loads((settings["vsa_checkpoint_path"] / "config.json").read_text())
    if not vsa_source.get("skill_flow_enabled", False) or vsa_source.get(
        "skill_flow_target", "canonical"
    ) != "canonical":
        raise ValueError(
            "Joint VSA checkpoint must provide the canonical skill-only trajectory route."
        )
    for field in ("skill_fsq_levels", "skill_vocab_size"):
        if predictor_source.get(field) != vsa_source.get(field):
            raise ValueError(
                f"Joint Predictor/VSA {field} mismatch: "
                f"{predictor_source.get(field)!r} != {vsa_source.get(field)!r}."
            )
    dataset_run_dir = settings["skillvla_dataset_dir"].parent
    verify = as_bool(_at(config, "fsq", "verify_source", default=True))
    predictor_space = _AUX.newtask_ft_code_space_id(
        predictor_source,
        predictor_checkpoint,
        dataset_run_dir,
        verify_fsq_source=verify,
    )
    vsa_space = _AUX.newtask_ft_code_space_id(
        vsa_source,
        settings["vsa_checkpoint_path"],
        dataset_run_dir,
        verify_fsq_source=verify,
    )
    if predictor_space != vsa_space:
        raise ValueError(
            f"Joint Predictor/VSA code-space mismatch: {predictor_space!r} != {vsa_space!r}."
        )

    predictor_source.setdefault("tokenizer_max_length", 200)
    missing = [field for field in _PREDICTOR_SHAPE_FIELDS if field not in predictor_source]
    if missing:
        raise ValueError(
            f"Joint Predictor checkpoint is missing module contract fields {missing}."
        )
    predictor_run, predictor_step = _VSA._checkpoint_run_and_step(predictor_checkpoint)
    weights = {
        "joint_gt_skill_weight": float(
            _at(config, "loss", "gt_skill_weight", default=1.0)
        ),
        "joint_gt_xyz_weight": float(
            _at(config, "loss", "gt_xyz_weight", default=1.0)
        ),
        "joint_route_hard_weight": float(
            _at(config, "loss", "route_hard_weight", default=1.0)
        ),
        "joint_route_ste_weight": float(
            _at(config, "loss", "route_ste_weight", default=0.1)
        ),
        "joint_route_timesteps": int(
            _at(config, "loss", "route_timesteps", default=2)
        ),
        "joint_action_to_predictor": as_bool(
            _at(config, "loss", "action_to_predictor", default=False)
        ),
        "joint_xyz_to_skill": as_bool(
            _at(config, "loss", "xyz_to_skill", default=False)
        ),
    }
    if any(value < 0 for key, value in weights.items() if key.endswith("_weight")):
        raise ValueError("Joint loss weights must be non-negative.")
    if weights["joint_route_timesteps"] <= 0:
        raise ValueError("loss.route_timesteps must be positive.")
    if weights["joint_xyz_to_skill"]:
        raise ValueError(
            "Joint keeps skill->XYZ gradient detached; loss.xyz_to_skill must remain false."
        )

    settings.update({field: predictor_source[field] for field in _PREDICTOR_SHAPE_FIELDS})
    settings.update(weights)
    settings.update(
        {
            "predictor_checkpoint_path": predictor_checkpoint,
            "skill_predictor_freeze_vlm": True,
            "skill_predictor_detach_vlm": True,
            "skill_predictor_lora": False,
            "skill_predictor_start_proprio_dim": int(
                vsa_source["input_features"]["observation.state"]["shape"][0]
            ),
            "joint_predictor_lr_scale": float(
                _at(config, "training", "optimizer", "predictor_lr_scale", default=1.0)
            ),
        }
    )
    settings["run_name"] = (
        f"{settings['run_name']}_joint_{predictor_run}_{predictor_step}"
    )
    settings["output_dir"] = (
        outputs_root / "skillVLA_NewTask_FT" / "Joint" / settings["run_name"]
    )
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
