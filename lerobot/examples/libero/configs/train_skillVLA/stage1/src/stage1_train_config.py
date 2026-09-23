#!/usr/bin/env python3
"""Resolve the supported Stage-1 Arch0--Arch20 family modes."""

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
    resolve_skillvla_dataset_run,
)

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "VSA" / "vsa_train_config.yaml"
SUPPORTED_ARCHITECTURES = (
    "arch0",
    "arch0_skill",
    "arch0_skill_chunk",
    "arch1",
    "arch1_skill",
    "arch1_skill_chunk",
    "arch2",
    "arch2_skill",
    "arch2_skill_chunk",
    "arch3",
    "arch3_skill",
    "arch3_skill_chunk",
    "arch4",
    "arch4_skill",
    "arch4_skill_chunk",
    "arch5",
    "arch5_skill",
    "arch5_skill_chunk",
    "arch6",
    "arch6_skill",
    "arch6_skill_chunk",
    "arch7",
    "arch7_skill",
    "arch7_skill_chunk",
    "arch8_1",
    "arch8_1_skill",
    "arch8_1_skill_chunk",
    "arch8_2",
    "arch8_2_skill",
    "arch8_2_skill_chunk",
    "arch9_1",
    "arch9_1_skill",
    "arch9_1_skill_chunk",
    "arch9_2",
    "arch9_2_skill",
    "arch9_2_skill_chunk",
    "arch10_1",
    "arch10_1_skill",
    "arch10_1_skill_chunk",
    "arch10_2",
    "arch10_2_skill",
    "arch10_2_skill_chunk",
    "arch11_1",
    "arch11_1_skill",
    "arch11_1_skill_chunk",
    "arch11_2",
    "arch11_2_skill",
    "arch11_2_skill_chunk",
    "arch12_1",
    "arch12_1_skill",
    "arch12_1_skill_chunk",
    "arch12_2",
    "arch12_2_skill",
    "arch12_2_skill_chunk",
    "arch13",
    "arch13_skill",
    "arch13_skill_chunk",
    "arch14",
    "arch14_skill",
    "arch14_skill_chunk",
    "arch15",
    "arch15_skill",
    "arch15_skill_chunk",
    "arch16",
    "arch16_skill",
    "arch16_skill_chunk",
    "arch16_align",
    "arch16_align_skill",
    "arch16_align_skill_chunk",
    "arch17",
    "arch17_skill",
    "arch17_skill_chunk",
    "arch17_align",
    "arch17_align_skill",
    "arch17_align_skill_chunk",
    "arch18",
    "arch18_skill",
    "arch18_skill_chunk",
    "arch18_align",
    "arch18_align_skill",
    "arch18_align_skill_chunk",
    "arch19",
    "arch19_skill",
    "arch19_skill_chunk",
    "arch20",
    "arch20_skill",
    "arch20_skill_chunk",
)
ARCH0_REVISION = "skillvla_real_v1"
ARCH1_REVISION = "fixed_visual_bottleneck_v1"
ARCH2_REVISION = "late_visual_bottleneck_v1"
ARCH3_REVISION = "layerwise_cond_bottleneck_v1"
ARCH4_REVISION = "layerwise_cond_bottleneck_core_exit_v1"
ARCH5_REVISION = "layerwise_cond_bottleneck_core_exit_uv_v1"
ARCH6_REVISION = "layerwise_cond_bottleneck_core_exit_latent_uv_v1"
ARCH7_REVISION = "layerwise_cond_bottleneck_core_exit_latent_xyz_v1"
ARCH8_1_REVISION = "layerwise_cond_bottleneck_uv_cond_xyz_v1"
ARCH8_2_REVISION = "layerwise_cond_bottleneck_uv_cond_xyz_cond_termination_v2"
ARCH9_1_REVISION = "layerwise_cond_bottleneck_wrist_skill_end_pose_v1"
ARCH9_2_REVISION = "layerwise_cond_bottleneck_wrist_skill_end_pose_cond_termination_v2"
ARCH10_1_REVISION = "layerwise_cond_bottleneck_wrist_proprio_expert_end_pose_v1"
ARCH10_2_REVISION = "layerwise_cond_bottleneck_wrist_proprio_expert_end_pose_cond_termination_v1"
ARCH11_1_REVISION = "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_end_pose_v1"
ARCH11_2_REVISION = "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_end_pose_cond_termination_v1"
ARCH12_1_REVISION = "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_v1"
ARCH12_2_REVISION = "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_cond_termination_v1"
ARCH13_REVISION = "layerwise_cond_bottleneck_xyz_cond_uv_v1"
ARCH14_REVISION = "layerwise_cond_bottleneck_xyz_cond_uv_expert_end_pose_v1"
ARCH15_REVISION = "layerwise_cond_bottleneck_xyz_skill_cond_uv_expert_end_pose_v1"
ARCH16_REVISION = "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_delta_v1"
ARCH17_REVISION = "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_delta_bridge_proprio_v1"
ARCH18_REVISION = "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_v1"
ARCH16_ALIGN_REVISION = "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_delta_align_v1"
ARCH17_ALIGN_REVISION = "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_delta_bridge_proprio_align_v1"
ARCH18_ALIGN_REVISION = "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_align_v1"
ARCH19_REVISION = "layerwise_cond_bottleneck_xyz_skill_cond_uv_expert_skill_delta_v1"
ARCH20_REVISION = "layerwise_cond_bottleneck_xyz_skill_cond_uv_v1"


def _at(config: dict, *path: str, default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _local_model_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return Path(resolve_path(project_root, path))
    if path.exists() or "models" not in path.parts:
        return path
    return project_root.joinpath(*path.parts[path.parts.index("models") :])


def _numeric_range(
    value,
    *,
    field: str,
    integer: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[int, int] | tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be a two-value [min, max] range.")
    cast = int if integer else float
    low, high = cast(value[0]), cast(value[1])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        raise ValueError(f"{field} values must be finite.")
    if low > high:
        raise ValueError(f"{field} minimum cannot exceed its maximum.")
    if minimum is not None and low < minimum:
        raise ValueError(f"{field} values must be >= {minimum}.")
    if maximum is not None and high > maximum:
        raise ValueError(f"{field} values must be <= {maximum}.")
    return low, high


def _read_dataset_contract(dataset_dir: Path, run_tag: str) -> dict:
    """Read skill geometry and jitter metadata without opening an FSQ checkpoint."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Stage-1 dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    levels = [int(level) for level in info.get("skill_fsq_levels", [])]
    if not levels or any(level <= 1 for level in levels):
        raise ValueError(f"Invalid skill_fsq_levels in {info_path}: {levels}")
    match = re.search(r"FSQ(\d+)", run_tag)
    if match and [int(digit) for digit in match.group(1)] != levels:
        raise ValueError(
            f"Dataset run says FSQ{match.group(1)}, but {info_path} says levels={levels}."
        )
    features = info.get("features", {})
    state_dim = int(features["observation.state"]["shape"][0])
    action_dim = int(features["action"]["shape"][0])
    jitter_pmax = int(info.get("skill_pmax", 0))
    directional_jitter = {
        name: int(info.get(f"skill_jitter_{name}_pmax", jitter_pmax))
        for name in ("early_start", "late_start", "early_end", "late_end")
    }
    if any(value < 0 or value > jitter_pmax for value in directional_jitter.values()):
        raise ValueError(
            f"Invalid directional jitter contract in {info_path}: "
            f"storage={jitter_pmax}, directional={directional_jitter}."
        )
    proprio_grounding = str(
        info.get("proprio_grounding", "none") or "none"
    ).strip().lower()
    if proprio_grounding not in {"none", "episode_start_xyz"}:
        raise ValueError(
            f"Invalid proprio_grounding in {info_path}: {proprio_grounding!r}."
        )
    focus_path = None
    recorded_focus_path = str(info.get("skill_focus_uv_path", "") or "").strip()
    if recorded_focus_path:
        recorded = Path(recorded_focus_path).expanduser()
        local = dataset_dir.parent / recorded.name
        if recorded.is_file():
            focus_path = recorded
        elif local.is_file():
            focus_path = local
    return {
        "levels": levels,
        "skill_code_space_id": str(
            info.get("skill_code_space_id", run_tag) or run_tag
        ).strip(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "proprio_grounding": proprio_grounding,
        "focus_uv_path": focus_path,
        "focus_uv_recorded_path": recorded_focus_path,
        "focus_uv_camera": str(info.get("skill_focus_uv_camera", "") or ""),
        "focus_uv_normalization": str(
            info.get("skill_focus_uv_normalization", "") or ""
        ),
        "skill_observed_max_length": int(
            info.get("skill_observed_max_length", 0)
        ),
        "jitter_pmax": jitter_pmax,
        "jitter_early_start_pmax": directional_jitter["early_start"],
        "jitter_late_start_pmax": directional_jitter["late_start"],
        "jitter_early_end_pmax": directional_jitter["early_end"],
        "jitter_late_end_pmax": directional_jitter["late_end"],
        "jitter_distribution": str(
            info.get("skill_jitter_distribution", "half_normal")
        ).replace("-", "_"),
    }


_PREDICTOR_CONTRACT_FIELDS = (
    "skill_predictor_vlm_variant",
    "skill_predictor_image_size",
    "skill_predictor_reader_tokens",
    "skill_predictor_reader_depth",
    "skill_predictor_reader_heads",
    "skill_predictor_all_layers",
    "skill_predictor_detach_vlm",
    "skill_predictor_lora",
    "skill_predictor_lora_targets",
    "skill_predictor_lora_rank",
    "skill_predictor_lora_alpha",
    "skill_predictor_lora_dropout",
    "skill_predictor_deadzone_frac",
    "skill_predictor_attend_image",
    "skill_predictor_attend_language",
    "tokenizer_max_length",
)


def _predictor_contract_from_checkpoint(
    checkpoint: Path,
    *,
    levels: list[int],
) -> dict:
    """Read the frozen predictor contract; Stage1 never trains this module."""
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-1 predictor config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Stage-1 predictor weights not found: {weights_path}")
    source = json.loads(config_path.read_text())
    if source.get("type") not in {"skill_expert", "skill_aux"}:
        raise ValueError(
            "Predictor checkpoint must be policy.type=skill_expert or skill_aux, got "
            f"{source.get('type')!r} at {checkpoint}."
        )
    if not source.get("train_skill_predictor", False):
        raise ValueError(f"Stage-1 checkpoint has no trained predictor: {checkpoint}")

    expected_geometry = {
        "skill_fsq_levels": levels,
        "skill_vocab_size": math.prod(levels),
    }
    mismatches = [
        f"{field}: checkpoint={source.get(field)!r}, dataset={value!r}"
        for field, value in expected_geometry.items()
        if source.get(field) != value
    ]
    if mismatches:
        raise ValueError("Predictor checkpoint contract mismatch: " + "; ".join(mismatches))
    missing = [field for field in _PREDICTOR_CONTRACT_FIELDS if field not in source]
    if missing:
        raise ValueError(
            "Predictor checkpoint does not record its complete frozen contract: "
            f"missing={missing} at {checkpoint}."
        )
    return {field: source[field] for field in _PREDICTOR_CONTRACT_FIELDS}


def build_settings(config: dict) -> dict:
    if config.get("stage1_component") not in (None, "VSA"):
        raise ValueError("Stage-1 VSA training requires stage1_component: VSA.")
    removed_sections = {"skill_predictor", "terminator"} & set(config)
    if removed_sections:
        raise ValueError(
            "Stage1 trains only the action model; remove co-training sections: "
            f"{sorted(removed_sections)}."
        )
    run_config = config.get("run", {})
    if not isinstance(run_config, dict):
        raise ValueError("run must be a mapping containing an optional suffix.")
    unknown_run_keys = set(run_config) - {"suffix"}
    if unknown_run_keys:
        raise ValueError(f"Unsupported run settings: {sorted(unknown_run_keys)}.")
    run_suffix = str(run_config.get("suffix", "") or "").strip().strip("_")
    if run_suffix and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_suffix) is None:
        raise ValueError(
            "run.suffix may contain only letters, numbers, '.', '_' and '-', "
            "and must start with a letter or number."
        )
    project_root = Path(str(config["project_root"])).expanduser()
    dataset_root = project_root / str(config.get("dataset_root", "dataset"))
    outputs_root = project_root / str(config.get("outputs_root", "outputs"))
    source = str(_at(config, "dataset", "source"))
    base_run_tag = str(_at(config, "dataset", "run"))
    skillvla_root = dataset_root / str(
        _at(config, "dataset", "skillvla_root", default="skillvla_dataset")
    )
    run_tag, dataset_relabeled = resolve_skillvla_dataset_run(
        skillvla_root / source,
        base_run_tag,
        _at(config, "dataset", "relabeled", default=""),
    )
    dataset_dir = skillvla_root / source / run_tag / "skillvla"
    contract = _read_dataset_contract(dataset_dir, run_tag)

    training_steps = int(
        _at(config, "training", "schedule", "steps", default=50000)
    )
    scheduler_mode = str(
        _at(
            config,
            "training",
            "schedule",
            "lr_mode",
            default="cosine_decay",
        )
    ).strip().lower()
    scheduler_warmup_steps = int(
        _at(config, "training", "schedule", "warmup_steps", default=1000)
    )
    scheduler_decay_steps = int(
        _at(config, "training", "schedule", "lr_decay_steps", default=30000)
    )
    if training_steps <= 0:
        raise ValueError("training.schedule.steps must be positive.")
    if scheduler_mode not in {"cosine_decay", "warmup_constant"}:
        raise ValueError(
            "training.schedule.lr_mode must be 'cosine_decay' or "
            f"'warmup_constant', got {scheduler_mode!r}."
        )
    if scheduler_warmup_steps < 0:
        raise ValueError("training.schedule.warmup_steps must be non-negative.")
    if scheduler_decay_steps <= 0:
        raise ValueError("training.schedule.lr_decay_steps must be positive.")

    pi_base = _local_model_path(
        project_root, str(_at(config, "warm_start", "pi_base", default="models/pi05_base"))
    )
    tokenizer_path = _local_model_path(
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
    dino_model = _local_model_path(
        project_root, str(_at(config, "vision", "dino_model", default="models/dinov3-vitl16"))
    )
    fsq_path = dataset_dir.parent / "FSQ.pt"
    if not (pi_base / "model.safetensors").is_file():
        raise FileNotFoundError(f"pi0.5 base checkpoint not found: {pi_base}")
    if not dino_model.is_dir():
        raise FileNotFoundError(f"DINO model not found: {dino_model}")
    training_skill_source = str(
        _at(config, "action_conditioning", "training_skill_source", default="gt")
    ).strip().lower()
    if training_skill_source not in {"gt", "predictor"}:
        raise ValueError(
            "action_conditioning.training_skill_source must be gt or predictor."
        )
    predictor_checkpoint_value = str(
        _at(config, "warm_start", "predictor_checkpoint", default="") or ""
    ).strip()
    predictor_checkpoint = (
        Path(resolve_path(project_root, predictor_checkpoint_value))
        if predictor_checkpoint_value
        else None
    )
    if training_skill_source == "predictor" and predictor_checkpoint is None:
        raise ValueError(
            "training_skill_source=predictor requires warm_start.predictor_checkpoint."
        )
    uses_skill_predictor = training_skill_source == "predictor"
    if uses_skill_predictor:
        required = ("config.json", "tokenizer_config.json", "tokenizer.json")
        missing = [name for name in required if not (tokenizer_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Local PaliGemma tokenizer is incomplete at {tokenizer_path}: missing {missing}."
            )
    freeze_vision_encoder = as_bool(
        _at(config, "vision", "freeze", default=False)
    )
    vision_config = config.get("vision", {})
    if not isinstance(vision_config, dict):
        raise ValueError("vision must be a mapping.")
    unknown_vision_keys = set(vision_config) - {
        "dino_model",
        "image_size",
        "freeze",
        "foveation",
    }
    if unknown_vision_keys:
        raise ValueError(f"Unsupported vision settings: {sorted(unknown_vision_keys)}.")
    foveation_config = vision_config.get("foveation", {})
    if not isinstance(foveation_config, dict):
        raise ValueError("vision.foveation must be a mapping.")
    unknown_foveation_keys = set(foveation_config) - {
        "enabled",
        "mode",
        "crop_size",
        "output_size",
        "inner_box",
        "shape",
        "sharp_size",
        "feather",
        "peripheral_mode",
        "blur_radius",
        "randomization",
    }
    if unknown_foveation_keys:
        raise ValueError(
            "Unsupported vision.foveation settings: "
            f"{sorted(unknown_foveation_keys)}."
        )
    foveated_vision_enabled = as_bool(foveation_config.get("enabled", False))
    foveation_mode = str(
        foveation_config.get("mode", "partial_fov")
    ).strip().lower()
    if foveation_mode not in {"partial_fov", "crop"}:
        raise ValueError("vision.foveation.mode must be partial_fov|crop.")
    foveation_crop_size = int(foveation_config.get("crop_size", 128))
    foveation_output_size = int(foveation_config.get("output_size", 224))
    if foveation_crop_size <= 0 or foveation_output_size <= 0:
        raise ValueError(
            "vision.foveation crop_size and output_size must be positive."
        )
    inner_box_config = foveation_config.get("inner_box", {})
    if not isinstance(inner_box_config, dict):
        raise ValueError("vision.foveation.inner_box must be a mapping.")
    unknown_inner_box_keys = set(inner_box_config) - {
        "enabled",
        "mode",
        "size",
        "line_width",
    }
    if unknown_inner_box_keys:
        raise ValueError(
            "Unsupported vision.foveation.inner_box settings: "
            f"{sorted(unknown_inner_box_keys)}."
        )
    foveation_inner_box_enabled = as_bool(
        inner_box_config.get("enabled", True)
    )
    foveation_inner_box_mode = str(
        inner_box_config.get("mode", "blur")
    ).strip().lower()
    if foveation_inner_box_mode not in {"box", "blur"}:
        raise ValueError("vision.foveation.inner_box.mode must be box|blur.")
    foveation_inner_box_size = int(inner_box_config.get("size", 32))
    foveation_inner_box_line_width = int(inner_box_config.get("line_width", 3))
    if foveation_inner_box_size <= 0 or foveation_inner_box_line_width <= 0:
        raise ValueError(
            "vision.foveation.inner_box size and line_width must be positive."
        )
    if foveation_inner_box_size > foveation_crop_size:
        raise ValueError(
            "vision.foveation.inner_box.size cannot exceed crop_size."
        )
    foveation_shape = str(foveation_config.get("shape", "square")).strip().lower()
    if foveation_shape not in {"square", "circle"}:
        raise ValueError("vision.foveation.shape must be square|circle.")
    foveation_sharp_size = int(foveation_config.get("sharp_size", 96))
    foveation_feather = int(foveation_config.get("feather", 20))
    foveation_peripheral_mode = str(
        foveation_config.get("peripheral_mode", "blur")
    ).strip().lower()
    foveation_peripheral_blur_radius = float(
        foveation_config.get("blur_radius", 8.0)
    )
    if foveation_sharp_size <= 0:
        raise ValueError("vision.foveation.sharp_size must be positive.")
    if foveation_feather < 0:
        raise ValueError("vision.foveation.feather must be non-negative.")
    if foveation_peripheral_mode not in {"blur", "black"}:
        raise ValueError("vision.foveation.peripheral_mode must be blur|black.")
    if foveation_mode == "crop" and foveation_peripheral_mode == "black":
        raise ValueError("vision.foveation.peripheral_mode=black requires partial_fov.")
    if (
        not math.isfinite(foveation_peripheral_blur_radius)
        or foveation_peripheral_blur_radius < 0
    ):
        raise ValueError("vision.foveation.blur_radius must be finite and non-negative.")

    randomization = foveation_config.get("randomization", {})
    if not isinstance(randomization, dict):
        raise ValueError("vision.foveation.randomization must be a mapping.")
    unknown_randomization_keys = set(randomization) - {
        "enabled",
        "color",
        "crop",
        "blur",
    }
    if unknown_randomization_keys:
        raise ValueError(
            "Unsupported vision.foveation.randomization settings: "
            f"{sorted(unknown_randomization_keys)}."
        )
    foveation_randomization_enabled = as_bool(
        randomization.get("enabled", False)
    )
    color_config = randomization.get("color", {})
    crop_config = randomization.get("crop", {})
    blur_config = randomization.get("blur", {})
    for name, section in (
        ("color", color_config),
        ("crop", crop_config),
        ("blur", blur_config),
    ):
        if not isinstance(section, dict):
            raise ValueError(
                f"vision.foveation.randomization.{name} must be a mapping."
            )
    unknown_color_keys = set(color_config) - {
        "enabled",
        "brightness",
        "contrast",
        "saturation",
        "hue",
    }
    unknown_crop_keys = set(crop_config) - {
        "enabled",
        "offset_px",
        "inner_box_offset_px",
    }
    unknown_blur_keys = set(blur_config) - {"enabled", "blur_radius"}
    if unknown_color_keys or unknown_crop_keys or unknown_blur_keys:
        raise ValueError(
            "Unsupported foveation randomization settings: "
            f"color={sorted(unknown_color_keys)}, crop={sorted(unknown_crop_keys)}, "
            f"blur={sorted(unknown_blur_keys)}."
        )
    foveation_color_enabled = as_bool(color_config.get("enabled", False))
    foveation_brightness = _numeric_range(
        color_config.get("brightness", [0.8, 1.2]),
        field="vision.foveation.randomization.color.brightness",
        minimum=0.0,
    )
    foveation_contrast = _numeric_range(
        color_config.get("contrast", [0.8, 1.2]),
        field="vision.foveation.randomization.color.contrast",
        minimum=0.0,
    )
    foveation_saturation = _numeric_range(
        color_config.get("saturation", [0.8, 1.2]),
        field="vision.foveation.randomization.color.saturation",
        minimum=0.0,
    )
    foveation_hue = _numeric_range(
        color_config.get("hue", [-0.15, 0.15]),
        field="vision.foveation.randomization.color.hue",
        minimum=-0.5,
        maximum=0.5,
    )
    foveation_crop_enabled = as_bool(crop_config.get("enabled", False))
    foveation_crop_offset = _numeric_range(
        crop_config.get("offset_px", [-24, 24]),
        field="vision.foveation.randomization.crop.offset_px",
        integer=True,
    )
    foveation_inner_box_offset = _numeric_range(
        crop_config.get("inner_box_offset_px", [-4, 4]),
        field="vision.foveation.randomization.crop.inner_box_offset_px",
        integer=True,
    )
    foveation_input_blur_enabled = as_bool(blur_config.get("enabled", False))
    foveation_input_blur_radius = _numeric_range(
        blur_config.get("blur_radius", [0.0, 4.0]),
        field="vision.foveation.randomization.blur.blur_radius",
        minimum=0.0,
    )
    if foveated_vision_enabled:
        if contract["focus_uv_path"] is None:
            recorded = contract["focus_uv_recorded_path"] or "<not recorded>"
            raise FileNotFoundError(
                "vision.foveation.enabled=true requires skill_focus_uv.npz, but "
                f"the dataset records {recorded!r} and no local artifact exists "
                f"beside {dataset_dir.parent}. Rebuild/relabel with focus_uv enabled."
            )
        if contract["focus_uv_normalization"] != "minus_one_to_one":
            raise ValueError(
                "Foveated training requires skill_focus_uv_normalization="
                f"'minus_one_to_one', got {contract['focus_uv_normalization']!r}."
            )
    expert_variant = str(
        _at(config, "architecture", "expert_variant", default="gemma_300m")
    )
    architecture_label = str(
        _at(config, "architecture", "name", default="arch0")
    ).strip().lower()
    architecture_config = config.get("architecture", {})
    if not isinstance(architecture_config, dict):
        raise ValueError("architecture must be a mapping.")
    supported_architecture_keys = {
        "name",
        "expert_variant",
        "max_state_dim",
        "max_action_dim",
        "chunk_size",
        "visual_bottleneck_tokens",
        "visual_bridge_last_n_layers",
        "focus_uv_loss_weight",
        "end_xyz_loss_weight",
        "spatial_loss_weight",
        "end_pose_mode",
    }
    unknown_architecture_keys = set(architecture_config) - supported_architecture_keys
    if unknown_architecture_keys:
        raise ValueError(
            "Stage1 exposes only the fixed architecture contracts; remove unsupported "
            f"architecture keys: {sorted(unknown_architecture_keys)}."
        )
    if architecture_label not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "architecture.name must be arch0|arch0_skill|arch0_skill_chunk|"
            "arch1|arch1_skill|arch1_skill_chunk|"
            "arch2|arch2_skill|arch2_skill_chunk|"
            "arch3|arch3_skill|arch3_skill_chunk|"
            "arch4|arch4_skill|arch4_skill_chunk|"
            "arch5|arch5_skill|arch5_skill_chunk|"
            "arch6|arch6_skill|arch6_skill_chunk|"
            "arch7|arch7_skill|arch7_skill_chunk|"
            "arch8_1|arch8_1_skill|arch8_1_skill_chunk|"
            "arch8_2|arch8_2_skill|arch8_2_skill_chunk|"
            "arch9_1|arch9_1_skill|arch9_1_skill_chunk|"
            "arch9_2|arch9_2_skill|arch9_2_skill_chunk|"
            "arch10_1|arch10_1_skill|arch10_1_skill_chunk|"
            "arch10_2|arch10_2_skill|arch10_2_skill_chunk|"
            "arch11_1|arch11_1_skill|arch11_1_skill_chunk|"
            "arch11_2|arch11_2_skill|arch11_2_skill_chunk|"
            "arch12_1|arch12_1_skill|arch12_1_skill_chunk|"
            "arch12_2|arch12_2_skill|arch12_2_skill_chunk|"
            "arch13|arch13_skill|arch13_skill_chunk|"
            "arch14|arch14_skill|arch14_skill_chunk|"
            "arch15|arch15_skill|arch15_skill_chunk|"
            "arch16|arch16_skill|arch16_skill_chunk|"
            "arch16_align|arch16_align_skill|arch16_align_skill_chunk|"
            "arch17|arch17_skill|arch17_skill_chunk|"
            "arch17_align|arch17_align_skill|arch17_align_skill_chunk|"
            "arch18|arch18_skill|arch18_skill_chunk|"
            "arch18_align|arch18_align_skill|arch18_align_skill_chunk|"
            "arch19|arch19_skill|arch19_skill_chunk|"
            "arch20|arch20_skill|arch20_skill_chunk, got "
            f"{architecture_label!r}."
        )
    is_arch1 = architecture_label == "arch1" or architecture_label.startswith("arch1_")
    is_arch2 = architecture_label == "arch2" or architecture_label.startswith("arch2_")  # not Arch20
    is_arch3 = architecture_label.startswith("arch3")
    is_arch4 = architecture_label.startswith("arch4")
    is_arch5 = architecture_label.startswith("arch5")
    is_arch6 = architecture_label.startswith("arch6")
    is_arch7 = architecture_label.startswith("arch7")
    is_arch8_1 = architecture_label.startswith("arch8_1")
    is_arch8_2 = architecture_label.startswith("arch8_2")
    is_arch9_1 = architecture_label.startswith("arch9_1")
    is_arch9_2 = architecture_label.startswith("arch9_2")
    is_arch10_1 = architecture_label.startswith("arch10_1")
    is_arch10_2 = architecture_label.startswith("arch10_2")
    is_arch11_1 = architecture_label.startswith("arch11_1")
    is_arch11_2 = architecture_label.startswith("arch11_2")
    is_arch12_1 = architecture_label.startswith("arch12_1")
    is_arch12_2 = architecture_label.startswith("arch12_2")
    # Arch16 = Arch11_1 with the Expert goal replaced by the skill displacement (end - start xyz);
    # Arch17 = Arch16 + current proprio in the bridge Expert layer(s). Both follow the wrist rules.
    # Arch18 = Arch17 with skill-start and skill-end xyz as two absolute Expert inputs.
    is_arch18 = architecture_label.startswith("arch18")
    is_arch17 = architecture_label.startswith("arch17")
    is_arch16 = architecture_label.startswith("arch16")
    is_skill_delta = is_arch16 or is_arch17 or is_arch18
    # Arch16_align/17_align/18_align = the same architectures plus a training-only head that names
    # the wrist patch holding the skill-end EEF; only the revision differs, so every other rule
    # below still sees them as Arch16/17/18.
    is_align = architecture_label.startswith(("arch16_align", "arch17_align", "arch18_align"))
    # Arch19 = Arch15 with the skill displacement (end - start xyz) as its Expert goal;
    # Arch20 = Arch15 without any Expert goal (= Arch13 + skill in Cond).
    is_arch20 = architecture_label.startswith("arch20")
    is_arch19 = architecture_label.startswith("arch19")
    is_arch15 = architecture_label.startswith("arch15")
    # "is_arch14" = the Expert-side goal family: Arch14, Arch15 (= Arch14 + skill in Cond) and Arch19.
    is_arch14 = architecture_label.startswith(("arch14", "arch15", "arch19"))
    # Arch14 = Arch13 + skill-end pose in the Expert AdaRMS; every Arch13 data/vision rule applies.
    is_arch13 = architecture_label.startswith(("arch13", "arch14", "arch15", "arch19", "arch20"))
    is_arch9 = is_arch9_1 or is_arch9_2
    is_arch10 = is_arch10_1 or is_arch10_2
    is_arch11 = is_arch11_1 or is_arch11_2
    is_arch12 = is_arch12_1 or is_arch12_2
    is_wrist_end_pose = is_arch9 or is_arch10 or is_arch11 or is_arch12 or is_skill_delta
    is_arch8 = is_arch8_1 or is_arch8_2
    is_layerwise = is_arch3 or is_arch4 or is_arch5 or is_arch6 or is_arch7 or is_arch8 or is_wrist_end_pose or is_arch13
    is_visual_bottleneck = is_arch1 or is_arch2
    architecture = (
        "layerwise_cond_bottleneck"
        if is_layerwise
        else ("fixed_visual_bottleneck" if is_visual_bottleneck else "cond_gemma")
    )
    architecture_revision = (
        ARCH20_REVISION if is_arch20 else
        ARCH19_REVISION if is_arch19 else
        ARCH18_ALIGN_REVISION if (is_arch18 and is_align) else
        ARCH17_ALIGN_REVISION if (is_arch17 and is_align) else
        ARCH16_ALIGN_REVISION if (is_arch16 and is_align) else
        ARCH18_REVISION if is_arch18 else
        ARCH17_REVISION if is_arch17 else
        ARCH16_REVISION if is_arch16 else
        ARCH15_REVISION if is_arch15 else
        ARCH14_REVISION if is_arch14 else
        ARCH13_REVISION if is_arch13 else
        ARCH12_2_REVISION if is_arch12_2 else
        ARCH12_1_REVISION if is_arch12_1 else
        ARCH11_2_REVISION if is_arch11_2 else
        ARCH11_1_REVISION if is_arch11_1 else
        ARCH10_2_REVISION if is_arch10_2 else
        ARCH10_1_REVISION if is_arch10_1 else
        ARCH9_2_REVISION if is_arch9_2 else
        ARCH9_1_REVISION if is_arch9_1 else
        ARCH8_2_REVISION if is_arch8_2 else
        ARCH8_1_REVISION if is_arch8_1 else
        ARCH7_REVISION if is_arch7 else
        ARCH6_REVISION if is_arch6 else
        ARCH5_REVISION if is_arch5 else
        ARCH4_REVISION if is_arch4 else
        ARCH3_REVISION if is_arch3 else
        ARCH2_REVISION if is_arch2 else
        ARCH1_REVISION if is_arch1 else ARCH0_REVISION
    )
    vision_conditioning_mode = (
        "layerwise_cond_bottleneck_cross_attention"
        if is_layerwise
        else (
            "fixed_bottleneck_cross_attention"
            if is_visual_bottleneck
            else "interleaved_cross_attention"
        )
    )
    cond_variant = expert_variant
    conditioning_route = "state_cond"

    max_state_dim = int(_at(config, "architecture", "max_state_dim", default=32))
    max_action_dim = int(_at(config, "architecture", "max_action_dim", default=32))
    if contract["state_dim"] > max_state_dim or contract["action_dim"] > max_action_dim:
        raise ValueError(
            "Dataset state/action dimensions exceed the configured pi0.5 projections: "
            f"dataset=({contract['state_dim']}, {contract['action_dim']}), "
            f"configured=({max_state_dim}, {max_action_dim})."
        )

    chunk_size = int(_at(config, "architecture", "chunk_size", default=10))
    requested_visual_bottleneck_tokens = int(
        _at(
            config,
            "architecture",
            "visual_bottleneck_tokens",
            default=4,
        )
    )
    if (is_visual_bottleneck or is_layerwise) and (
        requested_visual_bottleneck_tokens <= 0
        or (
            is_visual_bottleneck
            and requested_visual_bottleneck_tokens % 2 != 0
        )
    ):
        raise ValueError(
            "architecture.visual_bottleneck_tokens must be a positive even "
            "integer for Arch1/Arch2, or a positive integer for Arch3/Arch4/Arch5."
        )
    # Arch0 has no fixed visual bottleneck. Keep its serialized compatibility
    # value independent of an Arch1--Arch4 tuning value left in the YAML.
    visual_bottleneck_tokens = (
        requested_visual_bottleneck_tokens
        if (is_visual_bottleneck or is_layerwise)
        else 4
    )
    visual_bridge_last_n_layers = int(
        _at(
            config,
            "architecture",
            "visual_bridge_last_n_layers",
            default=1,
        )
    )
    if not 1 <= visual_bridge_last_n_layers <= 18:
        raise ValueError(
            "architecture.visual_bridge_last_n_layers must be within [1, 18]."
        )
    if (is_arch4 or is_arch5 or is_arch6 or is_arch7 or is_arch8 or is_wrist_end_pose or is_arch13) and visual_bridge_last_n_layers == 18:
        raise ValueError(
            "Arch4--Arch13 require visual_bridge_last_n_layers <= 17 so the "
            "skill-only motion core contains at least one Expert layer."
        )
    if not (is_arch2 or is_layerwise) and visual_bridge_last_n_layers != 1:
        raise ValueError(
            "architecture.visual_bridge_last_n_layers is Arch2/Arch3/Arch4-only; "
            "use 1 for Arch0/Arch1."
        )
    if "loss" in config:
        raise ValueError(
            "Stage1 uses a fixed flow objective; remove the legacy top-level "
            "loss key and control the trajectory auxiliary with "
            "cumulative_xyz_loss.enabled."
        )
    mask_actions_after_skill_end = as_bool(
        config.get("mask_actions_after_skill_end", False)
    )
    cumulative_xyz_config = config.get("cumulative_xyz_loss", {})
    if not isinstance(cumulative_xyz_config, dict):
        raise ValueError("cumulative_xyz_loss must be a mapping.")
    cumulative_xyz_loss_enabled = as_bool(
        cumulative_xyz_config.get("enabled", False)
    )
    cumulative_xyz_loss_weight = float(cumulative_xyz_config.get("weight", 0.5))
    if not math.isfinite(cumulative_xyz_loss_weight) or cumulative_xyz_loss_weight <= 0:
        raise ValueError("cumulative_xyz_loss.weight must be finite and positive.")
    skill_flow_config = config.get("skill_flow", {})
    if not isinstance(skill_flow_config, dict):
        raise ValueError("skill_flow must be a mapping.")
    unknown_skill_flow_keys = set(skill_flow_config) - {
        "weight",
        "chunk_multiplier",
        "latent_best_of_n",
    }
    if unknown_skill_flow_keys:
        raise ValueError(
            f"Unsupported skill_flow settings: {sorted(unknown_skill_flow_keys)}."
        )
    # Arch9--Arch12 use skill-end pose as an input, not an auxiliary spatial
    # readout. Ignore a leftover weight when switching from Arch5--Arch8.
    # ...except the _align labels, which spend it on their wrist patch objective.
    spatial_loss_weight = (
        None if (is_wrist_end_pose and not is_align) else architecture_config.get("spatial_loss_weight")
    )
    if spatial_loss_weight is not None and (
        "focus_uv_loss_weight" in architecture_config
        or "end_xyz_loss_weight" in architecture_config
    ):
        raise ValueError(
            "architecture.spatial_loss_weight cannot be combined with its older "
            "target-specific loss-weight keys."
        )
    focus_uv_loss_weight = float(
        architecture_config.get(
            "focus_uv_loss_weight",
            spatial_loss_weight if spatial_loss_weight is not None and (is_arch5 or is_arch6 or is_arch13) else 1.0,
        )
    )
    if not math.isfinite(focus_uv_loss_weight) or focus_uv_loss_weight <= 0:
        raise ValueError("architecture.focus_uv_loss_weight must be finite and positive.")
    if not (is_arch5 or is_arch6 or is_arch13) and focus_uv_loss_weight != 1.0:
        raise ValueError("architecture.focus_uv_loss_weight is configurable only for Arch5/Arch6/Arch13.")
    end_xyz_loss_weight = float(
        architecture_config.get(
            "end_xyz_loss_weight", spatial_loss_weight if spatial_loss_weight is not None and (is_arch7 or is_arch8) else 1.0
        )
    )
    if not math.isfinite(end_xyz_loss_weight) or end_xyz_loss_weight <= 0:
        raise ValueError("architecture.end_xyz_loss_weight must be finite and positive.")
    if not (is_arch7 or is_arch8) and end_xyz_loss_weight != 1.0:
        raise ValueError("architecture.end_xyz_loss_weight is configurable only for Arch7/Arch8.")
    wrist_patch_align_loss_weight = float(
        spatial_loss_weight if (is_align and spatial_loss_weight is not None) else 0.1
    )
    if not math.isfinite(wrist_patch_align_loss_weight) or wrist_patch_align_loss_weight <= 0:
        raise ValueError("architecture.spatial_loss_weight must be finite and positive.")
    if spatial_loss_weight is not None and not (
        is_arch5 or is_arch6 or is_arch7 or is_arch8 or is_arch13 or is_align
    ):
        raise ValueError(
            "architecture.spatial_loss_weight is configurable only for "
            "Arch5--Arch8/Arch13 and the _align labels."
        )
    if (is_arch5 or is_arch6 or is_arch7 or is_arch8 or is_arch13) and contract["focus_uv_path"] is None:
        raise FileNotFoundError(
            "Arch5--Arch8/Arch13 require skill_focus_uv.npz for occurrence indexing; rebuild the SkillVLA dataset "
            "with focus_uv.enabled=true."
        )
    if (is_arch5 or is_arch6 or is_arch8 or is_arch13) and contract["focus_uv_normalization"] != "minus_one_to_one":
        raise ValueError(
            "Arch5/Arch6/Arch8/Arch13 require skill_focus_uv_normalization='minus_one_to_one'; "
            f"got {contract['focus_uv_normalization']!r}."
        )
    if (is_arch8 or is_arch13) and foveated_vision_enabled:
        raise ValueError("Arch8/Arch13 use the original agent view: set vision.foveation.enabled=false.")
    end_pose_mode = str(architecture_config.get("end_pose_mode", "xyz")).strip().lower()
    if end_pose_mode not in {"xyz", "pose"}:
        raise ValueError("architecture.end_pose_mode must be xyz|pose.")
    if not (is_wrist_end_pose or is_arch13) and "end_pose_mode" in architecture_config:
        raise ValueError("architecture.end_pose_mode applies only to Arch9--Arch12.")
    if is_arch13 and not is_arch14 and end_pose_mode != "xyz":
        raise ValueError("Arch13 fixes architecture.end_pose_mode=xyz.")
    if is_arch14 and contract["state_dim"] < (3 if end_pose_mode == "xyz" else 6):
        raise ValueError("SkillVLA observation.state is too short for architecture.end_pose_mode.")
    if (is_skill_delta or is_arch19) and end_pose_mode != "xyz":
        raise ValueError("Arch16--Arch19 condition on skill start/end translations: set architecture.end_pose_mode: xyz.")
    if is_wrist_end_pose and foveated_vision_enabled:
        raise ValueError("Arch9--Arch12 are wrist-only: set vision.foveation.enabled=false.")
    if is_wrist_end_pose and contract["focus_uv_path"] is None:
        raise FileNotFoundError(
            "Arch9--Arch12 require skill_focus_uv.npz as the skill occurrence/end-state index."
        )
    if is_wrist_end_pose and contract["state_dim"] < (3 if end_pose_mode == "xyz" else 6):
        raise ValueError("SkillVLA observation.state is too short for architecture.end_pose_mode.")
    termination_config = config.get("termination_loss", {})
    if not isinstance(termination_config, dict):
        raise ValueError("termination_loss must be a mapping.")
    unknown_termination_keys = set(termination_config) - {"loss_weight", "target_sigma", "positive_weight"}
    if unknown_termination_keys:
        raise ValueError(f"Unsupported termination_loss settings: {sorted(unknown_termination_keys)}.")
    if termination_config and not (is_arch8_2 or is_arch9_2 or is_arch10_2 or is_arch11_2 or is_arch12_2):
        raise ValueError("termination_loss is configurable only for Arch8_2/Arch9_2/Arch10_2/Arch11_2/Arch12_2.")
    # Keep the serialized compatibility default at 1.0 when the block is
    # absent. The user-facing YAML advertises the recommended 0.1 setting for
    # new _2 runs explicitly, so old configs retain their original names.
    termination_loss_weight = float(termination_config.get("loss_weight", 1.0))
    termination_target_sigma = float(termination_config.get("target_sigma", 2.0))
    termination_positive_weight = float(termination_config.get("positive_weight", 2.0))
    if not math.isfinite(termination_loss_weight) or termination_loss_weight <= 0:
        raise ValueError("termination_loss.loss_weight must be finite and positive.")
    if not math.isfinite(termination_target_sigma) or termination_target_sigma < 0:
        raise ValueError("termination_loss.target_sigma must be finite and nonnegative.")
    if not math.isfinite(termination_positive_weight) or termination_positive_weight <= 0:
        raise ValueError("termination_loss.positive_weight must be finite and positive.")
    skill_flow_enabled = architecture_label in {
        "arch0_skill",
        "arch0_skill_chunk",
        "arch1_skill",
        "arch1_skill_chunk",
        "arch2_skill",
        "arch2_skill_chunk",
        "arch3_skill",
        "arch3_skill_chunk",
        "arch4_skill",
        "arch4_skill_chunk",
        "arch5_skill",
        "arch5_skill_chunk",
        "arch6_skill",
        "arch6_skill_chunk",
        "arch7_skill",
        "arch7_skill_chunk",
        "arch8_1_skill",
        "arch8_1_skill_chunk",
        "arch8_2_skill",
        "arch8_2_skill_chunk",
        "arch9_1_skill",
        "arch9_1_skill_chunk",
        "arch9_2_skill",
        "arch9_2_skill_chunk",
        "arch10_1_skill",
        "arch10_1_skill_chunk",
        "arch10_2_skill",
        "arch10_2_skill_chunk",
        "arch11_1_skill",
        "arch11_1_skill_chunk",
        "arch11_2_skill",
        "arch11_2_skill_chunk",
        "arch12_1_skill",
        "arch12_1_skill_chunk",
        "arch12_2_skill",
        "arch12_2_skill_chunk",
        "arch13_skill",
        "arch13_skill_chunk",
        "arch14_skill",
        "arch14_skill_chunk",
        "arch15_skill",
        "arch15_skill_chunk",
        "arch16_skill",
        "arch16_skill_chunk",
        "arch16_align_skill",
        "arch16_align_skill_chunk",
        "arch17_skill",
        "arch17_skill_chunk",
        "arch17_align_skill",
        "arch17_align_skill_chunk",
        "arch18_skill",
        "arch18_skill_chunk",
        "arch18_align_skill",
        "arch18_align_skill_chunk",
        "arch19_skill",
        "arch19_skill_chunk",
        "arch20_skill",
        "arch20_skill_chunk",
    }
    skill_flow_weight = float(skill_flow_config.get("weight", 1.0))
    if not math.isfinite(skill_flow_weight) or skill_flow_weight <= 0:
        raise ValueError("skill_flow.weight must be finite and positive.")
    skill_flow_chunk_multiplier = int(skill_flow_config.get("chunk_multiplier", 3))
    if skill_flow_chunk_multiplier <= 0:
        raise ValueError("skill_flow.chunk_multiplier must be positive.")
    latent_best_of_n = skill_flow_config.get("latent_best_of_n", {})
    if not isinstance(latent_best_of_n, dict):
        raise ValueError("skill_flow.latent_best_of_n must be a mapping.")
    unknown_latent_keys = set(latent_best_of_n) - {
        "enabled",
        "candidates",
        "top_k",
        "timesteps",
        "ranking",
        "fp32",
    }
    if unknown_latent_keys:
        raise ValueError(
            "Unsupported skill_flow.latent_best_of_n settings: "
            f"{sorted(unknown_latent_keys)}."
        )
    skill_flow_latent_best_of_n_enabled = as_bool(
        latent_best_of_n.get("enabled", False)
    )
    skill_flow_latent_candidates = int(latent_best_of_n.get("candidates", 5))
    skill_flow_latent_top_k = int(latent_best_of_n.get("top_k", 1))
    skill_flow_latent_assignment_timesteps = int(
        latent_best_of_n.get("timesteps", 2)
    )
    skill_flow_latent_ranking_route = str(
        latent_best_of_n.get("ranking", "main")
    ).strip().lower().replace("-", "_")
    # Sub-options are inert when the latent probe is disabled. This keeps a
    # ready-to-enable YAML block from changing or invalidating ordinary runs.
    skill_flow_latent_fp32 = (
        as_bool(latent_best_of_n.get("fp32", False))
        if skill_flow_latent_best_of_n_enabled
        else False
    )
    if skill_flow_latent_candidates <= 0:
        raise ValueError("skill_flow.latent_best_of_n.candidates must be positive.")
    if not 1 <= skill_flow_latent_top_k <= skill_flow_latent_candidates:
        raise ValueError(
            "skill_flow.latent_best_of_n.top_k must be within [1, candidates]."
        )
    if skill_flow_latent_assignment_timesteps <= 0:
        raise ValueError("skill_flow.latent_best_of_n.timesteps must be positive.")
    if skill_flow_latent_ranking_route not in {"main", "skill_only"}:
        raise ValueError(
            "skill_flow.latent_best_of_n.ranking must be main|skill_only."
        )
    if skill_flow_latent_best_of_n_enabled and architecture_label not in {
        "arch0_skill",
        "arch0_skill_chunk",
        "arch1_skill",
        "arch1_skill_chunk",
        "arch2_skill",
        "arch2_skill_chunk",
        "arch3_skill",
        "arch3_skill_chunk",
        "arch4_skill",
        "arch4_skill_chunk",
        "arch5_skill",
        "arch5_skill_chunk",
        "arch6_skill",
        "arch6_skill_chunk",
        "arch7_skill",
        "arch7_skill_chunk",
        "arch8_1_skill",
        "arch8_1_skill_chunk",
        "arch8_2_skill",
        "arch8_2_skill_chunk",
        "arch9_1_skill",
        "arch9_1_skill_chunk",
        "arch9_2_skill",
        "arch9_2_skill_chunk",
        "arch10_1_skill",
        "arch10_1_skill_chunk",
        "arch10_2_skill",
        "arch10_2_skill_chunk",
        "arch11_1_skill",
        "arch11_1_skill_chunk",
        "arch11_2_skill",
        "arch11_2_skill_chunk",
        "arch12_1_skill",
        "arch12_1_skill_chunk",
        "arch12_2_skill",
        "arch12_2_skill_chunk",
        "arch13_skill",
        "arch13_skill_chunk",
        "arch14_skill",
        "arch14_skill_chunk",
        "arch15_skill",
        "arch15_skill_chunk",
        "arch16_skill",
        "arch16_skill_chunk",
        "arch16_align_skill",
        "arch16_align_skill_chunk",
        "arch17_skill",
        "arch17_skill_chunk",
        "arch17_align_skill",
        "arch17_align_skill_chunk",
        "arch18_skill",
        "arch18_skill_chunk",
        "arch18_align_skill",
        "arch18_align_skill_chunk",
        "arch19_skill",
        "arch19_skill_chunk",
        "arch20_skill",
        "arch20_skill_chunk",
    }:
        raise ValueError(
            "skill_flow.latent_best_of_n is supported only for "
            "architecture.name=arch0_skill|arch0_skill_chunk|"
            "arch1_skill|arch1_skill_chunk|arch2_skill|arch2_skill_chunk|"
            "arch3_skill|arch3_skill_chunk|arch4_skill|arch4_skill_chunk."
        )
    skill_flow_target = (
        "extended_chunk"
        if architecture_label.endswith("_skill_chunk")
        else "canonical"
    )
    skill_flow_state_conditioned = False
    skill_flow_max_length = (
        int(contract["skill_observed_max_length"])
        if skill_flow_target == "canonical"
        else chunk_size * skill_flow_chunk_multiplier
    )
    if architecture_label in {
        "arch0_skill",
        "arch1_skill",
        "arch2_skill",
        "arch3_skill",
        "arch4_skill",
        "arch5_skill",
        "arch6_skill",
        "arch7_skill",
        "arch8_1_skill",
        "arch8_2_skill",
        "arch9_1_skill",
        "arch9_2_skill",
        "arch10_1_skill",
        "arch10_2_skill",
        "arch11_1_skill",
        "arch11_2_skill",
        "arch12_1_skill",
        "arch12_2_skill",
        "arch13_skill",
        "arch14_skill",
        "arch15_skill",
        "arch16_skill",
        "arch17_skill",
        "arch18_skill",
        "arch16_align_skill",
        "arch17_align_skill",
        "arch18_align_skill",
        "arch19_skill",
        "arch20_skill",
    }:
        if training_skill_source != "gt":
            raise ValueError(
                "architecture.name=*_skill currently requires "
                "action_conditioning.training_skill_source=gt."
            )
        if skill_flow_max_length <= 0:
            raise ValueError(
                "architecture.name=*_skill requires a positive "
                "skill_observed_max_length in the dataset info.json."
            )
    n_action_steps = int(
        _at(config, "execution", "action_steps", default=chunk_size)
    )
    transition_jitter = as_bool(
        _at(config, "transition_randomization", "enabled", default=True)
    )
    jitter_pmax = contract["jitter_pmax"] if transition_jitter else 0
    directional_jitter = {
        name: contract[f"jitter_{name}_pmax"] if transition_jitter else 0
        for name in ("early_start", "late_start", "early_end", "late_end")
    }
    jitter_distribution = contract["jitter_distribution"] if transition_jitter else "half_normal"
    if jitter_distribution not in {"half_normal", "uniform"}:
        raise ValueError(f"Unsupported dataset jitter distribution: {jitter_distribution!r}")

    batch_size = int(_at(config, "training", "dataloader", "batch_size", default=16))
    num_gpus = int(_at(config, "training", "dataloader", "gpus", default=1))
    base_lr = float(_at(config, "training", "optimizer", "base_lr", default=2.5e-5))
    optimizer_config = _at(config, "training", "optimizer", default={})
    if "dino_lr" in optimizer_config:
        raise ValueError(
            "training.optimizer.dino_lr was replaced by the relative dino_lr_scale."
        )
    if "terminator_lr_scale" in optimizer_config:
        raise ValueError(
            "training.optimizer.terminator_lr_scale was removed; Stage1 trains "
            "only the action model."
        )
    dino_lr_scale = float(
        _at(config, "training", "optimizer", "dino_lr_scale", default=0.1)
    )
    if dino_lr_scale <= 0.0:
        raise ValueError("training.optimizer.dino_lr_scale must be positive.")
    # Muon probe: 2D hidden matrices switch to Muon (AdamW keeps the rest).
    # base_lr/dino_lr_scale are reused unchanged (match_rms_adamw scaling).
    use_muon = as_bool(_at(config, "training", "optimizer", "muon", default=False))

    predictor_contract = {
        "skill_predictor_vlm_variant": "gemma_2b",
        "skill_predictor_image_size": 224,
        "skill_predictor_reader_tokens": 4,
        "skill_predictor_reader_depth": 2,
        "skill_predictor_reader_heads": 8,
        "skill_predictor_all_layers": False,
        "skill_predictor_detach_vlm": True,
        "skill_predictor_lora": False,
        "skill_predictor_lora_targets": "q,k,v,o",
        "skill_predictor_lora_rank": 8,
        "skill_predictor_lora_alpha": 16.0,
        "skill_predictor_lora_dropout": 0.0,
        "skill_predictor_deadzone_frac": 0.0,
        "skill_predictor_attend_image": True,
        "skill_predictor_attend_language": True,
        "tokenizer_max_length": 200,
    }
    if predictor_checkpoint is not None:
        predictor_contract = _predictor_contract_from_checkpoint(
            predictor_checkpoint,
            levels=contract["levels"],
        )
    run_name = f"bs{batch_size}_{source}_{run_tag}_{architecture_label}"
    if (is_visual_bottleneck or is_layerwise) and visual_bottleneck_tokens != 4:
        run_name = f"{run_name}_vtok{visual_bottleneck_tokens}"
    if (is_arch2 or is_layerwise) and visual_bridge_last_n_layers != 1:
        run_name = f"{run_name}_vlast{visual_bridge_last_n_layers}"
    if (is_arch5 or is_arch6 or is_arch13) and focus_uv_loss_weight != 1.0:
        run_name = f"{run_name}_uv{focus_uv_loss_weight:g}".replace(".", "p")
    if (is_arch7 or is_arch8) and end_xyz_loss_weight != 1.0:
        run_name = f"{run_name}_xyz{end_xyz_loss_weight:g}".replace(".", "p")
    if (is_arch8_2 or is_arch9_2 or is_arch10_2 or is_arch11_2 or is_arch12_2) and termination_loss_weight != 1.0:
        run_name = f"{run_name}_term{termination_loss_weight:g}".replace(".", "p")
    if (is_arch8_2 or is_arch9_2 or is_arch10_2 or is_arch11_2 or is_arch12_2) and termination_target_sigma != 2.0:
        run_name = f"{run_name}_ts{termination_target_sigma:g}".replace(".", "p")
    if (is_arch8_2 or is_arch9_2 or is_arch10_2 or is_arch11_2 or is_arch12_2) and termination_positive_weight != 2.0:
        run_name = f"{run_name}_pw{termination_positive_weight:g}".replace(".", "p")
    if is_wrist_end_pose or is_arch14:
        run_name = f"{run_name}_end{end_pose_mode}"
    if training_skill_source == "predictor":
        run_name = f"{run_name}_pretrained_predictor"
    if mask_actions_after_skill_end:
        run_name = f"{run_name}_skillendmask"
    if cumulative_xyz_loss_enabled:
        cumulative_weight_label = f"{cumulative_xyz_loss_weight:g}".replace(
            ".", "p"
        )
        run_name = f"{run_name}_cumxyz{cumulative_weight_label}"
    if skill_flow_latent_best_of_n_enabled:
        run_name = (
            f"{run_name}_zbest{skill_flow_latent_candidates}"
            f"k{skill_flow_latent_top_k}m{skill_flow_latent_assignment_timesteps}"
        )
        if skill_flow_latent_ranking_route == "main":
            run_name = f"{run_name}_rank"
        if skill_flow_latent_fp32:
            run_name = f"{run_name}_zfp32"
    if foveated_vision_enabled:
        foveation_tag = (
            "partial_fov" if foveation_mode == "partial_fov" else "crop_fov"
        )
        run_name = f"{run_name}_{foveation_tag}"
        if foveation_peripheral_mode == "black":
            run_name = f"{run_name}_black"
    if foveation_randomization_enabled:
        run_name = f"{run_name}_rand"
    if use_muon:
        # Muon A/B runs must never collide with the AdamW output directory.
        run_name = f"{run_name}_muon"
    if run_suffix:
        # The user suffix is always last so the automatic batch/architecture
        # naming contract stays machine-readable.
        run_name = f"{run_name}_{run_suffix}"

    levels = contract["levels"]
    training_config = config.get("training", {})
    if "vsa_debug_schedule" in training_config or "vsa_debug_steps" in training_config:
        raise ValueError(
            "training.vsa_debug_schedule/vsa_debug_steps were replaced by "
            "training.vsa_debug.{every,initial}."
        )
    vsa_debug_config = training_config.get("vsa_debug", {})
    if not isinstance(vsa_debug_config, dict):
        raise ValueError("training.vsa_debug must be a mapping.")
    vsa_debug_every = int(vsa_debug_config.get("every", 0))
    vsa_debug_initial = [
        int(step) for step in as_list(vsa_debug_config.get("initial", []))
    ]
    if vsa_debug_every < 0:
        raise ValueError("training.vsa_debug.every must be non-negative.")
    if any(step <= 0 for step in vsa_debug_initial):
        raise ValueError("training.vsa_debug.initial entries must be positive.")
    if sorted(set(vsa_debug_initial)) != vsa_debug_initial:
        raise ValueError(
            "training.vsa_debug.initial must be sorted and contain no duplicates."
        )
    periodic_debug_steps = (
        range(vsa_debug_every, training_steps + 1, vsa_debug_every)
        if vsa_debug_every > 0
        else ()
    )
    vsa_debug_schedule = sorted(
        set(vsa_debug_initial).union(periodic_debug_steps)
    )
    settings = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "skillvla_dataset_dir": dataset_dir,
        "dataset_relabeled": dataset_relabeled,
        "repo_id": f"dohyeon/{source}",
        "pi_base": pi_base,
        "fsq_path": fsq_path,
        "tokenizer_path": tokenizer_path,
        "dino_model_path": dino_model,
        "dino_image_size": int(_at(config, "vision", "image_size", default=224)),
        "freeze_vision_encoder": freeze_vision_encoder,
        "dino_lr_scale": dino_lr_scale,
        "foveated_vision_enabled": foveated_vision_enabled,
        "foveation_randomization_enabled": foveation_randomization_enabled,
        "foveation_mode": foveation_mode,
        "foveation_crop_size": foveation_crop_size,
        "foveation_output_size": foveation_output_size,
        "foveation_inner_box_enabled": foveation_inner_box_enabled,
        "foveation_inner_box_mode": foveation_inner_box_mode,
        "foveation_inner_box_size": foveation_inner_box_size,
        "foveation_inner_box_line_width": foveation_inner_box_line_width,
        "foveation_shape": foveation_shape,
        "foveation_sharp_size": foveation_sharp_size,
        "foveation_feather": foveation_feather,
        "foveation_peripheral_mode": foveation_peripheral_mode,
        "foveation_peripheral_blur_radius": foveation_peripheral_blur_radius,
        "foveation_color_enabled": foveation_color_enabled,
        "foveation_brightness_min": foveation_brightness[0],
        "foveation_brightness_max": foveation_brightness[1],
        "foveation_contrast_min": foveation_contrast[0],
        "foveation_contrast_max": foveation_contrast[1],
        "foveation_saturation_min": foveation_saturation[0],
        "foveation_saturation_max": foveation_saturation[1],
        "foveation_hue_min": foveation_hue[0],
        "foveation_hue_max": foveation_hue[1],
        "foveation_crop_enabled": foveation_crop_enabled,
        "foveation_crop_offset_min_px": foveation_crop_offset[0],
        "foveation_crop_offset_max_px": foveation_crop_offset[1],
        "foveation_inner_box_offset_min_px": foveation_inner_box_offset[0],
        "foveation_inner_box_offset_max_px": foveation_inner_box_offset[1],
        "foveation_input_blur_enabled": foveation_input_blur_enabled,
        "foveation_input_blur_min_radius": foveation_input_blur_radius[0],
        "foveation_input_blur_max_radius": foveation_input_blur_radius[1],
        "architecture": architecture,
        "architecture_label": architecture_label,
        "architecture_revision": architecture_revision,
        "vision_conditioning_mode": vision_conditioning_mode,
        "action_expert_variant": expert_variant,
        "cond_encoder_variant": cond_variant,
        "conditioning_route": conditioning_route,
        # The token count is the intentionally small Arch1--Arch4 capacity knob;
        # the remaining interface geometry stays fixed. Serializing every value
        # keeps checkpoints self-describing and preserves 4-token checkpoints.
        "visual_bottleneck_tokens": visual_bottleneck_tokens,
        "visual_bottleneck_width": 256,
        "visual_bottleneck_heads": 4,
        "visual_bridge_heads": 8,
        "visual_bridge_gate_init": 0.01,
        "visual_bridge_last_n_layers": visual_bridge_last_n_layers,
        "cond_focus_uv_loss_weight": focus_uv_loss_weight,
        "wrist_patch_align_loss_weight": wrist_patch_align_loss_weight,
        "cond_end_xyz_loss_weight": end_xyz_loss_weight,
        "bottleneck_termination_loss_weight": termination_loss_weight,
        "bottleneck_termination_target_sigma": termination_target_sigma,
        "bottleneck_termination_positive_weight": termination_positive_weight,
        "skill_end_pose_mode": end_pose_mode,
        "skill_fsq_levels": "[" + ",".join(str(level) for level in levels) + "]",
        "skill_vocab_size": math.prod(levels),
        "skill_code_space_id": contract["skill_code_space_id"],
        "transition_jitter_pmax": jitter_pmax,
        "transition_jitter_early_start_pmax": directional_jitter["early_start"],
        "transition_jitter_late_start_pmax": directional_jitter["late_start"],
        "transition_jitter_early_end_pmax": directional_jitter["early_end"],
        "transition_jitter_late_end_pmax": directional_jitter["late_end"],
        "transition_jitter_distribution": jitter_distribution,
        "training_skill_source": training_skill_source,
        "skill_predictor_checkpoint_path": predictor_checkpoint or "",
        "skill_predictor_all_layers": predictor_contract["skill_predictor_all_layers"],
        "skill_predictor_detach_vlm": predictor_contract["skill_predictor_detach_vlm"],
        "skill_predictor_lora": predictor_contract["skill_predictor_lora"],
        "skill_predictor_lora_targets": predictor_contract["skill_predictor_lora_targets"],
        "skill_predictor_lora_rank": predictor_contract["skill_predictor_lora_rank"],
        "skill_predictor_lora_alpha": predictor_contract["skill_predictor_lora_alpha"],
        "skill_predictor_lora_dropout": predictor_contract["skill_predictor_lora_dropout"],
        "skill_predictor_reader_tokens": predictor_contract["skill_predictor_reader_tokens"],
        "skill_predictor_reader_depth": predictor_contract["skill_predictor_reader_depth"],
        "skill_predictor_reader_heads": predictor_contract["skill_predictor_reader_heads"],
        "skill_predictor_deadzone_frac": predictor_contract["skill_predictor_deadzone_frac"],
        "skill_predictor_attend_image": predictor_contract["skill_predictor_attend_image"],
        "skill_predictor_attend_language": predictor_contract["skill_predictor_attend_language"],
        "tokenizer_max_length": predictor_contract["tokenizer_max_length"],
        "max_state_dim": max_state_dim,
        "max_action_dim": max_action_dim,
        "proprio_grounding": contract["proprio_grounding"],
        "chunk_size": chunk_size,
        "mask_actions_after_skill_end": mask_actions_after_skill_end,
        "cumulative_xyz_loss_enabled": cumulative_xyz_loss_enabled,
        "cumulative_xyz_loss_weight": cumulative_xyz_loss_weight,
        "skill_flow_enabled": skill_flow_enabled,
        "skill_flow_weight": skill_flow_weight,
        "skill_flow_max_length": skill_flow_max_length if skill_flow_enabled else 0,
        "skill_flow_target": skill_flow_target,
        "skill_flow_state_conditioned": skill_flow_state_conditioned,
        "skill_flow_chunk_multiplier": skill_flow_chunk_multiplier,
        "skill_flow_latent_best_of_n_enabled": skill_flow_latent_best_of_n_enabled,
        "skill_flow_latent_candidates": skill_flow_latent_candidates,
        "skill_flow_latent_top_k": skill_flow_latent_top_k,
        "skill_flow_latent_assignment_timesteps": skill_flow_latent_assignment_timesteps,
        "skill_flow_latent_ranking_route": skill_flow_latent_ranking_route,
        "skill_flow_latent_fp32": skill_flow_latent_fp32,
        # The probe deliberately fixes geometry/scale to avoid extra tuning knobs.
        "skill_flow_latent_dim": 2,
        "skill_flow_latent_distribution": "uniform_square",
        "skill_flow_latent_gain_init": 0.1,
        "n_action_steps": n_action_steps,
        "min_period": float(_at(config, "flow", "min_period", default=4e-3)),
        "max_period": float(_at(config, "flow", "max_period", default=4.0)),
        "time_sampling_beta_alpha": float(
            _at(config, "flow", "beta_alpha", default=1.5)
        ),
        "time_sampling_beta_beta": float(
            _at(config, "flow", "beta_beta", default=1.0)
        ),
        "time_sampling_scale": float(_at(config, "flow", "scale", default=0.999)),
        "time_sampling_offset": float(_at(config, "flow", "offset", default=0.001)),
        "pt_run_name": run_name,
        "pt_output_dir": outputs_root / "skillVLA_stage1" / (
            "VSA" if config.get("stage1_component") == "VSA" else ""
        ) / run_name,
        "batch_size": batch_size,
        "num_workers": int(
            _at(config, "training", "dataloader", "workers", default=2)
        ),
        "num_gpus": num_gpus,
        "gradient_checkpointing": as_bool(
            _at(config, "training", "gradient_checkpointing", default=True)
        ),
        "vsa_debug_steps": 0,
        "vsa_debug_schedule": "["
        + ",".join(str(step) for step in vsa_debug_schedule)
        + "]",
        "lr": base_lr * num_gpus,
        "use_muon": use_muon,
        "steps": training_steps,
        "scheduler_mode": scheduler_mode,
        "scheduler_warmup_steps": scheduler_warmup_steps,
        "scheduler_decay_steps": scheduler_decay_steps,
        "log_freq": int(
            _at(config, "training", "schedule", "log_every", default=100)
        ),
        "save_freq": int(
            _at(config, "training", "schedule", "save_every", default=5000)
        ),
        "wandb_enable": as_bool(
            _at(config, "logging", "wandb", "enable", default=True)
        ),
        "wandb_project": str(
            _at(config, "logging", "wandb", "project", default="VLA_stage1")
        ),
        "train_partition": ",".join(as_list(config.get("train_partition", ["big"]))) or "big",
        "train_qos": str(config.get("train_qos", "big_qos")),
        "train_gres": str(_at(config, "slurm", "gres", default="gpu:1")),
        "train_cpus_per_task": int(_at(config, "slurm", "cpus", default=16)),
        "train_mem": str(_at(config, "slurm", "memory", default="256G")),
        "train_time": str(_at(config, "slurm", "time", default="48:00:00")),
        "train_nodelist": str(config.get("train_nodelist", "")),
        "train_exclude_nodes": ",".join(as_list(config.get("train_exclude_nodes", []))),
    }
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--architecture",
        type=str.lower,
        choices=SUPPORTED_ARCHITECTURES,
        help="Override architecture.name without editing the YAML.",
    )
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    config = load_stage1_component_config(args.config)
    if args.architecture:
        config.setdefault("architecture", {})["name"] = args.architecture
    settings = build_settings(config)
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
