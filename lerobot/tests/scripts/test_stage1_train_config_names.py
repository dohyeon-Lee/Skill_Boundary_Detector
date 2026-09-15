import json
import sys
from pathlib import Path

import pytest


_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1/src"
)
sys.path.insert(0, str(_SRC))
from stage1_train_config import build_settings  # noqa: E402


def _config(tmp_path: Path, architecture: str = "arch0") -> dict:
    project = tmp_path / "project"
    run = "FSQ333_test"
    dataset = project / "dataset/skillvla_dataset/source" / run / "skillvla"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "skill_fsq_levels": [3, 3, 3],
                "skill_code_space_id": "FSQ333_test",
                "skill_observed_max_length": 120,
                "skill_pmax": 15,
                "skill_jitter_early_start_pmax": 10,
                "skill_jitter_late_start_pmax": 5,
                "skill_jitter_early_end_pmax": 10,
                "skill_jitter_late_end_pmax": 5,
                "skill_jitter_distribution": "half_normal",
                "features": {
                    "observation.state": {"shape": [8]},
                    "action": {"shape": [7]},
                },
            }
        )
    )
    pi_base = project / "models/pi05_base"
    dino = project / "models/dino"
    pi_base.mkdir(parents=True)
    dino.mkdir(parents=True)
    (pi_base / "model.safetensors").touch()
    return {
        "project_root": str(project),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "dataset": {
            "skillvla_root": "skillvla_dataset",
            "source": "source",
            "run": run,
        },
        "warm_start": {"pi_base": "models/pi05_base"},
        "vision": {"dino_model": "models/dino"},
        "architecture": {
            "name": architecture,
            "expert_variant": "gemma_300m",
            "max_state_dim": 32,
            "max_action_dim": 32,
            "chunk_size": 10,
        },
        "skill_flow": {
            "weight": 1.0,
            "chunk_multiplier": 3,
            "latent_best_of_n": {"enabled": False},
        },
        "training": {"optimizer": {"dino_lr_scale": 0.1}},
    }


@pytest.mark.parametrize(
    ("label", "enabled", "target", "length"),
    [
        ("arch0", False, "canonical", 0),
        ("arch0_skill", True, "canonical", 120),
        ("arch0_skill_chunk", True, "extended_chunk", 30),
        ("arch1", False, "canonical", 0),
        ("arch1_skill", True, "canonical", 120),
        ("arch1_skill_chunk", True, "extended_chunk", 30),
        ("arch2", False, "canonical", 0),
        ("arch2_skill", True, "canonical", 120),
        ("arch2_skill_chunk", True, "extended_chunk", 30),
    ],
)
def test_stage1_resolves_retained_arch0_and_arch1_modes(
    tmp_path: Path,
    label: str,
    enabled: bool,
    target: str,
    length: int,
) -> None:
    settings = build_settings(_config(tmp_path, label))

    is_arch1 = label.startswith("arch1")
    is_arch2 = label.startswith("arch2")
    is_visual_bottleneck = is_arch1 or is_arch2
    assert settings["architecture"] == (
        "fixed_visual_bottleneck" if is_visual_bottleneck else "cond_gemma"
    )
    assert settings["architecture_revision"] == (
        "late_visual_bottleneck_v1"
        if is_arch2
        else ("fixed_visual_bottleneck_v1" if is_arch1 else "skillvla_real_v1")
    )
    assert settings["vision_conditioning_mode"] == (
        "fixed_bottleneck_cross_attention"
        if is_visual_bottleneck
        else "interleaved_cross_attention"
    )
    assert settings["architecture_label"] == label
    assert settings["conditioning_route"] == "state_cond"
    assert settings["cond_encoder_variant"] == "gemma_300m"
    assert settings["skill_flow_enabled"] is enabled
    assert settings["skill_flow_target"] == target
    assert settings["skill_flow_state_conditioned"] is False
    assert settings["skill_flow_max_length"] == length
    assert settings["visual_bridge_last_n_layers"] == 1
    assert settings["pt_run_name"].endswith(f"_{label}")


def test_arch1_fixed_visual_interface_is_exported(tmp_path: Path) -> None:
    settings = build_settings(_config(tmp_path, "arch1"))

    assert settings["visual_bottleneck_tokens"] == 4
    assert settings["visual_bottleneck_width"] == 256
    assert settings["visual_bottleneck_heads"] == 4
    assert settings["visual_bridge_heads"] == 8
    assert settings["visual_bridge_gate_init"] == pytest.approx(0.01)


def test_arch2_bridge_depth_override_is_validated_and_named(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch2_skill")
    config["architecture"]["visual_bridge_last_n_layers"] = 4

    settings = build_settings(config)

    assert settings["visual_bridge_last_n_layers"] == 4
    assert "_arch2_skill_vlast4" in settings["pt_run_name"]


def test_removed_architecture_names_fail_at_resolution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="architecture.name"):
        build_settings(_config(tmp_path, "removed_mode"))


def test_removed_architecture_knobs_are_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["architecture"]["vsa"] = {"visual_latents_per_camera": 32}

    with pytest.raises(ValueError, match="unsupported architecture keys"):
        build_settings(config)


def test_skill_chunk_latent_probe_and_suffix(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch0_skill_chunk")
    config["mask_actions_after_skill_end"] = True
    config["skill_flow"]["latent_best_of_n"] = {
        "enabled": True,
        "candidates": 5,
        "top_k": 1,
        "timesteps": 2,
        "ranking": "main",
        "fp32": True,
    }
    settings = build_settings(config)

    assert settings["skill_flow_latent_best_of_n_enabled"] is True
    assert settings["skill_flow_latent_ranking_route"] == "main"
    assert settings["skill_flow_latent_fp32"] is True
    assert settings["pt_run_name"].endswith(
        "_arch0_skill_chunk_skillendmask_zbest5k1m2_rank_zfp32"
    )


def test_arch0_rejects_latent_probe(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch0")
    config["skill_flow"]["latent_best_of_n"] = {"enabled": True}
    with pytest.raises(ValueError, match="supported only"):
        build_settings(config)


def test_foveated_training_requires_and_exports_focus_contract(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch0")
    config["vision"]["foveation"] = {
        "enabled": True,
        "shape": "circle",
        "sharp_size": 80,
        "feather": 12,
        "blur_radius": 7.0,
        "randomization": {
            "enabled": True,
            "color": {
                "enabled": True,
                "brightness": [0.7, 1.3],
                "contrast": [0.8, 1.2],
                "saturation": [0.9, 1.1],
                "hue": [-0.2, 0.2],
            },
            "crop": {"enabled": True, "offset_px": [-16, 16]},
            "blur": {"enabled": True, "blur_radius": [0.5, 3.0]},
        },
    }
    project = Path(config["project_root"])
    run_dir = project / "dataset/skillvla_dataset/source/FSQ333_test"
    info_path = run_dir / "skillvla/meta/info.json"
    info = json.loads(info_path.read_text())
    focus_path = run_dir / "skill_focus_uv.npz"
    focus_path.touch()
    info.update(
        {
            "skill_focus_uv_path": str(focus_path),
            "skill_focus_uv_camera": "agentview",
            "skill_focus_uv_normalization": "minus_one_to_one",
        }
    )
    info_path.write_text(json.dumps(info))

    settings = build_settings(config)

    assert settings["foveated_vision_enabled"] is True
    assert settings["foveation_randomization_enabled"] is True
    assert settings["foveation_shape"] == "circle"
    assert settings["foveation_color_enabled"] is True
    assert settings["foveation_crop_enabled"] is True
    assert settings["foveation_input_blur_enabled"] is True
    assert settings["foveation_hue_min"] == pytest.approx(-0.2)
    assert settings["foveation_input_blur_max_radius"] == pytest.approx(3.0)
    assert settings["pt_run_name"].endswith("_arch0_partial_fov_rand")


def test_crop_inner_blur_training_contract_and_suffix(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch0_skill")
    config["vision"]["foveation"] = {
        "enabled": True,
        "mode": "crop",
        "crop_size": 128,
        "output_size": 224,
        "inner_box": {
            "enabled": True,
            "mode": "blur",
            "size": 32,
            "line_width": 3,
        },
        "shape": "square",
        "sharp_size": 96,
        "feather": 20,
        "blur_radius": 8.0,
        "randomization": {
            "enabled": True,
            "crop": {
                "enabled": True,
                "offset_px": [-40, 40],
                "inner_box_offset_px": [-4, 4],
            },
        },
    }
    project = Path(config["project_root"])
    run_dir = project / "dataset/skillvla_dataset/source/FSQ333_test"
    info_path = run_dir / "skillvla/meta/info.json"
    info = json.loads(info_path.read_text())
    focus_path = run_dir / "skill_focus_uv.npz"
    focus_path.touch()
    info.update(
        {
            "skill_focus_uv_path": str(focus_path),
            "skill_focus_uv_camera": "agentview",
            "skill_focus_uv_normalization": "minus_one_to_one",
        }
    )
    info_path.write_text(json.dumps(info))

    settings = build_settings(config)

    assert settings["foveation_mode"] == "crop"
    assert settings["foveation_crop_size"] == 128
    assert settings["foveation_output_size"] == 224
    assert settings["foveation_inner_box_enabled"] is True
    assert settings["foveation_inner_box_mode"] == "blur"
    assert settings["foveation_inner_box_size"] == 32
    assert settings["foveation_inner_box_offset_min_px"] == -4
    assert settings["foveation_inner_box_offset_max_px"] == 4
    assert settings["pt_run_name"].endswith("_arch0_skill_crop_fov_rand")


def test_foveated_training_rejects_dataset_without_focus_artifact(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "arch0")
    config["vision"]["foveation"] = {"enabled": True}

    with pytest.raises(FileNotFoundError, match="skill_focus_uv.npz"):
        build_settings(config)


def test_foveation_and_randomization_master_switches_are_independent(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "arch0_skill")
    config["vision"]["foveation"] = {
        "enabled": False,
        "randomization": {
            "enabled": False,
            "color": {"enabled": True},
            "crop": {"enabled": True},
            "blur": {"enabled": True},
        },
    }
    baseline = build_settings(config)
    assert baseline["foveated_vision_enabled"] is False
    assert baseline["foveation_randomization_enabled"] is False
    assert baseline["pt_run_name"].endswith("_arch0_skill")

    config["vision"]["foveation"]["randomization"]["enabled"] = True
    randomized = build_settings(config)
    assert randomized["foveated_vision_enabled"] is False
    assert randomized["foveation_randomization_enabled"] is True
    assert randomized["pt_run_name"].endswith("_arch0_skill_rand")


def test_stage1_can_use_but_never_train_an_external_frozen_predictor(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "arch0")
    project = Path(config["project_root"])
    predictor = project / "outputs/predictor/checkpoints/000100/pretrained_model"
    predictor.mkdir(parents=True)
    predictor_config = {
        "type": "skill_aux",
        "train_skill_predictor": True,
        "skill_fsq_levels": [3, 3, 3],
        "skill_vocab_size": 27,
        "skill_predictor_vlm_variant": "gemma_2b",
        "skill_predictor_image_size": 224,
        "skill_predictor_reader_tokens": 6,
        "skill_predictor_reader_depth": 3,
        "skill_predictor_reader_heads": 8,
        "skill_predictor_all_layers": True,
        "skill_predictor_detach_vlm": False,
        "skill_predictor_lora": True,
        "skill_predictor_lora_targets": "q,k,v,o",
        "skill_predictor_lora_rank": 8,
        "skill_predictor_lora_alpha": 16.0,
        "skill_predictor_lora_dropout": 0.0,
        "skill_predictor_deadzone_frac": 0.8,
        "skill_predictor_attend_image": True,
        "skill_predictor_attend_language": True,
        "tokenizer_max_length": 200,
    }
    (predictor / "config.json").write_text(json.dumps(predictor_config))
    (predictor / "model.safetensors").touch()
    tokenizer = project / "models/tokenizer"
    tokenizer.mkdir(parents=True)
    for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tokenizer / filename).write_text("{}")

    config["warm_start"].update(
        {"predictor_checkpoint": str(predictor), "tokenizer": str(tokenizer)}
    )
    config["action_conditioning"] = {"training_skill_source": "predictor"}
    settings = build_settings(config)

    assert settings["skill_predictor_checkpoint_path"] == predictor
    assert settings["skill_predictor_reader_tokens"] == 6
    assert settings["skill_predictor_lora"] is True
    assert "train_skill_predictor" not in settings
    assert settings["pt_run_name"].endswith("_arch0_pretrained_predictor")
