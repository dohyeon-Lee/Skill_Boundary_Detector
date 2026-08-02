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


def _config(tmp_path: Path) -> dict:
    project = tmp_path / "project"
    run = "FSQ333_125_std_pt_episodemean_80p_snap10_pmax15_halfnormal"
    dataset = project / "dataset/skillvla_dataset/source" / run / "skillvla"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "skill_fsq_levels": [3, 3, 3],
                "skill_pmax": 15,
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
        "architecture": {"name": "vsa_perceiver_crossattn"},
        "skill_predictor": {"train": False},
        "terminator": {"train": False},
        "training": {"optimizer": {"dino_lr_scale": 0.1}},
    }


def test_stage1_exports_single_architecture_and_relative_dino_lr(tmp_path: Path) -> None:
    config = _config(tmp_path)
    settings = build_settings(config)

    assert settings["architecture"] == "vsa_perceiver_crossattn"
    assert settings["vision_conditioning_mode"] == "residual_cross_attention"
    assert settings["include_state_in_visual_crossattn"] is True
    assert settings["include_skill_in_visual_crossattn"] is True
    assert settings["visual_crossattn_queries"] == "state + skill + action"
    assert settings["num_visual_latents_per_camera"] == 32
    assert settings["dino_lr_scale"] == 0.1
    assert settings["action_expert_variant"] == "gemma_300m"
    assert "conditioning_route" not in settings
    assert "cond_encoder_variant" not in settings
    assert "freeze_vision_encoder" not in settings
    assert "vsa_perceiver_crossattn" not in settings["pt_run_name"]
    assert "dino_tuned" not in settings["pt_run_name"]
    assert "sa18_lat32" not in settings["pt_run_name"]
    assert "residual_cross_attention_flow" in settings["pt_run_name"]
    assert settings["vsa_debug_schedule"] == "[]"

    config["architecture"]["include_state_in_visual_crossattn"] = True
    config["architecture"]["include_skill_in_visual_crossattn"] = True
    config["training"]["vsa_debug_schedule"] = [1, 100, 1000]
    included = build_settings(config)
    assert included["include_state_in_visual_crossattn"] is True
    assert included["include_skill_in_visual_crossattn"] is True
    assert included["visual_crossattn_queries"] == "state + skill + action"
    assert included["vsa_debug_schedule"] == "[1,100,1000]"

    config["architecture"]["visual_latents_per_camera"] = 64
    lat64 = build_settings(config)
    assert lat64["num_visual_latents_per_camera"] == 64
    assert "sa18_lat64" not in lat64["pt_run_name"]

    config["architecture"]["vision_conditioning_mode"] = "in_context_tokens"
    in_context = build_settings(config)
    assert in_context["visual_crossattn_queries"] == "ignored"
    assert "in_context_tokens_flow" in in_context["pt_run_name"]


def test_stage1_rejects_invalid_vsa_debug_schedule(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["training"]["vsa_debug_schedule"] = [100, 1, 100]

    with pytest.raises(ValueError, match="sorted and contain no duplicates"):
        build_settings(config)


@pytest.mark.parametrize(
    ("section", "field", "message"),
    [
        ("architecture", "conditioning_route", "Legacy Stage-1 architecture keys"),
        ("architecture", "cond_variant", "Legacy Stage-1 architecture keys"),
        ("vision", "freeze", "vision.freeze was removed"),
        ("optimizer", "dino_lr", "dino_lr was replaced"),
    ],
)
def test_stage1_rejects_legacy_config_keys(
    tmp_path: Path, section: str, field: str, message: str
) -> None:
    config = _config(tmp_path)
    if section == "optimizer":
        config["training"]["optimizer"][field] = 1e-5
    else:
        config[section][field] = "legacy" if field != "freeze" else False

    with pytest.raises(ValueError, match=message):
        build_settings(config)


def test_stage1_rejects_any_other_architecture(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["architecture"]["name"] = "state_skill_cond"

    with pytest.raises(ValueError, match="legacy architectures do not exist"):
        build_settings(config)
