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
from stage1_cond_train_config import (  # noqa: E402
    build_settings as build_legacy_cond_settings,
)


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
        "architecture": {
            "name": "arch2_2",
            "expert_variant": "gemma_300m",
            "vsa": {"visual_latents_per_camera": 32},
        },
        "skill_predictor": {"train": False},
        "terminator": {"train": False},
        "training": {"optimizer": {"dino_lr_scale": 0.1}},
    }


def test_stage1_exports_single_architecture_and_relative_dino_lr(tmp_path: Path) -> None:
    config = _config(tmp_path)
    settings = build_settings(config)

    assert settings["architecture"] == "vsa_perceiver_crossattn"
    assert settings["architecture_label"] == "arch2_2"
    assert settings["vision_conditioning_mode"] == "interleaved_cross_attention"
    assert settings["include_state_in_visual_crossattn"] is True
    assert settings["include_skill_in_visual_crossattn"] is True
    assert settings["visual_crossattn_queries"] == "state + skill + action"
    assert settings["num_visual_latents_per_camera"] == 32
    assert settings["visual_perceiver_width"] == 1024
    assert settings["dino_lr_scale"] == 0.1
    assert settings["action_expert_variant"] == "gemma_300m"
    assert settings["mask_actions_after_skill_end"] is False
    assert settings["cumulative_xyz_loss_enabled"] is False
    assert settings["cumulative_xyz_loss_weight"] == pytest.approx(0.5)
    assert settings["conditioning_route"] == "state_skill_cond"
    assert settings["cond_encoder_variant"] == "gemma_300m"
    assert settings["freeze_vision_encoder"] is False
    assert "vsa_perceiver_crossattn" not in settings["pt_run_name"]
    assert "dino_tuned" not in settings["pt_run_name"]
    assert "sa18_lat32" not in settings["pt_run_name"]
    assert "interleaved_cross_attention_flow" not in settings["pt_run_name"]
    assert settings["pt_run_name"].startswith("bs16_")
    assert settings["pt_run_name"].endswith("_arch2_2")
    assert settings["vsa_debug_schedule"] == "[]"
    assert settings["steps"] == 50_000
    assert settings["scheduler_mode"] == "cosine_decay"
    assert settings["scheduler_warmup_steps"] == 1_000
    assert settings["scheduler_decay_steps"] == 30_000

    config["training"]["vsa_debug"] = {
        "every": 5_000,
        "initial": [1, 100, 1000],
    }
    included = build_settings(config)
    assert included["include_state_in_visual_crossattn"] is True
    assert included["include_skill_in_visual_crossattn"] is True
    assert included["visual_crossattn_queries"] == "state + skill + action"
    assert included["vsa_debug_schedule"] == (
        "[1,100,1000,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000]"
    )

    config["architecture"]["vsa"]["visual_latents_per_camera"] = 64
    lat64 = build_settings(config)
    assert lat64["num_visual_latents_per_camera"] == 64
    assert "sa18_lat64" not in lat64["pt_run_name"]

    config["architecture"]["name"] = "arch3"
    in_context = build_settings(config)
    assert in_context["architecture_label"] == "arch3"
    assert in_context["vision_conditioning_mode"] == "in_context_tokens"
    assert in_context["visual_crossattn_queries"] == "ignored"
    assert "in_context_tokens_flow" not in in_context["pt_run_name"]
    assert in_context["pt_run_name"].startswith("bs16_")
    assert in_context["pt_run_name"].endswith("_arch3")

    config.setdefault("training", {}).setdefault("dataloader", {})[
        "batch_size"
    ] = 96
    batch96 = build_settings(config)
    assert batch96["pt_run_name"].startswith("bs96_")
    assert batch96["pt_run_name"].endswith("_arch3")


def test_skill_end_loss_mask_supports_jitter_and_has_distinct_name(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["mask_actions_after_skill_end"] = True
    settings = build_settings(config)

    assert settings["mask_actions_after_skill_end"] is True
    assert settings["transition_jitter_pmax"] == 15
    assert settings["pt_run_name"].endswith("_arch2_2_skillendmask")


def test_cumulative_xyz_auxiliary_has_weighted_distinct_name(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["mask_actions_after_skill_end"] = True
    config["cumulative_xyz_loss"] = {"enabled": True, "weight": 0.5}

    settings = build_settings(config)

    assert settings["cumulative_xyz_loss_enabled"] is True
    assert settings["cumulative_xyz_loss_weight"] == pytest.approx(0.5)
    assert settings["pt_run_name"].endswith(
        "_arch2_2_skillendmask_cumxyz0p5"
    )


def test_cumulative_xyz_auxiliary_rejects_endpoint_mix(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["loss"] = "flow_endpoint_xyz"
    config["cumulative_xyz_loss"] = {"enabled": True, "weight": 0.5}

    with pytest.raises(ValueError, match="cannot be combined"):
        build_settings(config)


def test_stage1_rejects_invalid_vsa_debug_initial_steps(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["training"]["vsa_debug"] = {"every": 5_000, "initial": [100, 1, 100]}

    with pytest.raises(ValueError, match="sorted and contain no duplicates"):
        build_settings(config)


def test_stage1_rejects_negative_vsa_debug_frequency(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["training"]["vsa_debug"] = {"every": -1, "initial": []}

    with pytest.raises(ValueError, match="vsa_debug.every must be non-negative"):
        build_settings(config)


def test_stage1_rejects_legacy_vsa_debug_keys(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["training"]["vsa_debug_schedule"] = [1, 100]

    with pytest.raises(ValueError, match="were replaced"):
        build_settings(config)


def test_stage1_exports_warmup_constant_schedule(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["training"]["schedule"] = {
        "steps": 50_000,
        "lr_mode": "warmup_constant",
        "warmup_steps": 1_000,
        "log_every": 100,
        "save_every": 5_000,
    }

    settings = build_settings(config)

    assert settings["scheduler_mode"] == "warmup_constant"
    assert settings["scheduler_warmup_steps"] == 1_000
    assert settings["scheduler_decay_steps"] == 30_000


def test_stage1_phase_batch_sampling_is_opt_in_and_uses_separate_run(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    ordinary = build_settings(config)
    assert ordinary["phase_batch_sampling_enabled"] is False
    assert ordinary["pt_run_name"].endswith("_arch2_2")

    config["batch_sampling"] = {
        "enabled": True,
        "focused_fraction": 0.8,
        "early_fraction": 0.4,
        "early_threshold": 0.2,
        "late_threshold": 0.7,
    }
    focused = build_settings(config)
    assert focused["phase_batch_sampling_enabled"] is True
    assert focused["phase_batch_focused_fraction"] == pytest.approx(0.8)
    assert focused["phase_batch_early_fraction"] == pytest.approx(0.4)
    assert focused["phase_batch_early_threshold"] == pytest.approx(0.2)
    assert focused["phase_batch_late_threshold"] == pytest.approx(0.7)
    assert focused["pt_run_name"].endswith("_arch2_2_phasefocus")


@pytest.mark.parametrize(
    "batch_sampling",
    [
        {"focused_fraction": -0.1},
        {"early_fraction": 1.1},
        {"early_threshold": 0.8, "late_threshold": 0.7},
    ],
)
def test_stage1_rejects_invalid_phase_batch_sampling(
    tmp_path: Path, batch_sampling: dict
) -> None:
    config = _config(tmp_path)
    config["batch_sampling"] = batch_sampling

    with pytest.raises(ValueError, match="batch_sampling"):
        build_settings(config)


@pytest.mark.parametrize(
    ("section", "field", "message"),
    [
        ("architecture", "conditioning_route", "fixed Cond-Gemma ablations"),
        ("architecture", "cond_variant", "fixed Cond-Gemma ablations"),
        ("architecture", "vision_conditioning_mode", "fixed Cond-Gemma ablations"),
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
        config[section][field] = "legacy"

    with pytest.raises(ValueError, match=message):
        build_settings(config)


def test_stage1_rejects_any_other_architecture(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["architecture"]["name"] = "state_skill_cond"

    with pytest.raises(
        ValueError,
        match=r"must be arch0\|arch0_1\|arch0_2\|arch0_2_sep\|arch0_3",
    ):
        build_settings(config)


def test_stage1_rejects_removed_arch2_alias(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["architecture"]["name"] = "arch2"

    with pytest.raises(ValueError, match="was split into arch2_1 and arch2_2"):
        build_settings(config)


@pytest.mark.parametrize(
    ("label", "revision", "mode", "memory_tokens"),
    [
        (
            "arch1_3",
            "visual_kv_uncompressed_v1",
            "uncompressed_visual_kv_self_attention",
            197,
        ),
        (
            "arch2_1",
            "visual_kv_perceiver_v1",
            "compressed_visual_kv_self_attention",
            32,
        ),
        (
            "arch2_2",
            "interleaved_direct1024_v3",
            "interleaved_cross_attention",
            32,
        ),
    ],
)
def test_vsa_arch1_3_arch2_family_resolves_exact_contract(
    tmp_path: Path,
    label: str,
    revision: str,
    mode: str,
    memory_tokens: int,
) -> None:
    config = _config(tmp_path)
    config["architecture"]["name"] = label
    settings = build_settings(config)

    assert settings["architecture_label"] == label
    assert settings["architecture_revision"] == revision
    assert settings["vision_conditioning_mode"] == mode
    assert settings["num_visual_latents_per_camera"] == memory_tokens
    assert settings["pt_run_name"].endswith(f"_{label}")


def test_stage1_appends_user_defined_run_suffix_last(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["run"] = {"suffix": "_custom-20_"}

    settings = build_settings(config)

    assert settings["pt_run_name"].endswith("_arch2_2_custom-20")

    config["batch_sampling"] = {"enabled": True}
    focused = build_settings(config)
    assert focused["pt_run_name"].endswith("_arch2_2_phasefocus_custom-20")


@pytest.mark.parametrize("suffix", ["has space", "../escape", "/absolute", "한글"])
def test_stage1_rejects_unsafe_run_suffix(tmp_path: Path, suffix: str) -> None:
    config = _config(tmp_path)
    config["run"] = {"suffix": suffix}

    with pytest.raises(ValueError, match="run.suffix may contain"):
        build_settings(config)


def test_unified_config_selects_arch0_and_applies_relative_dino_lr(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["architecture"]["name"] = "arch0"
    config["architecture"]["chunk_size"] = 10
    config["vision"]["freeze"] = False
    config["execution"] = {"action_steps": 10}
    config["training"] = {
        "dataloader": {"batch_size": 16, "workers": 2, "gpus": 1},
        "optimizer": {"base_lr": 2.5e-5, "dino_lr_scale": 0.1},
        "gradient_checkpointing": False,
        "schedule": {
            "steps": 1_000_000,
            "lr_decay_steps": 1_000_000,
            "log_every": 100,
            "save_every": 5_000,
        },
    }

    settings = build_settings(config)

    assert settings["architecture"] == "cond_gemma"
    assert settings["architecture_label"] == "arch0"
    assert settings["architecture_revision"] == "skillvla_real_v1"
    assert settings["conditioning_route"] == "state_cond"
    assert settings["cond_encoder_variant"] == "gemma_300m"
    assert settings["freeze_vision_encoder"] is False
    assert settings["dino_lr_scale"] == 0.1
    assert settings["n_action_steps"] == 10
    assert settings["batch_size"] == 16
    assert settings["num_workers"] == 2
    assert settings["steps"] == 1_000_000
    assert settings["scheduler_decay_steps"] == 1_000_000
    assert settings["log_freq"] == 100
    assert settings["save_freq"] == 5_000
    assert "dino_tuned_state_skill_cond_flow" not in settings["pt_run_name"]
    assert settings["pt_run_name"].startswith("bs16_")
    assert settings["pt_run_name"].endswith("_arch0")

    config["training"]["dataloader"]["batch_size"] = 64
    batch64 = build_settings(config)
    assert batch64["pt_run_name"].startswith("bs64_")
    assert batch64["pt_run_name"].endswith("_arch0")


@pytest.mark.parametrize(
    ("label", "revision"),
    [
        ("arch0_1", "expert_state_adarms_v1"),
        ("arch0_2", "cond_expert_state_adarms_v1"),
        ("arch0_2_sep", "cond_expert_separate_state_adarms_v1"),
        ("arch0_3", "wrist_cond_expert_state_adarms_v1"),
    ],
)
def test_arch0_state_location_ablations_resolve_distinct_contracts(
    tmp_path: Path, label: str, revision: str
) -> None:
    config = _config(tmp_path)
    config["architecture"]["name"] = label

    settings = build_settings(config)

    assert settings["architecture"] == "cond_gemma"
    assert settings["architecture_label"] == label
    assert settings["architecture_revision"] == revision
    assert settings["conditioning_route"] == "state_cond"
    assert settings["pt_run_name"].endswith(f"_{label}")


def test_cond_family_rejects_removed_conditioning_override(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["architecture"]["name"] = "arch0"
    config["architecture"]["arch1"] = {
        "conditioning_route": "VisO_StateO_SkillO"
    }

    with pytest.raises(ValueError, match="fixed Cond-Gemma ablations"):
        build_settings(config)


@pytest.mark.parametrize(
    ("label", "revision", "uses_perceiver"),
    [
        ("arch1_1", "expert_tokens_uncompressed_v1", False),
        ("arch1_2", "expert_tokens_perceiver_v1", True),
    ],
)
def test_cond_expert_token_ablations_have_distinct_contracts(
    tmp_path: Path, label: str, revision: str, uses_perceiver: bool
) -> None:
    config = _config(tmp_path)
    config["architecture"]["name"] = label
    settings = build_settings(config)

    assert settings["architecture"] == "cond_gemma"
    assert settings["architecture_label"] == label
    assert settings["architecture_revision"] == revision
    assert settings["conditioning_route"] == "state_skill_cond"
    assert settings["num_visual_latents_per_camera"] == 32
    assert settings["pt_run_name"].endswith(f"_{label}")
    assert (revision == "expert_tokens_perceiver_v1") is uses_perceiver


def test_arch1_old_name_requires_explicit_new_choice(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["architecture"]["name"] = "arch1"
    with pytest.raises(ValueError, match="was split into arch0, arch1_1, and arch1_2"):
        build_settings(config)


def test_legacy_cond_shim_preserves_already_submitted_run_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["architecture"] = {
        "name": "arch1",
        "revision": "skillvla_real_v1",
        "expert_variant": "gemma_300m",
        "cond_variant": "gemma_300m",
        "conditioning_route": "VisO_StateO_SkillO",
    }
    config["vision"]["freeze"] = False
    config["training"]["optimizer"] = {"base_lr": 2.5e-5, "dino_lr": None}

    settings = build_legacy_cond_settings(config)

    assert settings["conditioning_route"] == "state_skill_cond"
    assert settings["pt_run_name"].endswith("_arch1_VisO_StateO_SkillO")
    assert float(settings["dino_lr"]) == pytest.approx(settings["lr"])


@pytest.mark.parametrize("value", [0, -1])
def test_stage1_rejects_nonpositive_lr_decay_steps(
    tmp_path: Path, value: int
) -> None:
    config = _config(tmp_path)
    config.setdefault("training", {}).setdefault("schedule", {})[
        "lr_decay_steps"
    ] = value

    with pytest.raises(
        ValueError, match="training.schedule.lr_decay_steps must be positive"
    ):
        build_settings(config)

    config["architecture"]["name"] = "arch0"
    config["vision"]["freeze"] = False
    with pytest.raises(
        ValueError, match="training.schedule.lr_decay_steps must be positive"
    ):
        build_settings(config)


def test_vsa_rejects_frozen_shared_dino(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["vision"]["freeze"] = True

    with pytest.raises(ValueError, match="unsupported for Arch1_3--4"):
        build_settings(config)
