import json
import sys
from pathlib import Path

import pytest


CONFIG_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skills/src"
)
sys.path.insert(0, str(CONFIG_SRC))

from train_skills_config import train_settings  # noqa: E402


def _minimal_fsq_config(tmp_path: Path) -> dict:
    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "fsq_dataset_root": "FSQ_dataset",
        "target_dataset": "demo_full_full",
        "fsq_inputs_name": "FSQ_inputs",
        "skillset_seg_name": (
            "seg_demo_full_full_state_obs20_ck100000_"
            "std_episodemean_80p_trial"
        ),
        "skillset_name": "skillset",
        "fsq_levels": [3, 3, 3],
        "fsq_exp": "test_fsq_run",
        "fsq_autoencoder_mode": "zero",
    }


def _write_manifest(tmp_path: Path, config: dict) -> Path:
    skillset = (
        tmp_path
        / config["dataset_root"]
        / config["fsq_dataset_root"]
        / config["target_dataset"]
        / config["fsq_inputs_name"]
        / config["skillset_seg_name"]
        / config["skillset_name"]
    )
    skillset.mkdir(parents=True)
    manifest = {
        "dataset_name": "demo_full_full",
        "dataset_dir": str(tmp_path / "dataset/demo_full_full"),
        "policy_path": str(
            tmp_path
            / "outputs/DP/demo_full_full_state_obs20/checkpoints/100000/pretrained_model"
        ),
        "mode": "std",
        "detector": {
            "boundary_threshold_mode": "episode_mean",
            "boundary_threshold_scale": 0.8,
            "min_skills": 1,
            "min_skill_len": 10,
        },
        "probe": {
            "count": 24,
            "alpha": 0.1,
            "pca_variance": 0.95,
            "pca_stride": 3,
        },
        "action": {
            "mode": "dataset",
            "relative_exclude_joints": ["gripper"],
            "gripper_mode": "discrete",
            "gripper_indices": [-1],
            "gripper_values": [-1.0, 1.0],
            "gripper_threshold": 0.0,
        },
    }
    path = skillset / "skillset_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_fsq_selects_skillset_by_folders_and_reads_manifest(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_num_workers"] = 8
    config["fsq_val_num_workers"] = 0
    config["fsq_val_every"] = 25
    config["fsq_save_best_model"] = False
    config["fsq_lr_schedule"] = "constant"
    manifest_path = _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["skillset_manifest_path"] == manifest_path
    assert settings["skillset_dir"] == manifest_path.parent
    assert settings["dp_policy"] == "demo_full_full_state_obs20"
    assert settings["dp_checkpoint"] == "100000"
    assert settings["skillset_mode"] == "std"
    assert settings["skillset_boundary_threshold_mode"] == "episode_mean"
    assert settings["skillset_boundary_threshold_scale"] == 0.8
    assert settings["skillset_min_skill_len"] == 10
    assert settings["skillset_output_suffix"] == "_trial"
    assert settings["skillset_gripper_mode"] == "discrete"
    assert settings["fsq_num_workers"] == 8
    assert settings["fsq_val_num_workers"] == 0
    assert settings["fsq_val_every"] == 25
    assert settings["fsq_save_best_model"] is False
    assert settings["fsq_lr_schedule"] == "constant"
    assert settings["fsq_terminator_model"] == "default"
    assert settings["fsq_terminator_input_space"] == "both"
    assert settings["fsq_state_rnn_terminator"] is False
    assert settings["fsq_frame_cache_enabled"] is True
    assert settings["fsq_frame_cache_stage_local"] is True
    assert settings["fsq_frame_cache_local_root"] == ""
    assert settings["fsq_frame_cache_local_reserve_gb"] == 16
    assert settings["fsq_frame_cache_root"] == str(
        tmp_path
        / ".cache/fsq_frame_cache/dataset/demo_full_full/rgb_zstd_v2"
    )
    assert settings["fsq_frame_cache_dir"] == ""
    assert settings["fsq_run_name"] == (
        "zero1_recon_termDINO__pairOFF_routeOFF_loss__test_fsq_run"
    )


def test_fsq_frame_cache_completed_dir_can_be_injected_by_submitter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _minimal_fsq_config(tmp_path)
    _write_manifest(tmp_path, config)
    completed = tmp_path / "shared-cache/fingerprint"
    monkeypatch.setenv("FSQ_FRAME_CACHE_DIR", str(completed))

    settings = train_settings(config)

    assert settings["fsq_frame_cache_dir"] == str(completed)


def test_fsq_frame_cache_local_stage_settings_resolve(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_frame_cache_stage_local=False,
        fsq_frame_cache_local_root="/node-local/fsq",
        fsq_frame_cache_local_reserve_gb=7,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_frame_cache_stage_local"] is False
    assert settings["fsq_frame_cache_local_root"] == "/node-local/fsq"
    assert settings["fsq_frame_cache_local_reserve_gb"] == 7


def test_fsq_frame_cache_local_root_must_be_absolute(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_frame_cache_local_root"] = "relative/cache"
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match="must be empty or an absolute"):
        train_settings(config)


def test_fsq_frame_cache_local_reserve_must_be_nonnegative(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_frame_cache_local_reserve_gb"] = -1
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match="must be >= 0"):
        train_settings(config)


def test_fsq_selected_skillset_requires_manifest(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="skillset manifest not found"):
        train_settings(config)


def test_fsq_lr_schedule_rejects_unknown_value(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_lr_schedule"] = "linear"
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match=r"fsq_lr_schedule must be cosine\|constant"):
        train_settings(config)


def test_fsq_single_lr_controls_all_trainable_modules(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_lr"] = 3.0e-6
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_lr"] == "3e-06"
    assert settings["fsq_encoder_lr"] == settings["fsq_lr"]
    assert settings["fsq_reconstructor_lr"] == settings["fsq_lr"]
    assert settings["fsq_terminator_lr"] == settings["fsq_lr"]


def test_fsq_split_learning_rates_are_hidden(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_encoder_lr"] = 1.0e-4
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match="uses one fsq_lr"):
        train_settings(config)


def test_fsq_run_name_uses_mode_decoder_loss_and_optional_exp_suffix(
    tmp_path: Path,
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_exp="my_exact_name",
        fsq_run_name="legacy_template_must_be_ignored",
        fsq_terminator={
            "termination": True,
            "default_arch": "fusion",
            "vision_backbone": "resnet",
            "freeze_vision_encoder": False,
        },
        fsq_init_calibration={"enabled": True, "gain": 0.8, "samples": 4096},
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_exp"] == "my_exact_name"
    assert settings["fsq_decoder_name"] == "recon_termRES"
    assert settings["fsq_loss_name"] == "pairOFF_routeOFF_loss"
    assert settings["fsq_run_name"] == (
        "zero1_recon_termRES__pairOFF_routeOFF_loss__my_exact_name"
    )
    assert settings["fsq_output_dir"].name == settings["fsq_run_name"]


def test_fsq_exp_can_be_empty(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_exp"] = ""
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_run_name"] == (
        "zero1_recon_termDINO__pairOFF_routeOFF_loss"
    )


@pytest.mark.parametrize(
    ("reconstructor", "termination", "backbone", "decoder_name"),
    [
        (True, False, "dino", "recon_only"),
        (False, True, "dino", "termDINO_only"),
        (False, True, "resnet", "termRES_only"),
        (True, True, "dino", "recon_termDINO"),
        (True, True, "resnet", "recon_termRES"),
    ],
)
def test_fsq_decoder_name_covers_reconstructor_and_terminator_compositions(
    tmp_path: Path,
    reconstructor: bool,
    termination: bool,
    backbone: str,
    decoder_name: str,
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_exp="",
        fsq_decoder_reconstructor=reconstructor,
        fsq_terminator={
            "termination": termination,
            "vision_backbone": backbone,
        },
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_decoder_name"] == decoder_name
    assert settings["fsq_run_name"] == (
        f"zero1_{decoder_name}__pairOFF_routeOFF_loss"
    )


def test_fsq_loss_name_matches_pair_type_and_route_switch(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_exp="",
        fsq_terminator={"termination": False},
        fsq_pair_loss={"type": "contrastive"},
        fsq_route_loss={"enabled": True},
        fsq_boundary_aug_pmax=10,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_loss_name"] == "contrastiveON_routeON_loss"
    assert settings["fsq_run_name"] == (
        "zero1_recon_only__contrastiveON_routeON_loss"
    )


def test_fsq_pair_loss_type_none_disables_pair_loss_and_updates_name(
    tmp_path: Path,
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_exp="",
        fsq_terminator={"termination": False},
        fsq_pair_loss={
            "type": "none",
            "weight": 0.2,
            "inv_temperature": 7.5,
        },
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_pair_loss"] == "none"
    assert settings["fsq_pair_weight"] == "0.2"
    assert settings["fsq_pair_inv_temperature"] == "7.5"
    assert settings["fsq_loss_name"] == "pairOFF_routeOFF_loss"
    assert settings["fsq_run_name"] == "zero1_recon_only__pairOFF_routeOFF_loss"


@pytest.mark.parametrize("exp", ["../escape", "nested/run", "bad name"])
def test_fsq_exp_must_be_a_safe_optional_suffix(
    tmp_path: Path, exp: str
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_exp"] = exp
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match="fsq_exp"):
        train_settings(config)


def test_fsq_clean_model_options_resolve_to_internal_contract(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_autoencoder_mode="raw",
        fsq_decoder_reconstructor=True,
        fsq_terminator={
            "termination": True,
            "default_arch": "small",
            "vision_backbone": "dino",
            "freeze_vision_encoder": True,
        },
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_encoder_input_mode"] == "raw_state"
    assert settings["fsq_autoencoder_mode"] == "raw"
    assert settings["fsq_encoder_length_token"] is False
    assert settings["fsq_reconstructor_arch"] == "oneshot"
    assert settings["fsq_reconstructor_output_mode"] == "raw_state"
    assert settings["fsq_reconstructor_start_state"] is False
    assert settings["fsq_decoder_reconstructor"] is True
    assert settings["fsq_decoder_terminator_progress"] is False
    assert settings["fsq_decoder_terminator_termination"] is True
    assert settings["fsq_terminator_input_space"] == "both"
    assert settings["fsq_terminator_model"] == "default"
    assert settings["fsq_terminator_layers"] == 3
    assert settings["fsq_terminator_heads"] == 4
    assert settings["fsq_terminator_termination_only"] is True
    assert settings["fsq_state_rnn_terminator"] is False


def test_fsq_terminator_model_and_input_space_are_fixed(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_terminator_arch"] = "rnn"
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match="compact fsq_terminator mapping"):
        train_settings(config)


def test_fsq_dino_path_is_fixed_to_project_dinov3_s16(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_dino_model_path"] == str(
        tmp_path / "models/dinov3-vits16"
    )


def test_fsq_action_sequence_autoencoder_resolves_to_raw_matched_contract(
    tmp_path: Path,
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_autoencoder_mode="action",
        fsq_decoder_reconstructor=True,
        fsq_terminator={"termination": False},
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_autoencoder_mode"] == "action"
    assert settings["fsq_encoder_arch"] == "action_seq"
    assert settings["fsq_reconstructor_arch"] == "action_seq_transformer"
    assert settings["fsq_reconstructor_start_state"] is False
    assert settings["fsq_reconstructor_only"] is True
    assert settings["fsq_samples_per_skill"] == 2  # trainer makes it effective 1


def test_fsq_normalized_action_mapping_scales_gripper_and_names_weight(
    tmp_path: Path,
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.pop("fsq_autoencoder_mode")
    config.update(
        fsq_autoencoder={"mode": "norm_action", "gripper_weight": 0.1},
        fsq_exp="",
        fsq_decoder_reconstructor=True,
        fsq_terminator={"termination": False},
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_autoencoder_mode"] == "norm_action"
    assert settings["fsq_action_gripper_weight"] == "0.1"
    assert settings["fsq_encoder_arch"] == "action_seq"
    assert settings["fsq_reconstructor_arch"] == "action_seq_transformer"
    assert settings["fsq_run_name"] == (
        "norm_action01_recon_only__pairOFF_routeOFF_loss"
    )


@pytest.mark.parametrize("mode", ["raw", "zero", "action", "norm_action"])
def test_fsq_gripper_weight_is_available_in_every_autoencoder_mode(
    tmp_path: Path,
    mode: str,
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.pop("fsq_autoencoder_mode")
    config.update(
        fsq_autoencoder={"mode": mode, "gripper_weight": 0.1},
        fsq_exp="",
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_action_gripper_weight"] == "0.1"
    assert settings["fsq_run_name"].startswith(f"{mode}01_")


@pytest.mark.parametrize("mode", ["raw", "zero", "action", "norm_action"])
def test_fsq_start_state_adaln_is_available_in_every_autoencoder_mode(
    tmp_path: Path,
    mode: str,
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_autoencoder_mode=mode,
        fsq_start_state_conditioning="adaln",
        fsq_decoder_reconstructor=True,
        fsq_terminator={"termination": False},
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_start_state_conditioning"] == "adaln"
    assert settings["fsq_reconstructor_start_state"] is True
    assert "__inital_proprio_conditioned" in settings["fsq_run_name"]


def test_fsq_start_state_conditioning_rejects_unknown_mode(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_start_state_conditioning"] = "film"
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match="none\\|adaln"):
        train_settings(config)


def test_fsq_individual_architecture_keys_are_hidden(
    tmp_path: Path,
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_reconstructor_arch"] = "action_seq"
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match="selected only with fsq_autoencoder"):
        train_settings(config)


def test_fsq_causal_transformer_action_reconstructor_resolves(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_autoencoder_mode="action",
        fsq_decoder_reconstructor=True,
        fsq_terminator={"termination": False},
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_reconstructor_arch"] == "action_seq_transformer"
    assert settings["fsq_reconstructor_start_state"] is False
    assert settings["fsq_decoder_terminator_progress"] is False


def test_fsq_clean_decoder_interface_keeps_progress_opt_in(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_decoder_reconstructor=True,
        fsq_terminator={"termination": False},
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_decoder_terminator_progress"] is False
    assert settings["fsq_decoder_terminator_termination"] is False
    assert settings["fsq_reconstructor_only"] is True


@pytest.mark.parametrize(
    ("mode", "encoder_arch", "input_mode", "decoder_arch", "output_mode"),
    [
        ("raw", "spline", "raw_state", "oneshot", "raw_state"),
        ("zero", "spline", "zero_grounded", "oneshot", "zero_grounded"),
        (
            "action",
            "action_seq",
            "zero_grounded",
            "action_seq_transformer",
            "zero_grounded",
        ),
        (
            "norm_action",
            "action_seq",
            "zero_grounded",
            "action_seq_transformer",
            "zero_grounded",
        ),
    ],
)
def test_fsq_autoencoder_modes_are_indivisible_presets(
    tmp_path: Path,
    mode: str,
    encoder_arch: str,
    input_mode: str,
    decoder_arch: str,
    output_mode: str,
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_autoencoder_mode"] = mode
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_autoencoder_mode"] == mode
    assert settings["fsq_encoder_arch"] == encoder_arch
    assert settings["fsq_encoder_input_mode"] == input_mode
    assert settings["fsq_reconstructor_arch"] == decoder_arch
    assert settings["fsq_reconstructor_output_mode"] == output_mode
    assert settings["fsq_reconstructor_start_state"] is False


def test_fsq_resnet_vision_option_resolves(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_terminator={"vision_backbone": "ReSNet"},
        fsq_resnet_image_size=256,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_vision_backbone"] == "resnet"
    assert settings["fsq_resnet_image_size"] == 256
    assert settings["fsq_run_name"] == (
        "zero1_recon_termRES__pairOFF_routeOFF_loss__test_fsq_run"
    )


def test_fsq_fusion_terminator_option_does_not_rename_checkpoint(
    tmp_path: Path,
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_terminator"] = {"default_arch": "FuSiOn"}
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_terminator_default_arch"] == "fusion"
    assert settings["fsq_run_name"] == (
        "zero1_recon_termDINO__pairOFF_routeOFF_loss__test_fsq_run"
    )


def test_fsq_cond_terminator_option_is_rejected(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_terminator"] = {"default_arch": "cond"}
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match=r"must be small\|fusion"):
        train_settings(config)


def test_fsq_overlap_pair_settings_resolve(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_pair_loss={"type": "overlap", "weight": 0.2, "inv_temperature": 7.5},
        fsq_pair_warmup={"enabled": True, "epochs": 50, "ramp_epochs": 25},
        fsq_boundary_aug_pmax=10,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_pair_loss"] == "overlap"
    assert settings["fsq_pair_weight"] == "0.2"
    assert settings["fsq_pair_inv_temperature"] == "7.5"
    assert settings["fsq_pair_warmup"] is True
    assert settings["fsq_pair_warmup_epochs"] == 50
    assert settings["fsq_pair_ramp_epochs"] == 25
    assert settings["fsq_boundary_aug_pmax"] == 10
    assert settings["fsq_boundary_aug_early_start_pmax"] == 10
    assert settings["fsq_boundary_aug_late_start_pmax"] == 10
    assert settings["fsq_boundary_aug_early_end_pmax"] == 10
    assert settings["fsq_boundary_aug_late_end_pmax"] == 10
    assert settings["fsq_boundary_aug_distribution"] == "half_normal"


def test_fsq_boundary_distribution_is_fixed_to_half_normal(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_boundary_aug_distribution"] = "uniform"
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match="fixed to half_normal"):
        train_settings(config)


def test_fsq_action_and_end_loss_mappings_resolve(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_action_loss={"weight": 0.75},
        fsq_end_loss={"weight": 1.25, "target_sigma": 2.0},
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_action_loss_weight"] == "0.75"
    assert settings["fsq_end_loss_weight"] == "1.25"
    assert settings["fsq_end_target_sigma"] == "2.0"


def test_fsq_directional_boundary_windows_resolve_independently(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_pair_loss={"type": "js"},
        fsq_boundary_aug_early_start_pmax=2,
        fsq_boundary_aug_late_start_pmax=4,
        fsq_boundary_aug_early_end_pmax=6,
        fsq_boundary_aug_late_end_pmax=8,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_boundary_aug_pmax"] == 8
    assert settings["fsq_boundary_aug_early_start_pmax"] == 2
    assert settings["fsq_boundary_aug_late_start_pmax"] == 4
    assert settings["fsq_boundary_aug_early_end_pmax"] == 6
    assert settings["fsq_boundary_aug_late_end_pmax"] == 8


def test_fsq_zero_directional_boundary_window_disables_that_direction(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_pair_loss={"type": "js"},
        fsq_boundary_aug_early_start_pmax=0,
        fsq_boundary_aug_late_start_pmax=0,
        fsq_boundary_aug_early_end_pmax=0,
        fsq_boundary_aug_late_end_pmax=7,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_boundary_aug_pmax"] == 7
    assert settings["fsq_boundary_aug_early_start_pmax"] == 0
    assert settings["fsq_boundary_aug_late_end_pmax"] == 7


def test_fsq_js_pair_setting_resolves(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_pair_loss={"type": "js"},
        fsq_boundary_aug_pmax=10,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_pair_loss"] == "js"


def test_fsq_linear_contrastive_pair_setting_resolves(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_pair_loss={"type": "contrastive"},
        fsq_boundary_aug_pmax=10,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_pair_loss"] == "contrastive"


def test_fsq_action_sequence_contrastive_pair_setting_resolves(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_autoencoder_mode="action",
        fsq_pair_loss={"type": "contrastive"},
        fsq_boundary_aug_early_start_pmax=10,
        fsq_boundary_aug_late_start_pmax=5,
        fsq_boundary_aug_early_end_pmax=10,
        fsq_boundary_aug_late_end_pmax=5,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_encoder_arch"] == "action_seq"
    assert settings["fsq_reconstructor_arch"] == "action_seq_transformer"
    assert settings["fsq_pair_loss"] == "contrastive"
    assert settings["fsq_boundary_aug_pmax"] == 10


def test_fsq_route_loss_resolves_and_updates_run_name(
    tmp_path: Path,
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_autoencoder_mode="action",
        fsq_route_loss={"enabled": True},
        fsq_pair_loss={"inv_temperature": 7.5},
        fsq_decoder_reconstructor=True,
        fsq_terminator={"termination": False},
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_route_loss"] is True
    assert settings["fsq_pair_inv_temperature"] == "7.5"
    assert settings["fsq_run_name"] == (
        "action1_recon_only__pairOFF_routeON_loss__test_fsq_run"
    )


def test_legacy_reconstruction_route_key_maps_to_joint_route(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_reconstruction_route_loss"] = {"enabled": True}
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_route_loss"] is True


def test_fsq_route_loss_requires_reconstructor(
    tmp_path: Path,
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_route_loss={"enabled": True},
        fsq_decoder_reconstructor=False,
        fsq_terminator={"termination": True},
    )
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match="requires fsq_decoder_reconstructor=true"):
        train_settings(config)


def test_fsq_chunk_reconstructor_is_no_longer_selectable(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_reconstructor_arch"] = "chunk"
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match="remove hidden keys"):
        train_settings(config)


def test_bsq5_selects_distinct_tag_and_binary_latent_contract(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_quantizer="bsq",
        bsq_code_dim=5,
        fsq_entropy=False,
        fsq_pair_loss={"type": "js"},
        fsq_boundary_aug_pmax=10,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_quantizer"] == "bsq"
    assert settings["bsq_code_dim"] == 5
    assert settings["fsq_tag"] == "bsq5"
    assert settings["fsq_dim"] == 5
    assert settings["fsq_num_embeddings"] == 32
    assert settings["fsq_levels_str"] == "2 2 2 2 2"
    assert settings["fsq_run_name"] == (
        "zero1_recon_termDINO__jsON_routeOFF_loss__test_fsq_run"
    )


def test_bsq_rejects_fsq_entropy_objective(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(fsq_quantizer="bsq", bsq_code_dim=5, fsq_entropy=True)
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match="set fsq_entropy=false"):
        train_settings(config)


def test_fsq_entropy_conf_ceiling_resolves(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(fsq_entropy=True, fsq_entropy_conf_ceiling=0.1)
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_entropy"] is True
    assert settings["fsq_entropy_conf_ceiling"] == "0.1"


@pytest.mark.parametrize("ceiling", [-0.01, 1.01])
def test_fsq_entropy_conf_ceiling_rejects_values_outside_unit_interval(
    tmp_path: Path, ceiling: float
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_entropy_conf_ceiling"] = ceiling
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match="fsq_entropy_conf_ceiling must be in"):
        train_settings(config)


def test_fsq_init_calibration_does_not_mutate_exp_name(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_exp="recon",
        fsq_init_calibration={"enabled": True, "gain": 0.8, "samples": 4096},
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_init_calibration"] is True
    assert settings["fsq_init_calibration_gain"] == "0.8"
    assert settings["fsq_init_calibration_samples"] == 4096
    assert settings["fsq_run_name"] == (
        "zero1_recon_termDINO__pairOFF_routeOFF_loss__recon"
    )


def test_fsq_overlap_pair_requires_positive_boundary_window(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_pair_loss"] = {"type": "overlap"}
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match="at least one positive directional"):
        train_settings(config)


def test_fsq_job_reresolution_uses_exported_folder_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _minimal_fsq_config(tmp_path)
    manifest_path = _write_manifest(tmp_path, config)
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path / "dataset"))
    monkeypatch.setenv("DATASET_ROOT_NAME", "dataset")
    monkeypatch.setenv(
        "FSQ_DATASET_ROOT", str(tmp_path / "dataset/FSQ_dataset")
    )
    monkeypatch.setenv("FSQ_DATASET_ROOT_NAME", "FSQ_dataset")

    settings = train_settings(config)

    assert settings["skillset_manifest_path"] == manifest_path
    assert settings["dataset_root_name"] == "dataset"
    assert settings["fsq_dataset_root_name"] == "FSQ_dataset"


def test_fsq_job_reresolution_does_not_replace_compact_yaml_mappings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_init_calibration={"enabled": True, "gain": 0.8, "samples": 4096},
        fsq_terminator={
            "termination": False,
            "default_arch": "fusion",
            "vision_backbone": "resnet",
            "freeze_vision_encoder": True,
        },
        fsq_pair_loss={
            "type": "contrastive",
            "weight": 0.2,
            "inv_temperature": 7.5,
        },
        fsq_route_loss={"enabled": True},
        fsq_action_loss={"weight": 0.75},
        fsq_end_loss={"weight": 1.25, "target_sigma": 2.0},
        fsq_pair_warmup={"enabled": True, "epochs": 50, "ramp_epochs": 25},
        fsq_boundary_aug_pmax=10,
    )
    _write_manifest(tmp_path, config)

    # These are the flattened variables inherited by the Slurm job after the
    # submit-side resolver runs. They must not override structured YAML values
    # when train_fsq.sbatch resolves its immutable snapshot again.
    monkeypatch.setenv("FSQ_INIT_CALIBRATION", "true")
    monkeypatch.setenv("FSQ_PAIR_LOSS", "contrastive")
    monkeypatch.setenv("FSQ_ROUTE_LOSS", "true")
    monkeypatch.setenv("FSQ_PAIR_WARMUP", "true")

    settings = train_settings(config)

    assert settings["fsq_init_calibration_gain"] == "0.8"
    assert settings["fsq_init_calibration_samples"] == 4096
    assert settings["fsq_terminator_default_arch"] == "fusion"
    assert settings["fsq_vision_backbone"] == "resnet"
    assert settings["fsq_pair_weight"] == "0.2"
    assert settings["fsq_pair_inv_temperature"] == "7.5"
    assert settings["fsq_route_loss"] is True
    assert settings["fsq_action_loss_weight"] == "0.75"
    assert settings["fsq_end_loss_weight"] == "1.25"
    assert settings["fsq_end_target_sigma"] == "2.0"
    assert settings["fsq_pair_warmup_epochs"] == 50
    assert settings["fsq_pair_ramp_epochs"] == 25
