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
    config["fsq_state_rnn_terminator"] = True
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
    assert settings["fsq_state_rnn_terminator"] is True
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
        "demo_full_full_state_obs20_std_episodemean_80p_trial_fsq333"
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


def test_fsq_clean_model_options_resolve_to_internal_contract(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_encoder_input_mode="raw",
        fsq_encoder_arch="spline",
        fsq_decoder_reconstructor=True,
        fsq_decoder_terminator_progress=False,
        fsq_decoder_terminator_termination=True,
        fsq_terminator_input_space="state",
        fsq_terminator_arch="rnn",
        fsq_terminator_default_arch="small",
        fsq_reconstructor_arch="skill",
        fsq_reconstructor_output_mode="zero_grounded",
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_encoder_input_mode"] == "raw_state"
    assert settings["fsq_encoder_length_token"] is False
    assert settings["fsq_reconstructor_arch"] == "oneshot"
    assert settings["fsq_reconstructor_output_mode"] == "zero_grounded"
    assert settings["fsq_decoder_reconstructor"] is True
    assert settings["fsq_decoder_terminator_progress"] is False
    assert settings["fsq_decoder_terminator_termination"] is True
    assert settings["fsq_terminator_input_space"] == "state"
    assert settings["fsq_terminator_model"] == "rnn"
    assert settings["fsq_terminator_termination_only"] is True
    assert settings["fsq_state_rnn_terminator"] is True


def test_fsq_resnet_vision_option_resolves(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_vision_backbone="ReSNet",
        fsq_resnet_image_size=256,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_vision_backbone"] == "resnet"
    assert settings["fsq_resnet_image_size"] == 256
    assert settings["fsq_vision_suffix"] == "_resnet_frozen"
    assert settings["fsq_run_name"].endswith("_fsq333_resnet_frozen")


def test_fsq_fusion_terminator_option_is_selectable_and_names_checkpoint(
    tmp_path: Path,
) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_terminator_default_arch"] = "FuSiOn"
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_terminator_default_arch"] == "fusion"
    assert settings["fsq_run_name"].endswith("_fsq333_fusion")


def test_fsq_cond_terminator_option_is_rejected(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_terminator_default_arch"] = "cond"
    _write_manifest(tmp_path, config)

    with pytest.raises(ValueError, match=r"must be small\|fusion"):
        train_settings(config)


def test_fsq_overlap_pair_settings_resolve(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_encoder_arch="spline",
        fsq_pair_loss="overlap",
        fsq_pair_weight=0.2,
        fsq_pair_inv_temperature=7.5,
        fsq_pair_warmup=True,
        fsq_pair_warmup_epochs=50,
        fsq_pair_ramp_epochs=25,
        fsq_boundary_aug_pmax=10,
        fsq_boundary_aug_distribution="half-normal",
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


def test_fsq_directional_boundary_windows_resolve_independently(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_encoder_arch="spline",
        fsq_pair_loss="js",
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
        fsq_encoder_arch="spline",
        fsq_pair_loss="js",
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
        fsq_encoder_arch="spline",
        fsq_pair_loss="js",
        fsq_boundary_aug_pmax=10,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_pair_loss"] == "js"


def test_fsq_linear_contrastive_pair_setting_resolves(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_encoder_arch="spline",
        fsq_pair_loss="contrastive",
        fsq_boundary_aug_pmax=10,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_pair_loss"] == "contrastive"


def test_bsq5_selects_distinct_tag_and_binary_latent_contract(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_quantizer="bsq",
        bsq_code_dim=5,
        fsq_entropy=False,
        fsq_pair_loss="js",
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
    assert settings["fsq_run_name"].endswith("_bsq5")


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


def test_fsq_init_calibration_resolves_and_gets_distinct_run_name(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config.update(
        fsq_exp="recon",
        fsq_init_calibration=True,
        fsq_init_calibration_gain=0.8,
        fsq_init_calibration_samples=4096,
    )
    _write_manifest(tmp_path, config)

    settings = train_settings(config)

    assert settings["fsq_init_calibration"] is True
    assert settings["fsq_init_calibration_gain"] == "0.8"
    assert settings["fsq_init_calibration_samples"] == 4096
    assert settings["fsq_run_name"].endswith("_fsq333_recon_initcalg0p8n4096")


def test_fsq_overlap_pair_requires_positive_boundary_window(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)
    config["fsq_pair_loss"] = "overlap"
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
