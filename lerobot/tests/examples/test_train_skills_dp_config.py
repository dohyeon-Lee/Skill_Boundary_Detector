import sys
from pathlib import Path

import pytest


CONFIG_SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skills/src"
sys.path.insert(0, str(CONFIG_SRC))

from train_skills_config import (  # noqa: E402
    build_data_settings,
    dp_train_settings,
    load_config,
)


BUILD_DATA_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skills/build_data/build_data_config.yaml"
)
DP_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skills/DP/dp_config.yaml"
)
DP_TRAIN_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skills/DP/src/train_dp.sbatch"
)
BUILD_DATA_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skills/build_data/src/build_skillset.sbatch"
)


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "target_dataset": "demo_full_full",
        "dp_n_obs_steps": 4,
        "dp_future_action_horizon": 8,
    }


def test_dp_settings_resolve_only_dp_inputs(tmp_path: Path):
    settings = dp_train_settings(_config(tmp_path))

    assert settings["target_dataset"] == "demo_full_full"
    assert settings["dp_n_action_steps"] == 8
    assert settings["dp_action_sequence_mode"] == "future_only"
    assert settings["dp_trajectory_horizon"] == 8
    assert settings["dp_future_action_horizon"] == 8
    assert settings["dp_drop_n_last_frames"] == 7
    assert settings["dp_relative"] is False
    assert settings["dp_unet_size"] == "base"
    assert settings["dp_down_dims"] == [512, 1024, 2048]
    assert settings["dp_down_dims_arg"] == "[512,1024,2048]"
    assert settings["dp_amp"] is False
    assert settings["dp_proprio_grounding"] == "none"
    assert settings["raw_dataset_dir"] == tmp_path / "dataset" / "demo_full_full"


def test_future_only_dp_decouples_observation_and_action_horizons(tmp_path: Path):
    config = _config(tmp_path)
    config.update(
        dp_n_obs_steps=20,
        dp_future_action_horizon=24,
        dp_n_action_steps=5,
        dp_action_sequence_mode="future_only",
    )

    settings = dp_train_settings(config)

    assert settings["dp_n_obs_steps"] == 20
    assert settings["dp_future_action_horizon"] == 24
    assert settings["dp_n_action_steps"] == 5
    assert settings["dp_drop_n_last_frames"] == 23


def test_future_only_dp_defaults_execution_length_to_future_horizon(tmp_path: Path):
    config = _config(tmp_path)
    config["dp_action_sequence_mode"] = "future_only"

    settings = dp_train_settings(config)

    assert settings["dp_n_action_steps"] == 8


def test_history_reconstruction_derives_all_internal_temporal_settings(tmp_path: Path):
    config = _config(tmp_path)
    config.update(
        dp_n_obs_steps=20,
        dp_future_action_horizon=24,
        dp_action_sequence_mode="history_reconstruction",
    )

    settings = dp_train_settings(config)

    assert settings["dp_n_obs_steps"] == 20
    assert settings["dp_future_action_horizon"] == 24
    assert settings["dp_trajectory_horizon"] == 43
    assert settings["dp_n_action_steps"] == 24
    assert settings["dp_drop_n_last_frames"] == 23


def test_real_dp_yaml_exposes_only_two_numeric_temporal_knobs():
    config = load_config(DP_CONFIG)
    settings = dp_train_settings(config)

    assert "dp_horizon" not in config
    assert "dp_n_action_steps" not in config
    assert "dp_drop_n_last_frames" not in config
    expected_horizon = int(config["dp_future_action_horizon"])
    if config["dp_action_sequence_mode"] == "history_reconstruction":
        expected_horizon += int(config["dp_n_obs_steps"]) - 1
    assert settings["dp_trajectory_horizon"] == expected_horizon
    assert settings["dp_future_action_horizon"] == int(config["dp_future_action_horizon"])
    assert settings["dp_n_action_steps"] == int(config["dp_future_action_horizon"])
    assert settings["dp_drop_n_last_frames"] == int(config["dp_future_action_horizon"]) - 1
    assert settings["dp_amp"] is bool(config["dp_amp"])


def test_small_unet_adds_only_the_small_suffix(tmp_path: Path):
    config = _config(tmp_path)
    config["dp_unet_size"] = "small"

    settings = dp_train_settings(config)

    assert settings["dp_down_dims"] == [128, 256, 512]
    assert settings["dp_policy"] == "dp_demo_full_full_future_only_obs4_future8_small"


def test_episode_start_grounding_adds_grounded_suffix(tmp_path: Path):
    config = _config(tmp_path)
    config["dp_proprio_grounding"] = "episode_start_xyz"

    settings = dp_train_settings(config)

    assert settings["dp_proprio_grounding"] == "episode_start_xyz"
    assert settings["dp_policy"] == "dp_demo_full_full_grounded_future_only_obs4_future8"


def test_episode_start_grounding_rejects_relative_action_mode(tmp_path: Path):
    config = _config(tmp_path)
    config.update(dp_proprio_grounding="episode_start_xyz", dp_relative=True)

    with pytest.raises(ValueError, match="cannot be combined with dp_relative"):
        dp_train_settings(config)


def test_dp_settings_reject_unknown_unet_size(tmp_path: Path):
    config = _config(tmp_path)
    config["dp_unet_size"] = "tiny"

    with pytest.raises(ValueError, match="dp_unet_size must be one of"):
        dp_train_settings(config)


def test_dp_train_script_uses_bf16_toggle_and_inline_cuda_guard():
    script = DP_TRAIN_SCRIPT.read_text()

    assert "prepare_inline_cuda_guard" in script
    assert "require_cuda_or_requeue" not in script
    assert "ACCELERATE_MIXED_PRECISION=bf16" in script
    assert '--policy.use_amp="${DP_AMP}"' in script
    assert '--policy.down_dims="${DP_DOWN_DIMS_ARG}"' in script
    assert 'handle_inline_cuda_guard_exit "${TRAIN_STATUS}"' in script


def test_default_state_encoder_is_omitted_from_policy_name(tmp_path: Path):
    config = _config(tmp_path)

    state_settings = dp_train_settings(config)
    resnet_settings = dp_train_settings({**config, "dp_vision": "resnet"})

    assert state_settings["dp_policy"] == "dp_demo_full_full_future_only_obs4_future8"
    assert resnet_settings["dp_policy"] == "dp_demo_full_full_resnet_future_only_obs4_future8"


def test_dp_settings_reject_invalid_execution_length(tmp_path: Path):
    config = _config(tmp_path)
    config.update(dp_action_sequence_mode="future_only", dp_n_action_steps=9)

    with pytest.raises(ValueError, match=r"dp_n_action_steps must be in \[1, 8\]"):
        dp_train_settings(config)


def test_dp_settings_reject_removed_observation_aligned_mode(tmp_path: Path):
    config = _config(tmp_path)
    config["dp_action_sequence_mode"] = "observation_aligned"

    with pytest.raises(ValueError, match="dp_action_sequence_mode must be one of"):
        dp_train_settings(config)


def test_dp_settings_do_not_read_downstream_fsq_or_bsq_config(tmp_path: Path):
    config = _config(tmp_path)
    config.update(
        fsq_quantizer="not-a-mapping",
        fsq_autoencoder={"mode": "invalid"},
        bsq_code_dim="invalid",
    )

    settings = dp_train_settings(config)

    assert settings["target_dataset"] == "demo_full_full"
    assert not any(key.startswith("fsq_") or key.startswith("bsq_") for key in settings)


def test_dp_joint_relative_mode_is_preserved(tmp_path: Path):
    config = _config(tmp_path)
    config["dp_relative"] = True

    settings = dp_train_settings(config)

    assert settings["dp_relative"] is True


def test_build_data_settings_ignore_downstream_quantizer_and_autoencoder(tmp_path: Path):
    config = _config(tmp_path)
    config.update(
        fsq_quantizer="not-a-mapping",
        fsq_autoencoder={"mode": "invalid"},
        bsq_code_dim="invalid",
        skillset_mode="std",
    )

    settings = build_data_settings(config)

    assert settings["target_dataset"] == "demo_full_full"
    assert settings["fsq_inputs_dir"] == (
        tmp_path / "dataset" / "FSQ_dataset" / "demo_full_full" / "FSQ_inputs"
    )
    assert settings["skillset_mode"] == "std"
    assert "_std_episodemean_100p" in settings["skillset_seg_name"]
    assert "fsq_quantizer" not in settings
    assert "fsq_autoencoder_mode" not in settings
    assert "bsq_code_dim" not in settings


def test_real_build_data_yaml_resolves_without_fsq_config():
    config = load_config(BUILD_DATA_CONFIG)
    settings = build_data_settings(config)

    assert settings["target_dataset"] == config["target_dataset"]
    assert settings["dp_policy"] == config["dp_run_name"]
    assert settings["dp_checkpoint"] == str(config["dp_checkpoint"])
    assert settings["skillset_mode"] == config["skillset_mode"]


def test_build_inherits_grounding_from_checkpoint_instead_of_yaml():
    config_text = BUILD_DATA_CONFIG.read_text()
    script = BUILD_DATA_SCRIPT.read_text()

    assert "skillset_proprio_grounding" not in config_text
    assert "SKILLSET_PROPRIO_GROUNDING" not in script
    assert "--proprio_grounding" not in script
    assert "inherited automatically from the DP checkpoint" in script
