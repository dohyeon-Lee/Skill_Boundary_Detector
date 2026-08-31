import sys
from pathlib import Path


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


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "target_dataset": "demo_full_full",
        "dp_n_obs_steps": 4,
        "dp_horizon": 8,
    }


def test_dp_settings_resolve_only_dp_inputs(tmp_path: Path):
    settings = dp_train_settings(_config(tmp_path))

    assert settings["target_dataset"] == "demo_full_full"
    assert settings["dp_n_action_steps"] == 5
    assert settings["dp_relative"] is False
    assert settings["raw_dataset_dir"] == tmp_path / "dataset" / "demo_full_full"


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
    settings = build_data_settings(load_config(BUILD_DATA_CONFIG))

    assert settings["target_dataset"] == "langgap_ext_full_full"
    assert settings["dp_policy"] == "langgap_ext_full_full_state_obs20"
    assert settings["dp_checkpoint"] == "100000"
    assert settings["skillset_mode"] == "std"
