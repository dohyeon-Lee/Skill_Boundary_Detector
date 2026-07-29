import sys
from pathlib import Path

import pytest


_CONFIG_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/build_data/src"
)
sys.path.insert(0, str(_CONFIG_SRC))

from train_skillVLA_config import build_settings


def _config(tmp_path: Path, threshold_mode: str) -> dict:
    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "skillvla_dataset_root": "skillvla_dataset",
        "source_dataset": "libero_90_full_firsthalf",
        "skillvla_data_mode": "pt",
        "dp_policy_name": "libero_90_state_obs20",
        "dp_checkpoint": "100000",
        "fsq_run_name": "libero_90_std_global_fsq333_dino",
        "fsq_checkpoint": "0",
        "fsq_snap_to_supported": False,
        "skillset_mode": "std",
        "skillset_boundary_threshold_mode": threshold_mode,
    }


def test_episode_mean_has_disjoint_work_and_final_paths(tmp_path: Path) -> None:
    settings = build_settings(_config(tmp_path, "episode_mean"))

    assert settings["skillset_boundary_threshold_mode"] == "episode_mean"
    assert settings["skillset_global_threshold_source"] == ""
    assert settings["skillvla_seg_dir"].name.endswith("_episodemean_100p")
    assert "_pt_episodemean_100p_" in settings["run_tag"]
    assert "_ms1" not in settings["skillvla_seg_dir"].name
    assert "_ms1" not in settings["run_tag"]


def test_global_mean_has_explicit_threshold_identity(tmp_path: Path) -> None:
    settings = build_settings(_config(tmp_path, "global_mean"))

    assert settings["skillset_boundary_threshold_mode"] == "global_mean"
    assert settings["skillvla_seg_dir"].name.endswith("_globalmean_100p")
    assert "_pt_globalmean_100p_" in settings["run_tag"]


def test_scaled_mean_threshold_is_named_and_exported(tmp_path: Path) -> None:
    config = _config(tmp_path, "global_mean")
    config["skillset_boundary_threshold_scale"] = 0.8

    settings = build_settings(config)

    assert settings["skillset_boundary_threshold_scale"] == 0.8
    assert settings["skillvla_seg_dir"].name.endswith("_globalmean_80p")
    assert "_pt_globalmean_80p_" in settings["run_tag"]


def test_nondefault_min_skills_remains_in_identity(tmp_path: Path) -> None:
    config = _config(tmp_path, "episode_mean")
    config["skillset_min_skills"] = 2

    settings = build_settings(config)

    assert "_ms2_" in settings["skillvla_seg_dir"].name
    assert "_ms2_" in settings["run_tag"]


def test_ft_episode_mean_does_not_require_pt_global_threshold(tmp_path: Path) -> None:
    config = _config(tmp_path, "episode_mean")
    config["skillvla_data_mode"] = "ft"

    settings = build_settings(config)

    assert settings["skillset_global_threshold_source"] == ""
    assert settings["skillvla_seg_dir"].name.endswith("_episodemean_100p")


def test_invalid_boundary_threshold_mode_fails_early(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"episode_mean\|global_mean"):
        build_settings(_config(tmp_path, "median"))
