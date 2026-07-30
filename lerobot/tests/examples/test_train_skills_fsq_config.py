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
    assert settings["fsq_run_name"] == (
        "demo_full_full_state_obs20_std_episodemean_80p_trial_fsq333"
    )


def test_fsq_selected_skillset_requires_manifest(tmp_path: Path) -> None:
    config = _minimal_fsq_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="skillset manifest not found"):
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
