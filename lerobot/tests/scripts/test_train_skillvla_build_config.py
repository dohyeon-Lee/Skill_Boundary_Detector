import json
import sys
from pathlib import Path

import pytest


_CONFIG_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/build_data/src"
)
sys.path.insert(0, str(_CONFIG_SRC))

from train_skillVLA_config import build_settings
from check_jitter_contract import contract_matches


def _config(
    tmp_path: Path,
    threshold_mode: str,
    *,
    threshold_scale: float = 1.0,
    min_skills: int = 1,
    fsq_run_name: str = "libero_90_std_global_fsq333_dino",
    fsq_levels: list[int] | None = None,
    fsq_exp: str | None = None,
) -> dict:
    fsq_dir = tmp_path / "outputs" / "FSQ" / fsq_run_name
    fsq_dir.mkdir(parents=True, exist_ok=True)
    skillset_dir = (
        tmp_path
        / "dataset/FSQ_dataset/libero_90_full_full/FSQ_inputs"
        / "seg_libero_90_state_obs20_ck100000_std"
        / "skillset"
    )
    skillset_dir.mkdir(parents=True, exist_ok=True)
    (skillset_dir / "skillset_manifest.json").write_text(
        json.dumps(
            {
                "action": {
                    "mode": "dataset",
                    "relative_exclude_joints": ["gripper"],
                    "gripper_mode": "discrete",
                    "gripper_indices": [-1],
                    "gripper_values": [-1.0, 1.0],
                    "gripper_threshold": 0.0,
                },
                "detector": {
                    "boundary_threshold_mode": threshold_mode,
                    "boundary_threshold_scale": threshold_scale,
                    "eval_at_step": 7,
                    "min_skills": min_skills,
                    "min_skill_len": 10,
                    "n_gmm_components": 5,
                    "nms_dist": 20,
                    "replan_interval": 3,
                    "savgol_polyorder": 4,
                    "smooth_window": 7,
                },
                "mode": "std",
                "probe": {
                    "alpha": 0.1,
                    "count": 24,
                    "pca_scale_mode": "std",
                    "pca_stride": 3,
                    "pca_variance": 0.95,
                    "type": "pca_action",
                },
            }
        )
    )
    fsq_meta = {
        "fsq_dataset_root": "FSQ_dataset",
        "target_dataset": "libero_90_full_full",
        "fsq_inputs_name": "FSQ_inputs",
        "skillset_seg_name": "seg_libero_90_state_obs20_ck100000_std",
        "skillset_name": "skillset",
        "dp_run_name": "libero_90_state_obs20",
        "dp_checkpoint": "100000",
        "skillset_mode": "std",
        "skillset_min_skills": min_skills,
        "skillset_boundary_threshold_mode": threshold_mode,
        "skillset_boundary_threshold_scale": threshold_scale,
    }
    if fsq_levels is not None:
        fsq_meta["fsq_levels"] = fsq_levels
    if fsq_exp is not None:
        fsq_meta["fsq_exp"] = fsq_exp
    (fsq_dir / "fsq_meta.json").write_text(json.dumps(fsq_meta))
    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "skillvla_dataset_root": "skillvla_dataset",
        "source_dataset": "libero_90_full_firsthalf",
        "skillvla_data_mode": "pt",
        "fsq_run_name": fsq_run_name,
        "fsq_checkpoint": "0",
        "fsq_snap_to_supported": False,
    }


def test_episode_mean_has_disjoint_work_and_final_paths(tmp_path: Path) -> None:
    settings = build_settings(_config(tmp_path, "episode_mean"))

    assert settings["skillset_boundary_threshold_mode"] == "episode_mean"
    assert settings["skillset_global_threshold_source"] == ""
    assert settings["skillvla_seg_dir"].name.endswith("_episodemean_100p")
    assert settings["run_tag"].endswith("_pt")
    assert "std" not in settings["run_tag"]
    assert "episodemean" not in settings["run_tag"]
    assert "100p" not in settings["run_tag"]
    assert "halfnormal" not in settings["run_tag"]
    assert "_ms1" not in settings["skillvla_seg_dir"].name
    assert "_ms1" not in settings["run_tag"]
    assert settings["skillset_min_skill_len"] == 10
    assert "minlen" not in settings["skillvla_seg_dir"].name
    assert "minlen" not in settings["run_tag"]


def test_directional_jitter_is_exported_but_hidden_from_dataset_name(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "episode_mean")
    config.update(
        transition_jitter_early_start_pmax=10,
        transition_jitter_late_start_pmax=5,
        transition_jitter_early_end_pmax=10,
        transition_jitter_late_end_pmax=5,
    )

    settings = build_settings(config)

    assert settings["skill_pmax"] == 10
    assert settings["skill_early_start_pmax"] == 10
    assert settings["skill_late_start_pmax"] == 5
    assert settings["skill_early_end_pmax"] == 10
    assert settings["skill_late_end_pmax"] == 5
    assert "pmax" not in settings["run_tag"].lower()


def test_snap_settings_are_hidden_from_dataset_name(tmp_path: Path) -> None:
    config = _config(tmp_path, "episode_mean")
    config.update(fsq_snap_to_supported=True, fsq_snap_min_code_freq=10)

    settings = build_settings(config)

    assert settings["fsq_snap_to_supported"] is True
    assert settings["fsq_snap_min_code_freq"] == 10
    assert "snap" not in settings["run_tag"].lower()
    assert settings["run_tag"].endswith("_pt")


def test_custom_skillvla_output_suffix_is_appended_last(tmp_path: Path) -> None:
    config = _config(tmp_path, "episode_mean")
    config["skillvla_output_suffix"] = "_pmax15"

    settings = build_settings(config)

    assert settings["skillvla_output_suffix"] == "_pmax15"
    assert settings["run_tag"].endswith("_pt_pmax15")
    assert settings["skillvla_run_dir"].name == settings["run_tag"]


def test_invalid_skillvla_output_suffix_fails_early(tmp_path: Path) -> None:
    config = _config(tmp_path, "episode_mean")
    config["skillvla_output_suffix"] = "../pmax15"

    with pytest.raises(ValueError, match="skillvla_output_suffix"):
        build_settings(config)


def test_hidden_directional_contract_prevents_stale_dataset_reuse(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "skillvla"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta/info.json").write_text(
        json.dumps(
            {
                "skill_pmax": 10,
                "skill_jitter_early_start_pmax": 10,
                "skill_jitter_late_start_pmax": 5,
                "skill_jitter_early_end_pmax": 10,
                "skill_jitter_late_end_pmax": 5,
                "skill_jitter_distribution": "half_normal",
            }
        )
    )
    kwargs = {
        "storage_pmax": 10,
        "early_start_pmax": 10,
        "late_start_pmax": 5,
        "early_end_pmax": 10,
        "late_end_pmax": 5,
        "distribution": "half_normal",
    }

    assert contract_matches(dataset_dir, **kwargs) == (True, "")
    matched, detail = contract_matches(
        dataset_dir, **{**kwargs, "late_end_pmax": 4}
    )
    assert matched is False
    assert "skill_jitter_late_end_pmax" in detail


def test_exp_is_exported_but_omitted_from_skillvla_name(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        "episode_mean",
        fsq_run_name="motion_categories_v2",
        fsq_levels=[3, 3, 3],
        fsq_exp="motion_categories_v2",
    )

    settings = build_settings(config)

    assert settings["fsq_run_name"] == "motion_categories_v2"
    assert settings["fsq_exp"] == "motion_categories_v2"
    assert settings["fsq_levels_str"] == "3 3 3"
    assert settings["run_tag"] == "FSQ333_best_pt"
    assert "motion_categories_v2" not in settings["run_tag"]


def test_fsq_semantic_tags_come_from_structured_metadata(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        "episode_mean",
        fsq_run_name="unstable-folder-name",
        fsq_levels=[3, 3, 3],
        fsq_exp="custom",
    )
    meta_path = tmp_path / "outputs/FSQ/unstable-folder-name/fsq_meta.json"
    metadata = json.loads(meta_path.read_text())
    metadata.update(
        decoder_terminator_termination=True,
        vision_backbone="resnet",
        pair_loss="contrastive",
    )
    meta_path.write_text(json.dumps(metadata))

    settings = build_settings(config)

    assert settings["fsq_semantic_suffix"] == "_termRES_contrastive"
    assert settings["run_tag"] == "FSQ333_termRES_contrastive_best_pt"
    assert "custom" not in settings["run_tag"]


def test_recon_only_js_omits_terminator_tag(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        "episode_mean",
        fsq_run_name="folder-containing-termRES",
        fsq_levels=[3, 3, 3],
        fsq_exp="custom",
    )
    meta_path = tmp_path / "outputs/FSQ/folder-containing-termRES/fsq_meta.json"
    metadata = json.loads(meta_path.read_text())
    metadata.update(
        decoder_terminator_termination=False,
        vision_backbone="resnet",
        pair_loss="js",
    )
    meta_path.write_text(json.dumps(metadata))

    settings = build_settings(config)

    assert settings["fsq_semantic_suffix"] == "_js"
    assert settings["run_tag"] == "FSQ333_js_best_pt"
    assert "termRES" not in settings["run_tag"]
    assert "custom" not in settings["run_tag"]


def test_global_mean_has_explicit_threshold_identity(tmp_path: Path) -> None:
    settings = build_settings(_config(tmp_path, "global_mean"))

    assert settings["skillset_boundary_threshold_mode"] == "global_mean"
    assert settings["skillvla_seg_dir"].name.endswith("_globalmean_100p")
    assert settings["run_tag"].endswith("_pt")
    assert "globalmean" not in settings["run_tag"]
    assert "100p" not in settings["run_tag"]


def test_scaled_mean_threshold_is_named_and_exported(tmp_path: Path) -> None:
    settings = build_settings(_config(tmp_path, "global_mean", threshold_scale=0.8))

    assert settings["skillset_boundary_threshold_scale"] == 0.8
    assert settings["skillvla_seg_dir"].name.endswith("_globalmean_80p")
    assert settings["run_tag"].endswith("_pt")
    assert "globalmean" not in settings["run_tag"]
    assert "80p" not in settings["run_tag"]


def test_nondefault_min_skills_remains_in_identity(tmp_path: Path) -> None:
    settings = build_settings(_config(tmp_path, "episode_mean", min_skills=2))

    assert "_ms2_" in settings["skillvla_seg_dir"].name
    assert settings["run_tag"].endswith("_pt_ms2")


def test_ft_episode_mean_does_not_require_pt_global_threshold(tmp_path: Path) -> None:
    config = _config(tmp_path, "episode_mean")
    config["skillvla_data_mode"] = "ft"

    settings = build_settings(config)

    assert settings["skillset_global_threshold_source"] == ""
    assert settings["skillvla_seg_dir"].name.endswith("_episodemean_100p")


def test_invalid_boundary_threshold_mode_fails_early(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"fsq_meta.*episode_mean\|global_mean"):
        build_settings(_config(tmp_path, "median"))


def test_missing_fsq_metadata_fails_early(tmp_path: Path) -> None:
    config = _config(tmp_path, "episode_mean")
    (tmp_path / "outputs/FSQ/libero_90_std_global_fsq333_dino/fsq_meta.json").unlink()

    with pytest.raises(FileNotFoundError, match="fsq_meta.json"):
        build_settings(config)


def test_fsq_metadata_mode_beats_stale_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SKILLSET_MODE", "spherical")

    settings = build_settings(_config(tmp_path, "episode_mean"))

    assert settings["skillset_mode"] == "std"
    assert settings["skillset_probe_type"] == "pca_action"
    assert settings["skillset_pca_scale_mode"] == "std"


def test_segmentation_contract_comes_from_fsq_training_skillset(tmp_path: Path) -> None:
    config = _config(tmp_path, "episode_mean")
    config.update(
        {
            "skillset_action_mode": "anchor_relative",
            "skillset_gripper_mode": "continuous",
            "skillset_gripper_indices": [6, 13],
            "skillset_nms_dist": 99,
            "skillset_min_skill_len": 99,
        }
    )

    settings = build_settings(config)

    assert settings["skillset_action_mode"] == "dataset"
    assert settings["skillset_gripper_mode"] == "discrete"
    assert settings["skillset_gripper_indices"] == "-1"
    assert settings["skillset_nms_dist"] == 20
    assert settings["skillset_min_skill_len"] == 10
