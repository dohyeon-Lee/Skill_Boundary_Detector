import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lerobot.policies.skill_expert.processor_skill_expert import (
    load_skill_expert_normalization_stats,
)


_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage2/src/stage2_train_config.py"
)
_SPEC = importlib.util.spec_from_file_location("stage2_train_config", _CONFIG_PATH)
stage2_train_config = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(stage2_train_config)


def _write_checkpoint(directory: Path, config: dict, dataset_root: str | None = None) -> None:
    directory.mkdir(parents=True)
    (directory / "model.safetensors").touch()
    (directory / "config.json").write_text(json.dumps(config))
    if dataset_root is not None:
        (directory / "train_config.json").write_text(
            json.dumps({"dataset": {"root": dataset_root}})
        )


def _config(tmp_path: Path, *, policy_type: str = "skill_expert") -> dict:
    project = tmp_path / "project"
    dataset = (
        project
        / "dataset/skillvla_dataset/stage2_source/FSQ333_stage2/skillvla"
    )
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "skill_fsq_levels": [3, 3, 3],
                "proprio_grounding": "none",
                "features": {
                    "observation.state": {"shape": [8]},
                    "action": {"shape": [7]},
                },
            }
        )
    )
    dino = project / "models/dino"
    tokenizer = project / "models/paligemma-3b-pt-224-tokenizer"
    vlm_base = project / "models/pi05_base"
    dino.mkdir(parents=True)
    tokenizer.mkdir(parents=True)
    vlm_base.mkdir(parents=True)
    (vlm_base / "model.safetensors").touch()
    fsq = project / "dataset/skillvla_dataset/stage1_source/FSQ333_stage2/FSQ.pt"
    fsq.parent.mkdir(parents=True)
    fsq.touch()

    stage1_checkpoint = (
        project
        / "outputs/skillVLA_stage1/stage1_exact_name/checkpoints/last/pretrained_model"
    )
    _write_checkpoint(
        stage1_checkpoint,
        {
            "type": policy_type,
            "architecture": "cond_gemma",
            "architecture_revision": "skillvla_real_v1",
            "architecture_label": "arch0",
            "proprio_grounding": "none",
            "action_expert_variant": "gemma_300m",
            "cond_encoder_variant": "gemma_300m",
            "chunk_size": 10,
            "n_action_steps": 10,
            "max_state_dim": 32,
            "max_action_dim": 32,
            "num_inference_steps": 10,
            "min_period": 0.004,
            "max_period": 4.0,
            "time_sampling_beta_alpha": 1.5,
            "time_sampling_beta_beta": 1.0,
            "time_sampling_scale": 0.999,
            "time_sampling_offset": 0.001,
            "vision_backbone": "dino",
            "dino_model_path": str(dino),
            "dino_image_size": 224,
            "freeze_vision_encoder": True,
            "conditioning_route": "state_cond",
            "mask_actions_after_skill_end": True,
            "num_visual_latents_per_camera": 32,
            "visual_perceiver_width": 1024,
            "skill_vocab_size": 27,
            "skill_fsq_levels": [3, 3, 3],
            "transition_jitter_pmax": 15,
            "transition_jitter_early_start_pmax": 15,
            "transition_jitter_late_start_pmax": 7,
            "transition_jitter_early_end_pmax": 15,
            "transition_jitter_late_end_pmax": 7,
            "transition_jitter_distribution": "half_normal",
            "train_skill_predictor": False,
            "train_terminator": False,
            "fsq_path": str(fsq),
        },
        dataset_root=str(
            project / "dataset/skillvla_dataset/stage1_source/FSQ333_stage2/skillvla"
        ),
    )

    predictor_checkpoint = (
        project
        / "outputs/skillVLA_terminator/predictor_exact_name/checkpoints/last/pretrained_model"
    )
    _write_checkpoint(
        predictor_checkpoint,
        {
            "type": "skill_expert",
            "train_skill_predictor": True,
            "skill_fsq_levels": [3, 3, 3],
            "skill_vocab_size": 27,
            "skill_predictor_vlm_variant": "gemma_2b",
            "skill_predictor_image_size": 224,
            "skill_predictor_reader_tokens": 4,
            "skill_predictor_reader_depth": 2,
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
            "tokenizer_path": str(tokenizer),
            "tokenizer_max_length": 200,
            "fsq_path": str(fsq),
        },
    )
    return {
        "project_root": str(project),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "dataset": {
            "skillvla_root": "skillvla_dataset",
            "source": "stage2_source",
            "run": "FSQ333_stage2",
        },
        "warm_start": {
            "stage1_run": "stage1_exact_name",
            "checkpoint": "last",
        },
        "likelihood": {"layers": 4, "training_skill_source": "gt"},
    }


def _stage1_config_path(config: dict) -> Path:
    return (
        Path(config["project_root"])
        / "outputs/skillVLA_stage1/stage1_exact_name/checkpoints/last/pretrained_model/config.json"
    )


def _predictor_config_path(config: dict) -> Path:
    return (
        Path(config["project_root"])
        / "outputs/skillVLA_terminator/predictor_exact_name/checkpoints/last/pretrained_model/config.json"
    )


def test_stage2_resolver_reads_checkpoint_config_without_parsing_run_name(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["dataset"].pop("run")
    settings = stage2_train_config.build_settings(config)

    assert settings["stage1_checkpoint_path"].parts[-4:] == (
        "stage1_exact_name",
        "checkpoints",
        "last",
        "pretrained_model",
    )
    assert settings["vlm_base_path"].parts[-2:] == ("models", "pi05_base")
    assert settings["predictor_checkpoint_path"] == ""
    assert settings["architecture"] == "cond_gemma"
    assert settings["architecture_revision"] == "skillvla_real_v1"
    assert settings["architecture_label"] == "arch0"
    assert settings["proprio_grounding"] == "none"
    assert settings["skill_flow_enabled"] is False
    assert settings["stage2_mode"] == "likelihood"
    assert settings["skill_fsq_levels"] == "[3,3,3]"
    assert settings["likelihood_num_layers"] == 4
    assert settings["dsbc_noise_output_mode"] == "shared"
    assert settings["dsbc_noise_output_bound"] == pytest.approx(5.0)
    assert settings["dsbc_frs_num_steps"] == 10
    assert settings["training_skill_source"] == "gt"
    assert settings["cumulative_xyz_loss_enabled"] is False
    assert settings["cumulative_xyz_loss_weight"] == pytest.approx(0.5)
    assert settings["train_skill_predictor"] is True
    assert settings["train_terminator"] is False
    assert settings["mask_actions_after_skill_end"] is False
    assert settings["transition_jitter_pmax"] == 15
    assert settings["transition_jitter_early_start_pmax"] == 15
    assert settings["transition_jitter_late_start_pmax"] == 7
    assert settings["transition_jitter_early_end_pmax"] == 15
    assert settings["transition_jitter_late_end_pmax"] == 7
    assert settings["same_skill_batch_enabled"] is False
    assert settings["pt_run_name"] == "stage1_exact_name_last_vlmlast"


def test_stage2_base_vlm_is_default_without_predictor_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    settings = stage2_train_config.build_settings(config)

    assert settings["skill_predictor_lora"] is False
    assert settings["skill_predictor_detach_vlm"] is True
    assert settings["skill_predictor_all_layers"] is False
    assert settings["skill_predictor_deadzone_frac"] == 0.0
    assert settings["tokenizer_max_length"] == 200
    assert settings["tokenizer_path"].name == "paligemma-3b-pt-224-tokenizer"


def test_stage2_accepts_relabeled_folder_with_original_logical_code_space(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    project = Path(config["project_root"])
    original_dataset = (
        project / "dataset/skillvla_dataset/stage2_source/FSQ333_stage2/skillvla"
    )
    relabeled_dataset = original_dataset.parent.with_name(
        "FSQ333_stage2_relabeled"
    ) / "skillvla"
    relabeled_dataset.parent.mkdir(parents=True)
    original_dataset.rename(relabeled_dataset)
    info_path = relabeled_dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["skill_code_space_id"] = "FSQ333_stage2"
    info_path.write_text(json.dumps(info))

    stage1_path = _stage1_config_path(config)
    stage1 = json.loads(stage1_path.read_text())
    relabeled_fsq = (
        project
        / "dataset/skillvla_dataset/stage1_source/FSQ333_stage2_relabeled/FSQ.pt"
    )
    relabeled_fsq.parent.mkdir(parents=True)
    relabeled_fsq.touch()
    stage1["fsq_path"] = str(relabeled_fsq)
    stage1["skill_code_space_id"] = "FSQ333_stage2"
    stage1_path.write_text(json.dumps(stage1))
    config["dataset"]["run"] = "FSQ333_stage2_relabeled"

    settings = stage2_train_config.build_settings(config)

    assert settings["skill_code_space_id"] == "FSQ333_stage2"
    assert settings["skillvla_dataset_dir"] == relabeled_dataset


def test_stage2_predictor_module_fields_come_from_optional_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["warm_start"]["predictor"] = {
        "model_dir": "predictor_exact_name",
        "checkpoint": "last",
    }
    config["likelihood"]["training_skill_source"] = "predictor"

    settings = stage2_train_config.build_settings(config)

    assert settings["predictor_checkpoint_path"].parts[-4:] == (
        "predictor_exact_name",
        "checkpoints",
        "last",
        "pretrained_model",
    )
    assert settings["skill_predictor_lora"] is True
    assert settings["skill_predictor_detach_vlm"] is False
    assert settings["skill_predictor_all_layers"] is True
    assert settings["skill_predictor_deadzone_frac"] == 0.8


def test_stage2_inherits_conditioning_route_from_stage1_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    checkpoint_config = _stage1_config_path(config)
    stage1 = json.loads(checkpoint_config.read_text())
    for configured, resolved in (
        ("state_skill_cond", "state_skill_cond"),
        ("state_skill_only_cond", "state_skill_only_cond"),
        ("skill_cond", "skillonly_cond"),
        ("stateonly_cond", "stateonly_cond"),
        ("visiononly_cond", "visiononly_cond"),
    ):
        stage1["conditioning_route"] = configured
        checkpoint_config.write_text(json.dumps(stage1))
        settings = stage2_train_config.build_settings(config)
        assert settings["conditioning_route"] == resolved


def test_stage2_cumulative_xyz_loss_is_validated_exported_and_named(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["cumulative_xyz_loss"] = {"enabled": True, "weight": 0.25}

    settings = stage2_train_config.build_settings(config)

    assert settings["cumulative_xyz_loss_enabled"] is True
    assert settings["cumulative_xyz_loss_weight"] == pytest.approx(0.25)
    assert settings["pt_run_name"] == "stage1_exact_name_last_cumxyz0p25_vlmlast"

    config["cumulative_xyz_loss"] = {"enabled": True, "weight": 0.0}
    with pytest.raises(ValueError, match="weight must be finite and positive"):
        stage2_train_config.build_settings(config)


def test_stage2_layer_mix_gate_scale_and_scheduler_knobs(tmp_path: Path) -> None:
    config = _config(tmp_path)

    settings = stage2_train_config.build_settings(config)
    assert settings["likelihood_vlm_memory"] == "last"
    assert settings["likelihood_gate_lr_scale"] == pytest.approx(1.0)
    assert settings["scheduler_mode"] == "cosine_decay"
    assert settings["pt_run_name"] == "stage1_exact_name_last_vlmlast"

    config["likelihood"].update({"vlm_memory": "layer_mix", "gate_lr_scale": 10})
    config["training"] = {
        "schedule": {"lr_mode": "warmup_constant", "warmup_steps": 1000}
    }
    settings = stage2_train_config.build_settings(config)
    assert settings["likelihood_vlm_memory"] == "layer_mix"
    assert settings["likelihood_gate_lr_scale"] == pytest.approx(10.0)
    assert settings["scheduler_mode"] == "warmup_constant"
    assert settings["scheduler_warmup_steps"] == 1000
    assert settings["pt_run_name"] == "stage1_exact_name_last"

    config["likelihood"]["vlm_memory"] = "everything"
    with pytest.raises(ValueError, match="last|layer_mix"):
        stage2_train_config.build_settings(config)
    config["likelihood"]["vlm_memory"] = "layer_mix"
    config["training"]["schedule"]["lr_mode"] = "linear"
    with pytest.raises(ValueError, match="warmup_constant"):
        stage2_train_config.build_settings(config)


def test_stage2_dsbc_settings_are_exported_and_use_a_separate_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["stage2_mode"] = "dsbc"
    config["dsbc"] = {
        "noise_output_mode": "per_step",
        "noise_output_bound": 4.5,
        "frs_num_steps": 8,
        "anchor_seed": 17,
    }

    settings = stage2_train_config.build_settings(config)

    assert settings["stage2_mode"] == "dsbc"
    assert settings["dsbc_noise_output_mode"] == "per_step"
    assert settings["dsbc_noise_output_bound"] == pytest.approx(4.5)
    assert settings["dsbc_frs_num_steps"] == 8
    assert settings["dsbc_anchor_seed"] == 17
    assert settings["dsbc_reader"] == "final"
    assert settings["dsbc_latent_predictor_enabled"] is False
    assert settings["dsbc_latent_predictor_mode"] == "skill_start"
    assert settings["dsbc_latent_supervision"] == "main_chunk"
    assert settings["pt_run_name"] == "stage1_exact_name_last_dsbc_vlmlast"

    config["dsbc"]["noise_output_mode"] = "shared"
    settings = stage2_train_config.build_settings(config)
    assert settings["pt_run_name"] == (
        "stage1_exact_name_last_dsbc_shared_vlmlast"
    )

    config["cumulative_xyz_loss"] = {"enabled": True, "weight": 0.5}
    with pytest.raises(ValueError, match="unavailable in DSBC"):
        stage2_train_config.build_settings(config)


def test_stage2_all_layer_reader_and_latent_predictor_inherit_stage1_mode(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    checkpoint_config = _stage1_config_path(config)
    stage1 = json.loads(checkpoint_config.read_text())
    stage1.update(
        {
            "architecture_label": "arch0_skill",
            "skill_flow_enabled": True,
            "skill_flow_target": "canonical",
            "skill_flow_state_conditioned": False,
            "skill_flow_latent_best_of_n_enabled": True,
            "skill_flow_latent_candidates": 5,
            "skill_flow_latent_top_k": 1,
            "skill_flow_latent_assignment_timesteps": 2,
            "skill_flow_latent_ranking_route": "main",
            "skill_flow_latent_dim": 2,
            "skill_flow_latent_distribution": "uniform_square",
            "skill_flow_latent_gain_init": 0.1,
            "skill_flow_latent_fp32": True,
        }
    )
    checkpoint_config.write_text(json.dumps(stage1))
    config["stage2_mode"] = "dsbc"
    config["dsbc"] = {
        "reader": "all_layers",
        "noise_output_mode": "per_step",
        "latent_predictor": {
            "enabled": True,
            "mode": "per_chunk_final",
            "supervision": "skill_only",
            "loss_weight": 0.75,
            "timesteps": 3,
        },
    }

    settings = stage2_train_config.build_settings(config)

    assert settings["skill_flow_enabled"] is False
    assert settings["skill_flow_latent_best_of_n_enabled"] is True
    assert settings["skill_flow_latent_ranking_route"] == "main"
    assert settings["skill_flow_latent_fp32"] is True
    assert settings["dsbc_reader"] == "all_layers"
    assert settings["dsbc_latent_predictor_enabled"] is True
    assert settings["dsbc_latent_predictor_mode"] == "per_chunk_final"
    assert settings["dsbc_latent_supervision"] == "skill_only"
    assert settings["dsbc_latent_loss_weight"] == pytest.approx(0.75)
    assert settings["dsbc_latent_timesteps"] == 3
    assert settings["pt_run_name"] == (
        "stage1_exact_name_last_dsbc_allreader_zstep_zskillm3w0p75"
    )

    config["dsbc"]["latent_predictor"]["mode"] = "per_chunk_expert"
    settings = stage2_train_config.build_settings(config)
    assert settings["dsbc_latent_predictor_mode"] == "per_chunk_expert"
    assert settings["pt_run_name"] == (
        "stage1_exact_name_last_dsbc_allreader_zexpert_zskillm3w0p75"
    )


def test_stage2_skill_end_mask_is_fixed_to_full_chunk(tmp_path: Path) -> None:
    config = _config(tmp_path)

    settings = stage2_train_config.build_settings(config)
    assert settings["mask_actions_after_skill_end"] is False
    assert "mask" not in settings["pt_run_name"]

    # A legacy explicit false is accepted as the same fixed contract.
    config["mask_actions_after_skill_end"] = False
    settings = stage2_train_config.build_settings(config)
    assert settings["mask_actions_after_skill_end"] is False
    assert settings["pt_run_name"] == "stage1_exact_name_last_vlmlast"

    config["mask_actions_after_skill_end"] = True
    with pytest.raises(ValueError, match="complete action chunk"):
        stage2_train_config.build_settings(config)


def test_stage2_rejects_legacy_loss_selector(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["loss"] = "flow"

    with pytest.raises(ValueError, match="no longer has a 'loss' selector"):
        stage2_train_config.build_settings(config)


def test_stage2_exports_same_skill_different_task_batching(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["training"] = {
        "dataloader": {
            "batch_size": 16,
            "same_skill_different_task": {
                "enabled": True,
                "grouped_fraction": 0.5,
                "progress_temperature": 0.2,
            },
        }
    }

    settings = stage2_train_config.build_settings(config)

    assert settings["same_skill_batch_enabled"] is True
    assert settings["same_skill_batch_fraction"] == pytest.approx(0.5)
    assert settings["same_skill_progress_temperature"] == pytest.approx(0.2)
    assert settings["pt_run_name"] == "stage1_exact_name_last_vlmlast"


def test_stage2_name_uses_checkpoint_skill_source_and_manual_suffix(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["likelihood"]["training_skill_source"] = "predictor"
    config["warm_start"]["predictor"] = {
        "model_dir": "predictor_exact_name",
        "checkpoint": "last",
    }
    config["run"] = {"suffix": "ablation_a"}

    settings = stage2_train_config.build_settings(config)

    assert settings["pt_run_name"] == (
        "stage1_exact_name_last_vlmlast_ablation_a"
    )


def test_stage2_explicit_dataset_run_must_match_stage1(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["dataset"]["run"] = "FSQ333_different"

    with pytest.raises(ValueError, match="must match the Stage-1 dataset run"):
        stage2_train_config.build_settings(config)


def test_stage2_dataset_root_family_follows_stage1_lineage(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["dataset_root"] = "unrelated_global_root"

    settings = stage2_train_config.build_settings(config)

    assert settings["skillvla_dataset_dir"] == (
        Path(config["project_root"])
        / "dataset/skillvla_dataset/stage2_source/FSQ333_stage2/skillvla"
    )


def test_stage2_ignores_node_local_stage1_dataset_staging_path(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    checkpoint = _stage1_config_path(config).parent
    train_config_path = checkpoint / "train_config.json"
    train_config = json.loads(train_config_path.read_text())
    train_config["dataset"]["root"] = "/tmp/job_12345/skillvla"
    train_config_path.write_text(json.dumps(train_config))

    settings = stage2_train_config.build_settings(config)

    assert settings["skillvla_dataset_dir"] == (
        Path(config["project_root"])
        / "dataset/skillvla_dataset/stage2_source/FSQ333_stage2/skillvla"
    )


def test_stage2_rejects_legacy_auxiliary_section(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["auxiliary"] = {"terminator": {"train": True}}

    with pytest.raises(ValueError, match="no longer trains auxiliaries"):
        stage2_train_config.build_settings(config)


def test_stage2_gt_source_does_not_require_predictor(tmp_path: Path) -> None:
    config = _config(tmp_path)
    settings = stage2_train_config.build_settings(config)

    assert settings["training_skill_source"] == "gt"
    assert settings["predictor_checkpoint_path"] == ""


def test_stage2_predictor_source_requires_predictor_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["likelihood"]["training_skill_source"] = "predictor"

    with pytest.raises(ValueError, match="requires warm_start.predictor"):
        stage2_train_config.build_settings(config)


def test_stage2_rejects_predictor_without_trained_predictor(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["warm_start"]["predictor"] = {
        "model_dir": "predictor_exact_name",
        "checkpoint": "last",
    }
    predictor_config_path = _predictor_config_path(config)
    predictor = json.loads(predictor_config_path.read_text())
    predictor["train_skill_predictor"] = False
    predictor_config_path.write_text(json.dumps(predictor))

    with pytest.raises(ValueError, match="no trained predictor"):
        stage2_train_config.build_settings(config)


def test_stage2_rejects_legacy_vlm_source(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["warm_start"]["vlm_source"] = {
        "model_dir": "predictor_exact_name",
        "checkpoint": "last",
    }

    with pytest.raises(ValueError, match="no longer needed"):
        stage2_train_config.build_settings(config)


def test_stage2_predictor_source_rejects_full_path(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["warm_start"]["predictor"] = {"path": "/legacy/full/path"}

    with pytest.raises(ValueError, match="supports only model_dir and checkpoint"):
        stage2_train_config.build_settings(config)


def test_stage2_rejects_predictor_from_a_different_fsq_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["warm_start"]["predictor"] = {
        "model_dir": "predictor_exact_name",
        "checkpoint": "last",
    }
    project = Path(config["project_root"])
    other_fsq = project / "dataset/skillvla_dataset/stage1_source/FSQ333_other/FSQ.pt"
    other_fsq.parent.mkdir(parents=True)
    other_fsq.touch()
    predictor_config_path = _predictor_config_path(config)
    predictor = json.loads(predictor_config_path.read_text())
    predictor["fsq_path"] = str(other_fsq)
    predictor_config_path.write_text(json.dumps(predictor))

    with pytest.raises(ValueError, match="does not match the Stage-1 prior"):
        stage2_train_config.build_settings(config)


def test_stage2_resolver_rejects_non_stage1_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy.type=skill_expert"):
        stage2_train_config.build_settings(_config(tmp_path, policy_type="skill_vla"))


def test_stage2_resolver_rejects_vsa_architecture_prior(tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint_config = _stage1_config_path(config)
    stage1 = json.loads(checkpoint_config.read_text())
    stage1["architecture"] = "vsa_perceiver_crossattn"
    checkpoint_config.write_text(json.dumps(stage1))

    with pytest.raises(ValueError, match="cond_gemma"):
        stage2_train_config.build_settings(config)


@pytest.mark.parametrize(
    ("label", "revision", "target", "state_conditioned"),
    (
        ("arch0_skill", "skillvla_real_v1", "canonical", False),
        ("arch0_skill_chunk", "skillvla_real_v1", "extended_chunk", False),
        (
            "arch0_2_skill_chunk",
            "cond_expert_state_adarms_v1",
            "extended_chunk",
            True,
        ),
    ),
)
def test_stage2_accepts_stage1_skill_flow_architecture_aliases(
    tmp_path: Path,
    label: str,
    revision: str,
    target: str,
    state_conditioned: bool,
) -> None:
    config = _config(tmp_path)
    checkpoint_config = _stage1_config_path(config)
    stage1 = json.loads(checkpoint_config.read_text())
    stage1.update(
        {
            "architecture_label": label,
            "architecture_revision": revision,
            "skill_flow_enabled": True,
            "skill_flow_target": target,
            "skill_flow_state_conditioned": state_conditioned,
        }
    )
    checkpoint_config.write_text(json.dumps(stage1))

    settings = stage2_train_config.build_settings(config)

    assert settings["architecture_label"] == label
    assert settings["architecture_revision"] == revision
    assert settings["skill_flow_enabled"] is False


def test_stage2_rejects_mislabeled_skill_flow_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint_config = _stage1_config_path(config)
    stage1 = json.loads(checkpoint_config.read_text())
    stage1.update(
        {
            "architecture_label": "arch0_2_skill_chunk",
            "architecture_revision": "skillvla_real_v1",
            "skill_flow_enabled": True,
            "skill_flow_target": "extended_chunk",
            "skill_flow_state_conditioned": True,
        }
    )
    checkpoint_config.write_text(json.dumps(stage1))

    with pytest.raises(ValueError, match="contract mismatch"):
        stage2_train_config.build_settings(config)


def test_stage2_grounding_contract_is_inherited_and_validated(tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint_config = _stage1_config_path(config)
    stage1 = json.loads(checkpoint_config.read_text())
    stage1["proprio_grounding"] = "episode_start_xyz"
    checkpoint_config.write_text(json.dumps(stage1))
    dataset_info_path = (
        Path(config["project_root"])
        / "dataset/skillvla_dataset/stage2_source/FSQ333_stage2/skillvla/meta/info.json"
    )
    dataset_info = json.loads(dataset_info_path.read_text())
    dataset_info["proprio_grounding"] = "episode_start_xyz"
    dataset_info_path.write_text(json.dumps(dataset_info))

    settings = stage2_train_config.build_settings(config)
    assert settings["proprio_grounding"] == "episode_start_xyz"

    dataset_info["proprio_grounding"] = "none"
    dataset_info_path.write_text(json.dumps(dataset_info))
    with pytest.raises(ValueError, match="grounding must match"):
        stage2_train_config.build_settings(config)


def test_stage2_can_load_checkpoint_owned_normalization_stats(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    state_file = "policy_preprocessor_step_2_normalizer_processor.safetensors"
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "config": {},
                        "state_file": state_file,
                    }
                ]
            }
        )
    )
    save_file(
        {
            "observation.state.q01": torch.tensor([-1.0, -2.0]),
            "observation.state.q99": torch.tensor([1.0, 2.0]),
            "action.q01": torch.tensor([-0.5]),
            "action.q99": torch.tensor([0.5]),
        },
        checkpoint / state_file,
    )

    stats = load_skill_expert_normalization_stats(checkpoint)

    torch.testing.assert_close(
        stats["observation.state"]["q01"], torch.tensor([-1.0, -2.0])
    )
    torch.testing.assert_close(stats["action"]["q99"], torch.tensor([0.5]))
