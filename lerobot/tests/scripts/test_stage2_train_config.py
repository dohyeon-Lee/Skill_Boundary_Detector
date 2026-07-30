import importlib.util
import json
from pathlib import Path

import pytest


_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage2/src/stage2_train_config.py"
)
_SPEC = importlib.util.spec_from_file_location("stage2_train_config", _CONFIG_PATH)
stage2_train_config = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(stage2_train_config)


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
                "features": {
                    "observation.state": {"shape": [8]},
                    "action": {"shape": [7]},
                },
            }
        )
    )
    dino = project / "models/dino"
    tokenizer = project / "models/tokenizer"
    dino.mkdir(parents=True)
    tokenizer.mkdir(parents=True)
    fsq = project / "dataset/stage1/FSQ.pt"
    fsq.parent.mkdir(parents=True)
    fsq.touch()
    checkpoint = (
        project
        / "outputs/skillVLA_stage1/stage1_exact_name/checkpoints/last/pretrained_model"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").touch()
    (checkpoint / "train_config.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "root": str(
                        project
                        / "dataset/skillvla_dataset/stage1_source/FSQ333_stage2/skillvla"
                    )
                }
            }
        )
    )
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": policy_type,
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
                "skill_vocab_size": 27,
                "skill_fsq_levels": [3, 3, 3],
                "transition_jitter_pmax": 15,
                "transition_jitter_distribution": "half_normal",
                "train_skill_predictor": True,
                "skill_predictor_weight": 0.5,
                "skill_predictor_lr_scale": 1.0,
                "skill_predictor_all_layers": False,
                "skill_predictor_vlm_variant": "gemma_2b",
                "skill_predictor_image_size": 224,
                "skill_predictor_reader_tokens": 4,
                "skill_predictor_reader_depth": 2,
                "skill_predictor_reader_heads": 8,
                "skill_predictor_deadzone_frac": 0.0,
                "skill_predictor_attend_image": True,
                "skill_predictor_attend_language": True,
                "tokenizer_path": str(tokenizer),
                "tokenizer_max_length": 200,
                "train_terminator": True,
                "fsq_path": str(fsq),
                "terminator_freeze_vision_encoder": True,
                "terminator_lr_scale": 1.0,
                "terminator_end_target_sigma": 2.0,
                "terminator_end_pos_weight": 1.0,
            }
        )
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
        "warm_start": {"stage1_run": "stage1_exact_name", "checkpoint": "last"},
        "likelihood": {"layers": 4, "training_skill_source": "gt"},
    }


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
    assert settings["skill_fsq_levels"] == "[3,3,3]"
    assert settings["likelihood_num_layers"] == 4
    assert settings["training_skill_source"] == "gt"
    assert settings["action_loss_mode"] == "flow"
    assert settings["finetune_skill_predictor"] is False
    assert settings["finetune_terminator"] is False
    assert settings["same_skill_batch_enabled"] is False
    assert settings["pt_run_name"] == "stage1_exact_name_last_flow_gt_batchOFF"


def test_stage2_inherits_conditioning_route_from_stage1_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    checkpoint_config = (
        Path(config["project_root"])
        / "outputs/skillVLA_stage1/stage1_exact_name/checkpoints/last/pretrained_model/config.json"
    )
    stage1 = json.loads(checkpoint_config.read_text())
    stage1["conditioning_route"] = "state_skill_cond"
    checkpoint_config.write_text(json.dumps(stage1))

    settings = stage2_train_config.build_settings(config)

    assert settings["conditioning_route"] == "state_skill_cond"
    assert settings["pt_run_name"] == "stage1_exact_name_last_flow_gt_batchOFF"

    stage1["conditioning_route"] = "skill_cond"
    checkpoint_config.write_text(json.dumps(stage1))
    skill_only_settings = stage2_train_config.build_settings(config)
    assert skill_only_settings["conditioning_route"] == "skill_cond"


def test_stage2_loss_is_validated_exported_and_named(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["loss"] = "flow_endpoint_xyz"

    settings = stage2_train_config.build_settings(config)

    assert settings["action_loss_mode"] == "flow_endpoint_xyz"
    assert settings["pt_run_name"] == "stage1_exact_name_last_endpoint_gt_batchOFF"

    config["loss"] = "endpoint_only"
    with pytest.raises(ValueError, match="flow_endpoint_xyz"):
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
    assert settings["pt_run_name"] == "stage1_exact_name_last_flow_gt_batchON"


def test_stage2_same_skill_batch_keeps_random_slots_for_terminator(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["auxiliary"] = {"terminator": {"train": True}}
    config["training"] = {
        "dataloader": {
            "batch_size": 8,
            "same_skill_different_task": {
                "enabled": True,
                "grouped_fraction": 1.0,
                "progress_temperature": 0.1,
            },
        }
    }

    with pytest.raises(ValueError, match="at least one random dataloader slot"):
        stage2_train_config.build_settings(config)


def test_stage2_name_uses_checkpoint_skill_source_and_manual_suffix(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["likelihood"]["training_skill_source"] = "predictor"
    config["run"] = {"suffix": "ablation_a"}

    settings = stage2_train_config.build_settings(config)

    assert settings["pt_run_name"] == (
        "stage1_exact_name_last_flow_predictor_batchOFF_ablation_a"
    )


def test_stage2_explicit_dataset_run_must_match_stage1(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["dataset"]["run"] = "FSQ333_different"

    with pytest.raises(ValueError, match="must match the Stage-1 dataset run"):
        stage2_train_config.build_settings(config)


def test_stage2_optional_auxiliaries_are_named_and_exported(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["auxiliary"] = {
        "skill_predictor": {"train": True},
        "terminator": {"train": True},
    }
    checkpoint_config = (
        Path(config["project_root"])
        / "outputs/skillVLA_stage1/stage1_exact_name/checkpoints/last/pretrained_model/config.json"
    )
    stage1 = json.loads(checkpoint_config.read_text())
    stage1["skill_predictor_weight"] = 0.25
    stage1["skill_predictor_lr_scale"] = 0.5
    stage1["terminator_lr_scale"] = 0.75
    checkpoint_config.write_text(json.dumps(stage1))

    settings = stage2_train_config.build_settings(config)

    assert settings["finetune_skill_predictor"] is True
    assert settings["finetune_terminator"] is True
    assert settings["skill_predictor_weight"] == pytest.approx(0.25)
    assert settings["skill_predictor_lr_scale"] == pytest.approx(0.5)
    assert settings["terminator_lr_scale"] == pytest.approx(0.75)
    assert settings["pt_run_name"] == "stage1_exact_name_last_flow_gt_batchOFF"


@pytest.mark.parametrize(
    ("auxiliary", "match"),
    [
        ({"skill_predictor": {"train": True, "weight": 0.5}}, "only accepts 'train'"),
        ({"terminator": {"train": True, "lr_scale": 1.0}}, "only accepts 'train'"),
        ({"unknown": {"train": True}}, "Unknown Stage-2 auxiliary"),
    ],
)
def test_stage2_auxiliary_yaml_exposes_only_train(
    tmp_path: Path,
    auxiliary: dict,
    match: str,
) -> None:
    config = _config(tmp_path)
    config["auxiliary"] = auxiliary

    with pytest.raises(ValueError, match=match):
        stage2_train_config.build_settings(config)


def test_stage2_inherits_stage1_predictor_lora_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint_config = (
        Path(config["project_root"])
        / "outputs/skillVLA_stage1/stage1_exact_name/checkpoints/last/pretrained_model/config.json"
    )
    stage1 = json.loads(checkpoint_config.read_text())
    stage1.update(
        {
            "skill_predictor_detach_vlm": False,
            "skill_predictor_lora": True,
            "skill_predictor_lora_targets": "q,k,v,o",
            "skill_predictor_lora_rank": 8,
            "skill_predictor_lora_alpha": 16.0,
            "skill_predictor_lora_dropout": 0.0,
            "skill_predictor_lora_lr_scale": 10.0,
            "skill_predictor_all_layers": True,
            "skill_predictor_deadzone_frac": 0.8,
        }
    )
    checkpoint_config.write_text(json.dumps(stage1))

    settings = stage2_train_config.build_settings(config)

    assert settings["skill_predictor_lora"] is True
    assert settings["skill_predictor_detach_vlm"] is False
    assert settings["skill_predictor_all_layers"] is True
    assert settings["skill_predictor_deadzone_frac"] == 0.8
    assert settings["skill_predictor_lora_lr_scale"] == 10.0


def test_stage2_resolver_rejects_non_stage1_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy.type=skill_expert"):
        stage2_train_config.build_settings(_config(tmp_path, policy_type="skill_vla"))
