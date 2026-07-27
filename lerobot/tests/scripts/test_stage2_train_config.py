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
                "state_cond_mode": "broadcast",
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
    assert settings["finetune_skill_predictor"] is False
    assert settings["finetune_terminator"] is False
    assert settings["pt_run_name"] == "stage1_exact_name_stage2_likelihood4_gt"


def test_stage2_inherits_token_conditioning_from_stage1_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    checkpoint_config = (
        Path(config["project_root"])
        / "outputs/skillVLA_stage1/stage1_exact_name/checkpoints/last/pretrained_model/config.json"
    )
    stage1 = json.loads(checkpoint_config.read_text())
    stage1["state_cond_mode"] = "token"
    checkpoint_config.write_text(json.dumps(stage1))

    settings = stage2_train_config.build_settings(config)

    assert settings["state_cond_mode"] == "token"
    assert settings["pt_run_name"] == "stage1_exact_name_stage2_likelihood4_gt"


def test_stage2_explicit_dataset_run_must_match_stage1(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["dataset"]["run"] = "FSQ333_different"

    with pytest.raises(ValueError, match="must match the Stage-1 dataset run"):
        stage2_train_config.build_settings(config)


def test_stage2_optional_auxiliaries_are_named_and_exported(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["auxiliary"] = {
        "skill_predictor": {"train": True, "weight": 0.25, "lr_scale": 0.5},
        "terminator": {"train": True, "lr_scale": 0.75},
    }

    settings = stage2_train_config.build_settings(config)

    assert settings["finetune_skill_predictor"] is True
    assert settings["finetune_terminator"] is True
    assert settings["skill_predictor_weight"] == pytest.approx(0.25)
    assert settings["skill_predictor_lr_scale"] == pytest.approx(0.5)
    assert settings["terminator_lr_scale"] == pytest.approx(0.75)
    assert settings["pt_run_name"].endswith("_skillpred_term")


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
