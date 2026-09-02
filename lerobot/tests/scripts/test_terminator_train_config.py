from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/terminator/src/terminator_train_config.py"
)
SPEC = importlib.util.spec_from_file_location("terminator_train_config", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _config(
    tmp_path: Path,
    *,
    mode: str = "pt",
    terminator: bool = True,
    predictor: bool = True,
    predictor_checkpoint: str = "",
    terminator_checkpoint: str = "",
    dataset_source: str = "source",
) -> dict:
    run = "FSQ345_test"
    dataset_dir = (
        tmp_path / "dataset/skillvla_dataset" / dataset_source / run / "skillvla"
    )
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta/info.json").write_text(
        json.dumps(
            {
                "skill_fsq_levels": [3, 4, 5],
                "features": {
                    "observation.state": {"shape": [8]},
                    "action": {"shape": [7]},
                },
            }
        )
    )
    (dataset_dir.parent / "FSQ.pt").touch()

    pi_base = tmp_path / "models/pi05_base"
    pi_base.mkdir(parents=True)
    (pi_base / "model.safetensors").touch()
    tokenizer = tmp_path / "models/tokenizer"
    tokenizer.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tokenizer / name).write_text("{}")

    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "mode": mode,
        "dataset": {
            "skillvla_root": "skillvla_dataset",
            "source": dataset_source,
            "run": run,
        },
        "run": {"suffix": "test"},
        "warm_start": {
            "pi_base": "models/pi05_base",
            "tokenizer": "models/tokenizer",
            "predictor_checkpoint": predictor_checkpoint,
            "terminator_checkpoint": terminator_checkpoint,
        },
        "fsq_terminator": {
            "termination": terminator,
            "context": "prev_action",
            "default_arch": "fusion",
            "vision_backbone": "resnet",
            "freeze_vision_encoder": True,
        },
        "skill_predictor": {
            "train": predictor,
            "all_layers": True,
            "lora": {
                "enabled": True,
                "targets": "q,k,v,o",
                "rank": 8,
                "alpha": 16.0,
                "dropout": 0.0,
            },
            "reader": {
                "tokens": 4,
                "depth": 2,
                "heads": 8,
                "deadzone_frac": 0.8,
            },
            "token_access": {"image": True, "language": True},
        },
        "termination_loss": {"target_sigma": 2.0, "positive_weight": 1.0},
        "training": {
            "dataloader": {"batch_size": 2, "workers": 0, "gpus": 1},
            "optimizer": {
                "base_lr": 2.5e-5,
                "terminator_lr_scale": 1.0,
                "predictor_lr_scale": 1.0,
                "predictor_lora_lr_scale": 10.0,
                "grad_clip_norm": 1.0,
            },
            "schedule": {
                "steps": 10,
                "lr_mode": "warmup_constant",
                "warmup_steps": 1,
                "lr_decay_steps": 5,
                "log_every": 1,
                "save_every": 5,
            },
        },
        "logging": {"wandb": {"enable": True, "project": "VLA_auxiliary"}},
    }


def _write_auxiliary_checkpoint(
    tmp_path: Path,
    *,
    name: str,
    predictor: bool = True,
    terminator: bool = True,
    code_space_id: str = "FSQ345_test",
    terminator_context: str = "prev_action",
    terminator_arch: str = "fusion",
    terminator_vision_backbone: str = "resnet",
    terminator_freeze_vision_encoder: bool = True,
    training_batch_size: int = 2,
    dataset_source_lineage: list[str] | None = None,
    run_suffix_lineage: list[str] | None = None,
) -> str:
    checkpoint = tmp_path / "checkpoints" / name
    checkpoint.mkdir(parents=True)
    source = {
        "type": "skill_aux",
        "train_skill_predictor": predictor,
        "train_terminator": terminator,
        "skill_fsq_levels": [3, 4, 5],
        "skill_vocab_size": 60,
        "skill_code_space_id": code_space_id,
        "training_batch_size": training_batch_size,
        "dataset_source_lineage": dataset_source_lineage or ["pt_source"],
        "run_suffix_lineage": run_suffix_lineage or [],
        "fsq_path": str(tmp_path / f"dataset/source/{code_space_id}/FSQ.pt"),
        "terminator_context": terminator_context,
        "terminator_arch": terminator_arch,
        "terminator_vision_backbone": terminator_vision_backbone,
        "terminator_freeze_vision_encoder": terminator_freeze_vision_encoder,
        "terminator_termination_only": True,
        "terminator_end_target_sigma": 1.5,
        "terminator_end_pos_weight": 2.0,
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
        "tokenizer_path": str(tmp_path / "models/tokenizer"),
        "tokenizer_max_length": 200,
    }
    (checkpoint / "config.json").write_text(json.dumps(source))
    (checkpoint / "model.safetensors").touch()
    return str(checkpoint.relative_to(tmp_path))


def _write_relabeled_variant(config: dict, suffix: str = "relabeled_85k") -> Path:
    project = Path(config["project_root"])
    source = config["dataset"]["source"]
    source_run = config["dataset"]["run"]
    source_info = (
        project
        / config["dataset_root"]
        / config["dataset"]["skillvla_root"]
        / source
        / source_run
        / "skillvla/meta/info.json"
    )
    run = source_info.parents[2].parent / f"{source_run}_{suffix}"
    info_path = run / "skillvla/meta/info.json"
    info_path.parent.mkdir(parents=True)
    info = json.loads(source_info.read_text())
    info["skill_code_space_id"] = source_run
    info_path.write_text(json.dumps(info))
    (run / "FSQ.pt").touch()
    (run / "relabel_provenance.json").write_text(
        json.dumps({"source_run": source_run})
    )
    return run


@pytest.mark.parametrize(
    ("terminator", "predictor", "training_mode"),
    [
        (True, False, "terminator"),
        (False, True, "predictor"),
        (True, True, "predictor_terminator"),
    ],
)
def test_pt_target_combinations(tmp_path, terminator, predictor, training_mode):
    settings = MODULE.build_settings(
        _config(tmp_path, terminator=terminator, predictor=predictor)
    )

    assert settings["initialization_mode"] == "pt"
    assert settings["training_mode"] == training_mode
    assert settings["train_terminator"] is terminator
    assert settings["train_skill_predictor"] is predictor
    assert settings["wandb_project"] == "VLA_auxiliary"
    assert settings["output_dir"].parent.name == "skillVLA_terminator"
    assert settings["run_name"] == (
        f"bs2_FSQ345_test_source_{training_mode}_test"
    )


def test_all_targets_false_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Enable fsq_terminator"):
        MODULE.build_settings(
            _config(tmp_path, terminator=False, predictor=False)
        )


@pytest.mark.parametrize(
    "legacy_key",
    [
        "terminator",
        "image_only_terminator",
        "wrist_only_terminator",
        "state_only_terminator",
        "state_rnn_terminator",
    ],
)
def test_legacy_terminator_sections_are_rejected(tmp_path, legacy_key):
    config = _config(tmp_path)
    config[legacy_key] = {"train": True}
    with pytest.raises(ValueError, match="Legacy terminator sections were removed"):
        MODULE.build_settings(config)


def test_fsq_terminator_contract_is_exported(tmp_path):
    settings = MODULE.build_settings(_config(tmp_path))

    assert settings["terminator_context"] == "prev_action"
    assert settings["terminator_arch"] == "fusion"
    assert settings["terminator_vision_backbone"] == "resnet"
    assert settings["terminator_freeze_vision_encoder"] is True
    assert settings["terminator_termination_only"] is True


def test_auxiliary_training_uses_dataset_logical_code_space(tmp_path):
    config = _config(tmp_path)
    info_path = (
        tmp_path
        / "dataset/skillvla_dataset/source/FSQ345_test/skillvla/meta/info.json"
    )
    info = json.loads(info_path.read_text())
    info["skill_code_space_id"] = "FSQ345_original_taxonomy"
    info_path.write_text(json.dumps(info))

    settings = MODULE.build_settings(config)

    assert settings["skill_code_space_id"] == "FSQ345_original_taxonomy"


def test_terminator_only_selects_explicit_relabeled_dataset(tmp_path):
    config = _config(tmp_path, terminator=True, predictor=False)
    relabeled_run = _write_relabeled_variant(config)
    config["dataset"]["relabeled"] = "relabeled_85k"

    settings = MODULE.build_settings(config)

    assert settings["skillvla_dataset_dir"] == relabeled_run / "skillvla"
    assert settings["fsq_path"] == relabeled_run / "FSQ.pt"
    assert settings["skill_code_space_id"] == "FSQ345_test"
    assert settings["dataset_relabeled"] is True
    assert settings["dataset_relabel_ignored_for_predictor"] is False


@pytest.mark.parametrize("terminator", [False, True])
def test_predictor_training_ignores_relabeled_dataset(tmp_path, terminator):
    config = _config(tmp_path, terminator=terminator, predictor=True)
    _write_relabeled_variant(config)
    config["dataset"]["relabeled"] = "relabeled_85k"

    settings = MODULE.build_settings(config)

    original_run = (
        tmp_path / "dataset/skillvla_dataset/source/FSQ345_test"
    )
    assert settings["skillvla_dataset_dir"] == original_run / "skillvla"
    assert settings["fsq_path"] == original_run / "FSQ.pt"
    assert settings["dataset_relabeled"] is False
    assert settings["dataset_relabel_ignored_for_predictor"] is True


def test_pt_rejects_component_checkpoint(tmp_path):
    checkpoint = _write_auxiliary_checkpoint(tmp_path, name="predictor_pt")
    config = _config(tmp_path, predictor_checkpoint=checkpoint)

    with pytest.raises(ValueError, match="mode=pt must leave"):
        MODULE.build_settings(config)


def test_fsq_override_is_rejected(tmp_path):
    config = _config(tmp_path)
    config["warm_start"]["fsq"] = "models/some_other_fsq.pt"

    with pytest.raises(ValueError, match="FSQ checkpoint is always"):
        MODULE.build_settings(config)


def test_ft_requires_at_least_one_component_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="mode=ft requires"):
        MODULE.build_settings(_config(tmp_path, mode="ft"))


def test_ft_predictor_only_is_inferred_from_checkpoint_path(tmp_path):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="predictor_pt",
        predictor=True,
        terminator=False,
        training_batch_size=8,
        dataset_source_lineage=["libero_90_full_firsthalf_scene"],
        run_suffix_lineage=["pttag"],
    )
    settings = MODULE.build_settings(
        _config(
            tmp_path,
            mode="ft",
            predictor=False,
            terminator=True,
            predictor_checkpoint=checkpoint,
            dataset_source="libero_10_full_1",
        )
    )

    assert settings["initialization_mode"] == "ft"
    assert settings["train_skill_predictor"] is True
    assert settings["train_terminator"] is False
    assert settings["predictor_checkpoint_path"] == tmp_path / checkpoint
    assert settings["terminator_checkpoint_path"] == ""
    assert settings["batch_size"] == 8
    assert settings["training_batch_size"] == 8
    assert settings["run_name"] == (
        "bs8_FSQ345_test_libero_90_full_firsthalf_scene_"
        "libero_10_full_1_predictor_pttag_test"
    )
    assert settings["run_suffix_lineage"] == '["pttag", "test"]'


def test_ft_terminator_only_inherits_checkpoint_contract(tmp_path):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="terminator_pt",
        predictor=False,
        terminator=True,
        terminator_context="proprio",
        terminator_arch="small",
        terminator_vision_backbone="dino",
        terminator_freeze_vision_encoder=False,
    )
    settings = MODULE.build_settings(
        _config(
            tmp_path,
            mode="ft",
            predictor=True,
            terminator=False,
            terminator_checkpoint=checkpoint,
        )
    )

    assert settings["train_skill_predictor"] is False
    assert settings["train_terminator"] is True
    assert settings["terminator_context"] == "proprio"
    assert settings["terminator_arch"] == "small"
    assert settings["terminator_vision_backbone"] == "dino"
    assert settings["terminator_freeze_vision_encoder"] is False
    assert settings["terminator_end_target_sigma"] == 1.5
    assert settings["terminator_end_pos_weight"] == 2.0
    assert settings["run_name"] == "bs2_FSQ345_test_pt_source_source_terminator_test"


def test_ft_combines_different_predictor_and_terminator_checkpoints(tmp_path):
    predictor_checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="predictor_pt",
        predictor=True,
        terminator=False,
        dataset_source_lineage=["predictor_pt_source"],
    )
    terminator_checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="terminator_pt",
        predictor=False,
        terminator=True,
        dataset_source_lineage=["terminator_pt_source"],
    )
    settings = MODULE.build_settings(
        _config(
            tmp_path,
            mode="ft",
            predictor_checkpoint=predictor_checkpoint,
            terminator_checkpoint=terminator_checkpoint,
        )
    )

    assert settings["train_skill_predictor"] is True
    assert settings["train_terminator"] is True
    assert settings["predictor_checkpoint_path"] == tmp_path / predictor_checkpoint
    assert settings["terminator_checkpoint_path"] == tmp_path / terminator_checkpoint
    assert settings["run_name"] == (
        "bs2_FSQ345_test_predictor_pt_source_terminator_pt_source_"
        "source_predictor_terminator_test"
    )


def test_ft_rejects_different_component_pt_batch_sizes(tmp_path):
    predictor_checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="predictor_pt",
        predictor=True,
        terminator=False,
        training_batch_size=8,
    )
    terminator_checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="terminator_pt",
        predictor=False,
        terminator=True,
        training_batch_size=16,
    )
    config = _config(
        tmp_path,
        mode="ft",
        predictor_checkpoint=predictor_checkpoint,
        terminator_checkpoint=terminator_checkpoint,
    )

    with pytest.raises(ValueError, match="same PT batch size"):
        MODULE.build_settings(config)


def test_ft_rejects_checkpoint_without_requested_component(tmp_path):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path, name="terminator_pt", predictor=False, terminator=True
    )
    config = _config(
        tmp_path, mode="ft", predictor_checkpoint=checkpoint
    )

    with pytest.raises(ValueError, match="no trained predictor"):
        MODULE.build_settings(config)


def test_ft_rejects_different_code_space(tmp_path):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="predictor_pt",
        predictor=True,
        terminator=False,
        code_space_id="FSQ345_other",
    )
    config = _config(tmp_path, mode="ft", predictor_checkpoint=checkpoint)

    with pytest.raises(ValueError, match="code-space mismatch"):
        MODULE.build_settings(config)


def test_ft_ignores_pt_only_model_sections(tmp_path):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path, name="predictor_pt", predictor=True, terminator=False
    )
    config = _config(tmp_path, mode="ft", predictor_checkpoint=checkpoint)
    config["fsq_terminator"] = {"this_would_be_invalid_in_pt": True}
    config["skill_predictor"] = {"train": False, "reader": {"tokens": -123}}

    settings = MODULE.build_settings(config)

    assert settings["training_mode"] == "predictor"
    assert settings["skill_predictor_reader_tokens"] == 4


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("context", "state", "context must be"),
        ("default_arch", "gru", "default_arch must be"),
        ("vision_backbone", "vit", "vision_backbone must be"),
    ],
)
def test_invalid_terminator_contract_is_rejected(tmp_path, key, value, error):
    config = _config(tmp_path)
    config["fsq_terminator"][key] = value

    with pytest.raises(ValueError, match=error):
        MODULE.build_settings(config)
