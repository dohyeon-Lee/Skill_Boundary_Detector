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
    terminator: bool,
    predictor: bool,
    image_terminator: bool = False,
    wrist_terminator: bool = False,
    start_comparison_terminator: bool = False,
    start_comparison_image_only_terminator: bool = False,
    endpoint_oversampling: dict | None = None,
) -> dict:
    run = "FSQ345_test"
    dataset_dir = (
        tmp_path / "dataset/skillvla_dataset/source" / run / "skillvla"
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
        "dataset": {
            "skillvla_root": "skillvla_dataset",
            "source": "source",
            "run": run,
        },
        "run": {"suffix": "test"},
        "warm_start": {
            "pi_base": "models/pi05_base",
            "tokenizer": "models/tokenizer",
            "fsq": "",
            "predictor_checkpoint": "",
        },
        "terminator": {"train": terminator},
        "image_only_terminator": {"train": image_terminator},
        "wrist_only_terminator": {"train": wrist_terminator},
        "start_comparison_terminator": {
            "train": start_comparison_terminator
        },
        "start_comparison_image_only_terminator": {
            "train": start_comparison_image_only_terminator
        },
        "skill_predictor": {
            "train": predictor,
            "lora": {"enabled": True},
        },
        "training": {
            "dataloader": {"batch_size": 2, "workers": 0, "gpus": 1},
            "endpoint_oversampling": endpoint_oversampling or {"enabled": False},
            "schedule": {"steps": 10},
        },
        "logging": {"wandb": {"project": "VLA_terminator"}},
    }


@pytest.mark.parametrize(
    ("terminator", "image_terminator", "wrist_terminator", "predictor", "mode"),
    [
        (True, False, False, False, "terminator"),
        (False, True, False, False, "image_terminator"),
        (False, False, True, False, "wrist_terminator"),
        (False, False, False, True, "predictor"),
        (True, True, False, False, "terminator_image_terminator"),
        (True, False, True, False, "terminator_wrist_terminator"),
        (True, False, False, True, "terminator_predictor"),
        (False, True, True, False, "image_terminator_wrist_terminator"),
        (False, True, False, True, "image_terminator_predictor"),
        (False, False, True, True, "wrist_terminator_predictor"),
        (
            True,
            True,
            True,
            False,
            "terminator_image_terminator_wrist_terminator",
        ),
        (
            True,
            True,
            False,
            True,
            "terminator_image_terminator_predictor",
        ),
        (
            True,
            False,
            True,
            True,
            "terminator_wrist_terminator_predictor",
        ),
        (
            False,
            True,
            True,
            True,
            "image_terminator_wrist_terminator_predictor",
        ),
        (
            True,
            True,
            True,
            True,
            "terminator_image_terminator_wrist_terminator_predictor",
        ),
    ],
)
def test_yaml_switch_combinations(
    tmp_path, terminator, image_terminator, wrist_terminator, predictor, mode
):
    settings = MODULE.build_settings(
        _config(
            tmp_path,
            terminator=terminator,
            image_terminator=image_terminator,
            wrist_terminator=wrist_terminator,
            predictor=predictor,
        )
    )
    assert settings["training_mode"] == mode
    assert settings["train_terminator"] is terminator
    assert settings["train_image_only_terminator"] is image_terminator
    assert settings["train_wrist_only_terminator"] is wrist_terminator
    assert settings["train_skill_predictor"] is predictor
    assert settings["wandb_project"] == "VLA_terminator"
    assert settings["pt_output_dir"].parent.name == "skillVLA_terminator"
    assert settings["pt_run_name"] == (
        "bs2_source_FSQ345_test_test"
    )
    assert settings["terminator_endpoint_oversampling_enabled"] is False


def test_all_yaml_switches_false_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="all false"):
        MODULE.build_settings(
            _config(
                tmp_path,
                terminator=False,
                image_terminator=False,
                predictor=False,
            )
        )


def test_start_comparison_terminator_settings_are_exported(tmp_path):
    config = _config(
        tmp_path,
        terminator=False,
        image_terminator=False,
        wrist_terminator=False,
        start_comparison_terminator=True,
        predictor=False,
    )
    config["start_comparison_terminator"].update(
        {
            "freeze_vision": True,
            "end_target_sigma": 1.0,
            "end_pos_weight": 1.5,
        }
    )
    config["training"]["optimizer"] = {
        "start_comparison_terminator_lr_scale": 2.0
    }

    settings = MODULE.build_settings(config)

    assert settings["training_mode"] == "start_comparison_terminator"
    assert settings["train_start_comparison_terminator"] is True
    assert settings["start_comparison_terminator_freeze_vision_encoder"] is True
    assert settings["start_comparison_terminator_end_target_sigma"] == 1.0
    assert settings["start_comparison_terminator_end_pos_weight"] == 1.5
    assert settings["start_comparison_terminator_lr_scale"] == 2.0
    assert settings["pt_run_name"] == "bs2_source_FSQ345_test_startcmp_test"


def test_state_free_start_comparison_settings_are_exported(tmp_path):
    config = _config(
        tmp_path,
        terminator=False,
        start_comparison_image_only_terminator=True,
        predictor=False,
    )
    config["start_comparison_image_only_terminator"].update(
        {
            "freeze_vision": True,
            "end_target_sigma": 1.0,
            "end_pos_weight": 1.5,
        }
    )
    config["training"]["optimizer"] = {
        "start_comparison_image_only_terminator_lr_scale": 2.0
    }

    settings = MODULE.build_settings(config)

    assert settings["training_mode"] == "start_comparison_image_only_terminator"
    assert settings["train_start_comparison_image_only_terminator"] is True
    assert (
        settings["start_comparison_image_only_terminator_end_target_sigma"]
        == 1.0
    )
    assert settings["start_comparison_image_only_terminator_end_pos_weight"] == 1.5
    assert settings["start_comparison_image_only_terminator_lr_scale"] == 2.0
    assert settings["pt_run_name"] == "bs2_source_FSQ345_test_startcmp_img_test"


def test_endpoint_oversampling_settings_are_exported(tmp_path):
    settings = MODULE.build_settings(
        _config(
            tmp_path,
            terminator=True,
            predictor=False,
            endpoint_oversampling={
                "enabled": True,
                "exact_end_fraction": 0.3,
                "near_end_fraction": 0.2,
                "near_end_max_distance": 4,
            },
        )
    )

    assert settings["terminator_endpoint_oversampling_enabled"] is True
    assert settings["terminator_endpoint_exact_end_fraction"] == 0.3
    assert settings["terminator_endpoint_near_end_fraction"] == 0.2
    assert settings["terminator_endpoint_near_end_max_distance"] == 4
    assert settings["pt_run_name"] == "bs2_source_FSQ345_test_endpoint_os_test"


@pytest.mark.parametrize(
    "endpoint_oversampling",
    [
        {"enabled": True, "exact_end_fraction": -0.1},
        {"enabled": True, "near_end_fraction": 1.1},
        {
            "enabled": True,
            "exact_end_fraction": 0.6,
            "near_end_fraction": 0.5,
        },
        {"enabled": True, "near_end_max_distance": 0},
        {
            "enabled": True,
            "exact_end_fraction": 0.0,
            "near_end_fraction": 0.0,
        },
    ],
)
def test_invalid_endpoint_oversampling_settings_are_rejected(
    tmp_path, endpoint_oversampling
):
    with pytest.raises(ValueError):
        MODULE.build_settings(
            _config(
                tmp_path,
                terminator=True,
                predictor=False,
                endpoint_oversampling=endpoint_oversampling,
            )
        )


def test_endpoint_oversampling_requires_a_terminator(tmp_path):
    with pytest.raises(ValueError, match="requires terminator"):
        MODULE.build_settings(
            _config(
                tmp_path,
                terminator=False,
                image_terminator=False,
                predictor=True,
                endpoint_oversampling={"enabled": True},
            )
        )
