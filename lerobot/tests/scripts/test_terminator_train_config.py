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
    state_terminator: bool = False,
    state_rnn_terminator: bool = False,
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
        "state_only_terminator": {
            "train": state_terminator,
            "hidden_dim": 32,
            "num_layers": 2,
            "balance_positive_negative": False,
            "termination_only": False,
        },
        "state_rnn_terminator": {
            "train": state_rnn_terminator,
            "sequence_length": 8,
            "full_skill_sequence": False,
            "input_dim": 24,
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "balance_positive_negative": False,
            "termination_only": False,
        },
        "skill_predictor": {
            "train": predictor,
            "lora": {"enabled": True},
        },
        "training": {
            "dataloader": {"batch_size": 2, "workers": 0, "gpus": 1},
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


@pytest.mark.parametrize(
    ("state_terminator", "state_rnn_terminator", "mode"),
    [
        (True, False, "state_terminator"),
        (False, True, "state_rnn_terminator"),
        (True, True, "state_terminator_state_rnn_terminator"),
    ],
)
def test_state_terminator_yaml_switches(
    tmp_path,
    state_terminator,
    state_rnn_terminator,
    mode,
):
    config = _config(
        tmp_path,
        terminator=False,
        predictor=False,
        state_terminator=state_terminator,
        state_rnn_terminator=state_rnn_terminator,
    )

    settings = MODULE.build_settings(config)

    assert settings["training_mode"] == mode
    assert settings["train_state_only_terminator"] is state_terminator
    assert settings["train_state_rnn_terminator"] is state_rnn_terminator
    assert settings["state_only_terminator_hidden_dim"] == 32
    assert settings["state_rnn_terminator_sequence_length"] == 8
    assert settings["state_rnn_terminator_full_skill_sequence"] is False
    assert settings["state_rnn_terminator_input_dim"] == 24
    assert settings["state_rnn_terminator_hidden_dim"] == 32
    assert settings["state_only_terminator_termination_only"] is False
    assert settings["state_rnn_terminator_termination_only"] is False
    assert settings["state_only_terminator_balance_positive_negative"] is False
    assert settings["state_rnn_terminator_balance_positive_negative"] is False


def test_state_terminators_do_not_require_fsq_checkpoint(tmp_path):
    config = _config(
        tmp_path,
        terminator=False,
        predictor=False,
        state_terminator=True,
        state_rnn_terminator=True,
    )
    fsq_path = (
        tmp_path
        / "dataset/skillvla_dataset/source/FSQ345_test/FSQ.pt"
    )
    fsq_path.unlink()

    settings = MODULE.build_settings(config)

    assert settings["train_state_only_terminator"] is True
    assert settings["train_state_rnn_terminator"] is True
