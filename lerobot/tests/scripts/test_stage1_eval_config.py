import json
import sys
from pathlib import Path

import pytest


_EVAL_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_eval/src"
)
sys.path.insert(0, str(_EVAL_SRC))

from stage1_eval_config import build_settings


def _checkpoint_tree(
    tmp_path: Path,
    *,
    train_terminator: bool = True,
    train_predictor: bool = True,
) -> dict:
    project = tmp_path / "project"
    run = (
        project
        / "dataset_filtered/skillvla_dataset/libero_goal_full_firsthalf/FSQ333_run"
    )
    (run / "skillvla/meta").mkdir(parents=True)
    (run / "skillvla/meta/info.json").write_text("{}")
    (run / "FSQ.pt").touch()
    (project / "models/dino").mkdir(parents=True)
    (project / "models/tokenizer").mkdir(parents=True)

    model_dir = "FSQ333_run_vsa_stage1_dino_frozen_skillpred_term"
    policy_path = (
        project
        / "outputs_filtered/skillVLA_stage1"
        / model_dir
        / "checkpoints/last/pretrained_model"
    )
    policy_path.mkdir(parents=True)
    for name in (
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (policy_path / name).touch()
    (policy_path / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_expert",
                "fsq_path": str(run / "FSQ.pt"),
                "dino_model_path": str(project / "models/dino"),
                "terminator_dino_model_path": str(project / "models/dino"),
                "tokenizer_path": str(project / "models/tokenizer"),
                "train_skill_predictor": train_predictor,
                "train_terminator": train_terminator,
                "n_action_steps": 10,
                "chunk_size": 10,
            }
        )
    )
    return {
        "project_root": str(project),
        "dataset_root": "dataset_filtered",
        "outputs_root": "outputs_filtered",
        "model_dir": model_dir,
        "checkpoint": "last",
        "output_name": "smoke",
        "target_task": "libero_goal",
        "task_ids": [0, 1],
        "oracle": {"episode_exact": False, "advance_mode": "terminator"},
        "terminator": {"end_mode": "or"},
        "logging": {"wandb": {"enable": False}},
    }


def test_stage1_eval_uses_checkpoint_contract_and_local_output_root(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    settings = build_settings(config)

    assert settings["policy_path"].parts[-4:-2] == (
        config["model_dir"],
        "checkpoints",
    )
    assert settings["fsq_path"].name == "FSQ.pt"
    assert settings["skill_dataset_dir"].name == "skillvla"
    assert settings["eval_init_states_path"] == ""
    assert settings["eval_out_dir"] == (
        _EVAL_SRC.parent / "outputs/smoke"
    )
    assert settings["target_task"] == "libero_goal"
    assert settings["task_ids"] == "[0,1]"


def test_stage1_eval_requires_cotrained_terminator(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path, train_terminator=False)

    with pytest.raises(ValueError, match="train_terminator=true"):
        build_settings(config)


def test_episode_exact_mode_requires_init_state_map(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    config["oracle"]["episode_exact"] = True

    with pytest.raises(FileNotFoundError, match="episode_exact=true"):
        build_settings(config)


def test_multi_model_entries_can_compare_gt_and_predictor(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    model_dir = config.pop("model_dir")
    config["models"] = [
        {"model_dir": model_dir, "skill_source": "gt", "label": "GT"},
        {
            "model_dir": model_dir,
            "skill_source": "predictor",
            "label": "Predicted",
        },
    ]

    settings = build_settings(config)
    models = json.loads(settings["models_json"])

    assert settings["model_count"] == 2
    assert [model["label"] for model in models] == ["GT", "Predicted"]
    assert [model["skill_source"] for model in models] == ["gt", "predictor"]
    assert all(model["eval_init_states_path"] == "" for model in models)


def test_predictor_panel_requires_trained_predictor(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path, train_predictor=False)
    config["skill_source"] = "predictor"

    with pytest.raises(ValueError, match="no trained predictor"):
        build_settings(config)
