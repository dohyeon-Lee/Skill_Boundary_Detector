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
                "conditioning_route": "state_cond",
                "action_loss_mode": "flow",
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
    (policy_path / "train_config.json").write_text(
        json.dumps({"dataset": {"root": str(run / "skillvla")}})
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


def test_stage1_eval_reads_dataset_from_train_config_not_fsq_location(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    policy_path = (
        Path(config["project_root"])
        / config["outputs_root"]
        / "skillVLA_stage1"
        / config["model_dir"]
        / "checkpoints/last/pretrained_model"
    )
    external_fsq = Path(config["project_root"]) / "models/external_fsq/FSQ.pt"
    external_fsq.parent.mkdir(parents=True)
    external_fsq.touch()
    policy = json.loads((policy_path / "config.json").read_text())
    policy["fsq_path"] = str(external_fsq)
    (policy_path / "config.json").write_text(json.dumps(policy))

    settings = build_settings(config)

    assert settings["fsq_path"] == external_fsq
    assert settings["skill_dataset_dir"].parts[-2:] == ("FSQ333_run", "skillvla")
    assert settings["raw_dataset_dir"] == (
        Path(config["project_root"]) / "dataset_filtered/libero_goal_full_firsthalf"
    )


def test_stage1_eval_requires_cotrained_terminator(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path, train_terminator=False)

    with pytest.raises(ValueError, match="no trained terminator"):
        build_settings(config)


def test_stage1_eval_accepts_frozen_checkpoint_predictor(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path, train_predictor=False)
    config["skill_source"] = "predictor"
    policy_path = (
        Path(config["project_root"])
        / config["outputs_root"]
        / "skillVLA_stage1"
        / config["model_dir"]
        / "checkpoints/last/pretrained_model"
    )
    policy = json.loads((policy_path / "config.json").read_text())
    policy["training_skill_source"] = "predictor"
    (policy_path / "config.json").write_text(json.dumps(policy))

    settings = build_settings(config)

    assert json.loads(settings["models_json"])[0]["has_predictor"] is True


def test_gt_timed_eval_does_not_require_terminator(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path, train_terminator=False)
    config["skill_source"] = "gt"
    config["oracle"]["advance_mode"] = "gt"

    settings = build_settings(config)

    assert json.loads(settings["models_json"])[0]["has_terminator"] is False


def test_stage1_eval_reads_current_route_and_loss_contract(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    policy_path = (
        Path(config["project_root"])
        / config["outputs_root"]
        / "skillVLA_stage1"
        / config["model_dir"]
        / "checkpoints/last/pretrained_model"
    )
    policy = json.loads((policy_path / "config.json").read_text())
    policy["conditioning_route"] = "state_skill_cond"
    policy["action_loss_mode"] = "flow_endpoint_xyz"
    (policy_path / "config.json").write_text(json.dumps(policy))

    model = json.loads(build_settings(config)["models_json"])[0]

    assert model["conditioning_route"] == "state_skill_cond"
    assert model["action_loss_mode"] == "flow_endpoint_xyz"

    policy["conditioning_route"] = "skill_cond"
    (policy_path / "config.json").write_text(json.dumps(policy))
    skill_only_model = json.loads(build_settings(config)["models_json"])[0]
    assert skill_only_model["conditioning_route"] == "skill_cond"

    policy["conditioning_route"] = "stateonly_cond"
    (policy_path / "config.json").write_text(json.dumps(policy))
    state_only_model = json.loads(build_settings(config)["models_json"])[0]
    assert state_only_model["conditioning_route"] == "stateonly_cond"


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
