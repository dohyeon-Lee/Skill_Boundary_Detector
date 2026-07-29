import json
import sys
from pathlib import Path

import pytest


_EVAL_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage2_eval/src"
)
sys.path.insert(0, str(_EVAL_SRC))

from stage2_eval_config import build_settings


def _checkpoint_tree(tmp_path: Path, *, policy_type: str = "skill_vla_stage2") -> dict:
    project = tmp_path / "project"
    run = (
        project
        / "dataset_filtered/skillvla_dataset/libero_90_full_firsthalf/FSQ333_run"
    )
    skill_dataset = run / "skillvla"
    (skill_dataset / "meta").mkdir(parents=True)
    (skill_dataset / "meta/info.json").write_text("{}")
    (run / "FSQ.pt").touch()
    for directory in ("models/dino", "models/tokenizer"):
        (project / directory).mkdir(parents=True)

    model_dir = "stage1_050000_endpoint_gt_batchON"
    policy_path = (
        project
        / "outputs_filtered/skillVLA_stage2"
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
                "type": policy_type,
                "fsq_path": str(run / "FSQ.pt"),
                "dino_model_path": str(project / "models/dino"),
                "terminator_dino_model_path": str(project / "models/dino"),
                "tokenizer_path": str(project / "models/tokenizer"),
                "train_skill_predictor": True,
                "train_terminator": True,
                "n_action_steps": 10,
                "chunk_size": 10,
            }
        )
    )
    (policy_path / "train_config.json").write_text(
        json.dumps({"dataset": {"root": str(skill_dataset)}})
    )
    return {
        "project_root": str(project),
        "outputs_root": "outputs_filtered",
        "model_dir": model_dir,
        "checkpoint": "last",
        "output_name": "smoke",
        "target_task": "libero_90",
        "task_ids": [0, 1],
        "oracle": {"episode_exact": False, "advance_mode": "terminator"},
        "terminator": {"end_mode": "or"},
        "logging": {"wandb": {"enable": False}},
    }


def test_stage2_eval_uses_self_contained_checkpoint_contract(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    settings = build_settings(config)
    model = json.loads(settings["models_json"])[0]

    assert settings["policy_path"].parts[-4:-2] == (
        config["model_dir"],
        "checkpoints",
    )
    assert settings["skill_dataset_dir"].name == "skillvla"
    assert settings["raw_dataset_dir"].name == "libero_90_full_firsthalf"
    assert settings["eval_out_dir"] == _EVAL_SRC.parent / "outputs/smoke"
    assert model["skill_source"] == "gt"
    assert model["eval_init_states_path"] == ""


def test_stage2_eval_can_compare_gt_and_predictor(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    model_dir = config.pop("model_dir")
    config["models"] = [
        {"model_dir": model_dir, "skill_source": "gt", "label": "GT"},
        {
            "model_dir": model_dir,
            "skill_source": "predictor",
            "label": "Predictor",
        },
    ]

    settings = build_settings(config)
    models = json.loads(settings["models_json"])

    assert settings["model_count"] == 2
    assert [model["skill_source"] for model in models] == ["gt", "predictor"]
    assert [model["label"] for model in models] == ["GT", "Predictor"]


def test_stage2_eval_rejects_legacy_policy_type(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path, policy_type="skill_vla")

    with pytest.raises(ValueError, match="skill_vla_stage2"):
        build_settings(config)


def test_stage2_episode_exact_requires_init_state_map(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    config["oracle"]["episode_exact"] = True

    with pytest.raises(FileNotFoundError, match="episode_exact=true"):
        build_settings(config)
