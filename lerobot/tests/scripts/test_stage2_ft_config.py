from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/FT/src/ft_train_config.py"
)
SPEC = importlib.util.spec_from_file_location("stage2_ft_config", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _touch_checkpoint(path: Path, config: dict) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    for name in (
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (path / name).touch()
    (path / "train_config.json").write_text(
        json.dumps(
            {
                "policy": config,
                "batch_size": 32,
                "num_workers": 6,
                "steps": 500000,
                "log_freq": 100,
                "save_freq": 10000,
                "optimizer": {"lr": 2.5e-5},
                "scheduler": {
                    "type": "warmup_constant",
                    "num_warmup_steps": 1000,
                },
                "wandb": {"enable": True, "project": "VLA_stage2"},
            }
        )
    )


def _config(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    legacy_root = Path("/retired/server/Skill_Boundary_Detector")
    run_tag = "FSQ333_test"
    ft_run_tag = "FSQ333_ft"
    current = {
        "dino": tmp_path / "models/dinov3-vitl16",
        "tokenizer": tmp_path / "models/tokenizer",
        "vlm_base": tmp_path / "models/pi05_base",
        "stage1": tmp_path / "outputs/skillVLA_stage1/prior/checkpoints/last/pretrained_model",
        "predictor": tmp_path
        / "outputs/skillVLA_terminator/predictor/checkpoints/last/pretrained_model",
        "fsq": tmp_path / f"dataset/skillvla_dataset/old_source/{run_tag}/FSQ.pt",
        "terminator_dino": tmp_path / "models/terminator-dino",
    }
    for key, path in current.items():
        if key == "fsq":
            path.parent.mkdir(parents=True)
            path.write_bytes(b"same-fsq")
        else:
            path.mkdir(parents=True)

    parent = tmp_path / "outputs/skillVLA_stage2/parent/checkpoints/last/pretrained_model"
    _touch_checkpoint(
        parent,
        {
            "type": "skill_vla_stage2",
            "stage2_mode": "dsbc",
            "dsbc_latent_predictor_enabled": True,
            "dsbc_latent_predictor_mode": "per_chunk_expert",
            "training_skill_source": "gt",
            "train_terminator": False,
            "skill_fsq_levels": [3, 3, 3],
            "skill_vocab_size": 27,
            "max_state_dim": 32,
            "max_action_dim": 32,
            "dino_model_path": str(legacy_root / "models/dinov3-vitl16"),
            "tokenizer_path": str(legacy_root / "models/tokenizer"),
            "vlm_base_path": str(legacy_root / "models/pi05_base"),
            "stage1_checkpoint_path": str(
                legacy_root
                / "outputs/skillVLA_stage1/prior/checkpoints/last/pretrained_model"
            ),
            "skill_predictor_checkpoint_path": str(
                legacy_root
                / "outputs/skillVLA_terminator/predictor/checkpoints/last/pretrained_model"
            ),
            "fsq_path": str(
                legacy_root / f"dataset/skillvla_dataset/old_source/{run_tag}/FSQ.pt"
            ),
            "terminator_dino_model_path": str(
                legacy_root / "models/terminator-dino"
            ),
        },
    )

    dataset = tmp_path / f"dataset/skillvla_dataset/new_source/{ft_run_tag}/skillvla"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "repo_id": "skillvla/new_source",
                "skill_fsq_levels": [3, 3, 3],
                "features": {
                    "observation.state": {"shape": [8]},
                    "action": {"shape": [7]},
                },
            }
        )
    )
    (dataset.parent / "FSQ.pt").write_bytes(b"same-fsq")

    config = {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "dataset": {
            "skillvla_root": "skillvla_dataset",
            "source": "new_source",
            "run": "",
        },
        "warm_start": {
            "outputs_subdir": "skillVLA_stage2",
            "stage2_run": "parent",
            "checkpoint": "last",
        },
        "run": {"name": "portable_ft"},
    }
    return config, current


def test_ft_rebases_all_checkpoint_owned_project_paths(tmp_path: Path) -> None:
    config, current = _config(tmp_path)

    settings = MODULE.build_settings(config)

    assert settings["initialization_mode"] == "stage2"
    assert settings["skillvla_dataset_dir"].parent.name == "FSQ333_ft"
    assert settings["ft_train_scope"] == "noise_predictor+latent_predictor"
    assert settings["dsbc_latent_predictor_mode"] == "per_chunk_expert"
    assert settings["policy_dino_model_path"] == str(current["dino"])
    assert settings["policy_tokenizer_path"] == str(current["tokenizer"])
    assert settings["policy_vlm_base_path"] == str(current["vlm_base"])
    assert settings["policy_stage1_checkpoint_path"] == str(current["stage1"])
    assert settings["policy_skill_predictor_checkpoint_path"] == str(
        current["predictor"]
    )
    assert settings["policy_fsq_path"] == current["fsq"]
    assert settings["policy_terminator_dino_model_path"] == str(
        current["terminator_dino"]
    )


def test_ft_preserves_hub_model_references(tmp_path: Path) -> None:
    assert MODULE._relocate_checkpoint_reference(
        tmp_path,
        "namespace/model-name",
        field="dino_model_path",
        require_local=True,
    ) == "namespace/model-name"


def test_ft_rejects_missing_rebased_absolute_model_path(tmp_path: Path) -> None:
    legacy_path = "/retired/server/Skill_Boundary_Detector/models/missing-dino"

    with pytest.raises(FileNotFoundError) as exc_info:
        MODULE._relocate_checkpoint_reference(
            tmp_path,
            legacy_path,
            field="dino_model_path",
            require_local=True,
        )

    assert str(tmp_path / "models/missing-dino") in str(exc_info.value)
    assert legacy_path in str(exc_info.value)


def test_ft_rejects_non_dsbc_parent(tmp_path: Path) -> None:
    config, _ = _config(tmp_path)
    parent = (
        tmp_path
        / "outputs/skillVLA_stage2/parent/checkpoints/last/pretrained_model/config.json"
    )
    payload = json.loads(parent.read_text())
    payload["stage2_mode"] = "likelihood"
    parent.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="requires a DSBC parent"):
        MODULE.build_settings(config)


def test_ft_rejects_parent_without_latent_predictor(tmp_path: Path) -> None:
    config, _ = _config(tmp_path)
    parent = (
        tmp_path
        / "outputs/skillVLA_stage2/parent/checkpoints/last/pretrained_model/config.json"
    )
    payload = json.loads(parent.read_text())
    payload["dsbc_latent_predictor_enabled"] = False
    parent.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="latent predictor enabled"):
        MODULE.build_settings(config)


def test_ft_can_use_stage2_checkpoint_as_recipe_for_fresh_stage1_start(
    tmp_path: Path,
) -> None:
    config, current = _config(tmp_path)
    config["initialization"] = {"mode": "stage1"}
    config["run"] = {}

    settings = MODULE.build_settings(config)

    assert settings["initialization_mode"] == "stage1"
    assert settings["skillvla_dataset_dir"].parent.name == "FSQ333_ft"
    assert settings["stage2_checkpoint_path"].parent.name == "last"
    assert settings["stage2_train_config_path"] == (
        settings["stage2_checkpoint_path"] / "train_config.json"
    )
    assert settings["policy_stage1_checkpoint_path"] == str(current["stage1"])
    assert settings["batch_size"] == 32
    assert settings["num_workers"] == 6
    assert settings["steps"] == 500000
    assert settings["pt_run_name"].endswith("_new_source_ft_fresh")
    assert settings["pt_output_dir"].parent == tmp_path / "outputs/skillVLA_FT"


def test_ft_rejects_unknown_initialization_mode(tmp_path: Path) -> None:
    config = {"initialization": {"mode": "checkpoint_magic"}}

    with pytest.raises(ValueError, match=r"stage2\|stage1"):
        MODULE.build_settings(config)
