import json
import sys
from pathlib import Path


_STAGE1_SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/stage1/src"
sys.path.insert(0, str(_STAGE1_SRC))

from stage1_train_config import _read_dataset_contract, build_settings


def test_stage1_reads_skill_space_from_dataset_metadata(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "meta"
    metadata_dir.mkdir()
    (metadata_dir / "info.json").write_text(
        json.dumps(
            {
                "skill_fsq_levels": [3, 3, 3],
                "skill_pmax": 15,
                "skill_jitter_distribution": "half-normal",
                "features": {
                    "observation.state": {"shape": [8]},
                    "action": {"shape": [7]},
                },
            }
        )
    )

    contract = _read_dataset_contract(tmp_path, "FSQ333_example")

    assert contract == {
        "levels": [3, 3, 3],
        "state_dim": 8,
        "action_dim": 7,
        "jitter_pmax": 15,
        "jitter_distribution": "half_normal",
    }


def test_stage1_exports_stage3a_predictor_contract(tmp_path: Path) -> None:
    project = tmp_path / "project"
    run = "FSQ333_example"
    dataset = project / "dataset/skillvla_dataset/source" / run / "skillvla"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "skill_fsq_levels": [3, 3, 3],
                "skill_pmax": 15,
                "skill_jitter_distribution": "half_normal",
                "features": {
                    "observation.state": {"shape": [8]},
                    "action": {"shape": [7]},
                },
            }
        )
    )
    pi_base = project / "models/pi05_base"
    dino = project / "models/dino"
    tokenizer = project / "models/tokenizer"
    pi_base.mkdir(parents=True)
    dino.mkdir(parents=True)
    tokenizer.mkdir(parents=True)
    (pi_base / "model.safetensors").touch()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tokenizer / name).touch()

    config = {
        "project_root": str(project),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "dataset": {
            "skillvla_root": "skillvla_dataset",
            "source": "source",
            "run": run,
        },
        "warm_start": {
            "pi_base": "models/pi05_base",
            "tokenizer": "models/tokenizer",
        },
        "vision": {"dino_model": "models/dino", "freeze": True},
        "terminator": {"train": False},
        "run": {"suffix": "pred_lora_all_dz08"},
    }
    settings = build_settings(config)

    assert settings["skill_predictor_lora"] is True
    assert settings["skill_predictor_detach_vlm"] is False
    assert settings["skill_predictor_all_layers"] is True
    assert settings["skill_predictor_lora_targets"] == "q,k,v,o"
    assert settings["skill_predictor_lora_rank"] == 8
    assert settings["skill_predictor_lora_lr_scale"] == 10.0
    assert settings["skill_predictor_deadzone_frac"] == 0.8
    assert settings["action_loss_mode"] == "flow"
    assert settings["pt_run_name"].endswith("_flow_pred_lora_all_dz08")

    config["loss"] = "flow_endpoint_xyz"
    endpoint_settings = build_settings(config)
    assert endpoint_settings["pt_run_name"].endswith(
        "_flow_endpoint_xyz_pred_lora_all_dz08"
    )

    config["architecture"] = {"conditioning_route": "state_skill_cond"}
    routed_settings = build_settings(config)
    assert routed_settings["conditioning_route"] == "state_skill_cond"
    assert routed_settings["pt_run_name"].endswith(
        "_dino_frozen_state_skill_cond_flow_endpoint_xyz_pred_lora_all_dz08"
    )


def test_stage1_output_name_keeps_full_dataset_identity(tmp_path: Path) -> None:
    project = tmp_path / "project"
    run = "FSQ333_125_std_pt_episodemean_80p_snap10_pmax15_halfnormal"
    dataset = project / "dataset/skillvla_dataset/source" / run / "skillvla"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "skill_fsq_levels": [3, 3, 3],
                "skill_pmax": 15,
                "skill_jitter_distribution": "half_normal",
                "features": {
                    "observation.state": {"shape": [8]},
                    "action": {"shape": [7]},
                },
            }
        )
    )
    pi_base = project / "models/pi05_base"
    dino = project / "models/dino"
    pi_base.mkdir(parents=True)
    dino.mkdir(parents=True)
    (pi_base / "model.safetensors").touch()

    settings = build_settings(
        {
            "project_root": str(project),
            "dataset_root": "dataset",
            "outputs_root": "outputs",
            "dataset": {
                "skillvla_root": "skillvla_dataset",
                "source": "source",
                "run": run,
            },
            "warm_start": {"pi_base": "models/pi05_base"},
            "vision": {"dino_model": "models/dino", "freeze": True},
            "skill_predictor": {"train": False},
            "terminator": {"train": False},
        }
    )

    assert settings["pt_run_name"] == f"source_{run}_dino_frozen_state_cond_flow"
