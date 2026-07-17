import importlib.util
import json
from pathlib import Path

import pytest


_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples/libero/configs/train_skillVLA/stage0_eval/src/stage0_eval_config.py"
)
_SPEC = importlib.util.spec_from_file_location("stage0_eval_config", _CONFIG_PATH)
stage0_eval_config = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(stage0_eval_config)


def _config(tmp_path: Path) -> dict:
    run = "stage0_run"
    policy_path = tmp_path / "outputs/skillVLA_stage0" / run / "checkpoints/last/pretrained_model"
    policy_path.mkdir(parents=True)
    fsq_path = tmp_path / "dataset/skillvla_dataset/libero_90_full_full/fsq_run/FSQ.pt"
    fsq_path.parent.mkdir(parents=True)
    fsq_path.touch()
    (policy_path / "config.json").write_text(json.dumps({
        "fsq_path": str(fsq_path),
        "train_terminator": False,
        "terminator_dino_model_path": str(tmp_path / "models/dino"),
    }))
    return {
        "project_root": str(tmp_path),
        "outputs_root": "outputs",
        "models": [{
            "model_dir": run,
            "checkpoint": "last",
            "advance_mode": "terminator",
            "terminator_source": "auto",
            "label": "S0",
        }],
        "modes": "a,b,fsq",
        "skill_source": "gt",
        "target_task": "libero_90",
    }


def test_stage0_eval_expands_a_b_and_frozen_fsq(tmp_path: Path) -> None:
    settings = stage0_eval_config.build_settings(_config(tmp_path))
    panels = json.loads(settings["models_json"])

    assert [panel["label"] for panel in panels] == ["S0 [A]", "S0 [B]", "S0 [Frozen-FSQ]"]
    assert (panels[0]["drop_vlm"], panels[0]["keep_adapters"], panels[0]["runner"]) == (
        False, False, "stage0")
    assert (panels[1]["drop_vlm"], panels[1]["keep_adapters"], panels[1]["runner"]) == (
        True, True, "stage0")
    assert panels[2]["runner"] == "fsq"
    assert settings["use_gt_skill"] is True
    assert settings["models_per_row"] == 3


def test_stage0_eval_rejects_predicted_skill(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg["skill_source"] = "pred"

    with pytest.raises(ValueError, match="requires skill_source=gt"):
        stage0_eval_config.build_settings(cfg)


def test_stage0_eval_keeps_single_b_on_the_severed_route(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg["modes"] = "b"

    settings = stage0_eval_config.build_settings(cfg)
    panel = json.loads(settings["models_json"])[0]

    assert panel["drop_vlm"] is True
    assert panel["keep_adapters"] is True
