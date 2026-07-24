import importlib.util
import json
import re
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
        "stage0_vlm_residual": True,
        "vlm_cond": False,
        "vlm_expert": False,
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
        "modes": "conditional,unconditional",
        "skill_source": "gt",
        "target_task": "libero_90",
    }


def _write_stage3_eval(tmp_path: Path, models: list[dict]) -> tuple[Path, list[Path]]:
    paths = []
    for model in models:
        path = (
            tmp_path / "outputs/skillVLA_stage3" / model["model_dir"]
            / "checkpoints" / model.get("checkpoint", "last") / "pretrained_model"
        )
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps({"pt_stage": "skill"}))
        paths.append(path)
    config_path = tmp_path / "stage3_eval_config.yaml"
    config_path.write_text(
        "project_root: " + str(tmp_path) + "\n"
        "outputs_root: outputs\n"
        "models:\n"
        + "".join(
            f"  - {{model_dir: {model['model_dir']}, "
            f"checkpoint: \"{model.get('checkpoint', 'last')}\"}}\n"
            for model in models
        )
    )
    return config_path, paths


def test_stage0_eval_expands_conditional_and_unconditional(tmp_path: Path) -> None:
    settings = stage0_eval_config.build_settings(_config(tmp_path))
    panels = json.loads(settings["models_json"])

    assert [panel["label"] for panel in panels] == [
        "S0 [Conditional]", "S0 [Unconditional]",
    ]
    assert (panels[0]["drop_vlm"], panels[0]["keep_adapters"]) == (False, False)
    assert (panels[1]["drop_vlm"], panels[1]["keep_adapters"]) == (True, False)
    assert settings["use_gt_skill"] is True
    assert settings["models_per_row"] == 2


def test_stage0_eval_keeps_each_model_pair_on_one_row() -> None:
    panels = stage0_eval_config._expand_stage0_panels([
        {"label": "Exp2", "modes": "conditional,unconditional"},
        {"label": "Exp3", "modes": "conditional,unconditional"},
    ])

    assert [panel["label"] for panel in panels] == [
        "Exp2 [Conditional]", "Exp2 [Unconditional]",
        "Exp3 [Conditional]", "Exp3 [Unconditional]",
    ]


def test_stage0_eval_uses_checkpoint_predictor(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg["skill_source"] = "predicted"

    settings = stage0_eval_config.build_settings(cfg)

    assert settings["use_gt_skill"] is False
    assert settings["eval_skill_predictor_path"] == ""


def test_stage0_eval_uses_stage3_vlm_and_predictor(tmp_path: Path) -> None:
    stage3_config, paths = _write_stage3_eval(
        tmp_path, [{"model_dir": "stage3_run", "checkpoint": "030000"}]
    )
    cfg = _config(tmp_path)
    cfg["skill_source"] = "stage3"
    cfg["stage3_eval_config"] = str(stage3_config)

    settings = stage0_eval_config.build_settings(cfg)

    assert settings["use_gt_skill"] is False
    assert Path(settings["eval_skill_predictor_path"]) == paths[0]


def test_stage0_eval_rejects_multiple_stage3_predictors(tmp_path: Path) -> None:
    stage3_config, _ = _write_stage3_eval(
        tmp_path,
        [
            {"model_dir": "stage3_a", "checkpoint": "last"},
            {"model_dir": "stage3_b", "checkpoint": "last"},
        ],
    )
    cfg = _config(tmp_path)
    cfg["skill_source"] = "stage3"
    cfg["stage3_eval_config"] = str(stage3_config)

    with pytest.raises(ValueError, match="exactly one distinct Stage-3 checkpoint"):
        stage0_eval_config.build_settings(cfg)


def test_stage0_eval_rejects_gt_advance_with_predicted_skill(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg["skill_source"] = "predicted"
    cfg["models"][0]["advance_mode"] = "gt"

    with pytest.raises(ValueError, match="predicted skills must use terminator"):
        stage0_eval_config.build_settings(cfg)


def test_stage0_eval_keeps_single_unconditional_on_the_base_route(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg["modes"] = "unconditional"

    settings = stage0_eval_config.build_settings(cfg)
    panel = json.loads(settings["models_json"])[0]

    assert panel["drop_vlm"] is True
    assert panel["keep_adapters"] is False


def test_stage0_eval_uses_explicit_output_name_verbatim(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg["output_name"] = "my_stage0_eval"

    settings = stage0_eval_config.build_settings(cfg)

    assert settings["output_name"] == "my_stage0_eval"
    assert Path(settings["eval_out_dir"]).name == "my_stage0_eval"
    assert settings["wandb_run_name"] == "my_stage0_eval"


def test_stage0_eval_uses_resolution_time_when_output_name_is_blank(tmp_path: Path) -> None:
    settings = stage0_eval_config.build_settings(_config(tmp_path))

    assert re.fullmatch(r"\d{8}_\d{6}_\d{6}", settings["output_name"])
    assert Path(settings["eval_out_dir"]).name == settings["output_name"]


def test_stage0_eval_rejects_output_paths(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg["output_name"] = "nested/eval"

    with pytest.raises(ValueError, match="single folder name"):
        stage0_eval_config.build_settings(cfg)
