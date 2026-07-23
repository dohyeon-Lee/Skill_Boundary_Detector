import importlib.util
import json
from pathlib import Path

import pytest


_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples/libero/configs/train_skillVLA/stage0_pretrain_eval/src/stage0_pretrain_eval_config.py"
)
_SPEC = importlib.util.spec_from_file_location("stage0_pretrain_eval_config", _CONFIG_PATH)
stage0_pretrain_eval_config = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(stage0_pretrain_eval_config)


def _policy_path(root: Path, models_root: str, run: str, checkpoint: str = "last") -> Path:
    return root / "outputs" / models_root / run / "checkpoints" / checkpoint / "pretrained_model"


def _config(tmp_path: Path, *, skill_source: str) -> dict:
    run = "stage0_pretrain_run"
    policy_path = _policy_path(tmp_path, "skillVLA_stage0_pretrain", run)
    policy_path.mkdir(parents=True)
    fsq_path = tmp_path / "dataset/skillvla_dataset/source/fsq_run/FSQ.pt"
    fsq_path.parent.mkdir(parents=True)
    fsq_path.touch()
    (policy_path / "config.json").write_text(json.dumps({
        "fsq_path": str(fsq_path),
        "train_terminator": False,
    }))
    return {
        "project_root": str(tmp_path),
        "outputs_root": "outputs",
        "models": [{
            "model_dir": run,
            "checkpoint": "last",
            "advance_mode": "terminator",
            "terminator_source": "auto",
            "label": "S0P",
        }],
        "modes": "a",
        "skill_source": skill_source,
        "target_task": "libero_90",
    }


def _write_stage3_eval(tmp_path: Path, models: list[dict]) -> tuple[Path, list[Path]]:
    paths = []
    for model in models:
        path = _policy_path(
            tmp_path, "skillVLA_stage3", model["model_dir"], model.get("checkpoint", "last")
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


def test_predicted_skill_keeps_the_local_autoregressive_predictor(tmp_path: Path) -> None:
    settings = stage0_pretrain_eval_config.build_settings(
        _config(tmp_path, skill_source="predicted")
    )

    assert settings["eval_skill_predictor_path"] == ""


def test_stage3_skill_source_resolves_the_stage3_eval_checkpoint(tmp_path: Path) -> None:
    stage3_config, paths = _write_stage3_eval(
        tmp_path, [{"model_dir": "stage3_run", "checkpoint": "030000"}]
    )
    cfg = _config(tmp_path, skill_source="stage3")
    cfg["stage3_eval_config"] = str(stage3_config)

    settings = stage0_pretrain_eval_config.build_settings(cfg)

    assert Path(settings["eval_skill_predictor_path"]) == paths[0]
    assert settings["eval_terminator"] == "base"


def test_stage3_skill_source_rejects_distinct_stage3_models(tmp_path: Path) -> None:
    stage3_config, _ = _write_stage3_eval(
        tmp_path,
        [
            {"model_dir": "stage3_a", "checkpoint": "last"},
            {"model_dir": "stage3_b", "checkpoint": "last"},
        ],
    )
    cfg = _config(tmp_path, skill_source="stage3")
    cfg["stage3_eval_config"] = str(stage3_config)

    with pytest.raises(ValueError, match="exactly one distinct Stage-3 checkpoint"):
        stage0_pretrain_eval_config.build_settings(cfg)
