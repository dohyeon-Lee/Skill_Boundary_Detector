import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/PT/eval/src"
sys.path.insert(0, str(SRC))
import bspline_visualization_config as diagnostic


def test_visualization_overrides_episode_count_before_validation_without_editing_shared_config(monkeypatch):
    shared = {"n_episodes": 10, "eval_batch_size": 4, "max_parallel_tasks": 3,
              "eval_num_gpus": 10, "eval_max_workers_per_gpu": 2,
              "output_name": "example", "models": [{"model_dir": "bspline", "checkpoint": "050000"}],
              "oracle": {"repeat_episodes": False, "broadcast_single_episode": True},
              "terminator": {"end_threshold": .5}, "motion_plan": {"samples_per_skill": 6},
              "video": {"max_per_task": 3}, "logging": {"wandb": {"enable": True}}}
    original = deepcopy(shared)

    def base(config):
        assert config["n_episodes"] == config["eval_batch_size"] == config["max_parallel_tasks"] == 1
        assert config["models"] == original["models"] and config["oracle"] == original["oracle"]
        assert config["terminator"] == original["terminator"]
        assert config["eval_num_gpus"] == original["eval_num_gpus"]
        assert config["eval_max_workers_per_gpu"] == original["eval_max_workers_per_gpu"]
        assert config["output_name"] == "example_bspline_visualization"
        assert config["motion_plan"]["enabled"] and config["motion_plan"]["samples_per_skill"] == 6
        assert config["video"]["max_per_task"] == 1 and not config["logging"]["wandb"]["enable"]
        return {"models_json": json.dumps([{"label": "B2", "architecture_label": "fullskill_bspline_2_firstlayer"}])}

    monkeypatch.setattr(diagnostic.base, "build_settings", base)
    diagnostic.build_settings(shared)
    assert shared == original


@pytest.mark.parametrize("gpus, workers", [(10, 2), (2, 2), (1, 1), (3, 4), (2, 5)])
def test_visualization_preserves_shared_gpu_packing(gpus, workers):
    shared = {"eval_num_gpus": gpus, "eval_max_workers_per_gpu": workers}
    config = diagnostic.visualization_config(shared)
    assert config["eval_num_gpus"] == gpus
    assert config["eval_max_workers_per_gpu"] == workers
    assert config["n_episodes"] == 1
    assert shared == {"eval_num_gpus": gpus, "eval_max_workers_per_gpu": workers}


def test_visualization_keeps_gpu_packing_defaults_in_base_resolver():
    config = diagnostic.visualization_config({})
    assert "eval_num_gpus" not in config
    assert "eval_max_workers_per_gpu" not in config


def test_visualization_rejects_non_bspline_models(monkeypatch):
    monkeypatch.setattr(diagnostic.base, "build_settings", lambda cfg: {
        "models_json": json.dumps([{"label": "both3", "architecture_label": "both_3"}])})
    with pytest.raises(ValueError, match="Bspline checkpoints"):
        diagnostic.build_settings({})
