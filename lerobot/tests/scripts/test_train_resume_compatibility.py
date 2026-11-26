"""Load saved training state after retiring experimental sampler settings."""

import json
import sys

import pytest

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401


@pytest.mark.parametrize("enabled", [None, False, True])
def test_saved_training_config_loads_without_retired_sampling(tmp_path, monkeypatch, enabled):
    saved = {
        "dataset": {"repo_id": "test/dataset", "root": "/dataset"},
        "policy": {"type": "pi05", "push_to_hub": False},
        "steps": 100000,
        "batch_size": 16,
        "seed": 42,
    }
    if enabled is not None:
        saved["transition_oversampling"] = {
            "enabled": enabled, "percent": 25, "distribution": "uniform", "weight": 2,
        }
    path = tmp_path / "checkpoints/060000/pretrained_model/train_config.json"
    path.parent.mkdir(parents=True)
    original = json.dumps(saved)
    path.write_text(original)

    config = TrainPipelineConfig.from_pretrained(path, cli_args=["--resume=true", "--batch_size=8"])
    monkeypatch.setattr(sys, "argv", ["trainer", f"--config_path={path}"])
    config.validate()

    assert config.batch_size == 8
    assert config.steps == 100000 and config.seed == 42
    assert str(config.dataset.root) == "/dataset"
    assert config.policy.type == "pi05"
    assert config.checkpoint_path == path.parent.parent
    assert config.policy.pretrained_path == path.parent
    assert not hasattr(config, "transition_oversampling")
    assert path.read_text() == original
