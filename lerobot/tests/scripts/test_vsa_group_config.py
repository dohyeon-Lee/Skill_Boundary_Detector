"""Concurrent YAML selection keeps per-model contracts and one GPU allocation."""
import copy
import json
import sys

import pytest

from test_stage1_train_config_names import _config
import stage1_train_config as bootstrap


MODELS = [
    "UV_Bspline_skill_denoising_none_wovis_last_pi05vlm",
    "UV_Bspline_skill_denoising_none_wovis_nouv_last_pi05vlm",
]


def _group_config(tmp_path):
    config = _config(tmp_path, "arch8_1")
    config["architecture"]["name"] = MODELS[0]
    config["architecture"]["visual_bottleneck_tokens"] = 100
    return config


def test_concurrent_members_match_scalar_configs_without_mutating_yaml(tmp_path):
    config = _group_config(tmp_path)
    config["architecture"]["name"] = MODELS.copy()
    config["architecture"]["visual_bridge_last_n_layers"] = 1
    config["training"]["dataloader"] = {"batch_size": 16, "gpus": 1}
    config["slurm"] = {"gres": "gpu:B200:1"}
    original = copy.deepcopy(config)
    path = tmp_path / "vsa.yaml"
    settings = bootstrap.build_launch_settings(config, path)
    runs = json.loads(settings["stage1_group_runs_json"])
    assert len(runs) == 2 and config == original
    assert len({run["output_dir"] for run in runs}) == 2
    for name, run in zip(MODELS, runs, strict=True):
        scalar = copy.deepcopy(config)
        scalar["architecture"]["name"] = name
        expected = bootstrap.build_settings(scalar)
        assert run["architecture"] == expected["architecture_label"]
        assert run["output_dir"] == str(expected["pt_output_dir"])
        assert run["config"] == str(path.resolve())
        assert expected["batch_size"] == 16 and expected["num_gpus"] == 1
        assert expected["visual_bridge_last_n_layers"] == 1


@pytest.mark.parametrize("value", [[], [MODELS[0], MODELS[0].lower()], [MODELS[0], None]])
def test_invalid_selection_rejected_before_submission(tmp_path, value):
    config = _group_config(tmp_path)
    config["architecture"]["name"] = value
    with pytest.raises(ValueError, match="architecture.name"):
        bootstrap.build_launch_settings(config, tmp_path / "vsa.yaml")


def test_unknown_second_member_is_preflighted(tmp_path):
    config = _group_config(tmp_path)
    config["architecture"]["name"] = [MODELS[0], "unknown_model"]
    with pytest.raises(ValueError, match="architecture.name"):
        bootstrap.build_launch_settings(config, tmp_path / "vsa.yaml")


@pytest.mark.parametrize("gpus,gres", [(2, "gpu:B200:1"), (1, "gpu:B200:2")])
def test_concurrent_models_require_one_shared_gpu(tmp_path, gpus, gres):
    config = _group_config(tmp_path)
    config["architecture"]["name"] = MODELS.copy()
    config["training"]["dataloader"] = {"gpus": gpus}
    config["slurm"] = {"gres": gres}
    with pytest.raises(ValueError, match="one shared GPU"):
        bootstrap.build_launch_settings(config, tmp_path / "vsa.yaml")


def test_cli_override_selects_only_one_model_from_list(tmp_path, monkeypatch, capsys):
    config = _group_config(tmp_path)
    config["architecture"]["name"] = MODELS.copy()
    monkeypatch.setattr(bootstrap, "load_stage1_component_config", lambda _: config)
    monkeypatch.setattr(sys, "argv", ["stage1_train_config.py", "--shell", "--architecture", MODELS[1]])
    bootstrap.main()
    output = capsys.readouterr().out
    assert "STAGE1_GROUP_RUNS_JSON='[]'" in output
    assert "wovis_nouv_last_pi05vlm" in output
