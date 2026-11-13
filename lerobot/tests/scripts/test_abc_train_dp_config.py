from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lerobot.datasets.abc_native_dataset import ABCNativeDataset
from lerobot.scripts.lerobot_train import _make_policy_tail_sampler

ROOT = Path(__file__).resolve().parents[3]
ABC_DP_DIR = ROOT / "lerobot/examples/abc/configs/train_skills/DP"


def _load_resolver():
    path = ABC_DP_DIR / "src/train_dp_config.py"
    spec = importlib.util.spec_from_file_location("abc_train_dp_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RESOLVER = _load_resolver()


def _make_native_dataset_metadata(project_root: Path) -> None:
    dataset = project_root / "dataset_ABC/abc_sim_all"
    (dataset / "meta").mkdir(parents=True)
    names = [
        *(f"left_joint_{index}" for index in range(6)),
        "left_gripper",
        *(f"right_joint_{index}" for index in range(6)),
        "right_gripper",
    ]
    (dataset / "native_dataset.json").write_text(
        json.dumps(
            {
                "format": "sbd_abc_sim_native/v1",
                "task_sha256": {"grab_clutter": "a", "put_relative": "b"},
            }
        ),
        encoding="utf-8",
    )
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "total_tasks": 2,
                "features": {
                    "observation.state": {"shape": [14], "names": names},
                    "action": {"shape": [14], "names": names},
                },
            }
        ),
        encoding="utf-8",
    )


def _config(project_root: Path, **overrides):
    config = {
        "project_root": str(project_root),
        "dataset_root": "dataset_ABC",
        "outputs_root": "outputs_ABC",
        "dp_dataset_root": "dataset_ABC",
        "dp_dataset": "abc_sim_all",
        "dp_tasks": "grab_clutter",
        "dp_base_config": "base.yaml",
        "dp_n_obs_steps": 30,
        "dp_future_action_horizon": 36,
        "dp_history_encoder": "transformer",
        "dp_history_encoder_dim": 128,
        "dp_unet_size": "small",
        "dp_lr_scheduler": "constant",
        "dp_lr": 1e-4,
        "dp_amp": False,
        "dp_steps": 100000,
        "dp_save_freq": 10000,
    }
    config.update(overrides)
    return config


def test_abc_dp_resolves_grab_clutter_absolute_30hz_contract(tmp_path: Path) -> None:
    _make_native_dataset_metadata(tmp_path)

    settings = RESOLVER.build_settings(_config(tmp_path))

    assert settings["dp_tasks"] == ["grab_clutter"]
    assert json.loads(settings["dp_resolved_tasks"]) == ["grab_clutter"]
    assert settings["dp_task_filter"] is True
    assert settings["dp_n_obs_steps"] == 30
    assert settings["dp_future_action_horizon"] == 36
    assert settings["dp_n_action_steps"] == 36
    assert settings["dp_drop_n_last_frames"] == 35
    assert settings["dp_down_dims_arg"] == "[128, 256, 512]"
    assert settings["dp_lr_scheduler"] == "constant"
    assert settings["dp_lr"] == pytest.approx(1e-4)
    assert settings["dp_amp"] is False
    assert settings["dp_steps"] == 100000
    assert settings["dp_save_freq"] == 10000
    assert "task-grab_clutter_absolute_future_only_obs30_future36" in settings["dp_run_name"]
    assert settings["dp_final_checkpoint"].name == "pretrained_model"


def test_abc_dp_expands_global_task_groups(tmp_path: Path) -> None:
    _make_native_dataset_metadata(tmp_path)
    settings = RESOLVER.build_settings(
        _config(
            tmp_path,
            dp_tasks="group_multitask",
            abc_sim_task_groups={"group_multitask": ["put_relative", "grab_clutter"]},
        )
    )

    assert settings["dp_tasks"] == ["put_relative", "grab_clutter"]
    assert settings["dp_used_task_groups"] == ["group_multitask"]
    assert "group_multitask_absolute" in settings["dp_run_name"]


@pytest.mark.parametrize("retired", ["dp_proprio_grounding", "dp_relative"])
def test_abc_dp_rejects_grounding_and_relative_action_options(tmp_path: Path, retired: str) -> None:
    _make_native_dataset_metadata(tmp_path)
    with pytest.raises(ValueError, match="absolute state/action"):
        RESOLVER.build_settings(_config(tmp_path, **{retired: False}))


def test_abc_native_reader_owns_tail_filtering() -> None:
    # __new__ is sufficient because the ABC branch must return before reading
    # any instance fields. It protects compact task-subset indices from the
    # generic LeRobot episode sampler.
    dataset = object.__new__(ABCNativeDataset)
    assert _make_policy_tail_sampler(dataset, 35) is None


def test_generic_lerobot_dataset_keeps_episode_tail_sampler() -> None:
    dataset = SimpleNamespace(
        meta=SimpleNamespace(episodes={"dataset_from_index": [0, 10], "dataset_to_index": [10, 20]}),
        episodes=None,
    )
    sampler = _make_policy_tail_sampler(dataset, 3)
    assert sampler is not None
    assert len(sampler) == 14
    assert set(sampler.indices) == {*range(0, 7), *range(10, 17)}


def test_shipped_abc_dp_config_targets_grab_clutter() -> None:
    config = yaml.safe_load((ABC_DP_DIR / "dp_config.yaml").read_text())
    assert config["dp_dataset"] == "abc_sim_all"
    assert config["dp_tasks"] == "grab_clutter"
    assert config["dp_n_obs_steps"] == 30
    assert config["dp_future_action_horizon"] == 36
    assert config["dp_steps"] == 100000
    assert config["dp_save_freq"] == 10000
    assert "dp_relative" not in config
    assert "dp_proprio_grounding" not in config
