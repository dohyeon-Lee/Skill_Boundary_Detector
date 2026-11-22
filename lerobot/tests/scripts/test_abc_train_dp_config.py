from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from lerobot.datasets.abc_native_dataset import ABCNativeDataset
from lerobot.processor.yam_kinematics import to_yam_delta_pose_actions_numpy
from lerobot.scripts.lerobot_train import _make_policy_tail_sampler

ROOT = Path(__file__).resolve().parents[3]
ABC_DP_DIR = ROOT / "lerobot/skillVLA/abc/configs/train_skills/DP"


def _load_resolver():
    path = ABC_DP_DIR / "src/train_dp_config.py"
    spec = importlib.util.spec_from_file_location("abc_train_dp_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_stats_module():
    path = ABC_DP_DIR / "src/prepare_native_action_stats.py"
    spec = importlib.util.spec_from_file_location("abc_prepare_native_action_stats", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RESOLVER = _load_resolver()
STATS = _load_stats_module()


def _make_native_dataset_metadata(project_root: Path) -> None:
    dataset = project_root / "dataset_ABC/abc_sim_all"
    (dataset / "meta/episodes/chunk-000").mkdir(parents=True)
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
                "request_sha256": "synthetic-request",
                "task_sha256": {"grab_clutter": "a", "put_relative": "b"},
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for episode_index in range(8):
        task = "grab_clutter" if episode_index < 4 else "put_relative"
        rows.append(
            {
                "episode_index": episode_index,
                "length": 100 + episode_index,
                "source_task": task,
                "states_actions_path": f"{task}/episode_{episode_index}/states_actions.bin",
            }
        )
    pd.DataFrame(rows).to_parquet(
        dataset / "meta/episodes/chunk-000/file-000.parquet", index=False
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

    settings = RESOLVER.build_settings(_config(tmp_path, dp_action_mode="dataset"))

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
    assert settings["dp_action_mode"] == "dataset"
    assert settings["dp_use_relative_actions"] is False
    assert "task-grab_clutter_absolute_future_only_obs30_future36" in settings["dp_run_name"]
    assert settings["dp_final_checkpoint"].name == "pretrained_model"


def test_abc_dp_resolves_anchor_relative_arm_joints_with_absolute_grippers(
    tmp_path: Path,
) -> None:
    _make_native_dataset_metadata(tmp_path)

    settings = RESOLVER.build_settings(
        _config(
            tmp_path,
            dp_action_mode="anchor_relative",
            dp_relative_exclude_joints=["gripper"],
        )
    )

    assert settings["dp_action_mode"] == "anchor_relative"
    assert settings["dp_use_relative_actions"] is True
    assert settings["dp_use_delta_joint_actions"] is False
    assert settings["dp_relative_exclude_joints"] == ["gripper"]
    assert settings["dp_relative_exclude_joints_csv"] == "gripper"
    assert "task-grab_clutter_anchor-relative-joint_future_only" in settings["dp_run_name"]


def test_abc_dp_resolves_libero_analogous_delta_joint_mode(tmp_path: Path) -> None:
    _make_native_dataset_metadata(tmp_path)

    settings = RESOLVER.build_settings(
        _config(
            tmp_path,
            dp_action_mode="delta_joint",
            dp_relative_exclude_joints=["gripper"],
        )
    )

    assert settings["dp_use_relative_actions"] is False
    assert settings["dp_use_delta_joint_actions"] is True
    assert "task-grab_clutter_delta-joint_future_only" in settings["dp_run_name"]


def test_abc_dp_resolves_fk_delta_pose_mode(tmp_path: Path) -> None:
    _make_native_dataset_metadata(tmp_path)

    settings = RESOLVER.build_settings(
        _config(
            tmp_path,
            dp_action_mode="delta_pose",
            dp_relative_exclude_joints=["gripper"],
        )
    )

    assert settings["dp_use_relative_actions"] is False
    assert settings["dp_use_delta_joint_actions"] is False
    assert settings["dp_use_delta_pose_actions"] is True
    assert "task-grab_clutter_delta-pose-eef_future_only" in settings["dp_run_name"]


@pytest.mark.parametrize("action_mode", ["anchor_relative", "delta_joint", "delta_pose"])
def test_abc_dp_optimized_stats_chunks_match_original_offset_calculation(
    action_mode: str,
) -> None:
    rng = np.random.default_rng(42)
    states = rng.normal(size=(13, 14))
    actions = rng.normal(size=(13, 14))
    mask = np.asarray([index not in {6, 13} for index in range(14)])
    chunk_size = 7

    expected = []
    for offset in range(chunk_size):
        targets = actions[offset:].copy()
        if action_mode == "delta_pose":
            reference = states if offset == 0 else actions[offset - 1 : -1]
            relative = to_yam_delta_pose_actions_numpy(targets, reference)
        else:
            relative = targets
            if action_mode == "anchor_relative" or offset == 0:
                reference = states[: len(actions) - offset, mask]
            else:
                reference = actions[offset - 1 : -1, mask]
            relative[:, mask] -= reference
        expected.append(relative)

    actual = list(
        STATS._iter_episode_relative_chunks(
            states, actions, chunk_size, mask, action_mode
        )
    )

    assert len(actual) == len(expected)
    for optimized, original in zip(actual, expected, strict=True):
        np.testing.assert_allclose(optimized, original, rtol=0.0, atol=0.0)


def test_abc_dp_delta_pose_stats_run_fk_only_twice_per_episode(monkeypatch) -> None:
    rng = np.random.default_rng(7)
    states = rng.normal(size=(20, 14))
    actions = rng.normal(size=(20, 14))
    mask = np.asarray([index not in {6, 13} for index in range(14)])
    calls = 0
    original = STATS.to_yam_delta_pose_actions_numpy

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(STATS, "to_yam_delta_pose_actions_numpy", counted)
    chunks = list(STATS._iter_episode_relative_chunks(states, actions, 12, mask, "delta_pose"))

    assert len(chunks) == 12
    assert calls == 2


def test_abc_dp_anchor_relative_rejects_relative_grippers(tmp_path: Path) -> None:
    _make_native_dataset_metadata(tmp_path)
    with pytest.raises(ValueError, match="keep both gripper"):
        RESOLVER.build_settings(
            _config(
                tmp_path,
                dp_action_mode="anchor_relative",
                dp_relative_exclude_joints=[],
            )
        )


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


def test_abc_dp_materializes_reproducible_training_set_manifest(tmp_path: Path) -> None:
    _make_native_dataset_metadata(tmp_path)
    config = _config(
        tmp_path,
        dp_tasks="group_pair_2_eps",
        abc_sim_task_groups={"group_pair": ["grab_clutter", "put_relative"]},
        abc_sim_training_sets={
            "group_pair_2_eps": {
                "tasks": "group_pair",
                "episodes_per_task": 2,
                "sampling": "random",
                "seed": 42,
            }
        },
    )

    first = RESOLVER.build_settings(config)
    second = RESOLVER.build_settings(config)

    assert first["dp_tasks"] == ["grab_clutter", "put_relative"]
    assert first["dp_training_set"] == "group_pair_2_eps"
    assert first["dp_episode_filter"] is True
    assert first["dp_episode_count"] == 4
    assert first["dp_episode_selection_manifest"] == second["dp_episode_selection_manifest"]
    manifest = json.loads(Path(first["dp_episode_selection_manifest"]).read_text())
    assert manifest["seed"] == 42
    assert manifest["episodes_per_task"] == {"grab_clutter": 2, "put_relative": 2}
    assert {task: len(ids) for task, ids in manifest["episodes_by_task"].items()} == {
        "grab_clutter": 2,
        "put_relative": 2,
    }
    assert "group_pair_2_eps-sel" in first["dp_run_name"]


@pytest.mark.parametrize("retired", ["dp_proprio_grounding", "dp_relative"])
def test_abc_dp_rejects_grounding_and_relative_action_options(tmp_path: Path, retired: str) -> None:
    _make_native_dataset_metadata(tmp_path)
    with pytest.raises(ValueError, match="legacy grounding/relative"):
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


def test_shipped_abc_dp_config_targets_reproducible_singletask_group() -> None:
    config = yaml.safe_load((ABC_DP_DIR / "dp_config.yaml").read_text())
    assert config["dp_dataset"] == "abc_sim_all"
    assert config["dp_tasks"] == "group_singletask_1k_eps"
    assert config["dp_n_obs_steps"] == 30
    assert config["dp_future_action_horizon"] == 36
    assert config["dp_action_mode"] == "delta_pose"
    assert config["dp_relative_exclude_joints"] == ["gripper"]
    assert config["dp_steps"] == 100000
    assert config["dp_save_freq"] == 10000
    assert "dp_relative" not in config
    assert "dp_proprio_grounding" not in config
