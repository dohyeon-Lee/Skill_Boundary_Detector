from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_ORACLE = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/build_data/oracle_matching"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG = _load("restored_oracle_config", _ORACLE / "src/oracle_matching_config.py")
BUILDER = _load("restored_oracle_builder", _ORACLE / "src/build_init_states.py")


def _config(tmp_path, source="libero_10_full_full"):
    (tmp_path / "dataset_filtered" / source).mkdir(parents=True)
    (tmp_path / "libero_original_dataset/libero_10").mkdir(parents=True)
    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset_filtered",
        "source_dataset": source,
        "train_partition": ["suma_a6000"],
        "train_qos": "base_qos",
    }


def test_restored_config_resolves_portable_full_dataset_paths(tmp_path):
    settings = CONFIG.build_settings(_config(tmp_path), config_path=tmp_path / "oracle.yaml")
    assert settings["oracle_mode"] == "libero"
    assert settings["oracle_lerobot_dataset"] == tmp_path / "dataset_filtered/libero_10_full_full"
    assert settings["oracle_original_dataset"] == str(tmp_path / "libero_original_dataset/libero_10")
    assert settings["oracle_output_path"] == tmp_path / "dataset_filtered/skillvla_dataset/libero_10_full_full/eval_init_states.npz"
    assert settings["oracle_overwrite"] is False
    assert settings["oracle_partition"] == "suma_a6000"
    assert settings["oracle_qos"] == "base_qos"
    assert CONFIG.CONFIGS_ROOT == _ORACLE.parents[2]


def test_config_allows_auto_root_and_preserves_langgap_mode(tmp_path):
    cfg = _config(tmp_path, source="langgap_ext_full_full")
    cfg["dataset_root"] = "auto"
    cfg["task_ids"] = [2, 0, 2]
    settings = CONFIG.build_settings(cfg, config_path=tmp_path / "oracle.yaml")
    assert settings["oracle_mode"] == "langgap"
    assert settings["oracle_task_ids"] == "0,2"
    assert settings["oracle_original_dataset"] == ""


def test_config_rejects_ambiguous_auto_root(tmp_path):
    cfg = _config(tmp_path)
    (tmp_path / "dataset/libero_10_full_full").mkdir(parents=True)
    cfg["dataset_root"] = "auto"
    with pytest.raises(ValueError, match="exactly one root"):
        CONFIG.build_settings(cfg, config_path=tmp_path / "oracle.yaml")


@pytest.mark.parametrize("indices,method", [([1, 2, 3], "exact"), ([0, 2, 4], "subseq")])
def test_content_matching_preserves_original_exact_and_trimmed_paths(indices, method):
    actions = np.arange(35, dtype=np.float32).reshape(5, 7) / 100
    init = np.arange(9, dtype=np.float32)
    result = BUILDER.match_episode(actions[indices], [("scene.hdf5", "demo_2", actions, init)])
    assert result[4] == method
    assert result[0] == 0
    np.testing.assert_array_equal(result[3], init)


def _stub_source(monkeypatch, tmp_path, *, matches=True, overwrite=False):
    output = tmp_path / "mapping/eval_init_states.npz"
    args = ["build_init_states.py", "--lerobot_dataset", str(tmp_path / "data"),
            "--orig_dataset", str(tmp_path / "original"), "--out", str(output)]
    if overwrite:
        args.append("--overwrite")
    monkeypatch.setattr(sys, "argv", args)
    metadata = pd.DataFrame({"episode_index": [7], "tasks": [["pick object"]],
                             "data/chunk_index": [0], "data/file_index": [0]})
    actions = np.arange(35, dtype=np.float32).reshape(5, 7) / 100
    episode_actions = actions.copy() if matches else actions + 100
    monkeypatch.setattr(BUILDER, "load_episode_meta", lambda root: metadata)
    monkeypatch.setattr(BUILDER, "preload_demos", lambda root: {
        "pick object": [("KITCHEN_SCENE1_pick_object_demo.hdf5", "demo_2", actions, np.zeros(9))]
    })
    shard = pd.DataFrame({"episode_index": [7] * len(actions), "action": list(episode_actions)})
    monkeypatch.setattr(BUILDER.pd, "read_parquet", lambda path: shard)
    return output


def test_libero_builder_writes_complete_compatible_map(monkeypatch, tmp_path):
    output = _stub_source(monkeypatch, tmp_path)
    BUILDER.main()
    with np.load(output, allow_pickle=True) as result:
        np.testing.assert_array_equal(result["episode_index"], [7])
        assert result["demo"].tolist() == ["demo_2"]
        assert result["init_states"].shape == (1,)
        assert result["init_states"][0].shape == (9,)
        assert result["match_method"].tolist() == ["exact"]


def test_libero_builder_refuses_partial_map(monkeypatch, tmp_path):
    output = _stub_source(monkeypatch, tmp_path, matches=False)
    with pytest.raises(RuntimeError, match="incomplete episode-exact map"):
        BUILDER.main()
    assert not output.exists()


def test_libero_builder_preserves_existing_map_without_overwrite(monkeypatch, tmp_path):
    output = _stub_source(monkeypatch, tmp_path)
    output.parent.mkdir()
    output.write_bytes(b"existing map")
    with pytest.raises(FileExistsError, match="Output already exists"):
        BUILDER.main()
    assert output.read_bytes() == b"existing map"


def test_failed_overwrite_does_not_destroy_existing_map(monkeypatch, tmp_path):
    output = _stub_source(monkeypatch, tmp_path, matches=False, overwrite=True)
    output.parent.mkdir()
    output.write_bytes(b"existing map")
    with pytest.raises(RuntimeError, match="incomplete episode-exact map"):
        BUILDER.main()
    assert output.read_bytes() == b"existing map"
