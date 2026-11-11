from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_BUILD_SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/build_data/src"

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load("restored_oracle_builder", _BUILD_SRC / "build_init_states.py")
ENSURE = _load("ensure_episode_map_test", _BUILD_SRC / "ensure_episode_map.py")


def test_existing_episode_map_is_preserved_without_running_matcher(tmp_path, monkeypatch):
    output = tmp_path / "eval_init_states.npz"
    output.write_bytes(b"existing map")
    monkeypatch.setattr(ENSURE.subprocess, "run", lambda *a, **k: pytest.fail("must skip"))
    assert not ENSURE.ensure_episode_map(tmp_path / "source", tmp_path / "original", output, tmp_path)
    assert output.read_bytes() == b"existing map"


def test_missing_episode_map_is_built_then_reused(tmp_path, monkeypatch):
    source = tmp_path / "libero_90_full_full"
    original = tmp_path / "original"
    output = tmp_path / "skillvla/libero_90_full_full/eval_init_states.npz"
    calls = []

    def run(command, *, check):
        calls.append(command)
        assert check
        assert Path(command[2]).name == "build_init_states.py"
        assert command[command.index("--lerobot_dataset") + 1] == str(source)
        assert command[command.index("--orig_dataset") + 1] == str(original)
        assert not output.exists()
        np.savez(command[command.index("--out") + 1], episode_index=[0])

    monkeypatch.setattr(ENSURE.subprocess, "run", run)
    assert ENSURE.ensure_episode_map(source, original, output, tmp_path)
    assert not ENSURE.ensure_episode_map(source, original, output, tmp_path)
    assert len(calls) == 1
    with np.load(output) as data:
        assert data["episode_index"].tolist() == [0]


def test_failed_matching_never_publishes_partial_map(tmp_path, monkeypatch):
    output = tmp_path / "maps/eval_init_states.npz"

    def run(command, **kwargs):
        Path(command[command.index("--out") + 1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(ENSURE.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        ENSURE.ensure_episode_map(tmp_path / "source", tmp_path / "original", output, tmp_path)
    assert not output.exists()
    assert not any(path.is_dir() for path in output.parent.iterdir())


def test_success_without_map_is_not_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(ENSURE.subprocess, "run", lambda *a, **k: None)
    output = tmp_path / "maps/eval_init_states.npz"
    with pytest.raises(RuntimeError, match="without creating"):
        ENSURE.ensure_episode_map(tmp_path / "source", tmp_path / "original", output, tmp_path)
    assert not output.exists()


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
