"""Episode-exact init states in lerobot-eval (LIBERO benchmark stubbed out)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.scripts.lerobot_eval import _apply_exact_init_states


class _Env:
    def __init__(self, episode_index: int):
        self.unwrapped = self
        self.episode_index = episode_index
        self.init_states = False
        self._init_states = None
        self.init_state_id = -1


class _Vec:
    def __init__(self, n: int):
        self.envs = [_Env(i) for i in range(n)]
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def libero_stub(monkeypatch):
    tasks = [SimpleNamespace(name="task_a"), SimpleNamespace(name="task_b")]
    benchmark = types.ModuleType("libero.libero.benchmark")
    benchmark.get_benchmark_dict = lambda: {"libero_10": lambda: SimpleNamespace(tasks=tasks)}
    package = types.ModuleType("libero.libero")
    package.benchmark = benchmark
    monkeypatch.setitem(sys.modules, "libero", types.ModuleType("libero"))
    monkeypatch.setitem(sys.modules, "libero.libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", benchmark)


def _npz(tmp_path, rows):
    path = tmp_path / "eval_init_states.npz"
    states = np.empty(len(rows), dtype=object)
    for index, (_, _, state) in enumerate(rows):
        states[index] = np.asarray(state, dtype=np.float64)
    np.savez(
        path,
        episode_index=np.array([r[0] for r in rows], dtype=np.int32),
        scene_file=np.array([f"{r[1]}_demo.hdf5" for r in rows]),
        init_states=states,
    )
    return path


def test_rollout_k_starts_from_the_kth_matched_episode(tmp_path, libero_stub):
    path = _npz(tmp_path, [(7, "task_a", [7.0, 7.0]), (3, "task_a", [3.0, 3.0]), (5, "task_b", [5.0, 5.0])])
    envs = {"libero_10": {0: _Vec(2)}}
    _apply_exact_init_states(envs, str(path), n_episodes=2)
    for index, env in enumerate(envs["libero_10"][0].envs):
        assert env.init_states is True and env.init_state_id == index
        np.testing.assert_array_equal(env._init_states, [[3.0, 3.0], [7.0, 7.0]])  # sorted by episode


def test_tasks_without_enough_matches_are_dropped(tmp_path, libero_stub):
    path = _npz(tmp_path, [(3, "task_a", [3.0]), (4, "task_a", [4.0]), (5, "task_b", [5.0])])
    short = _Vec(2)
    envs = {"libero_10": {0: _Vec(2), 1: short}}
    _apply_exact_init_states(envs, str(path), n_episodes=2)
    assert list(envs["libero_10"]) == [0] and short.closed
    with pytest.raises(ValueError, match="No task"):
        _apply_exact_init_states({"libero_10": {1: _Vec(2)}}, str(path), n_episodes=2)


def test_panel_metrics_merge_into_the_stage1_chunk_file(tmp_path):
    import json

    from lerobot.scripts.lerobot_eval import _write_panel_metrics

    info = {"overall": {"pc_success": 50.0}, "per_task": [{"task_id": 0}], "per_group": {"x": 1}}
    path = _write_panel_metrics(info, str(tmp_path / "metrics"), "a", "w000_t0-0")
    _write_panel_metrics({**info, "overall": {"pc_success": 75.0}}, str(tmp_path / "metrics"), "b", "w000_t0-0")
    assert path.name == "eval_info_w000_t0-0.json"
    chunk = json.loads(path.read_text())
    assert list(chunk) == ["a", "b"] and set(chunk["a"]) == {"overall", "per_task"}
    assert chunk["b"]["overall"]["pc_success"] == 75.0


def test_wrist_panel_needs_annotation_and_its_own_switch(monkeypatch):
    from lerobot.scripts.lerobot_eval import _annotate_wrist_enabled

    monkeypatch.setenv("LEROBOT_EVAL_VIDEO_WRIST", "1")
    monkeypatch.delenv("LEROBOT_EVAL_ANNOTATE_VIDEOS", raising=False)
    assert not _annotate_wrist_enabled()
    monkeypatch.setenv("LEROBOT_EVAL_ANNOTATE_VIDEOS", "1")
    assert _annotate_wrist_enabled()
    monkeypatch.setenv("LEROBOT_EVAL_VIDEO_WRIST", "0")
    assert not _annotate_wrist_enabled()
