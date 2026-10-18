"""Episode outcome badges use the same success result as evaluation metrics."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.scripts import lerobot_skillvla_eval as skillvla_eval


def test_skill_html_serializes_success_failure_and_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(skillvla_eval, "_save_training_skill_examples", lambda **kwargs: {})
    policy = SimpleNamespace(config=SimpleNamespace(skill_fsq_levels=[3, 3, 3], chunk_size=10))
    path = skillvla_eval._write_task_skill_html(
        task_group="libero_10",
        task_id=0,
        task_description="A test task",
        records=[
            {"episode_index": 3, "success": np.bool_(True), "trace": []},
            {"episode_index": 9, "success": False, "trace": []},
            {"episode_index": 20, "trace": []},
        ],
        policy=policy,
        output_dir=tmp_path,
        train_samples=0,
        skill_latents_path=None,
        raw_dataset_dir=None,
        image_key=None,
    )
    source = Path(path).read_text()
    payload, _ = json.JSONDecoder().raw_decode(source.split("const DATA = ", 1)[1])
    assert [(ep["episode_index"], ep["success"]) for ep in payload["episodes"]] == [
        (3, True), (9, False), (20, None)
    ]
    assert "ep.success === true" in source
    assert "ep.success === false" in source
    assert 'success: "SUCCESS", failure: "FAILURE", unknown: "UNKNOWN"' in source
    assert ".result-success" in source and ".result-failure" in source


def test_skill_html_records_follow_metric_outcomes_across_batches(monkeypatch):
    class FakePolicy:
        def eval(self):
            return self

    class FakeVectorEnv:
        num_envs = 2
        envs = [SimpleNamespace(render=lambda: np.zeros((4, 4, 3), dtype=np.uint8))] * 2

    batches = [
        torch.tensor([[False, True, False, False], [False, False, False, False]]),
        torch.tensor([[False, False, False, False], [False, True, False, False]]),
    ]

    def fake_rollout(*, env, render_callback, **kwargs):
        for _ in range(4):
            render_callback(env)
        return {
            "done": torch.tensor([[False, False, True, True], [False, False, False, True]]),
            "reward": torch.zeros(2, 4),
            "success": batches.pop(0),
        }

    monkeypatch.setattr(skillvla_eval, "PreTrainedPolicy", FakePolicy)
    monkeypatch.setattr(skillvla_eval.gym.vector, "SyncVectorEnv", FakeVectorEnv)
    monkeypatch.setattr(skillvla_eval, "rollout", fake_rollout)
    info = skillvla_eval.eval_policy(
        env=FakeVectorEnv(),
        policy=FakePolicy(),
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=3,
        start_seed=1000,
        collect_skill_html=True,
    )
    records = info["skill_html_records"]
    assert [record["episode_index"] for record in records] == [0, 1, 2]
    assert [record["success"] for record in records] == [True, False, False]
    assert [record["success"] for record in records] == [ep["success"] for ep in info["per_episode"]]
    assert info["aggregated"]["pc_success"] == pytest.approx(100 / 3)
    assert not batches
