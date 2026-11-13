from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
EVAL_SRC = ROOT / "lerobot/examples/abc/configs/train_pi05/PT/eval/src"
if str(EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC))

from eval_config import load_eval_config  # noqa: E402
from pi05_adapter import Pi05PolicyAdapter, format_pi05_prompts, quantile_transform  # noqa: E402
from run_eval import HoldPolicy, resolve_dataset_tasks, rollout_over, video_frame  # noqa: E402


CONFIG_PATH = ROOT / "lerobot/examples/abc/configs/train_pi05/PT/eval/abc_sim_eval_config.yaml"


def test_eval_config_uses_global_slurm_and_excludes() -> None:
    config = load_eval_config(CONFIG_PATH)
    assert config.backend == "mjwarp"
    assert config.tasks == ("sim_spell_abc",)
    assert config.policy_type == "pi05"
    assert config.policy_checkpoint == ROOT / "models/pi05_base"
    assert config.policy_model_chunk_size == 30
    assert config.action_chunk_size == 15
    assert config.slurm_qos == "base_qos"
    assert "base_suma_rtx3090" in config.slurm_partition
    assert "node05" in config.slurm_exclude_nodes


def test_all_tasks_come_from_native_manifest() -> None:
    config = replace(load_eval_config(CONFIG_PATH), tasks=("all",))
    tasks = resolve_dataset_tasks(config)
    assert len(tasks) == 24
    assert "sim_spell_abc" in tasks
    marker = json.loads((config.dataset_dir / "native_dataset.json").read_text())
    assert tasks == list(marker["task_sha256"])


def test_hold_policy_single_and_batched_shapes() -> None:
    policy = HoldPolicy(chunk_size=15)
    single = np.arange(14, dtype=np.float32)
    batch = np.stack([single, single + 1])
    assert policy.infer({"state": single}).shape == (15, 14)
    actions = policy.infer({"state": batch})
    assert actions.shape == (2, 15, 14)
    np.testing.assert_array_equal(actions[:, 0], batch)


def test_rollout_over_handles_maintenance_tasks() -> None:
    assert rollout_over({"ever_success": True})
    assert not rollout_over({"ever_success": True, "ever_failed": False})
    assert rollout_over({"ever_success": True, "ever_failed": True})


def test_video_frame_concatenates_three_chw_cameras() -> None:
    images = {
        name: np.full((3, 8, 10), value, dtype=np.uint8)
        for value, name in enumerate(("top", "left", "right"))
    }
    frame = video_frame(images, ("top", "left", "right"))
    assert frame.shape == (8, 30, 3)
    assert frame.dtype == np.uint8


def test_pi05_quantiles_and_state_prompt_match_training_contract() -> None:
    raw = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    normalized = quantile_transform(raw, [0.0, 1.0], [2.0, 5.0])
    torch.testing.assert_close(normalized, torch.zeros_like(raw))
    torch.testing.assert_close(
        quantile_transform(normalized, [0.0, 1.0], [2.0, 5.0], inverse=True), raw
    )
    prompt = format_pi05_prompts(["sim_spell_abc"], normalized.numpy())
    assert prompt == ["Task: sim spell abc, State: 128 128;\nAction: "]


def test_pi05_adapter_maps_batched_abc_observation_to_absolute_actions() -> None:
    class FakeTokenizer:
        def __call__(self, prompts, **kwargs):
            assert prompts[0].startswith("Task: sim spell abc")
            return {
                "input_ids": torch.ones((len(prompts), 4), dtype=torch.int64),
                "attention_mask": torch.ones((len(prompts), 4), dtype=torch.int64),
            }

    class FakePolicy:
        config = SimpleNamespace(tokenizer_max_length=200)

        def predict_action_chunk(self, batch):
            assert set(key for key in batch if key.startswith("observation.images.")) == {
                "observation.images.base_0_rgb",
                "observation.images.left_wrist_0_rgb",
                "observation.images.right_wrist_0_rgb",
            }
            assert batch["observation.images.base_0_rgb"].shape == (2, 3, 8, 10)
            return torch.zeros((2, 3, 14), dtype=torch.float32)

    adapter = Pi05PolicyAdapter.__new__(Pi05PolicyAdapter)
    adapter.torch = torch
    adapter.device = torch.device("cpu")
    adapter.policy = FakePolicy()
    adapter.tokenizer = FakeTokenizer()
    adapter.config = SimpleNamespace(action_chunk_size=2)
    adapter.stats = {
        "observation.state": {
            "q01": np.zeros(14, dtype=np.float32),
            "q99": np.full(14, 2.0, dtype=np.float32),
        },
        "action": {
            "q01": np.zeros(14, dtype=np.float32),
            "q99": np.full(14, 2.0, dtype=np.float32),
        },
    }
    observation = {
        "state": np.ones((2, 14), dtype=np.float32),
        "prompt": ["sim spell abc", "sim spell abc"],
        "images": {
            key: np.zeros((2, 3, 8, 10), dtype=np.uint8)
            for key in ("top", "left", "right")
        },
    }
    actions = adapter.infer(observation)
    assert actions.shape == (2, 2, 14)
    np.testing.assert_allclose(actions, 1.0)
