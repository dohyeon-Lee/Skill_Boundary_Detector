from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
EVAL_SRC = ROOT / "lerobot/skillVLA/abc/configs/train_pi05/PT/eval/src"
if str(EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC))

from eval_config import load_eval_config, load_global_config, _global_path  # noqa: E402
from pi05_adapter import Pi05PolicyAdapter, format_pi05_prompts, quantile_transform, _load_quantiles  # noqa: E402
from run_eval import HoldPolicy, resolve_dataset_tasks, rollout_over, video_frame, run_sequential_task, run  # noqa: E402
import run_eval  # noqa: E402


CONFIG_PATH = ROOT / "lerobot/skillVLA/abc/configs/train_pi05/PT/eval/abc_sim_eval_config.yaml"


def test_eval_config_uses_global_slurm_and_excludes() -> None:
    config = load_eval_config(CONFIG_PATH)
    assert config.backend == "mujoco"
    assert config.tasks == ("grab_clutter",)
    assert config.policy_type == "pi05"
    assert config.policy_checkpoint.name == "pretrained_model"
    assert config.policy_checkpoint.parent.name == "050000"
    assert config.policy_model_chunk_size == 30
    assert config.action_chunk_size == 15
    global_config = load_global_config(_global_path(CONFIG_PATH))
    partitions = global_config["train_partition"]
    assert config.slurm_partition == tuple(partitions if isinstance(partitions, list) else [partitions])
    assert config.slurm_qos == global_config["train_qos"]
    assert config.max_steps == 3540
    assert config.parallel_worlds == 4
    assert config.prompt_prefix == ""
    assert config.output_dir.parent == EVAL_SRC.parent / "outputs"


def test_all_tasks_come_from_native_manifest(tmp_path) -> None:
    config = replace(load_eval_config(CONFIG_PATH), tasks=("all",), dataset_dir=tmp_path)
    marker = {"format": "sbd_abc_sim_native/v1", "task_sha256": {"grab_clutter": "abc", "sim_spell_abc": "def"}}
    (tmp_path / "native_dataset.json").write_text(json.dumps(marker))
    assert resolve_dataset_tasks(config) == ["grab_clutter", "sim_spell_abc"]


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


def test_checkpoint_quantiles_override_dataset_and_partial_files_fail(tmp_path):
    from safetensors.numpy import save_file
    values = {f"{kind}.{q}": np.full(14, v, dtype=np.float32)
              for kind in ("observation.state", "action") for q, v in (("q01", -2), ("q99", 3))}
    save_file(values, str(tmp_path / "policy_preprocessor_step_2_normalizer_processor.safetensors"))
    with pytest.raises(FileNotFoundError):
        _load_quantiles(tmp_path / "missing_dataset", tmp_path)
    save_file(values, str(tmp_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    result = _load_quantiles(tmp_path / "missing_dataset", tmp_path)
    np.testing.assert_array_equal(result["action"]["q99"], np.full(14, 3))


@pytest.mark.parametrize('save_video', [False, True])
def test_random_scenes_batch_inference_and_resume_completed_worlds(tmp_path, monkeypatch, save_video):
    created = []

    class FakeResult:
        def __init__(self, success):
            self.success = success

        def to_info(self, squeeze):
            return {"success": self.success, "ever_success": self.success, "reward": float(self.success)}

    class FakeEnv:
        def __init__(self):
            self.steps = 0
            self.closed = False
            self._last_randomization = {"object_count": len(created) + 1}
            created.append(self)

        def reset(self, seed, **kwargs):
            self.seed = seed
            return self.get_obs(), {}

        def get_obs(self):
            return {"state": np.full(14, self.seed, dtype=np.float32), "prompt": f"object {self.seed}",
                    "images": {name: np.zeros((3, 8, 10), dtype=np.uint8) for name in ("top", "left", "right")}}

        def evaluate_task(self):
            return FakeResult(self.seed % 2 == 0 and self.steps >= 1)

        def step(self, action, **kwargs):
            self.steps += 1

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "abc_sim", SimpleNamespace(make_env=lambda **kwargs: FakeEnv()))
    batches = []
    class Policy(HoldPolicy):
        def infer(self, observation):
            batches.append(len(observation["state"]))
            return super().infer(observation)
        def video_frame(self, world, env, images, cameras):
            return np.full((8, 30, 3), 77, np.uint8)

    recordings = {}
    class Writer:
        def __init__(self, path): self.frames = recordings.setdefault(str(path), [])
        def append_data(self, frame): self.frames.append(frame.copy())
        def close(self): pass
    monkeypatch.setattr(run_eval, '_video_writer', lambda path, config: Writer(path))
    def finish(raw, target, world, fps):
        world.update(raw_video_path=str(raw), video_path=str(target))
    monkeypatch.setattr(run_eval, 'finish_video', finish)

    config = replace(load_eval_config(CONFIG_PATH), num_worlds=3, parallel_worlds=2,
                     seed=1000, max_steps=3, save_video=save_video, output_dir=tmp_path/'output')
    spec = SimpleNamespace(name="grab_clutter", prompt="grab", evaluator_name="target_in_box")
    result = run_sequential_task(config, "grab_clutter", spec, Policy(2), tmp_path)
    assert [world["steps"] for world in result["worlds"]] == [1, 3, 1]
    assert result["successes"] == 2
    assert batches == [2, 1, 1]
    assert all(env.closed for env in created)
    assert (tmp_path / "partial_summary.json").exists()
    resumed = run_sequential_task(config, "grab_clutter", spec, Policy(2), tmp_path)
    assert resumed == result
    assert len(created) == 3
    if save_video:
        assert len(recordings) == 6
        for world in result['worlds']:
            raw, marked = recordings[world['raw_video_path']], recordings[world['camera_overlay_path']]
            assert len(raw) == len(marked) == world['steps'] + 1
            assert not any(frame.any() for frame in raw)
            assert all((frame == 77).all() for frame in marked[1:])


def test_output_contract_mismatch_prevents_loading_policy(tmp_path):
    config = replace(load_eval_config(CONFIG_PATH), output_dir=tmp_path)
    saved = config.json_dict()
    saved["max_steps"] = 1
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics/resolved_config.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="different evaluation contract"):
        run(config)


def test_abc_llm_checkpoint_config_can_be_loaded_without_changing_source():
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    source = {"type": "pi05", "gradient_checkpointing": True,
              "gradient_checkpointing_expert": False,
              "gradient_checkpointing_llm": True,
              "gradient_checkpointing_vision": False}
    migrated = PI05Config._migrate_legacy_config(source)
    assert migrated == source
