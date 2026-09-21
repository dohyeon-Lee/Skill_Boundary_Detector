"""pi05 proprio-grounding toggle, rollout grounding, and removed-LoRA checkpoint migration."""

from __future__ import annotations

import json

import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.skill_expert.processor_skill_expert import (
    EpisodeStartXYZGroundingProcessorStep,
)
from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline
from lerobot.scripts.lerobot_eval import _add_rollout_proprio_grounding
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE


def test_grounding_field_is_validated() -> None:
    assert PI05Config(device="cpu").proprio_grounding == "none"
    assert PI05Config(device="cpu", proprio_grounding="episode-start-xyz").proprio_grounding == "episode_start_xyz"
    with pytest.raises(ValueError, match="proprio_grounding"):
        PI05Config(device="cpu", proprio_grounding="world")


def _save_config(tmp_path, **extra) -> None:
    config = {"type": "pi05", "device": "cpu", "chunk_size": 10, **extra}
    (tmp_path / "config.json").write_text(json.dumps(config))


def test_legacy_lora_fields_are_dropped_on_load(tmp_path) -> None:
    _save_config(tmp_path, lora_enable=False, lora_rank=8, lora_alpha=16.0, lora_dropout=0.0,
                 lora_llm=True, lora_vision=False, lora_targets="q,k,v,o")
    config = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(config, PI05Config) and config.chunk_size == 10
    assert not hasattr(config, "lora_enable")


def test_lora_trained_checkpoints_are_rejected(tmp_path) -> None:
    _save_config(tmp_path, lora_enable=True)
    with pytest.raises(ValueError, match="removed LoRA probe"):
        PreTrainedConfig.from_pretrained(tmp_path)


def _pipeline() -> PolicyProcessorPipeline:
    normalizer = NormalizerProcessorStep(features={}, norm_map={}, stats={})
    return PolicyProcessorPipeline(
        steps=[normalizer], to_transition=lambda value: value, to_output=lambda value: value
    )


def test_rollout_grounding_is_inserted_only_for_grounded_pi05() -> None:
    plain = _pipeline()
    _add_rollout_proprio_grounding(plain, PI05Config(device="cpu"))
    assert len(plain.steps) == 1

    grounded = _pipeline()
    config = PI05Config(device="cpu", proprio_grounding="episode_start_xyz")
    _add_rollout_proprio_grounding(grounded, config)
    _add_rollout_proprio_grounding(grounded, config)  # idempotent
    assert [type(step) for step in grounded.steps] == [
        EpisodeStartXYZGroundingProcessorStep,
        NormalizerProcessorStep,
    ]


def test_rollout_reference_is_latched_per_episode_and_cleared_by_reset() -> None:
    pipeline = _pipeline()
    _add_rollout_proprio_grounding(pipeline, PI05Config(device="cpu", proprio_grounding="episode_start_xyz"))
    step = pipeline.steps[0]

    def ground(state):
        return step({TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor(state)}})[
            TransitionKey.OBSERVATION
        ][OBS_STATE]

    torch.testing.assert_close(ground([[1.0, 2.0, 3.0, 9.0]]), torch.tensor([[0.0, 0.0, 0.0, 9.0]]))
    torch.testing.assert_close(ground([[1.5, 2.0, 2.0, 8.0]]), torch.tensor([[0.5, 0.0, -1.0, 8.0]]))
    pipeline.reset()  # what rollout() does at the start of every episode batch
    torch.testing.assert_close(ground([[7.0, 7.0, 7.0, 1.0]]), torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
