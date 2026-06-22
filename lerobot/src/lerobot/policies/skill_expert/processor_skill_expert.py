#!/usr/bin/env python
"""Pre/post-processor pipeline for the Stage-1 SkillExpert policy.

Much simpler than the PI05/SkillVLA pipeline: there is no language, so no tokenizer and no
state-discretization-into-prompt. We normalize state/action (images stay [0, 1] via the
IDENTITY visual mode and are DINO-normalized inside the model) and preserve the skill
columns the policy needs (skill_sequence, skill_index).
"""

from typing import Any

import torch

from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    DONE,
    INFO,
    OBS_PREFIX,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    REWARD,
    TRUNCATED,
)

# Skill columns the SkillExpert needs from the dataset: current skill = skill_sequence[skill_index];
# skill_ds/skill_de give the within-skill offsets (boundary/cum/terminator targets). (The skill-progress
# token was removed — the action expert conditions on the skill code only.)
SKILL_EXPERT_BATCH_KEYS = ("skill_index", "skill_sequence", "skill_ds", "skill_de")


def skill_expert_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    observation = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
    complementary_data: dict[str, Any] = {}
    for key in ("task", "index", "task_index", "episode_index", *SKILL_EXPERT_BATCH_KEYS):
        if key in batch:
            complementary_data[key] = batch[key]
    complementary_data.update({k: v for k, v in batch.items() if "_is_pad" in k})
    return {
        TransitionKey.OBSERVATION: observation if observation else None,
        TransitionKey.ACTION: batch.get(ACTION),
        TransitionKey.REWARD: batch.get(REWARD, 0.0),
        TransitionKey.DONE: batch.get(DONE, False),
        TransitionKey.TRUNCATED: batch.get(TRUNCATED, False),
        TransitionKey.INFO: batch.get(INFO, {}),
        TransitionKey.COMPLEMENTARY_DATA: complementary_data,
    }


def skill_expert_transition_to_batch(transition: EnvTransition) -> dict[str, Any]:
    batch = {
        ACTION: transition.get(TransitionKey.ACTION),
        REWARD: transition.get(TransitionKey.REWARD, 0.0),
        DONE: transition.get(TransitionKey.DONE, False),
        TRUNCATED: transition.get(TransitionKey.TRUNCATED, False),
        INFO: transition.get(TransitionKey.INFO, {}),
    }
    batch.update(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
    observation = transition.get(TransitionKey.OBSERVATION)
    if isinstance(observation, dict):
        batch.update(observation)
    return batch


def make_skill_expert_pre_post_processors(
    config: SkillExpertConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,  # VISUAL=IDENTITY ([0,1]), STATE/ACTION=QUANTILES
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=skill_expert_batch_to_transition,
            to_output=skill_expert_transition_to_batch,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
