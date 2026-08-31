"""Pre/post processors for Stage-1 VSA and its isolated skill predictor."""

from __future__ import annotations

from typing import Any

import torch

from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
from lerobot.policies.skillVLA.dataset_skillVLA import (
    SKILL_CANONICAL_ACTION_IS_PAD,
    SKILL_CANONICAL_ACTION_LENGTH,
    SKILL_CANONICAL_ACTIONS,
    SKILL_PREVIOUS_ACTION,
    SKILL_PREVIOUS_ACTION_BOS,
)
from lerobot.policies.skillVLA.processor_skillVLA import (
    SAME_SKILL_PAIR_FALLBACK,
    SAME_SKILL_PAIR_ID,
    SKILL_CODE,
    SKILL_EFFECTIVE_DE,
    SKILL_PROGRESS,
    SKILL_START_IMAGE,
    SKILL_START_STATE,
    SKILL_START_WRIST_IMAGE,
    SkillVLAPrepareStateTokenizerProcessorStep,
    SkillVLAPreserveRawStateProcessorStep,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    DONE,
    INFO,
    OBS_PREFIX,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    REWARD,
    TRUNCATED,
)

SKILL_BATCH_KEYS = (
    "skill_index",
    "skill_sequence",
    "skill_sequence_len",
    "skill_ds",
    "skill_de",
    SKILL_EFFECTIVE_DE,
    SKILL_START_IMAGE,
    SKILL_START_WRIST_IMAGE,
    SKILL_START_STATE,
    SKILL_CODE,
    "skill_code_true",
    SKILL_PROGRESS,
    SAME_SKILL_PAIR_ID,
    SAME_SKILL_PAIR_FALLBACK,
    SKILL_PREVIOUS_ACTION,
    SKILL_PREVIOUS_ACTION_BOS,
    SKILL_CANONICAL_ACTIONS,
    SKILL_CANONICAL_ACTION_IS_PAD,
    SKILL_CANONICAL_ACTION_LENGTH,
)


@ProcessorStepRegistry.register(name="skill_expert_normalizer_processor_step")
class SkillExpertNormalizerProcessorStep(NormalizerProcessorStep):
    """Normalize the main chunk and canonical skill target with one action contract."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        normalized = super().__call__(transition)
        complementary = dict(
            normalized.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        )
        canonical = complementary.get(SKILL_CANONICAL_ACTIONS)
        if canonical is not None:
            canonical_tensor = torch.as_tensor(canonical)
            complementary[SKILL_CANONICAL_ACTIONS] = self._normalize_action(
                canonical_tensor, inverse=False
            )
            normalized[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return normalized


def skill_expert_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    observation = {key: value for key, value in batch.items() if key.startswith(OBS_PREFIX)}
    complementary_data = {
        key: batch[key]
        for key in (
            "task",
            "index",
            "task_index",
            "episode_index",
            *SKILL_BATCH_KEYS,
        )
        if key in batch
    }
    complementary_data.update(
        {key: value for key, value in batch.items() if "_is_pad" in key}
    )
    return {
        TransitionKey.OBSERVATION: observation or None,
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
    ]
    # The same raw inputs are needed both for terminator training and for
    # checkpoint terminator inference.  Stage 2 must therefore preserve them
    # even when the inherited terminator stays frozen.
    if config.train_terminator:
        input_steps.append(SkillVLAPreserveRawStateProcessorStep())
    normalizer_cls = (
        SkillExpertNormalizerProcessorStep
        if getattr(config, "skill_flow_enabled", False)
        else NormalizerProcessorStep
    )
    input_steps.append(
        normalizer_cls(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )
    )
    if config.uses_skill_predictor:
        state_stats = (dataset_stats or {}).get(OBS_STATE, {}) or {}
        input_steps.extend(
            [
                SkillVLAPrepareStateTokenizerProcessorStep(
                    max_state_dim=config.max_state_dim,
                    state_q01=state_stats.get("q01"),
                    state_q99=state_stats.get("q99"),
                ),
                TokenizerProcessorStep(
                    tokenizer_name=(
                        config.tokenizer_path or "google/paligemma-3b-pt-224"
                    ),
                    max_length=config.tokenizer_max_length,
                    padding_side="right",
                    padding="max_length",
                ),
            ]
        )
    input_steps.append(DeviceProcessorStep(device=config.device))
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline(
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=skill_expert_batch_to_transition,
            to_output=skill_expert_transition_to_batch,
        ),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
