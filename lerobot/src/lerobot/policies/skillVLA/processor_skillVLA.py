#!/usr/bin/env python

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.skillVLA.configuration_skillVLA import SkillVLAConfig
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
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_PREFIX,
    REWARD,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    TRUNCATED,
)


OBS_LANG_TO_ACTION_ATTENTION_MASK = "observation.language.to_action_attention_mask"
_STATE_ACTION_PREFIX_KEY = "skill_vla_state_action_prefix"
SKILL_VLA_BATCH_KEYS = (
    "skill_index",
    "skill_sequence",
    "skill_length_sequence",
    "skill_sequence_mask",
    "skill_sequence_len",
    "skill_ds",
    "skill_de",
    "skill_boundary",
    "skill_max_length",
    "skill_decoder_state",
    "skill_decoder_image",
)


def skill_vla_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    observation = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
    complementary_data = {}
    for key in ("task", "subtask", "index", "task_index", "episode_index", *SKILL_VLA_BATCH_KEYS):
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
        TransitionKey.COMPLEMENTARY_DATA: complementary_data if complementary_data else {},
    }


def skill_vla_transition_to_batch(transition: EnvTransition) -> dict[str, Any]:
    batch = {
        ACTION: transition.get(TransitionKey.ACTION),
        REWARD: transition.get(TransitionKey.REWARD, 0.0),
        DONE: transition.get(TransitionKey.DONE, False),
        TRUNCATED: transition.get(TransitionKey.TRUNCATED, False),
        INFO: transition.get(TransitionKey.INFO, {}),
    }

    comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
    if comp_data:
        batch.update(comp_data)

    observation = transition.get(TransitionKey.OBSERVATION)
    if isinstance(observation, dict):
        batch.update(observation)

    batch.pop(_STATE_ACTION_PREFIX_KEY, None)
    return batch


@dataclass
@ProcessorStepRegistry.register(name="skill_vla_preserve_raw_state_processor_step")
class SkillVLAPreserveRawStateProcessorStep(ProcessorStep):
    """Keep raw proprioceptive/image features for the FSQ skill decoder."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        comp_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})

        observation = transition.get(TransitionKey.OBSERVATION, {}) or {}
        if "skill_decoder_state" not in comp_data:
            state = observation.get(OBS_STATE)
            if state is not None:
                comp_data["skill_decoder_state"] = state.clone() if isinstance(state, torch.Tensor) else deepcopy(state)
        if "skill_decoder_image" not in comp_data:
            visual = observation.get("observation.dino.image")
            if visual is not None:
                comp_data["skill_decoder_image"] = visual.clone() if isinstance(visual, torch.Tensor) else deepcopy(visual)

        transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="skill_vla_prepare_state_tokenizer_processor_step")
class SkillVLAPrepareStateTokenizerProcessorStep(ProcessorStep):
    """Prepare the PI05 prompt and record where the non-task text starts."""

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for SkillVLA")

        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        state = deepcopy(state)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        task_prefixes = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            task_prefix = f"Task: {cleaned_text}, "
            state_action_suffix = f"State: {state_str};\nAction: "
            full_prompts.append(task_prefix + state_action_suffix)
            task_prefixes.append(task_prefix)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        transition[TransitionKey.COMPLEMENTARY_DATA][_STATE_ACTION_PREFIX_KEY] = task_prefixes
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="skill_vla_tokenizer_processor")
class SkillVLATokenizerProcessorStep(TokenizerProcessorStep):
    """Tokenize prompts and mark only `State: ...; Action:` as visible to action tokens."""

    def observation(self, observation):
        new_observation = super().observation(observation)

        task = self.get_task(self.transition)
        task_prefixes = self.transition[TransitionKey.COMPLEMENTARY_DATA].get(_STATE_ACTION_PREFIX_KEY)
        if task_prefixes is None:
            raise ValueError(f"{_STATE_ACTION_PREFIX_KEY} is required for SkillVLA token masking")

        prefix_tokenized = self._tokenize_text(task_prefixes)
        valid_mask = new_observation[OBS_LANGUAGE_ATTENTION_MASK]

        lang_to_action_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        for i in range(len(task)):
            full_len = int(valid_mask[i].sum().item())
            prefix_len = min(int(prefix_tokenized["attention_mask"][i].sum().item()), full_len)
            lang_to_action_mask[i, prefix_len:full_len] = True

        new_observation[OBS_LANG_TO_ACTION_ATTENTION_MASK] = lang_to_action_mask.to(
            device=valid_mask.device, dtype=torch.bool
        )
        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = super().transform_features(features)
        if OBS_LANG_TO_ACTION_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANG_TO_ACTION_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        return features


def make_skill_vla_pre_post_processors(
    config: SkillVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        SkillVLAPreserveRawStateProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        SkillVLAPrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        SkillVLATokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
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
            to_transition=skill_vla_batch_to_transition,
            to_output=skill_vla_transition_to_batch,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
