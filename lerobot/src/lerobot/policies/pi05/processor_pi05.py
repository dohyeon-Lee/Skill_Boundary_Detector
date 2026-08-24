#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
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
from lerobot.processor.eef_relative_action_processor import (
    EefRelativeActionsProcessorStep,
    EefRelativeToOscActionsProcessorStep,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def _load_pi05_eef_relative_action_stats(config: PI05Config) -> dict[str, torch.Tensor]:
    """Load and validate the relative target distribution for one PI0.5 chunk."""
    if not config.eef_relative_stats_path:
        raise ValueError(
            "use_eef_relative_actions=True needs eef_relative_stats_path "
            "(derived LIBERO dataset meta/relative_action_stats.json)"
        )
    path = Path(config.eef_relative_stats_path)
    if not path.is_file():
        raise FileNotFoundError(f"eef_relative_stats_path not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid EEF relative stats file {path}: {error}") from error

    expected_contract = {
        "representation": "eef_anchor_relative_so3",
        "storage_representation": "absolute_eef_command",
        "rotation_representation": "axis_angle_rotation_vector",
        "rotation_composition": "left_world",
    }
    mismatches = {
        key: (payload.get(key), expected)
        for key, expected in expected_contract.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Unsupported EEF relative stats contract in {path}: {mismatches}")

    if int(payload.get("chunk_size", 0)) < int(config.chunk_size):
        raise ValueError(
            f"EEF relative stats chunk_size={payload.get('chunk_size')} "
            f"< policy chunk_size={config.chunk_size}"
        )

    for key, configured in (
        ("osc_position_scale", config.eef_position_scale),
        ("osc_rotation_scale", config.eef_rotation_scale),
    ):
        try:
            stored = float(payload[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid or missing {key} in {path}") from error
        if not math.isclose(stored, configured, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(
                f"{key} mismatch: stats={stored}, policy={configured}. "
                "Execution must use the same OSC scale as dataset construction."
            )

    try:
        stats = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in payload["action"].items()
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid action statistics in {path}") from error
    invalid_shapes = {
        key: tuple(value.shape) for key, value in stats.items() if value.shape != (7,)
    }
    if invalid_shapes:
        raise ValueError(f"EEF relative action stats must be 7D, got {invalid_shapes}")
    return stats


def with_pi05_eef_relative_action_stats(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return dataset stats with only the action distribution replaced."""
    stats = dict(dataset_stats or {})
    stats[ACTION] = _load_pi05_eef_relative_action_stats(config)
    return stats


def reconnect_pi05_eef_relative_processors(preprocessor: Any, postprocessor: Any) -> None:
    """Restore the runtime link intentionally omitted from serialized configs."""
    relative_step = next(
        (step for step in preprocessor.steps if isinstance(step, EefRelativeActionsProcessorStep)),
        None,
    )
    osc_step = next(
        (step for step in postprocessor.steps if isinstance(step, EefRelativeToOscActionsProcessorStep)),
        None,
    )
    if relative_step is None or osc_step is None:
        raise RuntimeError("EEF-relative PI0.5 checkpoint is missing its paired processor steps.")
    osc_step.relative_step = relative_step


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    eef_relative_step = None
    if config.use_eef_relative_actions:
        if OBS_STATE not in config.input_features:
            raise ValueError("EEF-relative PI0.5 requires observation.state.")
        action_feature = config.output_features.get(ACTION)
        if action_feature is None or action_feature.shape != (7,):
            raise ValueError("EEF-relative PI0.5 requires a 7D action feature.")
        dataset_stats = with_pi05_eef_relative_action_stats(config, dataset_stats)
        eef_relative_step = EefRelativeActionsProcessorStep(enabled=True)

    # Add remaining processors
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        *([eef_relative_step] if eef_relative_step is not None else []),
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_path or "google/paligemma-3b-pt-224",
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
        *(
            [
                EefRelativeToOscActionsProcessorStep(
                    enabled=True,
                    position_scale=config.eef_position_scale,
                    rotation_scale=config.eef_rotation_scale,
                    relative_step=eef_relative_step,
                )
            ]
            if eef_relative_step is not None
            else []
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
