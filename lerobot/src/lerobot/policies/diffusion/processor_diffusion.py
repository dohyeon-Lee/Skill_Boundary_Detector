#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register("diffusion_relative_actions_processor")
class DiffusionRelativeActionsProcessorStep(RelativeActionsProcessorStep):
    """Relative-action conversion for Diffusion Policy batches.

    The official step assumes a single current state (B, D) — pi-style. DP batches carry an
    observation-history WINDOW (B, n_obs, D), so the conversion must anchor on the CURRENT
    state = the window's LAST step (matches the pi/VLA convention "chunk relative to the state
    at prediction time", which the SBD probe will also use). This subclass reduces the windowed
    state to its last step before delegating to the official conversion/caching logic."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {}) or {}
        state = observation.get(OBS_STATE)
        if state is not None and state.ndim >= 3:  # (B, n_obs, D) → anchor = 현재(마지막) 스텝
            transition = transition.copy()
            observation = dict(observation)
            observation[OBS_STATE] = state[..., -1, :]
            transition[TransitionKey.OBSERVATION] = observation
            new_tr = super().__call__(transition)
            # 원 윈도우 state 복원 (모델은 히스토리 전체를 조건으로 씀)
            observation = dict(new_tr.get(TransitionKey.OBSERVATION, {}) or {})
            observation[OBS_STATE] = state
            new_tr[TransitionKey.OBSERVATION] = observation
            return new_tr
        return super().__call__(transition)


def _load_relative_action_stats(config: DiffusionConfig) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """④-b의 meta/relative_action_stats.json 로드 → (action stats 텐서, exclude_joints, action_names).

    normalizer가 relative 분포를 봐야 하므로(순서: relative → normalize) absolute stats 대신
    이 파일의 stats를 action에 물린다. gripper 등 exclude dim은 ④-b가 absolute 분포 값을
    그대로 담고 있어(변환 함수가 그 dim을 absolute로 남기므로) 특례 없이 일관된다."""
    if not config.relative_stats_path:
        raise ValueError(
            "use_relative_actions=True needs relative_stats_path "
            "(dataset의 meta/relative_action_stats.json — ABC_dataset build ④-b 산출물)")
    path = Path(config.relative_stats_path)
    if not path.exists():
        raise FileNotFoundError(f"relative_stats_path not found: {path}")
    payload = json.loads(path.read_text())
    horizon_needed = int(config.horizon)
    if int(payload.get("chunk_size", 0)) < horizon_needed:
        raise ValueError(
            f"relative stats chunk_size={payload.get('chunk_size')} < policy horizon={horizon_needed} — "
            "긴 오프셋 분포가 빠져 정규화가 어긋남. compute_relative_action_stats.py를 "
            f"--chunk-size {horizon_needed} 이상으로 재계산하세요.")
    stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in payload["action"].items()}
    return stats, list(payload.get("exclude_joints") or []), list(payload.get("action_names") or [])


def make_diffusion_pre_post_processors(
    config: DiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a diffusion policy.

    The pre-processing pipeline prepares the input data for the model by:
    1. Renaming features.
    2. Normalizing the input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Moving the data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving the data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the diffusion policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = None
    if config.use_relative_actions:
        # relative → normalize 순서 (pi 계열과 동일). action stats는 relative 분포로 교체 —
        # 원본 dataset_stats(absolute)는 다른 feature용으로 그대로 두고 action만 스왑.
        rel_stats, exclude_joints, action_names = _load_relative_action_stats(config)
        dataset_stats = dict(dataset_stats or {})
        dataset_stats["action"] = rel_stats
        relative_step = DiffusionRelativeActionsProcessorStep(
            enabled=True, exclude_joints=exclude_joints, action_names=action_names or None)

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        *([relative_step] if relative_step is not None else []),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        # unnormalize → +anchor state (짝 스텝의 캐시) → absolute 복원 (추론/replay 소비자용)
        *([AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)]
          if relative_step is not None else []),
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
