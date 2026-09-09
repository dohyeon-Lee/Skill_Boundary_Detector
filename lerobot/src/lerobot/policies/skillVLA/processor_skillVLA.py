#!/usr/bin/env python

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE


# Skill-policy keys carried in complementary_data (model + closed-loop select_action consume these):
#   skill_start_*  : the VLM's skill-START view + state + FSQ code (from SkillVLADataset; training)
#   skill_decoder_*: RAW current obs for the FSQ terminator at inference (copied pre-normalization)
SKILL_START_IMAGE = "skill_start_image"
SKILL_START_WRIST_IMAGE = "skill_start_wrist_image"
SKILL_START_STATE = "skill_start_state"
SKILL_CODE = "skill_code"
SKILL_PROGRESS = "skill_progress"
SKILL_EFFECTIVE_DE = "skill_effective_de"
SAME_SKILL_PAIR_ID = "same_skill_pair_id"
SAME_SKILL_PAIR_FALLBACK = "same_skill_pair_fallback"

@dataclass
@ProcessorStepRegistry.register(name="skill_vla_preserve_raw_state_processor_step")
class SkillVLAPreserveRawStateProcessorStep(ProcessorStep):
    """Snapshot the RAW current obs (pre-normalization) for the FSQ terminator: raw state + raw
    3rd-person + wrist (DINO tokens if present, else RGB; the terminator auto-detects). Used at BOTH
    (a) closed-loop select_action, and (b) TRAINING — terminator co-training reads the raw current-frame
    state via skill_decoder_state (the FSQ normalizes it internally with its own min/max; feeding the
    quantile-normalized observation.state would double-normalize and desync train from eval). Runs
    before NormalizerProcessorStep so the snapshot is genuinely pre-normalization."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        comp_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
        observation = transition.get(TransitionKey.OBSERVATION, {}) or {}

        def _copy(v):
            return v.clone() if isinstance(v, torch.Tensor) else deepcopy(v)

        # Raw state (full observation.state) — always overwrite so a stale baked column can't shadow it.
        state = observation.get(OBS_STATE)
        if state is not None:
            comp_data["skill_decoder_state"] = _copy(state)
        # Raw images: prefer precomputed DINO tokens (token-FSQ), else RGB (image-FSQ); 3rd + wrist.
        third = observation.get("observation.dino.image") or observation.get("observation.images.image")
        if third is not None:
            comp_data["skill_decoder_image"] = _copy(third)
        wrist = observation.get("observation.dino.wrist") or observation.get("observation.images.wrist_image")
        if wrist is not None:
            comp_data["skill_decoder_wrist"] = _copy(wrist)

        transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

@dataclass
@ProcessorStepRegistry.register(name="skill_vla_prepare_state_tokenizer_processor_step")
class SkillVLAPrepareStateTokenizerProcessorStep(ProcessorStep):
    """PI05-style prompt (task text + discretized state) describing the VLM's SKILL-START obs.

    Training: discretize the raw ``skill_start_state`` (normalized here with observation.state's
    QUANTILES stats so it matches the normalizer). Inference: ``skill_start_state`` is absent, so
    discretize the current (already-normalized) observation.state — at a skill boundary the current
    frame *is* the skill start, and select_action snapshots these tokens for the rest of the skill.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    state_q01: object = None  # observation.state q01/q99 (np arrays) for normalizing skill_start_state
    state_q99: object = None

    def get_config(self) -> dict[str, Any]:
        # Persist q01/q99 (as lists) so a saved checkpoint is self-contained: on resume the step is
        # reconstructed WITH its quantiles instead of None (mirrors how the normalizer step persists
        # its stats). Without this the saved config is empty and resume crashes in _normalize_start_state.
        cfg: dict[str, Any] = {"max_state_dim": self.max_state_dim, "task_key": self.task_key}
        for name, val in (("state_q01", self.state_q01), ("state_q99", self.state_q99)):
            if val is not None:
                cfg[name] = np.asarray(val, dtype=np.float32).reshape(-1).tolist()
        return cfg

    def _normalize_start_state(self, state_np: np.ndarray) -> np.ndarray:
        if self.state_q01 is None or self.state_q99 is None:
            raise ValueError(
                "skill_start_state needs observation.state q01/q99 stats to be discretized "
                "(QUANTILES). Pass dataset_stats with quantile stats to make_skill_vla_pre_post_processors."
            )
        q01 = np.asarray(self.state_q01, dtype=np.float32).reshape(-1)[: state_np.shape[-1]]
        q99 = np.asarray(self.state_q99, dtype=np.float32).reshape(-1)[: state_np.shape[-1]]
        denom = np.where((q99 - q01) == 0, 1.0, q99 - q01)
        return 2.0 * (state_np - q01) / denom - 1.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        start_state = comp.get(SKILL_START_STATE)
        if start_state is not None:  # training: raw skill-start state → normalize to [-1, 1]
            sn = start_state.cpu().numpy() if isinstance(start_state, torch.Tensor) else np.asarray(start_state)
            state_np = self._normalize_start_state(sn)
        else:  # inference: current state, already normalized by the NormalizerProcessorStep
            state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
            if state is None:
                raise ValueError("State is required for SkillVLA")
            state_np = state.cpu().numpy()
        if state_np.ndim == 1:
            state_np = state_np[None, :]

        tasks = comp.get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
