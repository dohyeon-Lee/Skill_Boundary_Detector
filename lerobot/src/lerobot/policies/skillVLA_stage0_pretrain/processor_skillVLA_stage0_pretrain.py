from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.skillVLA.processor_skillVLA import (
    SkillVLAPreserveRawStateProcessorStep,
    SkillVLAPrepareStateTokenizerProcessorStep,
    skill_vla_batch_to_transition,
    skill_vla_transition_to_batch,
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
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_skillVLA_stage0_pretrain import SkillVLAStage0PretrainConfig
from .dataset_skillVLA_stage0_pretrain import (
    AR_FAST_TOKEN_MASK,
    AR_FAST_TOKENS,
    AR_STATE,
    AR_TASK,
)

AR_LANGUAGE_TOKENS = "stage0_pretrain_language_tokens"
AR_LANGUAGE_MASK = "stage0_pretrain_language_attention_mask"
AR_ACTION_TOKENS = "stage0_pretrain_action_tokens"
AR_ACTION_TOKEN_MASK = "stage0_pretrain_action_token_mask"


@dataclass
@ProcessorStepRegistry.register(name="skill_vla_stage0_pretrain_prompt_processor_step")
class Stage0PretrainPromptProcessorStep(SkillVLAPrepareStateTokenizerProcessorStep):
    """Use the exact pretraining prefix: Task + discretized state + newline, before BOS/skill."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = super().__call__(transition)
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        prompts = comp.get(self.task_key)
        if prompts is not None:
            comp[self.task_key] = [
                prompt[: -len("Action: ")] if prompt.endswith("Action: ") else prompt
                for prompt in prompts
            ]
        return transition


@dataclass
@ProcessorStepRegistry.register(name="skill_vla_stage0_pretrain_ar_processor_step")
class Stage0PretrainARProcessorStep(ProcessorStep):
    tokenizer_name: str
    tokenizer_max_length: int = 48
    fast_skip_tokens: int = 128
    fast_vocab_size: int = 1024
    state_q01: object = None
    state_q99: object = None
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        from transformers import AutoTokenizer  # noqa: PLC0415

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=True,
            add_eos_token=True,
            add_bos_token=False,
        )

    def get_config(self) -> dict[str, Any]:
        config = {
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_max_length": self.tokenizer_max_length,
            "fast_skip_tokens": self.fast_skip_tokens,
            "fast_vocab_size": self.fast_vocab_size,
        }
        for name, value in (("state_q01", self.state_q01), ("state_q99", self.state_q99)):
            if value is not None:
                config[name] = np.asarray(value, dtype=np.float32).reshape(-1).tolist()
        return config

    def _normalized_state(self, value: torch.Tensor) -> np.ndarray:
        if self.state_q01 is None or self.state_q99 is None:
            raise ValueError("Stage0-pretrain AR prompt requires observation.state q01/q99 stats.")
        state = value.detach().cpu().numpy().astype(np.float32)
        if state.ndim == 1:
            state = state[None]
        q01 = np.asarray(self.state_q01, dtype=np.float32).reshape(-1)[: state.shape[-1]]
        q99 = np.asarray(self.state_q99, dtype=np.float32).reshape(-1)[: state.shape[-1]]
        denominator = np.where((q99 - q01) == 0, 1.0, q99 - q01)
        return 2.0 * (state - q01) / denominator - 1.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
        tasks = comp.get(AR_TASK)
        states = comp.get(AR_STATE)
        raw_tokens = comp.get(AR_FAST_TOKENS)
        raw_masks = comp.get(AR_FAST_TOKEN_MASK)
        if tasks is None or states is None or raw_tokens is None or raw_masks is None:
            raise ValueError("Stage0-pretrain transition AR fields are missing from the batch.")
        if isinstance(tasks, str):
            tasks = [tasks]
        state = self._normalized_state(states)
        discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        prompts = []
        for task, values in zip(tasks, discretized, strict=True):
            cleaned = str(task).strip().replace("_", " ").replace("\n", " ")
            prompts.append(f"Task: {cleaned}, State: {' '.join(map(str, values))};\n")
        encoded = self._tokenizer(
            prompts,
            max_length=self.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        comp[AR_LANGUAGE_TOKENS] = encoded["input_ids"]
        comp[AR_LANGUAGE_MASK] = encoded["attention_mask"].bool()

        if raw_tokens.ndim == 1:
            raw_tokens = raw_tokens.unsqueeze(0)
            raw_masks = raw_masks.unsqueeze(0)
        raw_tokens, raw_masks = raw_tokens.long(), raw_masks.bool()
        valid = raw_tokens[raw_masks]
        if valid.numel() and (int(valid.min()) < 0 or int(valid.max()) >= self.fast_vocab_size):
            raise ValueError(
                f"Raw FAST IDs must be in [0,{self.fast_vocab_size}); got "
                f"[{int(valid.min())},{int(valid.max())}]."
            )
        prefix = [int(self._tokenizer.bos_token_id)] + list(
            self._tokenizer.encode("Action: ", add_special_tokens=False)
        )
        suffix = list(self._tokenizer.encode("|", add_special_tokens=False))
        batch_size, max_raw = raw_tokens.shape
        total = len(prefix) + max_raw + len(suffix)
        tokens = torch.zeros(batch_size, total, dtype=torch.long, device=raw_tokens.device)
        masks = torch.zeros(batch_size, total, dtype=torch.bool, device=raw_tokens.device)
        prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=raw_tokens.device)
        suffix_tensor = torch.tensor(suffix, dtype=torch.long, device=raw_tokens.device)
        for batch_index in range(batch_size):
            count = int(raw_masks[batch_index].sum())
            mapped = (
                self._tokenizer.vocab_size
                - 1
                - int(self.fast_skip_tokens)
                - raw_tokens[batch_index, :count]
            )
            end = len(prefix) + count
            tokens[batch_index, : len(prefix)] = prefix_tensor
            tokens[batch_index, len(prefix) : end] = mapped
            tokens[batch_index, end : end + len(suffix)] = suffix_tensor
            masks[batch_index, : end + len(suffix)] = True
        comp[AR_ACTION_TOKENS] = tokens
        comp[AR_ACTION_TOKEN_MASK] = masks
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_skill_vla_stage0_pretrain_pre_post_processors(
    config: SkillVLAStage0PretrainConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    state_stats = (dataset_stats or {}).get(OBS_STATE, {}) or {}
    tokenizer_name = config.tokenizer_path or "google/paligemma-3b-pt-224"
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        SkillVLAPreserveRawStateProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Stage0PretrainPromptProcessorStep(
            max_state_dim=config.max_state_dim,
            state_q01=state_stats.get("q01"),
            state_q99=state_stats.get("q99"),
        ),
        TokenizerProcessorStep(
            tokenizer_name=tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        Stage0PretrainARProcessorStep(
            tokenizer_name=tokenizer_name,
            tokenizer_max_length=config.tokenizer_max_length,
            fast_skip_tokens=config.fast_skip_tokens,
            fast_vocab_size=config.fast_vocab_size,
            state_q01=state_stats.get("q01"),
            state_q99=state_stats.get("q99"),
        ),
        DeviceProcessorStep(device=config.device),
    ]
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
            to_transition=skill_vla_batch_to_transition,
            to_output=skill_vla_transition_to_batch,
        ),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
