from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.policies.pi0_fast.processor_pi0_fast import (
    Pi0FastPrepareStateAndLanguageTokenizerProcessorStep,
)
from lerobot.policies.skillVLA.processor_skillVLA import (
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
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_skillVLA_pretrain import SkillVLAPretrainConfig
from .dataset_skillVLA_pretrain import PRETRAIN_FAST_TOKEN_MASK, PRETRAIN_FAST_TOKENS


@dataclass
@ProcessorStepRegistry.register(name="skill_vla_precomputed_fast_tokens_processor_step")
class SkillVLAPrecomputedFastTokensProcessorStep(ProcessorStep):
    """Map precomputed raw FAST IDs into PaliGemma IDs and add structural target tokens."""

    paligemma_tokenizer_name: str
    fast_skip_tokens: int = 128
    fast_vocab_size: int = 1024
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        from transformers import AutoTokenizer  # noqa: PLC0415

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.paligemma_tokenizer_name,
            trust_remote_code=True,
            add_eos_token=True,
            add_bos_token=False,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
        raw_tokens = complementary.get(PRETRAIN_FAST_TOKENS)
        raw_mask = complementary.get(PRETRAIN_FAST_TOKEN_MASK)
        if raw_tokens is None or raw_mask is None:
            raise ValueError("Precomputed full-skill FAST tokens/mask are missing from the batch.")
        if raw_tokens.ndim == 1:
            raw_tokens = raw_tokens.unsqueeze(0)
            raw_mask = raw_mask.unsqueeze(0)
        raw_tokens = raw_tokens.long()
        raw_mask = raw_mask.bool()
        valid_values = raw_tokens[raw_mask]
        if valid_values.numel() and (
            int(valid_values.min()) < 0 or int(valid_values.max()) >= self.fast_vocab_size
        ):
            raise ValueError(
                f"Raw FAST IDs must lie in [0,{self.fast_vocab_size}); got "
                f"[{int(valid_values.min())},{int(valid_values.max())}]."
            )

        bos = [int(self._tokenizer.bos_token_id)]
        prefix = bos + list(self._tokenizer.encode("Action: ", add_special_tokens=False))
        suffix = list(self._tokenizer.encode("|"))
        batch_size, max_raw = raw_tokens.shape
        max_total = len(prefix) + max_raw + len(suffix)
        tokens = torch.zeros(batch_size, max_total, dtype=torch.long, device=raw_tokens.device)
        masks = torch.zeros(batch_size, max_total, dtype=torch.bool, device=raw_tokens.device)
        prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=raw_tokens.device)
        suffix_tensor = torch.tensor(suffix, dtype=torch.long, device=raw_tokens.device)
        for batch_index in range(batch_size):
            count = int(raw_mask[batch_index].sum())
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

        complementary[ACTION_TOKENS] = tokens
        complementary[ACTION_TOKEN_MASK] = masks
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(self, features):
        return features


def make_skill_vla_pretrain_pre_post_processors(
    config: SkillVLAPretrainConfig,
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
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi0FastPrepareStateAndLanguageTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name=config.text_tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        SkillVLAPrecomputedFastTokensProcessorStep(
            paligemma_tokenizer_name=config.text_tokenizer_name,
            fast_skip_tokens=config.fast_skip_tokens,
            fast_vocab_size=config.fast_vocab_size,
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
