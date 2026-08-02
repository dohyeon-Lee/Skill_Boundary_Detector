"""Evaluation-only implementation for pre-residual-SA18 VSA checkpoints.

This module exists solely so Stage-1 evaluation on the skillVLA_VSA branch can
read checkpoints produced by the first Perceiver/cross-attention architecture.
Training never selects this class, and no missing tensors are partially loaded.
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaMLP,
    GemmaRotaryEmbedding,
)

from lerobot.policies.pi05.modeling_pi05 import OPENPI_ATTENTION_MASK_VALUE
from lerobot.policies.pi_gemma import PiGemmaRMSNorm, _gated_residual

from .vsa_perceiver_crossattn import (
    ActionOnlyNorm,
    GemmaCrossAttention,
    TokenSpecificNorm,
    _residual_by_token,
    make_expert_config,
)


LEGACY_VISUAL_LATENTS_PER_CAMERA = 8


class LegacyAlternatingExpertBlock(nn.Module):
    """Original even-self/odd-visual block with checkpoint-compatible names."""

    def __init__(
        self,
        config,
        layer_index: int,
        *,
        cross_attention: bool,
        include_state_in_visual_crossattn: bool = False,
        include_skill_in_visual_crossattn: bool = False,
    ):
        super().__init__()
        width = int(config.hidden_size)
        eps = float(config.rms_norm_eps)
        self.cross_attention = cross_attention
        self.include_state_in_visual_crossattn = include_state_in_visual_crossattn
        self.include_skill_in_visual_crossattn = include_skill_in_visual_crossattn
        self.attention_norm = (
            ActionOnlyNorm(width, eps)
            if cross_attention
            else TokenSpecificNorm(width, eps)
        )
        self.attention = (
            GemmaCrossAttention(config)
            if cross_attention
            else GemmaAttention(config=config, layer_idx=layer_index)
        )
        self.ffn_norm = TokenSpecificNorm(width, eps)
        self.mlp = GemmaMLP(config)

    def _visual_attention(
        self,
        context: Tensor,
        actions: Tensor,
        visual_memory: Tensor,
        time_condition: Tensor,
    ) -> tuple[Tensor, Tensor]:
        normalized_actions, action_gate = self.attention_norm(
            actions, time_condition
        )
        if not (
            self.include_state_in_visual_crossattn
            or self.include_skill_in_visual_crossattn
        ):
            attended_actions = self.attention(normalized_actions, visual_memory)
            return context, _gated_residual(actions, attended_actions, action_gate)

        state, skill = context.split((1, 1), dim=1)
        queries = []
        if self.include_state_in_visual_crossattn:
            queries.append(state)
        if self.include_skill_in_visual_crossattn:
            queries.append(skill)
        attended = self.attention(
            torch.cat((*queries, normalized_actions), dim=1), visual_memory
        )
        attended_context, attended_actions = attended.split(
            (len(queries), actions.shape[1]), dim=1
        )
        update_index = 0
        if self.include_state_in_visual_crossattn:
            state = state + attended_context[:, update_index : update_index + 1]
            update_index += 1
        if self.include_skill_in_visual_crossattn:
            skill = skill + attended_context[:, update_index : update_index + 1]
        context = torch.cat((state, skill), dim=1)
        actions = _gated_residual(actions, attended_actions, action_gate)
        return context, actions

    def forward(
        self,
        context: Tensor,
        actions: Tensor,
        visual_memory: Tensor,
        time_condition: Tensor,
        self_attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        if self.cross_attention:
            context, actions = self._visual_attention(
                context, actions, visual_memory, time_condition
            )
        else:
            normalized_context, normalized_actions, action_gate = self.attention_norm(
                context, actions, time_condition
            )
            attended, _ = self.attention(
                torch.cat((normalized_context, normalized_actions), dim=1),
                attention_mask=self_attention_mask,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            context, actions = _residual_by_token(
                context, actions, attended, action_gate
            )

        normalized_context, normalized_actions, action_gate = self.ffn_norm(
            context, actions, time_condition
        )
        transformed = self.mlp(
            torch.cat((normalized_context, normalized_actions), dim=1)
        )
        return _residual_by_token(context, actions, transformed, action_gate)


class LegacyVSAActionExpert(nn.Module):
    """Exact original 18-layer alternating VSA expert, for evaluation only."""

    def __init__(
        self,
        config=None,
        *,
        include_state_in_visual_crossattn: bool = False,
        include_skill_in_visual_crossattn: bool = False,
    ):
        super().__init__()
        self.config = make_expert_config() if config is None else config
        self.rotary_emb = GemmaRotaryEmbedding(self.config)
        self.blocks = nn.ModuleList(
            LegacyAlternatingExpertBlock(
                self.config,
                index,
                cross_attention=bool(index % 2),
                include_state_in_visual_crossattn=include_state_in_visual_crossattn,
                include_skill_in_visual_crossattn=include_skill_in_visual_crossattn,
            )
            for index in range(self.config.num_hidden_layers)
        )
        self.final_norm = PiGemmaRMSNorm(
            int(self.config.hidden_size),
            eps=float(self.config.rms_norm_eps),
            cond_dim=int(self.config.hidden_size),
        )
        self.gradient_checkpointing = False
        self.debug_enabled = False
        self.last_debug_stats: dict[str, float] = {}

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    @staticmethod
    def self_attention_mask(batch_size: int, action_tokens: int, device) -> Tensor:
        total = action_tokens + 2
        allowed = torch.ones(total, total, dtype=torch.bool, device=device)
        allowed[:2, 2:] = False
        additive = torch.where(allowed, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        return additive[None, None].expand(batch_size, 1, total, total)

    def forward(
        self,
        context: Tensor,
        actions: Tensor,
        visual_memory: Tensor,
        time_condition: Tensor,
    ) -> Tensor:
        if context.shape[1] != 2:
            raise ValueError(
                f"Legacy expert requires [state, skill], got {context.shape[1]} tokens."
            )
        batch_size = actions.shape[0]
        total = context.shape[1] + actions.shape[1]
        position_ids = torch.arange(total, device=actions.device, dtype=torch.long)
        position_ids = position_ids[None].expand(batch_size, -1)
        position_embeddings = self.rotary_emb(
            torch.cat((context, actions), dim=1), position_ids
        )
        attention_mask = self.self_attention_mask(
            batch_size, actions.shape[1], actions.device
        )
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                context, actions = torch.utils.checkpoint.checkpoint(
                    block,
                    context,
                    actions,
                    visual_memory,
                    time_condition,
                    attention_mask,
                    position_embeddings,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                context, actions = block(
                    context,
                    actions,
                    visual_memory,
                    time_condition,
                    attention_mask,
                    position_embeddings,
                )
        actions, _ = self.final_norm(actions, cond=time_condition)
        return actions
