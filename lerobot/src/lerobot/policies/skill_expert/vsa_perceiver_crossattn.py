"""Stage-1 DINO-Perceiver vision conditioning for the Gemma action expert.

All modes retain every pretrained expert self-attention layer. Vision is fused
either by odd-layer residual cross-attention, as in-context visual tokens, or by
a pooled global condition supplied to the existing action AdaRMS modules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaMLP,
    GemmaRotaryEmbedding,
)


from lerobot.policies.pi05.modeling_pi05 import OPENPI_ATTENTION_MASK_VALUE
from lerobot.policies.pi_gemma import PiGemmaRMSNorm, _gated_residual

from .configuration_skill_expert import (
    GLOBAL_VISUAL_ADARMS,
    IN_CONTEXT_TOKENS,
    RESIDUAL_CROSS_ATTENTION,
    VISION_CONDITIONING_MODES,
)


VISUAL_RESIDUAL_GATE_INIT = 0.1


def make_expert_config():
    """Return the fixed gemma_300m geometry used by Stage 1."""
    config = CONFIG_MAPPING["gemma"](
        head_dim=256,
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        vocab_size=257152,
        hidden_activation="gelu_pytorch_tanh",
        dtype="float32",
        attention_bias=False,
    )
    config._attn_implementation = "eager"  # noqa: SLF001
    return config


class PerceiverBlock(nn.Module):
    """Pre-LN latent-to-image cross-attention followed by a Pre-LN FFN."""

    def __init__(self, width: int = 384, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.latent_norm = nn.LayerNorm(width)
        self.image_norm = nn.LayerNorm(width)
        self.cross_attn = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
            nn.Dropout(dropout),
        )

    def forward(self, latents: Tensor, image_tokens: Tensor) -> Tensor:
        normalized_image = self.image_norm(image_tokens)
        attended, _ = self.cross_attn(
            self.latent_norm(latents),
            normalized_image,
            normalized_image,
            need_weights=False,
        )
        latents = latents + attended
        return latents + self.ffn(self.ffn_norm(latents))


class CameraPerceiverResampler(nn.Module):
    """Compress one camera's 197 DINO tokens into configurable visual latents."""

    def __init__(
        self,
        dino_width: int,
        expert_width: int = 1024,
        perceiver_width: int = 384,
        num_latents: int = 32,
    ):
        super().__init__()
        self.input_proj = nn.Linear(dino_width, perceiver_width)
        self.input_norm = nn.LayerNorm(perceiver_width)
        self.latents = nn.Parameter(torch.empty(1, num_latents, perceiver_width))
        nn.init.normal_(self.latents, std=perceiver_width**-0.5)
        self.blocks = nn.ModuleList([PerceiverBlock(perceiver_width) for _ in range(2)])
        self.output_proj = nn.Linear(perceiver_width, expert_width)
        self.output_norm = nn.LayerNorm(expert_width)

    def forward(self, image_tokens: Tensor) -> Tensor:
        image_tokens = self.input_norm(self.input_proj(image_tokens))
        latents = self.latents.expand(image_tokens.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, image_tokens)
        return self.output_norm(self.output_proj(latents))


class GemmaCrossAttention(nn.Module):
    """Gemma-300M GQA cross-attention without positional encoding on visual KV."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        width = int(config.hidden_size)
        use_bias = bool(config.attention_bias)
        self.q_proj = nn.Linear(width, self.num_heads * self.head_dim, bias=use_bias)
        self.k_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.v_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, width, bias=use_bias)
        # Runtime-only diagnostics. These attributes add no parameters or
        # state_dict entries and are enabled only on explicitly selected steps.
        self.debug_enabled = False
        self.last_debug_stats: dict[str, float] = {}

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        batch, query_tokens = query.shape[:2]
        memory_tokens = memory.shape[1]
        q = self.q_proj(query).view(
            batch, query_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(memory).view(
            batch, memory_tokens, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(memory).view(
            batch, memory_tokens, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        if self.num_kv_groups != 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        if self.debug_enabled:
            probabilities = weights.detach().float().clamp_min(1e-12)
            entropy = -(probabilities * probabilities.log()).sum(dim=-1)
            normalizer = max(float(torch.log(torch.tensor(memory_tokens)).item()), 1e-12)
            self.last_debug_stats = {
                "attention/entropy": float(entropy.mean().item()),
                "attention/normalized_entropy": float((entropy / normalizer).mean().item()),
                "attention/effective_memory_tokens": float(entropy.exp().mean().item()),
                "attention/max_probability": float(probabilities.amax(dim=-1).mean().item()),
            }
        else:
            self.last_debug_stats = {}
        output = torch.matmul(weights, v).transpose(1, 2).contiguous()
        return self.o_proj(output.view(batch, query_tokens, -1))


class TokenSpecificNorm(nn.Module):
    """Regular context RMSNorm and time-conditioned action AdaRMS."""

    def __init__(self, width: int, eps: float):
        super().__init__()
        self.context_norm = PiGemmaRMSNorm(width, eps=eps)
        self.action_norm = PiGemmaRMSNorm(width, eps=eps, cond_dim=width)

    def forward(
        self, context: Tensor, actions: Tensor, time_condition: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        context, _ = self.context_norm(context)
        actions, action_gate = self.action_norm(actions, cond=time_condition)
        return context, actions, action_gate


class ActionOnlyNorm(nn.Module):
    """Time-conditioned action norm for visual cross-attention queries."""

    def __init__(self, width: int, eps: float):
        super().__init__()
        self.action_norm = PiGemmaRMSNorm(width, eps=eps, cond_dim=width)

    def forward(
        self, actions: Tensor, time_condition: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        return self.action_norm(actions, cond=time_condition)


def _residual_by_token(
    context: Tensor,
    actions: Tensor,
    update: Tensor,
    action_gate: Tensor | None,
) -> tuple[Tensor, Tensor]:
    context_update, action_update = update.split((context.shape[1], actions.shape[1]), dim=1)
    context = context + context_update
    actions = _gated_residual(actions, action_update, action_gate)
    return context, actions


class ResidualVisualExpertBlock(nn.Module):
    """Pretrained self-attention, optional residual visual attention, then FFN."""

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
        self.layer_index = layer_index
        self.include_state_in_visual_crossattn = include_state_in_visual_crossattn
        self.include_skill_in_visual_crossattn = include_skill_in_visual_crossattn
        self.debug_enabled = False
        self.last_debug_stats: dict[str, float] = {}
        # Every layer keeps its pi0.5 self-attention. Odd layers additionally
        # receive a fresh visual cross-attention residual.
        self.self_attention_norm = TokenSpecificNorm(width, eps)
        self.self_attention = GemmaAttention(config=config, layer_idx=layer_index)
        self.visual_attention_norm = (
            ActionOnlyNorm(width, eps) if cross_attention else None
        )
        self.visual_cross_attention = (
            GemmaCrossAttention(config) if cross_attention else None
        )
        self.visual_residual_gate = (
            nn.Parameter(torch.tensor(VISUAL_RESIDUAL_GATE_INIT))
            if cross_attention
            else None
        )
        self.ffn_norm = TokenSpecificNorm(width, eps)
        self.mlp = GemmaMLP(config)

    @staticmethod
    def _rms(tensor: Tensor) -> Tensor:
        return tensor.detach().float().square().mean().sqrt()

    def _record_cross_update(
        self,
        *,
        actions: Tensor,
        attended_actions: Tensor,
        action_gate: Tensor | None,
        visual_scale: Tensor,
        state: Tensor | None = None,
        attended_state: Tensor | None = None,
        skill: Tensor | None = None,
        attended_skill: Tensor | None = None,
    ) -> None:
        """Record raw/applied visual updates without changing the forward path."""
        if not self.debug_enabled:
            self.last_debug_stats = {}
            return
        scaled_action = attended_actions * visual_scale
        action_applied = scaled_action if action_gate is None else scaled_action * action_gate
        action_rms = self._rms(actions)
        raw_action_rms = self._rms(attended_actions)
        applied_action_rms = self._rms(action_applied)
        stats = {
            **self.visual_cross_attention.last_debug_stats,
            "residual_gate/parameter": float(self.visual_residual_gate.detach().float().item()),
            "residual_gate/tanh_scale": float(visual_scale.detach().float().item()),
            "action/residual_rms": float(action_rms.item()),
            "action/raw_update_rms": float(raw_action_rms.item()),
            "action/applied_update_rms": float(applied_action_rms.item()),
            "action/raw_update_ratio": float((raw_action_rms / action_rms.clamp_min(1e-12)).item()),
            "action/applied_update_ratio": float(
                (applied_action_rms / action_rms.clamp_min(1e-12)).item()
            ),
        }
        if action_gate is not None:
            gate = action_gate.detach().float()
            stats.update(
                {
                    "action/gate_rms": float(gate.square().mean().sqrt().item()),
                    "action/gate_abs_mean": float(gate.abs().mean().item()),
                }
            )
        for name, residual, update in (
            ("state", state, attended_state),
            ("skill", skill, attended_skill),
        ):
            if residual is None or update is None:
                continue
            residual_rms = self._rms(residual)
            raw_update_rms = self._rms(update)
            applied_update_rms = self._rms(update * visual_scale)
            stats.update(
                {
                    f"{name}/residual_rms": float(residual_rms.item()),
                    f"{name}/raw_update_rms": float(raw_update_rms.item()),
                    f"{name}/applied_update_rms": float(applied_update_rms.item()),
                    f"{name}/applied_update_ratio": float(
                        (applied_update_rms / residual_rms.clamp_min(1e-12)).item()
                    ),
                }
            )
        self.last_debug_stats = stats

    def forward(
        self,
        context: Tensor,
        actions: Tensor,
        visual_memory: Tensor,
        time_condition: Tensor,
        self_attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        self.last_debug_stats = {}
        normalized_context, normalized_actions, action_gate = self.self_attention_norm(
            context, actions, time_condition
        )
        normalized = torch.cat((normalized_context, normalized_actions), dim=1)
        attended, _ = self.self_attention(
            normalized,
            attention_mask=self_attention_mask,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        context, actions = _residual_by_token(
            context, actions, attended, action_gate
        )

        if self.cross_attention:
            if (
                self.visual_attention_norm is None
                or self.visual_cross_attention is None
                or self.visual_residual_gate is None
            ):
                raise RuntimeError("Visual cross-attention layer is incomplete.")
            self.visual_cross_attention.debug_enabled = self.debug_enabled
            normalized_actions, action_gate = self.visual_attention_norm(
                actions, time_condition
            )
            visual_scale = torch.tanh(self.visual_residual_gate).to(actions.dtype)
            if (
                self.include_state_in_visual_crossattn
                or self.include_skill_in_visual_crossattn
            ):
                state, skill = context.split((1, 1), dim=1)
                context_queries = []
                if self.include_state_in_visual_crossattn:
                    context_queries.append(state)
                if self.include_skill_in_visual_crossattn:
                    context_queries.append(skill)
                visual_queries = torch.cat((*context_queries, normalized_actions), dim=1)
                visual_update = self.visual_cross_attention(visual_queries, visual_memory)
                attended_context, attended_actions = visual_update.split(
                    (len(context_queries), actions.shape[1]), dim=1
                )
                update_index = 0
                attended_state = None
                attended_skill = None
                if self.include_state_in_visual_crossattn:
                    attended_state = attended_context[:, update_index : update_index + 1]
                    state_before = state
                    state = state + attended_state * visual_scale
                    update_index += 1
                if self.include_skill_in_visual_crossattn:
                    attended_skill = attended_context[:, update_index : update_index + 1]
                    skill_before = skill
                    skill = skill + attended_skill * visual_scale
                self._record_cross_update(
                    actions=actions,
                    attended_actions=attended_actions,
                    action_gate=action_gate,
                    visual_scale=visual_scale,
                    state=state_before if self.include_state_in_visual_crossattn else None,
                    attended_state=attended_state,
                    skill=skill_before if self.include_skill_in_visual_crossattn else None,
                    attended_skill=attended_skill,
                )
                context = torch.cat((state, skill), dim=1)
                actions = _gated_residual(
                    actions, attended_actions * visual_scale, action_gate
                )
            else:
                attended_actions = self.visual_cross_attention(
                    normalized_actions, visual_memory
                )
                self._record_cross_update(
                    actions=actions,
                    attended_actions=attended_actions,
                    action_gate=action_gate,
                    visual_scale=visual_scale,
                )
                actions = _gated_residual(
                    actions, attended_actions * visual_scale, action_gate
                )

        normalized_context, normalized_actions, action_gate = self.ffn_norm(
            context, actions, time_condition
        )
        transformed = self.mlp(torch.cat((normalized_context, normalized_actions), dim=1))
        return _residual_by_token(context, actions, transformed, action_gate)


class VSAActionExpert(nn.Module):
    """18 pretrained self-attention layers with one selected vision fusion path."""

    def __init__(
        self,
        config=None,
        *,
        vision_conditioning_mode: str = RESIDUAL_CROSS_ATTENTION,
        include_state_in_visual_crossattn: bool = False,
        include_skill_in_visual_crossattn: bool = False,
    ):
        super().__init__()
        # ``config`` is an internal test seam; production always uses the fixed
        # geometry returned by ``make_expert_config``.
        self.config = make_expert_config() if config is None else config
        if vision_conditioning_mode not in VISION_CONDITIONING_MODES:
            raise ValueError(
                "Unsupported vision_conditioning_mode="
                f"{vision_conditioning_mode!r}; expected one of {VISION_CONDITIONING_MODES}."
            )
        self.vision_conditioning_mode = vision_conditioning_mode
        self.rotary_emb = GemmaRotaryEmbedding(self.config)
        self.blocks = nn.ModuleList(
            ResidualVisualExpertBlock(
                self.config,
                index,
                cross_attention=(
                    vision_conditioning_mode == RESIDUAL_CROSS_ATTENTION
                    and bool(index % 2)
                ),
                include_state_in_visual_crossattn=include_state_in_visual_crossattn,
                include_skill_in_visual_crossattn=include_skill_in_visual_crossattn,
            )
            for index in range(self.config.num_hidden_layers)
        )
        initializer_range = float(self.config.initializer_range)
        for block in self.blocks:
            modules = [block.self_attention, block.mlp]
            if block.visual_cross_attention is not None:
                modules.append(block.visual_cross_attention)
            for module in modules:
                for submodule in module.modules():
                    if isinstance(submodule, nn.Linear):
                        nn.init.normal_(submodule.weight, std=initializer_range)
                        if submodule.bias is not None:
                            nn.init.zeros_(submodule.bias)
        self.final_norm = PiGemmaRMSNorm(
            int(self.config.hidden_size),
            eps=float(self.config.rms_norm_eps),
            cond_dim=int(self.config.hidden_size),
        )
        self.gradient_checkpointing = False
        self.debug_enabled = False
        self.last_debug_stats: dict[str, float] = {}
        # Parameter-free runtime observability used by architecture sanity tests.
        self.last_sequence_length = 0
        self.last_position_ids: Tensor | None = None

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    @staticmethod
    def self_attention_mask(batch_size: int, action_tokens: int, device) -> Tensor:
        """State/skill cannot read actions; action queries can read every token."""
        total = action_tokens + 2
        allowed = torch.ones(total, total, dtype=torch.bool, device=device)
        allowed[:2, 2:] = False
        additive = torch.where(allowed, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        return additive[None, None].expand(batch_size, 1, total, total)

    @staticmethod
    def in_context_attention_mask(
        batch_size: int,
        visual_tokens: int,
        action_tokens: int,
        device,
    ) -> Tensor:
        """Block mask for [visual, state, skill, actions] in-context fusion.

        Visual queries read visual keys only. State/skill queries read visual
        and both condition tokens. Action queries read the complete sequence.
        Every allowed block is bidirectional.
        """
        context_tokens = visual_tokens + 2
        total = context_tokens + action_tokens
        allowed = torch.zeros(total, total, dtype=torch.bool, device=device)
        allowed[:visual_tokens, :visual_tokens] = True
        allowed[visual_tokens:context_tokens, :context_tokens] = True
        allowed[context_tokens:, :] = True
        additive = torch.where(allowed, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        return additive[None, None].expand(batch_size, 1, total, total)

    @staticmethod
    def position_ids_from_valid_mask(valid_mask: Tensor) -> Tensor:
        """Create continuous position IDs without deriving them from 2-D attention."""
        return (valid_mask.long().cumsum(dim=-1) - 1).clamp_min(0)

    def forward(
        self,
        context: Tensor,
        actions: Tensor,
        visual_memory: Tensor,
        time_condition: Tensor,
    ) -> Tensor:
        self.last_debug_stats = {}
        batch_size = actions.shape[0]
        if context.shape[1] != 2:
            raise ValueError(
                "Expert requires [state, skill] context, got "
                f"{context.shape[1]} tokens."
            )
        if self.vision_conditioning_mode == IN_CONTEXT_TOKENS:
            context = torch.cat((visual_memory, context), dim=1)
            attention_mask = self.in_context_attention_mask(
                batch_size,
                visual_memory.shape[1],
                actions.shape[1],
                actions.device,
            )
        else:
            attention_mask = self.self_attention_mask(
                batch_size, actions.shape[1], actions.device
            )
        total = context.shape[1] + actions.shape[1]
        valid_mask = torch.ones(
            batch_size, total, dtype=torch.bool, device=actions.device
        )
        position_ids = self.position_ids_from_valid_mask(valid_mask)
        self.last_sequence_length = total
        self.last_position_ids = position_ids.detach()
        position_embeddings = self.rotary_emb(
            torch.cat((context, actions), dim=1), position_ids
        )
        for layer_index, block in enumerate(self.blocks):
            block.debug_enabled = self.debug_enabled and block.cross_attention
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
            if self.debug_enabled and block.cross_attention:
                self.last_debug_stats.update(
                    {
                        f"cross_layer_{layer_index:02d}/{name}": value
                        for name, value in block.last_debug_stats.items()
                    }
                )
        actions, _ = self.final_norm(actions, cond=time_condition)
        if self.debug_enabled:
            final = actions.detach().float()
            self.last_debug_stats["expert/final_action_rms"] = float(
                final.square().mean().sqrt().item()
            )
        return actions
