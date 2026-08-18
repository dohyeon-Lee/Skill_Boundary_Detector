"""Selectable likelihood or FRS/DSBC Stage 2 on a frozen Cond-Gemma VSA.

Stage 2 assembles two independently trained frozen pieces:

* ``stage1_checkpoint_path``: a cond_gemma (Arch0-family) Stage-1 VSA prior.
  It carries no predictor or terminator of its own.
* ``skill_predictor_checkpoint_path``: any Stage-1/skill_aux checkpoint whose
  frozen VLM (and trained reader/head) is loaded completely into the predictor
  module built from that checkpoint's own architecture fields.

Likelihood mode trains the four extra blocks, VLM projection, and warm-started
action head. DSBC instead freezes the complete Stage-1 VSA and trains the same
extra path with a fresh initial-noise head. Terminators are attached externally
at evaluation time and are never part of Stage-2 training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import (
    OPENPI_ATTENTION_MASK_VALUE,
    make_att_2d_masks,
    pad_vector,
)
from lerobot.policies.pi_gemma import (
    GemmaAttention,
    GemmaMLP,
    PiGemmaRMSNorm,
    _gated_residual,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.skill_expert.cond_gemma import CondGemmaSkillExpert
from lerobot.policies.skill_expert.configuration_skill_expert import (
    COND_GEMMA_ARCHITECTURE,
    COND_GEMMA_ARCHITECTURE_REVISION,
    SKILLLESS_CONDITIONING_ROUTES,
    STATELESS_CONDITIONING_ROUTES,
    normalize_conditioning_route,
)
from lerobot.policies.skill_expert.modeling_skill_expert import (
    _PREDICTOR_CHECKPOINT_CONTRACT_FIELDS,
    SkillExpertPolicy,
    _load_complete_predictor_parameters,
    _load_pretrained_state_dict,
)
from lerobot.policies.skillVLA.dataset_skillVLA import (
    SAME_SKILL_PAIR_FALLBACK,
    SAME_SKILL_PAIR_ID,
    SKILL_PROGRESS,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    STAGE2_VLM_CACHE_ID,
)

from .configuration_skill_vla_stage2 import SkillVLAStage2Config

log = logging.getLogger(__name__)


class GemmaCrossAttention(nn.Module):
    """Cross-attention with gemma_300m's 8-query/1-KV-head geometry."""

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

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
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
        scores = scores.masked_fill(
            key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous()
        return self.o_proj(output.view(batch, query_tokens, -1))


def _identity_adarms(norm: PiGemmaRMSNorm) -> PiGemmaRMSNorm:
    """Make a fresh gated residual branch exactly zero at initialization."""
    if norm.dense is None:
        raise ValueError("Stage-2 likelihood norms must be adaptive.")
    nn.init.zeros_(norm.dense.weight)
    nn.init.zeros_(norm.dense.bias)
    return norm


class LikelihoodBlock(nn.Module):
    """Action self-attention -> frozen-VLM cross-attention -> Gemma FFN."""

    def __init__(self, config, layer_index: int):
        super().__init__()
        width = int(config.hidden_size)
        eps = float(config.rms_norm_eps)
        self.self_norm = _identity_adarms(PiGemmaRMSNorm(width, eps=eps, cond_dim=width))
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_index)
        self.cross_norm = _identity_adarms(PiGemmaRMSNorm(width, eps=eps, cond_dim=width))
        self.cross_attn = GemmaCrossAttention(config)
        self.ffn_norm = _identity_adarms(PiGemmaRMSNorm(width, eps=eps, cond_dim=width))
        self.mlp = GemmaMLP(config)

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        expert_condition: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        gate_rms: dict[str, Tensor] = {}

        residual = hidden
        normalized, gate = self.self_norm(hidden, cond=expert_condition)
        gate_rms["self"] = gate.detach().float().square().mean().sqrt()
        attended, _ = self.self_attn(
            normalized,
            attention_mask=None,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        hidden = _gated_residual(residual, attended, gate)

        residual = hidden
        normalized, gate = self.cross_norm(hidden, cond=expert_condition)
        gate_rms["cross"] = gate.detach().float().square().mean().sqrt()
        attended = self.cross_attn(normalized, memory, memory_key_padding_mask)
        hidden = _gated_residual(residual, attended, gate)

        residual = hidden
        normalized, gate = self.ffn_norm(hidden, cond=expert_condition)
        gate_rms["ffn"] = gate.detach().float().square().mean().sqrt()
        transformed = self.mlp(normalized)
        # Whether the language pathway is actually used is invisible in the
        # loss; expose the gate magnitudes so training logs can answer it.
        self._last_gate_rms = gate_rms
        return _gated_residual(residual, transformed, gate)


class SkillVLAStage2Pytorch(CondGemmaSkillExpert):
    """Frozen Stage-1 VSA plus four language-conditioned Stage-2 blocks."""

    def __init__(self, config: SkillVLAStage2Config):
        super().__init__(config)
        if self.skill_predictor is None:
            raise RuntimeError("Stage 2 requires the frozen VLM predictor module.")
        if self.fsq_term_train is not None:
            raise RuntimeError(
                "Stage 2 trains without a terminator; attach one at evaluation time."
            )
        expert_config = self.gemma_expert.model.config
        vlm_width = int(self.skill_predictor.vlm.language_model.config.hidden_size)
        self.vlm_to_expert_projection = nn.Linear(vlm_width, self.width)
        first_index = int(expert_config.num_hidden_layers)
        self.likelihood_blocks = nn.ModuleList(
            LikelihoodBlock(expert_config, first_index + index)
            for index in range(config.likelihood_num_layers)
        )
        action_feature = (config.output_features or {}).get(ACTION)
        self.real_action_dim = (
            int(action_feature.shape[0])
            if action_feature is not None
            else int(config.max_action_dim)
        )
        self.noise_out_proj = None
        if config.stage2_mode == "dsbc":
            self.noise_out_proj = nn.Linear(self.width, self.real_action_dim)
            anchor_generator = torch.Generator(device="cpu")
            anchor_generator.manual_seed(config.dsbc_anchor_seed)
            self.register_buffer(
                "dsbc_anchor_noise",
                torch.randn(
                    1,
                    config.chunk_size,
                    config.max_action_dim,
                    generator=anchor_generator,
                    dtype=torch.float32,
                ),
            )
        self.likelihood_layer_mix = None
        if config.likelihood_vlm_memory == "layer_mix":
            vlm_layers = int(
                self.skill_predictor.vlm.language_model.config.num_hidden_layers
            )
            # Biased toward the final layer so training starts from the
            # known-working last-hidden memory and only moves depth on demand.
            mix = torch.zeros(config.likelihood_num_layers, vlm_layers)
            mix[:, -1] = 5.0
            self.likelihood_layer_mix = nn.Parameter(mix)
        self._likelihood_gradient_checkpointing = False
        self._freeze_stage1_prior()

    def gradient_checkpointing_enable(self) -> None:
        # The frozen 18-layer prior and the VLM run under no_grad; checkpointing
        # helps only the four trainable likelihood blocks.
        self._likelihood_gradient_checkpointing = True

    def _freeze_stage1_prior(self) -> None:
        self.requires_grad_(False)
        self.vlm_to_expert_projection.requires_grad_(True)
        self.likelihood_blocks.requires_grad_(True)
        if self.config.stage2_mode == "likelihood":
            self.action_out_proj.requires_grad_(True)
        else:
            if self.noise_out_proj is None:
                raise RuntimeError("DSBC mode has no noise output head.")
            self.noise_out_proj.requires_grad_(True)
        if self.likelihood_layer_mix is not None:
            self.likelihood_layer_mix.requires_grad_(True)

    def train(self, mode: bool = True):
        # Stage 2 must not change stochastic behavior or running state anywhere
        # in the frozen prior or VLM; only fresh likelihood modules follow mode.
        nn.Module.train(self, mode)
        trainable_ids = {
            id(self.vlm_to_expert_projection),
            id(self.likelihood_blocks),
        }
        trainable_head = (
            self.action_out_proj
            if self.config.stage2_mode == "likelihood"
            else self.noise_out_proj
        )
        if trainable_head is not None:
            trainable_ids.add(id(trainable_head))
        for child in self.children():
            if id(child) not in trainable_ids:
                child.eval()
        return self

    def _prior_action_hidden(
        self,
        condition_tokens: Tensor,
        noisy_actions: Tensor,
        state: Tensor | None,
        skill_code: Tensor | None,
        time: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return (18-layer action hidden, expert AdaRMS condition) for any revision."""
        if self.uses_expert_context_tokens:
            expert_condition = self._expert_condition(time)
            hidden = self._run_expert_token_hidden(
                condition_tokens,
                self._expert_context_tokens(state, skill_code),
                noisy_actions,
                expert_condition,
                self._state_condition(state),
            )
            return hidden, expert_condition
        projected_state = self._project_state(state)
        expert_projected_state = self._project_expert_state(state, projected_state)
        condition_state = projected_state if self.uses_cond_state_adarms else None
        expert_condition = self._expert_condition(
            time, expert_projected_state, skill_code
        )
        condition_skill, expert_skill = self._skill_broadcasts(skill_code)
        hidden = self._run_joint_hidden(
            condition_tokens,
            noisy_actions,
            condition_state,
            expert_condition,
            condition_skill,
            expert_skill,
            self._condition_state_start_index(condition_tokens),
        )
        return hidden, expert_condition

    def _encode_likelihood_memory(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the frozen VLM memory in the configured layout."""
        if self.likelihood_layer_mix is None:
            return self.skill_predictor.encode_base_last_hidden(
                images, language_tokens, language_mask
            )
        return self.skill_predictor.encode_base_hidden_stack(
            images, language_tokens, language_mask
        )

    def _likelihood_memories(self, vlm_hidden: Tensor) -> list[Tensor]:
        """Project the VLM memory once per block; layer mixing happens first.

        The memory is timestep-independent, so callers compute this once per
        batch (or per rollout) and reuse it across every flow step.
        """
        projection = self.vlm_to_expert_projection
        if self.likelihood_layer_mix is None:
            memory = projection(vlm_hidden.to(projection.weight.dtype))
            return [memory] * len(self.likelihood_blocks)
        stack = vlm_hidden.to(projection.weight.dtype)
        weights = torch.softmax(self.likelihood_layer_mix.float(), dim=-1).to(
            stack.dtype
        )
        return [
            projection(torch.einsum("l,blnd->bnd", weights[index], stack))
            for index in range(len(self.likelihood_blocks))
        ]

    def encode_likelihood_memories(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> tuple[list[Tensor], Tensor]:
        """Encode and project the frozen VLM condition for inference reuse."""
        vlm_hidden, vlm_key_padding_mask = self._encode_likelihood_memory(
            images, language_tokens, language_mask
        )
        return self._likelihood_memories(vlm_hidden), vlm_key_padding_mask

    def _run_likelihood_blocks(
        self,
        prior_hidden: Tensor,
        memories: list[Tensor],
        vlm_key_padding_mask: Tensor,
        expert_condition: Tensor,
    ) -> Tensor:
        hidden = prior_hidden.to(self.working_dtype)
        position_ids = torch.arange(
            hidden.shape[1], device=hidden.device, dtype=torch.long
        )[None].expand(hidden.shape[0], -1)
        position_embeddings = self.gemma_expert.model.rotary_emb(hidden, position_ids)
        use_checkpoint = self._likelihood_gradient_checkpointing and self.training
        for block, memory in zip(self.likelihood_blocks, memories, strict=True):
            if use_checkpoint:
                hidden = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden,
                    memory,
                    vlm_key_padding_mask,
                    expert_condition,
                    position_embeddings,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden = block(
                    hidden,
                    memory,
                    vlm_key_padding_mask,
                    expert_condition,
                    position_embeddings,
                )
        return hidden

    def _likelihood_velocity(
        self,
        prior_hidden: Tensor,
        memories: list[Tensor],
        vlm_key_padding_mask: Tensor,
        expert_condition: Tensor,
    ) -> Tensor:
        hidden = self._run_likelihood_blocks(
            prior_hidden,
            memories,
            vlm_key_padding_mask,
            expert_condition,
        )
        return self.action_out_proj(hidden.to(self.working_dtype)).float()

    def _dsbc_noise_prediction(
        self,
        images: list[Tensor],
        vlm_start_images: list[Tensor],
        state: Tensor | None,
        skill_code: Tensor | None,
        language_tokens: Tensor,
        language_mask: Tensor,
        *,
        condition_tokens: Tensor | None = None,
        vlm_memory: tuple[list[Tensor], Tensor] | None = None,
    ) -> Tensor:
        """Predict the real-action part of the initial VSA noise."""
        if self.config.stage2_mode != "dsbc" or self.noise_out_proj is None:
            raise RuntimeError("DSBC noise prediction requires stage2_mode='dsbc'.")
        if state is not None:
            batch_size = state.shape[0]
        elif skill_code is not None:
            batch_size = skill_code.shape[0]
        elif images:
            batch_size = images[0].shape[0]
        else:
            raise ValueError("DSBC prediction requires state, skill, or image metadata.")

        selector_time = torch.ones(
            batch_size, dtype=torch.float32, device=language_tokens.device
        )
        anchor = self.dsbc_anchor_noise.expand(batch_size, -1, -1)
        with torch.no_grad():
            if condition_tokens is None:
                condition_tokens = self._condition_tokens(
                    images, batch_size=batch_size
                )
            prior_hidden, expert_condition = self._prior_action_hidden(
                condition_tokens, anchor, state, skill_code, selector_time
            )
            if vlm_memory is None:
                vlm_hidden, vlm_key_padding_mask = self._encode_likelihood_memory(
                    vlm_start_images, language_tokens, language_mask
                )
            else:
                vlm_hidden = None
                _, vlm_key_padding_mask = vlm_memory
        memories = (
            self._likelihood_memories(vlm_hidden)
            if vlm_memory is None
            else vlm_memory[0]
        )
        hidden = self._run_likelihood_blocks(
            prior_hidden,
            memories,
            vlm_key_padding_mask,
            expert_condition,
        )
        if self.config.dsbc_noise_output_mode == "shared":
            hidden = hidden.mean(dim=1)
        raw_prediction = self.noise_out_proj(
            hidden.to(self.working_dtype)
        ).float()
        noise_bound = float(
            getattr(self.config, "dsbc_noise_output_bound", 5.0)
        )
        return noise_bound * torch.tanh(raw_prediction)

    @staticmethod
    def _frs_state(
        real_state: Tensor,
        padding_noise: Tensor,
        time_value: float,
    ) -> Tensor:
        if padding_noise.shape[-1] == 0:
            return real_state
        return torch.cat(
            (real_state, padding_noise * time_value),
            dim=-1,
        )

    @torch.no_grad()
    def _frs_reverse_with_expert_context_cache(
        self,
        condition_tokens: Tensor,
        context_tokens: Tensor,
        actions: Tensor,
        padding_noise: Tensor,
        num_steps: int,
        condition_state: Tensor | None,
    ) -> Tensor:
        """Integrate the frozen Stage-1 VSA from action t=0 to noise t=1."""
        batch_size = actions.shape[0]
        n_prefix = condition_tokens.shape[1] + context_tokens.shape[1]
        n_action = actions.shape[1]
        device = actions.device
        prefix_cache = self._visual_context_cache(
            condition_tokens, context_tokens, condition_state
        )
        action_padding = torch.ones(
            batch_size, n_action, dtype=torch.bool, device=device
        )
        action_blocks = torch.tensor(
            [1] + [0] * (n_action - 1), dtype=torch.bool, device=device
        )[None].expand(batch_size, -1)
        action_attention = make_att_2d_masks(action_padding, action_blocks)
        prefix_visible = torch.ones(
            batch_size, n_action, n_prefix, dtype=torch.bool, device=device
        )
        full_attention = torch.cat((prefix_visible, action_attention), dim=2)[:, None]
        full_attention = torch.where(
            full_attention, 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        action_positions = n_prefix + torch.cumsum(action_padding, dim=1) - 1

        dt = 1.0 / num_steps
        real_state = actions[..., : self.real_action_dim].float()
        for step in range(num_steps):
            time_value = step * dt
            time = torch.full(
                (batch_size,), time_value, dtype=torch.float32, device=device
            )
            expert_condition = self._expert_condition(time)
            action_hidden = self._action_hidden_with_condition_cache(
                self._frs_state(real_state, padding_noise, time_value),
                expert_condition,
                None,
                prefix_cache,
                full_attention,
                action_positions,
            )
            velocity = self.action_out_proj(
                action_hidden.to(self.working_dtype)
            ).float()
            real_state = real_state + dt * velocity[..., : self.real_action_dim]
        return real_state

    @torch.no_grad()
    def _frs_reverse_with_condition_cache(
        self,
        condition_tokens: Tensor,
        actions: Tensor,
        state: Tensor | None,
        skill_code: Tensor | None,
        padding_noise: Tensor,
        num_steps: int,
    ) -> Tensor:
        """Action-to-noise integration for the standard Cond-Gemma route."""
        batch_size, n_condition = condition_tokens.shape[:2]
        n_chunk = actions.shape[1]
        device = actions.device
        projected_state = self._project_state(state)
        expert_projected_state = self._project_expert_state(
            state, projected_state
        )
        condition_state = projected_state if self.uses_cond_state_adarms else None
        condition_state_start_index = self._condition_state_start_index(
            condition_tokens
        )
        condition_skill, expert_skill = self._skill_broadcasts(skill_code)

        condition_padding = torch.ones(
            batch_size, n_condition, dtype=torch.bool, device=device
        )
        condition_blocks = torch.zeros_like(condition_padding)
        condition_attention = make_att_2d_masks(
            condition_padding, condition_blocks
        )[:, None]
        condition_attention = torch.where(
            condition_attention, 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        condition_positions = torch.cumsum(condition_padding, dim=1) - 1
        condition_cache = self.cond_encoder.model.forward(
            inputs_embeds=condition_tokens,
            attention_mask=condition_attention,
            position_ids=condition_positions,
            past_key_values=None,
            use_cache=True,
            adarms_cond=condition_state,
            adarms_start_index=condition_state_start_index,
            broadcast_cond=condition_skill,
        ).past_key_values

        action_padding = torch.ones(
            batch_size, n_chunk, dtype=torch.bool, device=device
        )
        action_blocks = torch.tensor(
            [1] + [0] * (n_chunk - 1), dtype=torch.bool, device=device
        )[None].expand(batch_size, -1)
        action_attention = make_att_2d_masks(action_padding, action_blocks)
        condition_visible = condition_padding[:, None].expand(
            batch_size, n_chunk, n_condition
        )
        full_attention = torch.cat((condition_visible, action_attention), dim=2)[:, None]
        full_attention = torch.where(
            full_attention, 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        action_positions = n_condition + torch.cumsum(action_padding, dim=1) - 1

        dt = 1.0 / num_steps
        real_state = actions[..., : self.real_action_dim].float()
        for step in range(num_steps):
            time_value = step * dt
            time = torch.full(
                (batch_size,), time_value, dtype=torch.float32, device=device
            )
            expert_condition = self._expert_condition(
                time, expert_projected_state, skill_code
            )
            action_hidden = self._action_hidden_with_condition_cache(
                self._frs_state(real_state, padding_noise, time_value),
                expert_condition,
                expert_skill,
                condition_cache,
                full_attention,
                action_positions,
            )
            velocity = self.action_out_proj(
                action_hidden.to(self.working_dtype)
            ).float()
            real_state = real_state + dt * velocity[..., : self.real_action_dim]
        return real_state

    @torch.no_grad()
    def _frs_target_noise(
        self,
        images: list[Tensor],
        state: Tensor | None,
        skill_code: Tensor | None,
        actions: Tensor,
        *,
        padding_noise: Tensor | None = None,
        condition_tokens: Tensor | None = None,
    ) -> Tensor:
        """Build an online FRS real-action noise target with the frozen VSA."""
        if self.config.stage2_mode != "dsbc":
            raise RuntimeError("FRS targets are defined only in DSBC mode.")
        batch_size, chunk_size = actions.shape[:2]
        padding_dim = self.config.max_action_dim - self.real_action_dim
        if padding_noise is None:
            padding_noise = self.sample_noise(
                (batch_size, chunk_size, padding_dim), actions.device
            )
        expected_shape = (batch_size, chunk_size, padding_dim)
        if tuple(padding_noise.shape) != expected_shape:
            raise ValueError(
                f"FRS padding noise must have shape {expected_shape}, got "
                f"{tuple(padding_noise.shape)}."
            )
        if condition_tokens is None:
            condition_tokens = self._condition_tokens(
                images, batch_size=batch_size
            )
        num_steps = self.config.dsbc_frs_num_steps
        if self.uses_expert_context_tokens:
            return self._frs_reverse_with_expert_context_cache(
                condition_tokens,
                self._expert_context_tokens(state, skill_code),
                actions,
                padding_noise,
                num_steps,
                self._state_condition(state),
            )
        return self._frs_reverse_with_condition_cache(
            condition_tokens,
            actions,
            state,
            skill_code,
            padding_noise,
            num_steps,
        )

    @torch.no_grad()
    def _dsbc_action_flow_residual(
        self,
        condition_tokens: Tensor,
        state: Tensor | None,
        skill_code: Tensor | None,
        actions: Tensor,
        predicted_noise: Tensor,
        padding_noise: Tensor,
    ) -> Tensor:
        """Evaluate DSBC with the same flow residual logged by legacy Stage 2.

        This is a detached comparison metric only. The optimization objective
        remains the FRS-target noise MSE returned by ``dsbc_training_pair``.
        """
        if self.config.dsbc_noise_output_mode == "shared":
            predicted_noise = predicted_noise[:, None].expand(
                -1, actions.shape[1], -1
            )
        source = torch.cat((predicted_noise.float(), padding_noise.float()), dim=-1)
        time = self.sample_time(actions.shape[0], actions.device)
        x_t = (
            time[:, None, None] * source
            + (1.0 - time[:, None, None]) * actions.float()
        )
        target_velocity = source - actions.float()
        prior_hidden, expert_condition = self._prior_action_hidden(
            condition_tokens,
            x_t,
            state,
            skill_code,
            time,
        )
        predicted_velocity = self.action_out_proj(
            prior_hidden.to(self.working_dtype)
        ).float()
        return (
            target_velocity[..., : self.real_action_dim]
            - predicted_velocity[..., : self.real_action_dim]
        )

    def dsbc_training_pair(
        self,
        images: list[Tensor],
        vlm_start_images: list[Tensor],
        state: Tensor | None,
        skill_code: Tensor | None,
        actions: Tensor,
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return prediction, FRS target, and detached legacy action residual."""
        batch_size, chunk_size = actions.shape[:2]
        padding_dim = self.config.max_action_dim - self.real_action_dim
        padding_noise = self.sample_noise(
            (batch_size, chunk_size, padding_dim), actions.device
        )
        with torch.no_grad():
            condition_tokens = self._condition_tokens(
                images, batch_size=batch_size
            )
        target = self._frs_target_noise(
            images,
            state,
            skill_code,
            actions,
            padding_noise=padding_noise,
            condition_tokens=condition_tokens,
        )
        prediction = self._dsbc_noise_prediction(
            images,
            vlm_start_images,
            state,
            skill_code,
            language_tokens,
            language_mask,
            condition_tokens=condition_tokens,
        )
        action_residual = self._dsbc_action_flow_residual(
            condition_tokens,
            state,
            skill_code,
            actions,
            prediction.detach(),
            padding_noise,
        )
        return prediction, target, action_residual

    def forward(
        self,
        images: list[Tensor],
        vlm_start_images: list[Tensor],
        state: Tensor | None,
        skill_code: Tensor | None,
        actions: Tensor,
        language_tokens: Tensor,
        language_mask: Tensor,
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        if getattr(self.config, "stage2_mode", "likelihood") != "likelihood":
            raise RuntimeError(
                "The flow-residual forward is available only in likelihood mode; "
                "use dsbc_training_pair in DSBC mode."
            )
        self._last_vsa_debug_stats = {}
        batch_size = actions.shape[0]
        time = self.sample_time(batch_size, actions.device) if time is None else time
        self._last_flow_time = time.detach()
        source = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        source = source.to(actions.dtype)
        x_t = time[:, None, None] * source + (1.0 - time[:, None, None]) * actions
        target_velocity = source - actions

        with torch.no_grad():
            condition_tokens = self._condition_tokens(images, batch_size=batch_size)
            prior_hidden, expert_condition = self._prior_action_hidden(
                condition_tokens, x_t, state, skill_code, time
            )
            vlm_hidden, vlm_key_padding_mask = self._encode_likelihood_memory(
                vlm_start_images, language_tokens, language_mask
            )
        predicted_velocity = self._likelihood_velocity(
            prior_hidden,
            self._likelihood_memories(vlm_hidden),
            vlm_key_padding_mask,
            expert_condition,
        )
        if self.config.cumulative_xyz_loss_enabled:
            # x_t = action + t * target_velocity, hence the one-step clean-action
            # reconstruction is action_hat = x_t - t * predicted_velocity.
            self._last_predicted_actions = (
                x_t - time[:, None, None] * predicted_velocity
            )
        else:
            self._last_predicted_actions = None
        return target_velocity - predicted_velocity

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        vlm_start_images: list[Tensor],
        state: Tensor | None,
        skill_code: Tensor | None,
        language_tokens: Tensor,
        language_mask: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        vlm_memory: tuple[list[Tensor], Tensor] | None = None,
    ) -> Tensor:
        if getattr(self.config, "stage2_mode", "likelihood") == "dsbc":
            return self._sample_dsbc_actions(
                images,
                vlm_start_images,
                state,
                skill_code,
                language_tokens,
                language_mask,
                noise=noise,
                num_steps=num_steps,
                vlm_memory=vlm_memory,
            )
        num_steps = self.config.num_inference_steps if num_steps is None else num_steps
        if state is not None:
            batch_size, device = state.shape[0], state.device
        elif skill_code is not None:
            batch_size, device = skill_code.shape[0], skill_code.device
        elif images:
            batch_size, device = images[0].shape[0], images[0].device
        else:
            raise ValueError("Action sampling requires state, skill, or image batch metadata.")
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim), device
            )
        condition_tokens = self._condition_tokens(images, batch_size=batch_size)
        if vlm_memory is None:
            memories, vlm_key_padding_mask = self.encode_likelihood_memories(
                vlm_start_images, language_tokens, language_mask
            )
        else:
            memories, vlm_key_padding_mask = vlm_memory
        if self.uses_expert_context_tokens:
            return self._likelihood_sample_with_expert_context_cache(
                condition_tokens,
                self._expert_context_tokens(state, skill_code),
                noise,
                num_steps,
                memories,
                vlm_key_padding_mask,
                self._state_condition(state),
            )
        return self._likelihood_sample_with_condition_cache(
            condition_tokens,
            noise,
            state,
            skill_code,
            num_steps,
            memories,
            vlm_key_padding_mask,
        )

    @torch.no_grad()
    def _sample_dsbc_actions(
        self,
        images: list[Tensor],
        vlm_start_images: list[Tensor],
        state: Tensor | None,
        skill_code: Tensor | None,
        language_tokens: Tensor,
        language_mask: Tensor,
        *,
        noise: Tensor | None,
        num_steps: int | None,
        vlm_memory: tuple[list[Tensor], Tensor] | None,
    ) -> Tensor:
        """Select initial noise once, then run the completely frozen Stage-1 VSA."""
        predicted = self._dsbc_noise_prediction(
            images,
            vlm_start_images,
            state,
            skill_code,
            language_tokens,
            language_mask,
            vlm_memory=vlm_memory,
        )
        if self.config.dsbc_noise_output_mode == "shared":
            predicted = predicted[:, None].expand(-1, self.config.chunk_size, -1)
        batch_size = predicted.shape[0]
        expected_shape = (
            batch_size,
            self.config.chunk_size,
            self.config.max_action_dim,
        )
        if noise is None:
            initial_noise = self.sample_noise(expected_shape, predicted.device)
        else:
            if tuple(noise.shape) != expected_shape:
                raise ValueError(
                    f"DSBC padding-noise reservoir must have shape {expected_shape}, "
                    f"got {tuple(noise.shape)}."
                )
            initial_noise = noise.float().clone()
        initial_noise[..., : self.real_action_dim] = predicted
        return super().sample_actions(
            images,
            state,
            skill_code,
            noise=initial_noise,
            num_steps=num_steps,
        )

    def _likelihood_sample_with_expert_context_cache(
        self,
        condition_tokens: Tensor,
        context_tokens: Tensor,
        noise: Tensor,
        num_steps: int,
        memories: list[Tensor],
        vlm_key_padding_mask: Tensor,
        condition_state: Tensor | None = None,
    ) -> Tensor:
        """Stage-1 expert-context sampler with likelihood blocks before the head."""
        batch_size = noise.shape[0]
        n_prefix = condition_tokens.shape[1] + context_tokens.shape[1]
        n_action = noise.shape[1]
        device = noise.device
        prefix_cache = self._visual_context_cache(
            condition_tokens, context_tokens, condition_state
        )
        action_padding = torch.ones(
            batch_size, n_action, dtype=torch.bool, device=device
        )
        action_blocks = torch.tensor(
            [1] + [0] * (n_action - 1), dtype=torch.bool, device=device
        )[None].expand(batch_size, -1)
        action_attention = make_att_2d_masks(action_padding, action_blocks)
        prefix_visible = torch.ones(
            batch_size, n_action, n_prefix, dtype=torch.bool, device=device
        )
        full_attention = torch.cat(
            (prefix_visible, action_attention), dim=2
        )[:, None]
        full_attention = torch.where(
            full_attention, 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        action_positions = n_prefix + torch.cumsum(action_padding, dim=1) - 1

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = torch.full(
                (batch_size,), 1.0 + step * dt, dtype=torch.float32, device=device
            )
            expert_condition = self._expert_condition(time)
            prior_hidden = self._action_hidden_with_condition_cache(
                x_t,
                expert_condition,
                None,
                prefix_cache,
                full_attention,
                action_positions,
            )
            velocity = self._likelihood_velocity(
                prior_hidden,
                memories,
                vlm_key_padding_mask,
                expert_condition,
            )
            x_t = x_t + dt * velocity
        return x_t

    def _likelihood_sample_with_condition_cache(
        self,
        condition_tokens: Tensor,
        noise: Tensor,
        state: Tensor | None,
        skill_code: Tensor | None,
        num_steps: int,
        memories: list[Tensor],
        vlm_key_padding_mask: Tensor,
    ) -> Tensor:
        """Stage-1 condition-cache sampler with likelihood blocks before the head."""
        batch_size, n_condition = condition_tokens.shape[:2]
        n_chunk = noise.shape[1]
        device = noise.device
        projected_state = self._project_state(state)
        expert_projected_state = self._project_expert_state(state, projected_state)
        condition_state = projected_state if self.uses_cond_state_adarms else None
        condition_state_start_index = self._condition_state_start_index(
            condition_tokens
        )
        condition_skill, expert_skill = self._skill_broadcasts(skill_code)

        condition_padding = torch.ones(
            batch_size, n_condition, dtype=torch.bool, device=device
        )
        condition_blocks = torch.zeros_like(condition_padding)
        condition_attention = make_att_2d_masks(
            condition_padding, condition_blocks
        )[:, None]
        condition_attention = torch.where(
            condition_attention, 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        condition_positions = torch.cumsum(condition_padding, dim=1) - 1
        condition_cache = self.cond_encoder.model.forward(
            inputs_embeds=condition_tokens,
            attention_mask=condition_attention,
            position_ids=condition_positions,
            past_key_values=None,
            use_cache=True,
            adarms_cond=condition_state,
            adarms_start_index=condition_state_start_index,
            broadcast_cond=condition_skill,
        ).past_key_values

        action_padding = torch.ones(batch_size, n_chunk, dtype=torch.bool, device=device)
        action_block_starts = [1] + [0] * (n_chunk - 1)
        action_blocks = torch.tensor(
            action_block_starts, dtype=torch.bool, device=device
        )[None].expand(batch_size, -1)
        action_attention = make_att_2d_masks(action_padding, action_blocks)
        condition_visible = condition_padding[:, None].expand(
            batch_size, n_chunk, n_condition
        )
        full_attention = torch.cat((condition_visible, action_attention), dim=2)[:, None]
        full_attention = torch.where(full_attention, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        action_positions = n_condition + torch.cumsum(action_padding, dim=1) - 1

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = torch.full(
                (batch_size,), 1.0 + step * dt, dtype=torch.float32, device=device
            )
            expert_condition = self._expert_condition(
                time, expert_projected_state, skill_code
            )
            prior_hidden = self._action_hidden_with_condition_cache(
                x_t,
                expert_condition,
                expert_skill,
                condition_cache,
                full_attention,
                action_positions,
            )
            velocity = self._likelihood_velocity(
                prior_hidden,
                memories,
                vlm_key_padding_mask,
                expert_condition,
            )
            x_t = x_t + dt * velocity
        return x_t


# Geometry and prior-architecture fields the Stage-1 checkpoint must agree on.
_STAGE1_CONTRACT_FIELDS = (
    "action_expert_variant",
    "cond_encoder_variant",
    "conditioning_route",
    "chunk_size",
    "n_action_steps",
    "max_state_dim",
    "max_action_dim",
    "num_inference_steps",
    "time_sampling_beta_alpha",
    "time_sampling_beta_beta",
    "time_sampling_scale",
    "time_sampling_offset",
    "min_period",
    "max_period",
    "vision_backbone",
    "dino_image_size",
    "freeze_vision_encoder",
    "skill_vocab_size",
    "skill_fsq_levels",
    "transition_jitter_pmax",
    "transition_jitter_distribution",
)
# Fields absent from older Stage-1 configs; when missing they unambiguously
# carry that era's defaults, so Stage 2 adopts its own configured value.
_STAGE1_OPTIONAL_CONTRACT_FIELDS = (
    "num_visual_latents_per_camera",
    "visual_perceiver_width",
    "mask_actions_after_skill_end",
    "freeze_vision_encoder",
)


class SkillVLAStage2Policy(SkillExpertPolicy):
    """Stage-2 policy assembled from a frozen prior and predictor VLM."""

    config_class = SkillVLAStage2Config
    name = "skill_vla_stage2"

    def __init__(
        self,
        config: SkillVLAStage2Config,
        *,
        initialize_from_sources: bool = True,
        **kwargs,
    ):
        del kwargs
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = SkillVLAStage2Pytorch(config)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(device=config.device, dtype=self._torch_dtype())
        if initialize_from_sources:
            self._initialize_from_stage1(config.stage1_checkpoint_path)
            self._initialize_predictor(config.skill_predictor_checkpoint_path)
        self.model._freeze_stage1_prior()
        self.reset()

    def reset(self) -> None:
        super().reset()
        # Eval-only tensors. They are deliberately ordinary attributes rather
        # than buffers so checkpoints never persist rollout-specific memory.
        self._eval_vlm_cache_ids: Tensor | None = None
        self._eval_vlm_cache: tuple[list[Tensor], Tensor] | None = None

    @torch.no_grad()
    def _cached_eval_vlm_memory(
        self,
        start_images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
        cache_ids: Tensor | None,
    ) -> tuple[list[Tensor], Tensor] | None:
        """Reuse projected VLM memory until an environment starts a new skill.

        Direct policy callers that do not provide ``STAGE2_VLM_CACHE_ID`` retain
        the uncached behavior. The LIBERO rollout wrapper supplies one monotonic
        generation per vector-environment row.
        """
        if cache_ids is None:
            return None
        batch_size = language_tokens.shape[0]
        cache_ids = cache_ids.to(
            device=language_tokens.device, dtype=torch.long
        ).reshape(-1)
        if cache_ids.numel() != batch_size:
            raise ValueError(
                f"{STAGE2_VLM_CACHE_ID} must have {batch_size} entries, got "
                f"{cache_ids.numel()}."
            )

        cached_ids = self._eval_vlm_cache_ids
        cached_memory = self._eval_vlm_cache
        cache_is_compatible = (
            cached_ids is not None
            and cached_memory is not None
            and cached_ids.shape == cache_ids.shape
            and cached_ids.device == cache_ids.device
            and len(cached_memory[0]) == len(self.model.likelihood_blocks)
            and cached_memory[1].shape[0] == batch_size
        )
        if not cache_is_compatible:
            cached_memory = self.model.encode_likelihood_memories(
                start_images, language_tokens, language_mask
            )
            self._eval_vlm_cache = (
                [memory.detach() for memory in cached_memory[0]],
                cached_memory[1].detach(),
            )
            self._eval_vlm_cache_ids = cache_ids.detach().clone()
            return self._eval_vlm_cache

        assert cached_ids is not None and cached_memory is not None
        stale = cache_ids.ne(cached_ids)
        if not bool(stale.any()):
            return cached_memory

        indices = stale.nonzero(as_tuple=False).flatten()
        refreshed = self.model.encode_likelihood_memories(
            [image.index_select(0, indices) for image in start_images],
            language_tokens.index_select(0, indices),
            language_mask.index_select(0, indices),
        )
        shapes_match = (
            len(refreshed[0]) == len(cached_memory[0])
            and all(
                new.shape[1:] == old.shape[1:]
                for new, old in zip(refreshed[0], cached_memory[0], strict=True)
            )
            and refreshed[1].shape[1:] == cached_memory[1].shape[1:]
        )
        if not shapes_match:
            refreshed = self.model.encode_likelihood_memories(
                start_images, language_tokens, language_mask
            )
            self._eval_vlm_cache = (
                [memory.detach() for memory in refreshed[0]],
                refreshed[1].detach(),
            )
            self._eval_vlm_cache_ids = cache_ids.detach().clone()
            return self._eval_vlm_cache

        for old, new in zip(cached_memory[0], refreshed[0], strict=True):
            old.index_copy_(0, indices, new)
        cached_memory[1].index_copy_(0, indices, refreshed[1])
        cached_ids.index_copy_(0, indices, cache_ids.index_select(0, indices))
        return cached_memory

    def _initialize_from_stage1(self, checkpoint_path: str | Path | None) -> None:
        path = Path(str(checkpoint_path or ""))
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Stage-1 config not found: {config_path}")
        stage1_config = json.loads(config_path.read_text())
        if stage1_config.get("type") != "skill_expert":
            raise ValueError(
                f"Stage 2 requires a skill_expert checkpoint, got {stage1_config.get('type')!r}."
            )
        saved_architecture = stage1_config.get("architecture") or (
            COND_GEMMA_ARCHITECTURE if "conditioning_route" in stage1_config else ""
        )
        if saved_architecture != COND_GEMMA_ARCHITECTURE:
            raise ValueError(
                "Stage 2 is implemented on the cond_gemma Stage-1 prior; got "
                f"architecture={saved_architecture!r} at {path}."
            )
        saved_revision = str(
            stage1_config.get("architecture_revision", COND_GEMMA_ARCHITECTURE_REVISION)
        )
        if saved_revision != self.config.architecture_revision:
            raise ValueError(
                "Stage-1 architecture_revision mismatch: "
                f"stage1={saved_revision!r}, stage2={self.config.architecture_revision!r}."
            )
        mismatches = []
        for field in _STAGE1_CONTRACT_FIELDS:
            if field not in stage1_config and field in _STAGE1_OPTIONAL_CONTRACT_FIELDS:
                continue
            expected = stage1_config.get(field)
            actual = getattr(self.config, field)
            if field == "conditioning_route":
                expected = normalize_conditioning_route(str(expected))
                actual = normalize_conditioning_route(str(actual))
            elif field == "skill_fsq_levels":
                expected = [int(value) for value in (expected or [])]
                actual = [int(value) for value in (actual or [])]
            if expected != actual:
                mismatches.append(f"{field}: stage1={expected!r}, stage2={actual!r}")
        if mismatches:
            raise ValueError("Stage-1 architecture mismatch: " + "; ".join(mismatches))

        loaded = _load_pretrained_state_dict(
            path, {}, architecture=COND_GEMMA_ARCHITECTURE
        )
        if loaded is None:
            raise FileNotFoundError(f"Stage-1 model weights not found: {path}")
        state_dict, is_pi05 = loaded
        if is_pi05:
            raise ValueError("Stage 2 cannot initialize directly from a pi0.5 checkpoint.")
        # The predictor is loaded completely from its own checkpoint, and Stage 2
        # never carries a terminator; drop those tensors if the prior has them.
        dropped_prefixes = (
            "model.skill_predictor.",
            "model.fsq_term_train.",
            "model.fsq_image_term_train.",
            "model.fsq_wrist_term_train.",
        )
        state_dict = {
            key: value.to(self._torch_dtype())
            for key, value in state_dict.items()
            if not key.startswith(dropped_prefixes)
        }
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed_prefixes = (
            "model.vlm_to_expert_projection.",
            "model.likelihood_blocks.",
            "model.likelihood_layer_mix",
            "model.noise_out_proj.",
            "model.dsbc_anchor_noise",
            "model.skill_predictor.",
        )
        invalid_missing = [
            key for key in missing if not key.startswith(allowed_prefixes)
        ]
        if invalid_missing or unexpected:
            raise RuntimeError(
                "Stage-1 checkpoint mismatch: "
                f"missing={sorted(invalid_missing)}, unexpected={sorted(unexpected)}"
            )
        log.info(
            "Stage 2 <- Stage-1 prior %s: loaded=%d, fresh=%d.",
            path,
            len(state_dict),
            len(missing),
        )
        if self.config.stage2_mode == "likelihood":
            # Snapshot the warm-started head so training logs can separate "head
            # re-fitting" from genuine likelihood-block usage.
            self._initial_action_head = (
                self.model.action_out_proj.weight.detach().float().cpu().clone()
            )

    def _initialize_predictor(self, checkpoint_path: str | Path | None) -> None:
        predictor = self.model.skill_predictor
        if predictor is None:
            raise RuntimeError("Stage 2 has no predictor module to initialize.")
        path = Path(str(checkpoint_path or ""))
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Stage-2 predictor config not found: {config_path}")
        source_config = json.loads(config_path.read_text())
        if source_config.get("type") not in {"skill_expert", "skill_aux"}:
            raise ValueError(
                "Predictor source must be a skill_expert or skill_aux checkpoint, got "
                f"{source_config.get('type')!r}."
            )
        if not source_config.get("train_skill_predictor", False):
            raise ValueError("Predictor source has no trained predictor.")
        mismatches = [
            f"{field}: predictor={source_config.get(field)!r}, "
            f"stage2={getattr(self.config, field)!r}"
            for field in _PREDICTOR_CHECKPOINT_CONTRACT_FIELDS
            if source_config.get(field) != getattr(self.config, field)
        ]
        if mismatches:
            raise ValueError(
                "Predictor module contract mismatch: " + "; ".join(mismatches)
            )
        loaded = _load_complete_predictor_parameters(predictor, path)
        predictor.requires_grad_(False).eval()
        log.info(
            "Stage 2 <- frozen predictor %s: loaded %d tensors.",
            path,
            loaded,
        )

    def _likelihood_usage_metrics(self) -> dict[str, float]:
        """Report whether the language pathway is actually being used.

        Gates start at exactly zero, so near-zero values here mean the blocks
        are inactive and any loss improvement came from the action head alone.
        """
        blocks = getattr(self.model, "likelihood_blocks", None)
        if blocks is None:
            return {}
        metrics: dict[str, float] = {}
        for kind in ("self", "cross", "ffn"):
            weight_values = []
            gate_values = []
            for block in blocks:
                norm = getattr(block, f"{kind}_norm")
                weight_values.append(
                    norm.dense.weight.detach().float().square().mean().sqrt()
                )
                last = getattr(block, "_last_gate_rms", None)
                if last is not None and kind in last:
                    gate_values.append(last[kind])
            metrics[f"stage2/gate_weight_rms/{kind}"] = float(
                torch.stack(weight_values).mean().item()
            )
            if gate_values:
                metrics[f"stage2/gate_value_rms/{kind}"] = float(
                    torch.stack(gate_values).mean().item()
                )
        metrics["stage2/vlm_projection_weight_rms"] = float(
            self.model.vlm_to_expert_projection.weight.detach()
            .float()
            .square()
            .mean()
            .sqrt()
            .item()
        )
        layer_mix = getattr(self.model, "likelihood_layer_mix", None)
        if layer_mix is not None:
            weights = torch.softmax(layer_mix.detach().float(), dim=-1)
            depth = torch.arange(
                1, weights.shape[1] + 1, dtype=torch.float32, device=weights.device
            )
            metrics["stage2/layer_mix/last_layer_weight"] = float(
                weights[:, -1].mean().item()
            )
            metrics["stage2/layer_mix/mean_depth"] = float(
                (weights * depth).sum(dim=-1).mean().item()
            )
        initial_head = getattr(self, "_initial_action_head", None)
        if initial_head is not None:
            head = self.model.action_out_proj.weight.detach().float().cpu()
            reference_rms = float(initial_head.square().mean().sqrt().item())
            metrics["stage2/action_head_drift_rel"] = float(
                (head - initial_head).square().mean().sqrt().item()
            ) / max(reference_rms, 1e-12)
        noise_head = getattr(self.model, "noise_out_proj", None)
        if noise_head is not None:
            metrics["dsbc/noise_head_weight_rms"] = float(
                noise_head.weight.detach().float().square().mean().sqrt().item()
            )
        return metrics

    def get_optim_params(self) -> list[dict]:
        stage2_mode = getattr(getattr(self, "config", None), "stage2_mode", "likelihood")
        trainable_head = (
            self.model.action_out_proj
            if stage2_mode == "likelihood"
            else self.model.noise_out_proj
        )
        if trainable_head is None:
            raise RuntimeError(f"Stage-2 mode {stage2_mode!r} has no trainable head.")
        trainable = [
            parameter
            for module in (
                self.model.vlm_to_expert_projection,
                self.model.likelihood_blocks,
                trainable_head,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        layer_mix = getattr(self.model, "likelihood_layer_mix", None)
        if layer_mix is not None and layer_mix.requires_grad:
            trainable.append(layer_mix)
        expected = {id(parameter) for parameter in trainable}
        actual = {
            id(parameter)
            for parameter in self.parameters()
            if parameter.requires_grad
        }
        if actual != expected:
            raise RuntimeError("Stage-2 trainable-parameter freeze contract was violated.")

        gate_lr_scale = float(getattr(self.config, "likelihood_gate_lr_scale", 1.0))
        if gate_lr_scale == 1.0:
            return [{"params": trainable}]
        # Bootstrap parameters of the language pathway learn faster to escape
        # the zero-gate cold start: gate dense layers, the VLM projection, and
        # the layer mix.
        boosted_ids = {
            id(parameter)
            for parameter in self.model.vlm_to_expert_projection.parameters()
        }
        for block in self.model.likelihood_blocks:
            for kind in ("self_norm", "cross_norm", "ffn_norm"):
                norm = getattr(block, kind, None)
                if norm is not None and norm.dense is not None:
                    boosted_ids.update(
                        id(parameter) for parameter in norm.dense.parameters()
                    )
        if layer_mix is not None:
            boosted_ids.add(id(layer_mix))
        boosted = [p for p in trainable if id(p) in boosted_ids]
        main_parameters = [p for p in trainable if id(p) not in boosted_ids]
        groups = [{"params": main_parameters}]
        if boosted:
            groups.append(
                {
                    "params": boosted,
                    "lr": self.config.optimizer_lr * gate_lr_scale,
                }
            )
        return groups

    def isolated_auxiliary_step(
        self,
        batch: dict,
        accelerator,
        grad_clip_norm: float,
        current_lr: float | None = None,
    ) -> dict:
        """Stage 2 keeps the predictor frozen; there is no auxiliary training."""
        del batch, accelerator, grad_clip_norm, current_lr
        return {}

    def _same_skill_batch_metrics(
        self, batch: dict, conditioning_skill: Tensor
    ) -> dict[str, float]:
        """Validate and report the sampler's post-jitter pairing contract.

        Pair construction is based on the dataset's jittered GT code.  When
        ``training_skill_source=predictor``, the extra conditioning metric shows
        how often the predictor preserved the intended same-skill relation.
        """
        if not bool(getattr(self.config, "same_skill_batch_enabled", False)):
            return {}
        required = (
            SAME_SKILL_PAIR_ID,
            SAME_SKILL_PAIR_FALLBACK,
            "skill_code",
            "task_index",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(
                "same_skill_batch_enabled=True but grouped sampler metadata is "
                f"missing: {missing}."
            )

        device = conditioning_skill.device
        pair_ids = batch[SAME_SKILL_PAIR_ID].to(device).view(-1).long()
        fallback = batch[SAME_SKILL_PAIR_FALLBACK].to(device).view(-1).bool()
        jittered_gt = batch["skill_code"].to(device).view(-1).long()
        task = batch["task_index"].to(device).view(-1).long()
        requested = (pair_ids >= 0) | fallback
        constructed = pair_ids >= 0
        positions = torch.nonzero(constructed, as_tuple=False).view(-1)
        if positions.numel() % 2:
            raise ValueError("Grouped sampler produced an odd number of pair samples.")
        pairs = positions.view(-1, 2)
        valid_gt = torch.ones(pairs.shape[0], dtype=torch.bool, device=device)
        valid_conditioning = valid_gt.clone()
        if pairs.numel() > 0:
            valid_gt &= pair_ids[pairs[:, 0]] == pair_ids[pairs[:, 1]]
            valid_gt &= jittered_gt[pairs[:, 0]] == jittered_gt[pairs[:, 1]]
            valid_gt &= task[pairs[:, 0]] != task[pairs[:, 1]]
            valid_conditioning = valid_gt & (
                conditioning_skill[pairs[:, 0]]
                == conditioning_skill[pairs[:, 1]]
            )

        batch_size = max(pair_ids.numel(), 1)
        requested_count = requested.float().sum().clamp_min(1.0)
        metrics = {
            "batch_sampling/requested_fraction": requested.float().mean().item(),
            "batch_sampling/constructed_fraction": constructed.float().mean().item(),
            "batch_sampling/effective_after_jitter_fraction": (
                2.0 * valid_gt.float().sum() / batch_size
            ).item(),
            "batch_sampling/effective_conditioning_fraction": (
                2.0 * valid_conditioning.float().sum() / batch_size
            ).item(),
            "batch_sampling/effective_of_requested": (
                2.0 * valid_gt.float().sum() / requested_count
            ).item(),
            "batch_sampling/fallback_fraction": fallback.float().mean().item(),
            "batch_sampling/unique_jittered_skills": float(
                torch.unique(jittered_gt).numel()
            ),
        }
        progress = batch.get(SKILL_PROGRESS)
        if pairs.numel() > 0 and progress is not None:
            progress = progress.to(device).view(-1).float()
            metrics["batch_sampling/jittered_progress_gap"] = (
                progress[pairs[:, 0]] - progress[pairs[:, 1]]
            ).abs().mean().item()
        return metrics

    def _forward_dsbc(self, batch: dict, reduction: str):
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        real_dim = self.config.output_features[ACTION].shape[0]
        if real_dim != self.model.real_action_dim:
            raise RuntimeError(
                "DSBC real-action dimension changed after model construction: "
                f"{real_dim} != {self.model.real_action_dim}."
            )
        device = actions.device
        route = normalize_conditioning_route(self.config.conditioning_route)
        state = (
            None
            if route in STATELESS_CONDITIONING_ROUTES
            else pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        )
        skill_code = (
            None
            if route in SKILLLESS_CONDITIONING_ROUTES
            else self._training_skill_code(batch)
        )
        images = self._collect_images(batch)
        vlm_start_images = self._predictor_start_images(batch)
        prediction, target, action_residual = self.model.dsbc_training_pair(
            images,
            vlm_start_images,
            state,
            skill_code,
            actions,
            batch[OBS_LANGUAGE_TOKENS].to(device),
            batch[OBS_LANGUAGE_ATTENTION_MASK].to(device),
        )
        valid = self._valid_action_steps(actions, batch)
        valid_float = valid.to(target.dtype).unsqueeze(-1)
        valid_per_sample = valid.sum(dim=1).clamp(min=1).to(target.dtype)

        action_squared_error = action_residual.square()
        valid_steps = valid.sum().clamp(min=1).to(target.dtype)
        action_loss = (action_squared_error * valid_float).sum() / (
            valid_steps * real_dim
        )
        action_loss_per_dim = (
            action_squared_error * valid_float
        ).sum(dim=(0, 1)) / valid_steps

        if self.config.dsbc_noise_output_mode == "shared":
            shared_target = (target * valid_float).sum(dim=1) / valid_per_sample[:, None]
            squared_error = (prediction - shared_target).square()
            per_sample = squared_error.mean(dim=-1)
            noise_loss = per_sample.mean()
            loss_per_dim = squared_error.mean(dim=0)
            metric_prediction = prediction[:, None].expand_as(target)
        else:
            squared_error = (prediction - target).square()
            per_sample = (squared_error * valid_float).sum(dim=(1, 2)) / (
                valid_per_sample * real_dim
            )
            noise_loss = (squared_error * valid_float).sum() / (
                valid_steps * real_dim
            )
            loss_per_dim = (squared_error * valid_float).sum(dim=(0, 1)) / valid_steps
            metric_prediction = prediction

        # The selector is supervised on the chunk mean in shared mode and on
        # each valid FRS target in per-step mode. Keep those statistics separate
        # from the unaveraged FRS distribution so the tanh bound can be audited.
        valid_frs_target = target.detach().float()[valid]
        supervision_target = (
            shared_target.detach().float()
            if self.config.dsbc_noise_output_mode == "shared"
            else valid_frs_target
        )
        noise_bound = float(
            getattr(self.config, "dsbc_noise_output_bound", 5.0)
        )

        def _abs_noise_stats(values: Tensor) -> tuple[float, float, float, float]:
            absolute = values.abs().reshape(-1)
            if absolute.numel() == 0:
                return 0.0, 0.0, 0.0, 0.0
            quantiles = torch.quantile(
                absolute,
                torch.tensor([0.95, 0.99], device=absolute.device),
            )
            return (
                quantiles[0].item(),
                quantiles[1].item(),
                absolute.max().item(),
                absolute.gt(noise_bound).float().mean().item(),
            )

        frs_p95, frs_p99, frs_max, frs_outside = _abs_noise_stats(
            valid_frs_target
        )
        target_p95, target_p99, target_max, target_outside = _abs_noise_stats(
            supervision_target
        )
        supervision_target_rms = (
            supervision_target.square().mean().sqrt().item()
            if supervision_target.numel() > 0
            else 0.0
        )
        valid_frs_target_rms = (
            valid_frs_target.square().mean().sqrt().item()
            if valid_frs_target.numel() > 0
            else 0.0
        )

        loss_dict = {
            # Same flow-velocity MSE key as legacy Stage 2 for W&B comparison;
            # this detached diagnostic is not the DSBC optimization objective.
            "action_loss": action_loss.detach().item(),
            "loss_per_dim": action_loss_per_dim.detach().cpu().tolist(),
            "gt_noise_loss": noise_loss.detach().item(),
            "gt_noise_loss_per_dim": loss_per_dim.detach().cpu().tolist(),
            # Retain the shorter aliases for scripts consuming early DSBC runs.
            "noise_loss": noise_loss.detach().item(),
            "noise_loss_per_dim": loss_per_dim.detach().cpu().tolist(),
            "dsbc/target_rms": target.detach().float().square().mean().sqrt().item(),
            "dsbc/prediction_rms": (
                metric_prediction.detach().float().square().mean().sqrt().item()
            ),
            "dsbc/supervision_target_rms": supervision_target_rms,
            "dsbc/frs_target_valid_rms": valid_frs_target_rms,
            "dsbc/supervision_target_abs_p95": target_p95,
            "dsbc/supervision_target_abs_p99": target_p99,
            "dsbc/supervision_target_abs_max": target_max,
            "dsbc/supervision_target_outside_bound_fraction": target_outside,
            "dsbc/frs_target_abs_p95": frs_p95,
            "dsbc/frs_target_abs_p99": frs_p99,
            "dsbc/frs_target_abs_max": frs_max,
            "dsbc/frs_target_outside_bound_fraction": frs_outside,
            "dsbc/noise_output_bound": noise_bound,
            "dsbc/frs_displacement_rms": (
                (target.detach() - actions[..., :real_dim].float())
                .square()
                .mean()
                .sqrt()
                .item()
            ),
            "dsbc/output_shared": float(
                self.config.dsbc_noise_output_mode == "shared"
            ),
            "dsbc/frs_num_steps": float(self.config.dsbc_frs_num_steps),
            "stage2/skill_source_predictor": float(
                self.config.training_skill_source == "predictor"
            ),
        }
        jitter_fraction = getattr(self, "_last_transition_jitter_fraction", None)
        if jitter_fraction is not None:
            loss_dict["regime/transition_jitter_fraction"] = (
                jitter_fraction.detach().item()
            )
        predicted_accuracy = getattr(self, "_last_predicted_skill_accuracy", None)
        if predicted_accuracy is not None:
            loss_dict["conditioning/predictor_acc_vs_jittered_gt"] = (
                predicted_accuracy.detach().item()
            )
            loss_dict["conditioning/unique_predicted_skills"] = float(
                self._last_unique_predicted_skills
            )
        predicted_diff = getattr(self, "_last_predicted_diff_from_current", None)
        if predicted_diff is not None:
            loss_dict["conditioning/predicted_diff_from_current_gt"] = (
                predicted_diff.detach().item()
            )
        if skill_code is not None:
            loss_dict.update(self._same_skill_batch_metrics(batch, skill_code))
        loss_dict.update(self._likelihood_usage_metrics())
        if reduction == "none":
            return per_sample, loss_dict
        return noise_loss, loss_dict

    def forward(self, batch: dict, reduction: str = "mean"):
        if getattr(self.config, "stage2_mode", "likelihood") == "dsbc":
            return self._forward_dsbc(batch, reduction)
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        real_dim = self.config.output_features[ACTION].shape[0]
        device = actions.device
        route = normalize_conditioning_route(self.config.conditioning_route)
        state = (
            None
            if route in STATELESS_CONDITIONING_ROUTES
            else pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        )
        skill_code = (
            None
            if route in SKILLLESS_CONDITIONING_ROUTES
            else self._training_skill_code(batch)
        )
        # Keep the frozen VSA on the current observation, while the VLM memory
        # sees the same jittered skill-start observation used to build the
        # language prompt and GT skill code.
        images = self._collect_images(batch)
        vlm_start_images = self._predictor_start_images(batch)
        batch_sampling_metrics = (
            self._same_skill_batch_metrics(batch, skill_code)
            if skill_code is not None
            else {}
        )
        residual = self.model(
            images,
            vlm_start_images,
            state,
            skill_code,
            actions,
            batch[OBS_LANGUAGE_TOKENS].to(device),
            batch[OBS_LANGUAGE_ATTENTION_MASK].to(device),
        )[..., :real_dim]
        squared_error = residual.square()
        valid = self._valid_action_steps(actions, batch)
        valid_float = valid.to(squared_error.dtype).unsqueeze(-1)
        valid_per_sample = valid.sum(dim=1).clamp(min=1).to(squared_error.dtype)
        per_sample = (squared_error * valid_float).sum(dim=(1, 2)) / (
            valid_per_sample * real_dim
        )
        valid_steps = valid.sum().clamp(min=1).to(squared_error.dtype)
        action_loss = (squared_error * valid_float).sum() / (
            valid_steps * real_dim
        )
        loss_per_dim = (squared_error * valid_float).sum(dim=(0, 1)) / valid_steps
        cumulative_xyz_loss = None
        cumulative_xyz_raw_loss = None
        action_objective = action_loss
        objective_per_sample = per_sample
        if self.config.cumulative_xyz_loss_enabled:
            predicted_actions = self.model._last_predicted_actions
            if predicted_actions is None:
                raise RuntimeError(
                    "cumulative XYZ loss did not receive reconstructed predicted actions."
                )
            (
                cumulative_xyz_loss,
                cumulative_xyz_raw_loss,
                cumulative_xyz_per_sample,
                _cumulative_xyz_raw_per_sample,
            ) = self._cumulative_xyz_loss(
                predicted_actions[..., :real_dim],
                actions[..., :real_dim],
                valid,
            )
            cumulative_weight = self.config.cumulative_xyz_loss_weight
            action_objective = action_loss + cumulative_weight * cumulative_xyz_loss
            objective_per_sample = (
                per_sample + cumulative_weight * cumulative_xyz_per_sample
            )
        loss_dict = {
            "action_loss": action_loss.detach().item(),
            "loss_per_dim": loss_per_dim.detach().cpu().tolist(),
            "stage2/skill_source_predictor": float(
                self.config.training_skill_source == "predictor"
            ),
        }
        jitter_fraction = getattr(self, "_last_transition_jitter_fraction", None)
        if jitter_fraction is not None:
            loss_dict["regime/transition_jitter_fraction"] = (
                jitter_fraction.detach().item()
            )
        predicted_accuracy = getattr(self, "_last_predicted_skill_accuracy", None)
        if predicted_accuracy is not None:
            loss_dict["conditioning/predictor_acc_vs_jittered_gt"] = (
                predicted_accuracy.detach().item()
            )
            loss_dict["conditioning/unique_predicted_skills"] = float(
                self._last_unique_predicted_skills
            )
        predicted_diff = getattr(self, "_last_predicted_diff_from_current", None)
        if predicted_diff is not None:
            loss_dict["conditioning/predicted_diff_from_current_gt"] = (
                predicted_diff.detach().item()
            )
        loss_dict.update(batch_sampling_metrics)
        loss_dict.update(self._likelihood_usage_metrics())
        if cumulative_xyz_loss is not None and cumulative_xyz_raw_loss is not None:
            cumulative_weight = self.config.cumulative_xyz_loss_weight
            weighted_cumulative = cumulative_weight * cumulative_xyz_loss
            loss_dict.update(
                {
                    "cumulative_xyz/raw": cumulative_xyz_raw_loss.detach().item(),
                    "cumulative_xyz/normalized": cumulative_xyz_loss.detach().item(),
                    "cumulative_xyz/weighted": weighted_cumulative.detach().item(),
                    "cumulative_xyz/weight": float(cumulative_weight),
                    "cumulative_xyz/horizon_normalizer_mean": (
                        (valid.sum(dim=1).float() + 1.0) / 2.0
                    ).mean().item(),
                    "cumulative_xyz/to_flow_ratio": (
                        weighted_cumulative.detach()
                        / action_loss.detach().clamp(
                            min=torch.finfo(action_loss.dtype).eps
                        )
                    ).item(),
                    "action_objective": action_objective.detach().item(),
                    "action_flow_weight": 1.0,
                    "action_cumulative_xyz_weight": float(cumulative_weight),
                }
            )
        if reduction == "none":
            return objective_per_sample, loss_dict
        return action_objective, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **kwargs) -> Tensor:
        self.eval()
        device = next(self.parameters()).device
        route = normalize_conditioning_route(self.config.conditioning_route)
        state = (
            None
            if route in STATELESS_CONDITIONING_ROUTES
            else pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        )
        skill_code = (
            None
            if route in SKILLLESS_CONDITIONING_ROUTES
            else self._skill_code(batch)
        )
        start_images = self._predictor_start_images(batch)
        language_tokens = batch[OBS_LANGUAGE_TOKENS].to(device)
        language_mask = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device)
        vlm_memory = self._cached_eval_vlm_memory(
            start_images,
            language_tokens,
            language_mask,
            batch.get(STAGE2_VLM_CACHE_ID),
        )
        actions = self.model.sample_actions(
            self._collect_images(batch),
            start_images,
            state,
            skill_code,
            language_tokens,
            language_mask,
            vlm_memory=vlm_memory,
            **kwargs,
        )
        real_dim = self.config.output_features[ACTION].shape[0]
        return actions[..., :real_dim]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path,
        *,
        config=None,
        strict: bool = True,
        **kwargs,
    ):
        """Load a complete, self-contained Stage-2 checkpoint."""
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        return PreTrainedPolicy.from_pretrained.__func__(
            cls,
            pretrained_name_or_path,
            config=config,
            strict=strict,
            initialize_from_sources=False,
            **kwargs,
        )
