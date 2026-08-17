"""BayesVLA-style Stage-2 likelihood refinement on the frozen cond_gemma Stage-1 prior.

Stage 2 assembles two independently trained frozen pieces:

* ``stage1_checkpoint_path``: a cond_gemma (Arch0-family) Stage-1 VSA prior.
  It carries no predictor or terminator of its own.
* ``skill_predictor_checkpoint_path``: any Stage-1/skill_aux checkpoint whose
  frozen VLM (and trained reader/head) is loaded completely into the predictor
  module built from that checkpoint's own architecture fields.

Only the four likelihood blocks, the VLM projection, and the action head train.
Terminators are attached externally at evaluation time via
``load_external_terminator`` and are never part of Stage-2 training.
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
    """Frozen cond_gemma Stage-1 prior plus four action-only likelihood blocks."""

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
        self.action_out_proj.requires_grad_(True)
        if self.likelihood_layer_mix is not None:
            self.likelihood_layer_mix.requires_grad_(True)

    def train(self, mode: bool = True):
        # Stage 2 must not change stochastic behavior or running state anywhere
        # in the frozen prior or VLM; only fresh likelihood modules follow mode.
        nn.Module.train(self, mode)
        trainable_ids = {
            id(self.vlm_to_expert_projection),
            id(self.likelihood_blocks),
            id(self.action_out_proj),
        }
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

    def _likelihood_velocity(
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
        return self.action_out_proj(hidden.to(self.working_dtype)).float()

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
    ) -> Tensor:
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
        vlm_hidden, vlm_key_padding_mask = self._encode_likelihood_memory(
            vlm_start_images, language_tokens, language_mask
        )
        memories = self._likelihood_memories(vlm_hidden)
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
    """Likelihood policy assembled from a frozen prior and a frozen predictor VLM."""

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
        return metrics

    def get_optim_params(self) -> list[dict]:
        trainable = [
            parameter
            for module in (
                self.model.vlm_to_expert_projection,
                self.model.likelihood_blocks,
                self.model.action_out_proj,
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

    def forward(self, batch: dict, reduction: str = "mean"):
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
        actions = self.model.sample_actions(
            self._collect_images(batch),
            self._predictor_start_images(batch),
            state,
            skill_code,
            batch[OBS_LANGUAGE_TOKENS].to(device),
            batch[OBS_LANGUAGE_ATTENTION_MASK].to(device),
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
