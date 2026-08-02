"""Original skillVLA_real condition-Gemma Stage-1 architecture.

This module intentionally preserves the old two-stream implementation:
DINO -> condition Gemma and noisy actions -> pi0.5 Gemma expert.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from transformers import AutoModel

from lerobot.policies.pi05.modeling_pi05 import (
    OPENPI_ATTENTION_MASK_VALUE,
    compute_layer_complete,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    layernorm_forward,
    make_att_2d_masks,
    sample_beta,
)

from .configuration_skill_expert import (
    SKILLLESS_CONDITIONING_ROUTES,
    STATELESS_CONDITIONING_ROUTES,
    VISIONLESS_CONDITIONING_ROUTES,
    SkillExpertConfig,
)
from .modeling_skill_predictor import FrozenVLMSkillPredictor
from .modeling_utils import build_fsq_terminator, build_gemma


class CondGemmaSkillExpert(nn.Module):
    """DINO condition stream + fully trainable pi0.5 action expert."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__()
        self.config = config
        expert_config = get_gemma_config(config.action_expert_variant)
        self.width = expert_config.width

        if config.conditioning_route in VISIONLESS_CONDITIONING_ROUTES:
            # The transformer still needs a condition sequence to carry state
            # AdaRMS and skill broadcasts. This learned seed contains no
            # observation information and replaces all vision tokens.
            self.dino = None
            self.n_register_tokens = 0
            self.image_proj = None
            self.visionless_condition_token = nn.Parameter(
                torch.zeros(1, 1, self.width)
            )
        else:
            self.dino = AutoModel.from_pretrained(config.dino_model_path)
            if config.freeze_vision_encoder:
                self.dino.requires_grad_(False)
                self.dino.eval()
            self.n_register_tokens = int(
                getattr(self.dino.config, "num_register_tokens", 0)
            )
            self.image_proj = nn.Linear(int(self.dino.config.hidden_size), self.width)
            self.register_parameter("visionless_condition_token", None)
        self.register_buffer(
            "_image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        # Stateless routes deliberately have no state parameters in the VSA
        # graph. This also avoids an unused trainable projection under DDP.
        self.state_proj = (
            None
            if config.conditioning_route in STATELESS_CONDITIONING_ROUTES
            else nn.Linear(config.max_state_dim, self.width)
        )
        # Skill-free routes likewise omit the otherwise unused projection.
        self.skill_proj = (
            None
            if config.conditioning_route in SKILLLESS_CONDITIONING_ROUTES
            else nn.Linear(len(config.skill_fsq_levels), self.width)
        )
        levels = torch.tensor(config.skill_fsq_levels, dtype=torch.long)
        strides = torch.ones_like(levels)
        for index in range(1, len(config.skill_fsq_levels)):
            strides[index] = strides[index - 1] * config.skill_fsq_levels[index - 1]
        self.register_buffer("_fsq_levels", levels, persistent=False)
        self.register_buffer("_fsq_strides", strides, persistent=False)
        self.register_buffer("_fsq_half", (levels - 1).float() / 2.0, persistent=False)

        self.action_in_proj = nn.Linear(config.max_action_dim, self.width)
        self.action_out_proj = nn.Linear(self.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(self.width, self.width)
        self.time_mlp_out = nn.Linear(self.width, self.width)

        self.cond_encoder = build_gemma(
            config.cond_encoder_variant,
            use_adarms=config.conditioning_route not in STATELESS_CONDITIONING_ROUTES,
        )
        self.gemma_expert = build_gemma(config.action_expert_variant, use_adarms=True)
        self.skill_predictor = (
            FrozenVLMSkillPredictor(config) if config.uses_skill_predictor else None
        )
        self.fsq_term_train = None
        if config.train_terminator:
            terminator = build_fsq_terminator(
                config.fsq_path,
                dino_model_path=config.terminator_dino_model_path,
            )
            if config.terminator_freeze_vision_encoder is not None:
                terminator.freeze_vision_encoder = bool(
                    config.terminator_freeze_vision_encoder
                )
            terminator.requires_grad_(True).train()
            if terminator.freeze_vision_encoder:
                terminator.vision_encoder.requires_grad_(False).eval()
            self.fsq_term_train = terminator.to(dtype=torch.float32)
        self._last_predicted_actions: Tensor | None = None
        # The current trainer calls this common observability surface.  Keeping
        # it empty adds no work to the original skillVLA_real forward path.
        self._last_vsa_debug_stats: dict[str, float] = {}
        self._vsa_debug_active = False
        self._gradient_checkpointing = False

    @property
    def working_dtype(self) -> torch.dtype:
        return self.action_in_proj.weight.dtype

    def set_training_step(self, step: int) -> None:
        _ = step

    def gradient_checkpointing_enable(self) -> None:
        self._gradient_checkpointing = True
        if hasattr(self.cond_encoder, "gradient_checkpointing_enable"):
            self.cond_encoder.gradient_checkpointing_enable()
        if hasattr(self.gemma_expert, "gradient_checkpointing_enable"):
            self.gemma_expert.gradient_checkpointing_enable()
        if self.skill_predictor is not None and self.config.train_skill_predictor:
            self.skill_predictor.gradient_checkpointing_enable()
        if (
            self.dino is not None
            and not self.config.freeze_vision_encoder
            and hasattr(self.dino, "gradient_checkpointing_enable")
        ):
            self.dino.gradient_checkpointing_enable()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.dino is not None and self.config.freeze_vision_encoder:
            self.dino.eval()
        if self.skill_predictor is not None and not self.config.train_skill_predictor:
            # Frozen checkpoint predictors must be deterministic while supplying
            # the action-conditioning code during Stage-1 training.
            self.skill_predictor.eval()
        if (
            self.fsq_term_train is not None
            and self.fsq_term_train.freeze_vision_encoder
        ):
            self.fsq_term_train.vision_encoder.eval()
        return self

    def sample_noise(self, shape, device) -> Tensor:
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_time(self, batch_size: int, device) -> Tensor:
        time = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            batch_size,
            device,
        )
        time = time * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def _image_features(self, image: Tensor) -> Tensor:
        if self.dino is None:
            raise RuntimeError(
                f"{self.config.conditioning_route} has no vision encoder in the VSA graph."
            )
        image = image.float()
        image = F.interpolate(
            image,
            size=(self.config.dino_image_size, self.config.dino_image_size),
            mode="bilinear",
            align_corners=False,
        )
        image = (image - self._image_mean.float()) / self._image_std.float()
        image = image.to(dtype=next(self.dino.parameters()).dtype)
        context = torch.no_grad() if self.config.freeze_vision_encoder else nullcontext()
        with context:
            hidden = self.dino(image).last_hidden_state
        cls_token = hidden[:, :1]
        patch_tokens = hidden[:, 1 + self.n_register_tokens :]
        return torch.cat((cls_token, patch_tokens), dim=1)

    def _condition_tokens(
        self, images: list[Tensor], *, batch_size: int | None = None
    ) -> Tensor:
        if self.config.conditioning_route in VISIONLESS_CONDITIONING_ROUTES:
            if batch_size is None:
                if not images:
                    raise ValueError("Visionless conditioning requires batch_size.")
                batch_size = images[0].shape[0]
            return self.visionless_condition_token.expand(batch_size, -1, -1)
        if self.image_proj is None:
            raise RuntimeError("Vision-conditioned route has no image projection.")
        tokens = [
            self.image_proj(self._image_features(image).to(self.working_dtype))
            for image in images
        ]
        return torch.cat(tokens, dim=1)

    def _code_to_zq(self, skill_code: Tensor) -> Tensor:
        index = skill_code.reshape(-1, 1).long()
        level_ids = (
            torch.div(index, self._fsq_strides[None], rounding_mode="floor")
            % self._fsq_levels[None]
        )
        return (level_ids.float() - self._fsq_half[None]) / self._fsq_half[None]

    def _skill_embedding(self, skill_code: Tensor) -> Tensor:
        if self.skill_proj is None:
            raise RuntimeError(
                f"{self.config.conditioning_route} has no skill projection in the VSA graph."
            )
        z_q = self._code_to_zq(skill_code).to(self.working_dtype)
        return self.skill_proj(z_q)

    def _state_condition(self, state: Tensor | None) -> Tensor | None:
        """Project state for cond AdaRMS, or omit it in stateless routes."""
        if self.config.conditioning_route in STATELESS_CONDITIONING_ROUTES:
            return None
        if state is None or self.state_proj is None:
            raise ValueError(
                f"{self.config.conditioning_route} requires robot state conditioning."
            )
        return self.state_proj(state.to(self.working_dtype))

    def _skill_broadcasts(
        self, skill_code: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None]:
        """Return ``(condition_stream, action_stream)`` skill broadcasts."""
        if self.config.conditioning_route in SKILLLESS_CONDITIONING_ROUTES:
            return None, None
        if skill_code is None:
            raise ValueError(
                f"{self.config.conditioning_route} requires skill conditioning."
            )
        skill = self._skill_embedding(skill_code)
        if self.config.conditioning_route == "state_cond":
            return None, skill
        return skill, None

    def terminator_predict(
        self,
        true_code: Tensor,
        raw_state: Tensor,
        image: Tensor,
        wrist_image: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the independent FSQ terminator on current raw observations."""
        terminator = self.fsq_term_train
        if terminator is None:
            raise RuntimeError("Terminator training is disabled.")
        device = next(terminator.parameters()).device
        dtype = next(terminator.parameters()).dtype
        state = raw_state.to(device=device, dtype=dtype)[
            ..., : int(terminator.state_dim)
        ]
        z_q = self._code_to_zq(true_code.to(self._fsq_strides.device)).to(
            device=device, dtype=dtype
        )
        return terminator(
            z_q,
            state,
            image.to(device=device, dtype=dtype),
            wrist_image.to(device=device, dtype=dtype),
        )

    def _time_condition(self, timestep: Tensor) -> Tensor:
        condition = create_sinusoidal_pos_embedding(
            timestep,
            self.width,
            self.config.min_period,
            self.config.max_period,
            device=timestep.device,
        ).to(self.working_dtype)
        condition = F.silu(self.time_mlp_in(condition))
        return F.silu(self.time_mlp_out(condition))

    def _expert_condition(self, timestep: Tensor) -> Tensor:
        """Keep the pi0.5 action expert's AdaRMS input strictly time-only."""
        return self._time_condition(timestep)

    def _run_joint_hidden(
        self,
        condition_tokens: Tensor,
        noisy_actions: Tensor,
        condition_state: Tensor | None,
        expert_condition: Tensor,
        condition_skill: Tensor | None,
        expert_skill: Tensor | None,
    ) -> Tensor:
        """Return the normalized action hidden after all 18 Stage-1 layer pairs."""
        action_tokens = self.action_in_proj(noisy_actions.to(self.working_dtype))
        n_chunk = action_tokens.shape[1]
        batch_size, n_condition = condition_tokens.shape[:2]
        n_action = action_tokens.shape[1]
        device = action_tokens.device

        padding_mask = torch.ones(
            batch_size, n_condition + n_action, dtype=torch.bool, device=device
        )
        # Condition tokens form one bidirectional block. Action tokens form the
        # second bidirectional block and can read the full condition stream.
        block_starts = [0] * n_condition + [1] + [0] * (n_chunk - 1)
        block_mask = torch.tensor(block_starts, dtype=torch.bool, device=device)
        block_mask = block_mask[None].expand(batch_size, -1)
        attention_mask = make_att_2d_masks(padding_mask, block_mask)[:, None]
        attention_mask = torch.where(
            attention_mask, 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        position_ids = torch.cumsum(padding_mask, dim=1) - 1

        streams = [condition_tokens, action_tokens]
        adarms_conditions = [condition_state, expert_condition]
        broadcast_conditions = [condition_skill, expert_skill]
        condition_shim = SimpleNamespace(
            model=SimpleNamespace(language_model=self.cond_encoder.model)
        )
        use_checkpoint = self._gradient_checkpointing and self.training
        for layer_index in range(self.gemma_expert.model.config.num_hidden_layers):
            if use_checkpoint:
                streams = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete,
                    layer_index,
                    streams,
                    attention_mask,
                    position_ids,
                    adarms_conditions,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    paligemma=condition_shim,
                    gemma_expert=self.gemma_expert,
                    broadcast_cond=broadcast_conditions,
                )
            else:
                streams = compute_layer_complete(
                    layer_index,
                    streams,
                    attention_mask,
                    position_ids,
                    adarms_conditions,
                    paligemma=condition_shim,
                    gemma_expert=self.gemma_expert,
                    broadcast_cond=broadcast_conditions,
                )

        action_hidden, _ = layernorm_forward(
            self.gemma_expert.model.norm, streams[1], expert_condition
        )
        return action_hidden[:, -n_chunk:]

    def _run_joint(
        self,
        condition_tokens: Tensor,
        noisy_actions: Tensor,
        condition_state: Tensor | None,
        expert_condition: Tensor,
        condition_skill: Tensor | None,
        expert_skill: Tensor | None,
    ) -> Tensor:
        """Run Stage 1 and project its normalized action hidden to flow velocity."""
        action_hidden = self._run_joint_hidden(
            condition_tokens,
            noisy_actions,
            condition_state,
            expert_condition,
            condition_skill,
            expert_skill,
        )
        return self.action_out_proj(action_hidden.to(self.working_dtype)).float()

    def forward(
        self,
        images: list[Tensor],
        state: Tensor | None,
        skill_code: Tensor | None,
        actions: Tensor,
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Return the signed flow residual; its square is the action-flow MSE."""
        batch_size = actions.shape[0]
        time = self.sample_time(batch_size, actions.device) if time is None else time
        source = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        source = source.to(actions.dtype)
        x_t = time[:, None, None] * source + (1.0 - time[:, None, None]) * actions
        target_velocity = source - actions
        condition_skill, expert_skill = self._skill_broadcasts(skill_code)

        predicted_velocity = self._run_joint(
            self._condition_tokens(images, batch_size=batch_size),
            x_t,
            self._state_condition(state),
            self._expert_condition(time),
            condition_skill,
            expert_skill,
        )
        if self.config.action_loss_mode == "flow_endpoint_xyz":
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
        state: Tensor | None,
        skill_code: Tensor | None,
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
        return self._sample_with_condition_cache(
            condition_tokens, noise, state, skill_code, num_steps
        )

    def _sample_with_condition_cache(
        self,
        condition_tokens: Tensor,
        noise: Tensor,
        state: Tensor | None,
        skill_code: Tensor | None,
        num_steps: int,
    ) -> Tensor:
        """Encode the condition stream once, then Euler-integrate the action flow."""
        batch_size, n_condition = condition_tokens.shape[:2]
        n_chunk = noise.shape[1]
        device = noise.device
        condition_state = self._state_condition(state)
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
        full_attention = torch.where(
            full_attention, 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        action_positions = n_condition + torch.cumsum(action_padding, dim=1) - 1

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = torch.full(
                (batch_size,), 1.0 + step * dt, dtype=torch.float32, device=device
            )
            action_hidden = self._action_hidden_with_condition_cache(
                x_t,
                self._expert_condition(time),
                expert_skill,
                condition_cache,
                full_attention,
                action_positions,
            )
            velocity = self.action_out_proj(action_hidden.to(self.working_dtype)).float()
            x_t = x_t + dt * velocity
        return x_t

    def _action_hidden_with_condition_cache(
        self,
        noisy_actions: Tensor,
        expert_condition: Tensor,
        expert_skill: Tensor | None,
        condition_cache,
        attention_mask: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """Run only the 18-layer action stream against a cached condition stream."""
        action_tokens = self.action_in_proj(noisy_actions.to(self.working_dtype))
        hidden = self.gemma_expert.model.forward(
            inputs_embeds=action_tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=copy.deepcopy(condition_cache),
            use_cache=False,
            adarms_cond=expert_condition,
            broadcast_cond=expert_skill,
        ).last_hidden_state
        return hidden

