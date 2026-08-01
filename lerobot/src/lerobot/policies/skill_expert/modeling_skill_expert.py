"""Stage-1 vision-state-action prior with an isolated skill-prediction auxiliary.

The action path has no VLM or language input. Usually DINO and a fresh 18-layer
condition transformer encode the current cameras; the visionless route replaces
camera tokens with one learned condition seed. Depending on the selected
experiment, robot state either modulates that condition stream through AdaRMS or
is omitted from the action path entirely. The 18-layer pi0.5 action expert
retains its original time-only AdaRMS input. The selected skill is broadcast
into either the action stream or the condition stream, or omitted in the
skill-free routes.
Optionally, a frozen pi0.5
VLM base with a skill-only LoRA feeds a separate
SkillReader/SkillHead optimizer; its graph and gradient clipping are disjoint from
the action path.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from transformers import AutoModel

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.lora import route_plain_to_base
from lerobot.policies.pi05.modeling_pi05 import (
    OPENPI_ATTENTION_MASK_VALUE,
    compute_layer_complete,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    layernorm_forward,
    make_att_2d_masks,
    pad_vector,
    sample_beta,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from .configuration_skill_expert import (
    SKILLLESS_CONDITIONING_ROUTES,
    STATELESS_CONDITIONING_ROUTES,
    VISIONLESS_CONDITIONING_ROUTES,
    SkillExpertConfig,
    normalize_conditioning_route,
)
from .modeling_utils import build_fsq_terminator, build_gemma, load_raw_state_dict
from .modeling_skill_predictor import FrozenVLMSkillPredictor

log = logging.getLogger(__name__)


class SkillExpertPytorch(nn.Module):
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
        self._gradient_checkpointing = False

    @property
    def working_dtype(self) -> torch.dtype:
        return self.action_in_proj.weight.dtype

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


def _map_pi05_key(key: str, *, include_predictor_vlm: bool = False) -> str | None:
    """Map the pi0.5 action prior and, when enabled, its frozen predictor VLM."""
    expert_prefix = "paligemma_with_expert.gemma_expert."
    if key.startswith(expert_prefix):
        suffix = key[len(expert_prefix) :]
        if suffix.startswith("lm_head"):
            return None
        return f"model.gemma_expert.{suffix}"
    vlm_prefix = "paligemma_with_expert.paligemma.model."
    if include_predictor_vlm and key.startswith(vlm_prefix):
        return "model.skill_predictor.vlm." + key.removeprefix(vlm_prefix)
    if key.startswith("paligemma_with_expert."):
        return None
    for projection in (
        "action_in_proj.",
        "action_out_proj.",
        "time_mlp_in.",
        "time_mlp_out.",
    ):
        if key.startswith(projection):
            return f"model.{key}"
    if key.startswith("action_time_mlp_in."):
        return "model.time_mlp_in." + key.removeprefix("action_time_mlp_in.")
    if key.startswith("action_time_mlp_out."):
        return "model.time_mlp_out." + key.removeprefix("action_time_mlp_out.")
    return None


def _build_state_dict(
    raw: dict, *, include_predictor_vlm: bool = False
) -> tuple[dict, bool]:
    is_pi05 = any(key.startswith("paligemma_with_expert.") for key in raw)
    if is_pi05:
        mapped = {}
        for key, value in raw.items():
            mapped_key = _map_pi05_key(
                key, include_predictor_vlm=include_predictor_vlm
            )
            if mapped_key is not None:
                mapped[mapped_key] = value
        predictor_embed = "model.skill_predictor.vlm.language_model.embed_tokens.weight"
        pi05_lm_head = "paligemma_with_expert.paligemma.lm_head.weight"
        if (
            include_predictor_vlm
            and predictor_embed not in mapped
            and pi05_lm_head in raw
        ):
            mapped[predictor_embed] = raw[pi05_lm_head].clone()
        return mapped, True
    return {
        key if key.startswith("model.") else f"model.{key}": value
        for key, value in raw.items()
    }, False


def _load_pretrained_state_dict(
    path: str | Path,
    kwargs: dict,
    *,
    include_predictor_vlm: bool = False,
) -> tuple[dict, bool] | None:
    """Selectively load the pi0.5 action prior and optional frozen predictor VLM."""
    local_path = Path(path)
    safetensors_path = local_path if local_path.is_file() else local_path / "model.safetensors"
    if safetensors_path.is_file():
        from safetensors import safe_open  # noqa: PLC0415

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as checkpoint:
            keys = list(checkpoint.keys())
            is_pi05 = any(key.startswith("paligemma_with_expert.") for key in keys)
            if is_pi05:
                mapped = {}
                for key in keys:
                    mapped_key = _map_pi05_key(
                        key, include_predictor_vlm=include_predictor_vlm
                    )
                    if mapped_key is not None:
                        mapped[mapped_key] = checkpoint.get_tensor(key)
                predictor_embed = (
                    "model.skill_predictor.vlm.language_model.embed_tokens.weight"
                )
                pi05_lm_head = "paligemma_with_expert.paligemma.lm_head.weight"
                if (
                    include_predictor_vlm
                    and predictor_embed not in mapped
                    and pi05_lm_head in keys
                ):
                    mapped[predictor_embed] = checkpoint.get_tensor(pi05_lm_head)
                return mapped, True

    raw = load_raw_state_dict(path, kwargs)
    return None if raw is None else _build_state_dict(
        raw, include_predictor_vlm=include_predictor_vlm
    )


_PREDICTOR_CHECKPOINT_CONTRACT_FIELDS = (
    "skill_vocab_size",
    "skill_fsq_levels",
    "skill_predictor_vlm_variant",
    "skill_predictor_image_size",
    "skill_predictor_reader_tokens",
    "skill_predictor_reader_depth",
    "skill_predictor_reader_heads",
    "skill_predictor_all_layers",
    "skill_predictor_detach_vlm",
    "skill_predictor_lora",
    "skill_predictor_lora_targets",
    "skill_predictor_lora_rank",
    "skill_predictor_lora_alpha",
    "skill_predictor_lora_dropout",
    "skill_predictor_deadzone_frac",
    "skill_predictor_attend_image",
    "skill_predictor_attend_language",
    "tokenizer_max_length",
)


def _is_learned_predictor_key(key: str) -> bool:
    """The pi0.5 predictor base is already loaded; overlay learned Stage-1 parts."""
    return key.startswith(("reader.", "head.")) or ".adapters.skill." in key


def _load_learned_predictor_parameters(
    predictor: FrozenVLMSkillPredictor,
    checkpoint_path: str | Path,
) -> int:
    """Copy only reader/head/skill-LoRA tensors without materializing a 4B checkpoint."""
    from safetensors import safe_open  # noqa: PLC0415

    path = Path(checkpoint_path)
    weights_path = path if path.is_file() else path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Stage-1 predictor weights not found: {weights_path}")

    prefix = "model.skill_predictor."
    target_state = predictor.state_dict()
    expected = {key for key in target_state if _is_learned_predictor_key(key)}
    with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
        source = {
            key.removeprefix(prefix)
            for key in checkpoint.keys()
            if key.startswith(prefix)
            and _is_learned_predictor_key(key.removeprefix(prefix))
        }
        missing = expected - source
        unexpected = source - expected
        if missing or unexpected:
            raise RuntimeError(
                "Stage-1 predictor tensor mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        with torch.no_grad():
            for key in sorted(expected):
                value = checkpoint.get_tensor(prefix + key)
                target = target_state[key]
                if value.shape != target.shape:
                    raise RuntimeError(
                        f"Stage-1 predictor shape mismatch for {key}: "
                        f"checkpoint={tuple(value.shape)}, model={tuple(target.shape)}"
                    )
                target.copy_(value.to(device=target.device, dtype=target.dtype))
    return len(expected)


def _load_complete_predictor_parameters(
    predictor: FrozenVLMSkillPredictor,
    checkpoint_path: str | Path,
) -> int:
    """Load one complete predictor without materializing unrelated Stage-1 tensors."""
    from safetensors import safe_open  # noqa: PLC0415

    path = Path(checkpoint_path)
    weights_path = path if path.is_file() else path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Stage-1 predictor weights not found: {weights_path}")

    prefix = "model.skill_predictor."
    target_state = predictor.state_dict()
    expected = set(target_state)
    with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
        source = {
            key.removeprefix(prefix)
            for key in checkpoint.keys()
            if key.startswith(prefix)
        }
        missing = expected - source
        unexpected = source - expected
        if missing or unexpected:
            raise RuntimeError(
                "Complete Stage-1 predictor tensor mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        with torch.no_grad():
            for key in sorted(expected):
                value = checkpoint.get_tensor(prefix + key)
                target = target_state[key]
                if value.shape != target.shape:
                    raise RuntimeError(
                        f"Stage-1 predictor shape mismatch for {key}: "
                        f"checkpoint={tuple(value.shape)}, model={tuple(target.shape)}"
                    )
                target.copy_(value.to(device=target.device, dtype=target.dtype))
    return len(expected)


def _load_complete_terminator_parameters(
    terminator: nn.Module,
    checkpoint_path: str | Path,
) -> int:
    """Load one complete co-trained terminator without unrelated Stage-1 tensors."""
    from safetensors import safe_open  # noqa: PLC0415

    path = Path(checkpoint_path)
    weights_path = path if path.is_file() else path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Stage-1 terminator weights not found: {weights_path}")

    prefix = "model.fsq_term_train."
    target_state = terminator.state_dict()
    expected = set(target_state)
    with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
        source = {
            key.removeprefix(prefix)
            for key in checkpoint.keys()
            if key.startswith(prefix)
        }
        missing = expected - source
        unexpected = source - expected
        if missing or unexpected:
            raise RuntimeError(
                "Complete Stage-1 terminator tensor mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        with torch.no_grad():
            for key in sorted(expected):
                value = checkpoint.get_tensor(prefix + key)
                target = target_state[key]
                if value.shape != target.shape:
                    raise RuntimeError(
                        f"Stage-1 terminator shape mismatch for {key}: "
                        f"checkpoint={tuple(value.shape)}, model={tuple(target.shape)}"
                    )
                target.copy_(value.to(device=target.device, dtype=target.dtype))
    return len(expected)


class SkillExpertPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for the Stage-1 VSA prior."""

    config_class = SkillExpertConfig
    name = "skill_expert"

    def __init__(self, config: SkillExpertConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = SkillExpertPytorch(config)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(device=config.device, dtype=self._torch_dtype())
        if self.model.fsq_term_train is not None:
            # Match FSQ training/inference numerics; this auxiliary remains fp32.
            self.model.fsq_term_train.to(dtype=torch.float32)
        self.reset()

    def _torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32

    def reset(self) -> None:
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def _initialize_frozen_skill_predictor(
        self, checkpoint_path: str | Path | None
    ) -> None:
        predictor = self.model.skill_predictor
        if predictor is None:
            raise RuntimeError("Predicted-skill training requires a predictor module.")
        path = Path(str(checkpoint_path or ""))
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Stage-1 predictor config not found: {config_path}")
        source_config = json.loads(config_path.read_text())
        if source_config.get("type") != "skill_expert":
            raise ValueError(
                "Predictor source must be a skill_expert checkpoint, got "
                f"{source_config.get('type')!r}."
            )
        if not source_config.get("train_skill_predictor", False):
            raise ValueError("Stage-1 predictor source has no trained predictor.")
        mismatches = [
            f"{field}: checkpoint={source_config.get(field)!r}, "
            f"current={getattr(self.config, field)!r}"
            for field in _PREDICTOR_CHECKPOINT_CONTRACT_FIELDS
            if source_config.get(field) != getattr(self.config, field)
        ]
        if mismatches:
            raise ValueError(
                "Stage-1 predictor architecture mismatch: " + "; ".join(mismatches)
            )
        loaded = _load_learned_predictor_parameters(predictor, path)
        predictor.requires_grad_(False).eval()
        log.info(
            "Stage 1 <- frozen predictor %s: loaded %d learned tensors.",
            path,
            loaded,
        )

    def load_external_skill_predictor(
        self, checkpoint_path: str | Path | None
    ) -> None:
        """Override an existing predictor or attach one to a predictor-free VSA."""
        if self.model.skill_predictor is not None:
            self._initialize_frozen_skill_predictor(checkpoint_path)
            return

        path = Path(str(checkpoint_path or ""))
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Stage-1 predictor config not found: {config_path}")
        source_config = json.loads(config_path.read_text())
        if source_config.get("type") != "skill_expert":
            raise ValueError(
                "Predictor source must be a skill_expert checkpoint, got "
                f"{source_config.get('type')!r}."
            )
        if not source_config.get("train_skill_predictor", False):
            raise ValueError("Stage-1 predictor source has no trained predictor.")
        mismatches = [
            f"{field}: checkpoint={source_config.get(field)!r}, "
            f"current={getattr(self.config, field)!r}"
            for field in _PREDICTOR_CHECKPOINT_CONTRACT_FIELDS
            if source_config.get(field) != getattr(self.config, field)
        ]
        if mismatches:
            raise ValueError(
                "Stage-1 predictor architecture mismatch: " + "; ".join(mismatches)
            )

        predictor = FrozenVLMSkillPredictor(self.config).to(dtype=self._torch_dtype())
        loaded = _load_complete_predictor_parameters(predictor, path)
        device = next(self.model.parameters()).device
        predictor.to(device=device)
        predictor.requires_grad_(False).eval()
        self.model.skill_predictor = predictor
        log.info(
            "Stage 1 <- attached external predictor %s: loaded %d tensors.",
            path,
            loaded,
        )

    def load_external_terminator(
        self, checkpoint_path: str | Path | None
    ) -> None:
        """Attach or override the co-trained terminator used during evaluation."""
        path = Path(str(checkpoint_path or ""))
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Stage-1 terminator config not found: {config_path}")
        source_config = json.loads(config_path.read_text())
        if source_config.get("type") != "skill_expert":
            raise ValueError(
                "Terminator source must be a skill_expert checkpoint, got "
                f"{source_config.get('type')!r}."
            )
        if not source_config.get("train_terminator", False):
            raise ValueError("Stage-1 terminator source has no trained terminator.")
        if source_config.get("skill_fsq_levels") != self.config.skill_fsq_levels:
            raise ValueError(
                "Stage-1 terminator FSQ mismatch: "
                f"checkpoint={source_config.get('skill_fsq_levels')!r}, "
                f"current={self.config.skill_fsq_levels!r}."
            )

        terminator = self.model.fsq_term_train
        if terminator is None:
            terminator = build_fsq_terminator(
                self.config.fsq_path,
                dino_model_path=self.config.terminator_dino_model_path,
            ).to(dtype=torch.float32)
        loaded = _load_complete_terminator_parameters(terminator, path)
        device = next(self.model.parameters()).device
        terminator.to(device=device, dtype=torch.float32)
        terminator.requires_grad_(False).eval()
        self.model.fsq_term_train = terminator
        log.info(
            "Stage 1 <- external terminator %s: loaded %d tensors.",
            path,
            loaded,
        )

    def get_optim_params(self) -> list[dict]:
        """VSA groups plus a separate-LR terminator; predictor stays excluded."""
        terminator = getattr(self.model, "fsq_term_train", None)
        terminator_parameters = (
            [parameter for parameter in terminator.parameters() if parameter.requires_grad]
            if terminator is not None
            else []
        )
        excluded_ids = {id(parameter) for parameter in terminator_parameters}

        dino_parameters = []
        if (
            self.model.dino is not None
            and not self.config.freeze_vision_encoder
            and self.config.dino_lr is not None
        ):
            dino_parameters = [
                parameter
                for parameter in self.model.dino.parameters()
                if parameter.requires_grad
            ]
            excluded_ids.update(id(parameter) for parameter in dino_parameters)

        base_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in excluded_ids
        ]
        groups = [{"params": base_parameters}] if base_parameters else []
        if dino_parameters:
            groups.append({"params": dino_parameters, "lr": self.config.dino_lr})
        if terminator_parameters:
            groups.append(
                {
                    "params": terminator_parameters,
                    "lr": self.config.optimizer_lr
                    * self.config.terminator_lr_scale,
                }
            )
        return groups

    def isolated_main_optimizer_grad_groups(self) -> dict[str, list[nn.Parameter]]:
        """Clip terminator gradients independently from the disjoint VSA graph."""
        terminator = getattr(self.model, "fsq_term_train", None)
        if not getattr(self.config, "train_terminator", False) or terminator is None:
            return {}
        params = [parameter for parameter in terminator.parameters() if parameter.requires_grad]
        return {"terminator": params} if params else {}

    def _collect_images(self, batch: dict) -> list[Tensor]:
        device = next(self.parameters()).device
        present = [key for key in self.config.image_features if key in batch]
        if not present:
            raise ValueError(
                f"No image features in batch; expected one of {list(self.config.image_features)}."
            )
        images = []
        for key in present:
            image = batch[key].to(device=device).float()
            if image.ndim == 4 and image.shape[1] != 3 and image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            images.append(image)
        return images

    def _skill_code(self, batch: dict) -> Tensor:
        # SkillVLADataset already resolves one coherent transition-jitter draw for
        # skill-start images/state and its target code. Reuse that same code for VSA.
        if "skill_code" in batch:
            code = batch["skill_code"].view(-1).long()
            if "skill_sequence" in batch and "skill_index" in batch:
                sequence = batch["skill_sequence"].long()
                current_index = batch["skill_index"].long().view(-1).clamp(
                    0, sequence.shape[1] - 1
                )
                current_code = sequence.gather(1, current_index[:, None]).squeeze(1)
                self._last_transition_jitter_fraction = (
                    code != current_code
                ).float().mean()
            else:
                self._last_transition_jitter_fraction = torch.zeros(
                    (), device=code.device
                )
            return code.clamp(0, self.config.skill_vocab_size - 1)

        # Compatibility fallback for raw frame datasets and direct policy tests.
        sequence = batch["skill_sequence"].long()
        index = batch["skill_index"].long().reshape(-1)
        index = index.clamp(0, sequence.shape[1] - 1)
        if self.training and self.config.transition_jitter_pmax > 0:
            from lerobot.policies.skillVLA.skill_jitter import choose_jitter_torch  # noqa: PLC0415

            original_index = index
            index, _ = choose_jitter_torch(
                index,
                batch["skill_ds"].long().reshape(-1),
                batch["skill_de"].long().reshape(-1),
                batch["skill_sequence_len"].long().reshape(-1),
                self.config.transition_jitter_pmax,
                distribution=self.config.transition_jitter_distribution,
            )
            index = index.clamp(0, sequence.shape[1] - 1)
            self._last_transition_jitter_fraction = (index != original_index).float().mean()
        else:
            self._last_transition_jitter_fraction = torch.zeros((), device=index.device)
        return sequence.gather(1, index[:, None]).squeeze(1).clamp(
            0, self.config.skill_vocab_size - 1
        )

    def _predictor_start_images(self, batch: dict) -> list[Tensor]:
        device = next(self.parameters()).device
        images = []
        for key in ("skill_start_image", "skill_start_wrist_image"):
            if key not in batch:
                raise ValueError(
                    f"Missing {key!r}; Stage-1 predictor training requires SkillVLADataset."
                )
            image = batch[key].to(device=device).float()
            if image.ndim == 4 and image.shape[1] != 3 and image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            images.append(image)
        return images

    def _skill_predictor_loss(self, batch: dict) -> tuple[Tensor, float]:
        predictor = self.model.skill_predictor
        if predictor is None:
            raise RuntimeError("Skill predictor is disabled.")
        target = batch["skill_code"].to(next(self.parameters()).device).view(-1).long()
        target = target.clamp(0, self.config.skill_vocab_size - 1)
        return predictor.loss(
            self._predictor_start_images(batch),
            batch[OBS_LANGUAGE_TOKENS].to(target.device),
            batch[OBS_LANGUAGE_ATTENTION_MASK].to(target.device),
            target,
        )

    @torch.no_grad()
    def _predicted_training_skill_code(self, batch: dict) -> Tensor:
        """Predict the held skill from the dataset's jittered skill-start view."""
        predictor = self.model.skill_predictor
        if predictor is None:
            raise RuntimeError("Predicted-skill training has no loaded predictor.")
        device = next(self.parameters()).device
        return predictor.predict(
            self._predictor_start_images(batch),
            batch[OBS_LANGUAGE_TOKENS].to(device),
            batch[OBS_LANGUAGE_ATTENTION_MASK].to(device),
        ).view(-1).long()

    def _training_skill_code(self, batch: dict) -> Tensor:
        # Resolve the coherent post-jitter GT only as an audit label and for the
        # legacy route.  Predictor mode never feeds this label to the VSA.
        jittered_gt = self._skill_code(batch)
        self._last_predicted_skill_accuracy = None
        self._last_predicted_diff_from_current = None
        self._last_unique_predicted_skills = None
        if getattr(self.config, "training_skill_source", "gt") == "gt":
            return jittered_gt

        predicted = self._predicted_training_skill_code(batch).clamp(
            0, self.config.skill_vocab_size - 1
        )
        jittered_gt = jittered_gt.to(predicted.device)
        self._last_predicted_skill_accuracy = (
            predicted == jittered_gt
        ).float().mean()
        self._last_unique_predicted_skills = torch.unique(predicted).numel()
        if "skill_code_true" in batch:
            current_gt = batch["skill_code_true"].to(predicted.device).view(-1).long()
            self._last_predicted_diff_from_current = (
                predicted != current_gt
            ).float().mean()
        return predicted

    @torch.no_grad()
    def predict_skill_code(self, batch: dict) -> Tensor:
        """Predict a skill from the current observation at a runtime boundary."""
        predictor = self.model.skill_predictor
        if predictor is None:
            raise RuntimeError(
                "This checkpoint has no loaded Stage-1 skill predictor."
            )
        device = next(self.parameters()).device
        return predictor.predict(
            self._collect_images(batch),
            batch[OBS_LANGUAGE_TOKENS].to(device),
            batch[OBS_LANGUAGE_ATTENTION_MASK].to(device),
        ).long()

    def _terminator_loss(self, batch: dict) -> tuple[Tensor, Tensor, Tensor]:
        """Stage-0-matched progress SmoothL1 + soft end BCE on current raw obs."""
        required = (
            "skill_code_true",
            "skill_ds",
            "skill_de",
            "skill_decoder_state",
            "observation.images.image",
            "observation.images.wrist_image",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(
                "train_terminator=True requires SkillVLADataset and the raw-state "
                f"processor; missing={missing}."
            )
        true_code = batch["skill_code_true"].view(-1).long().clamp(
            0, self.config.skill_vocab_size - 1
        )
        progress_prediction, termination_logits = self.model.terminator_predict(
            true_code,
            batch["skill_decoder_state"],
            batch["observation.images.image"],
            batch["observation.images.wrist_image"],
        )
        distance_from_start = batch["skill_ds"].float().view(-1).to(
            progress_prediction.device
        )
        distance_to_end = batch["skill_de"].float().view(-1).to(
            progress_prediction.device
        )
        progress_target = (
            distance_from_start
            / (distance_from_start + distance_to_end).clamp_min(1.0)
        ).clamp(0.0, 1.0)
        sigma = self.config.terminator_end_target_sigma
        termination_target = (
            torch.exp(-(distance_to_end.square()) / (2.0 * sigma**2))
            if sigma > 0
            else (distance_to_end == 0).float()
        )
        progress_loss = F.smooth_l1_loss(
            progress_prediction, progress_target.to(progress_prediction.dtype)
        )
        positive_weight = torch.tensor(
            self.config.terminator_end_pos_weight,
            device=termination_logits.device,
            dtype=termination_logits.dtype,
        )
        termination_loss = F.binary_cross_entropy_with_logits(
            termination_logits,
            termination_target.to(termination_logits.dtype),
            pos_weight=positive_weight,
        )
        return progress_loss + termination_loss, progress_loss, termination_loss

    def isolated_auxiliary_step(
        self,
        batch: dict,
        accelerator,
        grad_clip_norm: float,
        current_lr: float | None = None,
    ) -> dict:
        """Train only skill LoRA/reader/head after the isolated VSA optimizer step."""
        predictor = self.model.skill_predictor
        if not self.config.train_skill_predictor or predictor is None:
            return {}
        params = predictor.auxiliary_parameters()
        if not hasattr(self, "_skill_predictor_optimizer"):
            reader_head = predictor.reader_head_parameters()
            lora = predictor.lora_parameters()
            parameter_groups = [
                {
                    "params": reader_head,
                    "lr": self.config.optimizer_lr
                    * self.config.skill_predictor_lr_scale,
                    "lr_scale": self.config.skill_predictor_lr_scale,
                    "group_name": "reader_head",
                }
            ]
            if lora:
                parameter_groups.append(
                    {
                        "params": lora,
                        "lr": self.config.optimizer_lr
                        * self.config.skill_predictor_lora_lr_scale,
                        "lr_scale": self.config.skill_predictor_lora_lr_scale,
                        "group_name": "skill_lora",
                    }
                )
            self._skill_predictor_optimizer = torch.optim.AdamW(
                parameter_groups,
                betas=self.config.optimizer_betas,
                eps=self.config.optimizer_eps,
                weight_decay=self.config.optimizer_weight_decay,
            )
        optimizer = self._skill_predictor_optimizer
        if current_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = current_lr * float(group["lr_scale"])

        previous_requires_grad = [parameter.requires_grad for parameter in params]
        for parameter in params:
            parameter.requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        try:
            with accelerator.autocast():
                raw_loss, accuracy = self._skill_predictor_loss(batch)
                objective = self.config.skill_predictor_weight * raw_loss
            accelerator.backward(objective)
            if grad_clip_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(params, grad_clip_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params, float("inf"), error_if_nonfinite=False
                )
            optimizer.step()
            return {
                "skill_predictor/loss": raw_loss.detach().item(),
                "skill_predictor/objective_loss": objective.detach().item(),
                "skill_predictor/skill_acc": float(accuracy),
                "skill_predictor/weight": self.config.skill_predictor_weight,
                "skill_predictor/grad_norm": float(
                    grad_norm.detach().item()
                    if torch.is_tensor(grad_norm)
                    else grad_norm
                ),
                "skill_predictor/lr": optimizer.param_groups[0]["lr"],
                "skill_predictor/lora_lr": next(
                    (
                        group["lr"]
                        for group in optimizer.param_groups
                        if group.get("group_name") == "skill_lora"
                    ),
                    0.0,
                ),
                "skill_predictor/lora_layers": float(predictor.lora_layer_count),
                "skill_predictor/all_layers": float(
                    self.config.skill_predictor_all_layers
                ),
                "skill_predictor/deadzone_frac": float(
                    self.config.skill_predictor_deadzone_frac
                ),
            }
        finally:
            optimizer.zero_grad(set_to_none=True)
            for parameter, old_value in zip(
                params, previous_requires_grad, strict=True
            ):
                parameter.requires_grad_(old_value)

    @staticmethod
    def _valid_action_steps(actions: Tensor, batch: dict) -> Tensor:
        """Match Stage-0 unconditional supervision: mask episode padding only.

        A chunk may cross a skill boundary. Those tail actions remain real dataset
        targets even though the conditioning skill is the one active at the chunk
        start; ``skill_de`` must therefore not shorten or rewrite the target.
        """
        valid = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        if "action_is_pad" in batch:
            valid &= ~batch["action_is_pad"].to(actions.device).bool()
        return valid

    @staticmethod
    def _endpoint_xyz_loss(
        predicted_actions: Tensor,
        target_actions: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Stage0 Exp4-1 endpoint: MSE of accumulated valid XYZ deltas.

        Errors may cancel across intermediate timesteps; only the final chunk
        displacement matters. Every sample has equal weight, irrespective of
        the number of valid steps in its episode-clipped chunk.
        """
        if predicted_actions.shape[-1] < 3 or target_actions.shape[-1] < 3:
            raise ValueError("endpoint_xyz loss requires at least three action dimensions.")
        sample_valid = valid.any(dim=1)
        if not bool(sample_valid.any()):
            raise ValueError("endpoint_xyz loss received a batch with no valid action steps.")
        step_valid = valid.to(predicted_actions.dtype).unsqueeze(-1)
        endpoint_error = (
            (predicted_actions[..., :3] - target_actions[..., :3]) * step_valid
        ).sum(dim=1)
        per_sample = endpoint_error.square().mean(dim=-1)
        selected = sample_valid.to(per_sample.dtype)
        loss = (per_sample * selected).sum() / selected.sum().clamp(min=1.0)
        return loss, per_sample

    def forward(self, batch: dict, reduction: str = "mean"):
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        real_dim = self.config.output_features[ACTION].shape[0]
        route = normalize_conditioning_route(
            getattr(self.config, "conditioning_route", "state_cond")
        )
        state = (
            None
            if route in STATELESS_CONDITIONING_ROUTES
            else pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        )
        if route in SKILLLESS_CONDITIONING_ROUTES:
            skill_code = None
            self._last_transition_jitter_fraction = torch.zeros(
                (), device=actions.device
            )
            self._last_predicted_skill_accuracy = None
            self._last_predicted_diff_from_current = None
            self._last_unique_predicted_skills = None
        else:
            skill_code = self._training_skill_code(batch)
        images = (
            []
            if route in VISIONLESS_CONDITIONING_ROUTES
            else self._collect_images(batch)
        )
        residual = self.model(images, state, skill_code, actions)[..., :real_dim]
        squared_error = residual.square()
        valid = self._valid_action_steps(actions, batch)
        valid_float = valid.to(squared_error.dtype).unsqueeze(-1)
        valid_per_sample = valid.sum(dim=1).clamp(min=1).to(squared_error.dtype)
        per_sample = (squared_error * valid_float).sum(dim=(1, 2)) / (
            valid_per_sample * real_dim
        )
        valid_steps = valid.sum().clamp(min=1).to(squared_error.dtype)
        action_loss = (squared_error * valid_float).sum() / (valid_steps * real_dim)
        loss_per_dim = (squared_error * valid_float).sum(dim=(0, 1)) / valid_steps
        endpoint_loss = None
        action_objective = action_loss
        objective_per_sample = per_sample
        if self.config.action_loss_mode == "flow_endpoint_xyz":
            predicted_actions = self.model._last_predicted_actions
            if predicted_actions is None:
                raise RuntimeError(
                    "flow_endpoint_xyz did not receive reconstructed predicted actions."
                )
            endpoint_loss, endpoint_per_sample = self._endpoint_xyz_loss(
                predicted_actions[..., :real_dim],
                actions[..., :real_dim],
                valid,
            )
            action_objective = 0.5 * action_loss + 0.5 * endpoint_loss
            objective_per_sample = 0.5 * per_sample + 0.5 * endpoint_per_sample
        loss_dict = {
            "action_loss": action_loss.detach().item(),
            "loss_per_dim": loss_per_dim.detach().cpu().tolist(),
            "conditioning/skill_source_predictor": float(
                getattr(self.config, "training_skill_source", "gt") == "predictor"
            ),
            "regime/transition_jitter_fraction": self._last_transition_jitter_fraction.detach().item(),
        }
        if self._last_predicted_skill_accuracy is not None:
            loss_dict["conditioning/predictor_acc_vs_jittered_gt"] = (
                self._last_predicted_skill_accuracy.detach().item()
            )
            loss_dict["conditioning/unique_predicted_skills"] = float(
                self._last_unique_predicted_skills
            )
        if self._last_predicted_diff_from_current is not None:
            loss_dict["conditioning/predicted_diff_from_current_gt"] = (
                self._last_predicted_diff_from_current.detach().item()
            )
        if endpoint_loss is not None:
            loss_dict.update(
                {
                    "endpoint_xyz_loss": endpoint_loss.detach().item(),
                    "action_objective": action_objective.detach().item(),
                    "action_flow_weight": 0.5,
                    "action_endpoint_weight": 0.5,
                }
            )
        terminator_loss = None
        if self.config.train_terminator and self.model.fsq_term_train is not None:
            terminator_loss, progress_loss, termination_loss = self._terminator_loss(batch)
            loss_dict.update(
                {
                    "terminator/loss": terminator_loss.detach().item(),
                    "terminator/progress": progress_loss.detach().item(),
                    "terminator/termination": termination_loss.detach().item(),
                }
            )
        if reduction == "none":
            return objective_per_sample, loss_dict
        total = action_objective
        if terminator_loss is not None:
            total = total + terminator_loss
            loss_dict["loss_total"] = total.detach().item()
        return total, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **kwargs) -> Tensor:
        self.eval()
        route = normalize_conditioning_route(
            getattr(self.config, "conditioning_route", "state_cond")
        )
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
        images = (
            []
            if route in VISIONLESS_CONDITIONING_ROUTES
            else self._collect_images(batch)
        )
        actions = self.model.sample_actions(images, state, skill_code, **kwargs)
        real_dim = self.config.output_features[ACTION].shape[0]
        return actions[..., :real_dim]

    @torch.no_grad()
    def select_action(self, batch: dict, **kwargs) -> Tensor:
        self.eval()
        if not self._action_queue:
            actions = self.predict_action_chunk(batch, **kwargs)
            self._action_queue.extend(
                actions[:, : self.config.n_action_steps].transpose(0, 1)
            )
        return self._action_queue.popleft()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path,
        *,
        config=None,
        strict: bool = False,
        **kwargs,
    ):
        """Load either the pi0.5 expert prior or a complete Stage-1 checkpoint."""
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        policy = cls(config, **kwargs)
        loaded = _load_pretrained_state_dict(
            pretrained_name_or_path,
            kwargs,
            include_predictor_vlm=config.uses_skill_predictor,
        )
        if loaded is None:
            raise FileNotFoundError(f"Could not load Stage-1 initialization: {pretrained_name_or_path}")

        state_dict, is_pi05 = loaded
        if is_pi05 and not any(key.startswith("model.gemma_expert.") for key in state_dict):
            raise RuntimeError("The pi0.5 checkpoint contains no action-expert weights.")
        if is_pi05:
            state_dict, routed = route_plain_to_base(
                state_dict, set(policy.state_dict())
            )
            if routed:
                log.info(
                    "Stage-1 pi0.5 initialization routed %d predictor VLM tensors "
                    "into LoRA base projections.",
                    routed,
                )
        if is_pi05 and config.uses_skill_predictor:
            expected_vlm = {
                f"model.skill_predictor.vlm.{key}"
                for key in policy.model.skill_predictor.vlm.state_dict()
                if ".adapters." not in key
            }
            missing_vlm = expected_vlm - set(state_dict)
            if missing_vlm:
                raise RuntimeError(
                    "The pi0.5 checkpoint is incomplete for the frozen predictor VLM; "
                    f"missing={sorted(missing_vlm)[:20]}"
                )
        state_dict = {
            key: value.to(policy._torch_dtype()) for key, value in state_dict.items()
        }
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        if not is_pi05 and (missing or unexpected):
            raise RuntimeError(
                "Stage-1 checkpoint mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        if is_pi05 and config.training_skill_source == "predictor":
            policy._initialize_frozen_skill_predictor(
                config.skill_predictor_checkpoint_path
            )
        source = "pi0.5 action expert" if is_pi05 else "Stage-1 checkpoint"
        log.info(
            "Stage 1 <- %s: loaded=%d, fresh=%d, unexpected=%d.",
            source,
            len(state_dict),
            len(missing),
            len(unexpected),
        )
        return policy
