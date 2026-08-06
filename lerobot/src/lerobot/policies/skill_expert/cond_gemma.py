"""Condition-Gemma Stage-1 architecture and its state-location ablations.

Arch0 preserves the skillVLA_real two-stream implementation. Arch0_1--0_3
change where projected state enters Cond/Expert AdaRMS; Arch0_2_sep alone
removes projection sharing while preserving an identical initial function.
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
from transformers.cache_utils import DynamicCache
from transformers.models.gemma import modeling_gemma

from lerobot.policies.pi05.modeling_pi05 import (
    OPENPI_ATTENTION_MASK_VALUE,
    compute_layer_complete,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    layernorm_forward,
    make_att_2d_masks,
    sample_beta,
)
from lerobot.policies.pi_gemma import PiGemmaRMSNorm, _gated_residual

from .configuration_skill_expert import (
    COND_GEMMA_EXPERT_TOKENS_REVISION,
    COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION,
    COND_GEMMA_SEPARATE_DUAL_STATE_REVISION,
    COND_GEMMA_WRIST_DUAL_STATE_REVISION,
    COND_STATE_ADARMS_REVISIONS,
    EXPERT_STATE_ADARMS_REVISIONS,
    SKILLLESS_CONDITIONING_ROUTES,
    STATELESS_CONDITIONING_ROUTES,
    VISIONLESS_CONDITIONING_ROUTES,
    SkillExpertConfig,
)
from .modeling_skill_predictor import FrozenVLMSkillPredictor
from .modeling_utils import build_fsq_terminator, build_gemma
from .vsa_perceiver_crossattn import CameraPerceiverResampler


EXPERT_TOKEN_REVISIONS = frozenset(
    {
        COND_GEMMA_EXPERT_TOKENS_REVISION,
        COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION,
    }
)


def expert_token_attention_contract(
    batch_size: int,
    visual_tokens: int,
    action_tokens: int,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """Build the fixed [visual | state, skill | actions] mask and positions."""
    total = visual_tokens + 2 + action_tokens
    padding = torch.ones(batch_size, total, dtype=torch.bool, device=device)
    block_starts = (
        [0] * visual_tokens + [1, 0] + [1] + [0] * (action_tokens - 1)
    )
    blocks = torch.tensor(block_starts, dtype=torch.bool, device=device)[None].expand(
        batch_size, -1
    )
    attention = make_att_2d_masks(padding, blocks)[:, None]
    attention = torch.where(attention, 0.0, OPENPI_ATTENTION_MASK_VALUE)
    positions = torch.cumsum(padding, dim=1) - 1
    return attention, positions


def _mixed_visual_expert_layer(
    layer_index: int,
    streams: list[Tensor],
    attention_mask: Tensor,
    position_ids: Tensor,
    expert_condition: Tensor | None,
    *,
    cond_encoder,
    gemma_expert,
    context_input_norm: nn.Module,
    context_post_attention_norm: nn.Module,
) -> tuple[list[Tensor], Tensor, Tensor]:
    """Run one joint layer over visual, expert-context, and optional action streams.

    The visual stream uses the Cond-Gemma block. State/skill and actions share
    the pi0.5 expert attention/MLP weights, but only actions use timestep AdaRMS.
    Returned K/V are already RoPE-encoded and can be retained as an inference
    prefix when ``streams`` contains visual and context only.
    """
    if len(streams) not in {2, 3}:
        raise ValueError(f"Expected 2 or 3 streams, got {len(streams)}.")
    if len(streams) == 3 and expert_condition is None:
        raise ValueError("The action stream requires a timestep condition.")

    cond_layer = cond_encoder.model.layers[layer_index]
    expert_layer = gemma_expert.model.layers[layer_index]
    normalized: list[Tensor] = []
    gates: list[Tensor | None] = []

    hidden, gate = layernorm_forward(cond_layer.input_layernorm, streams[0], None)
    normalized.append(hidden)
    gates.append(gate)
    hidden, gate = layernorm_forward(context_input_norm, streams[1], None)
    normalized.append(hidden)
    gates.append(gate)
    if len(streams) == 3:
        hidden, gate = layernorm_forward(
            expert_layer.input_layernorm, streams[2], expert_condition
        )
        normalized.append(hidden)
        gates.append(gate)

    layers = [cond_layer, expert_layer, expert_layer]
    query_states: list[Tensor] = []
    key_states: list[Tensor] = []
    value_states: list[Tensor] = []
    for hidden, layer in zip(normalized, layers, strict=False):
        input_shape = hidden.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_states.append(
            layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
        )
        key_states.append(
            layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
        )
        value_states.append(
            layer.self_attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
        )

    query = torch.cat(query_states, dim=2)
    key = torch.cat(key_states, dim=2)
    value = torch.cat(value_states, dim=2)
    rotary_input = torch.zeros(
        query.shape[0],
        query.shape[2],
        query.shape[-1],
        device=query.device,
        dtype=query.dtype,
    )
    cos, sin = cond_encoder.model.rotary_emb(rotary_input, position_ids)
    query, key = modeling_gemma.apply_rotary_pos_emb(
        query, key, cos, sin, unsqueeze_dim=1
    )
    attention_output, _ = modeling_gemma.eager_attention_forward(
        cond_layer.self_attn,
        query,
        key,
        value,
        attention_mask,
        scaling=cond_layer.self_attn.scaling,
    )
    attention_output = attention_output.reshape(
        query.shape[0], -1, query.shape[1] * query.shape[-1]
    )

    outputs: list[Tensor] = []
    start = 0
    for stream_index, (residual, layer, gate) in enumerate(
        zip(streams, layers, gates, strict=False)
    ):
        end = start + residual.shape[1]
        projected = layer.self_attn.o_proj(
            attention_output[:, start:end].to(layer.self_attn.o_proj.weight.dtype)
        )
        hidden = _gated_residual(residual, projected, gate)
        post_residual = hidden
        if stream_index == 0:
            hidden, mlp_gate = layernorm_forward(
                cond_layer.post_attention_layernorm, hidden, None
            )
        elif stream_index == 1:
            hidden, mlp_gate = layernorm_forward(
                context_post_attention_norm, hidden, None
            )
        else:
            hidden, mlp_gate = layernorm_forward(
                expert_layer.post_attention_layernorm, hidden, expert_condition
            )
        hidden = layer.mlp(hidden.to(layer.mlp.up_proj.weight.dtype))
        outputs.append(_gated_residual(post_residual, hidden, mlp_gate))
        start = end
    return outputs, key, value


def compute_expert_token_layer(
    layer_index: int,
    streams: list[Tensor],
    attention_mask: Tensor,
    position_ids: Tensor,
    expert_condition: Tensor,
    *,
    cond_encoder,
    gemma_expert,
    context_input_norm: nn.Module,
    context_post_attention_norm: nn.Module,
) -> list[Tensor]:
    """Checkpoint-friendly wrapper for the three-stream training layer."""
    outputs, _, _ = _mixed_visual_expert_layer(
        layer_index,
        streams,
        attention_mask,
        position_ids,
        expert_condition,
        cond_encoder=cond_encoder,
        gemma_expert=gemma_expert,
        context_input_norm=context_input_norm,
        context_post_attention_norm=context_post_attention_norm,
    )
    return outputs


class CondGemmaSkillExpert(nn.Module):
    """DINO condition stream + fully trainable pi0.5 action expert."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__()
        self.config = config
        expert_config = get_gemma_config(config.action_expert_variant)
        self.width = expert_config.width
        self.uses_expert_context_tokens = (
            config.architecture_revision in EXPERT_TOKEN_REVISIONS
        )
        self.uses_visual_perceiver = (
            config.architecture_revision
            == COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION
        )
        self.uses_cond_state_adarms = (
            config.architecture_revision in COND_STATE_ADARMS_REVISIONS
            and config.conditioning_route not in STATELESS_CONDITIONING_ROUTES
        )
        self.uses_expert_state_adarms = (
            config.architecture_revision in EXPERT_STATE_ADARMS_REVISIONS
        )
        self.uses_separate_state_projections = (
            config.architecture_revision
            == COND_GEMMA_SEPARATE_DUAL_STATE_REVISION
        )
        self.uses_wrist_only_cond_state = (
            config.architecture_revision == COND_GEMMA_WRIST_DUAL_STATE_REVISION
        )

        if config.conditioning_route in VISIONLESS_CONDITIONING_ROUTES:
            # The transformer still needs a condition sequence to carry state
            # AdaRMS and skill broadcasts. This learned seed contains no
            # observation information and replaces all vision tokens.
            self.dino = None
            self.n_register_tokens = 0
            self.image_proj = None
            self.top_resampler = None
            self.wrist_resampler = None
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
            dino_width = int(self.dino.config.hidden_size)
            self.image_proj = (
                None
                if self.uses_visual_perceiver
                else nn.Linear(dino_width, self.width)
            )
            if self.uses_visual_perceiver:
                self.top_resampler = CameraPerceiverResampler(
                    dino_width,
                    self.width,
                    perceiver_width=config.visual_perceiver_width,
                    num_latents=config.num_visual_latents_per_camera,
                )
                self.wrist_resampler = CameraPerceiverResampler(
                    dino_width,
                    self.width,
                    perceiver_width=config.visual_perceiver_width,
                    num_latents=config.num_visual_latents_per_camera,
                )
            else:
                self.top_resampler = None
                self.wrist_resampler = None
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
        # Arch0_2_sep is identical to Arch0_2 except for this independent
        # Expert-side state embedding. Start from an exact copy so the probe
        # isolates gradient sharing rather than initialization differences.
        # ``state_proj`` remains the Cond path.
        self.expert_state_proj = (
            copy.deepcopy(self.state_proj)
            if self.uses_separate_state_projections
            else None
        )
        # Skill-free routes likewise omit the otherwise unused projection.
        self.skill_proj = (
            None
            if config.conditioning_route in SKILLLESS_CONDITIONING_ROUTES
            else nn.Linear(len(config.skill_fsq_levels), self.width)
        )
        self.state_norm = (
            PiGemmaRMSNorm(self.width) if self.uses_expert_context_tokens else None
        )
        self.skill_norm = (
            PiGemmaRMSNorm(self.width) if self.uses_expert_context_tokens else None
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
            use_adarms=self.uses_cond_state_adarms,
        )
        self.gemma_expert = build_gemma(config.action_expert_variant, use_adarms=True)
        if self.uses_expert_context_tokens:
            eps = float(self.gemma_expert.model.config.rms_norm_eps)
            self.context_input_norms = nn.ModuleList(
                [PiGemmaRMSNorm(self.width, eps=eps) for _ in range(expert_config.depth)]
            )
            self.context_post_attention_norms = nn.ModuleList(
                [PiGemmaRMSNorm(self.width, eps=eps) for _ in range(expert_config.depth)]
            )
        else:
            self.context_input_norms = nn.ModuleList()
            self.context_post_attention_norms = nn.ModuleList()
        self.skill_predictor = (
            FrozenVLMSkillPredictor(config) if config.uses_skill_predictor else None
        )
        self.fsq_term_train = None
        self.fsq_image_term_train = None
        if config.train_terminator:
            terminator = build_fsq_terminator(config.fsq_path)
            if config.terminator_freeze_vision_encoder is not None:
                terminator.freeze_vision_encoder = bool(
                    config.terminator_freeze_vision_encoder
                )
            terminator.requires_grad_(True).train()
            if terminator.freeze_vision_encoder:
                terminator.vision_encoder.requires_grad_(False).eval()
            self.fsq_term_train = terminator.to(dtype=torch.float32)
        self._last_predicted_actions: Tensor | None = None
        self._last_flow_time: Tensor | None = None
        # The current trainer calls this common observability surface.  Keeping
        # it empty adds no work to the original skillVLA_real forward path.
        self._last_vsa_debug_stats: dict[str, float] = {}
        self._vsa_training_step: int | None = None
        self._vsa_debug_active = False
        self._gradient_checkpointing = False

    @property
    def working_dtype(self) -> torch.dtype:
        return self.action_in_proj.weight.dtype

    def set_training_step(self, step: int) -> None:
        self._vsa_training_step = int(step)
        scheduled = self._vsa_training_step in self.config.vsa_debug_schedule
        initial = 0 < self._vsa_training_step <= self.config.vsa_debug_steps
        self._vsa_debug_active = self.training and (scheduled or initial)

    @staticmethod
    def _rms(tensor: Tensor) -> Tensor:
        return tensor.detach().float().square().mean().sqrt()

    @classmethod
    def _latent_debug_stats(cls, latents: Tensor, name: str) -> dict[str, float]:
        values = latents.detach().float()
        token_count = values.shape[1]
        normalized = F.normalize(values, dim=-1, eps=1e-12)
        cosine = normalized @ normalized.transpose(-1, -2)
        off_diagonal = ~torch.eye(
            token_count, dtype=torch.bool, device=values.device
        )[None]
        pairwise = cosine.masked_select(off_diagonal)
        centered = values - values.mean(dim=1, keepdim=True)
        gram = centered @ centered.transpose(-1, -2) / max(values.shape[-1], 1)
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
        probabilities = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        effective_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        )
        return {
            f"visual/{name}/pair_cosine_abs_mean": float(pairwise.abs().mean().item()),
            f"visual/{name}/effective_rank_fraction": float(
                (effective_rank / token_count).mean().item()
            ),
            f"visual/{name}/token_spread_rms": float(
                centered.square().mean().sqrt().item()
            ),
            f"visual/{name}/batch_spread_rms": float(
                (values - values.mean(dim=0, keepdim=True)).square().mean().sqrt().item()
            ),
        }

    def _record_visual_debug(self, condition_tokens: Tensor) -> None:
        if not self._vsa_debug_active:
            return
        if condition_tokens.shape[1] % 2 != 0:
            return
        top, wrist = condition_tokens.chunk(2, dim=1)
        top_centroid = F.normalize(top.detach().float().mean(dim=1), dim=-1)
        wrist_centroid = F.normalize(wrist.detach().float().mean(dim=1), dim=-1)
        self._last_vsa_debug_stats.update(
            {
                **self._latent_debug_stats(top, "top_latents"),
                **self._latent_debug_stats(wrist, "wrist_latents"),
                "visual/cross_camera/centroid_cosine": float(
                    (top_centroid * wrist_centroid).sum(dim=-1).mean().item()
                ),
            }
        )

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
        if self.uses_visual_perceiver:
            if len(images) != 2:
                raise ValueError(
                    f"Arch1_2 requires [top, wrist] images, got {len(images)}."
                )
            if self.top_resampler is None or self.wrist_resampler is None:
                raise RuntimeError("Arch1_2 has no camera Perceiver resamplers.")
            top, wrist = images
            if top.shape[0] != wrist.shape[0]:
                raise ValueError("Top and wrist image batches must have the same size.")
            # Match Arch2--4 exactly: one shared DINO call, then camera-specific
            # 1024-wide Perceiver Resamplers.
            prepared = torch.cat((top, wrist), dim=0).float()
            prepared = F.interpolate(
                prepared,
                size=(self.config.dino_image_size, self.config.dino_image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            prepared = (
                prepared - self._image_mean.float()
            ) / self._image_std.float()
            prepared = prepared.to(dtype=next(self.dino.parameters()).dtype)
            context = (
                torch.no_grad()
                if self.config.freeze_vision_encoder
                else nullcontext()
            )
            with context:
                hidden = self.dino(prepared).last_hidden_state
            top_hidden, wrist_hidden = hidden.split(top.shape[0], dim=0)

            def strip_registers(camera_hidden: Tensor) -> Tensor:
                tokens = torch.cat(
                    (
                        camera_hidden[:, :1],
                        camera_hidden[:, 1 + self.n_register_tokens :],
                    ),
                    dim=1,
                )
                if tokens.shape[1] != 197:
                    raise RuntimeError(
                        "DINO must produce CLS + 196 patch tokens after register "
                        f"removal; got {tokens.shape[1]}."
                    )
                return tokens

            top_tokens = strip_registers(top_hidden).to(self.working_dtype)
            wrist_tokens = strip_registers(wrist_hidden).to(self.working_dtype)
            return torch.cat(
                (
                    self.top_resampler(top_tokens),
                    self.wrist_resampler(wrist_tokens),
                ),
                dim=1,
            )
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

    def _expert_context_tokens(
        self, state: Tensor | None, skill_code: Tensor | None
    ) -> Tensor:
        """Return normalized [state, skill] tokens for Arch1_1/Arch1_2."""
        if not self.uses_expert_context_tokens:
            raise RuntimeError("Expert context tokens are disabled for Arch0.")
        if state is None or skill_code is None:
            raise ValueError("Arch1_1/Arch1_2 require both state and skill inputs.")
        if (
            self.state_proj is None
            or self.skill_proj is None
            or self.state_norm is None
            or self.skill_norm is None
        ):
            raise RuntimeError("Expert context projections are incomplete.")
        state_token, _ = self.state_norm(
            self.state_proj(state.to(self.working_dtype))
        )
        skill_token, _ = self.skill_norm(self._skill_embedding(skill_code))
        return torch.stack((state_token, skill_token), dim=1)

    def _project_state(self, state: Tensor | None) -> Tensor | None:
        """Return the Cond projection, also shared by Expert outside Arch0_2_sep."""
        if self.config.conditioning_route in STATELESS_CONDITIONING_ROUTES:
            return None
        if state is None or self.state_proj is None:
            raise ValueError(
                f"{self.config.conditioning_route} requires robot state conditioning."
            )
        return self.state_proj(state.to(self.working_dtype))

    def _project_expert_state(
        self,
        state: Tensor | None,
        shared_projection: Tensor | None,
    ) -> Tensor | None:
        """Return the Expert state embedding, shared except in Arch0_2_sep."""
        if not self.uses_expert_state_adarms:
            return None
        if self.expert_state_proj is None:
            return shared_projection
        if state is None:
            raise ValueError(
                f"{self.config.architecture_label} requires robot state conditioning."
            )
        return self.expert_state_proj(state.to(self.working_dtype))

    def _state_condition(self, state: Tensor | None) -> Tensor | None:
        """Project state only when this revision applies Cond-Gemma AdaRMS."""
        projected_state = self._project_state(state)
        return projected_state if self.uses_cond_state_adarms else None

    def _condition_state_start_index(
        self, condition_tokens: Tensor
    ) -> int | None:
        """Return the wrist-token boundary for Arch0_3's Cond AdaRMS."""
        if not self.uses_wrist_only_cond_state:
            return None
        if condition_tokens.shape[1] % 2:
            raise ValueError(
                "Arch0_3 requires equal top/wrist condition sequences; got "
                f"{condition_tokens.shape[1]} total tokens."
            )
        return condition_tokens.shape[1] // 2

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

    def _expert_condition(
        self,
        timestep: Tensor,
        projected_state: Tensor | None = None,
    ) -> Tensor:
        """Build Expert AdaRMS input from time and, for Arch0_1--0_3, state."""
        condition = self._time_condition(timestep)
        if not self.uses_expert_state_adarms:
            return condition
        if projected_state is None:
            raise ValueError(
                f"{self.config.architecture_label} requires projected state in Expert AdaRMS."
            )
        return condition + projected_state.to(condition.dtype)

    def _run_joint_hidden(
        self,
        condition_tokens: Tensor,
        noisy_actions: Tensor,
        condition_state: Tensor | None,
        expert_condition: Tensor,
        condition_skill: Tensor | None,
        expert_skill: Tensor | None,
        condition_state_start_index: int | None = None,
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
        adarms_start_indices = [condition_state_start_index, None]
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
                    adarms_start_index=adarms_start_indices,
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
                    adarms_start_index=adarms_start_indices,
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
        condition_state_start_index: int | None = None,
    ) -> Tensor:
        """Run Stage 1 and project its normalized action hidden to flow velocity."""
        action_hidden = self._run_joint_hidden(
            condition_tokens,
            noisy_actions,
            condition_state,
            expert_condition,
            condition_skill,
            expert_skill,
            condition_state_start_index,
        )
        return self.action_out_proj(action_hidden.to(self.working_dtype)).float()

    def _run_expert_token_hidden(
        self,
        condition_tokens: Tensor,
        context_tokens: Tensor,
        noisy_actions: Tensor,
        expert_condition: Tensor,
    ) -> Tensor:
        """Joint Arch1_1/Arch1_2 forward with the fixed three-block mask."""
        if context_tokens.shape[1] != 2:
            raise ValueError(
                "Expert context must contain exactly [state, skill] tokens; got "
                f"{context_tokens.shape[1]}."
            )
        action_tokens = self.action_in_proj(noisy_actions.to(self.working_dtype))
        batch_size = action_tokens.shape[0]
        n_visual = condition_tokens.shape[1]
        n_action = action_tokens.shape[1]
        device = action_tokens.device
        # [visual] reads visual; [state, skill] reads visual + both context
        # tokens; [actions] reads everything and is bidirectional internally.
        attention_mask, position_ids = expert_token_attention_contract(
            batch_size, n_visual, n_action, device
        )
        # One continuous coordinate system across visual/context/action tokens.
        streams = [condition_tokens, context_tokens, action_tokens]
        use_checkpoint = self._gradient_checkpointing and self.training
        for layer_index in range(self.gemma_expert.model.config.num_hidden_layers):
            kwargs = {
                "cond_encoder": self.cond_encoder,
                "gemma_expert": self.gemma_expert,
                "context_input_norm": self.context_input_norms[layer_index],
                "context_post_attention_norm": self.context_post_attention_norms[
                    layer_index
                ],
            }
            if use_checkpoint:
                streams = torch.utils.checkpoint.checkpoint(
                    compute_expert_token_layer,
                    layer_index,
                    streams,
                    attention_mask,
                    position_ids,
                    expert_condition,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    **kwargs,
                )
            else:
                streams = compute_expert_token_layer(
                    layer_index,
                    streams,
                    attention_mask,
                    position_ids,
                    expert_condition,
                    **kwargs,
                )
        action_hidden, _ = layernorm_forward(
            self.gemma_expert.model.norm, streams[2], expert_condition
        )
        return action_hidden

    def _run_expert_token_joint(
        self,
        condition_tokens: Tensor,
        context_tokens: Tensor,
        noisy_actions: Tensor,
        expert_condition: Tensor,
    ) -> Tensor:
        hidden = self._run_expert_token_hidden(
            condition_tokens, context_tokens, noisy_actions, expert_condition
        )
        return self.action_out_proj(hidden.to(self.working_dtype)).float()

    def _predict_velocity_from_condition(
        self,
        condition_tokens: Tensor,
        noisy_actions: Tensor,
        state: Tensor | None,
        skill_code: Tensor | None,
        time: Tensor,
    ) -> Tensor:
        """Run the post-vision path so scheduled probes can reuse encoded images."""
        expert_state_representation = None
        if self.uses_expert_context_tokens:
            expert_condition = self._expert_condition(time)
            context_tokens = self._expert_context_tokens(state, skill_code)
            predicted_velocity = self._run_expert_token_joint(
                condition_tokens,
                context_tokens,
                noisy_actions,
                expert_condition,
            )
            state_representation = context_tokens[:, :1]
            skill_representation = context_tokens[:, 1:]
        else:
            projected_state = self._project_state(state)
            expert_projected_state = self._project_expert_state(
                state, projected_state
            )
            if self.uses_separate_state_projections:
                expert_state_representation = expert_projected_state
            condition_state = (
                projected_state if self.uses_cond_state_adarms else None
            )
            expert_condition = self._expert_condition(
                time, expert_projected_state
            )
            condition_skill, expert_skill = self._skill_broadcasts(skill_code)
            state_representation = projected_state
            skill_representation = (
                condition_skill if condition_skill is not None else expert_skill
            )
            predicted_velocity = self._run_joint(
                condition_tokens,
                noisy_actions,
                condition_state,
                expert_condition,
                condition_skill,
                expert_skill,
                self._condition_state_start_index(condition_tokens),
            )
        if self._vsa_debug_active:
            tensors = {
                "visual_memory": condition_tokens,
                "noisy_actions": noisy_actions,
                "flow_prediction": predicted_velocity,
                "action_condition": expert_condition,
            }
            if state_representation is not None:
                tensors["state_token"] = state_representation
            if expert_state_representation is not None:
                tensors["expert_state_token"] = expert_state_representation
            if skill_representation is not None:
                tensors["skill_token"] = skill_representation
            self._last_vsa_debug_stats.update(
                {
                    f"activation/{name}_rms": float(self._rms(tensor).item())
                    for name, tensor in tensors.items()
                }
            )
        return predicted_velocity

    @torch.no_grad()
    def _input_sensitivity_stats(
        self,
        *,
        predicted_velocity: Tensor,
        condition_tokens: Tensor,
        noisy_actions: Tensor,
        state: Tensor | None,
        skill_code: Tensor | None,
        time: Tensor,
    ) -> dict[str, float]:
        """Perturb one modality at a time while keeping flow noise/time fixed."""
        if predicted_velocity.shape[0] < 2 or condition_tokens.shape[1] % 2 != 0:
            return {}
        top, wrist = condition_tokens.chunk(2, dim=1)
        variants: dict[str, tuple[Tensor, Tensor | None, Tensor | None]] = {
            "top_image_shuffle": (
                torch.cat((top.roll(1, dims=0), wrist), dim=1),
                state,
                skill_code,
            ),
            "wrist_image_shuffle": (
                torch.cat((top, wrist.roll(1, dims=0)), dim=1),
                state,
                skill_code,
            ),
            "both_images_shuffle": (
                condition_tokens.roll(1, dims=0),
                state,
                skill_code,
            ),
        }
        if state is not None:
            variants["state_shuffle"] = (
                condition_tokens,
                state.roll(1, dims=0),
                skill_code,
            )
        if skill_code is not None:
            variants["skill_shuffle"] = (
                condition_tokens,
                state,
                skill_code.roll(1, dims=0),
            )

        baseline = predicted_velocity.detach().float()
        baseline_rms = self._rms(baseline).clamp_min(1e-12)
        previous_debug = self._vsa_debug_active
        previous_checkpointing = self._gradient_checkpointing
        self._vsa_debug_active = False
        self._gradient_checkpointing = False
        try:
            stats = {}
            for name, (memory, perturbed_state, perturbed_skill) in variants.items():
                perturbed = self._predict_velocity_from_condition(
                    memory,
                    noisy_actions,
                    perturbed_state,
                    perturbed_skill,
                    time,
                ).float()
                difference_rms = self._rms(perturbed - baseline)
                stats[f"sensitivity/{name}/output_delta_rms"] = float(
                    difference_rms.item()
                )
                stats[f"sensitivity/{name}/relative_output_delta"] = float(
                    (difference_rms / baseline_rms).item()
                )
            return stats
        finally:
            self._vsa_debug_active = previous_debug
            self._gradient_checkpointing = previous_checkpointing

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
        self._last_vsa_debug_stats = {}
        batch_size = actions.shape[0]
        time = self.sample_time(batch_size, actions.device) if time is None else time
        self._last_flow_time = time.detach()
        source = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        source = source.to(actions.dtype)
        x_t = time[:, None, None] * source + (1.0 - time[:, None, None]) * actions
        target_velocity = source - actions
        condition_tokens = self._condition_tokens(images, batch_size=batch_size)
        self._record_visual_debug(condition_tokens)
        predicted_velocity = self._predict_velocity_from_condition(
            condition_tokens, x_t, state, skill_code, time
        )
        if self._vsa_debug_active:
            original_stats = dict(self._last_vsa_debug_stats)
            sensitivity = self._input_sensitivity_stats(
                predicted_velocity=predicted_velocity,
                condition_tokens=condition_tokens,
                noisy_actions=x_t,
                state=state,
                skill_code=skill_code,
                time=time,
            )
            self._last_vsa_debug_stats = {**original_stats, **sensitivity}
        if (
            self.config.action_loss_mode == "flow_endpoint_xyz"
            or self.config.cumulative_xyz_loss_enabled
        ):
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
        if self.uses_expert_context_tokens:
            return self._sample_with_expert_context_cache(
                condition_tokens,
                self._expert_context_tokens(state, skill_code),
                noise,
                num_steps,
            )
        return self._sample_with_condition_cache(
            condition_tokens, noise, state, skill_code, num_steps
        )

    def _visual_context_cache(
        self, condition_tokens: Tensor, context_tokens: Tensor
    ) -> DynamicCache:
        """Cache the timestep-independent visual + [state, skill] prefix."""
        batch_size, n_visual = condition_tokens.shape[:2]
        device = condition_tokens.device
        prefix_padding = torch.ones(
            batch_size, n_visual + 2, dtype=torch.bool, device=device
        )
        prefix_blocks = torch.tensor(
            [0] * n_visual + [1, 0], dtype=torch.bool, device=device
        )[None].expand(batch_size, -1)
        prefix_attention = make_att_2d_masks(
            prefix_padding, prefix_blocks
        )[:, None]
        prefix_attention = torch.where(
            prefix_attention, 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        prefix_positions = torch.cumsum(prefix_padding, dim=1) - 1
        streams = [condition_tokens, context_tokens]
        cache = DynamicCache(config=self.gemma_expert.model.config)
        for layer_index in range(self.gemma_expert.model.config.num_hidden_layers):
            streams, key, value = _mixed_visual_expert_layer(
                layer_index,
                streams,
                prefix_attention,
                prefix_positions,
                None,
                cond_encoder=self.cond_encoder,
                gemma_expert=self.gemma_expert,
                context_input_norm=self.context_input_norms[layer_index],
                context_post_attention_norm=self.context_post_attention_norms[
                    layer_index
                ],
            )
            cache.update(key, value, layer_index)
        return cache

    def _sample_with_expert_context_cache(
        self,
        condition_tokens: Tensor,
        context_tokens: Tensor,
        noise: Tensor,
        num_steps: int,
    ) -> Tensor:
        """Euler integration with visual/state/skill encoded exactly once."""
        batch_size = noise.shape[0]
        n_prefix = condition_tokens.shape[1] + context_tokens.shape[1]
        n_action = noise.shape[1]
        device = noise.device
        prefix_cache = self._visual_context_cache(condition_tokens, context_tokens)
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
                (batch_size,),
                1.0 + step * dt,
                dtype=torch.float32,
                device=device,
            )
            action_hidden = self._action_hidden_with_condition_cache(
                x_t,
                self._expert_condition(time),
                None,
                prefix_cache,
                full_attention,
                action_positions,
            )
            velocity = self.action_out_proj(
                action_hidden.to(self.working_dtype)
            ).float()
            x_t = x_t + dt * velocity
        return x_t

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
                self._expert_condition(time, expert_projected_state),
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
