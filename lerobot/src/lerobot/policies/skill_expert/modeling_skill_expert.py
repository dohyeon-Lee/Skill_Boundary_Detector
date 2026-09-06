"""Stage-1 DINO-Perceiver VSA prior with selectable vision conditioning."""

from __future__ import annotations

import copy
import json
import logging
import re
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import split_param_groups_for_muon
from lerobot.policies.pi05.lora import route_plain_to_base
from lerobot.policies.pi05.modeling_pi05 import (
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    pad_vector,
    sample_beta,
)
from lerobot.policies.pi_gemma import PiGemmaRMSNorm
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.skillVLA.dataset_skillVLA import (
    SKILL_CANONICAL_ACTION_IS_PAD,
    SKILL_CANONICAL_ACTIONS,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    SKILL_FLOW_MODE_LATENT_OVERRIDE,
    SKILL_FLOW_NOISE_OVERRIDE,
)

from .configuration_skill_expert import (
    COND_GEMMA_ARCHITECTURE,
    COND_GEMMA_ARCHITECTURE_REVISION,
    COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION,
    COMPRESSED_VISUAL_KV_SELF_ATTENTION,
    GLOBAL_VISUAL_ADARMS,
    IN_CONTEXT_TOKENS,
    INTERLEAVED_CROSS_ATTENTION,
    LEGACY_RESIDUAL_CROSS_ATTENTION,
    LEGACY_RESIDUAL_VSA_REVISION,
    SKILLLESS_CONDITIONING_ROUTES,
    STATELESS_CONDITIONING_ROUTES,
    VSA_ARCHITECTURE,
    VSA_ARCHITECTURE_REVISION,
    UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION,
    VISIONLESS_CONDITIONING_ROUTES,
    SkillExpertConfig,
    normalize_conditioning_route,
)
from .cond_gemma import CondGemmaSkillExpert
from .modeling_utils import (
    build_fsq_image_only_terminator,
    build_fsq_terminator,
    build_trainable_fsq_terminator,
    load_raw_state_dict,
)
from .modeling_skill_predictor import FrozenVLMSkillPredictor
from .vsa_perceiver_crossattn import (
    CameraPerceiverResampler,
    VSAActionExpert,
)

log = logging.getLogger(__name__)


class SkillExpertPytorch(nn.Module):
    """Shared DINO, camera-specific Perceivers, and one cross-attention expert."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__()
        self.config = config
        self.width = get_gemma_config(config.action_expert_variant).width
        self.dino = AutoModel.from_pretrained(config.dino_model_path)
        self.dino.requires_grad_(True)
        self.n_register_tokens = int(getattr(self.dino.config, "num_register_tokens", 0))
        dino_width = int(self.dino.config.hidden_size)
        self.uses_uncompressed_visual_memory = (
            config.vision_conditioning_mode
            == UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION
        )
        if self.uses_uncompressed_visual_memory:
            if dino_width != self.width:
                raise ValueError(
                    "Arch1_3 requires direct DINO/expert width equality; got "
                    f"{dino_width} and {self.width}."
                )
            self.top_resampler = None
            self.wrist_resampler = None
        else:
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

        self.state_proj = nn.Linear(config.max_state_dim, self.width)
        self.state_norm = PiGemmaRMSNorm(self.width)
        self.skill_proj = nn.Linear(len(config.skill_fsq_levels), self.width)
        self.skill_norm = PiGemmaRMSNorm(self.width)
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

        # This is the only mode-specific parameter outside the expert. Starting
        # from zero makes the first global-AdaRMS forward exactly timestep-only.
        self.visual_condition_projection = None
        if config.vision_conditioning_mode == GLOBAL_VISUAL_ADARMS:
            self.visual_condition_projection = nn.Linear(self.width * 2, self.width)
            nn.init.zeros_(self.visual_condition_projection.weight)
            nn.init.zeros_(self.visual_condition_projection.bias)

        expert_class = VSAActionExpert
        evaluation_revision = str(getattr(config, "eval_vsa_revision", ""))
        if evaluation_revision:
            # Explicit evaluation-only compatibility. Training configs never set
            # this runtime field, so new Stage-1 runs remain on the current path.
            from .legacy_vsa_eval import (  # noqa: PLC0415
                LegacyResidualSA18VSAActionExpert,
                LegacyVSAActionExpert,
            )

            expert_class = (
                LegacyResidualSA18VSAActionExpert
                if evaluation_revision == LEGACY_RESIDUAL_VSA_REVISION
                else LegacyVSAActionExpert
            )
        expert_kwargs = {
            "include_state_in_visual_crossattn": config.include_state_in_visual_crossattn,
            "include_skill_in_visual_crossattn": config.include_skill_in_visual_crossattn,
        }
        if not evaluation_revision or evaluation_revision == LEGACY_RESIDUAL_VSA_REVISION:
            expert_kwargs["vision_conditioning_mode"] = config.vision_conditioning_mode
        self.expert = expert_class(**expert_kwargs)
        self.skill_predictor = (
            FrozenVLMSkillPredictor(config) if config.uses_skill_predictor else None
        )
        if self.skill_predictor is not None:
            self.skill_predictor.requires_grad_(False).eval()
        self.fsq_term_train = None
        self.fsq_image_term_train = None
        if config.train_terminator:
            terminator = build_fsq_terminator(config.fsq_path)
            if config.terminator_freeze_vision_encoder is not None:
                terminator.freeze_vision_encoder = bool(
                    config.terminator_freeze_vision_encoder
                )
            self.fsq_term_train = (
                terminator.to(dtype=torch.float32).requires_grad_(False).eval()
            )
        self._last_predicted_actions: Tensor | None = None
        self._last_flow_time: Tensor | None = None
        self._last_vsa_debug_stats: dict[str, float] = {}
        self._vsa_training_step: int | None = None
        self._vsa_debug_active = False
        self._gradient_checkpointing = False

    @property
    def working_dtype(self) -> torch.dtype:
        return self.action_in_proj.weight.dtype

    def set_training_step(self, step: int) -> None:
        """Enable expensive diagnostics only on explicitly configured steps."""
        self._vsa_training_step = int(step)
        scheduled = self._vsa_training_step in self.config.vsa_debug_schedule
        initial = 0 < self._vsa_training_step <= self.config.vsa_debug_steps
        self._vsa_debug_active = self.training and (scheduled or initial)
        self.expert.debug_enabled = self._vsa_debug_active

    @staticmethod
    def _rms(tensor: Tensor) -> Tensor:
        return tensor.detach().float().square().mean().sqrt()

    @classmethod
    def _latent_debug_stats(cls, latents: Tensor, name: str) -> dict[str, float]:
        """Measure token diversity; unlike LayerNorm moments this detects collapse."""
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

    @classmethod
    def _visual_debug_stats(
        cls,
        top_tokens: Tensor,
        wrist_tokens: Tensor,
        top_memory: Tensor,
        wrist_memory: Tensor,
    ) -> dict[str, float]:
        stats = {
            **cls._latent_debug_stats(top_memory, "top_latents"),
            **cls._latent_debug_stats(wrist_memory, "wrist_latents"),
        }
        top_patch = top_tokens[:, 1:].detach().float()
        wrist_patch = wrist_tokens[:, 1:].detach().float()
        top_dino_spread = (
            (top_patch - top_patch.mean(dim=1, keepdim=True))
            .square().mean().sqrt()
        )
        wrist_dino_spread = (
            (wrist_patch - wrist_patch.mean(dim=1, keepdim=True))
            .square().mean().sqrt()
        )
        stats.update(
            {
                "visual/top_latents/spread_retention_vs_dino": float(
                    stats["visual/top_latents/token_spread_rms"]
                    / max(float(top_dino_spread.item()), 1e-12)
                ),
                "visual/wrist_latents/spread_retention_vs_dino": float(
                    stats["visual/wrist_latents/token_spread_rms"]
                    / max(float(wrist_dino_spread.item()), 1e-12)
                ),
            }
        )
        top_centroid = F.normalize(top_memory.detach().float().mean(dim=1), dim=-1)
        wrist_centroid = F.normalize(wrist_memory.detach().float().mean(dim=1), dim=-1)
        stats.update(
            {
                "visual/cross_camera/centroid_cosine": float(
                    (top_centroid * wrist_centroid).sum(dim=-1).mean().item()
                ),
            }
        )
        return stats

    def gradient_checkpointing_enable(self) -> None:
        self._gradient_checkpointing = True
        self.expert.gradient_checkpointing_enable()
        if hasattr(self.dino, "gradient_checkpointing_enable"):
            self.dino.gradient_checkpointing_enable()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.skill_predictor is not None:
            # Predictor is an optional frozen input provider, never a Stage1 target.
            self.skill_predictor.eval()
        if self.fsq_term_train is not None:
            self.fsq_term_train.eval()
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

    def _prepare_image(self, image: Tensor) -> Tensor:
        image = image.float()
        return F.interpolate(
            image,
            size=(self.config.dino_image_size, self.config.dino_image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def _strip_register_tokens(self, hidden: Tensor) -> Tensor:
        cls_token = hidden[:, :1]
        patch_tokens = hidden[:, 1 + self.n_register_tokens :]
        tokens = torch.cat((cls_token, patch_tokens), dim=1)
        if tokens.shape[1] != 197:
            raise RuntimeError(
                "DINO must produce CLS + 196 patch tokens after register removal; "
                f"got {tokens.shape[1]} (registers={self.n_register_tokens})."
            )
        return tokens

    def encode_visual_memory(self, images: list[Tensor]) -> Tensor:
        """Run shared DINO once and each camera-specific resampler once."""
        if len(images) != 2:
            raise ValueError(f"Stage 1 requires [top, wrist] images, got {len(images)}.")
        top, wrist = images
        if top.shape[0] != wrist.shape[0]:
            raise ValueError("Top and wrist image batches must have the same size.")
        prepared = torch.cat((self._prepare_image(top), self._prepare_image(wrist)), dim=0)
        prepared = (prepared - self._image_mean.float()) / self._image_std.float()
        prepared = prepared.to(dtype=next(self.dino.parameters()).dtype)
        hidden = self.dino(prepared).last_hidden_state
        top_hidden, wrist_hidden = hidden.split(top.shape[0], dim=0)
        top_tokens = self._strip_register_tokens(top_hidden)
        wrist_tokens = self._strip_register_tokens(wrist_hidden)
        if self.uses_uncompressed_visual_memory:
            top_memory = top_tokens.to(self.working_dtype)
            wrist_memory = wrist_tokens.to(self.working_dtype)
        else:
            if self.top_resampler is None or self.wrist_resampler is None:
                raise RuntimeError("Compressed visual mode has no Perceiver resamplers.")
            top_memory = self.top_resampler(top_tokens.to(self.working_dtype))
            wrist_memory = self.wrist_resampler(wrist_tokens.to(self.working_dtype))
        if self._vsa_debug_active:
            self._last_vsa_debug_stats.update(
                self._visual_debug_stats(
                    top_tokens, wrist_tokens, top_memory, wrist_memory
                )
            )
        return torch.cat((top_memory, wrist_memory), dim=1)

    def _code_to_zq(self, skill_code: Tensor) -> Tensor:
        index = skill_code.reshape(-1, 1).long()
        level_ids = (
            torch.div(index, self._fsq_strides[None], rounding_mode="floor")
            % self._fsq_levels[None]
        )
        return (level_ids.float() - self._fsq_half[None]) / self._fsq_half[None]

    def _skill_embedding(self, skill_code: Tensor) -> Tensor:
        z_q = self._code_to_zq(skill_code).to(self.working_dtype)
        token, _ = self.skill_norm(self.skill_proj(z_q))
        return token[:, None]

    def _state_embedding(self, state: Tensor) -> Tensor:
        token, _ = self.state_norm(self.state_proj(state.to(self.working_dtype)))
        return token[:, None]

    def _context_tokens(self, state: Tensor, skill_code: Tensor) -> Tensor:
        return torch.cat(
            (self._state_embedding(state), self._skill_embedding(skill_code)), dim=1
        )

    def terminator_predict(
        self,
        true_code: Tensor,
        raw_state: Tensor | None,
        image: Tensor | None,
        wrist_image: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Run the independent FSQ terminator on current raw observations."""
        terminator = self.fsq_term_train
        if terminator is None:
            raise RuntimeError("Terminator training is disabled.")
        device = next(terminator.parameters()).device
        dtype = next(terminator.parameters()).dtype
        context_mode = str(getattr(terminator, "context_mode", "proprio"))
        if context_mode == "none":
            state = None
        else:
            if raw_state is None:
                raise ValueError(f"{context_mode} terminator requires context input.")
            state = raw_state.to(device=device, dtype=dtype)[
                ..., : int(terminator.state_dim)
            ]
        camera_mode = str(getattr(terminator, "camera_mode", "both"))
        if camera_mode in {"both", "top"} and image is None:
            raise ValueError("Terminator requires a top image.")
        if camera_mode in {"both", "wrist"} and wrist_image is None:
            raise ValueError("Terminator requires a wrist image.")
        z_q = self._code_to_zq(true_code.to(self._fsq_strides.device)).to(
            device=device, dtype=dtype
        )
        return terminator(
            z_q,
            state,
            None if image is None else image.to(device=device, dtype=dtype),
            None
            if wrist_image is None
            else wrist_image.to(device=device, dtype=dtype),
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

    def _pool_visual_memory(self, visual_memory: Tensor) -> Tensor:
        """Mean-pool top/wrist latents independently, then concatenate features."""
        camera_tokens = self.config.num_visual_latents_per_camera
        if visual_memory.shape[1] != camera_tokens * 2:
            raise ValueError(
                "Global visual AdaRMS requires equal top/wrist memories; got "
                f"{visual_memory.shape[1]} tokens for {camera_tokens} latents/camera."
            )
        top, wrist = visual_memory.split(
            (camera_tokens, camera_tokens), dim=1
        )
        return torch.cat((top.mean(dim=1), wrist.mean(dim=1)), dim=-1)

    def _action_condition(
        self, visual_memory: Tensor, time_condition: Tensor
    ) -> Tensor:
        """Return the condition supplied to every action AdaRMS module."""
        if self.config.vision_conditioning_mode != GLOBAL_VISUAL_ADARMS:
            return time_condition
        if self.visual_condition_projection is None:
            raise RuntimeError("global_visual_adarms requires visual_condition_projection.")
        pooled = self._pool_visual_memory(visual_memory)
        visual_condition = self.visual_condition_projection(pooled)
        if self._vsa_debug_active:
            visual_rms = self._rms(visual_condition)
            time_rms = self._rms(time_condition)
            self._last_vsa_debug_stats.update(
                {
                    "conditioning/global_visual_rms": float(visual_rms.item()),
                    "conditioning/time_rms": float(time_rms.item()),
                    "conditioning/visual_to_time_rms_ratio": float(
                        (visual_rms / time_rms.clamp_min(1e-12)).item()
                    ),
                }
            )
        return time_condition + visual_condition

    def _run_joint_hidden(
        self,
        visual_memory: Tensor,
        noisy_actions: Tensor,
        state: Tensor,
        skill_code: Tensor,
        time_condition: Tensor,
    ) -> Tensor:
        """Return action hidden states after 18 self-attention expert blocks."""
        action_tokens = self.action_in_proj(noisy_actions.to(self.working_dtype))
        context_tokens = self._context_tokens(state, skill_code)
        action_condition = self._action_condition(visual_memory, time_condition)
        action_hidden = self.expert(
            context_tokens,
            action_tokens,
            visual_memory,
            action_condition,
        )
        if self._vsa_debug_active:
            tensors = {
                "visual_memory": visual_memory,
                "state_token": context_tokens[:, :1],
                "skill_token": context_tokens[:, 1:],
                "action_input": action_tokens,
                "action_hidden": action_hidden,
                "action_condition": action_condition,
            }
            self._last_vsa_debug_stats.update(
                {
                    f"activation/{name}_rms": float(
                        tensor.detach().float().square().mean().sqrt().item()
                    )
                    for name, tensor in tensors.items()
                }
            )
            self._last_vsa_debug_stats.update(self.expert.last_debug_stats)
        return action_hidden

    @torch.no_grad()
    def _input_sensitivity_stats(
        self,
        *,
        predicted_velocity: Tensor,
        visual_memory: Tensor,
        noisy_actions: Tensor,
        state: Tensor,
        skill_code: Tensor,
        time_condition: Tensor,
    ) -> dict[str, float]:
        """Measure input reliance with deterministic batch-roll perturbations."""
        if predicted_velocity.shape[0] < 2:
            return {}
        previous_debug = self.expert.debug_enabled
        self.expert.debug_enabled = False
        try:
            top, wrist = visual_memory.split(visual_memory.shape[1] // 2, dim=1)
            variants = {
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
                    visual_memory.roll(1, dims=0),
                    state,
                    skill_code,
                ),
                "state_shuffle": (visual_memory, state.roll(1, dims=0), skill_code),
                "skill_shuffle": (visual_memory, state, skill_code.roll(1, dims=0)),
            }
            baseline = predicted_velocity.detach().float()
            baseline_rms = self._rms(baseline).clamp_min(1e-12)
            stats = {}
            for name, (memory, perturbed_state, perturbed_skill) in variants.items():
                perturbed = self._run_joint(
                    memory,
                    noisy_actions,
                    perturbed_state,
                    perturbed_skill,
                    time_condition,
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
            self.expert.debug_enabled = previous_debug

    def _run_joint(
        self,
        visual_memory: Tensor,
        noisy_actions: Tensor,
        state: Tensor,
        skill_code: Tensor,
        time_condition: Tensor,
    ) -> Tensor:
        """Run Stage 1 and project its normalized action hidden to flow velocity."""
        action_hidden = self._run_joint_hidden(
            visual_memory,
            noisy_actions,
            state,
            skill_code,
            time_condition,
        )
        return self.action_out_proj(action_hidden.to(self.working_dtype)).float()

    def forward(
        self,
        images: list[Tensor],
        state: Tensor,
        skill_code: Tensor,
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
        visual_memory = self.encode_visual_memory(images)
        time_condition = self._time_condition(time)
        predicted_velocity = self._run_joint(
            visual_memory,
            x_t,
            state,
            skill_code,
            time_condition,
        )
        if self._vsa_debug_active:
            original_stats = dict(self._last_vsa_debug_stats)
            sensitivity = self._input_sensitivity_stats(
                predicted_velocity=predicted_velocity,
                visual_memory=visual_memory,
                noisy_actions=x_t,
                state=state,
                skill_code=skill_code,
                time_condition=time_condition,
            )
            # Probe forwards intentionally disable layer instrumentation; retain
            # the original forward's stats and append only sensitivity values.
            self._last_vsa_debug_stats = {**original_stats, **sensitivity}
        if getattr(self.config, "cumulative_xyz_loss_enabled", False):
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
        state: Tensor,
        skill_code: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        num_steps = self.config.num_inference_steps if num_steps is None else num_steps
        batch_size, device = state.shape[0], state.device
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim), device
            )
        visual_memory = self.encode_visual_memory(images)
        context_tokens = self._context_tokens(state, skill_code)
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = torch.full(
                (batch_size,), 1.0 + step * dt, dtype=torch.float32, device=device
            )
            action_tokens = self.action_in_proj(x_t.to(self.working_dtype))
            action_hidden = self.expert(
                context_tokens,
                action_tokens,
                visual_memory,
                self._action_condition(visual_memory, self._time_condition(time)),
            )
            velocity = self.action_out_proj(action_hidden.to(self.working_dtype)).float()
            x_t = x_t + dt * velocity
        return x_t


def _map_pi05_cond_key(
    key: str, *, include_predictor_vlm: bool = False
) -> str | None:
    """Original skillVLA_real pi0.5 -> condition-Gemma mapping."""
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


def _map_pi05_key(
    key: str,
    *,
    architecture: str = VSA_ARCHITECTURE,
    vision_conditioning_mode: str = INTERLEAVED_CROSS_ATTENTION,
    include_predictor_vlm: bool = False,
) -> str | None:
    """Apply the architecture-specific pi0.5 initialization contract."""
    if architecture == COND_GEMMA_ARCHITECTURE:
        return _map_pi05_cond_key(
            key, include_predictor_vlm=include_predictor_vlm
        )
    layer_match = re.fullmatch(
        r"paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)\."
        r"(self_attn|mlp|input_layernorm|post_attention_layernorm)\.(.+)",
        key,
    )
    if layer_match:
        layer_index = int(layer_match.group(1))
        component = layer_match.group(2)
        suffix = layer_match.group(3)
        replaces_self_attention = (
            vision_conditioning_mode == INTERLEAVED_CROSS_ATTENTION
            and bool(layer_index % 2)
        )
        if component == "self_attn":
            if replaces_self_attention:
                return None
            return f"model.expert.blocks.{layer_index}.self_attention.{suffix}"
        if component == "mlp":
            return f"model.expert.blocks.{layer_index}.mlp.{suffix}"
        if component == "input_layernorm" and replaces_self_attention:
            return None
        norm_name = (
            "self_attention_norm" if component == "input_layernorm" else "ffn_norm"
        )
        return f"model.expert.blocks.{layer_index}.{norm_name}.action_norm.{suffix}"
    final_norm = "paligemma_with_expert.gemma_expert.model.norm."
    if key.startswith(final_norm):
        return "model.expert.final_norm." + key.removeprefix(final_norm)
    if key.startswith("paligemma_with_expert.gemma_expert."):
        return None
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
    raw: dict,
    *,
    architecture: str,
    vision_conditioning_mode: str = INTERLEAVED_CROSS_ATTENTION,
    include_predictor_vlm: bool = False,
) -> tuple[dict, bool]:
    is_pi05 = any(key.startswith("paligemma_with_expert.") for key in raw)
    if is_pi05:
        mapped = {}
        for key, value in raw.items():
            mapped_key = _map_pi05_key(
                key,
                architecture=architecture,
                vision_conditioning_mode=vision_conditioning_mode,
                include_predictor_vlm=include_predictor_vlm,
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
    architecture: str,
    vision_conditioning_mode: str = INTERLEAVED_CROSS_ATTENTION,
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
                        key,
                        architecture=architecture,
                        vision_conditioning_mode=vision_conditioning_mode,
                        include_predictor_vlm=include_predictor_vlm,
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
        raw,
        architecture=architecture,
        vision_conditioning_mode=vision_conditioning_mode,
        include_predictor_vlm=include_predictor_vlm,
    )


def _allowed_pi05_missing_key(key: str, config: SkillExpertConfig) -> bool:
    """Return whether a target tensor is intentionally new relative to pi0.5."""
    if config.architecture == COND_GEMMA_ARCHITECTURE:
        if key.startswith(
            (
                "model.dino.",
                "model.image_proj.",
                "model.visionless_condition_token",
                "model.state_proj.",
                "model.expert_state_proj.",
                "model.skill_proj.",
                "model.cond_encoder.",
                "model.state_norm.",
                "model.skill_norm.",
                "model.expert_skill_norm.",
                "model.expert_skill_gain",
                "model.mode_latent_mlp.",
                "model.mode_latent_gain",
                "model.context_input_norms.",
                "model.context_post_attention_norms.",
                "model.top_resampler.",
                "model.wrist_resampler.",
            )
        ):
            return True
        if config.uses_skill_predictor and (
            key.startswith(
                (
                    "model.skill_predictor.reader.",
                    "model.skill_predictor.head.",
                )
            )
            or ".adapters.skill." in key
        ):
            return True
        return config.train_terminator and key.startswith("model.fsq_term_train.")
    if key.startswith(
        (
            "model.dino.",
            "model.top_resampler.",
            "model.wrist_resampler.",
            "model.state_proj.",
            "model.state_norm.",
            "model.skill_proj.",
            "model.skill_norm.",
        )
    ):
        return True
    if re.fullmatch(
        r"model\.expert\.blocks\.\d+\."
        r"(self_attention_norm|ffn_norm)\.context_norm\..+",
        key,
    ):
        return True
    if config.vision_conditioning_mode == INTERLEAVED_CROSS_ATTENTION and re.fullmatch(
        r"model\.expert\.blocks\.\d+\."
        r"(visual_attention_norm|visual_cross_attention)(\..+)?",
        key,
    ):
        return True
    if (
        config.vision_conditioning_mode == GLOBAL_VISUAL_ADARMS
        and key.startswith("model.visual_condition_projection.")
    ):
        return True
    if config.uses_skill_predictor and (
        key.startswith(
            (
                "model.skill_predictor.reader.",
                "model.skill_predictor.head.",
            )
        )
        or ".adapters.skill." in key
    ):
        return True
    return config.train_terminator and key.startswith("model.fsq_term_train.")


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
# Skill geometry is tied to the dataset/FSQ codebook the target was trained on,
# so it must match on both overlay paths.
_PREDICTOR_SKILL_GEOMETRY_FIELDS = ("skill_vocab_size", "skill_fsq_levels")
# The remaining fields only describe the predictor module's own shape. When the
# target owns no predictor they are unused defaults, so a predictor-free target
# adopts them from the source checkpoint instead of imposing its own.
_PREDICTOR_MODULE_FIELDS = tuple(
    field
    for field in _PREDICTOR_CHECKPOINT_CONTRACT_FIELDS
    if field not in _PREDICTOR_SKILL_GEOMETRY_FIELDS
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
    *,
    prefix: str = "model.fsq_term_train.",
    label: str = "terminator",
) -> int:
    """Load one complete co-trained terminator without unrelated Stage-1 tensors."""
    from safetensors import safe_open  # noqa: PLC0415

    path = Path(checkpoint_path)
    weights_path = path if path.is_file() else path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Stage-1 terminator weights not found: {weights_path}")

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
                f"Complete Stage-1 {label} tensor mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        with torch.no_grad():
            for key in sorted(expected):
                value = checkpoint.get_tensor(prefix + key)
                target = target_state[key]
                if value.shape != target.shape:
                    raise RuntimeError(
                        f"Stage-1 {label} shape mismatch for {key}: "
                        f"checkpoint={tuple(value.shape)}, model={tuple(target.shape)}"
                    )
                target.copy_(value.to(device=target.device, dtype=target.dtype))
    return len(expected)


class SkillExpertPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for VSA or the original condition-Gemma Stage 1."""

    config_class = SkillExpertConfig
    name = "skill_expert"

    def __init__(self, config: SkillExpertConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        log.info(
            "Stage-1 experiment architecture: %s",
            config.architecture_label or "legacy-unlabeled",
        )
        if config.architecture == COND_GEMMA_ARCHITECTURE:
            self.model = CondGemmaSkillExpert(config)
            log.info("Stage-1 architecture: condition Gemma + pi0.5 Gemma expert")
            if not self.model.uses_expert_context_tokens:
                log.info("Conditioning route: %s", config.conditioning_route)
                if self.model.uses_wrist_only_cond_state:
                    cond_state_target = "Cond-Gemma wrist tokens only"
                elif self.model.uses_cond_state_adarms:
                    cond_state_target = "all Cond-Gemma tokens"
                else:
                    cond_state_target = "disabled"
                log.info("State conditioning (Cond): %s", cond_state_target)
                log.info(
                    "State conditioning (Expert): %s",
                    "time + projected-state AdaRMS"
                    if self.model.uses_expert_state_adarms
                    else "time-only AdaRMS",
                )
                if self.model.uses_expert_state_adarms:
                    log.info(
                        "State projections (Cond/Expert): %s",
                        "separate"
                        if self.model.uses_separate_state_projections
                        else "shared",
                    )
                if self.model.uses_expert_skill_adarms:
                    log.info(
                        "Skill conditioning: expert AdaRMS "
                        "(RMS-normalized, summed with timestep, %s)",
                        "zero-init scalar gain"
                        if self.model.uses_zero_init_skill_gain
                        else "unit gain",
                    )
                else:
                    targets = [
                        name
                        for name, active in (
                            ("Cond-Gemma", self.model.uses_cond_skill_broadcast),
                            ("expert", self.model.uses_expert_skill_broadcast),
                        )
                        if active
                    ]
                    log.info(
                        "Skill conditioning: %s layerwise broadcast",
                        " + ".join(targets) if targets else "disabled",
                    )
                log.info("Expert sequence: noisy actions only")
            else:
                if self.model.uses_expert_state_token:
                    log.info("Condition-Gemma state/skill conditioning: disabled")
                    log.info("Expert sequence: state + skill + noisy actions")
                    log.info("Expert mask: [visual | state, skill | actions]")
                else:
                    log.info("Conditioning route: %s", config.conditioning_route)
                    log.info("State conditioning (Cond): all Cond-Gemma tokens")
                    log.info(
                        "Skill conditioning: one expert in-context token (%s)",
                        "isolated from vision"
                        if self.model.uses_isolated_skill_token
                        else "reads the visual prefix",
                    )
                    log.info("Expert sequence: skill + noisy actions")
                    log.info(
                        "Expert mask: %s",
                        "[visual | skill (visual-blind) | actions]"
                        if self.model.uses_isolated_skill_token
                        else "[visual | skill | actions]",
                    )
                log.info("Timestep AdaRMS: action tokens only")
                if (
                    config.architecture_revision
                    == COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION
                ):
                    log.info(
                        "Condition-Gemma visual tokens: %d top + %d wrist Perceiver latents",
                        config.num_visual_latents_per_camera,
                        config.num_visual_latents_per_camera,
                    )
                else:
                    log.info("Condition-Gemma visual tokens: uncompressed DINO tokens")
        else:
            self.model = SkillExpertPytorch(config)
            log.info("Vision conditioning mode: %s", config.vision_conditioning_mode)
            if config.vision_conditioning_mode in {
                INTERLEAVED_CROSS_ATTENTION,
                LEGACY_RESIDUAL_CROSS_ATTENTION,
            }:
                visual_query_tokens = []
                if config.include_state_in_visual_crossattn:
                    visual_query_tokens.append("state")
                if config.include_skill_in_visual_crossattn:
                    visual_query_tokens.append("skill")
                visual_query_label = (
                    " + ".join((*visual_query_tokens, "action"))
                    if visual_query_tokens
                    else "action-only"
                )
                log.info("Visual cross-attention queries: %s", visual_query_label)
            else:
                log.info(
                    "Visual cross-attention query switches are ignored in %s mode.",
                    config.vision_conditioning_mode,
                )
            if config.vision_conditioning_mode == IN_CONTEXT_TOKENS:
                visual_tokens = 2 * config.num_visual_latents_per_camera
                log.info(
                    "Expert sequence: %d visual + 1 state + 1 skill + H action",
                    visual_tokens,
                )
                log.info("Visual cross-attention modules: disabled")
            elif config.vision_conditioning_mode == GLOBAL_VISUAL_ADARMS:
                log.info("Visual aggregation: per-camera mean pooling")
                log.info("Visual injection: timestep condition + global visual condition")
                log.info("Visual cross-attention modules: disabled")
                log.info("Visual tokens in expert sequence: false")
            elif config.vision_conditioning_mode in {
                UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION,
                COMPRESSED_VISUAL_KV_SELF_ATTENTION,
            }:
                memory_tokens = (
                    394
                    if config.vision_conditioning_mode
                    == UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION
                    else 2 * config.num_visual_latents_per_camera
                )
                log.info(
                    "Visual fusion: fixed %d-token memory as K/V in all 18 expert self-attention layers",
                    memory_tokens,
                )
                log.info("Visual query/update path: disabled")
                log.info("Visual-side Transformer parameters: none")
            evaluation_revision = str(getattr(config, "eval_vsa_revision", ""))
            if evaluation_revision:
                log.info(
                    "VSA checkpoint layout: %s (historical evaluation-only)",
                    evaluation_revision,
                )
            elif config.vision_conditioning_mode == INTERLEAVED_CROSS_ATTENTION:
                log.info("Expert attention layout: even self-attention / odd visual cross-attention")
                log.info("Visual cross-attention residual gate: none")
            elif config.vision_conditioning_mode == LEGACY_RESIDUAL_CROSS_ATTENTION:
                log.info("Historical expert attention layout: SA18 + residual visual cross-attention")
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(device=config.device, dtype=self._torch_dtype())
        if (
            config.skill_flow_latent_best_of_n_enabled
            and self.model.mode_latent_mlp is not None
        ):
            log.info(
                "Mode-latent precision: projection=%s gain=%s expert_output=%s",
                next(self.model.mode_latent_mlp.parameters()).dtype,
                self.model.mode_latent_gain.dtype,
                self.model.working_dtype,
            )
        if self.model.fsq_term_train is not None:
            # Match FSQ training/inference numerics; this auxiliary remains fp32.
            self.model.fsq_term_train.to(dtype=torch.float32)
        counts = self.parameter_counts()
        if config.architecture == COND_GEMMA_ARCHITECTURE:
            log.info(
                "Stage-1 parameters: total=%.1fM trainable=%.1fM dino=%.1fM "
                "cond=%.1fM expert=%.1fM",
                counts["total"] / 1e6,
                counts["trainable"] / 1e6,
                counts["dino"] / 1e6,
                counts["conditioner"] / 1e6,
                counts["expert"] / 1e6,
            )
        else:
            log.info(
                "Stage-1 parameters: total=%.1fM trainable=%.1fM dino=%.1fM "
                "perceivers=%.1fM expert=%.1fM",
                counts["total"] / 1e6,
                counts["trainable"] / 1e6,
                counts["dino"] / 1e6,
                counts["perceivers"] / 1e6,
                counts["expert"] / 1e6,
            )
        self.reset()

    def set_training_step(self, step: int) -> None:
        """Receive the true optimizer step so resumed runs keep the debug schedule."""
        self.model.set_training_step(step)

    def parameter_counts(self) -> dict[str, int]:
        def count(module: nn.Module | None, *, trainable: bool = False) -> int:
            if module is None:
                return 0
            return sum(
                parameter.numel()
                for parameter in module.parameters()
                if not trainable or parameter.requires_grad
            )

        if getattr(self.config, "architecture", VSA_ARCHITECTURE) == COND_GEMMA_ARCHITECTURE:
            return {
                "total": count(self),
                "trainable": count(self, trainable=True),
                "dino": count(self.model.dino),
                "conditioner": count(self.model.cond_encoder),
                "expert": count(self.model.gemma_expert),
            }
        return {
            "total": count(self),
            "trainable": count(self, trainable=True),
            "dino": count(self.model.dino),
            "perceivers": count(self.model.top_resampler) + count(self.model.wrist_resampler),
            "expert": count(self.model.expert),
        }

    def training_debug_metrics(self) -> dict[str, float]:
        if not self.model._vsa_debug_active:
            return {}

        def grad_stats(module: nn.Module | None) -> tuple[float, float, float]:
            if module is None:
                return 0.0, 0.0, 0.0
            parameters = [
                parameter for parameter in module.parameters() if parameter.requires_grad
            ]
            if not parameters:
                return 0.0, 0.0, 0.0
            parameter_squared_sum = torch.stack(
                [parameter.detach().float().square().sum() for parameter in parameters]
            ).sum()
            parameter_count = sum(parameter.numel() for parameter in parameters)
            parameter_rms = (parameter_squared_sum / max(parameter_count, 1)).sqrt()
            gradients = [
                parameter.grad.detach().float()
                for parameter in parameters
                if parameter.grad is not None
            ]
            if not gradients:
                return 0.0, float(parameter_rms.item()), 0.0
            gradient_squared_sum = torch.stack(
                [gradient.square().sum() for gradient in gradients]
            ).sum()
            gradient_count = sum(gradient.numel() for gradient in gradients)
            gradient_rms = (gradient_squared_sum / max(gradient_count, 1)).sqrt()
            return (
                float(gradient_rms.item()),
                float(parameter_rms.item()),
                float((gradient_rms / parameter_rms.clamp_min(1e-12)).item()),
            )

        def module_list(*items: nn.Module | None) -> nn.ModuleList:
            return nn.ModuleList([item for item in items if item is not None])

        if self.config.architecture == COND_GEMMA_ARCHITECTURE:
            modules = {
                "dino": self.model.dino,
                "conditioner": self.model.cond_encoder,
                "expert": self.model.gemma_expert,
                "state_path": module_list(
                    self.model.state_proj,
                    self.model.expert_state_proj,
                    self.model.state_norm,
                ),
                "skill_path": module_list(
                    self.model.skill_proj, self.model.skill_norm
                ),
                "action_io": module_list(
                    self.model.action_in_proj, self.model.action_out_proj
                ),
                "time_mlp": module_list(
                    self.model.time_mlp_in, self.model.time_mlp_out
                ),
            }
            if self.model.expert_state_proj is not None:
                modules["cond_state_projection"] = self.model.state_proj
                modules["expert_state_projection"] = (
                    self.model.expert_state_proj
                )
            if self.model.mode_latent_mlp is not None:
                modules["mode_latent"] = self.model.mode_latent_mlp
            context_norms = module_list(
                self.model.context_input_norms,
                self.model.context_post_attention_norms,
            )
            if any(True for _ in context_norms.parameters()):
                modules["expert_context_norms"] = context_norms
        else:
            cross_attention = nn.ModuleList(
                [
                    block.visual_cross_attention
                    for block in self.model.expert.blocks
                    if block.cross_attention
                ]
            )
            self_attention = nn.ModuleList(
                [block.self_attention for block in self.model.expert.blocks]
            )
            expert_mlps = nn.ModuleList(
                [block.mlp for block in self.model.expert.blocks]
            )
            visual_residual_gates = nn.ParameterList(
                [
                    gate
                    for block in self.model.expert.blocks
                    if (gate := getattr(block, "visual_residual_gate", None)) is not None
                ]
            )
            modules = {
                "dino": self.model.dino,
                "state_path": module_list(
                    self.model.state_proj, self.model.state_norm
                ),
                "skill_path": module_list(
                    self.model.skill_proj, self.model.skill_norm
                ),
                "expert_self_attention": self_attention,
                "expert_mlp": expert_mlps,
                "action_io": module_list(
                    self.model.action_in_proj, self.model.action_out_proj
                ),
                "time_mlp": module_list(
                    self.model.time_mlp_in, self.model.time_mlp_out
                ),
            }
            if len(cross_attention) > 0:
                modules["expert_cross_attention"] = cross_attention
            if len(visual_residual_gates) > 0:
                modules["expert_visual_residual_gates"] = visual_residual_gates
            if self.model.visual_condition_projection is not None:
                modules["visual_condition_projection"] = (
                    self.model.visual_condition_projection
                )
        if self.model.top_resampler is not None:
            modules["top_resampler"] = self.model.top_resampler
        if self.model.wrist_resampler is not None:
            modules["wrist_resampler"] = self.model.wrist_resampler
        metrics = {}
        for name, module in modules.items():
            gradient_rms, parameter_rms, ratio = grad_stats(module)
            metrics[f"vsa_debug/gradient/preclip/{name}_grad_rms"] = gradient_rms
            metrics[f"vsa_debug/parameter/{name}_rms"] = parameter_rms
            metrics[f"vsa_debug/gradient/preclip/{name}_to_parameter_rms_ratio"] = ratio
        # The scheduled training update is complete. Avoid carrying debug mode
        # into checkpoint-time evaluation or any auxiliary forward.
        self.model._vsa_debug_active = False
        if hasattr(self.model, "expert"):
            self.model.expert.debug_enabled = False
        return metrics

    def _torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32

    def reset(self) -> None:
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._mode_latent_cache: Tensor | None = None
        self._mode_latent_skill_code: Tensor | None = None
        self._last_hindsight_mode_noise: Tensor | None = None
        self._last_eval_baseline_mode_latent: Tensor | None = None
        self._last_eval_mode_latent: Tensor | None = None

    @torch.no_grad()
    def _inference_mode_latent(self, skill_code: Tensor) -> Tensor | None:
        """Keep one sampled mode code while the active discrete skill is unchanged."""
        if not getattr(self.config, "skill_flow_latent_best_of_n_enabled", False):
            return None
        flat_code = skill_code.detach().reshape(-1)
        needs_new_cache = (
            self._mode_latent_cache is None
            or self._mode_latent_skill_code is None
            or self._mode_latent_cache.shape[0] != flat_code.shape[0]
            or self._mode_latent_cache.device != flat_code.device
        )
        if needs_new_cache:
            self._mode_latent_cache = self.model.sample_mode_latent(
                (flat_code.shape[0],), flat_code.device
            )
            self._mode_latent_skill_code = flat_code.clone()
            return self._mode_latent_cache
        changed = flat_code != self._mode_latent_skill_code
        if bool(changed.any()):
            fresh = self.model.sample_mode_latent(
                (flat_code.shape[0],), flat_code.device
            )
            self._mode_latent_cache = torch.where(
                changed[:, None], fresh, self._mode_latent_cache
            )
            self._mode_latent_skill_code = flat_code.clone()
        return self._mode_latent_cache

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
        if source_config.get("type") not in {"skill_expert", "skill_aux"}:
            raise ValueError(
                "Predictor source must be a skill_expert or skill_aux checkpoint, got "
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
                "Stage-1 predictor module contract mismatch: " + "; ".join(mismatches)
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
        if source_config.get("type") not in {"skill_expert", "skill_aux"}:
            raise ValueError(
                "Predictor source must be a skill_expert or skill_aux checkpoint, got "
                f"{source_config.get('type')!r}."
            )
        if not source_config.get("train_skill_predictor", False):
            raise ValueError("Stage-1 predictor source has no trained predictor.")
        # This branch attaches a predictor to a target that has none of its own,
        # so its unused skill_predictor_* defaults must not dictate the module
        # shape. Only the skill geometry has to agree; the module itself is
        # rebuilt from the source checkpoint that trained these weights.
        mismatches = [
            f"{field}: checkpoint={source_config.get(field)!r}, "
            f"current={getattr(self.config, field)!r}"
            for field in _PREDICTOR_SKILL_GEOMETRY_FIELDS
            if source_config.get(field) != getattr(self.config, field)
        ]
        if mismatches:
            raise ValueError(
                "Stage-1 predictor skill geometry mismatch: " + "; ".join(mismatches)
            )
        predictor_config = copy.deepcopy(self.config)
        for field in _PREDICTOR_MODULE_FIELDS:
            if field in source_config:
                setattr(predictor_config, field, source_config[field])

        predictor = FrozenVLMSkillPredictor(predictor_config).to(
            dtype=self._torch_dtype()
        )
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
        if source_config.get("type") not in {"skill_expert", "skill_aux"}:
            raise ValueError(
                "Terminator source must be a skill_expert or skill_aux checkpoint, got "
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

        # Rebuild from the source checkpoint's complete terminator contract.
        # Reusing the target policy's attached module is incorrect when an
        # external overlay changes context (proprio/prev_action/none), camera
        # selection, fusion architecture, or vision backbone (ResNet/DINO).
        def optional_bool(key: str) -> bool | None:
            value = source_config.get(key)
            return None if value is None else bool(value)

        terminator = build_trainable_fsq_terminator(
            self.config.fsq_path,
            termination_only=optional_bool("terminator_termination_only"),
            context=source_config.get("terminator_context"),
            cameras=source_config.get("terminator_cameras", "both"),
            default_arch=source_config.get("terminator_arch"),
            vision_backbone=source_config.get("terminator_vision_backbone"),
            freeze_vision_encoder=optional_bool(
                "terminator_freeze_vision_encoder"
            ),
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

    def load_external_image_only_terminator(
        self, checkpoint_path: str | Path | None
    ) -> None:
        """Attach the image-only terminator trained by ``skill_aux``."""
        path = Path(str(checkpoint_path or ""))
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Stage-1 image-only terminator config not found: {config_path}"
            )
        source_config = json.loads(config_path.read_text())
        if source_config.get("type") != "skill_aux":
            raise ValueError(
                "Image-only terminator source must be a skill_aux checkpoint, got "
                f"{source_config.get('type')!r}."
            )
        if not source_config.get("train_image_only_terminator", False):
            raise ValueError(
                "Stage-1 image-only terminator source has no trained image-only "
                "terminator."
            )
        if source_config.get("skill_fsq_levels") != self.config.skill_fsq_levels:
            raise ValueError(
                "Stage-1 image-only terminator FSQ mismatch: "
                f"checkpoint={source_config.get('skill_fsq_levels')!r}, "
                f"current={self.config.skill_fsq_levels!r}."
            )

        terminator = getattr(self.model, "fsq_image_term_train", None)
        if terminator is None:
            terminator = build_fsq_image_only_terminator(
                self.config.fsq_path,
                termination_only=bool(
                    source_config.get("image_only_terminator_termination_only", False)
                ),
            ).to(dtype=torch.float32)
        loaded = _load_complete_terminator_parameters(
            terminator,
            path,
            prefix="model.fsq_image_term_train.",
            label="image-only terminator",
        )
        device = next(self.model.parameters()).device
        terminator.to(device=device, dtype=torch.float32)
        terminator.requires_grad_(False).eval()
        self.model.fsq_image_term_train = terminator
        log.info(
            "Stage 1 <- external image-only terminator %s: loaded %d tensors.",
            path,
            loaded,
        )

    def image_only_terminator_predict(
        self,
        true_code: Tensor,
        image: Tensor,
        wrist_image: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run an attached image-only terminator on current camera frames."""
        terminator = getattr(self.model, "fsq_image_term_train", None)
        if terminator is None:
            raise RuntimeError("Image-only terminator is unavailable.")
        device = next(terminator.parameters()).device
        dtype = next(terminator.parameters()).dtype
        z_q = self.model._code_to_zq(  # noqa: SLF001
            true_code.to(self.model._fsq_strides.device)  # noqa: SLF001
        ).to(device=device, dtype=dtype)
        return terminator(
            z_q,
            image.to(device=device, dtype=dtype),
            wrist_image.to(device=device, dtype=dtype),
        )

    # I/O interface layers stay on AdamW under the Muon probe, following the
    # standard Muon convention of excluding embeddings and heads. Non-2D
    # tensors (norms, biases, DINO patch/cls embeddings) are excluded by shape.
    _MUON_ADAMW_ONLY_NAME_PARTS = (
        "embed",
        "action_in_proj",
        "action_out_proj",
        "state_proj",
        "skill_proj",
        "image_proj",
    )

    def _maybe_split_param_groups_for_muon(self, groups: list[dict]) -> list[dict]:
        """With use_muon, split each group into Muon(2D)/AdamW halves; else no-op."""
        if not getattr(self.config, "use_muon", False):
            return groups
        adamw_only_ids = {
            id(parameter)
            for name, parameter in self.named_parameters()
            if any(part in name for part in self._MUON_ADAMW_ONLY_NAME_PARTS)
        }
        return split_param_groups_for_muon(groups, adamw_only_ids)

    def get_optim_params(self) -> list[dict]:
        """Return only action-model and DINO groups; auxiliaries are frozen."""
        for name in (
            "skill_predictor",
            "fsq_term_train",
            "fsq_image_term_train",
            "fsq_wrist_term_train",
        ):
            auxiliary = getattr(self.model, name, None)
            if auxiliary is not None:
                auxiliary.requires_grad_(False).eval()
        if getattr(self.config, "architecture", VSA_ARCHITECTURE) == COND_GEMMA_ARCHITECTURE:
            return self._get_cond_gemma_optim_params()
        dino_parameters = [
            parameter for parameter in self.model.dino.parameters() if parameter.requires_grad
        ]
        excluded_ids = {id(parameter) for parameter in dino_parameters}

        base_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in excluded_ids
        ]
        groups = (
            [{"params": base_parameters, "group_name": "vsa", "lr_scale": 1.0}]
            if base_parameters
            else []
        )
        if not dino_parameters:
            raise RuntimeError("The fully trainable Stage-1 DINO optimizer group is empty.")
        groups.append(
            {
                "params": dino_parameters,
                "lr": self.config.optimizer_lr * self.config.dino_lr_scale,
                "lr_scale": self.config.dino_lr_scale,
                "group_name": "dino",
            }
        )
        groups = self._maybe_split_param_groups_for_muon(groups)
        for group in groups:
            log.info(
                "Stage-1 optimizer group %s: %.1fM params, lr_scale=%g, lr=%g",
                group.get("group_name", "unnamed"),
                sum(parameter.numel() for parameter in group["params"]) / 1e6,
                float(group.get("lr_scale", 1.0)),
                float(group.get("lr", self.config.optimizer_lr)),
            )
        return groups

    def _get_cond_gemma_optim_params(self) -> list[dict]:
        """Build Cond-Gemma action-model and relative-DINO-LR groups only."""
        excluded_ids: set[int] = set()
        dino_parameters = []
        if (
            self.model.dino is not None
            and not self.config.freeze_vision_encoder
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
        groups = (
            [{"params": base_parameters, "group_name": "cond_gemma", "lr_scale": 1.0}]
            if base_parameters
            else []
        )
        if dino_parameters:
            # dino_lr is retained only so historical saved configs remain loadable.
            # New unified Stage-1 runs always use the relative dino_lr_scale.
            dino_lr = (
                self.config.dino_lr
                if self.config.dino_lr is not None
                else self.config.optimizer_lr * self.config.dino_lr_scale
            )
            groups.append(
                {
                    "params": dino_parameters,
                    "lr": dino_lr,
                    "lr_scale": dino_lr / self.config.optimizer_lr,
                    "group_name": "dino",
                }
            )
        groups = self._maybe_split_param_groups_for_muon(groups)
        for group in groups:
            log.info(
                "Stage-1 optimizer group %s: %.1fM params, lr_scale=%g, lr=%g",
                group.get("group_name", "unnamed"),
                sum(parameter.numel() for parameter in group["params"]) / 1e6,
                float(group.get("lr_scale", 1.0)),
                float(group.get("lr", self.config.optimizer_lr)),
            )
        return groups

    def _collect_images(self, batch: dict) -> list[Tensor]:
        device = next(self.parameters()).device
        camera_keys = (
            "observation.images.image",
            "observation.images.wrist_image",
        )
        missing = [key for key in camera_keys if key not in batch]
        if missing:
            raise ValueError(
                "Stage 1 requires ordered top and wrist cameras; "
                f"missing={missing}."
            )
        images = []
        for key in camera_keys:
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
            directional_values = {
                "early_start_pmax": getattr(
                    self.config, "transition_jitter_early_start_pmax", -1
                ),
                "late_start_pmax": getattr(
                    self.config, "transition_jitter_late_start_pmax", -1
                ),
                "early_end_pmax": getattr(
                    self.config, "transition_jitter_early_end_pmax", -1
                ),
                "late_end_pmax": getattr(
                    self.config, "transition_jitter_late_end_pmax", -1
                ),
            }
            directional_kwargs = (
                directional_values
                if any(value >= 0 for value in directional_values.values())
                else {}
            )
            index, _ = choose_jitter_torch(
                index,
                batch["skill_ds"].long().reshape(-1),
                batch["skill_de"].long().reshape(-1),
                batch["skill_sequence_len"].long().reshape(-1),
                self.config.transition_jitter_pmax,
                distribution=self.config.transition_jitter_distribution,
                **directional_kwargs,
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

    def _valid_action_steps(self, actions: Tensor, batch: dict) -> Tensor:
        """Return action offsets supervised by the selected loss-mask contract."""
        valid = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        if "action_is_pad" in batch:
            action_is_pad = batch["action_is_pad"].to(actions.device).bool()
            if action_is_pad.ndim != 2 or action_is_pad.shape[0] != actions.shape[0]:
                raise ValueError(
                    "action_is_pad must have shape [B,T], got "
                    f"{tuple(action_is_pad.shape)} for actions {tuple(actions.shape)}."
                )
            if action_is_pad.shape[1] < actions.shape[1]:
                raise ValueError(
                    "action_is_pad is shorter than the requested action horizon: "
                    f"{action_is_pad.shape[1]} < {actions.shape[1]}."
                )
            valid &= ~action_is_pad[:, : actions.shape[1]]
        if getattr(self.config, "mask_actions_after_skill_end", False):
            boundary_key = (
                "skill_effective_de"
                if "skill_effective_de" in batch
                else "skill_de"
            )
            if boundary_key not in batch:
                raise KeyError(
                    "mask_actions_after_skill_end=true requires batch['skill_effective_de'] "
                    "or batch['skill_de']."
                )
            if (
                getattr(self.config, "transition_jitter_pmax", 0) > 0
                and boundary_key != "skill_effective_de"
            ):
                raise KeyError(
                    "Transition jitter with skill-end loss masking requires "
                    "batch['skill_effective_de'] from SkillVLADataset."
                )
            distance_to_end = batch[boundary_key].to(actions.device).long().reshape(-1)
            if distance_to_end.shape[0] != actions.shape[0]:
                raise ValueError(
                    "skill_de batch size does not match actions: "
                    f"{distance_to_end.shape[0]} != {actions.shape[0]}."
                )
            if bool((distance_to_end < 0).any()):
                raise ValueError("skill_de must be non-negative.")
            offsets = torch.arange(actions.shape[1], device=actions.device).unsqueeze(0)
            valid &= offsets <= distance_to_end.unsqueeze(1)
        return valid

    @staticmethod
    def _cumulative_xyz_loss(
        predicted_actions: Tensor,
        target_actions: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Prefix cumulative XYZ error with one horizon-scale normalization.

        Prefixes retain equal weight in the raw trajectory objective. The raw
        per-sample mean is divided once by ``(valid_steps + 1) / 2`` so its
        expected scale matches stepwise MSE for independent delta errors,
        without applying distinct 1/t weights to individual prefixes.
        """
        if predicted_actions.shape[-1] < 3 or target_actions.shape[-1] < 3:
            raise ValueError("cumulative XYZ loss requires at least three action dimensions.")
        sample_valid = valid.any(dim=1)
        if not bool(sample_valid.any()):
            raise ValueError("cumulative XYZ loss received a batch with no valid action steps.")
        valid_float = valid.float()
        delta_error = (
            predicted_actions[..., :3].float() - target_actions[..., :3].float()
        ) * valid_float.unsqueeze(-1)
        prefix_error = delta_error.cumsum(dim=1)
        prefix_mse = prefix_error.square().mean(dim=-1)
        valid_count = valid.sum(dim=1).clamp(min=1).to(prefix_mse.dtype)
        raw_per_sample = (prefix_mse * valid_float).sum(dim=1) / valid_count
        horizon_normalizer = (valid_count + 1.0) / 2.0
        normalized_per_sample = raw_per_sample / horizon_normalizer
        selected = sample_valid.to(prefix_mse.dtype)
        raw_loss = (raw_per_sample * selected).sum() / selected.sum().clamp(min=1.0)
        normalized_loss = (
            (normalized_per_sample * selected).sum()
            / selected.sum().clamp(min=1.0)
        )
        return normalized_loss, raw_loss, normalized_per_sample, raw_per_sample

    @staticmethod
    def _masked_action_mse(
        squared_error: Tensor,
        valid: Tensor,
        *,
        sample_mask: Tensor | None = None,
        step_slice: slice | None = None,
        dim_slice: slice | None = None,
    ) -> Tensor:
        """Average selected action errors without letting padding change the scale."""
        values = squared_error
        selected_valid = valid
        if step_slice is not None:
            values = values[:, step_slice]
            selected_valid = selected_valid[:, step_slice]
        if dim_slice is not None:
            values = values[..., dim_slice]
        if sample_mask is not None:
            selected_valid = selected_valid & sample_mask[:, None]
        denominator = selected_valid.sum() * values.shape[-1]
        numerator = (
            values * selected_valid.to(values.dtype).unsqueeze(-1)
        ).sum()
        denominator = denominator.to(values.dtype)
        return torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1),
            torch.full_like(numerator, float("nan")),
        )

    @classmethod
    def _action_diagnostic_losses(
        cls,
        squared_error: Tensor,
        valid: Tensor,
        flow_time: Tensor | None,
    ) -> dict[str, float]:
        """Architecture-neutral diagnostics for Stage-1 ablation comparisons."""
        tensor_metrics: dict[str, Tensor] = {}
        if flow_time is not None:
            time = flow_time.detach().float().reshape(-1)
            if time.shape[0] != squared_error.shape[0]:
                raise ValueError(
                    "Flow timestep batch size does not match action residual batch size."
                )
            bins = (
                ("t_0_025", 0.0, 0.25),
                ("t_025_050", 0.25, 0.5),
                ("t_050_075", 0.5, 0.75),
                ("t_075_100", 0.75, float("inf")),
            )
            for name, lower, upper in bins:
                value = cls._masked_action_mse(
                    squared_error,
                    valid,
                    sample_mask=(time >= lower) & (time < upper),
                )
                tensor_metrics[f"flow_timestep/{name}_loss"] = value

        if squared_error.shape[-1] == 7:
            for name, dims in (
                ("translation", slice(0, 3)),
                ("rotation", slice(3, 6)),
                ("gripper", slice(6, 7)),
            ):
                value = cls._masked_action_mse(
                    squared_error, valid, dim_slice=dims
                )
                tensor_metrics[f"action_component/{name}_loss"] = value

        horizon = squared_error.shape[1]
        boundaries = (0, round(horizon / 3), round(2 * horizon / 3), horizon)
        for name, start, end in zip(
            ("early", "middle", "late"), boundaries[:-1], boundaries[1:], strict=True
        ):
            if start == end:
                continue
            value = cls._masked_action_mse(
                squared_error, valid, step_slice=slice(start, end)
            )
            tensor_metrics[f"action_horizon/{name}_loss"] = value
        if not tensor_metrics:
            return {}
        # A single transfer keeps these diagnostics from introducing one GPU
        # synchronization per metric on every optimizer step.
        values = torch.stack(tuple(tensor_metrics.values())).detach().float().cpu().tolist()
        return dict(zip(tensor_metrics, values, strict=True))

    def _skill_flow_training_target(
        self, batch: dict
    ) -> tuple[Tensor, Tensor]:
        """Return the configured auxiliary trajectory and its padding mask."""
        if self.config.skill_flow_target == "canonical":
            if SKILL_CANONICAL_ACTIONS not in batch:
                raise KeyError(
                    "arch0_skill requires batch['skill_canonical_actions']; "
                    "construct training data with SkillVLADataset."
                )
            if SKILL_CANONICAL_ACTION_IS_PAD not in batch:
                raise KeyError(
                    "arch0_skill requires batch['skill_canonical_action_is_pad']."
                )
            actions = pad_vector(
                batch[SKILL_CANONICAL_ACTIONS], self.config.max_action_dim
            )
            is_pad = batch[SKILL_CANONICAL_ACTION_IS_PAD].to(
                actions.device
            ).bool()
            return actions, is_pad
        if self.config.skill_flow_target == "extended_chunk":
            horizon = int(self.config.skill_flow_max_length)
            if batch[ACTION].shape[1] < horizon:
                raise ValueError(
                    "Extended skill-flow target is shorter than its configured "
                    f"horizon: {batch[ACTION].shape[1]} < {horizon}."
                )
            actions = pad_vector(
                batch[ACTION][:, :horizon], self.config.max_action_dim
            )
            return actions, ~self._valid_action_steps(actions, batch)
        raise RuntimeError(
            f"Unsupported skill_flow_target={self.config.skill_flow_target!r}."
        )

    def _skill_flow_noise(self, actions: Tensor, main_noise: Tensor) -> Tensor:
        """Build auxiliary noise while preserving the existing sharing contract."""
        if self.config.skill_flow_target == "canonical":
            return self.model.sample_noise(actions.shape, actions.device).to(actions.dtype)
        if (
            main_noise.shape[0] != actions.shape[0]
            or main_noise.shape[2] != actions.shape[2]
            or main_noise.shape[1] > actions.shape[1]
        ):
            raise ValueError(
                "Main and extended skill-flow noise shapes are incompatible: "
                f"main={tuple(main_noise.shape)}, extended={tuple(actions.shape)}."
            )
        tail_length = actions.shape[1] - main_noise.shape[1]
        tail_noise = self.model.sample_noise(
            (actions.shape[0], tail_length, actions.shape[2]), actions.device
        )
        return torch.cat(
            (main_noise.to(actions.dtype), tail_noise.to(actions.dtype)), dim=1
        )

    @staticmethod
    def _repeat_top_k(tensor: Tensor | None, top_k: int) -> Tensor | None:
        """Repeat B entries as [b0*k, b1*k, ...] without changing order."""
        if tensor is None or top_k == 1:
            return tensor
        return tensor[:, None].expand(-1, top_k, *tensor.shape[1:]).reshape(
            tensor.shape[0] * top_k, *tensor.shape[1:]
        )

    @staticmethod
    def _masked_flow_per_sample(
        residual: Tensor, valid: Tensor, real_dim: int
    ) -> Tensor:
        squared = residual[..., :real_dim].square()
        valid_float = valid.to(squared.dtype).unsqueeze(-1)
        valid_count = valid.sum(dim=1).to(squared.dtype)
        if bool((valid_count == 0).any()):
            raise ValueError("Flow batch contains an empty trajectory.")
        return (squared * valid_float).sum(dim=(1, 2)) / (valid_count * real_dim)

    def _finish_mode_latent_assignment(
        self,
        candidates: Tensor,
        scores: Tensor,
        *,
        ranking_route: str,
    ) -> tuple[Tensor, dict[str, float]]:
        """Select the best candidates and expose route-agnostic diagnostics."""
        top_k = int(self.config.skill_flow_latent_top_k)
        selected_indices = scores.topk(top_k, dim=1, largest=False).indices
        selected = candidates.gather(
            1,
            selected_indices[..., None].expand(
                -1, -1, int(self.config.skill_flow_latent_dim)
            ),
        )
        if candidates.shape[1] > 1:
            two_best = scores.topk(2, dim=1, largest=False).values
            margin = two_best[:, 1] - two_best[:, 0]
        else:
            margin = torch.zeros_like(scores[:, 0])
        flat_selected = selected.reshape(-1, selected.shape[-1]).float()
        stats = {
            "mode_latent/candidate_loss_mean": float(scores.mean().item()),
            "mode_latent/selected_loss_mean": float(
                scores.gather(1, selected_indices).mean().item()
            ),
            "mode_latent/best_margin_mean": float(margin.mean().item()),
            "mode_latent/selected_x_mean": float(flat_selected[:, 0].mean().item()),
            "mode_latent/selected_x_std": float(
                flat_selected[:, 0].std(unbiased=False).item()
            ),
            "mode_latent/selected_y_mean": float(flat_selected[:, 1].mean().item()),
            "mode_latent/selected_y_std": float(
                flat_selected[:, 1].std(unbiased=False).item()
            ),
            "mode_latent/selected_radius_mean": float(
                flat_selected.square().sum(dim=1).sqrt().mean().item()
            ),
            "mode_latent/candidates": float(candidates.shape[1]),
            "mode_latent/top_k": float(top_k),
            "mode_latent/assignment_timesteps": float(
                self.config.skill_flow_latent_assignment_timesteps
            ),
            "mode_latent/ranking_main": float(ranking_route == "main"),
        }
        return selected, stats

    @torch.no_grad()
    def _select_skill_flow_mode_latents(
        self,
        actions: Tensor,
        skill_code: Tensor,
        action_is_pad: Tensor,
        noise: Tensor,
        main_time: Tensor,
        state: Tensor | None,
        real_dim: int,
    ) -> tuple[Tensor, dict[str, float]]:
        """Select per-sample z candidates using M-timestep skill-only FM loss."""
        candidates_n = int(self.config.skill_flow_latent_candidates)
        timesteps_n = int(self.config.skill_flow_latent_assignment_timesteps)
        batch_size = actions.shape[0]
        candidates = self.model.sample_mode_latent(
            (batch_size, candidates_n), actions.device
        )
        assignment_times = [main_time]
        assignment_times.extend(
            self.model.sample_time(batch_size, actions.device)
            for _ in range(timesteps_n - 1)
        )
        valid = ~action_is_pad
        scores = torch.zeros(
            batch_size, candidates_n, dtype=torch.float32, device=actions.device
        )
        kwargs = {"noise": noise}
        if getattr(self.config, "skill_flow_state_conditioned", False):
            kwargs["state"] = state
        # Looping over candidates keeps peak memory at the ordinary batch size;
        # these assignment passes intentionally build no backward graph.
        for time in assignment_times:
            for candidate_index in range(candidates_n):
                residual = self.model.skill_only_flow_residual(
                    actions,
                    skill_code,
                    action_is_pad,
                    time=time,
                    mode_latent=candidates[:, candidate_index],
                    **kwargs,
                )
                scores[:, candidate_index] += self._masked_flow_per_sample(
                    residual, valid, real_dim
                ).float()
        scores /= float(timesteps_n)
        return self._finish_mode_latent_assignment(
            candidates, scores, ranking_route="skill_only"
        )

    @torch.no_grad()
    def _select_main_flow_mode_latents(
        self,
        actions: Tensor,
        images: list[Tensor],
        state: Tensor | None,
        skill_code: Tensor,
        valid: Tensor,
        noise: Tensor,
        main_time: Tensor,
        real_dim: int,
    ) -> tuple[Tensor, dict[str, float]]:
        """Select z by the deployed vision/state/skill action-chunk FM loss."""
        candidates_n = int(self.config.skill_flow_latent_candidates)
        timesteps_n = int(self.config.skill_flow_latent_assignment_timesteps)
        batch_size = actions.shape[0]
        candidates = self.model.sample_mode_latent(
            (batch_size, candidates_n), actions.device
        )
        assignment_times = [main_time]
        assignment_times.extend(
            self.model.sample_time(batch_size, actions.device)
            for _ in range(timesteps_n - 1)
        )
        scores = torch.zeros(
            batch_size, candidates_n, dtype=torch.float32, device=actions.device
        )

        # Candidate assignment is graph-free. Encode the images once and reuse
        # that memory for all N x M post-vision passes; otherwise main-route
        # ranking would redundantly run DINO for every candidate.
        previous_debug = bool(getattr(self.model, "_vsa_debug_active", False))
        previous_checkpointing = bool(
            getattr(self.model, "_gradient_checkpointing", False)
        )
        self.model._vsa_debug_active = False
        self.model._gradient_checkpointing = False
        try:
            condition_tokens = self.model._condition_tokens(
                images, batch_size=batch_size
            )
            source = noise.to(actions.dtype)
            target_velocity = source - actions
            for time in assignment_times:
                x_t = (
                    time[:, None, None] * source
                    + (1.0 - time[:, None, None]) * actions
                )
                for candidate_index in range(candidates_n):
                    predicted_velocity = self.model._predict_velocity_from_condition(
                        condition_tokens,
                        x_t,
                        state,
                        skill_code,
                        time,
                        candidates[:, candidate_index],
                    )
                    residual = target_velocity - predicted_velocity
                    scores[:, candidate_index] += self._masked_flow_per_sample(
                        residual, valid, real_dim
                    ).float()
        finally:
            self.model._vsa_debug_active = previous_debug
            self.model._gradient_checkpointing = previous_checkpointing

        scores /= float(timesteps_n)
        return self._finish_mode_latent_assignment(
            candidates, scores, ranking_route="main"
        )

    def forward(self, batch: dict, reduction: str = "mean"):
        if batch[ACTION].shape[1] < self.config.chunk_size:
            raise ValueError(
                "Training action horizon is shorter than chunk_size: "
                f"{batch[ACTION].shape[1]} < {self.config.chunk_size}."
            )
        # *_skill_chunk asks the dataset for a longer auxiliary horizon. The
        # rollout/main flow contract remains exactly chunk_size steps.
        base_actions = pad_vector(
            batch[ACTION][:, : self.config.chunk_size],
            self.config.max_action_dim,
        )
        real_dim = self.config.output_features[ACTION].shape[0]
        if self.config.architecture == COND_GEMMA_ARCHITECTURE:
            route = normalize_conditioning_route(self.config.conditioning_route)
            base_state = (
                None
                if route in STATELESS_CONDITIONING_ROUTES
                else pad_vector(batch[OBS_STATE], self.config.max_state_dim)
            )
            base_skill_code = (
                None
                if route in SKILLLESS_CONDITIONING_ROUTES
                else self._training_skill_code(batch)
            )
            base_images = (
                []
                if route in VISIONLESS_CONDITIONING_ROUTES
                else self._collect_images(batch)
            )
        else:
            base_state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
            base_skill_code = self._training_skill_code(batch)
            base_images = self._collect_images(batch)
        base_valid = self._valid_action_steps(base_actions, batch)
        latent_best_of_n = bool(
            getattr(self.config, "skill_flow_latent_best_of_n_enabled", False)
        )
        mode_latent_stats: dict[str, float] = {}
        selected_mode_latent = None
        skill_flow_actions = None
        skill_flow_is_pad = None
        skill_flow_noise = None
        top_k = 1
        if latent_best_of_n:
            if base_skill_code is None:
                raise RuntimeError("latent Best-of-N requires skill conditioning.")
            skill_flow_actions, skill_flow_is_pad = self._skill_flow_training_target(
                batch
            )
            main_time = self.model.sample_time(base_actions.shape[0], base_actions.device)
            main_noise = self.model.sample_noise(base_actions.shape, base_actions.device).to(
                base_actions.dtype
            )
            skill_flow_noise = self._skill_flow_noise(skill_flow_actions, main_noise)
            ranking_route = str(
                getattr(
                    self.config,
                    "skill_flow_latent_ranking_route",
                    "skill_only",
                )
            )
            if ranking_route == "main":
                selected_mode_latent, mode_latent_stats = (
                    self._select_main_flow_mode_latents(
                        base_actions,
                        base_images,
                        base_state,
                        base_skill_code,
                        base_valid,
                        main_noise,
                        main_time,
                        real_dim,
                    )
                )
            else:
                selected_mode_latent, mode_latent_stats = (
                    self._select_skill_flow_mode_latents(
                        skill_flow_actions,
                        base_skill_code,
                        skill_flow_is_pad,
                        skill_flow_noise,
                        main_time,
                        base_state,
                        real_dim,
                    )
                )
            top_k = int(self.config.skill_flow_latent_top_k)
            mode_latent = selected_mode_latent.reshape(
                base_actions.shape[0] * top_k,
                int(self.config.skill_flow_latent_dim),
            )
            actions = self._repeat_top_k(base_actions, top_k)
            valid = self._repeat_top_k(base_valid, top_k)
            state = self._repeat_top_k(base_state, top_k)
            skill_code = self._repeat_top_k(base_skill_code, top_k)
            images = [self._repeat_top_k(image, top_k) for image in base_images]
            residual = self.model(
                images,
                state,
                skill_code,
                actions,
                noise=self._repeat_top_k(main_noise, top_k),
                time=self._repeat_top_k(main_time, top_k),
                mode_latent=mode_latent,
            )[..., :real_dim]
        else:
            actions = base_actions
            valid = base_valid
            state = base_state
            skill_code = base_skill_code
            images = base_images
            residual = self.model(images, state, skill_code, actions)[..., :real_dim]
        squared_error = residual.square()
        valid_float = valid.to(squared_error.dtype).unsqueeze(-1)
        valid_per_sample = valid.sum(dim=1).clamp(min=1).to(squared_error.dtype)
        main_per_selected = (squared_error * valid_float).sum(dim=(1, 2)) / (
            valid_per_sample * real_dim
        )
        per_sample = (
            main_per_selected.reshape(base_actions.shape[0], top_k).mean(dim=1)
            if top_k > 1
            else main_per_selected
        )
        valid_steps = valid.sum().clamp(min=1).to(squared_error.dtype)
        action_loss = (squared_error * valid_float).sum() / (valid_steps * real_dim)
        cumulative_xyz_loss = None
        cumulative_xyz_raw_loss = None
        action_objective = action_loss
        objective_per_sample = per_sample
        if getattr(self.config, "cumulative_xyz_loss_enabled", False):
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
            cumulative_weight = getattr(
                self.config, "cumulative_xyz_loss_weight", 0.5
            )
            if top_k > 1:
                cumulative_xyz_per_sample = cumulative_xyz_per_sample.reshape(
                    base_actions.shape[0], top_k
                ).mean(dim=1)
            action_objective = action_loss + cumulative_weight * cumulative_xyz_loss
            objective_per_sample = per_sample + cumulative_weight * cumulative_xyz_per_sample
        skill_flow_loss = None
        skill_flow_per_sample = None
        if getattr(self.config, "skill_flow_enabled", False):
            if self.config.architecture != COND_GEMMA_ARCHITECTURE:
                raise RuntimeError("skill_flow_enabled requires the Cond-Gemma Arch0 model.")
            if skill_flow_actions is None or skill_flow_is_pad is None:
                skill_flow_actions, skill_flow_is_pad = (
                    self._skill_flow_training_target(batch)
                )
            shared_time = getattr(self.model, "_last_flow_time", None)
            if shared_time is None:
                raise RuntimeError("Main Arch0 flow did not expose its sampled timestep.")
            skill_flow_kwargs = {"time": shared_time}
            if latent_best_of_n:
                if skill_flow_noise is None or selected_mode_latent is None:
                    raise RuntimeError("latent Best-of-N assignment state is missing.")
                skill_flow_actions = self._repeat_top_k(skill_flow_actions, top_k)
                skill_flow_is_pad = self._repeat_top_k(skill_flow_is_pad, top_k)
                skill_flow_kwargs["noise"] = self._repeat_top_k(
                    skill_flow_noise, top_k
                )
                skill_flow_kwargs["mode_latent"] = selected_mode_latent.reshape(
                    base_actions.shape[0] * top_k,
                    int(self.config.skill_flow_latent_dim),
                )
            elif self.config.skill_flow_target == "extended_chunk":
                main_noise = getattr(self.model, "_last_flow_noise", None)
                if main_noise is None:
                    raise RuntimeError(
                        "Main Arch0 flow did not expose its sampled noise."
                    )
                skill_flow_kwargs["noise"] = self._skill_flow_noise(
                    skill_flow_actions, main_noise
                )
            if getattr(self.config, "skill_flow_state_conditioned", False):
                skill_flow_kwargs["state"] = state
            skill_residual = self.model.skill_only_flow_residual(
                skill_flow_actions,
                skill_code,
                skill_flow_is_pad,
                **skill_flow_kwargs,
            )[..., :real_dim]
            skill_squared_error = skill_residual.square()
            skill_valid = ~skill_flow_is_pad
            skill_valid_float = skill_valid.to(skill_squared_error.dtype).unsqueeze(-1)
            skill_valid_per_sample = skill_valid.sum(dim=1).to(
                skill_squared_error.dtype
            )
            if bool((skill_valid_per_sample == 0).any()):
                raise ValueError("Skill-flow batch contains an empty trajectory.")
            # First average timesteps within each trajectory, then average the
            # batch. Long skills therefore do not receive a larger coefficient.
            skill_flow_per_sample = (
                (skill_squared_error * skill_valid_float).sum(dim=(1, 2))
                / (skill_valid_per_sample * real_dim)
            )
            if top_k > 1:
                skill_flow_per_sample = skill_flow_per_sample.reshape(
                    base_actions.shape[0], top_k
                ).mean(dim=1)
            skill_flow_loss = skill_flow_per_sample.mean()
            skill_flow_weight = float(self.config.skill_flow_weight)
            action_objective = action_objective + skill_flow_weight * skill_flow_loss
            objective_per_sample = (
                objective_per_sample + skill_flow_weight * skill_flow_per_sample
            )
        loss_dict = {
            "action_loss": action_loss.detach().item(),
            "conditioning/skill_source_predictor": float(
                getattr(self.config, "training_skill_source", "gt") == "predictor"
            ),
            "regime/transition_jitter_fraction": (
                self._last_transition_jitter_fraction.detach().item()
            ),
        }
        if latent_best_of_n:
            loss_dict.update(mode_latent_stats)
            loss_dict["mode_latent/gain"] = float(
                self.model.mode_latent_gain.detach().float().item()
            )
        if getattr(self.config, "mask_actions_after_skill_end", False):
            # Report the physical dataset batch once; top-K replication is an
            # optimization detail and must not change masking statistics.
            unpadded = torch.ones_like(base_valid)
            if "action_is_pad" in batch:
                unpadded &= ~batch["action_is_pad"].to(base_valid.device).bool()[
                    :, : base_valid.shape[1]
                ]
            unpadded_count = unpadded.sum().clamp(min=1)
            boundary_masked = unpadded & ~base_valid
            loss_dict.update(
                {
                    "loss_mask/skill_end_masked_fraction": (
                        boundary_masked.sum().float() / unpadded_count
                    ).detach().item(),
                    "loss_mask/valid_action_steps_mean": base_valid.sum(dim=1)
                    .float()
                    .mean()
                    .detach()
                    .item(),
                }
            )
            if "skill_effective_de" in batch and "skill_de" in batch:
                effective_de = batch["skill_effective_de"].to(base_valid.device).float()
                original_de = batch["skill_de"].to(base_valid.device).float()
                loss_dict.update(
                    {
                        "loss_mask/effective_boundary_changed_fraction": (
                            effective_de != original_de
                        )
                        .float()
                        .mean()
                        .detach()
                        .item(),
                        "loss_mask/effective_minus_original_de_mean": (
                            effective_de - original_de
                        )
                        .mean()
                        .detach()
                        .item(),
                    }
                )
        loss_dict.update(
            self._action_diagnostic_losses(
                squared_error,
                valid,
                getattr(self.model, "_last_flow_time", None),
            )
        )
        loss_dict.update(
            {
                f"vsa_debug/{name}": value
                for name, value in self.model._last_vsa_debug_stats.items()
            }
        )
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
        if cumulative_xyz_loss is not None and cumulative_xyz_raw_loss is not None:
            cumulative_weight = getattr(
                self.config, "cumulative_xyz_loss_weight", 0.5
            )
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
                        / action_loss.detach().clamp(min=torch.finfo(action_loss.dtype).eps)
                    ).item(),
                    "action_objective": action_objective.detach().item(),
                    "action_flow_weight": 1.0,
                    "action_cumulative_xyz_weight": float(cumulative_weight),
                }
            )
        if skill_flow_loss is not None and skill_flow_per_sample is not None:
            weighted_skill_flow = self.config.skill_flow_weight * skill_flow_loss
            if skill_flow_is_pad is None:
                raise RuntimeError("Skill-flow loss is missing its padding mask.")
            loss_dict.update(
                {
                    "skill_flow/loss": skill_flow_loss.detach().item(),
                    "skill_flow/weighted": weighted_skill_flow.detach().item(),
                    "skill_flow/weight": float(self.config.skill_flow_weight),
                    "skill_flow/valid_steps_mean": (~skill_flow_is_pad)
                    .sum(dim=1)
                    .float()
                    .mean()
                    .item(),
                    "skill_flow/extended_chunk": float(
                        self.config.skill_flow_target == "extended_chunk"
                    ),
                    "skill_flow/state_conditioned": float(
                        self.config.skill_flow_state_conditioned
                    ),
                    "skill_flow/to_main_flow_ratio": (
                        weighted_skill_flow.detach()
                        / action_loss.detach().clamp(
                            min=torch.finfo(action_loss.dtype).eps
                        )
                    ).item(),
                    "skill_flow/total_objective": action_objective.detach().item(),
                }
            )
        if reduction == "none":
            return objective_per_sample, loss_dict
        return action_objective, loss_dict

    @torch.no_grad()
    def select_hindsight_mode_latent(
        self,
        batch: dict,
        target_actions: Tensor,
        target_valid: Tensor,
        *,
        grid_size: int = 3,
        timesteps: int = 2,
        aggregate_windows: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Choose the Stage-1 mode z with minimum GT action-flow residual.

        This evaluation-only oracle mirrors main-route latent assignment used
        during Stage-1 training.  All z candidates share one Gaussian source
        noise and fixed FM scoring times.  The selected z and that exact source
        noise are then consumed together by the next action-chunk rollout.
        Candidate zero is the ordinary sampled per-skill latent, so the oracle
        cannot score worse than that baseline under this scoring objective.
        When ``aggregate_windows`` is true, every batch row is a different
        window from one skill. One common z is selected by averaging its
        residual over all valid actions in all of those windows.
        """
        if not getattr(
            self.config, "skill_flow_latent_best_of_n_enabled", False
        ):
            raise ValueError(
                "Hindsight latent selection requires a latent-enabled Stage-1 "
                "checkpoint."
            )
        if int(self.config.skill_flow_latent_dim) != 2:
            raise ValueError(
                "The grid hindsight oracle currently requires a 2-D mode latent."
            )
        grid_size = int(grid_size)
        timesteps = int(timesteps)
        if grid_size < 2:
            raise ValueError("Hindsight latent grid_size must be at least 2.")
        if timesteps <= 0:
            raise ValueError("Hindsight latent timesteps must be positive.")

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
        if skill_code is None:
            raise ValueError("Hindsight latent selection requires an active skill.")
        images = (
            []
            if route in VISIONLESS_CONDITIONING_ROUTES
            else self._collect_images(batch)
        )
        batch_size = int(skill_code.shape[0])
        real_action_dim = int(self.config.output_features[ACTION].shape[0])
        expected = (batch_size, int(self.config.chunk_size), real_action_dim)
        target_actions = target_actions.to(device=device, dtype=torch.float32)
        target_valid = target_valid.to(device=device, dtype=torch.bool)
        if tuple(target_actions.shape) != expected:
            raise ValueError(
                f"Hindsight target_actions must have shape {expected}, got "
                f"{tuple(target_actions.shape)}."
            )
        if tuple(target_valid.shape) != expected[:2]:
            raise ValueError(
                f"Hindsight target_valid must have shape {expected[:2]}, got "
                f"{tuple(target_valid.shape)}."
            )
        if bool((target_valid.sum(dim=1) == 0).any()):
            raise ValueError("Every hindsight sample needs at least one valid action.")

        aggregate_windows = bool(aggregate_windows)
        if aggregate_windows and not bool(
            (skill_code == skill_code[:1]).all()
        ):
            raise ValueError(
                "Aggregated hindsight windows must all use the same skill code."
            )
        baseline_skill = skill_code[:1] if aggregate_windows else skill_code
        baseline = self._inference_mode_latent(baseline_skill)
        if baseline is None:
            raise RuntimeError("Stage-1 mode-latent sampler returned no latent.")
        axis = torch.linspace(-1.0, 1.0, grid_size, device=device)
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
        candidate_batch_size = 1 if aggregate_windows else batch_size
        grid = grid.unsqueeze(0).expand(candidate_batch_size, -1, -1)
        candidates = torch.cat((baseline[:, None].float(), grid), dim=1)

        actions = pad_vector(target_actions, self.config.max_action_dim).float()
        source = self.model.sample_noise(actions.shape, device).float()
        self._last_hindsight_mode_noise = (
            source[:1] if aggregate_windows else source
        ).detach().clone()
        condition_tokens = self.model._condition_tokens(
            images, batch_size=batch_size
        )
        target_velocity = source - actions
        time_values = torch.arange(
            1, timesteps + 1, device=device, dtype=torch.float32
        ) / float(timesteps + 1)
        denominator = target_valid.sum(dim=1).float() * float(real_action_dim)
        scores = torch.zeros(
            candidate_batch_size,
            candidates.shape[1],
            device=device,
            dtype=torch.float32,
        )

        previous_debug = bool(getattr(self.model, "_vsa_debug_active", False))
        previous_checkpointing = bool(
            getattr(self.model, "_gradient_checkpointing", False)
        )
        self.model._vsa_debug_active = False
        self.model._gradient_checkpointing = False
        try:
            for candidate_index in range(candidates.shape[1]):
                candidate_latent = candidates[:, candidate_index]
                mode_latent = (
                    candidate_latent.expand(batch_size, -1)
                    if aggregate_windows
                    else candidate_latent
                )
                candidate_score = torch.zeros(
                    batch_size, device=device, dtype=torch.float32
                )
                for time_value in time_values:
                    time = torch.full(
                        (batch_size,),
                        float(time_value.item()),
                        device=device,
                        dtype=torch.float32,
                    )
                    x_t = (
                        time[:, None, None] * source
                        + (1.0 - time[:, None, None]) * actions
                    )
                    predicted_velocity = self.model._predict_velocity_from_condition(
                        condition_tokens,
                        x_t,
                        state,
                        skill_code,
                        time,
                        mode_latent,
                    ).float()
                    residual = (
                        target_velocity[..., :real_action_dim]
                        - predicted_velocity[..., :real_action_dim]
                    )
                    candidate_score += (
                        residual.square()
                        * target_valid.to(residual.dtype).unsqueeze(-1)
                    ).sum(dim=(1, 2)) / denominator
                candidate_score = candidate_score / float(timesteps)
                if aggregate_windows:
                    # Weight by the number of valid scalar targets rather than
                    # giving a short final window the same weight as a full one.
                    scores[0, candidate_index] = (
                        candidate_score * denominator
                    ).sum() / denominator.sum()
                else:
                    scores[:, candidate_index] = candidate_score
        finally:
            self.model._vsa_debug_active = previous_debug
            self.model._gradient_checkpointing = previous_checkpointing

        best_indices = scores.argmin(dim=1)
        selected = candidates.gather(
            1,
            best_indices[:, None, None].expand(-1, 1, candidates.shape[-1]),
        ).squeeze(1)
        self._last_eval_baseline_mode_latent = baseline.detach().clone()
        return selected, scores

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **kwargs) -> Tensor:
        self.eval()
        if self.config.architecture == COND_GEMMA_ARCHITECTURE:
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
            images = (
                []
                if route in VISIONLESS_CONDITIONING_ROUTES
                else self._collect_images(batch)
            )
        else:
            state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
            skill_code = self._skill_code(batch)
            images = self._collect_images(batch)
        mode_latent = kwargs.get("mode_latent")
        if (
            skill_code is not None
            and mode_latent is None
            and getattr(self.config, "skill_flow_latent_best_of_n_enabled", False)
        ):
            mode_latent = self._inference_mode_latent(skill_code)
            kwargs["mode_latent"] = mode_latent
        self._last_eval_baseline_mode_latent = (
            None if mode_latent is None else mode_latent.detach().clone()
        )
        latent_override = batch.get(SKILL_FLOW_MODE_LATENT_OVERRIDE)
        if latent_override is not None:
            if mode_latent is None:
                raise ValueError(
                    "A mode-latent override was provided to a latent-disabled policy."
                )
            latent_override = latent_override.to(
                device=mode_latent.device, dtype=mode_latent.dtype
            )
            if tuple(latent_override.shape) != tuple(mode_latent.shape):
                raise ValueError(
                    "Mode-latent override shape mismatch: "
                    f"expected {tuple(mode_latent.shape)}, got "
                    f"{tuple(latent_override.shape)}."
                )
            use_override = torch.isfinite(latent_override).all(
                dim=-1, keepdim=True
            )
            mode_latent = torch.where(use_override, latent_override, mode_latent)
            kwargs["mode_latent"] = mode_latent
        self._last_eval_mode_latent = (
            None if mode_latent is None else mode_latent.detach().clone()
        )

        noise_override = batch.get(SKILL_FLOW_NOISE_OVERRIDE)
        if noise_override is not None:
            batch_size = (
                int(state.shape[0]) if state is not None else int(skill_code.shape[0])
            )
            expected_noise_shape = (
                batch_size,
                int(self.config.chunk_size),
                int(self.config.max_action_dim),
            )
            noise_override = noise_override.to(device=next(self.parameters()).device)
            if tuple(noise_override.shape) != expected_noise_shape:
                raise ValueError(
                    "Flow-noise override shape mismatch: "
                    f"expected {expected_noise_shape}, got "
                    f"{tuple(noise_override.shape)}."
                )
            fallback_noise = self.model.sample_noise(
                expected_noise_shape, noise_override.device
            )
            use_override = torch.isfinite(noise_override).all(
                dim=(1, 2), keepdim=True
            )
            kwargs["noise"] = torch.where(
                use_override, noise_override, fallback_noise
            )
        actions = self.model.sample_actions(images, state, skill_code, **kwargs)
        real_dim = self.config.output_features[ACTION].shape[0]
        return actions[..., :real_dim]

    @torch.no_grad()
    def predict_skill_only_action_chunk(
        self,
        batch: dict,
        *,
        horizon: int | None = None,
        **kwargs,
    ) -> Tensor:
        """Generate actions through the auxiliary skill-only training route."""
        self.eval()
        if self.config.architecture != COND_GEMMA_ARCHITECTURE:
            raise RuntimeError(
                "Skill-only rollout is available only for Cond-Gemma architectures."
            )
        if not getattr(self.config, "skill_flow_enabled", False):
            raise RuntimeError(
                "Checkpoint has no trained skill-only flow path; use arch0_skill, "
                "arch0_skill_chunk, or arch0_2_skill_chunk."
            )
        skill_code = self._skill_code(batch)
        state = None
        if getattr(self.config, "skill_flow_state_conditioned", False):
            state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        if (
            "mode_latent" not in kwargs
            and getattr(self.config, "skill_flow_latent_best_of_n_enabled", False)
        ):
            kwargs["mode_latent"] = self._inference_mode_latent(skill_code)
        actions = self.model.sample_skill_only_actions(
            skill_code=skill_code,
            state=state,
            horizon=horizon,
            **kwargs,
        )
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
        """Load an exact Stage-1 checkpoint or initialize either architecture from pi0.5."""
        local_config = Path(pretrained_name_or_path) / "config.json"
        raw_config = json.loads(local_config.read_text()) if local_config.is_file() else {}
        saved_architecture = raw_config.get("architecture")
        if saved_architecture is None and "conditioning_route" in raw_config:
            # skillVLA_real checkpoints predate the explicit architecture field.
            saved_architecture = COND_GEMMA_ARCHITECTURE
        if raw_config.get("type") == "skill_expert":
            requested_architecture = (
                getattr(config, "architecture", None)
                if config is not None
                else saved_architecture
            )
            if saved_architecture != requested_architecture:
                raise ValueError(
                    "Stage-1 checkpoint architecture mismatch: "
                    f"checkpoint={saved_architecture!r}, "
                    f"requested={requested_architecture!r}."
                )
            if saved_architecture == COND_GEMMA_ARCHITECTURE:
                saved_revision = str(
                    raw_config.get(
                        "architecture_revision", COND_GEMMA_ARCHITECTURE_REVISION
                    )
                )
                requested_revision = str(
                    getattr(config, "architecture_revision", saved_revision)
                    if config is not None
                    else saved_revision
                )
                if saved_revision != requested_revision:
                    raise ValueError(
                        "cond_gemma checkpoint architecture_revision mismatch: "
                        f"checkpoint={saved_revision!r}, "
                        f"requested={requested_revision!r}."
                    )
                saved_route = normalize_conditioning_route(
                    raw_config.get("conditioning_route", "state_cond")
                )
                requested_route = normalize_conditioning_route(
                    getattr(config, "conditioning_route", saved_route)
                    if config is not None
                    else saved_route
                )
                if saved_route != requested_route:
                    raise ValueError(
                        "cond_gemma checkpoint conditioning_route mismatch: "
                        f"checkpoint={saved_route!r}, requested={requested_route!r}."
                    )
            else:
                eval_legacy_vsa = bool(
                    config is not None and getattr(config, "eval_legacy_vsa", False)
                )
                saved_revision = str(raw_config.get("architecture_revision", ""))
                requested_revision = str(
                    getattr(config, "architecture_revision", saved_revision)
                    if config is not None
                    else saved_revision
                )
                if not eval_legacy_vsa and not saved_revision:
                    raise ValueError(
                        "This checkpoint does not match the current VSA revision and "
                        "requires the historical evaluation contract: "
                        f"{pretrained_name_or_path}."
                    )
                if not eval_legacy_vsa and saved_revision != requested_revision:
                    raise ValueError(
                        "Stage-1 VSA checkpoint architecture_revision mismatch: "
                        f"checkpoint={saved_revision!r}, "
                        f"requested={requested_revision!r}."
                    )
                if not eval_legacy_vsa and config is not None:
                    saved_mode = str(
                        raw_config.get(
                            "vision_conditioning_mode", INTERLEAVED_CROSS_ATTENTION
                        )
                    )
                    requested_mode = str(
                        getattr(
                            config,
                            "vision_conditioning_mode",
                            INTERLEAVED_CROSS_ATTENTION,
                        )
                    )
                    if saved_mode != requested_mode:
                        raise ValueError(
                            "Stage-1 checkpoint vision_conditioning_mode mismatch: "
                            f"checkpoint={saved_mode!r}, requested={requested_mode!r}. "
                            "Cross-mode checkpoint conversion is not supported."
                        )
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
            if saved_architecture == COND_GEMMA_ARCHITECTURE:
                config.architecture = COND_GEMMA_ARCHITECTURE
                config.architecture_revision = str(
                    raw_config.get(
                        "architecture_revision", COND_GEMMA_ARCHITECTURE_REVISION
                    )
                )
        policy = cls(config, **kwargs)
        loaded = _load_pretrained_state_dict(
            pretrained_name_or_path,
            kwargs,
            architecture=config.architecture,
            vision_conditioning_mode=getattr(
                config,
                "vision_conditioning_mode",
                INTERLEAVED_CROSS_ATTENTION,
            ),
            include_predictor_vlm=config.uses_skill_predictor,
        )
        if loaded is None:
            raise FileNotFoundError(f"Could not load Stage-1 initialization: {pretrained_name_or_path}")

        state_dict, is_pi05 = loaded
        expert_prefix = (
            "model.gemma_expert."
            if config.architecture == COND_GEMMA_ARCHITECTURE
            else "model.expert."
        )
        if is_pi05 and not any(key.startswith(expert_prefix) for key in state_dict):
            raise RuntimeError("The pi0.5 checkpoint contains no compatible action-expert weights.")
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
        # Most Stage-1 tensors follow the configured model dtype. Optional
        # FP32 islands (currently the mode-latent projection/gain) instead use
        # their destination dtype so save/load does not silently requantize
        # them through BF16.
        target_state = policy.state_dict()
        state_dict = {
            key: value.to(
                target_state[key].dtype if key in target_state else policy._torch_dtype()
            )
            for key, value in state_dict.items()
        }
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(
                "Stage-1 initialization produced unexpected tensors: "
                f"{sorted(unexpected)}"
            )
        if not is_pi05 and missing:
            raise RuntimeError(
                "Stage-1 checkpoint mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        if is_pi05:
            disallowed_missing = sorted(
                key
                for key in missing
                if not _allowed_pi05_missing_key(key, config)
            )
            if disallowed_missing:
                raise RuntimeError(
                    "The pi0.5 warm-start is incomplete outside the explicit "
                    "Stage-1 new-parameter allowlist: "
                    f"missing={disallowed_missing}"
                )
        if is_pi05 and config.training_skill_source == "predictor":
            policy._initialize_frozen_skill_predictor(
                config.skill_predictor_checkpoint_path
            )
        source = "explicit pi0.5 initialization" if is_pi05 else "exact Stage-1 checkpoint"
        log.info(
            "Stage 1 <- %s: loaded=%d, fresh=%d, unexpected=%d.",
            source,
            len(state_dict),
            len(missing),
            len(unexpected),
        )
        return policy
