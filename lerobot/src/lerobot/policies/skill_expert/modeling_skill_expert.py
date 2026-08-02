"""Stage-1 DINO-Perceiver VSA prior with selectable vision conditioning."""

from __future__ import annotations

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
from lerobot.policies.pi05.lora import route_plain_to_base
from lerobot.policies.pi05.modeling_pi05 import (
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    pad_vector,
    sample_beta,
)
from lerobot.policies.pi_gemma import PiGemmaRMSNorm
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from .configuration_skill_expert import (
    GLOBAL_VISUAL_ADARMS,
    IN_CONTEXT_TOKENS,
    RESIDUAL_CROSS_ATTENTION,
    VSA_ARCHITECTURE_REVISION,
    SkillExpertConfig,
)
from .modeling_utils import build_fsq_terminator, load_raw_state_dict
from .modeling_skill_predictor import FrozenVLMSkillPredictor
from .vsa_perceiver_crossattn import (
    VISUAL_RESIDUAL_GATE_INIT,
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
        self.top_resampler = CameraPerceiverResampler(
            dino_width,
            self.width,
            num_latents=config.num_visual_latents_per_camera,
        )
        self.wrist_resampler = CameraPerceiverResampler(
            dino_width,
            self.width,
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
        if getattr(config, "eval_legacy_vsa", False):
            # Explicit evaluation-only compatibility. Training configs never set
            # this runtime flag, so the Stage-1 architecture remains single-path.
            from .legacy_vsa_eval import LegacyVSAActionExpert  # noqa: PLC0415

            expert_class = LegacyVSAActionExpert
        expert_kwargs = {
            "include_state_in_visual_crossattn": config.include_state_in_visual_crossattn,
            "include_skill_in_visual_crossattn": config.include_skill_in_visual_crossattn,
        }
        if not getattr(config, "eval_legacy_vsa", False):
            expert_kwargs["vision_conditioning_mode"] = config.vision_conditioning_mode
        self.expert = expert_class(**expert_kwargs)
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
            f"visual/{name}/pair_cosine_mean": float(pairwise.mean().item()),
            f"visual/{name}/pair_cosine_abs_mean": float(pairwise.abs().mean().item()),
            f"visual/{name}/pair_cosine_max": float(pairwise.max().item()),
            f"visual/{name}/effective_rank": float(effective_rank.mean().item()),
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
                "visual/top_dino/patch_spread_rms": float(top_dino_spread.item()),
                "visual/wrist_dino/patch_spread_rms": float(wrist_dino_spread.item()),
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
        top_normalized = F.normalize(top_memory.detach().float(), dim=-1, eps=1e-12)
        wrist_normalized = F.normalize(wrist_memory.detach().float(), dim=-1, eps=1e-12)
        cross_camera = top_normalized @ wrist_normalized.transpose(-1, -2)
        top_centroid = F.normalize(top_memory.detach().float().mean(dim=1), dim=-1)
        wrist_centroid = F.normalize(wrist_memory.detach().float().mean(dim=1), dim=-1)
        stats.update(
            {
                "visual/cross_camera/pair_cosine_mean": float(cross_camera.mean().item()),
                "visual/cross_camera/pair_cosine_abs_mean": float(
                    cross_camera.abs().mean().item()
                ),
                "visual/cross_camera/pair_cosine_max": float(cross_camera.amax().item()),
                "visual/cross_camera/centroid_cosine": float(
                    (top_centroid * wrist_centroid).sum(dim=-1).mean().item()
                ),
            }
        )
        return stats

    def gradient_checkpointing_enable(self) -> None:
        self._gradient_checkpointing = True
        self.expert.gradient_checkpointing_enable()
        if self.skill_predictor is not None and self.config.train_skill_predictor:
            self.skill_predictor.gradient_checkpointing_enable()
        if hasattr(self.dino, "gradient_checkpointing_enable"):
            self.dino.gradient_checkpointing_enable()

    def train(self, mode: bool = True):
        super().train(mode)
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
        return time_condition + self.visual_condition_projection(pooled)

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
                    f"activation/{name}_{stat}": float(value.detach().float().item())
                    for name, tensor in tensors.items()
                    for stat, value in (
                        ("mean", tensor.detach().float().mean()),
                        ("std", tensor.detach().float().std()),
                        ("rms", tensor.detach().float().square().mean().sqrt()),
                    )
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


def _map_pi05_key(key: str, *, include_predictor_vlm: bool = False) -> str | None:
    """Apply the explicit pi0.5 -> new VSA initialization contract."""
    layer_match = re.fullmatch(
        r"paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)\."
        r"(self_attn|mlp|input_layernorm|post_attention_layernorm)\.(.+)",
        key,
    )
    if layer_match:
        layer_index = int(layer_match.group(1))
        component = layer_match.group(2)
        suffix = layer_match.group(3)
        if component == "self_attn":
            return f"model.expert.blocks.{layer_index}.self_attention.{suffix}"
        if component == "mlp":
            return f"model.expert.blocks.{layer_index}.mlp.{suffix}"
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


def _allowed_pi05_missing_key(key: str, config: SkillExpertConfig) -> bool:
    """Return whether a target tensor is intentionally new relative to pi0.5."""
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
    if config.vision_conditioning_mode == RESIDUAL_CROSS_ATTENTION and re.fullmatch(
        r"model\.expert\.blocks\.\d+\."
        r"(visual_attention_norm|visual_cross_attention|visual_residual_gate)(\..+)?",
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
        log.info("Vision conditioning mode: %s", config.vision_conditioning_mode)
        if config.vision_conditioning_mode == RESIDUAL_CROSS_ATTENTION:
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
        if getattr(config, "eval_legacy_vsa", False):
            log.info("VSA checkpoint layout: legacy alternating (evaluation-only)")
        elif config.vision_conditioning_mode == RESIDUAL_CROSS_ATTENTION:
            log.info("Visual residual gate initialization: %.3f", VISUAL_RESIDUAL_GATE_INIT)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(device=config.device, dtype=self._torch_dtype())
        if self.model.fsq_term_train is not None:
            # Match FSQ training/inference numerics; this auxiliary remains fp32.
            self.model.fsq_term_train.to(dtype=torch.float32)
        counts = self.parameter_counts()
        log.info(
            "Stage-1 parameters: total=%.1fM trainable=%.1fM dino=%.1fM "
            "perceivers=%.1fM expert=%.1fM auxiliaries=%.1fM",
            counts["total"] / 1e6,
            counts["trainable"] / 1e6,
            counts["dino"] / 1e6,
            counts["perceivers"] / 1e6,
            counts["expert"] / 1e6,
            counts["auxiliaries"] / 1e6,
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

        auxiliaries = count(self.model.skill_predictor) + count(self.model.fsq_term_train)
        return {
            "total": count(self),
            "trainable": count(self, trainable=True),
            "dino": count(self.model.dino),
            "perceivers": count(self.model.top_resampler) + count(self.model.wrist_resampler),
            "expert": count(self.model.expert),
            "auxiliaries": auxiliaries,
        }

    def training_debug_metrics(self) -> dict[str, float]:
        if not self.model._vsa_debug_active:
            return {}

        def grad_stats(module: nn.Module) -> tuple[float, float]:
            gradients = [
                parameter.grad.detach().float()
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            if not gradients:
                return 0.0, 0.0
            squared_sum = torch.stack(
                [gradient.square().sum() for gradient in gradients]
            ).sum()
            count = sum(gradient.numel() for gradient in gradients)
            return (
                float(squared_sum.sqrt().item()),
                float((squared_sum / max(count, 1)).sqrt().item()),
            )

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
                block.visual_residual_gate
                for block in self.model.expert.blocks
                if block.visual_residual_gate is not None
            ]
        )

        modules = {
            "dino": self.model.dino,
            "top_resampler": self.model.top_resampler,
            "wrist_resampler": self.model.wrist_resampler,
            "state_path": nn.ModuleList(
                [self.model.state_proj, self.model.state_norm]
            ),
            "skill_path": nn.ModuleList(
                [self.model.skill_proj, self.model.skill_norm]
            ),
            "expert_total": self.model.expert,
            "expert_cross_attention": cross_attention,
            "expert_visual_residual_gates": visual_residual_gates,
            "expert_self_attention": self_attention,
            "expert_mlp": expert_mlps,
            "action_io": nn.ModuleList(
                [self.model.action_in_proj, self.model.action_out_proj]
            ),
            "time_mlp": nn.ModuleList(
                [self.model.time_mlp_in, self.model.time_mlp_out]
            ),
        }
        if self.model.visual_condition_projection is not None:
            modules["visual_condition_projection"] = (
                self.model.visual_condition_projection
            )
        metrics = {}
        for name, module in modules.items():
            norm, rms = grad_stats(module)
            metrics[f"vsa_debug/gradient/preclip/{name}_norm"] = norm
            metrics[f"vsa_debug/gradient/preclip/{name}_rms"] = rms
        # The scheduled training update is complete. Avoid carrying debug mode
        # into checkpoint-time evaluation or any auxiliary forward.
        self.model._vsa_debug_active = False
        self.model.expert.debug_enabled = False
        return metrics

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
                "Stage-1 predictor module contract mismatch: " + "; ".join(mismatches)
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
        """Main VSA, mandatory 0.1x DINO, and optional terminator groups."""
        terminator = getattr(self.model, "fsq_term_train", None)
        terminator_parameters = (
            [parameter for parameter in terminator.parameters() if parameter.requires_grad]
            if terminator is not None
            else []
        )
        excluded_ids = {id(parameter) for parameter in terminator_parameters}

        dino_parameters = [
            parameter for parameter in self.model.dino.parameters() if parameter.requires_grad
        ]
        excluded_ids.update(id(parameter) for parameter in dino_parameters)

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
        if terminator_parameters:
            groups.append(
                {
                    "params": terminator_parameters,
                    "lr": self.config.optimizer_lr
                    * self.config.terminator_lr_scale,
                    "lr_scale": self.config.terminator_lr_scale,
                    "group_name": "terminator",
                }
            )
        for group in groups:
            log.info(
                "Stage-1 optimizer group %s: %.1fM params, lr_scale=%g, lr=%g",
                group.get("group_name", "unnamed"),
                sum(parameter.numel() for parameter in group["params"]) / 1e6,
                float(group.get("lr_scale", 1.0)),
                float(group.get("lr", self.config.optimizer_lr)),
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
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        skill_code = self._training_skill_code(batch)
        images = self._collect_images(batch)
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
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        skill_code = self._skill_code(batch)
        images = self._collect_images(batch)
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
        """Load an exact new-Stage1 checkpoint or explicitly initialize from pi0.5."""
        local_config = Path(pretrained_name_or_path) / "config.json"
        raw_config = json.loads(local_config.read_text()) if local_config.is_file() else {}
        if raw_config.get("type") == "skill_expert":
            if raw_config.get("architecture") != "vsa_perceiver_crossattn":
                raise ValueError(
                    "This branch cannot load legacy Stage-1 checkpoints. Expected "
                    "architecture='vsa_perceiver_crossattn'; use the original branch "
                    f"for {pretrained_name_or_path}."
                )
            eval_legacy_vsa = bool(
                config is not None and getattr(config, "eval_legacy_vsa", False)
            )
            if (
                raw_config.get("architecture_revision") != VSA_ARCHITECTURE_REVISION
                and not eval_legacy_vsa
            ):
                raise ValueError(
                    "This checkpoint predates the residual-SA18 VSA revision and "
                    "cannot be resumed by the new architecture: "
                    f"{pretrained_name_or_path}."
                )
            if not eval_legacy_vsa and config is not None:
                saved_mode = str(
                    raw_config.get(
                        "vision_conditioning_mode", RESIDUAL_CROSS_ATTENTION
                    )
                )
                requested_mode = str(
                    getattr(
                        config,
                        "vision_conditioning_mode",
                        RESIDUAL_CROSS_ATTENTION,
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
        policy = cls(config, **kwargs)
        loaded = _load_pretrained_state_dict(
            pretrained_name_or_path,
            kwargs,
            include_predictor_vlm=config.uses_skill_predictor,
        )
        if loaded is None:
            raise FileNotFoundError(f"Could not load Stage-1 initialization: {pretrained_name_or_path}")

        state_dict, is_pi05 = loaded
        if is_pi05 and not any(key.startswith("model.expert.") for key in state_dict):
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
        state_dict = {
            key: value.to(policy._torch_dtype()) for key, value in state_dict.items()
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
