"""The retained Arch0 DINO/Cond-Gemma/Action-Expert implementation."""

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
    FIXED_VISUAL_BOTTLENECK_ARCHITECTURE,
    SkillExpertConfig,
)
from .modeling_skill_predictor import FrozenVLMSkillPredictor
from .modeling_utils import build_fsq_terminator, build_gemma


class CondGemmaSkillExpert(nn.Module):
    """DINO condition stream + fully trainable pi0.5 action expert."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__()
        self.config = config
        self.width = get_gemma_config(config.action_expert_variant).width
        # Stage 2 subclasses this module and still reads these two flags. They
        # are constants now, not architecture switches.
        self.uses_expert_context_tokens = False
        self.uses_cond_state_adarms = True

        self.dino = AutoModel.from_pretrained(config.dino_model_path)
        if config.freeze_vision_encoder:
            self.dino.requires_grad_(False)
            self.dino.eval()
        self.n_register_tokens = int(
            getattr(self.dino.config, "num_register_tokens", 0)
        )
        self.image_proj = nn.Linear(int(self.dino.config.hidden_size), self.width)
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
        self.skill_proj = nn.Linear(len(config.skill_fsq_levels), self.width)
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
        if config.skill_flow_latent_best_of_n_enabled:
            self.mode_latent_mlp = nn.Sequential(
                nn.Linear(config.skill_flow_latent_dim, self.width),
                nn.SiLU(),
                nn.Linear(self.width, self.width),
            )
            self.mode_latent_gain = nn.Parameter(
                torch.tensor(float(config.skill_flow_latent_gain_init))
            )
        else:
            # Keep disabled runs and historical checkpoints parameter-identical.
            self.mode_latent_mlp = None
            self.register_parameter("mode_latent_gain", None)

        # Arch1 deliberately removes the 18-layer condition Gemma. Its compact
        # DINO interface is implemented by FixedVisualBottleneckSkillExpert.
        self.cond_encoder = (
            None
            if config.architecture == FIXED_VISUAL_BOTTLENECK_ARCHITECTURE
            else build_gemma(config.cond_encoder_variant, use_adarms=True)
        )
        self.gemma_expert = build_gemma(config.action_expert_variant, use_adarms=True)
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
        self._last_flow_noise: Tensor | None = None
        # The current trainer calls this common observability surface.  Keeping
        # it empty adds no work to the original skillVLA_real forward path.
        self._last_vsa_debug_stats: dict[str, float] = {}
        self._vsa_training_step: int | None = None
        self._vsa_debug_active = False
        self._gradient_checkpointing = False

    @property
    def working_dtype(self) -> torch.dtype:
        # Compact input/output projections are deliberately kept in FP32 (see
        # ``_apply``), so their dtype is no longer the transformer compute
        # dtype.  The expert remains in the configured BF16/FP32 working dtype.
        expert = getattr(self, "gemma_expert", None)
        if expert is not None:
            try:
                return next(expert.parameters()).dtype
            except StopIteration:
                pass
        # Lightweight unit-test stubs may omit the full Gemma expert.
        return self.action_in_proj.weight.dtype

    def _apply(self, fn, recurse: bool = True):
        """Apply device/dtype moves while preserving small adaptive paths.

        Stage 1 casts the complete policy to BF16 before constructing AdamW.
        Keeping the compact projections themselves in BF16 loses ordinary
        optimizer-sized updates when their initialized weights are relatively
        large (most visibly ``skill_proj`` and ``state_proj``).  Retain FP32
        master parameters for these inexpensive paths while converting their
        activations back to the transformer's working dtype at the boundary.
        """
        super()._apply(fn, recurse=recurse)
        for name in (
            "image_proj",
            "state_proj",
            "skill_proj",
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ):
            module = getattr(self, name, None)
            if module is not None:
                module.to(dtype=torch.float32)
        if (
            self.config.skill_flow_latent_fp32
            and self.mode_latent_mlp is not None
            and self.mode_latent_gain is not None
        ):
            # Stage 1 normally casts the complete model to BF16 before the
            # optimizer is built. Restore this small path afterwards so both
            # its parameters and AdamW moments remain FP32. Preserve Parameter
            # identity in case a device-only move happens after optimizer setup.
            self.mode_latent_mlp.to(dtype=torch.float32)
            self.mode_latent_gain.data = self.mode_latent_gain.data.float()
            if self.mode_latent_gain.grad is not None:
                self.mode_latent_gain.grad.data = (
                    self.mode_latent_gain.grad.data.float()
                )
        return self

    def _action_tokens(self, actions: Tensor) -> Tensor:
        """Project actions in FP32, then enter the expert working dtype."""
        parameter = next(self.action_in_proj.parameters(), None)
        projection_dtype = actions.dtype if parameter is None else parameter.dtype
        projected = self.action_in_proj(
            actions.to(dtype=projection_dtype)
        )
        return projected.to(self.working_dtype)

    def _action_velocity(self, hidden: Tensor) -> Tensor:
        """Run the output head in FP32 and expose float flow velocity."""
        parameter = next(self.action_out_proj.parameters(), None)
        projection_dtype = hidden.dtype if parameter is None else parameter.dtype
        return self.action_out_proj(
            hidden.to(dtype=projection_dtype)
        ).float()

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
        if self.cond_encoder is not None and hasattr(
            self.cond_encoder, "gradient_checkpointing_enable"
        ):
            self.cond_encoder.gradient_checkpointing_enable()
        if hasattr(self.gemma_expert, "gradient_checkpointing_enable"):
            self.gemma_expert.gradient_checkpointing_enable()
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
        if self.skill_predictor is not None:
            # Predictor is an optional frozen input provider, never a Stage1 target.
            self.skill_predictor.eval()
        if self.fsq_term_train is not None:
            self.fsq_term_train.eval()
        return self

    def sample_noise(self, shape, device) -> Tensor:
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_mode_latent(self, batch_shape, device) -> Tensor:
        """Sample mode codes from the fixed square U[-1, 1]^2 prior."""
        shape = tuple(int(value) for value in batch_shape) + (
            int(self.config.skill_flow_latent_dim),
        )
        return torch.empty(shape, dtype=torch.float32, device=device).uniform_(-1.0, 1.0)

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
        self,
        images: list[Tensor],
        *,
        batch_size: int | None = None,
        skill_code: Tensor | None = None,
    ) -> Tensor:
        del batch_size, skill_code
        if len(images) != 2:
            raise ValueError(f"Arch0 requires [top, wrist] images, got {len(images)}.")
        tokens = [
            self.image_proj(
                self._image_features(image).to(dtype=self.image_proj.weight.dtype)
            ).to(self.working_dtype)
            for image in images
        ]
        return torch.cat(tokens, dim=1)

    def _code_to_zq(self, skill_code: Tensor) -> Tensor:
        # Stage-2 self-routed skill learning may pass normalized FSQ
        # coordinates directly.  Integer flat codes keep the historical path;
        # floating [B,D] coordinates preserve gradients through the frozen VSA.
        if skill_code.is_floating_point():
            expected = int(self._fsq_levels.numel())
            if skill_code.ndim != 2 or skill_code.shape[-1] != expected:
                raise ValueError(
                    "Continuous skill coordinates must have shape [B,D], got "
                    f"{tuple(skill_code.shape)} for D={expected}."
                )
            return skill_code.float()
        index = skill_code.reshape(-1, 1).long()
        level_ids = (
            torch.div(index, self._fsq_strides[None], rounding_mode="floor")
            % self._fsq_levels[None]
        )
        return (level_ids.float() - self._fsq_half[None]) / self._fsq_half[None]

    def _skill_embedding(self, skill_code: Tensor) -> Tensor:
        z_q = self._code_to_zq(skill_code).to(dtype=self.skill_proj.weight.dtype)
        return self.skill_proj(z_q).to(self.working_dtype)

    def _project_state(self, state: Tensor | None) -> Tensor | None:
        """Project state into the Cond-Gemma AdaRMS channel."""
        if state is None:
            raise ValueError("Arch0 requires robot state conditioning.")
        projected = self.state_proj(
            state.to(dtype=next(self.state_proj.parameters()).dtype)
        )
        return projected.to(self.working_dtype)

    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor | None:
        """Build the Cond AdaRMS input; spatially conditioned variants extend it."""
        del focus_uv, skill_code, end_pose
        return self._project_state(state)

    def _project_expert_state(
        self,
        state: Tensor | None,
        shared_projection: Tensor | None,
    ) -> Tensor | None:
        """Arch0 has no direct Expert-side state AdaRMS input."""
        del state, shared_projection
        return None

    def _state_condition(self, state: Tensor | None) -> Tensor | None:
        return self._project_state(state)

    def _condition_state_start_index(
        self, condition_tokens: Tensor
    ) -> int | None:
        del condition_tokens
        return None

    def _skill_broadcasts(
        self, skill_code: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None]:
        """Arch0 broadcasts skill only through the Action Expert."""
        if skill_code is None:
            raise ValueError("Arch0 requires skill conditioning.")
        return None, self._skill_embedding(skill_code)

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
        ).to(dtype=self.time_mlp_in.weight.dtype)
        condition = F.silu(self.time_mlp_in(condition))
        return F.silu(self.time_mlp_out(condition)).to(self.working_dtype)

    def _mode_latent_condition(self, mode_latent: Tensor | None) -> Tensor | None:
        """Project the 2D mode code into the Action Expert AdaRMS space."""
        if mode_latent is None:
            return None
        if not self.config.skill_flow_latent_best_of_n_enabled:
            raise ValueError("mode_latent was provided while latent Best-of-N is disabled.")
        if self.mode_latent_mlp is None or self.mode_latent_gain is None:
            raise RuntimeError("Latent Best-of-N modules are missing.")
        expected = int(self.config.skill_flow_latent_dim)
        if mode_latent.ndim != 2 or mode_latent.shape[1] != expected:
            raise ValueError(
                f"mode_latent must have shape [B,{expected}], got {tuple(mode_latent.shape)}."
            )
        latent_dtype = (
            torch.float32
            if self.config.skill_flow_latent_fp32
            else self.working_dtype
        )
        projected = self.mode_latent_mlp(mode_latent.to(latent_dtype))
        rms = projected.float().square().mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        projected = projected * rms.to(projected.dtype)
        projected = projected * self.mode_latent_gain.to(projected.dtype)
        return projected.to(self.working_dtype)

    def _expert_condition(
        self,
        timestep: Tensor,
        projected_state: Tensor | None = None,
        skill_code: Tensor | None = None,
        mode_latent: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        """Build Arch0 Expert AdaRMS input from flow time and optional mode z."""
        del projected_state, skill_code, end_pose
        condition = self._time_condition(timestep)
        mode_condition = self._mode_latent_condition(mode_latent)
        if mode_condition is not None:
            condition = condition + mode_condition.to(condition.dtype)
        return condition

    def _run_joint_hidden(
        self,
        condition_tokens: Tensor,
        noisy_actions: Tensor,
        condition_state: Tensor | None,
        expert_condition: Tensor,
        condition_skill: Tensor | None,
        expert_skill: Tensor | None,
        condition_state_start_index: int | None = None,
        *,
        return_all_layers: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return the final action hidden and, optionally, every layer's hidden.

        The optional stack is used only by the frozen Stage-2 FRS reader.  The
        normal Stage-1 path does not retain intermediate activations.
        """
        action_tokens = self._action_tokens(noisy_actions)
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
        layer_action_hidden: list[Tensor] = []
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
            if return_all_layers:
                normalized, _ = layernorm_forward(
                    self.gemma_expert.model.norm, streams[1], expert_condition
                )
                layer_action_hidden.append(normalized[:, -n_chunk:])

        action_hidden, _ = layernorm_forward(
            self.gemma_expert.model.norm, streams[1], expert_condition
        )
        action_hidden = action_hidden[:, -n_chunk:]
        if return_all_layers:
            return action_hidden, torch.stack(layer_action_hidden, dim=1)
        return action_hidden

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
        return self._action_velocity(action_hidden)

    def _predict_velocity_from_condition(
        self,
        condition_tokens: Tensor,
        noisy_actions: Tensor,
        state: Tensor | None,
        skill_code: Tensor | None,
        time: Tensor,
        mode_latent: Tensor | None = None,
        focus_uv: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        """Run the post-vision path so scheduled probes can reuse encoded images."""
        state_representation = self._project_condition_state(
            state, focus_uv, skill_code, end_pose
        )
        expert_condition = self._expert_condition(
            time,
            projected_state=state_representation,
            skill_code=skill_code,
            mode_latent=mode_latent,
            end_pose=end_pose,
        )
        condition_skill, expert_skill = self._skill_broadcasts(skill_code)
        skill_representation = expert_skill
        predicted_velocity = self._run_joint(
            condition_tokens,
            noisy_actions,
            state_representation,
            expert_condition,
            condition_skill,
            expert_skill,
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
        mode_latent: Tensor | None = None,
        focus_uv: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> dict[str, float]:
        """Perturb one modality at a time while keeping flow noise/time fixed."""
        if self.config.architecture_label.startswith(
            ("arch9_1", "arch9_2", "arch10_1", "arch10_2", "arch11_1", "arch11_2", "arch12_1", "arch12_2")
        ):
            if predicted_velocity.shape[0] < 2:
                return {}
            variants = {"wrist_image_shuffle": (condition_tokens.roll(1, 0), state, skill_code, end_pose)}
            if state is not None:
                variants["state_shuffle"] = (condition_tokens, state.roll(1, 0), skill_code, end_pose)
            if skill_code is not None:
                variants["skill_shuffle"] = (condition_tokens, state, skill_code.roll(1, 0), end_pose)
            if end_pose is not None:
                variants["end_pose_shuffle"] = (condition_tokens, state, skill_code, end_pose.roll(1, 0))
            baseline_rms = self._rms(predicted_velocity.detach().float()).clamp_min(1e-12)
            previous_debug = self._vsa_debug_active
            previous_checkpointing = self._gradient_checkpointing
            self._vsa_debug_active = False
            self._gradient_checkpointing = False
            try:
                stats = {}
                for name, (memory, perturbed_state, perturbed_skill, perturbed_pose) in variants.items():
                    perturbed = self._predict_velocity_from_condition(
                        memory, noisy_actions, perturbed_state, perturbed_skill,
                        time, mode_latent, end_pose=perturbed_pose,
                    ).float()
                    difference_rms = self._rms(perturbed - predicted_velocity.detach().float())
                    stats[f"sensitivity/{name}/output_delta_rms"] = float(difference_rms.item())
                    stats[f"sensitivity/{name}/relative_output_delta"] = float((difference_rms / baseline_rms).item())
                return stats
            finally:
                self._vsa_debug_active = previous_debug
                self._gradient_checkpointing = previous_checkpointing
        if predicted_velocity.shape[0] < 2 or condition_tokens.shape[1] % 2 != 0:
            return {}
        top, wrist = condition_tokens.chunk(2, dim=1)
        variants: dict[
            str,
            tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None],
        ] = {
            "top_image_shuffle": (
                torch.cat((top.roll(1, dims=0), wrist), dim=1),
                state,
                skill_code,
                focus_uv,
                end_pose,
            ),
            "wrist_image_shuffle": (
                torch.cat((top, wrist.roll(1, dims=0)), dim=1),
                state,
                skill_code,
                focus_uv,
                end_pose,
            ),
            "both_images_shuffle": (
                condition_tokens.roll(1, dims=0),
                state,
                skill_code,
                focus_uv,
                end_pose,
            ),
        }
        if state is not None:
            variants["state_shuffle"] = (
                condition_tokens,
                state.roll(1, dims=0),
                skill_code,
                focus_uv,
                end_pose,
            )
        if skill_code is not None:
            variants["skill_shuffle"] = (
                condition_tokens,
                state,
                skill_code.roll(1, dims=0),
                focus_uv,
                end_pose,
            )
        if focus_uv is not None:
            variants["focus_uv_shuffle"] = (
                condition_tokens,
                state,
                skill_code,
                focus_uv.roll(1, dims=0),
                end_pose,
            )
        if end_pose is not None:
            variants["end_pose_shuffle"] = (
                condition_tokens,
                state,
                skill_code,
                focus_uv,
                end_pose.roll(1, dims=0),
            )

        baseline = predicted_velocity.detach().float()
        baseline_rms = self._rms(baseline).clamp_min(1e-12)
        previous_debug = self._vsa_debug_active
        previous_checkpointing = self._gradient_checkpointing
        self._vsa_debug_active = False
        self._gradient_checkpointing = False
        try:
            stats = {}
            for name, (
                memory,
                perturbed_state,
                perturbed_skill,
                perturbed_uv,
                perturbed_pose,
            ) in variants.items():
                perturbed = self._predict_velocity_from_condition(
                    memory,
                    noisy_actions,
                    perturbed_state,
                    perturbed_skill,
                    time,
                    mode_latent,
                    perturbed_uv,
                    perturbed_pose,
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
        mode_latent: Tensor | None = None,
        focus_uv: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        """Return the signed flow residual; its square is the action-flow MSE."""
        self._last_vsa_debug_stats = {}
        batch_size = actions.shape[0]
        if self.config.skill_flow_latent_best_of_n_enabled and mode_latent is None:
            mode_latent = self.sample_mode_latent((batch_size,), actions.device)
        time = self.sample_time(batch_size, actions.device) if time is None else time
        self._last_flow_time = time.detach()
        source = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        source = source.to(actions.dtype)
        self._last_flow_noise = source.detach()
        x_t = time[:, None, None] * source + (1.0 - time[:, None, None]) * actions
        target_velocity = source - actions
        condition_tokens = self._condition_tokens(
            images, batch_size=batch_size, skill_code=skill_code
        )
        self._record_visual_debug(condition_tokens)
        predicted_velocity = self._predict_velocity_from_condition(
            condition_tokens, x_t, state, skill_code, time, mode_latent, focus_uv, end_pose
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
                mode_latent=mode_latent,
                focus_uv=focus_uv,
                end_pose=end_pose,
            )
            self._last_vsa_debug_stats = {**original_stats, **sensitivity}
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
        state: Tensor | None,
        skill_code: Tensor | None,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        mode_latent: Tensor | None = None,
        focus_uv: Tensor | None = None,
        end_pose: Tensor | None = None,
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
        if self.config.skill_flow_latent_best_of_n_enabled and mode_latent is None:
            mode_latent = self.sample_mode_latent((batch_size,), device)
        condition_tokens = self._condition_tokens(
            images, batch_size=batch_size, skill_code=skill_code
        )
        return self._sample_with_condition_cache(
            condition_tokens, noise, state, skill_code, num_steps, mode_latent,
            focus_uv=focus_uv,
            end_pose=end_pose,
        )

    @torch.no_grad()
    def sample_skill_only_actions(
        self,
        skill_code: Tensor,
        state: Tensor | None = None,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        horizon: int | None = None,
        mode_latent: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        """Sample the auxiliary skill-flow route without visual/Cond-Gemma input.

        ``end_pose`` is the skill-end EEF pose for architectures whose Expert AdaRMS is
        goal-conditioned; passing it reproduces the condition the route was trained with.

        This is the inference counterpart of :meth:`skill_only_flow_residual`.
        It uses the requested canonical trajectory length (within its padded
        training maximum) for a canonical ``*_skill`` mode and the complete configured
        extended-chunk length for ``*_skill_chunk``.
        """
        if not getattr(self.config, "skill_flow_enabled", False):
            raise RuntimeError(
                "Skill-only action sampling requires a skill-flow architecture."
            )
        if skill_code is None:
            raise ValueError("Skill-only action sampling requires skill_code.")
        num_steps = self.config.num_inference_steps if num_steps is None else num_steps
        if int(num_steps) <= 0:
            raise ValueError("num_steps must be positive.")
        batch_size, device = skill_code.shape[0], skill_code.device
        if self.config.skill_flow_latent_best_of_n_enabled and mode_latent is None:
            mode_latent = self.sample_mode_latent((batch_size,), device)
        configured_horizon = int(self.config.skill_flow_max_length)
        if horizon is None:
            horizon = configured_horizon
        horizon = int(horizon)
        if not 0 < horizon <= configured_horizon:
            raise ValueError(
                "Skill-only horizon must be within the trained padded horizon: "
                f"got {horizon}, configured maximum {configured_horizon}."
            )
        if (
            self.config.skill_flow_target == "extended_chunk"
            and horizon != configured_horizon
        ):
            raise ValueError(
                "Extended-chunk skill flow must use its complete trained horizon "
                f"({configured_horizon}), got {horizon}."
            )
        expected_shape = (batch_size, horizon, self.config.max_action_dim)
        if noise is None:
            noise = self.sample_noise(expected_shape, device)
        elif tuple(noise.shape) != expected_shape:
            raise ValueError(
                "Skill-only noise must match the configured auxiliary horizon: "
                f"expected {expected_shape}, got {tuple(noise.shape)}."
            )

        condition_skill, expert_skill = self._skill_broadcasts(skill_code)
        if condition_skill is not None or expert_skill is None:
            raise RuntimeError(
                "Skill-only sampling requires expert-only skill broadcast."
            )
        del state

        # Training treats every valid auxiliary trajectory token as one
        # bidirectional block. At inference the generated horizon has no pad
        # tokens, so reproduce that exact mask over the full horizon.
        valid = torch.ones(
            batch_size, horizon, dtype=torch.bool, device=device
        )
        block_starts = torch.zeros_like(valid)
        block_starts[:, 0] = True
        attention_mask = make_att_2d_masks(valid, block_starts)[:, None]
        attention_mask = torch.where(
            attention_mask, 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        position_ids = torch.arange(horizon, device=device)[None].expand(
            batch_size, -1
        )

        dt = -1.0 / int(num_steps)
        x_t = noise.float()
        for step in range(int(num_steps)):
            time = torch.full(
                (batch_size,),
                1.0 + step * dt,
                dtype=torch.float32,
                device=device,
            )
            action_tokens = self._action_tokens(x_t)
            expert_condition = self._expert_condition(
                time, mode_latent=mode_latent, end_pose=end_pose
            )
            hidden = self._skill_only_expert_hidden(
                action_tokens,
                attention_mask,
                position_ids,
                expert_condition,
                expert_skill,
            )
            velocity = self._action_velocity(hidden)
            x_t = x_t + dt * velocity
        return x_t

    def _sample_with_condition_cache(
        self,
        condition_tokens: Tensor,
        noise: Tensor,
        state: Tensor | None,
        skill_code: Tensor | None,
        num_steps: int,
        mode_latent: Tensor | None = None,
        *,
        focus_uv: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        """Encode the condition stream once, then Euler-integrate the action flow."""
        batch_size, n_condition = condition_tokens.shape[:2]
        n_chunk = noise.shape[1]
        device = noise.device
        projected_state = self._project_condition_state(
            state, focus_uv, skill_code, end_pose
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
            adarms_cond=projected_state,
            adarms_start_index=None,
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
                self._expert_condition(time, mode_latent=mode_latent, end_pose=end_pose),
                expert_skill,
                condition_cache,
                full_attention,
                action_positions,
            )
            velocity = self._action_velocity(action_hidden)
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
        action_tokens = self._action_tokens(noisy_actions)
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

    def _skill_only_flow_residual(
        self,
        actions: Tensor,
        skill_code: Tensor,
        action_is_pad: Tensor,
        *,
        time: Tensor,
        noise: Tensor | None = None,
        state: Tensor | None = None,
        mode_latent: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        """Training-only skill flow over a canonical or extended trajectory.

        This deliberately bypasses image encoding, Cond-Gemma, and its KV
        cache. Both retained skill-flow modes bypass robot state. Arch9--Arch12
        keep their skill-end EEF pose in the Expert AdaRMS condition so the auxiliary
        trajectory and deployed action stream use the same goal condition.
        The Action Expert, action projections, timestep path, layerwise skill
        broadcast, and output head are the exact same modules as the rollout
        route.
        """
        if actions.ndim != 3 or action_is_pad.shape != actions.shape[:2]:
            raise ValueError(
                "Canonical actions/mask must have shapes [B,T,D] and [B,T], got "
                f"{tuple(actions.shape)} and {tuple(action_is_pad.shape)}."
            )
        if time.shape != (actions.shape[0],):
            raise ValueError(
                f"Shared flow time must have shape {(actions.shape[0],)}, got {tuple(time.shape)}."
            )
        if self.config.skill_flow_latent_best_of_n_enabled and mode_latent is None:
            mode_latent = self.sample_mode_latent((actions.shape[0],), actions.device)
        valid = ~action_is_pad.to(device=actions.device, dtype=torch.bool)
        if bool((valid.sum(dim=1) == 0).any()):
            raise ValueError("Every canonical skill trajectory needs at least one valid step.")

        source = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        source = source.to(actions.dtype)
        x_t = time[:, None, None] * source + (1.0 - time[:, None, None]) * actions
        target_velocity = source - actions

        action_tokens = self._action_tokens(x_t)
        # All real trajectory tokens are one bidirectional block. Padding is
        # invisible both as key and query, and is also excluded from the loss.
        block_starts = torch.zeros_like(valid)
        block_starts[:, 0] = True
        attention_mask = make_att_2d_masks(valid, block_starts)[:, None]
        attention_mask = torch.where(
            attention_mask, 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        position_ids = (torch.cumsum(valid, dim=1) - 1).clamp_min(0)

        condition_skill, expert_skill = self._skill_broadcasts(skill_code)
        if condition_skill is not None or expert_skill is None:
            raise RuntimeError(
                "Skill-only flow requires expert-only layerwise skill broadcast."
            )
        del state
        expert_condition = self._expert_condition(
            time, mode_latent=mode_latent, end_pose=end_pose
        )
        hidden = self._skill_only_expert_hidden(
            action_tokens,
            attention_mask,
            position_ids,
            expert_condition,
            expert_skill,
        )
        predicted_velocity = self._action_velocity(hidden)
        return target_velocity - predicted_velocity

    def _skill_only_expert_hidden(
        self,
        action_tokens: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        expert_condition: Tensor,
        expert_skill: Tensor,
    ) -> Tensor:
        """Run the layers used by the training-only skill-motion route.

        Arch0, Arch1, and Arch3 retain the complete Action Expert here. Arch2
        and Arch4 override this narrow hook so their auxiliary predictions exit
        after the pure motion-core layers, before any visual/proprio bridge.
        The final Expert norm and shared action head remain common.
        """
        return self.gemma_expert.model.forward(
            inputs_embeds=action_tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            adarms_cond=expert_condition,
            broadcast_cond=expert_skill,
        ).last_hidden_state

    def skill_only_flow_residual(
        self,
        actions: Tensor,
        skill_code: Tensor,
        action_is_pad: Tensor,
        *,
        time: Tensor,
        noise: Tensor | None = None,
        state: Tensor | None = None,
        mode_latent: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        """Run the enabled Stage-1 skill-flow auxiliary route."""
        if not getattr(self.config, "skill_flow_enabled", False):
            raise RuntimeError("The canonical skill-flow path is disabled.")
        return self._skill_only_flow_residual(
            actions,
            skill_code,
            action_is_pad,
            time=time,
            noise=noise,
            state=state,
            mode_latent=mode_latent,
            end_pose=end_pose,
        )
