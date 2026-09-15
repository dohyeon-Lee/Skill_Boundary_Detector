"""Arch1: fixed DINO bottleneck shared by every Action Expert layer."""

from __future__ import annotations

import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from lerobot.policies.pi05.modeling_pi05 import (
    OPENPI_ATTENTION_MASK_VALUE,
    layernorm_forward,
    make_att_2d_masks,
)
from lerobot.policies.pi_gemma import _gated_residual, add_broadcast_condition

from .cond_gemma import CondGemmaSkillExpert
from .configuration_skill_expert import SkillExpertConfig


class FixedVisualBottleneckSkillExpert(CondGemmaSkillExpert):
    """DINO + one skill-aware visual bottleneck + an expert-only Gemma.

    DINO memory is compressed once per action chunk into four 256-D tokens.
    Those exact tokens are exposed to all 18 expert layers through one shared
    cross-attention adapter. There is no condition Gemma and no direct DINO to
    expert route. Skill controls visual selection through the bottleneck query,
    but the bottleneck output itself is an attention-weighted visual value.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        if self.cond_encoder is not None:
            raise RuntimeError("Arch1 must not allocate a condition Gemma.")

        expert_depth = int(self.gemma_expert.model.config.num_hidden_layers)
        bottleneck_width = int(config.visual_bottleneck_width)
        dino_width = int(self.image_proj.in_features)

        # Replace Arch0's direct DINO -> expert-width projection with the
        # intentionally narrow Arch1 interface.
        self.image_proj = nn.Linear(dino_width, bottleneck_width)
        self.visual_camera_embedding = nn.Parameter(
            torch.zeros(2, bottleneck_width)
        )
        self.visual_bottleneck_queries = nn.Parameter(
            torch.empty(config.visual_bottleneck_tokens, bottleneck_width)
        )
        self.visual_skill_query = nn.Linear(
            len(config.skill_fsq_levels), bottleneck_width, bias=False
        )
        self.visual_bottleneck_attention = nn.MultiheadAttention(
            embed_dim=bottleneck_width,
            num_heads=config.visual_bottleneck_heads,
            batch_first=True,
        )
        self.visual_bottleneck_norm = nn.LayerNorm(bottleneck_width)

        # One adapter is reused at every layer. Only the scalar residual gate
        # is layer-specific, so repeated access does not create 18 independent
        # vision-to-action parameter paths.
        self.visual_bridge_query_norm = nn.LayerNorm(self.width)
        self.visual_bridge_attention = nn.MultiheadAttention(
            embed_dim=self.width,
            num_heads=config.visual_bridge_heads,
            kdim=bottleneck_width,
            vdim=bottleneck_width,
            batch_first=True,
        )
        self.visual_bridge_gates = nn.Parameter(
            torch.full((expert_depth,), float(config.visual_bridge_gate_init))
        )

        # Proprioception no longer justifies a second 18-layer Gemma. It enters
        # the existing expert AdaRMS channel through this small projection.
        self.state_proj = nn.Sequential(
            nn.Linear(config.max_state_dim, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
        )

        nn.init.normal_(self.visual_bottleneck_queries, std=0.02)
        nn.init.normal_(self.visual_camera_embedding, std=0.02)
        self.uses_expert_context_tokens = False
        self.uses_cond_state_adarms = False

    def _condition_tokens(
        self,
        images: list[Tensor],
        *,
        batch_size: int | None = None,
        skill_code: Tensor | None = None,
    ) -> Tensor:
        """Compress both cameras once; skill changes selection, not values."""
        if len(images) != 2:
            raise ValueError(f"Arch1 requires [top, wrist] images, got {len(images)}.")
        if skill_code is None:
            raise ValueError("Arch1 visual bottleneck requires skill_code.")
        inferred_batch = int(images[0].shape[0])
        if batch_size is not None and int(batch_size) != inferred_batch:
            raise ValueError(
                f"Arch1 image batch is {inferred_batch}, got batch_size={batch_size}."
            )

        memories = []
        for camera_index, image in enumerate(images):
            features = self._image_features(image).to(self.working_dtype)
            projected = self.image_proj(features)
            projected = projected + self.visual_camera_embedding[camera_index][None, None]
            memories.append(projected)
        visual_memory = torch.cat(memories, dim=1)

        z_q = self._code_to_zq(skill_code).to(self.working_dtype)
        skill_query = self.visual_skill_query(z_q)[:, None]
        queries = self.visual_bottleneck_queries[None].expand(
            inferred_batch, -1, -1
        )
        queries = queries + skill_query
        # No query residual is added: the returned tokens must be derived from
        # DINO values, while skill may only alter the attention weights.
        bottleneck, _ = self.visual_bottleneck_attention(
            queries,
            visual_memory,
            visual_memory,
            need_weights=False,
        )
        return self.visual_bottleneck_norm(bottleneck)

    def _record_visual_debug(self, condition_tokens: Tensor) -> None:
        if not self._vsa_debug_active:
            return
        self._last_vsa_debug_stats.update(
            self._latent_debug_stats(condition_tokens, "fixed_bottleneck")
        )
        self._last_vsa_debug_stats["visual/bridge/gate_abs_mean"] = float(
            self.visual_bridge_gates.detach().float().tanh().abs().mean().item()
        )

    def _project_state(self, state: Tensor | None) -> Tensor:
        if state is None:
            raise ValueError("Arch1 requires robot state conditioning.")
        projected = self.state_proj(state.to(self.working_dtype))
        inverse_rms = projected.float().square().mean(
            dim=-1, keepdim=True
        ).add(1e-6).rsqrt()
        return projected * inverse_rms.to(projected.dtype)

    def _project_expert_state(
        self,
        state: Tensor | None,
        shared_projection: Tensor | None,
    ) -> Tensor:
        if shared_projection is not None:
            return shared_projection
        return self._project_state(state)

    def _expert_condition(
        self,
        timestep: Tensor,
        projected_state: Tensor | None = None,
        skill_code: Tensor | None = None,
        mode_latent: Tensor | None = None,
    ) -> Tensor:
        del skill_code
        condition = self._time_condition(timestep)
        if projected_state is not None:
            condition = condition + projected_state.to(condition.dtype)
        mode_condition = self._mode_latent_condition(mode_latent)
        if mode_condition is not None:
            condition = condition + mode_condition.to(condition.dtype)
        return condition

    def _expert_layer_with_visual_bridge(
        self,
        layer_index: int,
        hidden: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        expert_condition: Tensor,
        expert_skill: Tensor,
        visual_tokens: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        layer = self.gemma_expert.model.layers[layer_index]

        residual = hidden
        normalized, gate = layernorm_forward(
            layer.input_layernorm, hidden, expert_condition
        )
        normalized = add_broadcast_condition(normalized, expert_skill)
        attended, _ = layer.self_attn(
            normalized,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        hidden = _gated_residual(residual, attended, gate)

        bridge_query = self.visual_bridge_query_norm(hidden)
        bridge_output, _ = self.visual_bridge_attention(
            bridge_query,
            visual_tokens,
            visual_tokens,
            need_weights=False,
        )
        layer_gate = self.visual_bridge_gates[layer_index].tanh().to(hidden.dtype)
        hidden = hidden + layer_gate * bridge_output

        residual = hidden
        normalized, gate = layernorm_forward(
            layer.post_attention_layernorm, hidden, expert_condition
        )
        normalized = layer.mlp(normalized.to(layer.mlp.up_proj.weight.dtype))
        return _gated_residual(residual, normalized, gate)

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
        del condition_state, condition_state_start_index
        if condition_skill is not None or expert_skill is None:
            raise RuntimeError("Arch1 requires expert-only skill broadcast.")

        hidden = self.action_in_proj(noisy_actions.to(self.working_dtype))
        batch_size, horizon = hidden.shape[:2]
        valid = torch.ones(batch_size, horizon, dtype=torch.bool, device=hidden.device)
        block_starts = torch.zeros_like(valid)
        block_starts[:, 0] = True
        attention_mask = make_att_2d_masks(valid, block_starts)[:, None]
        attention_mask = torch.where(
            attention_mask, 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        position_ids = torch.arange(horizon, device=hidden.device)[None].expand(
            batch_size, -1
        )
        position_embeddings = self.gemma_expert.model.rotary_emb(
            hidden, position_ids
        )

        use_checkpoint = self._gradient_checkpointing and self.training
        layer_action_hidden: list[Tensor] = []
        for layer_index in range(
            int(self.gemma_expert.model.config.num_hidden_layers)
        ):
            if use_checkpoint:
                hidden = torch.utils.checkpoint.checkpoint(
                    self._expert_layer_with_visual_bridge,
                    layer_index,
                    hidden,
                    attention_mask,
                    position_ids,
                    expert_condition,
                    expert_skill,
                    condition_tokens,
                    position_embeddings,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden = self._expert_layer_with_visual_bridge(
                    layer_index,
                    hidden,
                    attention_mask,
                    position_ids,
                    expert_condition,
                    expert_skill,
                    condition_tokens,
                    position_embeddings,
                )
            if return_all_layers:
                normalized, _ = layernorm_forward(
                    self.gemma_expert.model.norm, hidden, expert_condition
                )
                layer_action_hidden.append(normalized)

        action_hidden, _ = layernorm_forward(
            self.gemma_expert.model.norm, hidden, expert_condition
        )
        if return_all_layers:
            return action_hidden, torch.stack(layer_action_hidden, dim=1)
        return action_hidden

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
    ) -> dict[str, float]:
        if predicted_velocity.shape[0] < 2:
            return {}
        variants: dict[str, tuple[Tensor, Tensor | None, Tensor | None]] = {
            "visual_bottleneck_shuffle": (
                condition_tokens.roll(1, dims=0),
                state,
                skill_code,
            )
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
                    mode_latent,
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

    def _sample_with_condition_cache(
        self,
        condition_tokens: Tensor,
        noise: Tensor,
        state: Tensor | None,
        skill_code: Tensor | None,
        num_steps: int,
        mode_latent: Tensor | None = None,
    ) -> Tensor:
        """Reuse fixed visual tokens while integrating the action flow."""
        projected_state = self._project_state(state)
        condition_skill, expert_skill = self._skill_broadcasts(skill_code)
        batch_size = int(noise.shape[0])
        dt = -1.0 / int(num_steps)
        x_t = noise.float()
        for step in range(int(num_steps)):
            time = torch.full(
                (batch_size,),
                1.0 + step * dt,
                dtype=torch.float32,
                device=noise.device,
            )
            expert_condition = self._expert_condition(
                time,
                projected_state=projected_state,
                skill_code=skill_code,
                mode_latent=mode_latent,
            )
            hidden = self._run_joint_hidden(
                condition_tokens,
                x_t,
                None,
                expert_condition,
                condition_skill,
                expert_skill,
            )
            velocity = self.action_out_proj(hidden.to(self.working_dtype)).float()
            x_t = x_t + dt * velocity
        return x_t
