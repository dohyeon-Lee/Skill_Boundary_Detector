"""Arch1/Arch2 fixed DINO visual-bottleneck Action Experts."""

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
    """DINO + one proprio-conditioned visual bottleneck + an expert-only Gemma.

    Each camera's DINO memory is compressed independently once per action chunk.
    Half of the configured queries may read only top-view tokens and half may
    read only wrist-view tokens. The resulting narrow 256-D token set is exposed
    to all 18 expert layers through one shared cross-attention adapter. There is
    no condition Gemma and no direct DINO-to-expert route. Fixed learnable queries
    select visual values; proprioception modulates the resulting tokens with
    FiLM. Skill remains on the Action Expert's layerwise broadcast path, matching
    Arch0's separation.
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
        self.visual_state_film = nn.Linear(self.width, 2 * bottleneck_width)
        nn.init.zeros_(self.visual_state_film.weight)
        nn.init.zeros_(self.visual_state_film.bias)

        nn.init.normal_(self.visual_bottleneck_queries, std=0.02)
        nn.init.normal_(self.visual_camera_embedding, std=0.02)
        self.uses_expert_context_tokens = False
        # Stage 2 uses this capability flag to forward the shared state
        # projection into _run_joint_hidden. Arch1 consumes it as visual FiLM,
        # not as Cond-Gemma AdaRMS (there is no condition Gemma).
        self.uses_cond_state_adarms = True

    def _apply(self, fn, recurse: bool = True):
        """Apply dtype moves while keeping small adaptive paths in FP32.

        Stage 1 casts the complete policy to BF16 before constructing AdamW.
        At the 0.01 initialization used here, a 2.5e-5 optimizer step is below
        one BF16 quantization interval and was rounded away on every update.
        Keeping the 18-scalar bridge gate and the zero-initialized state-FiLM
        projection in FP32 lets these adaptive paths learn while the rest of
        the model remains in the configured working dtype.
        """
        super()._apply(fn, recurse=recurse)
        self.visual_state_film.to(dtype=torch.float32)
        self.visual_bridge_gates.data = self.visual_bridge_gates.data.float()
        if self.visual_bridge_gates.grad is not None:
            self.visual_bridge_gates.grad.data = (
                self.visual_bridge_gates.grad.data.float()
            )
        return self

    def _condition_tokens(
        self,
        images: list[Tensor],
        *,
        batch_size: int | None = None,
        skill_code: Tensor | None = None,
    ) -> Tensor:
        """Compress top and wrist independently with fixed visual queries."""
        if len(images) != 2:
            raise ValueError(f"Arch1 requires [top, wrist] images, got {len(images)}.")
        del skill_code
        inferred_batch = int(images[0].shape[0])
        if batch_size is not None and int(batch_size) != inferred_batch:
            raise ValueError(
                f"Arch1 image batch is {inferred_batch}, got batch_size={batch_size}."
            )

        memories = []
        for camera_index, image in enumerate(images):
            features = self._image_features(image).to(
                dtype=self.image_proj.weight.dtype
            )
            projected = self.image_proj(features).to(self.working_dtype)
            projected = projected + self.visual_camera_embedding[camera_index][None, None]
            memories.append(projected)

        queries = self.visual_bottleneck_queries[None].expand(
            inferred_batch, -1, -1
        )
        camera_count = len(memories)
        if queries.shape[1] % camera_count != 0:
            raise RuntimeError(
                "Arch1 visual_bottleneck_tokens must divide evenly across "
                f"{camera_count} cameras; got {queries.shape[1]}."
            )
        tokens_per_camera = queries.shape[1] // camera_count
        bottlenecks = []
        for camera_index, visual_memory in enumerate(memories):
            start = camera_index * tokens_per_camera
            end = start + tokens_per_camera
            # The shared attention weights keep the interface small, while
            # separate calls make camera coverage structural: top queries can
            # never collapse onto wrist patches and vice versa. No query
            # residual is added, so returned tokens still derive from vision.
            bottleneck, _ = self.visual_bottleneck_attention(
                queries[:, start:end],
                visual_memory,
                visual_memory,
                need_weights=False,
            )
            bottlenecks.append(bottleneck)
        return self.visual_bottleneck_norm(torch.cat(bottlenecks, dim=1))

    def _record_visual_debug(self, condition_tokens: Tensor) -> None:
        if not self._vsa_debug_active:
            return
        self._last_vsa_debug_stats.update(
            self._latent_debug_stats(condition_tokens, "fixed_bottleneck")
        )
        if condition_tokens.shape[1] % 2 == 0:
            top_tokens, wrist_tokens = condition_tokens.chunk(2, dim=1)
            self._last_vsa_debug_stats.update(
                self._latent_debug_stats(top_tokens, "fixed_bottleneck_top")
            )
            self._last_vsa_debug_stats.update(
                self._latent_debug_stats(wrist_tokens, "fixed_bottleneck_wrist")
            )
            top_centroid = torch.nn.functional.normalize(
                top_tokens.detach().float().mean(dim=1), dim=-1
            )
            wrist_centroid = torch.nn.functional.normalize(
                wrist_tokens.detach().float().mean(dim=1), dim=-1
            )
            self._last_vsa_debug_stats[
                "visual/fixed_bottleneck/top_wrist_centroid_cosine_abs_mean"
            ] = float(
                (top_centroid * wrist_centroid).sum(dim=-1).abs().mean().item()
            )
        raw_gates = self._active_visual_bridge_gates().detach().float()
        gates = raw_gates.tanh()
        initial = float(self.config.visual_bridge_gate_init)
        self._last_vsa_debug_stats.update(
            {
                "bridge_gate/value_abs_mean": float(gates.abs().mean().item()),
                "bridge_gate/layer_std": float(
                    gates.std(unbiased=False).item()
                ),
                "bridge_gate/update_from_init_rms": float(
                    (raw_gates - initial).square().mean().sqrt().item()
                ),
            }
        )

    def _layer_uses_visual_bridge(self, layer_index: int) -> bool:
        """Whether this Expert layer may read visual/proprio bottleneck tokens."""
        del layer_index
        return True

    def _active_visual_bridge_gates(self) -> Tensor:
        """Return only gates that participate in this architecture's forward."""
        return self.visual_bridge_gates

    def _project_state(self, state: Tensor | None) -> Tensor:
        if state is None:
            raise ValueError("Arch1 requires robot state conditioning.")
        projected = self.state_proj(
            state.to(dtype=next(self.state_proj.parameters()).dtype)
        )
        inverse_rms = projected.float().square().mean(
            dim=-1, keepdim=True
        ).add(1e-6).rsqrt()
        return (projected * inverse_rms.to(projected.dtype)).to(self.working_dtype)

    def _project_expert_state(
        self,
        state: Tensor | None,
        shared_projection: Tensor | None,
    ) -> None:
        """Arch1 keeps proprioception off the Action Expert AdaRMS path."""
        del state, shared_projection
        return None

    def _apply_visual_state_film(
        self,
        visual_tokens: Tensor,
        projected_state: Tensor | None,
    ) -> Tensor:
        """Modulate the four visual tokens with the shared proprio projection."""
        if projected_state is None:
            raise ValueError("Arch1 visual FiLM requires robot state conditioning.")
        film = self.visual_state_film(
            projected_state.to(self.visual_state_film.weight.dtype)
        )
        scale, shift = film.chunk(2, dim=-1)
        scale = scale[:, None].to(visual_tokens.dtype)
        shift = shift[:, None].to(visual_tokens.dtype)
        conditioned = visual_tokens * (1.0 + scale) + shift
        if self._vsa_debug_active:
            self._last_vsa_debug_stats.update(
                self._latent_debug_stats(
                    conditioned, "fixed_bottleneck_state_film"
                )
            )
            self._last_vsa_debug_stats.update(
                {
                    "visual/state_film/scale_rms": float(self._rms(scale).item()),
                    "visual/state_film/shift_rms": float(self._rms(shift).item()),
                }
            )
        return conditioned

    def _expert_condition(
        self,
        timestep: Tensor,
        projected_state: Tensor | None = None,
        skill_code: Tensor | None = None,
        mode_latent: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        del projected_state, skill_code, end_pose
        condition = self._time_condition(timestep)
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
        visual_tokens: Tensor | None,
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

        if self._layer_uses_visual_bridge(layer_index):
            if visual_tokens is None:
                raise RuntimeError(
                    f"Expert layer {layer_index} requires visual bottleneck tokens."
                )
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
        del condition_state_start_index
        if condition_skill is not None or expert_skill is None:
            raise RuntimeError("Arch1 requires expert-only skill broadcast.")
        condition_tokens = self._apply_visual_state_film(
            condition_tokens, condition_state
        )

        hidden = self._action_tokens(noisy_actions)
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
        focus_uv: Tensor | None = None,
    ) -> dict[str, float]:
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
                if name == "both_images_shuffle":
                    # Retain the original Arch1 aggregate metric name without
                    # paying for a duplicate perturbation forward.
                    stats["sensitivity/visual_bottleneck_shuffle/output_delta_rms"] = (
                        stats[f"sensitivity/{name}/output_delta_rms"]
                    )
                    stats[
                        "sensitivity/visual_bottleneck_shuffle/relative_output_delta"
                    ] = stats[f"sensitivity/{name}/relative_output_delta"]
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
        *,
        focus_uv: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        """Reuse fixed visual tokens while integrating the action flow."""
        del end_pose
        projected_state = self._project_condition_state(state, focus_uv, skill_code)
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
                projected_state,
                expert_condition,
                condition_skill,
                expert_skill,
            )
            velocity = self._action_velocity(hidden)
            x_t = x_t + dt * velocity
        return x_t


class LateVisualBottleneckSkillExpert(FixedVisualBottleneckSkillExpert):
    """Arch2: a pure skill-motion core followed by a shallow visual bridge.

    The visual/proprio bottleneck is architecturally invisible to the first
    ``depth - visual_bridge_last_n_layers`` Action-Expert layers. Main action
    flow continues through the terminal bridge layers. The training-only
    ``*_skill`` route exits at that boundary and uses the shared final norm and
    action head, directly supervising the reusable motion core.
    """

    @property
    def visual_bridge_start_layer(self) -> int:
        depth = int(self.gemma_expert.model.config.num_hidden_layers)
        return depth - int(self.config.visual_bridge_last_n_layers)

    def _layer_uses_visual_bridge(self, layer_index: int) -> bool:
        return int(layer_index) >= self.visual_bridge_start_layer

    def _active_visual_bridge_gates(self) -> Tensor:
        return self.visual_bridge_gates[self.visual_bridge_start_layer :]

    def _skill_only_expert_hidden(
        self,
        action_tokens: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        expert_condition: Tensor,
        expert_skill: Tensor,
    ) -> Tensor:
        """Run only Arch2's pure motion-core prefix for auxiliary flow."""
        hidden = action_tokens
        position_embeddings = self.gemma_expert.model.rotary_emb(
            hidden, position_ids
        )
        use_checkpoint = self._gradient_checkpointing and self.training
        for layer_index in range(self.visual_bridge_start_layer):
            if use_checkpoint:
                hidden = torch.utils.checkpoint.checkpoint(
                    self._expert_layer_with_visual_bridge,
                    layer_index,
                    hidden,
                    attention_mask,
                    position_ids,
                    expert_condition,
                    expert_skill,
                    None,
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
                    None,
                    position_embeddings,
                )
        hidden, _ = layernorm_forward(
            self.gemma_expert.model.norm, hidden, expert_condition
        )
        return hidden
