"""Arch3 layerwise Cond-Gemma bottleneck Action Expert.

Arch3 restores the Arch0 condition Gemma, but removes its wide joint-attention
connection to the Action Expert.  A small set of latent tokens reads each
successive condition layer and is updated recurrently.  Only the configured
terminal Expert layers may read the matching updated latent state.
"""

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
from .wrist_patch_alignment import (
    WristPatchAlignmentHead,
    decompose_wrist_patch_alignment,
)
from .configuration_skill_expert import (
    LAYERWISE_COND_BOTTLENECK_UV_COND_XYZ_TERMINATION_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_SKILL_END_POSE_TERMINATION_REVISION,
    LIT_COND_END_GOAL_ARCH_LABELS,
    SkillExpertConfig,
)


class _LayerwiseConditionReader(nn.Module):
    """One residual latent update from a single Cond-Gemma layer."""

    def __init__(self, latent_width: int, condition_width: int, heads: int):
        super().__init__()
        self.capture_cross_attention = False
        self.last_cross_attention: Tensor | None = None
        self.cross_norm = nn.LayerNorm(latent_width)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_width,
            num_heads=heads,
            kdim=condition_width,
            vdim=condition_width,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(latent_width)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=latent_width,
            num_heads=heads,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(latent_width)
        self.mlp = nn.Sequential(
            nn.Linear(latent_width, 4 * latent_width),
            nn.SiLU(),
            nn.Linear(4 * latent_width, latent_width),
        )

    def forward(self, latent: Tensor, condition_hidden: Tensor) -> Tensor:
        query = self.cross_norm(latent)
        attention_options = (
            {"need_weights": True, "average_attn_weights": False}
            if self.capture_cross_attention
            else {"need_weights": False}
        )
        update, attention = self.cross_attention(
            query, condition_hidden, condition_hidden, **attention_options
        )
        self.last_cross_attention = attention if self.capture_cross_attention else None
        latent = latent + update
        normalized = self.self_norm(latent)
        update, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        latent = latent + update
        return latent + self.mlp(self.mlp_norm(latent))


class LayerwiseCondBottleneckSkillExpert(CondGemmaSkillExpert):
    """Cond-Gemma -> recurrent narrow tokens -> terminal Expert bridges.

    The condition and Action Expert depths are matched.  After condition layer
    ``i`` the recurrent bottleneck becomes ``Z_i``.  Expert layer ``i`` reads
    exactly ``Z_i`` when it belongs to the configured terminal bridge range.
    Thus ``last_n=1`` exposes only ``Z_18`` to Expert layer 18, while
    ``last_n=18`` establishes the full one-to-one ``Z_i`` mapping.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        if self.cond_encoder is None:
            raise RuntimeError("Arch3 requires a condition Gemma.")

        condition_depth = int(self.cond_encoder.model.config.num_hidden_layers)
        expert_depth = int(self.gemma_expert.model.config.num_hidden_layers)
        if condition_depth != expert_depth:
            raise RuntimeError(
                "Arch3 requires matching Cond/Expert depths, got "
                f"{condition_depth} and {expert_depth}."
            )
        latent_width = int(config.visual_bottleneck_width)
        self.layerwise_latent_queries = nn.Parameter(
            torch.empty(config.visual_bottleneck_tokens, latent_width)
        )
        self.layerwise_condition_memory_norm = nn.LayerNorm(self.width)
        self.layerwise_condition_readers = nn.ModuleList(
            [
                _LayerwiseConditionReader(
                    latent_width,
                    self.width,
                    int(config.visual_bottleneck_heads),
                )
                for _ in range(condition_depth)
            ]
        )

        # The bridge projection is shared across participating Expert layers;
        # only its small residual gain is layer-specific.  Consequently a
        # larger connection depth grants repeated access without creating a
        # separate wide Cond-to-Expert parameter path at every layer.
        self.visual_bridge_query_norm = nn.LayerNorm(self.width)
        self.visual_bridge_attention = nn.MultiheadAttention(
            embed_dim=self.width,
            num_heads=int(config.visual_bridge_heads),
            kdim=latent_width,
            vdim=latent_width,
            batch_first=True,
        )
        self.visual_bridge_gates = nn.Parameter(
            torch.full((expert_depth,), float(config.visual_bridge_gate_init))
        )
        self._capture_visual_bridge_attention = False
        self._captured_visual_bridge_attention: dict[int, Tensor] = {}
        nn.init.normal_(self.layerwise_latent_queries, std=0.02)

    def _apply(self, fn, recurse: bool = True):
        """Keep small update-sensitive parameters in FP32 under BF16 training."""
        super()._apply(fn, recurse=recurse)
        # These modules are the entire narrow Cond-to-Expert interface.  Their
        # optimizer-sized updates are vulnerable to BF16 rounding in exactly
        # the same way as the compact state/skill paths, while keeping them in
        # FP32 is inexpensive relative to either Gemma.
        self.layerwise_condition_memory_norm.to(dtype=torch.float32)
        self.layerwise_condition_readers.to(dtype=torch.float32)
        self.visual_bridge_query_norm.to(dtype=torch.float32)
        self.visual_bridge_attention.to(dtype=torch.float32)
        self.layerwise_latent_queries.data = self.layerwise_latent_queries.data.float()
        if self.layerwise_latent_queries.grad is not None:
            self.layerwise_latent_queries.grad.data = (
                self.layerwise_latent_queries.grad.data.float()
            )
        self.visual_bridge_gates.data = self.visual_bridge_gates.data.float()
        if self.visual_bridge_gates.grad is not None:
            self.visual_bridge_gates.grad.data = self.visual_bridge_gates.grad.data.float()
        return self

    @property
    def visual_bridge_start_layer(self) -> int:
        depth = int(self.gemma_expert.model.config.num_hidden_layers)
        return depth - int(self.config.visual_bridge_last_n_layers)

    def _layer_uses_visual_bridge(self, layer_index: int) -> bool:
        return int(layer_index) >= self.visual_bridge_start_layer

    def _active_visual_bridge_gates(self) -> Tensor:
        return self.visual_bridge_gates[self.visual_bridge_start_layer :]

    @staticmethod
    def _sequence_geometry(hidden: Tensor, model: nn.Module):
        batch_size, length = hidden.shape[:2]
        valid = torch.ones(
            batch_size, length, dtype=torch.bool, device=hidden.device
        )
        block_starts = torch.zeros_like(valid)
        attention_mask = make_att_2d_masks(valid, block_starts)[:, None]
        attention_mask = torch.where(
            attention_mask, 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        position_ids = torch.arange(length, device=hidden.device)[None].expand(
            batch_size, -1
        )
        position_embeddings = model.rotary_emb(hidden, position_ids)
        return attention_mask, position_ids, position_embeddings

    def _condition_layer_with_latent(
        self,
        layer_index: int,
        condition_hidden: Tensor,
        latent: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        condition_state: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        layer = self.cond_encoder.model.layers[layer_index]
        residual = condition_hidden
        normalized, gate = layernorm_forward(
            layer.input_layernorm, condition_hidden, condition_state
        )
        attended, _ = layer.self_attn(
            normalized,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        condition_hidden = _gated_residual(residual, attended, gate)
        residual = condition_hidden
        normalized, gate = layernorm_forward(
            layer.post_attention_layernorm, condition_hidden, condition_state
        )
        normalized = layer.mlp(normalized.to(layer.mlp.up_proj.weight.dtype))
        condition_hidden = _gated_residual(residual, normalized, gate)

        memory = self.layerwise_condition_memory_norm(condition_hidden.float())
        latent = self._update_layerwise_latent(layer_index, latent, memory)
        return condition_hidden, latent

    def _update_layerwise_latent(
        self, layer_index: int, latent: Tensor, memory: Tensor
    ) -> Tensor:
        """Update the recurrent interface; specialised modes may partition queries."""
        return self.layerwise_condition_readers[layer_index](latent, memory)

    def _action_bridge_latent(self, layer_latent: Tensor) -> Tensor:
        """Return the bottleneck tokens consumed by the ordinary action bridge."""
        return layer_latent

    def _aligned_bridge_residual(
        self,
        layer_index: int,
        bridge_query: Tensor,
        layer_latent: Tensor,
    ) -> Tensor | None:
        """Optional separately gated residual from dedicated alignment tokens."""
        del layer_index, bridge_query, layer_latent
        return None

    def _encode_layerwise_latents(
        self,
        condition_tokens: Tensor,
        condition_state: Tensor | None,
    ) -> list[Tensor]:
        if condition_state is None:
            raise ValueError("Arch3 requires state for Cond-Gemma AdaRMS.")
        condition_hidden = condition_tokens
        attention_mask, position_ids, position_embeddings = self._sequence_geometry(
            condition_hidden, self.cond_encoder.model
        )
        latent = self.layerwise_latent_queries.to(
            device=condition_hidden.device
        )[None].expand(condition_hidden.shape[0], -1, -1)
        initial_latent = latent
        layer_latents: list[Tensor] = []
        update_rms: list[Tensor] = []
        use_checkpoint = self._gradient_checkpointing and self.training
        for layer_index in range(
            int(self.cond_encoder.model.config.num_hidden_layers)
        ):
            previous = latent
            if use_checkpoint:
                condition_hidden, latent = torch.utils.checkpoint.checkpoint(
                    self._condition_layer_with_latent,
                    layer_index,
                    condition_hidden,
                    latent,
                    attention_mask,
                    position_ids,
                    condition_state,
                    position_embeddings,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                condition_hidden, latent = self._condition_layer_with_latent(
                    layer_index,
                    condition_hidden,
                    latent,
                    attention_mask,
                    position_ids,
                    condition_state,
                    position_embeddings,
                )
            layer_latents.append(latent)
            if self._vsa_debug_active:
                update_rms.append(self._rms(latent - previous))

        if self._vsa_debug_active:
            self._last_vsa_debug_stats.update(
                self._latent_debug_stats(latent, "layerwise_bottleneck_final")
            )
            self._last_vsa_debug_stats.update(
                {
                    "visual/layerwise_bottleneck/initial_to_final_delta_rms": float(
                        self._rms(latent - initial_latent).item()
                    ),
                    "visual/layerwise_bottleneck/layer_update_rms_mean": float(
                        torch.stack(update_rms).mean().item()
                    ),
                }
            )
        self._on_final_condition_hidden(condition_hidden)
        self._on_final_bottleneck_latent(latent)
        return layer_latents

    def _on_final_condition_hidden(self, hidden: Tensor) -> None:
        """Optional auxiliary readout; the deployed action path ignores it."""
        del hidden

    def _on_final_bottleneck_latent(self, latent: Tensor) -> None:
        """Optional auxiliary readout; the deployed action path ignores it."""
        del latent

    def _expert_layer_with_latent_bridge(
        self,
        layer_index: int,
        hidden: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        expert_condition: Tensor,
        expert_skill: Tensor,
        layer_latent: Tensor | None,
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
            if layer_latent is None:
                raise RuntimeError(
                    f"Expert layer {layer_index} requires its Arch3 latent state."
                )
            attention_options = (
                {"need_weights": True, "average_attn_weights": False}
                if self._capture_visual_bridge_attention
                else {"need_weights": False}
            )
            bridge_query = self.visual_bridge_query_norm(hidden.float())
            action_latent = self._action_bridge_latent(layer_latent)
            bridge_output, bridge_attention = self.visual_bridge_attention(
                bridge_query,
                action_latent,
                action_latent,
                **attention_options,
            )
            if self._capture_visual_bridge_attention:
                if bridge_attention is None:
                    raise RuntimeError("Visual bridge attention capture returned no weights.")
                self._captured_visual_bridge_attention[layer_index] = bridge_attention
            gate_value = self.visual_bridge_gates[layer_index].tanh()
            bridge_residual = gate_value * bridge_output
            aligned_residual = self._aligned_bridge_residual(
                layer_index, bridge_query, layer_latent
            )
            if aligned_residual is not None:
                bridge_residual = bridge_residual + aligned_residual
            hidden = (hidden.float() + bridge_residual).to(self.working_dtype)

        residual = hidden
        normalized, gate = layernorm_forward(
            layer.post_attention_layernorm, hidden, expert_condition
        )
        normalized = layer.mlp(normalized.to(layer.mlp.up_proj.weight.dtype))
        return _gated_residual(residual, normalized, gate)

    def _run_expert_with_layerwise_latents(
        self,
        noisy_actions: Tensor,
        expert_condition: Tensor,
        expert_skill: Tensor,
        layer_latents: list[Tensor],
        *,
        return_all_layers: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        hidden = self._action_tokens(noisy_actions)
        attention_mask, position_ids, position_embeddings = self._sequence_geometry(
            hidden, self.gemma_expert.model
        )
        use_checkpoint = self._gradient_checkpointing and self.training
        layer_action_hidden: list[Tensor] = []
        for layer_index in range(
            int(self.gemma_expert.model.config.num_hidden_layers)
        ):
            layer_latent = (
                layer_latents[layer_index]
                if self._layer_uses_visual_bridge(layer_index)
                else None
            )
            if use_checkpoint:
                hidden = torch.utils.checkpoint.checkpoint(
                    self._expert_layer_with_latent_bridge,
                    layer_index,
                    hidden,
                    attention_mask,
                    position_ids,
                    expert_condition,
                    expert_skill,
                    layer_latent,
                    position_embeddings,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden = self._expert_layer_with_latent_bridge(
                    layer_index,
                    hidden,
                    attention_mask,
                    position_ids,
                    expert_condition,
                    expert_skill,
                    layer_latent,
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

    def _diagnostic_action_forward(
        self,
        condition_tokens: Tensor,
        state: Tensor | None,
        skill_code: Tensor | None,
        end_pose: Tensor | None,
        *,
        probe_time: float,
        noisy_actions: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Run one deterministic flow point and return velocity plus layer latents."""
        batch_size = int(condition_tokens.shape[0])
        if noisy_actions is None:
            noisy_actions = torch.zeros(
                batch_size,
                int(self.config.chunk_size),
                int(self.config.max_action_dim),
                device=condition_tokens.device,
                dtype=torch.float32,
            )
        expected = (
            batch_size,
            int(self.config.chunk_size),
            int(self.config.max_action_dim),
        )
        if tuple(noisy_actions.shape) != expected:
            raise ValueError(
                f"Diagnostic noisy_actions must have shape {expected}, got "
                f"{tuple(noisy_actions.shape)}."
            )
        projected_state = self._project_condition_state(
            state, None, skill_code, end_pose
        )
        condition_skill, expert_skill = self._skill_broadcasts(skill_code)
        if condition_skill is not None or expert_skill is None:
            raise RuntimeError(
                "Layerwise action diagnostics require expert-only skill broadcast."
            )
        layer_latents = self._encode_layerwise_latents(
            condition_tokens, projected_state
        )
        time = torch.full(
            (batch_size,),
            float(probe_time),
            dtype=torch.float32,
            device=condition_tokens.device,
        )
        expert_condition = self._expert_condition(
            time,
            projected_state=projected_state,
            skill_code=skill_code,
            end_pose=end_pose,
        )
        action_hidden = self._run_expert_with_layerwise_latents(
            noisy_actions,
            expert_condition,
            expert_skill,
            layer_latents,
        )
        return self._action_velocity(action_hidden), layer_latents

    @torch.no_grad()
    def action_vision_attention_diagnostics(
        self,
        condition_tokens: Tensor,
        state: Tensor | None,
        skill_code: Tensor | None,
        end_pose: Tensor | None,
        *,
        probe_time: float = 0.5,
        noisy_actions: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compose Action->bottleneck and bottleneck->condition MHA weights.

        This is a direct two-hop attention diagnostic at every active visual
        bridge layer. It deliberately bypasses the auxiliary alignment head.
        When several terminal bridge layers are active, their maps are averaged
        in proportion to the absolute learned bridge gate.
        """
        if self.training:
            raise RuntimeError("Action attention diagnostics require model.eval().")
        self._captured_visual_bridge_attention.clear()
        self._capture_visual_bridge_attention = True
        for reader in self.layerwise_condition_readers:
            reader.capture_cross_attention = True
            reader.last_cross_attention = None
        try:
            velocity, _ = self._diagnostic_action_forward(
                condition_tokens,
                state,
                skill_code,
                end_pose,
                probe_time=probe_time,
                noisy_actions=noisy_actions,
            )
            layer_maps: list[Tensor] = []
            action_to_latent: list[Tensor] = []
            layer_indices: list[int] = []
            layer_gates: list[Tensor] = []
            for layer_index in range(
                self.visual_bridge_start_layer,
                int(self.gemma_expert.model.config.num_hidden_layers),
            ):
                bridge = self._captured_visual_bridge_attention.get(layer_index)
                reader = self.layerwise_condition_readers[layer_index]
                condition = reader.last_cross_attention
                if bridge is None or condition is None:
                    raise RuntimeError(
                        f"Missing captured attention at bridge layer {layer_index}."
                    )
                # [B,H,T,Q] -> [B,T,Q], [B,H,Q,S] -> [B,Q,S]
                action_weights = bridge.float().mean(dim=1)
                condition_weights = condition.float().mean(dim=1)
                composed = torch.bmm(action_weights, condition_weights)
                composed = composed / composed.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                layer_maps.append(composed)
                action_to_latent.append(action_weights)
                layer_indices.append(layer_index)
                layer_gates.append(
                    self.visual_bridge_gates[layer_index].detach().float().tanh().abs()
                )

            gate_weights = torch.stack(layer_gates)
            if float(gate_weights.sum()) <= 1e-12:
                gate_weights = torch.ones_like(gate_weights)
            gate_weights = gate_weights / gate_weights.sum()
            stacked_maps = torch.stack(layer_maps, dim=1)
            stacked_action = torch.stack(action_to_latent, dim=1)
            combined = (stacked_maps * gate_weights[None, :, None, None]).sum(dim=1)
            combined_action = (
                stacked_action * gate_weights[None, :, None, None]
            ).sum(dim=1)
            return {
                "condition_attention": combined.detach(),
                "action_to_latent": combined_action.detach(),
                "per_layer_condition_attention": stacked_maps.detach(),
                "bridge_layer_indices": torch.tensor(
                    layer_indices, device=combined.device, dtype=torch.long
                ),
                "bridge_gate_weights": gate_weights.detach(),
                "velocity": velocity.detach(),
            }
        finally:
            self._capture_visual_bridge_attention = False
            self._captured_visual_bridge_attention.clear()
            for reader in self.layerwise_condition_readers:
                reader.capture_cross_attention = False
                reader.last_cross_attention = None

    def action_vision_gradient_saliency(
        self,
        condition_tokens: Tensor,
        state: Tensor | None,
        skill_code: Tensor | None,
        end_pose: Tensor | None,
        *,
        action_dims: int,
        probe_time: float = 0.5,
        noisy_actions: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Gradient norm of each timestep's output magnitude by condition token.

        One VJP is evaluated per action timestep. The resulting norm over the
        token channel dimension is a local sensitivity measure and does not use
        the auxiliary spatial-alignment head.
        """
        if self.training:
            raise RuntimeError("Action gradient diagnostics require model.eval().")
        if not 0 < int(action_dims) <= int(self.config.max_action_dim):
            raise ValueError(
                f"action_dims must be in 1..{self.config.max_action_dim}, got {action_dims}."
            )
        tokens = condition_tokens.detach().requires_grad_(True)
        velocity, _ = self._diagnostic_action_forward(
            tokens,
            state,
            skill_code,
            end_pose,
            probe_time=probe_time,
            noisy_actions=noisy_actions,
        )
        saliency: list[Tensor] = []
        timesteps = int(velocity.shape[1])
        for timestep in range(timesteps):
            magnitude = velocity[:, timestep, :action_dims].float().square().sum(
                dim=-1
            ).add(1e-12).sqrt().sum()
            gradient = torch.autograd.grad(
                magnitude,
                tokens,
                retain_graph=timestep + 1 < timesteps,
                create_graph=False,
                allow_unused=False,
            )[0]
            saliency.append(gradient.float().square().sum(dim=-1).sqrt())
        return {
            "condition_token_saliency": torch.stack(saliency, dim=1).detach(),
            "velocity": velocity.detach(),
        }

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
            raise RuntimeError("Arch3 requires expert-only skill broadcast.")
        layer_latents = self._encode_layerwise_latents(
            condition_tokens, condition_state
        )
        return self._run_expert_with_layerwise_latents(
            noisy_actions,
            expert_condition,
            expert_skill,
            layer_latents,
            return_all_layers=return_all_layers,
        )

    def _record_visual_debug(self, condition_tokens: Tensor) -> None:
        super()._record_visual_debug(condition_tokens)
        if not self._vsa_debug_active:
            return
        raw_gates = self._active_visual_bridge_gates().detach().float()
        gates = raw_gates.tanh()
        initial = float(self.config.visual_bridge_gate_init)
        self._last_vsa_debug_stats.update(
            {
                "bridge_gate/value_abs_mean": float(gates.abs().mean().item()),
                "bridge_gate/layer_std": float(gates.std(unbiased=False).item()),
                "bridge_gate/update_from_init_rms": float(
                    (raw_gates - initial).square().mean().sqrt().item()
                ),
            }
        )

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
        """Cache the recurrent visual interface once per generated chunk."""
        projected_state = self._project_condition_state(
            state, focus_uv, skill_code, end_pose
        )
        condition_skill, expert_skill = self._skill_broadcasts(skill_code)
        if condition_skill is not None or expert_skill is None:
            raise RuntimeError("Arch3 requires expert-only skill broadcast.")
        layer_latents = self._encode_layerwise_latents(
            condition_tokens, projected_state
        )
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
                end_pose=end_pose,
            )
            hidden = self._run_expert_with_layerwise_latents(
                x_t,
                expert_condition,
                expert_skill,
                layer_latents,
            )
            velocity = self._action_velocity(hidden)
            x_t = x_t + dt * velocity
        return x_t


class CoreExitLayerwiseCondBottleneckSkillExpert(LayerwiseCondBottleneckSkillExpert):
    """Arch4: Arch3's deployed path with skill flow exiting before the bridges.

    The auxiliary trajectory is decoded from the pure skill-motion prefix via
    the same final norm and action head as the deployed 18-layer action path.
    No condition token, robot state, or visual bridge enters this prefix.
    """

    def _skill_only_expert_hidden(
        self,
        action_tokens: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        expert_condition: Tensor,
        expert_skill: Tensor,
    ) -> Tensor:
        hidden = action_tokens
        position_embeddings = self.gemma_expert.model.rotary_emb(hidden, position_ids)
        use_checkpoint = self._gradient_checkpointing and self.training
        for layer_index in range(self.visual_bridge_start_layer):
            if use_checkpoint:
                hidden = torch.utils.checkpoint.checkpoint(
                    self._expert_layer_with_latent_bridge,
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
                hidden = self._expert_layer_with_latent_bridge(
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


class _AuxiliaryUVAlignedCoreExitLayerwiseCondBottleneckSkillExpert(
    CoreExitLayerwiseCondBottleneckSkillExpert
):
    """Shared training-only UV readout for Arch5 and Arch6."""

    _focus_source = "cond"

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        readout_width = (
            self.width if self._focus_source == "cond"
            else int(config.visual_bottleneck_width)
        )
        self.focus_uv_token_norm = nn.LayerNorm(readout_width)
        self.focus_uv_token_score = nn.Linear(readout_width, 1)
        self.focus_uv_head = nn.Sequential(
            nn.LayerNorm(readout_width),
            nn.Linear(readout_width, readout_width // 2),
            nn.SiLU(),
            nn.Linear(readout_width // 2, 2),
            nn.Tanh(),
        )
        self._final_uv_tokens: Tensor | None = None

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.focus_uv_token_norm.to(dtype=torch.float32)
        self.focus_uv_token_score.to(dtype=torch.float32)
        self.focus_uv_head.to(dtype=torch.float32)
        return self

    def _on_final_condition_hidden(self, hidden: Tensor) -> None:
        if self._focus_source == "cond":
            self._final_uv_tokens = hidden if self.training else None

    def _on_final_bottleneck_latent(self, latent: Tensor) -> None:
        if self._focus_source == "bottleneck":
            self._final_uv_tokens = latent if self.training else None

    def predict_training_focus_uv(self) -> Tensor:
        tokens = self._final_uv_tokens
        self._final_uv_tokens = None
        if tokens is None:
            raise RuntimeError("UV readout requires a preceding training condition forward.")
        normalized = self.focus_uv_token_norm(tokens.float())
        weights = self.focus_uv_token_score(normalized).softmax(dim=1)
        pooled = (weights * normalized).sum(dim=1)
        return self.focus_uv_head(pooled)


class UVAlignedCoreExitLayerwiseCondBottleneckSkillExpert(
    _AuxiliaryUVAlignedCoreExitLayerwiseCondBottleneckSkillExpert
):
    """Arch5: Arch4 action path plus a final Cond-hidden UV readout."""


class BottleneckUVAlignedCoreExitLayerwiseCondBottleneckSkillExpert(
    _AuxiliaryUVAlignedCoreExitLayerwiseCondBottleneckSkillExpert
):
    """Arch6: Arch4 action path plus a final bottleneck-token UV readout."""

    _focus_source = "bottleneck"


class BottleneckXYZAlignedCoreExitLayerwiseCondBottleneckSkillExpert(
    CoreExitLayerwiseCondBottleneckSkillExpert
):
    """Arch7: Arch6 action path with a training-only skill-end XYZ readout.

    The readout supervises the final bottleneck tokens that the terminal visual
    bridge actually consumes. It does not change the deployed action path.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        width = int(config.visual_bottleneck_width)
        self.end_xyz_token_norm = nn.LayerNorm(width)
        self.end_xyz_token_score = nn.Linear(width, 1)
        self.end_xyz_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 3),
        )
        self._final_xyz_tokens: Tensor | None = None

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.end_xyz_token_norm.to(dtype=torch.float32)
        self.end_xyz_token_score.to(dtype=torch.float32)
        self.end_xyz_head.to(dtype=torch.float32)
        return self

    def _on_final_bottleneck_latent(self, latent: Tensor) -> None:
        self._final_xyz_tokens = latent if self.training else None

    def predict_training_end_xyz(self) -> Tensor:
        tokens = self._final_xyz_tokens
        self._final_xyz_tokens = None
        if tokens is None:
            raise RuntimeError("XYZ readout requires a preceding training condition forward.")
        normalized = self.end_xyz_token_norm(tokens.float())
        weights = self.end_xyz_token_score(normalized).softmax(dim=1)
        pooled = (weights * normalized).sum(dim=1)
        return self.end_xyz_head(pooled)


class UVConditionedBottleneckXYZSkillExpert(
    BottleneckXYZAlignedCoreExitLayerwiseCondBottleneckSkillExpert
):
    """Arch8_1: Arch7 with skill-end image UV in Cond-Gemma AdaRMS.

    UV is added to the existing robot-state condition at every Cond layer.
    It never enters the skill-only Expert prefix or the Expert time condition.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        self.focus_uv_condition = nn.Sequential(
            nn.Linear(2, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        # Start from the same AdaRMS signal as Arch7 while the UV branch learns.
        nn.init.zeros_(self.focus_uv_condition[-1].weight)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.focus_uv_condition.to(dtype=torch.float32)
        return self

    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        del skill_code, end_pose
        projected_state = self._project_state(state)
        if focus_uv is None:
            raise ValueError("Arch8 requires skill-end focus UV for Cond-Gemma AdaRMS.")
        if focus_uv.ndim != 2 or focus_uv.shape != (projected_state.shape[0], 2):
            raise ValueError(
                "Arch8 focus UV must have shape [batch, 2], got "
                f"{tuple(focus_uv.shape)}."
            )
        if not bool(torch.isfinite(focus_uv).all()):
            raise ValueError("Arch8 focus UV must be finite.")
        uv_condition = self.focus_uv_condition(focus_uv.float())
        return projected_state + uv_condition.to(projected_state.dtype)


class XYZConditionedBottleneckUVSkillExpert(
    BottleneckUVAlignedCoreExitLayerwiseCondBottleneckSkillExpert
):
    """Arch13: Arch8_1 with the spatial input/readout direction reversed.

    The fixed skill-end EEF XYZ augments proprio in Cond-Gemma AdaRMS, while
    the final recurrent visual bottleneck predicts the corresponding top-view
    focus UV. XYZ does not enter the pure skill-only Expert prefix.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        self.end_xyz_condition = nn.Sequential(
            nn.Linear(3, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        # Preserve Arch6/Arch8_1's initial proprio-only Cond signal.
        nn.init.zeros_(self.end_xyz_condition[-1].weight)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.end_xyz_condition.to(dtype=torch.float32)
        return self

    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        del focus_uv, skill_code
        projected_state = self._project_state(state)
        if end_pose is None:
            raise ValueError("Arch13 requires skill-end EEF XYZ for Cond-Gemma AdaRMS.")
        if end_pose.ndim != 2 or end_pose.shape != (projected_state.shape[0], 3):
            raise ValueError(
                "Arch13 skill-end EEF XYZ must have shape [batch, 3], got "
                f"{tuple(end_pose.shape)}."
            )
        if not bool(torch.isfinite(end_pose).all()):
            raise ValueError("Arch13 skill-end EEF XYZ must be finite.")
        xyz_condition = self.end_xyz_condition(end_pose.float())
        return projected_state + xyz_condition.to(projected_state.dtype)


class XYZConditionedBottleneckUVExpertEndPoseSkillExpert(
    XYZConditionedBottleneckUVSkillExpert
):
    """Arch14: Arch13 plus Arch11-style skill-end EEF pose in the Action Expert AdaRMS.

    Arch13 keeps the skill-end pose on the Cond side only, so its skill-only
    Expert prefix is goal-free. Arch14 also adds the same fixed per-occurrence
    pose to the Expert AdaRMS condition. ``_expert_condition`` is shared by the
    deployed route and the skill-only route, so both are goal-conditioned, exactly
    as in Arch9--Arch11. ``skill_end_pose_mode`` selects XYZ (3) or XYZ+axis-angle
    (6); the same pose vector enters Cond-Gemma and the Expert.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        pose_width = 3 if config.skill_end_pose_mode == "xyz" else 6
        if pose_width != 3:
            # Arch13 builds a 3-D Cond projection; pose mode widens it.
            self.end_xyz_condition = nn.Sequential(
                nn.Linear(pose_width, self.width),
                nn.SiLU(),
                nn.Linear(self.width, self.width, bias=False),
            )
            nn.init.zeros_(self.end_xyz_condition[-1].weight)
        # Same module name/shape as Arch9--Arch11 so NewTask FT freezes it with the
        # rest of the skill-only route. Zero-init keeps the pi0.5 Expert signal at start.
        self.end_pose_condition = nn.Sequential(
            nn.Linear(pose_width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        nn.init.zeros_(self.end_pose_condition[-1].weight)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.end_pose_condition.to(dtype=torch.float32)
        return self

    def _pose_width(self) -> int:
        return 3 if self.config.skill_end_pose_mode == "xyz" else 6

    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        del focus_uv, skill_code
        projected_state = self._project_state(state)
        expected = self._pose_width()
        if end_pose is None:
            raise ValueError("Arch14 requires the skill-end EEF pose for Cond-Gemma AdaRMS.")
        if end_pose.ndim != 2 or end_pose.shape != (projected_state.shape[0], expected):
            raise ValueError(
                f"Arch14 skill-end EEF pose must have shape [batch, {expected}], got "
                f"{tuple(end_pose.shape)}."
            )
        if not bool(torch.isfinite(end_pose).all()):
            raise ValueError("Arch14 skill-end EEF pose must be finite.")
        pose_condition = self.end_xyz_condition(end_pose.float())
        return projected_state + pose_condition.to(projected_state.dtype)

    def _expert_condition(
        self, timestep: Tensor, projected_state: Tensor | None = None,
        skill_code: Tensor | None = None, mode_latent: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        condition = super()._expert_condition(
            timestep, projected_state=projected_state, skill_code=skill_code,
            mode_latent=mode_latent,
        )
        if end_pose is None:
            # Same contract as Arch9--Arch11: callers that have no goal (legacy skill-only
            # sampling) get the pose-free condition instead of a crash.
            return condition
        expected = self._pose_width()
        if end_pose.ndim != 2 or end_pose.shape != (condition.shape[0], expected):
            raise ValueError(
                f"Arch14 end pose must have shape [batch, {expected}], got {tuple(end_pose.shape)}."
            )
        if not bool(torch.isfinite(end_pose).all()):
            raise ValueError("Arch14 end pose must be finite.")
        projected_pose = self.end_pose_condition(end_pose.float())
        return condition + projected_pose.to(condition.dtype)


class XYZSkillConditionedBottleneckUVExpertEndPoseSkillExpert(
    XYZConditionedBottleneckUVExpertEndPoseSkillExpert
):
    """Arch15: Arch14 plus the skill in the Cond-Gemma AdaRMS.

    Arch14's Cond-Gemma sees proprio and the skill-end pose but not the skill; the
    skill reaches only the Action Expert broadcast. Arch15 adds the FSQ skill
    coordinates to the Cond AdaRMS input through the same zero-initialised
    projection Arch9/Arch11 use, so the visual reader can be skill-aware. The
    Expert side (skill broadcast + end-pose AdaRMS) is unchanged, hence the
    skill-only route is identical to Arch14's.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        self.cond_skill_condition = nn.Sequential(
            nn.Linear(len(config.skill_fsq_levels), self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        nn.init.zeros_(self.cond_skill_condition[-1].weight)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.cond_skill_condition.to(dtype=torch.float32)
        return self

    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        projected = super()._project_condition_state(state, focus_uv, skill_code, end_pose)
        if skill_code is None:
            raise ValueError("Arch15 requires the skill for Cond-Gemma AdaRMS.")
        coordinates = self._code_to_zq(skill_code)
        projected_skill = self.cond_skill_condition(coordinates.float())
        return projected + projected_skill.to(projected.dtype)


class UVConditionedBottleneckXYZTerminationSkillExpert(
    UVConditionedBottleneckXYZSkillExpert
):
    """Arch8_2: Arch8_1 plus a termination readout from final Cond hidden.

    The legacy v1 revision retains its bottleneck-token readout so existing
    checkpoints remain loadable without changing parameter shapes.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        self._termination_source = (
            "bottleneck"
            if config.architecture_revision
            == LAYERWISE_COND_BOTTLENECK_UV_COND_XYZ_TERMINATION_REVISION
            else "cond"
        )
        width = (
            int(config.visual_bottleneck_width)
            if self._termination_source == "bottleneck"
            else self.width
        )
        self.termination_token_norm = nn.LayerNorm(width)
        self.termination_token_score = nn.Linear(width, 1)
        self.termination_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )
        self._final_termination_tokens: Tensor | None = None
        self.last_termination_probability: Tensor | None = None

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.termination_token_norm.to(dtype=torch.float32)
        self.termination_token_score.to(dtype=torch.float32)
        self.termination_head.to(dtype=torch.float32)
        return self

    def _termination_logits(self, tokens: Tensor) -> Tensor:
        normalized = self.termination_token_norm(tokens.float())
        weights = self.termination_token_score(normalized).softmax(dim=1)
        pooled = (weights * normalized).sum(dim=1)
        return self.termination_head(pooled).squeeze(-1)

    def _capture_termination_tokens(self, tokens: Tensor) -> None:
        self._final_termination_tokens = tokens if self.training else None
        self.last_termination_probability = (
            None if self.training else self._termination_logits(tokens).sigmoid().detach()
        )

    def _on_final_condition_hidden(self, hidden: Tensor) -> None:
        super()._on_final_condition_hidden(hidden)
        if self._termination_source == "cond":
            self._capture_termination_tokens(hidden)

    def _on_final_bottleneck_latent(self, latent: Tensor) -> None:
        super()._on_final_bottleneck_latent(latent)
        if self._termination_source == "bottleneck":
            self._capture_termination_tokens(latent)

    def predict_training_termination_logits(self) -> Tensor:
        tokens = self._final_termination_tokens
        self._final_termination_tokens = None
        if tokens is None:
            raise RuntimeError("Termination readout requires a preceding training condition forward.")
        return self._termination_logits(tokens)


class WristEndPoseLayerwiseCondBottleneckSkillExpert(
    CoreExitLayerwiseCondBottleneckSkillExpert
):
    """Arch10_1: wrist-only vision and Expert end-pose AdaRMS.

    Cond-Gemma keeps only its ordinary proprio AdaRMS input. The existing
    Expert skill broadcast is retained, and both the deployed and skill-only
    paths receive the same fixed per-occurrence end-pose AdaRMS condition.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        self.end_pose_condition = nn.Sequential(
            nn.Linear(3 if config.skill_end_pose_mode == "xyz" else 6, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        nn.init.zeros_(self.end_pose_condition[-1].weight)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.end_pose_condition.to(dtype=torch.float32)
        return self

    def _condition_tokens(
        self, images: list[Tensor], *, batch_size: int | None = None,
        skill_code: Tensor | None = None,
    ) -> Tensor:
        del batch_size, skill_code
        if len(images) != 1:
            raise ValueError(
                f"Arch9--Arch11 require only the wrist camera, got {len(images)} images."
            )
        features = self._image_features(images[0])
        return self.image_proj(
            features.to(dtype=self.image_proj.weight.dtype)
        ).to(self.working_dtype)

    def _expert_condition(
        self, timestep: Tensor, projected_state: Tensor | None = None,
        skill_code: Tensor | None = None, mode_latent: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        condition = super()._expert_condition(
            timestep, projected_state=projected_state, skill_code=skill_code,
            mode_latent=mode_latent,
        )
        if end_pose is None:
            return condition
        expected = 3 if self.config.skill_end_pose_mode == "xyz" else 6
        if end_pose.ndim != 2 or end_pose.shape != (condition.shape[0], expected):
            raise ValueError(
                "Arch9--Arch11 end pose must have shape "
                f"[batch, {expected}], got {tuple(end_pose.shape)}."
            )
        if not bool(torch.isfinite(end_pose).all()):
            raise ValueError("Arch9--Arch11 end pose must be finite.")
        projected_pose = self.end_pose_condition(end_pose.float())
        return condition + projected_pose.to(condition.dtype)


class WristSkillEndPoseLayerwiseCondBottleneckSkillExpert(
    WristEndPoseLayerwiseCondBottleneckSkillExpert
):
    """Arch9_1: Arch10_1 plus skill AdaRMS on the Cond-Gemma path."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        self.cond_skill_condition = nn.Sequential(
            nn.Linear(len(config.skill_fsq_levels), self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        nn.init.zeros_(self.cond_skill_condition[-1].weight)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.cond_skill_condition.to(dtype=torch.float32)
        return self

    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        del focus_uv, end_pose
        projected_state = self._project_state(state)
        if skill_code is None:
            raise ValueError("Arch9_1 requires skill for Cond-Gemma AdaRMS.")
        coordinates = self._code_to_zq(skill_code)
        projected_skill = self.cond_skill_condition(coordinates.float())
        return projected_state + projected_skill.to(projected_state.dtype)


class WristSkillEndPoseTerminationSkillExpert(
    WristSkillEndPoseLayerwiseCondBottleneckSkillExpert
):
    """Arch9_2: Arch9_1 plus a termination readout from final Cond hidden.

    The legacy v1 revision retains its bottleneck-token readout so existing
    checkpoints remain loadable without changing parameter shapes.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        self._termination_source = (
            "bottleneck"
            if config.architecture_revision
            == LAYERWISE_COND_BOTTLENECK_WRIST_SKILL_END_POSE_TERMINATION_REVISION
            else "cond"
        )
        width = (
            int(config.visual_bottleneck_width)
            if self._termination_source == "bottleneck"
            else self.width
        )
        self.termination_token_norm = nn.LayerNorm(width)
        self.termination_token_score = nn.Linear(width, 1)
        self.termination_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )
        self._final_termination_tokens: Tensor | None = None
        self.last_termination_probability: Tensor | None = None

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.termination_token_norm.to(dtype=torch.float32)
        self.termination_token_score.to(dtype=torch.float32)
        self.termination_head.to(dtype=torch.float32)
        return self

    def _termination_logits(self, tokens: Tensor) -> Tensor:
        normalized = self.termination_token_norm(tokens.float())
        weights = self.termination_token_score(normalized).softmax(dim=1)
        pooled = (weights * normalized).sum(dim=1)
        return self.termination_head(pooled).squeeze(-1)

    def _capture_termination_tokens(self, tokens: Tensor) -> None:
        self._final_termination_tokens = tokens if self.training else None
        self.last_termination_probability = (
            None if self.training else self._termination_logits(tokens).sigmoid().detach()
        )

    def _on_final_condition_hidden(self, hidden: Tensor) -> None:
        super()._on_final_condition_hidden(hidden)
        if self._termination_source == "cond":
            self._capture_termination_tokens(hidden)

    def _on_final_bottleneck_latent(self, latent: Tensor) -> None:
        super()._on_final_bottleneck_latent(latent)
        if self._termination_source == "bottleneck":
            self._capture_termination_tokens(latent)

    def predict_training_termination_logits(self) -> Tensor:
        tokens = self._final_termination_tokens
        self._final_termination_tokens = None
        if tokens is None:
            raise RuntimeError("Termination readout requires a preceding training condition forward.")
        return self._termination_logits(tokens)


class WristEndPoseTerminationSkillExpert(
    WristEndPoseLayerwiseCondBottleneckSkillExpert
):
    """Arch10_2: Arch10_1 plus final-Cond termination prediction."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        width = self.width
        self.termination_token_norm = nn.LayerNorm(width)
        self.termination_token_score = nn.Linear(width, 1)
        self.termination_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )
        self._final_termination_tokens: Tensor | None = None
        self.last_termination_probability: Tensor | None = None

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.termination_token_norm.to(dtype=torch.float32)
        self.termination_token_score.to(dtype=torch.float32)
        self.termination_head.to(dtype=torch.float32)
        return self

    def _termination_logits(self, tokens: Tensor) -> Tensor:
        normalized = self.termination_token_norm(tokens.float())
        weights = self.termination_token_score(normalized).softmax(dim=1)
        pooled = (weights * normalized).sum(dim=1)
        return self.termination_head(pooled).squeeze(-1)

    def _on_final_condition_hidden(self, hidden: Tensor) -> None:
        super()._on_final_condition_hidden(hidden)
        self._final_termination_tokens = hidden if self.training else None
        self.last_termination_probability = (
            None
            if self.training
            else self._termination_logits(hidden).sigmoid().detach()
        )

    def predict_training_termination_logits(self) -> Tensor:
        tokens = self._final_termination_tokens
        self._final_termination_tokens = None
        if tokens is None:
            raise RuntimeError(
                "Termination readout requires a preceding training condition forward."
            )
        return self._termination_logits(tokens)


class WristCondSkillEndPoseLayerwiseCondBottleneckSkillExpert(
    WristSkillEndPoseLayerwiseCondBottleneckSkillExpert
):
    """Arch11_1: wrist-only Cond/Expert conditioning with skill-end pose.

    Cond-Gemma receives proprio, skill coordinates, and the fixed skill-end
    EEF pose through AdaRMS.  The Action Expert keeps the ordinary skill
    broadcast and the same end-pose AdaRMS used by Arch9/Arch10.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        pose_width = 3 if config.skill_end_pose_mode == "xyz" else 6
        self.cond_end_pose_condition = nn.Sequential(
            nn.Linear(pose_width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        nn.init.zeros_(self.cond_end_pose_condition[-1].weight)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.cond_end_pose_condition.to(dtype=torch.float32)
        return self

    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        projected_state = super()._project_condition_state(
            state, focus_uv, skill_code, end_pose
        )
        expected = 3 if self.config.skill_end_pose_mode == "xyz" else 6
        if end_pose is None:
            raise ValueError("Arch11 requires skill-end pose for Cond-Gemma AdaRMS.")
        if end_pose.ndim != 2 or end_pose.shape != (projected_state.shape[0], expected):
            raise ValueError(
                "Arch11 end pose must have shape "
                f"[batch, {expected}], got {tuple(end_pose.shape)}."
            )
        if not bool(torch.isfinite(end_pose).all()):
            raise ValueError("Arch11 end pose must be finite.")
        pose_condition = self.cond_end_pose_condition(end_pose.float())
        return projected_state + pose_condition.to(projected_state.dtype)


class WristCondSkillEndPoseTerminationSkillExpert(
    WristCondSkillEndPoseLayerwiseCondBottleneckSkillExpert
):
    """Arch11_2: Arch11_1 plus final-Cond termination prediction."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        width = self.width
        self.termination_token_norm = nn.LayerNorm(width)
        self.termination_token_score = nn.Linear(width, 1)
        self.termination_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )
        self._final_termination_tokens: Tensor | None = None
        self.last_termination_probability: Tensor | None = None

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.termination_token_norm.to(dtype=torch.float32)
        self.termination_token_score.to(dtype=torch.float32)
        self.termination_head.to(dtype=torch.float32)
        return self

    def _termination_logits(self, tokens: Tensor) -> Tensor:
        normalized = self.termination_token_norm(tokens.float())
        weights = self.termination_token_score(normalized).softmax(dim=1)
        pooled = (weights * normalized).sum(dim=1)
        return self.termination_head(pooled).squeeze(-1)

    def _on_final_condition_hidden(self, hidden: Tensor) -> None:
        super()._on_final_condition_hidden(hidden)
        self._final_termination_tokens = hidden if self.training else None
        self.last_termination_probability = (
            None
            if self.training
            else self._termination_logits(hidden).sigmoid().detach()
        )

    def predict_training_termination_logits(self) -> Tensor:
        tokens = self._final_termination_tokens
        self._final_termination_tokens = None
        if tokens is None:
            raise RuntimeError(
                "Termination readout requires a preceding training condition forward."
            )
        return self._termination_logits(tokens)


class WristCondSkillEndPoseExpertSkillLayerwiseCondBottleneckSkillExpert(
    CoreExitLayerwiseCondBottleneckSkillExpert
):
    """Arch12_1: Arch11 Cond inputs without Expert end-pose AdaRMS.

    Cond-Gemma receives proprio, skill, and skill-end pose.  The Action Expert
    receives only its existing layerwise skill broadcast, so the skill-only
    route remains independent of the end pose.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        pose_width = 3 if config.skill_end_pose_mode == "xyz" else 6
        skill_width = len(config.skill_fsq_levels)
        self.cond_skill_condition = nn.Sequential(
            nn.Linear(skill_width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        self.cond_end_pose_condition = nn.Sequential(
            nn.Linear(pose_width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        nn.init.zeros_(self.cond_skill_condition[-1].weight)
        nn.init.zeros_(self.cond_end_pose_condition[-1].weight)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.cond_skill_condition.to(dtype=torch.float32)
        self.cond_end_pose_condition.to(dtype=torch.float32)
        return self

    def _condition_tokens(
        self, images: list[Tensor], *, batch_size: int | None = None,
        skill_code: Tensor | None = None,
    ) -> Tensor:
        del batch_size, skill_code
        if len(images) != 1:
            raise ValueError(
                f"Arch12 requires only the wrist camera, got {len(images)} images."
            )
        features = self._image_features(images[0])
        return self.image_proj(
            features.to(dtype=self.image_proj.weight.dtype)
        ).to(self.working_dtype)

    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        del focus_uv
        projected_state = self._project_state(state)
        if skill_code is None:
            raise ValueError("Arch12 requires skill for Cond-Gemma AdaRMS.")
        expected = 3 if self.config.skill_end_pose_mode == "xyz" else 6
        if end_pose is None:
            raise ValueError("Arch12 requires skill-end pose for Cond-Gemma AdaRMS.")
        if end_pose.ndim != 2 or end_pose.shape != (projected_state.shape[0], expected):
            raise ValueError(
                "Arch12 end pose must have shape "
                f"[batch, {expected}], got {tuple(end_pose.shape)}."
            )
        if not bool(torch.isfinite(end_pose).all()):
            raise ValueError("Arch12 end pose must be finite.")
        coordinates = self._code_to_zq(skill_code)
        skill_condition = self.cond_skill_condition(coordinates.float())
        pose_condition = self.cond_end_pose_condition(end_pose.float())
        return projected_state + (
            skill_condition + pose_condition
        ).to(projected_state.dtype)


class WristCondSkillEndPoseExpertSkillTerminationSkillExpert(
    WristCondSkillEndPoseExpertSkillLayerwiseCondBottleneckSkillExpert
):
    """Arch12_2: Arch12_1 plus final-Cond termination prediction."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        width = self.width
        self.termination_token_norm = nn.LayerNorm(width)
        self.termination_token_score = nn.Linear(width, 1)
        self.termination_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )
        self._final_termination_tokens: Tensor | None = None
        self.last_termination_probability: Tensor | None = None

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.termination_token_norm.to(dtype=torch.float32)
        self.termination_token_score.to(dtype=torch.float32)
        self.termination_head.to(dtype=torch.float32)
        return self

    def _termination_logits(self, tokens: Tensor) -> Tensor:
        normalized = self.termination_token_norm(tokens.float())
        weights = self.termination_token_score(normalized).softmax(dim=1)
        pooled = (weights * normalized).sum(dim=1)
        return self.termination_head(pooled).squeeze(-1)

    def _on_final_condition_hidden(self, hidden: Tensor) -> None:
        super()._on_final_condition_hidden(hidden)
        self._final_termination_tokens = hidden if self.training else None
        self.last_termination_probability = (
            None
            if self.training
            else self._termination_logits(hidden).sigmoid().detach()
        )

    def predict_training_termination_logits(self) -> Tensor:
        tokens = self._final_termination_tokens
        self._final_termination_tokens = None
        if tokens is None:
            raise RuntimeError(
                "Termination readout requires a preceding training condition forward."
            )
        return self._termination_logits(tokens)


class WristSkillDeltaGoalSkillExpert(WristCondSkillEndPoseLayerwiseCondBottleneckSkillExpert):
    """Arch16: Arch11_1 whose Action Expert goal is the skill displacement.

    The policy packs ``end_pose = [skill-end xyz, skill-end xyz - skill-start xyz]``. Cond-Gemma
    keeps the absolute goal (it lives next to the current proprio, so the remaining distance is
    recoverable there), while the Expert AdaRMS receives ONLY the displacement. LIBERO actions are
    end-effector deltas, so a skill trajectory is shaped by how far and in which direction the
    skill moves, not by where in the workspace it happens; an absolute-free goal therefore keeps
    the motion core translation invariant and makes the state-free skill-only route well posed.
    ``_expert_condition`` is shared, so the deployed and the skill-only route see the same goal.
    """

    @staticmethod
    def _split_goal(end_pose: Tensor | None, label: str) -> tuple[Tensor, Tensor]:
        if end_pose is None:
            raise ValueError(f"{label} requires the packed [skill-end xyz, skill displacement] goal.")
        if end_pose.ndim != 2 or end_pose.shape[1] != 6:
            raise ValueError(
                f"{label} goal must have shape [batch, 6] = [end xyz, end - start xyz], got "
                f"{tuple(end_pose.shape)}."
            )
        return end_pose[:, :3], end_pose[:, 3:]

    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        absolute_goal, _ = self._split_goal(end_pose, "Arch16/Arch17 Cond-Gemma")
        return super()._project_condition_state(state, focus_uv, skill_code, absolute_goal)

    def _expert_condition(
        self, timestep: Tensor, projected_state: Tensor | None = None,
        skill_code: Tensor | None = None, mode_latent: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        if end_pose is None:
            # Same contract as Arch9--Arch11 for legacy goal-free skill-only sampling.
            displacement = None
        else:
            _, displacement = self._split_goal(end_pose, "Arch16/Arch17 Expert")
        return super()._expert_condition(
            timestep, projected_state=projected_state, skill_code=skill_code,
            mode_latent=mode_latent, end_pose=displacement,
        )


class WristSkillDeltaGoalBridgeProprioSkillExpert(WristSkillDeltaGoalSkillExpert):
    """Arch17: Arch16 plus current proprio in the terminal bridge Expert layer(s).

    The motion-core prefix (layers before the visual bridge) stays state-free, so the skill-only
    route, which exits before the bridge layers, is untouched. Only the AdaRMS of the bridge
    layers is shifted by a zero-initialised proprio projection; the final Expert norm keeps the
    proprio-free condition because both routes decode through it.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        self.bridge_proprio_condition = nn.Sequential(
            nn.Linear(config.max_state_dim, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        nn.init.zeros_(self.bridge_proprio_condition[-1].weight)
        self._bridge_condition_shift: Tensor | None = None

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.bridge_proprio_condition.to(dtype=torch.float32)
        return self

    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        # Every deployed-route forward (training, cached sampling, sensitivity probes) projects
        # the Cond state right before running the Expert with the SAME state, so this is where
        # the bridge-layer shift for that forward is prepared.
        if state is None:
            raise ValueError("Arch17 requires robot state for its bridge-layer proprio AdaRMS.")
        shift = self.bridge_proprio_condition(
            state.to(dtype=next(self.bridge_proprio_condition.parameters()).dtype)
        )
        self._bridge_condition_shift = shift.to(self.working_dtype)
        return super()._project_condition_state(state, focus_uv, skill_code, end_pose)

    def _input_sensitivity_stats(self, **kwargs) -> dict[str, float]:
        # The probes re-project perturbed states. Put the real forward's shift back afterwards:
        # with gradient checkpointing the bridge layer is recomputed during backward and must
        # read the same proprio condition it used in the forward pass.
        shift = self._bridge_condition_shift
        try:
            return super()._input_sensitivity_stats(**kwargs)
        finally:
            self._bridge_condition_shift = shift

    def _expert_layer_with_latent_bridge(
        self,
        layer_index: int,
        hidden: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        expert_condition: Tensor,
        expert_skill: Tensor,
        layer_latent: Tensor | None,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        if self._layer_uses_visual_bridge(layer_index):
            shift = self._bridge_condition_shift
            if shift is None or shift.shape[0] != expert_condition.shape[0]:
                raise RuntimeError("Arch17 bridge layer ran without its proprio condition.")
            expert_condition = expert_condition + shift.to(expert_condition.dtype)
        return super()._expert_layer_with_latent_bridge(
            layer_index, hidden, attention_mask, position_ids, expert_condition,
            expert_skill, layer_latent, position_embeddings,
        )


class WristSkillStartEndGoalBridgeProprioSkillExpert(WristSkillDeltaGoalBridgeProprioSkillExpert):
    """Arch18: Arch17 with the skill start and end given to the Expert as two absolute inputs.

    The policy packs ``end_pose = [skill-end xyz, skill-start xyz]``. The Expert AdaRMS condition
    is ``time + end_pose_condition(end) + start_pose_condition(start)``: the first two terms are
    exactly Arch11_1's, the start projection is new and zero-initialised. Unlike Arch16/Arch17 the
    motion core is no longer translation invariant; in exchange the bridge layer, which also sees
    the absolute proprio, can relate "where I am" to "where this skill began and must end".
    Cond-Gemma and the bridge-layer proprio are unchanged from Arch17.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        self.start_pose_condition = nn.Sequential(
            nn.Linear(3, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        nn.init.zeros_(self.start_pose_condition[-1].weight)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.start_pose_condition.to(dtype=torch.float32)
        return self

    def _expert_condition(
        self, timestep: Tensor, projected_state: Tensor | None = None,
        skill_code: Tensor | None = None, mode_latent: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        # Skip Arch16's displacement override: the absolute end goes through Arch11_1's own
        # Expert end-pose projection, and the start gets its own.
        arch11_condition = super(WristSkillDeltaGoalSkillExpert, self)._expert_condition
        if end_pose is None:
            return arch11_condition(
                timestep, projected_state=projected_state, skill_code=skill_code,
                mode_latent=mode_latent,
            )
        end_xyz, start_xyz = self._split_goal(end_pose, "Arch18 Expert")
        condition = arch11_condition(
            timestep, projected_state=projected_state, skill_code=skill_code,
            mode_latent=mode_latent, end_pose=end_xyz,
        )
        if not bool(torch.isfinite(start_xyz).all()):
            raise ValueError("Arch18 skill-start xyz must be finite.")
        return condition + self.start_pose_condition(start_xyz.float()).to(condition.dtype)


class _LITChunkEndStateMixin:
    """LIT-style training head that decodes chunk-end state from final latents only."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        latent_width = int(config.visual_bottleneck_width)
        output_width = int(config.chunk_end_state_dim)
        self.chunk_end_state_token_norm = nn.LayerNorm(latent_width)
        self.chunk_end_state_token_score = nn.Linear(latent_width, 1)
        self.chunk_end_state_head = nn.Sequential(
            nn.LayerNorm(latent_width),
            nn.Linear(latent_width, latent_width),
            nn.SiLU(),
            nn.Linear(latent_width, output_width),
        )
        self._final_chunk_end_state_latents: Tensor | None = None
        # LIT_1--3 and LIT_6 deliberately remove the skill-end XYZ projection
        # from Cond-Gemma. LIT_4/5 retain their parent's end-goal projection.
        if config.architecture_label not in LIT_COND_END_GOAL_ARCH_LABELS:
            # Keep an Identity under the historical module name because parent
            # _apply methods still move that module to fp32.
            self.cond_end_pose_condition = nn.Identity()

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.chunk_end_state_token_norm.to(dtype=torch.float32)
        self.chunk_end_state_token_score.to(dtype=torch.float32)
        self.chunk_end_state_head.to(dtype=torch.float32)
        return self

    def _on_final_bottleneck_latent(self, latent: Tensor) -> None:
        super()._on_final_bottleneck_latent(latent)
        self._final_chunk_end_state_latents = latent if self.training else None

    def predict_training_chunk_end_state(self) -> Tensor:
        """Return normalized grounded state ``[xyz, axis-angle, gripper]``."""
        latents = self._final_chunk_end_state_latents
        self._final_chunk_end_state_latents = None
        if latents is None:
            raise RuntimeError(
                "Chunk-end state prediction requires a preceding training condition forward."
            )
        normalized = self.chunk_end_state_token_norm(latents.float())
        weights = self.chunk_end_state_token_score(normalized).softmax(dim=1)
        pooled = (weights * normalized).sum(dim=1)
        return self.chunk_end_state_head(pooled)


class _LITGoalFreeCondMixin:
    """Cond-Gemma sees current proprio + skill, never the provided Expert goal."""

    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        del end_pose
        # LIT_1 retains Arch17/18's proprio shift only in terminal Expert
        # bridge layers. LIT_2/3 do not own this module and skip this branch.
        bridge = getattr(self, "bridge_proprio_condition", None)
        if bridge is not None:
            if state is None:
                raise ValueError("LIT_1 requires robot state for bridge-layer proprio AdaRMS.")
            shift = bridge(state.to(dtype=next(bridge.parameters()).dtype))
            self._bridge_condition_shift = shift.to(self.working_dtype)
        return WristSkillEndPoseLayerwiseCondBottleneckSkillExpert._project_condition_state(
            self, state, focus_uv, skill_code, None
        )


class _LITProprioOnlyCondMixin:
    """Cond-Gemma sees current proprio only; all task context stays Expert-only."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        # The parent owns this projection for LIT_2/5. LIT_6 deliberately does
        # not condition Cond-Gemma on the skill, so do not retain dead trainable
        # parameters under the historical module name.
        self.cond_skill_condition = nn.Identity()

    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None, end_pose: Tensor | None = None,
    ) -> Tensor:
        del focus_uv, skill_code, end_pose
        return self._project_state(state)


class _BothCameraConditionMixin:
    """Read equal top/wrist token sequences through the shared vision tower."""

    def _condition_tokens(
        self, images: list[Tensor], *, batch_size: int | None = None,
        skill_code: Tensor | None = None,
    ) -> Tensor:
        del batch_size, skill_code
        if len(images) != 2:
            raise ValueError(f"Both_LIT requires [top, wrist] images, got {len(images)}.")
        camera_tokens = [
            self.image_proj(
                self._image_features(image).to(dtype=self.image_proj.weight.dtype)
            ).to(self.working_dtype)
            for image in images
        ]
        if camera_tokens[0].shape[1] != camera_tokens[1].shape[1]:
            raise ValueError(
                "Both_LIT requires equal top/wrist token lengths, got "
                f"{camera_tokens[0].shape[1]} and {camera_tokens[1].shape[1]}."
            )
        return torch.cat(camera_tokens, dim=1)


class WristSkillStartEndGoalSkillExpert(WristSkillDeltaGoalSkillExpert):
    """Arch18's start/end Expert goal without Arch17's bridge proprio."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        self.start_pose_condition = nn.Sequential(
            nn.Linear(3, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        nn.init.zeros_(self.start_pose_condition[-1].weight)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.start_pose_condition.to(dtype=torch.float32)
        return self

    def _expert_condition(
        self, timestep: Tensor, projected_state: Tensor | None = None,
        skill_code: Tensor | None = None, mode_latent: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        arch11_condition = super(WristSkillDeltaGoalSkillExpert, self)._expert_condition
        if end_pose is None:
            return arch11_condition(
                timestep, projected_state=projected_state, skill_code=skill_code,
                mode_latent=mode_latent,
            )
        end_xyz, start_xyz = self._split_goal(end_pose, "LIT_2 Expert")
        condition = arch11_condition(
            timestep, projected_state=projected_state, skill_code=skill_code,
            mode_latent=mode_latent, end_pose=end_xyz,
        )
        if not bool(torch.isfinite(start_xyz).all()):
            raise ValueError("LIT_2 skill-start xyz must be finite.")
        return condition + self.start_pose_condition(start_xyz.float()).to(condition.dtype)


class WristOnlyLIT1SkillExpert(
    _LITChunkEndStateMixin,
    _LITGoalFreeCondMixin,
    WristSkillStartEndGoalBridgeProprioSkillExpert,
):
    """Wrist LIT head; Expert gets skill, start/end XYZ, and bridge proprio."""


class BothLIT1SkillExpert(
    _LITChunkEndStateMixin,
    _BothCameraConditionMixin,
    _LITGoalFreeCondMixin,
    WristSkillStartEndGoalBridgeProprioSkillExpert,
):
    """Top+wrist counterpart of WristOnlyLIT1SkillExpert."""


class WristOnlyLIT2SkillExpert(
    _LITChunkEndStateMixin,
    _LITGoalFreeCondMixin,
    WristSkillStartEndGoalSkillExpert,
):
    """Wrist LIT head; Expert gets skill and start/end XYZ, without proprio."""


class BothLIT2SkillExpert(
    _LITChunkEndStateMixin,
    _BothCameraConditionMixin,
    _LITGoalFreeCondMixin,
    WristSkillStartEndGoalSkillExpert,
):
    """Top+wrist counterpart of WristOnlyLIT2SkillExpert."""


class WristOnlyLIT3SkillExpert(
    _LITChunkEndStateMixin,
    _LITGoalFreeCondMixin,
    WristCondSkillEndPoseLayerwiseCondBottleneckSkillExpert,
):
    """Wrist LIT head; Expert gets skill and end XYZ only."""


class BothLIT3SkillExpert(
    _LITChunkEndStateMixin,
    _BothCameraConditionMixin,
    _LITGoalFreeCondMixin,
    WristCondSkillEndPoseLayerwiseCondBottleneckSkillExpert,
):
    """Top+wrist counterpart of WristOnlyLIT3SkillExpert."""


class WristOnlyLIT4SkillExpert(
    _LITChunkEndStateMixin,
    WristSkillStartEndGoalBridgeProprioSkillExpert,
):
    """LIT_1 plus skill-end XYZ in the Cond-Gemma AdaRMS path."""


class BothLIT4SkillExpert(
    _LITChunkEndStateMixin,
    _BothCameraConditionMixin,
    WristSkillStartEndGoalBridgeProprioSkillExpert,
):
    """Top+wrist counterpart of WristOnlyLIT4SkillExpert."""


class WristOnlyLIT5SkillExpert(
    _LITChunkEndStateMixin,
    WristSkillStartEndGoalSkillExpert,
):
    """LIT_4 without current proprio in the terminal Expert bridge."""


class BothLIT5SkillExpert(
    _LITChunkEndStateMixin,
    _BothCameraConditionMixin,
    WristSkillStartEndGoalSkillExpert,
):
    """Top+wrist counterpart of WristOnlyLIT5SkillExpert."""


class BothLIT6SkillExpert(
    _LITChunkEndStateMixin,
    _BothCameraConditionMixin,
    _LITProprioOnlyCondMixin,
    WristSkillStartEndGoalSkillExpert,
):
    """Both_LIT_5 with proprio-only Cond-Gemma conditioning."""


class _WristPatchAlignedSkillExpert:
    """Training-only mixin: name the wrist patch that holds the skill-end EEF.

    Arch16--Arch18 see the wrist camera alone, and their goal reaches the Action Expert as three
    raw numbers. Nothing forces the visual stack to know where that goal *is* in the image, so it
    can settle for whatever texture drives the action head. This head asks the question directly:
    one query pooled from the bottleneck scores every patch of the condition sequence, and the
    answer is the patch the goal projects into (see ``wrist_patch_target``). The readout is
    discarded at inference -- only the features it shaped are deployed.

    The condition sequence is ``[wrist CLS, 196 wrist patches]`` in row-major order, so the patch
    features are ``condition_hidden[:, 1:]`` and the gradient reaches DINO through Cond-Gemma.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        self.wrist_patch_align_head = WristPatchAlignmentHead(
            int(config.visual_bottleneck_width), self.width
        )
        self._final_patch_tokens: Tensor | None = None
        self._final_patch_query: Tensor | None = None
        self._capture_patch_alignment = False

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.wrist_patch_align_head.to(dtype=torch.float32)
        return self

    def _on_final_condition_hidden(self, hidden: Tensor) -> None:
        super()._on_final_condition_hidden(hidden)
        self._final_patch_tokens = (
            hidden[:, 1:]
            if self.training or self._capture_patch_alignment
            else None
        )

    def _on_final_bottleneck_latent(self, latent: Tensor) -> None:
        super()._on_final_bottleneck_latent(latent)
        self._final_patch_query = (
            latent if self.training or self._capture_patch_alignment else None
        )

    def predict_training_wrist_patch_logits(self) -> Tensor:
        """``[batch, patches]`` logits over the wrist patch grid; consumes the stashed forward."""
        tokens, query = self._final_patch_tokens, self._final_patch_query
        self._final_patch_tokens = self._final_patch_query = None
        if tokens is None or query is None:
            raise RuntimeError(
                "The wrist patch readout requires a preceding training condition forward."
            )
        return self.wrist_patch_align_head(query, tokens)

    @torch.no_grad()
    def wrist_patch_alignment_diagnostics(
        self,
        condition_tokens: Tensor,
        condition_state: Tensor,
    ) -> dict[str, Tensor]:
        """Decompose the eval-time pooled heatmap into per-bottleneck contributions.

        The alignment head pools Q bottleneck tokens before its linear query
        projection. Because the projection is affine and the pooling weights sum
        to one, its final patch logits equal the sum of Q weighted per-query
        logits. This method exposes that exact decomposition without switching
        the policy to training mode.
        """
        if self.training:
            raise RuntimeError("Alignment diagnostics require model.eval().")
        self._capture_patch_alignment = True
        try:
            self._encode_layerwise_latents(condition_tokens, condition_state)
            tokens, query = self._final_patch_tokens, self._final_patch_query
        finally:
            self._capture_patch_alignment = False
            self._final_patch_tokens = self._final_patch_query = None
        if tokens is None or query is None:
            raise RuntimeError("Alignment diagnostic capture produced no hidden states.")

        diagnostics = decompose_wrist_patch_alignment(
            self.wrist_patch_align_head, query, tokens
        )
        direct_logits = self.wrist_patch_align_head(query, tokens)
        if not torch.allclose(
            diagnostics["pooled_logits"], direct_logits, atol=2e-5, rtol=2e-5
        ):
            raise RuntimeError("Per-query alignment decomposition does not reconstruct the head.")
        return {name: value.detach() for name, value in diagnostics.items()}


class WristSkillDeltaGoalAlignSkillExpert(
    _WristPatchAlignedSkillExpert, WristSkillDeltaGoalSkillExpert
):
    """Arch16_align: Arch16 plus the training-only wrist patch alignment head."""


class WristSkillDeltaGoalBridgeProprioAlignSkillExpert(
    _WristPatchAlignedSkillExpert, WristSkillDeltaGoalBridgeProprioSkillExpert
):
    """Arch17_align: Arch17 plus the training-only wrist patch alignment head."""


class WristSkillStartEndGoalBridgeProprioAlignSkillExpert(
    _WristPatchAlignedSkillExpert, WristSkillStartEndGoalBridgeProprioSkillExpert
):
    """Arch18_align: Arch18 plus the training-only wrist patch alignment head."""


class WristOnly1SkillExpert(
    _WristPatchAlignedSkillExpert,
    WristCondSkillEndPoseExpertSkillLayerwiseCondBottleneckSkillExpert,
):
    """WristOnly_1: wrist + proprio/skill/end-XYZ Cond, skill-only Expert.

    This is Arch12_1's routing with Arch18-align's spatial supervision: the
    Action Expert receives only the FSQ skill, while one training-only head
    asks the final bottleneck to locate the raw skill-end XYZ in the wrist
    patch grid.  The deployed action path has no alignment-head dependency.
    """


class _DedicatedAlignmentInterfaceMixin:
    """Reserve alignment queries while leaving the remaining queries action-driven.

    Dedicated queries and action queries use the same reader parameters but are
    updated in separate calls, so reader self-attention cannot leak the spatial
    auxiliary loss across the partition. The ordinary visual bridge sees only
    action queries. A separate zero-initialised gate lets the Expert use the
    aligned queries if action training finds them useful.
    """

    _alignment_token_count = 1

    def __init__(self, config: SkillExpertConfig):
        if int(config.visual_bottleneck_tokens) != 100:
            raise ValueError(
                f"{config.architecture_label} fixes visual_bottleneck_tokens=100 so its "
                f"alignment/action split remains {self._alignment_token_count}/"
                f"{100 - self._alignment_token_count}; got {config.visual_bottleneck_tokens}."
            )
        super().__init__(config)
        latent_width = int(config.visual_bottleneck_width)
        expert_depth = int(self.gemma_expert.model.config.num_hidden_layers)
        self.visual_align_bridge_attention = nn.MultiheadAttention(
            embed_dim=self.width,
            num_heads=int(config.visual_bridge_heads),
            kdim=latent_width,
            vdim=latent_width,
            batch_first=True,
        )
        self.visual_align_bridge_gates = nn.Parameter(torch.zeros(expert_depth))

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.visual_align_bridge_attention.to(dtype=torch.float32)
        self.visual_align_bridge_gates.data = self.visual_align_bridge_gates.data.float()
        if self.visual_align_bridge_gates.grad is not None:
            self.visual_align_bridge_gates.grad.data = (
                self.visual_align_bridge_gates.grad.data.float()
            )
        return self

    def _alignment_memories(self, memory: Tensor) -> tuple[Tensor, ...]:
        if self._alignment_token_count != 1:
            raise NotImplementedError
        return (memory,)

    def _update_layerwise_latent(
        self, layer_index: int, latent: Tensor, memory: Tensor
    ) -> Tensor:
        reader = self.layerwise_condition_readers[layer_index]
        memories = self._alignment_memories(memory)
        if len(memories) != self._alignment_token_count:
            raise RuntimeError(
                "Dedicated alignment memory count does not match its query count."
            )
        aligned = [
            reader(latent[:, index : index + 1], camera_memory)
            for index, camera_memory in enumerate(memories)
        ]
        action = reader(latent[:, self._alignment_token_count :], memory)
        return torch.cat([*aligned, action], dim=1)

    def _action_bridge_latent(self, layer_latent: Tensor) -> Tensor:
        return layer_latent[:, self._alignment_token_count :]

    def _aligned_bridge_residual(
        self,
        layer_index: int,
        bridge_query: Tensor,
        layer_latent: Tensor,
    ) -> Tensor:
        aligned = layer_latent[:, : self._alignment_token_count]
        output, _ = self.visual_align_bridge_attention(
            bridge_query,
            aligned,
            aligned,
            need_weights=False,
        )
        return self.visual_align_bridge_gates[layer_index].tanh() * output


class WristOnly2SkillExpert(
    _DedicatedAlignmentInterfaceMixin,
    _WristPatchAlignedSkillExpert,
    WristCondSkillEndPoseExpertSkillLayerwiseCondBottleneckSkillExpert,
):
    """WristOnly_2: one aligned query plus action-only recurrent queries."""

    def _on_final_bottleneck_latent(self, latent: Tensor) -> None:
        super()._on_final_bottleneck_latent(latent)
        if self.training or self._capture_patch_alignment:
            self._final_patch_query = latent[:, :1]


class _DualPatchAlignedSkillExpert:
    """Top/wrist counterpart of ``_WristPatchAlignedSkillExpert``.

    Both cameras pass through the shared DINO/Cond tower, but their patch
    tokens never mix inside the readout.  Independent heads pool the
    bottleneck queries independently and score only their own camera patches.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        query_width = int(config.visual_bottleneck_width)
        self.agent_patch_align_head = WristPatchAlignmentHead(query_width, self.width)
        self.wrist_patch_align_head = WristPatchAlignmentHead(query_width, self.width)
        self._final_agent_patch_tokens: Tensor | None = None
        self._final_wrist_patch_tokens: Tensor | None = None
        self._final_dual_patch_query: Tensor | None = None

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.agent_patch_align_head.to(dtype=torch.float32)
        self.wrist_patch_align_head.to(dtype=torch.float32)
        return self

    def _condition_tokens(
        self, images: list[Tensor], *, batch_size: int | None = None,
        skill_code: Tensor | None = None,
    ) -> Tensor:
        del batch_size, skill_code
        if len(images) != 2:
            raise ValueError(f"Both_1 requires [top, wrist] images, got {len(images)}.")
        camera_tokens = [
            self.image_proj(
                self._image_features(image).to(dtype=self.image_proj.weight.dtype)
            ).to(self.working_dtype)
            for image in images
        ]
        if camera_tokens[0].shape[1] != camera_tokens[1].shape[1]:
            raise ValueError(
                "Both_1 requires equal top/wrist patch grids, got token lengths "
                f"{camera_tokens[0].shape[1]} and {camera_tokens[1].shape[1]}."
            )
        return torch.cat(camera_tokens, dim=1)

    def _on_final_condition_hidden(self, hidden: Tensor) -> None:
        super()._on_final_condition_hidden(hidden)
        if not self.training:
            self._final_agent_patch_tokens = None
            self._final_wrist_patch_tokens = None
            return
        # [top CLS, top patches, wrist CLS, wrist patches]
        if hidden.shape[1] < 4 or (hidden.shape[1] - 2) % 2:
            raise RuntimeError(
                "Both_1 expected two equal [CLS + patch] camera sequences, got "
                f"{hidden.shape[1]} condition tokens."
            )
        patches = (hidden.shape[1] - 2) // 2
        self._final_agent_patch_tokens = hidden[:, 1 : 1 + patches]
        self._final_wrist_patch_tokens = hidden[:, 2 + patches :]

    def _on_final_bottleneck_latent(self, latent: Tensor) -> None:
        super()._on_final_bottleneck_latent(latent)
        self._final_dual_patch_query = latent if self.training else None

    def predict_training_camera_patch_logits(self) -> tuple[Tensor, Tensor]:
        """Return independent ``(top, wrist)`` patch logits and consume the stash."""
        top_tokens = self._final_agent_patch_tokens
        wrist_tokens = self._final_wrist_patch_tokens
        query = self._final_dual_patch_query
        self._final_agent_patch_tokens = None
        self._final_wrist_patch_tokens = None
        self._final_dual_patch_query = None
        if top_tokens is None or wrist_tokens is None or query is None:
            raise RuntimeError(
                "Both_1 patch readout requires a preceding training condition forward."
            )
        return (
            self.agent_patch_align_head(query, top_tokens),
            self.wrist_patch_align_head(query, wrist_tokens),
        )


class Both1SkillExpert(
    _DualPatchAlignedSkillExpert,
    WristCondSkillEndPoseExpertSkillLayerwiseCondBottleneckSkillExpert,
):
    """Both_1: top+wrist Cond with independent patch targets; skill-only Expert."""


class XYZSkillConditionedBottleneckUVExpertSkillDeltaSkillExpert(
    XYZSkillConditionedBottleneckUVExpertEndPoseSkillExpert
):
    """Arch19: Arch15 whose Action Expert goal is the skill displacement (Arch16-style).

    The policy packs ``end_pose = [skill-end xyz, skill-end xyz - skill-start xyz]`` exactly as for
    Arch16. Cond-Gemma keeps Arch15's input (proprio + absolute skill-end xyz + skill) and the
    bottleneck keeps its top-view focus UV head; only the Expert AdaRMS goal changes, through
    Arch14's ``end_pose_condition``, to the translation-invariant displacement. The deployed and
    the skill-only route share ``_expert_condition`` and therefore the same goal.
    """


    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        absolute_goal, _ = WristSkillDeltaGoalSkillExpert._split_goal(end_pose, "Arch19 Cond-Gemma")
        return super()._project_condition_state(state, focus_uv, skill_code, absolute_goal)

    def _expert_condition(
        self, timestep: Tensor, projected_state: Tensor | None = None,
        skill_code: Tensor | None = None, mode_latent: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        if end_pose is None:
            # Same contract as Arch14 for legacy goal-free skill-only sampling.
            displacement = None
        else:
            _, displacement = WristSkillDeltaGoalSkillExpert._split_goal(end_pose, "Arch19 Expert")
        return super()._expert_condition(
            timestep, projected_state=projected_state, skill_code=skill_code,
            mode_latent=mode_latent, end_pose=displacement,
        )


class Both2SkillExpert(
    _DedicatedAlignmentInterfaceMixin,
    _DualPatchAlignedSkillExpert,
    WristCondSkillEndPoseExpertSkillLayerwiseCondBottleneckSkillExpert,
):
    """Both_2: camera-specific aligned queries plus 98 action-driven queries."""

    _alignment_token_count = 2

    def _alignment_memories(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        if memory.shape[1] < 4 or memory.shape[1] % 2:
            raise RuntimeError(
                "Both_2 expected two equal [CLS + patch] camera memories, got "
                f"{memory.shape[1]} tokens."
            )
        camera_tokens = memory.shape[1] // 2
        return memory[:, :camera_tokens], memory[:, camera_tokens:]

    def _on_final_bottleneck_latent(self, latent: Tensor) -> None:
        super()._on_final_bottleneck_latent(latent)
        self._final_dual_patch_query = latent[:, :2] if self.training else None

    def predict_training_camera_patch_logits(self) -> tuple[Tensor, Tensor]:
        top_tokens = self._final_agent_patch_tokens
        wrist_tokens = self._final_wrist_patch_tokens
        query = self._final_dual_patch_query
        self._final_agent_patch_tokens = None
        self._final_wrist_patch_tokens = None
        self._final_dual_patch_query = None
        if top_tokens is None or wrist_tokens is None or query is None:
            raise RuntimeError(
                "Both_2 patch readout requires a preceding training condition forward."
            )
        return (
            self.agent_patch_align_head(query[:, :1], top_tokens),
            self.wrist_patch_align_head(query[:, 1:2], wrist_tokens),
        )


class XYZSkillConditionedBottleneckUVSkillExpert(XYZConditionedBottleneckUVSkillExpert):
    """Arch20: Arch15 without any Expert goal, i.e. Arch13 plus the skill in the Cond-Gemma AdaRMS.

    Cond-Gemma receives proprio, the skill-end xyz and the skill exactly as in Arch15, and the
    bottleneck predicts the same top-view focus UV. The Expert AdaRMS keeps Arch13's goal-free
    input, so the Expert is conditioned on the skill alone (through the Expert skill broadcast) and
    the skill-only route needs no pose at all.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__(config)
        # Same zero-initialised projection as Arch15, so training starts from Arch13's signal.
        self.cond_skill_condition = nn.Sequential(
            nn.Linear(len(config.skill_fsq_levels), self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        nn.init.zeros_(self.cond_skill_condition[-1].weight)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.cond_skill_condition.to(dtype=torch.float32)
        return self

    def _project_condition_state(
        self, state: Tensor | None, focus_uv: Tensor | None = None,
        skill_code: Tensor | None = None,
        end_pose: Tensor | None = None,
    ) -> Tensor:
        projected = super()._project_condition_state(state, focus_uv, skill_code, end_pose)
        if skill_code is None:
            raise ValueError("Arch20 requires the skill for Cond-Gemma AdaRMS.")
        coordinates = self._code_to_zq(skill_code)
        projected_skill = self.cond_skill_condition(coordinates.float())
        return projected + projected_skill.to(projected.dtype)
