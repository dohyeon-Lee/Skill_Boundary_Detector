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
from .configuration_skill_expert import SkillExpertConfig


class _LayerwiseConditionReader(nn.Module):
    """One residual latent update from a single Cond-Gemma layer."""

    def __init__(self, latent_width: int, condition_width: int, heads: int):
        super().__init__()
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
        update, _ = self.cross_attention(
            query,
            condition_hidden,
            condition_hidden,
            need_weights=False,
        )
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
        latent = self.layerwise_condition_readers[layer_index](latent, memory)
        return condition_hidden, latent

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
            bridge_output, _ = self.visual_bridge_attention(
                self.visual_bridge_query_norm(hidden.float()),
                layer_latent,
                layer_latent,
                need_weights=False,
            )
            gate_value = self.visual_bridge_gates[layer_index].tanh()
            hidden = (
                hidden.float() + gate_value * bridge_output
            ).to(self.working_dtype)

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
    ) -> Tensor:
        """Cache the recurrent visual interface once per generated chunk."""
        projected_state = self._project_state(state)
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
