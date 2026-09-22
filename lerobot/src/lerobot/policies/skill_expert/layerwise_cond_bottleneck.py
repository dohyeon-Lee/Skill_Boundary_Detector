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
from .configuration_skill_expert import (
    LAYERWISE_COND_BOTTLENECK_UV_COND_XYZ_TERMINATION_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_SKILL_END_POSE_TERMINATION_REVISION,
    SkillExpertConfig,
)


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
