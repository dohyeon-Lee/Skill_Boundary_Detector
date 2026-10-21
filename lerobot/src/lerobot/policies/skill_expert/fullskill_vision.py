"""LIT7 full-skill motion core with a separate, held skill-start visual bridge."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lerobot.configs.uv_fullskill import bspline_base_label

from .layerwise_cond_bottleneck import BothLIT7SkillExpert


class FullskillVisSkillExpert(BothLIT7SkillExpert):
    """Held start Z18 -> one prefix layer; live vision -> selected late layers.

    Both observations use the SAME DINO/Cond/recurrent bottleneck weights.
    V2 bridges both read 98 ordinary queries plus a separate gated 2-query
    alignment path. Start/current bridges have independent parameters. Patch
    heads are shared and supervise both observations. V1 checkpoints retain
    their original 98-query-only start bridge and current-only patch loss.
    A prepared start latent is reused by main and full-skill flows. Each
    checkpointed layer receives that tensor explicitly, so recomputation never
    reads a subsequently replaced mutable start cache.
    """

    def __init__(self, config):
        super().__init__(config)
        label = bspline_base_label(config.architecture_label).removesuffix("_preskill").removesuffix("_none").removesuffix("_current")
        self.split_layer = self.visual_bridge_start_layer
        depth = int(self.gemma_expert.model.config.num_hidden_layers)
        self.start_bridge_layer = (
            self.split_layer - 1
            if label.split("_lastvis", 1)[0].endswith("_lastlayer")
            else 0
        )
        if label.endswith("_lastvis_last"):
            self.current_vision_route = "final_last"
            self.current_bridge_layers = (depth - 1,)
        elif label.endswith("_lastvis_first"):
            self.current_vision_route = "final_first"
            self.current_bridge_layers = (self.split_layer,)
        else:
            self.current_vision_route = (
                "final_all" if label.endswith("_lastvis") else "layerwise"
            )
            self.current_bridge_layers = tuple(range(self.split_layer, depth))
        self.direct_current_proprio = label.startswith("fullskill_vis_2_")
        self.symmetric_start_vision = bool(getattr(config, "uses_symmetric_start_vision", False))
        self.start_bridge_query_norm = nn.LayerNorm(self.width)
        self.start_bridge_attention = nn.MultiheadAttention(
            self.width, int(config.visual_bridge_heads),
            kdim=int(config.visual_bottleneck_width),
            vdim=int(config.visual_bottleneck_width), batch_first=True,
        )
        self.start_bridge_gate = nn.Parameter(torch.tensor(float(config.visual_bridge_gate_init)))
        if self.symmetric_start_vision:
            self.start_align_bridge_attention = nn.MultiheadAttention(
                self.width, int(config.visual_bridge_heads),
                kdim=int(config.visual_bottleneck_width),
                vdim=int(config.visual_bottleneck_width), batch_first=True,
            )
            # Same initialization as the current alignment reader.
            self.start_align_bridge_gate = nn.Parameter(torch.zeros(()))
        self._skill_start_latent: Tensor | None = None
        self._bridge_condition_shift: Tensor | None = None
        if self.direct_current_proprio:
            # Same adapter/initialization as Both3, applied to every late layer
            # independently of its vision read. The prefix stays current-state-free.
            self.bridge_proprio_condition = nn.Sequential(
                nn.Linear(config.max_state_dim, self.width), nn.SiLU(),
                nn.Linear(self.width, self.width, bias=False),
            )
            nn.init.zeros_(self.bridge_proprio_condition[-1].weight)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.start_bridge_query_norm.to(dtype=torch.float32)
        self.start_bridge_attention.to(dtype=torch.float32)
        self.start_bridge_gate.data = self.start_bridge_gate.data.float()
        if self.start_bridge_gate.grad is not None:
            self.start_bridge_gate.grad.data = self.start_bridge_gate.grad.data.float()
        if self.symmetric_start_vision:
            self.start_align_bridge_attention.to(dtype=torch.float32)
            self.start_align_bridge_gate.data = self.start_align_bridge_gate.data.float()
            if self.start_align_bridge_gate.grad is not None:
                self.start_align_bridge_gate.grad.data = self.start_align_bridge_gate.grad.data.float()
        if self.direct_current_proprio:
            self.bridge_proprio_condition.to(dtype=torch.float32)
        return self

    def _project_condition_state(self, state, focus_uv=None, skill_code=None, end_pose=None):
        if self.direct_current_proprio:
            if state is None:
                raise ValueError("fullskill_vis_2 requires current proprio for the late Expert layers.")
            self._bridge_condition_shift = self.bridge_proprio_condition(state.float())
        return super()._project_condition_state(state, focus_uv, skill_code, end_pose)

    def _action_layer_condition(self, layer_index, expert_condition):
        if not self.direct_current_proprio or int(layer_index) < self.split_layer:
            return expert_condition
        shift = self._bridge_condition_shift
        if shift is None or shift.shape != expert_condition.shape:
            raise RuntimeError("fullskill_vis_2 late layer ran without its current proprio condition.")
        # Materialize BEFORE checkpointing: recomputation must not consult a
        # mutable shift overwritten by a later start/diagnostic/skill forward.
        return expert_condition + shift.to(expert_condition.dtype)

    def _input_sensitivity_stats(self, **kwargs):
        shift = self._bridge_condition_shift
        try:
            return super()._input_sensitivity_stats(**kwargs)
        finally:
            self._bridge_condition_shift = shift

    def encode_skill_start(self, images, state, skill_code, end_pose) -> Tensor:
        """Encode start-time observations once, before the current condition pass."""
        tokens = self._condition_tokens(images, batch_size=state.shape[0], skill_code=skill_code)
        condition = self._project_condition_state(state, None, skill_code, end_pose)
        return self._encode_layerwise_latents(tokens, condition)[-1]

    def set_skill_start_latent(self, latent: Tensor | None):
        self._skill_start_latent = latent

    def _required_start_latent(self) -> Tensor:
        if self._skill_start_latent is None:
            raise RuntimeError("Fullskill visual models need their skill-start observation before generating actions.")
        return self._skill_start_latent

    def _layer_uses_visual_bridge(self, layer_index):
        return int(layer_index) == self.start_bridge_layer or int(layer_index) in self.current_bridge_layers

    def _active_visual_bridge_gates(self):
        # Start vision has its own gate; inactive late-layer gates are not reads.
        return self.visual_bridge_gates[list(self.current_bridge_layers)]

    def _visual_bridge_update(self, layer_index, hidden, layer_latent):
        if int(layer_index) != self.start_bridge_layer:
            return super()._visual_bridge_update(layer_index, hidden, layer_latent)
        query = self.start_bridge_query_norm(hidden.float())
        memory = self._action_bridge_latent(layer_latent).float()
        update, _ = self.start_bridge_attention(query, memory, memory, need_weights=False)
        residual = self.start_bridge_gate.tanh() * update
        if self.symmetric_start_vision:
            aligned = layer_latent[:, :self._alignment_token_count].float()
            align_update, _ = self.start_align_bridge_attention(
                query, aligned, aligned, need_weights=False,
            )
            residual = residual + self.start_align_bridge_gate.tanh() * align_update
        return residual

    def _skill_only_layer_latent(self, layer_index):
        return self._required_start_latent() if int(layer_index) == self.start_bridge_layer else None

    def _run_expert_with_layerwise_latents(self, noisy_actions, expert_condition,
                                         expert_skill, layer_latents, **kwargs):
        latents = list(layer_latents)
        if self.current_vision_route != "layerwise":
            for index in self.current_bridge_layers:
                latents[index] = layer_latents[-1]
        latents[self.start_bridge_layer] = self._required_start_latent()
        return super()._run_expert_with_layerwise_latents(
            noisy_actions, expert_condition, expert_skill, latents, **kwargs,
        )

    def _record_visual_debug(self, condition_tokens):
        super()._record_visual_debug(condition_tokens)
        if self._vsa_debug_active:
            self._last_vsa_debug_stats["start_vision/bridge_gate"] = float(
                self.start_bridge_gate.detach().float().tanh().item()
            )
            self._last_vsa_debug_stats["start_vision/expert_layer"] = self.start_bridge_layer + 1
            if self.symmetric_start_vision:
                self._last_vsa_debug_stats["start_vision/align_bridge_gate"] = float(
                    self.start_align_bridge_gate.detach().float().tanh().item()
                )


# Import/checkpoint tooling written for the original first-layer model can keep
# using this class name. The persisted architecture label selects the variant.
FullskillVis1SkillExpert = FullskillVisSkillExpert
