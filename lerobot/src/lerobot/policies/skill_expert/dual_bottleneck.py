"""Two current-observation bottlenecks over a single pretrained VLM forward.

The shared Action Expert prefix also learns one independent 30-token fullskill
flow point. Action inference never constructs that auxiliary denoising chain.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from lerobot.configs.uv_fullskill import DUAL_BOTTLENECK_LAST_LABEL
from lerobot.policies.pi05.modeling_pi05 import layernorm_forward
from .independent_skill_planner import IndependentSkillPlannerExpert
from .layerwise_cond_bottleneck import _LayerwiseConditionReader
from .pi05_vlm_vision import Pi05VLMVanilaSkillExpert


class DualBottleneckSkillExpert(Pi05VLMVanilaSkillExpert):
    def __init__(self, config):
        super().__init__(config)
        if self.split_layer != 9 or len(self.gemma_expert.model.layers) != 18:
            raise ValueError("Dual bottleneck fixes the Expert split at 9+9.")
        width = int(config.visual_bottleneck_width)
        self.start_pose_condition = nn.Identity()  # No trainable skill-start XYZ path.
        self.lower_layerwise_latent_queries = nn.Parameter(
            torch.randn(config.visual_bottleneck_tokens, width) * .02
        )
        self.lower_layerwise_condition_readers = nn.ModuleList([
            _LayerwiseConditionReader(width, self.cond_encoder.model.config.hidden_size,
                                      config.visual_bottleneck_heads)
            for _ in self.cond_encoder.model.layers
        ])
        # Independent bridges/heads as well as independent recurrent readers.
        for name in ("visual_bridge_query_norm", "visual_bridge_attention", "visual_bridge_gates",
                     "visual_align_bridge_attention", "visual_align_bridge_gates",
                     "agent_patch_align_head", "wrist_patch_align_head", "agent_end_state_head"):
            setattr(self, "lower_" + name, copy.deepcopy(getattr(self, name)))
        self.bspline_noise_input = nn.Linear(8, self.width)
        self.bspline_source_norm = nn.LayerNorm(self.width)
        self.bspline_plan_projection = nn.Linear(self.width, width)
        self.bspline_proprio_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 8))
        self._last_lower_latent = self._final_lower_aux_query = None
        self._fullskill_flow_loss = None

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for name, module in self.named_children():
            if name.startswith(("lower_", "bspline_")):
                module.to(dtype=torch.float32)
        for name, parameter in self.named_parameters(recurse=False):
            if name.startswith("lower_"):
                parameter.data = parameter.data.float()
                if parameter.grad is not None:
                    parameter.grad.data = parameter.grad.data.float()
        return self

    def _initial_condition_latent(self, batch_size):
        upper = super()._initial_condition_latent(batch_size)
        lower = self.lower_layerwise_latent_queries[None].expand(batch_size, -1, -1)
        return torch.cat((lower, upper), dim=1)

    def _update_layerwise_latent(self, index, packed, memory):
        lower, upper = packed.chunk(2, dim=1)
        # Gemma and its normalized image memory are computed once. Only the
        # small recurrent readers differ; checkpointing replays them together.
        upper = super()._update_layerwise_latent(index, upper, memory)
        reader = self.lower_layerwise_condition_readers[index]
        aligned = [reader.read_condition(lower[:, i:i + 1], camera)
                   for i, camera in enumerate(self._alignment_memories(memory))]
        action = reader.read_condition(lower[:, 3:], memory)
        lower = reader.mix_latents(torch.cat((*aligned, action), dim=1))
        return torch.cat((lower, upper), dim=1)

    def _on_final_bottleneck_latent(self, packed):
        lower, upper = packed.chunk(2, dim=1)
        self._last_lower_latent = lower
        self._final_lower_aux_query = lower[:, :3] if self.training else None
        super()._on_final_bottleneck_latent(upper)

    def _expert_condition(self, timestep, projected_state=None, skill_code=None,
                          mode_latent=None, end_pose=None):
        uv, valid, _ = self._split_uv_goal(end_pose)
        if self._current_proprio_condition is None:
            raise RuntimeError("Dual bottleneck requires current proprio before Expert execution.")
        self._split_time_condition = self._time_condition(timestep)
        return (self._split_time_condition
                + self._current_proprio_condition.to(self.working_dtype)
                + (self.end_pose_condition(uv) * valid).to(self.working_dtype))

    def _action_layer_condition(self, layer_index, expert_condition):
        if layer_index < self.split_layer:
            return expert_condition
        if self._split_time_condition is None:
            raise RuntimeError("Missing action flow time.")
        return self._split_time_condition

    def _layer_uses_visual_bridge(self, layer_index):
        if self.config.architecture_label == DUAL_BOTTLENECK_LAST_LABEL:
            # Same 9+9 split and recurrent readers, but only the final layer
            # of each half reads its bottleneck (also in the auxiliary prefix).
            return int(layer_index) in (self.split_layer - 1, len(self.gemma_expert.model.layers) - 1)
        return 0 <= int(layer_index) < len(self.gemma_expert.model.layers)

    @property
    def visual_bridge_start_layer(self):
        return 0

    def _active_visual_bridge_gates(self):
        if self.config.architecture_label == DUAL_BOTTLENECK_LAST_LABEL:
            return torch.cat((self.lower_visual_bridge_gates[self.split_layer - 1:self.split_layer],
                              self.visual_bridge_gates[-1:]))
        return torch.cat((self.lower_visual_bridge_gates[:self.split_layer],
                          self.visual_bridge_gates[self.split_layer:]))

    def _visual_bridge_update(self, layer_index, hidden, latent):
        if layer_index >= self.split_layer:
            return super()._visual_bridge_update(layer_index, hidden, latent)
        query = self.lower_visual_bridge_query_norm(hidden.float())
        action = self._action_bridge_latent(latent)
        output, weights = self.lower_visual_bridge_attention(
            query, action, action, need_weights=self._capture_visual_bridge_attention,
            average_attn_weights=False,
        )
        if self._capture_visual_bridge_attention:
            self._captured_visual_bridge_attention[layer_index] = weights
        aligned = latent[:, :3]
        alignment, _ = self.lower_visual_align_bridge_attention(query, aligned, aligned, need_weights=False)
        return (self.lower_visual_bridge_gates[layer_index].tanh() * output
                + self.lower_visual_align_bridge_gates[layer_index].tanh() * alignment)

    def _run_expert_with_layerwise_latents(self, noisy_actions, expert_condition,
                                         expert_skill, layer_latents, *, return_all_layers=False):
        lower, upper = layer_latents[-1].chunk(2, dim=1)
        hidden = self._action_tokens(noisy_actions)
        mask, positions, rotary = self._sequence_geometry(hidden, self.gemma_expert.model)
        late_condition = self._action_layer_condition(self.split_layer, expert_condition)
        all_hidden = []
        depth = len(self.gemma_expert.model.layers)
        for index in range(depth):
            early = index < self.split_layer
            condition = expert_condition if early else late_condition
            args = (index, hidden, mask, positions, condition,
                    expert_skill if early else None, lower if early else upper, rotary)
            if self._checkpoint_layer("action_expert", index, depth):
                hidden = torch.utils.checkpoint.checkpoint(
                    self._expert_layer_with_latent_bridge, *args,
                    use_reentrant=False, preserve_rng_state=True,
                )
            else:
                hidden = self._expert_layer_with_latent_bridge(*args)
            if return_all_layers:
                all_hidden.append(layernorm_forward(self.gemma_expert.model.norm, hidden, condition)[0])
        hidden = layernorm_forward(self.gemma_expert.model.norm, hidden, late_condition)[0]
        return (hidden, torch.stack(all_hidden, dim=1)) if return_all_layers else hidden

    def _planner_forward(self, inputs, time, skill_code, end_pose, lower_latent):
        previous_time = self._split_time_condition
        try:
            condition = self._expert_condition(time, skill_code=skill_code, end_pose=end_pose)
        finally:
            self._split_time_condition = previous_time
        _, skill = self._skill_broadcasts(skill_code)
        hidden = inputs.to(self.working_dtype)
        mask, positions, rotary = self._sequence_geometry(hidden, self.gemma_expert.model)
        for index in range(self.split_layer):
            args = (index, hidden, mask, positions, condition, skill, lower_latent, rotary)
            if self._checkpoint_layer("skill_planner", index, self.split_layer):
                hidden = torch.utils.checkpoint.checkpoint(
                    self._expert_layer_with_latent_bridge, *args,
                    use_reentrant=False, preserve_rng_state=True,
                )
            else:
                hidden = self._expert_layer_with_latent_bridge(*args)
        return self.bspline_plan_projection(self.bspline_source_norm(hidden.float()))

    _flow_readout = IndependentSkillPlannerExpert._flow_readout
    _flow_supervision = IndependentSkillPlannerExpert._flow_supervision
    _sample_denoising_plan = IndependentSkillPlannerExpert._sample_denoising_plan

    def forward(self, *args, fullskill_target=None, **kwargs):
        self._fullskill_flow_loss = None
        try:
            residual = super().forward(*args, **kwargs)
            if fullskill_target is not None:
                skill_code = args[2] if len(args) > 2 else kwargs["skill_code"]
                self._fullskill_flow_loss = self._flow_supervision(
                    fullskill_target, skill_code, kwargs.get("end_pose"), self._last_lower_latent,
                )
            return residual
        finally:
            # Auxiliary heads retain only their three queries. Do not cache
            # the whole training observation graph between iterations.
            self._last_lower_latent = None

    def take_fullskill_flow_loss(self):
        loss, self._fullskill_flow_loss = self._fullskill_flow_loss, None
        return loss

    def predict_training_dual_auxiliaries(self, *, include_xyz=True):
        query = self._final_lower_aux_query
        top, wrist = self._final_agent_patch_tokens, self._final_wrist_patch_tokens
        self._final_lower_aux_query = None
        if query is None or top is None or wrist is None:
            raise RuntimeError("Dual auxiliary heads require a training observation forward.")
        upper = self.predict_training_top_wrist_patch_and_xyz(include_xyz=include_xyz)
        lower = (self.lower_agent_patch_align_head(query[:, :1], top),
                 self.lower_wrist_patch_align_head(query[:, 1:2], wrist),
                 self.lower_agent_end_state_head(query[:, 2].float()) if include_xyz else None)
        return upper, lower

    def _input_sensitivity_stats(self, **kwargs):
        saved = self._last_lower_latent, self._final_lower_aux_query
        try:
            return super()._input_sensitivity_stats(**kwargs)
        finally:
            self._last_lower_latent, self._final_lower_aux_query = saved

    def generate_visualization_plan(self, skill_code, end_pose, *, start_latent=None):
        # Explicit opt-in visualization only; ordinary inference has no FM work.
        latent = self._last_lower_latent if start_latent is None else start_latent
        if latent is None:
            raise RuntimeError("Encode the current observation before visualizing the auxiliary plan.")
        return self._sample_denoising_plan(skill_code, end_pose, latent)[1]
