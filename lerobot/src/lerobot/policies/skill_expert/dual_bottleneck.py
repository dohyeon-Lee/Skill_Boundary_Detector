"""Two current-observation bottlenecks over a single pretrained VLM forward.

The shared Action Expert learns an independent fullskill flow point, read out
after L9 by default or L18 on opt-in. Action inference skips that denoising chain.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from lerobot.configs.uv_fullskill import DUAL_BOTTLENECK_LAST_LABEL, DUAL_BOTTLENECK_NOPROPRIO_LAST_LABEL
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
        if config.dual_omits_current_expert_proprio:
            self.bridge_proprio_condition = nn.Identity()
        if config.dual_uses_start_proprio:
            self.start_proprio_condition = nn.Sequential(
                nn.Linear(getattr(config, "skill_plan_state_dim", 8), self.width), nn.SiLU(), nn.Linear(self.width, self.width, bias=False),
            )
            nn.init.zeros_(self.start_proprio_condition[-1].weight)
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
        plan_state_dim = int(getattr(config, "skill_plan_state_dim", 8))
        self.plan_state_dim = plan_state_dim
        self.bspline_noise_input = nn.Linear(plan_state_dim, self.width)
        self.bspline_source_norm = nn.LayerNorm(self.width)
        self.bspline_plan_projection = nn.Linear(self.width, width)
        self.bspline_proprio_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, plan_state_dim)
        )
        self._last_lower_latent = self._final_lower_aux_query = None
        self._fullskill_flow_loss = None

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for name, module in self.named_children():
            if name.startswith(("lower_", "bspline_", "start_proprio_")):
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
        action = reader.read_condition(lower[:, self._alignment_token_count:], memory)
        lower = reader.mix_latents(torch.cat((*aligned, action), dim=1))
        return torch.cat((lower, upper), dim=1)

    def _on_final_bottleneck_latent(self, packed):
        lower, upper = packed.chunk(2, dim=1)
        # Keep the historical cache/API for visualization. The final readout
        # needs both towers, while the default prefix only needs the lower one.
        self._last_lower_latent = packed if self.config.dual_fullskill_readout == "final" else lower
        self._final_lower_aux_query = (
            lower[:, :self._alignment_token_count] if self.training else None
        )
        super()._on_final_bottleneck_latent(upper)
        from .bottleneck_video import capture_latent
        capture_latent(self, lower, tower="lower", prefix="lower_")

    def _split_uv_goal(self, goal):
        if getattr(self.config, "bimanual_spatial_auxiliary", False):
            if goal is None or goal.ndim != 2 or goal.shape[1] != 6:
                raise ValueError("ABC top-UV goal must be [left UV2, right UV2, valid2].")
            if not bool(torch.isfinite(goal).all()):
                raise ValueError("ABC top-UV packed goal must be finite.")
            return goal[:, :4].float(), goal[:, 4:6].float(), goal.new_zeros(goal.shape[0], 3)
        if self.config.dual_uses_start_proprio:
            if goal is None or goal.ndim != 2 or goal.shape[1] != 11:
                raise ValueError("This dual variant requires [UV2, valid1, normalized skill-start proprio8].")
            if not bool(torch.isfinite(goal).all()):
                raise ValueError("Dual skill-start proprio goal must be finite.")
            # The inherited VLM prefix consumes only UV/valid. No start XYZ adapter.
            return goal[:, :2].float(), goal[:, 2:3].float(), goal.new_zeros(goal.shape[0], 3)
        return super()._split_uv_goal(goal)

    def _project_condition_state(self, state, focus_uv=None, skill_code=None, end_pose=None):
        if not self.config.dual_omits_current_expert_proprio:
            return super()._project_condition_state(state, focus_uv, skill_code, end_pose)
        if state is None or skill_code is None:
            raise ValueError("The VLM still requires current proprio and skill.")
        self._current_proprio_condition = self._bridge_condition_shift = None
        uv, valid, _ = self._split_uv_goal(end_pose)
        prompt_state = (
            state[:, : int(getattr(self.config, "skill_plan_state_dim", state.shape[-1]))]
            if getattr(self.config, "bimanual_spatial_auxiliary", False)
            else state
        )
        return prompt_state, self._code_to_zq(skill_code), uv, valid

    def _expert_condition(self, timestep, projected_state=None, skill_code=None,
                          mode_latent=None, end_pose=None):
        uv, valid, _ = self._split_uv_goal(end_pose)
        self._split_time_condition = self._time_condition(timestep)
        condition = self._split_time_condition
        if not self.config.dual_omits_current_expert_proprio:
            if self._current_proprio_condition is None:
                raise RuntimeError("Dual bottleneck requires current proprio before Expert execution.")
            condition = condition + self._current_proprio_condition.to(self.working_dtype)
        if getattr(self.config, "bimanual_spatial_auxiliary", False):
            safe_uv = uv * valid.repeat_interleave(2, dim=-1)
            goal_mask = valid.any(dim=-1, keepdim=True).to(uv.dtype)
            condition = condition + (
                self.end_pose_condition(safe_uv) * goal_mask
            ).to(self.working_dtype)
        else:
            condition = condition + (self.end_pose_condition(uv) * valid).to(self.working_dtype)
        if self.config.dual_uses_start_proprio:
            condition = condition + self.start_proprio_condition(end_pose[:, 3:11].float()).to(self.working_dtype)
        return condition

    def _action_layer_condition(self, layer_index, expert_condition):
        if layer_index < self.split_layer:
            return expert_condition
        if self._split_time_condition is None:
            raise RuntimeError("Missing action flow time.")
        return self._split_time_condition

    def _layer_uses_visual_bridge(self, layer_index):
        if self.config.dual_uses_start_proprio:
            return self.split_layer <= int(layer_index) < len(self.gemma_expert.model.layers)
        if self.config.architecture_label in (DUAL_BOTTLENECK_LAST_LABEL, DUAL_BOTTLENECK_NOPROPRIO_LAST_LABEL):
            # Same 9+9 split and recurrent readers, but only the final layer
            # of each half reads its bottleneck (also in the auxiliary prefix).
            return int(layer_index) in (self.split_layer - 1, len(self.gemma_expert.model.layers) - 1)
        return 0 <= int(layer_index) < len(self.gemma_expert.model.layers)

    @property
    def visual_bridge_start_layer(self):
        return 0

    def _active_visual_bridge_gates(self):
        if self.config.dual_uses_start_proprio:
            return self.visual_bridge_gates[self.split_layer:]
        if self.config.architecture_label in (DUAL_BOTTLENECK_LAST_LABEL, DUAL_BOTTLENECK_NOPROPRIO_LAST_LABEL):
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
        aligned = latent[:, :self._alignment_token_count]
        alignment, _ = self.lower_visual_align_bridge_attention(query, aligned, aligned, need_weights=False)
        return (self.lower_visual_bridge_gates[layer_index].tanh() * output
                + self.lower_visual_align_bridge_gates[layer_index].tanh() * alignment)

    def _run_expert_with_layerwise_latents(self, noisy_actions, expert_condition,
                                         expert_skill, layer_latents, *, return_all_layers=False):
        lower, upper = layer_latents[-1].chunk(2, dim=1)
        if self.config.dual_bottleneck_layerwise:
            # Keep all depths for the auxiliary flow as well as the deployed
            # action route. The final visual heads
            # continue to supervise only the final bottleneck queries.
            self._last_lower_latent = torch.stack(
                [packed if self.config.dual_fullskill_readout == "final" else packed.chunk(2, dim=1)[0]
                 for packed in layer_latents], dim=1,
            )
        hidden = self._action_tokens(noisy_actions)
        mask, positions, rotary = self._sequence_geometry(hidden, self.gemma_expert.model)
        late_condition = self._action_layer_condition(self.split_layer, expert_condition)
        all_hidden = []
        depth = len(self.gemma_expert.model.layers)
        for index in range(depth):
            early = index < self.split_layer
            if self.config.dual_bottleneck_layerwise:
                lower, upper = layer_latents[index].chunk(2, dim=1)
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
            late_condition = self._action_layer_condition(self.split_layer, condition)
        finally:
            self._split_time_condition = previous_time
        _, skill = self._skill_broadcasts(skill_code)
        hidden = inputs.to(self.working_dtype)
        mask, positions, rotary = self._sequence_geometry(hidden, self.gemma_expert.model)
        final_readout = self.config.dual_fullskill_readout == "final"
        depth = len(self.gemma_expert.model.layers) if final_readout else self.split_layer
        for index in range(depth):
            early = index < self.split_layer
            latent = lower_latent[:, index] if lower_latent.ndim == 4 else lower_latent
            if final_readout:
                lower, upper = latent.chunk(2, dim=1)
                latent = lower if early else upper
            args = (index, hidden, mask, positions, condition if early else late_condition,
                    skill if early else None, latent, rotary)
            if self._checkpoint_layer("skill_planner", index, depth):
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
        if getattr(self.config, "bimanual_spatial_auxiliary", False):
            query = self._final_lower_aux_query
            top = self._final_agent_patch_tokens
            left = self._final_wrist_patch_tokens
            right = self._final_right_wrist_patch_tokens
            self._final_lower_aux_query = None
            if query is None or top is None or left is None or right is None:
                raise RuntimeError("ABC dual auxiliary heads require a training observation forward.")
            # Keep local references before the upper readout consumes the
            # shared camera-token caches.
            upper = self.predict_training_bimanual_patch_and_xyz(include_xyz=include_xyz)
            lower = (
                self.lower_agent_patch_align_head(query[:, 0:1], top),
                self.lower_agent_patch_align_head(query[:, 1:2], top),
                self.lower_wrist_patch_align_head(query[:, 2:3], left),
                self.lower_wrist_patch_align_head(query[:, 3:4], right),
                self.lower_agent_end_state_head(query[:, 4].float()) if include_xyz else None,
            )
            return upper, lower
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

    def predict_training_bimanual_patch_and_xyz(self, *, include_xyz=True):
        query = self._final_dual_patch_query
        top = self._final_agent_patch_tokens
        left = self._final_wrist_patch_tokens
        right = self._final_right_wrist_patch_tokens
        self._final_dual_patch_query = None
        self._final_agent_patch_tokens = None
        self._final_wrist_patch_tokens = None
        self._final_right_wrist_patch_tokens = None
        if query is None or top is None or left is None or right is None:
            raise RuntimeError("ABC bimanual readout requires a preceding training VLM forward.")
        return (
            self.agent_patch_align_head(query[:, 0:1], top),
            self.agent_patch_align_head(query[:, 1:2], top),
            self.wrist_patch_align_head(query[:, 2:3], left),
            self.wrist_patch_align_head(query[:, 3:4], right),
            self.agent_end_state_head(query[:, 4].float()) if include_xyz else None,
        )

    def _input_sensitivity_stats(self, **kwargs):
        saved = self._last_lower_latent, self._final_lower_aux_query
        try:
            return super()._input_sensitivity_stats(**kwargs)
        finally:
            self._last_lower_latent, self._final_lower_aux_query = saved

    def generate_visualization_plan(self, skill_code, end_pose, *, start_latent=None, batch_indices=None):
        # Explicit opt-in visualization only; ordinary inference has no FM work.
        latent = self._last_lower_latent if start_latent is None else start_latent
        if latent is None:
            raise RuntimeError("Encode the current observation before visualizing the auxiliary plan.")
        previous_proprio = self._current_proprio_condition
        try:
            # Codes/latents are already selected by the video caller; the shared
            # proprio cache still describes the entire action batch.
            if batch_indices is not None and previous_proprio is not None:
                self._current_proprio_condition = previous_proprio[batch_indices]
            return self._sample_denoising_plan(skill_code, end_pose, latent)[1]
        finally:
            self._current_proprio_condition = previous_proprio
