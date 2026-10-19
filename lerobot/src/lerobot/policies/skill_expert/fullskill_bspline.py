"""H-token flow with a supervised, fixed-size plan at the Expert split.

The plan is regenerated at each flow time after the prefix Expert layers.
The final ``visual_bridge_last_n_layers`` layers read current vision; only
their first layer reads the plan. Held start vision enters the first or last
prefix Expert layer, independently of where the current-vision suffix starts.
It is not an additional noisy-action sequence or an independently sampled flow.
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint
from torch import nn

from lerobot.policies.pi05.modeling_pi05 import layernorm_forward

from .fullskill_vision import FullskillVisSkillExpert
from .layerwise_cond_bottleneck import _LayerwiseConditionReader, _LITGoalFreeCondMixin


class FullskillBsplineSkillExpert(FullskillVisSkillExpert):
    def __init__(self, config):
        super().__init__(config)
        depth = int(self.gemma_expert.model.config.num_hidden_layers)
        self.split_layer = depth - int(config.visual_bridge_last_n_layers)
        if not 1 <= self.split_layer < depth:
            raise ValueError("Bspline requires at least one prefix and one late Expert layer.")
        # Lastlayer means the last PREFIX layer, not the final action layer:
        # L9 for a 9+9 split, L17 for 17+1. Keep existing firstlayer weights intact.
        self.start_bridge_layer = (
            self.split_layer - 1 if config.architecture_label.endswith("_lastlayer") else 0
        )
        self.variant = int(config.architecture_label.split("_")[2])
        if self.variant in (5, 6):
            # Same goal-free Cond path as Both4/LIT1--3. The Expert retains its
            # separate start/end XYZ adapters; remove only the dead Cond goal.
            self.cond_end_pose_condition = nn.Identity()
        self.direct_current_proprio = self.variant in (2, 3, 6)
        if self.direct_current_proprio:
            self.bridge_proprio_condition = nn.Sequential(
                nn.Linear(config.max_state_dim, self.width), nn.SiLU(),
                nn.Linear(self.width, self.width, bias=False),
            )
            nn.init.zeros_(self.bridge_proprio_condition[-1].weight)
        plan_width = int(config.visual_bottleneck_width)
        self.bspline_queries = nn.Parameter(torch.randn(config.bspline_tokens, plan_width) * 0.02)
        self.bspline_source_norm = nn.LayerNorm(self.width)
        self.bspline_generator = _LayerwiseConditionReader(plan_width, self.width, 4)
        self.bspline_proprio_head = nn.Sequential(nn.LayerNorm(plan_width), nn.Linear(plan_width, 8))
        self.bspline_read_norm = nn.LayerNorm(self.width)
        self.bspline_reader = nn.MultiheadAttention(
            self.width, int(config.visual_bridge_heads),
            kdim=plan_width, vdim=plan_width, batch_first=True,
        )
        self.bspline_read_gate = nn.Parameter(torch.tensor(float(config.visual_bridge_gate_init)))
        self._bspline_prediction = None
        self._bspline_time_condition = None

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Small adapters remain FP32, as do the existing bottleneck readers.
        for name in (
            "bspline_source_norm", "bspline_generator", "bspline_proprio_head",
            "bspline_read_norm", "bspline_reader",
        ):
            getattr(self, name).to(dtype=torch.float32)
        for parameter in (self.bspline_queries, self.bspline_read_gate):
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()
        return self

    def _project_condition_state(self, state, focus_uv=None, skill_code=None, end_pose=None):
        if self.variant in (5, 6):
            # Both held-start and live-current vision use proprio + skill only.
            # Do not alter the packed goal consumed by _expert_condition below.
            return _LITGoalFreeCondMixin._project_condition_state(
                self, state, focus_uv, skill_code, end_pose,
            )
        return super()._project_condition_state(state, focus_uv, skill_code, end_pose)

    def _expert_condition(self, timestep, projected_state=None, skill_code=None,
                          mode_latent=None, end_pose=None):
        # Keep a pure time condition for variants 3--6. Neither XYZ adapter nor
        # the final Expert norm may reintroduce direct late-layer XYZ inputs.
        self._bspline_time_condition = self._time_condition(timestep)
        return super()._expert_condition(
            timestep, projected_state, skill_code, mode_latent, end_pose,
        )

    def _action_layer_condition(self, layer_index, expert_condition):
        if layer_index < self.split_layer:
            return expert_condition
        condition = expert_condition if self.variant == 2 else self._bspline_time_condition
        if condition is None:
            raise RuntimeError("Bspline action route requires its current flow time condition.")
        condition = condition.to(expert_condition.dtype)
        if self.direct_current_proprio:
            shift = self._bridge_condition_shift
            if shift is None or shift.shape != condition.shape:
                raise RuntimeError("Bspline variants 2/3/6 require current proprio in the late Expert.")
            condition = condition + shift.to(condition.dtype)
        return condition

    def _make_and_read_plan(self, hidden):
        queries = self.bspline_queries[None].expand(hidden.shape[0], -1, -1)
        plan = self.bspline_generator(queries, self.bspline_source_norm(hidden.float()))
        prediction = self.bspline_proprio_head(plan)
        query = self.bspline_read_norm(hidden.float())
        update, _ = self.bspline_reader(query, plan, plan, need_weights=False)
        # No detach: action gradients also shape the plan and prefix layers.
        return hidden + (self.bspline_read_gate.tanh() * update).to(hidden.dtype), prediction

    def _run_expert_with_layerwise_latents(self, noisy_actions, expert_condition,
                                         expert_skill, layer_latents, *, return_all_layers=False):
        hidden = self._action_tokens(noisy_actions)
        mask, positions, rotary = self._sequence_geometry(hidden, self.gemma_expert.model)
        start_latent = self._required_start_latent()
        all_hidden = []
        selective = min(self.split_layer, int(self.config.skill_flow_checkpoint_layers))
        late_condition = self._action_layer_condition(self.split_layer, expert_condition)
        for index in range(int(self.gemma_expert.model.config.num_hidden_layers)):
            if index == self.split_layer:
                if self._gradient_checkpointing and self.training:
                    hidden, prediction = torch.utils.checkpoint.checkpoint(
                        self._make_and_read_plan, hidden, use_reentrant=False, preserve_rng_state=False,
                    )
                else:
                    hidden, prediction = self._make_and_read_plan(hidden)
                self._bspline_prediction = prediction
            condition = expert_condition if index < self.split_layer else late_condition
            # Only variant 2 retains direct skill in the late Expert layers.
            skill = expert_skill if index < self.split_layer or self.variant == 2 else None
            latent = start_latent if index == self.start_bridge_layer else (
                layer_latents[index] if index >= self.split_layer else None
            )
            checkpoint_layer = self.training and (
                self._gradient_checkpointing or self.split_layer - selective <= index < self.split_layer
            )
            args = (index, hidden, mask, positions, condition, skill, latent, rotary)
            if checkpoint_layer:
                hidden = torch.utils.checkpoint.checkpoint(
                    self._expert_layer_with_latent_bridge, *args,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                hidden = self._expert_layer_with_latent_bridge(*args)
            if return_all_layers:
                normalized, _ = layernorm_forward(self.gemma_expert.model.norm, hidden, condition)
                all_hidden.append(normalized)
        hidden, _ = layernorm_forward(self.gemma_expert.model.norm, hidden, late_condition)
        return (hidden, torch.stack(all_hidden, dim=1)) if return_all_layers else hidden

    def take_bspline_prediction(self):
        prediction = self._bspline_prediction
        self._bspline_prediction = None
        if prediction is None:
            raise RuntimeError("Bspline proprio loss requested before an action forward.")
        return prediction

    def _input_sensitivity_stats(self, **kwargs):
        prediction, time = self._bspline_prediction, self._bspline_time_condition
        try:
            return super()._input_sensitivity_stats(**kwargs)
        finally:
            self._bspline_prediction, self._bspline_time_condition = prediction, time
