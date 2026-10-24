"""H-token flow with a supervised, fixed-size plan at the Expert split.

The plan is regenerated at each flow time after the prefix Expert layers.
The final ``visual_bridge_last_n_layers`` layers form the action suffix; only
their first layer reads the plan. Current vision is layerwise by default, or
the final bottleneck feeds all/first/last suffix layers in lastvis variants.
Held start vision enters the first or last
prefix Expert layer, independently of where the current-vision suffix starts.
The ``_current`` variant instead shares live Z18 with the prefix bridge,
without encoding or caching a separate skill-start observation.
It is not an additional noisy-action sequence or an independently sampled flow.
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint
from torch import nn

from lerobot.policies.pi05.modeling_pi05 import layernorm_forward

from .fullskill_vision import FullskillVisSkillExpert
from .layerwise_cond_bottleneck import _LayerwiseConditionReader, _LITGoalFreeCondMixin
from lerobot.configs.uv_fullskill import bspline_base_label


class FullskillBsplineSkillExpert(FullskillVisSkillExpert):
    def __init__(self, config):
        super().__init__(config)
        depth = int(self.gemma_expert.model.config.num_hidden_layers)
        self.split_layer = depth - int(config.visual_bridge_last_n_layers)
        if not 1 <= self.split_layer < depth:
            raise ValueError("Bspline requires at least one prefix and one late Expert layer.")
        # Lastlayer means the last PREFIX layer, not the final action layer:
        # L9 for a 9+9 split, L17 for 17+1. Keep existing firstlayer weights intact.
        base_label = bspline_base_label(config.architecture_label)
        self.start_bridge_layer = (
            self.split_layer - 1
            if base_label.split("_lastvis", 1)[0].endswith("_lastlayer") else 0
        )
        self.variant = int(base_label.split("_")[2])
        self.top_uv_goal = bool(getattr(config, "uses_top_uv_goal", False))
        if self.top_uv_goal:
            # Same zero-initialized MLP -> AdaRMS route, now with a 2-D goal.
            # Retain the separate 3-D start adapter; diffaux switches the Top
            # auxiliary readout below, independently of this input adapter.
            for name in ("cond_end_pose_condition", "end_pose_condition"):
                input_dim = 3 if name == "end_pose_condition" and getattr(config, "uses_predicted_expert_xyz", False) else 2
                module = nn.Sequential(nn.Linear(input_dim, self.width), nn.SiLU(),
                                       nn.Linear(self.width, self.width, bias=False))
                nn.init.zeros_(module[-1].weight)
                setattr(self, name, module)
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
        if getattr(config, "trains_agent_end_state_prediction", False):
            # Keep the dedicated Top query (index 0) and 1+1+98 interface.
            # Replace ONLY its training readout, not the gated Expert reader.
            self.agent_patch_align_head = nn.Identity()
            output_dim = 3 if getattr(config, "uses_predicted_expert_xyz", False) else 8
            self.agent_end_state_head = nn.Sequential(nn.LayerNorm(plan_width), nn.Linear(plan_width, output_dim))
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
        self._preskill_plan = None
        self._preskill_valid = None

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
        if hasattr(self, "agent_end_state_head"):
            self.agent_end_state_head.to(dtype=torch.float32)
        return self

    def _update_layerwise_latent(self, layer_index, latent, memory):
        if not getattr(self.config, "uses_joint_bottleneck_self_attention", False):
            return super()._update_layerwise_latent(layer_index, latent, memory)
        reader = self.layerwise_condition_readers[layer_index]
        # Preserve Top-only / Wrist-only / both-camera cross-attention. Only
        # replace the three independent self-attention calls with one joint call.
        aligned = [
            reader.read_condition(latent[:, index:index + 1], camera_memory)
            for index, camera_memory in enumerate(self._alignment_memories(memory))
        ]
        action = reader.read_condition(latent[:, self._alignment_token_count:], memory)
        return reader.mix_latents(torch.cat([*aligned, action], dim=1))

    def predict_training_end_state_and_wrist_patch(self):
        """Consume the same two auxiliary queries: Top state8/XYZ3, wrist patch CE."""
        query = self._final_dual_patch_query
        wrist_tokens = self._final_wrist_patch_tokens
        self._final_dual_patch_query = None
        self._final_agent_patch_tokens = None
        self._final_wrist_patch_tokens = None
        if query is None or wrist_tokens is None:
            raise RuntimeError("Diffaux readout requires a preceding training condition forward.")
        return (self.agent_end_state_head(query[:, 0].float()),
                self.wrist_patch_align_head(query[:, 1:2], wrist_tokens))

    def _condition_on_predicted_xyz(self, condition, final_latent):
        """Read the actual encoded observation, never a GT endpoint or stale cache."""
        if not getattr(self.config, "uses_predicted_expert_xyz", False):
            return condition
        xyz = self.agent_end_state_head(final_latent[:, 0].float())
        if self.config.detaches_predicted_expert_xyz:
            xyz = xyz.detach()
        q01 = xyz.new_tensor(self.config.chunk_end_state_q01[:3])
        q99 = xyz.new_tensor(self.config.chunk_end_state_q99[:3])
        # XYZ targets use quantile normalization; both the old start/end
        # Expert adapters consume episode-grounded metres.
        xyz = (xyz + 1) * 0.5 * (q99 - q01).clamp_min(1e-6) + q01
        return condition + self.end_pose_condition(xyz).to(condition.dtype)

    def _project_condition_state(self, state, focus_uv=None, skill_code=None, end_pose=None):
        if self.top_uv_goal:
            uv, valid, _ = self._split_uv_goal(end_pose)
            if state is None:
                raise ValueError("Top-UV Bspline requires current proprio.")
            projected = _LITGoalFreeCondMixin._project_condition_state(
                self, state, None, skill_code, None,
            )
            return projected + (self.cond_end_pose_condition(uv) * valid).to(projected.dtype)
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
        if self.top_uv_goal:
            uv, valid, start_xyz = self._split_uv_goal(end_pose)
            condition = self._bspline_time_condition
            if getattr(self.config, "uses_predicted_expert_xyz", False):
                # Training constructs this before encoding the condition stream.
                # Add the predicted goal once the final current latent exists.
                return condition + self.start_pose_condition(start_xyz).to(condition.dtype)
            return (condition + (self.end_pose_condition(uv) * valid).to(condition.dtype)
                    + self.start_pose_condition(start_xyz).to(condition.dtype))
        return super()._expert_condition(
            timestep, projected_state, skill_code, mode_latent, end_pose,
        )

    @staticmethod
    def _split_uv_goal(goal):
        # Six values preserve the existing opaque goal transport contract.
        # The validity bit masks the WHOLE adapter, including its input bias.
        if goal is None or goal.ndim != 2 or goal.shape[1] != 6:
            raise ValueError("Top-UV goal must be [end UV2, UV-valid1, start XYZ3].")
        if not bool(torch.isfinite(goal).all()):
            raise ValueError("Top-UV packed goal must be finite.")
        return goal[:, :2].float(), goal[:, 2:3].float(), goal[:, 3:].float()

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

    def _generate_plan(self, hidden):
        queries = self.bspline_queries[None].expand(hidden.shape[0], -1, -1)
        return self.bspline_generator(queries, self.bspline_source_norm(hidden.float()))

    def set_preskill_plan(self, plan, valid):
        self._preskill_plan = None if plan is None else plan.detach()
        self._preskill_valid = None if valid is None else valid.detach().bool()

    @torch.no_grad()
    def encode_preskill_plan(self, images, state, skill_code, end_pose):
        """Tau=1, prefix ONLY: memory never reads memory (including during training).

        A fixed Gaussian template makes train/eval extraction reproducible without
        changing the global RNG. This is direct proprio supervision, not a second
        denoising chain. No late action layers or auxiliary patch loss are run.
        """
        training_flags = [(module, module.training) for module in self.modules()]
        shift, time = self._bridge_condition_shift, self._bspline_time_condition
        debug = self._vsa_debug_active
        try:
            self.eval()
            self._vsa_debug_active = False
            tokens = self._condition_tokens(images, batch_size=state.shape[0], skill_code=skill_code)
            projected = self._project_condition_state(state, None, skill_code, end_pose)
            latent = self._encode_layerwise_latents(tokens, projected)[-1]
            _, skill = self._skill_broadcasts(skill_code)
            condition = self._expert_condition(
                torch.ones(state.shape[0], device=state.device),
                projected_state=projected, skill_code=skill_code, end_pose=end_pose,
            )
            condition = self._condition_on_predicted_xyz(condition, latent)
            generator = torch.Generator(device="cpu").manual_seed(0)
            noise = torch.randn(
                (1, self.config.chunk_size, self.config.max_action_dim), generator=generator, dtype=torch.float32,
            ).to(state.device).expand(state.shape[0], -1, -1)
            hidden = self._action_tokens(noise)
            mask, positions, rotary = self._sequence_geometry(hidden, self.gemma_expert.model)
            for index in range(self.split_layer):
                hidden = self._expert_layer_with_latent_bridge(
                    index, hidden, mask, positions, condition, skill,
                    latent if index == self.start_bridge_layer else None, rotary,
                )
            return self._generate_plan(hidden).detach()
        finally:
            self._bridge_condition_shift, self._bspline_time_condition = shift, time
            self._vsa_debug_active = debug
            for module, training in training_flags:
                module.training = training

    def _make_and_read_plan(self, hidden, previous_plan=None, previous_valid=None):
        plan = self._generate_plan(hidden)
        prediction = self.bspline_proprio_head(plan)
        if bspline_base_label(self.config.architecture_label).endswith("_current_none"):
            # Auxiliary plan MSE is retained, but action reads ONLY the H-token
            # main path. In particular no cross-attention or output bias runs.
            return hidden, prediction
        if getattr(self.config, "uses_bspline_preskill_memory", False):
            if previous_plan is None or previous_valid is None:
                raise RuntimeError("Preskill plan inputs must be prepared, including the BOS validity mask.")
            plan = previous_plan.detach()
        query = self.bspline_read_norm(hidden.float())
        update, _ = self.bspline_reader(query, plan, plan, need_weights=False)
        if getattr(self.config, "uses_bspline_preskill_memory", False):
            # Mask the entire update, including MHA output bias: zero memory
            # alone is not equivalent to having no previous skill at BOS.
            update = update * previous_valid[:, None, None].to(update.dtype)
        return hidden + (self.bspline_read_gate.tanh() * update).to(hidden.dtype), prediction

    def _run_expert_with_layerwise_latents(self, noisy_actions, expert_condition,
                                         expert_skill, layer_latents, *, return_all_layers=False):
        expert_condition = self._condition_on_predicted_xyz(expert_condition, layer_latents[-1])
        hidden = self._action_tokens(noisy_actions)
        mask, positions, rotary = self._sequence_geometry(hidden, self.gemma_expert.model)
        prefix_latent = (
            layer_latents[-1] if getattr(self.config, "uses_current_bspline_vision", False)
            else self._required_start_latent()
        )
        all_hidden = []
        selective = min(self.split_layer, int(self.config.skill_flow_checkpoint_layers))
        late_condition = self._action_layer_condition(self.split_layer, expert_condition)
        for index in range(int(self.gemma_expert.model.config.num_hidden_layers)):
            if index == self.split_layer:
                if self._gradient_checkpointing and self.training:
                    hidden, prediction = torch.utils.checkpoint.checkpoint(
                        self._make_and_read_plan, hidden, self._preskill_plan, self._preskill_valid,
                        use_reentrant=False, preserve_rng_state=False,
                    )
                else:
                    hidden, prediction = self._make_and_read_plan(hidden, self._preskill_plan, self._preskill_valid)
                self._bspline_prediction = prediction
            condition = expert_condition if index < self.split_layer else late_condition
            # Only variant 2 retains direct skill in the late Expert layers.
            skill = expert_skill if index < self.split_layer or self.variant == 2 else None
            latent = None
            if index == self.start_bridge_layer:
                latent = prefix_latent
            elif index in self.current_bridge_layers:
                latent = layer_latents[index] if self.current_vision_route == "layerwise" else layer_latents[-1]
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
