"""A single action flow: final current vision at all layers, or only L18."""
from __future__ import annotations

import torch
from torch import nn

from .layerwise_cond_bottleneck import BothLIT7SkillExpert, _LITGoalFreeCondMixin
from .fullskill_bspline import FullskillBsplineSkillExpert


class VanilaSkillExpert(BothLIT7SkillExpert):
    """Aux3 vision interface without any full-skill planner or plan reader.

    Non-visual conditions reach every Expert layer (and the final norm).
    Start images retain the baseline's training-only auxiliary supervision;
    inference encodes only the current observation.
    """
    _alignment_token_count = 3
    _split_uv_goal = staticmethod(FullskillBsplineSkillExpert._split_uv_goal)

    def __init__(self, config):
        super().__init__(config)
        for name in ("cond_end_pose_condition", "end_pose_condition"):
            if name == "end_pose_condition" and config.omits_skill_expert_uv_goal:
                setattr(self, name, nn.Identity())
                continue
            adapter = nn.Sequential(nn.Linear(2, self.width), nn.SiLU(),
                                    nn.Linear(self.width, self.width, bias=False))
            nn.init.zeros_(adapter[-1].weight)
            setattr(self, name, adapter)
        self.bridge_proprio_condition = nn.Sequential(
            nn.Linear(config.max_state_dim, self.width), nn.SiLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        nn.init.zeros_(self.bridge_proprio_condition[-1].weight)
        self.agent_end_state_head = nn.Sequential(
            nn.LayerNorm(config.visual_bottleneck_width), nn.Linear(config.visual_bottleneck_width, 3),
        )
        self._current_proprio_condition = None

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.bridge_proprio_condition.to(dtype=torch.float32)
        self.agent_end_state_head.to(dtype=torch.float32)
        return self

    def _project_condition_state(self, state, focus_uv=None, skill_code=None, end_pose=None):
        if state is None:
            raise ValueError("Vanila requires current proprio.")
        uv, valid, _ = self._split_uv_goal(end_pose)
        self._current_proprio_condition = self.bridge_proprio_condition(state.float())
        projected = _LITGoalFreeCondMixin._project_condition_state(self, state, None, skill_code, None)
        return projected + (self.cond_end_pose_condition(uv) * valid).to(projected.dtype)

    def _expert_condition(self, timestep, projected_state=None, skill_code=None,
                          mode_latent=None, end_pose=None):
        uv, valid, start = self._split_uv_goal(end_pose)
        if self._current_proprio_condition is None:
            raise RuntimeError("Vanila requires current observation encoding before action generation.")
        condition = self._time_condition(timestep)
        condition = condition + self.start_pose_condition(start).to(condition.dtype)
        condition = condition + self._current_proprio_condition.to(condition.dtype)
        if not self.config.omits_skill_expert_uv_goal:
            condition = condition + (self.end_pose_condition(uv) * valid).to(condition.dtype)
        return condition

    def _run_expert_with_layerwise_latents(self, noisy_actions, expert_condition,
                                         expert_skill, layer_latents, **kwargs):
        # Every participating Expert layer reads Z18, never its corresponding Zi.
        return super()._run_expert_with_layerwise_latents(
            noisy_actions, expert_condition, expert_skill,
            [layer_latents[-1]] * len(layer_latents), **kwargs,
        )

    def _input_sensitivity_stats(self, **kwargs):
        # Diagnostic forwards must not replace the real observation's state or
        # the differentiable auxiliary-head inputs consumed by the policy.
        names = ("_current_proprio_condition", "_bridge_condition_shift",
                 "_final_dual_patch_query", "_final_agent_patch_tokens", "_final_wrist_patch_tokens")
        saved = {name: getattr(self, name, None) for name in names}
        try:
            return super()._input_sensitivity_stats(**kwargs)
        finally:
            for name, value in saved.items():
                setattr(self, name, value)

    def _alignment_memories(self, memory):
        return (*super()._alignment_memories(memory), memory)

    def _update_layerwise_latent(self, layer_index, latent, memory):
        reader = self.layerwise_condition_readers[layer_index]
        aligned = [reader.read_condition(latent[:, i:i + 1], camera)
                   for i, camera in enumerate(self._alignment_memories(memory))]
        action = reader.read_condition(latent[:, 3:], memory)
        return reader.mix_latents(torch.cat([*aligned, action], dim=1))

    def _on_final_bottleneck_latent(self, latent):
        super()._on_final_bottleneck_latent(latent)
        self._final_dual_patch_query = latent[:, :3] if self.training else None

    # This shared readout does not construct a Bspline module or any plan tensors.
    predict_training_top_wrist_patch_and_xyz = FullskillBsplineSkillExpert.predict_training_top_wrist_patch_and_xyz

    def encode_skill_start(self, images, state, skill_code, end_pose):
        tokens = self._condition_tokens(images, batch_size=state.shape[0], skill_code=skill_code)
        condition = self._project_condition_state(state, None, skill_code, end_pose)
        return self._encode_layerwise_latents(tokens, condition)[-1]

    def set_skill_start_latent(self, latent):
        # Start-camera encoding supervises auxiliary heads only, not actions.
        del latent
