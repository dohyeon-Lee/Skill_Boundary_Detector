"""Start-only aux3 planners, independent of the action noise/current observation.

All modes share the Action Expert's prefix weights. The action suffix reads
their 30 plan latents except in `_none`, which trains only a one-pass flow
auxiliary. Only the policy owns inference caches; training uses fresh graphs.
"""
from __future__ import annotations

import torch
import torch.utils.checkpoint
from torch import nn

from .fullskill_bspline import FullskillBsplineSkillExpert


class IndependentSkillPlannerExpert(FullskillBsplineSkillExpert):
    def __init__(self, config):
        super().__init__(config)
        self.planner_mode = config.skill_planner_mode
        self.reads_independent_plan = config.reads_bspline_plan
        self.prefix_vision = config.uses_skill_prefix_vision
        self.bspline_plan_projection = nn.Linear(self.width, int(config.visual_bottleneck_width))
        if self.planner_mode == "denoising":
            self.bspline_noise_input = nn.Linear(8, self.width)
        else:
            count = config.chunk_size if self.planner_mode == "learnable_query" else config.bspline_tokens
            self.bspline_learnable_input = nn.Parameter(torch.randn(count, self.width) * .02)
        if self.planner_mode == "learnable_query":
            del self.bspline_plan_projection
        else:
            # Direct modes really have no separate output queries/generator.
            del self.bspline_queries, self.bspline_generator
        self.set_independent_plan(None, None)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for name in ("bspline_plan_projection", "bspline_noise_input"):
            module = getattr(self, name, None)
            if module is not None:
                module.to(dtype=torch.float32)
        parameter = getattr(self, "bspline_learnable_input", None)
        if parameter is not None:
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()
        return self

    def set_independent_plan(self, plan, prediction):
        self._independent_plan = plan
        self._independent_prediction = prediction
        if plan is None:
            self._bspline_prediction = None

    def _plan_reader_inputs(self):
        # Explicit checkpoint arguments prevent recomputation from reading a
        # cache overwritten by a later forward/skill/diagnostic call.
        return self._independent_plan, self._independent_prediction

    def _make_and_read_plan(self, hidden, plan=None, prediction=None):
        if not self.reads_independent_plan:
            return hidden, None
        if plan is None or prediction is None:
            raise RuntimeError("Independent planner must run before Action denoising.")
        query = self.bspline_read_norm(hidden.float())
        update, _ = self.bspline_reader(query, plan, plan, need_weights=False)
        return hidden + (self.bspline_read_gate.tanh() * update).to(hidden.dtype), prediction

    def _layer_uses_visual_bridge(self, layer_index):
        if not self.prefix_vision and int(layer_index) < self.split_layer:
            return False
        return super()._layer_uses_visual_bridge(layer_index)

    def _planner_forward(self, inputs, time, skill_code, end_pose, start_latent):
        """Prefix only: no current-state projection, current vision, or Action hidden."""
        previous_time = self._bspline_time_condition
        try:
            condition = self._expert_condition(time, skill_code=skill_code, end_pose=end_pose)
        finally:
            self._bspline_time_condition = previous_time
        _, skill = self._skill_broadcasts(skill_code)
        hidden = inputs.to(next(self.gemma_expert.parameters()).dtype)
        condition = condition.to(hidden.dtype)
        skill = None if skill is None else skill.to(hidden.dtype)
        mask, positions, rotary = self._sequence_geometry(hidden, self.gemma_expert.model)
        for index in range(self.split_layer):
            args = (index, hidden, mask, positions, condition, skill,
                    start_latent if self.prefix_vision and index == self.start_bridge_layer else None, rotary)
            if self.training and self._gradient_checkpointing:
                hidden = torch.utils.checkpoint.checkpoint(
                    self._expert_layer_with_latent_bridge, *args,
                    use_reentrant=False, preserve_rng_state=True,
                )
            else:
                hidden = self._expert_layer_with_latent_bridge(*args)
        if self.planner_mode == "learnable_query":
            return self._generate_plan(hidden)
        return self.bspline_plan_projection(self.bspline_source_norm(hidden.float()))

    def _flow_readout(self, state, time, skill_code, end_pose, start_latent):
        plan = self._planner_forward(self.bspline_noise_input(state.float()), time, skill_code, end_pose, start_latent)
        return plan, self.bspline_proprio_head(plan)

    def _flow_supervision(self, target, skill_code, end_pose, start_latent):
        expected = (skill_code.shape[0], int(self.config.bspline_tokens), 8)
        if tuple(target.shape) != expected or not bool(torch.isfinite(target).all()):
            raise ValueError(f"Planner flow target must be finite {expected} normalized proprio.")
        device = skill_code.device
        noise = self.sample_noise(target.shape, device).float()
        time = self.sample_time(target.shape[0], device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * target
        _, velocity = self._flow_readout(x_t, time, skill_code, end_pose, start_latent)
        return (velocity - (noise - target)).square().mean(dim=(1, 2))

    def generate_independent_plan(self, skill_code, end_pose, *, start_latent=None, target=None):
        """Return plan latent, clean normalized proprio, and optional per-sample FM loss.

        Reader modes provide a freely generated plan, including during training.
        `_none` returns no plan/prediction: one FM forward with a target, and no
        work at all without a target. GT-noised states never enter Action.
        """
        if not self.reads_independent_plan and target is None:
            return None, None, None
        latent = (
            self._required_start_latent() if start_latent is None else start_latent
        ) if self.prefix_vision else None
        if not self.reads_independent_plan:
            return None, None, self._flow_supervision(target, skill_code, end_pose, latent)
        batch, device = skill_code.shape[0], skill_code.device
        time = torch.ones(batch, device=device, dtype=torch.float32)
        if self.planner_mode != "denoising":
            inputs = self.bspline_learnable_input[None].expand(batch, -1, -1)
            plan = self._planner_forward(inputs, time, skill_code, end_pose, latent)
            return plan, self.bspline_proprio_head(plan), None

        count = int(self.config.bspline_tokens)
        state = self.sample_noise((batch, count, 8), device).float()
        steps = int(self.config.num_inference_steps)
        for index in range(steps):
            time = torch.full((batch,), 1.0 - index / steps, device=device)
            _, velocity = self._flow_readout(state, time, skill_code, end_pose, latent)
            state = state - velocity / steps
        # Read the CLEAN final state at t=0, never a noisy intermediate latent.
        plan, _ = self._flow_readout(state, torch.zeros(batch, device=device), skill_code, end_pose, latent)
        flow_loss = None
        if target is not None:
            flow_loss = self._flow_supervision(target, skill_code, end_pose, latent)
        return plan, state, flow_loss
