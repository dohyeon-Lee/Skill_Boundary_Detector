"""Current-state 4H motion base with an H-token, vision-conditioned velocity residual.

Notion specification: https://app.notion.com/p/3f8ffbb8f49380acb08ec1bf50cb9098
The padding mask supervises the base loss ONLY; all 4H tokens remain visible in
both training and inference. No observed past actions or GT future lengths are inputs.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor, nn
from torch.func import functional_call

from lerobot.policies.pi05.modeling_pi05 import layernorm_forward
from .configuration_skill_expert import DIRECT_PATCH_ARCH_LABELS, WINDOW_PATCH_ARCH_LABELS
from .layerwise_cond_bottleneck import Both3SkillExpert, LayerwiseCondBottleneckSkillExpert


class ResidualPatchSkillExpert(Both3SkillExpert):
    """Reuse Both3's dedicated 1+1+98 visual interface and six direct-input contracts."""

    def __init__(self, config):
        super().__init__(config)
        if config.architecture_label not in WINDOW_PATCH_ARCH_LABELS:
            raise ValueError("Expected residual_patch_1..6 or direct_patch_1..6, optionally suffixed _dGD.")
        self.detach_action_base = config.architecture_label.endswith("_dGD")
        self.residual_variant = int(config.architecture_label.removesuffix("_dGD").rsplit("_", 1)[1])
        if self.visual_bridge_start_layer != 17:
            raise ValueError("Window patch requires exactly 17 base layers and one terminal layer.")
        if config.architecture_label not in DIRECT_PATCH_ARCH_LABELS:
            self.residual_action_head = nn.Linear(self.width, config.max_action_dim)
            nn.init.zeros_(self.residual_action_head.weight)
            nn.init.zeros_(self.residual_action_head.bias)
        self._last_skill_window_residual = None
        self._last_residual_diagnostics = {}
        self._window_debug_condition_tokens = None

    @staticmethod
    def _read_only_module(module, *args, **kwargs):
        """Read shared weights without parameter gradients; keep input gradients.

        Unlike no_grad/output.detach/requires_grad_ toggles, this preserves the
        terminal layer's backward and the base loss's original parameter graph.
        No parameter copies are registered and state_dict stays unchanged.
        """
        parameters = {name: parameter.detach() for name, parameter in module.named_parameters()}
        return functional_call(module, parameters, args, kwargs)

    def _terminal_norm(self, hidden, condition):
        norm = self.gemma_expert.model.norm
        if self.detach_action_base:
            norm = partial(self._read_only_module, norm)
        return layernorm_forward(norm, hidden, condition)[0]

    def _terminal_shared_velocity(self, hidden):
        if not self.detach_action_base:
            return self._action_velocity(hidden)
        head = self.action_out_proj
        parameter = next(head.parameters(), None)
        dtype = hidden.dtype if parameter is None else parameter.dtype
        return self._read_only_module(head, hidden.to(dtype=dtype)).float()

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if hasattr(self, "residual_action_head"):
            self.residual_action_head.to(dtype=torch.float32)
        return self

    def _expert_layer_with_latent_bridge(self, *args, **kwargs):
        # The caller supplies base/terminal AdaRMS explicitly. Do not apply
        # Arch17's terminal proprio shift a second time.
        return LayerwiseCondBottleneckSkillExpert._expert_layer_with_latent_bridge(
            self, *args, **kwargs
        )

    def _residual_conditions(self, time, state, end_pose):
        if state is None:
            raise ValueError("Window base requires current proprio.")
        end, start = self._split_goal(end_pose, self.config.architecture_label)
        time_condition = self._time_condition(time)
        prop = self.bridge_proprio_condition(state.float()).to(time_condition.dtype)
        base_condition = time_condition + prop
        if self.residual_variant <= 5:
            base_condition = base_condition + self.end_pose_condition(end.float()).to(base_condition.dtype)
        if self.residual_variant <= 4:
            base_condition = base_condition + self.start_pose_condition(start.float()).to(base_condition.dtype)
        last_condition = time_condition
        if self.residual_variant in {1, 2}:
            last_condition = (last_condition
                + self.end_pose_condition(end.float()).to(last_condition.dtype)
                + self.start_pose_condition(start.float()).to(last_condition.dtype))
        if self.residual_variant in {1, 3}:
            last_condition = last_condition + prop
        return base_condition, last_condition

    def _run_residual_velocity(self, noisy_window, time, state, skill_code, end_pose, layer_latents):
        h = self.config.chunk_size
        if noisy_window.shape[1] != 4 * h:
            raise ValueError("Motion window must have exactly 4H tokens.")
        _, skill = self._skill_broadcasts(skill_code)
        base_condition, last_condition = self._residual_conditions(time, state, end_pose)
        hidden = self._action_tokens(noisy_window)
        attention, positions, rotary = self._sequence_geometry(hidden, self.gemma_expert.model)
        checkpoint_n = min(max(int(self.config.skill_flow_checkpoint_layers), 0), 17)
        checkpoint_forward = self.training and torch.is_grad_enabled()
        for i in range(17):
            args = (i, hidden, attention, positions, base_condition, skill, None, rotary)
            if checkpoint_forward and (self._gradient_checkpointing or i >= 17 - checkpoint_n):
                hidden = torch.utils.checkpoint.checkpoint(
                    self._expert_layer_with_latent_bridge, *args,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                hidden = self._expert_layer_with_latent_bridge(*args)
        normalized_base, _ = layernorm_forward(self.gemma_expert.model.norm, hidden, base_condition)
        base_velocity = self._action_velocity(normalized_base)
        # Attached variants train the base through both action paths. _dGD
        # reads the same values but blocks every direct action->base route,
        # including shared time/skill/proprio/XYZ conditioning projections.
        central = hidden[:, h:2 * h]
        central_base = base_velocity[:, h:2 * h]
        terminal_skill = skill
        if self.detach_action_base:
            central = central.detach()
            central_base = central_base.detach()
            last_condition = last_condition.detach()
            terminal_skill = skill.detach()
        central_rotary = tuple(value[:, h:2 * h] for value in rotary)
        args = (17, central, attention[:, :, h:2 * h, h:2 * h],
                positions[:, h:2 * h], last_condition, terminal_skill, layer_latents[17], central_rotary)
        if checkpoint_forward and self._gradient_checkpointing:
            central = torch.utils.checkpoint.checkpoint(
                self._expert_layer_with_latent_bridge, *args,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            central = self._expert_layer_with_latent_bridge(*args)
        normalized_last = self._terminal_norm(central, last_condition)
        terminal_velocity = self._terminal_velocity(normalized_last)
        # Keep the 4H auxiliary graph in base_velocity, not in the action
        # readout. This also avoids materializing zero base grads that AdamW
        # could decay during an action-only update with an unfrozen base.
        velocity = (base_velocity.detach() if self.detach_action_base else base_velocity).clone()
        velocity[:, h:2 * h] = self._combine_central_velocity(central_base, terminal_velocity)
        return velocity, base_velocity, terminal_velocity

    def _terminal_velocity(self, normalized_last):
        return self.residual_action_head(normalized_last.float())

    def _combine_central_velocity(self, base, terminal):
        return base + terminal

    def _residual_visual_cache(self, images, state, skill_code, end_pose):
        tokens = self._condition_tokens(images, batch_size=state.shape[0], skill_code=skill_code)
        if self._vsa_debug_active:
            # Reuse the REAL forward's DINO tokens. Probes re-run Cond/Expert,
            # not DINO, and never receive independently sampled noise or time.
            self._window_debug_condition_tokens = tokens.detach()
        self._record_visual_debug(tokens)
        projected = self._project_condition_state(state, None, skill_code, end_pose)
        return self._encode_layerwise_latents(tokens, projected)

    def _predict_window_velocity_from_condition(
        self, condition_tokens, noisy_window, state, skill_code, time,
        mode_latent=None, focus_uv=None, end_pose=None,
    ):
        del mode_latent, focus_uv
        projected = self._project_condition_state(state, None, skill_code, end_pose)
        latents = self._encode_layerwise_latents(condition_tokens, projected)
        velocity, _, _ = self._run_residual_velocity(
            noisy_window, time, state, skill_code, end_pose, latents,
        )
        h = self.config.chunk_size
        return velocity[:, h:2 * h]

    @torch.no_grad()
    def _window_input_sensitivity_stats(self, **kwargs):
        # Probe Cond forwards overwrite the patch-loss readout caches. Keep
        # the original graph-bearing tensors for the pending auxiliary loss
        # and backward (including checkpoint recomputation).
        cache_names = (
            "_bridge_condition_shift", "_final_agent_patch_tokens",
            "_final_wrist_patch_tokens", "_final_dual_patch_query",
        )
        original = {name: getattr(self, name) for name in cache_names if hasattr(self, name)}
        device = kwargs["noisy_actions"].device
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        try:
            # Diagnostics must not advance the RNG used by later training
            # updates, even if a Cond/Expert module consumes random values.
            with torch.random.fork_rng(devices=cuda_devices):
                return self._input_sensitivity_stats(
                    **kwargs, velocity_predictor=self._predict_window_velocity_from_condition,
                )
        finally:
            for name, value in original.items():
                setattr(self, name, value)

    def forward(self, images, state, skill_code, actions, noise=None, time=None,
                mode_latent=None, focus_uv=None, end_pose=None, *, window_actions=None):
        del mode_latent, focus_uv
        if window_actions is None:
            raise ValueError("Window training requires the current-frame 4H action window.")
        h = self.config.chunk_size
        if actions.shape[1] != h or window_actions.shape != (actions.shape[0], 4 * h, actions.shape[2]):
            raise ValueError("Window training action/window shapes do not match H/4H.")
        time = self.sample_time(actions.shape[0], actions.device) if time is None else time
        source = self.sample_noise(window_actions.shape, actions.device).to(actions.dtype)
        if noise is not None:
            source[:, h:2 * h] = noise.to(source.dtype)
        targets = window_actions.clone()
        # Preserve the ORIGINAL main action targets, including any cross-skill
        # tail that its existing action mask allows. Base uses its own skill mask.
        targets[:, h:2 * h] = actions
        noisy = time[:, None, None] * source + (1 - time[:, None, None]) * targets
        target_velocity = source - targets
        self._last_flow_time = time.detach()
        self._last_flow_noise = source[:, h:2 * h].detach()
        self._last_vsa_debug_stats = {}
        self._window_debug_condition_tokens = None
        latents = self._residual_visual_cache(images, state, skill_code, end_pose)
        velocity, base, delta = self._run_residual_velocity(noisy, time, state, skill_code, end_pose, latents)
        self._last_skill_window_residual = target_velocity - base
        self._last_residual_diagnostics = {
            "base": base[:, h:2 * h].detach(), "delta": delta.detach(),
            "target": target_velocity[:, h:2 * h].detach(),
        }
        self._last_predicted_actions = (
            noisy[:, h:2 * h] - time[:, None, None] * velocity[:, h:2 * h]
            if self.config.cumulative_xyz_loss_enabled else None
        )
        if self._vsa_debug_active:
            try:
                if self._window_debug_condition_tokens is None:
                    raise RuntimeError("Window sensitivity probes require cached visual tokens.")
                self._last_vsa_debug_stats.update(self._window_input_sensitivity_stats(
                    predicted_velocity=velocity[:, h:2 * h],
                    condition_tokens=self._window_debug_condition_tokens,
                    noisy_actions=noisy,
                    state=state, skill_code=skill_code, time=time, end_pose=end_pose,
                ))
            finally:
                self._window_debug_condition_tokens = None
        return target_velocity[:, h:2 * h] - velocity[:, h:2 * h]

    def take_skill_window_residual(self):
        value = self._last_skill_window_residual
        self._last_skill_window_residual = None
        if value is None:
            raise RuntimeError("No window base training forward is available.")
        return value

    def residual_diagnostics(self, valid, real_dim):
        d = self._last_residual_diagnostics
        self._last_residual_diagnostics = {}
        if not d:
            return {}
        base, delta, target = (d[k][..., :real_dim].float() for k in ("base", "delta", "target"))
        def mean(x):
            return torch.where(valid[..., None], x, 0).sum() / (valid.sum() * real_dim).clamp_min(1)
        base_mse = mean((target - base).square())
        final_mse = mean((target - base - delta).square())
        base_rms, delta_rms = mean(base.square()).sqrt(), mean(delta.square()).sqrt()
        values = {
            "base_velocity_rms": base_rms, "residual_velocity_rms": delta_rms,
            "residual_to_base_ratio": delta_rms / base_rms.clamp_min(1e-8),
            "base_velocity_mse": base_mse, "final_velocity_mse": final_mse,
            "mse_improvement": base_mse - final_mse,
        }
        scalars = torch.stack(list(values.values())).detach().cpu().tolist()
        return {"residual/" + k: v for k, v in zip(values, scalars, strict=True)}

    @torch.no_grad()
    def sample_actions(self, images, state, skill_code, noise=None, num_steps=None,
                       mode_latent=None, focus_uv=None, end_pose=None):
        del mode_latent, focus_uv
        if state is None:
            raise ValueError("Window sampling requires current proprio.")
        h = self.config.chunk_size
        shape = (state.shape[0], 4 * h, self.config.max_action_dim)
        if noise is None:
            window = self.sample_noise(shape, state.device).float()
        elif tuple(noise.shape) == shape:
            window = noise.float().clone()
        elif tuple(noise.shape) == (shape[0], h, shape[2]):
            window = self.sample_noise(shape, state.device).float()
            window[:, h:2 * h] = noise.float()
        else:
            raise ValueError("Window sampling noise must have H or 4H tokens.")
        latents = self._residual_visual_cache(images, state, skill_code, end_pose)
        steps = self.config.num_inference_steps if num_steps is None else int(num_steps)
        if steps <= 0:
            raise ValueError("num_steps must be positive.")
        dt = -1.0 / steps
        for i in range(steps):
            time = torch.full((shape[0],), 1 + i * dt, device=window.device, dtype=torch.float32)
            velocity, _, _ = self._run_residual_velocity(window, time, state, skill_code, end_pose, latents)
            window = window + dt * velocity
        return window[:, h:2 * h]

    @torch.no_grad()
    def sample_skill_only_actions(self, *args, **kwargs):
        raise ValueError("Window patch uses a joint 4H rollout; canonical skill-only sampling is undefined.")
