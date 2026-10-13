"""Current-state 4H motion base with an H-token, vision-conditioned velocity residual.

Notion specification: https://app.notion.com/p/3f8ffbb8f49380acb08ec1bf50cb9098
The padding mask supervises the base loss ONLY; all 4H tokens remain visible in
both training and inference. No observed past actions or GT future lengths are inputs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lerobot.policies.pi05.modeling_pi05 import layernorm_forward
from .configuration_skill_expert import RESIDUAL_PATCH_ARCH_LABELS
from .layerwise_cond_bottleneck import Both3SkillExpert, LayerwiseCondBottleneckSkillExpert


class ResidualPatchSkillExpert(Both3SkillExpert):
    """Reuse Both3's dedicated 1+1+98 visual interface and six direct-input contracts."""

    def __init__(self, config):
        super().__init__(config)
        if config.architecture_label not in RESIDUAL_PATCH_ARCH_LABELS:
            raise ValueError("Expected residual_patch_1 through residual_patch_6.")
        self.residual_variant = int(config.architecture_label.rsplit("_", 1)[1])
        if self.visual_bridge_start_layer != 17:
            raise ValueError("Residual_patch requires exactly 17 base layers and one terminal layer.")
        self.residual_action_head = nn.Linear(self.width, config.max_action_dim)
        nn.init.zeros_(self.residual_action_head.weight)
        nn.init.zeros_(self.residual_action_head.bias)
        self._last_skill_window_residual = None
        self._last_residual_diagnostics = {}

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
            raise ValueError("Residual base requires current proprio.")
        end, start = self._split_goal(end_pose, "Residual_patch")
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
            raise ValueError("Residual motion window must have exactly 4H tokens.")
        _, skill = self._skill_broadcasts(skill_code)
        base_condition, last_condition = self._residual_conditions(time, state, end_pose)
        hidden = self._action_tokens(noisy_window)
        attention, positions, rotary = self._sequence_geometry(hidden, self.gemma_expert.model)
        checkpoint_n = min(max(int(self.config.skill_flow_checkpoint_layers), 0), 17)
        for i in range(17):
            args = (i, hidden, attention, positions, base_condition, skill, None, rotary)
            if self.training and (self._gradient_checkpointing or i >= 17 - checkpoint_n):
                hidden = torch.utils.checkpoint.checkpoint(
                    self._expert_layer_with_latent_bridge, *args,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                hidden = self._expert_layer_with_latent_bridge(*args)
        normalized_base, _ = layernorm_forward(self.gemma_expert.model.norm, hidden, base_condition)
        base_velocity = self._action_velocity(normalized_base)
        # Last layer receives the SAME central noisy-token hidden state. No
        # detach: action loss trains the motion base through both paths.
        central = hidden[:, h:2 * h]
        central_rotary = tuple(value[:, h:2 * h] for value in rotary)
        args = (17, central, attention[:, :, h:2 * h, h:2 * h],
                positions[:, h:2 * h], last_condition, skill, layer_latents[17], central_rotary)
        if self.training and self._gradient_checkpointing:
            central = torch.utils.checkpoint.checkpoint(
                self._expert_layer_with_latent_bridge, *args,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            central = self._expert_layer_with_latent_bridge(*args)
        normalized_last, _ = layernorm_forward(self.gemma_expert.model.norm, central, last_condition)
        delta = self.residual_action_head(normalized_last.float())
        velocity = base_velocity.clone()
        velocity[:, h:2 * h] = base_velocity[:, h:2 * h] + delta
        return velocity, base_velocity, delta

    def _residual_visual_cache(self, images, state, skill_code, end_pose):
        tokens = self._condition_tokens(images, batch_size=state.shape[0], skill_code=skill_code)
        self._record_visual_debug(tokens)
        projected = self._project_condition_state(state, None, skill_code, end_pose)
        return self._encode_layerwise_latents(tokens, projected)

    def forward(self, images, state, skill_code, actions, noise=None, time=None,
                mode_latent=None, focus_uv=None, end_pose=None, *, window_actions=None):
        del mode_latent, focus_uv
        if window_actions is None:
            raise ValueError("Residual training requires the current-frame 4H action window.")
        h = self.config.chunk_size
        if actions.shape[1] != h or window_actions.shape != (actions.shape[0], 4 * h, actions.shape[2]):
            raise ValueError("Residual training action/window shapes do not match H/4H.")
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
        return target_velocity[:, h:2 * h] - velocity[:, h:2 * h]

    def take_skill_window_residual(self):
        value = self._last_skill_window_residual
        self._last_skill_window_residual = None
        if value is None:
            raise RuntimeError("No residual base training forward is available.")
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
            raise ValueError("Residual sampling requires current proprio.")
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
            raise ValueError("Residual sampling noise must have H or 4H tokens.")
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
        raise ValueError("Residual_patch uses a joint 4H rollout; canonical skill-only sampling is undefined.")
