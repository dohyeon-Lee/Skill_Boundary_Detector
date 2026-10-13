"""Direct 4H/H flow readouts sharing the original action output projection.

Reuse the residual family's window, conditioning, masking and rollout. The sole
architectural differences are a shared head and replacement (not addition) of
the central base velocity by the last-layer prediction.
"""

import torch

from .configuration_skill_expert import DIRECT_PATCH_ARCH_LABELS
from .residual_skill_expert import ResidualPatchSkillExpert


class DirectPatchSkillExpert(ResidualPatchSkillExpert):
    def __init__(self, config):
        if config.architecture_label not in DIRECT_PATCH_ARCH_LABELS:
            raise ValueError("Expected direct_patch_1 through direct_patch_6.")
        super().__init__(config)

    def _terminal_velocity(self, normalized_last):
        # The SAME original projection reads both the 17- and 18-layer hidden.
        # No new head, copy, zero initialization or explicit velocity residual.
        return self._action_velocity(normalized_last)

    def _combine_central_velocity(self, base, terminal):
        return terminal

    def residual_diagnostics(self, valid, real_dim):
        # Preserve the common window-forward interface, but use an independent
        # direct/ section and measure direct outputs, never residual magnitudes.
        data = self._last_residual_diagnostics
        self._last_residual_diagnostics = {}
        if not data:
            return {}
        base, direct, target = (data[k][..., :real_dim].float() for k in ("base", "delta", "target"))

        def mean(x):
            return torch.where(valid[..., None], x, 0).sum() / (valid.sum() * real_dim).clamp_min(1)

        base_mse = mean((target - base).square())
        direct_mse = mean((target - direct).square())
        values = {
            "base_velocity_rms": mean(base.square()).sqrt(),
            "direct_velocity_rms": mean(direct.square()).sqrt(),
            "direct_minus_base_rms": mean((direct - base).square()).sqrt(),
            "base_velocity_mse": base_mse,
            "final_velocity_mse": direct_mse,
            "mse_improvement": base_mse - direct_mse,
        }
        scalars = torch.stack(list(values.values())).detach().cpu().tolist()
        return {"direct/" + k: v for k, v in zip(values, scalars, strict=True)}
