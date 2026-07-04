"""Stage-2 VLM skill head — REGRESSION (FSQ-style) decode of the skill-query hidden into an FSQ code.

The skill is an FSQ code: a flat index in [0, prod(levels)) ≡ a per-dim grid coordinate. This head
REGRESSES the (normalized) FSQ grid coordinate z_q ∈ [-1, 1]^D and rounds to the nearest per-dim level,
mirroring the FSQ encoder's quantization. Unlike a per-dim categorical head, this preserves FSQ's
ordinal/metric structure (nearby codes ⇒ nearby z_q ⇒ similar motion), so near-misses stay gentle.

  - train : MSE between predicted z_q and the GT code's normalized grid coordinate, with an optional
            per-dim DEAD-ZONE (``deadzone_frac`` > 0): once a dim's |pred - target| is within a margin
            (the code already rounds correctly, with a safety band), that dim's loss is zeroed → training
            stops nudging a code that already decodes right (faster convergence, no post-correct drift).
            The round is NOT on the gradient path (loss is on the continuous prediction) → no straight-
            through needed here. (Finetuning, where the predicted vector feeds a frozen decoder, adds an
            STE/round path on top of this head.)
  - infer : round predicted z_q to the nearest per-dim level → flat code.

Geometry matches the FSQ codebook / ``SkillVLAPytorch._code_to_z`` EXACTLY (little-endian strides,
half = (L-1)/2), so the predicted flat code lives in the same space as the GT codes and the downstream
code→z_q conditioning, and the regression axes coincide with the true FSQ axes (ordinal alignment).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SkillHead(nn.Module):
    def __init__(self, hidden_dim: int, fsq_levels: list[int], deadzone_frac: float = 0.0):
        super().__init__()
        self.levels = list(fsq_levels)
        D = len(self.levels)
        self.proj = nn.Linear(hidden_dim, D)  # hidden → D continuous FSQ coords (pre-tanh)
        half = torch.tensor([(L - 1) / 2.0 for L in self.levels], dtype=torch.float32)
        self.register_buffer("_half", half, persistent=False)
        # Dead-zone margin (normalized coord frame) = frac × rounding-bucket half-width. bucket half-width
        # = 1/(L-1) (grid spacing 2/(L-1)) → frac=1 zeros the loss as soon as the code rounds right; frac<1
        # keeps training until pred is comfortably inside the bucket (safety band). Codebook-independent.
        self.deadzone_frac = float(deadzone_frac)
        lv = torch.tensor(self.levels, dtype=torch.float32)
        self.register_buffer("_deadzone_margin", self.deadzone_frac / (lv - 1.0).clamp(min=1.0), persistent=False)
        # little-endian strides (match the FSQ codebook / _code_to_z): stride_i = prod(levels[:i]).
        strides = torch.ones(D, dtype=torch.long)
        for i in range(1, D):
            strides[i] = strides[i - 1] * self.levels[i - 1]
        self.register_buffer("_strides", strides, persistent=False)
        self.register_buffer("_levels", torch.tensor(self.levels, dtype=torch.long), persistent=False)

    def _pred_z(self, hidden: Tensor) -> Tensor:
        """(B, hidden) → predicted normalized FSQ coord z_q ∈ (-1, 1)^D."""
        return torch.tanh(self.proj(hidden))

    def _code_to_norm_z(self, code: Tensor) -> Tensor:
        """flat code (B,) → normalized grid coord (B, D) ∈ [-1, 1] (little-endian; matches _code_to_z/half)."""
        idx = code.long().view(-1, 1)
        level_id = (idx // self._strides) % self._levels  # (B, D) ∈ [0, L)
        return (level_id.float() - self._half) / self._half

    def loss(self, hidden: Tensor, code: Tensor) -> Tensor:
        """Mean (dead-zoned) MSE between predicted z_q and the GT code's normalized grid coordinate.

        deadzone_frac == 0 → plain MSE. Else per-dim SE is zeroed where |pred - target| < margin (the code
        already decodes right with a safety band); the mean is over the SAME B×D count as plain MSE, so the
        total loss → 0 as more dims land correct (stops updating solved codes)."""
        pred = self._pred_z(hidden)
        target = self._code_to_norm_z(code).to(pred.dtype)
        if self.deadzone_frac <= 0.0:
            return F.mse_loss(pred, target)
        se = (pred - target) ** 2
        inside = (pred - target).abs() < self._deadzone_margin.to(pred.dtype)
        return torch.where(inside, torch.zeros_like(se), se).mean()

    @torch.no_grad()
    def decode(self, hidden: Tensor) -> Tensor:
        """Round predicted z_q to the nearest per-dim level → flat FSQ code (B,)."""
        pred = self._pred_z(hidden).float()
        level_id = torch.round((pred + 1.0) * self._half).clamp_(min=0.0)  # (pred+1)*half ∈ [0, L-1]
        level_id = torch.minimum(level_id, (self._levels - 1).to(level_id.dtype)).long()
        return (level_id * self._strides).sum(dim=-1)
