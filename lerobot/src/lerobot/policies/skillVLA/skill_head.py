"""Stage-2 VLM skill head — decode the skill-query hidden into an FSQ skill code.

The skill is an FSQ code: a flat index in [0, prod(levels)) ≡ a per-dim tuple (d_0..d_{D-1}) with
d_i in [0, levels[i]). The head predicts ONE categorical per dim (sizes = levels), so:
  - train : CE per dim against the GT code's per-dim decomposition, averaged over dims
  - infer : argmax per dim → flat code (fed to the action expert's skill embedding)

The flat↔per-dim mapping is a self-consistent mixed-radix (row-major, like np.unravel_index): the
head only needs an invertible mapping, since the action expert keys its skill embedding by the flat
code (an integer lookup), not by FSQ geometry.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SkillHead(nn.Module):
    def __init__(self, hidden_dim: int, fsq_levels: list[int]):
        super().__init__()
        self.levels = list(fsq_levels)
        self.proj = nn.Linear(hidden_dim, sum(self.levels))
        # row-major strides: code = sum_i d_i * stride_i, stride_i = prod(levels[i+1:])
        strides: list[int] = []
        acc = 1
        for lv in reversed(self.levels):
            strides.append(acc)
            acc *= lv
        self.register_buffer("_strides", torch.tensor(list(reversed(strides)), dtype=torch.long), persistent=False)

    def _logits(self, hidden: Tensor) -> list[Tensor]:
        """(B, hidden) → list of D tensors (B, levels[d])."""
        return list(torch.split(self.proj(hidden), self.levels, dim=-1))

    def _unravel(self, code: Tensor) -> Tensor:
        """flat code (B,) → per-dim indices (B, D)."""
        code = code.long().view(-1, 1)
        return (code // self._strides) % torch.tensor(self.levels, device=code.device)

    def _ravel(self, per_dim: Tensor) -> Tensor:
        """per-dim indices (B, D) → flat code (B,)."""
        return (per_dim.long() * self._strides).sum(dim=-1)

    def loss(self, hidden: Tensor, code: Tensor) -> Tensor:
        """Mean over dims of per-dim CE (hidden vs GT flat code)."""
        target = self._unravel(code)  # (B, D)
        logits = self._logits(hidden)
        return torch.stack([F.cross_entropy(logits[d], target[:, d]) for d in range(len(self.levels))]).mean()

    @torch.no_grad()
    def decode(self, hidden: Tensor) -> Tensor:
        """argmax per dim → flat FSQ code (B,)."""
        per_dim = torch.stack([lg.argmax(dim=-1) for lg in self._logits(hidden)], dim=-1)  # (B, D)
        return self._ravel(per_dim)
