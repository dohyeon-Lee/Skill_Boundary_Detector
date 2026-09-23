"""Training-only patch-alignment readout: which wrist patch holds the skill-end EEF.

The point is the vision encoder, not the answer. Forcing a bottleneck query to pick out the patch
that contains the goal makes the visual features carry *where* things are, instead of only what the
action head happens to need. Nothing here runs at inference: ``arch16_align``/``arch17_align``/
``arch18_align`` deploy exactly like ``arch16``/``arch17``/``arch18``.

The head is a single dot product between one pooled bottleneck query and every patch feature, so
its logits are a heat map that can be laid over the wrist frame when debugging. Targets come from
:mod:`wrist_patch_target`, which also decides which frames carry loss at all.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from lerobot.policies.skill_expert.wrist_patch_target import soft_targets


class WristPatchAlignmentHead(nn.Module):
    """Pooled bottleneck query x patch features -> one logit per patch.

    ``query_width`` is the bottleneck latent's width and ``patch_width`` the width of the visual
    patch tokens; ``width`` is the small space the two are compared in.
    """

    def __init__(self, query_width: int, patch_width: int, *, width: int = 128) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(query_width)
        self.query_score = nn.Linear(query_width, 1)
        self.query_proj = nn.Linear(query_width, width)
        self.patch_norm = nn.LayerNorm(patch_width)
        self.patch_proj = nn.Linear(patch_width, width)
        self.scale = 1.0 / math.sqrt(width)

    def forward(self, query_tokens: Tensor, patch_features: Tensor) -> Tensor:
        """``[batch, patches]`` logits from ``[batch, tokens, query_width]`` and ``[batch, patches, patch_width]``."""
        if query_tokens.ndim != 3 or patch_features.ndim != 3:
            raise ValueError(
                "Expected [batch, tokens, width] tensors, got "
                f"{tuple(query_tokens.shape)} and {tuple(patch_features.shape)}."
            )
        if query_tokens.shape[0] != patch_features.shape[0]:
            raise ValueError(
                "Query and patch batches differ: "
                f"{query_tokens.shape[0]} != {patch_features.shape[0]}."
            )
        normalized = self.query_norm(query_tokens.float())
        weights = self.query_score(normalized).softmax(dim=1)
        query = self.query_proj((weights * normalized).sum(dim=1))
        patches = self.patch_proj(self.patch_norm(patch_features.float()))
        return torch.einsum("bd,bpd->bp", query, patches) * self.scale


def wrist_patch_alignment_loss(
    logits: Tensor,
    label: Tensor,
    valid: Tensor,
    *,
    grid: int,
    sigma: float,
) -> tuple[Tensor, dict[str, float]]:
    """Soft cross-entropy over the patch grid, averaged over the frames that carry loss.

    Frames where the goal is behind the camera, outside the frame, or inside the gripper's own
    patch are dropped (``valid`` is False) exactly as ``mask_actions_after_skill_end`` drops the
    actions past a skill's end. The result is divided by ``ln(patches)`` so a chance-level head
    starts at ~1.0 whatever the grid, which keeps one shared ``spatial_loss_weight`` meaningful.
    """
    patches = grid * grid
    if logits.shape[-1] != patches:
        raise ValueError(f"Expected {patches} logits for a {grid}x{grid} grid, got {logits.shape[-1]}.")
    logits = logits.float()
    counted = int(valid.sum())
    metrics = {"wrist_patch/valid_fraction": counted / max(int(valid.numel()), 1)}
    if counted == 0:
        # Keep the parameters in the graph so DDP still sees a gradient for this head.
        return logits.sum() * 0.0, metrics

    logits, label = logits[valid], label[valid]
    targets = soft_targets(label, grid=grid, sigma=sigma)
    loss = -(targets * logits.log_softmax(dim=-1)).sum(dim=-1).mean() / math.log(patches)

    with torch.no_grad():
        predicted = logits.argmax(dim=-1)
        distance = torch.maximum(
            (predicted // grid - label // grid).abs(), (predicted % grid - label % grid).abs()
        ).float()
        metrics.update(
            {
                "wrist_patch/cell_accuracy": float((predicted == label).float().mean()),
                "wrist_patch/within_one_cell": float((distance <= 1.0).float().mean()),
                "wrist_patch/cell_distance": float(distance.mean()),
            }
        )
    return loss, metrics
