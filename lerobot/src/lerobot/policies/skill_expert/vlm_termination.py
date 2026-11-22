"""Termination objective over the shared VSA VLM; never resample VSA frames."""
import torch
from torch.nn import functional as F

from lerobot.policies.skillVLA.vlm_termination_supervision import (
    DISTANCE, TERMINAL, PAST_CODES, PAST_VALID, PAST_UV, PAST_TERMINAL, PAST_DISTANCE, termination_targets,
)


def vlm_termination_loss(model, config, batch, state):
    logits, memory = model.take_vlm_termination_outputs()
    if logits is None or memory is None or logits.shape != (state.shape[0],):
        raise RuntimeError("VLM termination requires the current action forward's prefix outputs.")
    target, valid = termination_targets(
        batch[DISTANCE].to(logits.device), sigma=config.vlm_termination_target_sigma,
        post_end_steps=config.vlm_termination_post_end_steps,
    )
    positive_weight = logits.new_tensor(config.vlm_termination_positive_weight)
    current = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight, reduction="none")
    current = current * valid
    # Exactly the standalone sampler's k final-frame repeats, confined to the
    # termination loss. Action/fullskill sampling and loss stay unchanged.
    terminal = batch[TERMINAL].to(logits.device).bool().reshape(-1)
    weights = 1 + config.vlm_termination_post_end_steps * terminal.float()
    current = current * weights
    condition_weight = (valid.float() * weights).sum()
    extra = torch.zeros_like(current)
    rows, slots = batch[PAST_VALID].to(logits.device).bool().nonzero(as_tuple=True)
    probabilities = []
    # Bound the extra prefix batch by the normal VSA batch size. Cached SigLIP
    # embeddings retain gradients; past conditions never run an Action Expert.
    for start in range(0, rows.numel(), state.shape[0]):
        r, s = rows[start:start + state.shape[0]], slots[start:start + state.shape[0]]
        code = batch[PAST_CODES].to(logits.device)[r, s]
        uv = batch[PAST_UV].to(logits.device)[r, s]
        goal = torch.cat((uv, torch.ones_like(uv[:, :1]), uv.new_zeros(uv.shape[0], 3)), dim=-1)
        past = model.vlm_termination_from_image_tokens(memory[r], state[r], code, goal)
        extra_target, extra_valid = termination_targets(
            batch[PAST_DISTANCE].to(logits.device)[r, s], sigma=config.vlm_termination_target_sigma,
            post_end_steps=config.vlm_termination_post_end_steps,
        )
        losses = F.binary_cross_entropy_with_logits(past, extra_target,
                                                  pos_weight=positive_weight, reduction="none")
        past_terminal = batch[PAST_TERMINAL].to(logits.device)[r, s].float()
        extra_weight = extra_valid.float() * (1 + config.vlm_termination_post_end_steps * past_terminal)
        losses = losses * extra_weight
        condition_weight = condition_weight + extra_weight.sum()
        extra = extra.index_add(0, r, losses)
        positives = extra_valid & (extra_target >= .5)
        if positives.any():
            probabilities.append(past.detach().sigmoid()[positives])
    # A condition is one termination example, irrespective of how many
    # current/past skills are evaluated at a frame. Preserve reduction='none'
    # while making its mean the global weighted BCE over valid conditions.
    scale = state.shape[0] / condition_weight.clamp_min(1)
    current, extra = current * scale, extra * scale
    with torch.no_grad():
        predicted, truth = logits.sigmoid() >= .5, target >= .5
        count = valid.sum().clamp_min(1)
        tp = (predicted & truth & valid).sum().float()
        fp = (predicted & ~truth & valid).sum().float()
        fn = (~predicted & truth & valid).sum().float()
        metrics = {
            "vlm_termination/loss": (current + extra).mean().item(),
            "vlm_termination/current_loss": current.mean().item(),
            "vlm_termination/post_end_loss": extra.mean().item(),
            "vlm_termination/current_valid_fraction": valid.float().mean().item(),
            "vlm_termination/positive_fraction": (truth & valid).sum().float().div(count).item(),
            "vlm_termination/accuracy": ((predicted == truth) & valid).sum().float().div(count).item(),
            "vlm_termination/precision": (tp / (tp + fp).clamp_min(1)).item(),
            "vlm_termination/recall": (tp / (tp + fn).clamp_min(1)).item(),
            "vlm_termination/extra_conditions_per_frame": rows.numel() / state.shape[0],
            "vlm_termination/post_end_conditions_per_frame": (
                (batch[PAST_DISTANCE].to(logits.device)[rows, slots] <= 0).sum().item() / state.shape[0]
                if rows.numel() else 0.0
            ),
        }
        if probabilities:
            metrics["vlm_termination/post_end_probability"] = torch.cat(probabilities).mean().item()
    return current + extra, metrics
