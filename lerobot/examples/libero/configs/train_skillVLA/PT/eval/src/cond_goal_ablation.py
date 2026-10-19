"""Eval-only interventions at the Cond goal projection, never the Expert goal.

Both3 passes episode-grounded XYZ in metres to this projection. Off removes
the projected embedding (including its learned bias), not merely its input.
Offset changes only this projection's input; all original tensors stay intact.
"""

from __future__ import annotations

import math


def cond_goal_settings(mode="normal", offset_m=None):
    if mode is None:
        mode = "normal"
    if not isinstance(mode, str):
        raise ValueError('cond_goal_mode must be a string; quote "off" in YAML.')
    mode = mode.strip().lower()
    if mode not in {"normal", "off", "offset"}:
        raise ValueError("cond_goal_mode must be normal|off|offset.")
    if offset_m is None:
        offset_m = [0.0, 0.0, 0.0]
    if not isinstance(offset_m, (list, tuple)) or len(offset_m) != 3:
        raise ValueError("cond_goal_offset_m must contain three finite metre offsets.")
    offset = [float(value) for value in offset_m]
    if not all(math.isfinite(value) for value in offset):
        raise ValueError("cond_goal_offset_m must contain three finite metre offsets.")
    if (mode == "offset") != any(value != 0.0 for value in offset):
        raise ValueError("A nonzero cond_goal_offset_m is required exactly when cond_goal_mode=offset.")
    return mode, offset


def install_cond_goal_ablation(model, *, mode="normal", offset_m=None):
    """Install a nonpersistent hook; no checkpoint weights or RNG are changed."""
    mode, offset = cond_goal_settings(mode, offset_m)
    if mode == "normal":
        return None
    import torch

    config = model.config
    if config.architecture_label != "both_3" or config.skill_end_pose_mode != "xyz":
        raise ValueError("Cond-only goal interventions currently require a Both3 XYZ checkpoint.")
    projection = getattr(model, "cond_end_pose_condition", None)
    if projection is None or isinstance(projection, torch.nn.Identity):
        raise ValueError("The selected model has no learned Cond goal projection.")
    if model.training:
        raise ValueError("Cond goal ablations are evaluation-only; call eval() first.")
    if getattr(model, "_eval_cond_goal_hook", None) is not None:
        raise ValueError("A Cond goal intervention is already installed on this model.")

    def ensure_eval(module):
        if module.training:
            raise RuntimeError("A Cond goal ablation must never run during training.")

    if mode == "off":
        def remove_embedding(module, inputs, output):
            ensure_eval(module)
            return torch.zeros_like(output)
        handle = projection.register_forward_hook(remove_embedding)
    else:
        def shift_goal(module, inputs):
            ensure_eval(module)
            xyz = inputs[0]
            if xyz.ndim != 2 or xyz.shape[-1] != 3:
                raise ValueError("Cond goal offset expects [B,3] raw-metre XYZ.")
            return (xyz + xyz.new_tensor(offset), *inputs[1:])
        handle = projection.register_forward_pre_hook(shift_goal)
    model._eval_cond_goal_hook = handle
    return handle
