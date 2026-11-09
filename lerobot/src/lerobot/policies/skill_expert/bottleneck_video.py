"""Read-only auxiliary-head capture for rendered action chunks, never training.

No hooks, parameters, training stashes or extra vision forwards are installed.
The context is deliberately limited to the real action call, not plan probes.
"""
from contextlib import contextmanager
import math

import numpy as np
import torch


@contextmanager
def capture_bottleneck(model, enabled=True):
    previous = getattr(model, "_video_aux_capture", None)
    capture = {"outputs": {}} if enabled and model is not None and not getattr(model, "training", True) else None
    if model is not None:
        model._video_aux_capture = capture
    try:
        yield capture
    finally:
        if capture is not None:
            capture.pop("top", None)  # Do not retain full GPU memory between chunks.
        if model is not None:
            if previous is None:
                del model._video_aux_capture
            else:
                model._video_aux_capture = previous


def capture_condition(model, hidden, *, cls_tokens=True):
    capture = getattr(model, "_video_aux_capture", None)
    if capture is None or model.training:
        return
    # This is the final image memory used by the trained patch heads.
    half = hidden.shape[1] // 2
    capture["top"] = hidden[:, 1:half].detach() if cls_tokens else hidden[:, :half].detach()


@torch.no_grad()
def capture_latent(model, latent, *, tower="upper", prefix=""):
    capture = getattr(model, "_video_aux_capture", None)
    if capture is None or model.training or getattr(model, "_alignment_token_count", 0) != 3:
        return
    top = capture.get("top")
    patch_head = getattr(model, prefix + "agent_patch_align_head", None)
    xyz_head = getattr(model, prefix + "agent_end_state_head", None)
    values = {}
    if top is not None and callable(patch_head):
        logits = patch_head(latent[:, :1].detach(), top).float()
        side = math.isqrt(logits.shape[-1])
        if side and side * side == logits.shape[-1]:
            index = logits.argmax(-1)
            row, col = index // side, index % side
            box = torch.stack((col, row, col + 1, row + 1), -1).float() * (2 / side) - 1
            box[~torch.isfinite(logits).all(-1)] = float("nan")
            values["top_patch_box_uv"] = box.cpu().numpy()
    if callable(xyz_head) and (not getattr(model.config, "uses_gemma_xyz_readout", False)
                               or getattr(model.config, "uses_separate_vision_tower_xyz", False)):
        values["xyz_normalized"] = xyz_head(latent[:, 2].detach().float()).float().cpu().numpy()
    capture["outputs"][tower] = values


def decode_auxiliaries(outputs, config, origin=None):
    """Decode normalized XYZ once to world metres; never substitute a GT goal."""
    decoded = {}
    lo = np.asarray(getattr(config, "chunk_end_state_q01", None), dtype=float).reshape(-1)[:3]
    hi = np.asarray(getattr(config, "chunk_end_state_q99", None), dtype=float).reshape(-1)[:3]
    grounded = getattr(config, "proprio_grounding", "none") == "episode_start_xyz"
    for tower, values in outputs.items():
        row = {}
        # A cropped local patch is not a global rollout patch. These models
        # normally prohibit foveation; skip rather than misproject old variants.
        if "top_patch_box_uv" in values and not getattr(config, "foveated_vision_enabled", False):
            row["top_patch_box_uv"] = values["top_patch_box_uv"]
        if ("xyz_normalized" in values and lo.size == hi.size == 3
                and np.isfinite([lo, hi]).all() and (hi > lo).all()
                and (not grounded or origin is not None)):
            xyz = (values["xyz_normalized"][..., :3] + 1) * .5 * (hi - lo) + lo
            if grounded:
                xyz = xyz + np.asarray(origin).reshape(-1, 3)
            row["xyz_world"] = xyz
        if row:
            decoded[tower] = row
    return decoded
