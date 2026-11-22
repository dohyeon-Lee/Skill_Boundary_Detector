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
            capture.pop("wrist", None)
            capture.pop("right_wrist", None)
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
    bimanual = getattr(getattr(model, "config", None), "bimanual_spatial_auxiliary", False)
    names = ("top", "wrist", "right_wrist") if bimanual else ("top", "wrist")
    if hidden.shape[1] % len(names):
        raise ValueError("Video auxiliary memory must split evenly across cameras")
    width = hidden.shape[1] // len(names)
    for index, name in enumerate(names):
        start = index * width + int(cls_tokens)
        capture[name] = hidden[:, start:(index + 1) * width].detach()


def _patch_box(logits):
    side = math.isqrt(logits.shape[-1])
    if not side or side * side != logits.shape[-1]:
        return None
    index = logits.argmax(-1)
    row, col = index // side, index % side
    box = torch.stack((col, row, col + 1, row + 1), -1).float() * (2 / side) - 1
    box[~torch.isfinite(logits).all(-1)] = float("nan")
    return box.cpu().numpy()


@torch.no_grad()
def capture_latent(model, latent, *, tower="upper", prefix=""):
    capture = getattr(model, "_video_aux_capture", None)
    if capture is None or model.training:
        return
    if getattr(model.config, "bimanual_spatial_auxiliary", False):
        # Match the trained ABC ordering: Top-L, Top-R, Wrist-L, Wrist-R, XYZ6.
        if getattr(model, "_alignment_token_count", 0) != 5:
            return
        values = {}
        for memory_name, query_index, head_name, field in (
            ("top", 0, "agent_patch_align_head", "top_left_patch_box_uv"),
            ("top", 1, "agent_patch_align_head", "top_right_patch_box_uv"),
            ("wrist", 2, "wrist_patch_align_head", "left_patch_box_uv"),
            ("right_wrist", 3, "wrist_patch_align_head", "right_patch_box_uv"),
        ):
            memory, head = capture.get(memory_name), getattr(model, prefix + head_name, None)
            if memory is not None and callable(head):
                box = _patch_box(head(latent[:, query_index:query_index + 1].detach(), memory).float())
                if box is not None:
                    values[field] = box
        head = getattr(model, prefix + "agent_end_state_head", None)
        if callable(head):
            values["xyz_normalized"] = head(latent[:, 4].detach().float()).float().cpu().numpy()
        capture["outputs"][tower] = values
        return
    if getattr(model, "_alignment_token_count", 0) != 3:
        return
    xyz_head = getattr(model, prefix + "agent_end_state_head", None)
    values = {}
    for camera, query_index, head_name in (("top", 0, "agent_patch_align_head"),
                                            ("wrist", 1, "wrist_patch_align_head")):
        memory = capture.get(camera)
        patch_head = getattr(model, prefix + head_name, None)
        if memory is None or not callable(patch_head):
            continue
        logits = patch_head(latent[:, query_index:query_index + 1].detach(), memory).float()
        box = _patch_box(logits)
        if box is not None:
            values[f"{camera}_patch_box_uv"] = box
    if callable(xyz_head) and (not getattr(model.config, "uses_gemma_xyz_readout", False)
                               or getattr(model.config, "uses_separate_vision_tower_xyz", False)):
        values["xyz_normalized"] = xyz_head(latent[:, 2].detach().float()).float().cpu().numpy()
    capture["outputs"][tower] = values


def decode_auxiliaries(outputs, config, origin=None):
    """Decode normalized XYZ once to world metres; never substitute a GT goal."""
    if getattr(config, "bimanual_spatial_auxiliary", False):
        decoded = {}
        # ABC supervises absolute world XYZ6 with separate bounds, not joint14
        # plan quantiles and not LIBERO's episode-grounded proprio bounds.
        lo = np.asarray(getattr(config, "bimanual_goal_xyz_q01", None), dtype=float).reshape(-1)
        hi = np.asarray(getattr(config, "bimanual_goal_xyz_q99", None), dtype=float).reshape(-1)
        for tower, values in outputs.items():
            row = {name: value for name, value in values.items()
                   if name.endswith("_patch_box_uv") and not getattr(config, "foveated_vision_enabled", False)}
            if ("xyz_normalized" in values and lo.size == hi.size == 6
                    and np.isfinite([lo, hi]).all() and (hi > lo).all()):
                xyz = (np.asarray(values["xyz_normalized"]).reshape(-1, 6) + 1) * .5 * (hi - lo) + lo
                row["xyz_world"] = xyz.reshape(-1, 2, 3)
            if row:
                decoded[tower] = row
        return decoded
    decoded = {}
    lo = np.asarray(getattr(config, "chunk_end_state_q01", None), dtype=float).reshape(-1)[:3]
    hi = np.asarray(getattr(config, "chunk_end_state_q99", None), dtype=float).reshape(-1)[:3]
    grounded = getattr(config, "proprio_grounding", "none") == "episode_start_xyz"
    for tower, values in outputs.items():
        row = {}
        # A cropped local patch is not a global rollout patch. These models
        # normally prohibit foveation; skip rather than misproject old variants.
        for field in ("top_patch_box_uv", "wrist_patch_box_uv"):
            if field in values and not getattr(config, "foveated_vision_enabled", False):
                row[field] = values[field]
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
