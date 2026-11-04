"""Independent activation-checkpoint selection without changing parameter names."""
from __future__ import annotations

from functools import lru_cache
from types import MethodType

COMPONENTS = ("action_expert", "skill_planner", "vlm", "vision")


def validate_selection(value):
    if value is None or isinstance(value, bool):
        return
    if type(value) is int and value >= 0:
        return
    if isinstance(value, list) and all(type(item) is int and item > 0 for item in value):
        if len(set(value)) == len(value):
            return
    raise ValueError("Checkpoint selection must be false/0, true, a non-negative last-N count, or unique 1-based layer numbers.")


def parse_checkpointing(value):
    """Legacy bool keeps its old semantics; a mapping independently controls all streams."""
    if isinstance(value, bool):
        return value, {name: None for name in COMPONENTS}
    if not isinstance(value, dict) or set(value) - set(COMPONENTS):
        raise ValueError("training.gradient_checkpointing must be a bool or a mapping of action_expert, skill_planner, vlm, vision.")
    selections = {name: value.get(name, False) for name in COMPONENTS}
    for selection in selections.values():
        if selection is None:
            raise ValueError("Explicit checkpoint selections cannot be null; use false to disable.")
        validate_selection(selection)
    return False, selections


@lru_cache(maxsize=128, typed=True)
def _indices(value, depth):
    if isinstance(value, bool):
        return frozenset(range(depth)) if value else frozenset()
    if isinstance(value, int):
        if value > depth:
            raise ValueError(f"Checkpoint last-N count {value} exceeds this stream's {depth} layers.")
        return frozenset(range(depth - value, depth))
    if any(number > depth for number in value):
        raise ValueError(f"Checkpoint layer selection {value} exceeds this stream's {depth} layers.")
    return frozenset(number - 1 for number in value)


def checkpoint_indices(value, depth):
    validate_selection(value)
    if value is None:
        raise ValueError("Resolve legacy checkpoint defaults before selecting layers.")
    return _indices(tuple(value) if isinstance(value, list) else value, depth)


def _vision_forward(module, *args, **kwargs):
    import torch
    import torch.utils.checkpoint
    original = module._sbd_checkpoint_original_forward
    if module.training and torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(original, *args, use_reentrant=False,
                                                 preserve_rng_state=True, **kwargs)
    return original(*args, **kwargs)


def configure_vision_checkpointing(model, selection):
    """Wrap selected DINO/SigLIP blocks in place; state_dict keys stay unchanged."""
    import torch.nn as nn
    blocks = None
    for path in ("vision_model.encoder.layers", "encoder.layers", "encoder.layer", "layer", "layers"):
        candidate = model
        for part in path.split("."):
            candidate = getattr(candidate, part, None)
        if isinstance(candidate, nn.ModuleList):
            blocks = candidate
            break
    if blocks is None:
        if selection is False or selection == 0 or selection == []:
            return
        raise ValueError("Cannot locate vision encoder blocks for selective checkpointing.")
    selected = checkpoint_indices(selection, len(blocks))
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    for index, block in enumerate(blocks):
        if hasattr(block, "_sbd_checkpoint_original_forward"):
            block.forward = block._sbd_checkpoint_original_forward
            del block._sbd_checkpoint_original_forward
        if index in selected:
            block._sbd_checkpoint_original_forward = block.forward
            block.forward = MethodType(_vision_forward, block)
