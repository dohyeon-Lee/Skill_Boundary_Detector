"""One Top-camera crop shared by VSA images, UV conditions and UV objectives."""
from dataclasses import replace
import math

import torch

from .predictor_image_crop import PredictorTopCrop

TOP_IMAGE = "observation.images.image"
UV_FIELDS = (
    ("skill_focus_uv", "skill_focus_valid", "skill_focus_clipped"),
    ("skill_chunk_end_focus_uv", "skill_chunk_end_focus_valid", None),
    ("vlm_termination_past_uv", "vlm_termination_past_uv_valid", None),
)


def map_uv(crop, value, *, inverse=False):
    """Transform any number of UV points per sample with that sample's crop."""
    shape = value.shape
    if shape[0] != crop.x.numel() or shape[-1] != 2:
        raise ValueError(f"Top crop expects UV [B,...,2], got {shape}.")
    count = math.prod(shape[1:-1])
    expanded = replace(crop, x=crop.x.repeat_interleave(count), y=crop.y.repeat_interleave(count))
    if inverse:
        return expanded.to_original_uv(value.reshape(-1, 2)).reshape(shape)
    uv, valid, clipped = expanded.to_model_uv_with_clipping(value.reshape(-1, 2))
    return uv.reshape(shape), valid.reshape(shape[:-1]), clipped.reshape(shape[:-1])


def prepare_top_crop(batch, config, *, training, crop=None):
    """Return a copy. Wrist images, XYZ, proprio, actions and start images stay intact."""
    if not getattr(config, "vsa_top_crop_enabled", False):
        return batch, None
    if "_vsa_top_crop" in batch:
        return batch, batch["_vsa_top_crop"]
    image = batch[TOP_IMAGE]
    crop = crop or PredictorTopCrop.sample(
        image, zoom=config.vsa_top_crop_zoom, training=training,
        model_size=config.dino_image_size,
    )
    result = dict(batch)
    result[TOP_IMAGE] = crop.apply(image)
    result["_vsa_top_crop"] = crop
    for key, valid_key, clipped_key in UV_FIELDS:
        if key not in batch:
            continue
        uv, valid, clipped = map_uv(crop, batch[key])
        result[key] = uv
        if valid_key in batch:
            result[valid_key] = batch[valid_key].bool() & valid
        if clipped_key is not None:
            result[clipped_key] = batch.get(clipped_key, torch.zeros_like(clipped)).bool() | clipped
    if "skill_focus_uv_pixels" in result:
        result["skill_focus_uv_pixels"] = (result["skill_focus_uv"] + 1) * (crop.model_size - 1) / 2
    return result, crop
