"""One explicit crop draw shared by Predictor pixels and absolute-UV supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class PredictorTopCrop:
    height: int
    width: int
    expanded_height: int
    expanded_width: int
    y: Tensor
    x: Tensor
    model_size: int

    @classmethod
    def sample(cls, image: Tensor, *, zoom: float, training: bool, model_size: int):
        channels_first = image.shape[1] == 3
        b = image.shape[0]
        h, w = image.shape[-2:] if channels_first else image.shape[1:3]
        if min(h, w, model_size) < 2:
            raise ValueError("Predictor UV coordinates need image dimensions >= 2.")
        eh, ew = round(h * zoom), round(w * zoom)
        if training:
            y = torch.randint(eh - h + 1, (b,), device=image.device)
            x = torch.randint(ew - w + 1, (b,), device=image.device)
        else:
            y = torch.full((b,), (eh - h) // 2, device=image.device, dtype=torch.long)
            x = torch.full((b,), (ew - w) // 2, device=image.device, dtype=torch.long)
        return cls(h, w, eh, ew, y, x, model_size)

    def apply(self, image: Tensor) -> Tensor:
        channels_first = image.shape[1] == 3
        image = image if channels_first else image.permute(0, 3, 1, 2)
        b, _, h, w = image.shape
        if (h, w) != (self.height, self.width) or b != self.x.numel():
            raise ValueError("Predictor crop and image shapes do not match.")
        expanded = F.interpolate(
            image.float(), size=(self.expanded_height, self.expanded_width),
            mode="bilinear", align_corners=False,
        )
        cropped = expanded.permute(0, 2, 3, 1)[
            torch.arange(b, device=image.device)[:, None, None],
            self.y[:, None, None] + torch.arange(h, device=image.device)[None, :, None],
            self.x[:, None, None] + torch.arange(w, device=image.device)[None, None, :],
        ]
        return cropped.permute(0, 3, 1, 2) if channels_first else cropped

    def _geometry(self, uv: Tensor):
        uv = uv.float().reshape(-1, 2)
        if uv.shape[0] != self.x.numel():
            raise ValueError("Predictor crop needs one UV coordinate per image.")
        original_span = uv.new_tensor([self.width - 1, self.height - 1])
        zoom_scale = uv.new_tensor([
            self.expanded_width / self.width, self.expanded_height / self.height,
        ])
        offset = torch.stack((self.x, self.y), dim=-1).to(device=uv.device, dtype=uv.dtype)
        # Exactly mirror resize_with_pad_torch, including integer size/padding
        # and align_corners=False half-pixel geometry. The trained UV is in the
        # FINAL model image, not just in the pre-resize native crop.
        ratio = max(self.width / self.model_size, self.height / self.model_size)
        rh, rw = int(self.height / ratio), int(self.width / ratio)
        resize_scale = uv.new_tensor([rw / self.width, rh / self.height])
        padding = uv.new_tensor([(self.model_size - rw) // 2, (self.model_size - rh) // 2])
        return uv, original_span, zoom_scale, offset, resize_scale, padding

    def to_model_uv_with_clipping(
        self, original_uv: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Map original UV to the nearest finite point in the sampled crop.

        A random zoom/crop is an image augmentation, not a reason to remove a
        spatial supervision target.  Original-frame points and transformed
        crop points are therefore clamped to their respective image bounds.
        Only non-finite coordinates remain invalid.  ``clipped`` records that
        the requested point was outside either bound before clamping.
        """
        uv, span, zoom, offset, resize, pad = self._geometry(original_uv)
        finite = torch.isfinite(uv).all(-1)
        safe_uv = torch.where(finite[:, None], uv, torch.zeros_like(uv))
        original_clipped = (safe_uv.abs() > 1).any(-1)
        safe_uv = safe_uv.clamp(-1.0, 1.0)
        pixels = (safe_uv + 1) * span / 2
        cropped = (pixels + .5) * zoom - .5 - offset
        model_pixels = (cropped + .5) * resize - .5 + pad
        model_uv = 2 * model_pixels / (self.model_size - 1) - 1
        crop_clipped = (
            (cropped < 0).any(-1)
            | (cropped > span).any(-1)
            | (model_uv.abs() > 1).any(-1)
        )
        valid = finite & torch.isfinite(model_uv).all(-1)
        return model_uv.clamp(-1.0, 1.0), valid, original_clipped | crop_clipped

    def to_model_uv(self, original_uv: Tensor) -> tuple[Tensor, Tensor]:
        """Map original UV to a clamped crop target and return finite validity."""
        model_uv, valid, _ = self.to_model_uv_with_clipping(original_uv)
        return model_uv, valid

    def to_original_uv(self, model_uv: Tensor) -> Tensor:
        """Invert the actual resize/zoom/crop back to the VSA's original-frame UV."""
        uv, span, zoom, offset, resize, pad = self._geometry(model_uv)
        model_pixels = (uv + 1) * (self.model_size - 1) / 2
        cropped = (model_pixels - pad + .5) / resize - .5
        original = (cropped + offset + .5) / zoom - .5
        return 2 * original / span - 1
