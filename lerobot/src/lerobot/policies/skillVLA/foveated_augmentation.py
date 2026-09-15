"""CPU-side foveated image construction for SkillVLA training samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def _pair(
    value: Any, *, name: str, integer: bool = False
) -> tuple[float, float] | tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value range, got {value!r}.")
    cast = int if integer else float
    low, high = cast(value[0]), cast(value[1])
    if low > high:
        raise ValueError(f"{name} minimum cannot exceed its maximum.")
    return low, high


@dataclass(frozen=True)
class FoveatedVisionAugmentationConfig:
    enabled: bool = False
    randomization_enabled: bool = False
    mode: str = "partial_fov"
    crop_size: int = 128
    output_size: int = 224
    inner_box_enabled: bool = True
    inner_box_mode: str = "blur"
    inner_box_size: int = 32
    inner_box_line_width: int = 3
    shape: str = "square"
    sharp_size: int = 96
    feather: int = 20
    peripheral_blur_radius: float = 8.0
    color_enabled: bool = False
    brightness: tuple[float, float] = (0.8, 1.2)
    contrast: tuple[float, float] = (0.8, 1.2)
    saturation: tuple[float, float] = (0.8, 1.2)
    hue: tuple[float, float] = (-0.15, 0.15)
    crop_enabled: bool = False
    crop_offset_px: tuple[int, int] = (-24, 24)
    inner_box_offset_px: tuple[int, int] = (-4, 4)
    input_blur_enabled: bool = False
    input_blur_radius: tuple[float, float] = (0.0, 4.0)

    @classmethod
    def from_mapping(
        cls, raw: dict[str, Any] | None
    ) -> FoveatedVisionAugmentationConfig:
        raw = dict(raw or {})
        config = cls(
            enabled=_as_bool(raw.get("enabled", False), name="foveation.enabled"),
            randomization_enabled=_as_bool(
                raw.get("randomization_enabled", False),
                name="foveation.randomization_enabled",
            ),
            mode=str(raw.get("mode", "partial_fov")).strip().lower(),
            crop_size=int(raw.get("crop_size", 128)),
            output_size=int(raw.get("output_size", 224)),
            inner_box_enabled=_as_bool(
                raw.get("inner_box_enabled", True),
                name="foveation.inner_box_enabled",
            ),
            inner_box_mode=str(raw.get("inner_box_mode", "blur")).strip().lower(),
            inner_box_size=int(raw.get("inner_box_size", 32)),
            inner_box_line_width=int(raw.get("inner_box_line_width", 3)),
            shape=str(raw.get("shape", "square")).strip().lower(),
            sharp_size=int(raw.get("sharp_size", 96)),
            feather=int(raw.get("feather", 20)),
            peripheral_blur_radius=float(raw.get("peripheral_blur_radius", 8.0)),
            color_enabled=_as_bool(
                raw.get("color_enabled", False), name="foveation.color_enabled"
            ),
            brightness=_pair(raw.get("brightness", (0.8, 1.2)), name="brightness"),
            contrast=_pair(raw.get("contrast", (0.8, 1.2)), name="contrast"),
            saturation=_pair(raw.get("saturation", (0.8, 1.2)), name="saturation"),
            hue=_pair(raw.get("hue", (-0.15, 0.15)), name="hue"),
            crop_enabled=_as_bool(
                raw.get("crop_enabled", False), name="foveation.crop_enabled"
            ),
            crop_offset_px=_pair(
                raw.get("crop_offset_px", (-24, 24)),
                name="crop_offset_px",
                integer=True,
            ),
            inner_box_offset_px=_pair(
                raw.get("inner_box_offset_px", (-4, 4)),
                name="inner_box_offset_px",
                integer=True,
            ),
            input_blur_enabled=_as_bool(
                raw.get("input_blur_enabled", False),
                name="foveation.input_blur_enabled",
            ),
            input_blur_radius=_pair(
                raw.get("input_blur_radius", (0.0, 4.0)),
                name="input_blur_radius",
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in {"partial_fov", "crop"}:
            raise ValueError(
                f"foveation mode must be partial_fov|crop, got {self.mode!r}."
            )
        if self.crop_size <= 0 or self.output_size <= 0:
            raise ValueError("foveation crop_size and output_size must be positive.")
        if self.inner_box_mode not in {"box", "blur"}:
            raise ValueError(
                "foveation inner_box_mode must be box|blur, got "
                f"{self.inner_box_mode!r}."
            )
        if self.inner_box_size <= 0 or self.inner_box_line_width <= 0:
            raise ValueError(
                "foveation inner_box_size and inner_box_line_width must be positive."
            )
        if self.inner_box_size > self.crop_size:
            raise ValueError("foveation inner_box_size cannot exceed crop_size.")
        if self.shape not in {"square", "circle"}:
            raise ValueError(
                f"foveation shape must be square|circle, got {self.shape!r}."
            )
        if self.sharp_size <= 0:
            raise ValueError("foveation sharp_size must be positive.")
        if self.feather < 0:
            raise ValueError("foveation feather must be non-negative.")
        if self.peripheral_blur_radius < 0:
            raise ValueError(
                "foveation peripheral_blur_radius must be non-negative."
            )
        for name, bounds in (
            ("brightness", self.brightness),
            ("contrast", self.contrast),
            ("saturation", self.saturation),
            ("input_blur_radius", self.input_blur_radius),
        ):
            if bounds[0] < 0:
                raise ValueError(f"foveation {name} values must be non-negative.")
        if self.hue[0] < -0.5 or self.hue[1] > 0.5:
            raise ValueError("foveation hue values must stay within [-0.5, 0.5].")


def _sample_uniform(bounds: tuple[float, float]) -> float:
    low, high = bounds
    if low == high:
        return float(low)
    return float(torch.empty(()).uniform_(float(low), float(high)).item())


def _sample_integer(bounds: tuple[int, int]) -> int:
    low, high = bounds
    if low == high:
        return int(low)
    return int(torch.randint(int(low), int(high) + 1, ()).item())


def _tensor_to_rgb(image: torch.Tensor) -> tuple[np.ndarray, torch.dtype]:
    tensor = torch.as_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"Foveation expects one CHW/HWC image, got {tuple(tensor.shape)}.")
    if tensor.shape[0] in {1, 3}:
        chw = tensor
    elif tensor.shape[-1] in {1, 3}:
        chw = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Foveation expects an RGB image, got {tuple(tensor.shape)}.")
    if chw.shape[0] != 3:
        raise ValueError("Foveation currently requires three-channel RGB images.")
    if chw.device.type != "cpu":
        raise ValueError("Dataset-side foveation expects CPU image tensors.")
    dtype = chw.dtype
    if dtype == torch.uint8:
        uint8 = chw
    else:
        uint8 = (chw.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return uint8.permute(1, 2, 0).contiguous().numpy(), dtype


def _rgb_to_tensor(image: np.ndarray, *, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(
        2, 0, 1
    )
    if dtype == torch.uint8:
        return tensor
    return tensor.to(dtype=dtype).div_(255.0)


def _color(
    image: np.ndarray,
    *,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> np.ndarray:
    result = Image.fromarray(image, mode="RGB")
    result = ImageEnhance.Brightness(result).enhance(brightness)
    result = ImageEnhance.Contrast(result).enhance(contrast)
    result = ImageEnhance.Color(result).enhance(saturation)
    if hue:
        hsv = np.asarray(result.convert("HSV"), dtype=np.uint8).copy()
        hsv[..., 0] = (
            hsv[..., 0].astype(np.int16) + int(round(hue * 255.0))
        ).astype(np.uint8)
        result = Image.fromarray(hsv, mode="HSV").convert("RGB")
    return np.asarray(result, dtype=np.uint8)


def _blur(image: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return image
    return np.asarray(
        Image.fromarray(image, mode="RGB").filter(
            ImageFilter.GaussianBlur(radius=radius)
        ),
        dtype=np.uint8,
    )


def _foveate(
    image: np.ndarray,
    *,
    center_xy: tuple[int, int],
    config: FoveatedVisionAugmentationConfig,
    sharp_size: int | None = None,
) -> np.ndarray:
    blurred = _blur(image, config.peripheral_blur_radius).astype(np.float32)
    source = image.astype(np.float32)
    height, width = image.shape[:2]
    x0, y0 = float(center_xy[0]), float(center_xy[1])
    yy, xx = np.mgrid[:height, :width]
    if config.shape == "circle":
        distance = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    else:
        distance = np.maximum(np.abs(xx - x0), np.abs(yy - y0))
    half = float(config.sharp_size if sharp_size is None else sharp_size) / 2.0
    if config.feather == 0:
        alpha = (distance <= half).astype(np.float32)
    else:
        alpha = np.clip(
            (half + float(config.feather) - distance) / float(config.feather),
            0.0,
            1.0,
        ).astype(np.float32)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    output = alpha[..., None] * source + (1.0 - alpha[..., None]) * blurred
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def _crop_bounds(
    *,
    center_xy: tuple[int, int],
    crop_size: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"foveation crop_size={crop_size} exceeds source image {width}x{height}."
        )
    left = int(round(float(center_xy[0]) - crop_size / 2.0))
    top = int(round(float(center_xy[1]) - crop_size / 2.0))
    left = int(np.clip(left, 0, width - crop_size))
    top = int(np.clip(top, 0, height - crop_size))
    return left, top, left + crop_size, top + crop_size


def _box_bounds_inside(
    *,
    center_xy: tuple[int, int],
    box_size: int,
    outer_bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    outer_left, outer_top, outer_right, outer_bottom = outer_bounds
    if box_size > outer_right - outer_left or box_size > outer_bottom - outer_top:
        raise ValueError(
            f"foveation inner_box_size={box_size} exceeds its outer crop."
        )
    left = int(round(float(center_xy[0]) - box_size / 2.0))
    top = int(round(float(center_xy[1]) - box_size / 2.0))
    left = int(np.clip(left, outer_left, outer_right - box_size))
    top = int(np.clip(top, outer_top, outer_bottom - box_size))
    return left, top, left + box_size, top + box_size


def _crop_focus(
    image: np.ndarray,
    *,
    crop_center_xy: tuple[int, int],
    inner_center_xy: tuple[int, int],
    config: FoveatedVisionAugmentationConfig,
) -> np.ndarray:
    """Build the production crop+inner-cue input used by Stage 1."""
    height, width = image.shape[:2]
    bounds = _crop_bounds(
        center_xy=crop_center_xy,
        crop_size=config.crop_size,
        height=height,
        width=width,
    )
    crop = np.asarray(Image.fromarray(image, mode="RGB").crop(bounds), dtype=np.uint8)
    inner = None
    if config.inner_box_enabled:
        inner = _box_bounds_inside(
            center_xy=inner_center_xy,
            box_size=config.inner_box_size,
            outer_bounds=bounds,
        )
        if config.inner_box_mode == "blur":
            local_center = (
                int(round((inner[0] + inner[2]) / 2.0 - bounds[0])),
                int(round((inner[1] + inner[3]) / 2.0 - bounds[1])),
            )
            crop = _foveate(
                crop,
                center_xy=local_center,
                config=config,
                sharp_size=config.inner_box_size,
            )

    resampling = getattr(Image, "Resampling", Image).BICUBIC
    output = Image.fromarray(crop, mode="RGB").resize(
        (config.output_size, config.output_size),
        resample=resampling,
    )
    if config.inner_box_enabled and config.inner_box_mode == "box":
        assert inner is not None
        scale = float(config.output_size) / float(config.crop_size)
        box = (
            int(round((inner[0] - bounds[0]) * scale)),
            int(round((inner[1] - bounds[1]) * scale)),
            int(round((inner[2] - bounds[0]) * scale)) - 1,
            int(round((inner[3] - bounds[1]) * scale)) - 1,
        )
        ImageDraw.Draw(output).rectangle(
            box,
            outline=(255, 0, 0),
            width=config.inner_box_line_width,
        )
    return np.asarray(output, dtype=np.uint8)


def augment_camera_pair(
    top_image: torch.Tensor,
    wrist_image: torch.Tensor,
    focus_uv: torch.Tensor | np.ndarray | None,
    config: FoveatedVisionAugmentationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Foveate top view; share color/input-blur draws with the wrist view."""
    if not config.enabled and not config.randomization_enabled:
        return top_image, wrist_image
    top, top_dtype = _tensor_to_rgb(top_image)
    wrist, wrist_dtype = _tensor_to_rgb(wrist_image)

    if config.randomization_enabled and config.color_enabled:
        parameters = {
            "brightness": _sample_uniform(config.brightness),
            "contrast": _sample_uniform(config.contrast),
            "saturation": _sample_uniform(config.saturation),
            "hue": _sample_uniform(config.hue),
        }
        top = _color(top, **parameters)
        wrist = _color(wrist, **parameters)
    if config.randomization_enabled and config.input_blur_enabled:
        radius = _sample_uniform(config.input_blur_radius)
        top = _blur(top, radius)
        wrist = _blur(wrist, radius)

    if config.enabled:
        uv = np.asarray(
            focus_uv.detach().cpu() if torch.is_tensor(focus_uv) else focus_uv,
            dtype=np.float32,
        ).reshape(-1)
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise ValueError(
                f"skill_focus_uv must contain two finite values, got {uv}."
            )
        height, width = top.shape[:2]
        center_x = int(
            round(float(np.clip((uv[0] + 1.0) * 0.5, 0.0, 1.0)) * (width - 1))
        )
        center_y = int(
            round(
                float(np.clip((uv[1] + 1.0) * 0.5, 0.0, 1.0))
                * (height - 1)
            )
        )
        canonical_center = (center_x, center_y)
        crop_center = canonical_center
        inner_center = canonical_center
        if config.randomization_enabled and config.crop_enabled:
            crop_center = (
                int(
                    np.clip(
                        center_x + _sample_integer(config.crop_offset_px),
                        0,
                        width - 1,
                    )
                ),
                int(
                    np.clip(
                        center_y + _sample_integer(config.crop_offset_px),
                        0,
                        height - 1,
                    )
                ),
            )
            if config.mode == "crop" and config.inner_box_enabled:
                inner_center = (
                    int(
                        np.clip(
                            center_x + _sample_integer(config.inner_box_offset_px),
                            0,
                            width - 1,
                        )
                    ),
                    int(
                        np.clip(
                            center_y + _sample_integer(config.inner_box_offset_px),
                            0,
                            height - 1,
                        )
                    ),
                )
        if config.mode == "crop":
            top = _crop_focus(
                top,
                crop_center_xy=crop_center,
                inner_center_xy=inner_center,
                config=config,
            )
        else:
            top = _foveate(top, center_xy=crop_center, config=config)
    return (
        _rgb_to_tensor(top, dtype=top_dtype),
        _rgb_to_tensor(wrist, dtype=wrist_dtype),
    )
