"""Small frame helpers used by Stage-1 multi-panel evaluation videos."""

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(px: int) -> ImageFont.FreeTypeFont:
    candidates = (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/google-droid-sans-fonts/DroidSans-Bold.ttf",
        "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
        "/usr/share/fonts/urw-base35/NimbusSans-Bold.otf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, px)
    fonts = list(Path("/usr/share/fonts").rglob("*.ttf"))
    if fonts:
        return ImageFont.truetype(str(fonts[0]), px)
    return ImageFont.load_default()


def even(value: int) -> int:
    return value - value % 2


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    reader = imageio.get_reader(str(path))
    fps = float(reader.get_meta_data().get("fps", 10) or 10)
    frames = [np.asarray(frame)[:, :, :3] for frame in reader]
    reader.close()
    return frames, fps


def label_bar(
    width: int,
    height: int,
    text: str,
    font: ImageFont.FreeTypeFont,
) -> np.ndarray:
    image = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    fitted_font = font
    while draw.textlength(text, font=fitted_font) > width - 8 and fitted_font.size > 8:
        if not hasattr(fitted_font, "path"):
            break
        fitted_font = ImageFont.truetype(fitted_font.path, fitted_font.size - 1)
    text_width = draw.textlength(text, font=fitted_font)
    draw.text(
        ((width - text_width) / 2, max(0, (height - fitted_font.size) / 2 - 1)),
        text,
        fill=(245, 245, 245),
        font=fitted_font,
    )
    return np.asarray(image)


def make_panel(frame: np.ndarray, height: int, bar: np.ndarray) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    panel_width = even(max(2, round(frame_width * height / frame_height)))
    resized = cv2.resize(
        frame,
        (panel_width, height),
        interpolation=cv2.INTER_AREA,
    )
    if bar.shape[1] != panel_width:
        bar = cv2.resize(
            bar,
            (panel_width, bar.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.vstack([bar, resized])
