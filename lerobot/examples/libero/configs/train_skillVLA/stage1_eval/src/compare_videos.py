"""Frame helpers shared by the Stage-1 and pi05 multi-panel evaluation videos."""

import logging
import os
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


def stitch_panel_videos(
    panels: list[tuple[Path, str]],
    output_dir: Path,
    columns: int,
    *,
    task_names: set[str] | None = None,
) -> int:
    """Stitch per-panel episode videos into one labelled grid per episode.

    ``panels`` is ``[(<panel>/videos, label), ...]`` in display order. Each panel video is shown
    under its label bar; rows hold ``columns`` panels (0 = all in one row). An episode is stitched
    only once every panel has it, and an existing output is kept, so concurrent Slurm elements can
    call this repeatedly. Returns the number of videos written.
    """
    if len(panels) < 2:
        return 0
    grid_columns = columns if columns > 0 else len(panels)
    height = 256
    bar_height = even(max(20, height // 9))
    font = load_font(int(bar_height * 0.62))
    first_dir = panels[0][0]
    written = 0
    for task_dir in sorted(path for path in first_dir.glob("*") if path.is_dir()):
        if task_names is not None and task_dir.name not in task_names:
            continue
        for first_video in sorted(task_dir.glob("eval_episode_*.mp4")):
            videos = [directory / task_dir.name / first_video.name for directory, _ in panels]
            if not all(video.is_file() for video in videos):
                continue
            destination = output_dir / task_dir.name / first_video.name
            # Concurrent fanout jobs may both reach a completed task; the first
            # finished stitch wins and later jobs skip it.
            if destination.is_file() and destination.stat().st_size > 0:
                continue
            reads = [read_video(video) for video in videos]
            frame_sets = [read[0] for read in reads]
            if any(not frames for frames in frame_sets):
                continue
            bars = []
            for (_, label), frames in zip(panels, frame_sets, strict=True):
                frame_height, frame_width = frames[0].shape[:2]
                width = even(max(2, round(frame_width * height / frame_height)))
                bars.append(label_bar(width, bar_height, label, font))
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f"{destination.stem}.tmp{os.getpid()}.mp4")
            writer = imageio.get_writer(
                str(temporary), fps=reads[0][1], codec="libx264", quality=8, macro_block_size=None,
            )
            for frame_index in range(max(len(frames) for frames in frame_sets)):
                tiles = [
                    make_panel(frames[min(frame_index, len(frames) - 1)], height, bar)
                    for frames, bar in zip(frame_sets, bars, strict=True)
                ]
                frame_rows = [
                    np.hstack(tiles[start : start + grid_columns])
                    for start in range(0, len(tiles), grid_columns)
                ]
                max_width = max(row.shape[1] for row in frame_rows)
                frame_rows = [
                    row if row.shape[1] == max_width
                    else np.pad(row, ((0, 0), (0, max_width - row.shape[1]), (0, 0)))
                    for row in frame_rows
                ]
                frame = np.vstack(frame_rows)
                frame = frame[: frame.shape[0] - frame.shape[0] % 2, : frame.shape[1] - frame.shape[1] % 2]
                writer.append_data(frame)
            writer.close()
            temporary.replace(destination)
            written += 1
    logging.getLogger(__name__).info("Wrote %d side-by-side videos to %s.", written, output_dir)
    return written
