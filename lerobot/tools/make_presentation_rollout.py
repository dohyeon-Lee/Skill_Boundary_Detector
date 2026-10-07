#!/usr/bin/env python3
"""Turn an annotated SkillVLA eval MP4 into a presentation-only rollout.

The eval renderer bakes the rollout, VSA inputs, goal marker, gauges, and
banners into one frame. This tool crops the rollout panel, removes the red
goal crosshair, and rebuilds only the presentation overlays from the saved
skill-trace HTML. No simulator or model execution is needed.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_SKILL_HUE_PERMUTATION = (
    3, 14, 10, 17, 5, 0, 7, 21, 19, 18, 23, 25, 4, 2,
    22, 6, 11, 20, 9, 26, 15, 16, 8, 1, 13, 24, 12,
)


def _skill_color(skill_id: int) -> tuple[int, int, int]:
    """Stable categorical color whose nearby/non-nearby IDs remain separated."""
    hue_index = _SKILL_HUE_PERMUTATION[int(skill_id) % len(_SKILL_HUE_PERMUTATION)]
    hue = hue_index / len(_SKILL_HUE_PERMUTATION)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.88)
    return tuple(round(channel * 255) for channel in (red, green, blue))


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _load_trace(path: Path, episode_index: int) -> tuple[str, list[dict]]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^const DATA = (\{.*\});$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not find embedded DATA in {path}.")
    data = json.loads(match.group(1))
    episode = next(
        (
            row
            for row in data.get("episodes", [])
            if int(row.get("episode_index", -1)) == int(episode_index)
        ),
        None,
    )
    if episode is None:
        raise ValueError(f"Episode {episode_index} is absent from {path}.")
    skills = sorted(episode.get("skills", []), key=lambda row: int(row["start_t"]))
    if not skills:
        raise ValueError(f"Episode {episode_index} has no skills in {path}.")
    return str(data.get("task_description", "")), skills


def _skill_at(skills: list[dict], timestep: int) -> int:
    current = skills[0]
    for skill in skills[1:]:
        if int(skill["start_t"]) > timestep:
            break
        current = skill
    return int(current["token"])


def _remove_goal_marker(frame_bgr: np.ndarray) -> np.ndarray:
    """Remove the eval renderer's (255, 64, 64) crosshair after MP4 decoding."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    red, green, blue = cv2.split(rgb)
    mask = (
        (red > 200)
        & (green > 15)
        & (green < 135)
        & (blue < 135)
        & ((red.astype(np.int16) - green.astype(np.int16)) > 90)
    ).astype(np.uint8) * 255
    if not np.any(mask):
        return frame_bgr
    # MP4 chroma subsampling leaves a faint halo beyond the nominal two-pixel
    # marker. Replace each compact marker-colored component with a filled mask
    # covering the complete 9 px crosshair arm and its compression halo.
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    marker_mask = np.zeros_like(mask)
    for component in range(1, component_count):
        x, y, width, height, area = stats[component]
        if area < 6 or width > 28 or height > 28:
            continue
        center_x, center_y = np.rint(centroids[component]).astype(int)
        cv2.circle(marker_mask, (center_x, center_y), 14, 255, thickness=-1)
    if not np.any(marker_mask):
        return frame_bgr
    return cv2.inpaint(frame_bgr, marker_mask, 5, cv2.INPAINT_TELEA)


def _remove_rollout_badge(frame_bgr: np.ndarray) -> np.ndarray:
    """Remove the legacy camera-panel label baked into annotated eval clips."""
    height, width = frame_bgr.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    # Remove the few chroma-subsampled rows that can bleed in from the green or
    # red outcome bar immediately above the rollout panel in the source MP4.
    mask[: min(height, 3), :] = 255
    # _annotate_eval_video draws this badge at (0, 0), with a height of 9% of
    # the original rollout and a width determined by the word "ROLLOUT".
    mask[: min(height, max(18, round(height * 0.10))), : min(width, round(width * 0.36))] = 255
    return cv2.inpaint(frame_bgr, mask, 7, cv2.INPAINT_TELEA)


def _render_frame(
    rollout_bgr: np.ndarray,
    *,
    skill_id: int,
    prompt: str,
    output_size: int,
    tint_alpha: float,
) -> np.ndarray:
    rollout = cv2.cvtColor(rollout_bgr, cv2.COLOR_BGR2RGB)
    rollout = cv2.resize(
        rollout,
        (output_size, output_size),
        interpolation=cv2.INTER_CUBIC,
    )
    color = np.asarray(_skill_color(skill_id), dtype=np.float32)
    rollout = np.clip(
        rollout.astype(np.float32) * (1.0 - tint_alpha) + color * tint_alpha,
        0,
        255,
    ).astype(np.uint8)

    prompt_h = max(54, round(output_size * 0.075))
    canvas = Image.new(
        "RGB",
        (output_size, output_size + prompt_h),
        (20, 20, 20),
    )
    canvas.paste(Image.fromarray(rollout), (0, 0))
    draw = ImageDraw.Draw(canvas)

    skill_font = _font(max(36, round(output_size * 0.068)))
    label = f"Skill {skill_id}"
    label_box = draw.textbbox((0, 0), label, font=skill_font, stroke_width=2)
    label_w = label_box[2] - label_box[0]
    label_h = label_box[3] - label_box[1]
    label_xy = (
        (output_size - label_w) / 2,
        (output_size - label_h) / 2 - label_box[1],
    )
    draw.text(
        label_xy,
        label,
        fill=(255, 255, 255),
        font=skill_font,
        stroke_width=3,
        stroke_fill=(10, 10, 10),
    )

    prompt_font = _font(max(16, round(prompt_h * 0.37)))
    prompt_box = draw.textbbox((0, 0), prompt, font=prompt_font)
    while prompt_box[2] - prompt_box[0] > output_size - 24 and getattr(prompt_font, "size", 0) > 12:
        prompt_font = _font(prompt_font.size - 1)
        prompt_box = draw.textbbox((0, 0), prompt, font=prompt_font)
    prompt_w = prompt_box[2] - prompt_box[0]
    prompt_text_h = prompt_box[3] - prompt_box[1]
    draw.text(
        (
            (output_size - prompt_w) / 2,
            output_size + (prompt_h - prompt_text_h) / 2 - prompt_box[1],
        ),
        prompt,
        fill=(240, 240, 240),
        font=prompt_font,
    )
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", type=Path)
    parser.add_argument("skill_trace_html", type=Path)
    parser.add_argument("output_video", type=Path)
    parser.add_argument("--episode-index", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--rollout-size", type=int, default=256)
    parser.add_argument("--top-bar-height", type=int, default=25)
    parser.add_argument("--output-size", type=int, default=768)
    parser.add_argument("--tint-alpha", type=float, default=0.20)
    args = parser.parse_args()
    if not 0.0 <= args.tint_alpha <= 1.0:
        raise ValueError("--tint-alpha must be in [0, 1].")

    prompt, skills = _load_trace(args.skill_trace_html, args.episode_index)
    capture = cv2.VideoCapture(str(args.input_video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open {args.input_video}.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS in {args.input_video}: {fps}.")

    prompt_h = max(54, round(args.output_size * 0.075))
    output_height = args.output_size + prompt_h
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args.output_size}x{output_height}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output_video),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    if encoder.stdin is None:
        raise RuntimeError("ffmpeg stdin was not created.")

    frame_index = 0
    try:
        while True:
            ok, full_frame = capture.read()
            if not ok:
                break
            y0 = args.top_bar_height
            y1 = y0 + args.rollout_size
            rollout = full_frame[y0:y1, : args.rollout_size]
            if rollout.shape[:2] != (args.rollout_size, args.rollout_size):
                raise ValueError(
                    f"Rollout crop is {rollout.shape[:2]}, expected "
                    f"{(args.rollout_size, args.rollout_size)}."
                )
            rollout = _remove_goal_marker(rollout)
            rollout = _remove_rollout_badge(rollout)
            skill_id = _skill_at(skills, frame_index * args.frame_stride)
            rendered = _render_frame(
                rollout,
                skill_id=skill_id,
                prompt=prompt,
                output_size=args.output_size,
                tint_alpha=args.tint_alpha,
            )
            encoder.stdin.write(rendered.tobytes())
            frame_index += 1
    finally:
        capture.release()
        encoder.stdin.close()
    return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}.")
    if frame_index == 0:
        raise ValueError(f"No frames decoded from {args.input_video}.")
    print(f"Wrote {frame_index} frames to {args.output_video}")


if __name__ == "__main__":
    main()
