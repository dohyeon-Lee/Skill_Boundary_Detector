#!/usr/bin/env python3
"""Preview deterministic zoom + random crop on a real recorded skill-start image.

CPU only. This experiment is NOT wired into any training/evaluation pipeline.
Raw training images are decoded using LeRobot v3 episode/video metadata; no
simulator, VLM, artificial image generation, or dataset modification is needed.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import random
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
PROJECT = next(p for p in HERE.parents if (p / "lerobot").is_dir() and (p / "dataset_filtered").is_dir())
DEFAULT_DATASET = PROJECT / (
    "dataset_filtered/skillvla_dataset/libero_90_full_full/"
    "FSQ333_contrastive_2000_pt_grounded_fov_new/skillvla"
)
CAMERA = "observation.images.image"


def recorded_frame(dataset: Path, episode: int, skill_index: int, frame: int | None):
    import pandas as pd

    info = json.loads((dataset / "meta/info.json").read_text())
    video_prefix = f"videos/{CAMERA}"
    columns = [
        "episode_index", "data/chunk_index", "data/file_index", "length", "tasks",
        f"{video_prefix}/chunk_index", f"{video_prefix}/file_index",
        f"{video_prefix}/from_timestamp",
    ]
    episodes = pd.concat(
        [pd.read_parquet(p, columns=columns) for p in sorted((dataset / "meta/episodes").rglob("*.parquet"))],
        ignore_index=True,
    )
    matches = episodes[episodes["episode_index"] == episode]
    if len(matches) != 1:
        raise ValueError(f"Expected one metadata record for episode {episode}, found {len(matches)}")
    meta = matches.iloc[0]
    data_path = dataset / info["data_path"].format(
        chunk_index=int(meta["data/chunk_index"]), file_index=int(meta["data/file_index"]),
    )
    rows = pd.read_parquet(data_path, columns=[
        "episode_index", "frame_index", "skill_initial_frame", "skill_sequence",
    ])
    rows = rows[rows["episode_index"] == episode].sort_values("frame_index")
    if rows.empty:
        raise ValueError(f"No data rows for episode {episode}")
    first = rows.iloc[0]
    starts = [int(value) for value in first["skill_initial_frame"] if int(value) >= 0]
    if frame is None:
        if not 0 <= skill_index < len(starts):
            raise ValueError(f"skill-index must be between 0 and {len(starts) - 1}")
        frame = starts[skill_index]
    if not 0 <= frame < int(meta["length"]):
        raise ValueError(f"Frame {frame} is outside episode length {int(meta['length'])}")
    actual_skill = max(i for i, start in enumerate(starts) if start <= frame)
    code = int(first["skill_sequence"][actual_skill])
    video = dataset / info["video_path"].format(
        video_key=CAMERA,
        chunk_index=int(meta[f"{video_prefix}/chunk_index"]),
        file_index=int(meta[f"{video_prefix}/file_index"]),
    )
    timestamp = float(meta[f"{video_prefix}/from_timestamp"]) + frame / float(info["fps"])
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to extract the recorded training image")
    result = subprocess.run([
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
        "-ss", f"{timestamp:.9f}", "-i", str(video), "-frames:v", "1",
        "-f", "image2pipe", "-vcodec", "png", "pipe:1",
    ], check=True, capture_output=True)
    image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
    return image, {
        "dataset": str(dataset), "episode_index": episode, "frame_index": frame,
        "skill_index": actual_skill, "skill_code": code,
        "task": str(meta["tasks"][0]), "camera": CAMERA,
        "source_video": str(video), "video_timestamp_seconds": timestamp,
        "fps": float(info["fps"]), "source_size": list(image.size),
    }


def zoom_and_crop(image: Image.Image, zoom: float, samples: int, seed: int):
    if not math.isfinite(zoom) or zoom <= 1.0:
        raise ValueError("zoom must be finite and greater than 1.0")
    if samples < 1:
        raise ValueError("samples must be positive")
    width, height = image.size
    expanded_size = (round(width * zoom), round(height * zoom))
    margin_x, margin_y = expanded_size[0] - width, expanded_size[1] - height
    if min(margin_x, margin_y) < 1:
        raise ValueError("zoom is too small to produce a movable crop window")
    candidates = (margin_x + 1) * (margin_y + 1)
    if samples > candidates:
        raise ValueError(f"Only {candidates} different integer crop offsets are available")
    expanded = image.resize(expanded_size, Image.Resampling.BILINEAR)
    center_x, center_y = margin_x // 2, margin_y // 2
    center = expanded.crop((center_x, center_y, center_x + width, center_y + height))
    rng = random.Random(seed)
    crops = []
    for index in rng.sample(range(candidates), samples):
        x, y = index % (margin_x + 1), index // (margin_x + 1)
        bounds = (x, y, x + width, y + height)
        crop = expanded.crop(bounds)
        assert crop.size == image.size
        assert 0 <= x <= margin_x and 0 <= y <= margin_y
        crops.append((crop, {"x": x, "y": y, "bounds_xyxy": list(bounds),
                             "offset_from_center_xy": [x - center_x, y - center_y]}))
    return expanded, center, crops, {
        "zoom_requested": zoom, "expanded_size": list(expanded_size),
        "zoom_effective_xy": [expanded_size[0] / width, expanded_size[1] / height],
        "output_size": list(image.size), "max_crop_offset_xy": [margin_x, margin_y],
        "center_crop_xy": [center_x, center_y], "interpolation": "bilinear",
        "padding": False, "seed": seed,
        "retained_field_of_view_fraction_xy": [width / expanded_size[0], height / expanded_size[1]],
    }


def font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default(size=size)


def comparison_sheet(original, center, crops, metadata):
    width, height = original.size
    gap, margin, header, footer, caption = 18, 24, 102, 48, 52
    items = [(original, "Original", f"{width} x {height}; recorded frame"),
             (center, "Zoom + center crop", "Fixed centered reference")]
    items += [(image, f"Random crop {i}", f"x={position['x']}, y={position['y']} px")
              for i, (image, position) in enumerate(crops, 1)]
    columns = min(4, len(items))
    rows = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (
        2 * margin + columns * width + (columns - 1) * gap,
        header + rows * (height + caption) + (rows - 1) * gap + footer,
    ), "white")
    draw = ImageDraw.Draw(sheet)
    expanded_w, expanded_h = metadata["expanded_size"]
    draw.text((margin, 16), f"Top-view crop preview | {metadata['zoom_requested']:.2f}x zoom",
              font=font(24), fill="#202020")
    draw.text((margin, 50), metadata["task"], font=font(15), fill="#3e3e3e")
    draw.text((margin, 74),
              f"{width}x{height} -> {expanded_w}x{expanded_h} -> {width}x{height} | "
              f"episode {metadata['episode_index']}, frame {metadata['frame_index']}, "
              f"skill {metadata['skill_code']} | seed {metadata['seed']}",
              font=font(14), fill="#5a5a5a")
    for i, (image, title, detail) in enumerate(items):
        x = margin + (i % columns) * (width + gap)
        y = header + (i // columns) * (height + caption + gap)
        draw.text((x, y), title, font=font(17), fill="#202020")
        draw.text((x, y + 24), detail, font=font(14), fill="#5a5a5a")
        sheet.paste(image, (x, y + caption))
    draw.text((margin, sheet.height - 29),
              "Only the top-view image changes. Same recorded scene and same 3D target; no padding.",
              font=font(14), fill="#4a4a4a")
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--skill-index", type=int, default=1)
    parser.add_argument("--frame", type=int, default=None, help="Override the skill-start frame")
    parser.add_argument("--zoom", type=float, default=1.10)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    original, source = recorded_frame(args.dataset.resolve(), args.episode, args.skill_index, args.frame)
    expanded, center, crops, augmentation = zoom_and_crop(original, args.zoom, args.samples, args.seed)
    metadata = {**source, **augmentation, "samples": []}
    output = args.output_dir or HERE / "outputs" / f"zoom_{round(args.zoom * 100):03d}"
    output.mkdir(parents=True, exist_ok=True)
    original.save(output / "original.png")
    expanded.save(output / "zoomed_canvas.png")
    center.save(output / "zoomed_center.png")
    for i, (image, position) in enumerate(crops, 1):
        filename = f"random_crop_{i:02d}.png"
        image.save(output / filename)
        metadata["samples"].append({"file": filename, **position})
    comparison_sheet(original, center, crops, metadata).save(output / "comparison.png")
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Source:", source["task"], f"episode={args.episode}, frame={source['frame_index']}")
    print("Resize / crop:", original.size, "->", expanded.size, "->", center.size)
    print("Random offsets:", [(p["x"], p["y"]) for _, p in crops])
    print("Comparison:", (output / "comparison.png").resolve())


if __name__ == "__main__":
    main()
