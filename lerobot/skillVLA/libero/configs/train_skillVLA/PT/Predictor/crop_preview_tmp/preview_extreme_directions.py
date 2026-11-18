#!/usr/bin/env python3
"""One PNG: one recorded scene per row, original/center/eight extreme shifts across columns."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw

from preview_topview_crop import DEFAULT_DATASET, HERE, font, recorded_frame
from preview_zoom_sweep import DEFAULT_EPISODES, fit_label, scene_map


# Directions describe IMAGE CONTENT movement relative to the centered zoom,
# not crop-window movement (which has the opposite sign).
DIRECTIONS = [
    ("up", 0, -1), ("down", 0, 1), ("left", -1, 0), ("right", 1, 0),
    ("up_left", -1, -1), ("up_right", 1, -1),
    ("down_left", -1, 1), ("down_right", 1, 1),
]


def extreme_crops(image: Image.Image, zoom: float):
    if not math.isfinite(zoom) or zoom <= 1:
        raise ValueError("zoom must be finite and greater than 1")
    width, height = image.size
    expanded_size = (round(width * zoom), round(height * zoom))
    margin_x, margin_y = expanded_size[0] - width, expanded_size[1] - height
    if min(margin_x, margin_y) < 2:
        raise ValueError("At least two extra pixels per axis are needed for eight distinct extremes")
    expanded = image.resize(expanded_size, Image.Resampling.BILINEAR)
    cx, cy = margin_x // 2, margin_y // 2
    center = expanded.crop((cx, cy, cx + width, cy + height))
    crops = []
    for name, dx, dy in DIRECTIONS:
        x = margin_x if dx < 0 else 0 if dx > 0 else cx
        y = margin_y if dy < 0 else 0 if dy > 0 else cy
        bounds = (x, y, x + width, y + height)
        crop = expanded.crop(bounds)
        crops.append((crop, {
            "direction": name, "direction_vector_xy": [dx, dy],
            "crop_xy": [x, y], "crop_bounds_xyxy": list(bounds),
            "content_shift_native_xy": [cx - x, cy - y],
        }))
    return expanded, center, crops, {
        "zoom_requested": zoom,
        "zoom_effective_xy": [expanded_size[0] / width, expanded_size[1] / height],
        "expanded_size": list(expanded_size), "native_crop_size": [width, height],
        "max_crop_offset_xy": [margin_x, margin_y], "center_crop_xy": [cx, cy],
        "direction_convention": "image_content_movement_relative_to_center",
        "random_crops": False, "padding": False,
    }


def comparison_sheet(records, zoom, size):
    margin, gap, header, row_labels, row_gap, footer = 22, 12, 151, 51, 24, 38
    headings = ["Original / no crop", "Center"] + [name.replace("_", "-").title() for name, _, _ in DIRECTIONS]
    width = 2 * margin + len(headings) * size + (len(headings) - 1) * gap
    row_height = row_labels + size
    height = header + len(records) * row_height + max(0, len(records) - 1) * row_gap + footer
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 15), f"Top-view {zoom:.2f}x | eight maximum shifts", font=font(29), fill="#202020")
    draw.text((margin, 56),
              f"One real source frame per row | every input {size} x {size} | directions mean image-content movement, not crop-window movement",
              font=font(17), fill="#555555")
    first = records[0]
    w, h = first["source_size"]
    ew, eh = first["expanded_size"]
    draw.text((margin, 83), f"Processing: {w}x{h} -> {ew}x{eh} zoom -> {w}x{h} crop -> {size}x{size} resize. No padding; no random sampling.",
              font=font(16), fill="#555555")
    for index, name in enumerate(headings):
        draw.text((margin + index * (size + gap), 119), name, font=font(18), fill="#202020")
    for row, record in enumerate(records):
        y = header + row * (row_height + row_gap)
        draw.text((margin, y),
                  f"{record['scene_layout']} | episode {record['episode_index']}, frame {record['frame_index']}",
                  font=font(17), fill="#202020")
        draw.text((margin, y + 25), fit_label(draw, record["task"], width - 2 * margin, size=16),
                  font=font(16), fill="#555555")
        for column, image in enumerate(record["images"]):
            assert image.size == (size, size)
            sheet.paste(image, (margin + column * (size + gap), y + row_labels))
    draw.text((margin, height - 27),
              "Original and centered zoom are references. The remaining eight columns are exact cardinal/diagonal extrema of the available crop range.",
              font=font(16), fill="#555555")
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episodes", type=int, nargs="+", default=DEFAULT_EPISODES)
    parser.add_argument("--skill-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--zoom", type=float, default=1.15)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if not math.isfinite(args.zoom) or args.zoom <= 1 or args.input_size < 1:
        parser.error("zoom must be finite and >1; input-size must be positive")
    if len(set(args.episodes)) != len(args.episodes):
        parser.error("episodes must not contain duplicates")
    dataset = args.dataset.resolve()
    mapping, mapping_path = scene_map(dataset)
    output = args.output_dir or HERE / "outputs" / f"extreme_zoom_{round(args.zoom * 100):03d}_multi_env"
    output.mkdir(parents=True, exist_ok=True)
    records, manifest_records = [], []
    for episode in args.episodes:
        raw, source = recorded_frame(dataset, episode, args.skill_index, args.frame)
        if raw.width != raw.height:
            raise ValueError("This preview expects square Top-view frames")
        scene = mapping.get(episode, "UNKNOWN_SCENE")
        match = re.match(r"(.+?_SCENE\d+)(?:_|$)", scene)
        layout = match.group(1) if match else scene
        expanded, center_native, crops_native, crop_info = extreme_crops(raw, args.zoom)
        metadata = {**source, **crop_info, "scene_layout": layout, "scene_file": scene,
                    "final_input_size": [args.input_size, args.input_size]}
        directory = output / f"{layout.lower()}_ep{episode:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        raw.save(directory / "source_native.png")
        expanded.save(directory / "zoomed_canvas.png")

        def resized(image):
            return image.resize((args.input_size, args.input_size), Image.Resampling.BILINEAR)

        original, center = resized(raw), resized(center_native)
        original.save(directory / "original.png")
        center.save(directory / "center.png")
        images, extremes = [original, center], []
        for crop, transform in crops_native:
            final = resized(crop)
            filename = f"shift_{transform['direction']}.png"
            final.save(directory / filename)
            transform["content_shift_input_xy"] = [
                transform["content_shift_native_xy"][0] * args.input_size / raw.width,
                transform["content_shift_native_xy"][1] * args.input_size / raw.height,
            ]
            extremes.append({**transform, "file": str((directory / filename).relative_to(output))})
            images.append(final)
        records.append({**metadata, "images": images})
        manifest_records.append({**metadata, "directory": directory.name, "extremes": extremes})
        print(f"{layout}: eight deterministic extremes; crop range {crop_info['max_crop_offset_xy']}", flush=True)
    comparison_sheet(records, args.zoom, args.input_size).save(output / "comparison_all.png")
    manifest = {
        "dataset": str(dataset), "scene_mapping": mapping_path,
        "zoom": args.zoom, "input_size": args.input_size,
        "extremes_per_scene": 8, "random_crops": False, "padding": False,
        "direction_convention": "image_content_movement_relative_to_center",
        "columns": ["original", "center"] + [name for name, _, _ in DIRECTIONS],
        "interpolation": "Pillow bilinear for preview",
        "note": "Matches model input dimensions/FOV; not a bit-exact tensor-preprocessing test.",
        "records": manifest_records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Combined PNG:", (output / "comparison_all.png").resolve())


if __name__ == "__main__":
    main()
