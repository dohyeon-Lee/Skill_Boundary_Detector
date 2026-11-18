#!/usr/bin/env python3
"""Compare centered zoom crops across recorded training scenes, without random crops."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from preview_topview_crop import CAMERA, DEFAULT_DATASET, HERE, font, recorded_frame


# These six real PT episodes belong to six DIFFERENT recorded scene layouts.
# The mapping is checked against eval_init_states.npz, not inferred from task prose.
DEFAULT_EPISODES = [0, 35, 318, 527, 957, 824]
DEFAULT_ZOOMS = [1.10, 1.20, 1.30, 1.40]


def center_zoom(image: Image.Image, zoom: float):
    if not math.isfinite(zoom) or zoom <= 1:
        raise ValueError("Each zoom must be finite and greater than 1")
    width, height = image.size
    expanded_size = (round(width * zoom), round(height * zoom))
    if expanded_size[0] <= width or expanded_size[1] <= height:
        raise ValueError("Zoom must produce at least one extra pixel in each dimension")
    expanded = image.resize(expanded_size, Image.Resampling.BILINEAR)
    x, y = (expanded.width - width) // 2, (expanded.height - height) // 2
    bounds = (x, y, x + width, y + height)
    return expanded.crop(bounds), {
        "zoom_requested": zoom,
        "expanded_size": list(expanded_size),
        "center_crop_bounds_xyxy": list(bounds),
        "crop_size_before_model_resize": [width, height],
        "remaining_fov_fraction_xy": [width / expanded.width, height / expanded.height],
    }


def scene_map(dataset: Path):
    path = dataset.parent.parent / "eval_init_states.npz"
    if not path.is_file():
        return {}, None
    with np.load(path, allow_pickle=True) as source:
        mapping = {int(ep): str(scene)
                   for ep, scene in zip(source["episode_index"], source["scene_file"], strict=True)}
    return mapping, str(path)


def fit_label(draw, text, max_width, size=15):
    chosen_font = font(size)
    if draw.textlength(text, font=chosen_font) <= max_width:
        return text
    while text and draw.textlength(text + "...", font=chosen_font) > max_width:
        text = text[:-1]
    return text + "..."


def overview(records, zooms, input_size, *, title):
    margin, gap, header, task_label, row_gap, footer = 22, 14, 104, 51, 24, 36
    columns = 1 + len(zooms)
    width = 2 * margin + columns * input_size + (columns - 1) * gap
    row_height = task_label + input_size
    height = header + len(records) * row_height + max(0, len(records) - 1) * row_gap + footer
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 12), title, font=font(23), fill="#202020")
    draw.text((margin, 44),
              f"Same frame across each row | all inputs {input_size} x {input_size} | center crops only",
              font=font(15), fill="#555555")
    headings = ["Original / no crop"] + [f"{z:.2f}x  (+{(z - 1) * 100:.0f}%)" for z in zooms]
    for column, heading in enumerate(headings):
        draw.text((margin + column * (input_size + gap), 76), heading,
                  font=font(17), fill="#202020")
    for row, record in enumerate(records):
        y = header + row * (row_height + row_gap)
        draw.text((margin, y),
                  f"{record['scene_layout']}  |  ep {record['episode_index']}, frame {record['frame_index']}",
                  font=font(16), fill="#222222")
        draw.text((margin, y + 24), fit_label(draw, record["task"], width - 2 * margin),
                  font=font(15), fill="#555555")
        for column, image in enumerate(record["images"]):
            assert image.size == (input_size, input_size)
            sheet.paste(image, (margin + column * (input_size + gap), y + task_label))
    draw.text((margin, height - 25),
              "Raw frame -> zoom -> native-size center crop -> model-size resize. No padding or random shifts.",
              font=font(14), fill="#555555")
    return sheet


def pair_image(original, cropped, record, zoom):
    size = original.width
    sheet = Image.new("RGB", (2 * size + 52, size + 104), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 10), record["scene_layout"], font=font(17), fill="#202020")
    draw.text((18, 39), "Original", font=font(16), fill="#202020")
    draw.text((size + 34, 39), f"Center crop {zoom:.2f}x", font=font(16), fill="#202020")
    sheet.paste(original, (18, 70))
    sheet.paste(cropped, (size + 34, 70))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episodes", type=int, nargs="+", default=DEFAULT_EPISODES)
    parser.add_argument("--zooms", type=float, nargs="+", default=DEFAULT_ZOOMS)
    parser.add_argument("--skill-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--rows-per-sheet", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.input_size <= 0 or args.rows_per_sheet <= 0:
        parser.error("input-size and rows-per-sheet must be positive")
    if len(set(args.episodes)) != len(args.episodes) or len(set(args.zooms)) != len(args.zooms):
        parser.error("episodes and zooms must not contain duplicates")
    if any(not math.isfinite(z) or z <= 1 for z in args.zooms):
        parser.error("zooms must be finite and greater than 1")
    dataset = args.dataset.resolve()
    mapping, mapping_path = scene_map(dataset)
    suffix = "_".join(f"{round(z * 100):03d}" for z in args.zooms)
    output = args.output_dir or HERE / "outputs" / f"center_sweep_{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    records, manifest_records = [], []
    for episode in args.episodes:
        raw, source = recorded_frame(dataset, episode, args.skill_index, args.frame)
        if raw.width != raw.height:
            raise ValueError("This preview expects square recorded Top-view images")
        scene = mapping.get(episode, "UNKNOWN_SCENE")
        match = re.match(r"(.+?_SCENE\d+)(?:_|$)", scene)
        layout = match.group(1) if match else scene
        source.update(scene_file=scene, scene_layout=layout, input_size=args.input_size)
        directory = output / f"{layout.lower()}_ep{episode:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        raw.save(directory / "source_native.png")
        original = raw.resize((args.input_size, args.input_size), Image.Resampling.BILINEAR)
        original.save(directory / "original_224.png" if args.input_size == 224 else directory / "original.png")
        images, transforms = [original], []
        for zoom in args.zooms:
            native_crop, transform = center_zoom(raw, zoom)
            final = native_crop.resize((args.input_size, args.input_size), Image.Resampling.BILINEAR)
            tag = f"{round(zoom * 100):03d}"
            final.save(directory / f"center_zoom_{tag}.png")
            pair_image(original, final, source, zoom).save(directory / f"pair_zoom_{tag}.png")
            transform["final_input_size"] = [args.input_size, args.input_size]
            transform["image_file"] = str((directory / f"center_zoom_{tag}.png").relative_to(output))
            transforms.append(transform)
            images.append(final)
        records.append({**source, "images": images})
        manifest_records.append({**source, "transforms": transforms})
        print(f"{layout}: episode={episode}, frame={source['frame_index']} | {source['task']}", flush=True)
    title = "Top-view center crop | zoom comparison"
    overview(records, args.zooms, args.input_size, title=title).save(output / "comparison_all.png")
    sheets = []
    for index, start in enumerate(range(0, len(records), args.rows_per_sheet), 1):
        path = output / f"comparison_{index:02d}.png"
        overview(records[start:start + args.rows_per_sheet], args.zooms, args.input_size,
                 title=f"{title} ({index})").save(path)
        sheets.append(path.name)
    manifest = {
        "camera": CAMERA, "dataset": str(dataset), "scene_mapping": mapping_path,
        "input_size": args.input_size, "zooms": args.zooms,
        "random_crops": False, "padding": False,
        "interpolation": "Pillow bilinear for preview",
        "note": "Matches model input dimensions/FOV; not a bit-exact tensor-preprocessing test.",
        "unique_scene_layouts": sorted({r["scene_layout"] for r in records}),
        "sheets": sheets, "records": manifest_records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Output:", output.resolve())


if __name__ == "__main__":
    main()
