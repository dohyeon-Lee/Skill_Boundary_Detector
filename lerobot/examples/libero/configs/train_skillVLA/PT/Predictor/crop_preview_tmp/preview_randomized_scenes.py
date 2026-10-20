#!/usr/bin/env python3
"""Six random Top-view crop samples per real recorded scene; CPU-only preview."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw

from preview_topview_crop import DEFAULT_DATASET, HERE, font, recorded_frame, zoom_and_crop
from preview_zoom_sweep import DEFAULT_EPISODES, fit_label, scene_map


def scene_sheet(original, center, crops, metadata):
    size = original.width
    margin, gap, header, caption, footer = 22, 16, 106, 49, 38
    items = [
        (original, "Original / no crop", f"{size} x {size}; full field of view"),
        (center, "Zoom / center crop", "x={}, y={} px".format(*metadata["center_crop_xy"])),
    ]
    items += [(image, f"Random crop {i}", f"x={position['x']}, y={position['y']} px")
              for i, (image, position) in enumerate(crops, 1)]
    columns = min(4, len(items))
    rows = math.ceil(len(items) / columns)
    width = 2 * margin + columns * size + (columns - 1) * gap
    height = header + rows * (caption + size) + (rows - 1) * gap + footer
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title = f"{metadata['scene_layout']} | {metadata['zoom_requested']:.2f}x random crop"
    draw.text((margin, 12), title, font=font(23), fill="#202020")
    draw.text((margin, 45), fit_label(draw, metadata["task"], width - 2 * margin),
              font=font(15), fill="#555555")
    native, expanded = metadata["source_size"][0], metadata["expanded_size"][0]
    draw.text((margin, 74),
              f"{native} -> {expanded} -> {native} crop -> {size} | "
              f"crop x/y: 0..{metadata['max_crop_offset_xy'][0]} px | "
              f"episode {metadata['episode_index']}, frame {metadata['frame_index']}, seed {metadata['seed']}",
              font=font(14), fill="#555555")
    for index, (image, title, detail) in enumerate(items):
        x = margin + (index % columns) * (size + gap)
        y = header + (index // columns) * (caption + size + gap)
        draw.text((x, y), title, font=font(16), fill="#202020")
        draw.text((x, y + 24), detail, font=font(13), fill="#555555")
        sheet.paste(image, (x, y + caption))
    draw.text((margin, height - 25),
              "Same real training frame and same 3D target. Top view only; no padding or color changes.",
              font=font(13), fill="#555555")
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episodes", type=int, nargs="+", default=DEFAULT_EPISODES)
    parser.add_argument("--skill-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--zoom", type=float, default=1.20)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if not math.isfinite(args.zoom) or args.zoom <= 1 or args.samples < 1 or args.input_size < 1:
        parser.error("zoom must be finite and >1; samples and input-size must be positive")
    if len(set(args.episodes)) != len(args.episodes):
        parser.error("episodes must not contain duplicates")
    dataset = args.dataset.resolve()
    mapping, mapping_path = scene_map(dataset)
    output = args.output_dir or HERE / "outputs" / f"random_zoom_{round(args.zoom * 100):03d}_multi_env"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for episode in args.episodes:
        raw, source = recorded_frame(dataset, episode, args.skill_index, args.frame)
        if raw.width != raw.height:
            raise ValueError("This preview expects square recorded Top-view images")
        scene = mapping.get(episode, "UNKNOWN_SCENE")
        match = re.match(r"(.+?_SCENE\d+)(?:_|$)", scene)
        layout = match.group(1) if match else scene
        # Different scenes get independent, reproducible uniform crop samples.
        scene_seed = args.seed + episode
        expanded, center_native, samples_native, crop_info = zoom_and_crop(
            raw, args.zoom, args.samples, scene_seed,
        )
        metadata = {**source, **crop_info, "scene_layout": layout, "scene_file": scene,
                    "final_input_size": [args.input_size, args.input_size], "samples": []}
        directory = output / f"{layout.lower()}_ep{episode:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        raw.save(directory / "source_native.png")
        expanded.save(directory / "zoomed_canvas.png")

        def resized(image):
            return image.resize((args.input_size, args.input_size), Image.Resampling.BILINEAR)

        original, center = resized(raw), resized(center_native)
        original.save(directory / "original.png")
        center.save(directory / "center.png")
        samples = []
        for index, (image, position) in enumerate(samples_native, 1):
            final = resized(image)
            filename = f"random_crop_{index:02d}.png"
            final.save(directory / filename)
            samples.append((final, position))
            metadata["samples"].append({"file": filename, **position})
        scene_sheet(original, center, samples, metadata).save(directory / "comparison.png")
        (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records.append({
            **metadata,
            "directory": directory.name,
            "comparison": str((directory / "comparison.png").relative_to(output)),
        })
        print(f"{layout}: {args.samples} random crops | offsets "
              f"{[(p['x'], p['y']) for _, p in samples]}", flush=True)
    manifest = {
        "dataset": str(dataset), "scene_mapping": mapping_path,
        "zoom": args.zoom, "samples_per_scene": args.samples,
        "input_size": args.input_size, "base_seed": args.seed,
        "padding": False, "random_crops": True,
        "interpolation": "Pillow bilinear for preview",
        "note": "Matches model input dimensions/FOV; not a bit-exact tensor-preprocessing test.",
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Output:", output.resolve())


if __name__ == "__main__":
    main()
