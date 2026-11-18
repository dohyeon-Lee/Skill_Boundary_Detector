#!/usr/bin/env python3
"""CPU-only GT endpoint/grid inspection; does NOT change Predictor training.

Task numbers use the exact LIBERO-90 benchmark order, not compact parquet task
indices. Episode ordering matches PT episode-exact eval. The first skill's
endpoint is min(frame_end, episode_length - 1), not frame_end - 1.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import itertools
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PROJECT = next(p for p in HERE.parents if (p / "dataset_filtered").is_dir())
sys.path.insert(0, str(HERE.parent / "crop_preview_tmp"))
from preview_topview_crop import DEFAULT_DATASET, font, recorded_frame  # noqa: E402


def grid_target(uv, grid):
    """Exactly the normalized-UV -> patch label rule in VSA top alignment."""
    uv = np.asarray(uv, dtype=np.float64)
    if not np.isfinite(uv).all() or (np.abs(uv) > 1).any():
        raise ValueError(f"Cannot quantize an invalid/out-of-frame UV: {uv}")
    cell = np.floor((uv + 1) * .5 * grid).astype(int).clip(0, grid - 1)
    center = -1 + (2 * cell + 1) / grid
    return cell, center


def center_crop(image, uv, *, zoom, size):
    """Mirror PredictorTopCrop's integer geometry and half-pixel UV mapping.

    PIL is used for display pixels only (not bit-exact torch interpolation).
    The continuous coordinate transform matches the square Mode3 input path.
    """
    w, h = image.size
    if w != h:
        raise ValueError("This diagnostic currently requires square Top images")
    ew, eh = round(w * zoom), round(h * zoom)
    x, y = (ew - w) // 2, (eh - h) // 2
    expanded = image.resize((ew, eh), Image.Resampling.BILINEAR)
    cropped = expanded.crop((x, y, x + w, y + h)).resize((size, size), Image.Resampling.BILINEAR)
    pixels = (np.asarray(uv) + 1) * np.array([w - 1, h - 1]) / 2
    pixels = (pixels + .5) * np.array([ew / w, eh / h]) - .5 - [x, y]
    pixels = (pixels + .5) * (size / w) - .5
    mapped = 2 * pixels / (size - 1) - 1
    return cropped, mapped, {"zoom": zoom, "expanded_size": [ew, eh], "crop_xy": [x, y]}


def annotated(image, uv, grid, *, display=448):
    cell, snapped = grid_target(uv, grid)
    picture = image.resize((display, display), Image.Resampling.NEAREST).convert("RGBA")
    overlay = Image.new("RGBA", picture.size)
    draw = ImageDraw.Draw(overlay)
    span = display - 1
    for index in range(grid + 1):
        position = index / grid * span
        draw.line((position, 0, position, span), fill=(255, 255, 255, 105), width=1)
        draw.line((0, position, span, position), fill=(255, 255, 255, 105), width=1)
    for row in range(grid):
        for column in range(grid):
            x, y = (column + .5) / grid * span, (row + .5) / grid * span
            draw.ellipse((x - 1.3, y - 1.3, x + 1.3, y + 1.3), fill=(255, 255, 255, 175))
    draw.line((span / 2, 0, span / 2, span), fill=(80, 220, 255, 175), width=2)
    draw.line((0, span / 2, span, span / 2), fill=(80, 220, 255, 175), width=2)
    point = (np.asarray(uv) + 1) * span / 2
    target = (snapped + 1) * span / 2
    draw.line((*point, *target), fill=(255, 255, 0, 255), width=3)
    x, y = target
    draw.rectangle((x - 7, y - 7, x + 7, y + 7), outline=(0, 0, 0, 255), width=4)
    draw.rectangle((x - 6, y - 6, x + 6, y + 6), outline=(255, 230, 0, 255), width=3)
    x, y = point
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 45, 65, 255), outline="white", width=1)
    return Image.alpha_composite(picture, overlay).convert("RGB"), cell, snapped


def uv_text(uv):
    return f"({uv[0]:+.3f}, {uv[1]:+.3f})"


def sheet(records, grid, *, endpoints_only=False):
    size, gap, margin = 448, 24, 26
    columns, rows = (len(records), 1) if endpoints_only else (3, len(records))
    row_height, header = size + 148, 128
    canvas = Image.new("RGB", (2 * margin + columns * size + (columns - 1) * gap,
                                header + rows * row_height + 55), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 16), f"LIBERO-90 | first-skill endpoint | {grid} x {grid} patch grid", font=font(27), fill="#202020")
    draw.text((margin, 57), "Red dot: exact GT UV   Yellow square: grid-center UV   Cyan axes: image-center origin", font=font(18), fill="#444444")
    draw.text((margin, 85), "GT dataset targets, NOT Predictor predictions. First exact demo of each task; no transition jitter.", font=font(17), fill="#555555")
    for row, record in enumerate(records):
        cases = [("Original endpoint frame", "endpoint", "original_uv")]
        if not endpoints_only:
            cases = [("Original skill-start frame", "start", "original_uv"),
                     ("Mode3 1.1x center crop (start)", "crop_start", "crop_uv"),
                     ("Original endpoint frame", "endpoint", "original_uv")]
        for column, (title, image_key, uv_key) in enumerate(cases):
            x = margin + (row if endpoints_only else column) * (size + gap)
            y = header + (0 if endpoints_only else row) * row_height
            draw.text((x, y), f"Task {record['task_id']} | {record['short_name']}", font=font(21), fill="#202020")
            draw.text((x, y + 31), title, font=font(18), fill="#444444")
            preview, cell, snapped = annotated(record[image_key], record[uv_key], grid)
            canvas.paste(preview, (x, y + 60))
            bottom = y + 60 + size
            draw.text((x, bottom + 9), f"GT {uv_text(record[uv_key])}  ->  grid {uv_text(snapped)}", font=font(17), fill="#202020")
            draw.text((x, bottom + 35), f"Cell (col,row)=({cell[0]},{cell[1]}) | ep {record['episode']} | skill {record['skill_code']}", font=font(16), fill="#555555")
            draw.text((x, bottom + 58), f"First skill: frames {record['start_frame']}..{record['end_frame']}", font=font(16), fill="#555555")
    draw.text((margin, canvas.height - 33), "Grid is the existing DINO patch-cell centers (not a newly selected Predictor quantization interval). v increases DOWN.", font=font(16), fill="#555555")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tasks", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--demos", type=int, default=10, help="First exact demos per task for coordinate-only overlap statistics")
    parser.add_argument("--grid", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    if args.demos < 1 or len(args.tasks) != len(set(args.tasks)):
        parser.error("demos must be positive; tasks must be unique")
    dataset = args.dataset.resolve()
    info = json.loads((dataset / "meta/info.json").read_text())
    dino_config = json.loads((PROJECT / "models/dinov3-vits16/config.json").read_text())
    size, patch = int(dino_config["image_size"]), int(dino_config["patch_size"])
    grid = args.grid or size // patch
    if grid < 2:
        parser.error("grid must be >= 2")
    task_names = runpy.run_path(str(PROJECT / "tools/lerobot-libero/libero/libero/benchmark/libero_suite_task_map.py"))["libero_task_map"]["libero_90"]
    mapping_path = dataset.parent.parent / "eval_init_states.npz"
    with np.load(mapping_path, allow_pickle=True) as source:
        episodes_by_scene = {}
        for episode, scene in zip(source["episode_index"], source["scene_file"], strict=True):
            episodes_by_scene.setdefault(str(scene).removesuffix("_demo.hdf5"), []).append(int(episode))
    uv_path = Path(info["skill_focus_uv_path"])
    if not uv_path.is_file():
        uv_path = dataset.parent / uv_path.name
    with np.load(uv_path, allow_pickle=False) as source:
        uv_data = {key: source[key].copy() for key in ("episode_id", "skill_index", "frame_start", "frame_end", "focus_uv", "focus_valid")}
    first_index = {}
    for index, (episode, rank) in enumerate(zip(uv_data["episode_id"], uv_data["skill_index"], strict=True)):
        old = first_index.get(int(episode))
        if old is None or rank < uv_data["skill_index"][old]:
            first_index[int(episode)] = index
    meta = pd.concat([pd.read_parquet(path, columns=["episode_index", "length"])
                      for path in sorted((dataset / "meta/episodes").rglob("*.parquet"))], ignore_index=True)
    lengths = dict(zip(meta["episode_index"].astype(int), meta["length"].astype(int), strict=True))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, coordinates = [], []
    short_names = {3: "butter at back", 4: "butter at front", 5: "chocolate pudding"}
    for task in args.tasks:
        name = task_names[task]
        episodes = sorted(episode for episode in episodes_by_scene.get(name, []) if episode in first_index)
        if not episodes:
            raise ValueError(f"No episode-exact GT data for Task {task}: {name}")
        for demo, episode in enumerate(episodes[:args.demos]):
            index = first_index[episode]
            uv = uv_data["focus_uv"][index].astype(float)
            if not uv_data["focus_valid"][index]:
                raise ValueError(f"Task {task}, episode {episode}: first endpoint is invalid/out of frame")
            cell, snapped = grid_target(uv, grid)
            end_frame = min(int(uv_data["frame_end"][index]), lengths[episode] - 1)
            entry = {"task_id": task, "demo_index": demo, "episode": episode,
                     "u": float(uv[0]), "v": float(uv[1]), "cell_col": int(cell[0]), "cell_row": int(cell[1]),
                     "grid_u": float(snapped[0]), "grid_v": float(snapped[1]), "endpoint_frame": end_frame}
            coordinates.append(entry)
            if demo:
                continue
            start_frame = int(uv_data["frame_start"][index])
            raw_start, source = recorded_frame(dataset, episode, 0, start_frame)
            raw_end, endpoint_source = recorded_frame(dataset, episode, 0, end_frame)
            start = raw_start.resize((size, size), Image.Resampling.BILINEAR)
            endpoint = raw_end.resize((size, size), Image.Resampling.BILINEAR)
            crop, crop_uv, crop_transform = center_crop(raw_start, uv, zoom=1.1, size=size)
            crop_cell, crop_snapped = grid_target(crop_uv, grid)
            record = {"task_id": task, "task_name": name, "short_name": short_names.get(task, name),
                      "episode": episode, "skill_index": int(uv_data["skill_index"][index]),
                      "skill_code": source["skill_code"], "start_frame": start_frame, "end_frame": end_frame,
                      "original_uv": uv, "crop_uv": crop_uv, "start": start, "endpoint": endpoint, "crop_start": crop,
                      "original_cell": cell.tolist(), "original_grid_uv": snapped.tolist(),
                      "crop_cell": crop_cell.tolist(), "crop_grid_uv": crop_snapped.tolist(),
                      "crop_transform": crop_transform, "start_source": source, "endpoint_source": endpoint_source}
            records.append(record)
            for key, image in (("native_start", raw_start), ("native_endpoint", raw_end), ("start", start),
                               ("endpoint", endpoint), ("mode3_center_start", crop)):
                image.save(output / f"task{task}_{key}.png")
            annotated(endpoint, uv, grid)[0].save(output / f"task{task}_endpoint_grid.png")
            print(f"Task {task} ep {episode} skill {source['skill_code']}: GT {uv_text(uv)} -> grid {uv_text(snapped)}", flush=True)
    sheet(records, grid, endpoints_only=True).save(output / "endpoints_tasks_3_4_5.png")
    sheet(records, grid).save(output / "comparison_all.png")
    with (output / "first_skill_uv_first_demos.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coordinates[0]))
        writer.writeheader()
        writer.writerows(coordinates)
    statistics = {}
    for task in args.tasks:
        counts = Counter((row["cell_col"], row["cell_row"]) for row in coordinates if row["task_id"] == task)
        statistics[str(task)] = {f"{col},{row}": count for (col, row), count in sorted(counts.items())}
    overlap = {f"{a} vs {b}": sorted(set(statistics[str(a)]) & set(statistics[str(b)]))
               for a, b in itertools.combinations(args.tasks, 2)}
    manifest = {"dataset": str(dataset), "focus_uv_path": str(uv_path), "episode_mapping": str(mapping_path),
                "grid": grid, "dino_image_size": size, "dino_patch_size": patch,
                "grid_uv_spacing": 2 / grid, "cell_width_model_pixels": size / grid,
                "note": "GT targets only; no predictor inference/training changes. Existing aux cell centers, NOT a new quantization choice.",
                "statistics_first_demos": statistics, "shared_cells_between_tasks": overlap,
                "records": [{key: value.tolist() if isinstance(value, np.ndarray) else value
                             for key, value in record.items() if not isinstance(value, Image.Image)} for record in records]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("GRID OCCUPANCY", json.dumps(statistics), flush=True)
    print("SHARED CELLS", json.dumps(overlap), flush=True)
    print("Overview:", output / "endpoints_tasks_3_4_5.png", flush=True)
    print("Start/crop/end comparison:", output / "comparison_all.png", flush=True)


if __name__ == "__main__":
    main()
