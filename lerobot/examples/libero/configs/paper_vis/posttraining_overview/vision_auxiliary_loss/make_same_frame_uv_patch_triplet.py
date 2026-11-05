#!/usr/bin/env python3
"""Create matched top-UV, top-patch, and wrist-patch paper assets.

The default selection is one frame already verified by the EEF projection
preview: LIBERO-90 episode 357, skill 0, frame 0.  All three outputs therefore
refer to the same skill-end EEF target and the same observation timestep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
PROJECT = next(parent for parent in HERE.parents if (parent / "dataset_filtered").is_dir())
sys.path.insert(0, str(PROJECT / "lerobot" / "src"))

from lerobot.datasets.video_utils import decode_video_frames  # noqa: E402


DEFAULT_DATASET = (
    PROJECT
    / "dataset_filtered/skillvla_dataset/libero_90_full_full"
    / "FSQ333_contrastive_2000_pt_grounded_fov_new/skillvla"
)
DEFAULT_MANIFEST = (
    PROJECT
    / "lerobot/examples/libero/configs/train_skills/skill_eval/eef_projection"
    / "outputs/skill_attention_preview/manifest.json"
)

BLUE = (73, 153, 218)
GRID = (245, 248, 250, 105)


def _frame(dataset: Path, episode: int, frame: int, video_key: str) -> np.ndarray:
    info = json.loads((dataset / "meta/info.json").read_text())
    columns = [
        "episode_index",
        f"videos/{video_key}/chunk_index",
        f"videos/{video_key}/file_index",
        f"videos/{video_key}/from_timestamp",
    ]
    metadata = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in sorted((dataset / "meta/episodes").rglob("*.parquet"))],
        ignore_index=True,
    ).set_index("episode_index")
    row = metadata.loc[int(episode)]
    video = dataset / str(info["video_path"]).format(
        video_key=video_key,
        chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
        file_index=int(row[f"videos/{video_key}/file_index"]),
    )
    timestamp = float(row[f"videos/{video_key}/from_timestamp"]) + int(frame) / float(info["fps"])
    decoded = decode_video_frames(video, [timestamp], tolerance_s=0.5 / float(info["fps"]))[0]
    return (
        (decoded.clamp(0.0, 1.0) * 255.0)
        .round()
        .to(dtype=__import__("torch").uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def _selection(manifest: Path, episode: int, skill: int, frame: int) -> tuple[dict, dict]:
    payload = json.loads(manifest.read_text())
    for record in payload["records"]:
        if int(record["episode_id"]) != episode or int(record["skill_index"]) != skill:
            continue
        for item in record["frames"]:
            if int(item["frame"]) == frame:
                return record, item
    raise ValueError(f"No preview record for episode={episode}, skill={skill}, frame={frame}")


def _dot_only(frame: np.ndarray, pixel: list[float], radius: int = 7) -> Image.Image:
    image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(image)
    x, y = map(float, pixel)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=BLUE)
    return image


def _patch_target(frame: np.ndarray, pixel: list[float], cell: list[int], grid: int) -> Image.Image:
    image = Image.fromarray(frame, mode="RGB").convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    row, column = map(int, cell)
    x0, y0 = column * width / grid, row * height / grid
    x1, y1 = (column + 1) * width / grid, (row + 1) * height / grid
    draw.rectangle((x0, y0, x1, y1), fill=(*BLUE, 125), outline=(*BLUE, 255), width=3)
    for index in range(1, grid):
        x = index * width / grid
        y = index * height / grid
        draw.line((x, 0, x, height), fill=GRID, width=1)
        draw.line((0, y, width, y), fill=GRID, width=1)
    composed = Image.alpha_composite(image, overlay).convert("RGB")
    draw = ImageDraw.Draw(composed)
    x, y = map(float, pixel)
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=BLUE)
    return composed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--episode", type=int, default=357)
    parser.add_argument("--skill", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()

    record, selected = _selection(args.manifest, args.episode, args.skill, args.frame)
    grid = int(json.loads(args.manifest.read_text())["patch_grid"])
    agent = _frame(args.dataset, args.episode, args.frame, "observation.images.image")
    wrist = _frame(args.dataset, args.episode, args.frame, "observation.images.wrist_image")
    agent_target = selected["projections"]["agent"]
    wrist_target = selected["projections"]["wrist"]
    if not agent_target["valid"] or not wrist_target["valid"]:
        raise ValueError("The selected target must be valid in both camera views.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "top_uv_endpoint": _dot_only(agent, agent_target["pixel"]),
        "top_patch_target": _patch_target(agent, agent_target["pixel"], agent_target["cell"], grid),
        "wrist_patch_target": _patch_target(wrist, wrist_target["pixel"], wrist_target["cell"], grid),
    }
    for name, image in outputs.items():
        image.save(args.output_dir / f"{name}.png", optimize=True)

    metadata = {
        "episode_id": args.episode,
        "skill_index": args.skill,
        "frame": args.frame,
        "endpoint_frame": record["endpoint_frame"],
        "endpoint_xyz": record["endpoint_xyz"],
        "task_description": record["task_description"],
        "patch_grid": grid,
        "top_target": agent_target,
        "wrist_target": wrist_target,
        "color_rgb": list(BLUE),
        "outputs": {name: f"{name}.png" for name in outputs},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
