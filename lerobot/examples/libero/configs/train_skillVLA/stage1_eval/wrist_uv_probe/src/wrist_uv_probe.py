#!/usr/bin/env python3
"""Check that a skill-end EEF xyz projects to the right pixel of the WRIST camera.

The agent-view focus targets (build_data/src/build_skill_focus_uv.py) use a world-fixed camera. The
wrist camera moves with the gripper, so its pose is ``EEF pose o constant offset`` (the offset comes
from the recorded model XML). Before wiring a wrist-UV head into a policy, this probe draws the
projection on real frames under every candidate image orientation and writes one HTML report, so
the right convention can be picked by eye.

Driven by ``../run_wrist_uv_probe.sh`` (YAML -> flags); the flags below are its interface.

Two numbers in the report verify the maths on their own:
* the EEF projects to a FIXED pixel in every frame (the camera is bolted to the gripper), and
* at a skill's last frame the target IS the EEF, so the two pixels must coincide.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

from lerobot.policies.skillVLA.focus_projection import (
    camera_transform_from_hand_pose,
    hand_camera_from_recorded_xml,
    rotation_from_axis_angle,
)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2] / "stage1_skill_eval" / "src"))

ORIENTATIONS = ("libero", "raw", "flip_y", "flip_xy")


def _pixels(column: float, row: float, *, height: int, width: int) -> dict[str, tuple[float, float]]:
    """The same projection under each candidate image orientation."""
    return {
        "raw": (column, row),
        "libero": (width - 1 - column, row),          # build_skill_focus_uv's stored orientation
        "flip_y": (column, height - 1 - row),
        "flip_xy": (width - 1 - column, height - 1 - row),
    }


def _project(transform: np.ndarray, xyz: np.ndarray, *, height: int, width: int):
    point = np.concatenate([np.asarray(xyz, dtype=np.float64), [1.0]])
    projected = np.asarray(transform, dtype=np.float64) @ point
    depth = float(projected[2])
    if not np.isfinite(depth) or depth <= 1e-8:
        return None, depth
    return _pixels(projected[0] / depth, projected[1] / depth, height=height, width=width), depth


def _in_view(pixel: tuple[float, float], *, height: int, width: int) -> bool:
    return 0.0 <= pixel[0] <= width - 1.0 and 0.0 <= pixel[1] <= height - 1.0


def _episode_video(dataset_dir: Path, video_key: str, episode_index: int) -> tuple[Path, float]:
    """(mp4 holding this episode, timestamp where it starts) from meta/episodes."""
    import pandas as pd

    frames = [
        pd.read_parquet(path) for path in sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    ]
    if not frames:
        raise FileNotFoundError(f"No episode metadata under {dataset_dir / 'meta' / 'episodes'}.")
    table = pd.concat(frames, ignore_index=True)
    rows = table[table["episode_index"] == episode_index]
    if rows.empty:
        raise KeyError(f"Episode {episode_index} is not in the dataset metadata.")
    row = rows.iloc[0]
    video = dataset_dir / "videos" / video_key / (
        f"chunk-{int(row[f'videos/{video_key}/chunk_index']):03d}"
    ) / f"file-{int(row[f'videos/{video_key}/file_index']):03d}.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")
    return video, float(row[f"videos/{video_key}/from_timestamp"])


def _decode(video: Path, timestamps: list[float]) -> list[np.ndarray]:
    """Nearest decoded frame (RGB array) for each timestamp, in one pass."""
    import av

    wanted = sorted(set(timestamps))
    frames: dict[float, np.ndarray] = {}
    pending = list(wanted)
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(wanted[0] - 1.0, 0) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            while pending and frame.time is not None and frame.time + 1e-6 >= pending[0]:
                frames[pending.pop(0)] = frame.to_ndarray(format="rgb24")
            if not pending:
                break
    if pending:
        raise RuntimeError(f"{video} ended before timestamps {pending}.")
    return [frames[min(wanted, key=lambda value: abs(value - stamp))] for stamp in timestamps]


def patch_cell(pixel: tuple[float, float], *, grid: int, height: int, width: int) -> tuple[int, int] | None:
    """(row, col) of the DINO patch holding this pixel - the classification target - or None."""
    column = int(np.floor(pixel[0] / width * grid))
    row = int(np.floor(pixel[1] / height * grid))
    if 0 <= row < grid and 0 <= column < grid:
        return row, column
    return None


def soft_targets(cell: tuple[int, int], *, grid: int, sigma: float) -> dict[tuple[int, int], float]:
    """Gaussian label smoothing over neighbouring cells (sigma in cells; 0 = one-hot)."""
    if sigma <= 0:
        return {cell: 1.0}
    reach = int(np.ceil(2 * sigma))
    weights = {}
    for row in range(max(cell[0] - reach, 0), min(cell[0] + reach + 1, grid)):
        for column in range(max(cell[1] - reach, 0), min(cell[1] + reach + 1, grid)):
            squared = (row - cell[0]) ** 2 + (column - cell[1]) ** 2
            weights[(row, column)] = float(np.exp(-squared / (2.0 * sigma ** 2)))
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _draw(
    image: np.ndarray, pixel: tuple[float, float], eef_pixel: tuple[float, float], path: Path,
    *, grid: int = 0, sigma: float = 0.0, inside: bool = True,
) -> None:
    """Red crosshair = the skill-end target, blue square = the EEF (a fixed pixel).

    With ``grid`` the DINO patch lattice is drawn too, and the classification target is shaded:
    the solid cell is the argmax label, the fainter ones its Gaussian-smoothed neighbours.
    """
    from PIL import Image, ImageDraw

    picture = Image.fromarray(image).convert("RGB")
    if grid > 0:
        height, width = picture.height, picture.width
        overlay = Image.new("RGBA", picture.size, (0, 0, 0, 0))
        painter = ImageDraw.Draw(overlay)
        cell = patch_cell(pixel, grid=grid, height=height, width=width) if inside else None
        if cell is not None:
            for (row, column), weight in soft_targets(cell, grid=grid, sigma=sigma).items():
                strength = weight / max(soft_targets(cell, grid=grid, sigma=sigma).values())
                painter.rectangle(
                    [(column * width / grid, row * height / grid),
                     ((column + 1) * width / grid - 1, (row + 1) * height / grid - 1)],
                    fill=(255, 90, 0, int(30 + 110 * strength)),
                )
        for index in range(1, grid):
            painter.line([(index * width / grid, 0), (index * width / grid, height)], fill=(255, 255, 255, 45))
            painter.line([(0, index * height / grid), (width, index * height / grid)], fill=(255, 255, 255, 45))
        picture = Image.alpha_composite(picture.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(picture)
    x, y = float(eef_pixel[0]), float(eef_pixel[1])
    draw.rectangle([(x - 5, y - 5), (x + 5, y + 5)], outline=(60, 140, 255), width=2)
    x, y = float(pixel[0]), float(pixel[1])
    draw.line([(x - 9, y), (x + 9, y)], fill=(255, 40, 40), width=2)
    draw.line([(x, y - 9), (x, y + 9)], fill=(255, 40, 40), width=2)
    draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], outline=(255, 220, 0), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    picture.save(path)


def _report(out_dir: Path, header: dict, samples: list[dict], orientations: list[str]) -> Path:
    """One page, one section per skill: its frames left to right, one row per image orientation."""
    groups: dict[tuple[int, int, int], list[dict]] = {}
    for sample in samples:
        groups.setdefault((sample["task"], sample["episode"], sample["skill"]), []).append(sample)

    sections = []
    for (task, episode, skill), items in sorted(groups.items()):
        items = sorted(items, key=lambda item: item["frame"])
        rows = []
        for orientation in orientations:
            cells = []
            for item in items:
                pixel = item["pixels"].get(orientation) if item["pixels"] else None
                if pixel is None:
                    cells.append('<td class="out"><div class="cap">behind camera</div></td>')
                    continue
                inside = _in_view(pixel, height=header["height"], width=header["width"])
                cell = patch_cell(pixel, grid=header["grid"], height=header["height"],
                                  width=header["width"]) if inside and header["grid"] else None
                label = f"cell {cell[0]},{cell[1]}" if cell else "absent"
                cells.append(
                    f'<td class="{"ok" if inside else "out"}">'
                    f'<img src="{html.escape(item["images"][orientation])}" width="200">'
                    f'<div class="cap">{pixel[0]:.0f}, {pixel[1]:.0f} &middot; {label}</div></td>'
                )
            label = orientation if len(orientations) > 1 else "wrist"
            rows.append(f'<tr><th class="side">{html.escape(label)}</th>{"".join(cells)}</tr>')
        heads = "".join(
            f'<th>frame {item["frame"]}{" (last)" if item["is_last"] else ""}'
            f'<br><span class="dim">{item["depth"]:.3f} m</span></th>'
            for item in items
        )
        sections.append(
            f'<section><h2>task {task} &middot; episode {episode} &middot; skill {skill}'
            f' <span class="dim">frames {items[0]["frame"]}&ndash;{items[-1]["frame"]}</span></h2>'
            f'<table><tr><th class="side"></th>{heads}</tr>{"".join(rows)}</table></section>'
        )

    checks = "".join(
        f"<li>episode {episode}: EEF pixel spread {spread[0]:.2f}, {spread[1]:.2f} px (must be ~0)</li>"
        for episode, spread in header["eef_spread"].items()
    )
    page = f"""<!doctype html><meta charset="utf-8"><title>Wrist UV probe</title>
<style>
 body {{ font-family: Arial, sans-serif; margin: 18px; color: #17202a; background: #f5f6f8; }}
 section {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 12px 14px;
            margin-bottom: 16px; overflow-x: auto; }}
 h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; margin: 0 0 10px; }}
 table {{ border-collapse: collapse; }}
 th, td {{ padding: 4px 6px; font-size: 12px; text-align: center; vertical-align: top; }}
 th {{ font-weight: 600; white-space: nowrap; }}
 th.side {{ text-align: right; color: #475467; }}
 td.ok .cap {{ color: #146c43; }}
 td.out .cap {{ color: #b42318; }}
 td.out img {{ opacity: 0.55; }}
 .cap {{ margin-top: 3px; }}
 .dim {{ color: #667085; font-weight: 400; }}
 .intro {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 10px 14px; }}
 ul {{ line-height: 1.6; margin: 6px 0; }}
</style>
<h1>Wrist UV probe</h1>
<div class="intro">
<p>Red crosshair: the skill-end EEF xyz projected into the wrist camera &mdash; it must stay on the
same physical spot while the camera moves. Blue square: the EEF itself, a fixed pixel, which the
crosshair meets on a skill's last frame.</p>
<ul>
 <li>dataset: {html.escape(header["dataset"])}</li>
 <li>suite: {html.escape(header["suite"])} &middot; camera: {html.escape(header["camera"])}
     &middot; EEF site: {html.escape(header["eef_site"])} (rotation: {html.escape(header["rotation_frame"])})
     &middot; video: {html.escape(header["video_key"])}</li>
 <li>image: {header["width"]}x{header["height"]} &middot; EEF pixel: {header["eef_pixel"]}
     &middot; patch grid: {header["grid"]}x{header["grid"]} (cell = {header["width"] // max(header["grid"], 1)} px,
     smoothing sigma {header["sigma"]} cells) &mdash; the shaded cell is the classification target,
     &quot;absent&quot; when the target is out of view</li>
 {checks}
</ul></div>
{"".join(sections)}
"""
    path = out_dir / "index.html"
    path.write_text(page)
    return path


def _select_episodes(task_of: dict[int, int], args: argparse.Namespace) -> list[int]:
    """--episode-ids, else one episode per --task-ids, else the first --episodes of the dataset."""
    def ids(text: str) -> list[int]:
        return [int(value) for value in str(text).split(",") if str(value).strip()]

    available = sorted(task_of)
    wanted = ids(args.episode_ids)
    if wanted:
        missing = [episode for episode in wanted if episode not in task_of]
        if missing:
            raise ValueError(f"episode_ids {missing} are not in the dataset (have {available}).")
        return wanted
    tasks = ids(args.task_ids)
    if tasks:
        by_task: dict[int, list[int]] = {}
        for episode, task in sorted(task_of.items()):
            by_task.setdefault(task, []).append(episode)
        missing = [task for task in tasks if task not in by_task]
        if missing:
            raise ValueError(f"task_ids {missing} have no episode (have {sorted(by_task)}).")
        return [by_task[task][0] for task in tasks]
    return available[: args.episodes]


def probe(args: argparse.Namespace) -> None:
    from skill_data import SkillEvaluationDataset  # noqa: PLC0415 - heavy import

    orientations = [name.strip() for name in args.orientations.split(",") if name.strip()]
    unknown = [name for name in orientations if name not in ORIENTATIONS]
    if not orientations or unknown:
        raise ValueError(f"--orientations must be a subset of {list(ORIENTATIONS)}; got {orientations}.")

    info = json.loads((args.skill_dataset_dir / "meta" / "info.json").read_text())
    feature = info["features"][args.video_key]["info"]
    height, width = int(feature["video.height"]), int(feature["video.width"])
    fps = float(info.get("fps", 20))

    dataset = SkillEvaluationDataset(
        skill_dataset_dir=args.skill_dataset_dir,
        skill_latents_path=args.skill_latents_path,
        eval_init_states_path=args.eval_init_states_path,
        original_dataset_dir=args.original_dataset_dir,
        suite_name=args.suite,
    )
    with np.load(args.skill_latents_path, allow_pickle=False) as source:
        episode_id = np.asarray(source["episode_id"])
        task_id = np.asarray(source["task_id"])
        frame_start = np.asarray(source["frame_start"])
        frame_end = np.asarray(source["frame_end"])
    task_of = {int(episode): int(task) for episode, task in zip(episode_id, task_id)}

    args.out.mkdir(parents=True, exist_ok=True)
    episodes = _select_episodes(task_of, args)
    samples: list[dict] = []
    eef_spread: dict[int, tuple[float, float]] = {}
    eef_pixel_seen = (float("nan"), float("nan"))
    print(f"{'episode':>7} {'skill':>5} {'frame':>6} {'depth':>7}  " +
          "  ".join(f"{name:>13}" for name in orientations))
    for episode in episodes:
        aligned = dataset.load_aligned_episode(episode)
        if not aligned.model_xml:
            raise ValueError(f"Episode {episode} has no recorded model XML.")
        hand_camera = hand_camera_from_recorded_xml(
            aligned.model_xml, camera_name=args.camera, eef_site_name=args.eef_site,
            rotation_frame=args.rotation_frame or None,
        )
        offset = (
            np.asarray(aligned.episode_start_xyz, dtype=np.float64)
            if dataset.proprio_grounding == "episode_start_xyz"
            else np.zeros(3)
        )
        states = np.asarray(aligned.filtered_states, dtype=np.float64)
        rows = [index for index, value in enumerate(episode_id) if int(value) == episode]
        video, video_start = _episode_video(args.skill_dataset_dir, args.video_key, episode)

        planned: list[tuple[int, int, np.ndarray, bool]] = []
        for row in rows[: args.skills_per_episode]:
            start, end = int(frame_start[row]), min(int(frame_end[row]), len(states) - 1)
            target = states[end, :3] + offset
            for frame in np.linspace(start, end, args.frames_per_skill, dtype=int):
                planned.append((row, int(frame), target, int(frame) == end))

        images = _decode(video, [video_start + frame / fps for _, frame, _, _ in planned])
        eef_pixels = []
        for (row, frame, target, is_last), image in zip(planned, images, strict=True):
            transform = camera_transform_from_hand_pose(
                hand_camera, states[frame, :3] + offset,
                rotation_from_axis_angle(states[frame, 3:6]), height=height, width=width,
            )
            pixels, depth = _project(transform, target, height=height, width=width)
            eef, _ = _project(transform, states[frame, :3] + offset, height=height, width=width)
            saved: dict[str, str] = {}
            for orientation in orientations:
                if pixels is None or eef is None:
                    continue
                name = f"{orientation}/ep{episode:03d}_skill{row:04d}_f{frame:05d}.png"
                _draw(
                    image, pixels[orientation], eef[orientation], args.out / name,
                    grid=args.patch_grid, sigma=args.soft_sigma,
                    inside=_in_view(pixels[orientation], height=height, width=width),
                )
                saved[orientation] = name
            if eef is not None:
                eef_pixels.append(eef[orientations[0]])
                eef_pixel_seen = eef[orientations[0]]
            samples.append({
                "episode": episode, "task": task_of.get(episode, -1),
                "skill": row, "frame": frame, "depth": depth,
                "is_last": is_last, "pixels": pixels, "images": saved,
            })
            shown = "behind camera" if pixels is None else "  ".join(
                f"{pixels[name][0]:6.1f},{pixels[name][1]:6.1f}" for name in orientations
            )
            print(f"{episode:>7} {row:>5} {frame:>6} {depth:>7.3f}  {shown}")
        spread = np.nanstd(np.asarray(eef_pixels, dtype=np.float64), axis=0)
        eef_spread[episode] = (float(spread[0]), float(spread[1]))
        print(f"  EEF pixel spread over frames: x={spread[0]:.2f} y={spread[1]:.2f} px (must be ~0)")

    report = _report(
        args.out,
        {
            "dataset": str(args.skill_dataset_dir), "suite": args.suite, "camera": args.camera,
            "eef_site": args.eef_site, "rotation_frame": args.rotation_frame,
            "grid": args.patch_grid, "sigma": args.soft_sigma,
            "video_key": args.video_key, "height": height, "width": width,
            "eef_spread": eef_spread,
            "eef_pixel": f"{eef_pixel_seen[0]:.1f}, {eef_pixel_seen[1]:.1f}",
        },
        samples,
        orientations,
    )
    print(f"\nReport: {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skill-dataset-dir", type=Path, required=True)
    parser.add_argument("--skill-latents-path", type=Path, required=True)
    parser.add_argument("--eval-init-states-path", type=Path, required=True)
    parser.add_argument("--original-dataset-dir", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--camera", default="robot0_eye_in_hand")
    parser.add_argument("--eef-site", default="gripper0_grip_site")
    parser.add_argument("--rotation-frame", default="robot0_right_hand",
                        help="frame of observation.state[3:6]; LIBERO records the hand body, not the site")
    parser.add_argument("--video-key", default="observation.images.wrist_image")
    parser.add_argument("--task-ids", default="", help="comma-separated LIBERO task ids to sample")
    parser.add_argument("--episode-ids", default="", help="comma-separated episode ids (wins over --task-ids)")
    parser.add_argument("--episodes", type=int, default=2, help="episode count when neither list is given")
    parser.add_argument("--skills-per-episode", type=int, default=3)
    parser.add_argument("--frames-per-skill", type=int, default=3)
    parser.add_argument("--orientations", default=",".join(ORIENTATIONS))
    parser.add_argument("--patch-grid", type=int, default=14,
                        help="DINO patch lattice drawn as the classification target (224/16 = 14; 0 = off)")
    parser.add_argument("--soft-sigma", type=float, default=0.7,
                        help="Gaussian label smoothing over neighbouring cells, in cells (0 = one-hot)")
    parser.add_argument("--out", type=Path, required=True)
    probe(parser.parse_args())


if __name__ == "__main__":
    main()
