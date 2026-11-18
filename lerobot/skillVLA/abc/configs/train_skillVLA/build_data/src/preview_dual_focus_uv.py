#!/usr/bin/env python3
"""Preview exact bimanual ABC EEF projections before defining the UV schema.

The official sim_224 episodes include the per-frame MuJoCo qpos and assembled
scene XML used for rendering. This tool projects each skill's endpoint
``left_grasp_site`` and ``right_grasp_site`` through the recorded top and wrist
cameras. Only endpoint UVs are drawn; current-frame EEF dots are intentionally
omitted so the preview matches the SkillVLA target definition.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw


LEFT_COLOR = (25, 225, 235)
RIGHT_COLOR = (238, 62, 210)
WHITE = (255, 255, 255)


@dataclass(frozen=True)
class Projection:
    xy: np.ndarray
    valid: bool
    clipped: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skillvla-dir",
        type=Path,
        required=True,
        help="SkillVLA run containing skillvla/native_dataset.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <skillvla-dir>/dual_focus_uv_preview.",
    )
    parser.add_argument("--task", default="grab_clutter")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--episode-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional native integer episode IDs; overrides automatic selection.",
    )
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def kinematics_only_xml(scene_xml: str) -> str:
    """Drop assets/geometry without changing the body's joint/qpos layout."""
    root = ET.fromstring(scene_xml)
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {"geom", "contact", "sensor"}:
                parent.remove(child)
    asset = root.find("asset")
    if asset is not None:
        root.remove(asset)
    return ET.tostring(root, encoding="unicode")


def compile_episode_model(episode_dir: Path, project_root: Path) -> mujoco.MjModel:
    scene_path = episode_dir / "scene_assembled.xml"
    if not scene_path.is_file():
        raise FileNotFoundError(f"Episode has no assembled scene XML: {scene_path}")
    scene_xml = scene_path.read_text(encoding="utf-8")
    local_assets = (project_root / "third_party/abc/abc_sim/models/assets").resolve()
    scene_xml = re.sub(
        r'(?<=\")/[^\"]*?/models/assets(?=[/\"])', str(local_assets), scene_xml
    )
    scene_xml = kinematics_only_xml(scene_xml)
    previous = Path.cwd()
    os.chdir(local_assets.parent)
    try:
        return mujoco.MjModel.from_xml_string(scene_xml)
    finally:
        os.chdir(previous)


def project_point(
    point_world: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    *,
    fovy: float,
    width: int,
    height: int,
) -> Projection:
    # MuJoCo cameras look along local -Z; +X is image-right and +Y image-up.
    point_camera = camera_rotation.T @ (point_world - camera_position)
    depth = float(-point_camera[2])
    if not np.isfinite(point_camera).all() or depth <= 1e-8:
        return Projection(np.asarray([math.nan, math.nan]), False, False)
    focal = 0.5 * height / math.tan(math.radians(fovy) / 2.0)
    x = width / 2.0 + focal * float(point_camera[0]) / depth
    y = height / 2.0 - focal * float(point_camera[1]) / depth
    finite = math.isfinite(x) and math.isfinite(y)
    in_view = finite and 0 <= x <= width - 1 and 0 <= y <= height - 1
    return Projection(np.asarray([x, y], dtype=np.float64), finite, finite and not in_view)


def episode_target_uv(
    model: mujoco.MjModel,
    scene_qpos: np.ndarray,
    annotation: dict,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scene_qpos.shape[1] != model.nq:
        raise ValueError(
            f"scene_qpos nq={scene_qpos.shape[1]}, compiled scene nq={model.nq}"
        )
    camera_names = ("top", "left", "right")
    camera_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        for name in camera_names
    ]
    site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_site"),
    ]
    if min(camera_ids) < 0 or min(site_ids) < 0:
        raise ValueError("Scene needs top/left/right cameras and left/right grasp sites.")
    fovys = [float(model.cam_fovy[camera_id]) for camera_id in camera_ids]
    data = mujoco.MjData(model)
    site_world = np.empty((len(scene_qpos), 2, 3), dtype=np.float64)
    camera_position = np.empty((len(scene_qpos), 3, 3), dtype=np.float64)
    camera_rotation = np.empty((len(scene_qpos), 3, 3, 3), dtype=np.float64)
    for frame_index, qpos in enumerate(scene_qpos):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for arm, site_id in enumerate(site_ids):
            site_world[frame_index, arm] = np.asarray(
                data.site_xpos[site_id], dtype=np.float64
            )
        for camera, camera_id in enumerate(camera_ids):
            camera_position[frame_index, camera] = np.asarray(
                data.cam_xpos[camera_id], dtype=np.float64
            )
            camera_rotation[frame_index, camera] = np.asarray(
                data.cam_xmat[camera_id], dtype=np.float64
            ).reshape(3, 3)

    # Slots: top-left target, top-right target, left-wrist/left target,
    # right-wrist/right target. Wrist cameras use their pose at the current
    # frame while the world-space point is fixed at the current skill endpoint.
    uv = np.full((len(scene_qpos), 4, 2), np.nan, dtype=np.float32)
    valid = np.zeros((len(scene_qpos), 4), dtype=np.bool_)
    clipped = np.zeros((len(scene_qpos), 4), dtype=np.bool_)
    for frame_index in range(len(scene_qpos)):
        order = int(annotation["skill_index"][frame_index])
        start = int(annotation["starts"][order])
        endpoint = min(
            start + int(annotation["lengths"][order]), annotation["length"] - 1
        )
        requests = (
            (0, 0),  # top camera -> left endpoint
            (0, 1),  # top camera -> right endpoint
            (1, 0),  # left wrist -> left endpoint
            (2, 1),  # right wrist -> right endpoint
        )
        for slot, (camera, arm) in enumerate(requests):
            result = project_point(
                site_world[endpoint, arm],
                camera_position[frame_index, camera],
                camera_rotation[frame_index, camera],
                fovy=fovys[camera],
                width=width,
                height=height,
            )
            uv[frame_index, slot] = result.xy
            valid[frame_index, slot] = result.valid
            clipped[frame_index, slot] = result.clipped
    return uv, valid, clipped


def annotation_for_episode(archive: np.lib.npyio.NpzFile, episode_id: int) -> dict:
    positions = np.flatnonzero(np.asarray(archive["episode_id"]) == episode_id)
    if len(positions) != 1:
        raise KeyError(f"Expected one annotation row for episode {episode_id}; got {len(positions)}")
    position = int(positions[0])
    start = int(archive["offsets"][position])
    length = int(archive["episode_length"][position])
    stop = start + length
    count = int(archive["skill_sequence_len"][position])
    return {
        "length": length,
        "skill_index": np.asarray(archive["skill_index"][start:stop], dtype=np.int32),
        "codes": np.asarray(archive["skill_sequence"][position, :count], dtype=np.int32),
        "starts": np.asarray(archive["skill_initial_frame"][position, :count], dtype=np.int32),
        "lengths": np.asarray(archive["skill_length_sequence"][position, :count], dtype=np.int32),
    }


def decode_camera_frames(
    video_path: Path, *, cameras: list[str], height: int, expected: int
):
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index >= expected:
                break
            array = frame.to_ndarray(format="rgb24")
            if array.shape[0] < height * len(cameras):
                raise ValueError(
                    f"Combined video is shorter than {len(cameras)} camera crops: {array.shape}"
                )
            yield {
                camera: array[position * height : (position + 1) * height]
                for position, camera in enumerate(cameras)
            }


def draw_marker(
    draw: ImageDraw.ImageDraw,
    xy: np.ndarray,
    *,
    color: tuple[int, int, int],
    scale: int,
    label: str,
    width: int,
    height: int,
    clipped: bool,
) -> None:
    if not np.isfinite(xy).all():
        return
    shown = np.asarray(xy, dtype=np.float64).copy()
    shown[0] = np.clip(shown[0], 0, width - 1)
    shown[1] = np.clip(shown[1], 0, height - 1)
    x, y = float(shown[0]) * scale, float(shown[1]) * scale
    radius = 10 * scale / 2
    box = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(box, outline=color, width=max(2, 2 * scale))
    cross = radius + 3 * scale
    draw.line((x - cross, y, x + cross, y), fill=color, width=max(1, scale))
    draw.line((x, y - cross, x, y + cross), fill=color, width=max(1, scale))
    draw.text(
        (x + radius + 2 * scale, y - radius),
        f"{label}{' (clip)' if clipped else ''}",
        fill=color,
        stroke_width=max(1, scale // 2),
        stroke_fill=(0, 0, 0),
    )


def overlay_frame(
    frames: dict[str, np.ndarray],
    *,
    frame_index: int,
    uv: np.ndarray,
    valid: np.ndarray,
    clipped: np.ndarray,
    annotation: dict,
    scale: int,
) -> Image.Image:
    camera_order = ("left", "top", "right")
    camera_titles = ("Left wrist", "Top", "Right wrist")
    source_height, source_width = frames["top"].shape[:2]
    title_height = 18 * scale
    panels = []
    for camera, title in zip(camera_order, camera_titles, strict=True):
        image = Image.fromarray(frames[camera]).resize(
            (source_width * scale, source_height * scale), Image.Resampling.NEAREST
        )
        panel = Image.new("RGB", (image.width, image.height + title_height), (12, 15, 20))
        panel.paste(image, (0, title_height))
        ImageDraw.Draw(panel).text((5 * scale, 3 * scale), title, fill=WHITE)
        panels.append(panel)
    image = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height))
    x_offset = 0
    for panel in panels:
        image.paste(panel, (x_offset, 0))
        x_offset += panel.width
    order = int(annotation["skill_index"][frame_index])
    skill_start = int(annotation["starts"][order])
    # Same endpoint convention as LIBERO: min(frame_end, episode_length - 1).
    endpoint = min(skill_start + int(annotation["lengths"][order]), annotation["length"] - 1)
    code = int(annotation["codes"][order])
    # panel offsets: left wrist, top, right wrist. Projection slots are
    # top-L, top-R, left-wrist-L, right-wrist-R.
    marker_specs = (
        (0, 2, "L endpoint", LEFT_COLOR),
        (1, 0, "L endpoint", LEFT_COLOR),
        (1, 1, "R endpoint", RIGHT_COLOR),
        (2, 3, "R endpoint", RIGHT_COLOR),
    )
    for panel_index, slot, label, color in marker_specs:
        panel = panels[panel_index]
        draw = ImageDraw.Draw(panel)
        draw_marker(
            draw,
            uv[frame_index, slot] + np.asarray([0.0, title_height / scale]),
            color=color,
            scale=scale,
            label=label,
            width=source_width,
            height=source_height + title_height / scale,
            clipped=bool(clipped[frame_index, slot]),
        )
    # Reassemble after drawing markers into each panel.
    x_offset = 0
    for panel in panels:
        image.paste(panel, (x_offset, 0))
        x_offset += panel.width

    line_height = 14 * scale
    panel_height = 2 * line_height + 8 * scale
    panel = Image.new("RGB", (image.width, panel_height), (12, 15, 20))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.text(
        (5 * scale, 3 * scale),
        f"frame {frame_index}/{annotation['length'] - 1}   skill {order}   "
        f"FSQ code {code}   endpoint {endpoint}",
        fill=WHITE,
    )
    names = ("Top-L", "Top-R", "Wrist-L", "Wrist-R")
    states = [
        f"{name}: {'valid' if valid[frame_index, slot] else 'invalid'}, "
        f"{'clipped' if clipped[frame_index, slot] else 'in-view'}"
        for slot, name in enumerate(names)
    ]
    panel_draw.text((5 * scale, 3 * scale + line_height), "   ".join(states), fill=(190, 198, 210))
    output = Image.new("RGB", (image.width, image.height + panel_height))
    output.paste(image, (0, 0))
    output.paste(panel, (0, image.height))
    return output


def open_output_video(output: Path, *, width: int, height: int, fps: float):
    container = av.open(str(output), mode="w")
    try:
        stream = container.add_stream("libx264", rate=max(1, round(fps)))
    except av.error.FFmpegError:
        stream = container.add_stream("mpeg4", rate=max(1, round(fps)))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "veryfast"}
    return container, stream


def encode_image(container, stream, image: Image.Image) -> None:
    frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)


def write_episode_preview(
    *,
    episode_id: int,
    episode_dir: Path,
    annotation: dict,
    project_root: Path,
    output_dir: Path,
    scale: int,
    fps: float,
) -> dict:
    metadata = load_json(episode_dir / "episode_metadata.json")
    width, height = int(metadata["image_width"]), int(metadata["image_height"])
    scene_qpos = np.load(episode_dir / "scene_qpos.npy", mmap_mode="r")
    if len(scene_qpos) != annotation["length"]:
        raise ValueError(
            f"Episode {episode_id}: scene_qpos={len(scene_qpos)}, annotations={annotation['length']}"
        )
    model = compile_episode_model(episode_dir, project_root)
    uv, valid, clipped = episode_target_uv(
        model, scene_qpos, annotation, width=width, height=height
    )

    source_video = episode_dir / "combined_camera-images-rgb.mp4"
    source_cameras = list(metadata.get("cameras") or ["top", "left", "right"])
    required_cameras = {"top", "left", "right"}
    if not required_cameras.issubset(source_cameras):
        raise ValueError(
            f"Episode {episode_id} needs top/left/right camera crops; got {source_cameras}"
        )
    frames = decode_camera_frames(
        source_video,
        cameras=source_cameras,
        height=height,
        expected=annotation["length"],
    )
    first = overlay_frame(
        next(frames), frame_index=0, uv=uv, valid=valid, clipped=clipped,
        annotation=annotation, scale=scale,
    )
    output_path = output_dir / f"episode_{episode_id}_dual_uv.mp4"
    container, stream = open_output_video(
        output_path, width=first.width, height=first.height, fps=fps
    )
    try:
        encode_image(container, stream, first)
        for frame_index, frame in enumerate(frames, start=1):
            encode_image(
                container,
                stream,
                overlay_frame(
                    frame, frame_index=frame_index, uv=uv, valid=valid,
                    clipped=clipped, annotation=annotation, scale=scale,
                ),
            )
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    wanted = {
        min(int(start + length), annotation["length"] - 1): order
        for order, (start, length) in enumerate(
            zip(annotation["starts"], annotation["lengths"], strict=True)
        )
    }
    endpoint_images = []
    for frame_index, frame_crops in enumerate(
        decode_camera_frames(
            source_video,
            cameras=source_cameras,
            height=height,
            expected=annotation["length"],
        )
    ):
        if frame_index not in wanted:
            continue
        image = overlay_frame(
            frame_crops,
            frame_index=frame_index,
            uv=uv,
            valid=valid,
            clipped=clipped,
            annotation=annotation,
            scale=1,
        )
        endpoint_images.append(image)
    if not endpoint_images:
        raise ValueError(f"Episode {episode_id} produced no skill endpoint previews")
    columns = 2
    rows = math.ceil(len(endpoint_images) / columns)
    card_width, card_height = endpoint_images[0].size
    sheet = Image.new(
        "RGB", (card_width * columns, card_height * rows), (28, 32, 40)
    )
    for index, image in enumerate(endpoint_images):
        sheet.paste(
            image,
            ((index % columns) * card_width, (index // columns) * card_height),
        )
    sheet_path = output_dir / f"episode_{episode_id}_skill_endpoints.jpg"
    sheet.save(sheet_path, quality=92)

    return {
        "episode_id": episode_id,
        "instruction": str(metadata.get("instruction", "")),
        "frames": annotation["length"],
        "skills": len(annotation["codes"]),
        "video": output_path.name,
        "sheet": sheet_path.name,
        "projection_names": ["top_left", "top_right", "left_wrist", "right_wrist"],
        "valid_fraction": np.mean(valid, axis=0).tolist(),
        "clipped_fraction": np.mean(clipped, axis=0).tolist(),
    }


def write_html(output_dir: Path, rows: list[dict]) -> Path:
    cards = []
    for row in rows:
        projection_summary = " · ".join(
            f"{name}: valid={valid:.3f}, clipped={clipped:.3f}"
            for name, valid, clipped in zip(
                row["projection_names"],
                row["valid_fraction"],
                row["clipped_fraction"],
                strict=True,
            )
        )
        cards.append(
            f"""<article class="card"><h2>episode {row['episode_id']}</h2>
<p>{html.escape(row['instruction'])}</p><p class="meta">{row['frames']} frames ·
{row['skills']} skills<br>{html.escape(projection_summary)}</p>
<video controls preload="metadata" src="{html.escape(row['video'])}"></video>
<details><summary>skill endpoint contact sheet</summary>
<img src="{html.escape(row['sheet'])}" alt="skill endpoint UV overlay"></details></article>"""
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>ABC dual focus UV preview</title><style>
body{{font-family:system-ui,sans-serif;background:#eef1f6;color:#172033;margin:0}}
header{{position:sticky;top:0;background:#fff;padding:18px 24px;border-bottom:1px solid #ccd3df;z-index:2}}
main{{max-width:1500px;margin:24px auto;padding:0 18px}}
.card{{background:#fff;border:1px solid #d8dee9;border-radius:12px;padding:18px;margin-bottom:24px;box-shadow:0 3px 14px #15203b16}}
h1,h2{{margin:0 0 8px}} .meta{{color:#58657a}}
video,img{{display:block;width:min(100%,1400px);margin:14px auto;border-radius:8px;background:#0c0f14}}
summary{{cursor:pointer;font-weight:650;margin-top:12px}} .left{{color:#00aeba}} .right{{color:#cf20b9}}
</style></head><body><header><h1>ABC dual focus-UV projection check</h1>
<span class="left">● Left EEF</span> &nbsp; <span class="right">● Right EEF</span>
&nbsp; ring/cross = current skill endpoint UV · Left wrist | Top | Right wrist</header>
<main>{''.join(cards)}</main></body></html>"""
    path = output_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.scale < 1:
        raise ValueError("--episodes and --scale must be positive")
    skillvla_dir = args.skillvla_dir.expanduser().resolve()
    output_dir = (args.output_dir or skillvla_dir / "dual_focus_uv_preview").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[7]
    native_dir = skillvla_dir / "skillvla"
    marker = load_json(native_dir / "native_dataset.json")
    raw_root = Path(marker["raw_root"]).expanduser()
    if not raw_root.is_absolute():
        raw_root = native_dir / raw_root
    raw_root = raw_root.resolve()
    annotation_path = native_dir / marker["skill_annotations"]

    with np.load(annotation_path, allow_pickle=False) as archive:
        available = [int(value) for value in archive["episode_id"]]
        if args.episode_ids:
            selected = list(args.episode_ids)
        else:
            selected = [
                episode_id
                for episode_id in available
                if str(marker["sources"][episode_id]["path"]).split("/", 1)[0] == args.task
            ][: args.episodes]
        if not selected:
            raise ValueError(f"No annotated episodes found for task {args.task!r}")

        reports = []
        for position, episode_id in enumerate(selected, start=1):
            episode_dir = raw_root / marker["sources"][episode_id]["path"]
            print(f"[{position}/{len(selected)}] projecting episode {episode_id}: {episode_dir}", flush=True)
            reports.append(
                write_episode_preview(
                    episode_id=episode_id,
                    episode_dir=episode_dir,
                    annotation=annotation_for_episode(archive, episode_id),
                    project_root=project_root,
                    output_dir=output_dir,
                    scale=args.scale,
                    fps=args.fps,
                )
            )

    (output_dir / "report.json").write_text(
        json.dumps({"task": args.task, "episodes": reports}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ABC dual-focus UV preview: {write_html(output_dir, reports)}")


if __name__ == "__main__":
    main()
