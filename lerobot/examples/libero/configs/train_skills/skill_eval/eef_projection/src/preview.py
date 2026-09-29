#!/usr/bin/env python3
"""Visualize skill-end attention targets in agent and wrist camera patches."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from lerobot.datasets.video_utils import decode_video_frames
from lerobot.policies.skillVLA.focus_projection import (
    camera_transform_from_recorded_xml,
    project_eef,
)
from lerobot.policies.skill_expert.wrist_patch_target import (
    WristCamera,
    patch_labels,
    project_into_wrist,
)

_HERE = Path(__file__).resolve().parent
_SKILL_EVAL_SRC = _HERE.parents[3] / "train_skillVLA" / "stage1_skill_eval" / "src"
sys.path.insert(0, str(_SKILL_EVAL_SRC))

from skill_data import SkillEvaluationDataset  # noqa: E402

log = logging.getLogger("skill_attention_preview")


class EpisodeFrameReader:
    """Decode only requested frames from one camera stream."""

    def __init__(self, dataset_dir: Path, *, video_key: str) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.video_key = str(video_key)
        info = json.loads((self.dataset_dir / "meta" / "info.json").read_text())
        if self.video_key not in (info.get("features") or {}):
            raise KeyError(f"Dataset has no video feature {self.video_key!r}.")
        self.fps = float(info["fps"])
        self.path_template = str(info["video_path"])
        files = sorted((self.dataset_dir / "meta" / "episodes").glob("**/*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No episode metadata under {self.dataset_dir / 'meta/episodes'}"
            )
        columns = [
            "episode_index",
            "length",
            f"videos/{self.video_key}/chunk_index",
            f"videos/{self.video_key}/file_index",
            f"videos/{self.video_key}/from_timestamp",
        ]
        self.index = pd.concat(
            [pd.read_parquet(path, columns=columns) for path in files],
            ignore_index=True,
        ).set_index("episode_index", drop=False)

    def episode_length(self, episode_id: int) -> int:
        return int(self.index.loc[int(episode_id), "length"])

    def frames(self, episode_id: int, frame_indices: list[int]) -> dict[int, np.ndarray]:
        row = self.index.loc[int(episode_id)]
        ordered = sorted({int(index) for index in frame_indices})
        video_path = self.dataset_dir / self.path_template.format(
            video_key=self.video_key,
            chunk_index=int(row[f"videos/{self.video_key}/chunk_index"]),
            file_index=int(row[f"videos/{self.video_key}/file_index"]),
        )
        base = float(row[f"videos/{self.video_key}/from_timestamp"])
        timestamps = [base + index / self.fps for index in ordered]
        decoded = decode_video_frames(video_path, timestamps, tolerance_s=0.5 / self.fps)
        images = (
            (decoded.clamp(0.0, 1.0) * 255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        return {index: images[position].copy() for position, index in enumerate(ordered)}


def _available_task_ids(dataset, *, episodes_per_task: int) -> list[int]:
    counts: dict[int, int] = {}
    for episode_id, source in dataset.sources.items():
        if episode_id in dataset._rows_by_episode and episode_id in dataset.episode_meta.index:
            counts[source.task_id] = counts.get(source.task_id, 0) + 1
    return sorted(
        task_id for task_id, count in counts.items() if count >= episodes_per_task
    )


def _selected_task_ids(raw: str, available: list[int]) -> list[int]:
    if raw.strip().lower() == "all":
        selected = available
    else:
        requested = [int(value) for value in json.loads(raw)]
        available_set = set(available)
        selected = [value for value in requested if value in available_set]
        skipped = [value for value in requested if value not in available_set]
        if skipped:
            log.warning("Skipping task IDs without enough exact episodes: %s", skipped)
    if not selected:
        raise RuntimeError("No selected task has enough episode-exact skill episodes.")
    return selected


def _frame_indices(start: int, end: int, count: int) -> list[int]:
    """Even, unique samples that always contain both skill boundaries."""
    if start > end:
        raise ValueError(f"Invalid frame interval [{start}, {end}].")
    if start == end:
        return [start]
    return sorted(
        {
            int(round(value))
            for value in np.linspace(start, end, min(count, end - start + 1))
        }
    )


def _cell(pixel: np.ndarray | tuple[float, float], *, grid: int, height: int, width: int) -> tuple[int, int] | None:
    x, y = float(pixel[0]), float(pixel[1])
    column = int(np.floor(x / width * grid))
    row = int(np.floor(y / height * grid))
    return (row, column) if 0 <= row < grid and 0 <= column < grid else None


def _soft_weights(cell: tuple[int, int], *, grid: int, sigma: float) -> np.ndarray:
    rows, columns = np.mgrid[:grid, :grid]
    if sigma <= 0:
        weights = np.zeros((grid, grid), dtype=np.float64)
        weights[cell] = 1.0
        return weights
    squared = (rows - cell[0]) ** 2 + (columns - cell[1]) ** 2
    weights = np.exp(-squared / (2.0 * sigma**2))
    return weights / weights.sum()


def _draw_attention_target(
    frame: np.ndarray,
    *,
    target_pixel: np.ndarray | tuple[float, float] | None,
    eef_pixel: np.ndarray | tuple[float, float] | None,
    target_visible: bool,
    grid: int,
    sigma: float,
    noisy_pixels: np.ndarray | None = None,
) -> np.ndarray:
    """Orange heatmap = patch target, red = skill end, blue = current EEF."""
    picture = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    height, width = picture.height, picture.width
    overlay = Image.new("RGBA", picture.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    target_cell = (
        _cell(target_pixel, grid=grid, height=height, width=width)
        if target_visible and target_pixel is not None
        else None
    )
    if target_cell is not None:
        weights = _soft_weights(target_cell, grid=grid, sigma=sigma)
        peak = float(weights.max())
        for row in range(grid):
            for column in range(grid):
                strength = float(weights[row, column] / peak)
                if strength < 0.02:
                    continue
                draw.rectangle(
                    [
                        (column * width / grid, row * height / grid),
                        ((column + 1) * width / grid - 1, (row + 1) * height / grid - 1),
                    ],
                    fill=(255, 105, 0, int(25 + 125 * strength)),
                )
    for index in range(1, grid):
        draw.line(
            [(index * width / grid, 0), (index * width / grid, height)],
            fill=(255, 255, 255, 55),
        )
        draw.line(
            [(0, index * height / grid), (width, index * height / grid)],
            fill=(255, 255, 255, 55),
        )
    picture = Image.alpha_composite(picture.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(picture)
    if eef_pixel is not None and _cell(eef_pixel, grid=grid, height=height, width=width) is not None:
        x, y = float(eef_pixel[0]), float(eef_pixel[1])
        draw.rectangle([(x - 5, y - 5), (x + 5, y + 5)], outline=(30, 180, 255), width=2)
    if target_cell is not None and target_pixel is not None:
        x, y = float(target_pixel[0]), float(target_pixel[1])
        draw.line([(x - 9, y), (x + 9, y)], fill=(255, 35, 35), width=2)
        draw.line([(x, y - 9), (x, y + 9)], fill=(255, 35, 35), width=2)
        draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], outline=(255, 235, 0), width=2)
    if noisy_pixels is not None:
        for pixel in np.asarray(noisy_pixels, dtype=np.float64):
            if not np.isfinite(pixel).all():
                continue
            if _cell(pixel, grid=grid, height=height, width=width) is None:
                continue
            x, y = float(pixel[0]), float(pixel[1])
            draw.ellipse(
                [(x - 2, y - 2), (x + 2, y + 2)],
                fill=(190, 45, 235),
                outline=(255, 255, 255),
                width=1,
            )
    return np.asarray(picture, dtype=np.uint8)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    Image.fromarray(image).save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _projection_record(
    pixel: np.ndarray | tuple[float, float] | None,
    *,
    visible: bool,
    valid: bool,
    grid: int,
    height: int,
    width: int,
    reason: str,
) -> dict:
    cell = _cell(pixel, grid=grid, height=height, width=width) if visible and pixel is not None else None
    return {
        "pixel": None if pixel is None else [float(pixel[0]), float(pixel[1])],
        "cell": None if cell is None else [int(cell[0]), int(cell[1])],
        "visible": bool(visible),
        "valid": bool(valid),
        "reason": str(reason),
    }


def _agent_projection(
    xyz: np.ndarray,
    transform: np.ndarray,
    *,
    grid: int,
    height: int,
    width: int,
) -> dict:
    homogeneous = np.concatenate([np.asarray(xyz, dtype=np.float64), np.ones(1)])
    depth = float((np.asarray(transform, dtype=np.float64) @ homogeneous)[2])
    if not np.isfinite(depth) or depth <= 1e-8:
        return _projection_record(
            None,
            visible=False,
            valid=False,
            grid=grid,
            height=height,
            width=width,
            reason="behind camera",
        )
    projected = project_eef(xyz, transform, height=height, width=width)
    reason = "active" if projected.valid else "outside image"
    return _projection_record(
        projected.raw_xy,
        visible=projected.valid,
        valid=projected.valid,
        grid=grid,
        height=height,
        width=width,
        reason=reason,
    )


def _wrist_projections(
    states: np.ndarray,
    target_xyz: np.ndarray,
    *,
    grid: int,
    height: int,
    width: int,
) -> tuple[list[dict], list[np.ndarray | None]]:
    count = len(states)
    eef_xyz = torch.as_tensor(states[:, :3], dtype=torch.float32)
    axis_angle = torch.as_tensor(states[:, 3:6], dtype=torch.float32)
    targets = torch.as_tensor(
        np.repeat(np.asarray(target_xyz, dtype=np.float32)[None], count, axis=0)
    )
    camera = WristCamera()
    labels, target_pixels, valid = patch_labels(
        eef_xyz,
        axis_angle,
        targets,
        camera,
        grid=grid,
        height=height,
        width=width,
    )
    _, depth = project_into_wrist(
        eef_xyz, axis_angle, targets, camera, height=height, width=width
    )
    eef_pixels, eef_depth = project_into_wrist(
        eef_xyz, axis_angle, eef_xyz, camera, height=height, width=width
    )
    records: list[dict] = []
    current_pixels: list[np.ndarray | None] = []
    for index in range(count):
        pixel = target_pixels[index].detach().cpu().numpy()
        in_view = bool(
            float(depth[index]) > 1e-8
            and _cell(pixel, grid=grid, height=height, width=width) is not None
        )
        is_valid = bool(valid[index])
        if not in_view:
            reason = "behind camera" if float(depth[index]) <= 1e-8 else "outside image"
        elif not is_valid:
            reason = "masked: same patch as current EEF"
        else:
            reason = "active"
        record = _projection_record(
            pixel,
            visible=in_view,
            valid=is_valid,
            grid=grid,
            height=height,
            width=width,
            reason=reason,
        )
        if is_valid:
            record["label"] = int(labels[index])
        records.append(record)
        eef_pixel = eef_pixels[index].detach().cpu().numpy()
        eef_visible = bool(
            float(eef_depth[index]) > 1e-8
            and _cell(eef_pixel, grid=grid, height=height, width=width) is not None
        )
        current_pixels.append(eef_pixel if eef_visible else None)
    return records, current_pixels


def _noise_key(std_m: float) -> str:
    millimetres = f"{float(std_m) * 1000.0:g}".replace(".", "p")
    return f"sigma_{millimetres}mm"


def _standard_noise(*, seed: int, uid: str, samples: int) -> np.ndarray:
    """Stable N(0,1) draws reused across levels, frames, and both cameras."""
    digest = hashlib.blake2b(
        f"{int(seed)}:{uid}:input_xyz_noise".encode("utf-8"), digest_size=8
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little", signed=False))
    return rng.normal(size=(int(samples), 3)).astype(np.float64)


def _project_agent_samples(
    xyz: np.ndarray,
    transform: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(
        [np.asarray(xyz, dtype=np.float64), np.ones((len(xyz), 1), dtype=np.float64)],
        axis=1,
    )
    projected = points @ np.asarray(transform, dtype=np.float64).T
    depth = projected[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        columns = projected[:, 0] / depth
        rows = projected[:, 1] / depth
    pixels = np.stack([(width - 1) - columns, rows], axis=1)
    visible = (
        np.isfinite(depth)
        & (depth > 1e-8)
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= width - 1.0)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= height - 1.0)
    )
    return pixels, visible


def _project_wrist_samples(
    state: np.ndarray,
    targets: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(targets)
    state = np.asarray(state, dtype=np.float32)
    eef = torch.as_tensor(np.repeat(state[None, :3], count, axis=0))
    axis_angle = torch.as_tensor(np.repeat(state[None, 3:6], count, axis=0))
    pixels, depth = project_into_wrist(
        eef,
        axis_angle,
        torch.as_tensor(np.asarray(targets, dtype=np.float32)),
        WristCamera(),
        height=height,
        width=width,
    )
    pixels_np = pixels.detach().cpu().numpy().astype(np.float64)
    depth_np = depth.detach().cpu().numpy().astype(np.float64)
    visible = (
        np.isfinite(depth_np)
        & (depth_np > 1e-8)
        & np.isfinite(pixels_np).all(axis=1)
        & (pixels_np[:, 0] >= 0.0)
        & (pixels_np[:, 0] <= width - 1.0)
        & (pixels_np[:, 1] >= 0.0)
        & (pixels_np[:, 1] <= height - 1.0)
    )
    return pixels_np, visible


def _noise_metrics(
    clean_pixel: np.ndarray | list[float] | tuple[float, float] | None,
    noisy_pixels: np.ndarray,
    visible: np.ndarray,
    *,
    grid: int,
    height: int,
    width: int,
) -> tuple[dict, dict]:
    total = int(len(noisy_pixels))
    visible = np.asarray(visible, dtype=bool)
    clean_cell = (
        _cell(clean_pixel, grid=grid, height=height, width=width)
        if clean_pixel is not None
        else None
    )
    shifts = np.empty(0, dtype=np.float64)
    changed = 0
    if clean_pixel is not None and bool(visible.any()):
        clean = np.asarray(clean_pixel, dtype=np.float64)
        shifts = np.linalg.norm(noisy_pixels[visible] - clean[None], axis=1)
        if clean_cell is not None:
            changed = sum(
                _cell(pixel, grid=grid, height=height, width=width) != clean_cell
                for pixel in noisy_pixels[visible]
            )
    visible_count = int(visible.sum())
    public = {
        "samples": total,
        "visible_fraction": visible_count / max(total, 1),
        "outside_fraction": (total - visible_count) / max(total, 1),
        "median_shift_px": float(np.median(shifts)) if len(shifts) else None,
        "p95_shift_px": float(np.percentile(shifts, 95)) if len(shifts) else None,
        "patch_change_fraction": changed / max(visible_count, 1),
    }
    aggregate = {
        "total": total,
        "visible": visible_count,
        "changed": int(changed),
        "shifts": shifts.tolist(),
    }
    return public, aggregate


def _merge_noise_aggregate(target: dict, update: dict) -> None:
    target["total"] += int(update["total"])
    target["visible"] += int(update["visible"])
    target["changed"] += int(update["changed"])
    target["shifts"].extend(float(value) for value in update["shifts"])


def _finalize_noise_aggregate(values: dict) -> dict:
    shifts = np.asarray(values["shifts"], dtype=np.float64)
    visible = int(values["visible"])
    total = int(values["total"])
    return {
        "samples": total,
        "visible_fraction": visible / max(total, 1),
        "outside_fraction": (total - visible) / max(total, 1),
        "median_shift_px": float(np.median(shifts)) if len(shifts) else None,
        "p95_shift_px": float(np.percentile(shifts, 95)) if len(shifts) else None,
        "patch_change_fraction": int(values["changed"]) / max(visible, 1),
    }


def _token_coord(token: int, levels: list[int]) -> list[int]:
    result = []
    base = 1
    for level in levels:
        result.append((int(token) // base) % int(level))
        base *= int(level)
    return result


def _report_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(str(payload["title"]))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#dce3ed;--blue:#1570ef;--orange:#f97316;--purple:#a21caf;--bg:#f4f7fb;--ok:#067647;--bad:#b42318}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
main{{max-width:1800px;margin:auto;padding:24px}}h1{{margin:0 0 5px;font-size:28px}}.lead{{color:var(--muted);margin:0 0 16px}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px 13px;margin-bottom:14px}}.red{{color:#e11d48}}.blue{{color:#0284c7}}.orange{{color:#ea580c}}.purple{{color:var(--purple)}}
.toolbar{{position:sticky;top:0;z-index:5;display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:#f4f7fbeF;padding:10px 0;backdrop-filter:blur(8px)}}select{{padding:7px 10px;border:1px solid #cbd5e1;border-radius:8px;background:#fff}}.summary{{color:var(--muted)}}
.card{{background:#fff;border:1px solid var(--line);border-radius:13px;margin:0 0 14px;overflow:hidden;box-shadow:0 5px 18px #23324b0a}}.card h2{{font-size:15px;margin:0;padding:10px 13px;background:#f8fafc;border-bottom:1px solid var(--line)}}.language{{color:var(--muted);font-weight:400}}
.scroller{{overflow-x:auto}}table{{border-collapse:collapse;min-width:100%;table-layout:fixed}}th,td{{border-right:1px solid var(--line);border-bottom:1px solid var(--line);padding:6px;text-align:center;vertical-align:top}}th{{font-size:12px;background:#fbfcfe;white-space:nowrap}}th.camera{{position:sticky;left:0;z-index:2;min-width:82px;background:#fff;font-size:13px}}td{{min-width:210px}}img{{width:200px;height:200px;object-fit:contain;background:#111;display:block;margin:auto}}.cap{{font-size:11px;margin-top:4px;color:var(--muted)}}.cap.ok{{color:var(--ok)}}.cap.bad{{color:var(--bad)}}.meta{{padding:8px 13px;color:var(--muted);font-size:12px}}.empty{{padding:60px;text-align:center;color:var(--muted)}}
</style></head><body><main>
<h1>{title}</h1>
<p class="lead">The same skill-end XYZ is projected into both cameras over the skill. No model inference.</p>
<div class="legend"><span class="orange">■ clean GT patch target</span><span class="red">＋ clean skill-end XYZ</span><span class="purple">● noisy input XYZ samples</span><span class="blue">□ current EEF</span><span>Noise never changes the supervision target.</span></div>
<div class="toolbar"><label>Task <select id="task"></select></label><label>FSQ code <select id="token"></select></label><label>Input XYZ noise σ <select id="noise"></select></label><span class="summary" id="summary"></span></div>
<div id="cards"></div>
</main><script>
const DATA={data}; const records=DATA.records;
const esc=(v)=>String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const task=document.getElementById('task'),token=document.getElementById('token'),noise=document.getElementById('noise');
function options(select,values,prefix){{select.innerHTML='<option value="all">all</option>'+values.map(v=>`<option value="${{v}}">${{prefix}}${{v}}</option>`).join('')}}
options(task,[...new Set(records.map(r=>r.task_id))].sort((a,b)=>a-b),'task ');options(token,[...new Set(records.map(r=>r.token))].sort((a,b)=>a-b),'code ');
noise.innerHTML=DATA.noise_levels.map(n=>`<option value="${{n.key}}">${{n.std_mm.toFixed(1)}} mm</option>`).join('');
function cap(p){{const where=p.pixel?`pixel (${{p.pixel[0].toFixed(1)}}, ${{p.pixel[1].toFixed(1)}})`:'no pixel';const cell=p.cell?`patch (${{p.cell[0]}}, ${{p.cell[1]}})`:'no patch';return `${{where}} · ${{cell}} · ${{esc(p.reason)}}`}}
function noiseCap(m){{const med=m.median_shift_px==null?'n/a':m.median_shift_px.toFixed(1),p95=m.p95_shift_px==null?'n/a':m.p95_shift_px.toFixed(1);return `noise: median ${{med}}px · p95 ${{p95}}px · patch changed ${{(100*m.patch_change_fraction).toFixed(1)}}% · outside ${{(100*m.outside_fraction).toFixed(1)}}%`}}
function cameraRow(record,name,label){{const key=noise.value;return `<tr><th class="camera">${{label}}</th>${{record.frames.map(f=>{{const p=f.projections[name],m=f.noise[name][key];return `<td><img loading="lazy" src="${{esc(f.images[name][key])}}"><div class="cap ${{p.valid?'ok':'bad'}}">${{cap(p)}}</div><div class="cap">${{noiseCap(m)}}</div></td>`}}).join('')}}</tr>`}}
function card(r){{const heads=r.frames.map(f=>`<th>frame ${{f.frame}}${{f.endpoint?' · endpoint':''}}</th>`).join('');return `<article class="card"><h2>task ${{r.task_id}} · episode ${{r.episode_id}} · skill ${{r.skill_index}} · code ${{r.token}} [${{r.code_coord.join(', ')}}] <span class="language">${{esc(r.task_description)}}</span></h2><div class="scroller"><table><tr><th class="camera"></th>${{heads}}</tr>${{cameraRow(r,'agent','agent')}}${{cameraRow(r,'wrist','wrist')}}</table></div><div class="meta">frames [${{r.frame_start}}, ${{r.frame_end}}) · endpoint frame ${{r.endpoint_frame}} · endpoint XYZ [${{r.endpoint_xyz.map(v=>v.toFixed(4)).join(', ')}}]</div></article>`}}
function render(){{const rows=records.filter(r=>(task.value==='all'||r.task_id===Number(task.value))&&(token.value==='all'||r.token===Number(token.value))),n=DATA.noise_stats[noise.value];document.getElementById('summary').textContent=`${{rows.length}} / ${{records.length}} skills · agent p95 ${{n.agent.p95_shift_px?.toFixed(1)??'n/a'}}px / patch Δ ${{(100*n.agent.patch_change_fraction).toFixed(1)}}% · wrist p95 ${{n.wrist.p95_shift_px?.toFixed(1)??'n/a'}}px / patch Δ ${{(100*n.wrist.patch_change_fraction).toFixed(1)}}%`;document.getElementById('cards').innerHTML=rows.length?rows.map(card).join(''):'<div class="empty">No matching skills.</div>'}}
task.onchange=render;token.onchange=render;noise.onchange=render;render();
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skill-dataset-dir", type=Path, required=True)
    parser.add_argument("--skill-latents-path", type=Path, required=True)
    parser.add_argument("--eval-init-states-path", type=Path, required=True)
    parser.add_argument("--original-dataset-dir", type=Path, required=True)
    parser.add_argument("--target-task", required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--episode-ids", default="[]")
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--episode-selection", choices=("first", "random"), required=True)
    parser.add_argument("--agent-camera", default="agentview")
    parser.add_argument("--agent-video-key", default="observation.images.image")
    parser.add_argument("--wrist-video-key", default="observation.images.wrist_image")
    parser.add_argument("--patch-grid", type=int, default=14)
    parser.add_argument("--soft-sigma", type=float, default=0.7)
    parser.add_argument("--frames-per-skill", type=int, default=5)
    parser.add_argument("--noise-std-m", default="[0.0,0.005,0.01,0.02,0.05]")
    parser.add_argument("--noise-samples", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    noise_std_m = sorted(float(value) for value in json.loads(args.noise_std_m))
    if (
        args.patch_grid <= 0
        or args.soft_sigma < 0
        or args.frames_per_skill < 2
        or args.noise_samples <= 0
        or not noise_std_m
        or any(not np.isfinite(value) or value < 0 for value in noise_std_m)
        or len(noise_std_m) != len(set(noise_std_m))
    ):
        raise ValueError(
            "Require patch_grid>0, soft_sigma>=0, frames_per_skill>=2, "
            "noise_samples>0, and unique finite non-negative noise levels."
        )

    dataset = SkillEvaluationDataset(
        skill_dataset_dir=args.skill_dataset_dir,
        skill_latents_path=args.skill_latents_path,
        eval_init_states_path=args.eval_init_states_path,
        original_dataset_dir=args.original_dataset_dir,
        suite_name=args.target_task,
    )
    info = json.loads((args.skill_dataset_dir / "meta" / "info.json").read_text())
    levels = [int(value) for value in info.get("skill_fsq_levels", [])]
    if not levels:
        raise ValueError("SkillVLA metadata has no skill_fsq_levels.")
    codebook_size = int(np.prod(levels, dtype=np.int64))

    available = _available_task_ids(dataset, episodes_per_task=args.episodes_per_task)
    task_ids = _selected_task_ids(args.task_ids, available)
    selected = dataset.select_episodes(
        task_ids=task_ids,
        episodes_per_task=args.episodes_per_task,
        selection=args.episode_selection,
        seed=args.seed,
        explicit_episode_ids=[int(value) for value in json.loads(args.episode_ids)],
    )
    occurrences = dataset.occurrences(selected)
    invalid = sorted({item.token for item in occurrences if not 0 <= item.token < codebook_size})
    if invalid:
        raise ValueError(f"Skill assignments exceed FSQ{levels}: {invalid}.")

    agent_reader = EpisodeFrameReader(args.skill_dataset_dir, video_key=args.agent_video_key)
    wrist_reader = EpisodeFrameReader(args.skill_dataset_dir, video_key=args.wrist_video_key)
    by_episode: dict[int, list] = defaultdict(list)
    for occurrence in occurrences:
        by_episode[occurrence.episode_id].append(occurrence)

    records: list[dict] = []
    totals = {"samples": 0, "agent_valid": 0, "wrist_visible": 0, "wrist_valid": 0}
    noise_levels = [
        {"key": _noise_key(std_m), "std_m": std_m, "std_mm": std_m * 1000.0}
        for std_m in noise_std_m
    ]
    noise_aggregate = {
        level["key"]: {
            camera: {"total": 0, "visible": 0, "changed": 0, "shifts": []}
            for camera in ("agent", "wrist")
        }
        for level in noise_levels
    }
    for episode_id in sorted(by_episode):
        episode_length = agent_reader.episode_length(episode_id)
        if wrist_reader.episode_length(episode_id) != episode_length:
            raise ValueError(f"Agent/wrist episode lengths differ for episode {episode_id}.")
        aligned = dataset.load_aligned_episode(episode_id)
        if not aligned.model_xml:
            raise ValueError(
                "Attention preview requires a recorded LIBERO model XML; "
                f"episode {episode_id} does not provide one."
            )
        states = np.asarray(aligned.filtered_states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] < 6:
            raise ValueError(
                f"Episode {episode_id} observation.state needs xyz + axis-angle, got {states.shape}."
            )
        sampled: dict[str, list[int]] = {}
        for occurrence in by_episode[episode_id]:
            endpoint = min(int(occurrence.frame_end), episode_length - 1)
            sampled[occurrence.uid] = _frame_indices(
                int(occurrence.frame_start), endpoint, args.frames_per_skill
            )
        needed = sorted({frame for values in sampled.values() for frame in values})
        agent_frames = agent_reader.frames(episode_id, needed)
        wrist_frames = wrist_reader.frames(episode_id, needed)
        agent_height, agent_width = agent_frames[needed[0]].shape[:2]
        wrist_height, wrist_width = wrist_frames[needed[0]].shape[:2]
        agent_transform = camera_transform_from_recorded_xml(
            aligned.model_xml,
            camera_name=args.agent_camera,
            height=agent_height,
            width=agent_width,
        )
        grounding = (
            np.asarray(aligned.episode_start_xyz, dtype=np.float64)
            if dataset.proprio_grounding == "episode_start_xyz"
            else np.zeros(3, dtype=np.float64)
        )
        if dataset.proprio_grounding not in {"none", "episode_start_xyz"}:
            raise ValueError(f"Unsupported proprio grounding: {dataset.proprio_grounding!r}.")

        for occurrence in by_episode[episode_id]:
            frames = sampled[occurrence.uid]
            endpoint_frame = min(int(occurrence.frame_end), episode_length - 1)
            endpoint_grounded = states[endpoint_frame, :3].copy()
            endpoint_world = endpoint_grounded + grounding
            standard_noise = _standard_noise(
                seed=args.seed, uid=occurrence.uid, samples=args.noise_samples
            )
            noise_deltas = {
                level["key"]: standard_noise * float(level["std_m"])
                for level in noise_levels
            }
            agent_noisy = {}
            for level in noise_levels:
                key = level["key"]
                pixels, visible = _project_agent_samples(
                    endpoint_world[None] + noise_deltas[key],
                    agent_transform,
                    height=agent_height,
                    width=agent_width,
                )
                agent_noisy[key] = (pixels, visible)
            wrist_records, wrist_eef_pixels = _wrist_projections(
                states[frames],
                endpoint_grounded,
                grid=args.patch_grid,
                height=wrist_height,
                width=wrist_width,
            )
            relative_dir = (
                Path("images")
                / f"task_{occurrence.task_id:02d}"
                / f"token_{occurrence.token:04d}"
                / occurrence.uid
            )
            frame_rows = []
            for position, frame_index in enumerate(frames):
                agent_target = _agent_projection(
                    endpoint_world,
                    agent_transform,
                    grid=args.patch_grid,
                    height=agent_height,
                    width=agent_width,
                )
                agent_eef = _agent_projection(
                    states[frame_index, :3] + grounding,
                    agent_transform,
                    grid=args.patch_grid,
                    height=agent_height,
                    width=agent_width,
                )
                wrist_target = wrist_records[position]
                images = {"agent": {}, "wrist": {}}
                noise_metrics = {"agent": {}, "wrist": {}}
                for level in noise_levels:
                    key, std_m = level["key"], float(level["std_m"])
                    agent_pixels, agent_visible = agent_noisy[key]
                    wrist_pixels, wrist_visible = _project_wrist_samples(
                        states[frame_index],
                        endpoint_grounded[None] + noise_deltas[key],
                        height=wrist_height,
                        width=wrist_width,
                    )
                    agent_metric, agent_update = _noise_metrics(
                        agent_target["pixel"],
                        agent_pixels,
                        agent_visible,
                        grid=args.patch_grid,
                        height=agent_height,
                        width=agent_width,
                    )
                    wrist_metric, wrist_update = _noise_metrics(
                        wrist_target["pixel"],
                        wrist_pixels,
                        wrist_visible,
                        grid=args.patch_grid,
                        height=wrist_height,
                        width=wrist_width,
                    )
                    _merge_noise_aggregate(noise_aggregate[key]["agent"], agent_update)
                    _merge_noise_aggregate(noise_aggregate[key]["wrist"], wrist_update)
                    noise_metrics["agent"][key] = agent_metric
                    noise_metrics["wrist"][key] = wrist_metric

                    agent_path = relative_dir / f"frame_{frame_index:04d}_agent_{key}.png"
                    wrist_path = relative_dir / f"frame_{frame_index:04d}_wrist_{key}.png"
                    _write_png(
                        args.output_dir / agent_path,
                        _draw_attention_target(
                            agent_frames[frame_index],
                            target_pixel=agent_target["pixel"],
                            eef_pixel=agent_eef["pixel"] if agent_eef["visible"] else None,
                            target_visible=agent_target["visible"],
                            noisy_pixels=(
                                None
                                if std_m == 0.0
                                else np.where(agent_visible[:, None], agent_pixels, np.nan)
                            ),
                            grid=args.patch_grid,
                            sigma=args.soft_sigma,
                        ),
                    )
                    _write_png(
                        args.output_dir / wrist_path,
                        _draw_attention_target(
                            wrist_frames[frame_index],
                            target_pixel=wrist_target["pixel"],
                            eef_pixel=wrist_eef_pixels[position],
                            target_visible=wrist_target["visible"],
                            noisy_pixels=(
                                None
                                if std_m == 0.0
                                else np.where(wrist_visible[:, None], wrist_pixels, np.nan)
                            ),
                            grid=args.patch_grid,
                            sigma=args.soft_sigma,
                        ),
                    )
                    images["agent"][key] = agent_path.as_posix()
                    images["wrist"][key] = wrist_path.as_posix()
                frame_rows.append(
                    {
                        "frame": int(frame_index),
                        "endpoint": int(frame_index) == endpoint_frame,
                        "images": images,
                        "projections": {"agent": agent_target, "wrist": wrist_target},
                        "noise": noise_metrics,
                    }
                )
                totals["samples"] += 1
                totals["agent_valid"] += int(agent_target["valid"])
                totals["wrist_visible"] += int(wrist_target["visible"])
                totals["wrist_valid"] += int(wrist_target["valid"])
            records.append(
                {
                    "task_id": int(occurrence.task_id),
                    "episode_id": int(occurrence.episode_id),
                    "skill_index": int(occurrence.skill_index),
                    "token": int(occurrence.token),
                    "code_coord": _token_coord(occurrence.token, levels),
                    "frame_start": int(occurrence.frame_start),
                    "frame_end": int(occurrence.frame_end),
                    "endpoint_frame": endpoint_frame,
                    "endpoint_xyz": endpoint_world.astype(float).tolist(),
                    "task_description": dataset.episode_task_description(episode_id),
                    "frames": frame_rows,
                }
            )
        log.info("Processed episode %s (%s skills)", episode_id, len(by_episode[episode_id]))

    samples = max(totals["samples"], 1)
    noise_stats = {
        level["key"]: {
            camera: _finalize_noise_aggregate(noise_aggregate[level["key"]][camera])
            for camera in ("agent", "wrist")
        }
        for level in noise_levels
    }
    payload = {
        "format": "skill_attention_target_preview_v1",
        "title": "Skill attention targets: agent + wrist",
        "levels": levels,
        "patch_grid": args.patch_grid,
        "soft_sigma": args.soft_sigma,
        "frames_per_skill": args.frames_per_skill,
        "agent_camera": args.agent_camera,
        "agent_video_key": args.agent_video_key,
        "wrist_video_key": args.wrist_video_key,
        "input_xyz_noise_contract": "Gaussian XYZ input noise; clean GT projection target",
        "noise_samples_per_level": args.noise_samples,
        "noise_levels": noise_levels,
        "noise_stats": noise_stats,
        "proprio_grounding": dataset.proprio_grounding,
        "stats": {
            "sample_count": totals["samples"],
            "agent_valid_fraction": totals["agent_valid"] / samples,
            "wrist_visible_fraction": totals["wrist_visible"] / samples,
            "wrist_valid_fraction": totals["wrist_valid"] / samples,
        },
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(args.output_dir / "manifest.json", json.dumps(payload, indent=2) + "\n")
    _atomic_text(args.output_dir / "index.html", _report_html(payload))
    log.info("Wrote %s skill records to %s", len(records), args.output_dir / "index.html")


if __name__ == "__main__":
    main()
