#!/usr/bin/env python3
"""Build a codebook-linked preview of endpoint-centered foveated top images.

This intentionally does not load an FSQ model or run a simulator. Segment
boundaries and codes come from an already-built SkillVLA dataset, top frames
come from its videos, and the recorded demo XML supplies the fixed agentview
camera needed to project each terminal EEF position into image pixels.
"""

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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from lerobot.datasets.video_utils import decode_video_frames
from lerobot.policies.skillVLA.focus_projection import (
    camera_transform_from_recorded_xml,
    project_eef,
)

_HERE = Path(__file__).resolve().parent
_SKILL_EVAL_SRC = _HERE.parents[2] / "train_skillVLA" / "stage1_skill_eval" / "src"
sys.path.insert(0, str(_SKILL_EVAL_SRC))

from skill_data import SkillEvaluationDataset  # noqa: E402

log = logging.getLogger("foveated_skill_preview")
VIDEO_KEY = "observation.images.image"


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


class EpisodeFrameReader:
    """Decode only requested top-view frames from a LeRobot video dataset."""

    def __init__(self, dataset_dir: Path, *, video_key: str = VIDEO_KEY) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.video_key = video_key
        info = json.loads((self.dataset_dir / "meta" / "info.json").read_text())
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
            f"videos/{video_key}/chunk_index",
            f"videos/{video_key}/file_index",
            f"videos/{video_key}/from_timestamp",
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


def _camera_transform_from_recorded_xml(
    model_xml: str, *, camera_name: str, height: int, width: int
) -> np.ndarray:
    return camera_transform_from_recorded_xml(
        model_xml, camera_name=camera_name, height=height, width=width
    )


def _project_eef(
    xyz: np.ndarray, transform: np.ndarray, *, height: int, width: int
) -> tuple[int, int]:
    return project_eef(xyz, transform, height=height, width=width).pixel_xy


def foveate_image(
    frame: np.ndarray,
    *,
    center_xy: tuple[int, int],
    shape: str,
    sharp_size: int,
    feather: int,
    blur_radius: float,
) -> np.ndarray:
    """Keep an endpoint-centered patch sharp and strongly blur its periphery."""
    image = np.asarray(frame, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB frame, got {image.shape}.")
    blurred = np.asarray(
        Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=blur_radius)),
        dtype=np.float32,
    )
    height, width = image.shape[:2]
    x0, y0 = (float(center_xy[0]), float(center_xy[1]))
    yy, xx = np.mgrid[:height, :width]
    if shape == "circle":
        distance = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    elif shape == "square":
        distance = np.maximum(np.abs(xx - x0), np.abs(yy - y0))
    else:
        raise ValueError(f"Unknown foveation shape: {shape!r}.")
    half = float(sharp_size) / 2.0
    if feather == 0:
        alpha = (distance <= half).astype(np.float32)
    else:
        alpha = np.clip((half + float(feather) - distance) / float(feather), 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    output = alpha[..., None] * image.astype(np.float32) + (1.0 - alpha[..., None]) * blurred
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def _crop_bounds(
    *,
    center_xy: tuple[int, int],
    crop_size: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    """Return a fixed square crop shifted inside the source at image edges."""
    crop_size = int(crop_size)
    if crop_size <= 0:
        raise ValueError("crop_size must be positive.")
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"crop_size={crop_size} exceeds source image size {width}x{height}."
        )
    left = int(round(float(center_xy[0]) - crop_size / 2.0))
    top = int(round(float(center_xy[1]) - crop_size / 2.0))
    left = int(np.clip(left, 0, width - crop_size))
    top = int(np.clip(top, 0, height - crop_size))
    return left, top, left + crop_size, top + crop_size


def _box_bounds_inside(
    *,
    center_xy: tuple[int, int],
    box_size: int,
    outer_bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Keep a smaller square fully inside an outer crop without padding."""
    outer_left, outer_top, outer_right, outer_bottom = outer_bounds
    outer_width = outer_right - outer_left
    outer_height = outer_bottom - outer_top
    if box_size <= 0:
        raise ValueError("inner box size must be positive.")
    if box_size > outer_width or box_size > outer_height:
        raise ValueError(
            f"inner box size {box_size} exceeds outer crop {outer_width}x{outer_height}."
        )
    left = int(round(float(center_xy[0]) - box_size / 2.0))
    top = int(round(float(center_xy[1]) - box_size / 2.0))
    left = int(np.clip(left, outer_left, outer_right - box_size))
    top = int(np.clip(top, outer_top, outer_bottom - box_size))
    return left, top, left + box_size, top + box_size


def crop_focus_image(
    frame: np.ndarray,
    *,
    center_xy: tuple[int, int],
    crop_size: int,
    output_size: int,
    inner_box_enabled: bool = True,
    inner_box_mode: str = "box",
    inner_box_center_xy: tuple[int, int] | None = None,
    inner_box_size: int = 32,
    inner_box_line_width: int = 3,
    inner_blur_shape: str = "square",
    inner_blur_feather: int = 20,
    inner_blur_radius: float = 8.0,
) -> np.ndarray:
    """Crop around the focus point and add a box or soft-blur inner cue."""
    image = np.asarray(frame, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB frame, got {image.shape}.")
    if output_size <= 0:
        raise ValueError("output_size must be positive.")
    bounds = _crop_bounds(
        center_xy=center_xy,
        crop_size=crop_size,
        height=image.shape[0],
        width=image.shape[1],
    )
    crop = Image.fromarray(image, mode="RGB").crop(bounds)
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    inner = None
    if inner_box_enabled:
        if inner_box_mode not in {"box", "blur"}:
            raise ValueError(f"Unknown inner box mode: {inner_box_mode!r}.")
        marker_center = center_xy if inner_box_center_xy is None else inner_box_center_xy
        inner = _box_bounds_inside(
            center_xy=marker_center,
            box_size=int(inner_box_size),
            outer_bounds=bounds,
        )
        if inner_box_mode == "blur":
            local_center = (
                int(round((inner[0] + inner[2]) / 2.0 - bounds[0])),
                int(round((inner[1] + inner[3]) / 2.0 - bounds[1])),
            )
            crop = Image.fromarray(
                foveate_image(
                    np.asarray(crop, dtype=np.uint8),
                    center_xy=local_center,
                    shape=inner_blur_shape,
                    sharp_size=int(inner_box_size),
                    feather=int(inner_blur_feather),
                    blur_radius=float(inner_blur_radius),
                ),
                mode="RGB",
            )
    output = crop.resize((int(output_size), int(output_size)), resample=resampling)
    if inner_box_enabled and inner_box_mode == "box":
        assert inner is not None
        left, top, _, _ = bounds
        scale = float(output_size) / float(crop_size)
        box = (
            int(round((inner[0] - left) * scale)),
            int(round((inner[1] - top) * scale)),
            int(round((inner[2] - left) * scale)) - 1,
            int(round((inner[3] - top) * scale)) - 1,
        )
        ImageDraw.Draw(output).rectangle(
            box,
            outline=(255, 0, 0),
            width=int(inner_box_line_width),
        )
    return np.asarray(output, dtype=np.uint8)


def draw_crop_window(
    frame: np.ndarray,
    *,
    center_xy: tuple[int, int],
    crop_size: int,
    inner_box_enabled: bool = True,
    inner_box_mode: str = "box",
    inner_box_center_xy: tuple[int, int] | None = None,
    inner_box_size: int = 32,
) -> np.ndarray:
    """Draw the actual edge-adjusted crop window for preview/debugging."""
    image = np.asarray(frame, dtype=np.uint8)
    bounds = _crop_bounds(
        center_xy=center_xy,
        crop_size=crop_size,
        height=image.shape[0],
        width=image.shape[1],
    )
    annotated = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(annotated)
    left, top, right, bottom = bounds
    draw.rectangle((left, top, right - 1, bottom - 1), outline=(255, 122, 0), width=3)
    if inner_box_enabled:
        marker_center = center_xy if inner_box_center_xy is None else inner_box_center_xy
        inner = _box_bounds_inside(
            center_xy=marker_center,
            box_size=int(inner_box_size),
            outer_bounds=bounds,
        )
        draw.rectangle(
            (inner[0], inner[1], inner[2] - 1, inner[3] - 1),
            outline=(255, 0, 0) if inner_box_mode == "box" else (0, 170, 255),
            width=2,
        )
    x, y = center_xy
    draw.ellipse(
        (x - 4, y - 4, x + 4, y + 4),
        fill=(255, 122, 0),
        outline=(0, 0, 0),
        width=1,
    )
    return np.asarray(annotated, dtype=np.uint8)


def _parse_numeric_range(
    raw: str,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> tuple[int, int] | tuple[float, float]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} must be JSON [min, max], got {raw!r}.") from exc
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be JSON [min, max], got {value!r}.")
    cast = int if integer else float
    low, high = cast(value[0]), cast(value[1])
    if low > high:
        raise ValueError(f"{field} minimum cannot exceed its maximum.")
    if minimum is not None and low < minimum:
        raise ValueError(f"{field} values must be >= {minimum}.")
    if maximum is not None and high > maximum:
        raise ValueError(f"{field} values must be <= {maximum}.")
    return low, high


def _rng_for(seed: int, uid: str, variant: str) -> np.random.Generator:
    """Create a stable per-occurrence RNG independent of processing order."""
    digest = hashlib.blake2b(
        f"{int(seed)}:{uid}:{variant}".encode("utf-8"), digest_size=8
    ).digest()
    return np.random.default_rng(int.from_bytes(digest, "little", signed=False))


def _sample_uniform(
    rng: np.random.Generator, bounds: tuple[float, float]
) -> float:
    low, high = bounds
    return float(low) if low == high else float(rng.uniform(low, high))


def _sample_integer(
    rng: np.random.Generator, bounds: tuple[int, int]
) -> int:
    low, high = bounds
    return int(low) if low == high else int(rng.integers(low, high + 1))


def color_randomize_image(
    frame: np.ndarray,
    *,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> np.ndarray:
    """Apply one sampled color transform to a full frame before foveation."""
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    image = ImageEnhance.Brightness(image).enhance(float(brightness))
    image = ImageEnhance.Contrast(image).enhance(float(contrast))
    image = ImageEnhance.Color(image).enhance(float(saturation))
    if hue:
        hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
        shift = int(round(float(hue) * 255.0))
        hsv[..., 0] = (
            hsv[..., 0].astype(np.int16) + shift
        ).astype(np.uint8)
        image = Image.fromarray(hsv, mode="HSV").convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def input_blur_image(frame: np.ndarray, *, radius: float) -> np.ndarray:
    """Blur a full source frame before extracting its foveated sharp region."""
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    return np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.uint8
    )


def _jitter_center(
    center_xy: tuple[int, int],
    *,
    dx: int,
    dy: int,
    height: int,
    width: int,
) -> tuple[int, int]:
    return (
        int(np.clip(int(center_xy[0]) + int(dx), 0, width - 1)),
        int(np.clip(int(center_xy[1]) + int(dy), 0, height - 1)),
    )


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


def _token_to_coord(token: int, levels: list[int]) -> list[int]:
    coordinate: list[int] = []
    base = 1
    for level in levels:
        coordinate.append((int(token) // base) % int(level))
        base *= int(level)
    return coordinate


def _report_html(payload: dict) -> str:
    payload = dict(payload)
    payload.setdefault("randomization", {"variants": []})
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(str(payload["title"]))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--ink:#182230;--muted:#667085;--line:#dfe5ec;--blue:#2563eb;--orange:#f97316;--bg:#f4f7fb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin:0 0 5px;font-size:30px}}.lead{{color:var(--muted);margin:0 0 20px}}
.toolbar{{display:flex;gap:12px;justify-content:space-between;align-items:center;flex-wrap:wrap;margin:0 0 16px}}.button-group{{display:flex;gap:7px;align-items:center;flex-wrap:wrap}}.group-label{{font-size:12px;font-weight:750;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.toggle{{border:1px solid #cbd5e1;background:#fff;color:#344054;border-radius:9px;padding:7px 11px;font-weight:650;cursor:pointer}}.toggle:hover{{border-color:#94a3b8}}.toggle.active{{background:var(--blue);border-color:var(--blue);color:#fff}}.toggle.augment.active{{background:#0f766e;border-color:#0f766e}}.toggle[hidden]{{display:none}}
.layout{{display:grid;grid-template-columns:minmax(360px,500px) 1fr;gap:18px;align-items:start}}.panel{{background:#fff;border:1px solid var(--line);border-radius:15px;padding:16px;box-shadow:0 7px 24px #23324b0b}}
.sticky{{position:sticky;top:14px}}.nav-title{{font-size:16px;margin:0 0 8px}}.nav-block[hidden]{{display:none}}svg{{width:100%;height:auto;display:block;background:#fbfdff;border-radius:12px}}
.edge{{stroke:#cfd8e5;stroke-width:2}}.node{{cursor:pointer;stroke:#fff;stroke-width:3;transition:.12s}}.node.empty{{fill:#d6dce5}}.node.used{{fill:var(--blue)}}.node.active{{fill:var(--orange);stroke:#7c2d12;stroke-width:4}}
.token-label{{font-size:11px;fill:#344054;pointer-events:none;text-anchor:middle}}.chips{{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}}.chip{{border:1px solid #cbd5e1;background:white;border-radius:8px;padding:5px 8px;cursor:pointer}}.chip.active{{background:var(--orange);color:#fff;border-color:var(--orange)}}
.selection{{margin:0 0 13px;font-weight:750;font-size:17px}}.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}}.card{{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#fff}}.variant{{border-top:1px solid var(--line)}}.variant:first-child{{border-top:0}}.variant-head{{display:flex;justify-content:space-between;gap:10px;padding:7px 10px;background:#f8fafc;font-size:12px;font-weight:750}}.variant-detail{{color:var(--muted);font-weight:500;text-align:right}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:var(--line)}}figure{{margin:0;background:#111}}img{{display:block;width:100%;height:auto}}figcaption{{padding:5px 8px;background:#fff;color:var(--muted);font-size:12px}}.meta{{padding:10px 12px}}.language{{margin-top:5px;color:#475467}}.empty-message{{padding:50px 10px;text-align:center;color:var(--muted)}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr}}.sticky{{position:static}}}}
</style></head><body><main>
<h1>{title}</h1>
<p class="lead">Endpoint EEF projection · paired randomization shared by skill start/end · no policy inference or FSQ metrics.<br><span id="summary"></span></p>
<div class="toolbar"><div class="button-group"><span class="group-label">Browse</span><button class="toggle view active" data-view="code">By code</button><button class="toggle view" data-view="task">By task</button></div>
<div class="button-group"><span class="group-label">Show below baseline</span><button class="toggle augment" data-variant="color">Color randomization</button><button class="toggle augment" data-variant="crop">Crop-position randomization</button><button class="toggle augment" data-variant="blur">Blur randomization</button></div></div>
<div class="layout"><section class="panel sticky"><div class="nav-block" id="codeNav"><h2 class="nav-title">FSQ code</h2><svg id="cube" viewBox="0 0 500 420" aria-label="FSQ codebook"></svg><div class="chips" id="codeChips"></div></div><div class="nav-block" id="taskNav" hidden><h2 class="nav-title">Task</h2><div class="chips" id="taskChips"></div></div></section>
<section class="panel"><div class="selection" id="selection"></div><div class="cards" id="cards"></div></section></div>
</main><script>
const DATA={data};
const records=DATA.records, levels=DATA.levels, count=levels.reduce((a,b)=>a*b,1);
const byToken=new Map(); for(const r of records){{if(!byToken.has(r.token))byToken.set(r.token,[]);byToken.get(r.token).push(r)}}
const byTask=new Map(); for(const r of records){{if(!byTask.has(r.task_id))byTask.set(r.task_id,[]);byTask.get(r.task_id).push(r)}}
const coord=(token)=>{{const c=[];let base=1;for(const l of levels){{c.push(Math.floor(token/base)%l);base*=l}}return c}};
const esc=(v)=>String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const availableVariants=new Set((DATA.randomization&&DATA.randomization.variants)||[]), shownVariants=new Set();
const focusSummary=DATA.foveation.mode==='crop'
 ? `crop ${{DATA.foveation.crop_size}}px → ${{DATA.foveation.output_size}}px · inner ${{DATA.foveation.inner_box_mode}} ${{DATA.foveation.inner_box_size}}px`
 : `partial foveation · sharp ${{DATA.foveation.sharp_size}}px · feather ${{DATA.foveation.feather}}px · peripheral blur ${{DATA.foveation.blur_radius}}`;
document.getElementById('summary').textContent=`${{records.length}} skills · ${{byTask.size}} tasks · FSQ[${{levels.join(', ')}}] · ${{focusSummary}}`;
let viewMode='code', selectedToken=[...byToken.keys()].sort((a,b)=>a-b)[0]??0, selectedTask=[...byTask.keys()].sort((a,b)=>a-b)[0]??0;
function point(c){{const a=(c[0]??0)-((levels[0]??1)-1)/2,b=(c[1]??0)-((levels[1]??1)-1)/2,z=(c[2]??0)-((levels[2]??1)-1)/2;return [250+82*(a-b),205+42*(a+b)-92*z]}}
function svgEl(name,attrs){{const e=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e}}
function drawCube(){{const svg=document.getElementById('cube');svg.innerHTML='';if(levels.length!==3){{const t=svgEl('text',{{x:250,y:205,'text-anchor':'middle',fill:'#667085'}});t.textContent=`${{levels.length}}D codebook: use buttons below`;svg.append(t);return}}
 const lookup=new Map();for(let t=0;t<count;t++)lookup.set(coord(t).join(','),t);
 for(let t=0;t<count;t++){{const c=coord(t),p=point(c);for(let axis=0;axis<3;axis++){{if(c[axis]+1>=levels[axis])continue;const n=[...c];n[axis]++;const q=point(n);svg.append(svgEl('line',{{x1:p[0],y1:p[1],x2:q[0],y2:q[1],class:'edge'}}))}}}}
 const order=[...Array(count).keys()].sort((a,b)=>point(coord(a))[1]-point(coord(b))[1]);for(const t of order){{const p=point(coord(t)),used=byToken.has(t);const g=svgEl('g',{{}}),circle=svgEl('circle',{{cx:p[0],cy:p[1],r:used?15:10,class:`node ${{used?'used':'empty'}} ${{t===selectedToken?'active':''}}`}});circle.addEventListener('click',()=>selectCode(t));const tip=svgEl('title',{{}});tip.textContent=`code ${{t}} · [${{coord(t).join(', ')}}] · ${{(byToken.get(t)||[]).length}} skills`;circle.append(tip);g.append(circle);const label=svgEl('text',{{x:p[0],y:p[1]+4,class:'token-label'}});label.textContent=t;g.append(label);svg.append(g)}}}}
function variantDetail(r,name){{const p=(r.randomization&&r.randomization[name])||{{}};if(name==='color')return `b=${{p.brightness?.toFixed(2)}} c=${{p.contrast?.toFixed(2)}} s=${{p.saturation?.toFixed(2)}} h=${{p.hue?.toFixed(3)}}`;if(name==='crop')return `crop Δ(${{p.dx}}, ${{p.dy}}) · inner Δ(${{p.inner_box_dx}}, ${{p.inner_box_dy}})`;if(name==='blur')return `input radius=${{p.blur_radius?.toFixed(2)}}`;return ''}}
function variantRow(r,name,label){{const pair=r.images&&r.images[name];if(!pair)return '';return `<div class="variant"><div class="variant-head"><span>${{label}}</span><span class="variant-detail">${{esc(variantDetail(r,name))}}</span></div><div class="pair"><figure><img loading="lazy" src="${{esc(pair.start)}}"><figcaption>skill start</figcaption></figure><figure><img loading="lazy" src="${{esc(pair.end)}}"><figcaption>skill end</figcaption></figure></div></div>`}}
function cards(items){{if(!items.length)return '<div class="empty-message">No selected skill occurrence is available.</div>';return items.map(r=>{{const isCrop=DATA.foveation.mode==='crop';let rows=isCrop?variantRow(r,'source','Source + actual crop window'):'';rows+=variantRow(r,'baseline',isCrop?'Baseline crop input':'Baseline partial foveation');if(shownVariants.has('color'))rows+=variantRow(r,'color',isCrop?'Color randomization → crop':'Color randomization → foveation');if(shownVariants.has('crop'))rows+=variantRow(r,'crop',isCrop?'Crop-position randomization → crop':'Crop-position randomization');if(shownVariants.has('blur'))rows+=variantRow(r,'blur',isCrop?'Input blur → crop':'Input blur → foveation');return `<article class="card">${{rows}}<div class="meta"><b>task ${{r.task_id}} · episode ${{r.episode_id}} · skill ${{r.skill_index}} · code ${{r.token}}</b><br>frames [${{r.frame_start}}, ${{r.frame_end}}) · baseline focus (${{r.focus_x}}, ${{r.focus_y}}) · endpoint [${{r.endpoint_xyz.map(v=>v.toFixed(3)).join(', ')}}]<div class="language">${{esc(r.task_description)}}</div></div></article>`}}).join('')}}
function render(){{let items;if(viewMode==='code'){{items=byToken.get(selectedToken)||[];document.getElementById('selection').textContent=`Code ${{selectedToken}} · [${{coord(selectedToken).join(', ')}}] · ${{items.length}} selected skills`}}else{{items=byTask.get(selectedTask)||[];const desc=items[0]?.task_description||'';document.getElementById('selection').textContent=`Task ${{selectedTask}} · ${{items.length}} skills · ${{desc}}`}}document.getElementById('cards').innerHTML=cards(items);drawCube();for(const b of document.querySelectorAll('#codeChips .chip'))b.classList.toggle('active',Number(b.dataset.token)===selectedToken);for(const b of document.querySelectorAll('#taskChips .chip'))b.classList.toggle('active',Number(b.dataset.task)===selectedTask)}}
function selectCode(t){{selectedToken=t;render()}}function selectTask(t){{selectedTask=t;render()}}
function setView(mode){{viewMode=mode;document.getElementById('codeNav').hidden=mode!=='code';document.getElementById('taskNav').hidden=mode!=='task';for(const b of document.querySelectorAll('.toggle.view'))b.classList.toggle('active',b.dataset.view===mode);render()}}
const codeChips=document.getElementById('codeChips');for(let t=0;t<count;t++){{const b=document.createElement('button');b.className='chip';b.dataset.token=t;b.textContent=`${{t}} (${{(byToken.get(t)||[]).length}})`;b.onclick=()=>selectCode(t);codeChips.append(b)}}
const taskChips=document.getElementById('taskChips');for(const t of [...byTask.keys()].sort((a,b)=>a-b)){{const items=byTask.get(t)||[],b=document.createElement('button');b.className='chip';b.dataset.task=t;b.textContent=`task ${{t}} (${{items.length}})`;b.title=items[0]?.task_description||'';b.onclick=()=>selectTask(t);taskChips.append(b)}}
for(const b of document.querySelectorAll('.toggle.view'))b.onclick=()=>setView(b.dataset.view);for(const b of document.querySelectorAll('.toggle.augment')){{const name=b.dataset.variant;b.hidden=!availableVariants.has(name);b.onclick=()=>{{if(shownVariants.has(name))shownVariants.delete(name);else shownVariants.add(name);b.classList.toggle('active',shownVariants.has(name));render()}}}}setView('code');
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-dataset-dir", type=Path, required=True)
    parser.add_argument("--skill-latents-path", type=Path, required=True)
    parser.add_argument("--eval-init-states-path", type=Path, required=True)
    parser.add_argument("--original-dataset-dir", type=Path, required=True)
    parser.add_argument("--target-task", required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--episode-ids", default="[]")
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--episode-selection", choices=("first", "random"), required=True)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--mode", choices=("partial_fov", "crop"), default="partial_fov")
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--inner-box-enabled", type=int, choices=(0, 1), default=1)
    parser.add_argument("--inner-box-mode", choices=("box", "blur"), default="box")
    parser.add_argument("--inner-box-size", type=int, default=32)
    parser.add_argument("--inner-box-line-width", type=int, default=3)
    parser.add_argument("--shape", choices=("square", "circle"), default="square")
    parser.add_argument("--sharp-size", type=int, default=96)
    parser.add_argument("--feather", type=int, default=20)
    parser.add_argument("--blur-radius", type=float, default=18.0)
    parser.add_argument("--random-color-enabled", type=int, choices=(0, 1), default=1)
    parser.add_argument("--random-color-brightness", default="[0.8,1.2]")
    parser.add_argument("--random-color-contrast", default="[0.8,1.2]")
    parser.add_argument("--random-color-saturation", default="[0.8,1.2]")
    parser.add_argument("--random-color-hue", default="[-0.05,0.05]")
    parser.add_argument("--random-crop-enabled", type=int, choices=(0, 1), default=1)
    parser.add_argument("--random-crop-offset-px", default="[-24,24]")
    parser.add_argument("--random-crop-inner-box-offset-px", default="[-4,4]")
    parser.add_argument("--random-blur-enabled", type=int, choices=(0, 1), default=1)
    parser.add_argument("--random-blur-radius", default="[0.0,4.0]")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    color_brightness = _parse_numeric_range(
        args.random_color_brightness,
        field="random_color_brightness",
        minimum=0.0,
    )
    color_contrast = _parse_numeric_range(
        args.random_color_contrast,
        field="random_color_contrast",
        minimum=0.0,
    )
    color_saturation = _parse_numeric_range(
        args.random_color_saturation,
        field="random_color_saturation",
        minimum=0.0,
    )
    color_hue = _parse_numeric_range(
        args.random_color_hue,
        field="random_color_hue",
        minimum=-0.5,
        maximum=0.5,
    )
    crop_offset = _parse_numeric_range(
        args.random_crop_offset_px,
        field="random_crop_offset_px",
        integer=True,
    )
    inner_box_offset = _parse_numeric_range(
        args.random_crop_inner_box_offset_px,
        field="random_crop_inner_box_offset_px",
        integer=True,
    )
    random_blur_radius = _parse_numeric_range(
        args.random_blur_radius,
        field="random_blur_radius",
        minimum=0.0,
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
    episode_ids = [int(value) for value in json.loads(args.episode_ids)]
    selected = dataset.select_episodes(
        task_ids=task_ids,
        episodes_per_task=args.episodes_per_task,
        selection=args.episode_selection,
        seed=args.seed,
        explicit_episode_ids=episode_ids,
    )
    occurrences = dataset.occurrences(selected)
    invalid = sorted({item.token for item in occurrences if not 0 <= item.token < codebook_size})
    if invalid:
        raise ValueError(f"Skill assignments exceed FSQ{levels}: {invalid}.")

    reader = EpisodeFrameReader(args.skill_dataset_dir)
    by_episode: dict[int, list] = defaultdict(list)
    for occurrence in occurrences:
        by_episode[occurrence.episode_id].append(occurrence)
    records: list[dict] = []
    processed = 0
    for episode_id in sorted(by_episode):
        episode_length = reader.episode_length(episode_id)
        aligned = dataset.load_aligned_episode(episode_id)
        if not aligned.model_xml:
            raise ValueError(
                "Foveated preview requires a recorded LIBERO model XML; "
                f"episode {episode_id} does not provide one."
            )
        # Every selected occurrence needs two frames; decode them together once.
        pairs = {
            occurrence.uid: (
                int(occurrence.frame_start),
                min(int(occurrence.frame_end), episode_length - 1),
            )
            for occurrence in by_episode[episode_id]
        }
        needed = sorted({frame for pair in pairs.values() for frame in pair})
        frames = reader.frames(episode_id, needed)
        sample = frames[needed[0]]
        height, width = sample.shape[:2]
        transform = _camera_transform_from_recorded_xml(
            aligned.model_xml,
            camera_name=args.camera,
            height=height,
            width=width,
        )
        for occurrence in by_episode[episode_id]:
            processed += 1
            start_frame, final_frame = pairs[occurrence.uid]
            endpoint_xyz = np.asarray(
                aligned.filtered_states[final_frame, :3], dtype=np.float64
            ).copy()
            if dataset.proprio_grounding == "episode_start_xyz":
                endpoint_xyz += np.asarray(aligned.episode_start_xyz, dtype=np.float64)
            elif dataset.proprio_grounding != "none":
                raise ValueError(
                    f"Unsupported proprio grounding: {dataset.proprio_grounding!r}."
                )
            center = _project_eef(endpoint_xyz, transform, height=height, width=width)
            relative_dir = (
                Path("images")
                / f"task_{occurrence.task_id:02d}"
                / f"token_{occurrence.token:04d}"
            )
            source_start = frames[start_frame]
            source_end = frames[final_frame]
            images: dict[str, dict[str, str]] = {}
            randomization: dict[str, dict[str, float | int]] = {}

            def focused(
                image: np.ndarray,
                focus_xy: tuple[int, int],
                inner_focus_xy: tuple[int, int],
            ) -> np.ndarray:
                if args.mode == "crop":
                    return crop_focus_image(
                        image,
                        center_xy=focus_xy,
                        crop_size=args.crop_size,
                        output_size=args.output_size,
                        inner_box_enabled=bool(args.inner_box_enabled),
                        inner_box_mode=args.inner_box_mode,
                        inner_box_center_xy=inner_focus_xy,
                        inner_box_size=args.inner_box_size,
                        inner_box_line_width=args.inner_box_line_width,
                        inner_blur_shape=args.shape,
                        inner_blur_feather=args.feather,
                        inner_blur_radius=args.blur_radius,
                    )
                return foveate_image(
                    image,
                    center_xy=focus_xy,
                    shape=args.shape,
                    sharp_size=args.sharp_size,
                    feather=args.feather,
                    blur_radius=args.blur_radius,
                )

            def write_variant(
                name: str,
                start_image: np.ndarray,
                end_image: np.ndarray,
                *,
                focus_xy: tuple[int, int] = center,
                inner_focus_xy: tuple[int, int] = center,
            ) -> None:
                suffix = "" if name == "baseline" else f"_{name}"
                start_path = relative_dir / f"{occurrence.uid}_start{suffix}.png"
                end_path = relative_dir / f"{occurrence.uid}_end{suffix}.png"
                _write_png(
                    args.output_dir / start_path,
                    focused(start_image, focus_xy, inner_focus_xy),
                )
                _write_png(
                    args.output_dir / end_path,
                    focused(end_image, focus_xy, inner_focus_xy),
                )
                images[name] = {
                    "start": start_path.as_posix(),
                    "end": end_path.as_posix(),
                }

            if args.mode == "crop":
                source_start_path = relative_dir / f"{occurrence.uid}_start_source.png"
                source_end_path = relative_dir / f"{occurrence.uid}_end_source.png"
                _write_png(
                    args.output_dir / source_start_path,
                    draw_crop_window(
                        source_start,
                        center_xy=center,
                        crop_size=args.crop_size,
                        inner_box_enabled=bool(args.inner_box_enabled),
                        inner_box_mode=args.inner_box_mode,
                        inner_box_center_xy=center,
                        inner_box_size=args.inner_box_size,
                    ),
                )
                _write_png(
                    args.output_dir / source_end_path,
                    draw_crop_window(
                        source_end,
                        center_xy=center,
                        crop_size=args.crop_size,
                        inner_box_enabled=bool(args.inner_box_enabled),
                        inner_box_mode=args.inner_box_mode,
                        inner_box_center_xy=center,
                        inner_box_size=args.inner_box_size,
                    ),
                )
                images["source"] = {
                    "start": source_start_path.as_posix(),
                    "end": source_end_path.as_posix(),
                }

            write_variant("baseline", source_start, source_end)

            if args.random_color_enabled:
                color_rng = _rng_for(args.seed, occurrence.uid, "color")
                color_params = {
                    "brightness": _sample_uniform(color_rng, color_brightness),
                    "contrast": _sample_uniform(color_rng, color_contrast),
                    "saturation": _sample_uniform(color_rng, color_saturation),
                    "hue": _sample_uniform(color_rng, color_hue),
                }
                # Apply to each complete source frame before the sharp region
                # and its blurred periphery are constructed.
                write_variant(
                    "color",
                    color_randomize_image(source_start, **color_params),
                    color_randomize_image(source_end, **color_params),
                )
                randomization["color"] = color_params

            if args.random_crop_enabled:
                crop_rng = _rng_for(args.seed, occurrence.uid, "crop")
                dx = _sample_integer(crop_rng, crop_offset)
                dy = _sample_integer(crop_rng, crop_offset)
                random_center = _jitter_center(
                    center,
                    dx=dx,
                    dy=dy,
                    height=height,
                    width=width,
                )
                inner_rng = _rng_for(args.seed, occurrence.uid, "inner_box")
                inner_dx = _sample_integer(inner_rng, inner_box_offset)
                inner_dy = _sample_integer(inner_rng, inner_box_offset)
                random_inner_center = _jitter_center(
                    center,
                    dx=inner_dx,
                    dy=inner_dy,
                    height=height,
                    width=width,
                )
                write_variant(
                    "crop",
                    source_start,
                    source_end,
                    focus_xy=random_center,
                    inner_focus_xy=random_inner_center,
                )
                randomization["crop"] = {
                    "dx": dx,
                    "dy": dy,
                    "focus_x": random_center[0],
                    "focus_y": random_center[1],
                    "inner_box_dx": inner_dx,
                    "inner_box_dy": inner_dy,
                    "inner_box_focus_x": random_inner_center[0],
                    "inner_box_focus_y": random_inner_center[1],
                }

            if args.random_blur_enabled:
                blur_rng = _rng_for(args.seed, occurrence.uid, "blur")
                sampled_radius = _sample_uniform(blur_rng, random_blur_radius)
                # This is input augmentation: blur the complete source first,
                # then form the foveated image. The ordinary peripheral blur
                # remains independently controlled by --blur-radius.
                write_variant(
                    "blur",
                    input_blur_image(source_start, radius=sampled_radius),
                    input_blur_image(source_end, radius=sampled_radius),
                )
                randomization["blur"] = {"blur_radius": sampled_radius}

            records.append(
                {
                    "uid": occurrence.uid,
                    "token": int(occurrence.token),
                    "coord": _token_to_coord(occurrence.token, levels),
                    "task_id": int(occurrence.task_id),
                    "task_description": dataset.task_descriptions.get(
                        occurrence.task_id, ""
                    ),
                    "episode_id": int(episode_id),
                    "skill_index": int(occurrence.skill_index),
                    "frame_start": int(occurrence.frame_start),
                    "frame_end": int(occurrence.frame_end),
                    "focus_x": int(center[0]),
                    "focus_y": int(center[1]),
                    "endpoint_xyz": [float(value) for value in endpoint_xyz],
                    "images": images,
                    "randomization": randomization,
                }
            )
            log.info(
                "[%d/%d] code=%d task=%d ep=%d skill=%d focus=(%d,%d)",
                processed,
                len(occurrences),
                occurrence.token,
                occurrence.task_id,
                episode_id,
                occurrence.skill_index,
                center[0],
                center[1],
            )

    payload = {
        "format": "foveated_skill_preview_v3",
        "title": f"Focused top preview · {args.skill_latents_path.parent.name}",
        "levels": levels,
        "selected_episodes": {str(key): value for key, value in selected.items()},
        "proprio_grounding": dataset.proprio_grounding,
        "foveation": {
            "camera": args.camera,
            "mode": args.mode,
            "crop_size": args.crop_size,
            "output_size": args.output_size,
            "inner_box_enabled": bool(args.inner_box_enabled),
            "inner_box_mode": args.inner_box_mode,
            "inner_box_size": args.inner_box_size,
            "inner_box_line_width": args.inner_box_line_width,
            "shape": args.shape,
            "sharp_size": args.sharp_size,
            "feather": args.feather,
            "blur_radius": args.blur_radius,
        },
        "randomization": {
            "variants": [
                name
                for name, enabled in (
                    ("color", args.random_color_enabled),
                    ("crop", args.random_crop_enabled),
                    ("blur", args.random_blur_enabled),
                )
                if enabled
            ],
            "color": {
                "brightness": list(color_brightness),
                "contrast": list(color_contrast),
                "saturation": list(color_saturation),
                "hue": list(color_hue),
            },
            "crop": {
                "offset_px": list(crop_offset),
                "inner_box_offset_px": list(inner_box_offset),
            },
            "blur": {"blur_radius": list(random_blur_radius)},
            "order": "full-frame randomization, then foveation",
        },
        "records": records,
    }
    _atomic_text(
        args.output_dir / "manifest.json",
        json.dumps(payload, indent=2, ensure_ascii=False),
    )
    _atomic_text(args.output_dir / "index.html", _report_html(payload))
    log.info("DONE -> %s", args.output_dir / "index.html")


if __name__ == "__main__":
    main()
