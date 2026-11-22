"""Saved ABC skill/termination traces shared by video annotation and HTML."""
from __future__ import annotations

import colorsys
import json
from pathlib import Path


SKILL_OVERLAY_VERSION = "abc_language_outcome_skill/v2"


def skill_color(code: int) -> tuple[int, int, int]:
    # Same deterministic categorical colors as LIBERO's _skill_banner_color.
    if code < 0:
        return (80, 80, 80)
    return tuple(round(x * 255) for x in colorsys.hsv_to_rgb(
        (int(code) * 0.618033988749895) % 1.0, .68, .78,
    ))


def load_skill_trace(world: dict, *, output: Path | None = None) -> list[dict]:
    filename = world.get("skill_trace_path")
    if not filename:
        return []
    path = Path(filename)
    if not path.is_absolute() and output is not None:
        path = output / path
    data = json.loads(path.read_text())
    skills = []
    by_order = {}
    for event in sorted(data.get("skills", []), key=lambda e: e["step"]):
        order, code = int(event["skill_order"]), int(event["skill_code"])
        item = {"order": order, "code": code, "start": int(event["step"]),
                "end": int(event.get("end_step", world["steps"])),
                "reason": event.get("end_reason", "episode_end"),
                "color": "#%02x%02x%02x" % skill_color(code), "points": []}
        skills.append(item)
        by_order[order] = item
    for step, order, probability in data.get("termination_trace", []):
        if int(order) in by_order:
            by_order[int(order)]["points"].append([int(step), float(probability)])
    return skills


def active_skill(skills: list[dict], frame: float) -> dict | None:
    # A new skill starts AFTER its activation observation. The termination
    # recorded at step N and the corresponding video frame belong to the old
    # skill; the first action of the next skill is step N+1.
    current = None
    for skill in skills:
        if current is not None and frame <= skill["start"]:
            break
        current = skill
    return current


def termination_widget(world: dict, settings: dict, *, output: Path) -> str:
    skills = load_skill_trace(world, output=output)
    if not skills:
        return ""
    overlay = world.get("video_overlay", {})
    payload = {"skills": skills, "threshold": float(settings.get("termination_threshold", .5)),
               "fps": float(overlay.get("fps", 30)),
               "stride": int(world.get("video_every_n_steps", overlay.get("video_every_n_steps", 1))),
               "steps": int(world["steps"])}
    # Data is embedded for file:// support, with no fetch or CDN dependency.
    encoded = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    return (f'<script type="application/json" class="skill-trace-data">{encoded}</script>'
            '<section class="termination-panel" hidden aria-label="Skill termination probability">'
            '<div class="termination-heading"><b class="active-skill"></b><span class="termination-value"></span></div>'
            '<canvas class="termination-chart" height="190" aria-label="Termination probability, threshold and video playhead"></canvas>'
            '<div class="skill-timeline" aria-label="Seek to a skill"></div>'
            '<p class="muted termination-help">Termination probability · dashed line: threshold · '
            'click the graph or a skill color to seek.</p></section>')


def termination_script() -> str:
    return Path(__file__).with_name("skill_trace_report.js").read_text()


TERMINATION_CSS = """
.termination-panel{margin-top:8px;background:#111a25;border:1px solid #344154;border-radius:8px;padding:10px}
.termination-heading{display:flex;flex-wrap:wrap;gap:6px 16px;justify-content:space-between;font-size:13px}
.termination-chart{display:block;width:100%;height:190px;cursor:crosshair;touch-action:pan-y}
.skill-timeline{display:flex;width:100%;gap:2px;height:24px}
.skill-timeline button{min-width:0;border:0;border-radius:3px;color:white;padding:0;font-size:11px;overflow:hidden;cursor:pointer;opacity:.65}
.skill-timeline button[aria-current=true]{opacity:1;outline:2px solid white;outline-offset:-2px}
.termination-help{margin:8px 0 0;font-size:11px}
"""
