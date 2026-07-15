#!/usr/bin/env python3
"""Visualize downloaded ABC-130k episodes as a readable HTML gallery.

Minimal footprint: only the **3rd-person (top) camera** is decoded, downscaled,
and turned into a short looping animation (WebP or GIF) per episode. The page
groups episodes by task and shows each episode's **language prompt** next to its
animation, so you can eyeball what you downloaded at a glance.

No system ffmpeg needed — decode (h264/hevc) and encode (webp/gif) both go
through PyAV. Reads the raw ``episode.mcap`` (the downloaded format) directly.

Run via ``../visualization/run.sh`` (see that + config.yaml).
"""

from __future__ import annotations

import argparse
import glob
import html
import io
import json
import os
import sys
from collections import defaultdict
from typing import Optional

import numpy as np

try:
    import yaml
except ImportError as e:  # pragma: no cover
    sys.exit("PyYAML required: uv pip install pyyaml  (%s)" % e)

# 3rd-person camera topics, in preference order (short-name -> mcap topic).
_VIEW_TOPICS = {
    "top": "/top-camera",
    "top_left": "/top-left-camera",
    "top_right": "/top-right-camera",
}


# ---------------------------------------------------------------------------
# Minimal mcap read: just the language prompt + one 3rd-person camera stream.
# (Purpose-built so an episode missing state/action topics still visualizes.)
# ---------------------------------------------------------------------------

def read_prompt_and_view(path: str, view_priority: list) -> Optional[dict]:
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    wanted_topics = [_VIEW_TOPICS[v] for v in view_priority if v in _VIEW_TOPICS]
    cams: dict[str, list] = {}      # topic -> list[(log_time, bytes)]
    codecs: dict[str, str] = {}
    task: Optional[str] = None

    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, msg, dec in reader.iter_decoded_messages():
            t = ch.topic
            if t == "/instruction":
                task = getattr(dec, "data", None) or task
            elif t in wanted_topics:
                cams.setdefault(t, []).append((msg.log_time, bytes(dec.data)))
                codecs.setdefault(t, getattr(dec, "format", "h264") or "h264")

    # Metadata fallback for the prompt (episode-metadata / session-metadata drift).
    if not task:
        with open(path, "rb") as f:
            for m in make_reader(f).iter_metadata():
                if m.name in ("episode-metadata", "session-metadata"):
                    md = dict(m.metadata)
                    task = md.get("task_name") or md.get("instruction")
                    break

    # Pick the highest-priority 3rd-person camera that is actually present.
    topic = next((_VIEW_TOPICS[v] for v in view_priority
                  if _VIEW_TOPICS.get(v) in cams), None)
    if topic is None:
        return None  # no 3rd-person view in this episode

    frames = [b for _, b in sorted(cams[topic], key=lambda x: x[0])]
    return {"task": task, "frames": frames, "codec": codecs[topic], "topic": topic}


# ---------------------------------------------------------------------------
# Decode (PyAV) — keep only the sampled frames, downscaled. Memory-bounded even
# for very long episodes (some ABC clips run minutes).
# ---------------------------------------------------------------------------

def decode_sampled(chunks: list, codec: str, target_frames: int, width: int) -> list:
    import av

    n = len(chunks)
    if n == 0:
        return []
    keep = np.linspace(0, n - 1, num=min(target_frames, n)).round().astype(int)
    keep_set = set(int(i) for i in keep)

    fmt = "hevc" if str(codec).lower() in ("h265", "hevc") else "h264"
    raw = b"".join(chunks)

    out: list = []
    h_out: Optional[int] = None
    container = av.open(io.BytesIO(raw), format=fmt)
    try:
        for i, frame in enumerate(container.decode(video=0)):
            if i not in keep_set:
                continue
            if h_out is None:  # lock aspect from the first kept frame
                w0, h0 = frame.width, frame.height
                h_out = max(2, int(round(h0 * width / max(1, w0))) // 2 * 2)
            arr = frame.reformat(width=width, height=h_out, format="rgb24").to_ndarray()
            out.append(arr)
    finally:
        container.close()
    return out


# ---------------------------------------------------------------------------
# Encode a short looping animation (PyAV): WebP (default, smaller) or GIF.
# ---------------------------------------------------------------------------

def write_anim(frames: list, out_path: str, fmt: str, fps: int) -> int:
    import av

    if not frames:
        return 0
    h, w = frames[0].shape[:2]
    # loop=0 (infinite) is a MUXER option — must be set on the container, not the
    # stream, or animated WebP bakes in loop=1 and plays exactly once in browsers.
    container = av.open(out_path, "w", options={"loop": "0"})
    try:
        if fmt == "gif":
            stream = container.add_stream("gif", rate=fps)
            stream.width, stream.height = w, h
            stream.pix_fmt = "rgb8"
            for img in frames:
                vf = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="rgb8")
                for pkt in stream.encode(vf):
                    container.mux(pkt)
        else:  # animated webp
            stream = container.add_stream("webp", rate=fps)
            stream.width, stream.height = w, h
            stream.pix_fmt = "yuv420p"
            stream.options = {"quality": "72"}
            for img in frames:
                vf = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p")
                for pkt in stream.encode(vf):
                    container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    finally:
        container.close()
    return len(frames)


# ---------------------------------------------------------------------------
# Episode discovery + path parsing
# ---------------------------------------------------------------------------

def find_episodes(in_dir: str, split: str) -> list:
    """Return sorted episode.mcap paths. Honors data/<split>/ if present."""
    roots = []
    if split and split not in ("all", "*"):
        roots.append(os.path.join(in_dir, "data", split))
    roots.append(in_dir)  # fallback: search everything under in_dir
    seen: set = set()
    found: list = []
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", "episode.mcap"), recursive=True):
            rp = os.path.realpath(p)
            if rp not in seen:
                seen.add(rp)
                found.append(p)
        if found:
            break
    return sorted(found)


def parse_ids(path: str) -> tuple:
    """(task_folder, episode_uuid) from .../<task>/episode_<uuid>/episode.mcap."""
    ep_dir = os.path.basename(os.path.dirname(path))
    task_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
    uuid = ep_dir[len("episode_"):] if ep_dir.startswith("episode_") else ep_dir
    return task_folder, uuid


def derive_group(path: str, in_dir: str) -> Optional[str]:
    """Category group from the download layout, or None if ungrouped.

    With the downloader's ``group_subdirs`` the layout is
    ``<in_dir>/<group>/data/<split>/<task>/episode_<uuid>/`` — so the group is the
    first path component under *in_dir* (None when that component is ``data``, i.e.
    the flat/ungrouped layout).
    """
    try:
        rel = os.path.relpath(path, in_dir)
    except ValueError:
        return None
    parts = rel.split(os.sep)
    return parts[0] if parts and parts[0] not in ("data", "..") else None


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#0f1115;--card:#1a1d24;--fg:#e6e8ee;--mut:#8b93a7;--acc:#6ea8fe}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;text-align:center}
header{padding:18px 24px 12px;background:var(--bg)}
header h1{margin:0 0 4px;font-size:18px}
header .sub{color:var(--mut);font-size:13px}
.tabs{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;padding:12px 16px;position:sticky;top:0;background:var(--bg);border-bottom:1px solid #262a33;z-index:3}
.tab{background:var(--card);color:var(--fg);border:1px solid #2b303b;border-radius:20px;padding:6px 14px;cursor:pointer;font:inherit;font-size:13px}
.tab:hover{border-color:var(--acc)}
.tab.active{background:var(--acc);color:#0b1020;border-color:var(--acc);font-weight:600}
.tab .badge{opacity:.7;font-size:11px;margin-left:5px}
.panel{padding-top:6px}
.task{padding:18px 24px 4px;max-width:1400px;margin:0 auto}
.task h2{font-size:15px;margin:0 0 2px;color:var(--acc)}
.task .cnt{color:var(--mut);font-size:12px;margin-bottom:10px}
.grid{display:flex;flex-wrap:wrap;justify-content:center;gap:14px;padding:0 24px 8px;max-width:1400px;margin:0 auto}
.card{flex:0 0 240px;background:var(--card);border:1px solid #262a33;border-radius:10px;overflow:hidden}
.card img{width:100%;display:block;background:#000;aspect-ratio:4/3;object-fit:contain}
.card .meta{padding:9px 11px;text-align:center}
.card .prompt{font-weight:600;margin:0 0 4px}
.card .sub{color:var(--mut);font-size:11.5px;word-break:break-all}
"""


def build_html(title: str, catalog: dict, anim_ext: str) -> str:
    """catalog = {group_or_None: {task_folder: [card, ...]}} -> tabbed HTML.

    Groups become clickable tabs (only the active group's panel is shown). With a
    single group the tab bar is omitted and everything renders inline.
    """
    def gkey(g):  # named groups A→Z, then misc/ungrouped last
        name = g or "~"
        return (g in (None, "misc"), name)

    groups = sorted(catalog, key=gkey)
    n_groups = len(groups)
    n_tasks = sum(len(t) for t in catalog.values())
    n_ep = sum(len(c) for t in catalog.values() for c in t.values())
    multi = n_groups > 1

    parts = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        "<meta name=viewport content='width=device-width,initial-scale=1'>",
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>",
        f"<header><h1>{html.escape(title)}</h1>",
        f"<div class=sub>{n_groups} groups &middot; {n_tasks} tasks &middot; "
        f"{n_ep} episodes &middot; top view only</div></header>",
    ]

    if multi:
        parts.append("<nav class=tabs>")
        for i, g in enumerate(groups):
            label = (g or "ungrouped").replace("_", " ")
            parts.append(
                f"<button class='tab{' active' if i == 0 else ''}' data-g='{i}'>"
                f"{html.escape(label)}<span class=badge>{len(catalog[g])}</span></button>"
            )
        parts.append("</nav>")

    for i, g in enumerate(groups):
        shown = "block" if (i == 0 or not multi) else "none"
        parts.append(f"<div class=panel data-g='{i}' style='display:{shown}'>")
        for task_folder in sorted(catalog[g]):
            cards = catalog[g][task_folder]
            pretty = task_folder.replace("_", " ")
            parts.append(
                f"<section class=task><h2>{html.escape(pretty)}</h2>"
                f"<div class=cnt>{html.escape(task_folder)} &middot; {len(cards)} ep</div></section>"
            )
            parts.append("<div class=grid>")
            for c in cards:
                prompt = html.escape(c["prompt"] or pretty)
                parts.append(
                    "<div class=card>"
                    f"<img loading=lazy src='{html.escape(c['rel'])}' alt='{prompt}'>"
                    f"<div class=meta><p class=prompt>{prompt}</p>"
                    f"<div class=sub>{html.escape(c['uuid'])}<br>{c['nframes']} frames shown</div>"
                    "</div></div>"
                )
            parts.append("</div>")
        parts.append("</div>")

    if multi:
        parts.append(
            "<script>"
            "document.querySelectorAll('.tab').forEach(function(b){"
            "b.onclick=function(){var g=b.dataset.g;"
            "document.querySelectorAll('.tab').forEach(function(t){t.classList.toggle('active',t.dataset.g===g)});"
            "document.querySelectorAll('.panel').forEach(function(p){p.style.display=(p.dataset.g===g)?'block':'none'});"
            "};});"
            "</script>"
        )
    parts.append("</body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="ABC-130k episode HTML visualizer (top view)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="list episodes to render, do nothing")
    ap.add_argument("--force", action="store_true", help="re-render every episode, ignoring the cache")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    in_dir = cfg.get("in_dir", "../abc_subset")
    split = cfg.get("split", "train")
    out_dir = cfg.get("out_dir", "../abc_viz")
    view_priority = cfg.get("view_priority", ["top", "top_left", "top_right"])
    anim_format = str(cfg.get("anim_format", "webp")).lower()
    if anim_format not in ("webp", "gif"):
        sys.exit(f"anim_format must be webp|gif, got {anim_format!r}")
    target_frames = int(cfg.get("target_frames", 48))
    playback_fps = int(cfg.get("playback_fps", 12))
    width = int(cfg.get("width", 256))
    title = cfg.get("title", "ABC-130k preview (top view)")
    max_ep = int(cfg.get("max_episodes", 0))   # cap via config.yaml (0 = all)

    eps = find_episodes(in_dir, split)
    if not eps:
        sys.exit(f"no episode.mcap found under {in_dir!r} "
                 f"(did you download yet? see ../download/)")
    if max_ep and max_ep > 0:
        # Cap PER TASK: up to max_ep episodes of each task (fewer if the task has
        # fewer). NOT a global cap.
        by_task: dict = defaultdict(list)
        for p in eps:
            by_task[parse_ids(p)[0]].append(p)
        eps = [p for tf in sorted(by_task) for p in sorted(by_task[tf])[:max_ep]]

    print(f"episodes to render: {len(eps)}  (view_priority={view_priority}, "
          f"{anim_format}, {target_frames}f@{playback_fps}fps, {width}px)")
    if args.dry_run:
        for p in eps:
            tf, uuid = parse_ids(p)
            print(f"  {tf}/{uuid}")
        return

    assets = os.path.join(out_dir, "assets")
    os.makedirs(assets, exist_ok=True)

    # Skip-if-exists cache. A tiny manifest records each rendered episode's card
    # (prompt / nframes / rel) keyed by task/uuid, plus the render params. On a
    # re-run with the SAME params, episodes whose asset already exists are reused
    # with NO mcap read and NO decode/encode. The check is a stat() + dict lookup
    # (microseconds) vs seconds per render, so it never costs more than it saves.
    # Params change (format/size/frames/fps/view) invalidates the whole cache.
    manifest_path = os.path.join(out_dir, ".viz_manifest.json")
    params = {"anim_format": anim_format, "width": width, "target_frames": target_frames,
              "playback_fps": playback_fps, "view_priority": view_priority}
    cache: dict = {}
    if os.path.exists(manifest_path) and not args.force:
        try:
            mf = json.load(open(manifest_path))
            if mf.get("params") == params:
                cache = mf.get("episodes", {})
        except Exception:  # noqa: BLE001 - corrupt/old manifest -> just re-render
            cache = {}

    catalog: dict = {}   # {group_or_None: {task_folder: [card, ...]}}
    manifest_eps: dict = {}
    ok = skip = fail = 0
    for k, path in enumerate(eps):
        task_folder, uuid = parse_ids(path)
        group = derive_group(path, in_dir)
        key = f"{task_folder}/{uuid}"
        fname = f"{task_folder}__{uuid}.{anim_format}"
        asset_path = os.path.join(assets, fname)

        # Reuse an already-rendered asset (same params) — skip the expensive work.
        if (not args.force and key in cache
                and os.path.exists(asset_path) and os.path.getsize(asset_path) > 0):
            card = dict(cache[key])
            catalog.setdefault(group, {}).setdefault(task_folder, []).append(card)
            manifest_eps[key] = card
            skip += 1
            continue

        try:
            ep = read_prompt_and_view(path, view_priority)
            if ep is None:
                print(f"[skip] no top view: {key}", file=sys.stderr)
                fail += 1
                continue
            frames = decode_sampled(ep["frames"], ep["codec"], target_frames, width)
            if not frames:
                print(f"[skip] no frames decoded: {key}", file=sys.stderr)
                fail += 1
                continue
            nf = write_anim(frames, asset_path, anim_format, playback_fps)
            card = {"prompt": ep["task"], "uuid": uuid,
                    "rel": os.path.join("assets", fname), "nframes": nf}
            catalog.setdefault(group, {}).setdefault(task_folder, []).append(card)
            manifest_eps[key] = card
            ok += 1
            print(f"[{k+1}/{len(eps)}] {key}  ({nf}f)")
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {key}: {type(e).__name__}: {e}", file=sys.stderr)
            fail += 1

    # Persist the manifest so the next run can skip these.
    try:
        with open(manifest_path, "w") as f:
            json.dump({"params": params, "episodes": manifest_eps}, f)
    except Exception:  # noqa: BLE001
        pass

    if not catalog:
        sys.exit("nothing rendered")
    index = os.path.join(out_dir, "index.html")
    with open(index, "w") as f:
        f.write(build_html(title, catalog, anim_format))
    print(f"\n[done] rendered={ok} skipped={skip} fail={fail}  ->  {os.path.realpath(index)}")


if __name__ == "__main__":
    main()
