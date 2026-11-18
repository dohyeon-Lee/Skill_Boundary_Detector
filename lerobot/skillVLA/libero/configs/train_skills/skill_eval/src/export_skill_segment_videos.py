"""Export the HTML skill player as standalone, PowerPoint-compatible MP4s.

No model inference or simulator is needed: cuts, timing, and source clips come
directly from the existing report. The source HTML and videos are never edited.

Pass --with-boundary-graph to append the report's detector graph and a synced
playhead, excluding the power=1 comparison and all legends (requires cairosvg).
"""

from __future__ import annotations

import argparse
import base64
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


PALETTE = ("#2563eb", "#7c3aed", "#db2777", "#ea580c", "#16a34a", "#0891b2")
DEFAULT_HTML = (
    Path(__file__).resolve().parents[5]
    / "analysis/skill_boundary_detection/libero/libero_90_full_full"
    / "standard_built_power2_tasks0_9_5ep/index.html"
)


class Players(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
        self.caption = ""
        self.in_caption = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if "cap" in classes:
            self.in_caption = True
            self.caption = ""
        if "episode-player" in classes:
            self.items.append({"caption": self.caption, **attrs})
        if "camera-panel" in classes:
            self.items[-1].setdefault("cameras", []).append({})
        if tag == "video" and "episode-video" in classes:
            self.items[-1]["source"] = attrs.get("data-src") or attrs.get("src")
            if self.items[-1].get("cameras"):
                self.items[-1]["cameras"][-1]["primary"] = True
        if "camera-canvas" in classes:
            self.items[-1]["cameras"][-1].update(attrs)
        if "camera-label" in classes:
            self.camera_label_pending = True
        if "timeline-canvas" in classes:
            self.items[-1]["timeline"] = attrs
        if tag == "img" and "timeline" in classes:
            self.items[-1]["timeline_svg"] = attrs["src"]

    def handle_data(self, data):
        if getattr(self, "camera_label_pending", False):
            self.items[-1]["cameras"][-1]["label"] = data
            self.camera_label_pending = False
        if self.in_caption:
            self.caption += data

    def handle_endtag(self, tag):
        if tag == "div":
            self.in_caption = False


def font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        # Matplotlib bundles these fonts even on headless compute/login nodes.
        import matplotlib

        path = Path(matplotlib.get_data_path()) / "fonts/ttf"
        return ImageFont.truetype(str(path / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def read_frame(stream, count):
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.read(count - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def boundary_graph(item, width):
    """Render the exact report SVG, minus the power=1 twin axis and legends."""
    import cairosvg

    encoded = item["timeline_svg"]
    if not encoded.startswith("data:image/svg+xml;base64,"):
        raise ValueError("Expected an embedded SVG boundary graph")
    root = ET.fromstring(base64.b64decode(encoded.split(",", 1)[1]))
    removed_axes = 0
    removed_legends = 0
    for parent in list(root.iter()):
        for element in list(parent):
            ident = element.get("id", "")
            if ident == "axes_2":
                # In the standard report this is exclusively the power=1
                # curve, its global-mean line, and its right-hand y-axis.
                parent.remove(element)
                removed_axes += 1
            elif ident.startswith("legend_"):
                parent.remove(element)
                removed_legends += 1
    if removed_axes not in (0, 1) or removed_legends != 1:
        raise ValueError(f"Unexpected standard graph layout: axes={removed_axes}, legends={removed_legends}")
    cleaned = ET.tostring(root)
    png = cairosvg.svg2png(bytestring=cleaned, output_width=width)
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    white = Image.new("RGBA", image.size, "white")
    white.alpha_composite(image)
    box = item["timeline"]
    top = float(re.search(r"--plot-top:([\d.]+)%", box["style"])[1])/100
    bottom = float(re.search(r"--plot-bottom:([\d.]+)%", box["style"])[1])/100
    return white.convert("RGB"), {
        "left": float(box["data-plot-left"]),
        "right": float(box["data-plot-right"]),
        "top": top,
        "bottom": bottom,
        "frames": int(box["data-n-frames"]),
    }


def export_one(item, html_path, output, width, flash_seconds, with_graph=False):
    if item.get("cameras"):
        raise ValueError("Use the ABC multi-camera exporter for this report")
    match = re.search(r"task(\d+)\s*·\s*episode\s+(\d+)", item["caption"])
    if not match:
        raise ValueError(f"Cannot identify task/episode: {item['caption']!r}")
    task, episode = map(int, match.groups())
    destination = output / f"task{task:02d}" / f"episode_{episode:07d}.mp4"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    source = (html_path.parent / unquote(item["source"])).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    skills = json.loads(item["data-skills"])
    total = int(item["data-n-frames"])
    fps = float(item["data-fps"])
    starts = [skill["start"] for skill in skills]
    if not skills or starts[0] != 0 or skills[-1]["end"] != total:
        raise ValueError(f"Incomplete skill coverage: {item['caption']}")
    if any(a["end"] != b["start"] for a, b in zip(skills, skills[1:])):
        raise ValueError(f"Non-contiguous skills: {item['caption']}")
    scale = width / 640
    gap, bar_h, footer_h, bottom = [round(v * scale) for v in (8, 36, 28, 8)]
    height = width + gap + bar_h + footer_h + bottom
    height += height % 2
    graph_y = height + gap
    graph = None
    if with_graph:
        graph, plot_box = boundary_graph(item, width)
        height = graph_y + graph.height
        height += height % 2
    fonts = {name: font(max(10, round(size * scale)), bold) for name, size, bold in (
        ("badge", 24, True), ("small", 14, False), ("bar", 16, True), ("flash", 32, True),
    )}
    bar_y = width + gap
    templates = []
    for active in range(len(skills)):
        template = Image.new("RGB", (width, height), "white")
        if graph is not None:
            template.paste(graph, (0, graph_y))
        draw = ImageDraw.Draw(template)
        for index, skill in enumerate(skills):
            x0, x1 = round(width*skill["start"]/total), round(width*skill["end"]/total)
            color = PALETTE[index % len(PALETTE)]
            rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            if index != active:
                rgb = tuple(round(.72*c + .28*255) for c in rgb)
            draw.rectangle((x0, bar_y, max(x0, x1-1), bar_y+bar_h-1), fill=rgb)
            draw.line((x1-1, bar_y, x1-1, bar_y+bar_h-1), fill="white")
            label = f"sk{index}"
            if draw.textlength(label, font=fonts["bar"]) + 8 < x1-x0:
                draw.text(((x0+x1)/2, bar_y+bar_h/2), label, anchor="mm", font=fonts["bar"], fill="white", stroke_width=1, stroke_fill="#333333")
            if index == active:
                draw.rectangle((x0+1, bar_y+1, max(x0+1, x1-2), bar_y+bar_h-2), outline="#111111", width=max(2, round(3*scale)))
        templates.append(template)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.mp4")
    if temporary.exists():
        raise FileExistsError(temporary)
    decoder_cmd = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-ss", item["data-episode-start"], "-i", str(source), "-an", "-frames:v", str(total), "-vf", f"fps={fps},scale={width}:{width}:flags=lanczos", "-filter_threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    encoder_cmd = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(temporary)]
    decoder = subprocess.Popen(decoder_cmd, stdout=subprocess.PIPE)
    encoder = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE)
    try:
        for frame_index in range(total):
            raw = read_frame(decoder.stdout, width*width*3)
            if len(raw) != width*width*3:
                raise RuntimeError(f"Short source: {source} at {frame_index}/{total}")
            active = max(0, bisect_right(starts, frame_index)-1)
            skill = skills[active]
            canvas = templates[active].copy()
            view = Image.frombytes("RGB", (width, width), raw).convert("RGBA")
            overlay = Image.new("RGBA", view.size)
            draw = ImageDraw.Draw(overlay)
            if active > 0 and (frame_index-skill["start"])/fps < flash_seconds:
                draw.rectangle((0, 0, width, width), fill=(37, 99, 235, 76))
                draw.text((width/2, width/2), "SKILL CHANGE", anchor="mm", font=fonts["flash"], fill="white", stroke_width=1, stroke_fill=(0, 0, 0, 150))
            pad = round(10*scale)
            title = f"Skill {skill['index']}"
            if skill.get("label") is not None:
                title += f" · tok{skill['label']}"
            box_w = max(draw.textlength(title, font=fonts["badge"]), draw.textlength(f"frames {skill['start']}–{skill['end']}", font=fonts["small"])) + 2*pad
            draw.rounded_rectangle((pad, pad, pad+box_w, pad+round(70*scale)), radius=round(7*scale), fill=(0, 0, 0, 184))
            draw.text((2*pad, pad+round(7*scale)), title, font=fonts["badge"], fill="white")
            draw.text((2*pad, pad+round(40*scale)), f"frames {skill['start']}–{skill['end']}", font=fonts["small"], fill="#dbeafe")
            canvas.paste(Image.alpha_composite(view, overlay).convert("RGB"), (0, 0))
            draw = ImageDraw.Draw(canvas)
            x = min(width-2, round(width*frame_index/total))
            draw.rectangle((x-1, bar_y, x+2, bar_y+bar_h-1), fill="#111111")
            draw.line((x, bar_y, x, bar_y+bar_h-1), fill="white", width=max(1, round(2*scale)))
            draw.text((1, bar_y+bar_h+round(5*scale)), f"frame {frame_index} / {total-1} · {frame_index/fps:.2f}s", font=fonts["small"], fill="#666666")
            if graph is not None:
                fraction = frame_index/max(1, plot_box["frames"])
                graph_x = round(width*(plot_box["left"] + (plot_box["right"]-plot_box["left"])*fraction))
                y0 = graph_y + round(graph.height*plot_box["top"])
                y1 = graph_y + round(graph.height*(1-plot_box["bottom"]))
                draw.line((graph_x, y0, graph_x, y1), fill="#ef4444", width=max(1, round(2*scale)))
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if decoder.wait() or encoder.wait():
            raise RuntimeError(f"ffmpeg failed: {destination}")
        temporary.rename(destination)
    finally:
        for process in (decoder, encoder):
            if process.poll() is None:
                process.terminate()
                process.wait()
        decoder.stdout.close()
        if not encoder.stdin.closed:
            encoder.stdin.close()
    print(f"{destination.relative_to(output)} ({total} frames, {len(skills)} skills)", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--output", type=Path, help="Default: <report>/presentation_videos[_with_boundary]")
    parser.add_argument("--with-boundary-graph", action="store_true", help="Append the report's power=2 graph without power=1 or legends; requires cairosvg")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--flash-seconds", type=float, default=.26)
    parser.add_argument("--limit", type=int, default=0, help="Export only the first N episodes (0 = all)")
    args = parser.parse_args()
    if args.width < 320 or args.width % 2 or args.workers < 1 or args.flash_seconds < 0:
        parser.error("width must be even and >=320; workers >=1; flash-seconds >=0")
    html_path = args.html.resolve()
    parsed = Players()
    parsed.feed(html_path.read_text())
    items = parsed.items[:args.limit] if args.limit else parsed.items
    if not items:
        raise ValueError(f"No episode players in {html_path}")
    folder = "presentation_videos_with_boundary" if args.with_boundary_graph else "presentation_videos"
    output = (args.output or html_path.parent/folder).resolve()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda item: export_one(item, html_path, output, args.width, args.flash_seconds, args.with_boundary_graph), items))
    print(f"Done: {len(results)} MP4s in {output}", flush=True)


if __name__ == "__main__":
    main()
