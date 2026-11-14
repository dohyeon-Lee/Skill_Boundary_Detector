#!/usr/bin/env python3
"""ABC evaluation layout, annotated videos and an offline HTML results browser."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


OVERLAY_VERSION = "abc_language_outcome/v1"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "task"


def panel_name(policy_type: str, checkpoint=None) -> str:
    step = Path(checkpoint).parent.name if checkpoint else ""
    label = f"_{int(step) // 1000}k" if step.isdigit() and int(step) % 1000 == 0 else ""
    return f"00_{slug(policy_type)}{label}"


def video_paths(output: Path, panel: str, task: str, index: int) -> tuple[Path, Path]:
    folder = output / "panels" / panel
    name = f"world_{index:03d}.mp4"
    return folder / "videos_raw" / slug(task) / name, folder / "videos" / slug(task) / name


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def annotation_banners(width: int, world: dict) -> tuple[np.ndarray, np.ndarray]:
    top = Image.new("RGB", (width, 26), (34, 139, 34) if world["success"] else (178, 34, 34))
    draw = ImageDraw.Draw(top)
    label = f"{'SUCCESS' if world['success'] else 'FAIL'}  |  Episode {world['world_index']:03d}"
    f = font(15)
    draw.text(((width - draw.textlength(label, font=f)) / 2, 4), label, font=f, fill="white")
    f = font(15)
    lines, line = [], ""
    for word in str(world.get("prompt", "")).split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=f) > width - 20:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    height = 10 + 20 * len(lines)
    bottom = Image.new("RGB", (width, height + height % 2), (22, 27, 34))
    draw = ImageDraw.Draw(bottom)
    for i, line in enumerate(lines):
        draw.text((10, 4 + 20 * i), line, font=f, fill="white")
    return np.asarray(top), np.asarray(bottom)


def annotate_video(source: Path, destination: Path, world: dict, fps: float = 30) -> dict:
    """Keep the raw input; atomically publish only a complete annotated video."""
    import av

    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Raw and annotated video paths must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".annotating.mp4")
    reader = imageio.get_reader(str(source), input_params=["-threads", "1"])
    writer = None
    frames = 0
    try:
        fps = float(reader.get_meta_data().get("fps", fps))
        for frame in reader:
            if writer is None:
                top, bottom = annotation_banners(frame.shape[1], world)
                writer = imageio.get_writer(str(temporary), fps=fps, codec="libx264", macro_block_size=1,
                                            ffmpeg_params=["-crf", "18", "-preset", "fast", "-threads", "1"])
            writer.append_data(np.concatenate([top, frame[:, :, :3], bottom], axis=0))
            frames += 1
    finally:
        reader.close()
        if writer is not None:
            writer.close()
    if not frames:
        raise ValueError(f"Empty source video: {source}")
    # A truncated encoder output must never replace a usable video.
    with av.open(str(temporary)) as encoded:
        encoded.streams.video[0].codec_context.thread_count = 1
        if sum(1 for _ in encoded.decode(video=0)) != frames:
            raise ValueError(f"Frame count changed during annotation: {source}")
    temporary.replace(destination)
    return {"version": OVERLAY_VERSION, "frames": frames, "fps": fps}


def finish_video(raw: Path, target: Path, world: dict, fps: float) -> None:
    world["raw_video_path"] = str(raw)
    world["video_overlay"] = annotate_video(raw, target, world, fps)
    world["video_path"] = str(target)


def build_result(policy: dict, dataset: str, tasks: list[dict], *, expected: int, wall_s=0.0) -> dict:
    total = sum(task["num_worlds"] for task in tasks)
    successes = sum(task["successes"] for task in tasks)
    return {"format": "sbd_abc_sim_eval/v1", "policy": policy, "dataset": dataset,
            "num_tasks": len(tasks), "num_worlds": total, "expected_worlds": expected,
            "successes": successes, "success_rate": successes / total if total else None,
            "wall_s": wall_s, "tasks": tasks}


def href(path: Path, root: Path) -> str:
    return html.escape(quote(os.path.relpath(path, root), safe="/"), quote=True)


def render_report(output: Path, result: dict) -> None:
    """Render saved episode outcomes only; never infer a missing success label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = result["tasks"]
    if not tasks:
        return
    fig, ax = plt.subplots(figsize=(9, max(2.8, len(tasks) * 0.5 + 1.5)))
    labels = [t["requested_task"].replace("_", " ") for t in tasks]
    rates = [100 * t["successes"] / t["num_worlds"] if t["num_worlds"] else 0 for t in tasks]
    ax.barh(labels, rates, color="#4c78a8")
    ax.invert_yaxis()
    ax.set_xlim(0, 126)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Success rate (%)")
    ax.set_title("ABC task success rates")
    for i, (rate, task) in enumerate(zip(rates, tasks, strict=True)):
        ax.text(rate + 1, i, f"{rate:g}% ({task['successes']}/{task['num_worlds']})", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "task_success_rates.tmp.png", dpi=150)
    plt.close(fig)
    (output / "task_success_rates.tmp.png").replace(output / "task_success_rates.png")

    worlds = [(task, w) for task in tasks for w in sorted(task["worlds"], key=lambda x: x["world_index"])]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    x = np.arange(len(worlds))
    ax.bar(x, [w["steps"] for _, w in worlds], color=["#228b22" if w["success"] else "#b22222" for _, w in worlds])
    ax.set_xlabel("Episode (task order, then episode index)")
    ax.set_ylabel("Executed actions")
    ax.set_title("Episode duration — green: success · red: failure")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "episode_steps.tmp.png", dpi=150)
    plt.close(fig)
    (output / "episode_steps.tmp.png").replace(output / "episode_steps.png")

    completed, success = result["num_worlds"], result["successes"]
    expected = result.get("expected_worlds", completed)
    summary_name = "summary.json" if (output / "metrics/summary.json").exists() else "partial_summary.json"
    cards = []
    for task, world in worlds:
        outcome = "success" if world["success"] else "failure"
        prompt = html.escape(str(world["prompt"]))
        video = world.get("video_path")
        video_html = (f'<video controls playsinline preload="none" src="{href(Path(video), output)}"></video>'
                      if video else '<p>No video recorded</p>')
        metric = output / "metrics/tasks" / slug(task["task"]) / f"world_{world['world_index']:03d}.json"
        cards.append(f'''<article data-outcome="{outcome}"><div class="card-head"><b>Episode {world['world_index']:03d}</b>
<span class="{outcome}">{outcome.upper()}</span></div>{video_html}<p>{prompt}</p>
<p class="muted">{html.escape(task['requested_task'])} · Seed {world['seed']} · {world['steps']:,} actions</p>
<a href="{href(metric, output)}">Episode metrics</a></article>''')
    task_rows = "".join(f"<tr><td>{html.escape(t['requested_task'])}</td><td>{t['successes']}/{t['num_worlds']}</td>"
                        f"<td>{100*t['successes']/t['num_worlds']:.1f}%</td></tr>" for t in tasks if t["num_worlds"])
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ABC evaluation — {html.escape(output.name)}</title><style>
body{{background:#10151d;color:#e8edf4;font:16px system-ui;margin:0 auto;padding:28px;max-width:1380px}}a{{color:#8ebcff}}
h1{{font-size:26px}}.muted{{color:#a3afbf;font-size:14px}}.stats,.charts,.videos{{display:grid;gap:18px}}
.stats{{grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}}.stats div,article,.chart{{background:#1a2330;border:1px solid #344154;border-radius:12px;padding:16px}}
.stats b{{display:block;font-size:28px}}.charts{{grid-template-columns:repeat(auto-fit,minmax(340px,1fr));margin:24px 0}}.chart img{{width:100%}}
.videos{{grid-template-columns:repeat(auto-fit,minmax(410px,1fr))}}video{{width:100%;background:#000;margin-top:14px}}.card-head{{display:flex;justify-content:space-between}}
.success{{color:#7de49d}}.failure{{color:#ff9292}}table{{border-collapse:collapse;margin:20px 0}}td,th{{padding:10px 24px 10px 0;text-align:left}}
input,select{{padding:10px;background:#1a2330;color:white;border:1px solid #506079;border-radius:6px;margin:0 10px 18px 0}}[hidden]{{display:none!important}}
@media(max-width:500px){{body{{padding:14px}}.videos,.charts{{grid-template-columns:1fr}}}}
</style><h1>ABC evaluation</h1><p>{html.escape(output.name)}</p>
<div class="stats"><div>Success rate<b>{100*success/completed if completed else 0:.1f}%</b></div><div>Successful<b>{success} / {completed}</b></div>
<div>Completed episodes<b>{completed} / {expected}</b></div><div>Failed<b>{completed-success}</b></div></div>
<p class="muted">Checkpoint: {html.escape(str(result['policy'].get('checkpoint', result['policy'].get('type', ''))))}</p>
<p><a href="metrics/{summary_name}">Summary JSON</a> · <a href="metrics/resolved_config.json">Evaluation config</a></p>
<div class="charts"><div class="chart"><a href="task_success_rates.png"><img src="task_success_rates.png" alt="Task success rates"></a></div>
<div class="chart"><a href="episode_steps.png"><img src="episode_steps.png" alt="Episode action counts by outcome"></a></div></div>
<table><tr><th>Task</th><th>Success / completed</th><th>Success rate</th></tr>{task_rows}</table>
<h2>Episodes</h2><select id="outcome"><option value="all">All outcomes</option><option value="success">Success only</option><option value="failure">Failure only</option></select>
<input id="search" type="search" placeholder="Search language, seed or episode" aria-label="Search episodes"><span id="count"></span>
<div class="videos">{''.join(cards)}</div><script>
function filter(){{let count=0;for(const card of document.querySelectorAll('article')){{card.hidden=!(
(outcome.value==='all'||card.dataset.outcome===outcome.value)&&card.textContent.toLowerCase().includes(search.value.toLowerCase()));if(!card.hidden)count++;}}document.getElementById('count').textContent=count+' episodes';}}
const outcome=document.getElementById('outcome'),search=document.getElementById('search');outcome.addEventListener('change',filter);search.addEventListener('input',filter);filter();
</script></html>'''
    target = output / "index.html"
    temporary = target.with_suffix(".html.tmp")
    temporary.write_text(text)
    temporary.replace(target)


def organize_existing(output: Path, workers=4) -> dict:
    """Migrate an idle completed run, preserving raw videos and numeric metrics."""
    output = output.resolve(strict=True)
    summary_path = output / "metrics/summary.json"
    if not summary_path.exists():
        summary_path = output / "summary.json"
    result = json.loads(summary_path.read_text())
    panel = panel_name(result["policy"]["type"], result["policy"].get("checkpoint"))
    originals = {str(p.relative_to(output)): json.loads(p.read_text()) for p in output.rglob("*.json")}
    backup = output / "metrics/layout_originals.json"
    if not backup.exists():
        atomic_json(backup, originals)
    jobs = []
    for task in result["tasks"]:
        metric_dir = output / "metrics/tasks" / slug(task["task"])
        for world in task["worlds"]:
            raw, target = video_paths(output, panel, task["task"], world["world_index"])
            if world.get("video_path"):
                source = Path(world["video_path"])
                if not raw.exists():
                    if source == target:
                        raise ValueError(f"Raw source missing: {raw}")
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    source.rename(raw)
                if world.get("video_overlay", {}).get("version") != OVERLAY_VERSION or not target.exists():
                    jobs.append((raw, target, world))
                else:
                    world["raw_video_path"] = str(raw)
                    world["video_path"] = str(target)
    def convert(job):
        raw, target, world = job
        finish_video(raw, target, world, 30)
        print(f"Annotated episode {world['world_index']:03d}: {'SUCCESS' if world['success'] else 'FAIL'}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(convert, jobs))
    for task in result["tasks"]:
        metric_dir = output / "metrics/tasks" / slug(task["task"])
        for world in task["worlds"]:
            atomic_json(metric_dir / f"world_{world['world_index']:03d}.json", world)
        atomic_json(metric_dir / "summary.json", task)
        old = output / slug(task["task"])
        if old.exists():
            # Metadata is preserved in layout_originals.json and the new metrics.
            for path in old.glob("*.json"):
                path.unlink()
            old.rmdir()  # Refuse to remove any unrecognized remaining file.
    resolved = output / "resolved_config.json"
    if resolved.exists():
        resolved.rename(output / "metrics/resolved_config.json")
    result.setdefault("expected_worlds", result["num_worlds"])
    atomic_json(output / "metrics/summary.json", result)
    if (output / "summary.json").exists():
        (output / "summary.json").unlink()
    render_report(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    organize_existing(args.output, args.workers)


if __name__ == "__main__":
    main()
