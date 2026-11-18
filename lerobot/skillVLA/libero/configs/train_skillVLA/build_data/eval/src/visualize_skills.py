#!/usr/bin/env python3
"""CPU-only start/end image report from canonical SkillVLA boundaries."""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import html
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import yaml

BUILD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BUILD / "src"))
from train_skillVLA_config import build_settings, load_config  # noqa: E402

CAMERA = "observation.images.image"


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def image_requests(raw: Path, info: dict, metadata: pd.DataFrame, latents: dict,
                   episode_ids: list[int]) -> tuple[list[dict], dict]:
    """Use episode-local boundary indices and v3 video-file offsets separately."""
    episodes, requests = [], defaultdict(set)
    for ep_id in episode_ids:
        row = metadata.loc[ep_id]
        length = int(row["length"])
        prefix = f"videos/{CAMERA}"
        video = raw / info["video_path"].format(
            video_key=CAMERA,
            chunk_index=int(row[f"{prefix}/chunk_index"]),
            file_index=int(row[f"{prefix}/file_index"]),
            episode_index=ep_id,
        )
        offset = float(row[f"{prefix}/from_timestamp"]) * float(info["fps"])
        if abs(offset - round(offset)) > 0.05:
            raise ValueError(f"Episode {ep_id}: video offset is not frame-aligned: {offset}")
        indices = np.flatnonzero(latents["episode_id"] == ep_id)
        indices = sorted(indices, key=lambda i: int(latents["skill_index"][i]))
        skills = []
        for i in indices:
            start, end = int(latents["frame_start"][i]), int(latents["frame_end"][i])
            if not 0 <= start < end <= length:
                raise ValueError(f"Episode {ep_id}: invalid skill boundary [{start}, {end})/{length}")
            # Match training goal/UV targets: use the transition boundary image,
            # clamping the final exclusive boundary to the last recorded frame.
            end_image = min(end, length - 1)
            skill = {
                "skill_index": int(latents["skill_index"][i]),
                "token": int(latents["tokens"][i]),
                "frame_start": start, "frame_end": end,
                "end_image_frame": end_image,
                "start_video_frame": int(round(offset)) + start,
                "end_video_frame": int(round(offset)) + end_image,
            }
            requests[video].update((skill["start_video_frame"], skill["end_video_frame"]))
            skills.append(skill)
        tasks = row["tasks"]
        task = tasks if isinstance(tasks, str) else " / ".join(map(str, tasks))
        episodes.append({"episode_id": ep_id, "task": task, "length": length,
                         "video": str(video), "skills": skills})
    return episodes, requests


def decode_images(requests: dict, width: int) -> dict:
    import cv2

    images = {}
    for path, wanted in requests.items():
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open top-view video: {path}")
            found = set()
            for index in range(max(wanted) + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                if index not in wanted:
                    continue
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((width, width), Image.Resampling.LANCZOS)
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=90)
                images[(str(path), index)] = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
                found.add(index)
            if found != wanted:
                raise RuntimeError(f"Missing video frames in {path}: {sorted(wanted - found)}")
        finally:
            cap.release()
        print(f"Decoded {len(wanted)} boundary frames: {path.name}", flush=True)
    return images


def render(run: Path, episodes: list[dict], images: dict) -> str:
    esc = html.escape
    count = sum(len(ep["skills"]) for ep in episodes)
    parts = ["""<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skill boundaries</title><style>
body{font:16px system-ui;background:#10151e;color:#e6edf5;margin:0;padding:28px;max-width:1300px;margin:auto}
h1{margin-bottom:8px}p{color:#a9b9cc;overflow-wrap:anywhere}summary{cursor:pointer;font-size:20px;padding:16px}
details{background:#192231;margin:20px 0;border-radius:12px}article{border-top:1px solid #344153;padding:18px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px;max-width:760px}.pair img{width:100%;max-width:320px;border-radius:6px}
figure{margin:0}figcaption{margin:8px 0;color:#b9cbe0}.token{color:#8dc9ff}.task{padding:0 16px}
select{background:#25354c;color:white;border:1px solid #52677e;padding:8px;border-radius:5px}
@media(max-width:480px){body{padding:10px}.pair{gap:8px}summary{font-size:17px}}
</style><h1>스킬 시작 · 종료 이미지</h1>""",
             f"<p>{esc(str(run))}</p><p>{len(episodes)} episodes · {count} skills · Top view</p>",
             "<p>구간은 [start, end)이며, 종료 이미지는 학습 목표와 동일하게 min(end, 마지막 프레임)을 표시합니다. "
             "따라서 전환 지점의 종료 이미지와 다음 스킬 시작 이미지는 같습니다.</p>",
             '<label>에피소드 <select id="episode"><option value="all">전체</option>']
    for ep in episodes:
        parts.append(f'<option value="ep{ep["episode_id"]}">Episode {ep["episode_id"]}</option>')
    parts.append('</select></label>')
    for ep in episodes:
        parts.append(f'<details open id="ep{ep["episode_id"]}"><summary>Episode {ep["episode_id"]} · '
                     f'{len(ep["skills"])} skills · {ep["length"]} frames</summary><p class="task">{esc(ep["task"])}</p>')
        for skill in ep["skills"]:
            parts.append(f'<article><b>Skill {skill["skill_index"]}</b> '
                         f'<span class="token">FSQ token {skill["token"]}</span> · '
                         f'[{skill["frame_start"]}, {skill["frame_end"]})<div class="pair">')
            for title, key, frame in (("시작", "start_video_frame", skill["frame_start"]),
                                      ("종료", "end_video_frame", skill["end_image_frame"])):
                image = images[(ep["video"], skill[key])]
                parts.append(f'<figure><figcaption>{title} · frame {frame}</figcaption>'
                             f'<img loading="lazy" src="{image}" alt="{title} frame {frame}"></figure>')
            parts.append('</div></article>')
        parts.append('</details>')
    parts.append("""<script>document.getElementById('episode').onchange=function(){
document.querySelectorAll('details').forEach(e=>{e.hidden=this.value!=='all'&&e.id!==this.value;});
};</script></html>""")
    return "\n".join(parts)


def render_comparison(runs: list[Path], groups: list[list[dict]], images: dict) -> str:
    """Align episodes and time axes, without treating skill indices as matches."""
    esc = html.escape
    labels = ["ft_own" if "_ft_own_" in run.name else "ft" if "_ft_" in run.name else run.name
              for run in runs]
    maps = [{ep["episode_id"]: ep for ep in group} for group in groups]
    if set(maps[0]) != set(maps[1]):
        raise ValueError("Comparison reports must contain the same episode IDs")
    parts = ["""<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FT skill boundary comparison</title><style>
*{box-sizing:border-box}body{font:15px system-ui;background:#10151e;color:#e6edf5;margin:0;padding:24px}
main{max-width:1600px;margin:auto}p{color:#adbed2;overflow-wrap:anywhere}summary{cursor:pointer;padding:16px;font-size:19px}
details{margin:20px 0;background:#172130;border-radius:10px}details>p{padding:0 16px}
.columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;padding:0 16px 16px}
.column{min-width:0;background:#111a28;border-radius:8px;padding:12px}.column h2{position:sticky;top:0;background:#111a28;margin:0;padding:8px 0;z-index:1}
.column:first-child{border-top:3px solid #72b7ff}.column:last-child{border-top:3px solid #edae73}
.timeline{display:flex;height:32px;width:100%;margin:10px 0}.timeline a{min-width:0;overflow:hidden;text-align:center;color:white;text-decoration:none;border-right:2px solid #111a28;line-height:32px}
article{border-top:1px solid #344153;padding:12px 0;scroll-margin-top:60px}article:target{outline:2px solid #87c6ff}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}.pair img{width:100%;max-width:320px;border-radius:5px}
figure{margin:0}figcaption{margin:8px 0;color:#b9cbe0}.boundaries{font-family:monospace;font-size:13px;min-height:32px}
select{background:#25354c;color:white;padding:8px;border:1px solid #52677e;border-radius:5px}
@media(max-width:800px){body{padding:8px}.columns{min-width:720px}details{overflow-x:auto}}
</style><main><h1>FT 스킬 분할 비교</h1>
<p>좌우는 같은 에피소드입니다. 타임라인은 같은 프레임 축을 사용하며, 클릭하면 해당 스킬로 이동합니다.
스킬 번호는 각 분할 내부의 순서입니다. 시작·종료 이미지에는 학습과 동일한 경계 프레임을 사용합니다.</p>"""]
    for i, run in enumerate(runs):
        count = sum(len(ep["skills"]) for ep in groups[i])
        parts.append(f'<p>{"왼쪽" if i == 0 else "오른쪽"}: <b>{esc(labels[i])}</b> · {count} skills · {esc(run.name)}</p>')
    parts.append('<label>에피소드 <select id="episode"><option value="all">전체</option>')
    for ep_id in maps[0]:
        parts.append(f'<option value="ep{ep_id}">Episode {ep_id}</option>')
    parts.append('</select></label>')
    for ep_id, left in maps[0].items():
        right = maps[1][ep_id]
        if any(left[key] != right[key] for key in ("length", "video", "task")):
            raise ValueError(f"Episode {ep_id}: comparison source images/tasks differ")
        boundaries = [{s["frame_end"] for s in ep["skills"][:-1]} for ep in (left, right)]
        difference = sorted(boundaries[0] ^ boundaries[1])
        parts.append(f'<details open id="ep{ep_id}"><summary>Episode {ep_id} · '
                     f'{len(left["skills"])} vs {len(right["skills"])} skills · {left["length"]} frames</summary>'
                     f'<p>{esc(left["task"])}</p><p>서로 다른 내부 경계: '
                     f'{", ".join(map(str, difference)) if difference else "없음"}</p><div class="columns">')
        for side, ep in enumerate((left, right)):
            parts.append(f'<section class="column"><h2>{esc(labels[side])} · {len(ep["skills"])} skills</h2><div class="timeline">')
            for skill in ep["skills"]:
                index = skill["skill_index"]
                color = f'hsl({(index * 67 + 205) % 360},45%,38%)'
                parts.append(f'<a style="flex:{skill["frame_end"] - skill["frame_start"]};background:{color}" '
                             f'href="#e{ep_id}-side{side}-s{index}" title="[{skill["frame_start"]}, {skill["frame_end"]})">{index}</a>')
            parts.append('</div><p class="boundaries">경계: ' +
                         ' → '.join(map(str, [s["frame_start"] for s in ep["skills"]] + [ep["length"]])) + '</p>')
            for skill in ep["skills"]:
                parts.append(f'<article id="e{ep_id}-side{side}-s{skill["skill_index"]}"><b>Skill {skill["skill_index"]}</b> · '
                             f'FSQ {skill["token"]} · [{skill["frame_start"]}, {skill["frame_end"]})<div class="pair">')
                for title, key, frame in (("시작", "start_video_frame", skill["frame_start"]),
                                          ("종료", "end_video_frame", skill["end_image_frame"])):
                    parts.append(f'<figure><figcaption>{title} · frame {frame}</figcaption>'
                                 f'<img loading="lazy" src="{images[(ep["video"], skill[key])]}" alt="{title} frame {frame}"></figure>')
                parts.append('</div></article>')
            parts.append('</section>')
        parts.append('</div></details>')
    parts.append("""<script>document.getElementById('episode').onchange=function(){
document.querySelectorAll('details').forEach(e=>{e.hidden=this.value!=='all'&&e.id!==this.value;});
};</script></main></html>""")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BUILD / "eval/visualize_config.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text()) or {}
    build_config = resolve(config_path.parent, config.get("build_config", "../train_skillVLA_config.yaml"))
    settings = build_settings(load_config(build_config))
    run = Path(settings["skillvla_run_dir"])
    override = str(config.get("dataset_run", "") or "").strip()
    if override:
        if Path(override).name != override or override in (".", ".."):
            raise ValueError("dataset_run must be one run folder name")
        run = run.parent / override
    raw = Path(settings["raw_dataset_dir"])
    info = json.loads((raw / "meta/info.json").read_text())
    if CAMERA not in info["features"]:
        raise ValueError(f"Dataset has no top-view camera: {CAMERA}")
    files = sorted((raw / "meta/episodes").rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {raw / 'meta/episodes'}")
    metadata = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True).set_index("episode_index")
    if not metadata.index.is_unique:
        raise ValueError("Episode metadata contains duplicate episode IDs")
    with np.load(run / "skill_latents.npz", allow_pickle=False) as archive:
        latents = {key: archive[key].copy() for key in archive.files}
    ids = config.get("episode_ids") or sorted(int(ep) for ep in metadata.index)
    ids = list(dict.fromkeys(int(ep) for ep in ids))
    missing = sorted(set(ids) - set(metadata.index))
    if missing:
        raise ValueError(f"Unknown episode_ids: {missing}")
    width = int(config.get("image_width", 320))
    if width < 1:
        raise ValueError("image_width must be positive")
    episodes, requests = image_requests(raw, info, metadata, latents, ids)
    compare_run, compare_episodes = None, None
    compare_name = str(config.get("compare_dataset_run", "") or "").strip()
    if compare_name:
        if Path(compare_name).name != compare_name or compare_name in (".", ".."):
            raise ValueError("compare_dataset_run must be one run folder name")
        compare_run = run.parent / compare_name
        with np.load(compare_run / "skill_latents.npz", allow_pickle=False) as archive:
            compare_latents = {key: archive[key].copy() for key in archive.files}
        compare_episodes, other_requests = image_requests(raw, info, metadata, compare_latents, ids)
        for video, frames in other_requests.items():
            requests[video].update(frames)
    if not requests:
        raise ValueError("Selected episodes contain no skills")
    images = decode_images(requests, width)
    output = resolve(config_path.parent, config.get("output_dir", "outputs")) / raw.name / run.name
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": str(run), "raw_dataset": str(raw), "camera": CAMERA,
                "endpoint_convention": "min(frame_end, episode_length - 1)",
                "episodes": episodes}
    reports = [("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2)),
               ("index.html", render(run, episodes, images))]
    if compare_run is not None:
        reports.append(("comparison.html", render_comparison([run, compare_run], [episodes, compare_episodes], images)))
        reports.append(("comparison_manifest.json", json.dumps(
            {"left": manifest, "right": {**manifest, "dataset": str(compare_run), "episodes": compare_episodes}},
            ensure_ascii=False, indent=2)))
    for filename, content in reports:
        target = output / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
    print(f"Report: {output / 'index.html'}", flush=True)
    if compare_run is not None:
        print(f"Comparison: {output / 'comparison.html'}", flush=True)
    print(f"Episodes: {len(episodes)}; skills: {sum(len(ep['skills']) for ep in episodes)}", flush=True)


if __name__ == "__main__":
    main()
