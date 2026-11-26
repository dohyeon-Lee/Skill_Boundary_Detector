"""Compare saved LIBERO FT skillsets, aligning subsets by original episode IDs."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "lerobot/src/lerobot").is_dir())
FT_DIR = Path(__file__).resolve().parents[1]
SHARED = FT_DIR.parents[1] / "train_skills"
sys.path.insert(0, str(SHARED / "src"))
from train_skills_config import (  # noqa: E402
    _merge_global, _read_yaml, resolve_skillvla_dataset_run, resolve_training_episodes,
)
from sbd_build_contract import validate_manifest  # noqa: E402


def episode_metadata(dataset):
    files = sorted((dataset / "meta/episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Episode metadata missing: {dataset}")
    rows = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if rows.episode_index.duplicated().any():
        raise ValueError(f"Duplicate episode IDs: {dataset}")
    return rows.set_index("episode_index").sort_index()


def original_episode_ids(dataset, seen=()):
    """Track subset reindexing; equal local IDs do not imply the same recording."""
    dataset = dataset.resolve()
    if dataset in seen:
        raise ValueError(f"Cyclic dataset provenance: {dataset}")
    ids = list(map(int, episode_metadata(dataset).index))
    info = json.loads((dataset / "meta/info.json").read_text())
    original = info.get("source_episode_indices")
    if original is None:
        return {ep: (dataset.name, ep) for ep in ids}
    if len(original) != len(ids) or len(set(original)) != len(original):
        raise ValueError(f"Invalid source_episode_indices: {dataset}")
    parent = dataset.parent / info["source_dataset"]
    parents = original_episode_ids(parent, (*seen, dataset))
    return {ep: parents[int(source)] for ep, source in zip(ids, original, strict=True)}


def find_skillset(run, contract):
    candidates = []
    for path in sorted((run.parent / "_work").glob("*/skillset/skillset_manifest.json")):
        try:
            validate_manifest(path, contract)
        except ValueError:
            continue
        manifest = json.loads(path.read_text())
        if manifest["detector"].get("boundary_threshold_mode") == "global_mean":
            reference = Path(manifest["detector"].get("global_threshold_path", ""))
            own = reference.parent.resolve() == path.parent.resolve()
            if own != ("_ft_own_" in run.name or "_pt_" in run.name):
                continue
        candidates.append((path.parent, manifest))
    if len(candidates) != 1:
        raise ValueError(f"Expected one matching saved SBD for {run}; found {[str(p) for p, _ in candidates]}")
    return candidates[0]


def load_build(root, data, label):
    source = root / data["source"]
    tag, _ = resolve_skillvla_dataset_run(source, data["run"], data.get("relabeled", ""))
    run = source / tag
    record = json.loads((run / "segmentation_contract.json").read_text())
    if record.get("stage") != "final":
        raise ValueError(f"Incomplete SkillVLA build: {run}")
    skillset, manifest = find_skillset(run, record["contract"])
    # Raw dataset locations come from the global root, not another host's path.
    raw = root.parent / data["source"]
    info = json.loads((run / "skillvla/meta/info.json").read_text())
    with np.load(run / "skill_latents.npz", allow_pickle=False) as z:
        bounds = {k: z[k].copy() for k in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "tokens")}
    provenance = json.loads((run / "fsq_source.json").read_text())
    return dict(label=label, source=data["source"], run=run, skillset=skillset,
                manifest=manifest, raw=raw, metadata=episode_metadata(raw),
                original=original_episode_ids(raw), bounds=bounds, info=info,
                fsq=(provenance["source_fsq_run_name"], str(provenance["source_fsq_checkpoint"])))


def saved_episode(build, episode):
    bounds = build["bounds"]
    rows = np.flatnonzero(bounds["episode_id"] == episode)
    rows = rows[np.argsort(bounds["skill_index"][rows])]
    if not len(rows):
        raise ValueError(f"No saved skills for episode {episode}: {build['run']}")
    skills = [(int(bounds["frame_start"][r]), int(bounds["frame_end"][r]), int(bounds["tokens"][r])) for r in rows]
    edges = [s for s, _, _ in skills] + [skills[-1][1]]
    with np.load(build["skillset"] / f"curves/ep{episode:07d}.npz", allow_pickle=False) as z:
        curve = {k: z[k].copy() for k in z.files}
    if not np.array_equal(curve.get("boundaries", []), edges):
        raise ValueError(f"Saved SBD boundaries differ from FT labels: {build['source']} episode {episode}")
    length = int(build["metadata"].loc[episode, "length"])
    if edges[0] != 0 or edges[-1] != length or any(s >= e for s, e, _ in skills):
        raise ValueError(f"Invalid episode coverage: {build['source']} episode {episode}")
    task = int(bounds["task_id"][rows[0]])
    return dict(episode_id=episode, task_id=task, length=length, skills=skills, curve=curve,
                threshold=float(np.asarray(curve.get("threshold_val", curve["mean_val"])).item()))


def selection(config_path, *, all_episodes=False):
    raw_config = _read_yaml(config_path)
    # Previewing data must not resolve/load warm-start checkpoints.
    cfg = _merge_global(config_path, {"dataset": raw_config["dataset"]})
    data = dict(cfg["dataset"])
    if all_episodes:
        data.update(n_episodes="all", episode_ids=[])
    primary = {**cfg, "dataset": data}
    root = Path(cfg["project_root"]) / cfg["dataset_root"] / data.get("skillvla_root", "skillvla")
    reference = load_build(root, data, "FT selection")
    episodes = resolve_training_episodes(primary, reference["run"] / "skillvla")
    if episodes is None:
        episodes = list(map(int, reference["metadata"].index))
    configs = raw_config.get("sbd_preview", {}).get("builds", [])
    if not configs:
        configs = [{**data, "label": "Current FT build"}]
    builds = [load_build(root, item, item["label"]) for item in configs]
    levels = reference["info"]["skill_fsq_levels"]
    if any(b["fsq"] != reference["fsq"] or b["info"]["skill_fsq_levels"] != levels for b in builds):
        raise ValueError("Compared builds must use the same FSQ encoder/checkpoint and levels")
    mappings = []
    for build in builds:
        inverse = {origin: ep for ep, origin in build["original"].items()}
        if len(inverse) != len(build["original"]):
            raise ValueError(f"Ambiguous original episode mapping: {build['raw']}")
        mappings.append(inverse)
    groups = []
    task_episodes = {}
    for task, episode in zip(reference["bounds"]["task_id"], reference["bounds"]["episode_id"]):
        task_episodes.setdefault(int(task), set()).add(int(episode))
    ordinals = {ep: position for ids in task_episodes.values() for position, ep in enumerate(sorted(ids))}
    for episode in episodes:
        origin = reference["original"][episode]
        ref = saved_episode(reference, episode)
        variants = []
        for build, mapping in zip(builds, mappings, strict=True):
            local = mapping.get(origin)
            result = saved_episode(build, local) if local is not None else None
            if result is not None and (result["length"], result["task_id"]) != (ref["length"], ref["task_id"]):
                raise ValueError(f"Different recording/task mapped to {origin}: {build['source']}")
            variants.append(result)
        groups.append(dict(episode_id=episode, original=list(origin), task_id=ref["task_id"],
                           task_episode=ordinals[episode], reference=ref, variants=variants))
    groups.sort(key=lambda group: (group["task_id"], group["task_episode"]))
    identity = dict(reference=str(reference["run"]), episodes=episodes, builds=[str(b["run"]) for b in builds])
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
    output = FT_DIR / "outputs" / f"sbd_comparison_{data['source']}_ep{len(episodes)}_{digest}"
    plan = dict(config=str(config_path), output=str(output), episodes=episodes,
                builds=[dict(label=b["label"], run=str(b["run"]), skillset=str(b["skillset"])) for b in builds],
                selection=dict(episode_selection=data.get("episode_selection", "global"),
                               n_episodes=data.get("n_episodes", "all"), episode_ids=data.get("episode_ids", [])))
    return plan, reference, builds, groups


STYLE = """<style>
*{box-sizing:border-box}body{background:#10151e;color:#e6edf5;max-width:1800px;margin:auto;padding:20px;text-align:center}
.card{background:#192231;color:#e6edf5;border-color:#344153;overflow:visible}.cap{font-size:16px;color:#c8d6e8;font-weight:600}
.player-status{color:#b4c6dc}.episode-player>.skill-track,.video-focus .skill-badge,.boundary-flash{display:none}
.comparison-columns{display:grid;grid-template-columns:repeat(var(--columns),minmax(0,1fr));gap:14px;margin-top:14px}
.comparison-column{min-width:0;background:#111a28;padding:12px;border-radius:8px}.comparison-column h3{margin:0 0 8px}
.comparison-current{height:32px;line-height:32px;font-weight:700}.comparison-caption{font-size:12px;color:#b7c9de;min-height:40px;overflow-wrap:anywhere}
.compare-track{display:flex;width:100%;height:28px;margin:8px 0}.compare-track button{min-width:0;border:1px solid #555;color:white;overflow:hidden;padding:0;cursor:pointer;text-shadow:0 1px 2px #000}
.compare-track button.active{outline:2px solid white;outline-offset:-3px}.compare-chart{position:relative;width:100%;line-height:0;background:white}
.card .compare-chart img{width:100%;max-width:100%;cursor:crosshair}.compare-playhead{position:absolute;top:var(--top);bottom:var(--bottom);width:2px;background:#ef4444;pointer-events:none}
.comparison-note{color:#b7c9de;max-width:1200px;margin:12px auto}.missing{padding:40px 12px;color:#a7b4c6}
@media(max-width:1000px){.comparison-columns{grid-template-columns:1fr}body{padding:8px}}
</style>"""

SCRIPT = """<script>
document.querySelectorAll('.comparison-columns').forEach(columns=>{
 const card=columns.closest('.card'),player=card.querySelector('.episode-player'),video=player.querySelector('video');
 const fps=Number(player.dataset.fps),start=Number(player.dataset.episodeStart),end=Number(player.dataset.episodeEnd);
 const frameNow=()=>Math.max(0,Math.floor((video.currentTime-start)*fps+1e-6));
 let animation;
 function update(){const frame=frameNow();columns.querySelectorAll('.comparison-column[data-skills]').forEach(column=>{
  const skills=JSON.parse(column.dataset.skills),skill=skills.find(s=>frame>=s[0]&&frame<s[1]);
  column.querySelector('.comparison-current').textContent=skill?`FSQ ${skill[2]} · frames ${skill[0]}–${skill[1]}`:'';
  column.querySelectorAll('.compare-track button').forEach((button,i)=>button.classList.toggle('active',skills[i]===skill));
  const chart=column.querySelector('.compare-chart'),ratio=Math.max(0,Math.min(1,frame/Number(chart.dataset.frames)));
  chart.querySelector('.compare-playhead').style.left=`${100*(Number(chart.dataset.left)+(Number(chart.dataset.right)-Number(chart.dataset.left))*ratio)}%`;
 });}
 async function seek(frame){if(!video.getAttribute('src'))await loadEpisode(player,false);video.currentTime=Math.min(end-.001,start+frame/fps);update();}
 columns.querySelectorAll('[data-frame]').forEach(button=>button.addEventListener('click',()=>seek(Number(button.dataset.frame))));
 columns.querySelectorAll('.compare-chart').forEach(chart=>chart.addEventListener('click',event=>{
  const rect=chart.getBoundingClientRect(),x=(event.clientX-rect.left)/rect.width,left=Number(chart.dataset.left),right=Number(chart.dataset.right);
  seek(Math.max(0,Math.min(Number(chart.dataset.frames)-1,(x-left)/(right-left)*Number(chart.dataset.frames))));
 }));
 function tick(){update();if(!video.paused)animation=requestAnimationFrame(tick);}
 video.addEventListener('play',()=>{cancelAnimationFrame(animation);tick();});
 video.addEventListener('pause',()=>{cancelAnimationFrame(animation);update();});
 video.addEventListener('seeking',update);video.addEventListener('timeupdate',update);update();
});
</script>"""


def render(plan, reference, builds, groups):
    sys.path.insert(0, str(SHARED / "skill_eval/src"))
    from codebook_visualizer import _episode_row, _load_episodes_meta
    from dp_skillset_eval import _episode_video_metadata
    from skillset_boundary_viz import fsq_grid_color, render_skillset_card, save_gallery

    output = Path(plan["output"])
    assets = output / "src"
    (assets / "timelines").mkdir(parents=True, exist_ok=True)
    levels = reference["info"]["skill_fsq_levels"]
    raw = reference["raw"]
    meta = _load_episodes_meta(raw)
    fps = float(json.loads((raw / "meta/info.json").read_text())["fps"])
    cards, extras, records = [], [], []
    for group in groups:
        episode, ref = group["episode_id"], group["reference"]
        video = _episode_video_metadata(raw, meta, reference["manifest"].get("image_key", "observation.images.image"),
                                        episode, output, fps, clip_cache_dir=assets / "videos")
        media = render_skillset_card(ref["skills"], None, None, episode_video=video, show_frames=False)
        media.pop("timeline_svg", None)
        marker = f"SBD_COMPARISON_{episode}_{len(cards)}"
        title = " / ".join(map(str, _episode_row(meta, episode)["tasks"]))
        caption = f"Task {ref['task_id']} · task 내 episode {group['task_episode']} · 원본 episode {group['original'][1]}"
        cards.append((f"Task {ref['task_id']} · {title}", caption + " · " + marker, media))
        columns, variants = [], []
        for index, (build, result) in enumerate(zip(builds, group["variants"], strict=True)):
            heading = f"<h3>{html.escape(build['label'])}</h3>"
            if result is None:
                columns.append(f"<section class='comparison-column'>{heading}<p class='missing'>이 원본 에피소드는 해당 빌드에 없습니다.</p></section>")
                variants.append(None)
                continue
            curve = result["curve"].copy()
            curve.update(code_labels_only=True, fsq_levels=levels)
            for key in ("denoising_gain", "delta_bic"):
                curve.pop(key, None)
            graph = render_skillset_card(result["skills"], None, curve, show_frames=False)
            path = assets / "timelines" / f"ep{episode:05d}_build{index}.svg"
            path.write_bytes(base64.b64decode(graph["timeline_svg"]))
            skills = result["skills"]
            encoded = html.escape(json.dumps(skills), quote=True)
            track = ''.join(f"<button type='button' data-frame='{s}' style='flex:{e-s};background:{fsq_grid_color(code, levels)}' title='FSQ {code} · [{s}, {e})'>{code}</button>" for s, e, code in skills)
            mask = build["manifest"]["detector"].get("initial_mask_frames", 0)
            columns.append(
                f"<section class='comparison-column' data-skills='{encoded}'>{heading}"
                f"<div class='comparison-caption'>local episode {result['episode_id']} · {len(skills)} skills · threshold {result['threshold']:.6g} · initial mask {mask}</div>"
                f"<div class='comparison-current'></div><div class='compare-track'>{track}</div>"
                f"<div class='compare-chart' data-frames='{graph['timeline_n_frames']}' data-left='{graph['timeline_plot_left']}' data-right='{graph['timeline_plot_right']}' "
                f"style='--top:{graph['timeline_plot_top']*100}%;--bottom:{graph['timeline_plot_bottom']*100}%'>"
                f"<img loading='lazy' src='{path.relative_to(output).as_posix()}' alt='{html.escape(build['label'], quote=True)} saved SBD'>"
                "<div class='compare-playhead'></div></div></section>")
            variants.append(dict(episode_id=result["episode_id"], skills=skills, threshold=result["threshold"], initial_mask_frames=mask))
        extras.append((marker, f"<div class='comparison-columns' style='--columns:{len(builds)}'>{''.join(columns)}</div>"))
        records.append({k: group[k] for k in ("episode_id", "original", "task_id", "task_episode") } | {"variants": variants})
        print(f"[SBD comparison] {len(cards)}/{len(groups)} · task {ref['task_id']} · episode {episode}", flush=True)
    save_gallery(output, f"LIBERO FT SBD 비교 · {len(groups)} episodes", cards)
    report = output / "index.html"
    content = report.read_text()
    for marker, extra in extras:
        content = content.replace(f" · {marker}</div>", "</div>" + extra, 1)
    # Put the common video above its three analysis columns within each card.
    # Existing gallery scripts keep the badge/time cursor and lazy video loader.
    content = content.replace("<body>", "<body data-server-no-reload>", 1)
    note = ("<p class='comparison-note'>같은 원본 에피소드의 저장된 분할을 비교합니다. DP 추론이나 재분할은 하지 않습니다. "
            "영상 재생·이동 시 각 타임라인이 함께 갱신됩니다. 색상 막대 또는 그래프를 클릭하면 해당 프레임으로 이동합니다. "
            "Task 내 episode 순번은 0부터 시작합니다. <a href='src/selection.json'>선택·빌드 기록</a></p>")
    content = content.replace("</h1>", "</h1>" + note, 1).replace("</head>", STYLE + "</head>", 1)
    # The generated columns precede the player in the original card markup;
    # moving them afterwards leaves one common player and equal-height rows.
    arrange = "<script>document.querySelectorAll('.comparison-columns').forEach(c=>c.closest('.card').append(c));</script>"
    content = content.replace("</body>", arrange + SCRIPT + "</body>", 1)
    report.write_text(content)
    (assets / "selection.json").write_text(json.dumps({**plan, "results": records}, ensure_ascii=False, indent=2))
    print(f"HTML: {report}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FT_DIR / "newtask_ft_common_config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all-episodes", action="store_true", help="Browse all episodes without changing the FT selection")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    plan, reference, builds, groups = selection(args.config.resolve(), all_episodes=args.all_episodes)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.dry_run:
        render(plan, reference, builds, groups)


if __name__ == "__main__":
    main()
