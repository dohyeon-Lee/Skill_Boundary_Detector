"""ABC video/skill-timeline report with one HTML entry and private report assets."""
from __future__ import annotations

import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import numpy as np


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        self.references.extend(value for key, value in attrs
                               if key in {"href", "src", "data-src", "poster"} and value)


def prepare_report_layout(settings):
    """Move an older report's generated files under src, preserving local links."""
    if not settings.get("report_root"):
        return
    root = Path(settings["report_root"])
    assets = Path(settings["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    moves = []
    for item in list(root.iterdir()):
        if item == assets:
            continue
        if item.name == "index.html" and 'name="abc-sbd-video-report"' in item.read_text():
            continue
        destination = assets / item.name
        if destination.exists():
            raise ValueError(f"Report layout conflict: {destination}")
        moves.append((item, destination))
    original_pages = []
    for source, destination in moves:
        pages = source.rglob("*.html") if source.is_dir() else [source]
        for page in pages:
            if page.suffix == ".html":
                target = destination / page.relative_to(source) if source.is_dir() else destination
                original_pages.append((page, target, page.read_text()))
    for source, destination in moves:
        source.rename(destination)
    for old, new, content in original_pages:
        parser = _Links()
        parser.feed(content)
        for reference in set(parser.references):
            url = urlsplit(reference)
            if url.scheme not in {"", "file"} or not url.path:
                continue
            target = Path(unquote(url.path))
            if not target.is_absolute():
                target = Path(os.path.normpath(old.parent / target))
            for source, destination in moves:
                try:
                    target = destination / target.relative_to(source)
                    break
                except ValueError:
                    pass
            replacement = quote(os.path.relpath(target, new.parent), safe="/.:@-_")
            if url.query:
                replacement += "?" + url.query
            if url.fragment:
                replacement += "#" + url.fragment
            content = content.replace(html.escape(reference, quote=False), html.escape(replacement, quote=False))
        new.write_text(content)


def render_video_report(settings, episode_rows, task_groups, manifest):
    from codebook_visualizer import _episode_row, _load_episodes_meta
    from dp_skillset_eval import _episode_video_metadata
    from skillset_boundary_viz import render_skillset_card, save_gallery

    root = Path(settings["report_root"])
    assets = Path(settings["output_dir"])
    dataset = Path(manifest["dataset_dir"])
    metadata = _load_episodes_meta(dataset)
    info = json.loads((dataset / "meta/info.json").read_text())
    fps = float(info.get("fps", 30))
    policy = json.loads((Path(manifest["policy_path"]) / "config.json").read_text())
    boundaries = {item["name"]: item for item in settings["boundary_variants"]}
    recipes = {item["name"]: item for item in settings["experiments"]}
    cards, exports, videos = [], [], set()
    rendered_episodes = 0
    total_episodes = sum(len(episodes) for episodes in task_groups.values())
    for task_id, episodes in task_groups.items():
        for episode in episodes:
            row = _episode_row(metadata, episode)
            task_names = dict(zip(map(int, row.get("task_indices", [])), map(str, row["tasks"])))
            label = f"task{task_id}: {task_names.get(task_id, ' / '.join(map(str, row['tasks'])))}"
            video = _episode_video_metadata(
                dataset, metadata, settings["report_image_key"], episode, root, fps,
                # DROID conversion already stores one MP4 per episode/camera.
                clip_cache_dir=None if manifest.get("evaluation_source") == "droid_subset" else assets / "videos",
            )
            video_path = (root / unquote(video["video_src"])).resolve()
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            videos.add(str(video_path))
            for name, derived in episode_rows[episode]:
                recipe = boundaries[recipes[name]["boundary"]]
                n_frames = int(np.asarray(derived["n_frames"]).item())
                cuts = np.asarray(derived["boundaries"], dtype=np.int64).tolist()
                edges = [0, *cuts, n_frames]
                skills = [(start, end, None) for start, end in zip(edges[:-1], edges[1:], strict=True)]
                ts = np.asarray(derived["replan_ts"])
                smooth = np.asarray(derived["smooth"])
                curve = dict(
                    replan_ts=ts, div_cos=derived["score"], sg_vals=smooth,
                    mean_val=float(np.asarray(derived["threshold"]).item()),
                    score_power=float(recipe.get("score_power", 1)),
                    score_metric=recipes[name]["metric"], peak_ts=np.asarray(cuts),
                    peak_vals=np.asarray([smooth[int(np.argmin(np.abs(ts - cut)))] for cut in cuts]),
                    n_frames=n_frames, terminal_mask_frames=int(recipe.get("terminal_mask_frames", 0)),
                    initial_mask_frames=int(recipe.get("initial_mask_frames", 0)),
                    initial_mask_until_frame=int(recipe.get("initial_mask_frames", 0)),
                    curvature_rejected_ts=np.asarray(derived.get("curvature_rejected_ts", [])),
                    curvature_rejected_values=np.asarray(derived.get("curvature_rejected_values", [])),
                    peak_curvature_threshold=float(np.asarray(derived.get("peak_curvature_threshold", 0)).item()),
                    peak_curvature_scale=float(recipe.get("peak_curvature_scale", 0)),
                )
                terminal = float(np.asarray(derived.get("semantic_terminal_boundary", np.nan)).item())
                if np.isfinite(terminal):
                    curve["terminal_mask_start_frame"] = int(terminal)
                comparison = derived.get("final_comparison")
                if comparison is not None:
                    curve["final_comparison"] = {
                        **curve, "div_cos": comparison["score"], "sg_vals": comparison["smooth"],
                        "mean_val": float(np.asarray(comparison["threshold"]).item()),
                        "peak_ts": np.asarray([], dtype=np.int64), "peak_vals": np.asarray([]),
                        "curvature_rejected_ts": np.asarray([]), "curvature_rejected_values": np.asarray([]),
                    }
                media = render_skillset_card(skills, None, curve, episode_video=video, show_frames=False)
                caption = f"task{task_id} · episode {episode} · {name} — {len(skills)} skills · {n_frames} frames"
                if comparison is not None:
                    caption += " · step 7 x0-hat: SBD / final 10: comparison only (shared step-7 boundaries)"
                no_valid_anchors = bool(derived.get("no_valid_boundary_anchors", False))
                if no_valid_anchors:
                    caption += " · 마스크 적용 후 유효 계산 지점 없음 (평균·곡률 계산에서 제외)"
                if recipe.get("peak_curvature_scale", 0):
                    caption += f" · curvature mean×{recipe['peak_curvature_scale']:g}, cutoff={curve['peak_curvature_threshold']:.4g}, rejected={len(curve['curvature_rejected_ts'])}"
                cards.append((label, caption, media))
                exports.append(dict(episode_id=episode, task_id=task_id, experiment=name,
                                    no_valid_boundary_anchors=no_valid_anchors,
                                    n_frames=n_frames, boundaries=cuts, skills=[dict(start=a, end=b) for a, b, _ in skills],
                                    curvature_scale=curve['peak_curvature_scale'],
                                    curvature_threshold=curve['peak_curvature_threshold'],
                                    curvature_peaks=[dict(frame=int(t), curvature=float(c)) for t, c in zip(
                                        derived.get('curvature_peak_ts', []), derived.get('peak_curvatures', []), strict=True)],
                                    curvature_rejected_peaks=curve['curvature_rejected_ts'].astype(int).tolist()))
            rendered_episodes += 1
            if rendered_episodes % 5 == 0 or rendered_episodes == total_episodes:
                print(f"[SBD video] episodes={rendered_episodes}/{total_episodes}", flush=True)
        print(f"[video report] task {task_id}: {len(episodes)} episodes", flush=True)
    domain = {"libero_subset": "LIBERO", "mimicgen_subset": "MimicGen", "droid_subset": "DROID"}.get(manifest.get("evaluation_source"), "ABC")
    task_label = settings.get("target_task", "") if domain != "ABC" else ', '.join(manifest['source_tasks'])
    title = f"{domain} {task_label} · obs{policy['n_obs_steps']} / future{policy['horizon']} · NMS "
    title += ", ".join(str(item.get("nms_frames", 0)) for item in boundaries.values())
    save_gallery(root, title, cards)
    report = root / "index.html"
    content = report.read_text()
    content = content.replace("<head>", '<head><meta name="abc-sbd-video-report" content="1">', 1)
    # Live Preview watches the workspace; unrelated edits must not interrupt playback.
    content = content.replace("<body>", "<body data-server-no-reload>", 1)
    statistics = quote(os.path.relpath(assets / "index.html", root), safe="/")
    content = content.replace("</h1>", f'</h1><p><a href="{statistics}">통계·실험 비교 리포트</a></p>', 1)
    report.write_text(content)
    (assets / "boundary_results.json").write_text(json.dumps({"episodes": exports}, indent=2) + "\n")
    (assets / "video_references.json").write_text(json.dumps(sorted(videos), indent=2) + "\n")
    (assets / "video_report_manifest.json").write_text(json.dumps({
        "report": "index.html", "episodes": len(episode_rows), "players": len(cards),
        "tasks": len(task_groups), "source_videos": len(videos), "reencoded": False,
    }, indent=2) + "\n")
    return report
