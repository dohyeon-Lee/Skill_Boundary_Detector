"""Shared rendering for the DP skill-boundary eval.

Per-episode card = boxed start/end frames for each skill (tight gap within a skill,
wide gap + coloured box between skills), followed by a timestep-scaled SVG timeline
containing the available SBD metrics and gripper signals.

Data-source agnostic on purpose: callers pass a per-episode ``skills`` list
``[(frame_start, frame_end, label_or_None), ...]``, the raw episode frames, and an
optional curve dict. Used by ``dp_skillset_eval`` for the
``train_skills/skill_eval/skill_boundary_detection`` visualizations (skills from
the FSQ_dataset skillset npz).

Requires ``train_skills/skill_eval/src`` on sys.path (for codebook_visualizer).
"""

from __future__ import annotations

import base64
import html as html_lib
import io
import json
from pathlib import Path

import numpy as np

from codebook_visualizer import _clip_frame_or_blank


def build_gmm_visualizations(
    raw_dir: str | Path | list[str | Path] | tuple[str | Path, ...],
    episode_ids,
    *,
    components: int = 5,
    max_projection_samples: int = 100_000,
) -> dict[int, dict]:
    """Build compact, video-synchronised 3-D views of cached SBD probe clouds.

    The GMM remains fitted in the complete descriptor space.  For display, every
    anchor cloud is centred independently before fitting one report-wide PCA.  This
    removes bulk trajectory translation and keeps the camera on the distribution so
    within-cloud mode separation is visible.  The display scale remains fixed across
    the report, and component identities are matched between adjacent anchors.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.mixture import GaussianMixture

    roots = (
        [Path(value) for value in raw_dir]
        if isinstance(raw_dir, (list, tuple))
        else [Path(raw_dir)]
    )
    loaded: dict[int, dict[str, np.ndarray]] = {}
    projection_rows = []
    for episode_id in episode_ids:
        filename = f"ep{int(episode_id):07d}.npz"
        path = next((root / filename for root in roots if (root / filename).is_file()), None)
        if path is None:
            continue
        with np.load(path, allow_pickle=False) as data:
            output = np.asarray(data["descriptors"], dtype=np.float32)
            if output.ndim != 3 or output.shape[1] < components:
                continue
            payload = {
                "replan_ts": np.asarray(data["replan_ts"], dtype=np.int64),
                "output": output,
            }
            if "input_descriptors" in data:
                before = np.asarray(data["input_descriptors"], dtype=np.float32)
                if before.shape == output.shape:
                    payload["input"] = before
            loaded[int(episode_id)] = payload
            centred_output = output - output.mean(axis=1, keepdims=True)
            projection_rows.append(centred_output.reshape(-1, output.shape[-1]))
    if not projection_rows:
        return {}

    projection_data = np.concatenate(projection_rows, axis=0).astype(np.float64)
    if len(projection_data) > max_projection_samples:
        take = np.linspace(
            0, len(projection_data) - 1, max_projection_samples, dtype=np.int64
        )
        projection_data = projection_data[take]
    projection_mean = projection_data.mean(axis=0)
    _u, _s, vh = np.linalg.svd(
        projection_data - projection_mean[None], full_matrices=False
    )
    projection = vh[:3]

    projected_payloads: dict[int, dict] = {}
    all_display_values = []
    for episode_id, payload in loaded.items():
        output = payload["output"].astype(np.float64)
        output_centres = output.mean(axis=1, keepdims=True)
        output_centred = output - output_centres
        output_3d = (output_centred - projection_mean[None, None]) @ projection.T
        before = payload.get("input")
        input_3d = (
            (
                before.astype(np.float64)
                - before.astype(np.float64).mean(axis=1, keepdims=True)
                - projection_mean[None, None]
            )
            @ projection.T
            if before is not None
            else None
        )
        labels, centers, counts = [], [], []
        previous_means = None
        for anchor_index, cloud in enumerate(output):
            model = GaussianMixture(
                n_components=int(components),
                covariance_type="diag",
                n_init=10,
                random_state=0,
                max_iter=300,
                reg_covar=1e-6,
            ).fit(cloud)
            raw_labels = model.predict(cloud)
            current_means = np.asarray(model.means_, dtype=np.float64)
            cloud_mean = cloud.mean(axis=0, keepdims=True)
            current_relative_means = current_means - cloud_mean
            if previous_means is None:
                center_3d = (
                    current_relative_means - projection_mean[None]
                ) @ projection.T
                order = np.argsort(center_3d[:, 0], kind="stable")
                current_to_stable = np.empty(components, dtype=np.int64)
                current_to_stable[order] = np.arange(components)
            else:
                row, col = linear_sum_assignment(
                    np.linalg.norm(
                        previous_means[:, None, :]
                        - current_relative_means[None, :, :],
                        axis=-1,
                    )
                )
                current_to_stable = np.empty(components, dtype=np.int64)
                current_to_stable[col] = row
            stable_means = np.empty_like(current_relative_means)
            for current_index, stable_index in enumerate(current_to_stable):
                stable_means[stable_index] = current_relative_means[current_index]
            stable_labels = current_to_stable[raw_labels]
            previous_means = stable_means
            labels.append(stable_labels.astype(int).tolist())
            centers.append(
                np.round((stable_means - projection_mean[None]) @ projection.T, 5).tolist()
            )
            counts.append(
                np.bincount(stable_labels, minlength=components).astype(int).tolist()
            )
        all_display_values.append(output_3d.reshape(-1, 3))
        if input_3d is not None:
            all_display_values.append(input_3d.reshape(-1, 3))
        projected_payloads[episode_id] = {
            "replanTs": payload["replan_ts"].astype(int).tolist(),
            "output": np.round(output_3d, 5).tolist(),
            "input": None if input_3d is None else np.round(input_3d, 5).tolist(),
            "labels": labels,
            "centers": centers,
            "counts": counts,
            "descriptorDim": int(output.shape[-1]),
            "components": int(components),
            "centering": "per_frame",
        }

    display = np.concatenate(all_display_values, axis=0)
    radius = float(np.percentile(np.linalg.norm(display, axis=1), 98.0))
    radius = max(radius, 1e-6)
    for payload in projected_payloads.values():
        payload["displayRadius"] = round(radius, 6)
    return projected_payloads


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88})
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_to_svg_b64(fig) -> str:
    """Serialize a timeline as SVG so long ABC episodes are not JPEG-size limited."""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _frame_to_b64(frame: np.ndarray, size: int, quality: int = 82) -> str:
    from PIL import Image

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize((size, size), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _skill_items_html(items: list[dict]) -> str:
    blocks = ["<div class='skill-grid'>"]
    for item in items:
        label = f"sk{int(item['index'])}"
        if item.get("label") is not None:
            label += f" · tok{item['label']}"
        size = int(item["thumb_size"])
        show_frames = bool(item.get("show_frames", True))
        start_jpeg = item.get("start_jpeg", "")
        end_jpeg = item.get("end_jpeg", "")
        src = html_lib.escape(str(item["video_src"]), quote=True)
        start_sec = float(item["start_sec"])
        end_sec = float(item["end_sec"])
        start_html = (
            "<figure>"
            f"<figcaption>start f{int(item['frame_start'])}</figcaption>"
            f"<img style='width:{size}px;height:{size}px' src='data:image/jpeg;base64,{start_jpeg}'>"
            "</figure>"
            if show_frames
            else ""
        )
        end_html = (
            "<figure>"
            f"<figcaption>end →f{int(item['frame_end'])}</figcaption>"
            f"<img style='width:{size}px;height:{size}px' src='data:image/jpeg;base64,{end_jpeg}'>"
            "</figure>"
            if show_frames
            else ""
        )
        poster = f" poster='data:image/jpeg;base64,{start_jpeg}'" if show_frames else ""
        blocks.append(
            "<div class='skill-item'>"
            f"<div class='skill-label'>{html_lib.escape(label)}</div>"
            "<div class='skill-triplet'>"
            f"{start_html}"
            "<div class='clip-slot'>"
            "<div class='clip-caption'>skill clip</div>"
            f"<video style='width:{size}px;height:{size}px' preload='none' muted playsinline controls "
            f"{poster} data-src='{src}' "
            f"data-start='{start_sec:.6f}' data-end='{end_sec:.6f}'></video>"
            "<button class='clip-play' type='button'>▶ load &amp; play</button>"
            "</div>"
            f"{end_html}"
            "</div></div>"
        )
    blocks.append("</div>")
    return "".join(blocks)


def _episode_video_html(video: dict, skills) -> str:
    """Interactive full-episode player whose badge/timeline follows skill cuts."""
    src = html_lib.escape(str(video["video_src"]), quote=True)
    fps = float(video["fps"])
    episode_start = float(video["episode_start_sec"])
    episode_end = float(video["episode_end_sec"])
    aspect_ratio = float(video.get("aspect_ratio", 4.0 / 3.0))
    object_position_y = float(video.get("object_position_y", 0.0))
    n_frames = max(1, int(skills[-1][1]) if skills else int(round((episode_end - episode_start) * fps)))
    palette = ("#2563eb", "#7c3aed", "#db2777", "#ea580c", "#16a34a", "#0891b2")
    payload = []
    segments = []
    for index, (frame_start, frame_end, label) in enumerate(skills):
        start = int(frame_start)
        end = int(frame_end)
        color = palette[index % len(palette)]
        item = {"index": index, "start": start, "end": end, "label": label}
        payload.append(item)
        width = max(0.0, 100.0 * (end - start) / n_frames)
        label_text = f"sk{index}" + (f" · tok{label}" if label is not None else "")
        segments.append(
            "<button class='skill-segment' type='button' "
            f"data-skill-index='{index}' data-frame-start='{start}' "
            f"style='flex-basis:{width:.6f}%;background:{color}' "
            f"title='{html_lib.escape(label_text)} · frames {start}–{end}'>"
            f"<span>{html_lib.escape(f'sk{index}')}</span></button>"
        )
    skills_json = html_lib.escape(json.dumps(payload, separators=(",", ":")), quote=True)
    return (
        "<div class='episode-player' "
        f"data-skills='{skills_json}' data-fps='{fps:.9g}' "
        f"data-episode-start='{episode_start:.9g}' data-episode-end='{episode_end:.9g}' "
        f"data-n-frames='{n_frames}'>"
        f"<div class='video-focus' style='--video-aspect:{aspect_ratio:.9g}'>"
        f"<div class='video-stage' style='aspect-ratio:{aspect_ratio:.9g}'>"
        "<video class='episode-video' preload='none' muted playsinline controls "
        f"data-src='{src}' style='object-position:center {object_position_y:.6f}%'></video>"
        "<div class='skill-badge'><strong>Skill 0</strong><span>waiting to load</span></div>"
        "<div class='boundary-flash'>SKILL CHANGE</div>"
        "<button class='video-expand' type='button' title='show only this video fullscreen'>"
        "⛶ 영상 크게 보기</button>"
        "<button class='episode-load' type='button'>▶ load &amp; play episode</button>"
        "</div></div>"
        "<div class='skill-track'>" + "".join(segments) + "<div class='skill-playhead'></div></div>"
        "<div class='player-status'><span class='frame-status'>frame 0</span>"
        "<span>click a coloured skill segment to seek</span></div>"
        "</div>"
    )


def _gmm_visualization_html(payload: dict) -> str:
    serialised = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    serialised = serialised.replace("</", "<\\/")
    before = "denoising 전·후" if payload.get("input") is not None else "denoising 후"
    return (
        "<div class='gmm-view'>"
        "<div class='gmm-heading'><strong>5-GMM action modes</strong>"
        f"<span>{html_lib.escape(before)} · frame-centered PC1–PC3</span></div>"
        "<canvas class='gmm-canvas' width='640' height='520'></canvas>"
        "<div class='gmm-status'>waiting for video</div>"
        "<div class='gmm-help'>drag: rotate · wheel: zoom · double click: reset</div>"
        f"<script type='application/json' class='gmm-data'>{serialised}</script>"
        "</div>"
    )


def save_gallery(out_dir: Path, title: str, cards: list[tuple], filename: str = "index.html") -> None:
    """Write an HTML gallery to out_dir/filename (default index.html).

    cards: [(task_label, caption, jpeg_b64_or_media_dict), ...]. When task_label is not None, a
    sticky section header is inserted whenever it changes, so episodes are grouped
    and visually separated per task. (task_label None → flat gallery as before.)
    filename: output HTML name within out_dir — set to e.g. '<dp>_ck<ckpt>.html' to keep multiple
    runs side by side in one shallow folder instead of overwriting index.html."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks, prev = [], "\0"
    for task_label, cap, media in cards:
        if task_label is not None and task_label != prev:
            if isinstance(task_label, (int, np.integer)):
                section_title = f"task{int(task_label):02d}"
            else:
                section_title = str(task_label)
            blocks.append(f"<h2 class='task'>{html_lib.escape(section_title)}</h2>")
        prev = task_label
        if isinstance(media, dict):
            if media.get("episode_video") is not None:
                images = _episode_video_html(media["episode_video"], media.get("skills", []))
            elif media.get("skill_items") is not None:
                images = _skill_items_html(media["skill_items"])
            elif media.get("frames_jpeg"):
                images = (
                    f"<img class='frames' src='data:image/jpeg;base64,{media['frames_jpeg']}'>"
                )
            else:
                images = ""
            if media.get("gmm_visualization") is not None:
                images = (
                    "<div class='episode-analysis-row'>" + images
                    + _gmm_visualization_html(media["gmm_visualization"])
                    + "</div>"
                )
            if media.get("timeline_svg"):
                timeline_n_frames = max(1, int(media.get("timeline_n_frames", 1)))
                timeline_plot_left = float(media.get("timeline_plot_left", 0.0))
                timeline_plot_right = float(media.get("timeline_plot_right", 1.0))
                timeline_plot_top = float(media.get("timeline_plot_top", 0.0))
                timeline_plot_bottom = float(media.get("timeline_plot_bottom", 0.0))
                images += (
                    "<div class='timeline-scroll'>"
                    "<div class='timeline-canvas' "
                    f"data-n-frames='{timeline_n_frames}' "
                    f"data-plot-left='{timeline_plot_left:.9g}' "
                    f"data-plot-right='{timeline_plot_right:.9g}' "
                    f"style='--plot-top:{100.0 * timeline_plot_top:.6f}%;"
                    f"--plot-bottom:{100.0 * timeline_plot_bottom:.6f}%'>"
                    f"<img class='timeline' src='data:image/svg+xml;base64,{media['timeline_svg']}'>"
                    "<div class='timeline-playhead' aria-hidden='true'></div>"
                    "</div></div>"
                )
        else:
            images = f"<img class='frames' src='data:image/jpeg;base64,{media}'>"
        blocks.append(
            f"<div class='card'><div class='cap'>{html_lib.escape(str(cap))}</div>"
            f"{images}</div>"
        )
    body = "".join(blocks)
    html = (
        "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>" + title + "</title><style>"
        "body{font-family:sans-serif;background:#f5f5f5;margin:0;padding:12px;}"
        "h1{font-size:18px;}"
        "h2.task{font-size:15px;color:#fff;background:#3367d6;margin:18px 0 4px;"
        "padding:5px 10px;border-radius:4px;position:sticky;top:0;z-index:2;}"
        ".grid{display:flex;flex-direction:column;gap:14px;}"
        ".card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;overflow-x:auto;}"
        ".cap{font-size:12px;color:#555;margin-bottom:4px;}"
        ".card img{display:block;max-width:none;}"
        ".card img.frames{margin-left:auto;margin-right:auto;}"
        ".skill-grid{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start;justify-content:center;}"
        ".skill-item{border:2px solid #2f6fd0;border-radius:5px;padding:5px;background:#fff;}"
        ".skill-label{font-size:11px;color:#2f6fd0;font-weight:600;margin-bottom:3px;}"
        ".skill-triplet{display:flex;gap:4px;align-items:flex-end;}"
        ".skill-triplet figure{margin:0;text-align:center;}"
        ".skill-triplet figcaption,.clip-caption{font-size:9px;color:#555;text-align:center;height:14px;}"
        ".skill-triplet img,.skill-triplet video{object-fit:cover;background:#333;}"
        ".clip-slot{position:relative;}"
        ".clip-play{position:absolute;left:50%;top:55%;transform:translate(-50%,-50%);"
        "font-size:10px;padding:5px 7px;border:0;border-radius:4px;background:rgba(0,0,0,.72);color:#fff;cursor:pointer;}"
        ".clip-slot.loaded .clip-play{display:none;}"
        ".episode-player{width:min(100%,760px);margin-left:auto;margin-right:auto;display:flex;"
        "flex-direction:column;align-items:center;}"
        ".episode-analysis-row{display:flex;flex-wrap:wrap;justify-content:center;align-items:flex-start;"
        "gap:18px;width:100%;}"
        ".episode-analysis-row .episode-player{width:min(100%,680px);margin:0;}"
        ".gmm-view{width:min(100%,680px);background:#0b1020;border:1px solid #26324d;border-radius:8px;"
        "padding:8px;color:#e5e7eb;box-sizing:border-box;}"
        ".gmm-heading{display:flex;justify-content:space-between;gap:8px;align-items:baseline;"
        "font-size:12px;margin:0 3px 5px}.gmm-heading span{font-size:10px;color:#94a3b8;}"
        ".gmm-canvas{display:block;width:100%;aspect-ratio:16/13;background:#080d19;border-radius:5px;"
        "cursor:grab;touch-action:none}.gmm-canvas:active{cursor:grabbing;}"
        ".gmm-status{font-size:10px;color:#cbd5e1;margin:5px 3px 0;font-variant-numeric:tabular-nums;}"
        ".gmm-help{font-size:9px;color:#64748b;margin:2px 3px 0;}"
        ".video-focus{width:min(100%,640px);background:#111;border-radius:8px;}"
        ".video-stage{width:min(100%,640px);position:relative;overflow:hidden;background:#111;border-radius:8px;}"
        ".episode-video{display:block;width:100%;height:100%;object-fit:cover;background:#111;}"
        ".skill-badge{position:absolute;left:10px;top:10px;display:flex;flex-direction:column;gap:2px;"
        "padding:7px 10px;border-radius:7px;background:rgba(0,0,0,.72);color:#fff;pointer-events:none;"
        "font-variant-numeric:tabular-nums;box-shadow:0 1px 4px rgba(0,0,0,.35);}"
        ".skill-badge strong{font-size:18px;line-height:1.05}.skill-badge span{font-size:10px;color:#dbeafe;}"
        ".boundary-flash{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;"
        "font-size:22px;font-weight:800;letter-spacing:.08em;color:#fff;background:rgba(37,99,235,.30);"
        "opacity:0;pointer-events:none;transition:opacity .16s ease;}"
        ".episode-player.skill-change .boundary-flash{opacity:1;}"
        ".episode-load{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);border:0;"
        "border-radius:7px;padding:10px 14px;background:rgba(0,0,0,.78);color:#fff;font-size:13px;cursor:pointer;}"
        ".episode-player.loaded .episode-load{display:none;}"
        ".video-expand{position:absolute;right:8px;top:8px;z-index:5;border:1px solid rgba(255,255,255,.45);"
        "border-radius:5px;padding:5px 8px;background:rgba(0,0,0,.68);color:#fff;font-size:10px;cursor:pointer;}"
        ".video-focus:fullscreen{width:100%;height:100%;max-width:none;border-radius:0;display:flex;"
        "align-items:center;justify-content:center;background:#000;padding:0;}"
        ".video-focus:fullscreen .video-stage{width:min(100vw,calc(100vh * var(--video-aspect)));"
        "max-width:none;border-radius:0;}"
        ".video-focus:fullscreen .skill-badge{left:18px;top:18px;transform:scale(1.25);transform-origin:top left;}"
        ".video-focus:fullscreen .video-expand{right:18px;top:18px;font-size:13px;padding:7px 10px;}"
        ".skill-track{width:min(100%,640px);height:30px;margin-top:7px;display:flex;position:relative;"
        "overflow:hidden;border-radius:5px;background:#d1d5db;box-shadow:inset 0 0 0 1px rgba(0,0,0,.16);}"
        ".skill-segment{min-width:2px;height:100%;padding:0;border:0;border-right:1px solid rgba(255,255,255,.7);"
        "color:#fff;font-size:10px;font-weight:700;overflow:hidden;cursor:pointer;opacity:.72;}"
        ".skill-segment.active{opacity:1;box-shadow:inset 0 0 0 3px #111;}"
        ".skill-segment span{padding:0 4px;white-space:nowrap;text-shadow:0 1px 2px #000;}"
        ".skill-playhead{position:absolute;top:0;bottom:0;left:0;width:2px;background:#fff;"
        "box-shadow:0 0 0 1px #111;pointer-events:none;z-index:3;}"
        ".player-status{width:min(100%,640px);display:flex;justify-content:space-between;color:#666;"
        "font-size:10px;margin-top:3px;font-variant-numeric:tabular-nums;}"
        ".timeline-scroll{width:100%;overflow-x:auto;margin-top:10px;padding-top:4px;"
        "border-top:1px solid #eee;display:flex;justify-content:safe center;}"
        ".timeline-canvas{position:relative;flex:0 0 auto;line-height:0;}"
        ".timeline-playhead{position:absolute;top:var(--plot-top,3%);bottom:var(--plot-bottom,12%);"
        "left:0;width:2px;transform:translateX(-1px);background:#ef4444;"
        "box-shadow:0 0 0 1px rgba(255,255,255,.9),0 0 5px rgba(0,0,0,.65);"
        "pointer-events:none;z-index:4;opacity:0;transition:opacity .15s ease;}"
        ".card.timeline-synced .timeline-playhead{opacity:.95;}"
        "</style></head><body><h1>" + title + "</h1><div class='grid'>" + body + "</div>"
        "<script>"
        "const skillVideos=[...document.querySelectorAll('.clip-slot video')];"
        "async function playSkill(slot){"
        "const video=slot.querySelector('video');"
        "skillVideos.forEach(v=>{if(v!==video)v.pause();});"
        "if(!video.getAttribute('src')){"
        "video.src=video.dataset.src;video.load();"
        "await new Promise((resolve,reject)=>{"
        "video.addEventListener('loadedmetadata',resolve,{once:true});"
        "video.addEventListener('error',reject,{once:true});});}"
        "video.currentTime=Number(video.dataset.start);slot.classList.add('loaded');"
        "try{await video.play();}catch(err){console.warn('skill clip play failed',err);}}"
        "document.querySelectorAll('.clip-play').forEach(button=>button.addEventListener('click',()=>playSkill(button.closest('.clip-slot'))));"
        "skillVideos.forEach(video=>{video.addEventListener('timeupdate',()=>{"
        "const end=Number(video.dataset.end),start=Number(video.dataset.start);"
        "if(video.currentTime>=end){video.currentTime=start;if(!video.paused)video.play();}});"
        "video.addEventListener('play',()=>{if(video.currentTime<Number(video.dataset.start)||video.currentTime>=Number(video.dataset.end))"
        "video.currentTime=Number(video.dataset.start);});});"
        "const episodePlayers=[...document.querySelectorAll('.episode-player')];"
        "const gmmColors=['#38bdf8','#a78bfa','#fb7185','#fbbf24','#34d399'];"
        "function rotateGmmPoint(point,state){const x=point[0]/state.radius,y=point[1]/state.radius,z=point[2]/state.radius;"
        "const cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);"
        "const x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1,z2=sp*y+cp*z1;return [x1,y1,z2];}"
        "function drawGmm(view,index){const data=view._gmmData,state=view._gmmState;if(!data||!data.output.length)return;"
        "index=Math.max(0,Math.min(data.output.length-1,index));view._gmmIndex=index;"
        "const canvas=view.querySelector('.gmm-canvas'),ctx=canvas.getContext('2d'),dpr=window.devicePixelRatio||1;"
        "const rect=canvas.getBoundingClientRect(),w=Math.max(320,rect.width),h=w*13/16;"
        "if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}"
        "ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#080d19';ctx.fillRect(0,0,w,h);"
        "const project=p=>{const q=rotateGmmPoint(p,state),persp=1/(1+.20*q[2]);return [w/2+q[0]*w*.34*state.zoom*persp,h/2-q[1]*h*.38*state.zoom*persp,q[2]];};"
        "ctx.lineWidth=1;[['#ef4444',[1.15,0,0]],['#22c55e',[0,1.15,0]],['#3b82f6',[0,0,1.15]]].forEach(a=>{const o=project([0,0,0]),e=project(a[1].map(v=>v*state.radius));ctx.strokeStyle=a[0]+'88';ctx.beginPath();ctx.moveTo(o[0],o[1]);ctx.lineTo(e[0],e[1]);ctx.stroke();});"
        "const output=data.output[index],input=data.input?data.input[index]:null,labels=data.labels[index];"
        "if(input){ctx.lineWidth=.6;for(let i=0;i<Math.min(input.length,output.length);i++){const a=project(input[i]),b=project(output[i]);ctx.strokeStyle='rgba(148,163,184,.16)';ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}"
        "for(const p of input){const q=project(p);ctx.fillStyle='rgba(203,213,225,.26)';ctx.beginPath();ctx.arc(q[0],q[1],2,0,Math.PI*2);ctx.fill();}}"
        "const order=output.map((p,i)=>({p:project(p),i})).sort((a,b)=>a.p[2]-b.p[2]);"
        "for(const item of order){const c=gmmColors[labels[item.i]%gmmColors.length],r=3.2+Math.max(-1,Math.min(1,item.p[2]))*.7;ctx.fillStyle=c+'d9';ctx.beginPath();ctx.arc(item.p[0],item.p[1],r,0,Math.PI*2);ctx.fill();}"
        "data.centers[index].forEach((p,i)=>{const q=project(p);ctx.fillStyle=gmmColors[i%gmmColors.length];ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(q[0],q[1],7,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#fff';ctx.font='bold 10px sans-serif';ctx.fillText('G'+i,q[0]+9,q[1]-7);});"
        "view.querySelector('.gmm-status').textContent=`anchor frame ${data.replanTs[index]} · centered ${data.descriptorDim}D→3D · `+data.counts[index].map((n,i)=>`G${i}:${n}`).join('  '); }"
        "function initGmm(view){if(view._gmmData)return;view._gmmData=JSON.parse(view.querySelector('.gmm-data').textContent);"
        "view._gmmState={yaw:-.55,pitch:.28,zoom:1,radius:Number(view._gmmData.displayRadius)||1};const canvas=view.querySelector('.gmm-canvas');"
        "let drag=false,lastX=0,lastY=0;canvas.addEventListener('pointerdown',e=>{drag=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);});"
        "canvas.addEventListener('pointermove',e=>{if(!drag)return;view._gmmState.yaw+=(e.clientX-lastX)*.01;view._gmmState.pitch=Math.max(-1.45,Math.min(1.45,view._gmmState.pitch+(e.clientY-lastY)*.01));lastX=e.clientX;lastY=e.clientY;drawGmm(view,view._gmmIndex||0);});"
        "canvas.addEventListener('pointerup',()=>drag=false);canvas.addEventListener('pointercancel',()=>drag=false);"
        "canvas.addEventListener('wheel',e=>{e.preventDefault();view._gmmState.zoom=Math.max(.35,Math.min(4,view._gmmState.zoom*Math.exp(-e.deltaY*.001)));drawGmm(view,view._gmmIndex||0);},{passive:false});"
        "canvas.addEventListener('dblclick',()=>{view._gmmState.yaw=-.55;view._gmmState.pitch=.28;view._gmmState.zoom=1;drawGmm(view,view._gmmIndex||0);});drawGmm(view,0);}"
        "function updateGmmForFrame(card,frame){const view=card?card.querySelector('.gmm-view'):null;if(!view)return;initGmm(view);"
        "const ts=view._gmmData.replanTs;let index=0;while(index+1<ts.length&&Math.abs(ts[index+1]-frame)<=Math.abs(ts[index]-frame))index++;if(index!==view._gmmIndex)drawGmm(view,index);}"
        "async function loadEpisode(player,autoplay=true){"
        "const video=player.querySelector('.episode-video');"
        "if(!video.getAttribute('src')){video.src=video.dataset.src;video.load();"
        "await new Promise((resolve,reject)=>{video.addEventListener('loadedmetadata',resolve,{once:true});"
        "video.addEventListener('error',reject,{once:true});});}"
        "player.classList.add('loaded');video.currentTime=Number(player.dataset.episodeStart);"
        "updateEpisodePlayer(player,true);if(autoplay){try{await video.play();}catch(err){console.warn('episode play failed',err);}}}"
        "function updateEpisodePlayer(player,force=false){"
        "const video=player.querySelector('.episode-video'),fps=Number(player.dataset.fps),"
        "start=Number(player.dataset.episodeStart),end=Number(player.dataset.episodeEnd),"
        "nFrames=Number(player.dataset.nFrames),skills=JSON.parse(player.dataset.skills);"
        "if(!Number.isFinite(video.currentTime))return;"
        "if(video.currentTime>=end){video.pause();video.currentTime=start;}"
        "const frame=Math.max(0,Math.min(nFrames-1,Math.floor((video.currentTime-start)*fps+1e-6)));"
        "let idx=skills.findIndex(skill=>frame>=skill.start&&frame<skill.end);if(idx<0)idx=Math.max(0,skills.length-1);"
        "const skill=skills[idx]||{index:0,start:0,end:nFrames,label:null};"
        "const previous=Number(player.dataset.activeSkill);"
        "if(force||previous!==idx){player.dataset.activeSkill=String(idx);"
        "player.querySelectorAll('.skill-segment').forEach((segment,i)=>segment.classList.toggle('active',i===idx));"
        "if(!force&&Number.isFinite(previous)){player.classList.remove('skill-change');void player.offsetWidth;"
        "player.classList.add('skill-change');setTimeout(()=>player.classList.remove('skill-change'),260);}}"
        "const badge=player.querySelector('.skill-badge');"
        "badge.querySelector('strong').textContent=`Skill ${skill.index}${skill.label===null?'':` · tok${skill.label}`}`;"
        "badge.querySelector('span').textContent=`frames ${skill.start}–${skill.end}`;"
        "player.querySelector('.skill-playhead').style.left=`${100*frame/Math.max(1,nFrames)}%`;"
        "const card=player.closest('.card'),timeline=card?card.querySelector('.timeline-canvas'):null;"
        "updateGmmForFrame(card,frame);"
        "if(timeline){const plotLeft=Number(timeline.dataset.plotLeft),plotRight=Number(timeline.dataset.plotRight),"
        "timelineFrames=Number(timeline.dataset.nFrames),ratio=Math.max(0,Math.min(1,frame/Math.max(1,timelineFrames)));"
        "const position=plotLeft+(plotRight-plotLeft)*ratio,playhead=timeline.querySelector('.timeline-playhead');"
        "playhead.style.left=`${100*position}%`;card.classList.add('timeline-synced');"
        "const scroller=timeline.closest('.timeline-scroll'),x=timeline.offsetLeft+timeline.offsetWidth*position;"
        "if(!video.paused&&scroller.scrollWidth>scroller.clientWidth){const visible=x-scroller.scrollLeft;"
        "if(visible>scroller.clientWidth*.82||visible<scroller.clientWidth*.18)"
        "scroller.scrollLeft=Math.max(0,x-scroller.clientWidth*.28);}}"
        "player.querySelector('.frame-status').textContent=`frame ${frame} / ${nFrames-1} · ${(frame/fps).toFixed(2)}s`; }"
        "function stopEpisodeAnimation(player){if(player._animationFrame){cancelAnimationFrame(player._animationFrame);"
        "player._animationFrame=null;}}"
        "function startEpisodeAnimation(player){stopEpisodeAnimation(player);const video=player.querySelector('.episode-video');"
        "const tick=()=>{updateEpisodePlayer(player);if(!video.paused&&!video.ended)player._animationFrame=requestAnimationFrame(tick);};tick();}"
        "async function toggleEpisodeFullscreen(player){const focus=player.querySelector('.video-focus');"
        "if(document.fullscreenElement===focus){await document.exitFullscreen();}else{await focus.requestFullscreen();}}"
        "function updateFullscreenButtons(){episodePlayers.forEach(player=>{const focus=player.querySelector('.video-focus');"
        "player.querySelector('.video-expand').textContent=document.fullscreenElement===focus?'↙ 원래 크기':'⛶ 영상 크게 보기';});}"
        "episodePlayers.forEach(player=>{const video=player.querySelector('.episode-video');"
        "player.querySelector('.episode-load').addEventListener('click',()=>loadEpisode(player,true));"
        "player.querySelector('.video-expand').addEventListener('click',()=>toggleEpisodeFullscreen(player));"
        "player.querySelectorAll('.skill-segment').forEach(segment=>segment.addEventListener('click',async()=>{"
        "await loadEpisode(player,false);video.currentTime=Number(player.dataset.episodeStart)+Number(segment.dataset.frameStart)/Number(player.dataset.fps);"
        "updateEpisodePlayer(player,true);}));"
        "video.addEventListener('timeupdate',()=>updateEpisodePlayer(player));"
        "video.addEventListener('seeking',()=>updateEpisodePlayer(player,true));"
        "video.addEventListener('play',()=>{episodePlayers.forEach(other=>{const v=other.querySelector('.episode-video');if(v!==video)v.pause();});"
        "const start=Number(player.dataset.episodeStart),end=Number(player.dataset.episodeEnd);"
        "if(video.currentTime<start||video.currentTime>=end)video.currentTime=start;startEpisodeAnimation(player);});"
        "video.addEventListener('pause',()=>stopEpisodeAnimation(player));"
        "video.addEventListener('ended',()=>stopEpisodeAnimation(player));});"
        "document.addEventListener('fullscreenchange',updateFullscreenButtons);"
        "</script></body></html>"
    )
    (out_dir / filename).write_text(html, encoding="utf-8")
    print(f"[eval] {title} → {out_dir / filename}")


def load_boundary_curve(curves_dir, ep: int) -> dict | None:
    """Load the per-episode multimodality curve npz written by build_skill_dataset
    (output_dir/curves/ep{ep:07d}.npz). Returns None if missing/unreadable so the
    skillset eval degrades to frames-only."""
    if not curves_dir:
        return None
    path = Path(curves_dir) / f"ep{int(ep):07d}.npz"
    if not path.is_file():
        return None
    try:
        z = np.load(str(path))
        return {k: z[k] for k in z.files}
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] boundary curve ep{ep}: {exc}")
        return None


def _skill_cut_frames(skills) -> list[int]:
    return [fs for fs, _fe, _lab in skills] + ([skills[-1][1]] if skills else [])


def _plot_skill_cuts(ax, skills, *, annotate: bool = False) -> None:
    cut_frames = _skill_cut_frames(skills)
    y_top = ax.get_ylim()[1]
    for k, frame in enumerate(cut_frames):
        ax.axvline(frame, color="#2f6fd0", linestyle=":", linewidth=1.0, alpha=0.85)
        if annotate and k < len(skills):
            ax.text(frame, y_top, f"sk{k}", fontsize=6, color="#2f6fd0", va="bottom", ha="left")


def _shade_excluded_tail(ax, curve: dict) -> None:
    if "n_frames" not in curve:
        return
    end = int(np.asarray(curve["n_frames"]).reshape(-1)[0])
    if "terminal_mask_start_frame" in curve:
        start = int(
            np.asarray(curve["terminal_mask_start_frame"]).reshape(-1)[0]
        )
        if start < end:
            ax.axvspan(
                max(0, start),
                end,
                color="#fca5a5",
                alpha=0.26,
                label="terminal motion excluded",
            )
        return
    if "last_valid_anchor" not in curve:
        return
    start = int(np.asarray(curve["last_valid_anchor"]).reshape(-1)[0]) + 1
    if start < end:
        ax.axvspan(start, end, color="0.75", alpha=0.25, label="incomplete future excluded")


def _shade_excluded_initial_motion(ax, curve: dict) -> None:
    if "initial_mask_until_frame" not in curve:
        return
    end = int(np.asarray(curve["initial_mask_until_frame"]).reshape(-1)[0])
    base_end = int(
        np.asarray(curve.get("initial_mask_base_end_frame", end)).reshape(-1)[0]
    )
    if base_end > 0:
        ax.axvspan(
            0,
            min(base_end, end),
            color="#93c5fd",
            alpha=0.22,
            label="initial motion excluded",
        )
    if end > base_end:
        ax.axvspan(
            base_end,
            end,
            color="#fdba74",
            alpha=0.28,
            label="reset recovery extension",
        )
    if "initial_motion_start_frame" in curve:
        start = int(np.asarray(curve["initial_motion_start_frame"]).reshape(-1)[0])
        ax.axvline(
            start,
            color="#0284c7",
            linestyle="-.",
            linewidth=0.9,
            alpha=0.9,
            label=f"motion start={start}",
        )


def _plot_raw_and_smooth(
    ax,
    curve: dict,
    *,
    raw_key: str,
    smooth_key: str,
    raw_label: str,
    ylabel: str,
    color: str,
    baseline: float,
    baseline_label: str,
    skills,
    annotate_cuts: bool = False,
    ylim=None,
) -> None:
    ts = np.asarray(curve["replan_ts"]).reshape(-1)
    raw = np.asarray(curve[raw_key]).reshape(-1)
    smooth = np.asarray(curve.get(smooth_key, raw)).reshape(-1)
    if len(ts):
        width = (ts[1] - ts[0]) * 0.8 if len(ts) > 1 else 4
        ax.bar(ts, raw, width=width, align="center", alpha=0.3, color=color, label=raw_label)
        ax.plot(ts, smooth, color=color, linewidth=1.8, label="smoothed")
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label=baseline_label)
    _shade_excluded_initial_motion(ax, curve)
    _shade_excluded_tail(ax, curve)
    if ylim is not None:
        ax.set_ylim(*[float(value) for value in np.asarray(ylim).reshape(-1)])
    _plot_skill_cuts(ax, skills, annotate=annotate_cuts)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.margins(x=0.01)


def plot_boundary_curve(ax, curve: dict, skills) -> None:
    """Multimodality (VF cos-divergence) curve: raw bars + SG-smoothed line + mean
    threshold + detected peaks, with the actual skill cuts (from the dataset) marked
    as vertical lines so the graph ties to the frame boxes above."""
    ts = np.asarray(curve["replan_ts"]).reshape(-1)
    raw = np.asarray(curve["div_cos"]).reshape(-1)
    sg = np.asarray(curve["sg_vals"]).reshape(-1)
    score_power = float(np.asarray(curve.get("score_power", 1.0)).reshape(-1)[0])
    # ``sg_vals`` is stored after applying the detector's score-power transform.
    # Put the raw bars in that same score space so bars, line, threshold, and
    # boundary peaks share one meaningful y-axis.
    raw_score = (
        raw
        if np.isclose(score_power, 1.0)
        else np.power(np.maximum(raw, 0.0), score_power)
    )
    mean_val = float(np.asarray(curve["mean_val"]).reshape(-1)[0])
    peak_ts = np.asarray(curve["peak_ts"]).reshape(-1)
    peak_vals = np.asarray(curve["peak_vals"]).reshape(-1)
    score_metric = str(
        np.asarray(curve.get("score_metric", "covariance_gated_cosine")).reshape(-1)[0]
    )
    if score_metric == "within_normalized_l2":
        raw_metric_label = "center distance / within-mode radius"
        ylabel = "GMM separation ratio"
    elif score_metric == "covariance_gated_angular_chord":
        raw_metric_label = "covariance-gated angular chord"
        ylabel = "angular chord score"
    elif score_metric == "strong_covariance_gated_cosine":
        raw_metric_label = "strong reliability-gated cosine"
        ylabel = "SBD detector score"
    else:
        raw_metric_label = "multimodality"
        ylabel = "SBD detector score"
    if len(ts):
        width = (ts[1] - ts[0]) * 0.8 if len(ts) > 1 else 4
        raw_label = (
            f"{raw_metric_label} (raw)"
            if np.isclose(score_power, 1.0)
            else f"{raw_metric_label} (raw^{score_power:g})"
        )
        ax.bar(ts, raw_score, width=width, align="center", alpha=0.35, color="tab:red", label=raw_label)
        ax.plot(ts, sg, color="tab:orange", linewidth=1.8, label="smoothed")
    hide_threshold = bool(
        np.asarray(curve.get("hide_score_threshold", False)).reshape(-1)[0]
    )
    if not hide_threshold:
        ax.axhline(
            mean_val,
            color="tab:orange",
            linestyle="--",
            linewidth=1.2,
            label=f"mean={mean_val:.6g}",
        )
    if len(peak_ts):
        ax.scatter(peak_ts, peak_vals, color="red", s=36, zorder=5, label="boundary peak")
    power1_ax = None
    if "score_power1_mean" in curve and not np.isclose(score_power, 1.0):
        power1_curve = np.power(
            np.maximum(sg, 0.0),
            1.0 / score_power,
        )
        power1_mean = float(np.asarray(curve["score_power1_mean"]).reshape(-1)[0])
        power1_ax = ax.twinx()
        power1_ax.plot(
            ts,
            power1_curve,
            color="tab:blue",
            linewidth=1.6,
            alpha=0.9,
            label="smoothed (power=1)",
        )
        if not hide_threshold:
            power1_ax.axhline(
                power1_mean,
                color="tab:blue",
                linestyle="--",
                linewidth=1.1,
                alpha=0.9,
                label=f"power=1 global mean={power1_mean:.6f}",
            )
        power1_ax.set_ylabel("SBD score (power=1)", fontsize=8, color="tab:blue")
        power1_ax.tick_params(axis="y", labelsize=7, colors="tab:blue")
        power1_ax.margins(y=0.08)
    _shade_excluded_initial_motion(ax, curve)
    _shade_excluded_tail(ax, curve)
    # Actual skill cuts used in the dataset (skill starts + final end).
    _plot_skill_cuts(ax, skills, annotate=True)
    ax.set_xlabel("frame", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    if power1_ax is not None:
        extra_handles, extra_labels = power1_ax.get_legend_handles_labels()
        handles.extend(extra_handles)
        labels.extend(extra_labels)
    ax.legend(handles, labels, fontsize=6, loc="upper right", ncol=2)
    ax.margins(x=0.01)


def plot_action_error(ax, curve: dict, skills) -> None:
    """Fresh-inference normalized action MSE aligned to the SBD frame axis."""
    ts = np.asarray(curve["action_error_replan_ts"]).reshape(-1)
    raw = np.asarray(curve["action_error_normalized_mse"]).reshape(-1)
    smooth = np.asarray(
        curve.get("action_error_normalized_mse_smooth", raw)
    ).reshape(-1)
    if len(ts):
        width = (ts[1] - ts[0]) * 0.8 if len(ts) > 1 else 4
        ax.bar(
            ts,
            raw,
            width=width,
            align="center",
            alpha=0.25,
            color="tab:blue",
            label="normalized action MSE (raw)",
        )
        ax.plot(ts, smooth, color="tab:blue", linewidth=1.8, label="5-anchor mean")
    _shade_excluded_tail(ax, curve)
    _plot_skill_cuts(ax, skills)
    ax.set_ylabel("normalized\naction MSE", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.margins(x=0.01)


def plot_denoising_gain(ax, curve: dict, skills) -> None:
    recut = "denoising_gain_threshold" in curve
    threshold = (
        float(np.asarray(curve["denoising_gain_threshold"]).reshape(-1)[0])
        if recut
        else 1.0
    )
    _plot_raw_and_smooth(
        ax,
        curve,
        raw_key="denoising_gain",
        smooth_key="denoising_gain_sg",
        raw_label="denoising gain (raw)",
        ylabel="denoising gain",
        color="tab:purple",
        baseline=threshold,
        baseline_label=(f"global mean={threshold:.4f}" if recut else "no contraction = 1"),
        skills=skills,
        annotate_cuts=recut,
        ylim=curve.get("denoising_gain_ylim"),
    )
    if recut:
        ax.axhline(
            1.0,
            color="black",
            linestyle=":",
            linewidth=0.9,
            alpha=0.6,
            label="no contraction = 1",
        )
        peak_ts = np.asarray(curve.get("denoising_gain_peak_ts", [])).reshape(-1)
        peak_vals = np.asarray(curve.get("denoising_gain_peak_vals", [])).reshape(-1)
        if len(peak_ts):
            ax.scatter(
                peak_ts,
                peak_vals,
                color="red",
                s=36,
                zorder=5,
                label="boundary peak",
            )
        ax.set_xlabel("frame", fontsize=8)

    power2_ax = None
    if "denoising_gain_power2_mean" in curve:
        ts = np.asarray(curve["replan_ts"]).reshape(-1)
        smooth = np.asarray(curve.get("denoising_gain_sg", curve["denoising_gain"])).reshape(-1)
        power2_curve = np.square(np.maximum(smooth, 0.0))
        power2_mean = float(
            np.asarray(curve["denoising_gain_power2_mean"]).reshape(-1)[0]
        )
        power2_ax = ax.twinx()
        power2_ax.plot(
            ts,
            power2_curve,
            color="tab:blue",
            linewidth=1.6,
            alpha=0.9,
            label="smoothed gain (power=2)",
        )
        power2_ax.axhline(
            power2_mean,
            color="tab:blue",
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
            label=f"power=2 global mean={power2_mean:.6f}",
        )
        power2_ax.set_ylabel("denoising gain²", fontsize=8, color="tab:blue")
        power2_ax.tick_params(axis="y", labelsize=7, colors="tab:blue")
        if "denoising_gain_power2_ylim" in curve:
            power2_ax.set_ylim(
                *[
                    float(value)
                    for value in np.asarray(curve["denoising_gain_power2_ylim"]).reshape(-1)
                ]
            )
        else:
            power2_ax.margins(y=0.08)

    handles, labels = ax.get_legend_handles_labels()
    if power2_ax is not None:
        extra_handles, extra_labels = power2_ax.get_legend_handles_labels()
        handles.extend(extra_handles)
        labels.extend(extra_labels)
    ax.legend(handles, labels, fontsize=6, loc="upper right", ncol=2)


def plot_delta_bic(ax, curve: dict, skills) -> None:
    _plot_raw_and_smooth(
        ax,
        curve,
        raw_key="delta_bic",
        smooth_key="delta_bic_sg",
        raw_label="ΔBIC (raw)",
        ylabel="ΔBIC (K1 - best K2..4)",
        color="tab:green",
        baseline=0.0,
        baseline_label="equal support = 0",
        skills=skills,
    )


def _plot_gripper_signal(ax, signal: np.ndarray, label: str, skills) -> None:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    ts = np.arange(len(signal))
    ax.plot(ts, signal, color="tab:green", linewidth=1.2, label=label)
    for fc in _skill_cut_frames(skills):
        ax.axvline(fc, color="#2f6fd0", linestyle=":", linewidth=0.9, alpha=0.8)
    ax.set_ylabel(label, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="upper right")


def _render_timeline(skills, curve, gripper_signal, gripper_labels) -> dict | None:
    """Render available SBD metrics + one axis per gripper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    gripper = None
    if gripper_signal is not None:
        gripper = np.asarray(gripper_signal, dtype=np.float32)
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        if gripper.ndim != 2:
            raise ValueError(f"gripper_signal must be (T, G), got {gripper.shape}")

    metric_plotters = []
    if curve is not None:
        if "div_cos" in curve:
            metric_plotters.append(plot_boundary_curve)
        if "action_error_normalized_mse" in curve:
            metric_plotters.append(plot_action_error)
        if "denoising_gain" in curve:
            metric_plotters.append(plot_denoising_gain)
        if "delta_bic" in curve:
            metric_plotters.append(plot_delta_bic)

    n_grippers = 0 if gripper is None else gripper.shape[1]
    n_plots = len(metric_plotters) + n_grippers
    if n_plots == 0:
        return None

    lengths = [skills[-1][1] if skills else 0]
    if curve is not None:
        if "n_frames" in curve:
            lengths.append(int(np.asarray(curve["n_frames"]).reshape(-1)[0]))
        ts = np.asarray(curve.get("replan_ts", [])).reshape(-1)
        if len(ts):
            lengths.append(int(ts[-1]) + 1)
    if gripper is not None:
        lengths.append(len(gripper))
    n_frames = max(1, *lengths)

    # Preserve the 0.02 inch/frame scale for long episodes, but give short
    # LIBERO episodes enough horizontal room to inspect peaks and skill cuts.
    # The x coordinates remain true frame indices; only the minimum display
    # width changes. Long ABC timelines stay vector-based and scrollable.
    fig_w = max(7.5, n_frames * 0.02)
    fig_h = 1.65 * n_plots + 0.45
    fig, axes = plt.subplots(n_plots, 1, figsize=(fig_w, fig_h), sharex=True, squeeze=False)
    axes = axes[:, 0]
    axis_i = 0
    for plot_metric in metric_plotters:
        plot_metric(axes[axis_i], curve, skills)
        axes[axis_i].set_xlabel("")
        axis_i += 1

    labels = list(gripper_labels or [])
    for grip_i in range(n_grippers):
        label = labels[grip_i] if grip_i < len(labels) else f"gripper[{grip_i}]"
        _plot_gripper_signal(axes[axis_i], gripper[:, grip_i], label, skills)
        axis_i += 1

    for ax in axes:
        ax.set_xlim(0, n_frames)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.margins(x=0)
    axes[-1].set_xlabel("timestep (100-step major ticks)", fontsize=8)
    plot_left = min(0.12, 1.1 / fig_w)
    plot_right = 0.995
    plot_top = 0.97
    plot_bottom = 0.12
    fig.subplots_adjust(
        left=plot_left,
        right=plot_right,
        top=plot_top,
        bottom=plot_bottom,
        hspace=0.22,
    )
    return {
        "svg": fig_to_svg_b64(fig),
        "n_frames": n_frames,
        # CSS top/bottom offsets are measured from the corresponding edge.
        "plot_left": plot_left,
        "plot_right": plot_right,
        "plot_top": 1.0 - plot_top,
        "plot_bottom": plot_bottom,
    }


def render_skillset_card(
    skills,
    raw,
    curve,
    thumb: int = 110,
    gripper_signal=None,
    gripper_labels=None,
    skill_videos=None,
    episode_video=None,
    gmm_visualization=None,
    show_frames: bool = True,
) -> dict[str, str]:
    """One episode → jpeg b64. ``skills`` = [(frame_start, frame_end, label_or_None), ...];
    label (e.g. FSQ token) is shown next to the skill index when not None. ``curve`` =
    the dict from load_boundary_curve (or None → frames-only, no graph row).

    No figure suptitle: the episode caption is rendered by save_gallery above the card,
    and a centred suptitle would overlap the frames (esp. for 2-skill episodes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _blank():
        return np.full((thumb,) * 2 + (3,), 80, np.uint8)

    have = raw is not None and len(raw)
    timeline = _render_timeline(
        skills,
        curve,
        gripper_signal=gripper_signal,
        gripper_labels=gripper_labels,
    )
    timeline_media = (
        {
            "timeline_svg": timeline["svg"],
            "timeline_n_frames": timeline["n_frames"],
            "timeline_plot_left": timeline["plot_left"],
            "timeline_plot_right": timeline["plot_right"],
            "timeline_plot_top": timeline["plot_top"],
            "timeline_plot_bottom": timeline["plot_bottom"],
        }
        if timeline is not None
        else {"timeline_svg": ""}
    )

    if episode_video is not None:
        return {
            "episode_video": episode_video,
            "skills": list(skills),
            "gmm_visualization": gmm_visualization,
            **timeline_media,
        }

    if skill_videos is not None:
        if len(skill_videos) != len(skills):
            raise ValueError(f"skill_videos length {len(skill_videos)} != skills length {len(skills)}")
        items = []
        for k, ((fs, fe, lab), video) in enumerate(zip(skills, skill_videos, strict=True)):
            start_jpeg = ""
            end_jpeg = ""
            if show_frames:
                start_frame = _clip_frame_or_blank(raw, fs, thumb) if have else _blank()
                end_frame = _clip_frame_or_blank(raw, max(0, fe - 1), thumb) if have else _blank()
                start_jpeg = _frame_to_b64(start_frame, thumb)
                end_jpeg = _frame_to_b64(end_frame, thumb)
            items.append({
                "index": k,
                "label": lab,
                "frame_start": fs,
                "frame_end": fe,
                "start_jpeg": start_jpeg,
                "end_jpeg": end_jpeg,
                "thumb_size": thumb,
                "show_frames": show_frames,
                **video,
            })
        return {"skill_items": items, **timeline_media}

    if not show_frames:
        return {"frames_jpeg": "", **timeline_media}

    n = max(1, len(skills))
    # Each skill gets a real container Axes whose spines are the blue boundary box.
    # This is deliberately different from deriving a FancyBboxPatch from child-Axes
    # positions: equal-aspect imshow axes can be repositioned during the final
    # savefig draw, which made those inferred boxes drift across the thumbnails on
    # very wide ABC figures.
    #
    # Wrap long episodes instead of growing one unbounded horizontal strip. JPEG's
    # maximum dimension is 65,500 px; ABC has episodes with 170+ skills, while
    # LIBERO generally stays on a single row. Fixed columns keep thumbnail sizing
    # consistent for both datasets and make every card renderable.
    max_skills_per_row = 8
    n_cols = min(n, max_skills_per_row)
    n_rows = (n + n_cols - 1) // n_cols
    per_skill_w = 4.0
    frame_w = per_skill_w * n_cols + 0.6 * (n_cols - 1) + 0.3
    fig = plt.figure(figsize=(frame_w, 2.1 * n_rows))
    gs_v = fig.add_gridspec(n_rows, 1, hspace=0.35)

    skill_rows = [gs_v[row].subgridspec(1, n_cols, wspace=0.18) for row in range(n_rows)]
    for k, (fs, fe, lab) in enumerate(skills):
        row, col = divmod(k, n_cols)
        cell = skill_rows[row][col]

        # The container owns the border, so it is guaranteed to move and scale
        # with the two child image axes during the final renderer pass.
        ax_box = fig.add_subplot(cell)
        ax_box.set_facecolor("none")
        ax_box.set_xticks([])
        ax_box.set_yticks([])
        ax_box.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax_box.spines.values():
            spine.set_color("#2f6fd0")
            spine.set_linewidth(1.3)
        ax_box.set_zorder(20)
        ax_box.patch.set_alpha(0.0)

        inner = cell.subgridspec(1, 2, wspace=0.04)
        ax_s = fig.add_subplot(inner[0])
        ax_e = fig.add_subplot(inner[1])
        s = _clip_frame_or_blank(raw, fs, thumb) if have else _blank()
        e = _clip_frame_or_blank(raw, max(0, fe - 1), thumb) if have else _blank()
        lab_str = f" tok{lab}" if lab is not None else ""
        ax_s.imshow(s); ax_s.axis("off"); ax_s.set_title(f"sk{k}{lab_str}\nstart f{fs}", fontsize=6)
        ax_e.imshow(e); ax_e.axis("off"); ax_e.set_title(f"end →f{fe}", fontsize=6)
    frames_jpeg = fig_to_b64(fig)
    return {"frames_jpeg": frames_jpeg, **timeline_media}
