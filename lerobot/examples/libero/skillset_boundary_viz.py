"""Shared rendering for the DP skill-boundary eval.

Per-episode card = boxed start/end frames for each skill (tight gap within a skill,
wide gap + coloured box between skills), followed by a timestep-scaled SVG timeline
containing the multimodality (VF cos-divergence) and gripper signals when available.

Data-source agnostic on purpose: callers pass a per-episode ``skills`` list
``[(frame_start, frame_end, label_or_None), ...]``, the raw episode frames, and an
optional curve dict. Used by:
  • train_skillVLA/build_data_eval (skills from the skillvla dataset parquet)
  • train_skills/DP_FSQ_eval        (skills from the FSQ_dataset skillset npz)

Requires ``examples/libero`` on sys.path (for codebook_visualizer); both callers add it.
"""

from __future__ import annotations

import base64
import html as html_lib
import io
from pathlib import Path

import numpy as np

from codebook_visualizer import _clip_frame_or_blank


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
            if media.get("skill_items") is not None:
                images = _skill_items_html(media["skill_items"])
            elif media.get("frames_jpeg"):
                images = (
                    f"<img class='frames' src='data:image/jpeg;base64,{media['frames_jpeg']}'>"
                )
            else:
                images = ""
            if media.get("timeline_svg"):
                images += (
                    "<div class='timeline-scroll'>"
                    f"<img class='timeline' src='data:image/svg+xml;base64,{media['timeline_svg']}'>"
                    "</div>"
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
        ".skill-grid{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start;}"
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
        ".timeline-scroll{overflow-x:auto;margin-top:10px;padding-top:4px;border-top:1px solid #eee;}"
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


def plot_boundary_curve(ax, curve: dict, skills) -> None:
    """Multimodality (VF cos-divergence) curve: raw bars + SG-smoothed line + mean
    threshold + detected peaks, with the actual skill cuts (from the dataset) marked
    as vertical lines so the graph ties to the frame boxes above."""
    ts = np.asarray(curve["replan_ts"]).reshape(-1)
    raw = np.asarray(curve["div_cos"]).reshape(-1)
    sg = np.asarray(curve["sg_vals"]).reshape(-1)
    mean_val = float(np.asarray(curve["mean_val"]).reshape(-1)[0])
    peak_ts = np.asarray(curve["peak_ts"]).reshape(-1)
    peak_vals = np.asarray(curve["peak_vals"]).reshape(-1)
    if len(ts):
        width = (ts[1] - ts[0]) * 0.8 if len(ts) > 1 else 4
        ax.bar(ts, raw, width=width, align="center", alpha=0.35, color="tab:red", label="multimodality (raw)")
        ax.plot(ts, sg, color="tab:orange", linewidth=1.8, label="smoothed")
    ax.axhline(mean_val, color="tab:orange", linestyle="--", linewidth=1.2, label=f"mean={mean_val:.4f}")
    if len(peak_ts):
        ax.scatter(peak_ts, peak_vals, color="red", s=36, zorder=5, label="boundary peak")
    # Actual skill cuts used in the dataset (skill starts + final end).
    cut_frames = [fs for fs, _fe, _lab in skills] + ([skills[-1][1]] if skills else [])
    y_top = ax.get_ylim()[1]
    for k, fc in enumerate(cut_frames):
        ax.axvline(fc, color="#2f6fd0", linestyle=":", linewidth=1.0, alpha=0.85)
        if k < len(skills):
            ax.text(fc, y_top, f"sk{k}", fontsize=6, color="#2f6fd0", va="bottom", ha="left")
    ax.set_xlabel("frame", fontsize=8)
    ax.set_ylabel("VF cos divergence", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.margins(x=0.01)


def _plot_gripper_signal(ax, signal: np.ndarray, label: str, skills) -> None:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    ts = np.arange(len(signal))
    ax.plot(ts, signal, color="tab:green", linewidth=1.2, label=label)
    cut_frames = [fs for fs, _fe, _lab in skills] + ([skills[-1][1]] if skills else [])
    for fc in cut_frames:
        ax.axvline(fc, color="#2f6fd0", linestyle=":", linewidth=0.9, alpha=0.8)
    ax.set_ylabel(label, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="upper right")


def _render_timeline(skills, curve, gripper_signal, gripper_labels) -> str | None:
    """Render VF + one axis per gripper with a fixed physical timestep spacing."""
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

    n_grippers = 0 if gripper is None else gripper.shape[1]
    n_plots = int(curve is not None) + n_grippers
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

    # 0.02 inch/frame at 110-DPI HTML viewing is about 2.2 px/frame. The SVG is
    # vector-based, so even a 10k-frame ABC episode remains valid and scrollable.
    fig_w = max(3.0, n_frames * 0.02)
    fig_h = 1.65 * n_plots + 0.45
    fig, axes = plt.subplots(n_plots, 1, figsize=(fig_w, fig_h), sharex=True, squeeze=False)
    axes = axes[:, 0]
    axis_i = 0
    if curve is not None:
        plot_boundary_curve(axes[axis_i], curve, skills)
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
    fig.subplots_adjust(left=min(0.12, 1.1 / fig_w), right=0.995, top=0.97, bottom=0.12, hspace=0.22)
    return fig_to_svg_b64(fig)


def render_skillset_card(
    skills,
    raw,
    curve,
    thumb: int = 110,
    gripper_signal=None,
    gripper_labels=None,
    skill_videos=None,
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
    timeline_svg = _render_timeline(
        skills,
        curve,
        gripper_signal=gripper_signal,
        gripper_labels=gripper_labels,
    )

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
        return {"skill_items": items, "timeline_svg": timeline_svg or ""}

    if not show_frames:
        return {"frames_jpeg": "", "timeline_svg": timeline_svg or ""}

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
    return {"frames_jpeg": frames_jpeg, "timeline_svg": timeline_svg or ""}
