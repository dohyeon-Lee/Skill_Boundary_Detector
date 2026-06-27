"""Shared rendering for the DP skill-boundary eval.

Per-episode card = boxed start/end frames for each skill (tight gap within a skill,
wide gap + coloured box between skills) with the multimodality (VF cos-divergence)
curve overlaid below when available, plus the HTML gallery wrapper.

Data-source agnostic on purpose: callers pass a per-episode ``skills`` list
``[(frame_start, frame_end, label_or_None), ...]``, the raw episode frames, and an
optional curve dict. Used by:
  • train_skillVLA/build_data_eval (skills from the skillvla dataset parquet)
  • train_skills/DP_FSQ_eval        (skills from the FSQ_dataset skillset npz)

Requires ``examples/libero`` on sys.path (for codebook_visualizer); both callers add it.
"""

from __future__ import annotations

import base64
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


def save_gallery(out_dir: Path, title: str, cards: list[tuple], filename: str = "index.html") -> None:
    """Write an HTML gallery to out_dir/filename (default index.html).

    cards: [(task_label, caption, jpeg_b64), ...]. When task_label is not None, a
    sticky section header is inserted whenever it changes, so episodes are grouped
    and visually separated per task. (task_label None → flat gallery as before.)
    filename: output HTML name within out_dir — set to e.g. '<dp>_ck<ckpt>.html' to keep multiple
    runs side by side in one shallow folder instead of overwriting index.html."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks, prev = [], "\0"
    for task_label, cap, b in cards:
        if task_label is not None and task_label != prev:
            blocks.append(f"<h2 class='task'>task {int(task_label):02d}</h2>")
        prev = task_label
        blocks.append(
            f"<div class='card'><div class='cap'>{cap}</div>"
            f"<img src='data:image/jpeg;base64,{b}'></div>"
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
        "</style></head><body><h1>" + title + "</h1><div class='grid'>" + body + "</div></body></html>"
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


def render_skillset_card(skills, raw, curve, thumb: int = 110) -> str:
    """One episode → jpeg b64. ``skills`` = [(frame_start, frame_end, label_or_None), ...];
    label (e.g. FSQ token) is shown next to the skill index when not None. ``curve`` =
    the dict from load_boundary_curve (or None → frames-only, no graph row).

    No figure suptitle: the episode caption is rendered by save_gallery above the card,
    and a centred suptitle would overlap the frames (esp. for 2-skill episodes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    def _blank():
        return np.full((thumb,) * 2 + (3,), 80, np.uint8)

    have = raw is not None and len(raw)
    n = max(1, len(skills))
    # Tight gap WITHIN each skill's start->end pair, wide gap BETWEEN skills, and a
    # coloured box per skill, so adjacent skills read as distinct groups instead of
    # one continuous strip. When the multimodality curve is available, it is overlaid
    # in a second row below the frames.
    #
    # FIXED inches per skill (start+end pair) so thumbnails are the SAME size regardless of skill count:
    # the card just grows wider and scrolls horizontally (.card has overflow-x:auto) instead of shrinking
    # the frames to fit a fixed width. 4.0 ≈ the well-sized 2-skill case (which only looked good before
    # because of the max(.,8.0) clamp stretching 2 skills across 8in). The curve row, if present, spans
    # the same width — that's fine; we no longer force the frames to match a fixed total width.
    per_skill_w = 4.0
    frame_w = per_skill_w * n + 0.6 * (n - 1) + 0.3
    # The multimodality curve keeps a FIXED width regardless of skill count — it must NOT stretch with
    # the (scrollable) frame strip. The figure is as wide as the frames need, but the curve axis is
    # pinned to curve_w inches (left-anchored) after layout (see the reposition below). 7.0 ≈ the
    # well-sized 2-skill graph the layout was tuned for.
    curve_w = 7.0
    if curve is not None:
        fig_w = max(frame_w, curve_w)
        fig = plt.figure(figsize=(fig_w, 3.7))
        gs_v = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.5], hspace=0.6)
        outer = gs_v[0].subgridspec(1, n, wspace=0.55)
        ax_curve = fig.add_subplot(gs_v[1])
    else:
        fig = plt.figure(figsize=(frame_w, 2.1))
        outer = fig.add_gridspec(1, n, wspace=0.55)
        ax_curve = None
    pair_axes = []
    for k, (fs, fe, lab) in enumerate(skills):
        inner = outer[k].subgridspec(1, 2, wspace=0.04)
        ax_s = fig.add_subplot(inner[0])
        ax_e = fig.add_subplot(inner[1])
        s = _clip_frame_or_blank(raw, fs, thumb) if have else _blank()
        e = _clip_frame_or_blank(raw, max(0, fe - 1), thumb) if have else _blank()
        lab_str = f" tok{lab}" if lab is not None else ""
        ax_s.imshow(s); ax_s.axis("off"); ax_s.set_title(f"sk{k}{lab_str}\nstart f{fs}", fontsize=6)
        ax_e.imshow(e); ax_e.axis("off"); ax_e.set_title(f"end →f{fe}", fontsize=6)
        pair_axes.append((ax_s, ax_e))
    if ax_curve is not None:
        plot_boundary_curve(ax_curve, curve, skills)
    # Boxes need final axes positions, so draw once before reading them.
    fig.canvas.draw()
    # Pin the curve to curve_w inches (left-anchored) so a wide frame strip doesn't stretch the graph.
    if ax_curve is not None:
        cp = ax_curve.get_position()
        ax_curve.set_position([cp.x0, cp.y0, min(cp.width, curve_w / fig_w), cp.height])
    for ax_s, ax_e in pair_axes:
        ps, pe = ax_s.get_position(), ax_e.get_position()
        x0, x1 = min(ps.x0, pe.x0), max(ps.x1, pe.x1)
        y0, y1 = min(ps.y0, pe.y0), max(ps.y1, pe.y1)
        padx, pad_top, pad_bot = 0.012, 0.14, 0.02
        fig.add_artist(FancyBboxPatch(
            (x0 - padx, y0 - pad_bot),
            (x1 - x0) + 2 * padx, (y1 - y0) + pad_top + pad_bot,
            boxstyle="round,pad=0.004", transform=fig.transFigure,
            fill=False, edgecolor="#2f6fd0", linewidth=1.3, zorder=20,
        ))
    return fig_to_b64(fig)
