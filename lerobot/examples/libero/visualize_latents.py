"""
Latent Space Visualization — t-SNE on skill VAE latent codes.

Loads skill_latents.npz produced by train_vae.py and plots t-SNE.
Color options:
  - 'length'     : skill duration (short vs long)
  - 'episode_id' : which episode the skill came from
  - 'cluster'    : KMeans cluster assignment

Optionally logs the plot to wandb.

Usage:
      
    # vae_lat32_hid128/skill_vae_latents_epoch0200 best
        
    python examples/libero/visualize_latents.py \
        --latents_path /scratch/mdorazi/Skill_Boundary_Detector/outputs/vae_sweep_epoch/vae_lat32_hid256/skill_vae_latents_epoch0800.npz \
        --output_dir /tmp/test_cluster_video \
        --color_by all \
        --n_clusters 10 \
        --dataset_dir /scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_10 \
        --skills_per_cluster 10 \
        --wandb_project SBD_vae \
        --wandb_run_name test_cluster_video
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tyro


@dataclass
class Args:
    latents_path: str = ""
    """Path to skill_latents.npz produced by train_vae.py."""
    output_dir: str = ""
    """Where to save the plot. Defaults to same directory as latents_path."""
    dataset_dir: str = ""
    """Path to dataset (e.g. libero_10) for episode_id → task_id mapping. Required for color_by=task."""
    skills_dir: str = ""
    """Path to skills directory (output of replay_demo.py). Used as fallback if latents npz lacks metadata."""

    # t-SNE
    perplexity: float = 30.0
    n_iter: int = 1000
    random_state: int = 42

    # Coloring
    color_by: str = "length"
    """'length', 'episode_id', or 'cluster'."""
    n_clusters: int = 10
    """Number of KMeans clusters (used when color_by='cluster')."""
    skill_index_bin: int = 3
    """Group skill indices into bins of this size for color_by=skill_index (0=no binning)."""

    # Cluster video
    skills_per_cluster: int = 10
    """Number of skills to sample per cluster for interactive video plot."""
    video_fps: int = 20
    """FPS for extracted skill video clips."""

    # Display
    point_size: float = 8.0
    alpha: float = 0.7
    figsize_w: float = 12.0
    figsize_h: float = 10.0

    # wandb
    wandb_project: str | None = None
    wandb_run_name: str = "latent_tsne"


def load_episode_task_map(dataset_dir: str) -> dict[int, int]:
    """Return {episode_id: task_id} by reading dataset episode metadata."""
    import pandas as pd
    from pathlib import Path as P
    meta_files = sorted((P(dataset_dir) / "meta" / "episodes").rglob("*.parquet"))
    meta = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    # task_id is stored in stats/task_index/min as a list
    ep_task = {}
    for _, row in meta.iterrows():
        ep_id = int(row["episode_index"])
        task_val = row["stats/task_index/min"]
        task_id = int(task_val[0]) if hasattr(task_val, "__len__") else int(task_val)
        ep_task[ep_id] = task_id
    return ep_task


def run_tsne(latents: np.ndarray, perplexity: float, n_iter: int, random_state: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    print(f"[t-SNE] Running on {latents.shape[0]} points (dim={latents.shape[1]}) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        verbose=1,
    )
    return tsne.fit_transform(latents)


def extract_skill_clip_b64(
    dataset_dir: str,
    episode_id: int,
    frame_start: int,
    frame_end: int,
    episodes_meta,
    fps: int = 20,
) -> str | None:
    """Extract a skill video clip and return base64-encoded mp4 string (using imageio)."""
    import tempfile, base64, os
    import imageio
    from pathlib import Path as P

    row = episodes_meta[episodes_meta["episode_index"] == episode_id]
    if row.empty:
        return None

    video_cols = [c for c in episodes_meta.columns if c.startswith("videos/") and c.endswith("/chunk_index")]
    if not video_cols:
        return None
    cam_key = video_cols[0].split("/")[1]

    chunk_idx = int(row.iloc[0][f"videos/{cam_key}/chunk_index"])
    file_idx  = int(row.iloc[0][f"videos/{cam_key}/file_index"])
    from_ts   = float(row.iloc[0][f"videos/{cam_key}/from_timestamp"])
    to_ts     = float(row.iloc[0][f"videos/{cam_key}/to_timestamp"])

    src = P(dataset_dir) / "videos" / cam_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
    if not src.exists():
        return None

    tmp_path = None
    try:
        # 스킬 구간의 절대 타임스탬프 계산
        skill_start_ts = from_ts + frame_start / fps
        skill_duration = max((frame_end - frame_start) / fps, 0.1)

        # ffmpeg seek으로 필요한 구간만 읽기 (전체 로드 방지)
        n_frames_needed = frame_end - frame_start
        reader = imageio.get_reader(
            str(src),
            format="ffmpeg",
            input_params=["-ss", f"{skill_start_ts:.3f}"],
        )
        frames = []
        for i, frame in enumerate(reader):
            if i >= n_frames_needed:
                break
            frames.append(frame)
        reader.close()

        if not frames:
            return None

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name

        writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264",
                                    output_params=["-crf", "28", "-preset", "fast"])
        for frame in frames:
            # Resize to max width 320
            h, w = frame.shape[:2]
            if w > 320:
                scale = 320 / w
                new_w, new_h = int(w * scale), int(h * scale)
                # Simple resize via slicing (no cv2 dependency)
                import PIL.Image
                img = PIL.Image.fromarray(frame).resize((new_w, new_h))
                frame = np.array(img)
            writer.append_data(frame)
        writer.close()

        with open(tmp_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return b64

    except Exception as e:
        print(f"    [warn] clip extraction failed for ep{episode_id}: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def make_interactive_cluster_plotly(
    xy: np.ndarray,
    cluster_labels: np.ndarray,
    episode_ids: np.ndarray,
    frame_starts: np.ndarray,
    frame_ends: np.ndarray,
    skill_idxs: np.ndarray,
    task_ids: np.ndarray | None,
    dataset_dir: str,
    skills_per_cluster: int = 10,
    fps: int = 20,
    point_size: float = 6.0,
    alpha: float = 0.7,
    random_state: int = 42,
):
    """Interactive Plotly: KMeans clusters colored, click a point → play skill video."""
    import plotly.graph_objects as go
    import pandas as pd
    from pathlib import Path as P

    rng = np.random.default_rng(random_state)
    unique_clusters = sorted(np.unique(cluster_labels))
    cmap_fn = cm.get_cmap("tab20", len(unique_clusters))

    # Load episodes_meta once
    meta_files = sorted((P(dataset_dir) / "meta" / "episodes").rglob("*.parquet"))
    episodes_meta = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)

    # Sample skills_per_cluster per cluster and extract videos
    print(f"[Cluster Video] Sampling {skills_per_cluster} skills/cluster × {len(unique_clusters)} clusters ...")
    sampled_indices = {}   # cluster → list of global indices
    for cl in unique_clusters:
        idxs = np.where(cluster_labels == cl)[0]
        chosen = rng.choice(idxs, size=min(skills_per_cluster, len(idxs)), replace=False)
        sampled_indices[cl] = chosen.tolist()

    # Extract videos for sampled skills
    # video_map: global_idx → base64 string
    video_map: dict[int, str] = {}
    all_sampled = [i for idxs in sampled_indices.values() for i in idxs]
    for gi in all_sampled:
        ep_id = int(episode_ids[gi])
        fs, fe = int(frame_starts[gi]), int(frame_ends[gi])
        b64 = extract_skill_clip_b64(dataset_dir, ep_id, fs, fe, episodes_meta, fps)
        if b64 is not None:
            video_map[gi] = b64
    print(f"[Cluster Video] Extracted {len(video_map)}/{len(all_sampled)} clips.")

    # Build traces: one per cluster
    # Non-sampled: small, no video. Sampled: larger marker, has video in customdata.
    traces = []
    for ci, cl in enumerate(unique_clusters):
        rgba = cmap_fn(ci)
        color_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{alpha})"
        sampled_set = set(sampled_indices[cl])
        mask_all = cluster_labels == cl

        # Non-sampled points (background)
        bg_idxs = [i for i in np.where(mask_all)[0] if i not in sampled_set]
        if bg_idxs:
            bg_idxs = np.array(bg_idxs)
            traces.append(go.Scatter(
                x=xy[bg_idxs, 0], y=xy[bg_idxs, 1],
                mode="markers",
                name=f"cluster {cl}",
                legendgroup=f"cl{cl}",
                marker=dict(size=point_size, color=color_str, opacity=0.4),
                hovertemplate=f"cluster {cl}<br>ep=%{{customdata[0]}} skill=%{{customdata[1]}}<extra></extra>",
                customdata=np.stack([episode_ids[bg_idxs], skill_idxs[bg_idxs]], axis=1),
                showlegend=True,
            ))

        # Sampled points (clickable, with video)
        s_idxs = [i for i in sampled_indices[cl] if i in video_map]
        if s_idxs:
            s_idxs = np.array(s_idxs)
            vid_b64_list = [video_map[i] for i in s_idxs]
            traces.append(go.Scatter(
                x=xy[s_idxs, 0], y=xy[s_idxs, 1],
                mode="markers",
                name=f"cluster {cl} (click)",
                legendgroup=f"cl{cl}",
                marker=dict(size=point_size * 2, color=color_str,
                            line=dict(width=1.5, color="white")),
                hovertemplate=f"cluster {cl} ▶ click to play<br>ep=%{{customdata[0]}} skill=%{{customdata[1]}}<extra></extra>",
                customdata=[[int(episode_ids[i]), int(skill_idxs[i]), v]
                            for i, v in zip(s_idxs, vid_b64_list)],
                showlegend=False,
            ))

    layout = go.Layout(
        title="KMeans Clusters — click highlighted point to play skill video",
        xaxis_title="t-SNE dim 1",
        yaxis_title="t-SNE dim 2",
        legend=dict(groupclick="toggleitem"),
        width=900, height=700,
    )
    fig = go.Figure(data=traces, layout=layout)

    # Inject JavaScript for click-to-play (side-by-side layout)
    click_js = """
<style>
  #main-layout {
    display: flex;
    align-items: flex-start;
    gap: 16px;
  }
  #main-layout .plotly-graph-div {
    flex: 0 0 auto;
  }
  #video-panel {
    flex: 0 0 340px;
    padding-top: 60px;
    text-align: center;
  }
  #video-hint {
    color: #888;
    font-size: 13px;
    margin-top: 8px;
  }
  #skill-video {
    display: none;
    width: 320px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
  }
  #video-label {
    color: #555;
    font-size: 12px;
    margin-top: 6px;
  }
</style>
<script>
document.addEventListener('DOMContentLoaded', function() {
  // 플롯과 비디오 패널을 나란히 배치
  var plotDiv = document.querySelector('.plotly-graph-div');
  if (!plotDiv) return;
  var wrapper = document.createElement('div');
  wrapper.id = 'main-layout';
  plotDiv.parentNode.insertBefore(wrapper, plotDiv);
  wrapper.appendChild(plotDiv);

  var panel = document.createElement('div');
  panel.id = 'video-panel';
  panel.innerHTML = '<p id="video-hint">▶ Clicking on a large point<br>plays the corresponding skill.</p>' +
                    '<video id="skill-video" controls></video>' +
                    '<p id="video-label"></p>';
  wrapper.appendChild(panel);

  plotDiv.on('plotly_click', function(data) {
    var pt = data.points[0];
    if (!pt.customdata || pt.customdata.length < 3) return;
    var ep  = pt.customdata[0];
    var si  = pt.customdata[1];
    var b64 = pt.customdata[2];
    if (typeof b64 !== 'string' || b64.length < 10) return;
    var vid   = document.getElementById('skill-video');
    var hint  = document.getElementById('video-hint');
    var label = document.getElementById('video-label');
    vid.src = 'data:video/mp4;base64,' + b64;
    vid.style.display = 'block';
    vid.play();
    hint.style.display = 'none';
    label.textContent = 'Episode ' + ep + '  |  Skill ' + si;
  });
});
</script>
"""
    # Write HTML manually to inject JS
    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
    html_str = html_str.replace("</body>", click_js + "\n</body>")
    return fig, html_str


def run_kmeans(latents: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    print(f"[KMeans] Clustering into {n_clusters} clusters ...")
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    return km.fit_predict(latents)


def compute_task_skill_colors(
    task_ids: np.ndarray,    # (N,) int
    skill_idxs: np.ndarray, # (N,) int
) -> tuple[np.ndarray, list]:
    """Compute per-point RGBA colors: task → base hue, skill_index → shade.

    Returns (colors (N,4), legend_handles).
    """
    import matplotlib.patches as mpatches
    # Sequential cmaps per task — each has a distinct hue
    task_cmaps = [
        "Blues", "Reds", "Greens", "Oranges", "Purples",
        "YlOrBr", "PuRd", "BuGn", "GnBu", "RdPu",
    ]
    unique_tasks = sorted(np.unique(task_ids))
    colors = np.zeros((len(task_ids), 4))
    legend_handles = []

    for ti, task in enumerate(unique_tasks):
        mask = task_ids == task
        cmap_name = task_cmaps[ti % len(task_cmaps)]
        cmap_fn = cm.get_cmap(cmap_name)

        idxs_in_task = skill_idxs[mask]
        max_idx = max(idxs_in_task.max(), 1)
        # Map skill_index to [1.0, 0.35] — low index = dark, high index = light
        norm = 1.0 - 0.65 * (idxs_in_task / max_idx)
        colors[mask] = np.array([cmap_fn(v) for v in norm])

        # Legend: show the darkest shade for this task
        legend_handles.append(mpatches.Patch(
            color=cmap_fn(1.0), label=f"task {task}"
        ))

    return colors, legend_handles


def make_interactive_plotly(
    xy: np.ndarray,           # (N, 2)
    task_ids: np.ndarray,     # (N,) int
    skill_idxs: np.ndarray,   # (N,) int
    episode_ids: np.ndarray,  # (N,) int
    point_size: float = 6.0,
    alpha: float = 0.7,
):
    """Build an interactive Plotly t-SNE figure.

    One trace per (task, skill_index) combination.
    Skill index slider filters which skill indices are visible.
    Each task has its own base color; skill index controls shade.
    """
    import plotly.graph_objects as go

    task_cmaps = [
        "Blues", "Reds", "Greens", "Oranges", "Purples",
        "YlOrBr", "PuRd", "BuGn", "GnBu", "RdPu",
    ]
    unique_tasks = sorted(np.unique(task_ids))
    unique_skill_idxs = sorted(np.unique(skill_idxs))
    max_skill = max(unique_skill_idxs) if unique_skill_idxs else 1

    # Build one trace per (task, skill_index)
    # trace order: task 변할 때마다 skill_index 전체
    traces = []
    for ti, task in enumerate(unique_tasks):
        cmap_fn = cm.get_cmap(task_cmaps[ti % len(task_cmaps)])
        for si in unique_skill_idxs:
            mask = (task_ids == task) & (skill_idxs == si)
            if not mask.any():
                continue
            shade = 1.0 - 0.65 * (si / max(max_skill, 1))
            rgba = cmap_fn(shade)
            color_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{alpha})"
            traces.append(go.Scatter(
                x=xy[mask, 0],
                y=xy[mask, 1],
                mode="markers",
                name=f"skill {si}",
                legendgroup=f"skill{si}",
                legendgrouptitle={"text": f"Skill {si}"} if ti == 0 else {},
                marker=dict(size=point_size, color=color_str),
                customdata=np.stack([
                    episode_ids[mask],
                    np.full(mask.sum(), task),
                    skill_idxs[mask],
                ], axis=1),
                hovertemplate=(
                    "ep=%{customdata[0]}  task=%{customdata[1]}  skill=%{customdata[2]}<br>"
                    "x=%{x:.2f}  y=%{y:.2f}<extra></extra>"
                ),
                visible=True,
                meta={"task": int(task), "skill": int(si)},
            ))

    # Slider steps — each step shows only traces belonging to that task
    # "all" + one step per task
    steps = [dict(
        method="update",
        args=[{"visible": [True] * len(traces)}],
        label="all",
    )]
    for task in unique_tasks:
        visible = [t.meta["task"] == int(task) for t in traces]
        steps.append(dict(
            method="update",
            args=[{"visible": visible}],
            label=f"task {task}",
        ))

    slider = [dict(
        active=0,
        currentvalue={"prefix": "task: "},
        pad={"t": 50},
        steps=steps,
    )]

    layout = go.Layout(
        title="t-SNE — task (slider) × skill index (legend toggle)",
        xaxis_title="t-SNE dim 1",
        yaxis_title="t-SNE dim 2",
        sliders=slider,
        legend=dict(groupclick="toggleitem", title={"text": "Skill index"}),
        width=900,
        height=700,
    )
    return go.Figure(data=traces, layout=layout)


def plot_tsne(
    xy: np.ndarray,           # (N, 2)
    color_vals: np.ndarray,   # (N,) — numeric values to color by
    color_label: str,
    title: str,
    figsize: tuple[float, float],
    point_size: float,
    alpha: float,
    is_categorical: bool = False,
    cmap: str = "viridis",
    precomputed_colors: np.ndarray | None = None,  # (N, 4) RGBA
    legend_handles: list | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    if precomputed_colors is not None:
        ax.scatter(xy[:, 0], xy[:, 1], c=precomputed_colors, s=point_size, alpha=alpha)
        if legend_handles:
            ax.legend(
                handles=legend_handles, title=color_label,
                bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7,
            )
    elif is_categorical:
        categories = np.unique(color_vals)
        cat_cmap = cm.get_cmap("tab20", len(categories))
        for i, cat in enumerate(categories):
            mask = color_vals == cat
            ax.scatter(
                xy[mask, 0], xy[mask, 1],
                s=point_size, alpha=alpha,
                color=cat_cmap(i), label=str(cat),
            )
        ax.legend(
            title=color_label, markerscale=2,
            bbox_to_anchor=(1.01, 1), loc="upper left",
            fontsize=7,
        )
    else:
        sc = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=color_vals, cmap=cmap,
            s=point_size, alpha=alpha,
        )
        plt.colorbar(sc, ax=ax, label=color_label)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    return fig


def main(args: Args) -> None:
    latents_path = Path(args.latents_path)
    output_dir = Path(args.output_dir) if args.output_dir else latents_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────────
    print(f"Loading {latents_path} ...")
    data = np.load(str(latents_path))
    latents     = data["latents"]       # (N, latent_dim)
    episode_ids = data["episode_id"]    # (N,)
    lengths     = data["length"]        # (N,)
    skill_idxs  = data["skill_index"]   # (N,)

    print(f"  {len(latents)} skills, latent_dim={latents.shape[1]}")
    print(f"  skill length — min:{lengths.min()}  max:{lengths.max()}  mean:{lengths.mean():.1f}")

    # ── t-SNE ──────────────────────────────────────────────────────────────────
    xy = run_tsne(latents, args.perplexity, args.n_iter, args.random_state)

    # ── Color values ───────────────────────────────────────────────────────────
    plots_to_save = []

    # ── episode_id → task_id mapping (if dataset_dir provided) ───────────────
    task_ids = None
    if args.dataset_dir:
        ep_task_map = load_episode_task_map(args.dataset_dir)
        task_ids = np.array([ep_task_map.get(int(ep), -1) for ep in episode_ids])
        print(f"  Tasks found: {np.unique(task_ids)}")

    color_configs = []
    if args.color_by == "cluster" or args.color_by == "all":
        cluster_labels = run_kmeans(latents, args.n_clusters, args.random_state)
        color_configs.append(("cluster", cluster_labels, f"KMeans cluster (k={args.n_clusters})", True))
    if args.color_by in ("length", "all"):
        color_configs.append(("length", lengths.astype(float), "skill length (frames)", False))

    if args.color_by in ("task", "all") and task_ids is not None:
        color_configs.append(("task", task_ids.astype(int), "task ID", True))
    elif args.color_by == "task" and task_ids is None:
        print("  [warn] --dataset_dir not set, skipping task coloring.")
    if args.color_by in ("task_skill", "all") and task_ids is not None:
        color_configs.append(("task_skill", None, "task (hue) + skill index (shade)", False))
    elif args.color_by == "task_skill" and task_ids is None:
        print("  [warn] --dataset_dir not set, skipping task_skill coloring.")
    if args.color_by in ("skill_index", "all"):
        if args.skill_index_bin > 0:
            binned = (skill_idxs.astype(int) // args.skill_index_bin).astype(float)
            color_configs.append(("skill_index", binned, f"skill index (bin={args.skill_index_bin})", False))
        else:
            color_configs.append(("skill_index", skill_idxs.astype(float), "skill index within episode", False))

    if not color_configs:
        # fallback: length
        color_configs.append(("length", lengths.astype(float), "skill length (frames)", False))

    _cmap_for_tag = {
        "skill_index": "Blues_r",
        "length": "viridis",
        "episode_id": "tab20",
        "task": "tab10",
        "cluster": "tab20",
    }

    for tag, color_vals, color_label, is_cat in color_configs:
        # task_skill은 plotly interactive로만 로깅 (matplotlib 정적 이미지 스킵)
        if tag == "task_skill":
            continue
        title = f"t-SNE of skill latents — colored by {tag}"
        if tag == "task_skill":
            precomp, handles = compute_task_skill_colors(task_ids, skill_idxs.astype(int))
            fig = plot_tsne(
                xy, None, color_label, title,
                figsize=(args.figsize_w, args.figsize_h),
                point_size=args.point_size,
                alpha=args.alpha,
                precomputed_colors=precomp,
                legend_handles=handles,
            )
        else:
            fig = plot_tsne(
                xy, color_vals, color_label, title,
                figsize=(args.figsize_w, args.figsize_h),
                point_size=args.point_size,
                alpha=args.alpha,
                is_categorical=is_cat,
                cmap=_cmap_for_tag.get(tag, "viridis"),
            )
        save_path = output_dir / f"tsne_{tag}.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")
        plots_to_save.append((tag, save_path))

    # Also save the xy coordinates + metadata for later use (cluster sampling etc.)
    np.savez(
        str(output_dir / "tsne_coords.npz"),
        xy=xy,
        latents=latents,
        episode_id=episode_ids,
        skill_index=skill_idxs,
        frame_start=data["frame_start"],
        frame_end=data["frame_end"],
        length=lengths,
        **({"cluster": cluster_labels} if args.color_by in ("cluster", "all") else {}),
    )
    print(f"  Saved t-SNE coords → {output_dir / 'tsne_coords.npz'}")

    # ── Interactive Plotly (task_skill) ────────────────────────────────────────
    plotly_fig = None
    if task_ids is not None:
        plotly_fig = make_interactive_plotly(
            xy, task_ids, skill_idxs.astype(int), episode_ids,
            point_size=args.point_size, alpha=args.alpha,
        )
        plotly_path = output_dir / "tsne_task_skill_interactive.html"
        plotly_fig.write_html(str(plotly_path))
        print(f"  Saved interactive plot → {plotly_path}")

    # ── Interactive Plotly (cluster + video) ───────────────────────────────────
    cluster_video_html = None
    _cluster_labels_computed = args.color_by in ("cluster", "all")
    if args.dataset_dir and _cluster_labels_computed:
        print(f"[Cluster Video] Building interactive cluster video plot ...")
        _, cluster_video_html = make_interactive_cluster_plotly(
            xy=xy,
            cluster_labels=cluster_labels,
            episode_ids=episode_ids,
            frame_starts=data["frame_start"],
            frame_ends=data["frame_end"],
            skill_idxs=skill_idxs.astype(int),
            task_ids=task_ids,
            dataset_dir=args.dataset_dir,
            skills_per_cluster=args.skills_per_cluster,
            fps=args.video_fps,
            point_size=args.point_size,
            alpha=args.alpha,
            random_state=args.random_state,
        )
        cluster_video_path = output_dir / "tsne_cluster_video.html"
        with open(str(cluster_video_path), "w") as f:
            f.write(cluster_video_html)
        print(f"  Saved cluster video plot → {cluster_video_path}")

    # ── wandb ──────────────────────────────────────────────────────────────────
    if args.wandb_project:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "n_skills": len(latents),
                "latent_dim": latents.shape[1],
                "perplexity": args.perplexity,
                "n_clusters": args.n_clusters,
                "color_by": args.color_by,
            },
        )
        print(f"  Logging to wandb: {[tag for tag, _ in plots_to_save]}")
        log_dict = {f"tsne/{tag}": wandb.Image(str(save_path)) for tag, save_path in plots_to_save}
        if plotly_fig is not None:
            log_dict["tsne/task_skill_interactive"] = wandb.Plotly(plotly_fig)
        if cluster_video_html is not None:
            log_dict["tsne/cluster_video"] = wandb.Html(cluster_video_html)
        run.log(log_dict, commit=True)
        run.finish()
        print(f"  Logged to wandb project '{args.wandb_project}'")


if __name__ == "__main__":
    main(tyro.cli(Args))
