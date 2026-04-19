"""
visualize_decoder.py — VAE 디코더 평가 (인터랙티브).

t-SNE 위의 점을 클릭하면 해당 스킬의 GT vs decoded 액션 비교 그래프가 표시된다.

Pipeline:
  1. tsne_coords.npz (visualize_latents 출력) 로드 → 2D 좌표 + 메타데이터
  2. VAE 모델 로드
  3. 클러스터별 N개 스킬 샘플링 → encode → decode
  4. GT vs decoded 비교 플롯을 base64 PNG로 인코딩
  5. Plotly HTML: 클릭 시 옆 패널에 비교 이미지 표시
  6. 요약 메트릭 (MSE boxplot, length scatter, task MSE) wandb 로깅

Usage:
    python examples/libero/visualize_decoder.py \
        --model_path   .../skill_vae.pt \
        --tsne_coords  .../tsne_coords.npz \
        --skills_dir   .../skill_dataset/skills \
        --output_dir   .../decoder_eval \
        --skills_per_cluster 10 \
        --wandb_project VAE_eval \
        --wandb_run_name decoder_eval
"""

from __future__ import annotations

import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).parent))

ACTION_DIM_NAMES = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]


# ── Args ──────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    model_path: str = ""
    """skill_vae.pt 경로"""
    tsne_coords: str = ""
    """visualize_latents가 저장한 tsne_coords.npz 경로"""
    skills_dir: str = ""
    """스킬 npz 디렉토리 (build_skill_dataset 출력)"""
    output_dir: str = ""
    """결과 저장 디렉토리. 미지정 시 model_path 부모."""
    skills_per_cluster: int = 10
    """클러스터당 샘플링할 스킬 수"""
    point_size: float = 6.0
    alpha: float = 0.7
    device: str = "cuda"
    seed: int = 42
    wandb_project: str | None = None
    wandb_run_name: str = "decoder_eval"
    no_html: bool = False
    """Skip interactive Plotly HTML generation and wandb upload."""


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_path: str, device: str):
    from skill_vae import SkillVAE, VAEConfig
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg: VAEConfig = ckpt["cfg"]
    model = SkillVAE(
        action_dim=cfg.action_dim,
        state_dim=cfg.state_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        stop_threshold=cfg.stop_threshold,
        max_decode_steps=cfg.max_decode_steps,
    )
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), cfg


# ── Skills index ──────────────────────────────────────────────────────────────

def build_skill_index(skills_dir: Path) -> dict[tuple[int, int], dict]:
    """(episode_id, skill_index) → {actions, init_state, task_id} 인덱스 빌드."""
    index = {}
    for f in skills_dir.rglob("*.npz"):
        d = np.load(str(f))
        key = (int(d["episode_id"]), int(d["skill_index"]))
        index[key] = {
            "actions":    d["actions"].astype(np.float32),
            "init_state": d["states"][0].astype(np.float32),
            "task_id":    int(d["task_id"]) if "task_id" in d else -1,
        }
    return index


# ── Comparison plot → base64 PNG ──────────────────────────────────────────────

def make_comparison_b64(gt: np.ndarray, pred: np.ndarray,
                         episode_id: int, skill_idx: int,
                         task_id: int, mse: float) -> str:
    n_dims = gt.shape[1]
    dim_names = ACTION_DIM_NAMES[:n_dims]
    t_gt   = np.arange(len(gt))
    t_pred = np.arange(len(pred))

    fig, axes = plt.subplots(n_dims, 1, figsize=(7, 1.4 * n_dims), sharex=False)
    if n_dims == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, dim_names)):
        ax.plot(t_gt,   gt[:, i],   color="tab:blue",   linewidth=1.0, label="GT")
        ax.plot(t_pred, pred[:, i], color="tab:orange", linewidth=1.0,
                linestyle="--", label="decoded")
        ax.set_ylabel(name, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("frame", fontsize=7)
    fig.suptitle(
        f"ep{episode_id:05d} | task{task_id:02d} | skill{skill_idx:02d} | "
        f"GT={len(gt)}  pred={len(pred)}  MSE={mse:.4f}",
        fontsize=8,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Summary plots ─────────────────────────────────────────────────────────────

def plot_mse_boxplot(mse_per_dim: np.ndarray, n_dims: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(mse_per_dim, tick_labels=ACTION_DIM_NAMES[:n_dims],
               patch_artist=True, medianprops=dict(color="red", linewidth=2))
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction MSE per action dimension")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_length_scatter(gt_lengths: list[int], pred_lengths: list[int]) -> plt.Figure:
    gt, pred = np.array(gt_lengths), np.array(pred_lengths)
    max_len = max(gt.max(), pred.max())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gt, pred, alpha=0.4, s=10, color="tab:blue")
    ax.plot([0, max_len], [0, max_len], "r--", linewidth=1, label="GT = pred")
    ax.set_xlabel("GT length (frames)")
    ax.set_ylabel("Decoded length (frames)")
    ax.set_title("Stop token quality")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_task_mse(task_mses: dict[int, list[float]]) -> plt.Figure:
    tasks = sorted(task_mses.keys())
    means = [np.mean(task_mses[t]) for t in tasks]
    stds  = [np.std(task_mses[t])  for t in tasks]
    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 0.4), 4))
    ax.bar(tasks, means, yerr=stds, capsize=3, color="steelblue", alpha=0.8)
    ax.set_xlabel("task ID")
    ax.set_ylabel("mean MSE")
    ax.set_title("Reconstruction MSE per task")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ── Interactive Plotly ────────────────────────────────────────────────────────

def make_interactive_decoder_plotly(
    xy: np.ndarray,
    cluster_labels: np.ndarray,
    episode_ids: np.ndarray,
    skill_idxs: np.ndarray,
    task_ids: np.ndarray | None,
    img_map: dict[int, str],   # global_idx → base64 PNG
    point_size: float,
    alpha: float,
) -> str:
    import plotly.graph_objects as go

    unique_clusters = sorted(np.unique(cluster_labels))
    cmap_fn = matplotlib.colormaps.get_cmap("tab20").resampled(len(unique_clusters))
    sampled_set = set(img_map.keys())

    traces = []
    for ci, cl in enumerate(unique_clusters):
        rgba = cmap_fn(ci)
        color_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{alpha})"
        mask_all = cluster_labels == cl

        # 배경 점 (클릭 불가)
        bg_idxs = np.array([i for i in np.where(mask_all)[0] if i not in sampled_set])
        if len(bg_idxs):
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

        # 샘플링된 점 (클릭 가능, 비교 이미지 포함)
        s_idxs = [i for i in np.where(mask_all)[0] if i in sampled_set]
        if s_idxs:
            s_idxs = np.array(s_idxs)
            traces.append(go.Scatter(
                x=xy[s_idxs, 0], y=xy[s_idxs, 1],
                mode="markers",
                name=f"cluster {cl} (click)",
                legendgroup=f"cl{cl}",
                marker=dict(size=point_size * 2.5, color=color_str,
                            line=dict(width=1.5, color="white")),
                hovertemplate=f"cluster {cl} ▶ click to compare<br>ep=%{{customdata[0]}} skill=%{{customdata[1]}}<extra></extra>",
                customdata=[[int(episode_ids[i]), int(skill_idxs[i]), img_map[i]]
                            for i in s_idxs],
                showlegend=False,
            ))

    layout = go.Layout(
        title="Decoder eval — click highlighted point to compare GT vs decoded actions",
        xaxis_title="t-SNE dim 1",
        yaxis_title="t-SNE dim 2",
        legend=dict(groupclick="toggleitem"),
        width=900, height=700,
    )
    fig = go.Figure(data=traces, layout=layout)

    click_js = """
<style>
  #main-layout { display:flex; align-items:flex-start; gap:16px; }
  #main-layout .plotly-graph-div { flex:0 0 auto; }
  #img-panel { flex:0 0 500px; padding-top:60px; text-align:center; }
  #img-hint { color:#888; font-size:13px; margin-top:8px; }
  #cmp-img { display:none; max-width:480px; border-radius:8px;
             box-shadow:0 2px 8px rgba(0,0,0,0.25); }
  #img-label { color:#555; font-size:12px; margin-top:6px; }
</style>
<script>
document.addEventListener('DOMContentLoaded', function() {
  var plotDiv = document.querySelector('.plotly-graph-div');
  if (!plotDiv) return;
  var wrapper = document.createElement('div');
  wrapper.id = 'main-layout';
  plotDiv.parentNode.insertBefore(wrapper, plotDiv);
  wrapper.appendChild(plotDiv);

  var panel = document.createElement('div');
  panel.id = 'img-panel';
  panel.innerHTML = '<p id="img-hint">▶ 큰 점을 클릭하면<br>GT vs decoded 비교가 표시됩니다.</p>' +
                    '<img id="cmp-img" />' +
                    '<p id="img-label"></p>';
  wrapper.appendChild(panel);

  plotDiv.on('plotly_click', function(data) {
    var pt = data.points[0];
    if (!pt.customdata || pt.customdata.length < 3) return;
    var ep  = pt.customdata[0];
    var si  = pt.customdata[1];
    var b64 = pt.customdata[2];
    if (typeof b64 !== 'string' || b64.length < 10) return;
    var img   = document.getElementById('cmp-img');
    var hint  = document.getElementById('img-hint');
    var label = document.getElementById('img-label');
    img.src = 'data:image/png;base64,' + b64;
    img.style.display = 'block';
    hint.style.display = 'none';
    label.textContent = 'Episode ' + ep + '  |  Skill ' + si;
  });
});
</script>
"""
    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
    html_str = html_str.replace("</body>", click_js + "\n</body>")
    return html_str


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    rng = np.random.default_rng(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path).parent / "decoder_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    model, cfg = load_model(args.model_path, device)
    print(f"  latent_dim={cfg.latent_dim}  action_dim={cfg.action_dim}  state_dim={cfg.state_dim}")

    print(f"Loading t-SNE coords: {args.tsne_coords}")
    coords = np.load(args.tsne_coords)
    xy            = coords["xy"]
    episode_ids   = coords["episode_id"]
    skill_idxs    = coords["skill_index"]
    cluster_labels = coords["cluster"] if "cluster" in coords else np.zeros(len(xy), dtype=int)
    task_ids      = coords["task_id"] if "task_id" in coords else None

    print(f"Loading skill index: {args.skills_dir}")
    skill_index = build_skill_index(Path(args.skills_dir))
    print(f"  {len(skill_index)} skills indexed")

    # ── 클러스터별 샘플링 ─────────────────────────────────────────────────────
    unique_clusters = sorted(np.unique(cluster_labels))
    sampled_indices = {}
    for cl in unique_clusters:
        idxs = np.where(cluster_labels == cl)[0]
        chosen = rng.choice(idxs, size=min(args.skills_per_cluster, len(idxs)), replace=False)
        sampled_indices[cl] = chosen.tolist()

    all_sampled = [i for idxs in sampled_indices.values() for i in idxs]
    print(f"Sampled {len(all_sampled)} skills ({args.skills_per_cluster}/cluster × {len(unique_clusters)} clusters)")

    # ── Encode → Decode → base64 PNG ─────────────────────────────────────────
    img_map: dict[int, str] = {}
    mse_list, gt_lengths, pred_lengths = [], [], []
    task_mses: dict[int, list[float]] = {}

    for gi in all_sampled:
        key = (int(episode_ids[gi]), int(skill_idxs[gi]))
        if key not in skill_index:
            continue
        skill = skill_index[key]
        gt, init_state = skill["actions"], skill["init_state"]
        task_id = skill["task_id"]

        z    = model.encode_numpy(gt, device=device)
        pred = model.decode_numpy(z, init_state, device=device)

        T_min = min(len(gt), len(pred))
        dim_mse = np.mean((gt[:T_min] - pred[:T_min]) ** 2, axis=0)
        mse = float(dim_mse.mean())

        mse_list.append(dim_mse)
        gt_lengths.append(len(gt))
        pred_lengths.append(len(pred))
        task_mses.setdefault(task_id, []).append(mse)

        b64 = make_comparison_b64(gt, pred, key[0], key[1], task_id, mse)
        img_map[gi] = b64

    mse_per_dim  = np.stack(mse_list)
    overall_mse  = float(mse_per_dim.mean())
    print(f"\nOverall MSE : {overall_mse:.4f}")
    for name, val in zip(ACTION_DIM_NAMES[:cfg.action_dim], mse_per_dim.mean(axis=0)):
        print(f"  {name:8s}: {val:.4f}")

    # ── Interactive HTML ──────────────────────────────────────────────────────
    html_str = None
    if not args.no_html:
        print("Building interactive plot ...")
        html_str = make_interactive_decoder_plotly(
            xy, cluster_labels, episode_ids, skill_idxs, task_ids,
            img_map, args.point_size, args.alpha,
        )
        html_path = output_dir / "decoder_interactive.html"
        with open(str(html_path), "w") as f:
            f.write(html_str)
        print(f"Saved → {html_path}")
    else:
        print("Skipping interactive HTML (--no_html)")

    # ── Summary plots ─────────────────────────────────────────────────────────
    fig_box  = plot_mse_boxplot(mse_per_dim, cfg.action_dim)
    fig_len  = plot_length_scatter(gt_lengths, pred_lengths)
    fig_task = plot_task_mse(task_mses)

    box_path  = output_dir / "mse_boxplot.png"
    len_path  = output_dir / "length_scatter.png"
    task_path = output_dir / "task_mse.png"
    fig_box.savefig(str(box_path),  dpi=120, bbox_inches="tight")
    fig_len.savefig(str(len_path),  dpi=120, bbox_inches="tight")
    fig_task.savefig(str(task_path), dpi=120, bbox_inches="tight")
    plt.close("all")

    # ── WandB ─────────────────────────────────────────────────────────────────
    if args.wandb_project:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_path": args.model_path,
                "n_sampled": len(img_map),
                "latent_dim": cfg.latent_dim,
                "overall_mse": overall_mse,
                "skills_per_cluster": args.skills_per_cluster,
            },
        )
        log_dict = {
            "decoder/mse_boxplot":    wandb.Image(str(box_path)),
            "decoder/length_scatter": wandb.Image(str(len_path)),
            "decoder/task_mse":       wandb.Image(str(task_path)),
            "metrics/overall_mse":    overall_mse,
            "metrics/mean_gt_len":    float(np.mean(gt_lengths)),
            "metrics/mean_pred_len":  float(np.mean(pred_lengths)),
        }
        if html_str is not None:
            log_dict["decoder/interactive"] = wandb.Html(html_str)
        for name, val in zip(ACTION_DIM_NAMES[:cfg.action_dim], mse_per_dim.mean(axis=0)):
            log_dict[f"metrics/mse_{name}"] = float(val)
        run.log(log_dict)
        run.finish()
        print(f"Logged to wandb '{args.wandb_project}'")


if __name__ == "__main__":
    main(tyro.cli(Args))
