"""
VQ-VAE Decoder evaluation.

Metrics (per-skill and per-codebook-entry):
  - Delta pose MSE      : MSE in raw decoder target units (EEF delta + gripper)
  - End timing error    : first step with sigmoid(end_logit) >= threshold vs GT
                          last step; falls back to argmax if threshold is missed
                          (signed: negative = early, positive = late)

Output:
  - wandb scalars + interactive HTML (dual bar chart, click entry for details)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from spline_vqae import SplineVQAE, SplineVQAEConfig
from train_vae import load_skill_files, load_skill_image_features, load_skill_images


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str, device: str) -> tuple[SplineVQAE, SplineVQAEConfig]:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg: SplineVQAEConfig = ckpt["cfg"]
    model = SplineVQAE(
        action_dim       = cfg.action_dim,
        state_dim        = cfg.state_dim,
        n_control        = cfg.n_control,
        spline_degree    = cfg.spline_degree,
        hidden_dim       = cfg.hidden_dim,
        latent_dim       = cfg.latent_dim,
        num_embeddings   = cfg.num_embeddings,
        num_layers       = cfg.num_layers,
        dropout          = 0.0,
        encoder_type     = cfg.encoder_type,
        decoder_rnn_type = cfg.decoder_rnn_type,
        commitment_cost  = cfg.commitment_cost,
        use_images       = cfg.use_images,
        image_model_name = cfg.image_model_name,
        image_feature_dim= cfg.image_feature_dim,
        image_size       = cfg.image_size,
        action_min       = cfg.action_min,
        action_max       = cfg.action_max,
        delta_min        = cfg.delta_min,
        delta_max        = cfg.delta_max,
        max_length       = cfg.max_length,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[EVAL] Loaded model epoch={ckpt.get('epoch','?')}  "
          f"codebook={cfg.num_embeddings}  latent={cfg.latent_dim}  use_images={cfg.use_images}")
    return model, cfg


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_decode_single(
    model: SplineVQAE,
    token: int,
    states: np.ndarray,      # (T, state_dim)
    image_feat: np.ndarray | None,  # (T, feat_dim) or None
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pred_deltas (T, delta_dim), pred_end_probs (T,))."""
    z_q = model.quantizer.embedding.weight[token].unsqueeze(0).to(device)  # (1, latent_dim)

    states_t = torch.from_numpy(states).float().unsqueeze(0).to(device)    # (1, T, state_dim)
    if image_feat is not None:
        img_np = image_feat.astype(np.float32)
        if img_np.ndim == 4 and img_np.shape[-1] in (1, 3):
            img_np = np.transpose(img_np / 255.0, (0, 3, 1, 2))  # (T,C,H,W)
        img_t = torch.from_numpy(img_np).float().unsqueeze(0).to(device)
    else:
        img_t = None
        if model.use_images:
            img_t = torch.zeros(
                1,
                len(states),
                model.image_feature_dim,
                dtype=torch.float32,
                device=device,
            )

    pred_d, pred_e = model.decode(z_q, states_t, img_t)
    pred_deltas    = pred_d.squeeze(0).cpu().numpy()          # (T, delta_dim)
    pred_end_probs = torch.sigmoid(pred_e).squeeze(0).cpu().numpy()  # (T,)
    return pred_deltas, pred_end_probs


def compute_skill_metrics(
    pred_deltas: np.ndarray,    # (T, delta_dim)
    gt_deltas: np.ndarray,      # (T, delta_dim) raw units
    pred_end_probs: np.ndarray, # (T,)
    gripper_dim_idx: int,       # which dim is gripper (-1 = last)
    end_threshold: float,
) -> dict:
    T, D = gt_deltas.shape
    gdim = (D + gripper_dim_idx) % D

    err2 = (pred_deltas - gt_deltas) ** 2
    mse_all      = float(np.mean(err2))
    mse_per_dim  = [float(np.mean(err2[:, d])) for d in range(D)]
    eef_mask     = [i for i in range(D) if i != gdim]
    mse_eef      = float(np.mean(err2[:, eef_mask]))
    mse_grip     = float(np.mean(err2[:, gdim]))

    # end timing
    gt_end_t = T - 1
    threshold_hits = np.flatnonzero(pred_end_probs >= end_threshold)
    missed_end = len(threshold_hits) == 0
    pred_end_t = int(threshold_hits[0]) if not missed_end else int(np.argmax(pred_end_probs))
    timing_err = pred_end_t - gt_end_t           # signed (neg=early, pos=late)

    target_end = np.zeros_like(pred_end_probs, dtype=np.float32)
    target_end[gt_end_t] = 1.0
    prob = np.clip(pred_end_probs, 1e-8, 1.0 - 1e-8)
    end_bce = float(np.mean(-(target_end * np.log(prob) + (1.0 - target_end) * np.log(1.0 - prob))))
    pred_bin = pred_end_probs >= end_threshold
    end_acc = float(np.mean(pred_bin == (target_end > 0.5)))

    return {
        "mse_all":     mse_all,
        "mse_per_dim": mse_per_dim,
        "mse_eef":     mse_eef,
        "mse_grip":    mse_grip,
        "timing_err":  timing_err,
        "timing_abs":  abs(timing_err),
        "end_bce":     end_bce,
        "end_acc":     end_acc,
        "end_missed":  float(missed_end),
        "pred_end_t":  pred_end_t,
        "length":      T,
    }


# ── trajectory plot ───────────────────────────────────────────────────────────

def make_skill_plot(traj: dict, dim_labels: list[str], skill_idx: int) -> str:
    """Render one skill's pred vs GT delta — 7 dims stacked vertically; return base64 JPEG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = len(dim_labels)
    fig, axes = plt.subplots(D, 1, figsize=(3.0, 1.2 * D), squeeze=False)
    t = np.arange(len(traj["gt"]))

    for d, (ax, label) in enumerate(zip(axes[:, 0], dim_labels)):
        ax.plot(t, traj["gt"][:, d],   color="#1976D2", linewidth=0.9, label="GT")
        ax.plot(t, traj["pred"][:, d], color="#D32F2F", linewidth=0.9, linestyle="--", label="Pred")
        ax.set_ylabel(label, fontsize=7, rotation=0, labelpad=28)
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        if d < D - 1:
            ax.set_xticks([])

    axes[0, 0].set_title(f"skill {skill_idx}", fontsize=8)
    axes[0, 0].legend(fontsize=6, loc="upper right", framealpha=0.6)
    fig.tight_layout(h_pad=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=70, bbox_inches="tight", pil_kwargs={"quality": 55})
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def make_entry_plots(skill_trajs: list[dict], dim_labels: list[str]) -> list[str]:
    """Generate one plot per skill; return list of base64 JPEGs."""
    return [make_skill_plot(tr, dim_labels, i) for i, tr in enumerate(skill_trajs)]


# ── HTML generation ───────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Decoder Eval</title>
<style>
  body  {{ font-family:sans-serif; background:#f5f5f5; margin:0; padding:12px; font-size:13px; }}
  h2    {{ margin:0 0 4px; font-size:15px; }}
  .summary {{ color:#555; margin-bottom:10px; font-size:12px; }}
  .chart-box {{
    background:#fff; border:1px solid #ddd; border-radius:6px;
    padding:10px; margin-bottom:10px; overflow-x:auto;
  }}
  .chart-box h4 {{ margin:0 0 6px; font-size:13px; color:#333; }}
  canvas {{ display:block; cursor:pointer; }}
  #panel {{
    background:#fff; border:1px solid #ddd; border-radius:6px;
    padding:10px; display:none; margin-top:4px;
  }}
  #panel h4 {{ margin:0 0 8px; }}
  table {{ border-collapse:collapse; font-size:12px; width:100%; }}
  th,td {{ border:1px solid #e0e0e0; padding:4px 8px; text-align:right; }}
  th {{ background:#f0f0f0; text-align:center; font-weight:600; }}
  td:first-child {{ text-align:left; }}
  .good {{ color:#2e7d32; }} .bad {{ color:#c62828; }}
  #skill-imgs {{
    display:flex; flex-wrap:nowrap; overflow-x:auto;
    gap:6px; margin-top:10px; padding-bottom:4px;
  }}
  .skill-img-wrap {{ flex-shrink:0; text-align:center; font-size:9px; color:#666; }}
  .skill-img-wrap img {{ display:block; width:120px; }}
</style>
</head>
<body>
<h2>VQ-VAE Decoder Evaluation</h2>
<div class="summary">{summary}</div>

<div class="chart-box">
  <h4>Delta Pose MSE per Codebook Entry &nbsp;<span style="font-weight:normal;font-size:11px;color:#888">(raw units, lower=better)</span></h4>
  <canvas id="c1"></canvas>
</div>
<div class="chart-box">
  <h4>|End Timing Error| per Codebook Entry &nbsp;<span style="font-weight:normal;font-size:11px;color:#888">(steps, lower=better)</span></h4>
  <canvas id="c2"></canvas>
</div>
<div id="panel">
  <h4 id="panel-title"></h4>
  <table id="panel-table"></table>
  <div id="skill-imgs"></div>
</div>

<script>
const ENTRY_DATA  = {entry_data_json};
const ENTRY_IMGS  = {entry_imgs_json};
const COUNTS      = {counts_json};
const DIM_LABELS  = {dim_labels_json};
const N           = COUNTS.length;

function makeCanvas(id, values, color, labelFn) {{
  const canvas = document.getElementById(id);
  const ctx    = canvas.getContext('2d');
  const BAR_W  = Math.max(2, Math.min(8, Math.floor(1400/N)));
  const PL=44, PR=10, PT=20, PB=30;
  canvas.width  = PL + N*BAR_W + PR;
  canvas.height = 180;

  const maxV = Math.max(...values.filter(v=>v!=null), 1e-9);
  const H    = canvas.height - PT - PB;

  function draw(sel) {{
    ctx.clearRect(0,0,canvas.width,canvas.height);
    // grid + axis
    ctx.strokeStyle='#bbb'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(PL,PT); ctx.lineTo(PL,PT+H); ctx.stroke();
    ctx.fillStyle='#888'; ctx.font='10px sans-serif'; ctx.textAlign='right';
    [0, maxV/2, maxV].forEach(v=>{{
      const y=PT+H-(v/maxV)*H;
      ctx.fillText(labelFn(v), PL-3, y+3);
      ctx.strokeStyle='#eee'; ctx.beginPath(); ctx.moveTo(PL,y); ctx.lineTo(canvas.width-PR,y); ctx.stroke();
    }});
    values.forEach((v,i)=>{{
      if(v==null) return;
      const x=PL+i*BAR_W, h=(v/maxV)*H, y=PT+H-h;
      ctx.fillStyle = (i===sel) ? '#f44336' : color;
      ctx.fillRect(x,y,BAR_W-1,h);
    }});
    ctx.fillStyle='#888'; ctx.textAlign='center'; ctx.font='9px sans-serif';
    for(let i=0;i<N;i+=50) ctx.fillText(i, PL+i*BAR_W+BAR_W/2, PT+H+12);
  }}

  let sel=-1;
  draw(sel);
  canvas.addEventListener('click', e=>{{
    const rect=canvas.getBoundingClientRect();
    const i=Math.floor((e.clientX-rect.left-PL)/BAR_W);
    if(i<0||i>=N||values[i]==null) return;
    sel=i; draw(sel); showPanel(i);
  }});
  return draw;
}}

const mseVals    = Array.from({{length:N}}, (_,i)=> ENTRY_DATA[i] ? ENTRY_DATA[i].mse_all : null);
const timingVals = Array.from({{length:N}}, (_,i)=> ENTRY_DATA[i] ? ENTRY_DATA[i].timing_abs : null);

makeCanvas('c1', mseVals,    '#1976D2', v=>v.toExponential(1));
makeCanvas('c2', timingVals, '#7B1FA2', v=>v.toFixed(1));

function showPanel(tok) {{
  const d = ENTRY_DATA[tok];
  if(!d) return;
  document.getElementById('panel-title').textContent =
    `Entry ${{tok}} — ${{COUNTS[tok]}} skills`;
  const dimRows = d.mse_per_dim.map((v,i) =>
    [`&nbsp;&nbsp;MSE ${{DIM_LABELS[i]}}`, v.toExponential(3)]);
    const rows = [
    ['# skills',               COUNTS[tok]],
    ['Delta MSE (all)',         d.mse_all.toExponential(3)],
    ...dimRows,
    ['|Timing error| (steps)', d.timing_abs.toFixed(2)],
    ['Timing error (signed)',  d.timing_err_signed.toFixed(2)],
    ['End accuracy',           d.end_acc.toFixed(3)],
    ['End threshold missed',   (100*d.end_missed).toFixed(1)+'%'],
    ['End BCE',                d.end_bce.toFixed(3)],
    ['Mean skill length',      d.mean_len.toFixed(1)],
  ];
  const tbl = document.getElementById('panel-table');
  tbl.innerHTML = '<tr><th>Metric</th><th>Value</th></tr>' +
    rows.map(([k,v])=>`<tr><td>${{k}}</td><td>${{v}}</td></tr>`).join('');
  const container = document.getElementById('skill-imgs');
  const imgs = ENTRY_IMGS[tok] || [];
  container.innerHTML = imgs.map((b64, i) =>
    `<div class="skill-img-wrap">sk${{i}}<img src="data:image/jpeg;base64,${{b64}}"></div>`
  ).join('');
  document.getElementById('panel').style.display='block';
  document.getElementById('panel').scrollIntoView({{behavior:'smooth',block:'nearest'}});
}}
</script>
</body>
</html>
"""


def build_html(
    per_skill: list[dict],
    per_skill_trajs: list[dict],
    tokens: np.ndarray,
    num_embeddings: int,
    summary_str: str,
    dim_labels: list[str],
    max_plot_samples: int,
) -> str:
    D = len(dim_labels)
    counts = [0] * num_embeddings
    for tok in tokens:
        counts[int(tok)] += 1

    entry_metric_lists: dict[int, list[dict]] = defaultdict(list)
    entry_traj_lists:   dict[int, list[dict]] = defaultdict(list)
    for m, tr, tok in zip(per_skill, per_skill_trajs, tokens):
        entry_metric_lists[int(tok)].append(m)
        entry_traj_lists[int(tok)].append(tr)

    entry_data: dict[int, dict]       = {}
    entry_imgs: dict[int, list[str]]  = {}
    active = sorted(entry_metric_lists)
    for tok in tqdm(active, desc="Rendering plots"):
        ms   = entry_metric_lists[tok]
        trs  = entry_traj_lists[tok][:max_plot_samples]
        entry_data[tok] = {
            "mse_all":           float(np.mean([m["mse_all"]   for m in ms])),
            "mse_per_dim":       [float(np.mean([m["mse_per_dim"][d] for m in ms])) for d in range(D)],
            "timing_abs":        float(np.mean([m["timing_abs"] for m in ms])),
            "timing_err_signed": float(np.mean([m["timing_err"] for m in ms])),
            "end_bce":           float(np.mean([m["end_bce"]   for m in ms])),
            "end_acc":           float(np.mean([m["end_acc"]   for m in ms])),
            "end_missed":        float(np.mean([m["end_missed"] for m in ms])),
            "mean_len":          float(np.mean([m["length"]    for m in ms])),
        }
        entry_imgs[tok] = make_entry_plots(trs, dim_labels)

    entry_data_indexed = [entry_data.get(i) for i in range(num_embeddings)]
    entry_imgs_indexed = [entry_imgs.get(i)  for i in range(num_embeddings)]

    return _HTML.format(
        summary          = summary_str,
        entry_data_json  = json.dumps(entry_data_indexed),
        entry_imgs_json  = json.dumps(entry_imgs_indexed),
        counts_json      = json.dumps(counts),
        dim_labels_json  = json.dumps(dim_labels),
    )


# ── wandb logging ─────────────────────────────────────────────────────────────

def log_wandb(per_skill: list[dict], tokens: np.ndarray, html_path: Path, args, dim_labels: list[str]) -> None:
    import wandb

    wandb.init(
        project = args.wandb_project,
        name    = args.wandb_run_name or Path(args.model_path).stem + "_decoder_eval",
        config  = vars(args),
        resume  = "allow",
    )

    mse_all      = [m["mse_all"]    for m in per_skill]
    mse_per_dim  = np.array([m["mse_per_dim"] for m in per_skill])  # (N, D)
    timing       = [m["timing_err"] for m in per_skill]
    timing_a     = [m["timing_abs"] for m in per_skill]
    end_bce      = [m["end_bce"]    for m in per_skill]
    end_acc      = [m["end_acc"]    for m in per_skill]
    end_missed   = [m["end_missed"] for m in per_skill]

    log_dict = {
        "eval/delta_mse_mean":      float(np.mean(mse_all)),
        "eval/delta_mse_std":       float(np.std(mse_all)),
        "eval/end_timing_err_mean": float(np.mean(timing)),
        "eval/end_timing_abs_mean": float(np.mean(timing_a)),
        "eval/end_timing_abs_std":  float(np.std(timing_a)),
        "eval/end_bce_mean":        float(np.mean(end_bce)),
        "eval/end_acc_mean":        float(np.mean(end_acc)),
        "eval/end_missed_frac":     float(np.mean(end_missed)),
        "eval/end_early_frac":      float(np.mean(np.array(timing) < 0)),
        "eval/end_late_frac":       float(np.mean(np.array(timing) > 0)),
        "eval/end_exact_frac":      float(np.mean(np.array(timing) == 0)),
        "eval/num_skills":          len(per_skill),
    }
    for d, label in enumerate(dim_labels):
        log_dict[f"eval/delta_mse_{label}"] = float(np.mean(mse_per_dim[:, d]))

    wandb.log(log_dict)
    wandb.log({"eval/visualizer": wandb.Html(str(html_path), inject=False)})
    wandb.finish()
    print(f"[wandb] logged to project '{args.wandb_project}'")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",     required=True)
    p.add_argument("--latents_path",   required=True,
                   help="skill_latents npz (provides tokens + metadata)")
    p.add_argument("--skills_dir",     required=True,
                   help="Directory of per-skill npz files")
    # images (pick one)
    p.add_argument("--image_features_path", default="",
                   help="Precomputed DINO features npz")
    p.add_argument("--dataset_dir", default="",
                   help="Raw video dataset dir (fallback if no precomputed features)")
    p.add_argument("--image_key", default="observation.images.image")
    # encoder args (must match training)
    p.add_argument("--eef_dims",       type=int, nargs="+", default=[0,1,2,3,4,5])
    p.add_argument("--gripper_action_dim", type=int, default=-1)
    p.add_argument("--zero_start_eef", action=argparse.BooleanOptionalAction, default=True)
    # misc
    p.add_argument("--device",         default="cuda")
    p.add_argument("--batch_size",     type=int, default=1,
                   help="Skills processed at once (keep 1 for variable-length simplicity)")
    p.add_argument("--max_plot_samples", type=int, default=50,
                   help="Max skills to plot per codebook entry")
    p.add_argument("--output_html",    default="")
    p.add_argument("--end_threshold",  type=float, default=None)
    p.add_argument("--wandb_project",  default="VAE_eval")
    p.add_argument("--wandb_run_name", default="")
    p.add_argument("--no_wandb",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # ── load model ──
    model, cfg = load_model(args.model_path, device)
    end_threshold = cfg.end_threshold if args.end_threshold is None else args.end_threshold

    # ── load skill data ──
    print("[EVAL] Loading skill files …")
    segments, decoder_states, decoder_targets, metadata, skill_orders = load_skill_files(
        Path(args.skills_dir),
        eef_dims          = args.eef_dims,
        gripper_action_dim= args.gripper_action_dim,
        zero_start_eef    = args.zero_start_eef,
    )
    N = len(segments)
    delta_dim  = decoder_targets[0].shape[-1]
    eef_count  = delta_dim - 1
    dim_labels = [f"d{i}" for i in range(eef_count)] + ["grip"]

    # ── load images ──
    images = None
    if cfg.use_images:
        if args.image_features_path:
            images = load_skill_image_features(Path(args.image_features_path), metadata)
            print(f"[EVAL] Using precomputed DINO features: {args.image_features_path}")
        elif args.dataset_dir:
            images = load_skill_images(Path(args.dataset_dir), metadata, args.image_key)
            print(f"[EVAL] Loaded raw images from {args.dataset_dir}")
        else:
            print("[EVAL] WARNING: model uses images but no image source given. "
                  "Decoder will use zero image features — metrics will be pessimistic.")

    # ── load tokens from latents npz ──
    lat = np.load(args.latents_path)
    tokens = lat["tokens"].astype(np.int32)  # (N,)
    assert len(tokens) == N, f"token count {len(tokens)} != skill count {N}"

    # ── run inference ──
    print(f"[EVAL] Running decoder on {N} skills …")
    per_skill:       list[dict] = []
    per_skill_trajs: list[dict] = []

    for i in tqdm(range(N)):
        tok       = int(tokens[i])
        states_i  = decoder_states[i]                   # (T, state_dim)
        gt_d_i    = decoder_targets[i]                  # (T, delta_dim) raw units
        img_i     = images[i] if images is not None else None

        pred_d, pred_p = run_decode_single(model, tok, states_i, img_i, device)
        T = len(states_i)
        pred_d = pred_d[:T]
        pred_p = pred_p[:T]

        metrics = compute_skill_metrics(pred_d, gt_d_i, pred_p, gripper_dim_idx=-1, end_threshold=end_threshold)
        per_skill.append(metrics)
        per_skill_trajs.append({"gt": gt_d_i, "pred": pred_d})

    # ── print summary ──
    mse_all     = np.mean([m["mse_all"]    for m in per_skill])
    mse_per_dim = np.mean(np.array([m["mse_per_dim"] for m in per_skill]), axis=0)
    t_abs       = np.mean([m["timing_abs"] for m in per_skill])
    t_mean      = np.mean([m["timing_err"] for m in per_skill])
    t_std       = np.std( [m["timing_err"] for m in per_skill])
    end_bce     = np.mean([m["end_bce"]    for m in per_skill])
    end_acc     = np.mean([m["end_acc"]    for m in per_skill])
    missed      = np.mean([m["end_missed"] for m in per_skill])
    early       = np.mean([m["timing_err"] < 0 for m in per_skill])
    late        = np.mean([m["timing_err"] > 0 for m in per_skill])

    per_dim_str = "  ".join(f"{label}={v:.4f}" for label, v in zip(dim_labels, mse_per_dim))
    summary = (
        f"N={N}  |  ΔPose MSE={mse_all:.4f}  [{per_dim_str}]  |  "
        f"End@{end_threshold:.2f} |err|={t_abs:.2f}steps  mean={t_mean:+.2f}±{t_std:.2f}  "
        f"early={early:.1%}  late={late:.1%} missed={missed:.1%}  |  "
        f"EndAcc={end_acc:.3f} EndBCE={end_bce:.3f}"
    )
    print(f"\n[EVAL] {summary}")

    # ── generate HTML ──
    if args.output_html:
        html_path = Path(args.output_html)
    else:
        html_path = Path(args.latents_path).with_suffix(".decoder_eval.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(per_skill, per_skill_trajs, tokens, cfg.num_embeddings, summary, dim_labels, args.max_plot_samples)
    html_path.write_text(html, encoding="utf-8")
    print(f"[EVAL] HTML saved → {html_path}")

    if not args.no_wandb:
        log_wandb(per_skill, tokens, html_path, args, dim_labels)


if __name__ == "__main__":
    main()
