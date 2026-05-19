"""
FSQ-AE Decoder evaluation.

Metrics per skill and per FSQ codebook index (0..124):
  - Delta pose MSE (first-step): immediate next-action prediction accuracy
  - Chunk MSE (chunk mode only): avg MSE across all K future steps in the chunk
  - End timing error: first step where sigmoid(end_logit) >= threshold vs GT (signed, steps)

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
from FSQ import SplineFSQAE, SplineFSQAEConfig
from train_FSQ import load_skill_files, load_dino_tokens, load_patch_flags


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str, device: str) -> tuple[SplineFSQAE, SplineFSQAEConfig]:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg: SplineFSQAEConfig = ckpt["cfg"]
    model = SplineFSQAE(
        action_dim           = cfg.action_dim,
        state_dim            = cfg.state_dim,
        n_control            = cfg.n_control,
        spline_degree        = cfg.spline_degree,
        hidden_dim           = cfg.hidden_dim,
        fsq_levels           = cfg.fsq_levels,
        num_layers           = cfg.num_layers,
        dropout              = 0.0,
        feat_dim             = cfg.feat_dim,
        n_tokens             = cfg.n_tokens,
        image_model_name     = getattr(cfg, "image_model_name", "/data2/dohyeon/SBD/models/dinov3-vits16"),
        image_size           = getattr(cfg, "image_size", 224),
        patch_grid           = getattr(cfg, "patch_grid", 8),
        n_patch_raw          = getattr(cfg, "n_patch_raw", 196),
        decoder_image_mode   = getattr(cfg, "decoder_image_mode", "dino_flags"),
        image_encoder_layers = getattr(cfg, "image_encoder_layers", 1),
        image_encoder_heads  = getattr(cfg, "image_encoder_heads", 4),
        decoder_output_mode  = cfg.decoder_output_mode,
        chunk_size           = cfg.chunk_size,
        max_length           = cfg.max_length,
        action_min           = cfg.action_min,
        action_max           = cfg.action_max,
        delta_min            = cfg.delta_min,
        delta_max            = cfg.delta_max,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    codebook_size = model.fsq.codebook_size
    print(f"[EVAL] Loaded FSQ model  fsq_levels={cfg.fsq_levels}  "
          f"codebook={codebook_size}  mode={cfg.decoder_output_mode}"
          + (f"  K={cfg.chunk_size}" if cfg.decoder_output_mode == "chunk" else ""))
    return model, cfg


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_decode_single(
    model: SplineFSQAE,
    z_q: np.ndarray,           # (D,)  FSQ latent vector
    states: np.ndarray,        # (T, state_dim)
    dec_tokens: np.ndarray,    # (T, n_tokens, feat_dim)
    patch_flags: np.ndarray,   # (T, n_patches, 2)
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pred_deltas, pred_end_probs).

    single_step mode: pred_deltas (T, A)
    chunk mode:       pred_deltas (T, K, A)
    pred_end_probs:   (T,) in single_step, (T, K) in chunk after sigmoid
    """
    T = len(states)
    z_t  = torch.from_numpy(z_q.astype(np.float32)).unsqueeze(0).to(device)
    s_t  = torch.from_numpy(states.astype(np.float32)).unsqueeze(0).to(device)
    d_t  = torch.from_numpy(dec_tokens[:T].astype(np.float32)).unsqueeze(0).to(device)
    p_t  = torch.from_numpy(patch_flags.astype(np.float32)).unsqueeze(0).to(device)
    progress = (torch.arange(T, dtype=torch.float32) / model.max_length).unsqueeze(0).to(device)

    pred_d, pred_e = model.decode(z_t, s_t, d_t, p_t, progress)
    return pred_d.squeeze(0).cpu().numpy(), torch.sigmoid(pred_e).squeeze(0).cpu().numpy()


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_skill_metrics(
    pred_deltas: np.ndarray,    # (T, A) or (T, K, A)
    gt_deltas: np.ndarray,      # (T, A)
    pred_end_probs: np.ndarray, # (T,) or (T, K)
    end_threshold: float,
) -> dict:
    T, A = gt_deltas.shape
    gdim = A - 1  # gripper is last dim

    # primary metric: immediate next-action (first step of each chunk)
    pred_d0 = pred_deltas[:, 0, :] if pred_deltas.ndim == 3 else pred_deltas

    err2 = (pred_d0 - gt_deltas) ** 2
    mse_all     = float(np.mean(err2))
    mse_per_dim = [float(np.mean(err2[:, d])) for d in range(A)]
    mse_eef     = float(np.mean(err2[:, :gdim]))
    mse_grip    = float(np.mean(err2[:, gdim]))

    # chunk MSE: compare pred_deltas[t, k] vs gt_deltas[t+k] for all valid (t,k)
    chunk_mse = None
    if pred_deltas.ndim == 3:
        K = pred_deltas.shape[1]
        t_idx = np.arange(T)[:, None] + np.arange(K)[None, :]  # (T, K)
        valid = (t_idx < T)                                      # (T, K)
        gt_exp = gt_deltas[np.minimum(t_idx, T - 1)]            # (T, K, A)
        chunk_err = ((pred_deltas - gt_exp) ** 2) * valid[..., None]
        n_valid = valid.sum() * A
        chunk_mse = float(chunk_err.sum() / n_valid) if n_valid > 0 else 0.0

    # end timing
    gt_end_t = T - 1
    if pred_end_probs.ndim == 2:
        K = pred_end_probs.shape[1]
        abs_t = np.arange(T)[:, None] + np.arange(K)[None, :]
        hits_2d = np.argwhere(pred_end_probs >= end_threshold)
        missed = len(hits_2d) == 0
        if missed:
            best = np.unravel_index(int(np.argmax(pred_end_probs)), pred_end_probs.shape)
            pred_end_t = int(abs_t[best])
        else:
            hit_abs = abs_t[hits_2d[:, 0], hits_2d[:, 1]]
            pred_end_t = int(hit_abs.min())
    else:
        hits = np.flatnonzero(pred_end_probs >= end_threshold)
        missed = len(hits) == 0
        pred_end_t = int(hits[0]) if not missed else int(np.argmax(pred_end_probs))
    timing_err = pred_end_t - gt_end_t

    prob = np.clip(pred_end_probs, 1e-8, 1.0 - 1e-8)
    if pred_end_probs.ndim == 2:
        K = pred_end_probs.shape[1]
        tgt_idx = np.arange(T)[:, None] + np.arange(K)[None, :]
        tgt = (tgt_idx == gt_end_t).astype(np.float32)
    else:
        tgt  = np.zeros(T, np.float32); tgt[gt_end_t] = 1.0
    end_bce = float(np.mean(-(tgt * np.log(prob) + (1.0 - tgt) * np.log(1.0 - prob))))
    end_acc = float(np.mean((pred_end_probs >= end_threshold) == (tgt > 0.5)))

    result = {
        "mse_all":     mse_all,
        "mse_per_dim": mse_per_dim,
        "mse_eef":     mse_eef,
        "mse_grip":    mse_grip,
        "timing_err":  timing_err,
        "timing_abs":  abs(timing_err),
        "end_bce":     end_bce,
        "end_acc":     end_acc,
        "end_missed":  float(missed),
        "pred_end_t":  pred_end_t,
        "length":      T,
    }
    if chunk_mse is not None:
        result["chunk_mse"] = chunk_mse
    return result


# ── trajectory plot ───────────────────────────────────────────────────────────

def make_skill_plot(traj: dict, dim_labels: list[str], skill_idx: int) -> str:
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


# ── HTML generation ───────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FSQ Decoder Eval</title>
<style>
  body  {{ font-family:sans-serif; background:#f5f5f5; margin:0; padding:12px; font-size:13px; }}
  h2    {{ margin:0 0 4px; font-size:15px; }}
  .summary {{ color:#555; margin-bottom:10px; font-size:12px; }}
  .chart-box {{ background:#fff; border:1px solid #ddd; border-radius:6px; padding:10px; margin-bottom:10px; overflow-x:auto; }}
  .chart-box h4 {{ margin:0 0 6px; font-size:13px; color:#333; }}
  canvas {{ display:block; cursor:pointer; }}
  #panel {{ background:#fff; border:1px solid #ddd; border-radius:6px; padding:10px; display:none; margin-top:4px; }}
  #panel h4 {{ margin:0 0 8px; }}
  table {{ border-collapse:collapse; font-size:12px; width:100%; }}
  th,td {{ border:1px solid #e0e0e0; padding:4px 8px; text-align:right; }}
  th {{ background:#f0f0f0; text-align:center; font-weight:600; }}
  td:first-child {{ text-align:left; }}
  #skill-imgs {{ display:flex; flex-wrap:nowrap; overflow-x:auto; gap:6px; margin-top:10px; padding-bottom:4px; }}
  .skill-img-wrap {{ flex-shrink:0; text-align:center; font-size:9px; color:#666; }}
  .skill-img-wrap img {{ display:block; width:120px; }}
</style>
</head>
<body>
<h2>FSQ-AE Decoder Evaluation</h2>
<div class="summary">{summary}</div>
<div class="chart-box">
  <h4>Delta Pose MSE per Codebook Entry &nbsp;<span style="font-weight:normal;font-size:11px;color:#888">(first-step, lower=better)</span></h4>
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
const ENTRY_DATA = {entry_data_json};
const ENTRY_IMGS = {entry_imgs_json};
const COUNTS     = {counts_json};
const DIM_LABELS = {dim_labels_json};
const N          = COUNTS.length;
const IS_CHUNK   = {is_chunk_json};

function makeCanvas(id, values, color, labelFn) {{
  const canvas = document.getElementById(id);
  const ctx    = canvas.getContext('2d');
  const BAR_W  = Math.max(2, Math.min(18, Math.floor(1400/N)));
  const PL=44, PR=10, PT=20, PB=30;
  canvas.width  = PL + N*BAR_W + PR;
  canvas.height = 180;
  const maxV = Math.max(...values.filter(v=>v!=null), 1e-9);
  const H = canvas.height - PT - PB;
  function draw(sel) {{
    ctx.clearRect(0,0,canvas.width,canvas.height);
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
    for(let i=0;i<N;i+=10) ctx.fillText(i, PL+i*BAR_W+BAR_W/2, PT+H+12);
  }}
  let sel=-1; draw(sel);
  canvas.addEventListener('click', e=>{{
    const rect=canvas.getBoundingClientRect();
    const i=Math.floor((e.clientX-rect.left-PL)/BAR_W);
    if(i<0||i>=N||values[i]==null) return;
    sel=i; draw(sel); showPanel(i);
  }});
}}

const mseVals    = Array.from({{length:N}}, (_,i)=> ENTRY_DATA[i] ? ENTRY_DATA[i].mse_all : null);
const timingVals = Array.from({{length:N}}, (_,i)=> ENTRY_DATA[i] ? ENTRY_DATA[i].timing_abs : null);
makeCanvas('c1', mseVals,    '#1976D2', v=>v.toExponential(1));
makeCanvas('c2', timingVals, '#7B1FA2', v=>v.toFixed(1));

function showPanel(tok) {{
  const d = ENTRY_DATA[tok]; if(!d) return;
  document.getElementById('panel-title').textContent = `Entry ${{tok}} — ${{COUNTS[tok]}} skills`;
  const dimRows = d.mse_per_dim.map((v,i)=>[`&nbsp;&nbsp;MSE ${{DIM_LABELS[i]}}`, v.toExponential(3)]);
  const rows = [
    ['# skills',               COUNTS[tok]],
    ['Delta MSE (first-step)', d.mse_all.toExponential(3)],
    ['Delta MSE EEF',          d.mse_eef.toExponential(3)],
    ['Delta MSE Gripper',      d.mse_grip.toExponential(3)],
    ...dimRows,
    ...(IS_CHUNK && d.chunk_mse != null ? [['Chunk MSE (all K)', d.chunk_mse.toExponential(3)]] : []),
    ['|Timing error| (steps)', d.timing_abs.toFixed(2)],
    ['Timing error (signed)',  d.timing_err_signed.toFixed(2)],
    ['End accuracy',           d.end_acc.toFixed(3)],
    ['End threshold missed',   (100*d.end_missed).toFixed(1)+'%'],
    ['End BCE',                d.end_bce.toFixed(3)],
    ['Mean skill length',      d.mean_len.toFixed(1)],
  ];
  document.getElementById('panel-table').innerHTML =
    '<tr><th>Metric</th><th>Value</th></tr>' +
    rows.map(([k,v])=>`<tr><td>${{k}}</td><td>${{v}}</td></tr>`).join('');
  const imgs = ENTRY_IMGS[tok] || [];
  document.getElementById('skill-imgs').innerHTML = imgs.map((b64,i)=>
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
    codebook_size: int,
    is_chunk: bool,
    summary_str: str,
    dim_labels: list[str],
    max_plot_samples: int,
) -> str:
    D = len(dim_labels)
    counts = [0] * codebook_size
    for tok in tokens:
        counts[int(tok)] += 1

    by_entry: dict[int, list] = defaultdict(list)
    by_entry_traj: dict[int, list] = defaultdict(list)
    for m, tr, tok in zip(per_skill, per_skill_trajs, tokens):
        by_entry[int(tok)].append(m)
        by_entry_traj[int(tok)].append(tr)

    entry_data: dict[int, dict] = {}
    entry_imgs: dict[int, list[str]] = {}
    for tok in tqdm(sorted(by_entry), desc="Rendering plots"):
        ms  = by_entry[tok]
        trs = by_entry_traj[tok][:max_plot_samples]
        ed: dict = {
            "mse_all":           float(np.mean([m["mse_all"]    for m in ms])),
            "mse_per_dim":       [float(np.mean([m["mse_per_dim"][d] for m in ms])) for d in range(D)],
            "mse_eef":           float(np.mean([m["mse_eef"]    for m in ms])),
            "mse_grip":          float(np.mean([m["mse_grip"]   for m in ms])),
            "timing_abs":        float(np.mean([m["timing_abs"] for m in ms])),
            "timing_err_signed": float(np.mean([m["timing_err"] for m in ms])),
            "end_bce":           float(np.mean([m["end_bce"]    for m in ms])),
            "end_acc":           float(np.mean([m["end_acc"]    for m in ms])),
            "end_missed":        float(np.mean([m["end_missed"] for m in ms])),
            "mean_len":          float(np.mean([m["length"]     for m in ms])),
        }
        if is_chunk and "chunk_mse" in ms[0]:
            ed["chunk_mse"] = float(np.mean([m["chunk_mse"] for m in ms]))
        entry_data[tok] = ed
        entry_imgs[tok] = [make_skill_plot(tr, dim_labels, i) for i, tr in enumerate(trs)]

    return _HTML.format(
        summary         = summary_str,
        entry_data_json = json.dumps([entry_data.get(i) for i in range(codebook_size)]),
        entry_imgs_json = json.dumps([entry_imgs.get(i) for i in range(codebook_size)]),
        counts_json     = json.dumps(counts),
        dim_labels_json = json.dumps(dim_labels),
        is_chunk_json   = json.dumps(is_chunk),
    )


# ── wandb ─────────────────────────────────────────────────────────────────────

def log_wandb(per_skill: list[dict], tokens: np.ndarray, html_path: Path, is_chunk: bool, args, dim_labels: list[str]) -> None:
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name or Path(args.model_path).stem,
               config=vars(args), resume="allow")
    mse_all     = [m["mse_all"]    for m in per_skill]
    mse_eef     = [m["mse_eef"]    for m in per_skill]
    mse_grip    = [m["mse_grip"]   for m in per_skill]
    mse_per_dim = np.array([m["mse_per_dim"] for m in per_skill])
    timing      = [m["timing_err"] for m in per_skill]
    timing_a    = [m["timing_abs"] for m in per_skill]
    log_dict = {
        "eval/delta_mse_mean":      float(np.mean(mse_all)),
        "eval/delta_mse_eef":       float(np.mean(mse_eef)),
        "eval/delta_mse_grip":      float(np.mean(mse_grip)),
        "eval/end_timing_err_mean": float(np.mean(timing)),
        "eval/end_timing_abs_mean": float(np.mean(timing_a)),
        "eval/end_bce_mean":        float(np.mean([m["end_bce"]    for m in per_skill])),
        "eval/end_acc_mean":        float(np.mean([m["end_acc"]    for m in per_skill])),
        "eval/end_missed_frac":     float(np.mean([m["end_missed"] for m in per_skill])),
        "eval/end_early_frac":      float(np.mean(np.array(timing) < 0)),
        "eval/end_late_frac":       float(np.mean(np.array(timing) > 0)),
        "eval/num_skills":          len(per_skill),
    }
    for d, label in enumerate(dim_labels):
        log_dict[f"eval/delta_mse_{label}"] = float(np.mean(mse_per_dim[:, d]))
    if is_chunk and "chunk_mse" in per_skill[0]:
        log_dict["eval/chunk_mse_mean"] = float(np.mean([m["chunk_mse"] for m in per_skill]))
    wandb.log(log_dict)
    wandb.log({"eval/visualizer": wandb.Html(str(html_path), inject=False)})
    wandb.finish()
    print(f"[wandb] logged to '{args.wandb_project}'")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",      required=True)
    p.add_argument("--latents_path",    required=True,
                   help="skill_latents.npz (provides latents + tokens + metadata)")
    p.add_argument("--skills_dir",      required=True,
                   help="Directory of per-skill npz files")
    p.add_argument("--dino_features",   default="",
                   help="Precomputed DINO token npz (N_total × n_tokens × feat_dim)")
    p.add_argument("--sam2_masks_dir",  default="",
                   help="Directory with per-skill SAM2 mask npz files")
    p.add_argument("--eef_dims",        type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--gripper_action_dim", type=int, default=-1)
    p.add_argument("--zero_start_eef",  action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--max_plot_samples", type=int, default=20)
    p.add_argument("--output_html",     default="")
    p.add_argument("--end_threshold",   type=float, default=None)
    p.add_argument("--wandb_project",   default="VAE_eval")
    p.add_argument("--wandb_run_name",  default="")
    p.add_argument("--no_wandb",        action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    model, cfg = load_model(args.model_path, device)
    end_threshold = cfg.end_threshold if args.end_threshold is None else args.end_threshold
    is_chunk      = cfg.decoder_output_mode == "chunk"
    codebook_size = model.fsq.codebook_size
    n_patches     = model.n_patches

    print("[EVAL] Loading skill files …")
    segments, dec_states, dec_targets, metadata = load_skill_files(
        Path(args.skills_dir),
        eef_dims           = args.eef_dims,
        gripper_action_dim = args.gripper_action_dim,
        zero_start_eef     = args.zero_start_eef,
    )
    N = len(segments)
    A = dec_targets[0].shape[-1]
    dim_labels = [f"d{i}" for i in range(A - 1)] + ["grip"]

    print("[EVAL] Loading DINO tokens …")
    if args.dino_features:
        dec_tokens_list = load_dino_tokens(Path(args.dino_features), metadata)
    else:
        print("[EVAL] WARNING: no --dino_features given — using zero tokens")
        n_tok, F = cfg.n_tokens, cfg.feat_dim
        dec_tokens_list = [np.zeros((m["length"], n_tok, F), np.float32) for m in metadata]

    print("[EVAL] Loading SAM2 patch flags …")
    if args.sam2_masks_dir:
        patch_flags_list = load_patch_flags(Path(args.sam2_masks_dir), metadata, n_patches)
    else:
        patch_flags_list = [np.zeros((m["length"], n_patches, 2), np.float32) for m in metadata]

    lat    = np.load(args.latents_path)
    latents = lat["latents"].astype(np.float32)  # (N, D) — z_q vectors
    tokens  = lat["tokens"].astype(np.int32)      # (N,)   — FSQ indices
    assert len(tokens) == N, f"token count {len(tokens)} != skill count {N}"

    print(f"[EVAL] Running decoder on {N} skills …")
    per_skill:       list[dict] = []
    per_skill_trajs: list[dict] = []

    for i in tqdm(range(N)):
        T        = int(metadata[i]["length"])
        states_i = dec_states[i][:T]
        gt_d_i   = dec_targets[i][:T]
        z_q_i    = latents[i]                      # (D,)
        dtok_i   = dec_tokens_list[i][:T]          # (T, n_tok, F)
        pf_i     = patch_flags_list[i][:T]         # (T, n_patches, 2)

        pred_d, pred_p = run_decode_single(model, z_q_i, states_i, dtok_i, pf_i, device)

        # trim to T (padding safeguard)
        if pred_d.ndim == 3:
            pred_d = pred_d[:T]
            pred_d0 = pred_d[:, 0, :]
        else:
            pred_d = pred_d[:T]
            pred_d0 = pred_d
        pred_p = pred_p[:T]

        metrics = compute_skill_metrics(pred_d, gt_d_i, pred_p, end_threshold)
        per_skill.append(metrics)
        per_skill_trajs.append({"gt": gt_d_i, "pred": pred_d0})

    mse_all  = np.mean([m["mse_all"]    for m in per_skill])
    mse_eef  = np.mean([m["mse_eef"]    for m in per_skill])
    mse_grip = np.mean([m["mse_grip"]   for m in per_skill])
    t_abs    = np.mean([m["timing_abs"] for m in per_skill])
    t_mean   = np.mean([m["timing_err"] for m in per_skill])
    t_std    = np.std( [m["timing_err"] for m in per_skill])
    end_bce  = np.mean([m["end_bce"]    for m in per_skill])
    end_acc  = np.mean([m["end_acc"]    for m in per_skill])
    missed   = np.mean([m["end_missed"] for m in per_skill])
    early    = np.mean([m["timing_err"] < 0 for m in per_skill])
    late     = np.mean([m["timing_err"] > 0 for m in per_skill])
    chunk_str = ""
    if is_chunk and "chunk_mse" in per_skill[0]:
        chunk_str = f"  ChunkMSE={np.mean([m['chunk_mse'] for m in per_skill]):.4f}"

    mse_dims = "  ".join(f"{l}={v:.4f}" for l, v in
                         zip(dim_labels, np.mean([m["mse_per_dim"] for m in per_skill], axis=0)))
    summary = (
        f"N={N}  mode={cfg.decoder_output_mode}  |  "
        f"ΔPose MSE={mse_all:.4f} (EEF={mse_eef:.4f} grip={mse_grip:.4f})  [{mse_dims}]{chunk_str}  |  "
        f"End@{end_threshold:.2f} |err|={t_abs:.2f}steps  mean={t_mean:+.2f}±{t_std:.2f}  "
        f"early={early:.1%} late={late:.1%} missed={missed:.1%}  |  "
        f"EndAcc={end_acc:.3f} EndBCE={end_bce:.3f}"
    )
    print(f"\n[EVAL] {summary}")

    html_path = Path(args.output_html) if args.output_html else Path(args.latents_path).with_suffix(".decoder_eval.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(per_skill, per_skill_trajs, tokens, codebook_size, is_chunk, summary, dim_labels, args.max_plot_samples)
    html_path.write_text(html, encoding="utf-8")
    print(f"[EVAL] HTML → {html_path}")

    if not args.no_wandb:
        log_wandb(per_skill, tokens, html_path, is_chunk, args, dim_labels)


if __name__ == "__main__":
    main()
