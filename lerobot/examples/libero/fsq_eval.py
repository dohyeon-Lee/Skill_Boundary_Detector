"""Unified FSQ evaluation: encoder + decoder in one pass.

Produces:
  - wandb scalars
      encoder : codebook/utilization_pct, skills_per_entry_{mean,std,max}
      decoder : decoder/chunk_action_mse_mean, recon_mse_{xyz,rpy,gripper},
                termination_err_mean (|timing| steps), early_rate, late_rate,
                progress_err_mean
  - one interactive HTML: left FSQ lattice cube, right per-entry bar charts
      (skill count, recon error, termination error, progress error); clicking an
      entry shows N sample skills, each with start/end frames and stacked
      GT-vs-pred plots for per-dim reconstruction, termination, and progress.

The skill_latents.npz used here is (re)generated from the evaluated checkpoint
and written next to the HTML (FSQ_eval/outputs).

Usage:
    python examples/libero/fsq_eval.py \
      --model_path   .../FSQ.pt \
      --skills_dir   .../skillset/skills \
      --dino_features .../dino_tokens.npz \
      --dino_features_wrist .../dino_tokens_wrist.npz \
      --dataset_dir  .../libero_90 \
      --output_dir   FSQ_eval/outputs/<run>/<epoch>
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

from codebook_visualizer import (  # noqa: E402
    _clip_frame_or_blank,
    _episode_row,
    _load_episodes_meta,
    _read_episode_clip,
    _resolve_image_key,
    _video_path,
)
from decoder_eval import load_model  # noqa: E402
from FSQ import spline_encode  # noqa: E402
from train_FSQ import load_dino_tokens, load_skill_files  # noqa: E402


# ── latent saving ──────────────────────────────────────────────────────────────

def save_latents(path: Path, latents, tokens, metadata):
    save: dict = {"latents": latents, "tokens": tokens}
    for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"):
        save[key] = np.array([m[key] for m in metadata])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **save)
    print(f"[fsq_eval] saved latents → {path}")


# ── decode + metrics ───────────────────────────────────────────────────────────

def _dim_groups(action_dim: int) -> dict[str, list[int]]:
    """xyz / rpy / gripper index groups (gripper = last dim)."""
    g = {"xyz": [0, 1, 2], "rpy": [3, 4, 5], "gripper": [action_dim - 1]}
    return {k: [i for i in v if i < action_dim] for k, v in g.items()}


def skill_metrics(delta, progress, term_prob, gt_actions, T, end_threshold, groups):
    """Per-skill metrics. Reconstruction MSE is computed over the whole chunk."""
    K, A = delta.shape[1], delta.shape[2]
    T_future = len(gt_actions)
    t_idx = np.arange(T)[:, None] + np.arange(K)[None, :]      # (T, K)
    valid = t_idx < T_future
    gt_exp = gt_actions[np.minimum(t_idx, T_future - 1)]       # (T, K, A)
    err = ((delta - gt_exp) ** 2) * valid[..., None]           # (T, K, A)
    n_valid = int(valid.sum())

    def group_mse(idxs: list[int]) -> float:
        if n_valid == 0 or not idxs:
            return 0.0
        return float(err[..., idxs].sum() / (n_valid * len(idxs)))

    chunk_mse = float(err.sum() / (n_valid * A)) if n_valid > 0 else 0.0

    # termination timing: first step crossing threshold, else argmax
    gt_end = T - 1
    hits = np.flatnonzero(term_prob[:T] >= end_threshold)
    pred_end = int(hits[0]) if len(hits) else int(np.argmax(term_prob[:T]))
    timing = pred_end - gt_end

    # progress regression error
    gt_prog = np.arange(T, dtype=np.float32) / max(T - 1, 1)
    prog_err = float(np.mean(np.abs(progress[:T] - gt_prog)))

    return {
        "chunk_mse":  chunk_mse,
        "mse_xyz":    group_mse(groups["xyz"]),
        "mse_rpy":    group_mse(groups["rpy"]),
        "mse_grip":   group_mse(groups["gripper"]),
        "timing":     timing,
        "timing_abs": abs(timing),
        "prog_err":   prog_err,
        "pred_end":   pred_end,
        "length":     T,
    }


# ── batched inference (fp16 clips → per-batch float32, no_grad) ─────────────────

@torch.no_grad()
def batched_encode(model, segments, lengths, device, batch_size):
    """Encode all skills (action-only) in length-bucketed batches. Returns latents (N,D), tokens (N,)."""
    N = len(segments)
    A = segments[0].shape[-1]
    nctrl, deg = model.n_control, model.spline_degree
    amin = model.action_min.cpu().numpy()
    amax = model.action_max.cpu().numpy()

    ctrl_norm = []
    for seg in segments:
        cp, _ = spline_encode(seg.astype(np.float32), nctrl, deg)
        cp = (cp - amin) / (amax - amin + 1e-8) * 2.0 - 1.0
        ctrl_norm.append(cp.astype(np.float32))

    latents = np.zeros((N, int(model.fsq.latent_dim)), np.float32)
    tokens = np.zeros(N, np.int32)
    order = sorted(range(N), key=lambda i: lengths[i])
    for s in tqdm(range(0, N, batch_size), desc="Encoding (batched)"):
        idxs = order[s:s + batch_size]
        B = len(idxs)
        ctrl = torch.zeros(B, nctrl, A)
        lens = torch.zeros(B, dtype=torch.long)
        for b, i in enumerate(idxs):
            ctrl[b] = torch.from_numpy(ctrl_norm[i])
            lens[b] = lengths[i]
        z_q, idx = model.encode(ctrl.to(device), lens.to(device))
        z_q = z_q.cpu().numpy()
        idx = idx.cpu().numpy()
        for b, i in enumerate(idxs):
            latents[i] = z_q[b]
            tokens[i] = idx[b]
    return latents, tokens


@torch.no_grad()
def batched_decode(model, latents, states, clips, clips_wrist, lengths, device, batch_size):
    """Decode all skills in length-bucketed batches.

    Returns per-skill lists sliced to T: deltas[i] (T,K,A), progresses[i] (T,),
    term_probs[i] (T,). GT progress is fed as the motion input (matches training).
    clips = 3rd-person tokens; clips_wrist = wrist tokens (None for single-camera models,
    where the terminator reads 3rd-person only).
    """
    N = len(latents)
    n_tokens, feat = model.n_tokens, model.feat_dim
    state_dim = model.state_dim
    D = int(model.fsq.latent_dim)
    deltas: list = [None] * N
    progs: list = [None] * N
    terms: list = [None] * N
    order = sorted(range(N), key=lambda i: lengths[i])
    for s in tqdm(range(0, N, batch_size), desc="Decoding (batched)"):
        idxs = order[s:s + batch_size]
        B = len(idxs)
        maxT = max(lengths[i] for i in idxs)
        z = torch.zeros(B, D)
        st = torch.zeros(B, maxT, state_dim)
        dec = torch.zeros(B, maxT, n_tokens, feat)
        dec_w = torch.zeros(B, maxT, n_tokens, feat) if clips_wrist is not None else None
        fp = torch.zeros(B, maxT)
        for b, i in enumerate(idxs):
            T = lengths[i]
            z[b] = torch.from_numpy(latents[i].astype(np.float32))
            st[b, :T] = torch.from_numpy(states[i][:T].astype(np.float32))
            dec[b, :T] = torch.from_numpy(clips[i][:T].astype(np.float32))
            if dec_w is not None:
                dec_w[b, :T] = torch.from_numpy(clips_wrist[i][:T].astype(np.float32))
            fp[b, :T] = torch.arange(T, dtype=torch.float32) / max(T - 1, 1)
        delta, prog, term_logits = model.decode(
            z.to(device), st.to(device), dec.to(device),
            dec_w.to(device) if dec_w is not None else None, fp.to(device))
        term = torch.sigmoid(term_logits)
        for b, i in enumerate(idxs):
            T = lengths[i]
            deltas[i] = delta[b, :T].cpu().numpy()
            progs[i] = prog[b, :T].cpu().numpy()
            terms[i] = term[b, :T].cpu().numpy()
    return deltas, progs, terms


# ── per-sample composite plot (start/end frame + recon/termination/progress) ────

def make_sample_plot(start_img, end_img, delta, progress, term_prob, gt_actions, T,
                     dim_labels, n_action_steps, end_threshold) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    A = delta.shape[2]
    K = delta.shape[1]
    n_rows = A + 2  # per-dim recon + termination + progress
    n_img_rows = 1     # row0: start / end frames
    img_h = 4.0     # inches per image row (≈ half figure width → fills the row)
    row_h = 1.0
    fig = plt.figure(figsize=(8.5, img_h * n_img_rows + row_h * n_rows))
    gs = GridSpec(n_img_rows + n_rows, 2, figure=fig,
                  height_ratios=[img_h] * n_img_rows + [row_h] * n_rows, wspace=0.03)

    # row 0: start / end frames (large, filling the row width)
    ax_s = fig.add_subplot(gs[0, 0]); ax_e = fig.add_subplot(gs[0, 1])
    ax_s.imshow(start_img); ax_s.set_title("start", fontsize=10); ax_s.axis("off")
    ax_e.imshow(end_img);   ax_e.set_title("end",   fontsize=10); ax_e.axis("off")

    base = n_img_rows
    t_full = np.arange(T)
    chunk_starts = np.arange(0, T, max(1, n_action_steps))

    # per-dim reconstruction (GT vs predicted chunks)
    for d in range(A):
        ax = fig.add_subplot(gs[base + d, :])
        ax.plot(t_full, gt_actions[:T, d], color="#0D47A1", linewidth=1.5, label="GT", zorder=3)
        for j, start in enumerate(chunk_starts):
            x = start + np.arange(K)
            m = x < len(gt_actions)
            if not np.any(m):
                continue
            ax.plot(x[m], delta[start, : int(m.sum()), d], color="#B71C1C",
                    linewidth=1.4, alpha=0.9, linestyle="--",
                    label="pred chunk" if j == 0 else None, zorder=4)
        ax.set_ylabel(dim_labels[d], fontsize=8, rotation=0, labelpad=26)
        ax.tick_params(labelsize=7); ax.grid(True, color="#eee", linewidth=0.6)
        ax.set_xticks([])
        if d == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

    # termination GT vs pred
    ax_t = fig.add_subplot(gs[base + A, :])
    gt_term = np.zeros(T); gt_term[T - 1] = 1.0
    ax_t.plot(t_full, gt_term, color="#0D47A1", linewidth=1.5, label="GT end")
    ax_t.plot(t_full, term_prob[:T], color="#B71C1C", linewidth=1.4, linestyle="--", label="pred prob")
    ax_t.axhline(end_threshold, color="#888", linewidth=0.8, linestyle=":")
    ax_t.set_ylabel("term", fontsize=8, rotation=0, labelpad=26)
    ax_t.set_ylim(-0.05, 1.05); ax_t.tick_params(labelsize=7)
    ax_t.grid(True, color="#eee", linewidth=0.6); ax_t.set_xticks([])
    ax_t.legend(fontsize=7, loc="upper left", framealpha=0.8)

    # progress GT vs pred
    ax_p = fig.add_subplot(gs[base + 1 + A, :])
    gt_prog = np.arange(T, dtype=np.float32) / max(T - 1, 1)
    ax_p.plot(t_full, gt_prog, color="#0D47A1", linewidth=1.5, label="GT")
    ax_p.plot(t_full, progress[:T], color="#B71C1C", linewidth=1.4, linestyle="--", label="pred")
    ax_p.set_ylabel("prog", fontsize=8, rotation=0, labelpad=26)
    ax_p.set_ylim(-0.05, 1.05); ax_p.tick_params(labelsize=7)
    ax_p.grid(True, color="#eee", linewidth=0.6)
    ax_p.legend(fontsize=7, loc="upper left", framealpha=0.8)

    fig.tight_layout(h_pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── HTML assembly ──────────────────────────────────────────────────────────────

_CSS = """
  body {font-family:sans-serif;background:#f5f5f5;margin:0;padding:12px;font-size:13px;}
  h1 {margin:0 0 4px;font-size:19px;color:#222;word-break:break-all;}
  h2 {margin:0 0 4px;font-size:16px;}
  .summary {color:#555;margin-bottom:10px;font-size:12px;}
  #top {display:flex;gap:12px;align-items:flex-start;}
  #cube-box {background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;}
  #charts {flex:1;min-width:360px;display:flex;flex-direction:column;gap:6px;}
  .chart-box {background:#fff;border:1px solid #ddd;border-radius:6px;padding:4px 6px;overflow-x:auto;}
  .chart-box h4 {margin:0 0 3px;font-size:11px;color:#333;}
  canvas {display:block;cursor:pointer;}
  #panel {margin-top:12px;background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px;display:none;}
  table {border-collapse:collapse;font-size:12px;margin-bottom:10px;}
  th,td {border:1px solid #e0e0e0;padding:4px 8px;text-align:right;}
  th {background:#f0f0f0;text-align:center;}
  td:first-child {text-align:left;}
  #samples {display:flex;flex-direction:row;flex-wrap:nowrap;gap:10px;overflow-x:auto;padding-bottom:8px;}
  .sample {flex:0 0 auto;}
  .sample img {display:block;width:300px;height:auto;border:1px solid #ddd;border-radius:4px;background:#fff;}
  .sample .cap {font-size:11px;color:#555;margin-bottom:3px;}
"""

# JS uses normal braces (no str.format) to keep it readable; data injected as JSON.
_JS = r"""
const N = COUNTS.length;
const LEVELS = FSQ_LEVELS;

// ── FSQ lattice cube ──────────────────────────────────────────────
const cubeCanvas = document.getElementById('cube');
const cubeCtx = cubeCanvas.getContext('2d');
cubeCanvas.width = 560; cubeCanvas.height = 560;
// Dimension-aware lattice: 1D line, 2D flat grid (e.g. [8,8]), 3D iso cube (e.g. [5,5,5] / [8,6,5]).
const NDIM = LEVELS.length;
const MAXL = Math.max.apply(null, LEVELS);
function idxToCoord(idx){const c=[];let r=idx;for(let d=0;d<NDIM;d++){c.push(r%LEVELS[d]);r=Math.floor(r/LEVELS[d]);}return c;}
// Center each axis, scale ALL axes by the same (MAXL-1) so non-cubic lattices keep true proportions.
function normCoord(c){const s=Math.max(1,MAXL-1)/2;return LEVELS.map((L,d)=>((c[d]||0)-(L-1)/2)/s);}
function projectCoord(c){
  const n=normCoord(c);
  const ox=cubeCanvas.width*0.5,oy=cubeCanvas.height*0.52;
  if(NDIM<=2){const xn=n[0]||0,yn=(n.length>1?n[1]:0);return [ox+xn*200, oy-yn*200, 0];}
  const xn=n[0]||0,yn=n[1]||0,zn=n[2]||0;
  const yaw=-0.63,pitch=0.46,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const xr=cy*xn-sy*yn,yr=sy*xn+cy*yn,zr=zn,scale=150;
  return [ox+xr*scale, oy+yr*scale*sp-zr*scale*cp, yr*cp+zr*sp];
}
function drawCube(sel){
  const ctx=cubeCtx; ctx.clearRect(0,0,cubeCanvas.width,cubeCanvas.height);
  const maxC=Math.max(...COUNTS,1);
  ctx.strokeStyle='rgba(120,120,120,0.45)'; ctx.lineWidth=1;
  const line=(a,b)=>{const pa=projectCoord(a),pb=projectCoord(b);ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke();};
  for(let i=0;i<N;i++){const c=idxToCoord(i);for(let d=0;d<NDIM;d++){if(c[d]+1<LEVELS[d]){const c2=c.slice();c2[d]+=1;line(c,c2);}}}
  const pts=[];
  for(let i=0;i<N;i++){const p=projectCoord(idxToCoord(i));pts.push({i,p,depth:p[2]});}
  pts.sort((a,b)=>a.depth-b.depth);
  pts.forEach(pt=>{
    const c=COUNTS[pt.i],active=c>0;
    const r=pt.i===sel?9:(active?4+7*Math.sqrt(c/maxC):2.6);
    ctx.beginPath();ctx.arc(pt.p[0],pt.p[1],r,0,Math.PI*2);
    ctx.fillStyle=pt.i===sel?'#f44336':(active?'rgba(25,118,210,0.78)':'rgba(180,180,180,0.4)');
    ctx.fill();
  });
  ctx.fillStyle='#777';ctx.font='10px sans-serif';
  ctx.fillText('levels = '+LEVELS.join(' x '),10,cubeCanvas.height-8);
}
cubeCanvas.addEventListener('click',e=>{
  const rect=cubeCanvas.getBoundingClientRect();
  const mx=(e.clientX-rect.left)*cubeCanvas.width/rect.width;
  const my=(e.clientY-rect.top)*cubeCanvas.height/rect.height;
  let best=-1,bd=1e9;
  for(let i=0;i<N;i++){const p=projectCoord(idxToCoord(i));const d=(p[0]-mx)**2+(p[1]-my)**2;if(d<bd){bd=d;best=i;}}
  if(best>=0&&bd<625&&COUNTS[best]>0)selectEntry(best);
});

// ── per-entry bar charts ──────────────────────────────────────────
const CHARTS=[
  {id:'c_count', vals:COUNTS,                title:'Skill count',        color:'#455A64', fmt:v=>v.toFixed(0)},
  {id:'c_recon', vals:VAL('chunk_mse'),      title:'Recon chunk MSE',    color:'#1976D2', fmt:v=>v.toExponential(1)},
  {id:'c_term',  vals:VAL('timing_abs'),     title:'Termination |err| (steps)', color:'#7B1FA2', fmt:v=>v.toFixed(1)},
  {id:'c_prog',  vals:VAL('prog_err'),       title:'Progress |err|',     color:'#2E7D32', fmt:v=>v.toFixed(3)},
];
function VAL(key){return Array.from({length:N},(_,i)=>ENTRY[i]?ENTRY[i][key]:null);}
function drawChart(ch,sel){
  const canvas=document.getElementById(ch.id),ctx=canvas.getContext('2d');
  const BW=Math.max(2,Math.min(16,Math.floor(1300/N))),PL=46,PR=10,PT=8,PB=16;
  canvas.width=PL+N*BW+PR; canvas.height=110;
  const vals=ch.vals,maxV=Math.max(...vals.filter(v=>v!=null),1e-12),H=canvas.height-PT-PB;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle='#bbb';ctx.beginPath();ctx.moveTo(PL,PT);ctx.lineTo(PL,PT+H);ctx.stroke();
  ctx.fillStyle='#888';ctx.font='9px sans-serif';ctx.textAlign='right';
  [0,maxV/2,maxV].forEach(v=>{const y=PT+H-(v/maxV)*H;ctx.fillText(ch.fmt(v),PL-3,y+3);});
  vals.forEach((v,i)=>{if(v==null)return;const x=PL+i*BW,h=(v/maxV)*H;ctx.fillStyle=(i===SEL)?'#f44336':ch.color;ctx.fillRect(x,PT+H-h,BW-1,h);});
}
function drawAllCharts(){CHARTS.forEach(ch=>drawChart(ch,SEL));}
CHARTS.forEach(ch=>{
  const canvas=document.getElementById(ch.id);
  canvas.addEventListener('click',e=>{
    const rect=canvas.getBoundingClientRect();const BW=Math.max(2,Math.min(16,Math.floor(1300/N)));
    const i=Math.floor((e.clientX-rect.left)*canvas.width/rect.width-46)/BW|0;
    if(i>=0&&i<N&&ENTRY[i])selectEntry(i);
  });
});

let SEL=-1;
function selectEntry(i){SEL=i;drawCube(SEL);drawAllCharts();showPanel(i);}
function showPanel(tok){
  const d=ENTRY[tok];if(!d)return;
  document.getElementById('panel-title').textContent='Entry '+tok+' — '+COUNTS[tok]+' skills';
  const rows=[
    ['# skills',COUNTS[tok]],
    ['Recon chunk MSE',d.chunk_mse.toExponential(3)],
    ['Recon MSE xyz',d.mse_xyz.toExponential(3)],
    ['Recon MSE rpy',d.mse_rpy.toExponential(3)],
    ['Recon MSE gripper',d.mse_grip.toExponential(3)],
    ['Termination |err| (steps)',d.timing_abs.toFixed(2)],
    ['Termination err (signed)',d.timing.toFixed(2)],
    ['Progress |err|',d.prog_err.toFixed(4)],
    ['Mean skill length',d.length.toFixed(1)],
  ];
  document.getElementById('panel-table').innerHTML='<tr><th>Metric</th><th>Value</th></tr>'+
    rows.map(r=>'<tr><td>'+r[0]+'</td><td>'+r[1]+'</td></tr>').join('');
  const imgs=SAMPLES[tok]||[];
  document.getElementById('samples').innerHTML=imgs.length
    ? imgs.map((b,i)=>'<div class="sample"><div class="cap">sample '+i+'</div><img src="data:image/jpeg;base64,'+b+'"></div>').join('')
    : '<div class="cap">No sample plots rendered for this entry (increase max_plot_entries).</div>';
  document.getElementById('panel').style.display='block';
  document.getElementById('panel').scrollIntoView({behavior:'smooth',block:'nearest'});
}
drawCube(-1);drawAllCharts();
"""


def build_html(title, summary, fsq_levels, counts, entry_data, samples, codebook_size) -> str:
    data_js = (
        "const SUMMARY=" + json.dumps(summary) + ";\n"
        "const FSQ_LEVELS=" + json.dumps(list(fsq_levels)) + ";\n"
        "const COUNTS=" + json.dumps(counts) + ";\n"
        "const ENTRY=" + json.dumps([entry_data.get(i) for i in range(codebook_size)]) + ";\n"
        "const SAMPLES=" + json.dumps([samples.get(i) for i in range(codebook_size)]) + ";\n"
    )
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<title>" + title + "</title><style>" + _CSS + "</style></head><body>"
        "<h1>" + title + "</h1>"
        "<div class='summary'>FSQ Unified Evaluation &nbsp;|&nbsp; " + summary + "</div>"
        "<div id='top'>"
        "  <div id='cube-box'><canvas id='cube'></canvas></div>"
        "  <div id='charts'>"
        "    <div class='chart-box'><h4>Skill count</h4><canvas id='c_count'></canvas></div>"
        "    <div class='chart-box'><h4>Reconstruction chunk MSE</h4><canvas id='c_recon'></canvas></div>"
        "    <div class='chart-box'><h4>Termination |error| (steps)</h4><canvas id='c_term'></canvas></div>"
        "    <div class='chart-box'><h4>Progress |error|</h4><canvas id='c_prog'></canvas></div>"
        "  </div>"
        "</div>"
        "<div id='panel'><h4 id='panel-title'></h4><table id='panel-table'></table>"
        "<div id='samples'></div></div>"
        "<script>" + data_js + _JS + "</script>"
        "</body></html>"
    )


# ── frame loading for sampled skills ───────────────────────────────────────────

def load_sample_frames(metadata, sample_ids, dataset_dir: Path, image_key: str, thumb: int):
    """Return {skill_idx: (start_rgb, end_rgb)} for the requested sample skill indices."""
    episodes_meta = _load_episodes_meta(dataset_dir)
    key = _resolve_image_key(episodes_meta, image_key)
    blank = np.full((thumb, thumb, 3), 80, np.uint8)

    ep_to_ids: dict[int, list[int]] = defaultdict(list)
    for i in sample_ids:
        ep_to_ids[int(metadata[i]["episode_id"])].append(i)

    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ep_id, ids in tqdm(sorted(ep_to_ids.items()), desc="Loading frames"):
        try:
            row = _episode_row(episodes_meta, ep_id)
            from_ts = float(row[f"videos/{key}/from_timestamp"])
            to_ts = float(row[f"videos/{key}/to_timestamp"])
            clip = _read_episode_clip(_video_path(dataset_dir, episodes_meta, ep_id, key),
                                      from_ts, to_ts, int(row["length"]))
            for i in ids:
                fs, fe = int(metadata[i]["frame_start"]), int(metadata[i]["frame_end"])
                frames[i] = (_clip_frame_or_blank(clip, fs, thumb),
                             _clip_frame_or_blank(clip, max(0, fe - 1), thumb))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] ep{ep_id}: {exc}")
            for i in ids:
                frames[i] = (blank, blank)
    return frames


# ── wandb ───────────────────────────────────────────────────────────────────────

def log_wandb(args, enc_stats, dec_means, html_path):
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name or Path(args.model_path).parent.name,
               config=vars(args), resume="allow")
    wandb.log({
        "codebook/utilization_pct":       enc_stats["utilization_pct"],
        "codebook/skills_per_entry_mean": enc_stats["skills_per_entry_mean"],
        "codebook/skills_per_entry_std":  enc_stats["skills_per_entry_std"],
        "codebook/skills_per_entry_max":  enc_stats["skills_per_entry_max"],
        "decoder/chunk_action_mse_mean":  dec_means["chunk_mse"],
        "decoder/recon_mse_xyz":          dec_means["mse_xyz"],
        "decoder/recon_mse_rpy":          dec_means["mse_rpy"],
        "decoder/recon_mse_gripper":      dec_means["mse_grip"],
        "decoder/termination_err_mean":   dec_means["timing_abs"],
        "decoder/early_rate":             dec_means["early_rate"],
        "decoder/late_rate":              dec_means["late_rate"],
        "decoder/progress_err_mean":      dec_means["prog_err"],
    })
    wandb.finish()
    print(f"[wandb] logged scalars to project '{args.wandb_project}' (HTML saved locally: {html_path})")


# ── main ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--skills_dir", required=True)
    p.add_argument("--dino_features", required=True)
    p.add_argument("--dino_features_wrist", default=None,
                   help="wrist DINO tokens npz (terminator's 2nd camera); only needed for dual-camera "
                        "models (terminator_use_wrist=True)")
    p.add_argument("--dataset_dir", required=True, help="LeRobot dataset dir (videos + meta) for frames")
    p.add_argument("--image_key", default="observation.images.image")
    p.add_argument("--output_dir", required=True, help="where the HTML is written (FSQ_eval/outputs/<run>/<epoch>)")
    p.add_argument("--latents_path", default="",
                   help="where to save/load skill_latents.npz (default: <output_dir>/skill_latents.npz). "
                        "Point this at the model checkpoint folder to keep latents next to FSQ.pt.")
    p.add_argument("--n_action_steps", type=int, default=5, help="chunk plot stride")
    p.add_argument("--max_plot_samples", type=int, default=5, help="sample skills per entry")
    p.add_argument("--max_plot_entries", type=int, default=0, help="0 = render all active entries")
    p.add_argument("--thumb_size", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=64, help="batch size for encode/decode inference")
    p.add_argument("--seed", type=int, default=42, help="seed for random sample selection per codebook entry")
    p.add_argument("--end_threshold", type=float, default=0.5)
    p.add_argument("--force_encode", action="store_true",
                   help="Re-encode latents even if skill_latents.npz already exists in output_dir.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb_project", default="VAE_eval")
    p.add_argument("--wandb_run_name", default="")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.model_path, device)
    levels = list(cfg.fsq_levels)
    codebook_size = int(model.fsq.codebook_size)

    use_wrist = bool(getattr(model, "terminator_use_wrist", False))
    print(f"[fsq_eval] loading skills / DINO (3rd-person{' + wrist' if use_wrist else ''}) ...")
    segments, dec_states, dec_targets, metadata = load_skill_files(Path(args.skills_dir))
    dec_tokens = load_dino_tokens(Path(args.dino_features), metadata)
    # Wrist is the dual terminator's 2nd camera — load it only when the model actually reads it.
    dec_tokens_wrist = None
    if use_wrist:
        if not args.dino_features_wrist:
            raise ValueError("Model has terminator_use_wrist=True but --dino_features_wrist was not given.")
        dec_tokens_wrist = load_dino_tokens(Path(args.dino_features_wrist), metadata)

    # ── encoder: reuse latents if already present for this run, else encode ──
    latents_path = Path(args.latents_path) if args.latents_path else out_dir / "skill_latents.npz"
    latents = tokens = None
    if latents_path.exists() and not args.force_encode:
        d = np.load(str(latents_path))
        # only trust the cache if it matches the current skill set
        if len(d["tokens"]) == len(metadata) and np.array_equal(
            d["frame_start"], np.array([m["frame_start"] for m in metadata])
        ):
            latents = d["latents"].astype(np.float32)
            tokens = d["tokens"].astype(np.int32)
            print(f"[fsq_eval] reusing existing latents → {latents_path} (--force_encode to redo)")
        else:
            print(f"[fsq_eval] existing latents at {latents_path} do not match skills; re-encoding")
    lengths = [int(m["length"]) for m in metadata]
    if latents is None:
        latents, tokens = batched_encode(model, segments, lengths, device, args.batch_size)
        save_latents(latents_path, latents, tokens, metadata)

    counts = [0] * codebook_size
    for t in tokens:
        counts[int(t)] += 1
    active = [c for c in counts if c > 0]
    enc_stats = {
        "utilization_pct":       100.0 * len(active) / codebook_size,
        "skills_per_entry_mean": float(np.mean(active)) if active else 0.0,
        "skills_per_entry_std":  float(np.std(active)) if active else 0.0,
        "skills_per_entry_max":  int(max(counts)) if counts else 0,
    }
    print(f"[fsq_eval] codebook {len(active)}/{codebook_size} used "
          f"({enc_stats['utilization_pct']:.1f}%)  mean={enc_stats['skills_per_entry_mean']:.1f}")

    # ── decoder: batched inference, then per-skill metrics ──
    action_dim = dec_targets[0].shape[-1]
    groups = _dim_groups(action_dim)
    dim_labels = [f"d{i}" for i in range(action_dim - 1)] + ["grip"]
    deltas, progresses, term_probs = batched_decode(
        model, latents, dec_states, dec_tokens, dec_tokens_wrist, lengths, device, args.batch_size)
    per_skill = [
        skill_metrics(deltas[i], progresses[i], term_probs[i], dec_targets[i], lengths[i],
                      args.end_threshold, groups)
        for i in range(len(metadata))
    ]

    keys = ["chunk_mse", "mse_xyz", "mse_rpy", "mse_grip", "timing_abs", "prog_err"]
    dec_means = {k: float(np.mean([s[k] for s in per_skill])) for k in keys}
    dec_means["early_rate"] = float(np.mean([s["timing"] < 0 for s in per_skill]))
    dec_means["late_rate"]  = float(np.mean([s["timing"] > 0 for s in per_skill]))
    print(f"[fsq_eval] chunk_mse={dec_means['chunk_mse']:.4e}  term|err|={dec_means['timing_abs']:.2f}  "
          f"early={dec_means['early_rate']:.1%} late={dec_means['late_rate']:.1%}  "
          f"prog_err={dec_means['prog_err']:.4f}")

    # ── aggregate per codebook entry ──
    by_entry: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(tokens):
        by_entry[int(t)].append(i)
    entry_data: dict[int, dict] = {}
    for tok, ids in by_entry.items():
        entry_data[tok] = {
            "chunk_mse":  float(np.mean([per_skill[i]["chunk_mse"] for i in ids])),
            "mse_xyz":    float(np.mean([per_skill[i]["mse_xyz"] for i in ids])),
            "mse_rpy":    float(np.mean([per_skill[i]["mse_rpy"] for i in ids])),
            "mse_grip":   float(np.mean([per_skill[i]["mse_grip"] for i in ids])),
            "timing_abs": float(np.mean([per_skill[i]["timing_abs"] for i in ids])),
            "timing":     float(np.mean([per_skill[i]["timing"] for i in ids])),
            "prog_err":   float(np.mean([per_skill[i]["prog_err"] for i in ids])),
            "length":     float(np.mean([per_skill[i]["length"] for i in ids])),
        }

    # ── choose entries/samples to render ──
    active_tokens = sorted(by_entry)
    if args.max_plot_entries > 0:
        plot_tokens = set(sorted(active_tokens, key=lambda t: (-counts[t], t))[: args.max_plot_entries])
    else:
        plot_tokens = set(active_tokens)
    rng = np.random.default_rng(args.seed)
    sample_ids: list[int] = []
    sample_by_tok: dict[int, list[int]] = {}
    for tok in plot_tokens:
        pool = by_entry[tok]
        n = min(args.max_plot_samples, len(pool))
        chosen = [pool[j] for j in rng.choice(len(pool), size=n, replace=False)] if n > 0 else []
        sample_by_tok[tok] = chosen
        sample_ids.extend(chosen)

    frames = ({} if args.max_plot_samples <= 0 else
              load_sample_frames(metadata, sample_ids, Path(args.dataset_dir), args.image_key, args.thumb_size))

    samples: dict[int, list[str]] = {}
    for tok in tqdm(sample_by_tok, desc="Rendering samples"):
        imgs = []
        for i in sample_by_tok[tok]:
            T = lengths[i]
            blank = np.full((args.thumb_size,) * 2 + (3,), 80, np.uint8)
            s_img, e_img = frames.get(i, (blank, blank))
            imgs.append(make_sample_plot(s_img, e_img, deltas[i], progresses[i], term_probs[i],
                                         dec_targets[i], T, dim_labels, args.n_action_steps, args.end_threshold))
        samples[tok] = imgs

    summary = (
        f"codebook {len(active)}/{codebook_size} ({enc_stats['utilization_pct']:.1f}%) | "
        f"skills/entry mean={enc_stats['skills_per_entry_mean']:.1f} max={enc_stats['skills_per_entry_max']} | "
        f"chunk MSE={dec_means['chunk_mse']:.3e} (xyz={dec_means['mse_xyz']:.2e} rpy={dec_means['mse_rpy']:.2e} "
        f"grip={dec_means['mse_grip']:.2e}) | term|err|={dec_means['timing_abs']:.2f} "
        f"early={dec_means['early_rate']:.0%} late={dec_means['late_rate']:.0%} | progress|err|={dec_means['prog_err']:.3f}"
    )
    title = f"{Path(args.model_path).parent.name} ({Path(args.model_path).stem})"
    html = build_html(title, summary, levels, counts, entry_data, samples, codebook_size)
    html_path = out_dir / "fsq_eval.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[fsq_eval] HTML → {html_path}")

    if not args.no_wandb:
        log_wandb(args, enc_stats, dec_means, html_path)


if __name__ == "__main__":
    main()
