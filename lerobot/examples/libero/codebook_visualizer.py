"""
FSQ codebook visualizer: interactive HTML + wandb logging.

Bar chart of skill counts per FSQ index (0..codebook_size-1).
Clicking a bar reveals start/end image pairs for skills mapped to that entry.

Usage:
    python codebook_visualizer.py \
        --latents_path .../skill_latents.npz \
        --dataset_dir  .../libero_90 \
        --wandb_project VAE_eval \
        --wandb_run_name my_run
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


# ── video-frame helpers (LeRobot dataset format) ──────────────────────────────

def _load_episodes_meta(dataset_dir: Path):
    import pandas as pd
    files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files under {dataset_dir / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _resolve_image_key(episodes_meta, image_key: str) -> str:
    video_cols = [c for c in episodes_meta.columns if c.startswith("videos/") and c.endswith("/chunk_index")]
    keys = [c.split("/")[1] for c in video_cols]
    if image_key and image_key in keys:
        return image_key
    if not keys:
        raise ValueError("No video camera keys found in episode metadata.")
    return keys[0]


def _episode_row(episodes_meta, episode_id: int):
    return episodes_meta[episodes_meta["episode_index"] == episode_id].iloc[0]


def _video_path(dataset_dir: Path, episodes_meta, episode_id: int, image_key: str) -> Path:
    row = _episode_row(episodes_meta, episode_id)
    chunk_idx = int(row[f"videos/{image_key}/chunk_index"])
    file_idx  = int(row[f"videos/{image_key}/file_index"])
    return dataset_dir / "videos" / image_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_episode_clip(vpath: Path, from_ts: float, to_ts: float, expected_len: int) -> np.ndarray:
    """Read only one episode span from a larger LeRobot video file."""
    from torchvision.io import read_video

    frames, _, _ = read_video(
        str(vpath),
        start_pts=from_ts,
        end_pts=to_ts - 0.001,
        pts_unit="sec",
        output_format="THWC",
    )
    arr = frames.numpy().astype(np.uint8)[..., :3]
    if len(arr) > expected_len:
        arr = arr[len(arr) - expected_len:]
    return arr


def _clip_frame_or_blank(clip: np.ndarray, idx: int, thumb_size: int) -> np.ndarray:
    if len(clip) == 0:
        return np.full((thumb_size, thumb_size, 3), 80, np.uint8)
    idx = int(np.clip(idx, 0, len(clip) - 1))
    return clip[idx]


def _img_to_b64(arr: np.ndarray, size: int, quality: int = 75) -> str:
    from PIL import Image
    img = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ── data collection ───────────────────────────────────────────────────────────

def collect_data(args) -> dict:
    """Returns serialisable dict with stats and base64 images for each entry."""
    data = np.load(args.latents_path)
    tokens       = data["tokens"].astype(np.int32)
    episode_ids  = data["episode_id"].astype(np.int64)
    skill_idxs   = data["skill_index"].astype(np.int64)
    frame_starts = data["frame_start"].astype(np.int64)
    frame_ends   = data["frame_end"].astype(np.int64)

    # FSQ codebook_size = product(levels); default 125 for [5,5,5]
    num_emb = args.num_embeddings or (int(tokens.max()) + 1)
    fsq_levels = args.fsq_levels
    if not fsq_levels:
        side = round(num_emb ** (1.0 / 3.0))
        fsq_levels = [side, side, side] if side ** 3 == num_emb else [num_emb, 1, 1]
    if int(np.prod(fsq_levels)) != num_emb:
        raise ValueError(f"fsq_levels product {int(np.prod(fsq_levels))} != num_embeddings {num_emb}")

    token_to_idxs: dict[int, list[int]] = defaultdict(list)
    for i, tok in enumerate(tokens):
        token_to_idxs[int(tok)].append(i)

    counts = [len(token_to_idxs.get(k, [])) for k in range(num_emb)]
    used   = sum(1 for c in counts if c > 0)
    active = [c for c in counts if c > 0]
    print(f"[VIZ] FSQ codebook: {used}/{num_emb} entries used  "
          f"| mean={np.mean(active):.1f}  max={max(counts)}  total skills={len(tokens)}")

    dataset_dir   = Path(args.dataset_dir)
    episodes_meta = _load_episodes_meta(dataset_dir)
    image_key     = _resolve_image_key(episodes_meta, args.image_key)
    vpath_cache: dict[int, Path] = {}

    def _vpath(ep_id: int) -> Path:
        if ep_id not in vpath_cache:
            vpath_cache[ep_id] = _video_path(dataset_dir, episodes_meta, ep_id, image_key)
        return vpath_cache[ep_id]

    active = sorted(k for k in token_to_idxs if token_to_idxs[k])
    selected_by_token = {tok: token_to_idxs[tok][: args.max_per_entry] for tok in active}

    ep_to_idxs: dict[int, list[int]] = defaultdict(list)
    for idxs in selected_by_token.values():
        for i in idxs:
            ep_to_idxs[int(episode_ids[i])].append(i)

    results: dict[int, dict] = {}
    blank_b64 = _img_to_b64(
        np.full((args.thumb_size, args.thumb_size, 3), 80, np.uint8), args.thumb_size
    )

    for ep_id, ep_idxs in tqdm(sorted(ep_to_idxs.items()), desc="Extracting thumbnails"):
        try:
            row = _episode_row(episodes_meta, ep_id)
            from_ts = float(row[f"videos/{image_key}/from_timestamp"])
            to_ts = float(row[f"videos/{image_key}/to_timestamp"])
            expected_len = int(row["length"])
            clip = _read_episode_clip(_vpath(ep_id), from_ts, to_ts, expected_len)
            for i in ep_idxs:
                fs = int(frame_starts[i])
                fe = int(frame_ends[i])
                s_b64 = _img_to_b64(_clip_frame_or_blank(clip, fs, args.thumb_size), args.thumb_size)
                e_b64 = _img_to_b64(_clip_frame_or_blank(clip, max(0, fe - 1), args.thumb_size), args.thumb_size)
                results[i] = {
                    "ep": ep_id,
                    "sk": int(skill_idxs[i]),
                    "fs": fs,
                    "fe": fe,
                    "s": s_b64,
                    "e": e_b64,
                }
        except Exception as exc:
            print(f"  [warn] ep{ep_id}: {exc}")
            for i in ep_idxs:
                results[i] = {
                    "ep": ep_id,
                    "sk": int(skill_idxs[i]),
                    "fs": int(frame_starts[i]),
                    "fe": int(frame_ends[i]),
                    "s": blank_b64,
                    "e": blank_b64,
                }

    entries: dict[str, list[dict]] = {
        str(tok): [results[i] for i in idxs]
        for tok, idxs in selected_by_token.items()
    }

    return {"num_emb": num_emb, "fsq_levels": fsq_levels, "counts": counts, "entries": entries,
            "total": int(len(tokens)), "used": used}


# ── HTML generation ───────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FSQ Codebook Visualizer</title>
<style>
  body {{ font-family: sans-serif; background:#f5f5f5; margin:0; padding:12px; }}
  h2   {{ margin:0 0 4px; font-size:15px; }}
  #stats {{ font-size:12px; color:#555; margin-bottom:10px; }}
  #chart-wrap {{
    width:100%; overflow-x:auto; background:#fff;
    border:1px solid #ddd; border-radius:6px; padding:10px; box-sizing:border-box;
  }}
  #cube-wrap {{
    margin-top:10px; background:#fff; border:1px solid #ddd;
    border-radius:6px; padding:10px; box-sizing:border-box;
  }}
  #cube-title {{ font-size:12px; color:#555; margin-bottom:6px; }}
  canvas {{ display:block; }}
  #panel {{
    margin-top:12px; background:#fff; border:1px solid #ddd;
    border-radius:6px; padding:10px; display:none;
  }}
  #panel h3 {{ margin:0 0 8px; font-size:14px; }}
  .grid {{
    display:flex; flex-wrap:wrap; gap:8px;
  }}
  .skill-card {{
    border:1px solid #e0e0e0; border-radius:4px;
    padding:4px; background:#fafafa; text-align:center;
    font-size:10px; color:#444;
  }}
  .skill-card .imgs {{ display:flex; gap:2px; }}
  .skill-card img {{ width:{thumb}px; height:{thumb}px; object-fit:cover; border-radius:2px; }}
  .label {{ margin-top:2px; }}
  .bar-selected {{ outline:2px solid #f44336; }}
</style>
</head>
<body>
<h2>FSQ Codebook Visualizer</h2>
<div id="stats">
  {total} skills &nbsp;|&nbsp; {used}/{num_emb} entries active &nbsp;|&nbsp;
  mean {mean:.1f} skills/entry &nbsp;|&nbsp; max {mx}
</div>
<div id="chart-wrap">
  <canvas id="chart"></canvas>
</div>
<div id="cube-wrap">
  <div id="cube-title">FSQ lattice view</div>
  <canvas id="cube"></canvas>
</div>
<div id="panel">
  <h3 id="panel-title"></h3>
  <div class="grid" id="grid"></div>
</div>

<script>
const COUNTS  = {counts_json};
const ENTRIES = {entries_json};
const LEVELS  = {fsq_levels_json};
const N       = COUNTS.length;
const MAX_PER = {max_per};

const canvas  = document.getElementById('chart');
const ctx     = canvas.getContext('2d');
const BAR_W   = Math.max(2, Math.min(8, Math.floor(1400 / N)));
const PAD_L   = 40, PAD_R = 10, PAD_T = 20, PAD_B = 30;

canvas.width  = PAD_L + N * BAR_W + PAD_R;
canvas.height = 200;

const maxC = Math.max(...COUNTS);

function drawChart(sel) {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const H = canvas.height - PAD_T - PAD_B;

  // y-axis
  ctx.strokeStyle = '#bbb'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD_L, PAD_T); ctx.lineTo(PAD_L, PAD_T+H); ctx.stroke();
  ctx.fillStyle='#888'; ctx.font='10px sans-serif'; ctx.textAlign='right';
  [0, Math.round(maxC/2), maxC].forEach(v => {{
    const y = PAD_T + H - (v/maxC)*H;
    ctx.fillText(v, PAD_L-3, y+3);
    ctx.strokeStyle='#eee'; ctx.beginPath(); ctx.moveTo(PAD_L,y); ctx.lineTo(canvas.width-PAD_R,y); ctx.stroke();
  }});

  COUNTS.forEach((c, i) => {{
    const x = PAD_L + i * BAR_W;
    const h = maxC > 0 ? (c / maxC) * H : 0;
    const y = PAD_T + H - h;
    ctx.fillStyle = (i === sel) ? '#f44336' : (c > 0 ? '#1976D2' : '#e0e0e0');
    ctx.fillRect(x, y, BAR_W - 1, h);
  }});

  // x-axis tick every 50
  ctx.fillStyle='#888'; ctx.textAlign='center'; ctx.font='9px sans-serif';
  for (let i=0; i<N; i+=50) {{
    const x = PAD_L + i*BAR_W + BAR_W/2;
    ctx.fillText(i, x, PAD_T+H+12);
  }}
}}

let selected = -1;
drawChart(selected);

function idxToCoord(idx) {{
  const lx = LEVELS[0], ly = LEVELS[1], lz = LEVELS[2];
  const x = idx % lx;
  const y = Math.floor(idx / lx) % ly;
  const z = Math.floor(idx / (lx * ly)) % lz;
  return [x, y, z];
}}

const cubeCanvas = document.getElementById('cube');
const cubeCtx = cubeCanvas.getContext('2d');
cubeCanvas.width = 560;
cubeCanvas.height = 420;

function projectCoord(coord) {{
  const [x, y, z] = coord;
  const lx = Math.max(1, LEVELS[0] - 1);
  const ly = Math.max(1, LEVELS[1] - 1);
  const lz = Math.max(1, LEVELS[2] - 1);
  const xn = lx > 0 ? (x / lx - 0.5) * 2.0 : 0.0;
  const yn = ly > 0 ? (y / ly - 0.5) * 2.0 : 0.0;
  const zn = lz > 0 ? (z / lz - 0.5) * 2.0 : 0.0;

  // Camera: slightly diagonal and above the lattice so all three axes are visible.
  const yaw = -0.63;
  const pitch = 0.46;
  const cyaw = Math.cos(yaw), syaw = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const xr = cyaw * xn - syaw * yn;
  const yr = syaw * xn + cyaw * yn;
  const zr = zn;

  const scale = 118;
  const cx = cubeCanvas.width * 0.50;
  const cy = cubeCanvas.height * 0.56;
  const sx = cx + xr * scale;
  const sy = cy + yr * scale * sp - zr * scale * cp;
  const depth = yr * cp + zr * sp;
  return [sx, sy, depth];
}}

function drawCube(sel) {{
  const ctx = cubeCtx;
  ctx.clearRect(0, 0, cubeCanvas.width, cubeCanvas.height);
  const maxC = Math.max(...COUNTS, 1);
  const lx = LEVELS[0], ly = LEVELS[1], lz = LEVELS[2];

  // Interior lattice lines make FSQ neighborhood distance readable.
  ctx.strokeStyle = 'rgba(120,120,120,0.50)';
  ctx.lineWidth = 1.2;
  const drawLine = (a, b) => {{
    const pa = projectCoord(a);
    const pb = projectCoord(b);
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.stroke();
  }};
  for (let x = 0; x < lx; x++) {{
    for (let y = 0; y < ly; y++) {{
      for (let z = 0; z < lz; z++) {{
        if (x + 1 < lx) drawLine([x, y, z], [x + 1, y, z]);
        if (y + 1 < ly) drawLine([x, y, z], [x, y + 1, z]);
        if (z + 1 < lz) drawLine([x, y, z], [x, y, z + 1]);
      }}
    }}
  }}

  const corners = [
    [0,0,0], [lx-1,0,0], [0,ly-1,0], [0,0,lz-1],
    [lx-1,ly-1,0], [lx-1,0,lz-1], [0,ly-1,lz-1], [lx-1,ly-1,lz-1],
  ];
  const edges = [[0,1],[0,2],[0,3],[1,4],[1,5],[2,4],[2,6],[3,5],[3,6],[4,7],[5,7],[6,7]];
  ctx.strokeStyle = 'rgba(70,70,70,0.72)';
  ctx.lineWidth = 1.8;
  edges.forEach(([a,b]) => {{
    const pa = projectCoord(corners[a]);
    const pb = projectCoord(corners[b]);
    ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
  }});

  const pts = [];
  for (let i = 0; i < N; i++) {{
    const coord = idxToCoord(i);
    const p = projectCoord(coord);
    pts.push({{i, coord, p, depth: p[2]}});
  }}
  pts.sort((a,b) => a.depth - b.depth);
  pts.forEach(pt => {{
    const c = COUNTS[pt.i];
    const active = c > 0;
    const r = pt.i === sel ? 9 : (active ? 4 + 7 * Math.sqrt(c / maxC) : 2.8);
    ctx.beginPath();
    ctx.arc(pt.p[0], pt.p[1], r, 0, Math.PI * 2);
    ctx.fillStyle = pt.i === sel ? '#f44336' : (active ? 'rgba(25,118,210,0.78)' : 'rgba(180,180,180,0.45)');
    ctx.fill();
    if (pt.i === sel) {{
      ctx.strokeStyle = '#8b0000';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = '#333';
      ctx.font = '11px sans-serif';
      ctx.fillText(`entry ${{pt.i}}  (${{pt.coord.join(',')}})`, 10, 18);
    }}
  }});
  ctx.fillStyle = '#777';
  ctx.font = '10px sans-serif';
  ctx.fillText(`levels = ${{LEVELS.join(' x ')}}`, 10, cubeCanvas.height - 10);
}}

cubeCanvas.addEventListener('click', e => {{
  const rect = cubeCanvas.getBoundingClientRect();
  const scaleX = cubeCanvas.width / rect.width;
  const scaleY = cubeCanvas.height / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;
  let best = -1, bestD = Infinity;
  for (let i = 0; i < N; i++) {{
    const p = projectCoord(idxToCoord(i));
    const d = (p[0] - mx) ** 2 + (p[1] - my) ** 2;
    if (d < bestD) {{ bestD = d; best = i; }}
  }}
  if (best >= 0 && bestD < 625) {{
    selectEntry(best);
  }}
}});

drawCube(selected);

function selectEntry(i) {{
  if (i < 0 || i >= N || COUNTS[i] === 0) return;
  selected = i;
  drawChart(selected);
  drawCube(selected);
  showPanel(i);
}}

canvas.addEventListener('click', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const i    = Math.floor((mx - PAD_L) / BAR_W);
  selectEntry(i);
}});

function showPanel(tok) {{
  const skills  = ENTRIES[tok] || [];
  const total   = COUNTS[tok];
  const panel   = document.getElementById('panel');
  const title   = document.getElementById('panel-title');
  const grid    = document.getElementById('grid');

  title.textContent = `Entry ${{tok}} — ${{total}} skills` +
    (total > MAX_PER ? ` (showing first ${{MAX_PER}})` : '');
  grid.innerHTML = '';

  skills.forEach(sk => {{
    const card = document.createElement('div');
    card.className = 'skill-card';
    card.innerHTML = `
      <div class="imgs">
        <img src="data:image/jpeg;base64,${{sk.s}}" title="start f${{sk.fs}}">
        <img src="data:image/jpeg;base64,${{sk.e}}" title="end f${{sk.fe}}">
      </div>
      <div class="label">ep${{sk.ep}} · sk${{sk.sk}}<br>f${{sk.fs}}→${{sk.fe}}</div>`;
    grid.appendChild(card);
  }});

  panel.style.display = 'block';
  panel.scrollIntoView({{behavior:'smooth', block:'nearest'}});
}}
</script>
</body>
</html>
"""


def generate_html(vdata: dict, max_per: int, thumb: int) -> str:
    counts = vdata["counts"]
    active = [c for c in counts if c > 0]
    return _HTML_TEMPLATE.format(
        thumb        = thumb,
        total        = vdata["total"],
        used         = vdata["used"],
        num_emb      = vdata["num_emb"],
        mean         = float(np.mean(active)) if active else 0.0,
        mx           = int(max(counts)),
        counts_json  = json.dumps(counts),
        entries_json = json.dumps(vdata["entries"]),
        fsq_levels_json = json.dumps(vdata["fsq_levels"]),
        max_per      = max_per,
    )


# ── wandb logging ─────────────────────────────────────────────────────────────

def log_to_wandb(html_path: Path, vdata: dict, args) -> None:
    import wandb

    wandb.init(
        project  = args.wandb_project,
        name     = args.wandb_run_name or Path(args.latents_path).stem,
        config   = vars(args),
        resume   = "allow",
    )

    # utilization summary scalars
    counts = vdata["counts"]
    active = [c for c in counts if c > 0]
    wandb.log({
        "codebook/total_skills":   vdata["total"],
        "codebook/entries_used":   vdata["used"],
        "codebook/entries_total":  vdata["num_emb"],
        "codebook/utilization_pct": 100.0 * vdata["used"] / vdata["num_emb"],
        "codebook/skills_per_entry_mean": float(np.mean(active)) if active else 0.0,
        "codebook/skills_per_entry_max":  int(max(counts)),
        "codebook/skills_per_entry_std":  float(np.std(active)) if active else 0.0,
    })

    # interactive HTML panel
    wandb.log({"codebook/visualizer": wandb.Html(str(html_path), inject=False)})
    print(f"[wandb] logged interactive HTML to project '{args.wandb_project}'")

    wandb.finish()


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_path",   required=True)
    p.add_argument("--dataset_dir",    required=True)
    p.add_argument("--image_key",      default="observation.images.image")
    p.add_argument("--num_embeddings", type=int, default=0,
                   help="Codebook size (0 = infer from data)")
    p.add_argument("--fsq_levels", type=int, nargs="*", default=[],
                   help="FSQ lattice levels, e.g. 5 5 5. Empty = infer cubic levels from num_embeddings.")
    p.add_argument("--max_per_entry",  type=int, default=50,
                   help="Max skills shown per codebook entry (shows all if entry has fewer)")
    p.add_argument("--thumb_size",     type=int, default=96,
                   help="Thumbnail size in pixels")
    p.add_argument("--output_html",    default="",
                   help="Path to save HTML (default: next to latents_path)")
    p.add_argument("--wandb_project",  default="VAE_eval")
    p.add_argument("--wandb_run_name", default="")
    p.add_argument("--no_wandb",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    vdata = collect_data(args)
    html  = generate_html(vdata, args.max_per_entry, args.thumb_size)

    # save HTML
    if args.output_html:
        html_path = Path(args.output_html)
    else:
        html_path = Path(args.latents_path).with_suffix(".codebook_vis.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    print(f"[VIZ] HTML saved → {html_path}")

    if not args.no_wandb:
        log_to_wandb(html_path, vdata, args)


if __name__ == "__main__":
    main()
