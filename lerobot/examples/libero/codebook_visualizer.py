"""
Codebook visualizer: generates an interactive HTML page and logs it to wandb.

The HTML shows a bar chart of skill counts per codebook entry.
Clicking a bar reveals the start/end image pairs for skills in that entry.

Usage:
    python codebook_visualizer.py \
        --latents_path .../spline_vqae_latents_epochXXXX.npz \
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
from train_vae import _load_episodes_meta, _resolve_image_key, _video_path


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_frame(reader, idx: int, thumb_size: int) -> np.ndarray:
    """Seek to a specific frame index in an open imageio reader."""
    try:
        return reader.get_data(idx)[..., :3].astype(np.uint8)
    except Exception:
        return np.full((thumb_size, thumb_size, 3), 80, np.uint8)


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

    num_emb = args.num_embeddings or (int(tokens.max()) + 1)

    token_to_idxs: dict[int, list[int]] = defaultdict(list)
    for i, tok in enumerate(tokens):
        token_to_idxs[int(tok)].append(i)

    counts = [len(token_to_idxs.get(k, [])) for k in range(num_emb)]
    used   = sum(1 for c in counts if c > 0)
    print(f"[VIZ] {used}/{num_emb} codebook entries used  "
          f"| mean={np.mean([c for c in counts if c>0]):.1f}  "
          f"max={max(counts)}  total skills={len(tokens)}")

    import imageio

    dataset_dir   = Path(args.dataset_dir)
    episodes_meta = _load_episodes_meta(dataset_dir)
    image_key     = _resolve_image_key(episodes_meta, args.image_key)
    vpath_cache: dict[int, Path] = {}

    def _vpath(ep_id: int) -> Path:
        if ep_id not in vpath_cache:
            vpath_cache[ep_id] = _video_path(dataset_dir, episodes_meta, ep_id, image_key)
        return vpath_cache[ep_id]

    entries: dict[str, list[dict]] = {}
    active = sorted(k for k in token_to_idxs if token_to_idxs[k])

    for tok in tqdm(active, desc="Extracting thumbnails"):
        idxs = token_to_idxs[tok][: args.max_per_entry]

        # Group by episode → open each video once, seek to needed frames
        ep_to_idxs: dict[int, list[int]] = defaultdict(list)
        for i in idxs:
            ep_to_idxs[int(episode_ids[i])].append(i)

        results: dict[int, dict] = {}
        blank_b64 = _img_to_b64(
            np.full((args.thumb_size, args.thumb_size, 3), 80, np.uint8), args.thumb_size)

        for ep_id, ep_idxs in ep_to_idxs.items():
            try:
                reader = imageio.get_reader(str(_vpath(ep_id)))
                try:
                    for i in ep_idxs:
                        fs = int(frame_starts[i])
                        fe = int(frame_ends[i])
                        s_b64 = _img_to_b64(_read_frame(reader, fs,       args.thumb_size), args.thumb_size)
                        e_b64 = _img_to_b64(_read_frame(reader, max(0, fe-1), args.thumb_size), args.thumb_size)
                        results[i] = {"ep": ep_id, "sk": int(skill_idxs[i]),
                                      "fs": fs, "fe": fe, "s": s_b64, "e": e_b64}
                finally:
                    reader.close()
            except Exception as exc:
                print(f"  [warn] ep{ep_id}: {exc}")
                for i in ep_idxs:
                    results[i] = {"ep": ep_id, "sk": int(skill_idxs[i]),
                                  "fs": int(frame_starts[i]), "fe": int(frame_ends[i]),
                                  "s": blank_b64, "e": blank_b64}

        entries[str(tok)] = [results[i] for i in idxs]

    return {"num_emb": num_emb, "counts": counts, "entries": entries,
            "total": int(len(tokens)), "used": used}


# ── HTML generation ───────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Codebook Visualizer</title>
<style>
  body {{ font-family: sans-serif; background:#f5f5f5; margin:0; padding:12px; }}
  h2   {{ margin:0 0 4px; font-size:15px; }}
  #stats {{ font-size:12px; color:#555; margin-bottom:10px; }}
  #chart-wrap {{
    width:100%; overflow-x:auto; background:#fff;
    border:1px solid #ddd; border-radius:6px; padding:10px; box-sizing:border-box;
  }}
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
<h2>Codebook Visualizer</h2>
<div id="stats">
  {total} skills &nbsp;|&nbsp; {used}/{num_emb} entries active &nbsp;|&nbsp;
  mean {mean:.1f} skills/entry &nbsp;|&nbsp; max {mx}
</div>
<div id="chart-wrap">
  <canvas id="chart"></canvas>
</div>
<div id="panel">
  <h3 id="panel-title"></h3>
  <div class="grid" id="grid"></div>
</div>

<script>
const COUNTS  = {counts_json};
const ENTRIES = {entries_json};
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

canvas.addEventListener('click', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const i    = Math.floor((mx - PAD_L) / BAR_W);
  if (i < 0 || i >= N || COUNTS[i] === 0) return;
  selected = i;
  drawChart(selected);
  showPanel(i);
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
