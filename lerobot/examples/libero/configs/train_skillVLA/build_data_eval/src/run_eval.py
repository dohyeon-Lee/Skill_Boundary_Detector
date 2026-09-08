#!/usr/bin/env python3
"""Unified SkillVLA-data evaluation, run off the FINAL artifacts only.

Inputs (no skillset / per-episode DINO needed):
  raw video    : {dataset_root}/{source_dataset}      (LeRobot videos + meta)
  skillvla/    : {run_dir}/skillvla                    (skill columns: ds, boundary, sequence, ...)
  assignments  : {run_dir}/skill_latents.npz            (saved encoder labels)

Evals (toggle via flags), each writing under {out_dir}/{name}/:
  skillset   : skill-boundary split — start/end frames per skill, laid horizontally
  fsq_recon  : legacy flag/output name for an encoder-only code-membership browser.
               It compares FSQ-training samples with the current SkillVLA data;
               no decoder or reconstruction metric is evaluated.

Skills are reconstructed from the skillvla dataset columns:
  skill_ds==0 marks a skill start, skill_boundary==1 marks a skill end,
  skill_sequence[skill_index] gives the FSQ token.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# examples/libero on path (fsq_eval, codebook_visualizer, train_FSQ helpers)
LIBERO_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(LIBERO_DIR))

from codebook_visualizer import (  # noqa: E402
    _clip_frame_or_blank,
    _episode_row,
    _load_episodes_meta,
    _read_episode_clip,
    _resolve_image_key,
    _video_path,
)
from skillset_boundary_viz import (  # noqa: E402
    fig_to_b64 as _fig_to_b64,
    load_boundary_curve,
    render_skillset_card,
    save_gallery as _save_gallery,
)


# ── shared loaders ──────────────────────────────────────────────────────────────

def load_skillvla_episodes(skillvla_dir: Path) -> pd.DataFrame:
    files = sorted((skillvla_dir / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet under {skillvla_dir / 'data'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def load_task_names(dataset_dir: Path) -> dict[int, str]:
    """task_index → language instruction from the raw dataset meta (tasks.parquet:
    index = task string, column = task_index). Missing/unreadable → {} (IDs only)."""
    try:
        t = pd.read_parquet(Path(dataset_dir) / "meta" / "tasks.parquet")
        return {int(v): str(k) for k, v in t["task_index"].items()}
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] tasks.parquet unreadable → task IDs only: {exc}")
        return {}


def reconstruct_skills(ep_df: pd.DataFrame) -> list[tuple[int, int, int]]:
    """Per-episode [(frame_start, frame_end, token), ...] from skillvla columns."""
    fr = ep_df["frame_index"].to_numpy()
    ds = ep_df["skill_ds"].to_numpy()
    bn = ep_df["skill_boundary"].to_numpy()
    si = ep_df["skill_index"].to_numpy()
    seq = np.asarray(ep_df["skill_sequence"].iloc[0]).reshape(-1)
    starts = sorted(fr[ds == 0].tolist())
    ends = sorted(fr[bn == 1].tolist())
    skills = []
    for fs, fe1 in zip(starts, ends):
        s_si = int(si[fr == fs][0])
        skills.append((int(fs), int(fe1) + 1, int(seq[s_si])))
    return skills


def select_episodes(df: pd.DataFrame, task_ids, n_episodes: int) -> list[tuple]:
    """Ordered [(task_label, [ep, ...]), ...].

    task_ids empty/None → first n_episodes episodes overall (task_label=None).
    Otherwise → n_episodes episodes per listed task (so several tasks are covered)."""
    if not task_ids:
        return [(None, sorted(df["episode_index"].unique())[:n_episodes])]
    groups = []
    for t in task_ids:
        sub = df[df["task_index"] == int(t)]
        groups.append((int(t), sorted(sub["episode_index"].unique())[:n_episodes]))
    return groups


def _cap(task_label, ep) -> str:
    return f"episode {ep}" if task_label is None else f"task{int(task_label):02d} · episode {ep}"


# _fig_to_b64 / _save_gallery are imported from skillset_boundary_viz (shared with train_skills/skill_eval).


# ── eval 1: DINO sanity ──────────────────────────────────────────────────────────

def eval_dino(df, dino: DinoNpz, frames_src, out_dir: Path, n_episodes: int,
              task_ids=None, n_cols: int = 8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = int(round(((dino.n_tokens - 1) ** 0.5)))
    cards = []
    for task_label, eps in select_episodes(df, task_ids, n_episodes):
        for ep in eps:
            clip = dino.episode_clip(ep)                          # (T, n_tokens, F)
            raw = frames_src(int(ep))                             # (T_raw, H, W, 3) or None
            T = len(clip)
            pca = patch_pca_rgb_clip(clip[:, 1:], grid)           # (T, 96, 96, 3) — per-episode basis
            idxs = np.linspace(0, T - 1, min(n_cols, T)).astype(int)
            fig, axes = plt.subplots(2, len(idxs), figsize=(1.6 * len(idxs), 3.4), squeeze=False)
            for c, t in enumerate(idxs):
                frame = _clip_frame_or_blank(raw, t, 96) if raw is not None and len(raw) else np.full((96, 96, 3), 80, np.uint8)
                axes[0][c].imshow(frame); axes[0][c].axis("off"); axes[0][c].set_title(f"t{t}", fontsize=7)
                axes[1][c].imshow(pca[t]); axes[1][c].axis("off")
            axes[0][0].set_ylabel("raw", fontsize=8); axes[1][0].set_ylabel("PCA", fontsize=8)
            fig.suptitle(_cap(task_label, ep), fontsize=9)
            cards.append((task_label, _cap(task_label, ep), _fig_to_b64(fig)))
    _save_gallery(out_dir, "DINO patch sanity", cards)


# ── eval 2: skillset split ───────────────────────────────────────────────────────

def eval_skillset(df, frames_src, out_dir: Path, n_episodes: int,
                  task_ids=None, thumb: int = 110, curves_dir=None, task_names=None):
    # Rendering (boxed frames + multimodality curve + gallery) is shared with
    # train_skills/skill_eval via skillset_boundary_viz; here the per-episode skills
    # come from the skillvla dataset parquet (reconstruct_skills).
    names = task_names or {}
    cards = []
    for task_label, eps in select_episodes(df, task_ids, n_episodes):
        section = task_label
        if task_label is not None and int(task_label) in names:
            section = f"task{int(task_label):02d} · {names[int(task_label)]}"
        for ep in eps:
            ep_df = df[df["episode_index"] == ep]
            skills = reconstruct_skills(ep_df)
            raw = frames_src(int(ep))
            curve = load_boundary_curve(curves_dir, ep)
            b64 = render_skillset_card(skills, raw, curve, thumb=thumb)
            cap = f"{_cap(task_label, ep)} — {len(skills)} skills"
            if task_label is None and len(ep_df):
                # flat gallery (no per-task sections) → put the task language on the card itself
                ti = int(ep_df["task_index"].iloc[0])
                lang = names.get(ti)
                cap = (f"task{ti:02d} · episode {ep}"
                       + (f" — {lang}" if lang else "")
                       + f" — {len(skills)} skills")
            cards.append((section, cap, b64))
    _save_gallery(out_dir, "Skill boundary split", cards)


# ── eval 3: FSQ patch ────────────────────────────────────────────────────────────

def eval_fsq_patch(df, dino: DinoNpz, frames_src, out_dir: Path, n_episodes: int,
                   n_samples: int = 3, seed: int = 42, task_ids=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = int(round(((dino.n_tokens - 1) ** 0.5)))
    rng = np.random.default_rng(seed)
    cards = []
    for task_label, eps in select_episodes(df, task_ids, n_episodes):
        for ep in eps:
            ep_df = df[df["episode_index"] == ep]
            skills = reconstruct_skills(ep_df)
            clip = dino.episode_clip(ep)
            raw = frames_src(int(ep))
            rows = len(skills)
            if rows == 0:
                continue
            pca = patch_pca_rgb_clip(clip[:, 1:], grid)           # per-episode basis (shared across frames)
            ncol = 2 + n_samples
            fig, axes = plt.subplots(rows, ncol, figsize=(1.5 * ncol, 1.5 * rows), squeeze=False)
            for r, (fs, fe, tok) in enumerate(skills):
                fi = _clip_frame_or_blank(raw, fs, 96) if raw is not None and len(raw) else np.full((96,96,3),80,np.uint8)
                ff = _clip_frame_or_blank(raw, max(0, fe-1), 96) if raw is not None and len(raw) else np.full((96,96,3),80,np.uint8)
                axes[r][0].imshow(fi); axes[r][0].axis("off"); axes[r][0].set_ylabel(f"sk{r}\ntok{tok}", fontsize=7)
                axes[r][1].imshow(ff); axes[r][1].axis("off")
                cand = list(range(fs, min(fe, len(clip))))
                pick = rng.choice(cand, size=min(n_samples, len(cand)), replace=False) if cand else []
                for c, t in enumerate(sorted(pick)):
                    axes[r][2 + c].imshow(pca[t]); axes[r][2 + c].axis("off")
                for c in range(len(pick), n_samples):
                    axes[r][2 + c].axis("off")
            axes[0][0].set_title("init", fontsize=7); axes[0][1].set_title("final", fontsize=7)
            fig.suptitle(f"{_cap(task_label, ep)} — patch PCA", fontsize=9)
            cards.append((task_label, _cap(task_label, ep), _fig_to_b64(fig)))
    _save_gallery(out_dir, "FSQ patch visualization", cards)


# ── eval 4: encoder codebook membership browser ──────────────────────────────────

def _load_assignment_records(latents_path: Path) -> tuple[list[dict], np.ndarray]:
    """Read the encoder labels saved by the SkillVLA builder.

    These are the labels actually consumed downstream (including configured
    supported-code snapping), so the report neither loads the FSQ model nor
    reruns its decoder.
    """
    if not latents_path.is_file():
        raise FileNotFoundError(f"Skill assignments not found: {latents_path}")
    with np.load(latents_path, allow_pickle=False) as data:
        required = {
            "tokens", "episode_id", "task_id", "skill_index",
            "frame_start", "frame_end", "length",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{latents_path} is missing assignment fields: {missing}")
        arrays = {key: np.asarray(data[key]) for key in required}
        latents = np.asarray(data["latents"], dtype=np.float32)
    n = len(arrays["tokens"])
    if any(len(value) != n for value in arrays.values()) or len(latents) != n:
        raise ValueError(f"Inconsistent assignment lengths in {latents_path}")
    records = [
        {
            "token": int(arrays["tokens"][i]),
            "episode_id": int(arrays["episode_id"][i]),
            "task_id": int(arrays["task_id"][i]),
            "skill_index": int(arrays["skill_index"][i]),
            "frame_start": int(arrays["frame_start"][i]),
            "frame_end": int(arrays["frame_end"][i]),
            "length": int(arrays["length"][i]),
        }
        for i in range(n)
    ]
    return records, latents


def _sample_memberships(records: list[dict], n_samples: int, max_entries: int,
                        seed: int) -> dict[int, list[dict]]:
    by_token: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_token[int(record["token"])].append(record)
    active = sorted(by_token)
    if max_entries > 0:
        active = sorted(active, key=lambda tok: (-len(by_token[tok]), tok))[:max_entries]
    rng = np.random.default_rng(seed)
    sampled: dict[int, list[dict]] = {}
    for token in active:
        pool = by_token[token]
        count = min(max(0, n_samples), len(pool))
        indices = rng.choice(len(pool), size=count, replace=False) if count else []
        sampled[token] = [dict(pool[int(index)]) for index in indices]
    return sampled


def _materialize_membership_images(
    sampled: dict[int, list[dict]], dataset_dir: Path, image_key: str,
    assets_dir: Path, relative_prefix: str, thumb: int,
) -> dict[int, list[dict]]:
    """Decode only sampled episodes and save small start/end JPEG assets."""
    from PIL import Image

    frames_src = make_frames_loader(dataset_dir, image_key)
    task_names = load_task_names(dataset_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[int, np.ndarray | None] = {}
    rendered: dict[int, list[dict]] = {}
    blank = np.full((thumb, thumb, 3), 80, np.uint8)
    for token, records in sampled.items():
        cards = []
        token_dir = assets_dir / f"code_{token:04d}"
        token_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            episode = int(record["episode_id"])
            if episode not in cache:
                cache[episode] = frames_src(episode)
            clip = cache[episode]
            start_frame = (
                _clip_frame_or_blank(clip, int(record["frame_start"]), thumb)
                if clip is not None and len(clip) else blank
            )
            end_frame = (
                _clip_frame_or_blank(clip, max(0, int(record["frame_end"]) - 1), thumb)
                if clip is not None and len(clip) else blank
            )
            stem = (
                f"task{int(record['task_id']):03d}_ep{episode:05d}_"
                f"skill{int(record['skill_index']):02d}_"
                f"f{int(record['frame_start']):04d}_{int(record['frame_end']):04d}"
            )
            start_path = token_dir / f"{stem}_start.jpg"
            end_path = token_dir / f"{stem}_end.jpg"
            Image.fromarray(start_frame).save(start_path, format="JPEG", quality=88)
            Image.fromarray(end_frame).save(end_path, format="JPEG", quality=88)
            rel_dir = f"{relative_prefix}/code_{token:04d}"
            card = dict(record)
            card.update({
                "language": task_names.get(int(record["task_id"]), ""),
                "start_image": f"{rel_dir}/{start_path.name}",
                "end_image": f"{rel_dir}/{end_path.name}",
            })
            cards.append(card)
        rendered[token] = cards
    return rendered


def _membership_html(*, levels: list[int], training_samples: dict[int, list[dict]],
                     target_samples: dict[int, list[dict]], training_name: str,
                     target_name: str, title: str) -> str:
    codebook_size = int(np.prod(levels))
    training = [training_samples.get(i, []) for i in range(codebook_size)]
    target = [target_samples.get(i, []) for i in range(codebook_size)]
    payload = (
        "const LEVELS=" + json.dumps(levels) + ";\n"
        "const TRAINING=" + json.dumps(training, ensure_ascii=False).replace("</", "<\\/") + ";\n"
        "const TARGET=" + json.dumps(target, ensure_ascii=False).replace("</", "<\\/") + ";\n"
        "const TRAINING_NAME=" + json.dumps(training_name, ensure_ascii=False) + ";\n"
        "const TARGET_NAME=" + json.dumps(target_name, ensure_ascii=False) + ";\n"
    )
    css = """
body{font-family:Inter,system-ui,sans-serif;background:#f5f6f8;color:#1f2937;margin:0;padding:18px}
h1{font-size:21px;margin:0 0 5px}.sub{color:#667085;margin-bottom:14px}
.legend{display:flex;gap:18px;align-items:center;margin:8px 0 14px;font-size:12px}.dot{width:11px;height:11px;border-radius:50%;display:inline-block;margin-right:5px}
#cubeBox,#panel{background:white;border:1px solid #dfe3e8;border-radius:10px;padding:12px;box-shadow:0 1px 2px #0000000d}
#cube{display:block;max-width:100%;height:auto;cursor:pointer}.hover{height:18px;color:#475467;font-size:12px;margin-top:3px}
#panel{display:none;margin-top:14px}#panel h2{font-size:17px;margin:0 0 12px}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:16px}.column{min-width:0}.column h3{font-size:14px;margin:0 0 9px;padding-bottom:7px;border-bottom:2px solid #e5e7eb}
.samples{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:9px}.card{border:1px solid #e4e7ec;border-radius:8px;padding:8px;background:#fcfcfd}
.images{display:grid;grid-template-columns:1fr 1fr;gap:5px}.images figure{margin:0}.images img{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:5px;background:#ddd}.images figcaption{text-align:center;font-size:10px;color:#667085;margin-top:2px}
.meta{font-size:11px;line-height:1.45;color:#475467;margin-top:6px}.lang{color:#101828;font-weight:600;margin-top:3px}.empty{color:#98a2b3;font-size:12px;padding:12px 2px}
@media(max-width:900px){.columns{grid-template-columns:1fr}}
"""
    js = r"""
const N=LEVELS.reduce((a,b)=>a*b,1), ND=LEVELS.length;
const canvas=document.getElementById('cube'),ctx=canvas.getContext('2d');canvas.width=760;canvas.height=620;
function coord(i){const c=[];for(let d=0;d<ND;d++){c.push(i%LEVELS[d]);i=Math.floor(i/LEVELS[d]);}return c}
const SLICE_LEVELS=LEVELS.slice(3),NS=Math.max(1,SLICE_LEVELS.reduce((a,b)=>a*b,1));
const COLS=Math.ceil(Math.sqrt(NS)),ROWS=Math.ceil(NS/COLS),CW=canvas.width/COLS,CH=(canvas.height-25)/ROWS;
function sliceIndex(c){let n=0,m=1;for(let d=3;d<ND;d++){n+=c[d]*m;m*=LEVELS[d]}return n}
function project(c){const si=sliceIndex(c),col=si%COLS,row=Math.floor(si/COLS),ox=(col+.5)*CW,oy=(row+.53)*CH;
 const dims=LEVELS.slice(0,3),maxL=Math.max(2,...dims),scale=Math.min(CW,CH)*.31;
 const val=d=>d<ND?((c[d]-(LEVELS[d]-1)/2)/Math.max(1,(maxL-1)/2)):0;
 if(ND===1)return[ox+val(0)*scale,oy,0];if(ND===2)return[ox+val(0)*scale,oy-val(1)*scale,0];
 const x=val(0),y=val(1),z=val(2),yaw=-.63,pitch=.46,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),xr=cy*x-sy*y,yr=sy*x+cy*y;
 return[ox+xr*scale,oy+yr*scale*sp-z*scale*cp,yr*cp+z*sp]}
function active(i){return TRAINING[i].length||TARGET[i].length}
function color(i){const a=TRAINING[i].length>0,b=TARGET[i].length>0;return a&&b?'#7c3aed':a?'#2563eb':b?'#f97316':'#d0d5dd'}
let selected=-1;
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='#c7cdd4';ctx.lineWidth=1;
 for(let i=0;i<N;i++){const c=coord(i);for(let d=0;d<Math.min(3,ND);d++)if(c[d]+1<LEVELS[d]){const q=c.slice();q[d]++;const a=project(c),b=project(q);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()}}
 if(NS>1){ctx.fillStyle='#667085';ctx.font='11px sans-serif';for(let s=0;s<NS;s++){const col=s%COLS,row=Math.floor(s/COLS),extra=[];let x=s;for(let d=3;d<ND;d++){extra.push(x%LEVELS[d]);x=Math.floor(x/LEVELS[d])}ctx.fillText('dims 4+ = ['+extra.join(', ')+']',col*CW+8,row*CH+15)}}
 const pts=[];for(let i=0;i<N;i++){const p=project(coord(i));pts.push({i,p})}pts.sort((a,b)=>a.p[2]-b.p[2]);
 for(const q of pts){const r=q.i===selected?12:(active(q.i)?7:3.2);ctx.beginPath();ctx.arc(q.p[0],q.p[1],r,0,Math.PI*2);ctx.fillStyle=q.i===selected?'#111827':color(q.i);ctx.fill();if(q.i===selected){ctx.strokeStyle='#fbbf24';ctx.lineWidth=4;ctx.stroke()}}
 ctx.fillStyle='#667085';ctx.font='11px sans-serif';ctx.fillText('levels = '+LEVELS.join(' × '),8,canvas.height-5)}
function nearest(e){const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*canvas.width/r.width,y=(e.clientY-r.top)*canvas.height/r.height;let bi=-1,bd=1e9;for(let i=0;i<N;i++){const p=project(coord(i)),d=(p[0]-x)**2+(p[1]-y)**2;if(d<bd){bd=d;bi=i}}return bd<900?bi:-1}
canvas.addEventListener('mousemove',e=>{const i=nearest(e);document.getElementById('hover').textContent=i>=0?'code '+i+' · coordinate ['+coord(i).join(', ')+']':''});
canvas.addEventListener('mouseleave',()=>document.getElementById('hover').textContent='');
canvas.addEventListener('click',e=>{const i=nearest(e);if(i>=0&&active(i)){selected=i;draw();show(i)}});
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function cards(items){if(!items.length)return'<div class="empty">No sampled skill in this code.</div>';return'<div class="samples">'+items.map(x=>`<article class="card"><div class="images"><figure><img loading="lazy" src="${esc(x.start_image)}"><figcaption>start</figcaption></figure><figure><img loading="lazy" src="${esc(x.end_image)}"><figcaption>end</figcaption></figure></div><div class="meta">task ${x.task_id} · episode ${x.episode_id} · skill ${x.skill_index}<br>frames [${x.frame_start}, ${x.frame_end}) · length ${x.length}${x.language?`<div class="lang">${esc(x.language)}</div>`:''}</div></article>`).join('')+'</div>'}
function show(i){document.getElementById('panelTitle').textContent='Code '+i+' · coordinate ['+coord(i).join(', ')+']';document.getElementById('trainTitle').textContent='FSQ training data · '+TRAINING_NAME;document.getElementById('targetTitle').textContent='Evaluated data · '+TARGET_NAME;document.getElementById('trainSamples').innerHTML=cards(TRAINING[i]);document.getElementById('targetSamples').innerHTML=cards(TARGET[i]);document.getElementById('panel').style.display='block'}
draw();const first=Array.from({length:N},(_,i)=>i).find(active);if(first!==undefined){selected=first;draw();show(first)}
"""
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>" + html.escape(title) +
        "</title><style>" + css + "</style></head><body><h1>" + html.escape(title) +
        "</h1><div class='sub'>Select a code to compare skills from FSQ training data and the evaluated SkillVLA dataset. "
        "This is encoder-label inspection only; no decoder or reconstruction metric is evaluated.</div>"
        "<div class='legend'><span><i class='dot' style='background:#7c3aed'></i>both</span>"
        "<span><i class='dot' style='background:#2563eb'></i>training only</span>"
        "<span><i class='dot' style='background:#f97316'></i>evaluated only</span></div>"
        "<div id='cubeBox'><canvas id='cube'></canvas><div id='hover' class='hover'></div></div>"
        "<section id='panel'><h2 id='panelTitle'></h2><div class='columns'>"
        "<div class='column'><h3 id='trainTitle'></h3><div id='trainSamples'></div></div>"
        "<div class='column'><h3 id='targetTitle'></h3><div id='targetSamples'></div></div>"
        "</div></section><script>" + payload + js + "</script></body></html>"
    )


def eval_fsq_membership(*, target_latents_path: Path, target_dataset_dir: Path,
                        training_latents_path: Path, training_dataset_dir: Path,
                        skillvla_dir: Path, out_dir: Path, image_key: str,
                        n_samples: int, max_entries: int, thumb: int, seed: int):
    info_path = skillvla_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"SkillVLA metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    levels = [int(value) for value in info.get("skill_fsq_levels", [])]
    if not levels:
        raise ValueError(f"skill_fsq_levels missing from {info_path}")
    codebook_size = int(np.prod(levels))

    target_records, _ = _load_assignment_records(target_latents_path)
    training_records, _ = _load_assignment_records(training_latents_path)
    for label, records in (("training", training_records), ("evaluated", target_records)):
        invalid = sorted({r["token"] for r in records if not 0 <= r["token"] < codebook_size})
        if invalid:
            raise ValueError(f"{label} assignments contain codes outside [0,{codebook_size}): {invalid}")

    train_pick = _sample_memberships(training_records, n_samples, max_entries, seed)
    target_pick = _sample_memberships(target_records, n_samples, max_entries, seed + 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cards = _materialize_membership_images(
        train_pick, training_dataset_dir, image_key,
        out_dir / "assets" / "training", "assets/training", thumb,
    )
    target_cards = _materialize_membership_images(
        target_pick, target_dataset_dir, image_key,
        out_dir / "assets" / "evaluated", "assets/evaluated", thumb,
    )
    report = _membership_html(
        levels=levels,
        training_samples=train_cards,
        target_samples=target_cards,
        training_name=training_dataset_dir.name,
        target_name=target_dataset_dir.name,
        title=f"FSQ encoder code membership · {skillvla_dir.parent.name}",
    )
    path = out_dir / "fsq_eval.html"
    path.write_text(report, encoding="utf-8")
    print(
        f"[eval] fsq codebook membership → {path} "
        f"(training={len(training_records)}, evaluated={len(target_records)}, decoder=off)"
    )


# ── main ────────────────────────────────────────────────────────────────────────

def make_frames_loader(dataset_dir: Path, image_key: str):
    episodes_meta = _load_episodes_meta(dataset_dir)
    key = _resolve_image_key(episodes_meta, image_key)

    def load(ep_id: int):
        try:
            row = _episode_row(episodes_meta, ep_id)
            from_ts = float(row[f"videos/{key}/from_timestamp"])
            to_ts = float(row[f"videos/{key}/to_timestamp"])
            return _read_episode_clip(_video_path(dataset_dir, episodes_meta, ep_id, key),
                                      from_ts, to_ts, int(row["length"]))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] frames ep{ep_id}: {exc}")
            return None
    return load


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skillvla_dir", required=True)
    p.add_argument("--target_latents", required=True,
                   help="Current run's skill_latents.npz")
    p.add_argument("--training_latents", required=True,
                   help="PT skill_latents.npz defining FSQ training-data membership")
    p.add_argument("--training_dataset_dir", required=True,
                   help="Raw dataset used to train FSQ, for source start/end frames")
    p.add_argument("--dataset_dir", required=True, help="raw LeRobot dataset (videos + meta)")
    p.add_argument("--image_key", default="observation.images.image")
    p.add_argument("--out_dir", required=True, help="{run_dir}/eval")
    p.add_argument("--boundary_curves_dir", default=None,
                   help="{skillset_dir}/curves with per-episode multimodality curves "
                        "(build_skill_dataset --dump_curves). Absent → skillset eval shows frames only.")
    p.add_argument("--run_skillset", action="store_true")
    p.add_argument("--run_fsq_recon", action="store_true")
    p.add_argument("--n_episodes", type=int, default=12,
                   help="episodes shown per visual eval; per task when --task_ids is given")
    p.add_argument("--task_ids", type=int, nargs="*", default=None,
                   help="restrict skillset visualization to these tasks (n_episodes each); "
                        "empty = first n_episodes overall")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--max_entries", type=int, default=0)
    p.add_argument("--thumb_size", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    df = load_skillvla_episodes(Path(args.skillvla_dir))
    frames_src = make_frames_loader(Path(args.dataset_dir), args.image_key)
    if args.run_skillset:
        eval_skillset(df, frames_src, out / "skillset", args.n_episodes, task_ids=args.task_ids,
                      curves_dir=args.boundary_curves_dir,
                      task_names=load_task_names(Path(args.dataset_dir)))
    if args.run_fsq_recon:
        eval_fsq_membership(
            target_latents_path=Path(args.target_latents),
            target_dataset_dir=Path(args.dataset_dir),
            training_latents_path=Path(args.training_latents),
            training_dataset_dir=Path(args.training_dataset_dir),
            skillvla_dir=Path(args.skillvla_dir),
            out_dir=out / "fsq_recon",
            image_key=args.image_key,
            n_samples=args.n_samples,
            max_entries=args.max_entries,
            thumb=args.thumb_size,
            seed=args.seed,
        )
    print(f"[eval] done → {out}")


if __name__ == "__main__":
    main()
