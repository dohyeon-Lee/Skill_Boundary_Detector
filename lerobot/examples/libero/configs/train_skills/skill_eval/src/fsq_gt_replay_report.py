#!/usr/bin/env python3
"""Race-safe manifest merge and interactive report for FSQ GT replays."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
from html import escape as escape_html
from pathlib import Path

# Signature fields every checkpoint of one comparison must share. model_path and
# latents_path are deliberately absent: they are what differs between checkpoints.
_COLLECTION_SIGNATURE_KEYS = (
    "format",
    "target_task",
    "selected_episodes",
    "seed",
)


def token_to_coord(token: int, levels: list[int]) -> list[int]:
    coord = []
    base = 1
    for level in levels:
        coord.append((int(token) // base) % int(level))
        base *= int(level)
    return coord


def report_payload(manifest: dict) -> dict:
    levels = [int(value) for value in manifest["levels"]]
    by_token: dict[int, list[dict]] = {}
    for record in manifest["records"].values():
        by_token.setdefault(int(record["token"]), []).append(record)
    skills = []
    for token, occurrences in sorted(by_token.items()):
        occurrences.sort(
            key=lambda row: (row["task_id"], row["episode_id"], row["frame_start"])
        )
        skills.append(
            {
                "token": token,
                "coord": token_to_coord(token, levels),
                "occurrences": occurrences,
            }
        )
    signature = manifest["signature"]
    return {
        "levels": levels,
        "run_name": manifest["run_name"],
        "model_name": manifest.get("model_name") or manifest["run_name"],
        "title": manifest.get("report_title") or "",
        "epoch_tag": manifest["epoch_tag"],
        "target_task": signature["target_task"],
        "task_ids": sorted(int(value) for value in signature["selected_episodes"]),
        "episode_count": sum(
            len(values) for values in signature["selected_episodes"].values()
        ),
        "occurrence_count": len(manifest["records"]),
        "train_codebook_counts": manifest.get("train_codebook_counts"),
        "train_codebook_used": manifest.get("train_codebook_used"),
        "train_codebook_effective": manifest.get("train_codebook_effective"),
        "skills": skills,
    }


def collection_payload(manifests: list[dict]) -> dict:
    if not manifests:
        raise ValueError("At least one FSQ replay manifest is required.")
    run_names = {manifest["run_name"] for manifest in manifests}
    model_names = {
        str(manifest.get("model_name") or "").strip()
        for manifest in manifests
        if str(manifest.get("model_name") or "").strip()
    }
    titles = {
        str(manifest.get("report_title") or "").strip()
        for manifest in manifests
        if str(manifest.get("report_title") or "").strip()
    }
    target_tasks = {
        manifest["signature"]["target_task"] for manifest in manifests
    }
    if len(run_names) != 1 or len(target_tasks) != 1:
        raise ValueError("FSQ replay checkpoints must share one run and task suite.")
    if len(model_names) > 1 or len(titles) > 1:
        raise ValueError("FSQ replay checkpoints must share one model name and title.")
    checkpoints = []
    for manifest in manifests:
        checkpoint = report_payload(manifest)
        prefix = Path("checkpoints") / checkpoint["epoch_tag"]
        for skill in checkpoint["skills"]:
            prefixed = []
            for occurrence in skill["occurrences"]:
                row = dict(occurrence)
                for key in ("start_image_path", "final_image_path"):
                    if row.get(key):
                        row[key] = (prefix / row[key]).as_posix()
                prefixed.append(row)
            skill["occurrences"] = prefixed
        checkpoints.append(checkpoint)
    return {
        "format": "fsq_gt_replay_collection_v1",
        "run_name": next(iter(run_names)),
        "model_name": next(iter(model_names), ""),
        "title": next(iter(titles), ""),
        "target_task": next(iter(target_tasks)),
        "checkpoints": checkpoints,
    }


def compare_payload(
    collection_dirs: list[str | Path], *, output_dir: str | Path
) -> dict:
    """Combine several finished run collections into one tabbed comparison.

    Media paths are rewritten relative to output_dir, so the comparison page can
    live anywhere without copying images. Models whose replay selection (task,
    episodes, seed) differs from the first model are kept but flagged mismatched:
    their cohesion numbers are not a fair comparison.
    """
    if not collection_dirs:
        raise ValueError("At least one collection directory is required.")
    output_dir = Path(output_dir)
    models: list[dict] = []
    titles: set[str] = set()
    reference_signature: dict | None = None
    for collection_dir in collection_dirs:
        collection_dir = Path(collection_dir)
        if collection_dir.resolve() == output_dir.resolve():
            continue
        available = completed_manifests(collection_dir)
        if not available:
            print(f"skipping {collection_dir}: no completed checkpoint manifest.")
            continue
        reference_tag = sorted(available, key=_tag_sort_key)[-1]
        manifests, excluded = _partition_compatible(available, reference_tag)
        payload = collection_payload(manifests)
        if payload.get("title"):
            titles.add(payload["title"])
        prefix = Path(os.path.relpath(collection_dir, output_dir))
        for checkpoint in payload["checkpoints"]:
            for skill in checkpoint["skills"]:
                for occurrence in skill["occurrences"]:
                    for key in ("start_image_path", "final_image_path"):
                        if occurrence.get(key):
                            occurrence[key] = (prefix / occurrence[key]).as_posix()
        signature = _comparable_signature(manifests[0])
        model = {
            "name": payload.get("model_name") or collection_dir.name,
            "run_name": payload["run_name"],
            "target_task": payload["target_task"],
            "checkpoints": payload["checkpoints"],
        }
        if excluded:
            model["excluded_epoch_tags"] = excluded
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            model["mismatched"] = True
        models.append(model)
    if not models:
        raise FileNotFoundError(
            "None of the given directories holds a completed checkpoint manifest."
        )
    if len(titles) > 1:
        raise ValueError(f"Compared FSQ runs disagree on report title: {sorted(titles)}.")
    default_title = output_dir.parent.name if output_dir.name == "compare" else output_dir.name
    return {
        "format": "fsq_gt_replay_compare_v1",
        "title": next(iter(titles), default_title),
        "models": models,
    }


def maybe_build_compare(
    collection_dirs: list[str | Path], *, output_dir: str | Path
) -> Path | None:
    """Build the multi-run comparison page once every run's collection is merged.

    Safe to call from every finishing run: a lock serializes builders and the
    page is only written when each collection dir has its merged collection.json.
    """
    output_dir = Path(output_dir)
    metrics = output_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    with (metrics / "compare.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not all(
            (Path(directory) / "metrics" / "collection.json").is_file()
            for directory in collection_dirs
        ):
            return None
        payload = compare_payload(collection_dirs, output_dir=output_dir)
        _atomic_json(metrics / "compare.json", payload)
        return write_html_report(output_dir, payload)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


_REPORT_DATA_CHUNK_BYTES = 16 * 1024 * 1024


def _script_json(value: object) -> str:
    """Serialize JSON so it is safe in both inline and external script elements."""
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


def _report_data_chunks(
    payload: dict, *, max_bytes: int = _REPORT_DATA_CHUNK_BYTES
) -> tuple[str, list[str]]:
    """Split report data into ordered JavaScript chunks.

    The HTML bootstraps an empty model list, then ordinary external scripts fill
    it before the UI script runs.  Checkpoints are normally added one skill at a
    time; an unusually large skill falls back to occurrence-at-a-time statements
    so every generated asset remains comfortably below editor/viewer limits.
    """
    if max_bytes <= 0:
        raise ValueError("Report data chunk size must be positive.")
    bootstrap = {key: value for key, value in payload.items() if key != "models"}
    bootstrap["models"] = []
    chunks: list[str] = []
    parts: list[str] = []
    part_bytes = 0

    def flush() -> None:
        nonlocal parts, part_bytes
        if parts:
            chunks.append("".join(parts))
            parts = []
            part_bytes = 0

    def add(statement: str) -> None:
        nonlocal part_bytes
        encoded_bytes = len(statement.encode("utf-8"))
        if encoded_bytes > max_bytes:
            raise ValueError(
                "One FSQ replay data record exceeds the configured report-data "
                f"chunk size ({encoded_bytes} > {max_bytes} bytes)."
            )
        if parts and part_bytes + encoded_bytes > max_bytes:
            flush()
        parts.append(statement)
        part_bytes += encoded_bytes

    data_ref = "window.FSQ_GT_REPLAY_DATA"
    for model_index, model in enumerate(payload["models"]):
        model_header = {
            key: value for key, value in model.items() if key != "checkpoints"
        }
        model_header["checkpoints"] = []
        add(f"{data_ref}.models.push({_script_json(model_header)});\n")
        model_ref = f"{data_ref}.models[{model_index}]"
        for checkpoint_index, checkpoint in enumerate(model["checkpoints"]):
            checkpoint_header = {
                key: value for key, value in checkpoint.items() if key != "skills"
            }
            checkpoint_header["skills"] = []
            add(f"{model_ref}.checkpoints.push({_script_json(checkpoint_header)});\n")
            skills_ref = f"{model_ref}.checkpoints[{checkpoint_index}].skills"
            for skill_index, skill in enumerate(checkpoint["skills"]):
                full_statement = f"{skills_ref}.push({_script_json(skill)});\n"
                if len(full_statement.encode("utf-8")) <= max_bytes:
                    add(full_statement)
                    continue
                skill_header = {
                    key: value for key, value in skill.items() if key != "occurrences"
                }
                skill_header["occurrences"] = []
                add(f"{skills_ref}.push({_script_json(skill_header)});\n")
                occurrences_ref = f"{skills_ref}[{skill_index}].occurrences"
                for occurrence in skill["occurrences"]:
                    add(f"{occurrences_ref}.push({_script_json(occurrence)});\n")
    flush()
    return _script_json(bootstrap), chunks


def write_html_report(output_dir: str | Path, payload: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "models" not in payload:
        if "checkpoints" not in payload:
            payload = {
                "format": "fsq_gt_replay_collection_v1",
                "run_name": payload["run_name"],
                "model_name": payload.get("model_name") or payload["run_name"],
                "title": payload.get("title") or "",
                "target_task": payload["target_task"],
                "checkpoints": [payload],
            }
        model = {
            "name": payload.get("model_name") or payload["run_name"],
            "run_name": payload["run_name"],
            "target_task": payload["target_task"],
            "checkpoints": payload["checkpoints"],
        }
        if payload.get("excluded_epoch_tags"):
            model["excluded_epoch_tags"] = payload["excluded_epoch_tags"]
        payload = {
            "format": "fsq_gt_replay_compare_v1",
            "title": payload.get("title") or "FSQ GT skill replay",
            "models": [model],
        }
    report_title = str(payload.get("title") or "FSQ GT skill replay")
    bootstrap, data_chunks = _report_data_chunks(payload)
    digest = hashlib.sha256()
    digest.update(bootstrap.encode("utf-8"))
    for chunk in data_chunks:
        digest.update(chunk.encode("utf-8"))
    generation = digest.hexdigest()[:12]
    data_paths = [
        output_dir / f"report-data-{generation}-{index:03d}.js"
        for index in range(len(data_chunks))
    ]
    for data_path, chunk in zip(data_paths, data_chunks, strict=True):
        temporary = data_path.with_name(data_path.name + ".tmp")
        temporary.write_text(chunk, encoding="utf-8")
        temporary.replace(data_path)
    data_scripts = "\n".join(
        f'<script src="{path.name}"></script>' for path in data_paths
    )
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>__REPORT_TITLE__</title>
  <style>
    :root{--ink:#17202a;--muted:#667085;--line:#d4dbe6;--blue:#2878b5;--red:#d62728}
    *{box-sizing:border-box} body{margin:0;font-family:Inter,Arial,sans-serif;background:#f4f6f9;color:var(--ink)}
    header{position:sticky;top:0;z-index:5;padding:12px 20px;background:#fff;border-bottom:1px solid var(--line)}
    h1{margin:0 0 5px;font-size:20px}.subtitle{color:var(--muted);font-size:12px}
    .tabs{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}.tab{padding:5px 11px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer;font-size:12px;font-weight:700;color:var(--ink)}.tab.active{background:var(--blue);border-color:var(--blue);color:#fff}
    .controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:9px}.control{display:flex;align-items:center;gap:7px;font-size:12px;font-weight:700}.control select,.range-input{padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:#fff}.range-input{width:72px}.range{display:flex;align-items:center;gap:5px}.tasks{display:flex;align-items:center;gap:5px;flex-wrap:wrap}.task-chip{padding:4px 7px;border:1px solid var(--line);border-radius:12px;background:#f8fafc;font-weight:500}.task-chip input{margin:0 4px 0 0}.small-button{padding:4px 7px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
    .layout{display:grid;grid-template-columns:minmax(420px,580px) 1fr;gap:16px;padding:16px;align-items:start}
    .sidebar{position:sticky;top:126px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px}
    .cube,.full-cube{width:100%;height:auto;display:block}.legend{display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
    .cube-modes{display:flex;gap:6px;margin:8px 0}.cube-mode.active{background:var(--blue);color:#fff;border-color:var(--blue)}
    .cube-section-title{margin:14px 0 0;padding-top:13px;border-top:1px solid var(--line);font-size:13px;font-weight:800}.cube-section-hint{margin:3px 0 0;color:var(--muted);font-size:11px}
    .grad{display:inline-block;width:110px;height:10px;background:linear-gradient(to right,rgb(253,235,232),rgb(136,8,8));border:1px solid var(--line);border-radius:3px;vertical-align:middle}
    .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}.selected{margin-top:9px;padding:9px;background:#f6f9fd;border-radius:7px;font-weight:700}
    .task-group{margin-bottom:18px}.task-group-title{display:flex;align-items:baseline;gap:7px;margin:0 0 7px;padding:8px 10px;background:#e9eef6;border:1px solid var(--line);border-radius:8px;font-size:13px;font-weight:800}.task-count{color:var(--muted);font-size:11px;font-weight:600}.occ-row{display:flex;align-items:flex-start;gap:10px;overflow-x:auto;padding:1px 1px 10px;scrollbar-gutter:stable}.occ{flex:0 0 340px;max-width:76vw;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}
    .occ-title{padding:9px 12px;background:#f8fafc;border-bottom:1px solid var(--line);font-size:13px;font-weight:700}
    .meta{padding:7px 12px;color:var(--muted);font-size:11px;border-bottom:1px solid var(--line)}
    .pair{display:flex;gap:6px;padding:8px}.pair figure{margin:0;flex:1;min-width:0}.pair img{display:block;width:100%;height:auto;border:1px solid var(--line)}.pair figcaption{text-align:center;color:var(--muted);font-size:10px;font-weight:700;margin-top:3px}.count{margin-left:6px;padding:1px 6px;border-radius:10px;background:#e8eef7;color:#35516f;font-size:11px}
    .empty{padding:32px;background:#fff;border:1px solid var(--line);border-radius:10px;text-align:center;color:var(--muted)}
    .main-col{min-width:0}.tables{display:grid;gap:12px;margin-bottom:16px}
    .panel{min-width:0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px}
    .panel>summary{cursor:pointer;font-size:13px;font-weight:800;list-style:none}.panel>summary::-webkit-details-marker{display:none}.panel>summary::before{content:'▾ ';color:var(--muted)}.panel:not([open])>summary::before{content:'▸ '}
    .hint{margin:4px 0 9px;color:var(--muted);font-size:11px;white-space:normal}
    .table-scroll{overflow-x:auto}
    table.usage{border-collapse:collapse;font-size:11px;white-space:nowrap}
    table.usage th,table.usage td{border:1px solid var(--line);padding:3px 8px;text-align:center}
    table.usage th{background:#eef3fa;font-weight:700}table.usage th.label,table.usage td.label{text-align:left;font-weight:700}
    table.usage td.label{background:#fafcff}table.usage td.total{font-weight:800;background:#f4f7fc}table.usage td.zero{color:#c3cad6}
    table.usage tr.summary td{background:#e9eef6;font-weight:800}table.usage tr.summary td.total{background:#dfe7f2}table.usage td.best{font-weight:800}
    .no-rows{padding:14px;color:var(--muted);font-size:12px}
    @media(max-width:1100px){.layout{grid-template-columns:1fr}.sidebar{position:relative;top:auto;max-width:620px}}
  </style>
</head>
<body>
<header><h1>__REPORT_TITLE__</h1><div class="subtitle" id="summary"></div>
  <div class="tabs" id="modelTabs" hidden></div>
  <div class="controls">
    <label class="control">Checkpoint <select id="checkpoint"></select></label>
    <div class="control"><span>Tasks</span><div class="tasks" id="tasks"></div><button class="small-button" id="allTasks" type="button">all</button><button class="small-button" id="clearTasks" type="button">clear</button></div>
    <label class="control">Skill position <select id="positionMode"><option value="all">all</option><option value="percent">percent</option><option value="id">skill ID</option></select></label>
    <div class="control range" id="positionRange" hidden><input class="range-input" id="positionStart" type="number"><span>to</span><input class="range-input" id="positionEnd" type="number"><span id="positionUnit"></span></div>
    <label class="control">Table order <select id="tableUnit"><option value="id">skill ID</option><option value="rank">rank in episode</option><option value="percent">percent (10% bins)</option></select></label>
  </div>
</header>
<main class="layout">
  <aside class="sidebar"><svg class="cube" viewBox="0 0 600 520"></svg>
    <div class="cube-modes"><button type="button" class="small-button cube-mode active" data-mode="usage">usage</button><button type="button" class="small-button cube-mode" data-mode="length">length</button><button type="button" class="small-button cube-mode" data-mode="count">count</button></div>
    <div class="legend" id="cubeLegend"></div>
    <div class="selected" id="selected"></div>
    <div class="cube-section-title">Full training skillset · count</div>
    <p class="cube-section-hint">All skills in this checkpoint's training latent artifact; unaffected by task and position filters.</p>
    <svg class="full-cube" viewBox="0 0 600 520"></svg>
    <div class="legend" id="fullCubeLegend"></div>
    <div class="selected" id="fullSelected">Click a used code to show its full-data count.</div>
  </aside>
  <div class="main-col">
    <div class="tables" id="tables"></div>
    <section id="content"></section>
  </div>
</main>
<script>window.FSQ_GT_REPLAY_DATA=__DATA_BOOTSTRAP__;</script>
__DATA_SCRIPTS__
<script>
const DATA=window.FSQ_GT_REPLAY_DATA, models=DATA.models;
let modelIndex=0,checkpointIndex=0,selectedTasks=new Set(),activeSkills=[],byToken=new Map(),positionByOccurrence=new Map(),positionMode='all';
let checkpoints=models.length?models[0].checkpoints:[];
let maximumSkillId=0;models.forEach(model=>model.checkpoints.forEach(cp=>cp.skills.forEach(skill=>skill.occurrences.forEach(o=>{maximumSkillId=Math.max(maximumSkillId,Number(o.skill_index)||0)}))));const positionRanges={percent:[0,100],id:[0,maximumSkillId]};
const esc=v=>String(v).replace(/[&<>'"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
const current=()=>checkpoints[checkpointIndex];
function coord(token,levels){const c=[];let base=1;for(const level of levels){c.push(Math.floor(token/base)%level);base*=level}return c}
function project(c,levels){const n=levels.length,maxL=Math.max(...levels),scaleDen=Math.max(1,maxL-1)/2;
  const v=levels.map((level,i)=>((c[i]||0)-(level-1)/2)/scaleDen),ox=300,oy=265;
  if(n<3)return[ox+(v[0]||0)*190,oy-(v[1]||0)*190,0];
  const yaw=-.63,pitch=.46,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const xr=cy*v[0]-sy*v[1],yr=sy*v[0]+cy*v[1],zr=v[2]||0;
  let x=ox+xr*145,y=oy+yr*145*sp-zr*145*cp,z=yr*cp+zr*sp;
  // Dims 3+ (e.g. BSQ [2,2,2,2,2]): classic hypercube rendering — each extra
  // dim shifts the base cube along its own ever-smaller diagonal, so all
  // corners land on distinct positions. No-op for 3D FSQ grids.
  const HD=[[46,-24,.3],[20,10,.15],[10,-6,.08]];
  for(let d=3;d<n;d++){const o=HD[Math.min(d-3,HD.length-1)];x+=(v[d]||0)*o[0];y+=(v[d]||0)*o[1];z+=(v[d]||0)*o[2]}
  return[x,y,z]}
let cubeMode='usage',selectedToken=-1;const COUNT_BORDER_THRESHOLD=10;
function cubeColor(t){const from=[253,235,232],to=[136,8,8];const rgb=from.map((v,i)=>Math.round(v+(to[i]-v)*t));return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`}
function renderCube(selected){selectedToken=selected;const levels=current().levels,svg=document.querySelector('.cube'),NS='http://www.w3.org/2000/svg';svg.innerHTML='';
  const make=(name,attrs)=>{const e=document.createElementNS(NS,name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));svg.appendChild(e);return e};
  const stats=new Map();byToken.forEach((skill,token)=>{const lengths=skill.occurrences.map(o=>Number(o.length));stats.set(Number(token),{count:skill.occurrences.length,meanLength:lengths.reduce((a,b)=>a+b,0)/lengths.length})});
  const metric=stat=>cubeMode==='length'?stat.meanLength:stat.count;
  let scale=null;
  if(cubeMode!=='usage'&&stats.size){const values=[...stats.values()].map(metric);scale={lo:Math.min(...values),hi:Math.max(...values)}}
  const total=levels.reduce((a,b)=>a*b,1);for(let t=0;t<total;t++){const c=coord(t,levels);for(let d=0;d<levels.length;d++){if(c[d]+1<levels[d]){const n=c.slice();n[d]++;const a=project(c,levels),b=project(n,levels);const hi=d>=3;make('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:hi?'rgba(120,120,170,.22)':'rgba(100,100,100,.42)','stroke-width':hi?0.8:1.2})}}}
  const points=[];for(let t=0;t<total;t++){const p=project(coord(t,levels),levels);points.push({t,p,used:byToken.has(t)})}points.sort((a,b)=>a.p[2]-b.p[2]);
  points.forEach(({t,p,used})=>{
    let fill=used?'#2878b5':'#d7dde8';
    if(used&&cubeMode!=='usage'){const value=metric(stats.get(t));const norm=scale.hi>scale.lo?(value-scale.lo)/(scale.hi-scale.lo):0.6;fill=cubeColor(0.1+0.9*norm)}
    const selectedCode=used&&t===selected;
    const countOutlined=used&&cubeMode==='count'&&stats.get(t).count>COUNT_BORDER_THRESHOLD;
    const stroke=selectedCode?'#d62728':cubeMode==='count'?(countOutlined?'#111':'none'):'#26384d';
    const strokeWidth=selectedCode?3:countOutlined?2:cubeMode==='count'?0:.8;
    const e=make('circle',{cx:p[0],cy:p[1],r:selectedCode?9:(used?6:3.5),fill,stroke,'stroke-width':strokeWidth,style:used?'cursor:pointer':'cursor:default'});
    if(used){const stat=stats.get(t);const title=document.createElementNS(NS,'title');title.textContent=`#${t} · ${stat.count} skills · mean length ${stat.meanLength.toFixed(1)} frames`;e.appendChild(title);e.addEventListener('click',()=>selectToken(t))}});
  renderCubeLegend(scale)}
function renderCubeLegend(scale){const box=document.getElementById('cubeLegend');
  if(cubeMode==='usage'){box.innerHTML='<span><i class="dot" style="background:#2878b5;border:2px solid #d62728"></i>selected</span><span><i class="dot" style="background:#2878b5"></i>used</span><span><i class="dot" style="background:#d7dde8"></i>unused</span>';return}
  const label=cubeMode==='length'?'mean skill length (frames)':'skill count';
  if(!scale){box.innerHTML=`<span>${label}: no used codes</span>`;return}
  const fmt=value=>cubeMode==='length'?value.toFixed(0):String(Math.round(value));
  const countBorders=cubeMode==='count'?`<span><i class="dot" style="background:#fff;border:2px solid #111"></i>&gt;${COUNT_BORDER_THRESHOLD} elements</span><span><i class="dot" style="background:#fff"></i>\\u2264${COUNT_BORDER_THRESHOLD} elements</span>`:'';
  box.innerHTML=`<span>${label}</span><span>${fmt(scale.lo)}</span><i class="grad"></i><span>${fmt(scale.hi)}</span>${countBorders}<span><i class="dot" style="background:#fff;border:2px solid #d62728"></i>selected</span><span><i class="dot" style="background:#d7dde8"></i>unused</span>`}
let selectedFullToken=-1;
function renderFullCube(selected=selectedFullToken){
  const cp=current(),levels=cp.levels,svg=document.querySelector('.full-cube'),NS='http://www.w3.org/2000/svg';svg.innerHTML='';
  const make=(name,attrs)=>{const e=document.createElementNS(NS,name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));svg.appendChild(e);return e};
  const total=levels.reduce((a,b)=>a*b,1),raw=Array.isArray(cp.train_codebook_counts)?cp.train_codebook_counts:[];
  const counts=Array.from({length:total},(_,token)=>Number(raw[token]||0)),usedCounts=counts.filter(count=>count>0);
  if(selected<0||selected>=total||counts[selected]<=0)selected=-1;
  selectedFullToken=selected;
  const scale=usedCounts.length?{lo:Math.min(...usedCounts),hi:Math.max(...usedCounts)}:null;
  for(let t=0;t<total;t++){const c=coord(t,levels);for(let d=0;d<levels.length;d++){if(c[d]+1<levels[d]){const n=c.slice();n[d]++;const a=project(c,levels),b=project(n,levels);const hi=d>=3;make('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:hi?'rgba(120,120,170,.22)':'rgba(100,100,100,.42)','stroke-width':hi?0.8:1.2})}}}
  const points=[];for(let t=0;t<total;t++){const p=project(coord(t,levels),levels);points.push({t,p,count:counts[t]})}points.sort((a,b)=>a.p[2]-b.p[2]);
  points.forEach(({t,p,count})=>{const used=count>0,selectedCode=used&&t===selected;let fill='#d7dde8';
    if(used){const norm=scale.hi>scale.lo?(count-scale.lo)/(scale.hi-scale.lo):0.6;fill=cubeColor(0.1+0.9*norm)}
    const countOutlined=used&&count>COUNT_BORDER_THRESHOLD,stroke=selectedCode?'#d62728':countOutlined?'#111':'none',strokeWidth=selectedCode?3:countOutlined?2:0;
    const e=make('circle',{cx:p[0],cy:p[1],r:selectedCode?9:(used?6:3.5),fill,stroke,'stroke-width':strokeWidth,style:used?'cursor:pointer':'cursor:default'});
    if(used){const title=document.createElementNS(NS,'title');title.textContent=`#${t} · ${count} full-data skills`;e.appendChild(title);e.addEventListener('click',()=>selectFullToken(t))}});
  renderFullCubeLegend(scale);
  const selectedBox=document.getElementById('fullSelected');
  selectedBox.textContent=selected>=0?`token #${selected} [${coord(selected,levels).join(', ')}] · ${counts[selected]} full-data elements`:usedCounts.length?'Click a used code to show its full-data count.':'Full-data token histogram unavailable.'}
function renderFullCubeLegend(scale){const box=document.getElementById('fullCubeLegend');
  if(!scale){box.innerHTML='<span>full skill count: unavailable</span>';return}
  box.innerHTML=`<span>full skill count</span><span>${Math.round(scale.lo)}</span><i class="grad"></i><span>${Math.round(scale.hi)}</span><span><i class="dot" style="background:#fff;border:2px solid #111"></i>&gt;${COUNT_BORDER_THRESHOLD} elements</span><span><i class="dot" style="background:#fff"></i>\\u2264${COUNT_BORDER_THRESHOLD} elements</span><span><i class="dot" style="background:#fff;border:2px solid #d62728"></i>selected</span><span><i class="dot" style="background:#d7dde8"></i>unused</span>`}
function selectFullToken(token){renderFullCube(Number(token))}
function buildPositionMap(cp){const groups=new Map(),result=new Map();cp.skills.flatMap(skill=>skill.occurrences).forEach(o=>{const key=`${o.task_id}:${o.episode_id}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(o)});groups.forEach(occurrences=>{occurrences.sort((a,b)=>Number(a.skill_index)-Number(b.skill_index)||Number(a.frame_start)-Number(b.frame_start));const total=occurrences.length;occurrences.forEach((o,index)=>result.set(o,{rank:index+1,total,percent:100*(index+.5)/total,id:Number(o.skill_index)}))});return result}
const positionCache=new Map();
function positionsFor(cp){if(!positionCache.has(cp))positionCache.set(cp,buildPositionMap(cp));return positionCache.get(cp)}
function entriesFor(cp){const map=positionsFor(cp);return cp.skills.flatMap(skill=>skill.occurrences).map(o=>({o,info:map.get(o)}))}
function positionMatches(o){const info=positionByOccurrence.get(o);if(!info||positionMode==='all')return true;const range=positionRanges[positionMode],low=Math.min(...range),high=Math.max(...range),value=positionMode==='percent'?info.percent:info.id;return value>=low&&value<=high}
let tableUnit='id';
function columnKey(info){return tableUnit==='percent'?Math.min(9,Math.floor(info.percent/10)):tableUnit==='rank'?info.rank:Number(info.id)}
function columnLabel(key){return tableUnit==='percent'?`${key*10}\\u2013${key*10+10}%`:tableUnit==='rank'?`${key}${['th','st','nd','rd'][key%10<4&&Math.floor(key/10)%10!==1?key%10:0]}`:`#${key}`}
const tally=list=>({codes:new Set(list.map(entry=>Number(entry.o.token))).size,occurrences:list.length});
const usageCell=(value,peak)=>value.occurrences?`<td style="background:rgba(40,120,181,${(0.05+0.4*value.codes/peak).toFixed(3)})" title="${value.occurrences} occurrences">${value.codes}</td>`:'<td class="zero">\\u00b7</td>';
function usageTable(rows){
  const filled=rows.filter(row=>row.entries.length);
  if(!filled.length)return '<div class="no-rows">No occurrences for this selection.</div>';
  const columns=[...new Set(filled.flatMap(row=>row.entries.map(entry=>columnKey(entry.info))))].sort((a,b)=>a-b);
  const counted=rows.map(row=>({row,cells:columns.map(key=>tally(row.entries.filter(entry=>columnKey(entry.info)===key))),total:tally(row.entries)}));
  const peak=Math.max(1,...counted.flatMap(item=>item.cells.map(cell=>cell.codes)));
  const head=`<tr><th class="label">checkpoint</th><th class="label">task</th>${columns.map(key=>`<th>${esc(columnLabel(key))}</th>`).join('')}<th>all orders</th></tr>`;
  const body=counted.map(({row,cells,total})=>`<tr class="${row.summary?'summary':''}"><td class="label">${esc(row.checkpoint)}</td><td class="label">${esc(row.task)}</td>${cells.map(value=>usageCell(value,peak)).join('')}<td class="total" title="${total.occurrences} occurrences">${total.codes}</td></tr>`).join('');
  return `<div class="table-scroll"><table class="usage"><thead>${head}</thead><tbody>${body}</tbody></table></div>`}
function checkpointTaskTable(){
  const perCheckpoint=checkpoints.map(cp=>({tag:cp.epoch_tag,entries:entriesFor(cp)}));
  const tasks=[...new Set(perCheckpoint.flatMap(item=>item.entries.map(entry=>Number(entry.o.task_id))))].sort((a,b)=>a-b);
  if(!tasks.length)return '<div class="no-rows">No occurrences.</div>';
  const counted=perCheckpoint.map(({tag,entries})=>({tag,cells:tasks.map(task=>tally(entries.filter(entry=>Number(entry.o.task_id)===task))),total:tally(entries)}));
  const peak=Math.max(1,...counted.flatMap(item=>item.cells.map(cell=>cell.codes)));
  const head=`<tr><th class="label">checkpoint</th>${tasks.map(task=>`<th>task ${task}</th>`).join('')}<th>all tasks</th></tr>`;
  const body=counted.map(({tag,cells,total})=>`<tr><td class="label">${esc(tag)}</td>${cells.map(value=>usageCell(value,peak)).join('')}<td class="total" title="${total.occurrences} occurrences">${total.codes}</td></tr>`).join('');
  return `<div class="table-scroll"><table class="usage"><thead>${head}</thead><tbody>${body}</tbody></table></div>`}
function cohesionStats(cp){
  const mean=values=>values.length?values.reduce((a,b)=>a+b,0)/values.length:0;
  const entries=entriesFor(cp);const cells=new Map();
  entries.forEach(entry=>{const key=`${Number(entry.o.task_id)}:${columnKey(entry.info)}`;if(!cells.has(key))cells.set(key,[]);cells.get(key).push(Number(entry.o.token))});
  const perCell=[...cells.values()].map(tokens=>{const counts=new Map();tokens.forEach(token=>counts.set(token,(counts.get(token)||0)+1));let entropy=0;counts.forEach(count=>{const p=count/tokens.length;entropy-=p*Math.log(p)});return{distinct:counts.size,effective:Math.exp(entropy)}});
  const size=cp.levels.reduce((a,b)=>a*b,1);const used=cp.train_codebook_used;
  const effective=mean(perCell.map(s=>s.effective));
  const globalEffective=cp.train_codebook_effective==null?null:Number(cp.train_codebook_effective);
  const utilization=used==null?null:100*Number(used)/size;
  return{effective,utilization,norm:globalEffective?effective/globalEffective:null,globalEffective,distinct:mean(perCell.map(s=>s.distinct)),occurrences:entries.length,usage:used==null?'?':`${used}/${size} (${utilization.toFixed(1)}%)`}}
function cohesionTable(){
  const stats=new Map();
  models.forEach((model,index)=>model.checkpoints.forEach(cp=>stats.set(`${index}:${cp.epoch_tag}`,cohesionStats(cp))));
  if(!stats.size)return '<div class="no-rows">No occurrences.</div>';
  const tagKey=tag=>{const match=/^epoch(\\d+)$/.exec(tag);return match?[0,Number(match[1]),'']:[1,0,tag]};
  const tags=[...new Set(models.flatMap(model=>model.checkpoints.map(cp=>cp.epoch_tag)))].sort((a,b)=>{const ka=tagKey(a),kb=tagKey(b);return ka[0]-kb[0]||ka[1]-kb[1]||String(ka[2]).localeCompare(String(kb[2]))});
  const head=`<tr><th class="label">checkpoint</th>${models.map(model=>`<th>${esc(model.name)}${model.mismatched?' \\u26a0':''}</th>`).join('')}</tr>`;
  const body=tags.map(tag=>{
    const cells=models.map((model,index)=>stats.get(`${index}:${tag}`));
    const ranked=[...new Set(cells.filter((cell,index)=>cell&&!models[index].mismatched).map(cell=>cell.effective))].sort((a,b)=>a-b);
    const rendered=cells.map((cell,index)=>{
      if(!cell)return '<td class="zero">\\u00b7</td>';
      let style='',cls='';
      if(!models[index].mismatched&&ranked.length){
        const rank=ranked.indexOf(cell.effective);
        const strength=ranked.length>1?1-rank/(ranked.length-1):1;
        const mix=(from,to)=>Math.round(from+(to-from)*strength);
        style=` style="background:rgb(${mix(250,105)},${mix(252,178)},${mix(250,105)})"`;
        if(rank===0)cls=' class="best"';
      }
      const utilization=cell.utilization==null?'?':`${cell.utilization.toFixed(1)}%`;
      const value=`${cell.effective.toFixed(2)} (${utilization})`;
      const normPart=cell.norm==null?'':`normalized ${cell.norm.toFixed(3)} \\u00b7 `;
      const globalPart=cell.globalEffective==null?'':`train-wide effective ${cell.globalEffective.toFixed(2)} \\u00b7 `;
      return `<td${cls}${style} title="${normPart}${globalPart}distinct ${cell.distinct.toFixed(2)} \\u00b7 codebook used (train) ${cell.usage} \\u00b7 ${cell.occurrences} occurrences">${value}</td>`}).join('');
    return `<tr><td class="label">${esc(tag)}</td>${rendered}</tr>`}).join('');
  return `<div class="table-scroll"><table class="usage"><thead>${head}</thead><tbody>${body}</tbody></table></div>`}
function taskRows(cp,entries){
  const tasks=[...new Set(entries.map(entry=>Number(entry.o.task_id)))].sort((a,b)=>a-b);
  const rows=tasks.map(task=>({checkpoint:cp.epoch_tag,task:`task ${task}`,entries:entries.filter(entry=>Number(entry.o.task_id)===task)}));
  if(rows.length>1)rows.push({checkpoint:cp.epoch_tag,task:'all tasks',entries,summary:true});
  return rows}
function renderTables(){
  const cp=current(),size=cp.levels.reduce((a,b)=>a*b,1);
  const selectionEntries=entriesFor(cp).filter(entry=>selectedTasks.has(Number(entry.o.task_id))&&positionMatches(entry.o));
  const selectionRows=[...selectedTasks].sort((a,b)=>a-b).map(task=>({checkpoint:cp.epoch_tag,task:`task ${task}`,entries:selectionEntries.filter(entry=>Number(entry.o.task_id)===task)}));
  if(selectionRows.length>1)selectionRows.push({checkpoint:cp.epoch_tag,task:'all tasks',entries:selectionEntries,summary:true});
  const overviewRows=checkpoints.flatMap(item=>taskRows(item,entriesFor(item)));
  document.getElementById('tables').innerHTML=
    `<details class="panel" open><summary>Cohesion \\u00b7 mean effective codes per cell (full-data codebook utilization) \\u00b7 lower effective is better</summary><p class="hint">Rows are checkpoints, columns are models; each cell shows "effective (utilization)". Effective is the mean entropy-based code count over task \\u00d7 skill-order cells ("Table order" above picks the order unit): two codes split 90/10 score 1.38 while 50/50 scores 2.00 \\u2014 lower means the same situation maps to fewer codes. Utilization is the percentage of the full codebook used by the checkpoint's entire training latent artifact, independent of this report's task_ids selection. Cells are shaded green by effective-value rank within each checkpoint row: the lowest effective value is darkest and later ranks fade toward white (ties share a rank); utilization does not affect the shading. Hover a value for normalized cohesion (effective \\u00f7 train-wide effective \\u2014 a diagnostic for codebook under-use when picking a checkpoint within one model, not a cross-model ranking), train-wide effective codes, distinct codes, codebook used, and occurrence count. Cohesion covers every model tab, task and order, ignoring the tab and filter controls. \\u26a0 marks models whose replay selection differs from the first model \\u2014 excluded from the highlight.</p>${cohesionTable()}</details>`+
    `<details class="panel" open><summary>Skill variety \\u00b7 checkpoint \\u00d7 task</summary><p class="hint">Distinct FSQ codes used in each task, over every skill order in that task. Independent of the controls above. Codebook size ${size}.</p>${checkpointTaskTable()}</details>`+
    `<details class="panel" open><summary>Codebook usage \\u00b7 current filters</summary><p class="hint">Distinct FSQ codes per checkpoint \\u00d7 task \\u00d7 skill order, limited to the checkpoint, tasks and skill-position range selected above. Cell titles show the occurrence count; "all orders" is the union over orders, not the column sum. Codebook size ${size}.</p>${usageTable(selectionRows)}</details>`+
    `<details class="panel" open><summary>Codebook usage \\u00b7 every checkpoint (unfiltered)</summary><p class="hint">The same counts over every checkpoint, task and skill order in this report, ignoring the controls above.</p>${usageTable(overviewRows)}</details>`}
function occurrenceCard(o){const info=positionByOccurrence.get(o),position=info?`${info.rank}/${info.total} · ${info.percent.toFixed(1)}%`:'';const figure=(src,label)=>src?`<figure><img loading="lazy" src="${esc(src)}" alt="${label}"><figcaption>${label}</figcaption></figure>`:'';return `<article class="occ"><div class="occ-title">episode ${o.episode_id} · skill ${o.skill_index}</div><div class="meta">position ${position} · frames [${o.frame_start}, ${o.frame_end}) · length ${o.length}</div><div class="pair">${figure(o.start_image_path,'GT start')}${figure(o.final_image_path,'GT end')}</div></article>`}
function selectToken(token){const skill=byToken.get(Number(token));if(!skill)return;renderCube(token);document.getElementById('selected').innerHTML=`token #${token} [${skill.coord.join(', ')}] <span class="count">${skill.occurrences.length} occurrences</span>`;const groups=new Map();skill.occurrences.forEach(o=>{const task=Number(o.task_id);if(!groups.has(task))groups.set(task,[]);groups.get(task).push(o)});document.getElementById('content').innerHTML=[...groups.entries()].sort((a,b)=>a[0]-b[0]).map(([task,occurrences])=>`<section class="task-group"><div class="task-group-title">Task ${task}: ${esc(occurrences[0].task_description||'')} <span class="task-count">${occurrences.length} videos</span></div><div class="occ-row">${occurrences.map(occurrenceCard).join('')}</div></section>`).join('')}
function renderTaskFilters(reset){const tasks=current().task_ids.map(Number);if(reset)selectedTasks=new Set(tasks);else selectedTasks=new Set([...selectedTasks].filter(t=>tasks.includes(t)));const box=document.getElementById('tasks');box.innerHTML=tasks.map(t=>`<label class="task-chip"><input type="checkbox" value="${t}" ${selectedTasks.has(t)?'checked':''}>${t}</label>`).join('');box.querySelectorAll('input').forEach(input=>input.addEventListener('change',()=>{const task=Number(input.value);if(input.checked)selectedTasks.add(task);else selectedTasks.delete(task);refresh()}))}
function refresh(){const model=models[modelIndex],cp=current();positionByOccurrence=positionsFor(cp);renderTables();renderFullCube();activeSkills=cp.skills.map(skill=>({...skill,occurrences:skill.occurrences.filter(o=>selectedTasks.has(Number(o.task_id))&&positionMatches(o))})).filter(skill=>skill.occurrences.length);byToken=new Map(activeSkills.map(skill=>[Number(skill.token),skill]));const tasks=[...selectedTasks].sort((a,b)=>a-b);const occurrences=activeSkills.reduce((sum,skill)=>sum+skill.occurrences.length,0),range=positionMode==='all'?'all':`${positionRanges[positionMode][0]}–${positionRanges[positionMode][1]}${positionMode==='percent'?'%':' ID'}`;document.getElementById('summary').textContent=`${model.name} · ${cp.epoch_tag} · GT start/end frames · ${model.target_task} · tasks ${tasks.length?tasks.join(', '):'none'} · position ${range} · ${occurrences} skill occurrences${(model.excluded_epoch_tags||[]).length?` \\u00b7 excluded (different replay settings): ${model.excluded_epoch_tags.join(', ')}`:''}`;if(activeSkills.length){const initial=activeSkills.slice().sort((a,b)=>b.occurrences.length-a.occurrences.length||a.token-b.token)[0].token;selectToken(initial)}else{renderCube(-1);document.getElementById('selected').textContent='No used code for the selected filters';document.getElementById('content').innerHTML='<div class="empty">No occurrences for the selected task and skill-position filters.</div>'}}
function configurePositionRange(){const range=document.getElementById('positionRange'),start=document.getElementById('positionStart'),end=document.getElementById('positionEnd'),unit=document.getElementById('positionUnit');range.hidden=positionMode==='all';if(positionMode==='all')return;const values=positionRanges[positionMode],percent=positionMode==='percent';start.min=0;end.min=0;start.max=percent?100:maximumSkillId;end.max=percent?100:maximumSkillId;start.step=percent?'0.1':'1';end.step=start.step;start.value=values[0];end.value=values[1];unit.textContent=percent?'%':'ID'}
const checkpointSelect=document.getElementById('checkpoint');
function renderCheckpointSelect(){checkpointSelect.innerHTML=checkpoints.map((cp,index)=>`<option value="${index}">${esc(cp.epoch_tag)}</option>`).join('');checkpointSelect.value=String(checkpointIndex)}
checkpointSelect.addEventListener('change',()=>{checkpointIndex=Number(checkpointSelect.value);selectedFullToken=-1;renderTaskFilters(false);refresh()});
function renderModelTabs(){const box=document.getElementById('modelTabs');if(models.length<2){box.hidden=true;return}box.hidden=false;box.innerHTML=models.map((model,index)=>`<button type="button" class="tab${index===modelIndex?' active':''}" data-index="${index}">${esc(model.name)}${model.mismatched?' \\u26a0':''}</button>`).join('');box.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>selectModel(Number(button.dataset.index))))}
function selectModel(index){if(index===modelIndex)return;modelIndex=index;checkpoints=models[index].checkpoints;checkpointIndex=0;selectedFullToken=-1;renderModelTabs();renderCheckpointSelect();renderTaskFilters(true);refresh()}
const positionModeSelect=document.getElementById('positionMode'),positionStart=document.getElementById('positionStart'),positionEnd=document.getElementById('positionEnd');positionModeSelect.addEventListener('change',()=>{positionMode=positionModeSelect.value;configurePositionRange();refresh()});positionStart.addEventListener('input',()=>{positionRanges[positionMode][0]=Number(positionStart.value);refresh()});positionEnd.addEventListener('input',()=>{positionRanges[positionMode][1]=Number(positionEnd.value);refresh()});configurePositionRange();
const tableUnitSelect=document.getElementById('tableUnit');tableUnitSelect.addEventListener('change',()=>{tableUnit=tableUnitSelect.value;renderTables()});
document.getElementById('allTasks').addEventListener('click',()=>{renderTaskFilters(true);refresh()});document.getElementById('clearTasks').addEventListener('click',()=>{selectedTasks.clear();renderTaskFilters(false);refresh()});
document.querySelectorAll('.cube-mode').forEach(button=>button.addEventListener('click',()=>{cubeMode=button.dataset.mode;document.querySelectorAll('.cube-mode').forEach(other=>other.classList.toggle('active',other===button));renderCube(selectedToken)}));
if(models.length&&checkpoints.length){renderModelTabs();renderCheckpointSelect();renderTaskFilters(true);refresh()}else{document.getElementById('content').innerHTML='<div class="empty">No completed checkpoints.</div>'}
</script></body></html>""".replace(
        "__REPORT_TITLE__", escape_html(report_title)
    ).replace("__DATA_BOOTSTRAP__", bootstrap).replace("__DATA_SCRIPTS__", data_scripts)
    path = output_dir / "index.html"
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(path)
    active_data_names = {data_path.name for data_path in data_paths}
    for stale_path in output_dir.glob("report-data-*.js"):
        if stale_path.name not in active_data_names:
            stale_path.unlink()
    return path


def maybe_merge_chunks(output_dir: str | Path, *, expected_chunks: int) -> Path | None:
    output_dir = Path(output_dir)
    metrics = output_dir / "metrics"
    chunks_dir = metrics / "chunks"
    metrics.mkdir(parents=True, exist_ok=True)
    with (metrics / "merge.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        paths = [chunks_dir / f"chunk_{index:03d}.json" for index in range(expected_chunks)]
        if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
            return None
        chunks = [json.loads(path.read_text()) for path in paths]
        if not all(chunk.get("completed", False) for chunk in chunks):
            return None
        signature = chunks[0]["signature"]
        request = chunks[0].get("request")
        levels = chunks[0]["levels"]
        records: dict[str, dict] = {}
        for index, chunk in enumerate(chunks):
            if (
                chunk["signature"] != signature
                or chunk.get("request") != request
                or chunk["levels"] != levels
            ):
                raise ValueError(f"FSQ GT replay chunk {index} contract mismatch.")
            if int(chunk["chunk_index"]) != index or int(chunk["chunk_count"]) != expected_chunks:
                raise ValueError(f"FSQ GT replay chunk {index} has invalid worker identity.")
            overlap = set(records) & set(chunk["records"])
            if overlap:
                raise ValueError(f"Duplicate replay occurrences: {sorted(overlap)[:5]}")
            records.update(chunk["records"])
        merged = {
            "signature": signature,
            "request": request,
            "run_name": chunks[0]["run_name"],
            "model_name": chunks[0].get("model_name") or chunks[0]["run_name"],
            "report_title": chunks[0].get("report_title") or "",
            "epoch_tag": chunks[0]["epoch_tag"],
            "levels": levels,
            "train_codebook_counts": chunks[0].get("train_codebook_counts"),
            "train_codebook_used": chunks[0].get("train_codebook_used"),
            "train_codebook_effective": chunks[0].get("train_codebook_effective"),
            "records": records,
            "completed": True,
        }
        _atomic_json(metrics / "manifest.json", merged)
        return write_html_report(output_dir, report_payload(merged))


def _comparable_signature(manifest: dict) -> dict:
    signature = manifest.get("signature") or {}
    return {key: signature.get(key) for key in _COLLECTION_SIGNATURE_KEYS}


def _tag_sort_key(tag: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"epoch(\d+)", tag)
    return (0, int(match.group(1)), "") if match else (1, 0, tag)


def _backfill_train_codebook_used(path: Path, manifest: dict) -> None:
    """Fill codebook-usage fields on manifests written before they existed."""
    if (
        manifest.get("train_codebook_counts") is not None
        and manifest.get("train_codebook_used") is not None
        and manifest.get("train_codebook_effective") is not None
    ):
        return
    latents_path = str((manifest.get("signature") or {}).get("latents_path") or "")
    if not latents_path or not Path(latents_path).is_file():
        return
    try:
        import numpy as np
    except ImportError:
        print(
            f"numpy unavailable; leaving codebook usage empty for {path}. "
            "Re-run with the project venv python to backfill it."
        )
        return
    tokens = np.asarray(np.load(latents_path)["tokens"], dtype=np.int64)
    levels = [int(value) for value in manifest.get("levels") or []]
    codebook_size = int(np.prod(levels)) if levels else 0
    counts = np.bincount(tokens, minlength=codebook_size)
    probabilities = counts[counts > 0] / tokens.size
    manifest["train_codebook_counts"] = [int(count) for count in counts]
    manifest["train_codebook_used"] = int((counts > 0).sum())
    manifest["train_codebook_effective"] = float(
        np.exp(-(probabilities * np.log(probabilities)).sum())
    )
    _atomic_json(path, manifest)


def completed_manifests(collection_dir: str | Path) -> dict[str, dict]:
    """Every finished per-checkpoint manifest on disk, keyed by epoch tag."""
    root = Path(collection_dir) / "checkpoints"
    manifests: dict[str, dict] = {}
    for path in sorted(root.glob("*/metrics/manifest.json")):
        if path.stat().st_size == 0:
            continue
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not manifest.get("completed", False):
            continue
        tag = str(manifest.get("epoch_tag") or "")
        if tag != path.parents[1].name:
            raise ValueError(
                f"Checkpoint manifest {path} declares epoch tag {tag!r}."
            )
        _backfill_train_codebook_used(path, manifest)
        manifests[tag] = manifest
    return manifests


def _partition_compatible(
    available: dict[str, dict], reference_tag: str
) -> tuple[list[dict], list[str]]:
    """Split checkpoints into those comparable with reference_tag and the rest."""
    reference = available[reference_tag]
    signature = _comparable_signature(reference)
    kept: list[dict] = []
    excluded: list[str] = []
    for tag in sorted(available, key=_tag_sort_key):
        manifest = available[tag]
        if (
            _comparable_signature(manifest) == signature
            and manifest.get("run_name") == reference.get("run_name")
            and manifest.get("levels") == reference.get("levels")
        ):
            kept.append(manifest)
        else:
            excluded.append(tag)
    return kept, excluded


def _write_collection(
    collection_dir: str | Path,
    available: dict[str, dict],
    manifests: list[dict],
    excluded: list[str],
) -> Path:
    collection_dir = Path(collection_dir)
    # Each per-checkpoint report was rendered whenever that checkpoint finished, so
    # re-render them from their manifests: adding a checkpoint later must not leave
    # the collection holding reports built by different versions of this file.
    for tag, manifest in available.items():
        write_html_report(
            collection_dir / "checkpoints" / tag, report_payload(manifest)
        )
    payload = collection_payload(manifests)
    if excluded:
        payload["excluded_epoch_tags"] = excluded
    _atomic_json(collection_dir / "metrics" / "collection.json", payload)
    return write_html_report(collection_dir, payload)


def rebuild_collection(
    collection_dir: str | Path, *, reference_tag: str | None = None
) -> tuple[Path, list[str], list[str]]:
    """Rebuild collection.json and index.html from the manifests already on disk."""
    collection_dir = Path(collection_dir)
    metrics = collection_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    with (metrics / "collection.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        available = completed_manifests(collection_dir)
        if not available:
            raise FileNotFoundError(
                f"No completed checkpoint manifest under {collection_dir / 'checkpoints'}."
            )
        if reference_tag is None:
            reference_tag = sorted(available, key=_tag_sort_key)[-1]
        elif reference_tag not in available:
            raise ValueError(
                f"Reference checkpoint {reference_tag!r} is not among the completed "
                f"checkpoints {sorted(available, key=_tag_sort_key)}."
            )
        manifests, excluded = _partition_compatible(available, reference_tag)
        path = _write_collection(collection_dir, available, manifests, excluded)
        return path, [manifest["epoch_tag"] for manifest in manifests], excluded


def maybe_merge_collection(
    collection_dir: str | Path, *, expected_epoch_tags: list[str]
) -> Path | None:
    collection_dir = Path(collection_dir)
    metrics = collection_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    with (metrics / "collection.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        available = completed_manifests(collection_dir)
        if any(tag not in available for tag in expected_epoch_tags):
            return None
        # Checkpoints from earlier runs share this collection directory, so merge
        # everything comparable rather than only what this run was asked to produce.
        manifests, excluded = _partition_compatible(available, expected_epoch_tags[0])
        dropped = [tag for tag in expected_epoch_tags if tag in excluded]
        if dropped:
            raise ValueError(
                "FSQ replay checkpoints of one run disagree on their replay "
                f"signature: {dropped}."
            )
        return _write_collection(collection_dir, available, manifests, excluded)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild one run's FSQ GT replay report, or combine several run "
            "collections into a tabbed model-comparison report with --compare."
        )
    )
    parser.add_argument("collection_dir", type=Path, nargs="?")
    parser.add_argument(
        "--reference",
        default=None,
        help="Epoch tag whose replay settings define comparability (default: newest).",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="+",
        default=None,
        help="Two or more run collection directories to combine into model tabs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Comparison output directory (default: <common parent>/compare).",
    )
    args = parser.parse_args()
    if args.compare is not None:
        if args.collection_dir is not None:
            raise SystemExit("Pass either a collection_dir or --compare, not both.")
        output_dir = args.output or (
            Path(os.path.commonpath([str(path) for path in args.compare])) / "compare"
        )
        payload = compare_payload(args.compare, output_dir=output_dir)
        _atomic_json(output_dir / "metrics" / "compare.json", payload)
        path = write_html_report(output_dir, payload)
        for model in payload["models"]:
            tags = ", ".join(item["epoch_tag"] for item in model["checkpoints"])
            note = " (mismatched replay settings)" if model.get("mismatched") else ""
            print(f"model {model['name']}{note}: {tags}")
        print(f"report: {path}")
        return
    if args.collection_dir is None:
        raise SystemExit("collection_dir is required unless --compare is used.")
    path, kept, excluded = rebuild_collection(
        args.collection_dir, reference_tag=args.reference
    )
    print(f"checkpoints: {', '.join(kept)}")
    if excluded:
        print(f"excluded (different replay settings): {', '.join(excluded)}")
    print(f"report: {path}")


if __name__ == "__main__":
    main()
