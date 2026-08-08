#!/usr/bin/env python3
"""Race-safe manifest merge and interactive report for FSQ GT replays."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
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
        "epoch_tag": manifest["epoch_tag"],
        "target_task": signature["target_task"],
        "task_ids": sorted(int(value) for value in signature["selected_episodes"]),
        "episode_count": sum(
            len(values) for values in signature["selected_episodes"].values()
        ),
        "occurrence_count": len(manifest["records"]),
        "skills": skills,
    }


def collection_payload(manifests: list[dict]) -> dict:
    if not manifests:
        raise ValueError("At least one FSQ replay manifest is required.")
    run_names = {manifest["run_name"] for manifest in manifests}
    target_tasks = {
        manifest["signature"]["target_task"] for manifest in manifests
    }
    if len(run_names) != 1 or len(target_tasks) != 1:
        raise ValueError("FSQ replay checkpoints must share one run and task suite.")
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
        "target_task": next(iter(target_tasks)),
        "checkpoints": checkpoints,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_html_report(output_dir: str | Path, payload: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "checkpoints" not in payload:
        payload = {
            "format": "fsq_gt_replay_collection_v1",
            "run_name": payload["run_name"],
            "target_task": payload["target_task"],
            "checkpoints": [payload],
        }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>FSQ GT skill replay</title>
  <style>
    :root{--ink:#17202a;--muted:#667085;--line:#d4dbe6;--blue:#2878b5;--red:#d62728}
    *{box-sizing:border-box} body{margin:0;font-family:Inter,Arial,sans-serif;background:#f4f6f9;color:var(--ink)}
    header{position:sticky;top:0;z-index:5;padding:12px 20px;background:#fff;border-bottom:1px solid var(--line)}
    h1{margin:0 0 5px;font-size:20px}.subtitle{color:var(--muted);font-size:12px}
    .controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:9px}.control{display:flex;align-items:center;gap:7px;font-size:12px;font-weight:700}.control select,.range-input{padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:#fff}.range-input{width:72px}.range{display:flex;align-items:center;gap:5px}.tasks{display:flex;align-items:center;gap:5px;flex-wrap:wrap}.task-chip{padding:4px 7px;border:1px solid var(--line);border-radius:12px;background:#f8fafc;font-weight:500}.task-chip input{margin:0 4px 0 0}.small-button{padding:4px 7px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
    .layout{display:grid;grid-template-columns:minmax(420px,580px) 1fr;gap:16px;padding:16px;align-items:start}
    .sidebar{position:sticky;top:126px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px}
    .cube{width:100%;height:auto;display:block}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
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
    table.usage tr.summary td{background:#e9eef6;font-weight:800}table.usage tr.summary td.total{background:#dfe7f2}
    .no-rows{padding:14px;color:var(--muted);font-size:12px}
    @media(max-width:1100px){.layout{grid-template-columns:1fr}.sidebar{position:relative;top:auto;max-width:620px}}
  </style>
</head>
<body>
<header><h1>FSQ GT skill replay</h1><div class="subtitle" id="summary"></div>
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
    <div class="legend"><span><i class="dot" style="background:#d62728"></i>selected</span><span><i class="dot" style="background:#2878b5"></i>used</span><span><i class="dot" style="background:#d7dde8"></i>unused</span></div>
    <div class="selected" id="selected"></div>
  </aside>
  <div class="main-col">
    <div class="tables" id="tables"></div>
    <section id="content"></section>
  </div>
</main>
<script>
const DATA=__DATA__, checkpoints=DATA.checkpoints;
let checkpointIndex=0,selectedTasks=new Set(),activeSkills=[],byToken=new Map(),positionByOccurrence=new Map(),positionMode='all';
const maximumSkillId=Math.max(0,...checkpoints.flatMap(cp=>cp.skills.flatMap(skill=>skill.occurrences.map(o=>Number(o.skill_index)))));const positionRanges={percent:[0,100],id:[0,maximumSkillId]};
const esc=v=>String(v).replace(/[&<>'"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
const current=()=>checkpoints[checkpointIndex];
function coord(token,levels){const c=[];let base=1;for(const level of levels){c.push(Math.floor(token/base)%level);base*=level}return c}
function project(c,levels){const n=levels.length,maxL=Math.max(...levels),scaleDen=Math.max(1,maxL-1)/2;
  const v=levels.map((level,i)=>((c[i]||0)-(level-1)/2)/scaleDen),ox=300,oy=265;
  if(n<3)return[ox+(v[0]||0)*190,oy-(v[1]||0)*190,0];
  const yaw=-.63,pitch=.46,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const xr=cy*v[0]-sy*v[1],yr=sy*v[0]+cy*v[1],zr=v[2]||0;return[ox+xr*145,oy+yr*145*sp-zr*145*cp,yr*cp+zr*sp]}
function renderCube(selected){const levels=current().levels,svg=document.querySelector('.cube'),NS='http://www.w3.org/2000/svg';svg.innerHTML='';
  const make=(name,attrs)=>{const e=document.createElementNS(NS,name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));svg.appendChild(e);return e};
  const total=levels.reduce((a,b)=>a*b,1);for(let t=0;t<total;t++){const c=coord(t,levels);for(let d=0;d<Math.min(3,levels.length);d++){if(c[d]+1<levels[d]){const n=c.slice();n[d]++;const a=project(c,levels),b=project(n,levels);make('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'rgba(100,100,100,.42)','stroke-width':1.2})}}}
  const points=[];for(let t=0;t<total;t++){const p=project(coord(t,levels),levels);points.push({t,p,used:byToken.has(t)})}points.sort((a,b)=>a.p[2]-b.p[2]);
  points.forEach(({t,p,used})=>{const e=make('circle',{cx:p[0],cy:p[1],r:t===selected?9:(used?6:3.5),fill:t===selected?'#d62728':(used?'#2878b5':'#d7dde8'),stroke:'#26384d','stroke-width':t===selected?2:.8,style:used?'cursor:pointer':'cursor:default'});if(used)e.addEventListener('click',()=>selectToken(t))})}
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
    `<details class="panel" open><summary>Skill variety \\u00b7 checkpoint \\u00d7 task</summary><p class="hint">Distinct FSQ codes used in each task, over every skill order in that task. Independent of the controls above. Codebook size ${size}.</p>${checkpointTaskTable()}</details>`+
    `<details class="panel" open><summary>Codebook usage \\u00b7 current filters</summary><p class="hint">Distinct FSQ codes per checkpoint \\u00d7 task \\u00d7 skill order, limited to the checkpoint, tasks and skill-position range selected above. Cell titles show the occurrence count; "all orders" is the union over orders, not the column sum. Codebook size ${size}.</p>${usageTable(selectionRows)}</details>`+
    `<details class="panel" open><summary>Codebook usage \\u00b7 every checkpoint (unfiltered)</summary><p class="hint">The same counts over every checkpoint, task and skill order in this report, ignoring the controls above.</p>${usageTable(overviewRows)}</details>`}
function occurrenceCard(o){const info=positionByOccurrence.get(o),position=info?`${info.rank}/${info.total} · ${info.percent.toFixed(1)}%`:'';const figure=(src,label)=>src?`<figure><img loading="lazy" src="${esc(src)}" alt="${label}"><figcaption>${label}</figcaption></figure>`:'';return `<article class="occ"><div class="occ-title">episode ${o.episode_id} · skill ${o.skill_index}</div><div class="meta">position ${position} · frames [${o.frame_start}, ${o.frame_end}) · length ${o.length}</div><div class="pair">${figure(o.start_image_path,'GT start')}${figure(o.final_image_path,'GT end')}</div></article>`}
function selectToken(token){const skill=byToken.get(Number(token));if(!skill)return;renderCube(token);document.getElementById('selected').innerHTML=`token #${token} [${skill.coord.join(', ')}] <span class="count">${skill.occurrences.length} occurrences</span>`;const groups=new Map();skill.occurrences.forEach(o=>{const task=Number(o.task_id);if(!groups.has(task))groups.set(task,[]);groups.get(task).push(o)});document.getElementById('content').innerHTML=[...groups.entries()].sort((a,b)=>a[0]-b[0]).map(([task,occurrences])=>`<section class="task-group"><div class="task-group-title">Task ${task}: ${esc(occurrences[0].task_description||'')} <span class="task-count">${occurrences.length} videos</span></div><div class="occ-row">${occurrences.map(occurrenceCard).join('')}</div></section>`).join('')}
function renderTaskFilters(reset){const tasks=current().task_ids.map(Number);if(reset)selectedTasks=new Set(tasks);else selectedTasks=new Set([...selectedTasks].filter(t=>tasks.includes(t)));const box=document.getElementById('tasks');box.innerHTML=tasks.map(t=>`<label class="task-chip"><input type="checkbox" value="${t}" ${selectedTasks.has(t)?'checked':''}>${t}</label>`).join('');box.querySelectorAll('input').forEach(input=>input.addEventListener('change',()=>{const task=Number(input.value);if(input.checked)selectedTasks.add(task);else selectedTasks.delete(task);refresh()}))}
function refresh(){const cp=current();positionByOccurrence=positionsFor(cp);renderTables();activeSkills=cp.skills.map(skill=>({...skill,occurrences:skill.occurrences.filter(o=>selectedTasks.has(Number(o.task_id))&&positionMatches(o))})).filter(skill=>skill.occurrences.length);byToken=new Map(activeSkills.map(skill=>[Number(skill.token),skill]));const tasks=[...selectedTasks].sort((a,b)=>a-b);const occurrences=activeSkills.reduce((sum,skill)=>sum+skill.occurrences.length,0),range=positionMode==='all'?'all':`${positionRanges[positionMode][0]}–${positionRanges[positionMode][1]}${positionMode==='percent'?'%':' ID'}`;document.getElementById('summary').textContent=`${DATA.run_name} · ${cp.epoch_tag} · GT start/end states · ${DATA.target_task} · tasks ${tasks.length?tasks.join(', '):'none'} · position ${range} · ${occurrences} skill occurrences${(DATA.excluded_epoch_tags||[]).length?` \\u00b7 excluded (different replay settings): ${DATA.excluded_epoch_tags.join(', ')}`:''}`;if(activeSkills.length){const initial=activeSkills.slice().sort((a,b)=>b.occurrences.length-a.occurrences.length||a.token-b.token)[0].token;selectToken(initial)}else{renderCube(-1);document.getElementById('selected').textContent='No used code for the selected filters';document.getElementById('content').innerHTML='<div class="empty">No occurrences for the selected task and skill-position filters.</div>'}}
function configurePositionRange(){const range=document.getElementById('positionRange'),start=document.getElementById('positionStart'),end=document.getElementById('positionEnd'),unit=document.getElementById('positionUnit');range.hidden=positionMode==='all';if(positionMode==='all')return;const values=positionRanges[positionMode],percent=positionMode==='percent';start.min=0;end.min=0;start.max=percent?100:maximumSkillId;end.max=percent?100:maximumSkillId;start.step=percent?'0.1':'1';end.step=start.step;start.value=values[0];end.value=values[1];unit.textContent=percent?'%':'ID'}
const checkpointSelect=document.getElementById('checkpoint');checkpointSelect.innerHTML=checkpoints.map((cp,index)=>`<option value="${index}">${esc(cp.epoch_tag)}</option>`).join('');checkpointSelect.addEventListener('change',()=>{checkpointIndex=Number(checkpointSelect.value);renderTaskFilters(false);refresh()});
const positionModeSelect=document.getElementById('positionMode'),positionStart=document.getElementById('positionStart'),positionEnd=document.getElementById('positionEnd');positionModeSelect.addEventListener('change',()=>{positionMode=positionModeSelect.value;configurePositionRange();refresh()});positionStart.addEventListener('input',()=>{positionRanges[positionMode][0]=Number(positionStart.value);refresh()});positionEnd.addEventListener('input',()=>{positionRanges[positionMode][1]=Number(positionEnd.value);refresh()});configurePositionRange();
const tableUnitSelect=document.getElementById('tableUnit');tableUnitSelect.addEventListener('change',()=>{tableUnit=tableUnitSelect.value;renderTables()});
document.getElementById('allTasks').addEventListener('click',()=>{renderTaskFilters(true);refresh()});document.getElementById('clearTasks').addEventListener('click',()=>{selectedTasks.clear();renderTaskFilters(false);refresh()});
if(checkpoints.length){renderTaskFilters(true);refresh()}else{document.getElementById('content').innerHTML='<div class="empty">No completed checkpoints.</div>'}
</script></body></html>""".replace("__DATA__", data)
    path = output_dir / "index.html"
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(path)
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
        levels = chunks[0]["levels"]
        records: dict[str, dict] = {}
        for index, chunk in enumerate(chunks):
            if chunk["signature"] != signature or chunk["levels"] != levels:
                raise ValueError(f"FSQ GT replay chunk {index} contract mismatch.")
            if int(chunk["chunk_index"]) != index or int(chunk["chunk_count"]) != expected_chunks:
                raise ValueError(f"FSQ GT replay chunk {index} has invalid worker identity.")
            overlap = set(records) & set(chunk["records"])
            if overlap:
                raise ValueError(f"Duplicate replay occurrences: {sorted(overlap)[:5]}")
            records.update(chunk["records"])
        merged = {
            "signature": signature,
            "run_name": chunks[0]["run_name"],
            "epoch_tag": chunks[0]["epoch_tag"],
            "levels": levels,
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
        description="Rebuild the combined FSQ GT replay report from finished checkpoints."
    )
    parser.add_argument("collection_dir", type=Path)
    parser.add_argument(
        "--reference",
        default=None,
        help="Epoch tag whose replay settings define comparability (default: newest).",
    )
    args = parser.parse_args()
    path, kept, excluded = rebuild_collection(
        args.collection_dir, reference_tag=args.reference
    )
    print(f"checkpoints: {', '.join(kept)}")
    if excluded:
        print(f"excluded (different replay settings): {', '.join(excluded)}")
    print(f"report: {path}")


if __name__ == "__main__":
    main()
