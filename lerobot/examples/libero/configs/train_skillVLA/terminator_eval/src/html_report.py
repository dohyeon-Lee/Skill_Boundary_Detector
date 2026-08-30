"""Interactive FSQ-cube report for multi-terminator skill rollouts."""

from __future__ import annotations

import json
from pathlib import Path


def write_html_report(output_dir: str | Path, payload: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html_path = output_dir / "index.html"
    html_text = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Multi-terminator skill evaluation</title>
  <style>
    :root { --ink:#17202a; --muted:#667085; --line:#d4dbe6; --blue:#2878b5; --red:#d62728; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,Arial,sans-serif; background:#f4f6f9; color:var(--ink); }
    header { position:sticky; top:0; z-index:5; padding:14px 20px; background:#fff; border-bottom:1px solid var(--line); }
    h1 { margin:0 0 5px; font-size:20px; }
    .subtitle,.muted { color:var(--muted); font-size:12px; }
    .layout { display:grid; grid-template-columns:minmax(440px,600px) 1fr; gap:16px; padding:16px; align-items:start; }
    .sidebar { position:sticky; top:86px; background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px; }
    .cube-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:8px; margin-bottom:8px; }
    .cube-panel { min-width:0; border:1px solid #e1e6ee; border-radius:8px; background:#fbfcfe; overflow:hidden; }
    .cube-slice-label { padding:5px 7px; border-bottom:1px solid #e1e6ee; color:#475467; font-size:11px; font-weight:700; text-align:center; }
    .cube { width:100%; height:auto; display:block; }
    .legend { display:flex; gap:14px; align-items:center; flex-wrap:wrap; font-size:12px; color:var(--muted); }
    .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; }
    .selected-info { margin-top:9px; padding:9px 10px; background:#f6f9fd; border-radius:7px; font-weight:700; }
    .content { min-width:0; }
    .empty { padding:32px; background:#fff; border:1px solid var(--line); border-radius:10px; text-align:center; color:var(--muted); }
    .occurrence { margin-bottom:16px; background:#fff; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
    .occ-title { padding:10px 12px; background:#f8fafc; border-bottom:1px solid var(--line); font-size:13px; font-weight:700; }
    .videos { padding:10px; overflow-x:auto; }
    .comparison-card { min-width:950px; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#fbfcfe; }
    .comparison-labels { display:grid; grid-template-columns:128fr repeat(5,256fr); }
    .comparison-label-spacer { border-bottom:3px solid transparent; background:#101318; }
    .comparison-label { padding:7px 8px; text-align:center; font-size:12px; font-weight:700; border-bottom:3px solid var(--branch,#98a2b3); }
    video { display:block; width:100%; background:#101318; object-fit:contain; }
    .final-frames { display:block; width:100%; object-fit:contain; border-top:2px solid var(--line); background:#101318; }
    .unavailable { display:grid; place-items:center; width:100%; aspect-ratio:4/3; background:#eceff3; color:#7b8492; padding:12px; text-align:center; font-size:12px; }
    .count { display:inline-block; margin-left:6px; padding:1px 6px; border-radius:10px; background:#e8eef7; color:#35516f; font-size:11px; }
    @media (max-width:1200px) {
      .layout { grid-template-columns:1fr; }
      .sidebar { position:relative; top:auto; max-width:650px; }
    }
  </style>
</head>
<body>
<header>
  <h1>Multi-terminator skill evaluation</h1>
  <div class="subtitle" id="summary"></div>
</header>
<main class="layout">
  <aside class="sidebar">
    <div class="cube-grid" id="cube-grid"></div>
    <div class="legend">
      <span><i class="dot" style="background:#d62728"></i>selected</span>
      <span><i class="dot" style="background:#2878b5"></i>used in selected episodes</span>
      <span><i class="dot" style="background:#d7dde8"></i>unused</span>
    </div>
    <div class="selected-info" id="selected-info"></div>
  </aside>
  <section class="content" id="content"></section>
</main>
<script>
const DATA = __DATA__;
const byToken = new Map(DATA.skills.map(s => [Number(s.token), s]));

function project(c, levels) {
  const lx=Math.max(1,levels[0]-1), ly=Math.max(1,levels[1]-1);
  const lz=levels.length>2 ? Math.max(1,levels[2]-1) : 1;
  const xn=(c[0]/lx-0.5)*2, yn=(c[1]/ly-0.5)*2;
  const zn=levels.length>2 ? (c[2]/lz-0.5)*2 : 0;
  const yaw=-0.63, pitch=0.46;
  const cyaw=Math.cos(yaw), syaw=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  const xr=cyaw*xn-syaw*yn, yr=syaw*xn+cyaw*yn;
  const scale=140;
  return [300+xr*scale, 275+yr*scale*sp-zn*scale*cp, yr*cp+zn*sp];
}

function coordForToken(token, levels) {
  const c=[]; let base=1;
  for (const level of levels) { c.push(Math.floor(token/base)%level); base*=level; }
  return c;
}

function extraSliceCoordinates(levels) {
  let slices=[[]];
  for(const level of levels.slice(3)) {
    const next=[];
    for(const prefix of slices) for(let value=0;value<level;value++) next.push([...prefix,value]);
    slices=next;
  }
  return slices;
}

function renderCubeSlice(svg, selectedToken, sliceCoord) {
  const levels=DATA.levels;
  const spatialLevels=[levels[0]||1,levels[1]||1,levels[2]||1];
  const maxToken=levels.reduce((a,b)=>a*b,1), NS="http://www.w3.org/2000/svg";
  svg.innerHTML="";
  const make=(name,attrs)=>{
    const el=document.createElementNS(NS,name);
    Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));
    svg.appendChild(el); return el;
  };
  for(let t=0;t<maxToken;t++) {
    const c=coordForToken(t,levels);
    if(c.slice(3).some((value,index)=>value!==sliceCoord[index])) continue;
    for(let d=0;d<Math.min(3,levels.length);d++) {
      const n=c.slice(); n[d]+=1;
      if(n[d]<levels[d]) {
        const [x1,y1]=project(c,spatialLevels),[x2,y2]=project(n,spatialLevels);
        make("line",{x1,y1,x2,y2,stroke:"rgba(100,100,100,.48)","stroke-width":1.5});
      }
    }
  }
  const BL=spatialLevels.map(l=>l-1);
  const corners=[0,1,2,3,4,5,6,7].map(b=>[BL[0]*(b&1),BL[1]*((b>>1)&1),(BL[2]||0)*((b>>2)&1)]);
  [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]].forEach(([a,b])=>{
    const [x1,y1]=project(corners[a],spatialLevels),[x2,y2]=project(corners[b],spatialLevels);
    make("line",{x1,y1,x2,y2,stroke:"rgba(50,50,50,.75)","stroke-width":2.2});
  });
  const pts=[];
  for(let t=0;t<maxToken;t++) {
    const c=coordForToken(t,levels);
    if(c.slice(3).some((value,index)=>value!==sliceCoord[index])) continue;
    const p=project(c,spatialLevels); pts.push({t,p,c,used:byToken.has(t)});
  }
  pts.sort((a,b)=>a.p[2]-b.p[2]);
  pts.forEach(({t,p,c,used})=>{
    const point=make("circle",{
      cx:p[0],cy:p[1],r:t===selectedToken?9:(used?6:3.8),
      fill:t===selectedToken?"#d62728":(used?"#2878b5":"#d7dde8"),
      stroke:t===selectedToken?"#8b0000":"#26384d",
      "stroke-width":t===selectedToken?2:.9,
      style:used?"cursor:pointer":"cursor:default"
    });
    if(used) point.addEventListener("click",()=>selectToken(t));
    const title=document.createElementNS(NS,"title");
    title.textContent=(used?`token #${t}: ${byToken.get(t).occurrences.length} occurrences`:`token #${t}: unused`)+` · [${c.join(", ")}]`;
    point.appendChild(title);
  });
}

function renderCube(selectedToken) {
  const grid=document.getElementById("cube-grid"), levels=DATA.levels;
  const NS="http://www.w3.org/2000/svg";
  grid.innerHTML="";
  for(const sliceCoord of extraSliceCoordinates(levels)) {
    const panel=document.createElement("div");
    panel.className="cube-panel";
    if(levels.length>3) {
      const label=document.createElement("div");
      label.className="cube-slice-label";
      label.textContent=sliceCoord.map((value,index)=>`dim ${index+4} = ${value}`).join(" · ");
      panel.appendChild(label);
    }
    const svg=document.createElementNS(NS,"svg");
    svg.setAttribute("class","cube");
    svg.setAttribute("viewBox","0 0 600 540");
    svg.setAttribute("aria-label",levels.length>3?`FSQ skill cube slice ${sliceCoord.join(",")}`:"FSQ skill cube");
    panel.appendChild(svg);
    grid.appendChild(panel);
    renderCubeSlice(svg,selectedToken,sliceCoord);
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
}

function comparisonCard(occ) {
  const labels=`<div class="comparison-label-spacer"></div>`+occ.branches.map(branch=>
    `<div class="comparison-label" style="--branch:${escapeHtml(branch.color||"#98a2b3")}">${escapeHtml(branch.label)}</div>`
  ).join("");
  if(!occ.comparison_path) {
    return `<div class="unavailable">Combined video was not generated for this legacy result.</div>`;
  }
  // The combined video has one 128px shared label gutter, five 256px branch
  // panels, and one 36px signal row per display model plus MAIN.
  const signalRows=(DATA.terminator_models||[]).length+1;
  const aspect=`${(128+5*256)/(256+36*signalRows)} / 1`;
  const finalFrames=occ.comparison_final_path
    ? `<img class="final-frames" style="aspect-ratio:${aspect}" loading="lazy" src="${escapeHtml(occ.comparison_final_path)}" alt="Final frame comparison" />`
    : "";
  const poster=occ.comparison_start_path
    ? ` poster="${escapeHtml(occ.comparison_start_path)}"`
    : "";
  // One MP4 contains all five synchronized branches. A single click therefore
  // starts GT, both exact-noise samples, early, and late together.
  return `<article class="comparison-card">
    <div class="comparison-labels">${labels}</div>
    <video controls preload="none" style="aspect-ratio:${aspect}"${poster} src="${escapeHtml(occ.comparison_path)}"></video>
    ${finalFrames}
  </article>`;
}

function selectToken(token) {
  const skill=byToken.get(Number(token)); if(!skill) return;
  renderCube(Number(token));
  document.getElementById("selected-info").innerHTML=
    `token #${skill.token} &nbsp; [${skill.coord.join(", ")}] <span class="count">${skill.occurrences.length} occurrences</span>`;
  const content=document.getElementById("content");
  if(!skill.occurrences.length) { content.innerHTML='<div class="empty">No occurrence.</div>'; return; }
  content.innerHTML=skill.occurrences.map(occ=>`<section class="occurrence">
    <div class="occ-title">Task ${occ.task_id}: ${escapeHtml(occ.task_description||"")}
    </div>
    <div class="videos">${comparisonCard(occ)}</div>
  </section>`).join("");
}

const displayLabels=(DATA.terminator_models||[]).map(model=>`${model.label}:${model.variant}`).join(", ");
document.getElementById("summary").textContent=
  `${DATA.model_label} · display terms ${displayLabels} · ${DATA.target_task} tasks ${DATA.task_ids.join(", ")} · ${DATA.selected_episode_count} episodes · ${DATA.occurrence_count} occurrences · shift ±${DATA.time_shift_offset}`;
const initial=DATA.skills.length?DATA.skills.slice().sort((a,b)=>b.occurrences.length-a.occurrences.length||a.token-b.token)[0].token:0;
if(DATA.skills.length) selectToken(initial); else { renderCube(-1); document.getElementById("content").innerHTML='<div class="empty">No skill occurrences selected.</div>'; }
</script>
</body>
</html>
""".replace("__DATA__", data_json)
    # A report may be opened through a web server as soon as the last array
    # worker finishes.  Replace an already-complete temporary file so readers
    # can never observe a truncated index.html.
    temporary = html_path.with_suffix(".html.tmp")
    temporary.write_text(html_text, encoding="utf-8")
    temporary.replace(html_path)
    return html_path
