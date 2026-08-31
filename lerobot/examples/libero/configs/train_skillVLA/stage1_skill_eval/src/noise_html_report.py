"""Interactive linked-codebook report for repeated exact-start trajectories."""

from __future__ import annotations

import json
from pathlib import Path


def write_noise_html_report(output_dir: str | Path, payload: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html_path = output_dir / "index.html"
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Stage-1 exact-start noise trajectories</title>
  <style>
    :root { --ink:#17202a; --muted:#667085; --line:#d4dbe6; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,Arial,sans-serif; background:#f4f6f9; color:var(--ink); }
    header { position:sticky; top:0; z-index:5; padding:13px 20px; background:#fff; border-bottom:1px solid var(--line); }
    h1 { margin:0 0 4px; font-size:20px; }
    .subtitle,.muted { color:var(--muted); font-size:12px; }
    .toolbar { display:flex; align-items:center; gap:8px; margin-top:8px; color:#475467; font-size:11px; }
    .toolbar input { width:150px; }
    .layout { display:grid; grid-template-columns:minmax(440px,600px) 1fr; gap:16px; padding:16px; align-items:start; }
    .sidebar { position:sticky; top:112px; padding:12px; border:1px solid var(--line); border-radius:10px; background:#fff; }
    .codebooks { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:10px; }
    .codebook-card { min-width:0; padding:8px; border:1px solid #e0e6ef; border-radius:8px; background:#fbfcfe; }
    .codebook-title { overflow:hidden; margin:0 0 2px; color:#344054; font-size:11px; font-weight:800; text-overflow:ellipsis; white-space:nowrap; }
    .cube { display:block; width:100%; height:auto; }
    .legend { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-top:8px; color:var(--muted); font-size:11px; }
    .dot { display:inline-block; width:9px; height:9px; margin-right:4px; border-radius:50%; }
    .selected-info { margin-top:9px; padding:9px 10px; border-radius:7px; background:#f6f9fd; font-size:12px; font-weight:700; line-height:1.45; }
    .content { min-width:0; }
    .empty { padding:32px; border:1px solid var(--line); border-radius:10px; background:#fff; text-align:center; color:var(--muted); }
    .sample { margin-bottom:16px; overflow:hidden; border:1px solid var(--line); border-radius:10px; background:#fff; }
    .sample-title { padding:9px 12px; border-bottom:1px solid var(--line); background:#f8fafc; font-size:13px; font-weight:800; }
    .sample-subtitle { margin-top:3px; color:var(--muted); font-size:11px; font-weight:500; }
    .model-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:10px; padding:10px; }
    .trajectory-panel { min-width:0; overflow:hidden; border:1px solid #dbe2eb; border-radius:8px; background:#0f1720; }
    .panel-title { display:flex; justify-content:space-between; gap:8px; padding:7px 9px; background:#f8fafc; color:#344054; font-size:11px; font-weight:800; }
    .panel-title span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .code-controls { display:flex; align-items:center; gap:9px; padding:6px 8px; border-top:1px solid #dbe2eb; background:#eef2f7; }
    .code-controls .hint { max-width:145px; color:#667085; font-size:9px; line-height:1.35; }
    .mini-codebook { display:flex; min-width:0; align-items:center; gap:7px; }
    .mini-cube { display:block; width:150px; max-width:46%; height:auto; border:1px solid #d3dae5; border-radius:6px; background:#fff; }
    .mini-caption { color:#344054; font-size:9px; font-weight:800; line-height:1.4; }
    .canvas-wrap { position:relative; width:100%; background:#111820; }
    canvas { display:block; width:100%; height:auto; }
    .panel-stats { display:flex; flex-wrap:wrap; gap:5px; padding:7px 8px; background:#f8fafc; }
    .badge { padding:2px 6px; border:1px solid #d7dee8; border-radius:10px; background:#fff; color:#526071; font-size:10px; }
    .count { display:inline-block; margin-left:5px; padding:1px 6px; border-radius:10px; background:#e8eef7; color:#35516f; font-size:10px; }
    @media (max-width:1200px) { .layout { grid-template-columns:1fr; } .sidebar { position:relative; top:auto; max-width:650px; } }
  </style>
</head>
<body>
<header>
  <h1>Exact-start policy-noise trajectories</h1>
  <div class="subtitle" id="summary"></div>
  <div class="toolbar"><label for="opacity">trajectory opacity</label><input id="opacity" type="range" min="0.05" max="0.65" step="0.02" value="0.24" /><span id="opacity-value">0.24</span><span>· color = tested code · green = start · colored dot = final position</span></div>
</header>
<main class="layout">
  <aside class="sidebar">
    <div class="codebooks" id="codebooks"></div>
    <div class="legend">
      <span><i class="dot" style="background:#d62728"></i>clicked</span>
      <span><i class="dot" style="background:#f39c4a"></i>linked</span>
      <span><i class="dot" style="background:#2878b5"></i>used</span>
      <span><i class="dot" style="background:#d7dde8"></i>unused</span>
    </div>
    <div class="selected-info" id="selected-info"></div>
  </aside>
  <section class="content" id="content"></section>
</main>
<script>
const DATA=__DATA__;
const skillSpaces=DATA.skill_spaces||[];
const spaceByModel=new Map(skillSpaces.map(space=>[Number(space.model_index),space]));
const allRecords=DATA.occurrences||[];
const recordByUid=new Map(allRecords.map(record=>[String(record.uid),record]));
let selectedSkill=null;
let trajectoryOpacity=0.24;
const emphasizedCodeByRecord=new Map();

function escapeHtml(value) {
  return String(value??"").replace(/[&<>'"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
}
function coordForToken(token,levels) {
  const coord=[]; let base=1;
  for(const level of levels) { coord.push(Math.floor(token/base)%level); base*=level; }
  return coord;
}
function project(coord,levels) {
  const lx=Math.max(1,levels[0]-1),ly=Math.max(1,levels[1]-1);
  const lz=levels.length>2?Math.max(1,levels[2]-1):1;
  const xn=(coord[0]/lx-.5)*2,yn=(coord[1]/ly-.5)*2;
  const zn=levels.length>2?(coord[2]/lz-.5)*2:0;
  const yaw=-.63,pitch=.46,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const xr=cy*xn-sy*yn,yr=sy*xn+cy*yn,scale=140;
  return [300+xr*scale,275+yr*scale*sp-zn*scale*cp,yr*cp+zn*sp];
}
function memberIds(skill) { return (skill.member_ids||[]).map(String); }
function rolloutToken(record,rollout) { return Number(rollout.eval_token??record.token); }
function evaluatedTokens(record) {
  const original=Number(record.token),declared=(record.evaluated_tokens||[]).map(Number);
  const inferred=(record.rollouts||[]).map(rollout=>rolloutToken(record,rollout));
  const unique=[...new Set([original,...declared,...inferred])];
  return [original,...unique.filter(token=>token!==original).sort((a,b)=>a-b)];
}
function tokenHue(token) { return ((Number(token)*137.508+205)%360+360)%360; }
function tokenColor(token,alpha=1,lightness=52) { return `hsla(${tokenHue(token).toFixed(1)},82%,${lightness}%,${alpha})`; }
function selectedMembers() {
  if(!selectedSkill) return new Set();
  const space=spaceByModel.get(Number(selectedSkill.model_index));
  const skill=space&&(space.skills||[]).find(item=>Number(item.token)===Number(selectedSkill.token));
  return new Set(skill?memberIds(skill):[]);
}
function renderCube(space,svg,members) {
  const levels=space.levels,byToken=new Map((space.skills||[]).map(skill=>[Number(skill.token),skill]));
  const maxToken=levels.reduce((a,b)=>a*b,1),NS="http://www.w3.org/2000/svg";
  svg.innerHTML="";
  const make=(name,attrs)=>{ const el=document.createElementNS(NS,name); Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v)); svg.appendChild(el); return el; };
  for(let token=0;token<maxToken;token++) {
    const coord=coordForToken(token,levels);
    for(let dim=0;dim<Math.min(3,levels.length);dim++) {
      const next=coord.slice(); next[dim]+=1;
      if(next[dim]<levels[dim]) { const [x1,y1]=project(coord,levels),[x2,y2]=project(next,levels); make("line",{x1,y1,x2,y2,stroke:"rgba(100,100,100,.48)","stroke-width":1.5}); }
    }
  }
  const bounds=levels.map(level=>level-1),corners=[0,1,2,3,4,5,6,7].map(bit=>[bounds[0]*(bit&1),bounds[1]*((bit>>1)&1),(bounds[2]||0)*((bit>>2)&1)]);
  [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]].forEach(([a,b])=>{ const [x1,y1]=project(corners[a],levels),[x2,y2]=project(corners[b],levels); make("line",{x1,y1,x2,y2,stroke:"rgba(50,50,50,.75)","stroke-width":2.2}); });
  const points=[];
  for(let token=0;token<maxToken;token++) points.push({token,p:project(coordForToken(token,levels),levels),used:byToken.has(token)});
  points.sort((a,b)=>a.p[2]-b.p[2]);
  points.forEach(({token,p,used})=>{
    const skill=byToken.get(token),overlap=skill?memberIds(skill).filter(uid=>members.has(uid)).length:0;
    const clicked=Boolean(selectedSkill)&&Number(selectedSkill.model_index)===Number(space.model_index)&&Number(selectedSkill.token)===token;
    const linked=!clicked&&overlap>0;
    const point=make("circle",{cx:p[0],cy:p[1],r:clicked?9:(linked?8:(used?6:3.8)),fill:clicked?"#d62728":(linked?"#f39c4a":(used?"#2878b5":"#d7dde8")),stroke:(clicked||linked)?"#8b1a1a":"#26384d","stroke-width":(clicked||linked)?2.2:.9,style:used?"cursor:pointer":"cursor:default"});
    if(used) point.addEventListener("click",()=>selectSkillCode(space.model_index,token));
    const title=document.createElementNS(NS,"title"); title.textContent=used?`code #${token}: ${memberIds(skill).length} exact skills${overlap?` · ${overlap} linked`:""}`:`code #${token}: unused`; point.appendChild(title);
  });
}
function renderMiniCube(record,svg) {
  const space=spaceByModel.get(Number(record.model_index)); if(!space) return;
  const levels=space.levels.map(Number),tested=new Set(evaluatedTokens(record)),original=Number(record.token),selected=emphasizedCodeByRecord.has(String(record.uid))?Number(emphasizedCodeByRecord.get(String(record.uid))):null;
  const NS="http://www.w3.org/2000/svg",maxToken=levels.reduce((a,b)=>a*b,1),visibleLevels=[levels[0]||1,levels[1]||1,levels[2]||1],hiddenCount=levels.slice(3).reduce((a,b)=>a*b,1);
  svg.innerHTML=""; svg.setAttribute("viewBox","0 0 600 540");
  const make=(name,attrs)=>{ const el=document.createElementNS(NS,name); Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v)); svg.appendChild(el); return el; };
  const visibleSize=visibleLevels.reduce((a,b)=>a*b,1);
  for(let token=0;token<visibleSize;token++) {
    const coord=coordForToken(token,visibleLevels);
    for(let dim=0;dim<3;dim++) { const next=coord.slice(); next[dim]+=1; if(next[dim]<visibleLevels[dim]) { const [x1,y1]=project(coord,visibleLevels),[x2,y2]=project(next,visibleLevels); make("line",{x1,y1,x2,y2,stroke:"rgba(100,110,125,.35)","stroke-width":1.4}); } }
  }
  const bounds=visibleLevels.map(level=>level-1),corners=[0,1,2,3,4,5,6,7].map(bit=>[bounds[0]*(bit&1),bounds[1]*((bit>>1)&1),bounds[2]*((bit>>2)&1)]);
  [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]].forEach(([a,b])=>{ const [x1,y1]=project(corners[a],visibleLevels),[x2,y2]=project(corners[b],visibleLevels); make("line",{x1,y1,x2,y2,stroke:"rgba(45,55,70,.72)","stroke-width":2.2}); });
  const points=[];
  for(let token=0;token<maxToken;token++) { const coord=coordForToken(token,levels),p=project(coord.slice(0,3),visibleLevels),hidden=coord.slice(3).reduce((value,item,index)=>value+item*levels.slice(3,index+3).reduce((a,b)=>a*b,1),0); if(hiddenCount>1) { const angle=2*Math.PI*hidden/hiddenCount-Math.PI/2,radius=Math.min(12,4+hiddenCount); p[0]+=Math.cos(angle)*radius; p[1]+=Math.sin(angle)*radius; } points.push({token,coord,p}); }
  points.sort((a,b)=>a.p[2]-b.p[2]);
  points.forEach(({token,coord,p})=>{ const active=tested.has(token),isOriginal=token===original,isSelected=token===selected,r=isSelected?10:(isOriginal?8:(active?6.5:3.2)); const point=make("circle",{cx:p[0],cy:p[1],r,fill:active?tokenColor(token,isSelected?1:.9,52):"#d7dde8",stroke:isSelected?"#111827":(isOriginal?"#111827":(active?tokenColor(token,1,31):"#8d99aa")),"stroke-width":isSelected?3.2:(isOriginal?2.5:.8),style:active?"cursor:pointer":"cursor:default",opacity:selected!==null&&!isSelected?(active ? 0.34 : 0.18):1});
    if(active) point.addEventListener("click",()=>{ const uid=String(record.uid),current=emphasizedCodeByRecord.get(uid); if(current!==undefined&&Number(current)===token) emphasizedCodeByRecord.delete(uid); else emphasizedCodeByRecord.set(uid,token); renderMiniCube(record,svg); const canvas=document.querySelector(`canvas[data-record-uid="${uid}"]`); if(canvas) drawRecordCanvas(canvas,record); });
    const title=document.createElementNS(NS,"title"); title.textContent=`code #${token} [${coord.join(", ")}]${isOriginal?" · assigned":""}${active?" · evaluated":" · not evaluated"}`; point.appendChild(title);
  });
  const caption=document.querySelector(`[data-mini-caption="${record.uid}"]`); if(caption) caption.innerHTML=selected===null?`assigned <b>#${original}</b><br>${tested.size}/${maxToken} codes visible`:`emphasized <b>#${selected}</b><br>click again for all`;
}
function renderMiniCubes() {
  document.querySelectorAll("svg[data-mini-record-uid]").forEach(svg=>{ const record=recordByUid.get(String(svg.dataset.miniRecordUid)); if(record) renderMiniCube(record,svg); });
}
function renderCodebooks() {
  const root=document.getElementById("codebooks"),members=selectedMembers(); root.innerHTML="";
  for(const space of skillSpaces) {
    const card=document.createElement("section"); card.className="codebook-card";
    const title=document.createElement("div"); title.className="codebook-title"; title.textContent=`${space.label} · [${space.levels.join("×")}]`; title.title=space.label;
    const svg=document.createElementNS("http://www.w3.org/2000/svg","svg"); svg.setAttribute("class","cube"); svg.setAttribute("viewBox","0 0 600 540");
    card.append(title,svg); root.appendChild(card); renderCube(space,svg,members);
  }
}
function rolloutStats(record,subset=null) {
  const rollouts=subset||(record.rollouts||[]),steps=rollouts.map(item=>Number(item.steps||0)).sort((a,b)=>a-b);
  const mean=steps.length?steps.reduce((a,b)=>a+b,0)/steps.length:0;
  const endpoints=rollouts.map(item=>{ const valid=(item.trajectory||[]).filter(Boolean); return valid.length?valid[valid.length-1]:null; }).filter(Boolean);
  let spread=0;
  if(endpoints.length) { const mx=endpoints.reduce((sum,p)=>sum+p[0],0)/endpoints.length,my=endpoints.reduce((sum,p)=>sum+p[1],0)/endpoints.length; spread=Math.sqrt(endpoints.reduce((sum,p)=>sum+(p[0]-mx)**2+(p[1]-my)**2,0)/endpoints.length); }
  const predicted=rollouts.filter(item=>item.stop_reason==="predicted_end").length;
  return {count:rollouts.length,mean,minimum:steps[0]||0,maximum:steps[steps.length-1]||0,spread,predicted};
}
function panelHtml(record) {
  const codes=evaluatedTokens(record),original=Number(record.token);
  const assignedRollouts=(record.rollouts||[]).filter(item=>rolloutToken(record,item)===original),stat=rolloutStats(record,assignedRollouts);
  return `<article class="trajectory-panel" data-record-panel="${escapeHtml(record.uid)}"><div class="panel-title"><span title="${escapeHtml(record.model_label)}">${escapeHtml(record.model_label)}</span><span>assigned #${record.token} · ${codes.length} tested codes · ${(record.rollouts||[]).length} rollouts</span></div><div class="canvas-wrap"><canvas data-record-uid="${escapeHtml(record.uid)}"></canvas></div><div class="code-controls"><div class="mini-codebook"><svg class="mini-cube" data-mini-record-uid="${escapeHtml(record.uid)}"></svg><div class="mini-caption" data-mini-caption="${escapeHtml(record.uid)}"></div></div><span class="hint">Click an evaluated point in this mini codebook to emphasize only that code's trajectories. Click it again to restore all.</span></div><div class="panel-stats"><span class="badge">assigned steps μ ${stat.mean.toFixed(1)} · ${stat.minimum}–${stat.maximum}</span><span class="badge">assigned spread ${stat.spread.toFixed(1)} px</span><span class="badge">assigned terminator ${stat.predicted}/${stat.count}</span></div></article>`;
}
function drawRecordCanvas(canvas,record) {
  const image=new Image();
  image.onload=()=>{
    canvas.width=image.naturalWidth; canvas.height=image.naturalHeight;
    const ctx=canvas.getContext("2d"); ctx.drawImage(image,0,0);
    const emphasized=emphasizedCodeByRecord.has(String(record.uid))?Number(emphasizedCodeByRecord.get(String(record.uid))):null;
    const rollouts=(record.rollouts||[]).slice().sort((a,b)=>{ if(emphasized===null) return 0; return Number(rolloutToken(record,a)===emphasized)-Number(rolloutToken(record,b)===emphasized); });
    rollouts.forEach(rollout=>{
      const token=rolloutToken(record,rollout),isEmphasized=emphasized===null||token===emphasized;
      const points=rollout.trajectory||[]; ctx.beginPath(); let active=false;
      for(const point of points) { if(!point) { active=false; continue; } if(!active) { ctx.moveTo(point[0],point[1]); active=true; } else ctx.lineTo(point[0],point[1]); }
      const alpha=emphasized===null?trajectoryOpacity:(isEmphasized?Math.max(.82,trajectoryOpacity):Math.max(.025,trajectoryOpacity*.12));
      ctx.strokeStyle=tokenColor(token,alpha,54); ctx.lineWidth=emphasized!==null&&isEmphasized?2.7:1.2; ctx.stroke();
      const valid=points.filter(Boolean); if(valid.length) { const end=valid[valid.length-1]; ctx.beginPath(); ctx.arc(end[0],end[1],emphasized!==null&&isEmphasized?2.8:1.55,0,2*Math.PI); ctx.fillStyle=tokenColor(token,Math.min(.96,alpha+.28),50); ctx.fill(); }
    });
    const first=rollouts.map(item=>(item.trajectory||[]).find(Boolean)).find(Boolean);
    if(first) { ctx.beginPath(); ctx.arc(first[0],first[1],3.3,0,2*Math.PI); ctx.fillStyle="#39d353"; ctx.fill(); ctx.strokeStyle="#102a17"; ctx.lineWidth=1; ctx.stroke(); }
  };
  image.src=record.start_image_path;
}
function drawVisibleCanvases() {
  document.querySelectorAll("canvas[data-record-uid]").forEach(canvas=>{ const record=recordByUid.get(String(canvas.dataset.recordUid)); if(record) drawRecordCanvas(canvas,record); });
}
function selectSkillCode(modelIndex,token) {
  const space=spaceByModel.get(Number(modelIndex)),skill=space&&(space.skills||[]).find(item=>Number(item.token)===Number(token)); if(!skill) return;
  selectedSkill={model_index:Number(modelIndex),token:Number(token)}; const members=new Set(memberIds(skill)); renderCodebooks();
  const links=skillSpaces.map(other=>{ const hits=(other.skills||[]).map(item=>({token:Number(item.token),count:memberIds(item).filter(uid=>members.has(uid)).length})).filter(item=>item.count>0).sort((a,b)=>b.count-a.count||a.token-b.token); return `${escapeHtml(other.label)}: ${hits.map(hit=>`#${hit.token} (${hit.count})`).join(", ")||"none"}`; }).join("<br>");
  document.getElementById("selected-info").innerHTML=`${escapeHtml(space.label)} · code #${skill.token} [${skill.coord.join(", ")}] <span class="count">${members.size} skills</span><div class="muted" style="margin-top:5px">${links}</div>`;
  const selected=allRecords.filter(record=>members.has(String(record.occurrence_uid))).sort((a,b)=>a.task_id-b.task_id||a.episode_id-b.episode_id||a.frame_start-b.frame_start||a.model_index-b.model_index);
  const grouped=new Map(); selected.forEach(record=>{ const key=String(record.occurrence_uid); if(!grouped.has(key)) grouped.set(key,[]); grouped.get(key).push(record); });
  const content=document.getElementById("content");
  if(!grouped.size) { content.innerHTML='<div class="empty">No matching exact skill occurrence.</div>'; return; }
  content.innerHTML=[...grouped.values()].map(records=>{ const first=records[0]; return `<section class="sample"><div class="sample-title">Task ${first.task_id} · environment episode ${first.episode_id} · skill ${first.skill_index} · frames ${first.frame_start}–${first.frame_end}<div class="sample-subtitle">${escapeHtml(first.task_description)}</div></div><div class="model-grid">${records.map(panelHtml).join("")}</div></section>`; }).join("");
  renderMiniCubes();
  drawVisibleCanvases();
}

document.getElementById("summary").textContent=`${(DATA.models||[]).map(model=>model.label).join(", ")} · ${DATA.target_task} tasks ${(DATA.task_ids||[]).join(", ")} · ${DATA.env_count} exact environments · ${DATA.noise_rollouts_per_env} paired noise rollouts/code · code probe ${DATA.code_probe_mode||"off"} · ${DATA.occurrence_count} GT skills`;
document.getElementById("opacity").addEventListener("input",event=>{ trajectoryOpacity=Number(event.target.value); document.getElementById("opacity-value").textContent=trajectoryOpacity.toFixed(2); drawVisibleCanvases(); });
const firstSpace=skillSpaces[0],initialSkill=firstSpace&&(firstSpace.skills||[]).slice().sort((a,b)=>memberIds(b).length-memberIds(a).length||a.token-b.token)[0];
if(initialSkill) selectSkillCode(firstSpace.model_index,initialSkill.token); else { renderCodebooks(); document.getElementById("content").innerHTML='<div class="empty">No evaluated skill occurrence.</div>'; }
</script>
</body>
</html>
""".replace("__DATA__", data_json)
    temporary = html_path.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(html_path)
    return html_path
