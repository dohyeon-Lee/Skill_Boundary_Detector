"""Interactive report for multi-policy rollouts stopped by a configurable MAIN."""

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
  <title>Stage-1 multi-policy skill evaluation</title>
  <style>
    :root { --ink:#17202a; --muted:#667085; --line:#d4dbe6; --blue:#2878b5; --red:#d62728; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,Arial,sans-serif; background:#f4f6f9; color:var(--ink); }
    header { position:sticky; top:0; z-index:5; padding:14px 20px; background:#fff; border-bottom:1px solid var(--line); }
    h1 { margin:0 0 5px; font-size:20px; }
    .subtitle,.muted { color:var(--muted); font-size:12px; }
    .success-heading { margin-top:10px; color:#475467; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:.04em; }
    .success-summary { display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:8px; margin-top:5px; }
    .success-card { padding:7px 10px; border:1px solid var(--line); border-radius:7px; background:#f8fafc; }
    .success-model { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#344054; font-size:11px; font-weight:700; }
    .success-metrics { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:5px; }
    .success-metric { position:relative; min-width:0; padding:5px 6px; border:1px solid #e4e7ec; border-radius:6px; background:#fff; }
    .success-metric.rank-1 { border-color:#d5a72c; background:#fff3bf; box-shadow:inset 0 0 0 1px rgba(213,167,44,.12); }
    .success-metric.rank-2 { border-color:#98a2b3; background:#e9edf2; box-shadow:inset 0 0 0 1px rgba(152,162,179,.12); }
    .success-kind { color:#667085; font-size:9px; font-weight:800; text-transform:uppercase; letter-spacing:.03em; }
    .success-value { margin-top:1px; color:#273142; font-size:15px; font-weight:800; white-space:nowrap; }
    .success-rate { margin-left:5px; color:#667085; font-size:11px; font-weight:600; }
    .success-rank { float:right; margin-left:4px; padding:1px 4px; border-radius:8px; color:#694f00; background:rgba(255,255,255,.72); font-size:8px; font-weight:900; letter-spacing:0; }
    .rank-2 .success-rank { color:#475467; }
    .review-toolbar { display:flex; align-items:center; gap:7px; flex-wrap:wrap; margin-top:8px; color:#475467; font-size:11px; }
    .review-help { margin-right:auto; }
    .review-button { appearance:none; padding:4px 8px; border:1px solid #98a2b3; border-radius:5px; background:#fff; color:#344054; font:inherit; font-weight:700; cursor:pointer; }
    .review-button:hover { background:#f2f4f7; }
    .review-button.danger { border-color:#f0a09a; color:#9c2f28; }
    .review-file { display:none; }
    .review-count { padding:2px 6px; border-radius:9px; background:#e8eef7; color:#35516f; font-weight:800; }
    .review-status { min-width:170px; color:#356341; }
    .review-status.error { color:#a12d27; }
    .layout { display:grid; grid-template-columns:minmax(440px,600px) 1fr; gap:16px; padding:16px; align-items:start; }
    .sidebar { position:sticky; top:170px; background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px; }
    .codebooks { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:10px; }
    .codebook-card { min-width:0; padding:8px; border:1px solid #e0e6ef; border-radius:8px; background:#fbfcfe; }
    .codebook-title { overflow:hidden; margin:0 0 2px; color:#344054; font-size:11px; font-weight:800; text-overflow:ellipsis; white-space:nowrap; }
    .cube { width:100%; height:auto; display:block; }
    .legend { display:flex; gap:14px; align-items:center; flex-wrap:wrap; font-size:12px; color:var(--muted); }
    .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; }
    .selected-info { margin-top:9px; padding:9px 10px; background:#f6f9fd; border-radius:7px; font-weight:700; }
    .content { min-width:0; }
    .token-success { margin-bottom:16px; padding:10px 12px 12px; background:#fff; border:1px solid var(--line); border-radius:10px; }
    .token-success-title { color:#344054; font-size:12px; font-weight:800; }
    .empty { padding:32px; background:#fff; border:1px solid var(--line); border-radius:10px; text-align:center; color:var(--muted); }
    .occurrence { margin-bottom:16px; background:#fff; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
    .occ-title { padding:10px 12px; background:#f8fafc; border-bottom:1px solid var(--line); font-size:13px; font-weight:700; }
    .videos { padding:10px; overflow-x:auto; }
    .comparison-card { min-width:950px; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#fbfcfe; }
    .comparison-labels { display:grid; grid-template-columns:128fr repeat(5,256fr); }
    .comparison-label-spacer { border-bottom:3px solid transparent; background:#101318; }
    .comparison-label { padding:7px 8px; text-align:center; font-size:12px; font-weight:700; border-bottom:3px solid var(--branch,#98a2b3); }
    .comparison-label.manual-success { background:#dff5e4; box-shadow:inset 0 0 0 2px #39a85a; }
    .comparison-label.manual-failure { background:#fde7e5; box-shadow:inset 0 0 0 2px #d94b43; }
    .manual-label { margin-left:5px; font-size:9px; font-weight:900; }
    video { display:block; width:100%; background:#101318; object-fit:contain; }
    .final-review-wrap { position:relative; display:block; width:100%; border-top:2px solid var(--line); background:#101318; overflow:hidden; }
    .final-frames { position:absolute; inset:0; display:block; width:100%; height:100%; object-fit:contain; cursor:crosshair; }
    .review-overlay { position:absolute; top:0; display:grid; place-items:end center; padding:5px; pointer-events:none; border:3px solid; }
    .review-overlay.manual-success { border-color:#32a852; background:rgba(72,205,105,.28); }
    .review-overlay.manual-failure { border-color:#d83d35; background:rgba(220,57,48,.25); }
    .review-overlay-tag { padding:3px 6px; border-radius:4px; color:#fff; background:rgba(16,24,40,.82); font-size:10px; font-weight:900; letter-spacing:.02em; }
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
  <h1>Stage-1 multi-policy skill evaluation</h1>
  <div class="subtitle" id="summary"></div>
  <div class="success-heading">Skill success (green / evaluated) · ID: exact + different noise · OOD: early + late · GT excluded</div>
  <div class="success-summary" id="success-summary"></div>
  <div class="review-toolbar">
    <span class="review-help">Human review: click a policy panel in the final image to toggle success/failure.</span>
    <span class="review-count" id="review-count">0 overrides</span>
    <button class="review-button" id="review-export" type="button">Export corrections</button>
    <label class="review-button" for="review-import">Import corrections</label>
    <input class="review-file" id="review-import" type="file" accept="application/json,.json" />
    <button class="review-button danger" id="review-clear" type="button">Clear corrections</button>
    <span class="review-status" id="review-status"></span>
  </div>
</header>
<main class="layout">
  <aside class="sidebar">
    <div class="codebooks" id="codebooks"></div>
    <div class="legend">
      <span><i class="dot" style="background:#d62728"></i>clicked code</span>
      <span><i class="dot" style="background:#f39c4a"></i>linked code(s)</span>
      <span><i class="dot" style="background:#2878b5"></i>used in selected episodes</span>
      <span><i class="dot" style="background:#d7dde8"></i>unused</span>
    </div>
    <div class="selected-info" id="selected-info"></div>
  </aside>
  <section class="content" id="content"></section>
</main>
<script>
const DATA = __DATA__;
const skillSpaces=(DATA.skill_spaces&&DATA.skill_spaces.length)?DATA.skill_spaces:[{
  model_index:0,label:(DATA.models&&DATA.models[0]&&DATA.models[0].label)||"skill space",
  levels:DATA.levels,skills:DATA.skills||[],
}];
const spaceByModel=new Map(skillSpaces.map(space=>[Number(space.model_index),space]));
const allOccurrences=(DATA.occurrences&&DATA.occurrences.length)
  ? DATA.occurrences
  : DATA.skills.flatMap(skill=>skill.occurrences||[]);
const occurrenceByUid = new Map(allOccurrences.map(occ=>[String(occ.uid),occ]));
const REVIEW_SCHEMA = "stage1_skill_eval_human_review_v1";
const STORAGE_KEY = `${REVIEW_SCHEMA}:${DATA.review_id||location.pathname}`;
let storageAvailable = true;
let selectedSkill = null;
let reviewServerAvailable = false;
let serverSaveChain = Promise.resolve();

function blankReview() {
  return {schema:REVIEW_SCHEMA,report_id:String(DATA.review_id||""),updated_at:null,corrections:{}};
}

function loadReview() {
  try {
    const raw=localStorage.getItem(STORAGE_KEY);
    if(!raw) return blankReview();
    const parsed=JSON.parse(raw);
    if(parsed.schema!==REVIEW_SCHEMA || typeof parsed.corrections!=="object" || !parsed.corrections) return blankReview();
    return {schema:REVIEW_SCHEMA,report_id:String(DATA.review_id||""),updated_at:parsed.updated_at||null,corrections:parsed.corrections};
  } catch(error) {
    storageAvailable=false;
    return blankReview();
  }
}

const REVIEW = loadReview();

function correctionKey(uid,branchName) {
  return `${uid}::${branchName}`;
}

function correctionFor(occ,branch) {
  const value=REVIEW.corrections[correctionKey(occ.uid,branch.name)];
  return value && typeof value.success==="boolean" ? value : null;
}

function effectiveSuccess(occ,branch) {
  const correction=correctionFor(occ,branch);
  return correction ? correction.success : Boolean(branch.green_tint);
}

function sanitizeCorrections(raw) {
  const clean={};
  for(const [key,value] of Object.entries(raw||{})) {
    const separator=key.lastIndexOf("::");
    if(separator<0) continue;
    const uid=key.slice(0,separator), branchName=key.slice(separator+2);
    const occ=occurrenceByUid.get(uid);
    const branch=occ && (occ.branches||[]).find(item=>item.name===branchName);
    const success=typeof value==="boolean"?value:(value&&value.success);
    if(!occ || !branch || branch.name==="gt" || branch.unavailable_reason!=null || typeof success!=="boolean") continue;
    if(success===Boolean(branch.green_tint)) continue;
    clean[key]={success,updated_at:String((value&&value.updated_at)||new Date().toISOString())};
  }
  return clean;
}

REVIEW.corrections=sanitizeCorrections(REVIEW.corrections);

function setReviewStatus(message,isError=false) {
  const status=document.getElementById("review-status");
  status.textContent=message;
  status.classList.toggle("error",Boolean(isError));
}

function reviewPayload() {
  return {
    schema:REVIEW_SCHEMA,
    report_id:String(DATA.review_id||""),
    updated_at:REVIEW.updated_at||new Date().toISOString(),
    corrections:REVIEW.corrections,
  };
}

function saveLocalReview() {
  try {
    localStorage.setItem(STORAGE_KEY,JSON.stringify(REVIEW));
    storageAvailable=true;
  } catch(error) {
    storageAvailable=false;
  }
  updateReviewCount();
}

function queueServerSave(message="Saved to server") {
  const payload=JSON.stringify(reviewPayload());
  serverSaveChain=serverSaveChain.catch(()=>{}).then(async()=>{
    const response=await fetch("./api/corrections",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:payload,
    });
    const result=await response.json().catch(()=>({}));
    if(!response.ok) throw new Error(result.error||`Review server returned HTTP ${response.status}.`);
    reviewServerAvailable=true;
    if((Date.parse(result.updated_at)||0)>=(Date.parse(REVIEW.updated_at)||0)) {
      REVIEW.updated_at=result.updated_at||REVIEW.updated_at;
    }
    saveLocalReview();
    setReviewStatus(message);
  }).catch(error=>{
    reviewServerAvailable=false;
    setReviewStatus(`Server save failed; kept in browser: ${error instanceof Error?error.message:String(error)}`,true);
  });
}

function saveReview(message="Saved") {
  saveLocalReview();
  if(reviewServerAvailable) {
    setReviewStatus("Saving to server…");
    queueServerSave(`${message} · server autosaved`);
  } else {
    setReviewStatus(
      storageAvailable?`${message} · browser autosaved`:`${message} · browser storage unavailable; export to keep it.`,
      !storageAvailable,
    );
  }
}

function updateReviewCount() {
  const count=Object.keys(REVIEW.corrections).length;
  document.getElementById("review-count").textContent=`${count} override${count===1?"":"s"}`;
}

function successStats(occurrences) {
  const groups={id:new Set(["policy","policy_alt_noise"]),ood:new Set(["policy_early","policy_late"])};
  const stats=(DATA.models||[]).map((model,modelIndex)=>({
    model_index:modelIndex,
    label:String(model.label||`model_${modelIndex}`),
    id:{success_count:0,total_count:0,success_rate:0,rank:null},
    ood:{success_count:0,total_count:0,success_rate:0,rank:null},
  }));
  for(const occ of occurrences||[]) {
    const stat=stats[Number(occ.model_index||0)];
    if(!stat) continue;
    for(const branch of occ.branches||[]) {
      const group=groups.id.has(branch.name)?"id":(groups.ood.has(branch.name)?"ood":null);
      if(!group || branch.unavailable_reason!=null) continue;
      stat[group].total_count+=1;
      stat[group].success_count+=effectiveSuccess(occ,branch)?1:0;
    }
  }
  for(const group of ["id","ood"]) {
    for(const stat of stats) {
      const metric=stat[group];
      metric.success_rate=metric.total_count?metric.success_count/metric.total_count:0;
    }
    const rates=[...new Set(stats.filter(stat=>stat[group].total_count>0).map(stat=>stat[group].success_rate))].sort((a,b)=>b-a).slice(0,2);
    for(const stat of stats) {
      if(stat[group].total_count>0) {
        const rank=rates.indexOf(stat[group].success_rate);
        stat[group].rank=rank>=0?rank+1:null;
      }
    }
  }
  return stats;
}

function reviewTimestamp(review) {
  const documentTime=Date.parse(review&&review.updated_at)||0;
  const correctionTimes=Object.values((review&&review.corrections)||{}).map(
    value=>Date.parse(value&&value.updated_at)||0,
  );
  return Math.max(documentTime,...correctionTimes,0);
}

async function connectReviewServer() {
  try {
    const response=await fetch("./api/corrections",{cache:"no-store"});
    if(!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload=await response.json();
    if(payload.schema!==REVIEW_SCHEMA || String(payload.report_id||"")!==String(DATA.review_id||"")) {
      throw new Error("Review server report ID does not match this HTML.");
    }
    const serverReview={
      updated_at:payload.updated_at||null,
      corrections:sanitizeCorrections(payload.corrections),
    };
    const localIsNewer=reviewTimestamp(REVIEW)>reviewTimestamp(serverReview);
    if(!localIsNewer) {
      REVIEW.corrections=serverReview.corrections;
      REVIEW.updated_at=serverReview.updated_at;
    }
    reviewServerAvailable=true;
    saveLocalReview();
    renderReviewedResults();
    if(localIsNewer) {
      queueServerSave("Browser corrections migrated to server");
    } else {
      setReviewStatus("Server autosave connected");
    }
  } catch(error) {
    reviewServerAvailable=false;
    setReviewStatus(
      storageAvailable?"Browser autosave only · start review_server.py for server saving":"No persistent storage available · start review_server.py or export corrections",
      !storageAvailable,
    );
  }
}

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

function memberIds(skill) {
  if(Array.isArray(skill.member_ids)) return skill.member_ids.map(String);
  return [...new Set((skill.occurrences||[]).map(occ=>String(occ.occurrence_uid||occ.uid)))];
}

function selectedMembers() {
  if(!selectedSkill) return new Set();
  const space=spaceByModel.get(Number(selectedSkill.model_index));
  const skill=space&&(space.skills||[]).find(item=>Number(item.token)===Number(selectedSkill.token));
  return new Set(skill?memberIds(skill):[]);
}

function renderCube(space,svg,members) {
  const levels=space.levels;
  const byToken=new Map((space.skills||[]).map(skill=>[Number(skill.token),skill]));
  const maxToken=levels.reduce((a,b)=>a*b,1), NS="http://www.w3.org/2000/svg";
  svg.innerHTML="";
  const make=(name,attrs)=>{
    const el=document.createElementNS(NS,name);
    Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));
    svg.appendChild(el); return el;
  };
  for(let t=0;t<maxToken;t++) {
    const c=coordForToken(t,levels);
    for(let d=0;d<Math.min(3,levels.length);d++) {
      const n=c.slice(); n[d]+=1;
      if(n[d]<levels[d]) {
        const [x1,y1]=project(c,levels),[x2,y2]=project(n,levels);
        make("line",{x1,y1,x2,y2,stroke:"rgba(100,100,100,.48)","stroke-width":1.5});
      }
    }
  }
  const BL=levels.map(l=>l-1);
  const corners=[0,1,2,3,4,5,6,7].map(b=>[BL[0]*(b&1),BL[1]*((b>>1)&1),(BL[2]||0)*((b>>2)&1)]);
  [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]].forEach(([a,b])=>{
    const [x1,y1]=project(corners[a],levels),[x2,y2]=project(corners[b],levels);
    make("line",{x1,y1,x2,y2,stroke:"rgba(50,50,50,.75)","stroke-width":2.2});
  });
  const pts=[];
  for(let t=0;t<maxToken;t++) {
    const p=project(coordForToken(t,levels),levels); pts.push({t,p,used:byToken.has(t)});
  }
  pts.sort((a,b)=>a.p[2]-b.p[2]);
  pts.forEach(({t,p,used})=>{
    const skill=byToken.get(t);
    const overlap=skill?memberIds(skill).filter(uid=>members.has(uid)).length:0;
    const clicked=Boolean(selectedSkill)
      && Number(selectedSkill.model_index)===Number(space.model_index)
      && Number(selectedSkill.token)===t;
    const linked=!clicked&&overlap>0;
    const point=make("circle",{
      cx:p[0],cy:p[1],r:clicked?9:(linked?8:(used?6:3.8)),
      fill:clicked?"#d62728":(linked?"#f39c4a":(used?"#2878b5":"#d7dde8")),
      stroke:(clicked||linked)?"#8b1a1a":"#26384d",
      "stroke-width":(clicked||linked)?2.2:.9,
      style:used?"cursor:pointer":"cursor:default"
    });
    if(used) point.addEventListener("click",()=>selectSkillCode(space.model_index,t));
    const title=document.createElementNS(NS,"title");
    title.textContent=used
      ? `code #${t}: ${memberIds(skill).length} GT skills${overlap?` · ${overlap} linked`:""}`
      : `code #${t}: unused`;
    point.appendChild(title);
  });
}

function renderCodebooks() {
  const root=document.getElementById("codebooks");
  root.innerHTML="";
  const members=selectedMembers();
  for(const space of skillSpaces) {
    const card=document.createElement("section");
    card.className="codebook-card";
    const title=document.createElement("div");
    title.className="codebook-title";
    title.textContent=`${space.label} · FSQ[${space.levels.join("×")}]`;
    title.title=space.label;
    const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");
    svg.setAttribute("class","cube");
    svg.setAttribute("viewBox","0 0 600 540");
    svg.setAttribute("aria-label",`${space.label} FSQ skill cube`);
    card.append(title,svg); root.appendChild(card);
    renderCube(space,svg,members);
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
}

function comparisonCard(occ) {
  const labels=`<div class="comparison-label-spacer"></div>`+occ.branches.map(branch=>{
    const correction=correctionFor(occ,branch);
    const manualClass=correction?(correction.success?" manual-success":" manual-failure"):"";
    const manualLabel=correction?`<span class="manual-label">${correction.success?"MANUAL SUCCESS":"MANUAL FAIL"}</span>`:"";
    return `<div class="comparison-label${manualClass}" style="--branch:${escapeHtml(branch.color||"#98a2b3")}">${escapeHtml(branch.label)}${manualLabel}</div>`;
  }).join("");
  if(!occ.comparison_path) {
    return `<div class="unavailable">Combined video was not generated for this legacy result.</div>`;
  }
  // The combined video has one 128px shared label gutter, five 256px branch
  // panels, and one signal row per display model plus the configured MAIN.
  const signalRows=(DATA.terminator_models||[]).filter(
    model=>model.model_index==null||Number(model.model_index)===Number(occ.model_index||0)
  ).length+1;
  const aspect=`${(128+5*256)/(256+36*signalRows)} / 1`;
  const totalWidth=128+5*256;
  const cameraPercent=100*256/(256+36*signalRows);
  const overlays=occ.branches.map((branch,index)=>{
    const correction=correctionFor(occ,branch);
    if(!correction) return "";
    const left=100*(128+index*256)/totalWidth;
    const width=100*256/totalWidth;
    const stateClass=correction.success?"manual-success":"manual-failure";
    const stateLabel=correction.success?"MANUAL SUCCESS":"MANUAL FAIL";
    return `<div class="review-overlay ${stateClass}" style="left:${left}%;width:${width}%;height:${cameraPercent}%"><span class="review-overlay-tag">${stateLabel}</span></div>`;
  }).join("");
  const finalFrames=occ.comparison_final_path
    ? `<div class="final-review-wrap" style="aspect-ratio:${aspect}"><img class="final-frames" data-occurrence-uid="${escapeHtml(occ.uid)}" loading="lazy" src="${escapeHtml(occ.comparison_final_path)}" alt="Final frame comparison; click a policy panel to review" />${overlays}</div>`
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

function successMetric(label,value) {
  const percent=(100*Number(value.success_rate||0)).toFixed(1);
  const rank=Number(value.rank||0);
  const rankClass=(rank===1||rank===2)?` rank-${rank}`:"";
  const rankBadge=(rank===1||rank===2)?`<span class="success-rank">#${rank}</span>`:"";
  return `<div class="success-metric${rankClass}"><div class="success-kind">${label} success rate${rankBadge}</div><div class="success-value">${value.success_count} / ${value.total_count}<span class="success-rate">${percent}%</span></div></div>`;
}

function successCards(stats) {
  return (stats||[]).map(stat=>`<div class="success-card" title="${escapeHtml(stat.label)}"><div class="success-model">${escapeHtml(stat.label)}</div><div class="success-metrics">${successMetric("ID",stat.id)}${successMetric("OOD",stat.ood)}</div></div>`).join("");
}

function selectSkillCode(modelIndex,token) {
  const space=spaceByModel.get(Number(modelIndex));
  const skill=space&&(space.skills||[]).find(item=>Number(item.token)===Number(token));
  if(!skill) return;
  selectedSkill={model_index:Number(modelIndex),token:Number(token)};
  const members=new Set(memberIds(skill));
  renderCodebooks();
  const links=skillSpaces.map(other=>{
    const hits=(other.skills||[]).map(item=>({
      token:Number(item.token),
      count:memberIds(item).filter(uid=>members.has(uid)).length,
    })).filter(item=>item.count>0).sort((a,b)=>b.count-a.count||a.token-b.token);
    return `${escapeHtml(other.label)}: ${hits.map(hit=>`#${hit.token} (${hit.count})`).join(", ")||"none"}`;
  }).join("<br>");
  const selectedOccurrences=allOccurrences.filter(
    occ=>members.has(String(occ.occurrence_uid||occ.uid))
  ).sort((a,b)=>a.task_id-b.task_id||a.episode_id-b.episode_id||a.frame_start-b.frame_start||a.model_index-b.model_index);
  document.getElementById("selected-info").innerHTML=
    `${escapeHtml(space.label)} · code #${skill.token} &nbsp; [${skill.coord.join(", ")}] <span class="count">${members.size} GT skills</span><div class="muted" style="margin-top:6px;line-height:1.5">${links}</div>`;
  const content=document.getElementById("content");
  if(!selectedOccurrences.length) { content.innerHTML='<div class="empty">No occurrence.</div>'; return; }
  const tokenSuccess=`<section class="token-success"><div class="token-success-title">Linked GT skill set · ID / OOD success</div><div class="success-summary">${successCards(successStats(selectedOccurrences))}</div></section>`;
  content.innerHTML=tokenSuccess+selectedOccurrences.map(occ=>`<section class="occurrence">
    <div class="occ-title">${escapeHtml(occ.model_label||"policy")} · code #${occ.token} · ${escapeHtml(occ.architecture_label||"")} · Task ${occ.task_id}: ${escapeHtml(occ.task_description||"")}
    </div>
    <div class="videos">${comparisonCard(occ)}</div>
  </section>`).join("");
}

function renderReviewedResults() {
  document.getElementById("success-summary").innerHTML=successCards(successStats(allOccurrences));
  updateReviewCount();
  if(selectedSkill) selectSkillCode(selectedSkill.model_index,selectedSkill.token);
}

function toggleCorrection(occ,branch) {
  const key=correctionKey(occ.uid,branch.name);
  const next=!effectiveSuccess(occ,branch);
  const changedAt=new Date().toISOString();
  if(next===Boolean(branch.green_tint)) {
    delete REVIEW.corrections[key];
  } else {
    REVIEW.corrections[key]={success:next,updated_at:changedAt};
  }
  REVIEW.updated_at=changedAt;
  saveReview(`${branch.label}: ${next?"manual success":"manual failure"}`);
  renderReviewedResults();
}

function handleFinalImageClick(event) {
  const image=event.target.closest(".final-frames");
  if(!image) return;
  const occ=occurrenceByUid.get(String(image.dataset.occurrenceUid));
  if(!occ) return;
  const rect=image.getBoundingClientRect();
  const sourceX=(event.clientX-rect.left)/rect.width*(128+5*256);
  const signalRows=(DATA.terminator_models||[]).filter(
    model=>model.model_index==null||Number(model.model_index)===Number(occ.model_index||0)
  ).length+1;
  const sourceY=(event.clientY-rect.top)/rect.height*(256+36*signalRows);
  if(sourceX<128 || sourceY>=256) {
    setReviewStatus("Click a policy camera panel, not the label/signal area.",true);
    return;
  }
  const branchIndex=Math.floor((sourceX-128)/256);
  const branch=(occ.branches||[])[branchIndex];
  if(!branch) return;
  if(branch.name==="gt") {
    setReviewStatus("GT is excluded from human-review success rates.",true);
    return;
  }
  if(branch.unavailable_reason!=null) {
    setReviewStatus("This branch was unavailable and cannot be reviewed.",true);
    return;
  }
  toggleCorrection(occ,branch);
}

function exportCorrections() {
  const payload={...reviewPayload(),
    exported_at:new Date().toISOString(),
  };
  const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob), link=document.createElement("a");
  link.href=url;
  link.download=`stage1_skill_eval_corrections_${DATA.review_id||"report"}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),0);
  setReviewStatus(`Exported ${Object.keys(REVIEW.corrections).length} overrides`);
}

async function importCorrections(file) {
  try {
    const payload=JSON.parse(await file.text());
    if(payload.schema!==REVIEW_SCHEMA) throw new Error("Unsupported corrections schema.");
    if(String(payload.report_id||"")!==String(DATA.review_id||"")) throw new Error("Corrections belong to a different evaluation report.");
    REVIEW.corrections=sanitizeCorrections(payload.corrections);
    REVIEW.updated_at=new Date().toISOString();
    saveReview(`Imported ${Object.keys(REVIEW.corrections).length} overrides`);
    renderReviewedResults();
  } catch(error) {
    setReviewStatus(error instanceof Error?error.message:String(error),true);
  }
}

const displayLabels=(DATA.terminator_models||[]).map(model=>{
  const policy=(DATA.models||[])[Number(model.model_index||0)];
  return `${policy?policy.label+"→":""}${model.label}:${model.variant}`;
}).join(", ");
const policyLabels=(DATA.models||[]).map(model=>model.label).join(", ");
const mainTerm=DATA.main_terminator||{};
const mainLabels=(DATA.main_terminators&&DATA.main_terminators.length)
  ? DATA.main_terminators.map(model=>{
      const policy=(DATA.models||[])[Number(model.model_index||0)];
      return `${policy?policy.label+"→":""}${model.label}:${model.variant}`;
    }).join(", ")
  : `${mainTerm.label||"FSQ_INIT"}:${mainTerm.variant||"fsq_initial"}`;
document.getElementById("summary").textContent=
  `${policyLabels} · MAIN ${mainLabels} · display-only ${displayLabels} · ${DATA.target_task} tasks ${DATA.task_ids.join(", ")} · ${DATA.selected_episode_count} episodes · ${DATA.occurrence_count} occurrences / ${DATA.evaluation_count} policy evaluations · shift ±${DATA.time_shift_offset}`;
document.getElementById("content").addEventListener("click",handleFinalImageClick);
document.getElementById("review-export").addEventListener("click",exportCorrections);
document.getElementById("review-import").addEventListener("change",event=>{
  const file=event.target.files&&event.target.files[0];
  if(file) importCorrections(file);
  event.target.value="";
});
document.getElementById("review-clear").addEventListener("click",()=>{
  const count=Object.keys(REVIEW.corrections).length;
  if(!count) { setReviewStatus("No corrections to clear."); return; }
  if(!confirm(`Clear all ${count} human-review overrides for this report?`)) return;
  REVIEW.corrections={};
  REVIEW.updated_at=new Date().toISOString();
  saveReview("All corrections cleared");
  renderReviewedResults();
});
document.getElementById("success-summary").innerHTML=successCards(successStats(allOccurrences));
updateReviewCount();
if(!storageAvailable) setReviewStatus("Browser storage unavailable; export corrections to keep them.",true);
const firstSpace=skillSpaces[0];
const initialSkill=firstSpace&&(firstSpace.skills||[]).slice().sort(
  (a,b)=>memberIds(b).length-memberIds(a).length||a.token-b.token
)[0];
if(initialSkill) selectSkillCode(firstSpace.model_index,initialSkill.token);
else { renderCodebooks(); document.getElementById("content").innerHTML='<div class="empty">No skill occurrences selected.</div>'; }
connectReviewServer();
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
