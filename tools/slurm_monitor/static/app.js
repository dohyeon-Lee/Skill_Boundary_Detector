"use strict";
const $ = id => document.getElementById(id);
const cards = new Map(), drafts = new Map();
const canceling = new Set();
let state = null, token = "", loading = false, toastTimer;
const deliveryNames = {none:"미전송",waiting:"전송 대기",sending:"전송 중",sent:"접수 성공 · 처리 여부 미확인",failed:"전송 실패",uncertain:"접수 여부 확인 필요"};
const date = value => value && !["Unknown","N/A","None"].includes(value) ? new Date(value).toLocaleString("ko-KR") : "—";

function toast(message, error=false) {
  clearTimeout(toastTimer); $("toast").textContent = message;
  $("toast").className = "notice" + (error ? " error" : ""); $("toast").hidden = false;
  toastTimer = setTimeout(() => $("toast").hidden = true, 6000);
}
async function post(path, data) {
  const response = await fetch(path, {method:"POST",headers:{"Content-Type":"application/json","X-Monitor-Token":token},body:JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "요청 실패");
  return result;
}
function badge(element, text, color="") {
  element.textContent=text;
  element.classList.remove("green","red","blue","amber");
  if(color) element.classList.add(color);
}
function detailLine(parent, label, text) {
  if (!text) return;
  const line=document.createElement("p"); line.textContent=label+": "+text; parent.append(line);
}
function draft(card, key) {
  drafts.set(key,{prompt:card.querySelector(".prompt").value,armed:card.querySelector(".armed").checked,output_label:card.querySelector(".output-label").value});
  card.querySelector(".draft-label").textContent="저장하지 않은 변경";
}
async function save(card, key) {
  const values = {prompt:card.querySelector(".prompt").value,armed:card.querySelector(".armed").checked,output_label:card.querySelector(".output-label").value};
  await post("/api/job",{key,values}); drafts.delete(key); await load();
}
function createCard(job) {
  const card=$("job-template").content.firstElementChild.cloneNode(true);
  card.querySelector(".cancel").addEventListener("click",async()=>{
    const current=state.jobs.find(row=>row.key===job.key);
    if(!current || !current.can_cancel || canceling.has(job.key))return;
    const warning=current.armed?"\n취소가 확인되면 저장된 종료 후 지시도 기존 설정대로 전송됩니다.":"";
    if(!confirm(`${current.display_name}\nJob #${current.job_id}을 취소할까요?${warning}`))return;
    canceling.add(job.key);updateCard(card,current);
    try{await post("/api/cancel",{key:job.key});toast("취소를 요청했습니다. Slurm에서 종료가 확인되면 상태가 바뀝니다.");}
    catch(error){toast(error.message,true);}
    finally{canceling.delete(job.key);await load();}
  });
  for (const selector of [".prompt",".armed",".output-label"]) card.querySelector(selector).addEventListener("input",()=>draft(card,job.key));
  card.querySelector(".save").addEventListener("click", async event=>{
    const button=event.currentTarget;button.disabled=true;
    try {await save(card,job.key);toast("job 지시를 저장했습니다.");} catch(error){toast(error.message,true);} finally {button.disabled=false;}
  });
  card.querySelector(".send").addEventListener("click",async event=>{
    const current=state.jobs.find(row=>row.key===job.key);
    if (current.delivery.status==="uncertain" && !confirm("이미 접수됐을 수 있습니다. Codex 세션을 확인했으며 다시 전송하시겠습니까?")) return;
    const button=event.currentTarget;button.disabled=true;
    try {if(drafts.has(job.key)) await save(card,job.key);await post("/api/send",{key:job.key});toast("Codex 전송을 요청했습니다.");await load();}
    catch(error){toast(error.message,true);}finally{button.disabled=false;}
  });
  return card;
}
function updateCard(card, job) {
  card.querySelector("h3").textContent=job.display_name;
  card.querySelector("h3").title=job.display_name;
  card.querySelector(".job-meta").textContent=`#${job.job_id} · ${job.job_name} · 경과 ${job.elapsed || "—"} / 제한 ${job.time_limit || "—"}`;
  badge(card.querySelector(".state"),job.state,job.state==="COMPLETED"?"green":job.terminal?"red":job.state==="PENDING"?"amber":"blue");
  card.querySelector(".location").textContent=job.terminal?`종료 ${date(job.ended_at)} · exit ${job.exit_code || "—"}`:job.reason || "";
  const cancelStatus=job.cancel?.status || "";
  const cancelButton=card.querySelector(".cancel");
  cancelButton.hidden=!job.can_cancel;
  cancelButton.disabled=canceling.has(job.key)||["requesting","requested"].includes(cancelStatus);
  cancelButton.textContent=cancelButton.disabled?"취소 요청 중":"Job 취소";
  const cancelNote=card.querySelector(".cancel-status");
  cancelNote.hidden=!cancelStatus||job.terminal;
  cancelNote.textContent=cancelStatus==="uncertain"?`취소 요청 결과 확인 필요 · ${job.cancel.error || "Slurm 상태를 확인하세요."}`:"취소 요청 중 · Slurm 종료 상태 확인을 기다립니다.";
  const progress=card.querySelector(".training-progress");
  progress.replaceChildren();
  progress.hidden=!(job.training_progress?.length);
  for(const item of job.training_progress || []) {
    const line=document.createElement("div");
    const label=job.training_progress.length>1?(item.output.split("/").pop()+" · "):"";
    const number=value=>Number(value).toLocaleString("ko-KR");
    const value=item.step===null?"스텝 확인 중":`${item.approximate?"약 ":""}${number(item.step)}`;
    const fraction=item.step!==null&&item.total?` · ${(item.step/item.total*100).toFixed(1)}%`:"";
    line.textContent=label+(job.state==="PENDING"?"대기 중 · 이전 학습 기록 ":"학습 스텝 ")+value+(item.total?` / ${number(item.total)}`:"")+fraction;
    line.title=item.source?`최근 기록: ${date(item.updated_at)}\n${item.source}`:"학습 로그가 기록되면 자동으로 표시합니다.";
    progress.append(line);
  }
  const status=job.delivery.status;
  let text=deliveryNames[status] || status;
  if(status==="none") text=job.armed?"자동 전송 설정됨 · 종료 대기":job.prompt?"자동 전송 꺼짐":"종료 후 지시 미설정";
  badge(card.querySelector(".delivery"),text,status==="sent"?"green":status==="failed"||status==="uncertain"?"red":status==="sending"||status==="waiting"?"blue":"");
  if (!drafts.has(job.key)) {
    card.querySelector(".prompt").value=job.prompt || "";
    card.querySelector(".armed").checked=Boolean(job.armed);
    card.querySelector(".output-label").value=job.output_label || "";
    card.querySelector(".draft-label").textContent="";
  }
  card.querySelector(".armed").disabled=(job.terminal&&!job.armed)||status==="sending";
  card.querySelector(".save").disabled=status==="sending";
  const sendButton=card.querySelector(".send");
  sendButton.hidden=!job.terminal;
  sendButton.disabled=["sent","sending","waiting"].includes(status)||!state.connection.configured;
  sendButton.textContent=["failed","uncertain"].includes(status)?"다시 전송":"지금 전송";
  card.querySelector(".saved-state").textContent=status==="sent"?`접수 ${date(job.delivery.finished_at)}`:job.terminal?"저장 후 ‘지금 전송’으로 전달하세요.":job.armed?"종료가 확인되면 한 번 전송합니다.":"자동 전송을 켜려면 지시와 함께 저장하세요.";
  card.querySelector(".prompt-summary").textContent=job.prompt?"종료 후 지시 보기 / 수정":"종료 후 할 일 작성";
  const paths=card.querySelector(".paths");paths.replaceChildren();
  detailLine(paths,"폴더명 출처",job.output_source || "아직 확인되지 않음 · 표시 이름을 직접 보정할 수 있습니다");
  for(const path of job.outputs || []) detailLine(paths,"출력",path);
  detailLine(paths,"설정 스냅샷",job.snapshot_path);
  detailLine(paths,"작업 폴더",job.work_dir);
  detailLine(paths,"제출",date(job.submitted_at));
  for(const path of job.log_paths || []) detailLine(paths,"로그",path);
  const history=card.querySelector(".history");history.replaceChildren();
  if(job.delivery.error) detailLine(history,"전송 오류",job.delivery.error);
  for(const record of [...(job.events || [])].reverse()) {
    const block=document.createElement("pre");
    block.textContent=`${date(record.finished_at)} · ${deliveryNames[record.status] || record.status}\n세션: ${record.thread_id}\n${record.receipt || record.error || ""}\n\n보낸 지시 원문:\n${record.prompt || ""}`;
    history.append(block);
  }
}
function render() {
  if(!state)return;
  const needle=$("search").value.trim().toLowerCase();
  const visible=state.jobs.filter(job=>[job.display_name,job.job_id,job.job_name,...(job.outputs||[])].join(" ").toLowerCase().includes(needle));
  for(const group of ["running","pending","ended"]) {
    $("count-"+group).textContent=state.jobs.filter(job=>job.group===group).length;
    const rows=visible.filter(job=>job.group===group);
    $("label-"+group).textContent=rows.length;
    $("empty-"+group).hidden=rows.length!==0;
    for(const [index,job] of rows.entries()) {
      let card=cards.get(job.key);if(!card){card=createCard(job);cards.set(job.key,card);}
      updateCard(card,job);
      if($(group).children[index]!==card)$(group).insertBefore(card,$(group).children[index]||null);
      card.hidden=false;
    }
  }
  const visibleKeys=new Set(visible.map(job=>job.key));
  for(const [key,card] of cards) if(!visibleKeys.has(key))card.hidden=true;
  $("count-sent").textContent=state.jobs.filter(job=>job.delivery.status==="sent").length;
  $("error").hidden=!state.error;$("error").textContent=state.error?"Slurm 조회 실패 · 마지막 확인 상태를 표시합니다.\n"+state.error:"";
  $("updated").textContent=state.polling?"Slurm 확인 중…":"마지막 확인 "+date(state.updated_at);
  $("owner").textContent=`${state.config.user} · ${state.config.interval}초마다 확인`;
  $("session-title").textContent=state.connection.session_title || state.config.thread_id || "전송 대상 세션을 설정하세요";
  $("session-state").textContent=state.connection.detail;
}
async function load(initial=false) {
  if(loading)return;
  loading=true;
  try {
    const response=await fetch("/api/state",{cache:"no-store"});
    if(!response.ok)throw new Error("모니터 연결 실패");
    state=await response.json();token=state.token;
    if(initial) {
      const form=$("settings-form");
      for(const key of ["thread_id","interval","codex_home","codex_command","codex_remote"])form.elements.namedItem(key).value=state.config[key]||"";
      if(!state.connection.configured)$("settings").hidden=false;
    }
    render();
  } catch(error){$("error").hidden=false;$("error").textContent=error.message;} finally{loading=false;}
}
$("settings-form").addEventListener("submit",async event=>{
  event.preventDefault();const values=Object.fromEntries(new FormData(event.currentTarget));values.interval=Number(values.interval);
  try{await post("/api/settings",values);toast("Codex 연결 설정을 저장했습니다.");await load();}catch(error){toast(error.message,true);}
});
$("toggle-settings").addEventListener("click",()=>$("settings").hidden=!$("settings").hidden);
$("search").addEventListener("input",render);
$("refresh").addEventListener("click",async()=>{try{await post("/api/refresh",{});toast("Slurm 새로고침을 요청했습니다.");await load();}catch(error){toast(error.message,true);}});
load(true);setInterval(load,5000);
