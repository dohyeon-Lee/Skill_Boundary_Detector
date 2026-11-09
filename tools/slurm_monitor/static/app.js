"use strict";
const $ = id => document.getElementById(id);
const cards = new Map(), drafts = new Map();
const canceling = new Set();
const API_BASE=location.pathname==="/slurm"||location.pathname.startsWith("/slurm/")?"/slurm":"";
const groups=["training-running","training-ended","running","pending","ended"];
const sectionsKey="sbd.slurm.sections.v1";
let sectionPreferences={};
try {
  const saved=JSON.parse(localStorage.getItem(sectionsKey));
  if(saved&&typeof saved==="object"&&!Array.isArray(saved))sectionPreferences=saved;
} catch {}
for(const group of [...groups,"cluster-queue"]) {
  const section=$("section-"+group);
  if(typeof sectionPreferences[group]==="boolean")section.open=sectionPreferences[group];
  section.addEventListener("toggle",()=>{
    sectionPreferences[group]=section.open;
    try {localStorage.setItem(sectionsKey,JSON.stringify(sectionPreferences));} catch {}
  });
}
let state = null, token = "", loading = false, toastTimer;
const deliveryNames = {none:"미전송",waiting:"전송 대기",sending:"전송 중",sent:"접수 성공 · 처리 여부 미확인",failed:"전송 실패",uncertain:"접수 여부 확인 필요"};
const date = value => value && !["Unknown","N/A","None"].includes(value) ? new Date(value).toLocaleString("ko-KR") : "—";

function toast(message, error=false) {
  clearTimeout(toastTimer); $("toast").textContent = message;
  $("toast").className = "notice" + (error ? " error" : ""); $("toast").hidden = false;
  toastTimer = setTimeout(() => $("toast").hidden = true, 6000);
}
async function post(path, data) {
  const response = await fetch(API_BASE+path, {method:"POST",headers:{"Content-Type":"application/json","X-Monitor-Token":token},body:JSON.stringify(data)});
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
  renderJobGPUs(card,job);
  card.querySelector("h3").textContent=job.display_name;
  card.querySelector("h3").title=job.display_name;
  const training=job.kind==="training";
  card.querySelector(".job-meta").textContent=training?
    `개별 학습 · GPU 할당 #${job.job_id} (${job.allocation_state}) · 시작 ${date(job.started_at)}`:
    `GPU 할당 #${job.job_id} · ${job.job_name} · 경과 ${job.elapsed || "—"} / 제한 ${job.time_limit || "—"}`;
  card.querySelector(".armed-label").textContent=training?"이 학습 종료 시 이 지시를 Codex에 자동 전송":"GPU 할당 종료 시 이 지시를 Codex에 자동 전송";
  badge(card.querySelector(".state"),job.state==="WAITING_FOR_MEMORY"?"VRAM 대기":job.state,job.state==="COMPLETED"?"green":job.terminal?"red":job.group==="pending"||job.state==="WAITING_FOR_MEMORY"?"amber":"blue");
  card.querySelector(".location").textContent=job.terminal?`종료 ${date(job.ended_at)} · exit ${job.exit_code ?? "—"}${job.reason?" · "+job.reason:""}`:job.reason || "";
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
  detailLine(paths,"학습 실행 기록",job.status_path);
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
function renderJobGPUs(jobCard,job) {
  const container=jobCard.querySelector(".job-gpus");
  container.replaceChildren();
  container.hidden=job.terminal||!["running","training-running"].includes(job.group);
  if(container.hidden)return;
  const devices=(state.gpu?.devices||[]).filter(device=>device.job_ids.includes(job.job_id));
  for(const device of devices) {
    const card=document.createElement("div");card.className="gpu-device";
    const title=document.createElement("strong");
    title.textContent=`${device.node} · GPU ${device.index} · ${device.name}`;
    const memory=document.createElement("div");memory.className="gpu-memory";
    const individual=job.kind==="training";
    const processes=(device.processes||[]).filter(process=>process.run_id===job.run_id);
    const used=individual?processes.reduce((sum,process)=>sum+process.used_gib,0):device.used_gib;
    const measured=!individual||typeof device.process_sampled_at==="number";
    memory.textContent=individual?(processes.length?`이 실행 사용 ${used.toFixed(1)} GiB · GPU 프로세스 ${processes.length}개 합계`:
      measured&&!device.process_error?"이 실행의 GPU 사용량을 아직 확인할 수 없습니다.":"이 실행의 VRAM 사용량 확인 중…"):
      `사용 ${device.used_gib.toFixed(1)} GiB  /  남음 ${device.free_gib.toFixed(1)} GiB  /  전체 ${device.total_gib.toFixed(1)} GiB`;
    const meter=document.createElement("progress");meter.max=device.total_gib;meter.value=used;
    meter.hidden=individual&&!processes.length;
    meter.setAttribute("aria-label",individual?"이 실행의 GPU 메모리 사용량":"GPU 전체 메모리 사용량");
    const note=document.createElement("span");note.className="muted";
    const stale=device.stale||(individual&&device.processes_stale);
    const sampled=individual&&measured?new Date(device.process_sampled_at*1000).toISOString():device.updated_at;
    note.textContent=`${individual?"해당 실행의 모델·자식 워커만 합산":"GPU 전체 사용량 · 공유 학습 합산"} · ${stale?"조회 지연 · 마지막 값 ":"확인 "}${date(sampled)}`;
    if(individual&&processes.length)note.title="PID "+processes.map(process=>process.pid).join(", ");
    card.append(title,memory,meter,note);container.append(card);
  }
  if(!devices.length) {
    const empty=document.createElement("span");empty.className="muted";
    empty.textContent=state.gpu?.errors?.length?"VRAM 조회 지연 · 다음 확인을 기다리는 중":"GPU 메모리 확인 중…";
    container.append(empty);
  }
}
function renderClusterQueue() {
  const data=state.cluster_queue, container=$("cluster-queue-groups");
  container.replaceChildren();
  $("cluster-queue-status").textContent=data?.updated_at?`마지막 확인 ${date(data.updated_at)} · 약 ${data.interval}초마다 읽기 전용 조회`:"Slurm 대기열 확인 중…";
  $("cluster-queue-error").hidden=!data?.error;
  $("cluster-queue-error").textContent=data?.error?"조회 지연 · 마지막 확인 값을 표시합니다.\n"+data.error:"";
  const unique=new Set((data?.groups||[]).flatMap(group=>group.pending.map(row=>row.job_id)));
  $("label-cluster-queue").textContent=data?.updated_at?`${unique.size}개 대기`:"";
  const element=(tag,text,className)=>{const node=document.createElement(tag);node.textContent=text;if(className)node.className=className;return node;};
  for(const group of data?.groups||[]) {
    const panel=document.createElement("section");panel.className="cluster-queue-panel";
    panel.append(element("h3",`${group.label} · ${group.pending.length}개 대기`));
    panel.append(element("p",group.nodes.map(node=>`${node.name}: 할당 ${node.allocated}/${node.total} GPU · 미할당 ${node.unallocated} · ${node.state}`).join(" / ")||"노드 정보 없음","muted"));
    const estimate=group.estimate;
    if(estimate?.available) {
      const position=estimate.first===estimate.last?`${estimate.first}번째`:`${estimate.first}~${estimate.last}번째 (동점 포함)`;
      panel.append(element("p",`지금 새 잡 제출 시 예상 ${position} · Priority 약 ${estimate.priority.toLocaleString("ko-KR")} · 앞에 ${estimate.ahead}개`,"queue-estimate"));
      panel.append(element("p",`${estimate.account} 계정 · ${estimate.basis} 실제 잡을 제출하지 않고 계산한 추정입니다.` ,"muted"));
    } else panel.append(element("p",estimate?.reason||"신규 잡 예상 순위 확인 중…","muted"));
    if(group.pending.length) {
      const scroll=document.createElement("div");scroll.className="queue-table-scroll";
      const table=document.createElement("table");table.className="queue-table";
      const head=document.createElement("thead"), headers=document.createElement("tr");
      for(const label of ["Priority 순위","Job","사용자","Priority","GPU 요청","Slurm 예상 시작","대기 사유"])headers.append(element("th",label));
      head.append(headers);table.append(head);
      const body=document.createElement("tbody");
      for(const job of group.pending) {
        const row=document.createElement("tr");if(job.mine)row.className="queue-mine";
        const position=job.rank_first===job.rank_last?`${job.rank_first}`:`${job.rank_first}~${job.rank_last}`;
        const start=["Unknown","N/A","None","(null)",""].includes(job.expected_start)?"미정":date(job.expected_start);
        const values=[position,`#${job.job_id} · ${job.name}`,job.user+(job.mine?" (내 잡)":""),job.priority.toLocaleString("ko-KR"),job.gpu_request,start,job.reason];
        for(const value of values)row.append(element("td",value));
        row.title=`대상 노드: ${job.eligible_nodes.join(", ")} · CPU ${job.cpus} · QoS ${job.qos||"기본값"}`;
        body.append(row);
      }
      table.append(body);scroll.append(table);panel.append(scroll);
    } else panel.append(element("p","이 GPU를 대상으로 하는 대기 잡이 없습니다.","muted"));
    container.append(panel);
  }
}
function render() {
  if(!state)return;
  renderClusterQueue();
  const needle=$("search").value.trim().toLowerCase();
  const visible=state.jobs.filter(job=>[job.display_name,job.job_id,job.job_name,...(job.outputs||[])].join(" ").toLowerCase().includes(needle));
  for(const group of groups) {
    if($("count-"+group)) $("count-"+group).textContent=state.jobs.filter(job=>job.group===group).length;
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
    const response=await fetch(API_BASE+"/api/state",{cache:"no-store"});
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
