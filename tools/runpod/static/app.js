"use strict";
let token = "", state = null, busy = false, configured = false;
let podRenderKey = "", resultRenderKey = "";
const stopping = new Set();
const $ = id => document.getElementById(id);
const money = value => value == null ? "—" : new Intl.NumberFormat("en-US", {style:"currency",currency:"USD",minimumFractionDigits:4}).format(value);
const pct = value => value == null ? "—" : `${value.toFixed(1)}%`;
const when = value => value ? new Date(value).toLocaleString() : "—";
function el(tag, text, cls) { const node=document.createElement(tag); if(text != null)node.textContent=text; if(cls)node.className=cls; return node; }
function notice(id, text) { $(id).textContent=text || ""; $(id).hidden=!text; }
async function api(path, body) {
  const response=await fetch(`/runpod/api/${path}`, body === undefined ? {cache:"no-store"} : {
    method:"POST",headers:{"Content-Type":"application/json","X-Sync-Token":token},body:JSON.stringify(body)});
  const value=await response.json(); if(!response.ok)throw new Error(value.error || `HTTP ${response.status}`); return value;
}
const labels={RUNNING:"실행 중",EXITED:"중단됨",TERMINATED:"종료됨",DELETED:"삭제됨",STOP_REQUESTED:"중단 확인 중",STARTING:"시작 중",PROVISIONING:"준비 중",running:"실행 중",pending:"대기 중",done:"완료",failed:"실패",stopped:"중단됨",pod_stopped:"Pod 종료",cancelled:"취소",final:"최종 결과",partial:"부분 결과",unverified:"완료 상태 미확인"};
function badge(value){ return el("span",labels[value] || value,`badge ${String(value).toLowerCase()}`); }
function renderPods(query) {
  const key=JSON.stringify([state.pods,query,[...stopping]]);if(key===podRenderKey)return;podRenderKey=key;
  const fragment=document.createDocumentFragment();
  for(const pod of state.pods) {
    if(!`${pod.name} ${pod.id} ${(pod.jobs || []).map(j=>j.name).join(" ")}`.toLowerCase().includes(query))continue;
    const card=el("article",null,"card"), top=el("div",null,"card-head"), title=el("div");
    title.append(el("h3",pod.name),el("div",`${pod.id} · ${pod.gpu_name || "GPU 정보 없음"} × ${pod.gpu_count || 0}`,"meta"));
    const controls=el("div",null,"card-actions"), stop=el("button",stopping.has(pod.id)?"중단 요청 중…":"Pod 중단","danger");
    stop.disabled=pod.status!=="RUNNING" || !pod.present || pod.stale || stopping.has(pod.id);
    stop.addEventListener("click",()=>stopPod(pod)); controls.append(badge(pod.status),stop); top.append(title,controls);card.append(top);
    const costs=el("div",null,"costs");
    for(const [label,value] of [["현재 요금 / 시간",money(pod.hourly_cost)],[pod.billing?.lifetime?"누적 집계 비용":"최근 30일 집계 비용",money(pod.billing?.totalAmount)],[["RUNNING","STOP_REQUESTED"].includes(pod.status)?"현재 실행분 추정":"마지막 실행분 추정",money(pod.session_estimate)]]) {
      const item=el("div",label,"muted");item.append(el("strong",value));costs.append(item);
    }
    card.append(costs);
    if(pod.billing)card.append(el("p",`집계: ${when(pod.billing.updated_at)} · GPU ${money(pod.billing.gpuAmount)} / CPU ${money(pod.billing.cpuAmount)} / 디스크 ${money(pod.billing.diskAmount)}. 반영 지연 및 별도 네트워크 볼륨 요금은 제외될 수 있습니다.`,"meta"));
    card.append(el("p","중단하면 GPU 작업이 멈춥니다. 저장소 요금은 남을 수 있으며 컨테이너 파일은 사라질 수 있습니다.","meta"));
    if(pod.gpus?.length && ["RUNNING","STOP_REQUESTED"].includes(pod.status)){const list=el("div",null,"gpu-list");for(const gpu of pod.gpus){const cell=el("div",`GPU ${gpu.index} · ${gpu.name}: ${(gpu.memory_used_mib/1024).toFixed(1)} / ${(gpu.memory_total_mib/1024).toFixed(1)} GiB · 사용률 ${gpu.util_pct}%`,"gpu");const bar=el("progress");bar.max=gpu.memory_total_mib;bar.value=gpu.memory_used_mib;bar.setAttribute("aria-label",`GPU ${gpu.index} 메모리`);cell.append(bar);list.append(cell);}card.append(list);}
    if(pod.jobs?.length){const table=el("table",null,"jobs"),head=el("tr");for(const text of ["잡 이름","상태","GPU","시작"]){head.append(el("th",text));}const thead=el("thead");thead.append(head);const tbody=el("tbody");for(const job of pod.jobs){const row=el("tr"),name=el("td",job.name),status=el("td");if(job.result_key){const result=state.results.find(r=>r.key===job.result_key);if(result){const link=el("a"," · 결과");link.href=`#result-${result.remote_id}-${pod.id}`;name.append(link);}}status.append(badge(job.state));row.append(name,status,el("td",job.gpus || "—"),el("td",when(job.started)));tbody.append(row);}table.append(thead,tbody);card.append(table);}else card.append(el("p",pod.ssh_error?"작업을 조회하지 못했습니다.":"등록된 작업이 없습니다.","muted"));
    for(const text of [pod.ssh_error,pod.billing_error,...(pod.warnings || []),pod.stale?"조회가 지연되어 이전 기록을 표시합니다.":""]){if(text)card.append(el("p",text,"inline-error"));}
    fragment.append(card);
  }
  $("empty-pods").hidden=fragment.childNodes.length>0;$("pods").replaceChildren(fragment);
}
function renderResults(query) {
  const key=JSON.stringify([state.results,query]);if(key===resultRenderKey)return;resultRenderKey=key;
  const fragment=document.createDocumentFragment();
  for(const result of [...state.results].sort((a,b)=>b.mtime-a.mtime)){
    if(!`${result.name} ${result.pod_name} ${result.models.map(m=>m.label).join(" ")}`.toLowerCase().includes(query))continue;
    const card=el("article",null,"card");card.id=`result-${result.remote_id}-${result.pod_id}`;
    const top=el("div",null,"card-head"),title=el("div");title.append(el("h3",result.name),el("div",`${result.pod_name} · 수집 ${when(result.collected_at)}`,"meta"));top.append(title,badge(result.phase));card.append(top,el("p",result.output,"meta"));
    const summary=el("div",null,"result-summary");for(const model of result.models){const cell=el("div",model.label,"model");cell.append(el("strong",pct(model.overall_pct)),el("span",`${model.task_count} tasks · 전체 성공률`,"muted"));summary.append(cell);}card.append(summary);
    if(result.chart_url){const link=el("a");link.href=result.chart_url;link.target="_blank";link.rel="noopener";const image=el("img",null,"result-chart");image.src=`${result.chart_url}?v=${result.fingerprint}`;image.alt=`${result.name}: 전체 및 task별 성공률`;image.loading="lazy";image.width=1100;image.height=550;link.append(image);card.append(link);}
    const details=el("details",null,"tasks");details.append(el("summary","Task별 성공률 보기"));
    for(const model of result.models){details.append(el("h4",model.label));const list=el("div",null,"task-list");for(const task of model.tasks){const row=el("div",null,"task-row"),bar=el("progress");bar.max=100;if(task.success_pct!=null)bar.value=task.success_pct;bar.setAttribute("aria-label",`${task.name} 성공률`);row.append(el("span",task.name),bar,el("span",pct(task.success_pct)));list.append(row);}details.append(list);}card.append(details);fragment.append(card);
  }
  $("empty-results").hidden=fragment.childNodes.length>0;$("results").replaceChildren(fragment);
}
function render(){const query=$("search").value.trim().toLowerCase();$("running-count").textContent=state.pods.filter(p=>p.status==="RUNNING").length;$("job-count").textContent=state.pods.filter(p=>p.status==="RUNNING").flatMap(p=>p.jobs || []).filter(j=>j.state==="running").length;$("result-count").textContent=state.results.length;renderPods(query);renderResults(query);}
async function load(){if(busy)return;busy=true;try{state=await api("state");token=state.token;notice("error",state.error);$("updated").textContent=`마지막 확인 ${when(state.updated_at)}`;if(!configured){for(const [key,value] of Object.entries(state.config))$("settings-form").elements[key].value=value;configured=true;}render();}catch(error){notice("error",error.message);}finally{busy=false;}}
async function stopPod(pod){if(!confirm(`${pod.name} (${pod.id})를 중단할까요?\n실행 중인 모든 작업이 멈추며 컨테이너 파일이 사라질 수 있습니다.\n마지막으로 결과 수집을 시도한 뒤 중단합니다.\n저장소 요금은 계속 발생할 수 있습니다.`))return;stopping.add(pod.id);render();try{const result=await api("stop",{pod_id:pod.id,confirm:pod.id});notice("message",[result.message,result.warning].filter(Boolean).join("\n"));await load();}catch(error){notice("error",error.message);}finally{stopping.delete(pod.id);render();}}
$("refresh").addEventListener("click",async()=>{try{await api("refresh",{});notice("message","새 조회를 요청했습니다.");setTimeout(load,1000);}catch(error){notice("error",error.message);}});
$("settings-form").addEventListener("submit",async event=>{event.preventDefault();const form=event.currentTarget.elements;try{await api("settings",{project_root:form.project_root.value,identity:form.identity.value,interval:Number(form.interval.value)});notice("message","조회 설정을 저장했습니다.");}catch(error){notice("error",error.message);}});
$("search").addEventListener("input",()=>{if(state)render();});load();setInterval(load,5000);
