"use strict";
const $ = (id) => document.getElementById(id);
const state = {config:null, source:null, destination:null, path:"", offset:0, listing:null,
  selected:new Set(), browseVersion:0, prepared:null, pending:false, searchTimer:null};
const statuses = {queued:"대기",running:"전송 중",completed:"완료",failed:"실패",cancelled:"취소",cancelling:"취소 중"};
function node(tag, text, className) {
  const element=document.createElement(tag); if(text!==undefined)element.textContent=text;
  if(className)element.className=className;return element;
}
function button(text, handler, className="button quiet") {
  const element=node("button",text,className);element.type="button";element.addEventListener("click",handler);return element;
}
function error(message) { $("error").textContent=message||"";$("error").classList.toggle("hidden",!message); }
async function api(path, body) {
  const options=body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json","X-Sync-Token":state.config.token},body:JSON.stringify(body)};
  const response=await fetch(path,options);const payload=await response.json();
  if(!response.ok)throw new Error(payload.error||"요청에 실패했습니다.");return payload;
}
function endpoint(id) {return state.config.endpoints.find(item=>item.id===id);}
function locationLabel(item) {return (item.host?item.host+":":"")+item.base;}
function formatSize(bytes) {
  if(bytes===null||bytes===undefined)return "폴더";
  const units=["B","KB","MB","GB","TB"];let index=0;
  while(bytes>=1024&&index<units.length-1){bytes/=1024;index++;}return bytes.toFixed(index?1:0)+" "+units[index];
}
function renderLocations() {
  $("source").value=state.source;$("destination").value=state.destination;
  $("source-path").textContent=locationLabel(endpoint(state.source));
  $("destination-path").textContent=locationLabel(endpoint(state.destination));
}
function covered(path) {return [...state.selected].some(parent=>path!==parent&&path.startsWith(parent+"/"));}
function selectionPath(item) {
  return item.match&&item.match!==item.name?item.path+"/"+item.match:item.path;
}
function renderSelection() {
  const count=state.selected.size;$("selected-count").textContent=count;$("selected-list").replaceChildren();
  $("prepare").disabled=!count||state.source===state.destination||state.pending;
  $("prepare").textContent=count?`${count}개 항목 전송 준비`:"항목을 선택하세요";
  if(!count){const empty=node("div",undefined,"empty");empty.append(node("strong","체크포인트를 골라 담으세요"),node("span","다른 폴더에 있는 항목도 함께 담을 수 있어요."));$("selected-list").append(empty);}
  for(const path of [...state.selected].sort()) {
    const item=node("div",undefined,"selected-item");item.append(node("code",path));
    const remove=button("×",()=>{state.selected.delete(path);renderSelection();renderFiles();},"remove");
    remove.setAttribute("aria-label",path+" 선택 해제");item.append(remove);$("selected-list").append(item);
  }
}
function breadcrumbs() {
  $("breadcrumbs").replaceChildren(button("프로젝트",()=>browse("")));
  let path="";
  for(const part of state.path.split("/").filter(Boolean)) {
    path=path?path+"/"+part:part;const target=path;
    $("breadcrumbs").append(node("span","/","crumb-divider"),button(part,()=>browse(target)));
  }
  $("add-current").disabled=!state.path;
}
function renderFiles() {
  if(!state.listing)return;
  const list=$("file-list");list.replaceChildren();
  const eligible=state.listing.entries.filter(item=>item.kind==="directory"||item.kind==="file");
  const picked=eligible.filter(item=>{const target=selectionPath(item);return state.selected.has(target)||covered(target);}).length;
  $("select-visible").disabled=!eligible.length;$("select-visible").checked=!!eligible.length&&picked===eligible.length;
  $("select-visible").indeterminate=picked>0&&picked<eligible.length;
  if(!state.listing.entries.length)list.append(node("div","표시할 항목이 없습니다.","empty"));
  for(const item of state.listing.entries) {
    const target=selectionPath(item);
    const selected=state.selected.has(target)||covered(target);
    const row=node("div",undefined,"file-row"+(selected?" selected":""));
    const check=node("input");check.type="checkbox";check.checked=selected;
    check.disabled=covered(target)||!["directory","file"].includes(item.kind);check.setAttribute("aria-label",target+" 선택");
    if(check.disabled&&!covered(target))row.classList.add("disabled");
    check.addEventListener("change",()=>{check.checked?state.selected.add(target):state.selected.delete(target);renderSelection();renderFiles();});
    row.append(check,node("span",item.kind==="directory"?"▱":item.kind==="symlink"?"↗":"▤","file-icon"));
    const name=item.kind==="directory"?button(item.name,()=>browse(item.path),"file-name folder"):node("span",item.name,"file-name");
    name.title=item.path;
    const metadata=item.match&&item.match!==item.name?`하위 일치 · ${item.match}`:
      item.kind==="symlink"?"링크":item.kind==="special"?"선택 불가":formatSize(item.size);
    row.append(name,node("span",metadata,"file-meta"));list.append(row);
  }
  const {total,offset,limit}=state.listing;$("item-count").textContent=`${total}개 항목`;
  $("page-info").textContent=total?`${offset+1}–${Math.min(total,offset+limit)} / ${total}`:"0개";
  $("previous").disabled=offset===0;$("next").disabled=offset+limit>=total;
}
async function browse(path=state.path,offset=0,keepSearch=false) {
  const version=++state.browseVersion;state.path=path;state.offset=offset;state.listing=null;
  if(!keepSearch)$("search").value="";breadcrumbs();$("file-list").replaceChildren(node("div","폴더를 불러오는 중…","empty"));
  try {
    const params=new URLSearchParams({endpoint:state.source,path,offset:String(offset),search:$("search").value});
    const listing=await api("/api/browse?"+params);
    if(version!==state.browseVersion)return;state.listing=listing;error("");renderFiles();
  } catch(issue) {if(version===state.browseVersion){$("file-list").replaceChildren(node("div","목록을 불러오지 못했습니다.","empty"));error(issue.message);}}
}
async function prepare() {
  if(state.pending||!state.selected.size)return;state.pending=true;renderSelection();error("");
  const request={source:state.source,destination:state.destination,paths:[...state.selected]};
  try {
    const plan=await api("/api/plan",request);state.prepared={request,plan};
    $("confirmation-route").textContent=endpoint(request.source).name+" → "+endpoint(request.destination).name;
    $("confirmation-list").replaceChildren();
    for(const item of plan.items){const element=node("div",undefined,"confirmation-item");element.append(node("strong",item.path),node("code",item.source),node("code","→ "+item.destination));$("confirmation-list").append(element);}
    $("collapsed").textContent=plan.collapsed?`${plan.collapsed}개 중복 선택을 제외하고 ${plan.items.length}개를 전송합니다.`:`${plan.items.length}개 항목 · 기존 상대 경로 유지`;
    $("start").disabled=false;$("start").textContent=`${plan.items.length}개 전송 시작`;$("confirmation").showModal();
  } catch(issue) {error(issue.message);}finally{state.pending=false;renderSelection();}
}
async function start() {
  const prepared=state.prepared;if(!prepared)return;$("start").disabled=true;$("start").textContent="대기열에 추가 중…";
  try {
    await api("/api/transfers",prepared.request);state.prepared=null;$("confirmation").close();
    state.selected.clear();renderSelection();renderFiles();await pollJobs();$("jobs").scrollIntoView({behavior:"smooth",block:"nearest"});
  } catch(issue){$("confirmation").close();error(issue.message);}finally{$("start").disabled=false;}
}
async function pollJobs() {
  try {
    const {jobs}=await api("/api/jobs");
    const expanded=new Set([...$("jobs").querySelectorAll("details[open]")].map(item=>item.dataset.id));
    $("jobs").replaceChildren();$("job-count").textContent=jobs.length;
    if(!jobs.length)$("jobs").append(node("div","전송 대기열이 비어 있습니다.","empty"));
    for(const job of jobs) {
      const card=node("article",undefined,"job");const heading=node("div",undefined,"job-header");
      heading.append(node("div",job.path,"job-title"),node("span",statuses[job.status]||job.status,"badge "+job.status));
      card.append(heading,node("div",job.source_name+" → "+job.destination_name,"job-route"));
      const bottom=node("div",undefined,"job-bottom");const progress=node("progress");progress.max=100;progress.value=job.progress;
      progress.setAttribute("aria-label",job.path+" 전송 진행률");bottom.append(progress,node("span",job.progress+"%","hint"));
      if(["queued","running","cancelling"].includes(job.status)) {
        const cancel=button("취소",async()=>{cancel.disabled=true;try{await api("/api/cancel",{id:job.id});await pollJobs();}catch(issue){error(issue.message);}},"button quiet");cancel.disabled=job.status==="cancelling";bottom.append(cancel);
      } else if(["failed","cancelled"].includes(job.status)) {
        bottom.append(button("다시 담기",()=>{if(state.selected.size&&!confirm("현재 선택을 비우고 이 항목을 다시 담을까요?"))return;
          state.source=job.source_id;state.destination=job.destination_id;state.selected=new Set([job.path]);renderLocations();renderSelection();browse(job.path.split("/").slice(0,-1).join("/"));}));
      }
      card.append(bottom,node("div",job.detail,"job-detail"));
      const details=node("details");details.dataset.id=job.id;details.open=expanded.has(job.id);details.append(node("summary","상세 로그"),node("pre",job.logs.join("\n")));card.append(details);$("jobs").append(card);
    }
  } catch(issue) {error("전송 서버와 연결이 끊겼습니다: "+issue.message);}
}
async function initialize() {
  try {
    state.config=await api("/api/config");const endpoints=state.config.endpoints;
    for(const id of ["source","destination"])for(const item of endpoints){const option=node("option",item.name);option.value=item.id;$(id).append(option);}
    state.source=endpoints[0].id;state.destination=endpoints[endpoints.length-1].id;
    $("mode").textContent=state.config.mode;$("concurrency").textContent=`동시 전송 ${state.config.concurrency}개 · 로컬 복사 ${state.config.streams} streams`;
    renderLocations();renderSelection();await browse();await pollJobs();setInterval(pollJobs,2500);
  } catch(issue) {error(issue.message);}
}
$("source").addEventListener("change",()=>{if(state.selected.size&&!confirm("출발 위치를 바꾸면 기존 선택이 비워집니다. 계속할까요?")){$("source").value=state.source;return;}
  state.source=$("source").value;state.selected.clear();renderLocations();renderSelection();browse("");});
$("destination").addEventListener("change",()=>{state.destination=$("destination").value;renderLocations();renderSelection();});
$("swap").addEventListener("click",()=>{if(state.selected.size&&!confirm("출발·도착을 바꾸면 기존 선택이 비워집니다. 계속할까요?"))return;
  [state.source,state.destination]=[state.destination,state.source];state.selected.clear();renderLocations();renderSelection();browse("");});
$("refresh").addEventListener("click",()=>browse(state.path,state.offset,true));
$("add-current").addEventListener("click",()=>{if(state.path){state.selected.add(state.path);renderSelection();renderFiles();}});
$("clear").addEventListener("click",()=>{state.selected.clear();renderSelection();renderFiles();});
$("search").addEventListener("input",()=>{clearTimeout(state.searchTimer);state.searchTimer=setTimeout(()=>browse(state.path,0,true),500);});
$("previous").addEventListener("click",()=>browse(state.path,Math.max(0,state.offset-200),true));
$("next").addEventListener("click",()=>browse(state.path,state.offset+200,true));
$("select-visible").addEventListener("change",()=>{if(!state.listing)return;const select=$("select-visible").checked;
  for(const item of state.listing.entries){const target=selectionPath(item);if(!["file","directory"].includes(item.kind)||covered(target))continue;select?state.selected.add(target):state.selected.delete(target);}renderSelection();renderFiles();});
$("prepare").addEventListener("click",prepare);$("start").addEventListener("click",start);$("dismiss").addEventListener("click",()=>$("confirmation").close());
initialize();
