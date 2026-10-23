"use strict";
const $ = id => document.getElementById(id);
const state = {config:null, operation:"dataset-pull", source:null, destination:null, path:"", offset:0,
  listing:null, selected:new Set(), browseVersion:0, prepared:null, pending:false, searchTimer:null, completedDeletions:new Set()};
const statuses = {queued:"대기", running:"실행 중", completed:"완료", failed:"실패", cancelled:"취소", cancelling:"취소 중"};
function node(tag, text, className) {
  const element=document.createElement(tag);if(text!==undefined)element.textContent=text;
  if(className)element.className=className;return element;
}
function button(text, handler, className="button quiet") {
  const element=node("button",text,className);element.type="button";element.addEventListener("click",handler);return element;
}
function error(message) {$("error").textContent=message||"";$("error").classList.toggle("hidden",!message);}
async function api(path, body) {
  const options=body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json","X-Sync-Token":state.config.token},body:JSON.stringify(body)};
  const response=await fetch(path,options),payload=await response.json();
  if(!response.ok)throw Error(payload.error||"요청에 실패했습니다.");return payload;
}
function endpoint(id) {return state.config.endpoints.find(item=>item.id===id);}
function route() {return state.config.routes.find(item=>item.id===state.operation);}
function deleting() {return state.operation.endsWith("-delete");}
function formatSize(bytes) {
  if(bytes===null||bytes===undefined)return "폴더";
  const units=["B","KiB","MiB","GiB","TiB"];let i=0;
  while(bytes>=1024&&i<units.length-1){bytes/=1024;i++;}return bytes.toFixed(i?1:0)+" "+units[i];
}
function covered(path) {return [...state.selected].some(parent=>path!==parent&&path.startsWith(parent+"/"));}
function renderSelection() {
  const count=state.selected.size;$("selected-count").textContent=count;$("selected-list").replaceChildren();
  $("prepare").disabled=!count||state.pending;$("prepare-watch").disabled=state.pending;
  $("prepare").textContent=count?`${count}개 항목 ${deleting()?"삭제":"전송"} 준비`:"항목을 선택하세요";
  $("prepare").classList.toggle("danger",deleting());
  if(!count)$("selected-list").append(node("div","필요한 파일·폴더를 골라 담으세요.","empty"));
  for(const path of [...state.selected].sort()) {
    const item=node("div",undefined,"selected-item");item.append(node("code",path));
    const remove=button("×",()=>{state.selected.delete(path);renderSelection();renderFiles();},"remove");
    remove.setAttribute("aria-label",path+" 선택 해제");item.append(remove);$("selected-list").append(item);
  }
}
function breadcrumbs() {
  $("breadcrumbs").replaceChildren(button("루트",()=>browse("")));let path="";
  for(const part of state.path.split("/").filter(Boolean)) {
    path=path?path+"/"+part:part;const target=path;
    $("breadcrumbs").append(node("span","/","crumb-divider"),button(part,()=>browse(target)));
  }
  $("add-current").disabled=!state.path;
}
function renderFiles() {
  if(!state.listing)return;
  const list=$("file-list");list.replaceChildren();const {entries,total,offset,limit,has_more}=state.listing;
  const eligible=entries.filter(item=>["directory","file"].includes(item.kind));
  const picked=eligible.filter(item=>state.selected.has(item.path)||covered(item.path)).length;
  $("select-visible").disabled=!eligible.length;$("select-visible").checked=!!eligible.length&&picked===eligible.length;
  $("select-visible").indeterminate=picked>0&&picked<eligible.length;
  if(!entries.length)list.append(node("div","표시할 항목이 없습니다.","empty"));
  for(const item of entries) {
    const selected=state.selected.has(item.path)||covered(item.path);
    const row=node("div",undefined,"file-row"+(selected?" selected":""));
    const check=node("input");check.type="checkbox";check.checked=selected;
    check.disabled=covered(item.path)||!["directory","file"].includes(item.kind);check.setAttribute("aria-label",item.path+" 선택");
    check.addEventListener("change",()=>{check.checked?state.selected.add(item.path):state.selected.delete(item.path);renderSelection();renderFiles();});
    const name=item.kind==="directory"?button(item.name,()=>browse(item.path),"file-name folder"):node("span",item.name,"file-name");
    name.title=item.description||item.path;
    row.append(check,node("span",item.kind==="directory"?"▱":"▤","file-icon"),name,
      node("span",item.description?"모델":formatSize(item.size),"file-meta"));list.append(row);
  }
  $("item-count").textContent=has_more?`${offset+entries.length}개 이상`:`${total}개 항목`;
  $("page-info").textContent=entries.length?`${offset+1}–${offset+entries.length}`:"0개";
  $("previous").disabled=offset===0;$("next").disabled=!has_more;
}
async function browse(path=state.path,offset=0,keepSearch=false) {
  const version=++state.browseVersion;state.path=path;state.offset=offset;state.listing=null;
  if(!keepSearch)$("search").value="";breadcrumbs();$("file-list").replaceChildren(node("div","목록을 불러오는 중…","empty"));
  $("browser-notice").classList.add("hidden");
  $("select-visible").disabled=true;$("next").disabled=true;$("previous").disabled=true;
  try {
    let attempt=path;
    while(version===state.browseVersion) {
      const params=new URLSearchParams({endpoint:state.source,path:attempt,offset:String(offset),search:$("search").value});
      let listing;
      try {listing=await api("/api/browse?"+params);}catch(issue) {
        if(version!==state.browseVersion)return;
        // Also handle a still-running older backend after the static UI is
        // refreshed. Only missing entries trigger fallback, never auth failures.
        if(state.source.startsWith("hub-")&&attempt&&/Entry Not Found|does not exist on /i.test(issue.message)) {
          attempt=attempt.split("/").slice(0,-1).join("/");offset=0;$("search").value="";continue;
        }
        throw issue;
      }
      if(version!==state.browseVersion)return;
      state.path=listing.path;state.offset=listing.offset;state.listing=listing;
      if(state.path!==path) {
        const missing=path.split("/").slice(0,(state.path?state.path.split("/").length:0)+1).join("/");
        for(const selected of state.selected)if(selected===missing||selected.startsWith(missing+"/"))state.selected.delete(selected);
        $("search").value="";
        $("browser-notice").textContent=`요청한 폴더가 없어 ${state.path||"저장소 루트"}로 이동했습니다.`;
        $("browser-notice").classList.toggle("hidden",false);renderSelection();
      }
      breadcrumbs();error("");renderFiles();return;
    }
  } catch(issue) {if(version===state.browseVersion){$("file-list").replaceChildren(node("div","목록을 불러오지 못했습니다.","empty"));error(issue.message);}}
}
async function changeOperation(operation) {
  state.operation=operation;$("operation").value=operation;state.source=route().source;state.destination=route().destination;
  state.selected.clear();state.prepared=null;state.browseVersion++;clearTimeout(state.searchTimer);
  for(const side of ["source","destination"]) {
    const item=endpoint(state[side]);$(side+"-name").textContent=item.name;$(side+"-path").textContent=item.base;
  }
  $("destination-name").textContent=deleting()?"선택한 원격 파일 삭제":endpoint(state.destination).name;
  $("route-arrow").textContent=deleting()?"×":"→";
  $("selection-title").textContent=deleting()?"삭제할 항목":"전송할 항목";
  $("selection-hint").textContent=deleting()?"Hub에서 삭제할 파일 목록·개수·용량을 먼저 확인합니다.":"전송 전에 서버 경로와 Hub 목적지를 확인할 수 있습니다.";
  $("browse-hint").textContent=deleting()?"파일·폴더를 선택하면 그 안의 원격 파일을 삭제합니다. 로컬 파일은 유지됩니다.":"폴더 이름은 열기, 체크박스는 선택입니다. 다른 폴더로 이동해도 선택은 유지됩니다.";
  const watch=operation==="watch";$("watch-panel").classList.toggle("hidden",!watch);$("browse-panel").classList.toggle("hidden",watch);
  error("");renderSelection();if(!watch)await browse("");
}
function requestForSelection() {
  const request={source:state.source,destination:state.destination,paths:[...state.selected],dry_run:$("dry-run").checked};
  if(state.operation==="watch") {
    request.paths=[$("watch-root").value.trim()||"outputs*"];
    request.options={root:$("watch-root").value.trim(), interval:Number($("watch-interval").value),keep:Number($("watch-keep").value),
      protect:$("watch-protect").value.trim(),once:$("watch-once").checked,prune_local:$("watch-prune").checked,squash:$("watch-squash").checked};
  }
  return request;
}
async function prepare() {
  if(state.pending||(!state.selected.size&&state.operation!=="watch"))return;
  state.pending=true;renderSelection();error("");const request=requestForSelection();
  try {
    const plan=await api("/api/plan",request),isDelete=request.source===request.destination;
    if(plan.plan_id)request.plan_id=plan.plan_id;
    state.prepared={request,plan,deleting:isDelete};
    $("confirmation-title").textContent=isDelete?"Hugging Face에서 삭제할까요?":"이 작업을 시작할까요?";
    $("confirmation-route").textContent=endpoint(request.source).name+(isDelete?" · 원격 파일 삭제":" → "+endpoint(request.destination).name);
    $("confirmation-list").replaceChildren();
    for(const item of plan.items) {
      const element=node("div",undefined,"confirmation-item");element.append(node("strong",item.path),
        node("code",item.source),node("code","→ "+item.destination));$("confirmation-list").append(element);
      if(isDelete) {
        element.replaceChildren(node("strong",item.repo),node("p",`${item.file_count}개 파일 · ${formatSize(item.size)}`,"delete-summary"));
        for(const path of item.paths)element.append(node("code",path,"delete-path"));
        const details=node("details");details.append(node("summary",`삭제할 파일 보기 (최대 20개 표시)`));
        const files=node("div",undefined,"delete-samples");
        for(const file of item.sample_files)files.append(node("code",file,"delete-path"));
        details.append(files);element.append(details);
      }
    }
    if(state.operation==="watch") {
      const options=request.options;
      $("confirmation-list").append(node("p",`확인 간격 ${options.interval}초 · ${options.once?"한 번만 확인":"계속 감시"} · 보호 step ${options.protect||"없음"}`,"hint"));
    }
    $("collapsed").textContent=`${plan.items.length}개 작업`+(plan.collapsed?` · 중복 ${plan.collapsed}개 제외`:"");
    $("notice").textContent=plan.notice;$("confirm-delete").checked=false;
    $("delete-confirmation").classList.toggle("hidden",!isDelete||request.dry_run);
    $("start").disabled=isDelete&&!request.dry_run;$("start").textContent=request.dry_run?"미리보기 실행":isDelete?"Hub에서 삭제":"시작";
    $("start").classList.toggle("danger",isDelete&&!request.dry_run);
    $("confirmation-error").classList.add("hidden");
    $("confirmation").showModal();
  } catch(issue) {error(issue.message);}finally{state.pending=false;renderSelection();}
}
async function start() {
  if(!state.prepared||$("start").disabled)return;$("start").disabled=true;
  try {
    await api("/api/transfers",{...state.prepared.request,confirm_delete:$("confirm-delete").checked});$("confirmation").close();state.prepared=null;
    state.selected.clear();renderSelection();renderFiles();await pollJobs();$("jobs").scrollIntoView({behavior:"smooth",block:"nearest"});
  } catch(issue){error(issue.message);$("confirmation-error").textContent=issue.message;$("confirmation-error").classList.toggle("hidden",false);$("start").disabled=false;}
}
async function pollJobs() {
  try {
    const {jobs}=await api("/api/jobs");let refreshPath=null;
    const expanded=new Set([...$("jobs").querySelectorAll("details[open]")].map(item=>item.dataset.id));
    $("jobs").replaceChildren();$("job-count").textContent=jobs.length;
    if(!jobs.length)$("jobs").append(node("div","작업 대기열이 비어 있습니다.","empty"));
    for(const job of jobs) {
      if(job.route.endsWith("-delete")&&job.status==="completed"&&!job.dry_run&&!state.completedDeletions.has(job.id)) {
        state.completedDeletions.add(job.id);
        if(state.source===job.source_id) {
          refreshPath=refreshPath??state.path;
          for(const path of job.paths) {
            if(refreshPath===path||refreshPath.startsWith(path+"/"))refreshPath=path.split("/").slice(0,-1).join("/");
            for(const selected of state.selected)if(selected===path||selected.startsWith(path+"/"))state.selected.delete(selected);
          }
        }
      }
      const card=node("article",undefined,"job"),heading=node("div",undefined,"job-header");
      heading.append(node("div",(job.dry_run?"[미리보기] ":"")+job.path,"job-title"),node("span",statuses[job.status]||job.status,"badge "+job.status));
      card.append(heading,node("div",job.source_name+" → "+job.destination_name,"job-route"));
      const bottom=node("div",undefined,"job-bottom"),progress=node("progress");progress.max=100;
      if(!["running","queued","cancelling"].includes(job.status))progress.value=job.status==="completed"?100:0;
      progress.setAttribute("aria-label",job.path+" 작업 상태");bottom.append(progress);
      if(["queued","running","cancelling"].includes(job.status)) {
        const cancel=button("취소",async()=>{cancel.disabled=true;try{await api("/api/cancel",{id:job.id});await pollJobs();}catch(issue){error(issue.message);}});
        cancel.disabled=job.status==="cancelling";bottom.append(cancel);
      } else if(["failed","cancelled"].includes(job.status)) {
        bottom.append(button("다시 준비",async()=>{
          if(state.selected.size&&!confirm("현재 선택을 비우고 이 항목을 다시 담을까요?"))return;
          await changeOperation(job.route);$("dry-run").checked=job.dry_run;
          if(job.route==="watch") {
            for(const key of ["root","interval","keep","protect"])$("watch-"+key).value=job.options[key];
            $("watch-once").checked=job.options.once;$("watch-prune").checked=job.options.prune_local;$("watch-squash").checked=job.options.squash;
          } else {for(const path of job.paths||[job.path])state.selected.add(path);renderSelection();renderFiles();}
        }));
      }
      card.append(bottom,node("div",job.detail,"job-detail"));const details=node("details");details.dataset.id=job.id;details.open=expanded.has(job.id);
      details.append(node("summary","상세 로그"),node("pre",job.logs.join("\n")));card.append(details);$("jobs").append(card);
    }
    if(refreshPath!==null){renderSelection();await browse(refreshPath);}
  }catch(issue){error("전송 서버와 연결이 끊겼습니다: "+issue.message);}
}
async function initialize() {
  try {
    state.config=await api("/api/config");
    for(const item of state.config.routes){const option=node("option",item.name);option.value=item.id;$("operation").append(option);}
    $("mode").textContent=state.config.mode;$("auth").textContent=state.config.authenticated?"● 서버 HF 토큰 설정됨":"서버에서 hf auth login 필요";
    $("concurrency").textContent=`동시 작업 ${state.config.concurrency}개`;
    await changeOperation(state.operation);await pollJobs();setInterval(pollJobs,2500);
  }catch(issue){error(issue.message);}
}
$("operation").addEventListener("change",async()=>{
  if(state.selected.size&&!confirm("작업을 바꾸면 기존 선택이 비워집니다. 계속할까요?")){$("operation").value=state.operation;return;}
  await changeOperation($("operation").value);
});
$("refresh").addEventListener("click",()=>browse(state.path,state.offset,true));
$("add-current").addEventListener("click",()=>{if(state.path){state.selected.add(state.path);renderSelection();renderFiles();}});
$("clear").addEventListener("click",()=>{state.selected.clear();renderSelection();renderFiles();});
$("search").addEventListener("input",()=>{clearTimeout(state.searchTimer);state.searchTimer=setTimeout(()=>browse(state.path,0,true),400);});
$("previous").addEventListener("click",()=>browse(state.path,Math.max(0,state.offset-200),true));
$("next").addEventListener("click",()=>browse(state.path,state.offset+200,true));
$("select-visible").addEventListener("change",()=>{
  if(!state.listing)return;const select=$("select-visible").checked;
  for(const item of state.listing.entries){if(!["file","directory"].includes(item.kind)||covered(item.path))continue;select?state.selected.add(item.path):state.selected.delete(item.path);}
  renderSelection();renderFiles();
});
$("prepare").addEventListener("click",prepare);$("prepare-watch").addEventListener("click",prepare);
$("start").addEventListener("click",start);$("dismiss").addEventListener("click",()=>$("confirmation").close());
$("confirm-delete").addEventListener("change",()=>{if(state.prepared?.deleting&&!state.prepared.request.dry_run)$("start").disabled=!$("confirm-delete").checked;});
initialize();
