"use strict";
let token = "", state = null, busy = false, configured = false, podRenderKey = "";
const stopping = new Set();
const disclosureStorage = "sbd.runpod.disclosures.v1";
const disclosures = (() => {
  try {
    return new Map(Object.entries(JSON.parse(localStorage.getItem(disclosureStorage) || "{}"))
      .filter(([, value]) => typeof value === "boolean"));
  } catch { return new Map(); }
})();
const $ = id => document.getElementById(id);
const money = value => value == null ? "—" : new Intl.NumberFormat("en-US", {style:"currency", currency:"USD", minimumFractionDigits:4}).format(value);
const pct = value => Number.isFinite(value) ? `${value.toFixed(1)}%` : "—";
const when = value => value ? new Date(value).toLocaleString() : "—";
function duration(value) {
  if (!Number.isFinite(value) || value < 0) return "—";
  const total = Math.floor(value), hours = Math.floor(total / 3600), minutes = Math.floor(total % 3600 / 60);
  return [hours ? `${hours}시간` : "", hours || minutes ? `${minutes}분` : "", `${total % 60}초`].filter(Boolean).join(" ");
}
function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function notice(id, text) { $(id).textContent = text || ""; $(id).hidden = !text; }
function rememberDisclosure(node) { disclosures.set(node.dataset.disclosureKey, node.open); }
function saveDisclosures() {
  try { localStorage.setItem(disclosureStorage, JSON.stringify(Object.fromEntries(disclosures))); } catch {}
}
function bindDisclosure(node, key, open) {
  node.dataset.disclosureKey = key;
  node.open = disclosures.has(key) ? disclosures.get(key) : open;
  node.addEventListener("toggle", () => {
    if (!node.isConnected) return;
    rememberDisclosure(node);
    saveDisclosures();
  });
}
function fold(key, label, cls, open = false) {
  const node = el("details", null, `disclosure ${cls || ""}`);
  const summary = el("summary"), heading = el("div", null, "fold-title"), body = el("div", null, "fold-body");
  if (typeof label === "string") heading.textContent = label;
  else heading.append(label);
  summary.append(heading);
  node.append(summary, body);
  bindDisclosure(node, key, open);
  return {node, summary, body};
}
function setAllDisclosures(open) {
  for (const node of document.querySelectorAll("main details[data-disclosure-key]")) {
    node.open = open;
    rememberDisclosure(node);
  }
  saveDisclosures();
}
async function api(path, body) {
  const response = await fetch(`/runpod/api/${path}`, body === undefined ? {cache:"no-store"} : {
    method:"POST", headers:{"Content-Type":"application/json", "X-Sync-Token":token}, body:JSON.stringify(body)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
const labels = {RUNNING:"실행 중", EXITED:"중단됨", STOPPED:"중단됨", TERMINATED:"종료됨", DELETED:"삭제됨", STOP_REQUESTED:"중단 확인 중", STARTING:"시작 중", PROVISIONING:"준비 중", running:"실행 중", pending:"대기 중", done:"완료", failed:"실패", stopped:"중단됨", pod_stopped:"Pod 종료", cancelled:"취소", final:"최종 결과", partial:"부분 결과", unverified:"완료 상태 미확인"};
function badge(value) { return el("span", labels[value] || value, `badge ${String(value).toLowerCase()}`); }
function successRow(name, value, overall = false) {
  const row = el("div", null, `task-row${overall ? " overall-row" : ""}`), bar = el("progress");
  bar.max = 100;
  bar.value = Number.isFinite(value) ? value : 0;
  bar.setAttribute("aria-label", `${name} 성공률${Number.isFinite(value) ? "" : " 미확인"}`);
  if (!Number.isFinite(value)) bar.className = "unknown";
  row.append(el("span", name), bar, el("span", pct(value), "success-value"));
  return row;
}
function renderResult(result, jobKey) {
  const label = el("div");
  label.append(el("h4", result.kind === "skill_boundary_detection" ? "스킬 경계 평가 결과" : "평가 결과"), el("div", result.name, "meta"));
  const view = fold(`${jobKey}:result:${result.key}`, label, "eval-result", true);
  view.summary.append(badge(result.phase));
  view.body.append(el("p", `수집 ${when(result.collected_at)}`, "meta"));
  if (result.kind === "skill_boundary_detection") {
    for (const [name, statistics] of Object.entries(result.thresholds || {})) {
      const condition = fold(`${jobKey}:result:${result.key}:condition:${name}`, name, "model-result", true);
      for (const [group, values] of Object.entries(statistics.groups || {})) {
        condition.body.append(el("p", `${group} · threshold ${values.threshold} · calibration ${values.episodes}개 에피소드 / ${values.points}개 유효 점수`));
      }
      condition.body.append(el("p", `보고서 대상 ${statistics.calibration?.evaluation_episodes ?? "—"}개 에피소드`, "meta"));
      view.body.append(condition.node);
    }
    view.body.append(el("p", "HTML 비교 보고서와 상세 지표는 결과 폴더에 저장됩니다.", "meta"));
  }
  for (const model of result.models || []) {
    const group = fold(`${jobKey}:result:${result.key}:model:${model.label}`, model.label, "model-result", true);
    group.summary.append(el("span", `${model.task_count} tasks`, "meta"));
    const list = el("div", null, "task-list");
    list.append(successRow("Overall · 전체", model.overall_pct, true));
    for (const task of model.tasks || []) list.append(successRow(task.name, task.success_pct));
    group.body.append(list);
    view.body.append(group.node);
  }
  const location = fold(`${jobKey}:result:${result.key}:location`, "결과 경로", "result-location");
  location.body.append(el("p", result.output, "meta"));
  view.body.append(location.node);
  return view.node;
}
function progressRow(name, progress, training = false) {
  const row = el("div", null, "progress-row"), bar = el("progress");
  bar.max = 100;
  if (Number.isFinite(progress.percent)) bar.value = progress.percent;
  bar.setAttribute("aria-label", `${name} ${training ? "학습" : "평가"} 진행률`);
  const count = value => Number.isFinite(value) ? value.toLocaleString() : "—";
  const done = training ? count(progress.completed_steps) : progress.completed_episodes;
  const total = training ? count(progress.total_steps) : progress.total_episodes;
  row.append(el("strong", name), el("span", `${done} / ${total}${training ? " step" : "회"} · ${pct(progress.percent)}`, "progress-count"), bar);
  return row;
}
function renderProgress(progress, key) {
  const view = fold(`${key}:progress`, "평가 진행 상황", "eval-progress", true);
  view.body.append(progressRow("전체 진행률", progress));
  view.body.append(el("p", "중간 성공률은 결과가 확인된 완료 에피소드 기준이며 최종 결과와 달라질 수 있습니다.", "meta"));
  if (progress.incomplete) view.body.append(el("p", "로그 범위를 넘어 일부 진행률을 확인할 수 없습니다.", "inline-error"));
  for (const model of progress.models || []) {
    const group = fold(`${key}:progress:model:${model.label}`, model.label, "model-progress", true);
    group.body.append(progressRow("평가 진행률", model));
    group.body.append(el("p", `태스크 ${model.completed_tasks} / ${model.total_tasks} 완료 · 성공 ${model.successes} / 집계 ${model.observed_episodes}회`, "meta"));
    const list = el("div", null, "task-list");
    list.append(successRow("Overall · 중간", model.success_pct, true));
    for (const task of model.tasks || []) {
      list.append(successRow(`${task.name} · ${task.completed_episodes}/${task.total_episodes}회`, task.success_pct));
    }
    group.body.append(list); view.body.append(group.node);
  }
  if (progress.updated_at) view.body.append(el("p", `로그 갱신 ${when(progress.updated_at * 1000)}`, "meta"));
  return view.node;
}
function renderTrainingProgress(progress, key) {
  const view = fold(`${key}:training-progress`, "학습 진행 상황", "train-progress", true);
  if (progress.runs.length > 1) view.body.append(progressRow("전체 진행률", progress, true));
  for (const run of progress.runs) {
    const group = fold(`${key}:training-progress:run:${run.label}`, run.label, "model-progress", true);
    group.body.append(progressRow("학습 진행률", run, true));
    group.body.append(el("p", `최근 저장 체크포인트: ${Number.isFinite(run.latest_checkpoint) ? `${run.latest_checkpoint.toLocaleString()} step` : "아직 확인되지 않음"}`, "meta training-checkpoint"));
    if (run.completed_steps == null) group.body.append(el("p", "현재 step을 확인할 수 없습니다. 로그 갱신을 기다립니다.", "meta"));
    view.body.append(group.node);
  }
  if (progress.updated_at) view.body.append(el("p", `로그 갱신 ${when(progress.updated_at * 1000)}`, "meta"));
  return view.node;
}
function renderJob(pod, job, results, archived = false) {
  const key = `pod:${pod.id}:job:${archived ? "archive:" : ""}${job.id}`;
  const title = el("div");
  title.append(el("h4", job.name), el("div", archived ? "잡 기록 없이 수집된 평가" : `잡 ${job.id}`, "meta"));
  const view = fold(key, title, "job-card");
  if (job.progress) view.summary.append(el("span", `진행 ${pct(job.progress.percent)}`, "badge partial job-progress-badge"));
  if (job.training_progress) view.summary.append(el("span", `학습 ${pct(job.training_progress.percent)}`, "badge partial job-training-badge"));
  if (Number.isFinite(job.elapsed_seconds)) view.summary.append(el("span", `소요 ${duration(job.elapsed_seconds)}`, "badge job-duration"));
  view.summary.append(badge(job.state));
  if (!archived) view.body.append(el("p", `GPU ${job.gpus || "—"} · 시작 ${when(job.started)}${job.finished ? ` · 종료 ${when(job.finished)}` : ""}`, "meta"));
  if (job.progress && !results.some(result => result.phase === "final")) view.body.append(renderProgress(job.progress, key));
  if (job.training_progress) view.body.append(renderTrainingProgress(job.training_progress, key));
  for (const result of results) view.body.append(renderResult(result, key));
  if (!job.progress && !results.length && (job.eval_output || String(job.script || "").includes("/eval/"))) {
    view.body.append(el("p", "아직 수집된 평가 결과가 없습니다.", "muted"));
  }
  return view.node;
}
function podLastUsed(pod) {
  const timestamp = value => Date.parse(value) || 0;
  return Math.max(timestamp(pod.created_at) || timestamp(pod.first_seen),
    ...(pod.jobs || []).map(job => timestamp(job.started)));
}
function renderPods(query) {
  const key = JSON.stringify([state.pods, state.results, query, [...stopping]]);
  if (key === podRenderKey) return;
  podRenderKey = key;
  // Capture native details changes even if their batched toggle event has not fired.
  for (const node of $("pods").querySelectorAll("details[data-disclosure-key]")) rememberDisclosure(node);
  const pods = [...state.pods], results = state.results || [];
  for (const result of results) {
    if (!pods.some(pod => pod.id === result.pod_id)) pods.push({id:result.pod_id, name:result.pod_name || result.pod_id,
      status:"DELETED", present:false, jobs:[], hourly_cost:0});
  }
  pods.sort((a, b) => podLastUsed(b) - podLastUsed(a));
  const fragment = document.createDocumentFragment();
  for (const pod of pods) {
    const podResults = results.filter(result => result.pod_id === pod.id).sort((a, b) => b.mtime - a.mtime);
    const searchable = [pod.name, pod.id, ...(pod.jobs || []).flatMap(job => [job.name, job.id, ...(job.progress?.models || []).map(model => model.label)]),
      ...podResults.flatMap(result => [result.name, ...(result.models || []).map(model => model.label)])].join(" ");
    if (!searchable.toLowerCase().includes(query)) continue;
    const title = el("div");
    title.append(el("h3", pod.name), el("div", `${pod.id} · ${pod.gpu_name || "GPU 정보 없음"} × ${pod.gpu_count || 0}`, "meta"));
    const card = fold(`pod:${pod.id}`, title, "card pod-card", pod.status === "RUNNING");
    const controls = el("div", null, "card-actions"), stop = el("button", stopping.has(pod.id) ? "중단 요청 중…" : "Pod 중단", "danger");
    stop.type = "button";
    stop.disabled = pod.status !== "RUNNING" || !pod.present || pod.stale || stopping.has(pod.id);
    stop.setAttribute("aria-label", `${pod.name} Pod 중단`);
    stop.addEventListener("click", event => { event.preventDefault(); event.stopPropagation(); stopPod(pod); });
    if (Number.isFinite(pod.elapsed_seconds)) {
      const elapsed = el("span", `전체 경과 ${duration(pod.elapsed_seconds)}`, "badge pod-duration");
      elapsed.title = "Pod 생성부터 현재 또는 중단 확인까지. 설치·평가·백업 시간을 포함하며 과금 시간과는 다릅니다.";
      controls.append(elapsed);
    }
    controls.append(badge(pod.status), stop);
    card.summary.append(controls);
    const costs = fold(`pod:${pod.id}:costs`, "비용", "pod-section", true), cells = el("div", null, "costs");
    for (const [label, value] of [["현재 요금 / 시간", money(pod.hourly_cost)],
      [pod.billing?.lifetime ? "누적 집계 비용" : "최근 30일 집계 비용", money(pod.billing?.totalAmount)],
      [["RUNNING", "STOP_REQUESTED"].includes(pod.status) ? "현재 실행분 추정" : "마지막 실행분 추정", money(pod.session_estimate)]]) {
      const item = el("div", label, "muted");
      item.append(el("strong", value));
      cells.append(item);
    }
    costs.body.append(cells);
    if (pod.billing) costs.body.append(el("p", `집계: ${when(pod.billing.updated_at)} · GPU ${money(pod.billing.gpuAmount)} / CPU ${money(pod.billing.cpuAmount)} / 디스크 ${money(pod.billing.diskAmount)}. 반영 지연 및 별도 네트워크 볼륨 요금은 제외될 수 있습니다.`, "meta"));
    costs.body.append(el("p", "중단하면 GPU 작업이 멈춥니다. 저장소 요금은 남을 수 있으며 컨테이너 파일은 사라질 수 있습니다.", "meta"));
    card.body.append(costs.node);
    if (pod.gpus?.length && ["RUNNING", "STOP_REQUESTED"].includes(pod.status)) {
      const gpuView = fold(`pod:${pod.id}:gpus`, `GPU 상태 (${pod.gpus.length})`, "pod-section", true), list = el("div", null, "gpu-list");
      for (const gpu of pod.gpus) {
        const cell = el("div", `GPU ${gpu.index} · ${gpu.name}: ${(gpu.memory_used_mib / 1024).toFixed(1)} / ${(gpu.memory_total_mib / 1024).toFixed(1)} GiB · 사용률 ${gpu.util_pct}%`, "gpu"), bar = el("progress");
        bar.max = gpu.memory_total_mib; bar.value = gpu.memory_used_mib;
        bar.setAttribute("aria-label", `GPU ${gpu.index} 메모리`);
        cell.append(bar); list.append(cell);
      }
      gpuView.body.append(list); card.body.append(gpuView.node);
    }
    const jobs = fold(`pod:${pod.id}:jobs`, "잡 목록", "pod-section", true), shown = new Set();
    for (const job of pod.jobs || []) {
      const matched = podResults.filter(result => result.key === job.result_key ||
        (result.job_ids || []).map(String).includes(String(job.id)) || (job.eval_output && result.output === job.eval_output));
      for (const result of matched) shown.add(result.key);
      jobs.body.append(renderJob(pod, job, matched));
    }
    // Keep archived evaluations visible under their Pod even without a saved job record.
    for (const result of podResults.filter(result => !shown.has(result.key))) {
      jobs.body.append(renderJob(pod, {id:result.remote_id, name:result.name, state:result.phase}, [result], true));
    }
    if (!jobs.body.childNodes.length) jobs.body.append(el("p", pod.ssh_error ? "작업을 조회하지 못했습니다." : "등록된 작업이 없습니다.", "muted"));
    card.body.append(jobs.node);
    const warnings = [pod.ssh_error, pod.billing_error, ...(pod.warnings || []), pod.stale ? "조회가 지연되어 이전 기록을 표시합니다." : ""].filter(Boolean);
    if (warnings.length) {
      const alerts = fold(`pod:${pod.id}:warnings`, `알림 (${warnings.length})`, "pod-section", true);
      for (const text of warnings) alerts.body.append(el("p", text, "inline-error"));
      card.body.append(alerts.node);
    }
    fragment.append(card.node);
  }
  $("empty-pods").hidden = fragment.childNodes.length > 0;
  $("pods").replaceChildren(fragment);
}
function render() {
  $("running-count").textContent = state.pods.filter(pod => pod.status === "RUNNING").length;
  $("job-count").textContent = state.pods.filter(pod => pod.status === "RUNNING").flatMap(pod => pod.jobs || []).filter(job => job.state === "running").length;
  $("result-count").textContent = (state.results || []).length;
  renderPods($("search").value.trim().toLowerCase());
}
async function load() {
  if (busy) return;
  busy = true;
  try {
    state = await api("state"); token = state.token;
    notice("error", state.error);
    $("updated").textContent = `마지막 확인 ${when(state.updated_at)}`;
    if (!configured) {
      for (const [key, value] of Object.entries(state.config)) $("settings-form").elements[key].value = value;
      configured = true;
    }
    render();
  } catch (error) { notice("error", error.message); }
  finally { busy = false; }
}
async function stopPod(pod) {
  if (!confirm(`${pod.name} (${pod.id})를 중단할까요?\n실행 중인 모든 작업이 멈추며 컨테이너 파일이 사라질 수 있습니다.\n마지막으로 결과 수집을 시도한 뒤 중단합니다.\n저장소 요금은 계속 발생할 수 있습니다.`)) return;
  stopping.add(pod.id); render();
  try {
    const result = await api("stop", {pod_id:pod.id, confirm:pod.id});
    notice("message", [result.message, result.warning].filter(Boolean).join("\n"));
    await load();
  } catch (error) { notice("error", error.message); }
  finally { stopping.delete(pod.id); render(); }
}
$("refresh").addEventListener("click", async () => {
  try { await api("refresh", {}); notice("message", "새 조회를 요청했습니다."); setTimeout(load, 1000); }
  catch (error) { notice("error", error.message); }
});
$("settings-form").addEventListener("submit", async event => {
  event.preventDefault(); const form = event.currentTarget.elements;
  try {
    await api("settings", {project_root:form.project_root.value, identity:form.identity.value, interval:Number(form.interval.value)});
    notice("message", "조회 설정을 저장했습니다.");
  } catch (error) { notice("error", error.message); }
});
$("search").addEventListener("input", () => { if (state) render(); });
$("expand-all").addEventListener("click", () => setAllDisclosures(true));
$("collapse-all").addEventListener("click", () => setAllDisclosures(false));
for (const [id, key, open] of [["overview", "overview", true], ["settings", "settings", false], ["pod-list", "pods", true], ["help", "help", false]]) {
  bindDisclosure($(id), key, open);
}
const storagePanel = document.querySelector(".storage-panel");
if (storagePanel) {
  const storage = fold("storage", "서버 용량", "storage-toggle");
  storagePanel.replaceWith(storage.node);
  storage.body.append(storagePanel);
}
load();
setInterval(load, 5000);
