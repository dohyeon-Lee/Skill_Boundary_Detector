/* Shared RunPod-style metrics for the RunPod, Slurm and Yonsei desks. */
'use strict';
(() => {
  const el = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  const pct = value => Number.isFinite(value) ? `${value.toFixed(1)}%` : '—';
  const count = value => Number.isFinite(value) ? value.toLocaleString() : '—';
  const storage = 'sbd.desk.metrics.disclosures.v1';
  let saved;
  try { saved = new Map(Object.entries(JSON.parse(localStorage.getItem(storage) || '{}'))); }
  catch { saved = new Map(); }
  function fold(key, label, cls = '', open = false) {
    const node = el('details', null, `desk-metrics disclosure ${cls}`);
    const summary = el('summary'), heading = el('div', null, 'fold-title'), body = el('div', null, 'fold-body');
    if (typeof label === 'string') heading.textContent = label;
    else heading.append(label);
    summary.append(heading); node.append(summary, body);
    node.dataset.disclosureKey = key;
    node.open = saved.has(key) ? saved.get(key) : open;
    node.addEventListener('toggle', () => {
      if (!node.isConnected) return;
      saved.set(key, node.open);
      try { localStorage.setItem(storage, JSON.stringify(Object.fromEntries(saved))); } catch {}
    });
    return {node, summary, body};
  }
  function capture() {
    for (const node of document.querySelectorAll('.desk-metrics[data-disclosure-key]')) {
      saved.set(node.dataset.disclosureKey, node.open);
    }
  }
  function successRow(name, value, overall = false) {
    const row = el('div', null, `desk-metrics task-row${overall ? ' overall-row' : ''}`), bar = el('progress');
    bar.max = 100; bar.value = Number.isFinite(value) ? value : 0;
    bar.setAttribute('aria-label', `${name} 성공률${Number.isFinite(value) ? '' : ' 미확인'}`);
    if (!Number.isFinite(value)) bar.className = 'unknown';
    row.append(el('span', name), bar, el('span', pct(value), 'success-value'));
    return row;
  }
  function progressRow(name, progress, training = false) {
    const row = el('div', null, 'desk-metrics progress-row'), bar = el('progress');
    bar.max = 100;
    if (Number.isFinite(progress.percent)) bar.value = progress.percent;
    bar.setAttribute('aria-label', `${name} ${training ? '학습' : '평가'} 진행률`);
    const done = training ? progress.completed_steps : progress.completed_episodes;
    const total = training ? progress.total_steps : progress.total_episodes;
    row.append(el('strong', name), el('span', `${progress.approximate ? '약 ' : ''}${count(done)} / ${count(total)}${training ? ' ' + (progress.unit || 'step') : '회'} · ${pct(progress.percent)}`, 'progress-count'), bar);
    return row;
  }
  function normalizeEval(value) {
    const counters = item => {
      const done = item.completed_episodes ?? item.completed ?? null, total = item.total_episodes ?? item.total ?? null;
      return {...item, completed_episodes:done, total_episodes:total,
        percent:item.percent ?? (Number.isFinite(done) && total > 0 ? 100 * done / total : null)};
    };
    return {...counters(value), models:(value.models || []).map(model => {
      const tasks = (model.tasks || []).map(task => ({...counters(task), name:task.name ?? `Task ${task.task_id}`}));
      return {...counters(model), tasks, total_tasks:model.total_tasks ?? tasks.length,
        completed_tasks:model.completed_tasks ?? tasks.filter(t => t.total_episodes > 0 && t.completed_episodes >= t.total_episodes).length,
        observed_episodes:model.observed_episodes ?? model.completed_episodes ?? model.completed ?? null};
    })};
  }
  function normalizeTraining(value) {
    const runs = (Array.isArray(value) ? value : value.runs || []).map(item => {
      const done = item.completed_steps ?? item.step ?? null, total = item.total_steps ?? item.total ?? null;
      return {...item, label:item.label || String(item.output || '').split('/').pop() || '학습',
        completed_steps:done, total_steps:total,
        percent:item.percent ?? (Number.isFinite(done) && total > 0 ? 100 * done / total : null)};
    });
    const known = runs.length && runs.every(r => Number.isFinite(r.completed_steps) && r.total_steps > 0);
    const sameUnit = new Set(runs.map(r => r.unit || 'step')).size <= 1;
    const done = known && sameUnit ? runs.reduce((n, r) => n + r.completed_steps, 0) : null;
    const total = known && sameUnit ? runs.reduce((n, r) => n + r.total_steps, 0) : null;
    return {runs, completed_steps:done, total_steps:total, unit:runs[0]?.unit || 'step',
      percent:total ? 100 * done / total : null, updated_at:Math.max(value.updated_at || 0, ...runs.map(r => r.updated_at || 0))};
  }
  function renderProgress(value, key, makeFold = fold) {
    const progress = normalizeEval(value), final = progress.phase === 'final';
    const view = makeFold(`${key}:progress`, final ? '평가 결과' : '평가 진행 상황', 'desk-metrics eval-progress', true);
    view.body.append(progressRow('전체 진행률', progress));
    view.body.append(el('p', final ? '완료된 전체 평가 결과입니다.' : '중간 성공률은 결과가 확인된 완료 에피소드 기준이며 최종 결과와 달라질 수 있습니다.', 'meta'));
    if (progress.incomplete) view.body.append(el('p', '로그 범위를 넘어 일부 진행률을 확인할 수 없습니다.', 'inline-error'));
    if (progress.at_least) view.body.append(el('p', '완료된 태스크 기준 집계입니다. 진행 중인 태스크는 포함되지 않을 수 있습니다.', 'meta'));
    for (const model of progress.models) {
      const group = makeFold(`${key}:progress:model:${model.label}`, model.label, 'desk-metrics model-progress', true);
      group.body.append(progressRow('평가 진행률', model));
      group.body.append(el('p', `태스크 ${count(model.completed_tasks)} / ${count(model.total_tasks)} 완료 · 성공 ${count(model.successes)} / 집계 ${count(model.observed_episodes)}회`, 'meta'));
      const list = el('div', null, 'task-list');
      list.append(successRow(final ? 'Overall · 전체' : 'Overall · 중간', model.success_pct, true));
      for (const task of model.tasks) list.append(successRow(`${task.name} · ${count(task.completed_episodes)}/${count(task.total_episodes)}회`, task.success_pct));
      group.body.append(list); view.body.append(group.node);
    }
    if (progress.updated_at) view.body.append(el('p', `로그 갱신 ${new Date(progress.updated_at * 1000).toLocaleString()}`, 'meta'));
    return view.node;
  }
  function renderTrainingProgress(value, key, makeFold = fold) {
    const progress = normalizeTraining(value), view = makeFold(`${key}:training-progress`, '학습 진행 상황', 'desk-metrics train-progress', true);
    if (progress.runs.length > 1) view.body.append(progressRow('전체 진행률', progress, true));
    for (const run of progress.runs) {
      const group = makeFold(`${key}:training-progress:run:${run.label}`, run.label, 'desk-metrics model-progress', true);
      group.body.append(progressRow('학습 진행률', run, true));
      group.body.append(el('p', `최근 저장 체크포인트: ${Number.isFinite(run.latest_checkpoint) ? count(run.latest_checkpoint) + ' ' + (run.unit || 'step') : '아직 확인되지 않음'}`, 'meta training-checkpoint'));
      if (run.completed_steps == null) group.body.append(el('p', '현재 step을 확인할 수 없습니다. 로그 갱신을 기다립니다.', 'meta'));
      view.body.append(group.node);
    }
    if (progress.updated_at) view.body.append(el('p', `로그 갱신 ${new Date(progress.updated_at * 1000).toLocaleString()}`, 'meta'));
    return view.node;
  }
  globalThis.DeskMetrics = {el, pct, fold, capture, successRow, progressRow, normalizeEval, normalizeTraining, renderProgress, renderTrainingProgress};
})();
