'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const states = {PENDING:'대기 중',RUNNING:'실행 중',LAUNCHING:'제출 중',ACCOUNTING:'종료 상태 확인 중',FINALIZING:'마무리 중',COMPLETED:'실행 완료',FAILED:'실패',DRY_RUN:'설정 확인'};
  const transfers = {TRANSFERRING:'결과 회수 중',COLLECTED:'결과 회수 완료',RETRY:'결과 회수 재시도 대기'};
  let token='',data={jobs:[]},busy=false;
  function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
  const pct=n=>Number.isFinite(n)?n.toFixed(1)+'%':'—';
  function render(){
    const opened=new Set([...document.querySelectorAll('details[open][data-key]')].map(n=>n.dataset.key));
    const jobs=data.jobs||[];
    $('active').textContent=jobs.filter(j=>!['COMPLETED','FAILED','DRY_RUN'].includes(j.state)).length;
    $('finished').textContent=jobs.filter(j=>j.state==='COMPLETED'&&j.evaluation).length;
    $('collected').textContent=jobs.filter(j=>j.collection?.state==='COLLECTED').length;
    $('failed').textContent=jobs.filter(j=>j.state==='FAILED').length;
    const needle=$('filter').value.toLowerCase();
    const list=jobs.filter(j=>JSON.stringify([j.evaluation?.output,j.script,j.job_ids,j.id]).toLowerCase().includes(needle));
    const fragment=document.createDocumentFragment();
    for(const job of list){
      const card=el('article',undefined,'card job'),head=el('div',undefined,'job-head');
      const name=job.evaluation?.output?.split('/').pop()||job.script.split('/').slice(-2).join('/');
      head.append(el('h3',name),el('span',states[job.state]||job.state||'연결 대기','badge '+(job.state||'')));card.append(head);
      card.append(el('p',new Date(job.created_at*1000).toLocaleString()+' · Slurm '+((job.job_ids||[]).join(', ')||'제출 대기'),'meta'));
      const p=job.progress;
      if(p?.total){
        const line=el('div',undefined,'progress-line');line.append(el('span',`완료 ${p.completed} / ${p.total} 에피소드 (${pct(p.percent)})`),el('strong','현재 성공률 '+pct(p.success_pct)));card.append(line);
        const bar=el('progress');bar.max=p.total;bar.value=p.completed;bar.setAttribute('aria-label','평가 진행률');card.append(bar);
        const details=el('details');details.dataset.key=job.id+'-tasks';details.open=opened.has(details.dataset.key);details.append(el('summary','모델·task별 성공률'));
        for(const model of p.models||[]){
          const panel=el('section',undefined,'model');panel.append(el('h4',model.label+' · '+pct(model.success_pct)));
          const grid=el('div',undefined,'task-grid');
          for(const task of model.tasks||[]){const row=el('div',undefined,'task');row.append(el('span',`Task ${task.task_id} · ${task.completed}/${task.total}`),el('strong',pct(task.success_pct)));grid.append(row);}
          panel.append(grid);details.append(panel);
        }card.append(details);
      }else if(job.evaluation){card.append(el('p',job.state==='COMPLETED'?'리포트 생성 완료':'평가 준비 중 · 에피소드 결과가 저장되면 진행률이 표시됩니다.'));}
      const c=job.collection||{};
      if(c.state){card.append(el('div',transfers[c.state]||c.state,'result-state'));
        if(c.path){const link=el('a','이 서버에서 결과 열기');link.href='/yonsei/results/'+encodeURIComponent(job.id)+'/';link.target='_blank';link.rel='noopener';card.append(link,el('p',c.path,'meta'));}
        if(c.error)card.append(el('p',c.error,'error'));
      }else if(job.evaluation&&job.state==='COMPLETED'){card.append(el('p','결과 회수 대기','result-state'));}
      if(job.connection_error)card.append(el('p','연결 확인 실패 · 마지막 확인 결과입니다: '+job.connection_error,'error'));
      if(job.error)card.append(el('p',job.error,'error'));
      for(const j of job.jobs||[]){if(!['COMPLETED','RUNNING','PENDING'].includes(j.state))card.append(el('p',`Slurm ${j.id}: ${j.state} (${j.exit_code})`,'error'));}
      fragment.append(card);
    }
    if(!list.length)fragment.append(el('p','아직 Yonsei 실행 기록이 없습니다. 평가 폴더에서 ./submit_eval.sh --yonsei 로 시작하세요.','empty'));
    $('jobs').replaceChildren(fragment);
  }
  async function update(){
    if(busy)return;busy=true;
    try{const r=await fetch('/yonsei/api/state',{cache:'no-store'});const value=await r.json();if(!r.ok)throw Error(value.error||r.status);data=value;token=value.token;
      $('updated').textContent=value.updated_at?'확인 '+new Date(value.updated_at*1000).toLocaleTimeString():'첫 확인 중…';
      const errors=(value.errors||[]).map(e=>e.host+': '+e.error);if(value.stale)errors.push('갱신이 지연되고 있습니다.');$('notice').hidden=!errors.length;$('notice').textContent=errors.join('\n');render();
    }catch(e){$('notice').hidden=false;$('notice').textContent='연결 실패: '+e.message;}finally{busy=false;}
  }
  $('refresh').addEventListener('click',async()=>{if(!token)return;try{const r=await fetch('/yonsei/api/refresh',{method:'POST',headers:{'X-Sync-Token':token}});if(!r.ok)throw Error(r.status);await update();}catch(e){$('notice').hidden=false;$('notice').textContent=e.message;}});
  $('filter').addEventListener('input',render);update();setInterval(update,5000);
})();
