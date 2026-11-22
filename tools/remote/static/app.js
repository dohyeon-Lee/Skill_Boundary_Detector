'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const states = {PENDING:'대기 중',RUNNING:'실행 중',LAUNCHING:'제출 중',ACCOUNTING:'종료 상태 확인 중',FINALIZING:'마무리 중',COMPLETED:'실행 완료',FAILED:'실패',CANCELLED:'취소됨',DRY_RUN:'설정 확인'};
  const transfers = {TRANSFERRING:'결과 회수 중',COLLECTED:'결과 회수 완료',RETRY:'결과 회수 재시도 대기'};
  let token='',data={jobs:[]},busy=false;
  const canceling=new Set();
  function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
  const pct=n=>Number.isFinite(n)?n.toFixed(1)+'%':'—';
  function render(){
    DeskMetrics.capture();
    const jobs=data.jobs||[];
    $('active').textContent=jobs.filter(j=>!['COMPLETED','FAILED','CANCELLED','DRY_RUN'].includes(j.state)).length;
    $('finished').textContent=jobs.filter(j=>j.state==='COMPLETED'&&j.evaluation).length;
    $('collected').textContent=jobs.filter(j=>j.collection?.state==='COLLECTED').length;
    $('failed').textContent=jobs.filter(j=>j.state==='FAILED').length;
    const needle=$('filter').value.toLowerCase();
    const list=jobs.filter(j=>JSON.stringify([j.evaluation?.output,j.script,j.job_ids,j.id]).toLowerCase().includes(needle));
    const fragment=document.createDocumentFragment();
    for(const job of list){
      const name=job.evaluation?.output?.split('/').pop()||job.training_progress?.find(p=>p.output)?.output.split('/').pop()||job.script.split('/').slice(-2).join('/');
      const title=el('div');title.append(el('h4',name),el('div','Slurm '+((job.job_ids||[]).join(', ')||'제출 대기'),'meta'));
      const view=DeskMetrics.fold('yonsei:'+job.id,title,'card job-card yonsei-job');
      const card=view.body,head=el('div',undefined,'card-actions');
      if(job.progress)head.append(el('span','진행 '+pct(job.progress.percent),'badge partial job-progress-badge'));
      if(job.training_progress?.length)head.append(el('span','학습 '+pct(DeskMetrics.normalizeTraining(job.training_progress).percent),'badge partial job-training-badge'));
      head.append(el('span',states[job.state]||job.state||'연결 대기','badge '+String(job.state||'').toLowerCase()));
      view.summary.append(head);
      if(job.can_cancel){
        const button=el('button',job.cancel?.status==='requested'?'취소 요청 중':'이 실행 취소','cancel danger');
        button.disabled=canceling.has(job.id)||job.cancel?.status==='requested';
        button.addEventListener('click',async()=>{
          if(!confirm(`${name}\nYonsei Slurm ${(job.job_ids||[]).join(', ')} 잡을 취소할까요? 다른 실행은 유지됩니다.`))return;
          canceling.add(job.id);button.disabled=true;
          try{
            const r=await fetch('/yonsei/api/cancel',{method:'POST',headers:{'Content-Type':'application/json','X-Sync-Token':token},body:JSON.stringify({id:job.id})});
            const value=await r.json();if(!r.ok)throw Error(value.error||r.status);
            job.cancel=value;button.textContent='취소 요청 중';await update();
          }catch(error){$('notice').hidden=false;$('notice').textContent='취소 요청 확인 필요: '+error.message;}
          finally{canceling.delete(job.id);button.disabled=job.cancel?.status==='requested';}
        });card.append(button);
      }
      if(job.cancel?.error)card.append(el('p',job.cancel.error,'error'));
      card.append(el('p',new Date(job.created_at*1000).toLocaleString()+' · Slurm '+((job.job_ids||[]).join(', ')||'제출 대기'),'meta'));
      const p=job.progress;
      if(job.training?.checkpoint_sync){
        const sync=job.checkpoint_sync||{},completed=Object.values(sync.completed||{});
        const selection=job.training.protected?.length?'보호 스텝만: '+job.training.protected.join(', '):'새 체크포인트 전체';
        card.append(el('p',`체크포인트 자동 전송 · ${selection} · 완료 ${completed.length}개`,'result-state'));
        if(sync.state==='TRANSFERRING')card.append(el('p',`${sync.current_step} 체크포인트 전송 중…`));
        if(sync.error)card.append(el('p','전송 재시도 대기: '+sync.error,'error'));
        for(const checkpoint of completed)card.append(el('p',`${checkpoint.step}: ${checkpoint.path}`,'meta'));
      }
      if(job.training_progress?.length)card.append(DeskMetrics.renderTrainingProgress(job.training_progress,'yonsei:'+job.id));
      else if(job.training)card.append(DeskMetrics.renderTrainingProgress([{step:null,total:job.training.total,unit:job.training.unit}],'yonsei:'+job.id));
      if(p?.total){
        card.append(DeskMetrics.renderProgress({...p,phase:job.state==='COMPLETED'?'final':'partial'},'yonsei:'+job.id));
      }else if(job.evaluation){card.append(el('p',job.state==='COMPLETED'?'리포트 생성 완료':'평가 준비 중 · 에피소드 결과가 저장되면 진행률이 표시됩니다.'));}
      const c=job.collection||{};
      if(c.state){card.append(el('div',transfers[c.state]||c.state,'result-state'));
        if(c.path){const link=el('a','이 서버에서 결과 열기');link.href='/yonsei/results/'+encodeURIComponent(job.id)+'/';link.target='_blank';link.rel='noopener';card.append(link,el('p',c.path,'meta'));}
        if(c.error)card.append(el('p',c.error,'error'));
      }else if(job.evaluation&&job.state==='COMPLETED'){card.append(el('p','결과 회수 대기','result-state'));}
      if(job.connection_error)card.append(el('p','연결 확인 실패 · 마지막 확인 결과입니다: '+job.connection_error,'error'));
      if(job.error)card.append(el('p',job.error,'error'));
      for(const j of job.jobs||[]){if(!['COMPLETED','RUNNING','PENDING'].includes(j.state))card.append(el('p',`Slurm ${j.id}: ${j.state} (${j.exit_code})`,'error'));}
      fragment.append(view.node);
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
