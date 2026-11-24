"""Continuous executed gripper commands and a video-synchronized HTML viewer."""

from __future__ import annotations

import html

import numpy as np


def gripper_signal(actions, *, sim_fps: float, video_fps: float, frame_stride: int,
                   chunks: list[dict] | None = None) -> dict:
    """Keep the postprocessed action before simulator clipping/sign conversion.

    Coffee's PandaGripper uses sign(action[6]): negative=open, positive=close,
    zero=hold. These values are signed action commands, not probabilities.
    """
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError('Gripper visualization requires finite action[T, 7]')
    if sim_fps <= 0 or video_fps <= 0 or frame_stride < 1:
        raise ValueError('Invalid gripper video timing')
    predicted = []
    for index, chunk in enumerate(chunks or []):
        start = int(chunk['start_step'])
        future = np.asarray(chunk['values'], dtype=float)
        if (future.ndim != 1 or not len(future) or not np.isfinite(future).all()
                or start < 0 or start >= len(values)
                or (predicted and start <= predicted[-1]['start_step'])):
            raise ValueError('Malformed or unordered gripper prediction chunk')
        stop = int(chunks[index + 1]['start_step']) if index + 1 < len(chunks) else len(values)
        used = min(len(future), stop - start, len(values) - start)
        predicted.append({'start_step': start, 'values': future.tolist(),
                          'executed_steps': used,
                          'executed_values': values[start:start + used, 6].astype(float).tolist()})
    return {
        'schema_version': 1,
        'source': 'postprocessed_action_before_simulator_clipping',
        'action_dimension': 6,
        'threshold': 0.0,
        'semantics': 'negative=open; positive=close; zero=hold (PandaGripper sign conversion)',
        'sim_fps': float(sim_fps),
        'video_frames_per_second': float(video_fps),
        'sim_steps_per_video_frame': int(frame_stride),
        'values': values[:, 6].astype(float).tolist(),
        'chunks': predicted,
    }


def gripper_player_html(video_url: str) -> str:
    return (
        '<div class="gripper-signal-player" style="max-width:1100px;margin:12px 0">'
        f'<video controls playsinline preload="metadata" src="{html.escape(video_url, quote=True)}" '
        'style="width:100%;max-height:65vh;display:block;background:#111"></video>'
        '<div style="padding:10px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px">'
        '<b>Gripper command — continuous output</b>'
        '<div style="font-size:12px;margin:4px 0">−1 open · +1 close · actual threshold: 0 '
        '(not 0.5). Click the graph to seek the video.</div>'
        '<div class="gripper-readout" style="font-family:monospace"></div>'
        '<svg class="gripper-chart" viewBox="0 0 1000 180" '
        'style="width:100%;height:180px;display:block;cursor:crosshair" '
        'role="img" aria-label="Continuous gripper action by simulator frame"></svg>'
        '<div class="gripper-chunks" style="margin-top:12px">'
        '<b>Predicted action chunk · gripper</b> '
        '<label><input class="gripper-follow" type="checkbox" checked> Follow video</label> '
        '<select class="gripper-chunk-selector" aria-label="Select replanning frame"></select>'
        '<div class="gripper-chunk-readout" style="font-family:monospace;margin-top:6px"></div>'
        '<svg class="gripper-chunk-chart" viewBox="0 0 1000 180" '
        'style="width:100%;height:180px;display:block" role="img" aria-label="Full predicted gripper chunk and executed prefix"></svg>'
        '<div style="font-size:12px">Orange: full predicted chunk · Blue: executed commands · '
        'Shaded: executed prefix. The remaining predictions were replaced by replanning.</div>'
        '</div></div></div>'
    )


GRIPPER_PLAYER_SCRIPT = r"""
function attachGripperSignal(root, signal) {
  if (!root || !signal || !signal.values.length) return;
  const video=root.querySelector('video'), svg=root.querySelector('svg');
  const readout=root.querySelector('.gripper-readout'), values=signal.values;
  const NS='http://www.w3.org/2000/svg', left=52, right=984, top=12, bottom=148;
  const last=Math.max(1,values.length-1);
  const lo=Math.min(-1,...values), hi=Math.max(1,...values);
  const x=f=>left+f/last*(right-left), y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);
  const add=(tag,attrs,text)=>{
    const el=document.createElementNS(NS,tag);
    Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));
    if(text!==undefined)el.textContent=text;
    svg.appendChild(el);return el;
  };
  for(const value of [-1,0,1]) {
    add('line',{x1:left,x2:right,y1:y(value),y2:y(value),stroke:value===0?'#475569':'#dbe3ec',
                'stroke-dasharray':value===0?'6 4':'2 4'});
    add('text',{x:left-8,y:y(value)+4,'text-anchor':'end','font-size':12,fill:'#334155'},value.toFixed(1));
  }
  for(let i=0;i<=4;i++) {
    const f=Math.round((values.length-1)*i/4);
    add('text',{x:x(f),y:169,'text-anchor':'middle','font-size':12,fill:'#334155'},String(f));
  }
  add('text',{x:right,y:179,'text-anchor':'end','font-size':10,fill:'#64748b'},'simulator frame');
  add('path',{d:values.map((v,i)=>(i?'L':'M')+x(i).toFixed(2)+','+y(v).toFixed(2)).join(' '),
              fill:'none',stroke:'#2563eb','stroke-width':2,'vector-effect':'non-scaling-stroke'});
  const cursor=add('line',{x1:left,x2:left,y1:top,y2:bottom,stroke:'#dc2626','stroke-width':1.5});
  const dot=add('circle',{cx:left,cy:y(values[0]),r:4,fill:'#dc2626'});
  const rate=signal.video_frames_per_second*signal.sim_steps_per_video_frame;
  const chunks=signal.chunks||[], chunkBox=root.querySelector('.gripper-chunks');
  const selector=root.querySelector('.gripper-chunk-selector'), follow=root.querySelector('.gripper-follow');
  const chunkSvg=root.querySelector('.gripper-chunk-chart'), chunkReadout=root.querySelector('.gripper-chunk-readout');
  chunkBox.hidden=!chunks.length;
  let shown=-1, chunkCursor=null, chunkX=null;
  for(let i=0;i<chunks.length;i++) {
    const option=document.createElement('option');option.value=i;
    option.textContent=`replan ${chunks[i].start_step}`;selector.appendChild(option);
  }
  function drawChunk(index) {
    shown=index;selector.value=String(index);chunkSvg.innerHTML='';
    const chunk=chunks[index], n=Math.max(1,chunk.values.length-1);
    const low=Math.min(-1,...chunk.values,...chunk.executed_values);
    const high=Math.max(1,...chunk.values,...chunk.executed_values);
    const xx=f=>left+f/n*(right-left), yy=v=>bottom-(v-low)/(high-low)*(bottom-top);
    chunkX=xx;
    const make=(tag,attrs,text)=>{
      const el=document.createElementNS(NS,tag);
      Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));
      if(text!==undefined)el.textContent=text;chunkSvg.appendChild(el);return el;
    };
    make('rect',{x:left,y:top,width:xx(Math.min(n,chunk.executed_steps)) - left,
                 height:bottom-top,fill:'#dbeafe'});
    for(const value of [-1,0,1]) {
      make('line',{x1:left,x2:right,y1:yy(value),y2:yy(value),stroke:value===0?'#475569':'#cbd5e1',
                   'stroke-dasharray':value===0?'6 4':'2 4'});
      make('text',{x:left-8,y:yy(value)+4,'text-anchor':'end','font-size':12,fill:'#334155'},value.toFixed(1));
    }
    for(const [series,color] of [[chunk.values,'#ea580c'],[chunk.executed_values,'#2563eb']]) {
      make('path',{d:series.map((v,i)=>(i?'L':'M')+xx(i).toFixed(2)+','+yy(v).toFixed(2)).join(' '),
                   fill:'none',stroke:color,'stroke-width':2});
      series.forEach((v,i)=>make('circle',{cx:xx(i),cy:yy(v),r:2.5,fill:color}));
    }
    for(let i=0;i<=4;i++) {
      const offset=Math.round((chunk.values.length-1)*i/4);
      make('text',{x:xx(offset),y:169,'text-anchor':'middle','font-size':12,fill:'#334155'},String(chunk.start_step+offset));
    }
    chunkCursor=make('line',{x1:left,x2:left,y1:top,y2:bottom,stroke:'#dc2626','stroke-width':1.5});
  }
  function updateChunk(frame) {
    if(!chunks.length)return;
    let index=Math.max(0,Number(selector.value)||0);
    if(follow.checked)for(let i=0;i<chunks.length;i++) {
      if(chunks[i].start_step<=frame)index=i;else break;
    }
    if(index!==shown)drawChunk(index);
    const chunk=chunks[index], offset=frame-chunk.start_step;
    chunkCursor.style.display=offset>=0&&offset<chunk.values.length?'':'none';
    chunkCursor.setAttribute('x1',chunkX(offset));chunkCursor.setAttribute('x2',chunkX(offset));
    const current=offset>=0&&offset<chunk.values.length?` · prediction here ${chunk.values[offset].toFixed(4)}`:'';
    chunkReadout.textContent=`replan frame ${chunk.start_step} · horizon ${chunk.values.length} · executed ${chunk.executed_steps} step(s)${current}`;
  }
  function update() {
    const frame=Math.max(0,Math.min(values.length-1,Math.floor(video.currentTime*rate+1e-6)));
    const value=values[frame];
    cursor.setAttribute('x1',x(frame));cursor.setAttribute('x2',x(frame));
    dot.setAttribute('cx',x(frame));dot.setAttribute('cy',y(value));
    readout.textContent=`frame ${frame} · simulation ${(frame/signal.sim_fps).toFixed(2)} s · command ${value.toFixed(4)} · ${value>0?'CLOSE':value<0?'OPEN':'HOLD'}`;
    updateChunk(frame);
  }
  selector.addEventListener('change',()=>{follow.checked=false;update();});
  follow.addEventListener('change',update);
  let animation=null;
  function animate(){update();if(!video.paused&&!video.ended)animation=requestAnimationFrame(animate);}
  video.addEventListener('play',()=>{if(animation!==null)cancelAnimationFrame(animation);animate();});
  for(const event of ['timeupdate','seeked','loadedmetadata','pause','ended'])video.addEventListener(event,update);
  svg.addEventListener('click',event=>{
    const point=svg.createSVGPoint();point.x=event.clientX;point.y=event.clientY;
    const local=point.matrixTransform(svg.getScreenCTM().inverse());
    const frame=Math.max(0,Math.min(values.length-1,(local.x-left)/(right-left)*last));
    video.currentTime=frame/rate;update();
  });
  update();
}
"""
