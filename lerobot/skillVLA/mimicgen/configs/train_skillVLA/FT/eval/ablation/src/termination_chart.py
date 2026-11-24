"""Dependency-free, video-synchronized plot of the actual per-step Term signal."""
import html
import json
import math


def chart_html(result, video_id):
    if not result.get('termination_graph', False):
        return ''
    segments = []
    for record in result.get('skill_trace', []):
        points = [[int(point['episode_timestep']), float(point['prob'])]
                  for point in record.get('end_probs', []) if math.isfinite(float(point['prob']))]
        if points:
            segments.append(dict(skill=int(record['source_skill_index'])+1, points=points))
    data = dict(steps=result['steps'], fps=result.get('sim_fps', 20),
                threshold=result['termination_threshold'], segments=segments,
                transitions=[dict(step=t['step'], skill=t['to_skill_number']) for t in result['transitions']],
                disabled_from=result.get('transition_step'))
    encoded = html.escape(json.dumps(data, separators=(',', ':')), quote=True)
    return (f'<div class="termination-panel"><p>Termination probability · every simulation step '
            f'({data["fps"]} Hz) · threshold {data["threshold"]:g}</p>'
            f'<canvas class="termination-chart" data-video="{video_id}" data-signal="{encoded}" '
            'aria-label="Per-step termination probabilities synchronized with the video"></canvas>'
            '<p class="muted">Click the graph to seek. Gray region: final skill, Terminator disabled.</p></div>')


CHART_SCRIPT = r'''
<script>
for (const canvas of document.querySelectorAll('.termination-chart')) {
  const video = document.getElementById(canvas.dataset.video);
  const data = JSON.parse(canvas.dataset.signal);
  const ctx = canvas.getContext('2d');
  const colors = ['#70d6b4', '#74b9ff', '#f8c76d'];
  let animation = 0;
  function draw() {
    const width = Math.max(300, canvas.getBoundingClientRect().width), height = 220;
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width*ratio) || canvas.height !== Math.round(height*ratio)) {
      canvas.width = Math.round(width*ratio); canvas.height = Math.round(height*ratio);
    }
    ctx.setTransform(ratio,0,0,ratio,0,0);
    ctx.clearRect(0,0,width,height);
    const left=42, right=16, top=28, bottom=35, plotWidth=width-left-right, plotHeight=height-top-bottom;
    const x = step => left + step/Math.max(1,data.steps)*plotWidth;
    const y = value => top + (1-value)*plotHeight;
    ctx.fillStyle='#171e28'; ctx.fillRect(left,top,plotWidth,plotHeight);
    if (data.disabled_from !== null) {
      ctx.fillStyle='#ffffff0c';
      ctx.fillRect(x(data.disabled_from),top,x(data.steps)-x(data.disabled_from),plotHeight);
    }
    ctx.font='12px sans-serif'; ctx.textAlign='right';
    for (const value of [0,.25,.5,.75,1]) {
      ctx.strokeStyle='#ffffff18'; ctx.beginPath();ctx.moveTo(left,y(value));ctx.lineTo(width-right,y(value));ctx.stroke();
      ctx.fillStyle='#b6c2d5';ctx.fillText(value.toFixed(2),left-7,y(value)+4);
    }
    ctx.textAlign='center';
    for (let i=0;i<=4;i++) {
      const step=data.steps*i/4;
      ctx.fillStyle='#b6c2d5';ctx.fillText(`${(step/data.fps).toFixed(1)}s`,x(step),height-12);
    }
    ctx.strokeStyle='#ff8a8a';ctx.setLineDash([5,4]);ctx.beginPath();
    ctx.moveTo(left,y(data.threshold));ctx.lineTo(width-right,y(data.threshold));ctx.stroke();ctx.setLineDash([]);
    for (let index=0;index<data.segments.length;index++) {
      const segment=data.segments[index];ctx.strokeStyle=colors[index%colors.length];ctx.lineWidth=1.6;
      ctx.beginPath();
      segment.points.forEach(([step,value],i)=>{
        if (!i || step-segment.points[i-1][0]>1) ctx.moveTo(x(step),y(value)); else ctx.lineTo(x(step),y(value));
      });ctx.stroke();
      ctx.fillStyle=ctx.strokeStyle;ctx.textAlign='left';
      ctx.fillText(`Skill ${segment.skill}`,left+index*90,17);
    }
    for (const transition of data.transitions) {
      ctx.strokeStyle='#f8c76d';ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();
      ctx.moveTo(x(transition.step),top);ctx.lineTo(x(transition.step),height-bottom);ctx.stroke();ctx.setLineDash([]);
    }
    const step=Math.min(data.steps,Math.max(0,video.currentTime*data.fps));
    ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x(step),top);ctx.lineTo(x(step),height-bottom);ctx.stroke();
    let probability=null, skill=null;
    for (const segment of data.segments) {
      const point=segment.points.find(p=>p[0]===Math.floor(step));
      if (point) { probability=point[1];skill=segment.skill; }
    }
    ctx.fillStyle='#e8edf5';ctx.textAlign='right';
    const status=probability===null?'Terminator off':`skill ${skill} · p=${probability.toFixed(3)}`;
    ctx.fillText(`step ${Math.floor(step)} · ${status}`,width-right,17);
  }
  function tick(){draw();animation=(!video.paused&&!video.ended)?requestAnimationFrame(tick):0;}
  video.addEventListener('play',()=>{if(!animation)tick();});
  for (const event of ['loadedmetadata','timeupdate','seeking','seeked','pause','ended']) video.addEventListener(event,draw);
  canvas.addEventListener('click',event=>{
    const rect=canvas.getBoundingClientRect();
    const fraction=Math.max(0,Math.min(1,(event.clientX-rect.left-42)/(rect.width-58)));
    if (Number.isFinite(video.duration)) video.currentTime=Math.min(video.duration,fraction*data.steps/data.fps);
    draw();
  });
  new ResizeObserver(draw).observe(canvas);
  draw();
}
</script>
'''
