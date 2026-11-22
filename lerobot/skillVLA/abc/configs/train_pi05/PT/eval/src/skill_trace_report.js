// The trace is embedded in the report so local files work without a server.
for (const article of document.querySelectorAll('article')) {
  const source = article.querySelector('.skill-trace-data');
  const video = article.querySelector('video');
  if (!source || !video) continue;
  const trace = JSON.parse(source.textContent);
  const panel = article.querySelector('.termination-panel');
  const canvas = panel.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const timeline = panel.querySelector('.skill-timeline');
  let active = trace.skills[0], animation = null;
  const toTime = frame => frame / trace.fps / trace.stride;
  const seek = frame => { video.currentTime = Math.min(video.duration || Infinity, toTime(frame)); draw(); };
  const buttons = trace.skills.map(skill => {
    const button = document.createElement('button');
    button.style.background = skill.color;
    button.style.flex = String(Math.max(1, skill.end - skill.start));
    button.textContent = String(skill.code);
    button.title = `Skill #${skill.order + 1}, ID ${skill.code} · frames ${skill.start}–${skill.end} · ${skill.reason}`;
    button.setAttribute('aria-label', button.title);
    button.addEventListener('click', () => seek(skill.start + (skill.order ? 1 : 0)));
    timeline.append(button);
    return button;
  });
  function draw() {
    if (panel.hidden) return;
    const frame = Math.min(trace.steps, Math.max(0, video.currentTime * trace.fps * trace.stride));
    active = trace.skills[0];
    for (const skill of trace.skills.slice(1)) {
      if (frame <= skill.start + 1e-6) break;
      active = skill;
    }
    const localFrame = Math.max(0, frame - active.start);
    let probability = null;
    for (const [step, p] of active.points) {
      if (step > frame + 1e-6) break;
      probability = p;
    }
    const label = panel.querySelector('.active-skill');
    label.textContent = `Skill #${active.order + 1} · ID ${active.code}`;
    label.style.color = active.color;
    panel.querySelector('.termination-value').textContent =
      `P(term) ${probability === null ? '—' : probability.toFixed(3)} · threshold ${trace.threshold} · ${active.reason}`;
    buttons.forEach((button, i) => button.setAttribute('aria-current', String(trace.skills[i] === active)));
    const width = Math.max(1, canvas.getBoundingClientRect().width), height = 190;
    const scale = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * scale) || canvas.height !== Math.round(height * scale)) {
      canvas.width = Math.round(width * scale); canvas.height = Math.round(height * scale);
    }
    ctx.setTransform(scale, 0, 0, scale, 0, 0); ctx.clearRect(0, 0, width, height);
    const left = 35, right = width - 12, top = 13, bottom = height - 30;
    const length = Math.max(1, active.end - active.start);
    const x = step => left + (step - active.start) / length * (right - left);
    const y = p => bottom - Math.max(0, Math.min(1, p)) * (bottom - top);
    ctx.font = '11px system-ui'; ctx.lineWidth = 1;
    for (const p of [0, .5, 1]) {
      ctx.strokeStyle = '#344154'; ctx.beginPath(); ctx.moveTo(left, y(p)); ctx.lineTo(right, y(p)); ctx.stroke();
      ctx.fillStyle = '#a3afbf'; ctx.fillText(String(p), 8, y(p) + 4);
    }
    ctx.strokeStyle = '#ffb86b'; ctx.setLineDash([5, 4]); ctx.beginPath();
    ctx.moveTo(left, y(trace.threshold)); ctx.lineTo(right, y(trace.threshold)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = active.color; ctx.lineWidth = 2; ctx.beginPath();
    active.points.forEach(([step, p], i) => i ? ctx.lineTo(x(step), y(p)) : ctx.moveTo(x(step), y(p)));
    ctx.stroke();
    const cursor = Math.max(left, Math.min(right, x(frame)));
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1; ctx.beginPath();
    ctx.moveTo(cursor, top); ctx.lineTo(cursor, bottom); ctx.stroke();
    if (probability !== null) {
      ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(cursor, y(probability), 3, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.fillStyle = '#a3afbf'; ctx.textAlign = 'left'; ctx.fillText('0', left, bottom + 16);
    ctx.textAlign = 'right'; ctx.fillText(String(length), right, bottom + 16);
    ctx.textAlign = 'center'; ctx.fillText(`Skill frame ${Math.round(localFrame)} / ${length}`, (left + right) / 2, height - 3);
    ctx.textAlign = 'left';
  }
  function reveal() { panel.hidden = false; draw(); }
  function tick() {
    draw();
    animation = video.paused || video.ended ? null : requestAnimationFrame(tick);
  }
  video.addEventListener('play', () => { reveal(); if (animation === null) tick(); });
  for (const event of ['loadedmetadata', 'timeupdate', 'seeking', 'seeked', 'pause', 'ended']) {
    video.addEventListener(event, reveal);
  }
  canvas.addEventListener('click', event => {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left - 35) / (rect.width - 47)));
    seek(active.start + ratio * (active.end - active.start));
  });
  new ResizeObserver(draw).observe(canvas);
}
