// One visible player per skill. Playback controls always operate on the whole
// selected code; segment bounds and looping remain independent for each clip.
class SkillVideoGrid {
  constructor(root, options = {}) {
    this.root = root;
    this.options = options;
    this.view = 'task';
    this.players = [];
    this.playing = false;
    this.frame = 0;
    this.lastUi = 0;
  }

  escape(value) {
    return String(value ?? '').replace(/[&<>"']/g,
      c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  }

  clear() {
    this.pauseAll();
    for (const p of this.players) {
      p.disposed = true;
      p.video.removeAttribute('src');
      p.video.load();
    }
    this.players = [];
    this.root.innerHTML = '';
    this.options.onViewChange?.('task');
  }

  mount(occurrences, title) {
    this.clear();
    const esc = value => this.escape(value);
    const sorted = [...occurrences].sort((a, b) => Number(a.task_id) - Number(b.task_id)
      || Number(a.episode_id) - Number(b.episode_id) || Number(a.frame_start) - Number(b.frame_start));
    let previousTask = null;
    const cards = sorted.map((o, index) => {
      const task = Number(o.task_id);
      const heading = task !== previousTask
        ? `<h3 class="skill-task-heading">Task ${task}: ${esc(o.task_description || '')}</h3>` : '';
      previousTask = task;
      const info = this.options.renderInfo ? this.options.renderInfo(o)
        : `<div class="skill-card-title">episode ${o.episode_id} · skill ${o.skill_index}</div>
           <div class="skill-card-meta">Task ${task} · ${esc(o.task_description || '')}<br>frames [${o.frame_start}, ${o.frame_end}) · ${o.length} frames</div>`;
      const player = o.video_path
        ? `<div class="skill-stage" data-stage="${index}">
             <video muted playsinline preload="metadata"${o.start_image_path ? ` poster="${esc(o.start_image_path)}"` : ''}></video>
             <button type="button" class="skill-stage-toggle" aria-label="Play or pause all skill videos"></button>
           </div>
           <div class="skill-clip-controls"><button type="button" class="skill-play-all">▶ Play all</button>
             <input class="skill-clip-seek" type="range" min="0" max="1000" value="0" aria-label="Seek this skill">
             <span class="skill-clip-time">0.00 / 0.00 s</span>
           </div><div class="skill-clip-status" role="status"></div>`
        : '<div class="skill-video-missing">Video unavailable in this report</div>';
      return `${heading}<article class="skill-card" data-skill-index="${index}">${info}${player}</article>`;
    }).join('');
    this.root.innerHTML = `<div class="skill-video-toolbar"><strong>${esc(title)} · ${sorted.length} skills</strong>
      <div class="skill-view-buttons"><button type="button" data-skill-view="task">By task</button>
        <button type="button" data-skill-view="all">All skills · full width</button></div>
      <button type="button" class="skill-play-all">▶ Play all</button>
      <span class="skill-video-hint">Any play button starts all clips · each skill loops</span></div>
      <div class="skill-grid">${cards}</div>`;
    this.root.querySelectorAll('[data-skill-view]').forEach(button =>
      button.addEventListener('click', () => this.setView(button.dataset.skillView)));
    this.root.querySelectorAll('.skill-play-all,.skill-stage-toggle').forEach(button =>
      button.addEventListener('click', () => this.toggleAll()));
    this.root.querySelectorAll('.skill-card').forEach(card => {
      const video = card.querySelector('video');
      if (!video) return;
      const o = sorted[Number(card.dataset.skillIndex)];
      const fps = Number(o.video_fps) || 30, offset = Number(o.video_from_timestamp) || 0;
      const p = {video, card, occurrence: o, fps, start: offset + Number(o.frame_start) / fps,
        end: offset + Number(o.frame_end) / fps, ready: false, disposed: false,
        seek: card.querySelector('.skill-clip-seek'), time: card.querySelector('.skill-clip-time'),
        status: card.querySelector('.skill-clip-status')};
      this.players.push(p);
      // Muted inline videos can start together even if some metadata arrives
      // after the click. They retain their own decoder and clip interval.
      video.muted = true;
      video.addEventListener('loadedmetadata', () => this.configure(p));
      video.addEventListener('timeupdate', () => { this.loop(p); this.updatePlayer(p); });
      video.addEventListener('ended', () => {
        if (!p.disposed && this.playing && p.ready) {
          video.currentTime = p.start;
          this.play(p);
        }
      });
      video.addEventListener('seeked', () => this.updatePlayer(p));
      video.addEventListener('error', () => {
        if (!p.disposed) p.status.textContent = 'Could not load this video';
      });
      p.seek.addEventListener('input', () => {
        if (!p.ready || p.disposed) return;
        const duration = p.end - p.start, last = Math.max(p.start, p.end - .5 / p.fps);
        video.currentTime = Math.min(last, p.start + Number(p.seek.value) / 1000 * duration);
        this.updatePlayer(p);
      });
      const path = String(o.video_path).split('#')[0];
      video.src = `${path}#t=${p.start},${p.end}`;
      video.load();
    });
    this.setView(this.view);
    this.updateButtons();
  }

  configure(p) {
    if (p.disposed) return;
    const video = p.video;
    if (Number.isFinite(video.duration) && video.duration > 0) p.end = Math.min(p.end, video.duration);
    if (!(p.end > p.start)) {
      p.status.textContent = 'This skill has no playable video interval';
      return;
    }
    const top = Math.max(0, Math.min(video.videoHeight - 1, Number(p.occurrence.video_crop_row_start) || 0));
    const bottom = Math.max(top + 1, Math.min(video.videoHeight,
      Number(p.occurrence.video_crop_row_end) || video.videoHeight));
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      const height = bottom - top;
      p.card.querySelector('.skill-stage').style.aspectRatio = `${video.videoWidth} / ${height}`;
      video.style.height = `${100 * video.videoHeight / height}%`;
      video.style.top = `${-100 * top / height}%`;
    }
    p.ready = true;
    video.currentTime = p.start;
    this.updatePlayer(p);
    if (this.playing) this.play(p);
  }

  play(p) {
    if (!p.ready || p.disposed || !this.playing) return;
    const result = p.video.play();
    result?.catch(error => {
      if (!p.disposed && this.playing && error.name !== 'AbortError')
        p.status.textContent = 'Playback could not start; click Play all to retry';
    });
  }

  loop(p) {
    if (!this.playing || !p.ready || p.disposed || p.video.seeking) return;
    if (p.video.currentTime >= p.end - .25 / p.fps || p.video.currentTime < p.start - .5 / p.fps) {
      p.video.currentTime = p.start;
      if (p.video.paused) this.play(p);
    }
  }

  tick(now) {
    this.frame = 0;
    if (!this.playing) return;
    const updateUi = now - this.lastUi >= 200;
    for (const p of this.players) {
      this.loop(p);
      if (updateUi) this.updatePlayer(p);
    }
    if (updateUi) this.lastUi = now;
    this.frame = requestAnimationFrame(time => this.tick(time));
  }

  updatePlayer(p) {
    if (!p.ready || p.disposed) return;
    const duration = p.end - p.start, elapsed = Math.max(0, Math.min(duration, p.video.currentTime - p.start));
    p.seek.value = String(Math.round(1000 * elapsed / duration));
    p.time.textContent = `${elapsed.toFixed(2)} / ${duration.toFixed(2)} s`;
  }

  playAll() {
    this.playing = true;
    for (const p of this.players) { this.loop(p); this.play(p); }
    if (!this.frame) this.frame = requestAnimationFrame(time => this.tick(time));
    this.updateButtons();
  }

  pauseAll() {
    this.playing = false;
    cancelAnimationFrame(this.frame);
    this.frame = 0;
    for (const p of this.players) p.video.pause();
    this.updateButtons();
  }

  toggleAll() { if (this.playing) this.pauseAll(); else this.playAll(); }

  updateButtons() {
    this.root.querySelectorAll('.skill-play-all').forEach(button => {
      button.textContent = this.playing ? 'Ⅱ Pause all' : '▶ Play all';
      button.setAttribute('aria-pressed', String(this.playing));
    });
  }

  setView(view) {
    this.view = view === 'all' ? 'all' : 'task';
    this.root.classList.toggle('skill-view-all', this.view === 'all');
    this.root.querySelectorAll('[data-skill-view]').forEach(button =>
      button.setAttribute('aria-pressed', String(button.dataset.skillView === this.view)));
    this.options.onViewChange?.(this.view);
  }
}
