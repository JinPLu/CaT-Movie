/* CaT-Movie project page — interactions. Data: window.CAT_DATA (data.js), paper tables below. */
(() => {
'use strict';
const D = window.CAT_DATA;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const RM = matchMedia('(prefers-reduced-motion: reduce)').matches;
const NS = 'http://www.w3.org/2000/svg';
const sv = (tag, attrs = {}, parent) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; };
const h = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const story = id => D.stories.find(s => s.id === id);
/* media (video, frames) come from a pinned CDN mirror on the published page; empty base = same origin */
const MEDIA = (document.querySelector('meta[name="cat-media"]') || {}).content || '';
if (MEDIA) document.addEventListener('error', e => { const el = e.target; if (!el || !/^(VIDEO|IMG|SOURCE)$/.test(el.tagName)) return; const u = el.getAttribute('src') || ''; if (u.startsWith(MEDIA)) el.setAttribute('src', u.slice(MEDIA.length)); const p = el.getAttribute('poster') || ''; if (p.startsWith(MEDIA)) el.setAttribute('poster', p.slice(MEDIA.length)); }, true);
const frame = (sid, k, kind) => `${MEDIA}f/${sid}/s${k}${kind === 'last' ? '-last' : ''}.jpg`;
const shotAt = (spans, t) => { for (let i = 0; i < spans.length; i++) if (t < spans[i][1]) return i; return spans.length - 1; };
const short = n => n.replace(/^Character /, 'C').replace(/^Object /, 'O').replace(/^Location /, 'L');
const cap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
const camLabel = cam => { if (!cam) return ''; const p = []; if (cam.shot_size) p.push(cap(cam.shot_size.replace(/ shot$/, ''))); if (cam.angle) p.push(cam.angle.replace(/ shot$/, '')); if (cam.movement) p.push(cam.movement); return p.join(' · '); };
const pad = (n, w = 2) => String(n).padStart(w, '0');
const winRange = (spans, w) => [spans[w.shots[0] - 1][0], spans[w.shots[w.shots.length - 1] - 1][1]];
const entLabel = (s, n) => {
  const c = s.plan.characters.find(x => x.name === n); if (c) return `${short(n)}, ${c.appearance.replace(/^an? /, '').split(',')[0]}`;
  const o = s.plan.objects.find(x => x.name === n); return o ? o.appearance.replace(/^an? /, '') : n;
};
const recallUse = (s, w) => w.refs.map((ref, ri) => {
  const src = s.plan.shots[ref.shot - 1]; const have = new Set(src.present.map(p => p.name)); const uses = [];
  w.shots.forEach((k, si) => { if (!(w.routing[si] || []).includes(ri)) return; const sh = s.plan.shots[k - 1];
    uses.push({ shot: k, shared: sh.present.map(p => p.name).filter(n => have.has(n)), place: sh.place === src.place }); });
  return { ref, ri, uses };
});
const purpose = (s, u) => { const names = [...new Set(u.uses.flatMap(x => x.shared))].sort((x, y) => /^Character/.test(y) - /^Character/.test(x));
  return names.length ? names.map(n => esc(entLabel(s, n))).join('; ') : (u.uses.some(x => x.place) ? 'the setting' : ''); };
const once = (node, cb, threshold = 0.25) => { if (!node) return; const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { io.disconnect(); cb(); } }), { threshold }); io.observe(node); };
const watch = (node, cb, threshold = 0.2, rootMargin = '0px') => { const io = new IntersectionObserver(es => es.forEach(e => cb(e.isIntersecting, e)), { threshold, rootMargin }); io.observe(node); return io; };
const COURT = 'film-003995-b0001-shot06';
const HERO = 'film-003995-b0001-shot06';
const run = (name, fn) => { try { fn(); } catch (e) { console.error('[' + name + ']', e); } };

/* ---------------- tooltip ---------------- */
const tip = $('#tip');
function showTip(html, ev) {
  tip.innerHTML = html; tip.classList.add('on');
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 16, y = ev.clientY + 16;
  if (x + r.width > innerWidth - 12) x = ev.clientX - r.width - 16;
  if (y + r.height > innerHeight - 12) y = ev.clientY - r.height - 16;
  tip.style.left = Math.max(8, x) + 'px'; tip.style.top = Math.max(8, y) + 'px';
}
const hideTip = () => tip.classList.remove('on');

/* ---------------- film grain ---------------- */
run('grain', function grain() {
  const c = $('#grain'); const ctx = c.getContext('2d');
  let frames = [], i = 0, last = 0;
  function build() {
    const w = Math.max(64, Math.ceil(innerWidth / 2)), hh = Math.max(64, Math.ceil(innerHeight / 2));
    c.width = w; c.height = hh; frames = [];
    for (let f = 0; f < (RM ? 1 : 4); f++) {
      const img = ctx.createImageData(w, hh); const d = img.data;
      for (let p = 0; p < d.length; p += 4) { const v = Math.random() * 255 | 0; d[p] = d[p + 1] = d[p + 2] = v; d[p + 3] = 255; }
      frames.push(img);
    }
    ctx.putImageData(frames[0], 0, 0);
  }
  build();
  let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(build, 200); });
  if (RM) return;
  (function loop(t) { if (t - last > 83) { last = t; i = (i + 1) % frames.length; ctx.putImageData(frames[i], 0, 0); } requestAnimationFrame(loop); })(0);
});

/* ---------------- nav progress + active section + theme toggle ---------------- */
run('nav', function nav() {
  const root = document.documentElement;
  const tBtn = $('#themeToggle'), tLab = $('#themeLabel'), tIcon = $('#themeIcon');
  const sunSvg = '<circle cx="8" cy="8" r="3.2"/><path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M12.6 3.4l-1.4 1.4M4.8 11.2l-1.4 1.4"/>';
  const moonSvg = '<path d="M13.2 10.2A5.5 5.5 0 0 1 5.8 2.8a5.5 5.5 0 1 0 7.4 7.4z"/>';
  function applyTheme(th) {
    root.dataset.theme = th;
    if (tLab) tLab.textContent = th === 'light' ? 'Dark' : 'Light';
    if (tIcon) tIcon.innerHTML = th === 'light' ? moonSvg : sunSvg;
    try { localStorage.setItem('cat-theme', th); } catch (e) {}
  }
  let saved = 'light';
  try { saved = localStorage.getItem('cat-theme') || 'light'; } catch (e) {}
  applyTheme(saved);
  if (tBtn) tBtn.addEventListener('click', () => applyTheme(root.dataset.theme === 'light' ? 'dark' : 'light'));

  const bar = $('#navProgress span'); const prog = $('#navProgress');
  const secs = $$('main > section'); const links = $$('.nav-links a');
  function ticks() {
    $$('b', prog).forEach(b => b.remove());
    const max = document.documentElement.scrollHeight - innerHeight;
    secs.slice(1).forEach(s => { const b = document.createElement('b'); b.style.left = (Math.min(1, s.offsetTop / max) * 100) + '%'; prog.appendChild(b); });
  }
  function onScroll() { const max = document.documentElement.scrollHeight - innerHeight; bar.style.width = (Math.min(1, scrollY / Math.max(1, max)) * 100) + '%'; }
  addEventListener('scroll', onScroll, { passive: true }); addEventListener('resize', ticks); addEventListener('load', ticks); setTimeout(ticks, 1500); onScroll();
  const nl = $('.nav-links'); const ind = h('span', 'nav-ind'); nl.prepend(ind);
  function mark(id) {
    let hit = null; links.forEach(a => { const on = a.getAttribute('href') === '#' + id; a.classList.toggle('on', on); if (on) hit = a; });
    if (!hit) { ind.style.opacity = 0; return; }
    ind.style.opacity = 1; ind.style.width = hit.offsetWidth + 'px'; ind.style.transform = `translate(${hit.offsetLeft}px,-50%)`;
  }
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) mark(e.target.dataset.nav || e.target.id); }), { rootMargin: '-45% 0px -50% 0px' });
  secs.forEach(s => io.observe(s));
  addEventListener('resize', () => { const a = $('.nav-links a.on'); if (a) mark(a.getAttribute('href').slice(1)); });
});

/* ---------------- hero: one long story; arcs under the timeline carry each recalled view into the shot that reads it ---------------- */
run('hero', function hero() {
  const s = story(HERO); const spans = s.spans.ours; const v = $('#heroVideo');
  const root = $('#top');
  root.addEventListener('pointermove', e => { const r = root.getBoundingClientRect(); root.style.setProperty('--mx', ((e.clientX - r.left) / r.width * 100) + '%'); root.style.setProperty('--my', ((e.clientY - r.top) / r.height * 100) + '%'); });
  $('#reelTitle').textContent = s.title;
  $('#reelStats').textContent = `${spans.length} shots, ${s.seconds.toFixed(1)} s, ${s.windows.length} generation windows`;
  const tl = $('#heroTimeline'); const shotBtns = []; const wins = [];
  s.windows.forEach((w, wi) => {
    const [a, b] = winRange(spans, w);
    const win = h('div', 'tl-win'); win.style.flex = `${b - a} 1 0`;
    win.appendChild(h('div', 'lab', `<span>Window ${wi + 1}</span><span>${(b - a).toFixed(1)} s</span>`));
    const strip = h('div', 'tl-strip');
    w.shots.forEach(k => {
      const [sa, sb] = spans[k - 1];
      const btn = h('button', 'tl-shot', `<span>${k}</span><i class="ph"></i>`);
      btn.type = 'button'; btn.style.flex = `${sb - sa} 1 0`; btn.style.backgroundImage = `url(${frame(HERO, k)})`;
      btn.setAttribute('aria-label', `Jump to shot ${k}`);
      btn.addEventListener('click', () => { v.currentTime = sa + 0.02; v.play().catch(() => {}); });
      strip.appendChild(btn); shotBtns[k - 1] = btn;
    });
    win.appendChild(strip); tl.appendChild(win); wins.push(win);
  });
  const arcSvg = sv('svg', { class: 'tl-arcs', 'aria-hidden': 'true' }, tl);
  let arcs = [];
  function drawArcs(wi) {
    arcSvg.innerHTML = ''; arcs = [];
    shotBtns.forEach(b => b.classList.remove('src'));
    wins.forEach(w => w.classList.toggle('narrow', w.getBoundingClientRect().width < 118));
    const r0 = tl.getBoundingClientRect(); if (!r0.width) return;
    arcSvg.setAttribute('viewBox', `0 0 ${r0.width.toFixed(1)} ${r0.height.toFixed(1)}`);
    recallUse(s, s.windows[wi]).forEach(u => u.uses.forEach(x => {
      const a = shotBtns[u.ref.shot - 1], b = shotBtns[x.shot - 1]; a.classList.add('src');
      const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
      const x1 = ra.left + ra.width / 2 - r0.left, x2 = rb.left + rb.width / 2 - r0.left, y = ra.bottom - r0.top + 3;
      const dip = Math.min(30, 12 + Math.abs(x2 - x1) * 0.05);
      const d = `M${x1.toFixed(1)} ${y.toFixed(1)} C${x1.toFixed(1)} ${(y + dip).toFixed(1)} ${x2.toFixed(1)} ${(y + dip).toFixed(1)} ${x2.toFixed(1)} ${y.toFixed(1)}`;
      const p = sv('path', { d, class: 'arc' }, arcSvg);
      p.style.setProperty('--len', p.getTotalLength().toFixed(1)); p.style.animationDelay = `${arcs.length * 110}ms`;
      const dot = sv('circle', { r: 3.2, class: 'arc-dot' }, arcSvg);
      if (!RM) sv('animateMotion', { dur: '1.5s', repeatCount: 'indefinite', path: d }, dot);
      arcs.push({ p, dot, to: x.shot });
    }));
  }
  const box = $('#heroRecall'), winEl = $('#castWin'), note = $('#castNote'); let items = [];
  function showWindow(wi) {
    const w = s.windows[wi]; const used = recallUse(s, w).filter(u => u.uses.length);
    winEl.textContent = `Window ${wi + 1} of ${s.windows.length}, shot${w.shots.length > 1 ? 's' : ''} ${w.shots[0]}${w.shots.length > 1 ? '–' + w.shots[w.shots.length - 1] : ''}`;
    winEl.classList.remove('bump'); void winEl.offsetWidth; winEl.classList.add('bump');
    box.innerHTML = ''; items = [];
    if (!used.length) { note.textContent = 'Nothing has been generated yet, so the first window starts from the storyboard text alone.';
      box.appendChild(h('div', 'rc-empty', `<b>No earlier shots to recall</b><span>Shots ${w.shots.join(', ')} are generated from their storyboard text only. From window 2 on, earlier shots of the same characters come back here.</span>`)); }
    else {
      note.textContent = 'Earlier shots this window recalls, and who or what each is recalled for. Lit: read by the shot now playing.';
      items = used.map((u, i) => { const el = h('div', 'rc', `<i style="background-image:url(${frame(HERO, u.ref.shot)})"></i><span><b>Shot ${u.ref.shot}</b><span>${purpose(s, u)}</span></span>`); el.style.setProperty('--i', i); box.appendChild(el); return { el, ri: u.ri }; });
    }
    drawArcs(wi);
  }
  const shotNo = $('#heroShotNo'), cam = $('#heroCam'), line = $('#heroLine'), lineNo = $('#heroLineNo'), tc = $('#heroTc'), passEl = $('#heroPass');
  let typeTimer = 0;
  function typeTo(text) {
    clearInterval(typeTimer);
    if (RM) { line.textContent = text; return; }
    let i = 0; line.textContent = '';
    typeTimer = setInterval(() => { i += 3; line.textContent = text.slice(0, i); if (i >= text.length) clearInterval(typeTimer); }, 16);
  }
  const prog = $('#heroProg'); const segs = s.windows.map(w => { const [a, b] = winRange(spans, w); const sp = document.createElement('span'); sp.style.flex = `${b - a} 1 0`; sp.appendChild(document.createElement('i'));
    w.shots.slice(1).forEach(k => { const t = h('b'); t.style.left = ((spans[k - 1][0] - a) / (b - a) * 100) + '%'; sp.appendChild(t); }); prog.appendChild(sp); return { a, b, fill: sp.firstChild }; });
  let cur = -1, curWin = -1;
  function update() {
    const t = v.currentTime || 0; const k = shotAt(spans, t);
    const fr = Math.floor(t * 24);
    tc.textContent = `00:${pad(Math.floor(t / 60) % 60)}:${pad(Math.floor(t) % 60)}:${pad(fr % 24)}`;
    if (k !== cur) {
      cur = k;
      shotNo.textContent = `Shot ${k + 1} / ${spans.length}`;
      if (!RM) { shotNo.classList.add('flash'); setTimeout(() => shotNo.classList.remove('flash'), 260); }
      cam.textContent = camLabel(s.plan.shots[k].camera);
      lineNo.textContent = `Shot ${k + 1}`;
      typeTo(s.requests.shots[k]);
      shotBtns.forEach((b, i) => b.classList.toggle('on', i === k));
      const wi = s.windows.findIndex(w => w.shots.includes(k + 1));
      if (wi !== curWin) { curWin = wi; wins.forEach((w, i) => w.classList.toggle('on', i === wi)); passEl.textContent = `Window ${wi + 1} / ${s.windows.length}`; showWindow(wi); }
      const w = s.windows[wi]; const reads = new Set(w.routing[w.shots.indexOf(k + 1)] || []);
      items.forEach(x => x.el.classList.toggle('on', reads.has(x.ri))); box.classList.toggle('live', reads.size > 0);
      arcs.forEach(a => { const on = a.to === k + 1; a.p.classList.toggle('on', on); a.dot.classList.toggle('on', on); });
    }
    const [a, b] = spans[k]; const ph = shotBtns[k].querySelector('.ph'); ph.style.left = (Math.min(1, (t - a) / (b - a)) * 100) + '%';
    segs.forEach(g => { g.fill.style.width = (Math.max(0, Math.min(1, (t - g.a) / (g.b - g.a))) * 100) + '%'; });
  }
  let raf = 0; const loop = () => { update(); raf = requestAnimationFrame(loop); };
  v.addEventListener('play', () => { cancelAnimationFrame(raf); loop(); });
  v.addEventListener('pause', () => cancelAnimationFrame(raf));
  v.addEventListener('seeked', update);
  const pBtn = $('#heroPlay'), pIcon = $('#heroPlayIcon'); let heroUserPaused = false;
  function syncBtn() { const p = v.paused; pIcon.innerHTML = p ? '<path d="M2 1v8l7-4z" fill="currentColor"/>' : '<path d="M2 1h2v8H2zM6 1h2v8H6z" fill="currentColor"/>'; pBtn.lastChild.textContent = p ? 'Play' : 'Pause'; pBtn.setAttribute('aria-label', p ? 'Play the film' : 'Pause the film'); }
  pBtn.addEventListener('click', () => { if (v.paused) { heroUserPaused = false; v.play().catch(() => {}); } else { heroUserPaused = true; v.pause(); } });
  v.addEventListener('play', syncBtn); v.addEventListener('pause', syncBtn);
  let rt = 0; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { if (curWin >= 0) { drawArcs(curWin); const kk = cur; cur = -1; update(); if (kk < 0) cur = -1; } }, 150); });
  addEventListener('load', () => { if (curWin >= 0) drawArcs(curWin); cur = -1; update(); });
  update();
  watch(v, vis => { if (vis && !heroUserPaused) v.play().catch(() => {}); else if (!vis) v.pause(); }, 0.15);
});

/* ---------------- paper tables ---------------- */
const T1 = [
  { g: 'Per-shot generation' },
  { name: 'Wan2.1, per shot', base: 'Wan2.1-T2V-1.3B', v: [0.998, 0.994, 0.219, 0.169, 0.310, 0.274, 0.694, 0.181, 0.474], fam: 'shot' },
  { name: 'Wan2.2, per shot', base: 'Wan2.2-TI2V-5B', v: [0.999, 0.996, 0.227, 0.275, 0.310, 0.387, 0.701, 0.174, 0.497], fam: 'shot' },
  { name: 'StoryMem', base: 'Wan2.2-I2V-A14B', v: [0.928, 0.719, 0.631, 0.453, 0.687, 0.484, 0.545, 0.179, 0.527], fam: 'shot' },
  { name: 'StoryDiffusion', base: 'Wan2.2-I2V-A14B', v: [0.905, 0.825, 0.300, 0.265, 0.245, 0.597, 0.535, 0.171, 0.468], fam: 'shot' },
  { name: 'IC-LoRA', base: 'Wan2.2-I2V-A14B', v: [0.976, 0.961, 0.168, 0.133, 0.406, 0.452, 0.663, 0.176, 0.484], fam: 'shot' },
  { g: 'Joint multi-shot generation' },
  { name: 'MultiShotMaster', base: 'Wan2.1-T2V-1.3B', v: [0.994, 0.990, 0.538, 0.254, 0.582, 0.371, 0.587, 0.178, 0.527], fam: 'joint' },
  { name: 'HoloCine', base: 'Wan2.2-T2V-A14B', v: [0.748, 0.967, 0.517, 0.512, 0.586, 0.694, 0.691, 0.195, 0.571], fam: 'joint' },
  { name: 'CaT-Movie', base: 'Wan2.2-TI2V-5B', v: [0.995, 0.974, 0.651, 0.497, 0.630, 0.468, 0.785, 0.186, 0.597], fam: 'ours' },
];
const T1COLS = ['Cut F1', 'Execution', 'Character', 'Object', 'Scene', 'Establish', 'Persist', 'Alignment', 'Avg.'];
const T1GROUPS = [['Transition control ↑', 2], ['Inter-shot consistency ↑', 3], ['State continuity ↑', 2], ['Text ↑', 1], ['', 1]];
const POINT_COLOR = { 'CaT-Movie': '#8aa8ff', HoloCine: '#d8b26e', MultiShotMaster: '#7cc4a4', StoryMem: '#e3907f', 'Wan2.1, per shot': '#666d7c', 'Wan2.2, per shot': '#7f8695', StoryDiffusion: '#a3a9b5', 'IC-LoRA': '#525867' };

/* ---------------- pareto chart ---------------- */
run('pareto', function pareto() {
  const host = $('#pareto'); const W = 620, H = 400, m = { l: 54, r: 16, t: 16, b: 46 };
  const x0 = 0.79, x1 = 1.012, y0 = 0.18, y1 = 0.64;
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r), Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  const pts = T1.filter(r => r.name).map(r => ({ name: r.name, fam: r.fam, tr: (r.v[0] + r.v[1]) / 2, co: (r.v[2] + r.v[3] + r.v[4]) / 3, r }));
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Scatter of inter-shot consistency against transition control for eight methods' }, host);
  const grid = sv('g', { class: 'grid' }, svg); const ax = sv('g', { class: 'axis' }, svg);
  [0.80, 0.85, 0.90, 0.95, 1.00].forEach(v => { sv('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }, grid); sv('text', { x: X(v), y: H - m.b + 18, 'text-anchor': 'middle' }, ax).textContent = v.toFixed(2); });
  [0.2, 0.3, 0.4, 0.5, 0.6].forEach(v => { sv('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid); sv('text', { x: m.l - 10, y: Y(v) + 4, 'text-anchor': 'end' }, ax).textContent = v.toFixed(1); });
  sv('path', { d: `M${m.l} ${m.t} V${H - m.b} H${W - m.r}`, fill: 'none' }, ax);
  sv('text', { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', style: 'fill:#c8ccd5' }, ax).textContent = 'Transition control: planned cuts made →';
  const yl = sv('text', { x: 0, y: 0, 'text-anchor': 'middle', transform: `translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)`, style: 'fill:#c8ccd5' }, ax); yl.textContent = 'Consistency: same characters, objects, places →';
  const base = pts.filter(p => p.fam !== 'ours'); const ours = pts.find(p => p.fam === 'ours');
  const front = base.slice().sort((a, b) => b.tr - a.tr).reduce((acc, p) => { if (!acc.length || p.co > acc[acc.length - 1].co) acc.push(p); return acc; }, []);
  const f = xv => Math.max(...base.filter(p => p.tr >= xv).map(p => p.co), y0);
  const bps = front.map(p => p.tr).filter(v => v < ours.tr).sort((a, b) => b - a);
  let poly = `M${X(bps[bps.length - 1])} ${Y(ours.co)} H${X(ours.tr)} V${Y(f(ours.tr))}`;
  bps.forEach((bx, i) => { poly += ` H${X(bx)}`; if (i < bps.length - 1) poly += ` V${Y(f(bx - 1e-6))}`; });
  sv('path', { d: poly + 'Z', fill: 'rgba(138,168,255,.14)', stroke: 'none' }, svg);
  let st = `M${X(front[0].tr)} ${Y(y0)} V${Y(front[0].co)}`;
  for (let i = 1; i < front.length; i++) st += ` H${X(front[i].tr)} V${Y(front[i].co)}`;
  st += ` H${X(x0)}`;
  const fp = sv('path', { d: st, fill: 'none', stroke: '#a3a9b5', 'stroke-width': 1.4, 'stroke-dasharray': '3 4', class: 'front' }, svg);
  const LBL = { 'StoryMem': [0, -16, 'middle'], HoloCine: [14, 5, 'start'], MultiShotMaster: [-14, 20, 'end'], 'CaT-Movie': [-16, 5, 'end'], StoryDiffusion: [12, 4, 'start'], 'IC-LoRA': [-12, 4, 'end'], 'Wan2.2, per shot': [-10, -12, 'end'], 'Wan2.1, per shot': [-10, 20, 'end'] };
  const nodes = [];
  pts.forEach((p, i) => {
    const g = sv('g', { class: 'pt' + (p.fam === 'ours' ? ' ours-pt' : ''), tabindex: 0, 'data-name': p.name, style: 'transform-box:fill-box;transform-origin:center' }, svg);
    const cx = X(p.tr), cy = Y(p.co), col = POINT_COLOR[p.name];
    if (p.fam === 'shot') sv('circle', { cx, cy, r: 7.5, fill: col, stroke: '#07080c', 'stroke-width': 2 }, g);
    else { const s = p.fam === 'ours' ? 12 : 9; if (p.fam === 'ours') sv('rect', { x: cx - s, y: cy - s, width: 2 * s, height: 2 * s, transform: `rotate(45 ${cx} ${cy})`, fill: 'none', stroke: col, 'stroke-opacity': .35, 'stroke-width': 8 }, g); sv('rect', { x: cx - s * .72, y: cy - s * .72, width: 1.44 * s, height: 1.44 * s, transform: `rotate(45 ${cx} ${cy})`, fill: col, stroke: '#07080c', 'stroke-width': 2 }, g); }
    const [dx, dy, an] = LBL[p.name];
    sv('text', { x: cx + dx, y: cy + dy, 'text-anchor': an, class: 'svg-text', style: `fill:${p.fam === 'shot' && p.name !== 'StoryMem' ? '#868d9d' : col};font-weight:${p.fam === 'shot' ? 500 : 600}` }, g).textContent = p.name;
    const v = p.r.v;
    const html = `<b>${esc(p.name)}</b><table><tr><td>Transition control</td><td>${p.tr.toFixed(3)}</td></tr><tr><td>Consistency</td><td>${p.co.toFixed(3)}</td></tr><tr><td>Cut F1 · Execution</td><td>${v[0].toFixed(3)} · ${v[1].toFixed(3)}</td></tr><tr><td>Char · Obj · Scene</td><td>${v[2].toFixed(2)} · ${v[3].toFixed(2)} · ${v[4].toFixed(2)}</td></tr><tr><td>Average</td><td>${v[8].toFixed(3)}</td></tr></table>`;
    g.addEventListener('pointermove', e => showTip(html, e)); g.addEventListener('pointerleave', () => { hideTip(); document.dispatchEvent(new CustomEvent('pick', { detail: { name: null, preview: true } })); });
    g.addEventListener('pointerenter', () => document.dispatchEvent(new CustomEvent('pick', { detail: { name: p.name, preview: true } })));
    g.addEventListener('click', () => { nodes.forEach(n => n.classList.toggle('pinned', n === g)); document.dispatchEvent(new CustomEvent('pick', { detail: { name: p.name, preview: false } })); });
    g.addEventListener('focus', () => { const r = g.getBoundingClientRect(); showTip(html, { clientX: r.right, clientY: r.top }); }); g.addEventListener('blur', hideTip);
    nodes.push(g);
  });
  once(host, () => {
    if (RM) return;
    const len = fp.getTotalLength(); fp.style.setProperty('--len', len); fp.classList.add('draw', 'run');
    nodes.forEach((n, i) => { n.style.animationDelay = (i * 70) + 'ms'; n.classList.add('pop', 'run'); });
  }, 0.4);
});

/* ---------------- one story, three generators (judge verdicts from the released CaT-MBench receipts, v17) ----------------
   While the panel is in view a playhead plays shot 1, 2, 3, 8 in all three rows; each verdict appears as the playhead reaches it. */
run('compare', function compare() {
  const host = $('#cmpGrid'); if (!host) return;
  const sid = 'cinedance-010942-b0000-shot03'; const s = story(sid); const SH = [1, 2, 3, 8];
  const HEAD = [['Medium close-up', 'The bearded man sits at a desk.'], ['Close-up', 'The officer looks to the side.'], ['Medium close-up', 'The bearded man looks up.'], ['Close-up', 'The officer looks to the side.']];
  /* cuts: verdicts for the planned cuts shot 1→2 and shot 2→3; marks: per shot, the judge's attribute verdict against the earlier shot of the same character (v17 receipts):
     kind 'person' = judged a different person, 'outfit' = same person, clothing judged different;
     pin = [x, y] tip on the head or the clothes, zoom = [x, y, width] square crop of the whole head (hair to chin) or the outfit */
  const ROWS = [
    { m: 'holocine', n: 'Joint generation', who: 'HoloCine', col: 'var(--holo)', cuts: ['new', 'new'], marks: { 2: { kind: 'person', pin: [.49, .27], zoom: [.49, .43, .46] }, 8: { kind: 'person', pin: [.40, .19], zoom: [.40, .34, .36] } },
      note: 'HoloCine cuts where the story asks, but the officer who returns in shot 8 is judged a different man from shot 2.' },
    { m: 'storymem', n: 'Memory-augmented', who: 'StoryMem', col: 'var(--mem)', cuts: ['new', 'lost'], marks: { 3: { kind: 'outfit', pin: [.42, .64], zoom: [.40, .69, .36] } },
      note: 'StoryMem keeps both faces, but the bearded man is judged to have changed clothes by shot 3, and shot 3 runs on from shot 2 with no detectable cut, so the planned new shot never happens. Every shot also keeps the same framing.' },
    { m: 'ours', n: 'CaT-Movie', who: 'ours', col: 'var(--ours)', cuts: ['new', 'new'], marks: {},
      note: 'CaT-Movie opens a new camera setup at both cuts, and both men are judged the same people throughout.' },
  ];
  const zoom = (img, cx, cy, c) => { const ch = c * 640 / 370; return `background-image:url(${img});background-size:${100 / c}% ${100 / ch}%;background-position:${((cx - c / 2) / (1 - c) * 100).toFixed(1)}% ${((cy - ch / 2) / (1 - ch) * 100).toFixed(1)}%`; };
  host.innerHTML = `<div class="qh"><span></span>${SH.map((k, i) => `<button type="button" class="cmp-col" data-k="${k}" aria-label="Play shot ${k} in all three"><b>Shot ${k}</b><em>${HEAD[i][0]}</em><q>${HEAD[i][1]}</q></button>`).join('')}</div>` +
    ROWS.map(r => `<div class="qr" data-m="${r.m}"><div class="qn" style="color:${r.col}">${r.n}<small>${r.who === 'ours' ? '' : r.who}</small></div><div class="qstrip${r.m === 'ours' ? ' ours' : ''}">${SH.map(k => {
      const img = `${MEDIA}f/qual/${r.m}-s${k}.jpg`; const mk = r.marks[k];
      return `<div class="qc" data-k="${k}" style="background-image:url(${img})">${mk ? `<span class="pin ${mk.kind}" style="left:${mk.pin[0] * 100}%;top:${mk.pin[1] * 100}%"><svg viewBox="0 0 20 24"><use href="#${mk.kind === 'outfit' ? 'i-outfit' : 'i-neq'}"/></svg></span><span class="zoom" style="${zoom(img, ...mk.zoom)}"></span>` : ''}<span class="tc mono"></span><i class="qbar"></i></div>`;
    }).join('')}${r.cuts.map((c, i) => `<svg class="qcut${c === 'lost' ? ' lost' : ''}" data-b="${i + 1}" style="left:${(i + 1) * 25}%" viewBox="0 0 24 24"><use href="#${c === 'lost' ? 'i-scissor-x' : 'i-scissor'}"/></svg>`).join('')}</div></div>`).join('') +
    `<div class="qlegend"><span><svg viewBox="0 0 20 24"><use href="#i-neq"/></svg>judged a different person</span><span><svg viewBox="0 0 20 24"><use href="#i-outfit"/></svg>same person, judged a different outfit</span><span><svg viewBox="0 0 24 24"><use href="#i-scissor"/></svg>planned cut, new camera setup</span><span><svg viewBox="0 0 24 24"><use href="#i-scissor-x"/></svg>planned cut not made</span><span>shots 4&ndash;7 not shown</span></div>`;

  /* playback: a cell plays its own shot (looping), or once when the playhead drives it */
  let live = [];
  function stop() { live.forEach(({ v, cell }) => { v.pause(); v.remove(); cell.classList.remove('playing'); $('.qbar', cell).style.width = '0'; $('.tc', cell).textContent = ''; }); live = []; }
  function start(cell, onDone) {
    const m = cell.closest('.qr').dataset.m; const k = +cell.dataset.k; const [a, b] = s.spans[m][k - 1];
    const v = document.createElement('video'); v.muted = true; v.playsInline = true; v.preload = 'auto'; v.src = `${MEDIA}v/${sid}/${m}.mp4#t=${a.toFixed(2)}`; v.className = 'qv';
    cell.prepend(v); cell.classList.add('playing'); const tc = $('.tc', cell), bar = $('.qbar', cell);
    const x = { v, cell, done: false };
    v.addEventListener('loadedmetadata', () => { v.currentTime = a + 0.02; v.play().catch(() => {}); }, { once: true });
    v.addEventListener('timeupdate', () => {
      if (!v.isConnected) return;
      if (v.currentTime >= b - 0.06) { if (onDone) { v.pause(); if (!x.done) { x.done = true; onDone(); } } else v.currentTime = a + 0.02; }
      const p = Math.max(0, Math.min(1, (v.currentTime - a) / (b - a)));
      tc.textContent = `${(v.currentTime - a).toFixed(1)} / ${(b - a).toFixed(1)} s`; bar.style.width = (p * 100) + '%';
    });
    live.push(x);
  }

  /* the playhead */
  const tourBtn = $('#cmpTour'), status = $('#cmpStatus');
  let ci = -1, touring = false, held = RM, timer = 0, guard = 0, hoverCell = null, hoverT = 0, pinnedCell = null;
  if (!RM) host.classList.add('tour');
  const revealCut = i => $$(`.qcut[data-b="${i}"]`, host).forEach(q => q.classList.add('in'));
  const revealMarks = i => $$(`.qc[data-k="${SH[i]}"]`, host).forEach(c => c.classList.add('marked'));
  function setBtn() {
    tourBtn.innerHTML = touring ? '<svg viewBox="0 0 14 14"><path d="M4 2.5v9M10 2.5v9" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>Pause' : '<svg viewBox="0 0 14 14"><path d="M3.5 2v10l8.5-5z" fill="currentColor"/></svg>Play all three';
    tourBtn.setAttribute('aria-pressed', String(touring));
    if (!touring) status.textContent = 'Click a frame to play that shot, or a shot heading to play it in all three.';
  }
  function playCol(i) {
    ci = i; stop(); revealCut(i); clearTimeout(guard);
    $$('.cmp-col', host).forEach((b, j) => b.classList.toggle('on', j === i));
    const cells = $$(`.qc[data-k="${SH[i]}"]`, host); let done = 0;
    status.innerHTML = `Shot ${SH[i]}, all three in sync <span class="dots">${SH.map((_, j) => `<i class="${j < i ? 'past' : j === i ? 'now' : ''}"></i>`).join('')}</span>`;
    const next = () => { clearTimeout(guard); clearTimeout(timer); stop(); revealMarks(i); timer = setTimeout(() => { if (!touring) return; const n = (i + 1) % SH.length; if (n === 0) host.classList.remove('tour'); playCol(n); }, 1300); };
    cells.forEach(c => start(c, () => { if (++done === cells.length) next(); }));
    const longest = Math.max(...ROWS.map(r => s.spans[r.m][SH[i] - 1][1] - s.spans[r.m][SH[i] - 1][0]));
    guard = setTimeout(next, (longest + 4) * 1000);
  }
  function startTour() { if (held || touring) return; touring = true; setBtn(); playCol(ci < 0 ? 0 : (ci + 1) % SH.length); }
  function stopTour(hold) {
    touring = false; clearTimeout(timer); clearTimeout(guard);
    if (hold) { held = true; host.classList.remove('tour'); }
    $$('.cmp-col', host).forEach(b => b.classList.remove('on')); setBtn();
  }
  tourBtn.addEventListener('click', () => { if (touring) { stopTour(true); stop(); } else { held = false; startTour(); } });
  watch(host, vis => { if (vis) startTour(); else if (touring) { stopTour(false); stop(); } }, 0.35);

  $$('.qc', host).forEach(c => { c.tabIndex = 0; c.setAttribute('role', 'button'); c.setAttribute('aria-label', `Play shot ${c.dataset.k} of ${c.closest('.qr').dataset.m}`);
    const toggle = () => { const was = !touring && pinnedCell === c; stopTour(true); stop(); hoverCell = null; pinnedCell = was ? null : c; if (!was) start(c); };
    c.addEventListener('click', toggle); c.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
    c.addEventListener('pointerenter', e => { if (e.pointerType !== 'mouse' || touring || !held || pinnedCell) return; clearTimeout(hoverT); hoverT = setTimeout(() => { stop(); hoverCell = c; start(c); }, 120); });
    c.addEventListener('pointerleave', e => { if (e.pointerType !== 'mouse') return; clearTimeout(hoverT); if (!pinnedCell && hoverCell === c) { stop(); hoverCell = null; } }); });
  $$('.cmp-col', host).forEach(b => b.addEventListener('click', () => {
    const k = b.dataset.k; const cells = $$(`.qc[data-k="${k}"]`, host);
    const was = !touring && live.length === cells.length && live.every(x => x.cell.dataset.k === k);
    stopTour(true); stop(); if (!was) { cells.forEach(c => start(c)); b.classList.add('on'); }
  }));

  /* reading the verdicts one generator at a time */
  const ctrl = $('#cmpCtrl'), note = $('#cmpNote');
  const btns = ROWS.map(r => { const b = h('button', 'cmp-btn', `<i style="background:${r.col}"></i>${r.n}${r.who === 'ours' ? '' : ` <span>${r.who}</span>`}`); b.type = 'button'; b.dataset.m = r.m; b.setAttribute('aria-pressed', 'false'); ctrl.appendChild(b); return b; });
  function focusRow(m) {
    btns.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.m === m)));
    $$('.qr', host).forEach(r => r.classList.toggle('dim', m != null && r.dataset.m !== m));
    note.classList.remove('swap'); void note.offsetWidth; note.classList.add('swap');
    note.textContent = m ? ROWS.find(r => r.m === m).note : 'Pick a generator to read what the judge found in its four shots.';
  }
  btns.forEach(b => b.addEventListener('click', () => focusRow(b.getAttribute('aria-pressed') === 'true' ? null : b.dataset.m)));
  focusRow(null); setBtn();
});

/* ---------------- how it works: plan once, then render window by window (Cable Car) ----------------
   Step 1: request lines, cast registry and shot table are linked on hover.
   Step 2: scrolling generates the windows one by one (pinned where there is room): an empty film strip fills in,
   and arcs carry each recalled earlier shot into the shots that read it. Elsewhere the windows are picked by click. */
run('howit', function howit() {
  const plan = $('#planBody'), wrap = $('#renderScroll'), stage = $('#renderStage'); if (!plan || !stage) return;
  const s = story(HERO); const spans = s.spans.ours; const N = s.windows.length;
  const dur = k => spans[k - 1][1] - spans[k - 1][0];
  const ents = s.plan.characters.filter(c => s.plan.shots.some(sh => sh.present.some(p => p.name === c.name)));
  const desc = c => c.appearance.replace(/^an? /, '');
  const label = n => /^Character/.test(n) ? short(n) : entLabel(s, n);
  const who = sh => sh.present.filter(p => /^Character/.test(p.name)).map(p => `<span class="hw-chip">${esc(short(p.name))}</span>`).join('') || '<span class="hw-none">none</span>';

  /* ---- step 1 ---- */
  $('#planNote').textContent = 'Qwen3.8-27B, used as is, turns the request into a storyboard: who each character is, once, and what each shot shows. Code then carries outfits and held items from shot to shot. Point at a character, a request line or a row to see how they link.';
  plan.innerHTML = `<div class="hw-plan">
    <div class="hw-col"><h4>The request, a shot list</h4><ol class="hw-req">${s.requests.shots.map((l, i) => `<li data-k="${i + 1}" style="--i:${i}">${esc(l).replace(/Character (\d+)/g, '<b data-e="Character $1">Character $1</b>')}</li>`).join('')}</ol></div>
    <div class="hw-col"><h4>The storyboard the planner writes</h4>
      <div class="hw-reg">${ents.map((c, i) => `<div class="hw-ent" data-e="${c.name}" style="--i:${i}"><b>${esc(short(c.name))}</b><span>${esc(desc(c))}${c.outfit ? `; ${esc(c.outfit)}` : ''}</span></div>`).join('')}</div>
      <div class="hw-scroll"><table class="hw-shots"><thead><tr><th>Shot</th><th>Camera</th><th>Who is on screen</th></tr></thead><tbody>${s.plan.shots.map((sh, i) => `<tr data-k="${sh.shot}" data-es="${sh.present.map(p => p.name).join('|')}" style="--i:${i}"><td>${sh.shot}</td><td>${esc(camLabel(sh.camera))}</td><td>${who(sh)}</td></tr>`).join('')}</tbody></table></div>
    </div></div>`;
  (function bindPlan() {
    const set = (li, tr, en) => { plan.classList.add('linking');
      $$('.hw-req li', plan).forEach(x => x.classList.toggle('hl', li(x))); $$('.hw-shots tr[data-k]', plan).forEach(x => x.classList.toggle('hl', tr(x))); $$('.hw-ent', plan).forEach(x => x.classList.toggle('hl', en(x))); };
    const clear = () => { plan.classList.remove('linking'); $$('.hl, b.m', plan).forEach(x => x.classList.remove('hl', 'm')); };
    const mark = names => $$('.hw-req b[data-e]', plan).forEach(b => b.classList.toggle('m', names.includes(b.dataset.e)));
    const scroller = $('.hw-scroll', plan);
    const reveal = () => { const first = $('.hw-shots tr.hl', plan); if (!first) return; const top = first.offsetTop - 34, bot = first.offsetTop + first.offsetHeight; if (top < scroller.scrollTop || bot > scroller.scrollTop + scroller.clientHeight) scroller.scrollTo({ top: Math.max(0, top - 20), behavior: RM ? 'auto' : 'smooth' }); };
    $$('.hw-ent', plan).forEach(e => { const n = e.dataset.e;
      e.addEventListener('pointerenter', () => { set(li => !!li.querySelector(`b[data-e="${n}"]`), tr => tr.dataset.es.split('|').includes(n), x => x === e); mark([n]); reveal(); }); e.addEventListener('pointerleave', clear); });
    $$('.hw-req li, .hw-shots tr[data-k]', plan).forEach(x => { const k = x.dataset.k;
      x.addEventListener('pointerenter', () => { const es = $(`.hw-shots tr[data-k="${k}"]`, plan).dataset.es.split('|'); set(li => li.dataset.k === k, tr => tr.dataset.k === k, en => es.includes(en.dataset.e)); mark(es); if (x.tagName === 'LI') reveal(); });
      x.addEventListener('pointerleave', clear); });
  })();
  watch(plan, v => { if (v) plan.classList.add('in'); }, 0.25);

  /* ---- step 2 ---- */
  stage.innerHTML = `<div class="rs-top"><div class="rs-count" id="rsCount"></div><p class="rs-note" id="rsNote"></p>
      <div class="rs-dots" role="group" aria-label="Generation windows">${s.windows.map((w, i) => `<button type="button" data-w="${i}" aria-label="Window ${i + 1}">${i + 1}</button>`).join('')}</div></div>
    <div class="pk${RM ? '' : ' packing'}">${s.windows.map((x, i) => { const d = x.shots.reduce((t, k) => t + dur(k), 0);
      return `<button type="button" class="pk-win" data-w="${i}" aria-label="Window ${i + 1}, ${d.toFixed(1)} seconds" style="flex:${d.toFixed(2)} 1 0"><span class="pk-lab"><b><span class="w-long">Window </span><span class="w-short">W</span>${i + 1}</b><span>${d.toFixed(1)} s</span></span><span class="pk-strip">${x.shots.map((k, j) => `<i style="flex:${dur(k).toFixed(2)} 1 0;--j:${j};background-image:url(${frame(HERO, k)})"><em>${k}</em></i>`).join('')}</span></button>`; }).join('')}</div>
    <div class="hw-stage" id="hwStage"></div>
    <p class="rs-hint" id="rsHint"></p>`;
  const winEls = $$('.pk-win', stage), dots = $$('.rs-dots button', stage), count = $('#rsCount'), note = $('#rsNote'), hint = $('#rsHint'), body = $('#hwStage');
  let wi = -1;
  function setWin(i, force) {
    if (i === wi && !force) return; wi = i; const w = s.windows[i]; const first = w.shots[0];
    winEls.forEach((e, j) => { e.classList.toggle('on', j === i); e.classList.toggle('done', j < i); e.setAttribute('aria-pressed', String(j === i)); });
    dots.forEach((d, j) => { d.classList.toggle('on', j === i); d.classList.toggle('done', j < i); });
    count.innerHTML = `Window ${i + 1}<span>of ${N}</span>`;
    const used = recallUse(s, w).filter(u => u.uses.length); const recalled = new Set(used.map(u => u.ref.shot));
    const range = w.shots.length > 1 ? `Shots ${w.shots[0]}–${w.shots[w.shots.length - 1]} are generated together.` : `Shot ${w.shots[0]} is generated on its own.`;
    note.textContent = used.length ? `${range} ${recalled.size === 1 ? 'One earlier shot is' : recalled.size + ' earlier shots are'} recalled, each only by the shots that show the same character, object or place.`
      : `${range} Nothing has been generated yet, so this window starts from the storyboard text alone.`;
    const past = first > 1 ? `<div class="past-grid">${Array.from({ length: first - 1 }, (_, j) => j + 1).map(k => `<div class="past${recalled.has(k) ? ' rec' : ''}" data-k="${k}"><i style="background-image:url(${frame(HERO, k)})"></i><span>${k}</span></div>`).join('')}</div>`
      : '<p class="hw-empty">Nothing has been generated yet, so window 1 starts from the storyboard text alone.</p>';
    const now = w.shots.map((k, si) => {
      const reads = used.flatMap(u => u.uses.filter(x => x.shot === k).map(x => ({ src: u.ref.shot, what: x.shared.length ? x.shared.map(label).join(', ') : 'setting' })));
      return `<div class="hw-shot" data-k="${k}" style="--i:${si}"><i style="background-image:url(${frame(HERO, k)})"></i><div><b>Shot ${k}<span>${esc(camLabel(s.plan.shots[k - 1].camera))}</span></b><span class="hw-q">${esc(s.requests.shots[k - 1].replace(/^[A-Za-z -]+shot\. /, ''))}</span><span class="hw-reads">${reads.length ? reads.map(r => `<span class="hw-chip ours" data-src="${r.src}">shot ${r.src} &middot; ${esc(r.what)}</span>`).join('') : '<span class="hw-none">reads no earlier shot</span>'}</span></div></div>`;
    }).join('');
    body.innerHTML = `<div class="hw-past"><h4>Earlier shots${used.length ? `, ${recalled.size} recalled` : ''}</h4>${past}</div><div class="hw-now"><h4>Generated together in window ${i + 1}</h4>${now}</div><svg class="hw-arcs" aria-hidden="true"></svg>`;
    $$('.hw-chip[data-src], .past.rec', body).forEach(x => { const src = x.dataset.src || x.dataset.k;
      x.addEventListener('pointerenter', () => { body.classList.add('tracing'); $$('.hw-arc', body).forEach(p => p.classList.toggle('hl', p.dataset.src === src)); $$(`.past[data-k="${src}"]`, body).forEach(p => p.classList.add('hl')); });
      x.addEventListener('pointerleave', () => { body.classList.remove('tracing'); $$('.hl', body).forEach(p => p.classList.remove('hl')); }); });
    requestAnimationFrame(drawArcs);
  }
  function drawArcs() {
    const svg = $('.hw-arcs', body); if (!svg) return;
    const r0 = body.getBoundingClientRect(); svg.setAttribute('viewBox', `0 0 ${r0.width.toFixed(1)} ${r0.height.toFixed(1)}`); svg.innerHTML = '';
    const per = new Map(), seenK = new Map(); let n = 0;
    $$('.hw-chip[data-src]', body).forEach(c => { const k = c.closest('.hw-shot').dataset.k; per.set(k, (per.get(k) || 0) + 1); });
    $$('.hw-chip[data-src]', body).forEach(chip => {
      const src = $(`.past[data-k="${chip.dataset.src}"] i`, body), dst = $('i', chip.closest('.hw-shot')); if (!src || !dst) return;
      const a = src.getBoundingClientRect(), b = dst.getBoundingClientRect();
      const tk = chip.closest('.hw-shot').dataset.k; const j = seenK.get(tk) || 0; seenK.set(tk, j + 1); const m = per.get(tk);
      const x1 = a.right - r0.left - 6, y1 = a.top + a.height / 2 - r0.top, x2 = b.left - r0.left - 5, y2 = b.top - r0.top + b.height * (j + 1) / (m + 1);
      const mx = (x1 + x2) / 2;
      const d = `M${x1.toFixed(1)} ${y1.toFixed(1)} C${mx.toFixed(1)} ${y1.toFixed(1)} ${mx.toFixed(1)} ${y2.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`;
      const p = sv('path', { d, class: 'hw-arc', 'data-src': chip.dataset.src }, svg);
      p.style.setProperty('--len', p.getTotalLength().toFixed(1)); p.style.animationDelay = `${120 + n * 120}ms`;
      sv('circle', { cx: x2.toFixed(1), cy: y2.toFixed(1), r: 3.5, class: 'hw-end', style: `animation-delay:${n * 120 + 800}ms` }, svg);
      if (!RM) { const dot = sv('circle', { r: 3, class: 'hw-dot' }, svg); sv('animateMotion', { dur: '2.2s', begin: `${(n * 0.12 + 0.9).toFixed(2)}s`, repeatCount: 'indefinite', path: d }, dot); }
      n++;
    });
  }
  /* pinned only where there is room and motion is welcome; otherwise the same stage is driven by clicks */
  let pinned = false;
  const canPin = () => !RM && innerWidth > 860 && innerHeight >= 700;
  const span = () => N * 0.58 * innerHeight;
  function fromScroll() {
    if (!pinned) return; const r = wrap.getBoundingClientRect();
    const p = Math.max(0, Math.min(0.9999, (72 - r.top) / span())); setWin(Math.floor(p * N));
  }
  function layout() {
    pinned = canPin(); wrap.classList.toggle('pinned', pinned); wrap.style.setProperty('--n', N);
    hint.textContent = pinned ? 'Keep scrolling to generate the next window, or pick one.' : 'Pick a window.';
    if (pinned) fromScroll(); else setWin(wi < 0 ? 1 : wi, true);
  }
  function go(i) {
    if (!pinned) { setWin(i); return; }
    scrollTo({ top: wrap.getBoundingClientRect().top + scrollY - 72 + (i + 0.5) / N * span(), behavior: RM ? 'auto' : 'smooth' });
  }
  [...winEls, ...dots].forEach(b => b.addEventListener('click', () => go(+b.dataset.w)));
  addEventListener('scroll', fromScroll, { passive: true });
  watch(stage, v => { if (v) $('.pk', stage).classList.remove('packing'); const svg = $('.hw-arcs', body); if (svg && svg.pauseAnimations) v ? svg.unpauseAnimations() : svg.pauseAnimations(); }, 0.2);
  let rt = 0; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { layout(); setWin(wi, true); }, 150); });
  setWin(0, true); layout();
});

/* ---------------- headings wipe in once; numbers count up ---------------- */
run('reveal', function reveal() {
  const heads = $$('main > section:not(.hero) .split h2');
  heads.forEach(hd => hd.classList.add('rv'));
  if (RM) heads.forEach(hd => hd.classList.add('in'));
  else { const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { $('h2', e.target).classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.3 }); heads.forEach(hd => io.observe(hd.closest('.split'))); }
  $$('[data-count]').forEach(n => once(n, () => {
    if (RM) return;
    const to = parseFloat(n.dataset.count), fmt = n.dataset.fmt, t0 = performance.now(), dur = 1400;
    const f = v => (to < 0 ? '−' : '') + (fmt === 'int' ? Math.round(Math.abs(v)).toLocaleString('en-US') : Math.abs(v).toFixed(+fmt));
    (function step(t) { const p = Math.min(1, (t - t0) / dur); const e = 1 - Math.pow(2, -10 * p); n.textContent = f(to * (p === 1 ? 1 : e)); if (p < 1) requestAnimationFrame(step); })(t0);
  }, 0.6));
});

/* ---------------- theater ---------------- */
const MET = [
  { key: 'ours', name: 'CaT-Movie', color: '#8aa8ff', kind: 'planner storyboard · recalled views' },
  { key: 'holocine', name: 'HoloCine', color: '#d8b26e', kind: 'joint generation' },
  { key: 'multishotmaster', name: 'MultiShotMaster', color: '#7cc4a4', kind: 'joint generation' },
  { key: 'storymem', name: 'StoryMem', color: '#e3907f', kind: 'memory-augmented' },
];
run('theater', function theater() {
  const rail = $('#storyRail'), screens = $('#screens'), bar = $('#shotbar'), handle = $('#wipeHandle');
  const playBtn = $('#tPlay'), timeEl = $('#tTime'), capK = $('#capK'), capLine = $('#capLine'), wipeSel = $('#wipeSel'), wipeLab = $('#wipeLab');
  let si = 0, mode = 'grid', focus = 'ours', wipeB = 'storymem', userPaused = false, inView = false, loaded = false, raf = 0, curShot = -1;
  const tiles = MET.map((m, i) => {
    const t = h('div', 'tile', `<video muted playsinline preload="none"></video><span class="name"><i style="background:${m.color}"></i>${m.name}</span><span class="shotno tnum">Shot 1</span><span class="kind">${m.kind}</span>`);
    t.dataset.m = m.key; t.tabIndex = 0; t.setAttribute('aria-label', `${m.name} video`); t.style.viewTransitionName = 'tile-' + m.key;
    t.addEventListener('click', () => { if (mode === 'wipe') return; focus = m.key; setMode('focus'); });
    screens.insertBefore(t, handle); const v = $('video', t); v.muted = true; v.playsInline = true; return { m, t, v, no: $('.shotno', t) };
  });
  const master = tiles[0].v;
  D.stories.forEach((s, i) => {
    const b = h('button', 'story' + (i === 0 ? ' on' : ''), `<span class="th" style="background-image:url(${MEDIA}f/${s.id}/s${s.cover}.jpg)"></span><span><b>${esc(s.title)}</b><small>${s.spans.ours.length} shots · ${s.seconds.toFixed(1)} s</small></span>`);
    b.addEventListener('click', () => { if (si === i) return; si = i; userPaused = false; $$('.story', rail).forEach((x, j) => x.classList.toggle('on', j === i)); load(true); });
    rail.appendChild(b);
  });
  function S() { return D.stories[si]; }
  function load(autoplay) {
    loaded = true; const s = S(); curShot = -1;
    tiles.forEach(({ m, v }) => { v.poster = `${MEDIA}p/${s.id}/${m.key}.jpg`; v.preload = 'auto'; v.src = `${MEDIA}v/${s.id}/${m.key}.mp4`; v.playbackRate = rate; v.load(); });
    bar.replaceChildren();
    s.spans.ours.forEach(([a, b], k) => { const btn = h('button', '', `<i></i><span>${k + 1}</span>`); btn.style.flex = `${b - a} 1 0`; btn.title = `Shot ${k + 1}: ${s.requests.shots[k]}`; btn.addEventListener('click', () => seekShot(k)); bar.appendChild(btn); });
    update(true);
    if (autoplay && !userPaused) play();
  }
  let rate = 1;
  function play() { tiles.forEach(({ v }) => v.play().catch(() => {})); setIcon(true); loop(); }
  function pause() { tiles.forEach(({ v }) => v.pause()); setIcon(false); cancelAnimationFrame(raf); }
  function setIcon(p) { $('#tPlayIcon').innerHTML = p ? '<path d="M3 2h3v10H3zM8 2h3v10H8z" fill="currentColor"/>' : '<path d="M3 1.5v11l9-5.5z" fill="currentColor"/>'; playBtn.classList.toggle('on', p); }
  function seekShot(k) { const s = S(); tiles.forEach(({ m, v }) => { v.currentTime = s.spans[m.key][k][0] + 0.03; }); update(true); }
  function loop() { cancelAnimationFrame(raf); const step = () => { update(); raf = requestAnimationFrame(step); }; raf = requestAnimationFrame(step); }
  function update(force) {
    const s = S(); const t = master.currentTime || 0;
    tiles.forEach(({ m, v, no }, i) => {
      if (i && !master.paused && !v.ended && Math.abs(v.currentTime - t) > 0.3 && t < (v.duration || 0) - 0.1) v.currentTime = t;
      const k = shotAt(s.spans[m.key], v.currentTime || 0); no.textContent = `Shot ${k + 1}/${s.spans[m.key].length}`;
    });
    const k = shotAt(s.spans.ours, t);
    timeEl.textContent = `${t.toFixed(1)} / ${(master.duration || s.seconds).toFixed(1)} s`;
    const btns = bar.children;
    for (let i = 0; i < btns.length; i++) { const [a, b] = s.spans.ours[i]; btns[i].classList.toggle('on', i === k); btns[i].firstChild.style.width = (Math.max(0, Math.min(1, (t - a) / (b - a))) * 100) + '%'; }
    if (k !== curShot || force) { curShot = k; capK.textContent = `Shot ${k + 1}/${s.spans.ours.length}`; capLine.textContent = s.requests.shots[k]; }
  }
  master.addEventListener('ended', () => { tiles.forEach(({ v }) => { v.currentTime = 0; }); if (!userPaused && inView) play(); });
  master.addEventListener('seeked', () => update(true));
  playBtn.addEventListener('click', () => { if (master.paused) { userPaused = false; play(); } else { userPaused = true; pause(); } });
  $('#tRestart').addEventListener('click', () => { seekShot(0); if (master.paused) { userPaused = false; play(); } });
  $$('.seg [data-rate]').forEach(b => b.addEventListener('click', () => { rate = +b.dataset.rate; $$('.seg [data-rate]').forEach(x => x.classList.toggle('on', x === b)); tiles.forEach(({ v }) => v.playbackRate = rate); }));
  function setMode(md) {
    if (document.startViewTransition && !RM && md !== mode && (md === 'focus' || mode === 'focus')) { document.startViewTransition(() => applyMode(md)); return; }
    applyMode(md);
  }
  function applyMode(md) {
    mode = md; $$('.seg [data-mode]').forEach(x => x.classList.toggle('on', x.dataset.mode === md));
    screens.classList.toggle('focus', md === 'focus'); screens.classList.toggle('wipe', md === 'wipe');
    tiles.forEach(({ m, t }) => { t.classList.toggle('big', md === 'focus' && m.key === focus); t.classList.toggle('wipe-a', md === 'wipe' && m.key === 'ours'); t.classList.toggle('wipe-b', md === 'wipe' && m.key === wipeB); });
    if (md === 'focus') { const big = tiles.find(x => x.m.key === focus); screens.prepend(big.t); }
    else tiles.forEach(x => screens.insertBefore(x.t, handle));
    wipeSel.hidden = wipeLab.hidden = md !== 'wipe';
    if (md === 'wipe' && !RM) sweep();
  }
  $$('.seg [data-mode]').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  screens.addEventListener('pointerover', e => { if (mode === 'grid' && e.pointerType === 'mouse' && e.target.closest('.tile')) screens.classList.add('grid-hover'); });
  screens.addEventListener('pointerleave', () => screens.classList.remove('grid-hover'));
  wipeSel.value = wipeB; wipeSel.addEventListener('change', () => { wipeB = wipeSel.value; setMode('wipe'); });
  function setCut(p) { screens.style.setProperty('--cut', (Math.max(0.03, Math.min(0.97, p)) * 100) + '%'); }
  function sweep() { const t0 = performance.now(); (function f(t) { const p = (t - t0) / 1600; if (p > 1) { setCut(.5); return; } setCut(.5 + Math.sin(p * Math.PI * 2) * .32); requestAnimationFrame(f); })(t0); }
  let dragging = false;
  screens.addEventListener('pointerdown', e => { if (mode !== 'wipe') return; dragging = true; screens.setPointerCapture(e.pointerId); move(e); });
  screens.addEventListener('pointermove', e => { if (mode === 'wipe' && (dragging || e.pointerType === 'mouse')) move(e); });
  screens.addEventListener('pointerup', () => dragging = false);
  function move(e) { const r = screens.getBoundingClientRect(); setCut((e.clientX - r.left) / r.width); }
  setCut(.5);
  // keyboard
  document.addEventListener('keydown', e => {
    if (!inView || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    const s = S();
    if (e.key === ' ') { e.preventDefault(); playBtn.click(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); seekShot(Math.min(s.spans.ours.length - 1, shotAt(s.spans.ours, master.currentTime) + 1)); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); seekShot(Math.max(0, shotAt(s.spans.ours, master.currentTime) - 1)); }
    else if (/^[1-4]$/.test(e.key)) { focus = MET[+e.key - 1].key; setMode('focus'); }
  });
  const root = $('#theaterRoot');
  tiles.forEach(({ m, v }) => { v.poster = `${MEDIA}p/${S().id}/${m.key}.jpg`; });
  watch(root, vis => { if (vis && !loaded) load(false); }, 0, '600px');
  watch(root, vis => { inView = vis; if (vis && !userPaused) play(); else if (!vis) pause(); }, 0.3);
});

/* ---------------- cut-gate demo ---------------- */
run('gate', function gate() {
  const J = $('#gateJoint'), M = $('#gateMem'), root = $('#gate'), sw = $('#gateSwitch');
  const zoom = (img, cx, cy) => { const c = 0.34, ch = c * 16 / 9; return `background-image:url(${img});background-size:${100 / c}% ${100 / ch}%;background-position:${((cx - c / 2) / (1 - c) * 100).toFixed(1)}% ${((cy - ch / 2) / (1 - ch) * 100).toFixed(1)}%`; };
  const neq = (x, y) => `<span class="neq" style="left:${x}%;top:${y}%"><svg width="22" height="26" viewBox="0 0 20 24"><use href="#i-neq"/></svg></span>`;
  J.innerHTML = `<div class="gstrip span2"><div class="gframe" style="background-image:url(${MEDIA}f/kitchen/joint-s03.jpg)"></div><div class="gframe" style="background-image:url(${MEDIA}f/kitchen/joint-s04.jpg)">${neq(47, 12)}<span class="zoom" style="${zoom(`${MEDIA}f/kitchen/joint-s04.jpg`, .47, .45)}"></span></div><svg class="scissor" style="left:50%" viewBox="0 0 24 24"><use href="#i-scissor"/></svg></div>` +
    `<div class="gstrip"><div class="gframe" style="background-image:url(${MEDIA}f/kitchen/joint-s06.jpg)">${neq(42, 12)}<span class="zoom" style="${zoom(`${MEDIA}f/kitchen/joint-s06.jpg`, .42, .45)}"></span></div></div>`;
  M.innerHTML = `<div class="gstrip"><div class="gframe" style="background-image:url(${MEDIA}f/kitchen/memory-s03.jpg)"></div></div>` +
    `<div class="gstrip"><div class="gframe" style="background-image:url(${MEDIA}f/kitchen/memory-s04.jpg)"></div><span class="gate-fail"><span>gate fails · setup kept</span></span></div>` +
    `<div class="gstrip"><div class="gframe" style="background-image:url(${MEDIA}f/kitchen/memory-s06.jpg)"></div></div>` +
    `<svg class="scissor" style="left:33.33%" viewBox="0 0 24 24"><use href="#i-scissor-x"/></svg>`;
  let touched = false;
  function set(on) {
    const rows = [$('#srOff'), $('#srOn')]; const before = rows.map(r => r.getBoundingClientRect().top);
    sw.setAttribute('aria-checked', on); root.classList.toggle('gate-on', on); $('#gateState').textContent = on ? 'on' : 'off';
    if (!RM) rows.forEach((r, i) => { const dy = before[i] - r.getBoundingClientRect().top; if (!dy) return; r.animate([{ transform: `translateY(${dy}px)` }, { transform: 'none' }], { duration: 380, easing: 'cubic-bezier(.2,0,0,1)' }); });
    $('#srOff').classList.toggle('dim', on); $('#srOn').classList.toggle('dim', !on);
    const op = $('#srOn .op'); if (on && !RM) { op.style.transform = 'scale(1.6)'; setTimeout(() => op.style.transform = '', 350); }
  }
  sw.addEventListener('click', () => { touched = true; set(sw.getAttribute('aria-checked') !== 'true'); });
  once(root, () => { if (RM) return; setTimeout(() => { if (!touched) set(true); }, 1800); }, 0.5);
});

/* ---------------- tables ---------------- */
run('mainTable', function mainTable() {
  const tb = $('#mainTable'); let sortK = null, dir = -1;
  const rows = T1.filter(r => r.name);
  const best = T1COLS.map((_, c) => { const vals = rows.map(r => r.v[c]).sort((a, b) => b - a); return [vals[0], vals.find(v => v < vals[0])]; });
  const rng = T1COLS.map((_, c) => { const vals = rows.map(r => r.v[c]); return [Math.min(...vals), Math.max(...vals)]; });
  function cell(r, c) { const v = r.v[c]; const cls = v === best[c][0] ? 'best' : v === best[c][1] ? 'second' : ''; const [lo, hi] = rng[c]; const w = 6 + 30 * (v - lo) / (hi - lo || 1); return `<td class="${cls}">${v.toFixed(3)}<span class="bar" style="width:${w}px"></span></td>`; }
  function row(r) { return `<tr class="${r.fam === 'ours' ? 'ours' : ''}"><td>${esc(r.name)}</td><td>${esc(r.base)}</td>${T1COLS.map((_, c) => cell(r, c)).join('')}</tr>`; }
  function render() {
    const head1 = `<tr><th class="grp" colspan="2"></th>${T1GROUPS.map(([g, n]) => `<th class="grp" colspan="${n}">${g}</th>`).join('')}</tr>`;
    const head2 = `<tr><th data-k="-1">Method</th><th data-k="-2">Video base</th>${T1COLS.map((c, i) => `<th data-k="${i}" class="${sortK === i ? 'sorted' : ''}" data-dir="${dir < 0 ? '↓' : '↑'}">${c}</th>`).join('')}</tr>`;
    let body;
    if (sortK === null) body = T1.map(r => r.g ? `<tr class="grp-row"><td colspan="11">${r.g}</td></tr>` : row(r)).join('');
    else body = rows.slice().sort((a, b) => dir * (a.v[sortK] - b.v[sortK])).map(row).join('');
    tb.innerHTML = `<thead>${head1}${head2}</thead><tbody>${body}</tbody>`;
    $$('th[data-k]', tb).forEach(th => th.addEventListener('click', () => { const k = +th.dataset.k; if (k < 0) { sortK = null; } else if (sortK === k) dir = -dir; else { sortK = k; dir = -1; } render(); }));
  }
  render();
  once(tb, () => { if (RM) return; $$('.bar', tb).forEach((b, i) => { b.style.transform = 'scaleX(0)'; requestAnimationFrame(() => { b.style.transitionDelay = (i % 9) * 40 + 'ms'; b.style.transform = ''; }); }); }, 0.3);
});
const T2 = [
  { name: 'CaT-Movie (ref. storyboard)', v: [0.958, 0.581, 0.585, 0.566, 0.615, 0.571], sig: [] , ours: true },
  { g: 'Trained without' },
  { name: 'identity lines', v: [0.942, 0.582, 0.607, 0.557, 0.653, 0.561], sig: [] },
  { name: 'routing', v: [0.923, 0.593, 0.513, 0.595, 0.609, 0.564], sig: [0] },
  { name: 'references', v: [0.966, 0.289, 0.342, 0.474, 0.696, 0.521], sig: [1, 2, 5] },
  { g: 'Reference selection' },
  { name: 'recent shots', v: [0.928, 0.488, 0.615, 0.471, 0.598, 0.563], sig: [0, 1, 3] },
  { name: 'CLIP retrieval', v: [0.953, 0.544, 0.538, 0.503, 0.534, 0.537], sig: [5] },
  { name: 'anchor + first views', v: [0.955, 0.569, 0.547, 0.546, 0.667, 0.566], sig: [] },
  { g: 'Storyboard' },
  { name: 'request only', v: [0.955, 0.525, 0.509, 0.546, 0.557, 0.541], sig: [] },
  { name: 'our planner', v: [0.963, 0.579, 0.530, 0.560, 0.708, 0.587], sig: [] },
];
run('ablTable', function ablTable() {
  const tb = $('#ablTable'); const cols = ['Exec.', 'Char.', 'Obj.', 'Scene', 'Persist', 'Avg.']; let mode = 'abs';
  const ref = T2[0].v;
  function render() {
    tb.innerHTML = `<thead><tr><th style="text-align:left;cursor:default">Setting</th>${cols.map(c => `<th style="cursor:default">${c}</th>`).join('')}</tr></thead><tbody>${T2.map(r => {
      if (r.g) return `<tr class="grp-row"><td colspan="7">${r.g}</td></tr>`;
      const cells = r.v.map((v, i) => {
        const sig = r.sig.includes(i) ? ' sig' : '';
        if (mode === 'abs' || r.ours) return `<td class="${sig.trim()}">${v.toFixed(3)}</td>`;
        const d = (v - ref[i]) * 100; return `<td class="${d > 0.05 ? 'delta-pos' : d < -0.05 ? 'delta-neg' : ''}${sig}">${d > 0 ? '+' : d < 0 ? '−' : '±'}${Math.abs(d).toFixed(1)}</td>`;
      }).join('');
      return `<tr class="${r.ours ? 'ours' : ''}"><td>${r.ours ? '' : '<span style="color:var(--mute)">· </span>'}${esc(r.name)}</td>${cells}</tr>`;
    }).join('')}</tbody>`;
  }
  render();
  $$('#ablMode button').forEach(b => b.addEventListener('click', () => { mode = b.dataset.m; $$('#ablMode button').forEach(x => x.classList.toggle('on', x === b)); render(); }));
});

/* ---------------- charts ---------------- */
function barChart(host, cfg) {
  const { W = 600, H = 300, m = { l: 46, r: 10, t: 14, b: 46 }, groups, series, yMin = 0, yMax = 1, ticks, fmt = v => (v * 100).toFixed(0) + '%' } = cfg;
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.label }, host);
  const Y = v => H - m.b - (v - yMin) / (yMax - yMin) * (H - m.t - m.b);
  const grid = sv('g', { class: 'grid' }, svg), ax = sv('g', { class: 'axis' }, svg);
  ticks.forEach(t => { sv('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }, grid); sv('text', { x: m.l - 8, y: Y(t) + 4, 'text-anchor': 'end' }, ax).textContent = fmt(t); });
  sv('path', { d: `M${m.l} ${H - m.b} H${W - m.r}`, fill: 'none' }, ax);
  const gw = (W - m.l - m.r) / groups.length; const bw = Math.min(30, (gw - 24) / series.length);
  const bars = [];
  groups.forEach((g, gi) => {
    const gx = m.l + gi * gw + (gw - bw * series.length - (series.length - 1) * 4) / 2;
    sv('text', { x: m.l + gi * gw + gw / 2, y: H - m.b + 20, 'text-anchor': 'middle', class: 'svg-text', style: `fill:${g.color || '#c8ccd5'};font-weight:${g.color ? 600 : 500}` }, svg).textContent = g.name;
    if (g.sub) sv('text', { x: m.l + gi * gw + gw / 2, y: H - m.b + 34, 'text-anchor': 'middle', class: 'svg-text', style: 'font-size:10px' }, svg).textContent = g.sub;
    series.forEach((s, si) => {
      const d = cfg.value(gi, si); if (!d) return; const x = gx + si * (bw + 4);
      const gEl = sv('g', { style: 'cursor:pointer' }, svg);
      const r = sv('rect', { x, y: Y(d.v), width: bw, height: Math.max(0, Y(yMin) - Y(d.v)), rx: 2, fill: cfg.fill(gi, si) }, gEl);
      if (d.lo != null) { sv('path', { d: `M${x + bw / 2} ${Y(d.lo)} V${Y(d.hi)} M${x + bw / 2 - 4} ${Y(d.lo)} h8 M${x + bw / 2 - 4} ${Y(d.hi)} h8`, stroke: '#ece8e0', 'stroke-opacity': .55, 'stroke-width': 1.2 }, gEl); }
      gEl.addEventListener('pointermove', e => showTip(cfg.tip(gi, si, d), e)); gEl.addEventListener('pointerleave', hideTip);
      bars.push({ r, y: Y(d.v), hgt: Math.max(0, Y(yMin) - Y(d.v)) });
    });
  });
  once(host, () => { if (RM) return; bars.forEach((b, i) => { b.r.setAttribute('y', Y(yMin)); b.r.setAttribute('height', 0); const t0 = performance.now() + i * 35; (function f(t) { const p = Math.max(0, Math.min(1, (t - t0) / 700)); const e = 1 - Math.pow(1 - p, 3); b.r.setAttribute('y', Y(yMin) - b.hgt * e); b.r.setAttribute('height', b.hgt * e); if (p < 1) requestAnimationFrame(f); })(performance.now()); }); }, 0.35);
}
run('returnChart', function returnChart() {
  const RT = { 'CaT-Movie': { rate: [0.6331, 0.6111, 0.6344], lo: [0.5710, 0.5257, 0.5592], hi: [0.7044, 0.7178, 0.7129] }, HoloCine: { rate: [0.6115, 0.4091, 0.4355], lo: [0.5369, 0.3209, 0.3402], hi: [0.6906, 0.5023, 0.5517] }, MultiShotMaster: { rate: [0.5791, 0.4444, 0.4892], lo: [0.5143, 0.3432, 0.3423], hi: [0.6558, 0.5489, 0.6172] }, StoryMem: { rate: [0.5827, 0.5960, 0.7312], lo: [0.5075, 0.5060, 0.6646], hi: [0.6771, 0.6835, 0.7873] } };
  const N = [278, 198, 186]; const bins = ['<10 s', '10–20 s', '≥20 s']; const names = Object.keys(RT);
  barChart($('#returnChart'), {
    label: 'Character checks passed by return time for four methods', groups: names.map(n => ({ name: n, color: n === 'CaT-Movie' ? 'var(--ours)' : null })), series: bins, yMin: 0.3, yMax: 0.8, ticks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    value: (g, s) => ({ v: RT[names[g]].rate[s], lo: RT[names[g]].lo[s], hi: RT[names[g]].hi[s] }),
    fill: (g, s) => names[g] === 'CaT-Movie' ? `rgba(61,105,235,${[.94, .64, .38][s]})` : `rgba(116,124,140,${[.86, .56, .32][s]})`,
    tip: (g, s, d) => `<b>${names[g]}</b> \u00b7 return ${bins[s]}<table><tr><td>pass rate</td><td>${(d.v * 100).toFixed(1)}%</td></tr><tr><td>95% interval</td><td>${(d.lo * 100).toFixed(1)}\u2013${(d.hi * 100).toFixed(1)}%</td></tr><tr><td>checks</td><td>${N[s]}</td></tr></table>`,
  });
});
run('dataCharts', function dataCharts() {
  function stack(host, parts, colors) {
    const st = h('div', 'stack'); parts.forEach(([n, v], i) => { const d = h('div', '', v >= 12 ? `${v.toFixed(1)}%` : ''); d.style.flex = `${v} 1 0`; d.style.background = colors[i]; d.title = `${n}: ${v.toFixed(1)}%`; d.addEventListener('pointermove', e => showTip(`<b>${n}</b> \u00b7 ${v.toFixed(1)}%`, e)); d.addEventListener('pointerleave', hideTip); st.appendChild(d); });
    const key = h('div', 'stack-key', parts.map(([n, v], i) => `<span><i style="background:${colors[i]}"></i>${n} ${v.toFixed(1)}%</span>`).join(''));
    host.append(st, key);
    once(host, () => { if (RM) return; const kids = Array.from(st.children); kids.forEach(k => { k.dataset.f = k.style.flex; k.style.flexGrow = 0.0001; }); requestAnimationFrame(() => requestAnimationFrame(() => kids.forEach((k, i) => { k.style.transitionDelay = i * 120 + 'ms'; k.style.flex = k.dataset.f; }))); }, 0.4);
  }
  stack($('#srcStack'), [['CinePile', 65.74], ['CineDance', 24.69], ['MovieChat', 9.57]], ['#5a82f7', '#d8b26e', '#7cc4a4']);
  stack($('#typeStack'), [['Live action', 84.92], ['3D / game', 9.35], ['2D', 5.73]], ['#b8c0cf', '#8d95a5', '#666d7c']);
  const ENT = { People: [8.21, 18.82, 33.10, 39.87], Objects: [5.98, 16.44, 17.26, 60.31], Places: [0.27, 81.90, 15.13, 2.70] };
  const host = $('#entStack'); const colors = ['#525867', '#7f8695', '#a3a9b5', '#8aa8ff'];
  Object.entries(ENT).forEach(([k, arr]) => { const lab = h('div', 'small', `<b style="color:var(--paper);font-weight:500">${k}</b>`); const st = h('div', 'stack'); arr.forEach((v, i) => { const d = h('div', '', v >= 9 ? `${v.toFixed(0)}%` : ''); d.style.flex = `${v} 1 0`; d.style.background = colors[i]; if (i === 3) d.style.color = '#081022'; d.addEventListener('pointermove', e => showTip(`<b>${k}</b> · ${['0', '1', '2', '3+'][i]} per block · ${v.toFixed(1)}%`, e)); d.addEventListener('pointerleave', hideTip); st.appendChild(d); }); host.append(lab, st); });
});

/* ---------------- lightbox, copy ---------------- */
run('misc', function misc() {
  $$('[data-zoom]').forEach(b => b.addEventListener('click', () => {
    const lb = h('div', 'lightbox', `<img src="${b.dataset.zoom}" alt=""><button type="button">Close</button>`);
    const close = () => { lb.remove(); document.removeEventListener('keydown', onKey); };
    const onKey = e => { if (e.key === 'Escape') close(); };
    lb.addEventListener('click', close); document.addEventListener('keydown', onKey); document.body.appendChild(lb);
  }));
  const toast = $('#toast');
  $('#copyBib').addEventListener('click', async () => {
    const txt = $('#bibText').textContent;
    try { await navigator.clipboard.writeText(txt); toast.textContent = 'BibTeX copied'; }
    catch { const r = document.createRange(); r.selectNodeContents($('#bibText')); const s = getSelection(); s.removeAllRanges(); s.addRange(r); toast.textContent = 'Selected — press ⌘C to copy'; }
    toast.classList.add('on'); setTimeout(() => toast.classList.remove('on'), 1800);
  });
});
window.CAT = { MEDIA, $, $$, sv, h, esc, once, watch, showTip, hideTip, story, frame, camLabel, short, RM, T1, POINT_COLOR, barChart, pad, winRange, shotAt, run };
})();
