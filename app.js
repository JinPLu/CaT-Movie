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
const frame = (sid, k, kind) => `f/${sid}/s${k}${kind === 'last' ? '-last' : ''}.jpg`;
const shotAt = (spans, t) => { for (let i = 0; i < spans.length; i++) if (t < spans[i][1]) return i; return spans.length - 1; };
const short = n => n.replace(/^Character /, 'C').replace(/^Object /, 'O').replace(/^Location /, 'L');
const cap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
const camLabel = cam => { if (!cam) return ''; const p = []; if (cam.shot_size) p.push(cap(cam.shot_size.replace(/ shot$/, ''))); if (cam.angle) p.push(cam.angle.replace(/ shot$/, '')); if (cam.movement) p.push(cam.movement); return p.join(' · '); };
const pad = (n, w = 2) => String(n).padStart(w, '0');
const winRange = (spans, w) => [spans[w.shots[0] - 1][0], spans[w.shots[w.shots.length - 1] - 1][1]];
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

/* ---------------- nav progress + active section ---------------- */
run('nav', function nav() {
  const bar = $('#navProgress span'); const prog = $('#navProgress');
  const secs = $$('main > section'); const links = $$('.nav-links a');
  function ticks() {
    $$('b', prog).forEach(b => b.remove());
    const max = document.documentElement.scrollHeight - innerHeight;
    secs.slice(1).forEach(s => { const b = document.createElement('b'); b.style.left = (Math.min(1, s.offsetTop / max) * 100) + '%'; prog.appendChild(b); });
  }
  function onScroll() { const max = document.documentElement.scrollHeight - innerHeight; bar.style.width = (Math.min(1, scrollY / Math.max(1, max)) * 100) + '%'; }
  addEventListener('scroll', onScroll, { passive: true }); addEventListener('resize', ticks); addEventListener('load', ticks); setTimeout(ticks, 1500); onScroll();
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id)); }), { rootMargin: '-45% 0px -50% 0px' });
  secs.forEach(s => io.observe(s));
});

/* ---------------- settle-in + counters ---------------- */
run('settle', function settle() {
  const els = $$('.settle');
  els.forEach(e => { if (e.getBoundingClientRect().top > innerHeight * 0.95) e.classList.add('pre'); });
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.remove('pre'); io.unobserve(e.target); } }), { threshold: 0.08 });
  els.forEach(e => io.observe(e));
  $$('[data-count]').forEach(n => once(n, () => {
    if (RM) return;
    const to = parseFloat(n.dataset.count), fmt = n.dataset.fmt, t0 = performance.now(), dur = 1500;
    const f = v => fmt === 'int' ? Math.round(v).toLocaleString('en-US') : v.toFixed(+fmt);
    (function step(t) { const p = Math.min(1, (t - t0) / dur); const e = 1 - Math.pow(2, -10 * p); n.textContent = f(to * (p === 1 ? 1 : e)); if (p < 1) requestAnimationFrame(step); })(t0);
  }, 0.6));
});

/* ---------------- hero ---------------- */
run('hero', function hero() {
  const s = story(HERO); const spans = s.spans.ours; const v = $('#heroVideo');
  const root = $('#top');
  root.addEventListener('pointermove', e => { const r = root.getBoundingClientRect(); root.style.setProperty('--mx', ((e.clientX - r.left) / r.width * 100) + '%'); root.style.setProperty('--my', ((e.clientY - r.top) / r.height * 100) + '%'); });
  $('#reelTitle').textContent = s.title; $('#reelStats').innerHTML = `${spans.length} shots · <b>${s.seconds.toFixed(1)} s</b> · ${s.windows.length} generation passes · 7 s window cap`;
  const tl = $('#heroTimeline'); const shotBtns = []; const wins = []; const refEls = [];
  const NS2 = 'http://www.w3.org/2000/svg'; const arcs = document.createElementNS(NS2, 'svg'); arcs.setAttribute('class', 'tl-arcs'); arcs.setAttribute('aria-hidden', 'true');
  s.windows.forEach((w, wi) => {
    const [a, b] = winRange(spans, w);
    const win = h('div', 'tl-win'); win.style.flex = `${b - a} 1 0`;
    win.appendChild(h('div', 'lab', `<span>Pass ${wi + 1}</span><span>${(b - a).toFixed(1)} s</span>`));
    const strip = h('div', 'tl-strip');
    w.shots.forEach(k => {
      const [sa, sb] = spans[k - 1];
      const btn = h('button', 'tl-shot', `<span>${k}</span><i class="ph"></i>`);
      btn.style.flex = `${sb - sa} 1 0`; btn.style.backgroundImage = `url(${frame(HERO, k)})`;
      btn.setAttribute('aria-label', `Jump to shot ${k}`);
      btn.addEventListener('click', () => { v.currentTime = sa + 0.02; v.play().catch(() => {}); });
      strip.appendChild(btn); shotBtns[k - 1] = btn;
    });
    win.appendChild(strip);
    const refs = h('div', 'tl-refs'); refEls[wi] = [];
    if (!w.refs.length) refs.appendChild(h('span', 'none', 'no history yet'));
    w.refs.forEach(r => {
      const d = h('div', 'tl-ref', `<b>${r.shot}</b>`); d.style.backgroundImage = `url(${frame(HERO, r.shot, r.kind)})`;
      d.title = `Recalled: shot ${r.shot} (${r.kind} frame)${r.entities.length ? ' · ' + r.entities.join(', ') : ' · location view'}`;
      d.addEventListener('pointerenter', () => shotBtns[r.shot - 1].classList.add('src')); d.addEventListener('pointerleave', () => shotBtns[r.shot - 1].classList.remove('src'));
      refs.appendChild(d); refEls[wi].push({ d, shot: r.shot });
    });
    win.appendChild(refs); tl.appendChild(win); wins.push(win);
  });
  tl.appendChild(arcs);
  /* in-frame progress, one segment per generation pass */
  const prog = $('#heroProg'); const segs = s.windows.map(w => { const [a, b] = winRange(spans, w); const sp = document.createElement('span'); sp.style.flex = `${b - a} 1 0`; sp.appendChild(document.createElement('i')); prog.appendChild(sp); return { a, b, fill: sp.firstChild }; });
  const shotNo = $('#heroShotNo'), cam = $('#heroCam'), line = $('#heroLine'), lineNo = $('#heroLineNo'), tc = $('#heroTc'), passEl = $('#heroPass');
  let cur = -1, curWin = -1, typeTimer = null;
  function typeTo(text) {
    clearInterval(typeTimer);
    if (RM) { line.textContent = text; return; }
    let i = 0; line.textContent = '';
    typeTimer = setInterval(() => { i += 3; line.textContent = text.slice(0, i); if (i >= text.length) clearInterval(typeTimer); }, 16);
  }
  function drawArcs() {
    while (arcs.firstChild) arcs.removeChild(arcs.firstChild);
    if (curWin < 0 || innerWidth < 640) return; const base = tl.getBoundingClientRect();
    refEls[curWin].forEach(({ d, shot }, n) => {
      const r1 = d.getBoundingClientRect(), r0 = shotBtns[shot - 1].getBoundingClientRect();
      const x1 = r1.left + r1.width / 2 - base.left, y1 = r1.top - base.top - 1, x0 = r0.left + r0.width / 2 - base.left, y0 = r0.bottom - base.top + 1;
      const p = document.createElementNS(NS2, 'path'); const dy = Math.max(14, (y1 - y0) * .9);
      p.setAttribute('d', `M${x0} ${y0} C${x0} ${y0 + dy}, ${x1} ${y1 - dy}, ${x1} ${y1}`); p.style.animationDelay = `${-n * .2}s`; arcs.appendChild(p);
    });
  }
  addEventListener('resize', () => requestAnimationFrame(drawArcs));
  function update() {
    const t = v.currentTime || 0; const k = shotAt(spans, t);
    const fr = Math.floor(t * 24);
    tc.textContent = `00:${pad(Math.floor(t / 60) % 60)}:${pad(Math.floor(t) % 60)}:${pad(fr % 24)}`;
    if (k !== cur) {
      cur = k;
      shotNo.textContent = `SHOT ${pad(k + 1)} / ${spans.length}`;
      if (!RM) { shotNo.classList.add('flash'); setTimeout(() => shotNo.classList.remove('flash'), 260); }
      cam.textContent = camLabel(s.plan.shots[k].camera);
      lineNo.textContent = `s${pad(k + 1)}`;
      typeTo(s.requests.shots[k]);
      shotBtns.forEach((b, i) => b.classList.toggle('on', i === k));
      const wi = s.windows.findIndex(w => w.shots.includes(k + 1));
      if (wi !== curWin) { curWin = wi; wins.forEach((w, i) => w.classList.toggle('on', i === wi)); passEl.textContent = `Pass ${wi + 1} / ${s.windows.length}`; drawArcs(); }
    }
    const [a, b] = spans[k]; const ph = shotBtns[k].querySelector('.ph'); ph.style.left = (Math.min(1, (t - a) / (b - a)) * 100) + '%';
    segs.forEach(g => { g.fill.style.width = (Math.max(0, Math.min(1, (t - g.a) / (g.b - g.a))) * 100) + '%'; });
  }
  let raf = 0; const loop = () => { update(); raf = requestAnimationFrame(loop); };
  v.addEventListener('play', () => { cancelAnimationFrame(raf); loop(); });
  v.addEventListener('pause', () => cancelAnimationFrame(raf));
  v.addEventListener('seeked', update);
  update(); setTimeout(drawArcs, 600); addEventListener('load', drawArcs);
  watch(v, vis => { if (vis) v.play().catch(() => {}); else v.pause(); }, 0.15);
});

/* ---------------- paradigms (schematic) ---------------- */
const PLANNED = ['C', 'A', 'B', 'A', 'C', 'D'];
const SETUP_NAME = { A: 'Close-up · her', B: 'Close-up · him', C: 'Two-shot', D: 'Over-shoulder' };
const WOMAN = [{ hair: '#d8b98a', top: '#8c7b74' }, { hair: '#5e3b2c', top: '#6f84b3' }, { hair: '#24201e', top: '#a65a5a' }, { hair: '#c9c2b8', top: '#55705f' }];
const MAN = [{ top: '#2f7a55' }, { top: '#a3542f' }, { top: '#43618a' }, { top: '#7d6c37' }];
const PARS = {
  pershot: { groups: [[0], [1], [2], [3], [4], [5]], carry: 'none', render: PLANNED, w: [0, 1, 0, 2, 3, 3], m: [0, 0, 1, 0, 2, 3], wNeq: [0, 1, 0, 1, 1, 1], mNeq: [0, 0, 1, 0, 1, 1],
    verdict: 'Executes nearly every planned cut, but rebuilds the cast from scratch in each shot.', who: 'Wan2.2 per shot (representative of four per-shot and keyframe pipelines)',
    metrics: [['0.996', 'cut execution'], ['0.227', 'character consistency'], ['0.497', 'average']], hl: ['Wan2.1, per shot', 'Wan2.2, per shot', 'StoryDiffusion', 'IC-LoRA'] },
  joint: { groups: [[0, 1, 2], [3, 4, 5]], carry: 'nothing', render: PLANNED, w: [0, 0, 0, 1, 1, 1], m: [0, 0, 0, 0, 1, 1], wNeq: [0, 0, 0, 1, 1, 0], mNeq: [0, 0, 0, 0, 1, 1],
    verdict: 'Coordinates camera cuts within a window, but resets character identity at the window seam.', who: 'HoloCine and MultiShotMaster (62–83% in-window vs. 28–30% cross-window)',
    metrics: [['0.97 / 0.99', 'cut execution'], ['0.52 / 0.54', 'character consistency'], ['0.57 / 0.53', 'average']], hl: ['HoloCine', 'MultiShotMaster'] },
  memory: { groups: [[0], [1], [2], [3], [4], [5]], carry: 'frame', render: ['C', 'A', 'B', 'B', 'C', 'D'], w: [0, 0, 0, 0, 0, 0], m: [0, 0, 0, 0, 0, 0], wNeq: [0, 0, 0, 0, 0, 0], mNeq: [0, 0, 0, 0, 0, 0],
    verdict: 'Preserves character identity, but raw frame recall copies prior framing across planned cuts.', who: 'StoryMem (repeats prior framing at 114 of 674 planned setup changes)',
    metrics: [['0.719', 'cut execution'], ['0.631', 'character consistency'], ['0.527', 'average']], hl: ['StoryMem'] },
  ours: { groups: [[0, 1], [2, 3], [4, 5]], carry: 'storyboard', render: PLANNED, w: [0, 0, 0, 0, 0, 0], m: [0, 0, 0, 0, 0, 0], wNeq: [0, 0, 0, 0, 0, 0], mNeq: [0, 0, 0, 0, 0, 0],
    verdict: 'Borrows identity from storyboard-indexed anchor views while taking state and framing from text.', who: 'CaT-Movie (matches HoloCine within windows and leads by +22.2 pts across windows)',
    metrics: [['0.974', 'cut execution'], ['0.651', 'character consistency'], ['0.597', 'average']], hl: ['CaT-Movie'] },
};
function person(g, cx, cy, s, look, isMan, back) {
  const skin = isMan ? '#6a4733' : '#e3bc9d';
  if (back) { sv('ellipse', { cx, cy: cy + 26 * s, rx: 26 * s, ry: 22 * s, fill: '#0b0d13' }, g); sv('circle', { cx, cy, r: 13 * s, fill: '#0b0d13' }, g); return; }
  sv('path', { d: `M${cx - 24 * s} ${cy + 46 * s} Q${cx - 24 * s} ${cy + 16 * s} ${cx} ${cy + 15 * s} Q${cx + 24 * s} ${cy + 16 * s} ${cx + 24 * s} ${cy + 46 * s} Z`, fill: look.top }, g);
  if (isMan) sv('path', { d: `M${cx - 22 * s} ${cy + 30 * s} H${cx + 22 * s}`, stroke: 'rgba(255,255,255,.18)', 'stroke-width': 3 * s }, g);
  else for (let i = 0; i < 3; i++) sv('path', { d: `M${cx - 20 * s} ${cy + (24 + i * 6) * s} H${cx + 20 * s}`, stroke: 'rgba(255,255,255,.14)', 'stroke-width': 1.6 * s }, g);
  sv('rect', { x: cx - 4 * s, y: cy + 8 * s, width: 8 * s, height: 9 * s, fill: skin }, g);
  if (!isMan) sv('path', { d: `M${cx - 14 * s} ${cy + 10 * s} Q${cx - 16 * s} ${cy - 16 * s} ${cx} ${cy - 15 * s} Q${cx + 16 * s} ${cy - 16 * s} ${cx + 14 * s} ${cy + 10 * s} Z`, fill: look.hair }, g);
  sv('circle', { cx, cy, r: 11 * s, fill: skin }, g);
  if (isMan) sv('path', { d: `M${cx - 11 * s} ${cy - 3 * s} Q${cx} ${cy - 17 * s} ${cx + 11 * s} ${cy - 3 * s} Q${cx} ${cy - 9 * s} ${cx - 11 * s} ${cy - 3 * s}Z`, fill: '#1c1512' }, g);
  else sv('path', { d: `M${cx - 11 * s} ${cy - 1 * s} Q${cx - 6 * s} ${cy - 15 * s} ${cx + 11 * s} ${cy - 4 * s} Q${cx + 2 * s} ${cy - 10 * s} ${cx - 11 * s} ${cy - 1 * s}Z`, fill: look.hair }, g);
}
function neqMark(g, x, y, s = 1) { const u = sv('use', { href: '#i-neq', x: x - 7 * s, y: y - 17 * s, width: 14 * s, height: 17 * s }, g); return u; }
function drawShot(svg, x, y, w, hh, setup, wl, ml, wNeq, mNeq, id) {
  const clip = sv('clipPath', { id }, svg); sv('rect', { x, y, width: w, height: hh, rx: 1.5 }, clip);
  const g = sv('g', { 'clip-path': `url(#${id})` }, svg);
  sv('rect', { x, y, width: w, height: hh, fill: '#252a36' }, g);
  sv('rect', { x, y, width: w, height: hh * 0.62, fill: '#2f3442' }, g);
  sv('rect', { x: x + w * 0.72, y: y + hh * 0.12, width: w * 0.12, height: hh * 0.28, fill: '#3a4152' }, g);
  const marks = [];
  if (setup === 'A') { person(g, x + w * 0.46, y + hh * 0.5, 1.75, wl, false); if (wNeq) marks.push([x + w * 0.46, y + hh * 0.12]); }
  if (setup === 'B') { person(g, x + w * 0.54, y + hh * 0.5, 1.75, ml, true); if (mNeq) marks.push([x + w * 0.54, y + hh * 0.16]); }
  if (setup === 'C') {
    person(g, x + w * 0.3, y + hh * 0.42, 0.95, wl, false); person(g, x + w * 0.7, y + hh * 0.42, 0.95, ml, true);
    sv('rect', { x: x + w * 0.12, y: y + hh * 0.74, width: w * 0.76, height: hh * 0.3, rx: 3, fill: '#d9d4c7', opacity: 0.85 }, g);
    if (wNeq) marks.push([x + w * 0.3, y + hh * 0.17]); if (mNeq) marks.push([x + w * 0.7, y + hh * 0.17]);
  }
  if (setup === 'D') { person(g, x + w * 0.66, y + hh * 0.44, 1.15, ml, true); person(g, x + w * 0.2, y + hh * 0.68, 1.9, wl, false, true); if (mNeq) marks.push([x + w * 0.66, y + hh * 0.13]); }
  marks.forEach(([mx, my]) => neqMark(svg, mx, my));
}
run('paradigms', function paradigms() {
  const root = $('#paradigms'); const stage = $('#parStage'); const read = $('#parRead'); const tabs = $$('.tab', root);
  let cur = 'pershot', timer = null, held = false;
  function render(p) {
    const P = PARS[p]; const fw = 124, fh = 74, gi = 3, G = 34, padS = 6;
    const widths = P.groups.map(g => padS * 2 + g.length * fw + (g.length - 1) * gi);
    const total = widths.reduce((a, b) => a + b, 0) + (P.groups.length - 1) * G;
    const W = Math.max(960, total + 24);
    let x = (W - total) / 2; const top = 76, stripH = fh + 22;
    const svg = sv('svg', { viewBox: `0 0 ${W} 232`, role: 'img', 'aria-label': `${P.who}: ${P.verdict}` });
    sv('text', { x: 8, y: 16, class: 'svg-text', 'letter-spacing': '0.12em' }, svg).textContent = 'PLANNED SHOTS';
    sv('text', { x: 8, y: top - 8, class: 'svg-text', 'letter-spacing': '0.12em' }, svg).textContent = '';
    const fx = []; // frame x positions
    P.groups.forEach((g, gi2) => {
      const sw = widths[gi2];
      const strip = sv('g', { class: 'fade-in', style: `animation-delay:${gi2 * 70}ms` }, svg);
      sv('rect', { x, y: top, width: sw, height: stripH, rx: 4, fill: '#000', stroke: p === 'ours' ? 'rgba(138,168,255,.55)' : '#2a3040' }, strip);
      for (let px = x + 5; px < x + sw - 8; px += 10) { sv('rect', { x: px, y: top + 3.5, width: 5, height: 4, rx: 1, fill: 'rgba(236,232,224,.16)' }, strip); sv('rect', { x: px, y: top + stripH - 7.5, width: 5, height: 4, rx: 1, fill: 'rgba(236,232,224,.16)' }, strip); }
      g.forEach((si, j) => {
        const fxx = x + padS + j * (fw + gi); fx[si] = fxx;
        drawShot(strip, fxx, top + 11, fw, fh, P.render[si], WOMAN[P.w[si]], MAN[P.m[si]], P.wNeq[si], P.mNeq[si], `c-${p}-${si}`);
      });
      if (gi2 < P.groups.length - 1) {
        const sx = x + sw + G / 2, sy = top + stripH / 2; const cg = sv('g', {}, svg);
        if (P.carry === 'none' || P.carry === 'nothing') {
          sv('path', { d: `M${sx - 5} ${sy - 5} l10 10 M${sx + 5} ${sy - 5} l-10 10`, stroke: '#5d6475', 'stroke-width': 1.6, 'stroke-linecap': 'round' }, cg);
        } else {
          sv('path', { d: `M${sx - 15} ${sy} H${sx + 15}`, stroke: p === 'ours' ? 'rgba(138,168,255,.5)' : 'rgba(227,144,127,.5)', 'stroke-dasharray': '2 3' }, cg);
          const mv = sv('g', { class: 'carrier' }, cg);
          if (P.carry === 'frame') { sv('rect', { x: sx - 8, y: sy - 6, width: 16, height: 11, rx: 1.5, fill: '#2f3442', stroke: '#e3907f' }, mv); sv('circle', { cx: sx - 2, cy: sy - 1, r: 2.3, fill: '#e3bc9d' }, mv); sv('circle', { cx: sx + 3, cy: sy - 1, r: 2.3, fill: '#6a4733' }, mv); }
          else { sv('rect', { x: sx - 9, y: sy - 8, width: 9, height: 12, rx: 1, fill: '#ece8e0' }, mv); [0, 1, 2].forEach(i => sv('path', { d: `M${sx - 7.5} ${sy - 5 + i * 3} h6`, stroke: '#3d5bb8', 'stroke-width': 1 }, mv)); sv('circle', { cx: sx + 5, cy: sy - 2, r: 4, fill: '#8aa8ff' }, mv); }
        }
      }
      x += sw + G;
    });
    // planned labels + seams labels
    PLANNED.forEach((s, i) => {
      const cx = fx[i] + fw / 2;
      sv('text', { x: cx, y: 36, 'text-anchor': 'middle', class: 'svg-text', fill: '#ece8e0', style: 'fill:#ece8e0;font-weight:600' }, svg).textContent = `Shot ${i + 1}`;
      sv('text', { x: cx, y: 51, 'text-anchor': 'middle', class: 'svg-text' }, svg).textContent = SETUP_NAME[s];
    });
    // cut marks between consecutive shots
    for (let i = 1; i < 6; i++) {
      const bx = (fx[i - 1] + fw + fx[i]) / 2; const same = P.render[i] === P.render[i - 1];
      sv('use', { href: same ? '#i-scissor-x' : '#i-scissor', x: bx - 9, y: top - 19, width: 18, height: 18, class: 'fade-in', style: `animation-delay:${200 + i * 60}ms` }, svg);
    }
    const carryText = { none: 'each shot is generated in isolation · no visual history crosses shot boundaries', nothing: 'multiple shots denoised per pass · no visual history crosses window seams', frame: 'unindexed historical frames cross the seam · identity stays bundled with prior framing and state', storyboard: 'compiled per-shot prompts + entity-routed anchor views cross every window seam' }[P.carry];
    sv('text', { x: W / 2, y: top + stripH + 28, 'text-anchor': 'middle', class: 'svg-text', style: `fill:${p === 'ours' ? '#8aa8ff' : '#868d9d'}` }, svg).textContent = `One film strip = one generation pass · ${carryText}`;
    stage.replaceChildren(svg);
    read.innerHTML = `<div class="verdict fade-in">${esc(P.verdict)}<small>${esc(P.who)} · Table 1, 92 test stories</small></div>` +
      P.metrics.map((m, i) => `<div class="metric fade-in" style="animation-delay:${80 + i * 60}ms"><b class="tnum${p === 'ours' ? ' k-ours' : ''}">${m[0]}</b><span>${m[1]}</span></div>`).join('');
    tabs.forEach(t => t.classList.toggle('on', t.dataset.p === p));
    tabs.forEach(t => { const b = $('.bar', t); if (b) { b.style.animation = 'none'; void b.offsetWidth; b.style.animation = ''; } });
    document.dispatchEvent(new CustomEvent('paradigm', { detail: P.hl }));
    cur = p;
  }
  const order = ['pershot', 'joint', 'memory', 'ours'];
  function schedule() { clearTimeout(timer); if (held || RM) return; timer = setTimeout(() => { render(order[(order.indexOf(cur) + 1) % 4]); schedule(); }, 6500); }
  tabs.forEach(t => t.addEventListener('click', () => { held = true; root.classList.add('held'); clearTimeout(timer); render(t.dataset.p); }));
  render('pershot');
  if (RM) root.classList.add('held');
  watch(root, vis => { if (vis) schedule(); else clearTimeout(timer); }, 0.35);
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
  sv('text', { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', style: 'fill:#c8ccd5' }, ax).textContent = 'Transition control →';
  const yl = sv('text', { x: 0, y: 0, 'text-anchor': 'middle', transform: `translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)`, style: 'fill:#c8ccd5' }, ax); yl.textContent = 'Inter-shot consistency →';
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
    const g = sv('g', { class: 'pt', tabindex: 0, 'data-name': p.name, style: 'transform-box:fill-box;transform-origin:center' }, svg);
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
  document.addEventListener('paradigm', e => nodes.forEach(n => n.classList.toggle('dim', !e.detail.includes(n.dataset.name) && n.dataset.name !== 'CaT-Movie')));
  once(host, () => {
    if (RM) return;
    const len = fp.getTotalLength(); fp.style.setProperty('--len', len); fp.classList.add('draw', 'run');
    nodes.forEach((n, i) => { n.style.animationDelay = (i * 70) + 'ms'; n.classList.add('pop', 'run'); });
  }, 0.4);
});

/* ---------------- method scrollytelling ---------------- */
run('method', function method() {
  const s = story(COURT); const spans = s.spans.ours; const plan = s.plan;
  const layers = $$('#scrolly .layer'); const steps = $$('#scrolly .step');
  let active = 1, win = 3, focusShot = null;
  /* 1 · request */
  const reqBox = $('#reqBox'); const reqMeta = $('#reqMeta');
  function showReq(t) {
    $$('#reqTabs .tab').forEach(b => b.classList.toggle('on', b.dataset.t === t));
    const R = s.requests;
    if (t === 'shots') {
      reqBox.innerHTML = `<p class="hd">SUBJECTS</p><div class="subj">${R.subjects.map(esc).join('<br>')}</div><p class="hd">SHOTS</p><div class="sl">${R.shots.map((l, i) => `<span class="n">s${pad(i + 1)}</span><span>${esc(l)}</span>`).join('')}</div>`;
    } else reqBox.innerHTML = `<p style="margin:0;font-size:${t === 'synopsis' ? 22 : 17}px;line-height:1.55">${esc(R[t])}</p>`;
    const words = (t === 'shots' ? R.shots.join(' ') + ' ' + R.subjects.join(' ') : R[t]).split(/\s+/).length;
    reqMeta.innerHTML = `<span><b>${words}</b> words</span><span><b>${plan.characters.length}</b> characters · <b>${plan.objects.length}</b> objects in the storyboard</span><span>CaT-Movie rendered this story from the <b>shot list</b></span>`;
  }
  $$('#reqTabs .tab').forEach(b => b.addEventListener('click', () => showReq(b.dataset.t)));
  showReq('shots');
  /* 2 · storyboard */
  const reg = $('#registry'); const tbl = $('#sbTable');
  const ents = [...plan.characters.map(c => ({ n: c.name, t: 'c', d: `${c.appearance}${c.outfit ? '; ' + c.outfit : ''}` })), ...plan.objects.map(o => ({ n: o.name, t: 'o', d: o.appearance })), ...plan.places.map(p => ({ n: p.name, t: 'p', d: p.appearance }))];
  ents.forEach(e => { const d = h('div', `ent ${e.t}`, `${esc(e.n)}<small>${esc(e.d.length > 90 ? e.d.slice(0, 88) + '…' : e.d)}</small>`); d.dataset.n = e.n; d.title = e.d; reg.appendChild(d); });
  tbl.innerHTML = `<colgroup><col class="c1"><col class="c2"><col class="c3"><col></colgroup><thead><tr><th>Shot</th><th>Camera</th><th>Present</th><th>Action and changes</th></tr></thead><tbody>${plan.shots.map(sh => {
    const pres = sh.present.map(p => `<span class="tag${/^Character/.test(p.name) ? ' c' : ''}" data-n="${esc(p.name)}">${esc(short(p.name))}${p.posture ? ' · ' + esc(p.posture) : ''}</span>`).join('');
    const chg = sh.changes.map(c => `<span class="tag chg">${esc(short(c.name))} ${esc(c.what)} → ${esc(Array.isArray(c.after) ? (c.after.join(', ') || 'empty-handed') : c.after)}</span>`).join('');
    const names = sh.present.map(p => p.name).concat(sh.place ? [sh.place] : []).join('|');
    return `<tr data-names="${esc(names)}"><td class="n">${pad(sh.shot)}</td><td class="cam">${esc(camLabel(sh.camera))}</td><td>${pres}</td><td>${esc(sh.action)}${chg ? '<br>' + chg : ''}</td></tr>`;
  }).join('')}</tbody>`;
  $$('.ent', reg).forEach(e => {
    e.addEventListener('pointerenter', () => { e.classList.add('hl'); $$('tbody tr', tbl).forEach(r => { const on = r.dataset.names.split('|').includes(e.dataset.n); r.classList.toggle('hl', on); r.classList.toggle('dim', !on); }); });
    e.addEventListener('pointerleave', () => { e.classList.remove('hl'); $$('tbody tr', tbl).forEach(r => r.classList.remove('hl', 'dim')); });
  });
  /* 3 · packing */
  const pk = $('#packSvg'); const T = spans[spans.length - 1][1]; const PX = t => 24 + t / T * 712;
  function drawPack(animate) {
    pk.replaceChildren();
    sv('text', { x: 24, y: 22, class: 'svg-text', 'letter-spacing': '0.12em' }, pk).textContent = 'STORYBOARD TIMELINE · PLANNED SHOT DURATIONS';
    spans.forEach(([a, b], i) => {
      sv('rect', { x: PX(a) + 1, y: 34, width: Math.max(2, PX(b) - PX(a) - 2), height: 30, rx: 3, fill: '#161a24', stroke: '#323a4b' }, pk);
      sv('text', { x: (PX(a) + PX(b)) / 2, y: 54, 'text-anchor': 'middle', class: 'svg-text', style: 'fill:#ece8e0' }, pk).textContent = i + 1;
    });
    for (let t = 0; t <= T; t += 5) { sv('path', { d: `M${PX(t)} 70 v5`, stroke: '#323a4b' }, pk); sv('text', { x: PX(t), y: 88, 'text-anchor': 'middle', class: 'svg-text' }, pk).textContent = t + ' s'; }
    sv('text', { x: 24, y: 128, class: 'svg-text', 'letter-spacing': '0.12em' }, pk).textContent = 'GENERATION PASSES · EACH STRIP IS DENOISED JOINTLY';
    s.windows.forEach((w, wi) => {
      const [a, b] = winRange(spans, w); const x = PX(a) + 3, wd = PX(b) - PX(a) - 6, y = 150, hh = 92;
      const g = sv('g', {}, pk);
      sv('rect', { x, y, width: wd, height: hh, rx: 5, fill: '#000', stroke: 'rgba(138,168,255,.45)' }, g);
      for (let px = x + 4; px < x + wd - 6; px += 9) { sv('rect', { x: px, y: y + 3, width: 4.5, height: 4, rx: 1, fill: 'rgba(236,232,224,.18)' }, g); sv('rect', { x: px, y: y + hh - 7, width: 4.5, height: 4, rx: 1, fill: 'rgba(236,232,224,.18)' }, g); }
      w.shots.forEach((k, j) => {
        const [sa, sb] = spans[k - 1]; const fx = PX(sa) + (j === 0 ? 5 : 1), fwid = PX(sb) - PX(sa) - (j === 0 ? 6 : 2) - (j === w.shots.length - 1 ? 4 : 0);
        const img = sv('image', { href: frame(COURT, k), x: fx, y: y + 11, width: Math.max(4, fwid), height: hh - 22, preserveAspectRatio: 'xMidYMid slice' }, g);
        if (animate && !RM) { img.style.transition = `transform .8s var(--ease) ${wi * 90 + j * 40}ms, opacity .8s ${wi * 90}ms`; img.style.transform = 'translateY(-118px)'; img.style.opacity = '.2'; requestAnimationFrame(() => requestAnimationFrame(() => { img.style.transform = 'none'; img.style.opacity = '1'; })); }
      });
      sv('text', { x: x + wd / 2, y: y + hh + 18, 'text-anchor': 'middle', class: 'svg-text', style: 'fill:#8aa8ff;font-weight:600' }, g).textContent = `Pass ${wi + 1}`;
      sv('text', { x: x + wd / 2, y: y + hh + 32, 'text-anchor': 'middle', class: 'svg-text' }, g).textContent = `${(b - a).toFixed(1)} s`;
    });
    const cx = PX(0) + 3; sv('path', { d: `M${cx} 300 v8 H${PX(7) + 3} v-8`, stroke: '#d690bb', fill: 'none', 'stroke-width': 1.4 }, pk);
    sv('text', { x: (cx + PX(7)) / 2, y: 326, 'text-anchor': 'middle', class: 'svg-text', style: 'fill:#d690bb' }, pk).textContent = '7 s window cap';
    const longest = Math.max(...s.windows.map(w => { const [a, b] = winRange(spans, w); return b - a; }));
    $('#packNote').textContent = `${s.windows.length} passes · ${spans.length} shots · longest pass ${longest.toFixed(1)} s`;
  }
  drawPack(false);
  /* 4 · recall */
  const rs = $('#recallSvg'); const pbox = $('#promptBox');
  function chips(host) { host.replaceChildren(); s.windows.forEach((w, i) => { const b = h('button', 'win-chip' + (i === win ? ' on' : ''), `Pass ${i + 1}`); b.addEventListener('click', () => { win = i; focusShot = null; refreshWin(); }); host.appendChild(b); }); }
  function highlight(text) { return esc(text).replace(/(Character \d+|Object \d+|Location \d+)/g, '<b>$1</b>').replace(/(By the end of the shot[^.]*\.)/g, '<span class="chgw">$1</span>'); }
  function drawRecall() {
    rs.replaceChildren(); const w = s.windows[win]; const start = w.shots[0];
    const hist = Array.from({ length: start - 1 }, (_, i) => i + 1);
    const refBy = new Map(w.refs.map((r, i) => [r.shot, { ...r, i }]));
    sv('text', { x: 20, y: 20, class: 'svg-text', 'letter-spacing': '0.12em' }, rs).textContent = hist.length ? 'GENERATED HISTORY · EARLIER PASSES' : 'GENERATED HISTORY';
    const hx = {}; let tw = 0, th = 0;
    if (!hist.length) {
      sv('text', { x: 380, y: 96, 'text-anchor': 'middle', style: 'font:600 22px var(--display);fill:#ece8e0' }, rs).textContent = 'First pass: nothing generated yet.';
      sv('text', { x: 380, y: 122, 'text-anchor': 'middle', class: 'svg-text' }, rs).textContent = 'The renderer opens from text alone, and training drops references for 20% of windows to match.';
    } else {
      tw = Math.min(118, (720 - (hist.length - 1) * 8) / hist.length); th = tw * 0.55;
      const totalW = hist.length * tw + (hist.length - 1) * 8; let x = (760 - totalW) / 2;
      hist.forEach(k => {
        const r = refBy.get(k); const g = sv('g', { style: `opacity:${r ? 1 : .22};transition:opacity .4s` }, rs);
        sv('image', { href: frame(COURT, k, r && r.kind), x, y: 32, width: tw, height: th, preserveAspectRatio: 'xMidYMid slice' }, g);
        sv('rect', { x, y: 32, width: tw, height: th, fill: 'none', stroke: r ? '#8aa8ff' : '#323a4b', 'stroke-width': r ? 2 : 1, rx: 2 }, g);
        sv('text', { x: x + 4, y: 44, style: 'font:600 9.5px var(--mono);fill:#fff;paint-order:stroke;stroke:#000;stroke-width:2.5px' }, g).textContent = k;
        if (r) {
          sv('text', { x: x + tw / 2, y: 32 + th + 14, 'text-anchor': 'middle', class: 'svg-text', style: 'fill:#8aa8ff;font-size:10px' }, g).textContent = r.entities.length ? r.entities.map(short).join(' ') : 'place';
          sv('text', { x: x + tw / 2, y: 32 + th + 27, 'text-anchor': 'middle', class: 'svg-text', style: 'font-size:9.5px' }, g).textContent = r.kind === 'last' ? 'last frame' : 'mid frame';
        }
        hx[k] = x + tw / 2; x += tw + 8;
      });
    }
    const m = w.shots.length; const bw = Math.min(230, (700 - (m - 1) * 18) / m), bh = bw * 0.55; const totalB = m * bw + (m - 1) * 18; let bx = (760 - totalB) / 2; const by = 236;
    sv('text', { x: 20, y: by - 22, class: 'svg-text', 'letter-spacing': '0.12em', style: 'fill:#8aa8ff' }, rs).textContent = `PASS ${win + 1} · DENOISED JOINTLY`;
    const curves = sv('g', {}, rs);
    w.shots.forEach((k, j) => {
      const g = sv('g', { style: 'cursor:pointer' }, rs);
      const sel = (focusShot ?? w.shots[0]) === k;
      sv('rect', { x: bx - 6, y: by - 6, width: bw + 12, height: bh + 12, rx: 6, fill: '#000', stroke: sel ? '#8aa8ff' : 'rgba(138,168,255,.35)', 'stroke-width': sel ? 2 : 1 }, g);
      sv('image', { href: frame(COURT, k), x: bx, y: by, width: bw, height: bh, preserveAspectRatio: 'xMidYMid slice' }, g);
      sv('text', { x: bx, y: by + bh + 22, style: 'font:600 12px var(--mono);fill:#ece8e0' }, g).textContent = `Shot ${k}`;
      sv('text', { x: bx + 12 + 7.4 * ('Shot ' + k).length, y: by + bh + 22, class: 'svg-text' }, g).textContent = camLabel(plan.shots[k - 1].camera);
      const reads = (w.routing[j] || []).map(i => w.refs[i]);
      sv('text', { x: bx, y: by + bh + 38, class: 'svg-text', style: 'font-size:10px' }, g).textContent = reads.length ? `reads ${reads.map(r => 'shot ' + r.shot).join(', ')}` : 'reads no recalled frame';
      g.addEventListener('click', () => { focusShot = k; drawRecall(); });
      (w.routing[j] || []).forEach((ri, n) => {
        const r = w.refs[ri]; const sx = hx[r.shot]; if (sx == null) return; const sy = 32 + th + 32; const tx = bx + bw / 2 + (n - ((w.routing[j].length - 1) / 2)) * 10, ty = by - 8;
        const p = sv('path', { d: `M${sx} ${sy} C${sx} ${sy + 60}, ${tx} ${ty - 70}, ${tx} ${ty}`, fill: 'none', stroke: '#8aa8ff', 'stroke-width': 1.5, 'stroke-opacity': (focusShot ?? w.shots[0]) === k ? .9 : .35 }, curves);
        if (!RM) { const L = 400; p.style.setProperty('--len', L); p.style.strokeDasharray = L; p.classList.add('draw', 'run'); p.style.animationDelay = (n * 120 + j * 160) + 'ms'; }
      });
      bx += bw + 18;
    });
    const k = focusShot ?? w.shots[0]; const j = w.shots.indexOf(k);
    const reads = (w.routing[j] || []).map(i => w.refs[i]);
    pbox.innerHTML = `<div style="color:var(--mute);margin-bottom:6px">Compiled prompt c<sub>${k}</sub> · click a shot to switch</div>${highlight(w.prompts[j])}` +
      (reads.length ? `<div style="color:var(--mute);margin:10px 0 4px">Identity lines of the frames shot ${k} reads</div>` + reads.map(r => `<div>shot ${r.shot}: ${highlight(r.line)}</div>`).join('') : '');
  }
  /* 5 · attention mask */
  const ms = $('#maskSvg'); const mv = $('#miniVideo'); const leg = $('#maskLegend');
  let loopRange = null;
  function drawMask() {
    ms.replaceChildren(); const w = s.windows[win]; const N = w.refs.length, M = w.shots.length;
    const blocks = []; w.refs.forEach((r, i) => blocks.push({ t: 'ref', i, u: 1, lab: `R${i + 1}` }));
    w.shots.forEach((k, j) => { blocks.push({ t: 'first', j, u: 1, lab: `${k}ᶠ` }); blocks.push({ t: 'rest', j, u: 3, lab: `${k}` }); });
    const U = blocks.reduce((a, b) => a + b.u, 0); const off = 44, size = 400 - off - 6, c = size / U;
    let acc = 0; blocks.forEach(b => { b.o = acc; acc += b.u; });
    const allow = (q, k) => {
      if (q.t === 'ref') return k.t === 'ref' && k.i === q.i ? 'ref' : null;
      if (k.t === 'ref') return (w.routing[q.j] || []).includes(k.i) ? 'ref' : null;
      if (k.j === q.j) return 'self';
      if (k.t === 'first') return 'first';
      return null;
    };
    const COL = { self: '#ece8e0', first: '#d8b26e', ref: '#8aa8ff' };
    const rows = [];
    blocks.forEach(q => {
      const rowG = sv('g', { class: 'mrow' }, ms);
      blocks.forEach(k => {
        const a = allow(q, k);
        sv('rect', { x: off + k.o * c + .5, y: off + q.o * c + .5, width: k.u * c - 1, height: q.u * c - 1, rx: 1.5, fill: a ? COL[a] : '#161a24', 'fill-opacity': a ? (q.t === 'ref' ? .45 : .92) : 1, class: 'cell' }, rowG);
      });
      sv('text', { x: off - 6, y: off + (q.o + q.u / 2) * c + 3.5, 'text-anchor': 'end', class: 'svg-text', style: `font-size:${Math.min(10.5, c * 0.9 + 3)}px;fill:${q.t === 'ref' ? '#8aa8ff' : '#ece8e0'}` }, ms).textContent = q.t === 'ref' ? q.lab : (q.t === 'first' ? '' : `S${w.shots[q.j]}`);
      const hit = sv('rect', { x: 0, y: off + q.o * c, width: 400, height: q.u * c, fill: 'transparent', style: 'cursor:pointer' }, ms);
      hit.addEventListener('pointerenter', () => focusRow(q));
      hit.addEventListener('pointerleave', () => focusRow(null));
      rows.push({ q, rowG });
    });
    blocks.forEach(k => sv('text', { x: off + (k.o + k.u / 2) * c, y: off - 8, 'text-anchor': 'middle', class: 'svg-text', style: `font-size:${Math.min(10.5, c * 0.9 + 3)}px;fill:${k.t === 'ref' ? '#8aa8ff' : '#ece8e0'}` }, ms).textContent = k.t === 'ref' ? k.lab : (k.t === 'first' ? 'ᶠ' : `S${w.shots[k.j]}`));
    sv('text', { x: 4, y: 12, class: 'svg-text', style: 'font-size:9.5px;letter-spacing:.1em' }, ms).textContent = 'QUERY ↓  KEY →';
    function focusRow(q) {
      rows.forEach(r => r.rowG.style.opacity = !q ? 1 : ((r.q.t === q.t && (r.q.i === q.i && q.t === 'ref' || (q.t !== 'ref' && r.q.j === q.j))) ? 1 : .18));
      legend(q);
      if (q && q.t !== 'ref') { const k = w.shots[q.j]; loopRange = spans[k - 1]; mv.currentTime = loopRange[0] + 0.02; }
      else if (!q) setLoop();
    }
    function legend(q) {
      const base = `<div><i style="background:#ece8e0"></i>own shot tokens</div><div><i style="background:#d8b26e"></i>first latent frame of a co-generated shot</div><div><i style="background:#8aa8ff"></i>routed recalled frame</div>`;
      let msg;
      if (!q) msg = N ? `Hover a row. ${M} shots and ${N} recalled frame${N > 1 ? 's' : ''} share this pass.` : 'First pass: no recalled frames, shots see each other only through first latent frames.';
      else if (q.t === 'ref') { const r = w.refs[q.i]; msg = `<b>R${q.i + 1}</b> = shot ${r.shot}, ${r.kind} frame${r.entities.length ? ' (' + r.entities.join(', ') + ')' : ''}. It attends only to itself and its identity line.`; }
      else { const k = w.shots[q.j]; const reads = (w.routing[q.j] || []).map(i => `R${i + 1} (shot ${w.refs[i].shot})`); const others = w.shots.filter(x => x !== k); msg = `<b>Shot ${k}</b> reads its own tokens${others.length ? ', the first latent frame of shot ' + others.join(' and ') : ''}${reads.length ? ', and ' + reads.join(', ') : ', and no recalled frame'}.`; }
      leg.innerHTML = base + `<div style="margin-top:6px;color:var(--paper)">${msg}</div>`;
    }
    legend(null);
    $('#miniWin').textContent = `PASS ${win + 1}`;
  }
  function setLoop() { loopRange = winRange(spans, s.windows[win]); if (mv.currentTime < loopRange[0] || mv.currentTime > loopRange[1]) mv.currentTime = loopRange[0] + 0.02; }
  mv.addEventListener('timeupdate', () => {
    if (!loopRange) return; if (mv.currentTime >= loopRange[1] - 0.05) mv.currentTime = loopRange[0] + 0.02;
    $('#miniShot').textContent = `SHOT ${pad(shotAt(spans, mv.currentTime) + 1)}`;
  });
  function refreshWin() { chips($('#winChips4')); chips($('#winChips5')); drawRecall(); drawMask(); setLoop(); }
  refreshWin();
  /* activation */
  function setStep(n) {
    if (n === active) return; active = n;
    layers.forEach(l => l.classList.toggle('on', +l.dataset.layer === n));
    steps.forEach(st => st.classList.toggle('on', +st.dataset.step === n));
    if (n === 3) drawPack(true);
    if (n === 4) drawRecall();
    if (n === 5) { setLoop(); mv.play().catch(() => {}); } else mv.pause();
  }
  const narrow = matchMedia('(max-width: 920px)').matches;
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) setStep(+e.target.dataset.step); }), { rootMargin: narrow ? '-74% 0px -22% 0px' : '-48% 0px -48% 0px' });
  steps.forEach(st => io.observe(st));
  watch($('#scrolly'), vis => { if (!vis) mv.pause(); else if (active === 5) mv.play().catch(() => {}); }, 0.05);
});

/* ---------------- theater ---------------- */
const MET = [
  { key: 'ours', name: 'CaT-Movie', color: '#8aa8ff', kind: 'planner storyboard · routed references' },
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
    t.dataset.m = m.key; t.tabIndex = 0; t.setAttribute('aria-label', `${m.name} video`);
    t.addEventListener('click', () => { if (mode === 'wipe') return; focus = m.key; setMode('focus'); });
    screens.insertBefore(t, handle); const v = $('video', t); v.muted = true; v.playsInline = true; return { m, t, v, no: $('.shotno', t) };
  });
  const master = tiles[0].v;
  D.stories.forEach((s, i) => {
    const b = h('button', 'story' + (i === 0 ? ' on' : ''), `<span class="th" style="background-image:url(f/${s.id}/s${s.cover}.jpg)"></span><span><b>${esc(s.title)}</b><small>${s.spans.ours.length} shots · ${s.seconds.toFixed(1)} s</small></span>`);
    b.addEventListener('click', () => { if (si === i) return; si = i; $$('.story', rail).forEach((x, j) => x.classList.toggle('on', j === i)); load(true); });
    rail.appendChild(b);
  });
  function S() { return D.stories[si]; }
  function load(autoplay) {
    loaded = true; const s = S(); curShot = -1;
    tiles.forEach(({ m, v }) => { v.preload = 'auto'; v.src = `v/${s.id}/${m.key}.mp4`; v.playbackRate = rate; v.load(); });
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
    mode = md; $$('.seg [data-mode]').forEach(x => x.classList.toggle('on', x.dataset.mode === md));
    screens.classList.toggle('focus', md === 'focus'); screens.classList.toggle('wipe', md === 'wipe');
    tiles.forEach(({ m, t }) => { t.classList.toggle('big', md === 'focus' && m.key === focus); t.classList.toggle('wipe-a', md === 'wipe' && m.key === 'ours'); t.classList.toggle('wipe-b', md === 'wipe' && m.key === wipeB); });
    if (md === 'focus') { const big = tiles.find(x => x.m.key === focus); screens.prepend(big.t); }
    else tiles.forEach(x => screens.insertBefore(x.t, handle));
    wipeSel.hidden = wipeLab.hidden = md !== 'wipe';
    if (md === 'wipe' && !RM) sweep();
  }
  $$('.seg [data-mode]').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
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
  watch(root, vis => { if (vis && !loaded) load(false); }, 0, '600px');
  watch(root, vis => { inView = vis; if (vis && !userPaused) play(); else if (!vis) pause(); }, 0.3);
});

/* ---------------- cut-gate demo ---------------- */
run('gate', function gate() {
  const J = $('#gateJoint'), M = $('#gateMem'), root = $('#gate'), sw = $('#gateSwitch');
  const zoom = (img, cx, cy) => { const c = 0.34, ch = c * 16 / 9; return `background-image:url(${img});background-size:${100 / c}% ${100 / ch}%;background-position:${((cx - c / 2) / (1 - c) * 100).toFixed(1)}% ${((cy - ch / 2) / (1 - ch) * 100).toFixed(1)}%`; };
  const neq = (x, y) => `<span class="neq" style="left:${x}%;top:${y}%"><svg width="22" height="26" viewBox="0 0 20 24"><use href="#i-neq"/></svg></span>`;
  J.innerHTML = `<div class="gstrip span2"><div class="gframe" style="background-image:url(f/kitchen/joint-s03.jpg)"></div><div class="gframe" style="background-image:url(f/kitchen/joint-s04.jpg)">${neq(47, 12)}<span class="zoom" style="${zoom('f/kitchen/joint-s04.jpg', .47, .45)}"></span></div><svg class="scissor" style="left:50%" viewBox="0 0 24 24"><use href="#i-scissor"/></svg></div>` +
    `<div class="gstrip"><div class="gframe" style="background-image:url(f/kitchen/joint-s06.jpg)">${neq(42, 12)}<span class="zoom" style="${zoom('f/kitchen/joint-s06.jpg', .42, .45)}"></span></div></div>`;
  M.innerHTML = `<div class="gstrip"><div class="gframe" style="background-image:url(f/kitchen/memory-s03.jpg)"></div></div>` +
    `<div class="gstrip"><div class="gframe" style="background-image:url(f/kitchen/memory-s04.jpg)"></div><span class="gate-fail"><span>gate fails · setup kept</span></span></div>` +
    `<div class="gstrip"><div class="gframe" style="background-image:url(f/kitchen/memory-s06.jpg)"></div></div>` +
    `<svg class="scissor" style="left:33.33%" viewBox="0 0 24 24"><use href="#i-scissor-x"/></svg>`;
  let touched = false;
  function set(on) {
    sw.setAttribute('aria-checked', on); root.classList.toggle('gate-on', on); $('#gateState').textContent = on ? 'on' : 'off';
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
    label: 'Character checks passed by return time for four methods', groups: names.map(n => ({ name: n, color: n === 'CaT-Movie' ? '#8aa8ff' : null })), series: bins, yMin: 0.3, yMax: 0.8, ticks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    value: (g, s) => ({ v: RT[names[g]].rate[s], lo: RT[names[g]].lo[s], hi: RT[names[g]].hi[s] }),
    fill: (g, s) => names[g] === 'CaT-Movie' ? `rgba(138,168,255,${[.95, .6, .32][s]})` : `rgba(236,232,224,${[.9, .55, .28][s]})`,
    tip: (g, s, d) => `<b>${names[g]}</b> · return ${bins[s]}<table><tr><td>pass rate</td><td>${(d.v * 100).toFixed(1)}%</td></tr><tr><td>95% interval</td><td>${(d.lo * 100).toFixed(1)}–${(d.hi * 100).toFixed(1)}%</td></tr><tr><td>checks</td><td>${N[s]}</td></tr></table>`,
  });
});
run('dataCharts', function dataCharts() {
  function stack(host, parts, colors) {
    const st = h('div', 'stack'); parts.forEach(([n, v], i) => { const d = h('div', '', v >= 12 ? `${v.toFixed(1)}%` : ''); d.style.flex = `${v} 1 0`; d.style.background = colors[i]; d.title = `${n}: ${v.toFixed(1)}%`; d.addEventListener('pointermove', e => showTip(`<b>${n}</b> · ${v.toFixed(1)}%`, e)); d.addEventListener('pointerleave', hideTip); st.appendChild(d); });
    const key = h('div', 'stack-key', parts.map(([n, v], i) => `<span><i style="background:${colors[i]}"></i>${n} ${v.toFixed(1)}%</span>`).join(''));
    host.append(st, key);
    once(host, () => { if (RM) return; const kids = Array.from(st.children); kids.forEach(k => { k.dataset.f = k.style.flex; k.style.flexGrow = 0.0001; }); requestAnimationFrame(() => requestAnimationFrame(() => kids.forEach((k, i) => { k.style.transitionDelay = i * 120 + 'ms'; k.style.flex = k.dataset.f; }))); }, 0.4);
  }
  stack($('#srcStack'), [['CinePile', 65.74], ['CineDance', 24.69], ['MovieChat', 9.57]], ['#8aa8ff', '#d8b26e', '#7cc4a4']);
  stack($('#typeStack'), [['Live action', 84.92], ['3D / game', 9.35], ['2D', 5.73]], ['#ece8e0', '#a3a9b5', '#666d7c']);
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
window.CAT = { $, $$, sv, h, esc, once, watch, showTip, hideTip, story, frame, camLabel, short, RM, T1, POINT_COLOR, barChart, pad, winRange, shotAt, run };
})();
