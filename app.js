/* CaT-Movie project page: core helpers, hero, comparison, gallery, method, cut gate, tables. Data: window.CAT_DATA (data.js). */
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
const run = (name, fn) => { try { fn(); } catch (e) { console.error('[' + name + ']', e); } };
document.documentElement.classList.add('js');

/* Media mirror: pinned CDN on the published build (meta cat-media), same origin otherwise; fall back to same origin on error */
const MEDIA = (document.querySelector('meta[name="cat-media"]') || {}).content || '';
if (MEDIA) {
  document.addEventListener('error', e => {
    const el = e.target;
    if (!el || !/^(VIDEO|IMG|SOURCE)$/.test(el.tagName)) return;
    const u = el.getAttribute('src') || '';
    if (u.startsWith(MEDIA)) el.setAttribute('src', u.slice(MEDIA.length));
    const p = el.getAttribute('poster') || '';
    if (p.startsWith(MEDIA)) el.setAttribute('poster', p.slice(MEDIA.length));
  }, true);
}

const frame = (sid, k, kind) => `${MEDIA}f/${sid}/s${k}${kind === 'last' ? '-last' : ''}.jpg`;
const shotAt = (spans, t) => { for (let i = 0; i < spans.length; i++) if (t < spans[i][1]) return i; return spans.length - 1; };
const short = n => n.replace(/^Character /, 'C').replace(/^Object /, 'O').replace(/^Location /, 'L');
const cap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
const camLabel = cam => {
  if (!cam) return '';
  const p = [];
  if (cam.shot_size) p.push(cap(cam.shot_size.replace(/ shot$/, '')));
  if (cam.angle) p.push(cam.angle.replace(/ shot$/, ''));
  if (cam.movement) p.push(cam.movement);
  return p.join(', ');
};
const pad = (n, w = 2) => String(n).padStart(w, '0');
const winRange = (spans, w) => [spans[w.shots[0] - 1][0], spans[w.shots[w.shots.length - 1] - 1][1]];
const entLabel = (s, n) => {
  const c = s.plan.characters.find(x => x.name === n);
  if (c) return `${short(n)}, ${c.appearance.replace(/^an? /, '').split(',')[0]}`;
  const o = s.plan.objects.find(x => x.name === n);
  return o ? o.appearance.replace(/^an? /, '') : n;
};
/* which recalled frame each target shot of a window reads, and what it shares with it */
const recallUse = (s, w) => w.refs.map((ref, ri) => {
  const src = s.plan.shots[ref.shot - 1];
  const have = new Set(src.present.map(p => p.name));
  const uses = [];
  w.shots.forEach((k, si) => {
    if (!(w.routing[si] || []).includes(ri)) return;
    const sh = s.plan.shots[k - 1];
    uses.push({ shot: k, shared: sh.present.map(p => p.name).filter(n => have.has(n)), place: sh.place === src.place });
  });
  return { ref, ri, uses };
});
const watch = (node, cb, threshold = 0.2, rootMargin = '0px') => {
  if (!node) return null;
  const io = new IntersectionObserver(es => es.forEach(e => cb(e.isIntersecting, e)), { threshold, rootMargin });
  io.observe(node);
  return io;
};
const once = (node, cb, threshold = 0.25) => {
  if (!node) return;
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { io.disconnect(); cb(); } }), { threshold });
  io.observe(node);
};

/* ---------- one reveal for the whole page: figures rise once, headings unveil once, charts draw once ---------- */
const revealCbs = new WeakMap();
const revealIO = new IntersectionObserver(es => es.forEach(e => {
  if (!e.isIntersecting) return;
  const el = e.target;
  revealIO.unobserve(el);
  el.classList.add('in');
  (revealCbs.get(el) || []).forEach(cb => run('reveal', cb));
  revealCbs.delete(el);
}), { rootMargin: '0px 0px -12% 0px' });
function onReveal(el, cb) {
  if (!el) return;
  if (el.classList.contains('in')) { cb(); return; }
  if (!revealCbs.has(el)) revealCbs.set(el, []);
  revealCbs.get(el).push(cb);
}
$$('.fig').forEach(f => revealIO.observe(f));
$$('main section:not(.hero) h2').forEach(hd => { hd.classList.add('rv'); revealIO.observe(hd); });
if (RM) $$('.fig, .rv').forEach(x => x.classList.add('in'));

/* charts are drawn at the pixel width of their column, so their text stays at body scale; redrawn when the column changes */
function chart(host, draw) {
  if (!host) return;
  const fig = host.closest('.fig') || host;
  let w0 = 0, intro = null, revealed = false, played = false;
  const play = () => { if (played || !revealed || !intro) return; played = true; if (!RM) intro(); };
  const render = () => {
    const W = Math.floor(host.getBoundingClientRect().width);
    if (W < 60 || W === w0) return;
    w0 = W;
    host.replaceChildren();
    intro = draw(W) || null;
    if (played) intro = null;
    play();
  };
  new ResizeObserver(() => requestAnimationFrame(render)).observe(host);
  onReveal(fig, () => { revealed = true; play(); });
}

/* text tabs: one control style for every switch on the page */
function setTab(group, btn) {
  $$('button', group).forEach(b => { const on = b === btn; b.classList.toggle('on', on); b.setAttribute('aria-pressed', String(on)); });
}

/* ---------- tooltip ---------- */
const tip = $('#tip');
function showTip(html, ev) {
  tip.innerHTML = html;
  tip.classList.add('on');
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 16, y = ev.clientY + 16;
  if (x + r.width > innerWidth - 12) x = ev.clientX - r.width - 16;
  if (y + r.height > innerHeight - 12) y = ev.clientY - r.height - 16;
  tip.style.left = Math.max(8, x) + 'px';
  tip.style.top = Math.max(8, y) + 'px';
}
const hideTip = () => tip.classList.remove('on');
addEventListener('scroll', hideTip, { passive: true });
const pick = (name, from) => document.dispatchEvent(new CustomEvent('pick', { detail: { name, from } }));

/* ---------- navigation and theme ---------- */
run('nav', function nav() {
  const root = document.documentElement;
  const tLab = $('#themeLabel'), tIcon = $('#themeIcon');
  const sunSvg = '<circle cx="8" cy="8" r="3.2"/><path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M12.6 3.4l-1.4 1.4M4.8 11.2l-1.4 1.4"/>';
  const moonSvg = '<path d="M13.2 10.2A5.5 5.5 0 0 1 5.8 2.8a5.5 5.5 0 1 0 7.4 7.4z"/>';
  function applyTheme(th) {
    root.dataset.theme = th;
    tLab.textContent = th === 'light' ? 'Dark' : 'Light';
    tIcon.innerHTML = th === 'light' ? moonSvg : sunSvg;
    try { localStorage.setItem('cat-theme', th); } catch (e) {}
  }
  let saved = 'light';
  try { saved = localStorage.getItem('cat-theme') || 'light'; } catch (e) {}
  applyTheme(saved === 'dark' ? 'dark' : 'light');
  $('#themeToggle').addEventListener('click', () => applyTheme(root.dataset.theme === 'light' ? 'dark' : 'light'));

  const links = $$('.nav-links a'), nl = $('.nav-links'), ind = h('span', 'nav-ind');
  nl.prepend(ind);
  function mark(id) {
    let hit = null;
    links.forEach(a => { const on = a.getAttribute('href') === '#' + id; a.classList.toggle('on', on); if (on) hit = a; });
    if (!hit) { ind.style.opacity = 0; return; }
    ind.style.opacity = 1;
    ind.style.width = hit.offsetWidth + 'px';
    ind.style.transform = `translateX(${hit.offsetLeft}px)`;
  }
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) mark(e.target.id); }), { rootMargin: '-45% 0px -50% 0px' });
  $$('main > section').forEach(s => io.observe(s));
  addEventListener('resize', () => { const a = $('.nav-links a.on'); if (a) mark(a.getAttribute('href').slice(1)); });
});

/* ---------- hero: film with window progress, three stories ---------- */
const HERO = 'cinedance-010942-b0000-shot03';
const HERO_REELS = [
  { id: 'cinedance-010942-b0000-shot03', label: 'Police Station', video: 'v/hero-police.mp4', poster: 'f/cinedance-010942-b0000-shot03/s1.jpg' },
  { id: 'cinedance-009077-b0004-shot01', label: 'Courtroom', video: 'v/hero-courtroom.mp4', poster: 'f/cinedance-009077-b0004-shot01/s1.jpg' },
  { id: 'film-003995-b0001-shot06', label: 'Cable Car', video: 'v/hero-cablecar.mp4', poster: 'f/film-003995-b0001-shot06/s1.jpg' },
];
run('hero', function hero() {
  const v = $('#heroVideo'), steps = $('#heroSteps'), line = $('#heroLine'), lineNo = $('#heroLineNo');
  const capEl = line.closest('.film-cap'), sw = $('#heroSwitch');
  let reelIdx = 0, s = story(HERO_REELS[0].id), spans = s.spans.ours, segs = [], pb = null;
  let userPaused = false, cur = -1, raf = 0;

  sw.innerHTML = '<span class="label">Story</span>' + HERO_REELS.map((r, i) => `<button type="button" data-i="${i}" class="${i === 0 ? 'on' : ''}" aria-pressed="${i === 0}">${esc(r.label)}</button>`).join('');
  $$('button', sw).forEach(b => b.addEventListener('click', () => {
    const idx = +b.dataset.i;
    if (idx === reelIdx) return;
    userPaused = false;
    loadReel(idx, true);
  }));

  function syncPlayBtn() {
    if (!pb) return;
    pb.innerHTML = v.paused
      ? '<svg viewBox="0 0 10 10"><path d="M2.5 1v8l6.5-4z" fill="currentColor"/></svg>'
      : '<svg viewBox="0 0 10 10"><path d="M2 1h2v8H2zM6 1h2v8H6z" fill="currentColor"/></svg>';
    pb.setAttribute('aria-label', v.paused ? 'Play the film' : 'Pause the film');
  }
  function buildSteps() {
    steps.replaceChildren();
    segs = s.windows.map((w, wi) => {
      const [a, b] = winRange(spans, w);
      const el = h('button', 'fs', `<span>${pad(wi + 1)}</span><i><u></u></i>`);
      el.type = 'button';
      el.style.flex = `${(b - a).toFixed(2)} 1 0`;
      el.setAttribute('aria-label', `Window ${wi + 1}, shots ${w.shots[0]}${w.shots.length > 1 ? ' to ' + w.shots[w.shots.length - 1] : ''}`);
      el.title = `Shots ${w.shots.join(', ')} generated together`;
      el.addEventListener('click', () => { userPaused = false; v.currentTime = a + 0.02; v.play().catch(() => {}); });
      steps.appendChild(el);
      return { el, a, b, fill: $('u', el) };
    });
    pb = h('button', 'ibtn');
    pb.type = 'button';
    pb.addEventListener('click', () => {
      if (v.paused) { userPaused = false; v.play().catch(() => {}); }
      else { userPaused = true; v.pause(); }
    });
    steps.appendChild(pb);
    syncPlayBtn();
  }
  function update() {
    const t = v.currentTime || 0;
    const k = shotAt(spans, t);
    segs.forEach(g => {
      const p = Math.max(0, Math.min(1, (t - g.a) / (g.b - g.a)));
      g.fill.style.width = (p * 100) + '%';
      g.el.classList.toggle('on', t >= g.a && t < g.b);
      g.el.classList.toggle('done', t >= g.b);
    });
    if (k !== cur) {
      cur = k;
      const wi = s.windows.findIndex(w => w.shots.includes(k + 1));
      lineNo.textContent = `Window ${wi + 1}, Shot ${k + 1}`;
      if (RM || !line.textContent) line.textContent = s.requests.shots[k];
      else {
        capEl.classList.add('swapping');
        setTimeout(() => { line.textContent = s.requests.shots[cur]; capEl.classList.remove('swapping'); }, 220);
      }
    }
  }
  function loadReel(idx, playNow) {
    reelIdx = idx;
    const r = HERO_REELS[idx];
    s = story(r.id);
    spans = s.spans.ours;
    cur = -1;
    line.textContent = '';
    setTab(sw, $$('button', sw)[idx]);
    v.setAttribute('aria-label', `CaT-Movie generation of the ${r.label} test story`);
    if (playNow || !(v.getAttribute('src') || '').endsWith(r.video)) {
      v.poster = MEDIA + r.poster;
      v.src = MEDIA + r.video;
      v.load();
    }
    buildSteps();
    update();
    if (playNow && !userPaused) v.play().catch(() => {});
  }
  const loop = () => { update(); raf = requestAnimationFrame(loop); };
  v.addEventListener('play', () => { syncPlayBtn(); cancelAnimationFrame(raf); loop(); });
  v.addEventListener('pause', () => { syncPlayBtn(); cancelAnimationFrame(raf); });
  v.addEventListener('seeked', update);
  buildSteps();
  update();
  watch(v, vis => { if (vis && !userPaused) v.play().catch(() => {}); else if (!vis) v.pause(); }, 0.15);
});

/* a pooled clip fades in only once it has a frame to show, so a cell never goes blank on a slow connection */
const READY = ['loadeddata', 'canplay', 'playing', 'seeked', 'timeupdate'];
function showWhenReady(v) {
  const on = () => { v.classList.add('live'); READY.forEach(ev => v.removeEventListener(ev, on)); };
  if (v.readyState >= 2) on(); else READY.forEach(ev => v.addEventListener(ev, on));
}

/* ---------- 01 comparison: Police Station shots 1, 2, 3, 8 by three generators ---------- */
run('compare', function compare() {
  const fig = $('#cmp'), host = $('#cmpGrid');
  if (!host) return;
  const s = story(HERO);
  const SH = [1, 2, 3, 8];
  const HEAD = [
    ['Medium close-up', 'The bearded man sits at a desk.'],
    ['Close-up', 'The officer looks to the side.'],
    ['Medium close-up', 'The bearded man looks up.'],
    ['Close-up', 'The officer looks to the side.'],
  ];
  /* marks follow the benchmark judge; each verdict carries the matching test-set number */
  const ROWS = [
    { m: 'holocine', who: 'HoloCine', kind: 'joint generation', col: 'var(--holo)', cuts: ['new', 'new'],
      marks: { 2: { kind: 'person', pin: [0.49, 0.27], zoom: [0.49, 0.43, 0.46] }, 8: { kind: 'person', pin: [0.40, 0.19], zoom: [0.40, 0.34, 0.36] } },
      say: 'Cuts where the story asks, but the officer in shots 2 and 8 comes out as two different men. Across a window boundary, joint generators keep only 28&ndash;30% of their characters, against 62&ndash;83% inside one window.' },
    { m: 'storymem', who: 'StoryMem', kind: 'memory-augmented', col: 'var(--mem)', cuts: ['new', 'lost'],
      marks: { 1: { kind: 'outfit', pin: [0.50, 0.62], zoom: [0.52, 0.66, 0.36] }, 3: { kind: 'outfit', pin: [0.42, 0.64], zoom: [0.40, 0.69, 0.36] } },
      say: 'Keeps both faces, but dresses the bearded man differently in shots 1 and 3, and shot 3 repeats the framing of shot 2 instead of cutting. Over the test set it repeats 114 of 674 planned setup changes.' },
    { m: 'ours', who: 'CaT-Movie', kind: 'ours', col: 'var(--ours)', cuts: ['new', 'new'], marks: {},
      say: 'Opens both planned camera setups and keeps both men the same across windows.' },
  ];
  const zoom = (img, cx, cy, c) => {
    const ch = c * 640 / 370;
    return `background-image:url(${img});background-size:${100 / c}% ${100 / ch}%;background-position:${((cx - c / 2) / (1 - c) * 100).toFixed(1)}% ${((cy - ch / 2) / (1 - ch) * 100).toFixed(1)}%`;
  };
  host.innerHTML =
    `<div class="cmp-heads">${SH.map((k, i) => `<button type="button" class="cmp-col" data-k="${k}" aria-label="Play shot ${k} in all three"><b>Shot ${k}<em>${HEAD[i][0]}</em></b><q>${HEAD[i][1]}</q></button>`).join('')}</div>` +
    ROWS.map(r => `<div class="cmp-row" data-m="${r.m}">
      <p class="cmp-say"><button type="button" class="cmp-gen" data-m="${r.m}" aria-pressed="false"><i style="background:${r.col}"></i>${r.who}</button><span class="cmp-kind">${r.kind}.</span>${r.say}</p>
      <div class="cmp-strip">${SH.map(k => {
        const img = `${MEDIA}f/qual/${r.m}-s${k}.jpg`;
        const mk = r.marks[k];
        return `<div class="qc" data-k="${k}"${mk ? ` data-pair="${r.m}-${mk.kind}"` : ''} style="background-image:url(${img})">${mk ? `<span class="pin ${mk.kind}" style="left:${mk.pin[0] * 100}%;top:${mk.pin[1] * 100}%"><svg viewBox="0 0 20 24"><use href="#${mk.kind === 'outfit' ? 'i-outfit' : 'i-neq'}"/></svg></span><span class="zoom" style="${zoom(img, ...mk.zoom)}"></span>` : ''}<i class="qbar"></i></div>`;
      }).join('')}${r.cuts.map((c, i) => `<svg class="qcut${c === 'lost' ? ' lost' : ''}" data-b="${i + 1}" style="left:calc(${(i + 1) * 25}% + ${(i + 1) * 2 - 4}px)" viewBox="0 0 24 24"><use href="#${c === 'lost' ? 'i-scissor-x' : 'i-scissor'}"/></svg>`).join('')}</div>
    </div>`).join('');

  const clip = (m, k) => `${MEDIA}v/cmp/${m}-s${k}.mp4`;
  const pool = new Map();
  const make = src => { const v = document.createElement('video'); v.muted = true; v.playsInline = true; v.preload = 'auto'; v.className = 'qv'; v.src = src; return v; };
  watch(fig, vis => { if (vis) ROWS.forEach(r => SH.forEach(k => { const src = clip(r.m, k); if (!pool.has(src)) pool.set(src, make(src)); })); }, 0, '900px');

  let live = [];
  function stop() {
    live.forEach(({ v, cell, src }) => {
      v.pause(); v.onended = null; v.loop = false; v.classList.remove('live'); v.remove();
      try { v.currentTime = 0; } catch (e) {}
      pool.set(src, v);
      cell.classList.remove('playing');
      $('.qbar', cell).style.width = '0';
    });
    live = [];
  }
  /* each clip is cut at its shot's end at the generator's native frame rate, so playback stops on the shot boundary */
  function start(cell, onDone) {
    const m = cell.closest('.cmp-row').dataset.m;
    const k = +cell.dataset.k;
    const [a, b] = s.spans[m][k - 1];
    const L = b - a;
    const src = clip(m, k);
    const v = pool.get(src) || make(src);
    pool.delete(src);
    v.loop = !onDone;
    try { v.currentTime = 0; } catch (e) {}
    cell.prepend(v);
    cell.classList.add('playing');
    showWhenReady(v);
    const bar = $('.qbar', cell);
    const x = { v, cell, src, done: false };
    const show = t => { bar.style.width = (Math.min(1, t / L) * 100) + '%'; };
    const rvfc = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
    (function tick() {
      if (!v.isConnected || x.done) return;
      show(v.currentTime || 0);
      if (rvfc) v.requestVideoFrameCallback(tick); else requestAnimationFrame(tick);
    })();
    if (onDone) v.onended = () => { if (x.done) return; x.done = true; show(L); onDone(); };
    v.play().catch(() => {});
    live.push(x);
  }

  const tourBtn = $('#cmpTour');
  let ci = -1, touring = false, held = RM, timer = 0, guard = 0, hoverCell = null, hoverT = 0, pinnedCell = null;
  if (!RM) fig.classList.add('tour');
  const revealCut = i => $$(`.qcut[data-b="${i}"]`, host).forEach(q => q.classList.add('in'));
  const revealMarks = i => $$(`.qc[data-k="${SH[i]}"]`, host).forEach(c => c.classList.add('marked'));
  function setBtn() {
    tourBtn.innerHTML = touring
      ? '<svg viewBox="0 0 14 14"><path d="M4 2.5v9M10 2.5v9" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>Pause'
      : '<svg viewBox="0 0 14 14"><path d="M3.5 2v10l8.5-5z" fill="currentColor"/></svg>Play shot by shot';
    tourBtn.setAttribute('aria-pressed', String(touring));
  }
  const heads = () => $$('.cmp-col', host);
  function playCol(i) {
    ci = i; stop(); revealCut(i); clearTimeout(guard);
    heads().forEach((b, j) => b.classList.toggle('on', j === i));
    const cells = $$(`.qc[data-k="${SH[i]}"]`, host);
    let done = 0;
    const next = () => {
      clearTimeout(guard); clearTimeout(timer); stop(); revealMarks(i);
      timer = setTimeout(() => {
        if (!touring) return;
        if (i === SH.length - 1) { stopTour(true); ci = -1; return; }
        playCol(i + 1);
      }, 1100);
    };
    cells.forEach(c => start(c, () => { if (++done === cells.length) next(); }));
    const longest = Math.max(...ROWS.map(r => s.spans[r.m][SH[i] - 1][1] - s.spans[r.m][SH[i] - 1][0]));
    guard = setTimeout(next, (longest + 4) * 1000);
  }
  function startTour() { if (held || touring) return; touring = true; setBtn(); playCol(ci < 0 ? 0 : (ci + 1) % SH.length); }
  function stopTour(hold) {
    touring = false; clearTimeout(timer); clearTimeout(guard);
    if (hold) { held = true; fig.classList.remove('tour'); }
    heads().forEach(b => b.classList.remove('on'));
    setBtn();
  }
  tourBtn.addEventListener('click', () => { if (touring) { stopTour(true); stop(); } else { held = false; pinnedCell = null; stop(); startTour(); } });
  watch(fig, vis => { if (vis) startTour(); else if (touring) { stopTour(false); stop(); } }, 0.35);

  $$('.qc', host).forEach(c => {
    c.tabIndex = 0; c.setAttribute('role', 'button');
    c.setAttribute('aria-label', `Play shot ${c.dataset.k} of ${c.closest('.cmp-row').dataset.m}`);
    const toggle = () => { const was = !touring && pinnedCell === c; stopTour(true); stop(); hoverCell = null; pinnedCell = was ? null : c; if (!was) start(c); };
    c.addEventListener('click', toggle);
    c.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
    c.addEventListener('pointerenter', e => { if (e.pointerType !== 'mouse' || touring || !held || pinnedCell) return; clearTimeout(hoverT); hoverT = setTimeout(() => { stop(); hoverCell = c; start(c); }, 100); });
    c.addEventListener('pointerleave', e => { if (e.pointerType !== 'mouse') return; clearTimeout(hoverT); if (!pinnedCell && hoverCell === c) { stop(); hoverCell = null; } });
  });
  heads().forEach(b => b.addEventListener('click', () => {
    const k = b.dataset.k;
    const cells = $$(`.qc[data-k="${k}"]`, host);
    const was = !touring && live.length === cells.length && live.every(x => x.cell.dataset.k === k);
    stopTour(true); stop(); pinnedCell = null;
    if (!was) { cells.forEach(c => start(c)); b.classList.add('on'); }
  }));
  $$('.qc[data-pair]', host).forEach(c => {
    const mates = () => $$(`.qc[data-pair="${c.dataset.pair}"]`, host);
    c.addEventListener('pointerenter', () => mates().forEach(x => x.classList.add('pair-hl')));
    c.addEventListener('pointerleave', () => mates().forEach(x => x.classList.remove('pair-hl')));
  });
  /* a generator name isolates its row and drops its marks again */
  const gens = $$('.cmp-gen', host);
  function focusRow(m) {
    gens.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.m === m)));
    $$('.cmp-row', host).forEach(r => r.classList.toggle('dim', m != null && r.dataset.m !== m));
    if (!m) return;
    stopTour(true);
    $$(`.cmp-row[data-m="${m}"] .qc[data-pair]`, host).forEach(c => { c.classList.remove('redrop'); void c.offsetWidth; c.classList.add('redrop'); });
    $$(`.cmp-row[data-m="${m}"] .qcut`, host).forEach(q => { q.classList.remove('in'); void q.getBoundingClientRect(); q.classList.add('in'); });
  }
  gens.forEach(b => b.addEventListener('click', () => focusRow(b.getAttribute('aria-pressed') === 'true' ? null : b.dataset.m)));
  setBtn();
});

/* ---------- 02 gallery: eight stories, four generators in lockstep ---------- */
const MET = [
  { key: 'ours', name: 'CaT-Movie', color: 'var(--ours)' },
  { key: 'holocine', name: 'HoloCine', color: 'var(--holo)' },
  { key: 'multishotmaster', name: 'MultiShotMaster', color: 'var(--msm)' },
  { key: 'storymem', name: 'StoryMem', color: 'var(--mem)' },
];
run('theater', function theater() {
  const fig = $('#theater'), tabsEl = $('#storyTabs'), screens = $('#screens'), bar = $('#shotbar'), handle = $('#wipeHandle');
  const playBtn = $('#tPlay'), timeEl = $('#tTime'), capK = $('#capK'), capLine = $('#capLine');
  const wipeSel = $('#wipeSel'), wipeLab = $('#wipeLab'), layout = $('#layoutTabs');
  let si = 0, mode = 'grid', focus = 'ours', wipeB = 'storymem', userPaused = false, inView = false, loaded = false, raf = 0, curShot = -1;

  const tiles = MET.map(m => {
    const t = h('div', 'tile', `<video muted playsinline preload="none"></video><span class="name"><i style="background:${m.color}"></i>${m.name}</span><span class="shotno">Shot 1</span>`);
    t.dataset.m = m.key; t.tabIndex = 0; t.setAttribute('role', 'button'); t.setAttribute('aria-label', `${m.name}: enlarge`);
    t.style.viewTransitionName = 'tile-' + m.key;
    const act = () => {
      if (mode === 'wipe') return;
      if (mode === 'focus' && focus === m.key) { setMode('grid'); return; }
      focus = m.key; setMode('focus');
    };
    t.addEventListener('click', act);
    t.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); act(); } });
    screens.insertBefore(t, handle);
    const v = $('video', t);
    v.muted = true; v.playsInline = true;
    return { m, t, v, no: $('.shotno', t) };
  });
  const master = tiles[0].v;

  tabsEl.innerHTML = D.stories.map((s, i) => `<button type="button" data-i="${i}" class="${i === 0 ? 'on' : ''}" aria-pressed="${i === 0}" title="${s.spans.ours.length} shots, ${s.seconds.toFixed(1)} s">${esc(s.title)}</button>`).join('');
  $$('button', tabsEl).forEach(b => b.addEventListener('click', () => {
    const i = +b.dataset.i;
    if (si === i) return;
    si = i; userPaused = false;
    setTab(tabsEl, b);
    load(true);
  }));

  const S = () => D.stories[si];
  function load(autoplay) {
    loaded = true;
    const s = S();
    curShot = -1;
    tiles.forEach(({ m, v }) => {
      v.poster = `${MEDIA}p/${s.id}/${m.key}.jpg`;
      v.preload = 'auto';
      v.src = `${MEDIA}v/${s.id}/${m.key}.mp4`;
      v.load();
    });
    bar.replaceChildren();
    s.spans.ours.forEach(([a, b], k) => {
      const btn = h('button', '', `<i></i><span>${k + 1}</span>`);
      btn.type = 'button';
      btn.style.flexGrow = (b - a).toFixed(2);
      btn.title = `Shot ${k + 1}: ${s.requests.shots[k]}`;
      btn.setAttribute('aria-label', `Shot ${k + 1}`);
      btn.addEventListener('click', () => { seekShot(k); if (master.paused) { userPaused = false; play(); } });
      bar.appendChild(btn);
    });
    update(true);
    if (autoplay && !userPaused) play();
  }
  function play() { tiles.forEach(({ v }) => v.play().catch(() => {})); setIcon(true); loop(); }
  function pause() { tiles.forEach(({ v }) => v.pause()); setIcon(false); cancelAnimationFrame(raf); }
  function setIcon(p) {
    $('#tPlayIcon').innerHTML = p ? '<path d="M3 2h3v10H3zM8 2h3v10H8z" fill="currentColor"/>' : '<path d="M3 1.5v11l9-5.5z" fill="currentColor"/>';
    playBtn.setAttribute('aria-label', p ? 'Pause' : 'Play');
  }
  function seekShot(k) {
    const s = S();
    tiles.forEach(({ m, v }) => { v.currentTime = s.spans[m.key][k][0] + 0.03; });
    update(true);
  }
  function loop() { cancelAnimationFrame(raf); const step = () => { update(); raf = requestAnimationFrame(step); }; raf = requestAnimationFrame(step); }
  function update(force) {
    const s = S(), t = master.currentTime || 0;
    tiles.forEach(({ m, v, no }, i) => {
      if (i && !master.paused && !v.ended && Math.abs(v.currentTime - t) > 0.3 && t < (v.duration || 0) - 0.1) v.currentTime = t;
      const k = shotAt(s.spans[m.key], v.currentTime || 0);
      no.textContent = `Shot ${k + 1}/${s.spans[m.key].length}`;
    });
    const k = shotAt(s.spans.ours, t);
    timeEl.textContent = `${t.toFixed(1)} / ${(master.duration || s.seconds).toFixed(1)} s`;
    const btns = bar.children;
    for (let i = 0; i < btns.length; i++) {
      const [a, b] = s.spans.ours[i];
      btns[i].classList.toggle('on', i === k);
      btns[i].firstChild.style.width = (Math.max(0, Math.min(1, (t - a) / (b - a))) * 100) + '%';
    }
    if (k !== curShot || force) { curShot = k; capK.textContent = `Shot ${k + 1} of ${s.spans.ours.length}`; capLine.textContent = s.requests.shots[k]; }
  }
  master.addEventListener('ended', () => { tiles.forEach(({ v }) => { v.currentTime = 0; }); if (!userPaused && inView) play(); });
  master.addEventListener('seeked', () => update(true));
  playBtn.addEventListener('click', () => { if (master.paused) { userPaused = false; play(); } else { userPaused = true; pause(); } });

  function setMode(md) {
    if (document.startViewTransition && !RM && md !== mode && (md === 'focus' || mode === 'focus')) {
      document.startViewTransition(() => applyMode(md));
      return;
    }
    applyMode(md);
  }
  function applyMode(md) {
    const prev = mode;
    mode = md;
    setTab(layout, $(`[data-mode="${md}"]`, layout));
    screens.classList.toggle('focus', md === 'focus');
    screens.classList.toggle('wipe', md === 'wipe');
    tiles.forEach(({ m, t }) => {
      t.classList.toggle('big', md === 'focus' && m.key === focus);
      t.classList.toggle('wipe-a', md === 'wipe' && m.key === 'ours');
      t.classList.toggle('wipe-b', md === 'wipe' && m.key === wipeB);
      t.setAttribute('aria-label', `${m.name}: ${md === 'focus' && m.key === focus ? 'back to grid' : 'enlarge'}`);
    });
    if (md === 'focus') screens.prepend(tiles.find(x => x.m.key === focus).t);
    else tiles.forEach(x => screens.insertBefore(x.t, handle));
    wipeLab.hidden = md !== 'wipe';
    if (md === 'wipe' && prev !== 'wipe' && !RM) sweep();
  }
  $$('button', layout).forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  wipeSel.value = wipeB;
  wipeSel.addEventListener('change', () => { wipeB = wipeSel.value; applyMode('wipe'); });
  function setCut(p) { screens.style.setProperty('--cut', (Math.max(0.03, Math.min(0.97, p)) * 100) + '%'); }
  function sweep() {
    const t0 = performance.now();
    (function f(t) { const p = (t - t0) / 1100; if (p > 1) { setCut(0.5); return; } setCut(0.5 + Math.sin(p * Math.PI * 2) * 0.22 * (1 - p)); requestAnimationFrame(f); })(t0);
  }
  let dragging = false;
  screens.addEventListener('pointerdown', e => { if (mode !== 'wipe') return; dragging = true; screens.setPointerCapture(e.pointerId); move(e); });
  screens.addEventListener('pointermove', e => { if (mode === 'wipe' && (dragging || e.pointerType === 'mouse')) move(e); });
  screens.addEventListener('pointerup', () => { dragging = false; });
  function move(e) { const r = screens.getBoundingClientRect(); setCut((e.clientX - r.left) / r.width); }
  setCut(0.5);

  document.addEventListener('keydown', e => { if (e.key === 'Escape' && mode !== 'grid' && inView) setMode('grid'); });
  fig.addEventListener('keydown', e => {
    if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    const s = S();
    if (e.key === ' ' && e.target === fig) { e.preventDefault(); playBtn.click(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); seekShot(Math.min(s.spans.ours.length - 1, shotAt(s.spans.ours, master.currentTime) + 1)); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); seekShot(Math.max(0, shotAt(s.spans.ours, master.currentTime) - 1)); }
  });

  tiles.forEach(({ m, v }) => { v.poster = `${MEDIA}p/${S().id}/${m.key}.jpg`; });
  watch(fig, vis => { if (vis && !loaded) load(false); }, 0, '600px');
  watch(fig, vis => {
    inView = vis;
    if (vis && !userPaused) play();
    else if (!vis) pause();
  }, 0.25);
});

/* ---------- 03 method: the plan for every shot (stages 1-3), recall in every window (stages 4-7) ---------- */
/* face crops for the four characters of Police Station, shared with the overview's storyboard */
const AVATAR = {
  'Character 1': { k: 1, sz: '355% 200%', pos: '22% 2%' },
  'Character 2': { k: 2, sz: '323% 182%', pos: '66% 5%' },
  'Character 3': { k: 10, sz: '277% 156%', pos: '66% 17%' },
  'Character 4': { k: 6, sz: '290% 164%', pos: '22% 18%' },
};
const avatarStyle = n => { const a = AVATAR[n]; return a ? `background-image:url(${frame(HERO, a.k)});background-size:${a.sz};background-position:${a.pos}` : ''; };
const charTip = (s, n) => { const c = s.plan.characters.find(x => x.name === n); return c ? `<b>${esc(short(n))}</b> ${esc(c.appearance.replace(/^an? /, ''))}${c.outfit ? `<br>wearing ${esc(c.outfit.replace(/^an? /, ''))}` : ''}` : esc(n); };
run('method', function method() {
  const planFig = $('#planFig'), renderFig = $('#renderFig');
  if (!planFig || !renderFig) return;
  const s = story(HERO);
  const spans = s.spans.ours;
  const N = s.windows.length;
  const dur = k => spans[k - 1][1] - spans[k - 1][0];
  const label = n => /^Character/.test(n) ? short(n) : entLabel(s, n);
  const KEY_SHOTS = [1, 2, 5, 8];

  /* the registry itself sits in the overview's storyboard; here each request line sits beside the plan compiled from it */
  let showAll = false;
  function renderPlan() {
    const pick = showAll ? s.plan.shots : s.plan.shots.filter(sh => KEY_SHOTS.includes(sh.shot));
    const changeTag = sh => {
      if (!sh.changes || !sh.changes.length) return '';
      const ch = sh.changes[0];
      return `<span class="tag chg">${esc(short(ch.name))} ${esc(ch.what)}: ${ch.before ? esc(ch.before.split(',')[0]) + ' → ' : ''}${esc(ch.after)}</span>`;
    };
    planFig.innerHTML = `<div class="plan-rows">
        <span class="label">Shot request</span><span></span><span class="label">Compiled shot plan</span>
        ${pick.map(sh => {
          const line = esc(s.requests.shots[sh.shot - 1]).replace(/Character (\d+)/g, '<b data-e="Character $1">Character $1</b>');
          const who = sh.present.filter(p => /^Character/.test(p.name)).map(p => `<span class="tag ours" data-e="${esc(p.name)}">${esc(short(p.name))}</span>`).join('');
          return `<div class="req card" data-k="${sh.shot}" tabindex="0"><span class="no">S${pad(sh.shot)}</span>${line}</div>
          <span class="plan-arrow" aria-hidden="true">→</span>
          <div class="shp card" data-k="${sh.shot}" data-es="${esc(sh.present.map(p => p.name).join('|'))}" tabindex="0">
            <div class="shp-head"><b>Shot ${pad(sh.shot)}</b><span>${esc(camLabel(sh.camera))}</span></div>
            <div class="chips-row">${who}${changeTag(sh)}</div>
          </div>`;
        }).join('')}
      </div>
      <div class="plan-foot"><p class="cap">Police Station, ${pick.length} of ${s.plan.shots.length} shots. Point at a character to find every shot that shows them.</p><button type="button" class="tbtn" id="planAll">${showAll ? 'Show 4 key shots' : `Show all ${s.plan.shots.length} shots`}<svg viewBox="0 0 10 10"><path d="${showAll ? 'M2 6.5 5 3.5l3 3' : 'M2 3.5 5 6.5l3-3'}" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg></button></div>`;
    $('#planAll', planFig).addEventListener('click', () => { showAll = !showAll; renderPlan(); });
  }
  const clear = () => { planFig.classList.remove('linking'); $$('.hl, b.m', planFig).forEach(x => x.classList.remove('hl', 'm')); };
  const byEnt = name => {
    planFig.classList.add('linking');
    $$('.req', planFig).forEach(x => x.classList.toggle('hl', !!x.querySelector(`b[data-e="${name}"]`)));
    $$('.shp', planFig).forEach(x => x.classList.toggle('hl', x.dataset.es.split('|').includes(name)));
    $$('.req b[data-e]', planFig).forEach(b => b.classList.toggle('m', b.dataset.e === name));
  };
  const byShot = k => {
    const card = $(`.shp[data-k="${k}"]`, planFig);
    const es = card ? card.dataset.es.split('|') : [];
    planFig.classList.add('linking');
    $$('.req, .shp', planFig).forEach(x => x.classList.toggle('hl', x.dataset.k === k));
    $$('.req b[data-e]', planFig).forEach(b => b.classList.toggle('m', es.includes(b.dataset.e)));
  };
  /* one delegated tracer: a character name or tag traces that character, anywhere else on a row traces that shot */
  const trace = el => {
    const ent = el.closest && el.closest('[data-e]'), row = el.closest && el.closest('[data-k]');
    if (ent && planFig.contains(ent)) byEnt(ent.dataset.e); else if (row && planFig.contains(row)) byShot(row.dataset.k); else clear();
  };
  planFig.addEventListener('pointerover', e => trace(e.target));
  planFig.addEventListener('focusin', e => trace(e.target));
  planFig.addEventListener('pointerleave', () => { clear(); hideTip(); });
  planFig.addEventListener('focusout', clear);
  planFig.addEventListener('pointermove', e => { const ent = e.target.closest('[data-e]'); if (ent) showTip(charTip(s, ent.dataset.e), e); else hideTip(); });
  renderPlan();

  /* Step 2: one timeline. Windows are drawn proportional to seconds at equal height; generated windows are in colour,
     later ones still blank; the recalled earlier shots light up in place and route down into the shots that read them. */
  renderFig.innerHTML = `<div class="rs-top">
      <div class="rs-count" id="rsCount"></div>
      <p class="rs-note" id="rsNote"></p>
      <div class="rs-nav"><button type="button" class="ibtn" data-d="-1" aria-label="Previous window"><svg viewBox="0 0 10 10"><path d="M6.5 1.5 3 5l3.5 3.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg></button><button type="button" class="ibtn" data-d="1" aria-label="Next window"><svg viewBox="0 0 10 10"><path d="M3.5 1.5 7 5 3.5 8.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg></button></div>
    </div>
    <div class="rs-stage" id="rsStage">
      <div class="tl" role="group" aria-label="Generation windows">${s.windows.map((w, i) => {
        const d = w.shots.reduce((t, k) => t + dur(k), 0);
        return `<button type="button" class="tl-win" data-w="${i}" aria-label="Window ${i + 1}, ${d.toFixed(1)} seconds" style="flex-grow:${d.toFixed(2)}"><span class="tl-lab"><b><span class="w-long">Window </span><span class="w-short">W</span>${i + 1}</b><span>${d.toFixed(1)} s</span></span><span class="tl-strip">${w.shots.map(k => `<i class="tl-shot" data-k="${k}" style="flex:${dur(k).toFixed(2)} 1 0;background-image:url(${frame(HERO, k)})"><em>${k}</em></i>`).join('')}</span></button>`;
      }).join('')}</div>
      <div class="rs-lane"></div>
      <div class="rs-cards" id="rsCards"></div>
      <svg class="rs-arcs" aria-hidden="true"></svg>
    </div>`;
  const wins = $$('.tl-win', renderFig), stage = $('#rsStage', renderFig), cards = $('#rsCards', renderFig);
  const count = $('#rsCount', renderFig), note = $('#rsNote', renderFig), arcs = $('.rs-arcs', renderFig);
  const navBtns = $$('.rs-nav .ibtn', renderFig);
  let wi = -1, touched = false, autoT = 0;

  function setWin(i, opts = {}) {
    if (i === wi || i < 0 || i >= N) return;
    const forward = i > wi;
    wi = i;
    const w = s.windows[i];
    wins.forEach((e, j) => {
      e.classList.toggle('past', j < i); e.classList.toggle('on', j === i); e.classList.toggle('future', j > i);
      e.setAttribute('aria-pressed', String(j === i));
      e.classList.remove('fill');
    });
    if (forward && !RM && opts.fill !== false) { void wins[i].offsetWidth; wins[i].classList.add('fill'); }
    navBtns[0].disabled = i === 0; navBtns[1].disabled = i === N - 1;
    count.innerHTML = `Window ${i + 1}<span>of ${N}</span>`;
    const used = recallUse(s, w).filter(u => u.uses.length);
    const recalled = new Set(used.map(u => u.ref.shot));
    $$('.tl-shot', renderFig).forEach(c => c.classList.toggle('rec', recalled.has(+c.dataset.k)));
    const range = w.shots.length > 1 ? `Shots ${w.shots[0]}–${w.shots[w.shots.length - 1]} render together.` : `Shot ${w.shots[0]} renders on its own.`;
    note.textContent = used.length
      ? `${range} ${recalled.size === 1 ? 'One earlier shot is' : recalled.size + ' earlier shots are'} recalled, each routed only into the shots that show its character or setting.`
      : `${range} Nothing has been generated yet, so the first window starts from the storyboard text alone.`;
    cards.innerHTML = w.shots.map(k => {
      const reads = used.flatMap(u => u.uses.filter(x => x.shot === k).map(x => ({ src: u.ref.shot, what: x.shared.length ? x.shared.map(label).join(', ') : 'setting' })));
      return `<div class="rs-card card" data-k="${k}">
        <i style="background-image:url(${frame(HERO, k)})"></i>
        <div>
          <b>Shot ${k}<span>${esc(camLabel(s.plan.shots[k - 1].camera))}</span></b>
          <span class="rs-q">${esc(s.requests.shots[k - 1].replace(/^[A-Za-z -]+shot\. /, ''))}</span>
          <span class="chips-row">${reads.length ? reads.map(r => `<span class="tag ours" data-src="${r.src}">← Shot ${r.src}: ${esc(r.what)}</span>`).join('') : '<span class="label">text conditioning only</span>'}</span>
        </div>
      </div>`;
    }).join('');
    if (!RM) { cards.classList.remove('swap'); void cards.offsetWidth; cards.classList.add('swap'); }
    $$('.tag[data-src], .tl-shot.rec', renderFig).forEach(x => {
      const src = x.dataset.src || x.dataset.k;
      x.onpointerenter = () => {
        stage.classList.add('tracing');
        $$('.rs-arc', arcs).forEach(p => p.classList.toggle('hl', p.dataset.src === src));
        $$(`.tl-shot[data-k="${src}"]`, renderFig).forEach(p => p.classList.add('hl'));
      };
      x.onpointerleave = () => { stage.classList.remove('tracing'); $$('.hl', renderFig).forEach(p => p.classList.remove('hl')); };
    });
    requestAnimationFrame(() => drawArcs(!RM));
  }
  function drawArcs(animate) {
    arcs.innerHTML = '';
    if (getComputedStyle($('.rs-lane', renderFig)).display === 'none') return;
    const r0 = stage.getBoundingClientRect();
    if (!r0.width) return;
    arcs.setAttribute('viewBox', `0 0 ${r0.width.toFixed(1)} ${r0.height.toFixed(1)}`);
    const chips = $$('.tag[data-src]', cards), per = new Map(), seen = new Map();
    chips.forEach(c => { const k = c.closest('.rs-card').dataset.k; per.set(k, (per.get(k) || 0) + 1); });
    chips.forEach((chip, n) => {
      const srcEl = $(`.tl-shot[data-k="${chip.dataset.src}"]`, renderFig), dstEl = chip.closest('.rs-card');
      if (!srcEl || !dstEl) return;
      const a = srcEl.getBoundingClientRect(), b = dstEl.getBoundingClientRect();
      const tk = dstEl.dataset.k, j = seen.get(tk) || 0, m = per.get(tk) || 1;
      seen.set(tk, j + 1);
      const x1 = a.left + a.width / 2 - r0.left, y1 = a.bottom - r0.top + 3;
      const x2 = b.left - r0.left + b.width * (j + 1) / (m + 1), y2 = b.top - r0.top - 4;
      const ym = (y1 + y2) / 2;
      const d = `M${x1.toFixed(1)} ${y1.toFixed(1)} C${x1.toFixed(1)} ${ym.toFixed(1)} ${x2.toFixed(1)} ${ym.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`;
      const p = sv('path', { d, class: 'rs-arc', 'data-src': chip.dataset.src }, arcs);
      const dot = sv('circle', { cx: x2.toFixed(1), cy: y2.toFixed(1), r: 3.2, class: 'rs-end' }, arcs);
      if (animate) {
        const L = p.getTotalLength();
        p.style.strokeDasharray = L;
        p.animate([{ strokeDashoffset: L }, { strokeDashoffset: 0 }], { duration: 550, delay: 120 + n * 60, easing: 'cubic-bezier(.2,.7,.2,1)', fill: 'backwards' });
        dot.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 200, delay: 600 + n * 60, fill: 'backwards' });
      }
    });
  }
  const take = () => { touched = true; clearTimeout(autoT); };
  wins.forEach(b => b.addEventListener('click', () => { take(); setWin(+b.dataset.w); }));
  navBtns.forEach(b => b.addEventListener('click', () => { take(); setWin(wi + +b.dataset.d); }));
  renderFig.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); take(); setWin(wi + (e.key === 'ArrowRight' ? 1 : -1)); const b = wins[wi]; if (b) b.focus(); }
  });
  let w0 = 0;
  new ResizeObserver(() => { const w = stage.getBoundingClientRect().width; if (Math.abs(w - w0) > 1) { w0 = w; requestAnimationFrame(() => drawArcs(false)); } }).observe(stage);
  /* on first view the story renders window by window once, then rests on Window 4 (Shots 8 and 9), the officer recall from the comparison */
  if (RM) setWin(3, { fill: false });
  else {
    setWin(0, { fill: false });
    onReveal(renderFig, () => {
      const go = j => { if (touched) return; setWin(j); if (j < 3) autoT = setTimeout(() => go(j + 1), 1500); };
      autoT = setTimeout(() => go(1), 1300);
    });
  }
});

/* ---------- 04 cut gate: the Kitchen example from the paper's teaser, playable ---------- */
run('gate', function gate() {
  const J = $('#gateJoint'), M = $('#gateMem'), root = $('#gate'), sw = $('#gateSwitch');
  if (!J || !M || !root || !sw) return;
  const F = k => `${MEDIA}f/teaser/${k}.jpg`;
  const crop = (img, [x0, y0, x1, y1]) => {
    const w = x1 - x0, hh = y1 - y0;
    return `background-image:url(${img});background-size:${(832 / w * 100).toFixed(1)}% ${(480 / hh * 100).toFixed(1)}%;background-position:${w < 832 ? (x0 / (832 - w) * 100).toFixed(1) : 0}% ${hh < 480 ? (y0 / (480 - hh) * 100).toFixed(1) : 0}%`;
  };
  const pin = (x, y) => `<span class="neq" style="left:${x}%;top:${y}%"><svg width="21" height="25" viewBox="0 0 20 24"><use href="#i-neq"/></svg></span>`;
  J.innerHTML = `<div class="gstrip span2"><div class="gframe" style="background-image:url(${F('joint-s03')})"></div><div class="gframe" style="background-image:url(${F('joint-s04')})">${pin(50, 24)}<span class="zoom" style="${crop(F('joint-s04'), [220, 0, 640, 480])}"></span></div><svg class="scissor" style="left:50%" viewBox="0 0 24 24"><use href="#i-scissor"/></svg></div>` +
    `<div class="gstrip"><div class="gframe" style="background-image:url(${F('joint-s06')})">${pin(41, 24)}<span class="zoom" style="${crop(F('joint-s06'), [205, 30, 505, 373])}"></span></div></div>`;
  M.innerHTML = `<div class="gstrip"><div class="gframe" style="background-image:url(${F('memory-s03')})"></div></div>` +
    `<div class="gstrip"><div class="gframe" style="background-image:url(${F('memory-s04')})"></div><span class="gate-fail"><span>gate fails, setup kept</span></span></div>` +
    `<div class="gstrip"><div class="gframe" style="background-image:url(${F('memory-s06')})"></div></div>` +
    `<svg class="scissor" style="left:calc(33.333% - 1.33px)" viewBox="0 0 24 24"><use href="#i-scissor-x"/></svg>`;

  const COLS = ['3', '4', '6'];
  const cells = $$('.gframe', root);
  let pinned = null, hoverT = 0;
  cells.forEach((c, i) => {
    c.dataset.clip = `${i < 3 ? 'joint' : 'memory'}-s0${COLS[i % 3]}`;
    c.dataset.col = COLS[i % 3];
    c.tabIndex = 0; c.setAttribute('role', 'button');
    c.setAttribute('aria-label', `Play shot ${COLS[i % 3]}, ${i < 3 ? 'joint generation' : 'memory-augmented'}`);
  });
  const heads = $$('.gate-cols .ch', root);
  const FROM = { 3: 2.5, 4: 0.5, 6: 1.5 }, pool = new Map();
  const make = cl => { const v = document.createElement('video'); v.className = 'gv'; v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'auto'; v.src = `${MEDIA}v/teaser/${cl}.mp4`; return v; };
  watch(root, vis => { if (vis) cells.forEach(c => { if (!pool.has(c.dataset.clip) && !$('.gv', c)) pool.set(c.dataset.clip, make(c.dataset.clip)); }); }, 0, '900px');
  function play(c) {
    if (c.classList.contains('playing')) return;
    const v = pool.get(c.dataset.clip) || make(c.dataset.clip);
    pool.delete(c.dataset.clip);
    try { v.currentTime = FROM[c.dataset.col]; } catch (e) {}
    c.prepend(v); c.classList.add('playing');
    showWhenReady(v);
    v.play().catch(() => {});
  }
  function halt(c) { const v = $('.gv', c); if (v) { v.pause(); v.classList.remove('live'); v.remove(); pool.set(c.dataset.clip, v); } c.classList.remove('playing'); }
  const haltAll = () => cells.forEach(halt);
  cells.forEach(c => {
    const toggle = () => { const was = pinned === c; haltAll(); heads.forEach(hd => hd.classList.remove('on')); pinned = was ? null : c; if (!was) play(c); };
    c.addEventListener('click', toggle);
    c.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
    c.addEventListener('pointerenter', e => { if (e.pointerType !== 'mouse' || pinned) return; clearTimeout(hoverT); hoverT = setTimeout(() => play(c), 40); });
    c.addEventListener('pointerleave', e => { if (e.pointerType !== 'mouse') return; clearTimeout(hoverT); if (!pinned) halt(c); });
  });
  heads.forEach(hd => hd.addEventListener('click', () => {
    const col = hd.dataset.col, was = pinned === col;
    haltAll(); heads.forEach(x => x.classList.toggle('on', !was && x === hd));
    pinned = was ? null : col;
    if (!was) cells.filter(c => c.dataset.col === col).forEach(play);
  }));
  watch(root, vis => { if (!vis) { haltAll(); pinned = null; heads.forEach(x => x.classList.remove('on')); } }, 0.1);

  let touched = false;
  function set(on) {
    const rows = [$('#srOff'), $('#srOn')];
    const before = rows.map(r => r.getBoundingClientRect().top);
    sw.setAttribute('aria-checked', on);
    root.classList.toggle('gate-on', on);
    $('#gateState').textContent = on ? 'on' : 'off';
    $('#srOff').classList.toggle('dim', on);
    $('#srOn').classList.toggle('dim', !on);
    if (!RM) rows.forEach((r, i) => {
      const dy = before[i] - r.getBoundingClientRect().top;
      if (dy) r.animate([{ transform: `translateY(${dy}px)` }, { transform: 'none' }], { duration: 380, easing: 'cubic-bezier(.2,0,0,1)' });
    });
  }
  sw.addEventListener('click', () => { touched = true; set(sw.getAttribute('aria-checked') !== 'true'); });
  once(root, () => { if (!RM) setTimeout(() => { if (!touched) set(true); }, 1600); }, 0.5);
});

/* ---------- 05 Table 1: sortable; columns carry their benchmark question, rows answer the Pareto chart ---------- */
const T1 = [
  { g: 'Per-shot generation' },
  { name: 'Wan2.1, per shot', base: 'Wan2.1-T2V-1.3B', input: 'one prompt per shot', v: [0.998, 0.994, 0.219, 0.169, 0.310, 0.274, 0.694, 0.181, 0.474], fam: 'shot' },
  { name: 'Wan2.2, per shot', base: 'Wan2.2-TI2V-5B', input: 'one prompt per shot', v: [0.999, 0.996, 0.227, 0.275, 0.310, 0.387, 0.701, 0.174, 0.497], fam: 'shot' },
  { name: 'StoryMem', base: 'Wan2.2-I2V-A14B', input: 'shot prompt and keyframe memory', v: [0.928, 0.719, 0.631, 0.453, 0.687, 0.484, 0.545, 0.179, 0.527], fam: 'shot' },
  { name: 'StoryDiffusion', base: 'Wan2.2-I2V-A14B', input: 'keyframes, then Wan2.2-I2V', v: [0.905, 0.825, 0.300, 0.265, 0.245, 0.597, 0.535, 0.171, 0.468], fam: 'shot' },
  { name: 'IC-LoRA', base: 'Wan2.2-I2V-A14B', input: 'three-panel grid, then Wan2.2-I2V', v: [0.976, 0.961, 0.168, 0.133, 0.406, 0.452, 0.663, 0.176, 0.484], fam: 'shot' },
  { g: 'Joint multi-shot generation' },
  { name: 'MultiShotMaster', base: 'Wan2.1-T2V-1.3B', input: 'story caption and shot captions', v: [0.994, 0.990, 0.538, 0.254, 0.582, 0.371, 0.587, 0.178, 0.527], fam: 'joint' },
  { name: 'HoloCine', base: 'Wan2.2-T2V-A14B', input: 'global caption and shot captions', v: [0.748, 0.967, 0.517, 0.512, 0.586, 0.694, 0.691, 0.195, 0.571], fam: 'joint' },
  { name: 'CaT-Movie', base: 'Wan2.2-TI2V-5B', input: 'its planner’s storyboard, from the shot-list request', v: [0.995, 0.974, 0.651, 0.497, 0.630, 0.468, 0.785, 0.186, 0.597], fam: 'ours' },
];
const T1COLS = ['Cut F1', 'Execution', 'Character', 'Object', 'Scene', 'Establish', 'Persist', 'Alignment', 'Avg.'];
const T1GROUPS = [['Transition control ↑', 2], ['Inter-shot consistency ↑', 3], ['State continuity ↑', 2], ['Text ↑', 1], ['', 1]];
const T1ASK = [
  ['Cut F1', 'Detected cuts matched to all 740 planned cuts within 0.125 s.'],
  ['Execution', 'Of the 674 cuts that change the setup: TransNetV2 finds a cut within 0.5 s and the judge sees a new camera setup.'],
  ['Character', '“Is C1 at S3 the same person as at S1?” Counts only across executed cuts.'],
  ['Object', '“Is O1 at S2 the same object as at S1?”'],
  ['Scene', '“Are S10 and S1 in the same place (L1)?”'],
  ['Establish', '“Is O1 folded by the end of S1?” A requested change must happen in its shot.'],
  ['Persist', '“Is O1 still folded in S2?” The change must hold when the entity returns.'],
  ['Text alignment', 'ViCLIP similarity of each shot to its request line.'],
  ['Average', 'Mean of the four aspects.'],
];
const POINT_COLOR = {
  'CaT-Movie': 'var(--ours)', HoloCine: 'var(--holo)', MultiShotMaster: 'var(--msm)', StoryMem: 'var(--mem)',
  'Wan2.1, per shot': 'var(--grey-b)', 'Wan2.2, per shot': 'var(--grey-a)', StoryDiffusion: 'var(--grey-c)', 'IC-LoRA': 'var(--grey-d)',
};
run('mainTable', function mainTable() {
  const tb = $('#mainTable');
  if (!tb) return;
  let sortK = null, dir = -1;
  const rows = T1.filter(r => r.name);
  const best = T1COLS.map((_, c) => { const vals = rows.map(r => r.v[c]).sort((a, b) => b - a); return [vals[0], vals.find(v => v < vals[0])]; });
  const rng = T1COLS.map((_, c) => { const vals = rows.map(r => r.v[c]); return [Math.min(...vals), Math.max(...vals)]; });
  const cell = (r, c) => {
    const v = r.v[c], cls = v === best[c][0] ? 'best' : v === best[c][1] ? 'second' : '';
    const [lo, hi] = rng[c];
    return `<td class="${cls}">${v.toFixed(3)}<span class="bar" style="width:${(6 + 28 * (v - lo) / (hi - lo || 1)).toFixed(1)}px"></span></td>`;
  };
  const row = r => `<tr class="${r.fam === 'ours' ? 'ours' : ''}" data-name="${esc(r.name)}"><td>${esc(r.name)}<small>${esc(r.base)}</small></td>${T1COLS.map((_, c) => cell(r, c)).join('')}</tr>`;
  function render() {
    const head1 = `<tr><th class="grp"></th>${T1GROUPS.map(([g, n]) => `<th class="grp" colspan="${n}">${g}</th>`).join('')}</tr>`;
    const head2 = `<tr><th data-k="-1">Method</th>${T1COLS.map((c, i) => `<th data-k="${i}" class="${sortK === i ? 'sorted' : ''}" data-dir="${dir < 0 ? '↓' : '↑'}" aria-sort="${sortK === i ? (dir < 0 ? 'descending' : 'ascending') : 'none'}">${c}</th>`).join('')}</tr>`;
    const body = sortK === null
      ? T1.map(r => r.g ? `<tr class="grp-row"><td colspan="10">${r.g}</td></tr>` : row(r)).join('')
      : rows.slice().sort((a, b) => dir * (a.v[sortK] - b.v[sortK])).map(row).join('');
    tb.innerHTML = `<thead>${head1}${head2}</thead><tbody>${body}</tbody>`;
    $$('th[data-k]', tb).forEach(th => {
      const k = +th.dataset.k;
      th.addEventListener('click', () => {
        if (k < 0) sortK = null;
        else if (sortK === k) dir = -dir;
        else { sortK = k; dir = -1; }
        hideTip(); render();
      });
      if (k >= 0) {
        th.addEventListener('pointermove', e => showTip(`<b>${T1ASK[k][0]}</b><br>${T1ASK[k][1]}`, e));
        th.addEventListener('pointerleave', hideTip);
      }
    });
    $$('tbody tr[data-name]', tb).forEach(tr => {
      const r = rows.find(x => x.name === tr.dataset.name);
      tr.addEventListener('pointerenter', () => pick(r.name, 'table'));
      tr.addEventListener('pointerleave', () => { pick(null, 'table'); hideTip(); });
      const nameCell = tr.firstElementChild;
      nameCell.addEventListener('pointermove', e => showTip(`<b>${esc(r.name)}</b><br>Input: ${esc(r.input)}<br>Video base: ${esc(r.base)}`, e));
      nameCell.addEventListener('pointerleave', hideTip);
    });
  }
  render();
  document.addEventListener('pick', e => { $$('tbody tr[data-name]', tb).forEach(tr => tr.classList.toggle('hl', tr.dataset.name === e.detail.name && e.detail.from !== 'table')); });
  onReveal(tb.closest('.fig'), () => {
    if (RM) return;
    $$('.bar', tb).forEach((b, i) => b.animate([{ transform: 'scaleX(0)' }, { transform: 'none' }], { duration: 650, delay: 150 + (i % 9) * 35, easing: 'cubic-bezier(.2,.7,.2,1)', fill: 'backwards' }));
  });
});

/* ---------- ablation table (Method drawer) ---------- */
const T2 = [
  { name: 'CaT-Movie, reference storyboard', v: [0.958, 0.581, 0.585, 0.566, 0.615, 0.571], sig: [], ours: true },
  { g: 'Trained without' },
  { name: 'identity lines', v: [0.942, 0.582, 0.607, 0.557, 0.653, 0.561], sig: [] },
  { name: 'routing', v: [0.923, 0.593, 0.513, 0.595, 0.609, 0.564], sig: [0] },
  { name: 'references', v: [0.966, 0.289, 0.342, 0.474, 0.696, 0.521], sig: [1, 2, 5] },
  { g: 'Reference selection' },
  { name: 'recent shots', v: [0.928, 0.488, 0.615, 0.471, 0.598, 0.563], sig: [0, 1, 3] },
  { name: 'CLIP retrieval', v: [0.953, 0.544, 0.538, 0.503, 0.534, 0.537], sig: [5] },
  { name: 'anchor and first views', v: [0.955, 0.569, 0.547, 0.546, 0.667, 0.566], sig: [] },
  { g: 'Storyboard' },
  { name: 'request only', v: [0.955, 0.525, 0.509, 0.546, 0.557, 0.541], sig: [] },
  { name: 'our planner', v: [0.963, 0.579, 0.530, 0.560, 0.708, 0.587], sig: [] },
];
run('ablTable', function ablTable() {
  const tb = $('#ablTable');
  if (!tb) return;
  const cols = ['Execution', 'Character', 'Object', 'Scene', 'Persist', 'Avg.'];
  let mode = 'abs';
  const ref = T2[0].v;
  function render() {
    tb.innerHTML = `<thead><tr><th>Setting</th>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>${T2.map(r => {
      if (r.g) return `<tr class="grp-row"><td colspan="7">${r.g}</td></tr>`;
      const cells = r.v.map((v, i) => {
        const sig = r.sig.includes(i) ? ' sig' : '';
        if (mode === 'abs' || r.ours) return `<td class="${sig.trim()}">${v.toFixed(3)}</td>`;
        const d = (v - ref[i]) * 100;
        return `<td class="${d > 0.05 ? 'up' : d < -0.05 ? 'dn' : ''}${sig}">${d > 0 ? '+' : d < 0 ? '−' : '±'}${Math.abs(d).toFixed(1)}</td>`;
      }).join('');
      return `<tr class="${r.ours ? 'ours' : ''}"><td>${esc(r.name)}</td>${cells}</tr>`;
    }).join('')}</tbody>`;
  }
  render();
  $$('#ablMode button').forEach(b => b.addEventListener('click', () => { mode = b.dataset.m; setTab($('#ablMode'), b); render(); }));
});

/* ---------- copy BibTeX ---------- */
run('misc', function misc() {
  const toast = $('#toast'), btn = $('#copyBib');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    const txt = $('#bibText').textContent;
    try { await navigator.clipboard.writeText(txt); toast.textContent = 'BibTeX copied'; }
    catch (e) {
      const r = document.createRange();
      r.selectNodeContents($('#bibText'));
      const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
      toast.textContent = 'Selected, press ⌘C to copy';
    }
    toast.classList.add('on');
    setTimeout(() => toast.classList.remove('on'), 1800);
  });
});

window.CAT = { MEDIA, $, $$, sv, h, esc, once, watch, onReveal, chart, setTab, showTip, hideTip, pick, RM, T1, POINT_COLOR, run, story, frame, short, camLabel, pad, HERO, avatarStyle, charTip, showWhenReady };
})();
