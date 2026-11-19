/* CaT-Movie project page — native, animated versions of every paper figure. Needs app.js (window.CAT). */
(() => {
'use strict';
const { MEDIA, $, $$, sv, h, esc, once, watch, showTip, hideTip, story, frame, camLabel, short, RM, T1, POINT_COLOR, barChart, pad, run } = window.CAT;
const STORY = 'film-003995-b0001-shot06';
const C = { ours: 'var(--ours)', oursDeep: 'var(--ours-deep)', cut: 'var(--cut)', holo: 'var(--holo)', msm: 'var(--msm)', mem: 'var(--mem)', paper: 'var(--paper)', text: 'var(--text)', mute: 'var(--mute)', line: 'var(--line)', line2: 'var(--line-2)', ink: 'var(--ink)', ink2: 'var(--ink-2)', ink3: 'var(--ink-3)', flame: '#e07b39', ice: 'var(--ours)' };
const T = (g, x, y, str, style = '', anchor = 'start') => { const t = sv('text', { x, y, 'text-anchor': anchor, style }, g); t.textContent = str; return t; };
const MONO = (size = 10, fill = C.mute, w = 500) => `font:${w} ${size}px var(--mono);fill:${fill};letter-spacing:.06em`;
const BODY = (size = 12, fill = C.text, w = 400) => `font:${w} ${size}px var(--body);fill:${fill}`;
const DISP = (size = 18, fill = C.paper, w = 700) => `font:${w} ${size}px var(--display);fill:${fill};letter-spacing:.01em`;
function wrap(str, n) { const words = str.split(/\s+/); const out = []; let line = ''; for (const w of words) { if ((line + ' ' + w).trim().length > n) { if (line) out.push(line); line = w; } else line = (line + ' ' + w).trim(); } if (line) out.push(line); return out; }
function snow(g, x, y, r = 5) { const s = sv('g', { stroke: C.ice, 'stroke-width': 1.3, 'stroke-linecap': 'round' }, g); for (let i = 0; i < 3; i++) { const a = i * Math.PI / 3; sv('path', { d: `M${x - r * Math.cos(a)} ${y - r * Math.sin(a)} L${x + r * Math.cos(a)} ${y + r * Math.sin(a)}` }, s); } return s; }
function flame(g, x, y, s = 1) { return sv('path', { d: `M${x} ${y + 6 * s} C${x - 5 * s} ${y + 6 * s} ${x - 6 * s} ${y + 1 * s} ${x - 3 * s} ${y - 3 * s} C${x - 2 * s} ${y - 1 * s} ${x - 1 * s} ${y} ${x} ${y - 1 * s} C${x} ${y - 4 * s} ${x + 1 * s} ${y - 6 * s} ${x + 3 * s} ${y - 7 * s} C${x + 2 * s} ${y - 3 * s} ${x + 6 * s} ${y - 1 * s} ${x + 5 * s} ${y + 3 * s} C${x + 4.5 * s} ${y + 5 * s} ${x + 3 * s} ${y + 6 * s} ${x} ${y + 6 * s}Z`, fill: C.flame }, g); }
function pill(g, x, y, w, hgt, label, opts = {}) {
  const gg = sv('g', {}, g);
  sv('rect', { x, y, width: w, height: hgt, rx: hgt / 2, fill: opts.fill || C.ink3, stroke: opts.stroke || C.line2 }, gg);
  T(gg, x + w / 2 + (opts.icon ? -6 : 0), y + hgt / 2 + 4.5, label, opts.style || BODY(12.5, C.paper, 500), 'middle');
  if (opts.icon === 'snow') snow(gg, x + w - 16, y + hgt / 2, 5);
  if (opts.icon === 'flame') flame(gg, x + w - 16, y + hgt / 2);
  if (opts.icon === 'code') T(gg, x + w - 18, y + hgt / 2 + 4.5, '{ }', MONO(11, C.ours, 600), 'middle');
  return gg;
}
function thumb(g, href, x, y, w, hgt, opts = {}) {
  const gg = sv('g', {}, g); const id = 'cl' + Math.random().toString(36).slice(2, 8);
  const cp = sv('clipPath', { id }, gg); sv('rect', { x, y, width: w, height: hgt, rx: opts.rx ?? 3 }, cp);
  sv('image', { href, x, y, width: w, height: hgt, preserveAspectRatio: 'xMidYMid slice', 'clip-path': `url(#${id})` }, gg);
  sv('rect', { x, y, width: w, height: hgt, rx: opts.rx ?? 3, fill: 'none', stroke: opts.stroke || 'rgba(236,232,224,.18)', 'stroke-width': opts.sw || 1 }, gg);
  return gg;
}
function avatar(g, href, cx, cy, r, fx, fy, crop) { // crop = face width fraction of a 480x264 frame
  const gg = sv('g', {}, g); const id = 'av' + Math.random().toString(36).slice(2, 8);
  const cp = sv('clipPath', { id }, gg); sv('circle', { cx, cy, r }, cp);
  const W = 2 * r / crop, H = W * 264 / 480;
  sv('image', { href, x: cx - fx * W, y: cy - fy * H, width: W, height: H, 'clip-path': `url(#${id})`, preserveAspectRatio: 'none' }, gg);
  sv('circle', { cx, cy, r, fill: 'none', stroke: 'rgba(236,232,224,.35)' }, gg);
  return gg;
}
function arrowDefs(svg) {
  const d = sv('defs', {}, svg);
  [['arr', C.mute], ['arrB', C.ours], ['arrP', C.cut]].forEach(([id, col]) => { const m = sv('marker', { id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, d); sv('path', { d: 'M0 0L10 5L0 10z', fill: col }, m); });
  const p = sv('pattern', { id: 'noise', width: 6, height: 6, patternUnits: 'userSpaceOnUse' }, d);
  for (let i = 0; i < 12; i++) sv('rect', { x: Math.floor(Math.random() * 6), y: Math.floor(Math.random() * 6), width: 1, height: 1, fill: `rgba(236,232,224,${(Math.random() * .7 + .2).toFixed(2)})` }, p);
  return d;
}

/* =============== 1 · profile card beside the Pareto chart =============== */
run('profile', function profile() {
  const host = $('#profile'); if (!host) return;
  const cols = ['Cut F1', 'Execution', 'Character', 'Object', 'Scene', 'Establish', 'Persist', 'Text alignment', 'Average'];
  const ours = T1.find(r => r.name === 'CaT-Movie');
  const FAM = { 'Wan2.1, per shot': 'Independent per-shot generation', 'Wan2.2, per shot': 'Independent per-shot generation', StoryMem: 'Memory-augmented, per shot', StoryDiffusion: 'Keyframe pipeline', 'IC-LoRA': 'Keyframe pipeline', MultiShotMaster: 'Joint multi-shot generation', HoloCine: 'Joint multi-shot generation', 'CaT-Movie': 'Storyboard planner and window renderer' };
  host.innerHTML = `<div class="pf-head"><i id="pfDot"></i><div><b id="pfName"></b><small id="pfFam"></small></div></div><div class="pf-axes"><div><span>Transition control</span><b id="pfTr"></b></div><div><span>Inter-shot consistency</span><b id="pfCo"></b></div></div><div class="pf-rows">${cols.map((c, i) => `<div class="pf-row${i === 8 ? ' avg' : ''}"><span>${c}</span><div class="pf-track"><i class="pf-fill"></i><b class="pf-tick"></b></div><em class="tnum"></em></div>`).join('')}</div><p class="small pf-note">Hover a point to preview it, click to pin. The tick on each bar marks CaT-Movie.</p>`;
  const rows = $$('.pf-row', host); let pinned = 'CaT-Movie';
  function show(name) {
    const r = T1.find(x => x.name === name); if (!r) return; const col = POINT_COLOR[name];
    $('#pfDot').style.background = col; $('#pfName').textContent = name; $('#pfName').style.color = name === 'CaT-Movie' ? C.ours : C.paper;
    $('#pfFam').textContent = `${FAM[name]} · ${r.base}`;
    $('#pfTr').textContent = ((r.v[0] + r.v[1]) / 2).toFixed(3); $('#pfCo').textContent = ((r.v[2] + r.v[3] + r.v[4]) / 3).toFixed(3);
    rows.forEach((row, i) => { const v = r.v[i]; $('.pf-fill', row).style.width = (v * 100) + '%'; $('.pf-fill', row).style.background = col; const tk = $('.pf-tick', row); tk.style.left = (ours.v[i] * 100) + '%'; tk.style.opacity = name === 'CaT-Movie' ? 0 : 1; const em = $('em', row); const d = (v - ours.v[i]) * 100; em.innerHTML = `${v.toFixed(3)}${name === 'CaT-Movie' ? '' : `<small class="${d >= 0 ? 'up' : 'dn'}">${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(1)}</small>`}`; });
  }
  document.addEventListener('pick', e => { if (e.detail.preview) show(e.detail.name || pinned); else { pinned = e.detail.name; show(pinned); } });
  show('CaT-Movie');
  once(host, () => { if (RM) return; rows.forEach((row, i) => { const f = $('.pf-fill', row); const w = f.style.width; f.style.transition = 'none'; f.style.width = '0'; requestAnimationFrame(() => requestAnimationFrame(() => { f.style.transition = `width .9s var(--ease) ${i * 60}ms, background .3s`; f.style.width = w; })); }); }, 0.4);
});

/* =============== 3 · CaT-MBench question bank + judging (paper Fig. 3 c, d) =============== */
run('judging', function judging() {
  const qTabs = $('#qTabs'); if (!qTabs) return;
  const Q = [
    ['Character consistency', 'Is C1 at S3 the same person as at S1?', 'Counts only when the transition into S3 opens its planned camera setup.'],
    ['Object consistency', 'Is O1 at S2 the same object as at S1?', 'Evaluates movable props explicitly requested in both shots.'],
    ['Scene consistency', 'Are S10 and S1 in the same place (L1)?', 'Checks whether the setting matches its first appearance across non-adjacent shots and camera angles.'],
    ['State establishment', 'Is O1 folded? (S1 end)', 'A requested within-shot state transition must become visible by the end of its shot.'],
    ['State persistence', 'Is O1 still folded? (S2)', 'The established state must still hold when the entity returns in a later shot.'],
    ['Cut execution', 'S1 → S2: cut to a new camera setup?', 'TransNetV2 must detect a cut within 0.5 s and the VLM judge must verify a new location, shot size, angle, or cast.'],
  ];
  const story0 = 'C1: dark-haired man, grey suit · O1: piece of paper · L1: wooden kitchen · S01: Medium shot. C1 writes on the paper, folds it, and hands it to C2.';
  const qCard = $('#qCard'); let qi = 0, qHeld = false;
  Q.forEach(([n], i) => { const b = h('button', 'qtab', n); b.addEventListener('click', () => { qHeld = true; setQ(i); }); qTabs.appendChild(b); });
  function setQ(i) { qi = i; $$('.qtab', qTabs).forEach((b, j) => b.classList.toggle('on', j === i)); qCard.innerHTML = `<div class="qsrc mono">${esc(story0)}</div><div class="qq fade-in"><span class="mono">Q</span>${esc(Q[i][1])}</div><div class="qwhy fade-in">${esc(Q[i][2])}</div>`; }
  setQ(0);
  const AD = [['Wan2.2, per shot', 'one prompt per shot'], ['MultiShotMaster', 'story caption + shot caption'], ['HoloCine', 'global caption + shot captions'], ['StoryMem', 'shot prompt + keyframe memory'], ['StoryDiffusion', 'keyframes → Wan2.2-I2V'], ['IC-LoRA', '3-panel grid → Wan2.2-I2V']];
  $('#adapters').innerHTML = AD.map(([m, f]) => `<div><b>${m}</b><span>${f}</span></div>`).join('');
  /* judging strip */
  const FR = [['s01_t00', 'S1', '0:00'], ['s01_t02', 'S1', '0:02'], ['s01_t04', 'S1', '0:04'], ['s02_t06', 'S2', '0:06'], ['s03_t09', 'S3', '0:09'], ['s03_t10', 'S3', '0:10'], ['s03_t11', 'S3', '0:11'], ['s04_t13', 'S4', '0:13'], ['s08_t45', 'S8', '0:45'], ['s08_t46', 'S8', '0:46'], ['s08_t47', 'S8', '0:47'], ['s09_t50', 'S9', '0:50']];
  const strip = $('#jStrip');
  strip.innerHTML = [0, 1, 2].map(r => `<div class="jrow">${FR.slice(r * 4, r * 4 + 4).map(([f, s, t], i) => `<div class="jf" data-i="${r * 4 + i}" style="background-image:url(${MEDIA}f/judge/${f}.jpg)"><span>${s} ${t}</span></div>`).join('')}</div>`).join('');
  const EX = [
    { hi: [[0, 1, 2], [4, 5, 6]], q: 'Is C1 at 0:09–0:12 the same person as at 0:00–0:04?', steps: [['Judge', 'No, face differs despite matching hair, build, and attire.', 'bad'], ['Gate', 'new camera setup ✓ (frame correlation −0.08)', 'ok'], ['Result', 'gate passed, character check counts and fails', 'bad']] },
    { hi: [[2], [3]], q: 'S1 → S2: does the planned cut open a new camera setup?', steps: [['TransNetV2', 'cut detected within 0.5 s ✓', 'ok'], ['Frame correlation', '0.30', 'neu'], ['Judge', 'new camera setup ✓', 'ok'], ['Result', 'cut executed, consistency checks across it are counted', 'ok']] },
  ];
  const flow = $('#jFlow'); let ei = 0, eTimer = null, eHeld = false, vis = false, typer = null;
  function setE(i) {
    ei = i; $$('#jEx button').forEach((b, j) => b.classList.toggle('on', j === i)); const E = EX[i];
    $$('.jf', strip).forEach(f => { const k = +f.dataset.i; f.classList.remove('a', 'b'); if (E.hi[0].includes(k)) f.classList.add('a'); if (E.hi[1].includes(k)) f.classList.add('b'); });
    flow.innerHTML = `<div class="jq"><span class="mono">ASK</span>${esc(E.q)}</div>` + E.steps.map(([k, v, c], n) => `<div class="jstep ${c}" style="--d:${n}"><span class="mono">${esc(k)}</span><b data-t="${esc(v)}"></b></div>`).join('');
    clearTimeout(typer); const bs = $$('.jstep b', flow); let n = 0;
    (function next() { if (n >= bs.length) return; const b = bs[n]; const txt = b.dataset.t; if (RM) { bs.forEach(x => x.textContent = x.dataset.t); return; } let c = 0; b.parentElement.classList.add('on'); (function ty() { c += 2; b.textContent = txt.slice(0, c); if (c < txt.length) typer = setTimeout(ty, 18); else { n++; typer = setTimeout(next, 380); } })(); })();
  }
  $$('#jEx button').forEach(b => b.addEventListener('click', () => { eHeld = true; setE(+b.dataset.e); }));
  setE(0);
  function cycle() { clearTimeout(eTimer); if (!vis || RM || eHeld) return; eTimer = setTimeout(() => { setE((ei + 1) % EX.length); cycle(); }, 7000); }
  setInterval(() => { if (vis && !qHeld && !RM) setQ((qi + 1) % Q.length); }, 3500);
  watch($('#judging'), v => { vis = v; if (v) { setE(ei); cycle(); } else { clearTimeout(eTimer); } }, 0.3);
});

/* =============== 5 · planner text and written vs executed (paper Fig. 7) =============== */
run('planner', function planner() {
  const a = $('#planText'), b = $('#writExec'); if (!a || !b) return;
  const D = { opening: { Syn: [0.1625, 0.4471], Para: [0.5649, 0.7091], Shot: [0.4833, 0.7304] }, held: { Syn: [0.0732, 0.2171], Para: [0.5721, 0.6797], Shot: [0.3122, 0.5122] } };
  const W = 600, H = 300, m = { l: 44, r: 10, t: 18, b: 58 }; const Y = v => H - m.b - v * (H - m.t - m.b);
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Share of reference opening states and held items stated by the request alone and with the planner' }, a);
  const grid = sv('g', { class: 'grid' }, svg), ax = sv('g', { class: 'axis' }, svg);
  [0, .25, .5, .75, 1].forEach(v => { sv('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid); sv('text', { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end' }, ax).textContent = (v * 100) + '%'; });
  const segs = []; const tiers = ['Syn', 'Para', 'Shot']; const TL = { Syn: 'synopsis', Para: 'paragraph', Shot: 'shot list' };
  [['opening', 'Opening states'], ['held', 'Held items']].forEach(([key, name], gi) => {
    const gx = m.l + 20 + gi * 275;
    T(svg, gx + 118, H - 14, name, BODY(12.5, C.paper, 500), 'middle');
    tiers.forEach((t, ti) => {
      const [r, tot] = D[key][t]; const x = gx + ti * 82, bw = 58;
      const g = sv('g', { style: 'cursor:pointer' }, svg);
      const r1 = sv('rect', { x, y: Y(r), width: bw, height: Y(0) - Y(r), fill: 'rgba(116,124,140,.42)', rx: 2 }, g);
      const r2 = sv('rect', { x, y: Y(tot), width: bw, height: Y(r) - Y(tot), fill: C.ours, rx: 2 }, g);
      const lab = T(g, x + bw / 2, Y(tot) - 7, `${Math.round(tot * 100)}%`, MONO(11, C.paper, 600), 'middle');
      T(svg, x + bw / 2, H - m.b + 16, t, MONO(10.5), 'middle');
      g.addEventListener('pointermove', e => showTip(`<b>${name} · ${TL[t]} requests</b><table><tr><td>stated by the request</td><td>${(r * 100).toFixed(1)}%</td></tr><tr><td>added by the planner</td><td>+${((tot - r) * 100).toFixed(1)}</td></tr><tr><td>stated in the compiled text</td><td>${(tot * 100).toFixed(1)}%</td></tr></table>`, e)); g.addEventListener('pointerleave', hideTip);
      segs.push({ r1, r2, lab, r, tot, x });
    });
  });
  once(a, () => { if (RM) return; segs.forEach((sg, i) => { const { r1, r2, lab, r, tot } = sg; r1.setAttribute('y', Y(0)); r1.setAttribute('height', 0); r2.setAttribute('y', Y(0)); r2.setAttribute('height', 0); lab.style.opacity = 0; const t0 = performance.now() + i * 90; (function f(t) { const p = Math.max(0, Math.min(1, (t - t0) / 1300)); const e1 = Math.min(1, p / .5), e2 = Math.max(0, (p - .5) / .5); const ez = x => 1 - Math.pow(1 - x, 3); const v1 = r * ez(e1), v2 = (tot - r) * ez(e2); r1.setAttribute('y', Y(v1)); r1.setAttribute('height', Y(0) - Y(v1)); r2.setAttribute('y', Y(v1 + v2)); r2.setAttribute('height', Y(v1) - Y(v1 + v2)); lab.setAttribute('y', Y(v1 + v2) - 7); lab.style.opacity = e2; if (p < 1) requestAnimationFrame(f); })(performance.now()); }); }, 0.35);
  /* written vs executed */
  const R = [['request_only', 'Request', 170, 79, 49, 42], ['reference', 'Reference', 171, 91, 59, 21], ['planner_27b', 'Our planner', 171, 108, 48, 15]];
  const svg2 = sv('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'State questions shown, stated but not shown, or unstated for three storyboards' }, b);
  const g2 = sv('g', { class: 'grid' }, svg2), ax2 = sv('g', { class: 'axis' }, svg2);
  [0, .25, .5, .75, 1].forEach(v => { sv('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g2); sv('text', { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end' }, ax2).textContent = (v * 100) + '%'; });
  const defs = sv('defs', {}, svg2); const hp = sv('pattern', { id: 'hatchP', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs); sv('rect', { width: 6, height: 6, fill: 'rgba(214,144,187,.18)' }, hp); sv('path', { d: 'M0 0V6', stroke: C.cut, 'stroke-width': 2 }, hp);
  const bars = [];
  R.forEach(([key, name, n, se, sn, ns], i) => {
    const x = m.l + 40 + i * 180, bw = 92; const f1 = se / n, f2 = sn / n, f3 = ns / n; const g = sv('g', { style: 'cursor:pointer' }, svg2);
    const a1 = sv('rect', { x, y: Y(f1), width: bw, height: Y(0) - Y(f1), fill: C.ours, rx: 2 }, g);
    const a2 = sv('rect', { x, y: Y(f1 + f2), width: bw, height: Y(f1) - Y(f1 + f2), fill: 'url(#hatchP)', rx: 2 }, g);
    const a3 = sv('rect', { x: x + .5, y: Y(1) + .5, width: bw - 1, height: Y(f1 + f2) - Y(1) - 1, fill: 'none', stroke: C.line2, 'stroke-dasharray': '3 3', rx: 2 }, g);
    T(svg2, x + bw / 2, H - m.b + 18, name, BODY(12.5, key === 'planner_27b' ? C.ours : C.paper, 600), 'middle');
    T(svg2, x + bw / 2, H - m.b + 33, `stated ${Math.round((se + sn) / n * 100)}%`, MONO(9.5), 'middle');
    T(svg2, x + bw / 2, H - m.b + 46, `${Math.round(se / (se + sn) * 100)}% of stated shown`, MONO(9.5), 'middle');
    g.addEventListener('pointermove', e => showTip(`<b>${name} storyboard</b><table><tr><td>stated and shown</td><td>${se} (${(f1 * 100).toFixed(0)}%)</td></tr><tr><td>stated, not shown</td><td>${sn} (${(f2 * 100).toFixed(0)}%)</td></tr><tr><td>unstated</td><td>${ns} (${(f3 * 100).toFixed(0)}%)</td></tr><tr><td>state questions</td><td>${n}</td></tr></table>`, e)); g.addEventListener('pointerleave', hideTip);
    const lab1 = T(g, x + bw / 2, Y(f1 / 2) + 5, Math.round(f1 * 100), 'font:600 14px var(--mono);fill:#081022', 'middle');
    const lab2 = T(g, x + bw / 2, Y(f1 + f2 / 2) + 5, Math.round(f2 * 100), `font:600 14px var(--mono);fill:${C.paper};paint-order:stroke;stroke:#0e1118;stroke-width:3px`, 'middle');
    const lab3 = T(g, x + bw / 2, Y(f1 + f2 + f3 / 2) + 5, Math.round(f3 * 100), MONO(13, C.mute, 600), 'middle');
    bars.push({ a1, a2, f1, f2, labs: [lab1, lab2, lab3] });
  });
  once(b, () => { if (RM) return; bars.forEach(({ a1, a2, f1, f2, labs }, i) => { labs.forEach((l, j) => l.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 400, delay: 700 + i * 150 + j * 120, fill: 'backwards' })); const t0 = performance.now() + i * 150; (function f(t) { const p = Math.max(0, Math.min(1, (t - t0) / 1000)); const e = 1 - Math.pow(1 - p, 3); a1.setAttribute('y', Y(f1 * e)); a1.setAttribute('height', Y(0) - Y(f1 * e)); a2.setAttribute('y', Y((f1 + f2) * e)); a2.setAttribute('height', Y(f1 * e) - Y((f1 + f2) * e)); if (p < 1) requestAnimationFrame(f); })(performance.now()); }); }, 0.35);
});

/* =============== 6 · annotation + entity grounding (paper Fig. 3 a, b) =============== */
run('grounding', function grounding() {
  const host = $('#ground'); if (!host) return;
  /* per shot: entities and, on every 4th training frame, each entity's box from its tracked mask (832x480; null = no mask) */
  const TRACK = [{"shot":1,"entities":["C1","O1"],"frames":[[[447,37,330,443],[0,413,456,67]],[[445,34,327,446],[0,407,453,73]],[[437,38,325,442],[0,404,444,69]],[[460,43,319,437],[0,92,652,375]],[[459,39,327,441],[0,342,468,119]],[[440,42,333,438],[0,339,447,117]],[[394,24,351,456],[0,361,485,93]],[[364,0,379,480],[0,352,470,104]],[[379,0,382,480],[0,369,471,93]],[[401,0,391,480],[0,375,488,90]],[[409,1,406,479],[7,395,492,85]],[[385,13,430,467],[47,400,690,80]],[[342,8,464,472],[88,443,386,37]],[[372,6,448,474],[126,456,349,24]]]},{"shot":2,"entities":["C2","O2","O3"],"frames":[[[145,71,303,409],[250,0,425,480],[408,80,133,393]],[[123,61,298,419],[249,0,487,480],[415,74,142,399]],[[104,61,287,419],[249,0,510,480],[428,71,150,409]],[[96,54,305,426],[242,0,517,480],[424,66,171,414]],[[109,50,325,430],[235,0,535,480],[439,57,178,423]],[[140,26,367,454],[238,0,562,480],[424,50,216,430]],[[162,5,399,475],[206,0,626,480],[426,65,226,408]],[[155,2,420,478],[169,0,663,480],[420,36,233,290]],[[118,0,432,480],[135,0,697,480],[397,33,255,447]],[[66,0,445,480],[113,0,719,480],[386,29,267,451]],[[29,0,462,480],[645,0,187,480],[393,29,248,451]],[[0,0,468,480],[679,0,153,480],[403,27,282,453]],[[0,0,444,480],[466,0,366,480],[407,28,315,452]],[[0,0,428,480],[339,0,493,480],[402,31,348,449]],[[0,0,435,480],[73,0,759,480],[398,33,388,447]],[[0,0,454,480],[73,0,759,480],[399,33,416,447]],[[0,0,471,480],null,[395,32,413,448]],[[0,0,470,480],null,[391,36,410,444]],[[0,0,460,480],null,[388,42,413,438]],[[0,0,471,480],null,[390,46,411,434]]]},{"shot":3,"entities":["C1"],"frames":[[[315,36,517,444]],[[361,21,471,459]],[[368,26,464,454]],[[333,54,499,426]],[[264,45,532,435]],[[251,35,540,445]],[[284,38,540,442]],[[342,34,490,446]],[[397,12,435,468]],[[405,13,427,467]],[[363,29,469,451]],[[296,2,536,478]],[[283,8,549,472]],[[315,17,517,463]],[[389,10,443,470]],[[430,3,402,477]],[[424,12,408,468]],[[380,30,452,450]],[[310,21,522,459]],[[284,15,548,465]]]}];
  const COL = {"C1": "#29B6AA", "O1": "#5B8FF9", "C2": "#FCB84C", "O2": "#B084D8", "O3": "#EE7D92"};
  const LAYERS = [['rgb', 'Frame', 'source shot'], ['boxes', 'Boxes', 'Qwen3.8-27B, then tracked'], ['masks', 'Masks', 'SAM 3.1, tracked'], ['depth', 'Depth', 'VGGT-Ω, relative']];
  host.innerHTML = `<div class="gh"><span class="k">Entity grounding</span><div class="seg" id="gSeg">${LAYERS.map(([k, n], i) => `<button data-l="${k}"${i === 1 ? ' class="on"' : ''}>${n}</button>`).join('')}</div></div>` +
    `<div class="gframes">${TRACK.map((t, i) => `<div class="gfr" data-s="${i + 1}"><div class="gvid"><video muted loop playsinline preload="none" poster="${MEDIA}v/ground/s${i + 1}-poster.jpg" src="${MEDIA}v/ground/s${i + 1}.mp4" aria-label="Shot ${i + 1} of an annotated block"></video></div><svg class="lay boxes" viewBox="0 0 832 480" preserveAspectRatio="none">${t.entities.map(n => `<g data-e="${n}"><rect class="bx" fill="none" stroke="${COL[n]}" stroke-width="5" pathLength="100"/><rect class="tagbg" width="${n.length * 22 + 14}" height="36" rx="4" fill="${COL[n]}"/><text style="font:700 24px var(--mono);fill:#081022">${n}</text></g>`).join('')}</svg><span class="gs mono">Shot ${i + 1}</span></div>`).join('')}</div>` +
    `<p class="small" id="gCap"></p>`;
  const clips = $$('.gfr', host).map((fr, i) => ({ v: $('video', fr), t: TRACK[i], gs: $$('.boxes g', fr), j: -1 }));
  function place(c) {
    const j = Math.min(c.t.frames.length - 1, Math.floor((c.v.currentTime || 0) * 16 / 4)); if (j === c.j) return; c.j = j;
    c.gs.forEach((g, e) => { const b = c.t.frames[j][e]; g.style.display = b ? '' : 'none'; if (!b) return; const [x, y, w, hh] = b;
      const [r, bg] = g.querySelectorAll('rect'); const tx = g.querySelector('text');
      r.setAttribute('x', x + 2); r.setAttribute('y', y + 2); r.setAttribute('width', Math.max(4, w - 4)); r.setAttribute('height', Math.max(4, hh - 4));
      const ty = Math.min(480, y + hh) - 44; bg.setAttribute('x', x + 6); bg.setAttribute('y', ty); tx.setAttribute('x', x + 13); tx.setAttribute('y', ty + 27); });
  }
  let raf = 0; const tick = () => { clips.forEach(place); raf = requestAnimationFrame(tick); };
  clips.forEach(place);
  let li = 1, timer = null, held = false, vis = false;
  function setL(k) { li = LAYERS.findIndex(l => l[0] === k); host.dataset.layer = k; $$('#gSeg button').forEach(b => b.classList.toggle('on', b.dataset.l === k)); $('#gCap').innerHTML = `<b>${LAYERS[li][1]}</b> · ${LAYERS[li][2]}. ${k === 'depth' ? 'Relative depth within each shot, lighter is farther, kept on every fourth frame.' : k === 'masks' ? 'Each entity is tracked through its shot; Character 1 keeps the same colour in shots 1 and 3. Masks are kept on every fourth frame.' : k === 'boxes' ? 'Boxes carry the storyboard registry identifiers and follow each tracked entity, linking C1 across shots 1 and 3.' : 'Three consecutive shots from one annotated scene block, at the film’s own frame rate.'}`; }
  $$('#gSeg button').forEach(b => b.addEventListener('click', () => { held = true; setL(b.dataset.l); }));
  setL('boxes');
  function cycle() { clearTimeout(timer); if (held || !vis || RM) return; timer = setTimeout(() => { setL(LAYERS[(li + 1) % 4][0]); cycle(); }, 2600); }
  watch(host, v => { vis = v; cycle(); clips.forEach(c => { if (v) { c.v.preload = 'auto'; c.v.play().catch(() => {}); } else c.v.pause(); }); cancelAnimationFrame(raf); if (v) tick(); }, 0.2);
  /* annotation card typed in, entity hover links to boxes */
  const an = $('#annot');
  const LINES = [['h', 'Entity registry'], ['e', 'C1', 'an older man with white hair; black suit, white shirt, dark red tie'], ['e', 'C2', 'a man with dark hair and a lean build; dark grey polo shirt'], ['e', 'O1', 'a pile of newspapers and magazines'], ['e', 'O2', 'a metal mesh cage or display case'], ['e', 'O3', 'a black armored suit and cowl inside the cage'], ['h', 'Shot-level annotations'], ['s', 'Shot 1', 'Medium shot, static. C1 stands near a pile of newspapers and looks down, then looks up.'], ['s', 'Shot 2', 'Medium shot, tracking. C2 walks towards a metal mesh cage containing a black armored suit.'], ['s', 'Shot 3', 'Medium shot, static. C1 speaks while looking at C2.'], ['h', 'State change in shot 1'], ['c', 'C1 posture', 'looking down → looking up and forward']];
  an.innerHTML = `<div class="gh"><span class="k">One-pass block annotation · Qwen3.7-Flash</span></div><div class="astrip">${[1, 2, 3].map(i => `<div style="background-image:url(${MEDIA}f/ground/s${i}-rgb.jpg)"><span class="mono">Shot ${i}</span></div>`).join('')}</div><div class="alines">${LINES.map(l => l[0] === 'h' ? `<p class="ah">${l[1]}</p>` : `<p class="a${l[0]}"${l[0] === 'e' ? ` data-e="${l[1]}"` : ''}><b>${l[1]}</b> ${esc(l[2])}</p>`).join('')}</div>`;
  $$('.ae', an).forEach(p => { p.addEventListener('pointerenter', () => { held = true; setL('boxes'); host.classList.add('focus'); $$(`.boxes g`, host).forEach(g => g.classList.toggle('hl', g.dataset.e === p.dataset.e)); }); p.addEventListener('pointerleave', () => { host.classList.remove('focus'); $$('.boxes g', host).forEach(g => g.classList.remove('hl')); }); });
  once(an, () => { if (RM) return; $$('.alines p', an).forEach((p, i) => { p.style.animation = `fadeIn .5s var(--ease) ${i * 90}ms both`; }); }, 0.3);
});

/* =============== 7 · composition sunburst (paper Fig. 3 centre; plot_dataset_composition.py, v17 data) =============== */
run('sunburst', function sunburst() {
  const host = $('#sunburst'); if (!host) return;
  const hex = c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16) / 255);
  const rgb = a => `rgb(${a.map(v => Math.round(Math.max(0, Math.min(1, v)) * 255)).join(',')})`;
  const blend = (a, k) => a.map(v => v * (1 - k) + k);
  const ROSE = hex('#B7799E'), BLUE = hex('#6E87B0'), dark = a => a.map(v => v * .8);
  const SH = 'labelled usable shots', N_SH = 288083;
  const QUADS = [
    { col: dark(ROSE), half: 'top', facets: [
      ['Scene', SH, N_SH, [['Indoor', 66.24, 190814], ['Outdoor', 33.53, 96597], [null, 0.23, 672, 'unclear']]],
      ['Characters in shot', SH, N_SH, [['Single character', 51.81, 149246], ['Multiple characters', 44.46, 128088], [null, 3.73, 10749, 'no character in shot']]]] },
    { col: dark(BLUE), half: 'bot', facets: [
      ['Evaluation dimension', 'scored questions, dev + test', 1361, [['Character', 65.17, 887], ['Object', 6.32, 86], ['Scene', 18.81, 256], ['State', 9.70, 132]]],
      ['Recurrence distance', 'cross-shot consistency questions', 1229, [['Adjacent', 23.68, 291], ['2–4 apart', 32.79, 403], ['≥5 apart', 43.53, 535]]]] },
    { col: BLUE, half: 'bot', facets: [
      ['Shots per story', 'stories', 122, [['6–9 shots', 52.46, 64], ['10–12 shots', 47.54, 58]]],
      ['State change type', 'annotated state changes', 307, [['Posture', 53.75, 165], ['Held object', 21.82, 67], ['State or clothing', 24.43, 75]]]] },
    { col: ROSE, half: 'top', facets: [
      ['Visual style', SH, N_SH, [['Live action', 84.99, 244828], ['Animation', 15.01, 43255]]],
      ['Character recurrence', SH, N_SH, [['First seen', 31.38, 90409], ['Previous shot', 28.07, 80865], ['Earlier shot', 36.82, 106060], [null, 3.73, 10749, 'no character in shot']]]] },
  ];
  const W = 760, H = 700, cx = W / 2, cy = H / 2, U = 360;
  const R = { c: .185 * U, i0: .185 * U, i1: .41 * U, o0: .415 * U, o1: .585 * U, disc: .635 * U, dot: .665 * U };
  const P = (r, deg) => { const a = deg * Math.PI / 180; return [cx + r * Math.sin(a), cy - r * Math.cos(a)]; };
  const wedge = (r0, r1, d0, d1) => { const [x0, y0] = P(r1, d0), [x1, y1] = P(r1, d1), [x2, y2] = P(r0, d1), [x3, y3] = P(r0, d0); const L = d1 - d0 > 180 ? 1 : 0; return `M${x0} ${y0}A${r1} ${r1} 0 ${L} 1 ${x1} ${y1}L${x2} ${y2}A${r0} ${r0} 0 ${L} 0 ${x3} ${y3}Z`; };
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Composition sunburst: training data in the upper half, CaT-MBench in the lower half' }, host);
  const gW = sv('g', { class: 'sb-wedges' }, svg), gL = sv('g', { class: 'sb-labels' }, svg);
  const segs = [], labels = [], facetsAll = [];
  let deg = 0;
  QUADS.forEach(q => q.facets.forEach(([name, unit, total, cats], k) => {
    const base = blend(q.col, .16 * k), d0 = deg, d1 = deg + 45; deg = d1;
    const fseg = sv('path', { d: wedge(R.i0, R.i1, d0, d1), fill: rgb(base), stroke: C.ink2, 'stroke-width': 1.6, class: 'sb-seg', style: `--d:${d0}` }, gW);
    const fac = { name, unit, total, d0, d1, el: fseg, cats: [] }; facetsAll.push(fac);
    const labelled = cats.filter(c => c[0]); const n = labelled.length; let a = d0;
    cats.forEach(([lab, share, count, what]) => {
      const b0 = a, b1 = a + 45 * share / 100; a = b1;
      const color = lab ? blend(base, .28 + (n > 1 ? .34 * labelled.findIndex(c => c[0] === lab) / (n - 1) : .14)) : null;
      const el = sv('path', { d: wedge(R.o0, R.o1, b0, b1), fill: lab ? rgb(color) : '#3b4252', stroke: C.ink2, 'stroke-width': 1.4, class: 'sb-seg', style: `--d:${b0}` }, gW);
      const seg = { lab, share, count, what, b0, b1, mid: (b0 + b1) / 2, el, fac, color, base };
      segs.push(seg); fac.cats.push(seg);
      if (lab) labels.push(seg);
    });
  }));
  /* shares inside the outer ring, along the radius */
  segs.forEach(sg => {
    if (!sg.lab || sg.share < 8) return; const [x, y] = P((R.o0 + R.o1) / 2, sg.mid);
    let rot = sg.mid - 90; if (rot > 90) rot -= 180; if (rot < -90) rot += 180;
    const L = sg.color.reduce((a, v, i) => a + v * [.299, .587, .114][i], 0);
    T(gW, x, y + 4.5, `${Math.round(sg.share)}%`, `font:500 13.5px var(--mono);fill:${L < .52 ? '#fff' : '#121520'};pointer-events:none`, 'middle').setAttribute('transform', `rotate(${rot} ${x} ${y})`);
  });
  /* facet names along the arc */
  facetsAll.forEach(f => {
    const mid = (f.d0 + f.d1) / 2, [x, y] = P((R.i0 + R.i1) / 2, mid); let rot = mid; if (mid > 90 && mid < 270) rot = mid - 180;
    const words = f.name === 'State change type' ? ['State change', 'type'] : f.name.split(' ').length > 1 && f.name.length > 10 ? [f.name.split(' ').slice(0, -1).join(' '), f.name.split(' ').slice(-1)[0]] : [f.name];
    const t = sv('text', { x, y: y - (words.length - 1) * 7 + 5, 'text-anchor': 'middle', transform: `rotate(${rot} ${x} ${y})`, style: 'font:700 13px var(--body);fill:#fff;paint-order:stroke;stroke:rgba(17,21,32,.48);stroke-width:2.5px;pointer-events:none' }, gW);
    words.forEach((w, i) => { const sp = sv('tspan', { x, dy: i ? 15 : 0 }, t); sp.textContent = w; });
  });
  /* quadrant boundaries */
  [0, 90, 180, 270].forEach(d => { const [x0, y0] = P(R.c, d), [x1, y1] = P(R.o1 + 2, d); sv('path', { d: `M${x0} ${y0}L${x1} ${y1}`, stroke: '#0b0e14', 'stroke-width': 5 }, gW); });
  [R.i1 + 1, R.o1 + 1].forEach(r => sv('circle', { cx, cy, r, fill: 'none', stroke: 'rgba(236,232,224,.12)', 'stroke-width': .8 }, gW));
  /* leader-line labels with angular repulsion (min 10 deg within and across facets) */
  const repel = (angs, gap, lo, hi) => { const y = angs.slice(); for (let i = 1; i < y.length; i++) if (y[i] - y[i - 1] < gap) y[i] = y[i - 1] + gap; if (y.length && y[y.length - 1] > hi) { y[y.length - 1] = hi; for (let i = y.length - 2; i >= 0; i--) y[i] = Math.min(y[i], y[i + 1] - gap); } if (y.length && y[0] < lo) { y[0] = lo; for (let i = 1; i < y.length; i++) if (y[i] - y[i - 1] < gap) y[i] = y[i - 1] + gap; } return y; };
  facetsAll.forEach(f => { const ls = f.cats.filter(c => c.lab).sort((a, b) => a.mid - b.mid); const ang = repel(ls.map(c => c.mid), 10, f.d0, f.d1); ls.forEach((c, i) => c.ang = ang[i]); });
  labels.sort((a, b) => a.ang - b.ang); const glob = repel(labels.map(l => l.ang), 10, -360, 720); labels.forEach((l, i) => l.ang = glob[i]);
  const txt = l => l.share < 8 ? `${l.lab} ${Math.round(l.share)}%` : l.lab;
  const lines = l => { const t = txt(l); const brk = { 'Single character': ['Single', 'character'], 'Multiple characters': ['Multiple', 'characters'], 'Held object': ['Held', 'object'], 'State or clothing': ['State or', 'clothing'] }[l.lab]; return brk || [t]; };
  const box = (l, er) => { const r = R.dot + er + 12, [x, y] = P(r, l.ang), ls = lines(l), w = Math.max(...ls.map(s => s.length)) * 7.6, hh = ls.length * 16; const s = Math.sin(l.ang * Math.PI / 180), c = Math.cos(l.ang * Math.PI / 180); if (Math.abs(s) >= Math.abs(c)) return s > 0 ? [x, x + w, y - hh / 2, y + hh / 2] : [x - w, x, y - hh / 2, y + hh / 2]; return c > 0 ? [x - w / 2, x + w / 2, y - hh, y] : [x - w / 2, x + w / 2, y, y + hh]; };
  const placed = []; labels.slice().sort((a, b) => Math.abs(Math.sin(b.ang * Math.PI / 180)) - Math.abs(Math.sin(a.ang * Math.PI / 180))).forEach(l => { let er = 0; for (; er < 120; er += 4) { const b = box(l, er); if (!placed.some(p => !(b[1] <= p[0] + 2 || b[0] >= p[1] - 2 || b[3] <= p[2] + 1 || b[2] >= p[3] - 1))) break; } l.er = er; placed.push(box(l, er)); });
  labels.forEach(l => {
    const r = R.dot + l.er; const [xr, yr] = P(R.o1, l.mid), [xe, ye] = P(r, l.mid), [xl, yl] = P(r, l.ang); const col = rgb(blend(l.base, .1));
    const g = sv('g', { class: 'sb-lab', style: `--d:${l.mid}` }, gL);
    sv('path', { d: `M${xr} ${yr}L${xe} ${ye}L${xl} ${yl}`, fill: 'none', stroke: col, 'stroke-width': 1.4, 'stroke-linecap': 'round' }, g);
    sv('circle', { cx: xl, cy: yl, r: 3.6, fill: '#0b0e14', stroke: rgb(l.base), 'stroke-width': 1.6 }, g);
    const s = Math.sin(l.ang * Math.PI / 180), c = Math.cos(l.ang * Math.PI / 180); const [xt, yt] = P(r + 10, l.ang); const ls = lines(l);
    let anchor = 'middle', y0 = yt; if (Math.abs(s) >= Math.abs(c)) { anchor = s > 0 ? 'start' : 'end'; y0 = yt - (ls.length - 1) * 8 + 5; } else if (c > 0) { y0 = yt - (ls.length - 1) * 16 - 2; } else { y0 = yt + 12; }
    const t = sv('text', { x: xt, y: y0, 'text-anchor': anchor, style: `font:600 13.5px var(--body);fill:${C.paper}` }, g);
    ls.forEach((w, i) => { const sp = sv('tspan', { x: xt, dy: i ? 16 : 0 }, t); sp.textContent = w; });
    l.lg = g;
  });
  /* centre */
  const cg = sv('g', {}, svg);
  sv('circle', { cx, cy, r: R.c - 4, fill: C.ink, stroke: C.line2 }, cg);
  const top = sv('g', {}, cg), bot = sv('g', {}, cg);
  sv('path', { d: `M${cx - 50} ${cy - 13}l6 -9l6 9h-4v8h-4v-8z`, fill: '#e9c3da' }, top); T(top, cx - 34, cy - 10, 'Dataset', DISP(21, '#e9c3da', 800));
  sv('path', { d: `M${cx - 58} ${cy + 13}l6 9l6 -9h-4v-8h-4v8z`, fill: '#b9cbff' }, bot); T(bot, cx - 42, cy + 27, 'Benchmark', DISP(21, '#b9cbff', 800));
  const info = sv('g', { style: 'opacity:0;transition:opacity .25s' }, cg);
  const iCat = T(info, cx, cy - 26, '', MONO(9, C.mute), 'middle'), iVal = T(info, cx, cy + 6, '', DISP(30, C.paper, 800), 'middle'), iLab = T(info, cx, cy + 24, '', BODY(11.5, C.paper, 500), 'middle'), iCnt = T(info, cx, cy + 39, '', MONO(9, C.mute), 'middle');
  function focus(list, cat, val, lab, cnt) { segs.forEach(sg => sg.el.classList.toggle('dim', !list.includes(sg.el))); facetsAll.forEach(f => f.el.classList.toggle('dim', !list.includes(f.el))); labels.forEach(l => l.lg.classList.toggle('dim', !list.includes(l.el))); top.style.opacity = bot.style.opacity = 0; info.style.opacity = 1; iCat.textContent = cat; iVal.textContent = val; iLab.textContent = lab; iCnt.textContent = cnt; }
  function reset() { [...segs, ...facetsAll].forEach(x => x.el.classList.remove('dim')); labels.forEach(l => l.lg.classList.remove('dim')); top.style.opacity = bot.style.opacity = 1; info.style.opacity = 0; }
  segs.forEach(sg => { sg.el.addEventListener('pointerenter', () => focus([sg.el, sg.fac.el], sg.fac.name.toUpperCase(), `${sg.share.toFixed(1)}%`, sg.lab || sg.what, `${sg.count.toLocaleString('en-US')} of ${sg.fac.total.toLocaleString('en-US')}`)); sg.el.addEventListener('pointerleave', reset); });
  facetsAll.forEach(f => { f.el.addEventListener('pointerenter', () => focus([f.el, ...f.cats.map(c => c.el)], f.unit.toUpperCase(), f.total.toLocaleString('en-US'), f.name, `${f.cats.filter(c => c.lab).length} categories`)); f.el.addEventListener('pointerleave', reset); });
  requestAnimationFrame(() => { try { const b = svg.getBBox(); if (b.width > 0) svg.setAttribute('viewBox', `${b.x - 8} ${b.y - 8} ${b.width + 16} ${b.height + 16}`); } catch (e) {} });
  once(host, () => { if (RM) return; svg.classList.add('sb-in'); }, 0.3);
});

/* =============== 8 · window distance (paper Fig. 6 right): line chart with values =============== */
run('distLines', function distLines() {
  const host = $('#distChart'); if (!host) return;
  const S = [['ours', 'CaT-Movie', C.ours, [46.3, 53.9, 59.8], [147, 297, 480], null, 'circle', ''], ['recent', 'Recent-shot recall', '#a3a9b5', [46.9, 54.9, 44.4], [147, 297, 480], [[0.7, -5.6, 9.4], [1.0, -5.1, 7.3], [-15.4, -22.7, -9.0]], 'square', '7 5'], ['no_refs', 'Trained without references', C.cut, [38.8, 31.3, 32.1], [147, 297, 480], [[-7.5, -20.2, 3.2], [-22.6, -30.6, -14.4], [-27.7, -35.3, -20.7]], 'tri', '2 5']];
  const W = 600, H = 300, m = { l: 46, r: 30, t: 22, b: 52 }; const X = i => m.l + 50 + i * ((W - m.l - m.r - 100) / 2), Y = v => H - m.b - (v - 25) / (65 - 25) * (H - m.t - m.b);
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Consistency pass rate by window distance for three settings' }, host);
  const grid = sv('g', { class: 'grid' }, svg), ax = sv('g', { class: 'axis' }, svg);
  [30, 40, 50, 60].forEach(v => { sv('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid); sv('text', { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end' }, ax).textContent = v + '%'; });
  ['Same window', '1 window apart', '≥2 windows apart'].forEach((n, i) => T(svg, X(i), H - m.b + 22, n, BODY(12.5, C.paper, 500), 'middle'));
  const anims = [];
  S.forEach(([k, name, col, v, n, d, mk, dash], si) => {
    const g = sv('g', {}, svg);
    const path = sv('path', { d: v.map((y, i) => `${i ? 'L' : 'M'}${X(i)} ${Y(y)}`).join(''), fill: 'none', stroke: col, 'stroke-width': k === 'ours' ? 3 : 2.2, 'stroke-dasharray': dash || 'none', 'stroke-linecap': 'round' }, g);
    v.forEach((y, i) => {
      const x = X(i), yy = Y(y); const pg = sv('g', { style: 'cursor:pointer' }, g);
      if (mk === 'circle') sv('circle', { cx: x, cy: yy, r: 6, fill: col, stroke: C.ink2, 'stroke-width': 2 }, pg);
      else if (mk === 'square') sv('rect', { x: x - 5.5, y: yy - 5.5, width: 11, height: 11, fill: col, stroke: C.ink2, 'stroke-width': 2 }, pg);
      else sv('path', { d: `M${x} ${yy - 7}L${x + 6.5} ${yy + 5}L${x - 6.5} ${yy + 5}Z`, fill: col, stroke: C.ink2, 'stroke-width': 2 }, pg);
      const [dx, dy, an] = k === 'ours' ? [[-12, 5, 'end'], [0, 22, 'middle'], [0, -14, 'middle']][i] : k === 'recent' ? [[0, -13, 'middle'], [0, -13, 'middle'], [0, 22, 'middle']][i] : [0, 23, 'middle'];
      T(pg, x + dx, yy + dy, y.toFixed(1), `font:${k === 'ours' ? 700 : 500} 12px var(--mono);fill:${col}`, an);
      pg.addEventListener('pointermove', e => showTip(`<b>${name}</b> · ${['same window', '1 window apart', '≥2 windows apart'][i]}<table><tr><td>pass rate</td><td>${y.toFixed(1)}%</td></tr><tr><td>checks, 3 seeds</td><td>${n[i]}</td></tr>${d ? `<tr><td>vs CaT-Movie</td><td>${d[i][0] > 0 ? '+' : ''}${d[i][0].toFixed(1)} [${d[i][1]}, ${d[i][2]}]</td></tr>` : ''}</table>`, e)); pg.addEventListener('pointerleave', hideTip);
    });
    anims.push({ path, g, si });
  });
  once(host, () => { if (RM) return; anims.forEach(({ path, g, si }) => { const L = path.getTotalLength(); const dash = path.getAttribute('stroke-dasharray'); path.style.strokeDasharray = `${L}`; path.style.strokeDashoffset = L; path.getBoundingClientRect(); path.style.transition = `stroke-dashoffset 1.3s var(--ease) ${si * 250}ms`; path.style.strokeDashoffset = 0; setTimeout(() => { path.style.strokeDasharray = dash === 'none' ? '' : dash; path.style.transition = ''; path.style.strokeDashoffset = ''; }, 1400 + si * 250); $$('g', g).forEach((pg, i) => pg.animate([{ opacity: 0, transform: 'scale(.4)' }, { opacity: 1, transform: 'none' }], { duration: 500, delay: si * 250 + i * 320, fill: 'both', easing: 'cubic-bezier(.2,.7,.1,1)' })); }); }, 0.35);
});

/* =============== 9 · block length histogram + curation yield (dataset statistics b, d) =============== */
run('dataStats', function dataStats() {
  const hb = $('#blockHist');
  if (hb) {
    const HB = [15459, 12498, 8388, 5959, 4301, 4242, 5642, 4826, 4864, 4136, 3100, 2046];
    const W = 600, H = 270, m = { l: 46, r: 12, t: 26, b: 44 }; const bw = (W - m.l - m.r) / 12, Y = v => H - m.b - v / 20000 * (H - m.t - m.b);
    const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Histogram of shots per scene block, mode one shot, median four' }, hb);
    const grid = sv('g', { class: 'grid' }, svg), ax = sv('g', { class: 'axis' }, svg);
    [0, 5000, 10000, 15000, 20000].forEach(v => { sv('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid); sv('text', { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end' }, ax).textContent = v ? (v / 1000) + 'k' : '0'; });
    const bars = HB.map((v, i) => { const x = m.l + i * bw + 5; const g = sv('g', { style: 'cursor:pointer' }, svg); const r = sv('rect', { x, y: Y(v), width: bw - 10, height: Y(0) - Y(v), rx: 2, fill: i === 0 ? C.ours : 'rgba(138,168,255,.55)' }, g); T(svg, x + (bw - 10) / 2, H - m.b + 17, i + 1, MONO(10.5), 'middle'); g.addEventListener('pointermove', e => showTip(`<b>${i + 1} shot${i ? 's' : ''} per block</b><table><tr><td>blocks</td><td>${v.toLocaleString('en-US')}</td></tr><tr><td>share</td><td>${(v / 75461 * 100).toFixed(1)}%</td></tr></table>`, e)); g.addEventListener('pointerleave', hideTip); return { r, v }; });
    T(svg, (W + m.l) / 2, H - 4, 'Shots per block', BODY(11.5, C.mute), 'middle');
    const mx = m.l + 3 * bw + bw / 2; const med = sv('g', { class: 'hb-med' }, svg); sv('path', { d: `M${mx} ${m.t - 6}V${Y(0)}`, stroke: C.cut, 'stroke-width': 2, 'stroke-dasharray': '6 5' }, med); T(med, mx + 8, m.t + 26, 'Median = 4', `font:700 13px var(--body);fill:${C.cut}`);
    T(svg, m.l + bw / 2, Y(15459) - 22, '15.5k', `font:700 12.5px var(--mono);fill:${C.ours}`, 'middle'); T(svg, m.l + bw / 2, Y(15459) - 8, '(mode)', MONO(9.5, C.ours), 'middle');
    once(hb, () => { if (RM) return; bars.forEach(({ r, v }, i) => r.animate([{ height: '0px', y: `${Y(0)}px` }, { height: `${Y(0) - Y(v)}px`, y: `${Y(v)}px` }], { duration: 700, delay: i * 45, easing: 'cubic-bezier(.2,.7,.1,1)', fill: 'backwards' })); med.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 600, delay: 700, fill: 'backwards' }); }, 0.35);
  }
  const cy = $('#yieldChart'); if (!cy) return;
  const D = [['Usable shots', 226648, 290267, '+28.1%'], ['Adjacent trainable pairs', 140120, 166550, '+18.9%']];
  const W = 600, H = 270, m = { l: 50, r: 12, t: 30, b: 44 }; const Y = v => H - m.b - v / 320000 * (H - m.t - m.b);
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Window-first versus shot-first filtering yield' }, cy);
  const defs = sv('defs', {}, svg); const hp = sv('pattern', { id: 'hatchW', width: 7, height: 7, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs); sv('rect', { width: 7, height: 7, fill: 'rgba(214,144,187,.22)' }, hp); sv('path', { d: 'M0 0V7', stroke: C.cut, 'stroke-width': 1.6 }, hp);
  const grid = sv('g', { class: 'grid' }, svg), ax = sv('g', { class: 'axis' }, svg);
  [0, 100000, 200000, 300000].forEach(v => { sv('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid); sv('text', { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end' }, ax).textContent = v ? (v / 1000) + 'k' : '0'; });
  const anim = [];
  D.forEach(([n, a, b, up], i) => {
    const gx = m.l + 40 + i * 270, bw = 86;
    const r1 = sv('rect', { x: gx, y: Y(a), width: bw, height: Y(0) - Y(a), fill: 'url(#hatchW)', stroke: C.cut, 'stroke-width': 1, rx: 2 }, svg);
    const r2 = sv('rect', { x: gx + bw + 6, y: Y(b), width: bw, height: Y(0) - Y(b), fill: C.ours, rx: 2 }, svg);
    T(svg, gx + bw / 2, Y(a) - 8, `${Math.round(a / 1000)}k`, MONO(11.5, C.paper, 600), 'middle');
    const l2 = T(svg, gx + bw * 1.5 + 6, Y(b) - 24, `${Math.round(b / 1000)}k`, `font:700 13px var(--mono);fill:${C.ours}`, 'middle'); const l3 = T(svg, gx + bw * 1.5 + 6, Y(b) - 9, up, `font:700 11.5px var(--mono);fill:${C.ours}`, 'middle');
    T(svg, gx + bw + 3, H - m.b + 20, n, BODY(12.5, C.paper, 500), 'middle');
    [r1, r2].forEach((r, j) => r.addEventListener('pointermove', e => showTip(`<b>${n}</b><table><tr><td>window-first</td><td>${a.toLocaleString('en-US')}</td></tr><tr><td>shot-first</td><td>${b.toLocaleString('en-US')}</td></tr><tr><td>gain</td><td>${up}</td></tr></table>`, e))); [r1, r2].forEach(r => r.addEventListener('pointerleave', hideTip));
    anim.push([r1, a, 0], [r2, b, 1], [l2, 0, 2], [l3, 0, 2]);
  });
  once(cy, () => { if (RM) return; anim.forEach(([el, v, k], i) => { if (k === 2) el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 500, delay: 900, fill: 'backwards' }); else el.animate([{ height: '0px', y: `${Y(0)}px` }, { height: `${Y(0) - Y(v)}px`, y: `${Y(v)}px` }], { duration: 800, delay: k * 300 + i * 60, easing: 'cubic-bezier(.2,.7,.1,1)', fill: 'backwards' }); }); }, 0.35);
});
})();
