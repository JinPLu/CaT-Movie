/* CaT-Movie project page: interactive Figure 2 architecture, Pareto + profile, Figure 3 benchmark & dataset pipelines, sunburst. */
(() => {
'use strict';
const { MEDIA, $, $$, sv, h, esc, once, watch, onReveal, chart, setTab, showTip, hideTip, pick, RM, T1, POINT_COLOR, run, story, frame, short, camLabel, pad, HERO, avatarStyle, charTip } = window.CAT;
const EASE = 'cubic-bezier(.2,.72,.2,1)';
const T = (g, x, y, str, cls = 'svg-text', style = '', anchor = 'start') => { const t = sv('text', { x, y, 'text-anchor': anchor, class: cls, style }, g); t.textContent = str; return t; };
const svgAt = (host, W, H, label) => sv('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': label }, host);

const grow = (els, origin = 'bottom', base = 0, step = 32) => els.forEach((el, i) => {
  el.style.transformBox = 'fill-box'; el.style.transformOrigin = origin;
  el.animate([{ transform: origin === 'left' ? 'scaleX(0)' : 'scaleY(0)' }, { transform: 'none' }], { duration: 600, delay: base + i * step, easing: EASE, fill: 'backwards' });
});
const drawLine = (p, delay = 0, dur = 700) => {
  const L = p.getTotalLength(), dash = p.getAttribute('stroke-dasharray');
  p.animate([{ strokeDasharray: `${L} ${L}`, strokeDashoffset: L }, { strokeDasharray: `${L} ${L}`, strokeDashoffset: 0 }], { duration: dur, delay, easing: EASE, fill: 'backwards' })
    .onfinish = () => { if (dash) p.setAttribute('stroke-dasharray', dash); };
};
const pop = (els, base = 0, step = 45) => els.forEach((el, i) => {
  el.style.transformBox = 'fill-box'; el.style.transformOrigin = 'center';
  el.animate([{ opacity: 0, transform: 'scale(.4)' }, { opacity: 1, transform: 'none' }], { duration: 360, delay: base + i * step, easing: EASE, fill: 'backwards' });
});
const fadeIn = (els, base = 0, step = 0) => els.forEach((el, i) => el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 380, delay: base + i * step, fill: 'backwards' }));
const gridY = (svg, m, W, Y, ticks, fmt) => {
  const grid = sv('g', { class: 'grid' }, svg), ax = sv('g', { class: 'axis' }, svg);
  ticks.forEach(v => { sv('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid); T(ax, m.l - 8, Y(v) + 4, fmt(v), 'svg-text', '', 'end'); });
  return ax;
};

/* =============== 04 Results: Pareto frontier + Linked Method Profile Card =============== */
run('pareto', function pareto() {
  const host = $('#pareto'), prof = $('#methodProf');
  if (!host) return;
  let nodes = [], pinned = null;
  const pts = T1.filter(r => r.name).map(r => ({
    name: r.name, fam: r.fam,
    tr: (r.v[0] + r.v[1]) / 2,
    co: (r.v[2] + r.v[3] + r.v[4]) / 3,
    st: (r.v[5] + r.v[6]) / 2,
    r,
  }));
  const oursPt = pts.find(p => p.fam === 'ours');

  function renderProfile(name) {
    if (!prof) return;
    const p = pts.find(x => x.name === name) || oursPt;
    const col = POINT_COLOR[p.name] || 'var(--ours)';
    const famLabel = p.fam === 'ours' ? 'Global Plan + Window Render' : p.fam === 'joint' ? 'Joint Multi-Shot Baseline' : 'Per-Shot / Memory Baseline';
    const axes = [
      ['Transition control', p.tr, oursPt.tr],
      ['Character consistency', p.r.v[2], oursPt.r.v[2]],
      ['Object & scene mean', (p.r.v[3] + p.r.v[4]) / 2, (oursPt.r.v[3] + oursPt.r.v[4]) / 2],
      ['State continuity', p.st, oursPt.st],
      ['CaT-MBench overall', p.r.v[8], oursPt.r.v[8]],
    ];
    prof.innerHTML = `<div class="prof-top">
        <span class="prof-fam">${famLabel}</span>
        <h4 class="prof-name"><i style="background:${col}"></i>${esc(p.name)}</h4>
        <p class="prof-meta">${esc(p.r.base)} &mdash; ${esc(p.r.input)}</p>
      </div>
      <div class="prof-bars">${axes.map(([lab, val, ref]) => `<div class="pbar-row">
        <div class="pbar-hd"><span>${lab}</span><b>${val.toFixed(3)}</b></div>
        <div class="pbar-tr">
          <div class="pbar-fill" style="width:${(val * 100).toFixed(1)}%;background:${col}"></div>
          ${p.fam !== 'ours' ? `<span class="pbar-ref" style="left:${(ref * 100).toFixed(1)}%" title="CaT-Movie: ${ref.toFixed(3)}"></span>` : ''}
        </div>
      </div>`).join('')}</div>
      <div class="prof-foot">${p.fam === 'ours' ? 'Hover or click any baseline point or table row to compare against CaT-Movie.' : 'Blue tick marks on each bar indicate CaT-Movie (5B) reference performance.'}</div>`;
  }

  const mark = name => {
    nodes.forEach(n => { n.classList.toggle('hl', n.dataset.name === name); n.classList.toggle('dim', !!name && n.dataset.name !== name); });
    renderProfile(name || pinned);
  };
  document.addEventListener('pick', e => { if (e.detail.from !== 'pareto') mark(e.detail.name || pinned); });
  renderProfile(null);

  chart(host, W => {
    const narrow = W < 480;
    const H = Math.round(Math.max(295, Math.min(390, W * 0.66)));
    const m = { l: 48, r: narrow ? 12 : 20, t: 14, b: 46 };
    const x0 = 0.785, x1 = 1.014, y0 = 0.17, y1 = 0.64;
    const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
    const Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
    const svg = svgAt(host, W, H, 'Inter-shot consistency against transition control for eight methods on 92 test stories');
    const grid = sv('g', { class: 'grid' }, svg), ax = sv('g', { class: 'axis' }, svg);
    (narrow ? [0.80, 0.90, 1.00] : [0.80, 0.85, 0.90, 0.95, 1.00]).forEach(v => { sv('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }, grid); T(ax, X(v), H - m.b + 18, v.toFixed(2), 'svg-text', '', 'middle'); });
    [0.2, 0.3, 0.4, 0.5, 0.6].forEach(v => { sv('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid); T(ax, m.l - 8, Y(v) + 4, v.toFixed(1), 'svg-text', '', 'end'); });
    sv('path', { d: `M${m.l} ${m.t} V${H - m.b} H${W - m.r}`, fill: 'none' }, ax);
    T(ax, (m.l + W - m.r) / 2, H - 8, narrow ? 'Transition control →' : 'Transition control (Cut F1 & Execution) →', 'svg-text', 'fill:var(--text)', 'middle');
    T(ax, 0, 0, narrow ? 'Consistency →' : 'Inter-shot consistency →', 'svg-text', 'fill:var(--text)', 'middle').setAttribute('transform', `translate(13 ${(m.t + H - m.b) / 2}) rotate(-90)`);

    const base = pts.filter(p => p.fam !== 'ours'), ours = oursPt;
    const front = base.slice().sort((a, b) => b.tr - a.tr).reduce((acc, p) => { if (!acc.length || p.co > acc[acc.length - 1].co) acc.push(p); return acc; }, []);
    const f = xv => Math.max(...base.filter(p => p.tr >= xv).map(p => p.co), y0);
    const bps = front.map(p => p.tr).filter(v => v < ours.tr).sort((a, b) => b - a);
    let poly = `M${X(bps[bps.length - 1])} ${Y(ours.co)} H${X(ours.tr)} V${Y(f(ours.tr))}`;
    bps.forEach((bx, i) => { poly += ` H${X(bx)}`; if (i < bps.length - 1) poly += ` V${Y(f(bx - 1e-6))}`; });
    const region = sv('path', { d: poly + 'Z', fill: 'var(--ours)', 'fill-opacity': 0.13, stroke: 'none' }, svg);
    let st = `M${X(front[0].tr)} ${Y(y0)} V${Y(front[0].co)}`;
    for (let i = 1; i < front.length; i++) st += ` H${X(front[i].tr)} V${Y(front[i].co)}`;
    st += ` H${X(x0)}`;
    const fp = sv('path', { d: st, fill: 'none', stroke: 'var(--grey-b)', 'stroke-width': 1.5, 'stroke-dasharray': '4 4' }, svg);

    const LBL = {
      StoryMem: [0, -14, 'middle'], HoloCine: [13, 5, 'start'], MultiShotMaster: [-13, 20, 'end'], 'CaT-Movie': [-16, 5, 'end'],
      StoryDiffusion: [11, 4, 'start'], 'IC-LoRA': [-11, 4, 'end'], 'Wan2.2, per shot': [-10, -10, 'end'], 'Wan2.1, per shot': [-10, 18, 'end'],
    };
    nodes = [];
    pts.forEach(p => {
      const g = sv('g', { class: 'pt', tabindex: 0, 'data-name': p.name }, svg);
      const cx = X(p.tr), cy = Y(p.co), col = POINT_COLOR[p.name];
      if (p.fam === 'shot') sv('circle', { cx, cy, r: 6.8, fill: col, stroke: 'var(--card)', 'stroke-width': 2 }, g);
      else {
        const s = p.fam === 'ours' ? 11.5 : 8.8;
        if (p.fam === 'ours') sv('rect', { x: cx - s, y: cy - s, width: 2 * s, height: 2 * s, transform: `rotate(45 ${cx} ${cy})`, fill: 'none', stroke: col, 'stroke-opacity': 0.32, 'stroke-width': 6.5 }, g);
        sv('rect', { x: cx - s * 0.72, y: cy - s * 0.72, width: 1.44 * s, height: 1.44 * s, transform: `rotate(45 ${cx} ${cy})`, fill: col, stroke: 'var(--card)', 'stroke-width': 2 }, g);
      }
      const [dx, dy, an] = LBL[p.name];
      const nm = narrow ? p.name.replace(', per shot', '') : p.name;
      T(g, cx + dx, cy + dy, nm, 'svg-text', `fill:${p.fam === 'shot' && p.name !== 'StoryMem' ? 'var(--mute)' : col};font-weight:${p.fam === 'shot' ? 450 : 600}`, an);
      const v = p.r.v;
      const html = `<b>${esc(p.name)}</b><table><tr><td>Transition control</td><td>${p.tr.toFixed(3)}</td></tr><tr><td>Inter-shot consistency</td><td>${p.co.toFixed(3)}</td></tr><tr><td>Character</td><td>${v[2].toFixed(3)}</td></tr><tr><td>Average</td><td>${v[8].toFixed(3)}</td></tr></table>`;
      g.addEventListener('pointermove', e => showTip(html, e));
      g.addEventListener('pointerenter', () => { mark(p.name); pick(p.name, 'pareto'); });
      g.addEventListener('pointerleave', () => { hideTip(); mark(pinned); pick(pinned, 'pareto'); });
      g.addEventListener('click', () => { pinned = pinned === p.name ? null : p.name; mark(pinned); pick(pinned, 'pareto'); });
      g.addEventListener('focus', () => { const r = g.getBoundingClientRect(); showTip(html, { clientX: r.right, clientY: r.top }); mark(p.name); pick(p.name, 'pareto'); });
      g.addEventListener('blur', () => { hideTip(); mark(pinned); pick(pinned, 'pareto'); });
      nodes.push(g);
    });
    mark(pinned);
    return () => {
      fadeIn([region], 450);
      drawLine(fp, 0, 820);
      pop(nodes.filter(n => n.dataset.name !== 'CaT-Movie'), 140, 55);
      pop(nodes.filter(n => n.dataset.name === 'CaT-Movie'), 740);
    };
  });
});

/* =============== 04 Results: Character retention by return gap =============== */
run('returnChart', function returnChart() {
  const host = $('#returnChart');
  if (!host) return;
  const RT = {
    'CaT-Movie': { rate: [0.6331, 0.6111, 0.6344], lo: [0.5710, 0.5257, 0.5592], hi: [0.7044, 0.7178, 0.7129] },
    HoloCine: { rate: [0.6115, 0.4091, 0.4355], lo: [0.5369, 0.3209, 0.3402], hi: [0.6906, 0.5023, 0.5517] },
    MultiShotMaster: { rate: [0.5791, 0.4444, 0.4892], lo: [0.5143, 0.3432, 0.3423], hi: [0.6558, 0.5489, 0.6172] },
    StoryMem: { rate: [0.5827, 0.5960, 0.7312], lo: [0.5075, 0.5060, 0.6646], hi: [0.6771, 0.6835, 0.7873] },
  };
  const N = [278, 198, 186], bins = ['under 10 s', '10–20 s', '20 s or more'], names = Object.keys(RT);
  const OP = [0.9, 0.58, 0.34];
  const fill = (g, s) => names[g] === 'CaT-Movie' ? `color-mix(in srgb,var(--ours) ${[95, 64, 38][s]}%,transparent)` : `color-mix(in srgb,var(--grey-b) ${OP[s] * 100}%,transparent)`;
  $('#rtLegend').innerHTML = bins.map((b, s) => `<span><i class="sq" style="background:color-mix(in srgb,var(--grey-b) ${OP[s] * 100}%,transparent)"></i>${b}</span>`).join('') + '<span><i class="sq" style="background:var(--ours)"></i>CaT-Movie</span>';
  chart(host, W => {
    const narrow = W < 520;
    const H = 280, m = { l: 44, r: 8, t: 12, b: narrow ? 50 : 36 };
    const yMin = 0.3, yMax = 0.8;
    const Y = v => H - m.b - (v - yMin) / (yMax - yMin) * (H - m.t - m.b);
    const svg = svgAt(host, W, H, 'Character checks passed by return time for four methods');
    gridY(svg, m, W, Y, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], v => Math.round(v * 100) + '%');
    sv('path', { d: `M${m.l} ${Y(yMin)} H${W - m.r}`, stroke: 'var(--rule)' }, svg);
    const gw = (W - m.l - m.r) / names.length, bw = Math.min(34, (gw - 28) / 3);
    const bars = [], whisk = [];
    names.forEach((n, gi) => {
      const cx = m.l + gi * gw + gw / 2, gx = cx - (bw * 3 + 8) / 2;
      const lines = narrow && n === 'MultiShotMaster' ? ['MultiShot', 'Master'] : [n];
      lines.forEach((ln, li) => T(svg, cx, H - m.b + 20 + li * 15, ln, 'svg-text', `fill:${n === 'CaT-Movie' ? 'var(--ours)' : 'var(--text)'};font-weight:${n === 'CaT-Movie' ? 600 : 450}`, 'middle'));
      bins.forEach((b, s) => {
        const d = RT[n], x = gx + s * (bw + 4);
        const g = sv('g', { style: 'cursor:pointer' }, svg);
        bars.push(sv('rect', { x, y: Y(d.rate[s]), width: bw, height: Y(yMin) - Y(d.rate[s]), rx: 3, fill: fill(gi, s) }, g));
        whisk.push(sv('path', { d: `M${x + bw / 2} ${Y(d.lo[s])} V${Y(d.hi[s])} M${x + bw / 2 - 3.5} ${Y(d.lo[s])} h7 M${x + bw / 2 - 3.5} ${Y(d.hi[s])} h7`, stroke: 'var(--paper)', 'stroke-opacity': 0.5, 'stroke-width': 1.2, fill: 'none' }, g));
        g.addEventListener('pointermove', e => showTip(`<b>${n}</b> (${b})<table><tr><td>Pass rate</td><td>${(d.rate[s] * 100).toFixed(1)}%</td></tr><tr><td>95% CI</td><td>${(d.lo[s] * 100).toFixed(1)}–${(d.hi[s] * 100).toFixed(1)}%</td></tr><tr><td>Checks</td><td>${N[s]}</td></tr></table>`, e));
        g.addEventListener('pointerleave', hideTip);
      });
    });
    return () => { grow(bars, 'bottom', 90, 28); fadeIn(whisk, 640, 14); };
  });
});

/* =============== 02 Method Workbench: Consistency by window distance =============== */
run('distChart', function distChart() {
  const host = $('#distChart');
  if (!host) return;
  const S = [
    ['ours', 'CaT-Movie', 'var(--ours)', [46.3, 53.9, 59.8], [147, 297, 480], null, 'circle', ''],
    ['recent', 'Latest-shot recall', 'var(--grey-a)', [46.9, 54.9, 44.4], [147, 297, 480], [[0.7, -5.6, 9.4], [1.0, -5.1, 7.3], [-15.4, -22.7, -9.0]], 'square', '7 5'],
    ['no_refs', 'No recalled views', 'var(--cut)', [38.8, 31.3, 32.1], [147, 297, 480], [[-7.5, -20.2, 3.2], [-22.6, -30.6, -14.4], [-27.7, -35.3, -20.7]], 'tri', '2 5'],
  ];
  const XL = ['same window', '1 window apart', '2 or more windows apart'];
  chart(host, W => {
    const H = 265, m = { l: 44, r: 18, t: 20, b: 38 };
    const padX = Math.min(68, (W - m.l - m.r) * 0.13);
    const X = i => m.l + padX + i * ((W - m.l - m.r - 2 * padX) / 2);
    const Y = v => H - m.b - (v - 25) / (65 - 25) * (H - m.t - m.b);
    const svg = svgAt(host, W, H, 'Consistency pass rate by window distance for three settings');
    gridY(svg, m, W, Y, [30, 40, 50, 60], v => v + '%');
    XL.forEach((n, i) => T(svg, X(i), H - m.b + 22, n, 'svg-text', 'fill:var(--text)', 'middle'));
    const lines = [], marks = [];
    S.forEach(([k, name, col, v, n, d, mk, dash]) => {
      const g = sv('g', {}, svg);
      lines.push(sv('path', { d: v.map((y, i) => `${i ? 'L' : 'M'}${X(i)} ${Y(y)}`).join(''), fill: 'none', stroke: col, 'stroke-width': k === 'ours' ? 2.8 : 2, 'stroke-dasharray': dash || null, 'stroke-linecap': 'round' }, g));
      v.forEach((y, i) => {
        const x = X(i), yy = Y(y), pg = sv('g', { style: 'cursor:pointer' }, g);
        if (mk === 'circle') sv('circle', { cx: x, cy: yy, r: 5.5, fill: col, stroke: 'var(--card)', 'stroke-width': 2 }, pg);
        else if (mk === 'square') sv('rect', { x: x - 5, y: yy - 5, width: 10, height: 10, fill: col, stroke: 'var(--card)', 'stroke-width': 2 }, pg);
        else sv('path', { d: `M${x} ${yy - 6.5}L${x + 6} ${yy + 4.5}L${x - 6} ${yy + 4.5}Z`, fill: col, stroke: 'var(--card)', 'stroke-width': 2 }, pg);
        const [dx, dy, an] = k === 'ours' ? [[-11, 5, 'end'], [0, 21, 'middle'], [0, -12, 'middle']][i] : k === 'recent' ? [[0, -12, 'middle'], [0, -12, 'middle'], [0, 21, 'middle']][i] : [0, 22, 'middle'];
        T(pg, x + dx, yy + dy, y.toFixed(1) + '%', 'svg-text', `fill:${col};font-weight:${k === 'ours' ? 600 : 450}`, an);
        pg.addEventListener('pointermove', e => showTip(`<b>${name}</b> (${XL[i]})<table><tr><td>Pass rate</td><td>${y.toFixed(1)}%</td></tr><tr><td>Checks (3 seeds)</td><td>${n[i]}</td></tr>${d ? `<tr><td>vs CaT-Movie</td><td>${d[i][0] > 0 ? '+' : ''}${d[i][0].toFixed(1)} [${d[i][1]}, ${d[i][2]}]</td></tr>` : ''}</table>`, e));
        pg.addEventListener('pointerleave', hideTip);
        marks.push(pg);
      });
    });
    return () => { lines.forEach((p, i) => drawLine(p, i * 90, 660)); pop(marks, 180, 38); };
  });
});

/* =============== 02 Method Workbench: Planner study charts =============== */
run('planner', function planner() {
  const a = $('#planText'), b = $('#writExec');
  if (!a || !b) return;
  const D = { opening: { Syn: [0.1625, 0.4471], Para: [0.5649, 0.7091], Shot: [0.4833, 0.7304] }, held: { Syn: [0.0732, 0.2171], Para: [0.5721, 0.6797], Shot: [0.3122, 0.5122] } };
  const TL = { Syn: 'synopsis', Para: 'paragraph', Shot: 'shot list' };
  chart(a, W => {
    const H = 265, m = { l: 42, r: 8, t: 20, b: 50 };
    const Y = v => H - m.b - v * (H - m.t - m.b);
    const svg = svgAt(a, W, H, 'Share of reference opening states and held items stated by request alone and with planner');
    gridY(svg, m, W, Y, [0, 0.25, 0.5, 0.75, 1], v => (v * 100) + '%');
    const gw = (W - m.l - m.r) / 2, bw = Math.min(48, (gw - 34) / 3), segs = [], labs = [];
    [['opening', 'Opening states'], ['held', 'Held items']].forEach(([key, name], gi) => {
      const cx = m.l + gi * gw + gw / 2;
      T(svg, cx, H - 10, name, 'svg-text', 'fill:var(--paper);font-weight:500', 'middle');
      ['Syn', 'Para', 'Shot'].forEach((t, ti) => {
        const [r, tot] = D[key][t], x = cx + (ti - 1) * (bw + 10) - bw / 2;
        const g = sv('g', { style: 'cursor:pointer' }, svg);
        segs.push(sv('rect', { x, y: Y(r), width: bw, height: Y(0) - Y(r), fill: 'var(--grey-c)', 'fill-opacity': 0.5, rx: 2.5 }, g));
        const top = sv('rect', { x, y: Y(tot), width: bw, height: Y(r) - Y(tot), fill: 'var(--ours)', rx: 2.5 }, g);
        labs.push(T(g, x + bw / 2, Y(tot) - 7, `${Math.round(tot * 100)}%`, 'svg-text', 'fill:var(--paper);font-weight:600;font-size:11.5px', 'middle'));
        T(svg, x + bw / 2, H - m.b + 17, TL[t], 'svg-text', 'font-size:11px', 'middle');
        g.addEventListener('pointermove', e => showTip(`<b>${name} (${TL[t]})</b><table><tr><td>Request alone</td><td>${(r * 100).toFixed(1)}%</td></tr><tr><td>Planner adds</td><td>+${((tot - r) * 100).toFixed(1)}%</td></tr><tr><td>Compiled total</td><td>${(tot * 100).toFixed(1)}%</td></tr></table>`, e));
        g.addEventListener('pointerleave', hideTip);
        segs.push(top);
      });
    });
    return () => { grow(segs, 'bottom', 0, 28); fadeIn(labs, 560, 28); };
  });

  const R = [['Request', 170, 79, 49, 42], ['Reference', 171, 91, 59, 21], ['Our planner', 171, 108, 48, 15]];
  chart(b, W => {
    const H = 265, m = { l: 42, r: 8, t: 16, b: 56 };
    const Y = v => H - m.b - v * (H - m.t - m.b);
    const svg = svgAt(b, W, H, 'State questions shown, stated not shown, or unstated');
    gridY(svg, m, W, Y, [0, 0.25, 0.5, 0.75, 1], v => (v * 100) + '%');
    const defs = sv('defs', {}, svg);
    const hp = sv('pattern', { id: 'hatchP', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
    sv('rect', { width: 6, height: 6, fill: 'var(--cut)', 'fill-opacity': 0.16 }, hp);
    sv('path', { d: 'M0 0V6', stroke: 'var(--cut)', 'stroke-width': 2 }, hp);
    const gw = (W - m.l - m.r) / 3, bw = Math.min(72, gw - 28), parts = [], labs = [];
    R.forEach(([name, n, se, sn, ns], i) => {
      const x = m.l + i * gw + (gw - bw) / 2, f1 = se / n, f2 = sn / n, f3 = ns / n;
      const g = sv('g', { style: 'cursor:pointer' }, svg);
      parts.push(sv('rect', { x, y: Y(f1), width: bw, height: Y(0) - Y(f1), fill: 'var(--ours)', rx: 2.5 }, g));
      parts.push(sv('rect', { x, y: Y(f1 + f2), width: bw, height: Y(f1) - Y(f1 + f2), fill: 'url(#hatchP)', rx: 2.5 }, g));
      sv('rect', { x: x + 0.5, y: Y(1) + 0.5, width: bw - 1, height: Y(f1 + f2) - Y(1) - 1, fill: 'none', stroke: 'var(--rule)', 'stroke-dasharray': '3 3', rx: 2.5 }, g);
      const ours = name === 'Our planner';
      T(svg, x + bw / 2, H - m.b + 18, name, 'svg-text', `fill:${ours ? 'var(--ours)' : 'var(--paper)'};font-weight:600`, 'middle');
      T(svg, x + bw / 2, H - m.b + 34, `${Math.round((se + sn) / n * 100)}% stated`, 'svg-text', 'font-size:11px', 'middle');
      labs.push(T(g, x + bw / 2, Y(f1 / 2) + 4, Math.round(f1 * 100) + '%', 'svg-text', 'fill:#fff;font-weight:600;font-size:12px', 'middle'));
      labs.push(T(g, x + bw / 2, Y(f1 + f2 / 2) + 4, Math.round(f2 * 100) + '%', 'svg-text', 'fill:var(--paper);font-weight:600;font-size:12px;paint-order:stroke;stroke:var(--card);stroke-width:3px', 'middle'));
      if (f3 >= 0.08) labs.push(T(g, x + bw / 2, Y(f1 + f2 + f3 / 2) + 4, Math.round(f3 * 100) + '%', 'svg-text', 'font-weight:600;font-size:11.5px', 'middle'));
      g.addEventListener('pointermove', e => showTip(`<b>${name} storyboard</b><table><tr><td>Stated &amp; shown</td><td>${se} (${(f1 * 100).toFixed(0)}%)</td></tr><tr><td>Stated, not shown</td><td>${sn} (${(f2 * 100).toFixed(0)}%)</td></tr><tr><td>Unstated</td><td>${ns} (${(f3 * 100).toFixed(0)}%)</td></tr></table>`, e));
      g.addEventListener('pointerleave', hideTip);
    });
    return () => { grow(parts, 'bottom', 0, 50); fadeIn(labs, 500, 22); };
  });
});

/* =============== 02 Method: Interactive System Architecture (Figure 2, live W1..W5 & Stages 1..7) =============== */
run('overview', function overview() {
  const fig = $('#ovFig');
  if (!fig) return;
  const s = story(HERO);
  const ICE = '<svg class="ico" viewBox="0 0 14 14" aria-label="frozen"><g stroke="var(--ours)" stroke-width="1.35" stroke-linecap="round" fill="none"><path d="M7 1.5v11M2.2 4.25l9.6 5.5M2.2 9.75l9.6-5.5"/><path d="M5.5 2.5 7 3.9l1.5-1.4M5.5 11.5 7 10.1l1.5 1.4"/></g></svg>';
  const FLAME = '<svg class="ico" viewBox="0 0 14 14" aria-label="trainable"><path d="M7 13.2c-2.9 0-4.4-2-4.4-4.1 0-2.5 2-3.6 2.5-6.3 1 .9 1.5 1.9 1.5 3 .7-.5 1.1-1.5 1-2.8 2 1.4 3.9 3.6 3.9 6 0 2.3-1.7 4.2-4.5 4.2z" fill="#e07b39"/></svg>';
  const navSvg = d => `<svg viewBox="0 0 10 10"><path d="${d}" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`;
  const ST_NAMES = ['Request', 'Storyboard', 'Compile', 'Recall', 'Encode', 'Denoise', 'Decode'];

  let WI = 3, stage = 0, reader = 8, touring = false, timers = [];

  const named = k => new Set(s.requests.shots[k - 1].match(/Character \d+/g) || []);
  const STOP = new Set(['with', 'that', 'this', 'from', 'into', 'while', 'their']);
  const posture = sh => { const c = sh.present.find(p => /^Character/.test(p.name) && p.posture); return c ? c.posture.split(',')[0] : ''; };
  const prov = (k, text) => {
    const req = s.requests.shots[k - 1].toLowerCase();
    const words = (text.toLowerCase().replace(/character \d+/g, '').match(/[a-z]+/g) || []).filter(w => w.length > 3 && !STOP.has(w));
    return words.some(w => req.includes(w)) ? '' : 'cmp';
  };
  const shotList = a => a.length ? a.join(', ').replace(/, (\d+)$/, ' and $1') : 'none';

  const ph = '<i class="ov-dn ph" aria-hidden="true"></i>', dn = '<i class="ov-dn" aria-hidden="true"></i>';
  fig.innerHTML = `<div class="ov-bar">
      <div class="ov-bar-left">
        <h3>Interactive Architecture</h3>
        <div class="tabs" id="ovWinTabs" role="group" aria-label="Active window">${s.windows.map((w, i) => `<button type="button" data-w="${i}" class="${i === WI ? 'on' : ''}" aria-pressed="${i === WI}">W${i + 1}</button>`).join('')}</div>
      </div>
      <div class="ov-ctl">
        <div class="ov-stages" id="ovTabs" role="group" aria-label="Pipeline stages">${ST_NAMES.map((n, i) => `<button type="button" data-n="${i + 1}" aria-pressed="false"><b>${i + 1}</b>${n}</button>`).join('')}</div>
        <button type="button" class="ibtn" data-d="-1" aria-label="Previous stage">${navSvg('M6.5 1.5 3 5l3.5 3.5')}</button>
        <button type="button" class="ibtn" data-d="1" aria-label="Next stage">${navSvg('M3.5 1.5 7 5 3.5 8.5')}</button>
        <button type="button" class="tbtn" id="ovPlay"></button>
      </div>
    </div>
    <div class="ov-map" id="ovMap">
      <div class="ov-reg ov-pl">
        <p class="ov-rt">CaT-Movie Planner<small>global once</small></p>
        <div class="ovn box" data-st="1" id="ovReqBox"></div>
        ${dn}
        <div class="ovn pill" data-st="1" id="ovLlm">Input Adaptation <small>Qwen3.8-27B</small>${ICE}</div>
        ${dn}
        <div class="ovn box" data-st="2" id="ovBoard"></div>
        ${dn}
        <div class="ovn pill" data-st="3" id="ovComp">Prompt Compiler <small>code</small></div>
      </div>
      ${ph}
      <div class="ov-top">
        <div class="ovn bare" data-st="4,7" id="ovHist"></div>
        <div class="ov-tf"><span>${FLAME}Trainable</span><span>${ICE}Frozen</span></div>
      </div>
      ${ph}
      <div class="ov-mid">
        <div class="ov-mid-a">
          <div class="ovn pill" data-st="4" id="ovRecall">Frame Recall</div>
          ${dn}
          <div class="ovn pill" data-st="5" id="ovEnc">VAE Encoder${ICE}</div>
        </div>
        <div class="ov-mid-b">
          <div class="ovn pill" data-st="6" id="ovT5">umT5${ICE}</div>
          <i class="ov-dn up" aria-hidden="true"></i>
          <div class="ovn ov-stack" data-st="3" id="ovPrompts"></div>
        </div>
      </div>
      ${ph}
      <div class="ov-reg ov-rd">
        <p class="ov-rt">CaT-Movie Renderer<small>Wan2.2-TI2V-5B, zero added parameters</small></p>
        <div class="ov-rd-in">
          <div class="ov-tokb ovn" data-st="5" id="ovTokBox">
            <div class="ov-tok-head"><span class="ovl" style="margin:0">Latent Sequence &amp; RoPE Time</span><div class="tabs" id="ovReader" role="group" aria-label="Reading shot"></div></div>
            <div class="chart" id="ovTok"></div>
          </div>
          <div class="ovn ov-dit" data-st="6" id="ovDit">
            <span class="ovl" style="margin-bottom:4px">Transformer Block &times; N</span>
            <div class="ov-layer" id="ovSelf"><i class="dot" style="background:var(--ours)"></i>Reference-Routed Sparse Self-Attention${FLAME}</div>
            <div class="ov-layer" id="ovCross"><i class="dot" style="background:var(--holo)"></i>Shot-wise Cross-Attention${FLAME}</div>
            <div class="ov-layer"><i class="dot" style="background:var(--grey-c)"></i>Feed-Forward${FLAME}</div>
          </div>
          <div class="ov-dec ovn" data-st="7" id="ovDec">
            <div class="ovn pill" data-st="7" id="ovDecPill">VAE Decoder${ICE}</div>
            <div class="ov-outs" id="ovOuts"></div>
          </div>
          <div class="ov-mreg ovn" data-st="6" id="ovMaskBox">
            <p class="ov-rt">Reference-Routed Attention Mask</p>
            <div id="ovMask"></div>
            <span class="legend">
              <span><i class="sq" style="background:var(--grey-a)"></i>within own shot</span>
              <span><i class="sq" style="background:var(--holo)"></i>other shot first frame</span>
              <span><i class="sq" style="background:var(--ours)"></i>routed recalled frame</span>
            </span>
          </div>
        </div>
      </div>
      <svg class="ov-links" aria-hidden="true"></svg>
    </div>
    <div class="ov-foot"><p class="cap" id="ovCap" aria-live="polite"></p></div>`;

  const winTabs = $('#ovWinTabs', fig), stageTabs = $('#ovTabs', fig), tabBtns = $$('button', stageTabs);
  const capEl = $('#ovCap', fig), playBtn = $('#ovPlay', fig), navBtns = $$('.ov-ctl .ibtn', fig);
  const map = $('#ovMap', fig), links = $('.ov-links', fig), tokHost = $('#ovTok', fig);

  function stageCopy() {
    const W = s.windows[WI], shs = W.shots, refShots = [...new Set(W.refs.map(r => r.shot))];
    const sLabel = shs.length > 1 ? `Shots ${shotList(shs)}` : `Shot ${shs[0]}`;
    const ST = [
      ['Request', `User request lines go to frozen Qwen3.8-27B, which expands the whole story into one structured storyboard in a single pass.`],
      ['Storyboard', `The storyboard registers recurring characters, objects, and places once, then writes per-shot camera framing, present entities, and state transitions (${sLabel} form Window ${WI + 1}).`],
      ['Compile', `Deterministic code propagates attire and held items across shots and compiles one self-contained prompt per shot plus ${W.refs.length ? `an identity line for each recalled view` : `no reference lines for cold-start Window 1`}.`],
      ['Recall', W.refs.length ? `Code indexes anchor views from already-rendered windows: Window ${WI + 1} recalls ${W.refs.length} frame${W.refs.length > 1 ? 's' : ''} from Shot${refShots.length > 1 ? 's' : ''} ${shotList(refShots)}.` : `Window 1 has no prior rendered windows, so Frame Recall passes an empty reference prefix.`],
      ['Encode', `Each shot is VAE-encoded independently so no latent spans a cut. ${W.refs.length ? `Recalled latents take RoPE offsets spaced 5 steps apart (p_k − 5m) right before the reading shot.` : `Target latents occupy contiguous RoPE positions starting at 0.`}`],
      ['Denoise', `Across all N transformer blocks, each shot attends within itself, to the first latent frame of sibling window shots, and solely to the recalled frames routed to it, while cross-attending only to its own prompt.`],
      ['Decode', `The frozen VAE decoder reconstructs ${sLabel}, appending them to the generated film history for subsequent windows.`],
    ];
    if (!stage) {
      return `<b>Window ${WI + 1} rendering pass (${sLabel}).</b> The planner writes the global storyboard once; the renderer turns ${shs.length} compiled prompt${shs.length > 1 ? 's' : ''} and ${W.refs.length} recalled anchor frame${W.refs.length === 1 ? '' : 's'} into ${sLabel}. Click any window (W1–W5) or stage (1–7), or press Play.`;
    }
    return `<b>${stage}. ${ST[stage - 1][0]}.</b> ${ST[stage - 1][1]}`;
  }

  let placeTok = () => {};

  function buildWindowDOM() {
    const W = s.windows[WI], shs = W.shots;
    if (!shs.includes(reader)) reader = shs[0];
    const refByShot = new Map();
    W.refs.forEach((r, idx) => { if (!refByShot.has(r.shot)) refByShot.set(r.shot, []); refByShot.get(r.shot).push(`R${idx + 1}`); });

    /* 1. Planner Request & Storyboard */
    $('#ovReqBox', fig).innerHTML = `<span class="ovl">Request (shot list)</span>${shs.slice(0, 2).map(k => `<p class="ov-q"><b>S${pad(k)}</b>${esc(s.requests.shots[k - 1])}</p>`).join('')}`;
    const rowIndices = [];
    const lo = Math.max(1, shs[0] - 1), hi = Math.min(s.plan.shots.length, lo + 3);
    for (let k = Math.max(1, hi - 3); k <= hi; k++) rowIndices.push(k);
    $('#ovBoard', fig).innerHTML = `<span class="ovl">Storyboard</span>
      <div class="ov-avs">${s.plan.characters.map(c => `<span class="ov-av" data-e="${esc(c.name)}"><i style="${avatarStyle(c.name)}"></i>${esc(short(c.name))}</span>`).join('')}</div>
      <div class="ov-th"><span>Shot</span><span>Entities</span><span>Shot plan</span></div>
      ${rowIndices.map(k => {
        const sh = s.plan.shots[k - 1], who = sh.present.filter(p => /^Character/.test(p.name)).map(p => p.name), pos = posture(sh), nm = named(k);
        return `<div class="ov-tr${shs.includes(k) ? ' cur' : ''}"><b>${pad(k)}</b><span class="who">${who.map(n => `<i class="${nm.has(n) ? '' : 'inf'}" style="${avatarStyle(n)}"></i>`).join('')}</span><span>${esc(camLabel(sh.camera))}${pos ? `, <span class="${prov(k, pos)}">${esc(pos.replace(/Character (\d+)/g, 'C$1'))}</span>` : ''}</span></div>`;
      }).join('')}
      <div class="ov-key"><span>Preserved</span><span class="inf">Inferred</span><span class="cmp">Completed</span></div>`;
    $$('.ov-av', fig).forEach(a => {
      a.addEventListener('pointermove', e => showTip(charTip(s, a.dataset.e), e));
      a.addEventListener('pointerleave', hideTip);
    });

    /* 2. Top Film History Strip (clickable W1..W5!) */
    $('#ovHist', fig).innerHTML = `<div class="ov-strip">${s.windows.map((w, wi) => {
      const cls = wi === WI ? ' now' : wi > WI ? ' future' : '';
      const lbl = wi === WI ? `W${wi + 1} (Current)` : wi > WI ? `W${wi + 1} <em>Planned</em>` : `Window ${wi + 1}`;
      return `<button type="button" class="hw${cls}" data-w="${wi}" style="--n:${w.shots.length}" title="Switch architecture view to Window ${wi + 1}">
        <span class="hw-head"><span>${lbl}</span></span>
        <span class="hw-cards">${w.shots.map(k => {
          const rbs = refByShot.get(k);
          return `<i class="hs${rbs ? ' rec' : ''}" data-k="${k}" style="background-image:url(${frame(HERO, k)})"><em>S${k}</em>${rbs ? `<b class="rb">${rbs[0]}</b>` : ''}${wi === WI ? '<span class="nz"></span>' : ''}</i>`;
        }).join('')}</span>
      </button>`;
    }).join('')}</div>`;
    $$('.hw', fig).forEach(b => b.addEventListener('click', () => { stopTour(); selectWindow(+b.dataset.w); }));

    /* 3. Compiled Prompts Stack */
    $('#ovPrompts', fig).innerHTML = `${W.refs.length ? `<span><b>c<sup>ref</sup></b>${W.refs.length} identity line${W.refs.length > 1 ? 's' : ''}</span>` : `<span><b>c<sup>ref</sup></b>none (cold start)</span>`}
      ${shs.map(k => `<span><b>c<sub>${k}</sub></b>Shot ${k} prompt</span>`).join('')}
      <small>[Entities] [Attire] [Camera] [Opening] [Action]</small>`;

    /* 4. Reader shot pills */
    const rdr = $('#ovReader', fig);
    rdr.innerHTML = shs.map(k => `<button type="button" data-k="${k}" class="${k === reader ? 'on' : ''}" aria-pressed="${k === reader}">Shot ${k}</button>`).join('');
    $$('button', rdr).forEach(b => b.addEventListener('click', () => setReader(+b.dataset.k)));

    /* 5. Decoded outputs */
    const outs = $('#ovOuts', fig);
    outs.style.setProperty('--sn', String(Math.max(2, shs.length)));
    outs.innerHTML = shs.map(k => `<div class="ov-out"><i style="background-image:url(${frame(HERO, k)})"><span class="nz"></span></i><b>Shot ${k}</b></div>`).join('');

    /* 6. Attention Mask SVG */
    buildMask();
    /* 7. Contiguous Latent + RoPE SVG */
    tokHost.replaceChildren();
    buildTokens();
    applyStage();
    capEl.innerHTML = stageCopy();
    requestAnimationFrame(drawLinks);
  }

  function buildMask() {
    const W = s.windows[WI], host = $('#ovMask', fig);
    host.replaceChildren();
    const blocks = [];
    W.refs.forEach((r, i) => blocks.push({ t: 'ref', i, u: 1, label: `R${i + 1} (Shot ${r.shot})` }));
    W.shots.forEach((k, j) => { blocks.push({ t: 'first', j, k, u: 1 }); blocks.push({ t: 'rest', j, k, u: 3 }); });
    let acc = 0; blocks.forEach(b => { b.o = acc; acc += b.u; });
    const c = Math.min(13, Math.floor(156 / Math.max(8, acc))), L = 40, Tp = 22, size = acc * c;
    const svg = sv('svg', { width: L + size + 4, height: Tp + size + 4, viewBox: `0 0 ${L + size + 4} ${Tp + size + 4}`, role: 'img', 'aria-label': `Attention mask for Window ${WI + 1}` }, host);
    const allow = (q, k) => {
      if (q.t === 'ref') return k.t === 'ref' && k.i === q.i ? 'ref' : null;
      if (k.t === 'ref') return (W.routing[q.j] || []).includes(k.i) ? 'ref' : null;
      if (k.j === q.j) return 'self';
      return k.t === 'first' ? 'first' : null;
    };
    const FILL = { self: 'var(--grey-a)', first: 'var(--holo)', ref: 'var(--ours)' };
    blocks.forEach(q => blocks.forEach(k => {
      const a = allow(q, k);
      const rect = sv('rect', {
        x: L + k.o * c + 0.6, y: Tp + q.o * c + 0.6, width: k.u * c - 1.2, height: q.u * c - 1.2, rx: 1.8,
        fill: a ? FILL[a] : 'var(--panel)', 'fill-opacity': a ? (q.t === 'ref' ? 0.55 : 0.88) : 1,
        class: 'mc', 'data-q': q.k || '', 'data-ref': k.t === 'ref' ? W.refs[k.i].shot : '',
      }, svg);
      rect.addEventListener('pointerenter', () => {
        rect.classList.add('hl');
        if (q.k && q.k !== reader) setReader(q.k);
      });
      rect.addEventListener('pointerleave', () => rect.classList.remove('hl'));
    }));
    const mLab = (x, y, base, supSub, isSup, col, an) => {
      const t = sv('text', { x, y, 'text-anchor': an, style: `font:600 11px var(--mono);fill:${col}` }, svg);
      t.textContent = base;
      const sp = sv('tspan', { dy: isSup ? -3.5 : 2.5, style: 'font-size:8.5px' }, t);
      sp.textContent = supSub;
    };
    if (W.refs.length) {
      mLab(L - 5, Tp + W.refs.length * c / 2 + 4, 'Q', 'ref', true, 'var(--ours)', 'end');
      mLab(L + W.refs.length * c / 2, Tp - 5, 'K', 'ref', true, 'var(--ours)', 'middle');
    }
    W.shots.forEach((k, j) => {
      const o = W.refs.length + j * 4;
      mLab(L - 5, Tp + (o + 2) * c + 4, 'Q', String(k), false, k === reader ? 'var(--ours)' : 'var(--paper)', 'end');
      mLab(L + (o + 2) * c, Tp - 5, 'K', String(k), false, 'var(--paper)', 'middle');
    });
  }

  function buildTokens() {
    const W = s.windows[WI], shs = W.shots;
    const Wd = 420, H = 64, padL = 8;
    const maxRefSlots = Math.max(2, W.refs.length);
    const totalSlots = maxRefSlots + 0.7 + shs.length * 5 + (shs.length - 1) * 0.35;
    const sw = (Wd - padL * 2) / totalSlots, sz = Math.min(18, Math.max(11, sw - 4));
    const X = slot => padL + slot * sw + (sw - sz) / 2;
    const yT = 18, yA = 47;
    const svg = sv('svg', { width: Wd, height: H, viewBox: `0 0 ${Wd} ${H}`, role: 'img', 'aria-label': 'Contiguous latent token sequence with RoPE time offsets' }, tokHost);
    const pat = sv('pattern', { id: 'ovNoisePat', width: 4, height: 4, patternUnits: 'userSpaceOnUse' }, sv('defs', {}, svg));
    [[0, 0], [2, 1], [1, 3], [3, 2]].forEach(([x, y]) => sv('rect', { x, y, width: 1.2, height: 1.2, fill: '#fff', 'fill-opacity': 0.85 }, pat));

    sv('path', { d: `M${padL} ${yA} H${Wd - padL}`, stroke: 'var(--rule)' }, svg);
    const refLabels = [];
    for (let m = 0; m < maxRefSlots; m++) {
      const cx = X(m) + sz / 2;
      sv('rect', { x: X(m), y: yT, width: sz, height: sz, rx: 3, fill: 'none', stroke: 'var(--rule)', 'stroke-dasharray': '2 2' }, svg);
      refLabels.push(T(svg, cx, yA + 13, `−${(maxRefSlots - m) * 5}`, 'svg-text', 'font:500 9.5px var(--mono);fill:var(--cut)', 'middle'));
    }
    const refGroups = W.refs.map((r, i) => {
      const g = sv('g', { class: 'tk-r' }, svg);
      sv('rect', { x: 0, y: yT, width: sz, height: sz, rx: 3, fill: 'var(--cut-soft)', stroke: 'var(--cut)', 'stroke-width': 1.5 }, g);
      T(g, sz / 2, yT - 4, `R${i + 1}`, 'svg-text', 'font:600 9.5px var(--mono);fill:var(--cut)', 'middle');
      return g;
    });

    const shotGroups = shs.map((k, j) => {
      const g = sv('g', { class: 'tk-s', 'data-k': k, style: 'cursor:pointer' }, svg);
      const baseSlot = maxRefSlots + 0.7 + j * 5.35;
      T(g, X(baseSlot + 2) + sz / 2, yT - 4, `Shot ${k}`, 'svg-text', 'font:600 10px var(--mono);fill:var(--paper)', 'middle');
      for (let q = 0; q < 5; q++) {
        const x = X(baseSlot + q);
        sv('rect', { x, y: yT, width: sz, height: sz, rx: 3, fill: q === 0 ? 'var(--holo)' : 'var(--ours-soft)', stroke: q === 0 ? 'var(--holo)' : 'var(--ours)', 'stroke-width': 1.2 }, g);
        sv('rect', { x, y: yT, width: sz, height: sz, rx: 3, fill: 'url(#ovNoisePat)', class: 'tk-n' }, g);
        if (q === 0 || q === 4) T(svg, x + sz / 2, yA + 13, String(j * 5 + q), 'svg-text', 'font:500 9.5px var(--mono)', 'middle');
      }
      g.addEventListener('click', () => setReader(k));
      return g;
    });

    placeTok = () => {
      const j = Math.max(0, shs.indexOf(reader)), reads = W.routing[j] || [], p0 = j * 5;
      refLabels.forEach((lbl, m) => {
        const idx = m - (maxRefSlots - reads.length);
        if (idx >= 0 && reads.length) {
          const offset = p0 - 5 * (reads.length - idx);
          lbl.textContent = offset < 0 ? `−${Math.abs(offset)}` : String(offset);
          lbl.style.opacity = 1;
        } else {
          lbl.textContent = '';
          lbl.style.opacity = 0;
        }
      });
      refGroups.forEach((g, i) => {
        const posInRead = reads.indexOf(i);
        const slot = posInRead < 0 ? 0 : (maxRefSlots - reads.length + posInRead);
        g.style.transform = `translate(${X(slot).toFixed(1)}px,0)`;
        g.style.opacity = posInRead < 0 ? 0 : 1;
      });
      shotGroups.forEach(g => { g.style.opacity = +g.dataset.k === reader ? 1 : 0.42; });
      $$('#ovMask rect.mc', fig).forEach(r => r.classList.toggle('hl', +r.dataset.q === reader));
    };
    placeTok();
  }

  function setReader(k) {
    reader = k;
    setTab($('#ovReader', fig), $(`[data-k="${k}"]`, $('#ovReader', fig)));
    placeTok();
  }

  function selectWindow(wi) {
    WI = wi;
    setTab(winTabs, $(`[data-w="${wi}"]`, winTabs));
    reader = s.windows[WI].shots[0];
    buildWindowDOM();
  }

  const rel = (el, r0) => {
    const r = el.getBoundingClientRect();
    return { l: r.left - r0.left, r: r.right - r0.left, t: r.top - r0.top, b: r.bottom - r0.top, cx: r.left - r0.left + r.width / 2, cy: r.top - r0.top + r.height / 2 };
  };
  let wireSeq = 0;
  function addPulse(grp, pathId, delay = 0) {
    if (RM) return;
    const c = sv('circle', { r: 3.3, class: 'pulse', opacity: 0 }, grp);
    const begin = `${delay.toFixed(2)}s`;
    sv('animate', { attributeName: 'opacity', values: '0;1;1;0', keyTimes: '0;0.12;0.86;1', dur: '1.8s', begin, repeatCount: 'indefinite' }, c);
    const am = sv('animateMotion', { dur: '1.8s', begin, repeatCount: 'indefinite', calcMode: 'spline', keyTimes: '0;1', keySplines: '0.3 0.1 0.25 1' }, c);
    sv('mpath', { href: `#${pathId}` }, am);
  }
  function drawLinks() {
    links.innerHTML = '';
    if (getComputedStyle(links).display === 'none') return;
    const r0 = map.getBoundingClientRect();
    if (!r0.width) return;
    links.setAttribute('viewBox', `0 0 ${r0.width.toFixed(1)} ${r0.height.toFixed(1)}`);
    const g = (st, cls = '') => sv('g', { 'data-st': st, class: cls }, links);
    const head = (e, x, y, dir) => {
      const d = dir === 'down' ? `M${x - 4} ${y - 6} L${x} ${y} L${x + 4} ${y - 6}` : `M${x - 6} ${y - 4} L${x} ${y} L${x - 6} ${y + 4}`;
      sv('path', { d, class: 'hd' }, e);
    };
    const curve = (a, b, st, cls = '') => {
      const x1 = a.r + 2, y1 = a.cy, x2 = b.l - 3, y2 = b.cy, mx = (x1 + x2) / 2;
      const e = g(st, cls), pid = `ovW_${++wireSeq}`;
      sv('path', { id: pid, d: `M${x1.toFixed(1)} ${y1.toFixed(1)} C${mx.toFixed(1)} ${y1.toFixed(1)} ${mx.toFixed(1)} ${y2.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`, class: 'wire' }, e);
      head(e, x2, y2, 'right');
      addPulse(e, pid, 0.1);
    };
    const vert = (x1, y1, x2, y2, st) => {
      const e = g(st), pid = `ovW_${++wireSeq}`, my = (y1 + y2) / 2;
      sv('path', { id: pid, d: `M${x1.toFixed(1)} ${y1.toFixed(1)} C${x1.toFixed(1)} ${my.toFixed(1)} ${x2.toFixed(1)} ${my.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`, class: 'wire' }, e);
      head(e, x2, y2, 'down');
      addPulse(e, pid, 0.15);
    };

    const board = rel($('#ovBoard', fig), r0), comp = rel($('#ovComp', fig), r0), prompts = rel($('#ovPrompts', fig), r0), t5 = rel($('#ovT5', fig), r0);
    const recall = rel($('#ovRecall', fig), r0), enc = rel($('#ovEnc', fig), r0), tok = rel($('#ovTokBox', fig), r0), dit = rel($('#ovDit', fig), r0), cross = rel($('#ovCross', fig), r0), dec = rel($('#ovDec', fig), r0);

    curve(board, recall, '4', 'rose');
    curve(comp, prompts, '3', 'rose');
    curve(t5, cross, '6');
    curve(enc, { l: tok.l - 3, cy: tok.cy }, '5');

    const recs = $$('.hs.rec', fig).map(el => rel(el, r0)), hist = rel($('#ovHist', fig), r0);
    if (recs.length) {
      const busY = hist.b + 7, bus = g('4', 'rose'), xs = [recall.cx, ...recs.map(a => a.cx)];
      recs.forEach((a, idx) => {
        const pid = `ovW_${++wireSeq}`;
        sv('path', { id: pid, d: `M${a.cx.toFixed(1)} ${(a.b + 2).toFixed(1)} V${busY.toFixed(1)} H${recall.cx.toFixed(1)} V${(recall.t - 3).toFixed(1)}`, class: 'wire' }, bus);
        if (idx === 0) addPulse(bus, pid, 0.05);
      });
      sv('path', { d: `M${Math.min(...xs).toFixed(1)} ${busY.toFixed(1)} H${Math.max(...xs).toFixed(1)}`, class: 'wire' }, bus);
      head(bus, recall.cx, recall.t - 3, 'down');
    }
    vert(dit.cx, tok.b + 2, dit.cx, dit.t - 3, '5,6');
    vert(dit.cx, dit.b + 2, dit.cx, dec.t - 3, '7');
    applyStage();
  }

  function applyStage() {
    $$('[data-st]', fig).forEach(el => {
      const match = stage > 0 && el.dataset.st.split(',').includes(String(stage));
      el.classList.toggle('on', match);
      $$('circle.pulse', el).forEach(c => { c.style.display = (stage === 0 || match) ? '' : 'none'; });
    });
  }
  function setStage(n) {
    stage = n;
    fig.dataset.stage = n;
    applyStage();
    setTab(stageTabs, n ? tabBtns[n - 1] : null);
    capEl.innerHTML = stageCopy();
    fig.classList.toggle('noisy', n >= 1 && n <= 5);
    fig.classList.toggle('denoising', n === 6);
    if (!RM) { capEl.classList.remove('swap'); void capEl.offsetWidth; capEl.classList.add('swap'); }
  }
  const setPlay = () => {
    playBtn.innerHTML = touring
      ? '<svg viewBox="0 0 14 14"><path d="M4 2.5v9M10 2.5v9" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>Pause'
      : '<svg viewBox="0 0 14 14"><path d="M3.5 2v10l8.5-5z" fill="currentColor"/></svg>Play pass';
    playBtn.setAttribute('aria-label', touring ? 'Pause pass' : 'Play pass');
  };
  function stopTour() { timers.forEach(clearTimeout); timers = []; touring = false; setPlay(); }
  const DWELL = 2800;
  function tour() {
    stopTour();
    touring = true; setPlay();
    const shs = s.windows[WI].shots;
    for (let n = 1; n <= 7; n++) timers.push(setTimeout(() => setStage(n), (n - 1) * DWELL));
    if (shs.length > 1) timers.push(setTimeout(() => setReader(shs[1]), 4 * DWELL + 1300));
    timers.push(setTimeout(() => { touring = false; setPlay(); setReader(shs[0]); setStage(0); }, 7 * DWELL + 300));
  }

  $$('button', winTabs).forEach(b => b.addEventListener('click', () => { stopTour(); selectWindow(+b.dataset.w); }));
  tabBtns.forEach(b => b.addEventListener('click', () => { stopTour(); const n = +b.dataset.n; setStage(n === stage ? 0 : n); }));
  navBtns.forEach(b => b.addEventListener('click', () => { stopTour(); const d = +b.dataset.d; setStage(stage === 0 ? (d > 0 ? 1 : 7) : Math.max(1, Math.min(7, stage + d))); }));
  playBtn.addEventListener('click', () => { if (touring) stopTour(); else tour(); });

  new ResizeObserver(() => requestAnimationFrame(drawLinks)).observe(map);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => requestAnimationFrame(drawLinks));

  setPlay();
  buildWindowDOM();
  setStage(0);
  onReveal(fig, () => { if (!RM) timers.push(setTimeout(tour, 450)); });
});

/* =============== 03 Benchmark: Visual Pipeline & Judge Replay (Figure 3 c, d) =============== */
run('benchFig', function benchFig() {
  const fig = $('#benchFig');
  if (!fig) return;
  const AD = [
    ['Wan2.1 / Wan2.2', 'one prompt per shot'],
    ['MultiShotMaster', 'story + shot captions'],
    ['HoloCine', 'global + shot captions'],
    ['StoryMem', 'prompt + keyframe memory'],
    ['StoryDiffusion', 'keyframes → Wan2.2-I2V'],
    ['IC-LoRA', '3-panel grid → Wan2.2-I2V'],
  ];
  const Q = [
    ['Character consistency', 'Is C1 at S3 the same person as at S1?', 0],
    ['Object consistency', 'Is O1 at S2 the same object as at S1?', null],
    ['Scene consistency', 'Are S10 and S1 in the same place (L1)?', null],
    ['State establishment', 'Is O1 folded by the end of S1?', null],
    ['State persistence', 'Is O1 still folded in S2?', null],
    ['Cut execution', 'S1 → S2: cut to a new camera setup?', 1],
  ];
  const FR = [
    ['s01_t00', 'S1 0:00'], ['s01_t02', 'S1 0:02'], ['s01_t04', 'S1 0:04'], ['s02_t06', 'S2 0:06'], ['s03_t09', 'S3 0:09'], ['s03_t10', 'S3 0:10'],
    ['s03_t11', 'S3 0:11'], ['s04_t13', 'S4 0:13'], ['s08_t45', 'S8 0:45'], ['s08_t46', 'S8 0:46'], ['s08_t47', 'S8 0:47'], ['s09_t50', 'S9 0:50'],
  ];
  const EX = [
    {
      q: 'Character check: Is C1 at 0:09–0:12 (Shot 3) the same person as at 0:00–0:04 (Shot 1)?',
      hi: [[0, 1, 2], [4, 5, 6]], roles: ['Anchor S1', 'Query S3'],
      steps: [
        ['Cut Gate', 'New camera setup confirmed (correlation −0.08)', 'ok'],
        ['Gemini 3.8 Flash', 'No: face differs, though hair, build, and suit match', 'bad'],
        ['Scored Verdict', 'Gate passes, so the failed character check counts (0)', 'bad'],
      ],
    },
    {
      q: 'Cut execution check: Does S1 → S2 transition to the planned new camera setup?',
      hi: [[2], [3]], roles: ['S1 end', 'S2 start'],
      steps: [
        ['TransNetV2', 'Visual cut boundary detected within 0.5 s', 'ok'],
        ['Frame Correlation', 'Boundary correlation 0.30 (passed to VLM)', ''],
        ['Gemini 3.8 Flash', 'New camera setup verified; cross-cut checks unlocked', 'ok'],
      ],
    },
  ];

  fig.innerHTML = `<div class="fig-head">
      <div><h3>How CaT-MBench builds questions and judges videos</h3><p class="fig-sub">Fair native input adapters paired with entity-indexed binary VLM checks</p></div>
      <button type="button" class="tbtn" id="benchReplay"></button>
    </div>
    <div class="bf-pipe">
      <div class="bf-box">
        <div class="bf-step-hd"><span class="bf-badge"><i>1</i>Held-Out Stories &amp; Native Inputs</span><span class="bf-model">Film-disjoint</span></div>
        <div class="bf-big"><b>122</b><span>stories (6–12 shots, up to 60 s): 30 dev + 92 test</span></div>
        <div class="mbar"><i style="flex:30 1 0;background:var(--grey-c)"></i><i style="flex:92 1 0;background:var(--ours)"></i></div>
        <div class="bf-req"><b>C1</b> dark-haired man, grey suit. <b>O1</b> piece of paper. <b>L1</b> wooden kitchen.<br><b>S01</b> Medium shot. C1 writes on the paper, folds it, and hands it to C2.</div>
        <div class="bf-adapters">${AD.map(([m, f]) => `<div class="bf-ad"><b>${m}</b><span>${f}</span></div>`).join('')}</div>
      </div>
      <div class="bf-box">
        <div class="bf-step-hd"><span class="bf-badge"><i>2</i>Entity Question Bank</span><span class="bf-model">GPT-6.1-sol</span></div>
        <div class="bf-qgrid">${Q.map(([t, q, ex]) => ex != null
          ? `<button type="button" class="bf-qitem click" data-ex="${ex}"><div><b>${t}</b><span>${esc(q)}</span></div><em>Replay ↓</em></button>`
          : `<div class="bf-qitem"><div><b>${t}</b><span>${esc(q)}</span></div></div>`).join('')}</div>
        <div class="bf-nums">
          <div><b>1,361</b><span>binary questions (349 dev, 1,012 test)</span></div>
          <div><b>995</b><span>planned cuts (674 of 740 test cuts change setup)</span></div>
        </div>
      </div>
    </div>
    <div class="bf-join"><span>↓ Sampled at 2 fps into Cut-Gated VLM Judge (TransNetV2 + Gemini 3.8 Flash) ↓</span></div>
    <div class="bf-judge">
      <div class="bf-jhead">
        <span class="bf-badge"><i>3</i>Worked Judge Trace (Wan2.2 per-shot video)</span>
        <div class="tabs" id="jEx" role="group" aria-label="Worked judge example">
          <button type="button" data-e="0" class="on" aria-pressed="true">Character Check</button>
          <button type="button" data-e="1" aria-pressed="false">Cut Execution Check</button>
        </div>
      </div>
      <div class="bf-rule"><span><b>Visible evidence only</b></span><span>Yes / No + timestamp + reason</span><span>Unclear defaults to No</span></div>
      <div class="jstrip" id="jStrip">${FR.map(([f, l], i) => `<div class="jf" data-i="${i}" style="background-image:url(${MEDIA}f/judge/${f}.jpg)"><span>${l}</span><i class="role"></i></div>`).join('')}</div>
      <div class="jflow" id="jFlow" aria-live="polite"></div>
    </div>`;

  const ex = $('#jEx', fig), flow = $('#jFlow', fig), strip = $('#jStrip', fig), qClicks = $$('.bf-qitem.click', fig), replay = $('#benchReplay', fig);
  let stepT = [], tourT = [], touring = false;
  function setE(i) {
    stepT.forEach(clearTimeout); stepT = [];
    setTab(ex, $$('button', ex)[i]);
    qClicks.forEach(b => b.classList.toggle('on', +b.dataset.ex === i));
    const E = EX[i];
    $$('.jf', strip).forEach(f => {
      const k = +f.dataset.i, isA = E.hi[0].includes(k), isB = E.hi[1].includes(k);
      f.classList.toggle('a', isA);
      f.classList.toggle('b', isB);
      $('i.role', f).textContent = isA ? E.roles[0] : isB ? E.roles[1] : '';
    });
    flow.innerHTML = `<p class="jq">${esc(E.q)}</p><div class="jsteps">${E.steps.map(([k, v, c]) => `<div class="jstep ${c}"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('')}</div>`;
    $$('.jstep', flow).forEach((st, n) => { if (RM) st.classList.add('on'); else stepT.push(setTimeout(() => st.classList.add('on'), 180 + n * 340)); });
  }
  const setReplay = () => { replay.innerHTML = touring ? 'Skip' : '<svg viewBox="0 0 14 14"><path d="M2.5 7a4.5 4.5 0 1 0 1.6-3.45M2.5 1.8v2.6h2.6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>Replay trace'; };
  function finish() { tourT.forEach(clearTimeout); tourT = []; touring = false; setReplay(); }
  function tour() {
    finish();
    touring = true; setReplay();
    setE(0);
    tourT.push(setTimeout(() => setE(1), 4200));
    tourT.push(setTimeout(() => { setE(0); finish(); }, 8200));
  }
  $$('button', ex).forEach(b => b.addEventListener('click', () => { finish(); setE(+b.dataset.e); }));
  qClicks.forEach(b => b.addEventListener('click', () => { finish(); setE(+b.dataset.ex); }));
  replay.addEventListener('click', () => { if (touring) finish(); else tour(); });
  setE(0); finish();
  onReveal(fig, () => { if (!RM) tour(); });
});

/* =============== 06 Dataset: Curation Funnel + 4-Layer Entity Grounding (Figure 3 a, b) =============== */
run('dataFig', function dataFig() {
  const fig = $('#dataFig'), pipe = $('#pipe'), host = $('#blk');
  if (!fig || !pipe || !host) return;
  const SRC = [['CinePile', 65.74, 'var(--ours)'], ['CineDance', 24.69, 'var(--grey-a)'], ['MovieChat', 9.57, 'var(--grey-c)']];
  const HB = [15459, 12498, 8388, 5959, 4301, 4242, 5642, 4826, 4864, 4136, 3100, 2046], mx = Math.max(...HB);
  const step = (n, chip, title, big, desc, mini) => `<li class="pp lit" data-i="${n - 1}"><div class="pp-top"><span class="pp-n">${n}</span><span class="pp-chip">${chip}</span></div><b class="pp-t">${title}</b>${big ? `<strong>${big}</strong>` : ''}<span class="pp-d">${desc}</span><div class="mini">${mini}</div></li>`;
  pipe.innerHTML = [
    step(1, 'Corpus', 'Cinematic videos', '19,535', 'raw film & scene sources', `<div class="mbar">${SRC.map(([nm, v, c]) => `<i style="flex:${v} 1 0;background:${c}" title="${nm} ${v.toFixed(1)}%"></i>`).join('')}</div><div class="mkey">${SRC.map(([nm, v, c]) => `<span><i style="background:${c}"></i>${nm} ${Math.round(v)}%</span>`).join('')}</div>`),
    step(2, 'TransNetV2', 'Shot detection', '358,302', 'clean shots after boundary & quality checks (332.8 h)', `<div class="pp-cuts">${[1, 2, 3].map(i => `<i style="background-image:url(${MEDIA}f/ground/s${i}-rgb.jpg)"></i>`).join('')}<svg style="left:calc(33.333% - 2px)" viewBox="0 0 24 24"><use href="#i-scissor"/></svg><svg style="left:calc(66.667% + 2px)" viewBox="0 0 24 24"><use href="#i-scissor"/></svg></div>`),
    step(3, 'Blocks', 'Scene grouping', '75,461', 'contiguous multi-shot blocks of up to 12 shots', `<div class="mhist">${HB.map((v, i) => `<i class="${i === 0 ? 'mode' : ''}" data-i="${i}" style="height:${(v / mx * 100).toFixed(1)}%"></i>`).join('')}<span class="med" style="left:${(3.5 / 12 * 100).toFixed(2)}%"><span>med 4</span></span></div><div class="mhist-x"><span>1 shot</span><span>12 shots</span></div>`),
    step(4, 'Quality', 'Shot-first filtering', '+28.1%', 'drops single bad shots rather than discarding whole windows', `<div class="myield"><div><p><span>window-first</span><em>227k</em></p><i style="width:${(226648 / 290267 * 100).toFixed(1)}%"></i></div><div><p><span>shot-first</span><em>290k</em></p><i style="width:100%"></i></div></div>`),
    step(5, 'Split', 'Film-disjoint split', '66,573<small>train</small>', 'blocks + 916 held-out; zero film overlap with CaT-MBench', `<div class="mbar split"><i style="flex:66573 1 0;background:var(--grey-b)"></i><i style="flex:916 1 0;min-width:5px"></i></div><div class="mkey"><span><i style="background:var(--grey-b)"></i>train 66,573</span><span><i style="background:var(--ours)"></i>held-out 916</span></div>`),
  ].join('');
  $$('.mhist i', pipe).forEach(b => {
    const i = +b.dataset.i;
    b.addEventListener('pointermove', e => showTip(`<b>${i + 1} shot${i ? 's' : ''} per block</b><table><tr><td>Blocks</td><td>${HB[i].toLocaleString('en-US')}</td></tr><tr><td>Share</td><td>${(HB[i] / 75461 * 100).toFixed(1)}%</td></tr></table>`, e));
    b.addEventListener('pointerleave', hideTip);
  });

  const TRACK = [
    { shot: 1, entities: ['C1', 'O1'], frames: [[[447, 37, 330, 443], [0, 413, 456, 67]], [[445, 34, 327, 446], [0, 407, 453, 73]], [[437, 38, 325, 442], [0, 404, 444, 69]], [[460, 43, 319, 437], [0, 92, 652, 375]], [[459, 39, 327, 441], [0, 342, 468, 119]], [[440, 42, 333, 438], [0, 339, 447, 117]], [[394, 24, 351, 456], [0, 361, 485, 93]], [[364, 0, 379, 480], [0, 352, 470, 104]], [[379, 0, 382, 480], [0, 369, 471, 93]], [[401, 0, 391, 480], [0, 375, 488, 90]], [[409, 1, 406, 479], [7, 395, 492, 85]], [[385, 13, 430, 467], [47, 400, 690, 80]], [[342, 8, 464, 472], [88, 443, 386, 37]], [[372, 6, 448, 474], [126, 456, 349, 24]]] },
    { shot: 2, entities: ['C2', 'O2', 'O3'], frames: [[[145, 71, 303, 409], [250, 0, 425, 480], [408, 80, 133, 393]], [[123, 61, 298, 419], [249, 0, 487, 480], [415, 74, 142, 399]], [[104, 61, 287, 419], [249, 0, 510, 480], [428, 71, 150, 409]], [[96, 54, 305, 426], [242, 0, 517, 480], [424, 66, 171, 414]], [[109, 50, 325, 430], [235, 0, 535, 480], [439, 57, 178, 423]], [[140, 26, 367, 454], [238, 0, 562, 480], [424, 50, 216, 430]], [[162, 5, 399, 475], [206, 0, 626, 480], [426, 65, 226, 408]], [[155, 2, 420, 478], [169, 0, 663, 480], [420, 36, 233, 290]], [[118, 0, 432, 480], [135, 0, 697, 480], [397, 33, 255, 447]], [[66, 0, 445, 480], [113, 0, 719, 480], [386, 29, 267, 451]], [[29, 0, 462, 480], [645, 0, 187, 480], [393, 29, 248, 451]], [[0, 0, 468, 480], [679, 0, 153, 480], [403, 27, 282, 453]], [[0, 0, 444, 480], [466, 0, 366, 480], [407, 28, 315, 452]], [[0, 0, 428, 480], [339, 0, 493, 480], [402, 31, 348, 449]], [[0, 0, 435, 480], [73, 0, 759, 480], [398, 33, 388, 447]], [[0, 0, 454, 480], [73, 0, 759, 480], [399, 33, 416, 447]], [[0, 0, 471, 480], null, [395, 32, 413, 448]], [[0, 0, 470, 480], null, [391, 36, 410, 444]], [[0, 0, 460, 480], null, [388, 42, 413, 438]], [[0, 0, 471, 480], null, [390, 46, 411, 434]]] },
    { shot: 3, entities: ['C1'], frames: [[[315, 36, 517, 444]], [[361, 21, 471, 459]], [[368, 26, 464, 454]], [[333, 54, 499, 426]], [[264, 45, 532, 435]], [[251, 35, 540, 445]], [[284, 38, 540, 442]], [[342, 34, 490, 446]], [[397, 12, 435, 468]], [[405, 13, 427, 467]], [[363, 29, 469, 451]], [[296, 2, 536, 478]], [[283, 8, 549, 472]], [[315, 17, 517, 463]], [[389, 10, 443, 470]], [[430, 3, 402, 477]], [[424, 12, 408, 468]], [[380, 30, 452, 450]], [[310, 21, 522, 459]], [[284, 15, 548, 465]]] },
  ];
  const COL = { C1: '#29B6AA', O1: '#5B8FF9', C2: '#FCB84C', O2: '#B084D8', O3: '#EE7D92' };
  const REG = [
    ['C1', 'older man, white hair, black suit & red tie'],
    ['C2', 'dark-haired man, dark grey polo shirt'],
    ['O1', 'pile of newspapers and magazines'],
    ['O2', 'metal mesh display cage'],
    ['O3', 'black armored suit inside cage'],
  ];
  const SHOT = [
    'Medium shot, static. C1 stands near a pile of newspapers and looks down, then looks up.',
    'Medium shot, tracking. C2 walks towards a metal mesh cage containing a black armored suit.',
    'Medium shot, static. C1 speaks while looking at C2.',
  ];
  const LAYERS = [
    ['rgb', 'RGB Frame', 'Three consecutive shots of one scene block at the source film’s native frame rate.'],
    ['boxes', 'Boxes (Qwen3.8-27B)', 'Qwen3.8-27B open-vocabulary detection & tracking; C1 retains the same registry ID across Shots 1 and 3.'],
    ['masks', 'Masks (SAM 3.1)', 'SAM 3.1 dense instance segmentation propagated across every shot.'],
    ['depth', 'Depth (VGGT-Ω)', 'VGGT-Ω relative depth maps (lighter is farther), stored every fourth frame.'],
  ];
  host.innerHTML = `<div class="blk-head">
      <div class="blk-head-item"><span class="pp-n">6</span><b>Block Storyboard Annotation</b><span class="pp-chip">Qwen3.7-Flash (1 pass)</span></div>
      <div class="blk-head-item"><span class="pp-n">7</span><b>Pixel Grounding</b><div class="tabs" id="gSeg" role="group" aria-label="Grounding layer">${LAYERS.map(([k, n]) => `<button type="button" data-l="${k}"${k === 'boxes' ? ' class="on" aria-pressed="true"' : ' aria-pressed="false"'}>${n}</button>`).join('')}</div></div>
    </div>
    <div class="blk-reg">${REG.map(([id, d]) => `<button type="button" data-e="${id}"><i style="background:${COL[id]}"></i><b>${id}</b><span>${esc(d)}</span></button>`).join('')}</div>
    <div class="blk-shots">${TRACK.map((t, i) => `<div class="blk-shot">
      <div class="gfr"><div class="gvid"><video muted loop playsinline preload="none" poster="${MEDIA}v/ground/s${i + 1}-poster.jpg" src="${MEDIA}v/ground/s${i + 1}.mp4" aria-label="Shot ${i + 1} of the annotated block"></video></div>
        <div class="gbx">${t.entities.map(n => `<div class="gb" data-e="${n}" style="border-color:${COL[n]}"><span style="background:${COL[n]}">${n}</span></div>`).join('')}</div>
        <span class="gs">Shot ${i + 1}</span></div>
      <p><b>Shot ${i + 1}</b>${esc(SHOT[i])}</p>${i === 0 ? '<p class="chg"><b>State change (C1):</b>looking down → looking up and forward</p>' : ''}
    </div>`).join('')}</div>
    <p class="cap" id="gCap"></p>`;

  const clips = $$('.blk-shot', host).map((el, i) => ({ v: $('video', el), t: TRACK[i], gs: $$('.gb', el), j: -1 }));
  function place(c) {
    const j = Math.min(c.t.frames.length - 1, Math.floor((c.v.currentTime || 0) * 16 / 4));
    if (j === c.j) return;
    c.j = j;
    c.gs.forEach((g, e) => {
      const b = c.t.frames[j][e];
      g.hidden = !b;
      if (!b) return;
      const [x, y, w, hh] = b;
      g.style.left = (x / 832 * 100).toFixed(2) + '%';
      g.style.top = (y / 480 * 100).toFixed(2) + '%';
      g.style.width = (w / 832 * 100).toFixed(2) + '%';
      g.style.height = (hh / 480 * 100).toFixed(2) + '%';
    });
  }
  let raf = 0;
  const tick = () => { clips.forEach(place); raf = requestAnimationFrame(tick); };
  clips.forEach(place);
  const seg = $('#gSeg', host);
  function setL(k) {
    const L = LAYERS.find(l => l[0] === k);
    host.dataset.layer = k;
    setTab(seg, $(`[data-l="${k}"]`, seg));
    $('#gCap', host).innerHTML = `<b>${L[1]}:</b> ${L[2]}`;
  }
  $$('.blk-reg button', host).forEach(p => {
    const focusEnt = () => {
      if (host.dataset.layer !== 'boxes') setL('boxes');
      host.classList.add('focus');
      $$('.blk-reg button', host).forEach(x => x.classList.toggle('hl', x === p));
      $$('.gb', host).forEach(g => g.classList.toggle('hl', g.dataset.e === p.dataset.e));
    };
    const clearEnt = () => {
      host.classList.remove('focus');
      $$('.blk-reg button, .gb', host).forEach(x => x.classList.remove('hl'));
    };
    p.addEventListener('pointerenter', focusEnt);
    p.addEventListener('pointerleave', clearEnt);
    p.addEventListener('click', () => { if (p.classList.contains('hl')) clearEnt(); else focusEnt(); });
  });
  watch(host, v => {
    clips.forEach(c => { if (v) { c.v.preload = 'auto'; c.v.play().catch(() => {}); } else c.v.pause(); });
    cancelAnimationFrame(raf);
    if (v) tick();
  }, 0.15);

  const replay = $('#dataReplay');
  let timers = [], touring = false;
  const setReplay = () => { replay.innerHTML = touring ? 'Stop tour' : '<svg viewBox="0 0 14 14"><path d="M2.5 7a4.5 4.5 0 1 0 1.6-3.45M2.5 1.8v2.6h2.6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>Cycle grounding layers'; };
  function finish() { timers.forEach(clearTimeout); timers = []; touring = false; setReplay(); }
  function tour() {
    finish();
    touring = true; setReplay();
    if (!RM) grow($$('.mbar i, .mhist i, .myield i', pipe), 'left', 0, 20);
    const seq = ['rgb', 'boxes', 'masks', 'depth', 'boxes'];
    seq.forEach((lk, idx) => timers.push(setTimeout(() => { setL(lk); if (idx === seq.length - 1) finish(); }, idx * 1450)));
  }
  $$('button', seg).forEach(b => b.addEventListener('click', () => { finish(); setL(b.dataset.l); }));
  replay.addEventListener('click', () => { if (touring) { finish(); setL('boxes'); } else tour(); });
  setL('boxes'); finish();
});

/* =============== 06 Dataset: Registered entities per block =============== */
run('entStack', function entStack() {
  const host = $('#entStack');
  if (!host) return;
  const ENT = { People: [8.21, 18.82, 33.10, 39.87], Objects: [5.98, 16.44, 17.26, 60.31], Places: [0.27, 81.90, 15.13, 2.70] };
  const COLS = [['var(--grey-d)', 'var(--paper)'], ['var(--grey-b)', '#fff'], ['var(--grey-a)', '#fff'], ['var(--ours)', '#fff']];
  host.innerHTML = Object.entries(ENT).map(([k, arr]) => `<span>${k}</span><div class="stack">${arr.map((v, i) => `<div data-i="${i}" style="flex:${v} 1 0;background:${COLS[i][0]};color:${COLS[i][1]}">${v >= 9 ? Math.round(v) + '%' : ''}</div>`).join('')}</div>`).join('');
  $$('.stack', host).forEach((st, r) => $$('div', st).forEach((d, i) => {
    const k = Object.keys(ENT)[r], v = ENT[k][i];
    d.addEventListener('pointermove', e => showTip(`<b>${k}</b>: ${['0', '1', '2', '3 or more'][i]} per block (${v.toFixed(1)}%)`, e));
    d.addEventListener('pointerleave', hideTip);
  }));
  onReveal(host.closest('.fig'), () => {
    if (RM) return;
    $$('.stack', host).forEach((st, r) => st.animate([{ clipPath: 'inset(0 100% 0 0 round 6px)' }, { clipPath: 'inset(0 0% 0 0 round 6px)' }], { duration: 620, delay: r * 110, easing: EASE, fill: 'backwards' }));
  });
});

/* =============== 06 Dataset: Composition sunburst (expanded, no white plate) =============== */
run('sunburst', function sunburst() {
  const host = $('#sunburst');
  if (!host) return;
  const hex = c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16) / 255);
  const rgb = a => `rgb(${a.map(v => Math.round(Math.max(0, Math.min(1, v)) * 255)).join(',')})`;
  const blend = (a, k) => a.map(v => v * (1 - k) + k);
  const ROSE = hex('#B7799E'), BLUE = hex('#6E87B0'), dark = a => a.map(v => v * 0.8);
  const SH = 'labelled usable shots', N_SH = 288083;
  const QUADS = [
    { half: 'data', col: dark(ROSE), facets: [
      ['Scene', SH, N_SH, [['Indoor', 66.24, 190814], ['Outdoor', 33.53, 96597], [null, 0.23, 672, 'unclear']]],
      ['Characters in shot', SH, N_SH, [['Single character', 51.81, 149246], ['Multiple characters', 44.46, 128088], [null, 3.73, 10749, 'no character in shot']]]] },
    { half: 'bench', col: dark(BLUE), facets: [
      ['Evaluation dimension', 'scored questions (dev + test)', 1361, [['Character', 65.17, 887], ['Object', 6.32, 86], ['Scene', 18.81, 256], ['State', 9.70, 132]]],
      ['Recurrence distance', 'cross-shot consistency questions', 1229, [['Adjacent', 23.68, 291], ['2–4 apart', 32.79, 403], ['≥5 apart', 43.53, 535]]]] },
    { half: 'bench', col: BLUE, facets: [
      ['Shots per story', 'stories', 122, [['6–9 shots', 52.46, 64], ['10–12 shots', 47.54, 58]]],
      ['State change type', 'annotated state changes', 307, [['Posture', 53.75, 165], ['Held object', 21.82, 67], ['State or clothing', 24.43, 75]]]] },
    { half: 'data', col: ROSE, facets: [
      ['Visual style', SH, N_SH, [['Live action', 84.99, 244828], ['Animation', 15.01, 43255]]],
      ['Character recurrence', SH, N_SH, [['First seen', 31.38, 90409], ['Previous shot', 28.07, 80865], ['Earlier shot', 36.82, 106060], [null, 3.73, 10749, 'no character in shot']]]] },
  ];
  const catColor = (base, cats, lab) => { const labelled = cats.filter(c => c[0]), n = labelled.length; return blend(base, 0.28 + (n > 1 ? 0.34 * labelled.findIndex(c => c[0] === lab) / (n - 1) : 0.14)); };

  function bars() {
    const half = key => QUADS.filter(q => q.half === key).flatMap(q => q.facets.map((f, k) => ({ f, base: blend(q.col, 0.16 * k) })));
    const block = (title, sub, list) => `<div><h4>${title}<small>${sub}</small></h4>${list.map(({ f: [name, unit, total, cats], base }) => `<div class="sbar"><span class="label">${name}</span>
      <div class="mbar" style="height:12px">${cats.map(([lab, share, count, what]) => `<i style="flex:${share} 1 0;background:${lab ? rgb(catColor(base, cats, lab)) : rgb(blend(base, 0.72))}" title="${esc(lab || what)} ${share.toFixed(1)}%"></i>`).join('')}</div>
      <div class="mkey">${cats.filter(c => c[0]).map(([lab, share]) => `<span><i style="background:${rgb(catColor(base, cats, lab))}"></i>${esc(lab)} ${Math.round(share)}%</span>`).join('')}</div></div>`).join('')}</div>`;
    host.innerHTML = `<div class="sbars">${block('Dataset', '288,083 labelled training shots', half('data'))}${block('Benchmark', '1,361 CaT-MBench questions across 122 stories', half('bench'))}</div>`;
    return () => grow($$('.mbar i', host), 'left', 0, 22);
  }

  function ring() {
    const W = 760, H = 700, cx = W / 2, cy = H / 2, U = 360;
    const R = { c: 0.185 * U, i0: 0.185 * U, i1: 0.41 * U, o0: 0.415 * U, o1: 0.585 * U, dot: 0.665 * U };
    const P = (r, deg) => { const a = deg * Math.PI / 180; return [cx + r * Math.sin(a), cy - r * Math.cos(a)]; };
    const wedge = (r0, r1, d0, d1) => {
      const [x0, y0] = P(r1, d0), [x1, y1] = P(r1, d1), [x2, y2] = P(r0, d1), [x3, y3] = P(r0, d0);
      const L = d1 - d0 > 180 ? 1 : 0;
      return `M${x0} ${y0}A${r1} ${r1} 0 ${L} 1 ${x1} ${y1}L${x2} ${y2}A${r0} ${r0} 0 ${L} 0 ${x3} ${y3}Z`;
    };
    const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Composition sunburst: training data in upper half, CaT-MBench in lower half' }, host);
    const gW = sv('g', {}, svg), gL = sv('g', {}, svg);
    const segs = [], labels = [], facetsAll = [];
    let deg = 0;
    QUADS.forEach(q => q.facets.forEach(([name, unit, total, cats], k) => {
      const base = blend(q.col, 0.16 * k), d0 = deg, d1 = deg + 45; deg = d1;
      const fseg = sv('path', { d: wedge(R.i0, R.i1, d0, d1), fill: rgb(base), stroke: 'var(--ink)', 'stroke-width': 1.6, class: 'sb-seg' }, gW);
      const fac = { name, unit, total, d0, d1, el: fseg, cats: [] }; facetsAll.push(fac);
      let a = d0;
      cats.forEach(([lab, share, count, what]) => {
        const b0 = a, b1 = a + 45 * share / 100; a = b1;
        const color = lab ? catColor(base, cats, lab) : null;
        const el = sv('path', { d: wedge(R.o0, R.o1, b0, b1), fill: lab ? rgb(color) : rgb(blend(base, 0.72)), stroke: 'var(--ink)', 'stroke-width': 1.4, class: 'sb-seg' }, gW);
        const seg = { lab, share, count, what, b0, b1, mid: (b0 + b1) / 2, el, fac, color, base };
        segs.push(seg); fac.cats.push(seg);
        if (lab) labels.push(seg);
      });
    }));
    segs.forEach(sg => {
      if (!sg.lab || sg.share < 8) return;
      const [x, y] = P((R.o0 + R.o1) / 2, sg.mid);
      let rot = sg.mid - 90; if (rot > 90) rot -= 180; if (rot < -90) rot += 180;
      const L = sg.color.reduce((a, v, i) => a + v * [0.299, 0.587, 0.114][i], 0);
      T(gW, x, y + 4.5, `${Math.round(sg.share)}%`, '', `font:600 13px var(--mono);font-variant-numeric:tabular-nums;fill:${L < 0.52 ? '#fff' : '#121520'};pointer-events:none`, 'middle').setAttribute('transform', `rotate(${rot} ${x} ${y})`);
    });
    facetsAll.forEach(f => {
      const mid = (f.d0 + f.d1) / 2, [x, y] = P((R.i0 + R.i1) / 2, mid);
      let rot = mid; if (mid > 90 && mid < 270) rot = mid - 180;
      const words = f.name === 'State change type' ? ['State change', 'type'] : f.name.split(' ').length > 1 && f.name.length > 10 ? [f.name.split(' ').slice(0, -1).join(' '), f.name.split(' ').slice(-1)[0]] : [f.name];
      const t = sv('text', { x, y: y - (words.length - 1) * 7 + 5, 'text-anchor': 'middle', transform: `rotate(${rot} ${x} ${y})`, style: 'font:600 13px var(--body);fill:#fff;paint-order:stroke;stroke:rgba(17,21,32,.42);stroke-width:2.5px;pointer-events:none' }, gW);
      words.forEach((w, i) => { const sp = sv('tspan', { x, dy: i ? 15 : 0 }, t); sp.textContent = w; });
    });
    [0, 90, 180, 270].forEach(d => { const [x0, y0] = P(R.c, d), [x1, y1] = P(R.o1 + 2, d); sv('path', { d: `M${x0} ${y0}L${x1} ${y1}`, stroke: 'var(--ink)', 'stroke-width': 5 }, gW); });
    const repel = (angs, gap, lo, hi) => {
      const y = angs.slice();
      for (let i = 1; i < y.length; i++) if (y[i] - y[i - 1] < gap) y[i] = y[i - 1] + gap;
      if (y.length && y[y.length - 1] > hi) { y[y.length - 1] = hi; for (let i = y.length - 2; i >= 0; i--) y[i] = Math.min(y[i], y[i + 1] - gap); }
      if (y.length && y[0] < lo) { y[0] = lo; for (let i = 1; i < y.length; i++) if (y[i] - y[i - 1] < gap) y[i] = y[i - 1] + gap; }
      return y;
    };
    facetsAll.forEach(f => { const ls = f.cats.filter(c => c.lab).sort((a, b) => a.mid - b.mid); const ang = repel(ls.map(c => c.mid), 10, f.d0, f.d1); ls.forEach((c, i) => { c.ang = ang[i]; }); });
    labels.sort((a, b) => a.ang - b.ang);
    const glob = repel(labels.map(l => l.ang), 10, -360, 720);
    labels.forEach((l, i) => { l.ang = glob[i]; });
    const txt = l => l.share < 8 ? `${l.lab} ${Math.round(l.share)}%` : l.lab;
    const lines = l => ({ 'Single character': ['Single', 'character'], 'Multiple characters': ['Multiple', 'characters'], 'Held object': ['Held', 'object'], 'State or clothing': ['State or', 'clothing'] }[l.lab] || [txt(l)]);
    const box = (l, er) => {
      const r = R.dot + er + 12, [x, y] = P(r, l.ang), ls = lines(l), w = Math.max(...ls.map(s => s.length)) * 7.6, hh = ls.length * 16;
      const s = Math.sin(l.ang * Math.PI / 180), c = Math.cos(l.ang * Math.PI / 180);
      if (Math.abs(s) >= Math.abs(c)) return s > 0 ? [x, x + w, y - hh / 2, y + hh / 2] : [x - w, x, y - hh / 2, y + hh / 2];
      return c > 0 ? [x - w / 2, x + w / 2, y - hh, y] : [x - w / 2, x + w / 2, y, y + hh];
    };
    const placed = [];
    labels.slice().sort((a, b) => Math.abs(Math.sin(b.ang * Math.PI / 180)) - Math.abs(Math.sin(a.ang * Math.PI / 180))).forEach(l => {
      let er = 0;
      for (; er < 120; er += 4) { const b = box(l, er); if (!placed.some(p => !(b[1] <= p[0] + 2 || b[0] >= p[1] - 2 || b[3] <= p[2] + 1 || b[2] >= p[3] - 1))) break; }
      l.er = er; placed.push(box(l, er));
    });
    labels.forEach(l => {
      const r = R.dot + l.er, [xr, yr] = P(R.o1, l.mid), [xe, ye] = P(r, l.mid), [xl, yl] = P(r, l.ang), col = rgb(blend(l.base, 0.08));
      const g = sv('g', { class: 'sb-lab' }, gL);
      sv('path', { d: `M${xr} ${yr}L${xe} ${ye}L${xl} ${yl}`, fill: 'none', stroke: col, 'stroke-width': 1.4, 'stroke-linecap': 'round' }, g);
      sv('circle', { cx: xl, cy: yl, r: 3.5, fill: 'var(--ink)', stroke: rgb(l.base), 'stroke-width': 1.6 }, g);
      const s = Math.sin(l.ang * Math.PI / 180), c = Math.cos(l.ang * Math.PI / 180), [xt, yt] = P(r + 10, l.ang), ls = lines(l);
      let anchor = 'middle', y0 = yt;
      if (Math.abs(s) >= Math.abs(c)) { anchor = s > 0 ? 'start' : 'end'; y0 = yt - (ls.length - 1) * 8 + 5; }
      else if (c > 0) y0 = yt - (ls.length - 1) * 16 - 2;
      else y0 = yt + 12;
      const t = sv('text', { x: xt, y: y0, 'text-anchor': anchor, style: 'font:500 13.5px var(--body);fill:var(--paper)' }, g);
      ls.forEach((w, i) => { const sp = sv('tspan', { x: xt, dy: i ? 16 : 0 }, t); sp.textContent = w; });
      l.lg = g;
    });

    /* Center ring directly on --ink canvas (no white backing plate) */
    const cg = sv('g', {}, svg);
    sv('circle', { cx, cy, r: R.c - 4, fill: 'var(--ink)', stroke: 'var(--rule)' }, cg);
    const top = sv('g', { style: 'transition:opacity .18s' }, cg), bot = sv('g', { style: 'transition:opacity .18s' }, cg);
    sv('path', { d: `M${cx - 48} ${cy - 13}l6 -9l6 9h-4v8h-4v-8z`, fill: 'var(--cut)' }, top);
    T(top, cx - 32, cy - 10, 'Dataset', '', 'font:600 17.5px var(--body);fill:var(--cut);letter-spacing:-.02em');
    sv('path', { d: `M${cx - 56} ${cy + 13}l6 9l6 -9h-4v-8h-4v8z`, fill: 'var(--ours)' }, bot);
    T(bot, cx - 40, cy + 25, 'Benchmark', '', 'font:600 16.5px var(--body);fill:var(--ours);letter-spacing:-.02em');
    const info = sv('g', { style: 'opacity:0;transition:opacity .18s' }, cg);
    const iCat = T(info, cx, cy - 25, '', '', 'font:500 10px var(--mono);fill:var(--mute)', 'middle');
    const iVal = T(info, cx, cy + 5, '', '', 'font:600 25px var(--mono);fill:var(--paper);letter-spacing:-.02em', 'middle');
    const iLab = T(info, cx, cy + 23, '', '', 'font:500 11.5px var(--body);fill:var(--paper)', 'middle');
    const iCnt = T(info, cx, cy + 38, '', '', 'font:500 10px var(--mono);fill:var(--mute)', 'middle');
    function focus(list, cat, val, lab, cnt) {
      segs.forEach(sg => sg.el.classList.toggle('dim', !list.includes(sg.el)));
      facetsAll.forEach(f => f.el.classList.toggle('dim', !list.includes(f.el)));
      labels.forEach(l => l.lg.classList.toggle('dim', !list.includes(l.el)));
      top.style.opacity = bot.style.opacity = 0; info.style.opacity = 1;
      iCat.textContent = cat; iVal.textContent = val; iLab.textContent = lab; iCnt.textContent = cnt;
    }
    function reset() {
      [...segs, ...facetsAll].forEach(x => x.el.classList.remove('dim'));
      labels.forEach(l => l.lg.classList.remove('dim'));
      top.style.opacity = bot.style.opacity = 1; info.style.opacity = 0;
    }
    segs.forEach(sg => {
      sg.el.addEventListener('pointerenter', () => focus([sg.el, sg.fac.el], sg.fac.name.toLowerCase(), `${sg.share.toFixed(1)}%`, sg.lab || sg.what, `${sg.count.toLocaleString('en-US')} / ${sg.fac.total.toLocaleString('en-US')}`));
      sg.el.addEventListener('pointerleave', reset);
    });
    facetsAll.forEach(f => {
      f.el.addEventListener('pointerenter', () => focus([f.el, ...f.cats.map(c => c.el)], f.unit, f.total.toLocaleString('en-US'), f.name, `${f.cats.filter(c => c.lab).length} categories`));
      f.el.addEventListener('pointerleave', reset);
    });
    try { const b = svg.getBBox(); if (b.width > 0) svg.setAttribute('viewBox', `${b.x - 8} ${b.y - 8} ${b.width + 16} ${b.height + 16}`); } catch (e) {}
    return () => {
      const all = [...facetsAll.map(f => ({ el: f.el, d: f.d0 })), ...segs.map(s => ({ el: s.el, d: s.b0 }))];
      all.forEach(({ el, d }) => { el.style.transformBox = 'view-box'; el.style.transformOrigin = `${cx}px ${cy}px`; el.animate([{ opacity: 0, transform: 'scale(.9)' }, { opacity: 1, transform: 'none' }], { duration: 480, delay: d * 0.85, easing: EASE, fill: 'backwards' }); });
      labels.forEach(l => l.lg.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 380, delay: 180 + l.mid * 0.85, fill: 'backwards' }));
    };
  }
  chart(host, W => (W < 560 ? bars() : ring()));
});
})();
