/* CaT-Movie project page: charts, annotated scene block, judge replay. Needs app.js (window.CAT). */
(() => {
'use strict';
const { MEDIA, $, $$, sv, h, esc, once, watch, onReveal, chart, setTab, showTip, hideTip, pick, RM, T1, POINT_COLOR, run, story, frame, short, camLabel, pad, HERO, avatarStyle, charTip } = window.CAT;
const EASE = 'cubic-bezier(.2,.7,.2,1)';
const T = (g, x, y, str, cls = 'svg-text', style = '', anchor = 'start') => { const t = sv('text', { x, y, 'text-anchor': anchor, class: cls, style }, g); t.textContent = str; return t; };
const svgAt = (host, W, H, label) => sv('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': label }, host);
/* bars grow from their baseline, lines draw, marks pop: the same three intros everywhere */
const grow = (els, origin = 'bottom', base = 0, step = 35) => els.forEach((el, i) => {
  el.style.transformBox = 'fill-box'; el.style.transformOrigin = origin;
  el.animate([{ transform: origin === 'left' ? 'scaleX(0)' : 'scaleY(0)' }, { transform: 'none' }], { duration: 650, delay: base + i * step, easing: EASE, fill: 'backwards' });
});
const drawLine = (p, delay = 0, dur = 750) => {
  const L = p.getTotalLength(), dash = p.getAttribute('stroke-dasharray');
  p.animate([{ strokeDasharray: `${L} ${L}`, strokeDashoffset: L }, { strokeDasharray: `${L} ${L}`, strokeDashoffset: 0 }], { duration: dur, delay, easing: EASE, fill: 'backwards' })
    .onfinish = () => { if (dash) p.setAttribute('stroke-dasharray', dash); };
};
const pop = (els, base = 0, step = 50) => els.forEach((el, i) => {
  el.style.transformBox = 'fill-box'; el.style.transformOrigin = 'center';
  el.animate([{ opacity: 0, transform: 'scale(.4)' }, { opacity: 1, transform: 'none' }], { duration: 380, delay: base + i * step, easing: EASE, fill: 'backwards' });
});
const fadeIn = (els, base = 0, step = 0) => els.forEach((el, i) => el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 400, delay: base + i * step, fill: 'backwards' }));
const gridY = (svg, m, W, Y, ticks, fmt) => {
  const grid = sv('g', { class: 'grid' }, svg), ax = sv('g', { class: 'axis' }, svg);
  ticks.forEach(v => { sv('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid); T(ax, m.l - 8, Y(v) + 4, fmt(v), 'svg-text', '', 'end'); });
  return ax;
};

/* =============== Results: consistency against transition control =============== */
run('pareto', function pareto() {
  const host = $('#pareto');
  if (!host) return;
  let nodes = [], pinned = null;
  const pts = T1.filter(r => r.name).map(r => ({ name: r.name, fam: r.fam, tr: (r.v[0] + r.v[1]) / 2, co: (r.v[2] + r.v[3] + r.v[4]) / 3, r }));
  const mark = name => nodes.forEach(n => { n.classList.toggle('hl', n.dataset.name === name); n.classList.toggle('dim', !!name && n.dataset.name !== name); });
  document.addEventListener('pick', e => { if (e.detail.from !== 'pareto') mark(e.detail.name || pinned); });
  chart(host, W => {
    const narrow = W < 540;
    const H = Math.round(Math.max(300, Math.min(430, W * 0.58)));
    const m = { l: 52, r: narrow ? 10 : 18, t: 12, b: 48 };
    const x0 = 0.79, x1 = 1.012, y0 = 0.18, y1 = 0.64;
    const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
    const Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
    const svg = svgAt(host, W, H, 'Inter-shot consistency against transition control for eight methods on 92 test stories');
    const grid = sv('g', { class: 'grid' }, svg), ax = sv('g', { class: 'axis' }, svg);
    (narrow ? [0.8, 0.9, 1.0] : [0.80, 0.85, 0.90, 0.95, 1.00]).forEach(v => { sv('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }, grid); T(ax, X(v), H - m.b + 18, v.toFixed(2), 'svg-text', '', 'middle'); });
    [0.2, 0.3, 0.4, 0.5, 0.6].forEach(v => { sv('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid); T(ax, m.l - 9, Y(v) + 4, v.toFixed(1), 'svg-text', '', 'end'); });
    sv('path', { d: `M${m.l} ${m.t} V${H - m.b} H${W - m.r}`, fill: 'none' }, ax);
    T(ax, (m.l + W - m.r) / 2, H - 8, narrow ? 'Transition control →' : 'Transition control: planned cuts made →', 'svg-text', 'fill:var(--text)', 'middle');
    T(ax, 0, 0, narrow ? 'Inter-shot consistency →' : 'Consistency: same characters, objects, places →', 'svg-text', 'fill:var(--text)', 'middle').setAttribute('transform', `translate(13 ${(m.t + H - m.b) / 2}) rotate(-90)`);

    const base = pts.filter(p => p.fam !== 'ours'), ours = pts.find(p => p.fam === 'ours');
    const front = base.slice().sort((a, b) => b.tr - a.tr).reduce((acc, p) => { if (!acc.length || p.co > acc[acc.length - 1].co) acc.push(p); return acc; }, []);
    const f = xv => Math.max(...base.filter(p => p.tr >= xv).map(p => p.co), y0);
    const bps = front.map(p => p.tr).filter(v => v < ours.tr).sort((a, b) => b - a);
    let poly = `M${X(bps[bps.length - 1])} ${Y(ours.co)} H${X(ours.tr)} V${Y(f(ours.tr))}`;
    bps.forEach((bx, i) => { poly += ` H${X(bx)}`; if (i < bps.length - 1) poly += ` V${Y(f(bx - 1e-6))}`; });
    const region = sv('path', { d: poly + 'Z', fill: 'var(--ours)', 'fill-opacity': 0.12, stroke: 'none' }, svg);
    let st = `M${X(front[0].tr)} ${Y(y0)} V${Y(front[0].co)}`;
    for (let i = 1; i < front.length; i++) st += ` H${X(front[i].tr)} V${Y(front[i].co)}`;
    st += ` H${X(x0)}`;
    const fp = sv('path', { d: st, fill: 'none', stroke: 'var(--grey-b)', 'stroke-width': 1.4, 'stroke-dasharray': '3 4' }, svg);

    const LBL = {
      StoryMem: [0, -15, 'middle'], HoloCine: [14, 5, 'start'], MultiShotMaster: [-14, 21, 'end'], 'CaT-Movie': [-17, 5, 'end'],
      StoryDiffusion: [12, 4, 'start'], 'IC-LoRA': [-12, 4, 'end'], 'Wan2.2, per shot': [-11, -10, 'end'], 'Wan2.1, per shot': [-11, 19, 'end'],
    };
    nodes = [];
    pts.forEach(p => {
      const g = sv('g', { class: 'pt', tabindex: 0, 'data-name': p.name }, svg);
      const cx = X(p.tr), cy = Y(p.co), col = POINT_COLOR[p.name];
      if (p.fam === 'shot') sv('circle', { cx, cy, r: 7, fill: col, stroke: 'var(--ink)', 'stroke-width': 2 }, g);
      else {
        const s = p.fam === 'ours' ? 12 : 9;
        if (p.fam === 'ours') sv('rect', { x: cx - s, y: cy - s, width: 2 * s, height: 2 * s, transform: `rotate(45 ${cx} ${cy})`, fill: 'none', stroke: col, 'stroke-opacity': 0.3, 'stroke-width': 7 }, g);
        sv('rect', { x: cx - s * 0.72, y: cy - s * 0.72, width: 1.44 * s, height: 1.44 * s, transform: `rotate(45 ${cx} ${cy})`, fill: col, stroke: 'var(--ink)', 'stroke-width': 2 }, g);
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
      fadeIn([region], 500);
      drawLine(fp, 0, 900);
      pop(nodes.filter(n => n.dataset.name !== 'CaT-Movie'), 150, 60);
      pop(nodes.filter(n => n.dataset.name === 'CaT-Movie'), 800);
    };
  });
});

/* =============== Results: character checks by return time =============== */
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
  const OP = [0.9, 0.58, 0.32];
  const fill = (g, s) => names[g] === 'CaT-Movie' ? `color-mix(in srgb,var(--ours) ${[95, 62, 36][s]}%,transparent)` : `color-mix(in srgb,var(--grey-b) ${OP[s] * 100}%,transparent)`;
  $('#rtLegend').innerHTML = bins.map((b, s) => `<span><i class="sq" style="background:color-mix(in srgb,var(--grey-b) ${OP[s] * 100}%,transparent)"></i>${b}</span>`).join('') + '<span><i class="sq" style="background:var(--ours)"></i>CaT-Movie</span>';
  chart(host, W => {
    const narrow = W < 520;
    const H = 290, m = { l: 44, r: 6, t: 12, b: narrow ? 50 : 36 };
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
        bars.push(sv('rect', { x, y: Y(d.rate[s]), width: bw, height: Y(yMin) - Y(d.rate[s]), rx: 2, fill: fill(gi, s) }, g));
        whisk.push(sv('path', { d: `M${x + bw / 2} ${Y(d.lo[s])} V${Y(d.hi[s])} M${x + bw / 2 - 3.5} ${Y(d.lo[s])} h7 M${x + bw / 2 - 3.5} ${Y(d.hi[s])} h7`, stroke: 'var(--paper)', 'stroke-opacity': 0.5, 'stroke-width': 1.2, fill: 'none' }, g));
        g.addEventListener('pointermove', e => showTip(`<b>${n}</b>, returning ${b} later<table><tr><td>pass rate</td><td>${(d.rate[s] * 100).toFixed(1)}%</td></tr><tr><td>95% interval</td><td>${(d.lo[s] * 100).toFixed(1)}–${(d.hi[s] * 100).toFixed(1)}%</td></tr><tr><td>checks</td><td>${N[s]}</td></tr></table>`, e));
        g.addEventListener('pointerleave', hideTip);
      });
    });
    return () => { grow(bars, 'bottom', 100, 30); fadeIn(whisk, 700, 15); };
  });
});

/* =============== Method drawer: consistency by window distance =============== */
run('distChart', function distChart() {
  const host = $('#distChart');
  if (!host) return;
  const S = [
    ['ours', 'CaT-Movie', 'var(--ours)', [46.3, 53.9, 59.8], [147, 297, 480], null, 'circle', ''],
    ['recent', 'Recent-shot recall', 'var(--grey-a)', [46.9, 54.9, 44.4], [147, 297, 480], [[0.7, -5.6, 9.4], [1.0, -5.1, 7.3], [-15.4, -22.7, -9.0]], 'square', '7 5'],
    ['no_refs', 'Trained without references', 'var(--cut)', [38.8, 31.3, 32.1], [147, 297, 480], [[-7.5, -20.2, 3.2], [-22.6, -30.6, -14.4], [-27.7, -35.3, -20.7]], 'tri', '2 5'],
  ];
  const XL = ['same window', '1 window apart', '2 or more apart'];
  chart(host, W => {
    const H = 280, m = { l: 44, r: 16, t: 22, b: 40 };
    const pad = Math.min(60, (W - m.l - m.r) * 0.12);
    const X = i => m.l + pad + i * ((W - m.l - m.r - 2 * pad) / 2);
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
        if (mk === 'circle') sv('circle', { cx: x, cy: yy, r: 5.5, fill: col, stroke: 'var(--ink)', 'stroke-width': 2 }, pg);
        else if (mk === 'square') sv('rect', { x: x - 5, y: yy - 5, width: 10, height: 10, fill: col, stroke: 'var(--ink)', 'stroke-width': 2 }, pg);
        else sv('path', { d: `M${x} ${yy - 6.5}L${x + 6} ${yy + 4.5}L${x - 6} ${yy + 4.5}Z`, fill: col, stroke: 'var(--ink)', 'stroke-width': 2 }, pg);
        const [dx, dy, an] = k === 'ours' ? [[-11, 5, 'end'], [0, 21, 'middle'], [0, -12, 'middle']][i] : k === 'recent' ? [[0, -12, 'middle'], [0, -12, 'middle'], [0, 21, 'middle']][i] : [0, 22, 'middle'];
        T(pg, x + dx, yy + dy, y.toFixed(1), 'svg-text', `fill:${col};font-weight:${k === 'ours' ? 600 : 450}`, an);
        pg.addEventListener('pointermove', e => showTip(`<b>${name}</b>, ${XL[i]}<table><tr><td>pass rate</td><td>${y.toFixed(1)}%</td></tr><tr><td>checks, 3 seeds</td><td>${n[i]}</td></tr>${d ? `<tr><td>against CaT-Movie</td><td>${d[i][0] > 0 ? '+' : ''}${d[i][0].toFixed(1)} [${d[i][1]}, ${d[i][2]}]</td></tr>` : ''}</table>`, e));
        pg.addEventListener('pointerleave', hideTip);
        marks.push(pg);
      });
    });
    return () => { lines.forEach((p, i) => drawLine(p, i * 90, 700)); pop(marks, 200, 40); };
  });
});

/* =============== Method drawer: what the planner adds =============== */
run('planner', function planner() {
  const a = $('#planText'), b = $('#writExec');
  if (!a || !b) return;
  const D = { opening: { Syn: [0.1625, 0.4471], Para: [0.5649, 0.7091], Shot: [0.4833, 0.7304] }, held: { Syn: [0.0732, 0.2171], Para: [0.5721, 0.6797], Shot: [0.3122, 0.5122] } };
  const TL = { Syn: 'synopsis', Para: 'paragraph', Shot: 'shot list' };
  chart(a, W => {
    const H = 270, m = { l: 44, r: 6, t: 20, b: 50 };
    const Y = v => H - m.b - v * (H - m.t - m.b);
    const svg = svgAt(a, W, H, 'Share of reference opening states and held items stated by the request alone and with the planner');
    gridY(svg, m, W, Y, [0, 0.25, 0.5, 0.75, 1], v => (v * 100) + '%');
    const gw = (W - m.l - m.r) / 2, bw = Math.min(56, (gw - 40) / 3), segs = [], labs = [];
    [['opening', 'Opening states'], ['held', 'Held items']].forEach(([key, name], gi) => {
      const cx = m.l + gi * gw + gw / 2;
      T(svg, cx, H - 10, name, 'svg-text', 'fill:var(--paper);font-weight:500', 'middle');
      ['Syn', 'Para', 'Shot'].forEach((t, ti) => {
        const [r, tot] = D[key][t], x = cx + (ti - 1) * (bw + 12) - bw / 2;
        const g = sv('g', { style: 'cursor:pointer' }, svg);
        segs.push(sv('rect', { x, y: Y(r), width: bw, height: Y(0) - Y(r), fill: 'var(--grey-c)', 'fill-opacity': 0.5, rx: 2 }, g));
        const top = sv('rect', { x, y: Y(tot), width: bw, height: Y(r) - Y(tot), fill: 'var(--ours)', rx: 2 }, g);
        labs.push(T(g, x + bw / 2, Y(tot) - 7, `${Math.round(tot * 100)}%`, 'svg-text', 'fill:var(--paper);font-weight:600', 'middle'));
        T(svg, x + bw / 2, H - m.b + 17, TL[t], 'svg-text', 'font-size:11.5px', 'middle');
        g.addEventListener('pointermove', e => showTip(`<b>${name}, ${TL[t]} requests</b><table><tr><td>stated by the request</td><td>${(r * 100).toFixed(1)}%</td></tr><tr><td>added by the planner</td><td>+${((tot - r) * 100).toFixed(1)}</td></tr><tr><td>in the compiled text</td><td>${(tot * 100).toFixed(1)}%</td></tr></table>`, e));
        g.addEventListener('pointerleave', hideTip);
        segs.push(top);
      });
    });
    return () => { grow(segs, 'bottom', 0, 30); fadeIn(labs, 600, 30); };
  });

  const R = [['Request', 170, 79, 49, 42], ['Reference', 171, 91, 59, 21], ['Our planner', 171, 108, 48, 15]];
  chart(b, W => {
    const H = 290, m = { l: 44, r: 6, t: 12, b: 64 };
    const Y = v => H - m.b - v * (H - m.t - m.b);
    const svg = svgAt(b, W, H, 'State questions shown, stated but not shown, or unstated for three storyboards');
    gridY(svg, m, W, Y, [0, 0.25, 0.5, 0.75, 1], v => (v * 100) + '%');
    const defs = sv('defs', {}, svg);
    const hp = sv('pattern', { id: 'hatchP', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
    sv('rect', { width: 6, height: 6, fill: 'var(--cut)', 'fill-opacity': 0.16 }, hp);
    sv('path', { d: 'M0 0V6', stroke: 'var(--cut)', 'stroke-width': 2 }, hp);
    const gw = (W - m.l - m.r) / 3, bw = Math.min(88, gw - 36), parts = [], labs = [];
    R.forEach(([name, n, se, sn, ns], i) => {
      const x = m.l + i * gw + (gw - bw) / 2, f1 = se / n, f2 = sn / n, f3 = ns / n;
      const g = sv('g', { style: 'cursor:pointer' }, svg);
      parts.push(sv('rect', { x, y: Y(f1), width: bw, height: Y(0) - Y(f1), fill: 'var(--ours)', rx: 2 }, g));
      parts.push(sv('rect', { x, y: Y(f1 + f2), width: bw, height: Y(f1) - Y(f1 + f2), fill: 'url(#hatchP)', rx: 2 }, g));
      sv('rect', { x: x + 0.5, y: Y(1) + 0.5, width: bw - 1, height: Y(f1 + f2) - Y(1) - 1, fill: 'none', stroke: 'var(--rule)', 'stroke-dasharray': '3 3', rx: 2 }, g);
      const ours = name === 'Our planner';
      T(svg, x + bw / 2, H - m.b + 18, name, 'svg-text', `fill:${ours ? 'var(--ours)' : 'var(--paper)'};font-weight:600`, 'middle');
      T(svg, x + bw / 2, H - m.b + 34, `stated ${Math.round((se + sn) / n * 100)}%`, 'svg-text', '', 'middle');
      T(svg, x + bw / 2, H - m.b + 49, `${Math.round(se / (se + sn) * 100)}% of it shown`, 'svg-text', '', 'middle');
      labs.push(T(g, x + bw / 2, Y(f1 / 2) + 4, Math.round(f1 * 100), 'svg-text', 'fill:var(--ink);font-weight:600;font-size:13px', 'middle'));
      labs.push(T(g, x + bw / 2, Y(f1 + f2 / 2) + 4, Math.round(f2 * 100), 'svg-text', 'fill:var(--paper);font-weight:600;font-size:13px;paint-order:stroke;stroke:var(--ink);stroke-width:3px', 'middle'));
      labs.push(T(g, x + bw / 2, Y(f1 + f2 + f3 / 2) + 4, Math.round(f3 * 100), 'svg-text', 'font-weight:600;font-size:13px', 'middle'));
      g.addEventListener('pointermove', e => showTip(`<b>${name} storyboard</b><table><tr><td>stated and shown</td><td>${se} (${(f1 * 100).toFixed(0)}%)</td></tr><tr><td>stated, not shown</td><td>${sn} (${(f2 * 100).toFixed(0)}%)</td></tr><tr><td>unstated</td><td>${ns} (${(f3 * 100).toFixed(0)}%)</td></tr><tr><td>state questions</td><td>${n}</td></tr></table>`, e));
      g.addEventListener('pointerleave', hideTip);
    });
    return () => { grow(parts, 'bottom', 0, 60); fadeIn(labs, 550, 25); };
  });
});

/* =============== Method: one window pass, laid out like the paper's overview; stages stepped, played once =============== */
run('overview', function overview() {
  const fig = $('#ovFig');
  if (!fig) return;
  const s = story(HERO), WI = 3, W = s.windows[WI], A = W.shots[0], B = W.shots[1];
  const recalled = new Set(W.refs.map(r => r.shot));
  const ICE = '<svg class="ico" viewBox="0 0 14 14" aria-label="frozen"><g stroke="var(--ours)" stroke-width="1.3" stroke-linecap="round" fill="none"><path d="M7 1.5v11M2.2 4.25l9.6 5.5M2.2 9.75l9.6-5.5"/><path d="M5.5 2.5 7 3.9l1.5-1.4M5.5 11.5 7 10.1l1.5 1.4"/></g></svg>';
  const FLAME = '<svg class="ico" viewBox="0 0 14 14" aria-label="fine-tuned"><path d="M7 13.2c-2.9 0-4.4-2-4.4-4.1 0-2.5 2-3.6 2.5-6.3 1 .9 1.5 1.9 1.5 3 .7-.5 1.1-1.5 1-2.8 2 1.4 3.9 3.6 3.9 6 0 2.3-1.7 4.2-4.5 4.2z" fill="#e07b39"/></svg>';
  const shotList = a => a.join(', ').replace(/, (\d+)$/, ' and $1');
  const STAGES = [
    ['Request', 'A request (a synopsis, a paragraph or, as here, a shot list) goes to frozen Qwen3.8-27B, which expands it into one structured storyboard in a single pass.'],
    ['Storyboard', `The storyboard registers each recurring character, object and place once, then plans every shot: camera, who is present and in what state, the action and any change. Shots ${A} and ${B} form Window 4.`],
    ['Compile', `Code carries attire and held items from shot to shot and compiles one self-contained prompt per shot, plus an identity line for each recalled frame. c<sub>${A}</sub> begins: “${esc(W.prompts[0].split('. ').slice(0, 1).join('. '))}.”`],
    ['Recall', `Code looks up each entity’s anchor view in the windows already generated: Window 4 recalls Shots ${shotList([...recalled])}. Shot ${A} reads two of them, Shot ${B} reads three.`],
    ['Encode', 'Each shot is VAE-encoded on its own, so no latent frame spans a cut. Recalled frames take time positions five apart, ending five before the first frame of the shot that reads them.'],
    ['Denoise', 'In every transformer block, each shot attends to itself, to the first latent frame of the other shot, and only to the recalled frames routed to it; it cross-attends only to its own prompt.'],
    ['Decode', `The VAE decoder returns Shots ${A} and ${B}. They join the generated history that Window 5 recalls from.`],
  ];
  const REST = `<b>One window pass.</b> The planner writes the storyboard once. For Window 4, the renderer turns two compiled prompts and ${W.refs.length} recalled frames into Shots ${A} and ${B}. Step through the stages, or play the pass.`;
  const avatar = n => `<i style="${avatarStyle(n)}"></i>`;
  /* where each storyboard detail comes from, read mechanically against the request line: characters the request does not name were
     added by the planner (inferred); a posture none of whose words appear in the request was filled in (completed) */
  const named = k => new Set(s.requests.shots[k - 1].match(/Character \d+/g) || []);
  const STOP = new Set(['with', 'that', 'this', 'from', 'into', 'while', 'their']);
  const posture = sh => { const c = sh.present.find(p => /^Character/.test(p.name) && p.posture); return c ? c.posture.split(',')[0] : ''; };
  const prov = (k, text) => { const req = s.requests.shots[k - 1].toLowerCase(); const words = (text.toLowerCase().replace(/character \d+/g, '').match(/[a-z]+/g) || []).filter(w => w.length > 3 && !STOP.has(w)); return words.some(w => req.includes(w)) ? '' : 'cmp'; };
  const row = k => {
    const sh = s.plan.shots[k - 1], who = sh.present.filter(p => /^Character/.test(p.name)).map(p => p.name), pos = posture(sh), nm = named(k);
    return `<div class="ov-tr${k === A || k === B ? ' cur' : ''}"><b>${pad(k)}</b><span class="who">${who.map(n => `<i class="${nm.has(n) ? '' : 'inf'}" style="${avatarStyle(n)}"></i>`).join('')}</span><span>${esc(camLabel(sh.camera))}${pos ? `; <span class="${prov(k, pos)}">${esc(pos.replace(/Character (\d+)/g, 'C$1'))}</span>` : ''}</span></div>`;
  };
  const ph = '<i class="ov-dn ph" aria-hidden="true"></i>', dn = '<i class="ov-dn" aria-hidden="true"></i>';
  const navSvg = d => `<svg viewBox="0 0 10 10"><path d="${d}" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`;
  fig.innerHTML = `<div class="fig-head"><h3>One rendering pass</h3>
      <div class="ov-ctl"><div class="tabs" id="ovTabs" role="group" aria-label="Stages">${STAGES.map(([n], i) => `<button type="button" data-n="${i + 1}" aria-pressed="false"><b>${i + 1}</b>${n}</button>`).join('')}</div>
        <button type="button" class="ibtn" data-d="-1" aria-label="Previous stage">${navSvg('M6.5 1.5 3 5l3.5 3.5')}</button><button type="button" class="ibtn" data-d="1" aria-label="Next stage">${navSvg('M3.5 1.5 7 5 3.5 8.5')}</button><button type="button" class="tbtn" id="ovPlay"></button></div></div>
    <div class="ov-map" id="ovMap">
      <div class="ov-reg ov-pl">
        <p class="ov-rt">CaT-Movie Planner</p>
        <div class="ovn bare" data-st="1"><span class="ovl">Request, a shot list</span>${[A, B].map(k => `<p class="ov-q"><b>S${pad(k)}</b>${esc(s.requests.shots[k - 1])}</p>`).join('')}</div>
        ${dn}
        <div class="ovn pill" data-st="1">Input adaptation <small>Qwen3.8-27B</small>${ICE}</div>
        ${dn}
        <div class="ovn bare" data-st="2" id="ovBoard"><span class="ovl">Storyboard</span>
          <div class="ov-avs">${s.plan.characters.map(c => `<span class="ov-av" data-e="${esc(c.name)}">${avatar(c.name)}${esc(short(c.name))}</span>`).join('')}</div>
          <div class="ov-th"><span>Shot</span><span>Entities</span><span>Shot plan</span></div>
          ${[A - 1, A, B, B + 1].map(row).join('')}
          <p class="ov-key"><span>Preserved</span><span class="inf">Inferred</span><span class="cmp">Completed</span></p>
        </div>
        ${dn}
        <div class="ovn pill" data-st="3" id="ovComp">Prompt compiler <small>code</small></div>
      </div>
      ${ph}
      <div class="ov-top">
        <div class="ovn bare" data-st="4,7" id="ovHist"><div class="ov-strip">${s.windows.map((w, wi) => wi > WI
          ? `<div class="hw plan" style="--n:${w.shots.length}"><span>Window ${wi + 1}</span><div>Planned</div></div>`
          : `<div class="hw${wi === WI ? ' now' : ''}" style="--n:${w.shots.length}"><span>${wi === WI ? 'Current window' : `Window ${wi + 1}`}</span><div>${w.shots.map(k => `<i class="hs${recalled.has(k) ? ' rec' : ''}" data-k="${k}" style="background-image:url(${frame(HERO, k)})"><em>${k}</em>${wi === WI ? '<span class="nz"></span>' : ''}</i>`).join('')}</div></div>`).join('')}</div></div>
        <span class="legend ov-tf"><span>${FLAME}trainable</span><span>${ICE}frozen</span></span>
      </div>
      ${ph}
      <div class="ov-mid">
        <div class="ov-mid-a"><div class="ovn pill" data-st="4" id="ovRecall">Frame recall</div>${dn}<div class="ovn pill" data-st="5" id="ovEnc">VAE encoder${ICE}</div></div>
        <div class="ov-gap"></div>
        <div class="ov-mid-b"><div class="ovn pill" data-st="6" id="ovT5">umT5${ICE}</div><i class="ov-dn up" aria-hidden="true"></i><div class="ovn ov-stack" data-st="3" id="ovPrompts"><span><b>c<sup>ref</sup></b>identity lines</span><span><b>c<sub>${A}</sub></b>Shot ${A} prompt</span><span><b>c<sub>${B}</sub></b>Shot ${B} prompt</span></div></div>
      </div>
      ${ph}
      <div class="ov-reg ov-rd">
        <p class="ov-rt">CaT-Movie Renderer<small>Wan2.2-TI2V-5B, fine-tuned without new parameters</small></p>
        <div class="ov-rd-in">
          <div class="ov-tokb ovn bare" data-st="5" id="ovTokBox"><div class="ov-tok-head"><span class="ovl">RoPE time index</span><div class="tabs" id="ovReader" role="group" aria-label="Reading shot"><button type="button" data-k="${A}" class="on" aria-pressed="true">read by Shot ${A}</button><button type="button" data-k="${B}" aria-pressed="false">Shot ${B}</button></div></div><div class="chart" id="ovTok"></div></div>
          <div class="ovn ov-dit" data-st="6" id="ovDit"><span class="ovl" style="margin:4px 0 0">Transformer block × N</span>
            <div class="ov-layer"><i class="dot" style="background:var(--ours)"></i>Reference-routed sparse self-attention${FLAME}</div>
            <div class="ov-layer" id="ovCross"><i class="dot" style="background:var(--holo)"></i>Shot-wise cross-attention${FLAME}</div>
            <div class="ov-layer"><i class="dot" style="background:var(--grey-c)"></i>Feed-forward${FLAME}</div>
          </div>
          <div class="ov-dec" id="ovDec"><div class="ovn pill" data-st="7">VAE decoder${ICE}</div><div class="ov-outs ovn bare" data-st="7">${[A, B].map(k => `<div class="ov-out"><i style="background-image:url(${frame(HERO, k)})"><span class="nz"></span></i><b>Shot ${k}</b></div>`).join('')}</div></div>
          <div class="ov-mreg ovn" data-st="6"><p class="ov-rt">Reference-routed attention mask</p><div id="ovMask"></div>
            <span class="legend"><span><i class="sq" style="background:var(--grey-a)"></i>own shot</span><span><i class="sq" style="background:var(--holo)"></i>other shot, first frame</span><span><i class="sq" style="background:var(--ours)"></i>routed recalled frame</span></span>
          </div>
        </div>
      </div>
      <svg class="ov-links" aria-hidden="true"></svg>
    </div>
    <div class="ov-foot"><p class="cap" id="ovCap" aria-live="polite"></p></div>`;

  $$('.ov-av', fig).forEach(a => {
    a.addEventListener('pointermove', e => showTip(charTip(s, a.dataset.e), e));
    a.addEventListener('pointerleave', hideTip);
  });

  /* Window 4's attention mask: recalled frames see only themselves; each shot sees itself, the other shot's first latent frame, and its routed recalled frames */
  (function mask() {
    const host = $('#ovMask', fig), blocks = [];
    W.refs.forEach((r, i) => blocks.push({ t: 'ref', i, u: 1 }));
    W.shots.forEach((k, j) => { blocks.push({ t: 'first', j, u: 1 }); blocks.push({ t: 'rest', j, u: 3 }); });
    let acc = 0; blocks.forEach(b => { b.o = acc; acc += b.u; });
    const c = 12, L = 46, Tp = 18, size = acc * c;
    const svg = sv('svg', { width: L + size, height: Tp + size, viewBox: `0 0 ${L + size} ${Tp + size}`, role: 'img', 'aria-label': 'Attention mask for Window 4' }, host);
    const allow = (q, k) => {
      if (q.t === 'ref') return k.t === 'ref' && k.i === q.i ? 'ref' : null;
      if (k.t === 'ref') return W.routing[q.j].includes(k.i) ? 'ref' : null;
      if (k.j === q.j) return 'self';
      return k.t === 'first' ? 'first' : null;
    };
    const FILL = { self: 'var(--grey-a)', first: 'var(--holo)', ref: 'var(--ours)' };
    blocks.forEach(q => blocks.forEach(k => { const a = allow(q, k); sv('rect', { x: L + k.o * c + 0.6, y: Tp + q.o * c + 0.6, width: k.u * c - 1.2, height: q.u * c - 1.2, rx: 1.5, fill: a ? FILL[a] : 'var(--raise)', 'fill-opacity': a ? (q.t === 'ref' ? 0.55 : 0.85) : 1 }, svg); }));
    const lab = (x, y, t, col, an) => T(svg, x, y, t, 'svg-text', `font-size:12px;fill:${col}`, an);
    lab(L - 6, Tp + W.refs.length * c / 2 + 4, `R1–R${W.refs.length}`, 'var(--ours)', 'end');
    lab(L + W.refs.length * c / 2, Tp - 6, 'R', 'var(--ours)', 'middle');
    W.shots.forEach((k, j) => { const o = W.refs.length + j * 4; lab(L - 6, Tp + (o + 2) * c + 4, `Shot ${k}`, 'var(--paper)', 'end'); lab(L + (o + 2) * c, Tp - 6, String(k), 'var(--paper)', 'middle'); });
  })();

  /* recalled frames on the RoPE time axis, as one reading shot sees them */
  let reader = A, place = () => {};
  const tokHost = $('#ovTok', fig);
  chart(tokHost, Wd => {
    const H = 74, lab = Wd < 420 ? 58 : 70, n = 26, sw = (Wd - lab - 4) / n, sz = Math.max(7, Math.min(14, sw - 3));
    const X = t => lab + (t + 15) * sw + (sw - sz) / 2;
    const yR = 14, yS = 38, yA = 62;
    const svg = sv('svg', { width: Wd, height: H, viewBox: `0 0 ${Wd} ${H}`, role: 'img', 'aria-label': 'Recalled frames placed on the time axis before the shot that reads them' }, tokHost);
    const pat = sv('pattern', { id: 'ovNoise', width: 4, height: 4, patternUnits: 'userSpaceOnUse' }, sv('defs', {}, svg));
    [[0, 0], [2, 1], [1, 3], [3, 2]].forEach(([x, y]) => sv('rect', { x, y, width: 1, height: 1, fill: '#fff', 'fill-opacity': 0.85 }, pat));
    T(svg, 0, yR + sz / 2 + 4, 'recalled', 'svg-text', 'fill:var(--ours)');
    T(svg, 0, yS + sz / 2 + 4, `Shots ${A}, ${B}`, 'svg-text', 'fill:var(--paper)');
    sv('path', { d: `M${X(-15)} ${yA} H${X(10)}`, stroke: 'var(--rule)' }, svg);
    [-15, -10, -5, 0, 5].forEach(t => { sv('path', { d: `M${X(t) + sz / 2} ${yA - 3} v6`, stroke: 'var(--rule)' }, svg); T(svg, X(t) + sz / 2, yA + 12, String(t).replace('-', '−'), 'svg-text', '', 'middle'); });
    const shots = W.shots.map((k, j) => {
      const g = sv('g', { class: 'tk-s', 'data-k': k, style: 'cursor:pointer' }, svg);
      for (let q = 0; q < 5; q++) {
        const x = X(j * 5 + q);
        sv('rect', { x, y: yS, width: sz, height: sz, rx: 2.5, fill: q === 0 ? 'var(--holo)' : 'var(--grey-b)' }, g);
        sv('rect', { x, y: yS, width: sz, height: sz, rx: 2.5, fill: 'url(#ovNoise)', class: 'tk-n' }, g);
      }
      g.addEventListener('click', () => setReader(k));
      return g;
    });
    const refs = W.refs.map((r, i) => {
      const g = sv('g', { class: 'tk-r' }, svg);
      sv('rect', { x: 0, y: yR, width: sz, height: sz, rx: 2.5, fill: 'var(--ours)' }, g);
      T(g, sz / 2, yR - 3, `R${i + 1}`, 'svg-text', 'font-size:11px;fill:var(--ours);font-weight:600', 'middle');
      return g;
    });
    place = () => {
      const j = W.shots.indexOf(reader), reads = W.routing[j], p = j * 5;
      refs.forEach((g, i) => { const m = reads.indexOf(i); const t = m < 0 ? -15 : p - 5 * (reads.length - m); g.style.transform = `translate(${X(t)}px,0)`; g.style.opacity = m < 0 ? 0 : 1; });
      shots.forEach(g => { g.style.opacity = +g.dataset.k === reader ? 1 : 0.35; });
    };
    place();
    requestAnimationFrame(drawLinks);
  });
  const readerTabs = $('#ovReader', fig);
  function setReader(k) { reader = k; setTab(readerTabs, $(`[data-k="${k}"]`, readerTabs)); place(); }
  $$('button', readerTabs).forEach(b => b.addEventListener('click', () => setReader(+b.dataset.k)));

  /* connectors between areas, drawn while the map is laid out side by side */
  const map = $('#ovMap', fig), links = $('.ov-links', fig);
  const rel = (el, r0) => { const r = el.getBoundingClientRect(); return { l: r.left - r0.left, r: r.right - r0.left, t: r.top - r0.top, b: r.bottom - r0.top, cx: r.left - r0.left + r.width / 2, cy: r.top - r0.top + r.height / 2 }; };
  function drawLinks() {
    links.innerHTML = '';
    if (getComputedStyle(links).display === 'none') return;
    const r0 = map.getBoundingClientRect();
    if (!r0.width) return;
    links.setAttribute('viewBox', `0 0 ${r0.width.toFixed(1)} ${r0.height.toFixed(1)}`);
    const g = st => sv('g', { 'data-st': st }, links);
    const head = (e, x, y, dir) => { const d = dir === 'down' ? `M${x - 4} ${y - 6} L${x} ${y} L${x + 4} ${y - 6}` : `M${x - 6} ${y - 4} L${x} ${y} L${x - 6} ${y + 4}`; sv('path', { d, class: 'hd' }, e); };
    const side = (a, b, st) => {
      const x1 = a.r + 2, y1 = a.cy, x2 = b.l - 3, y2 = b.cy, mx = (x1 + x2) / 2;
      const e = g(st);
      sv('path', { d: `M${x1} ${y1} C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}` }, e); head(e, x2, y2, 'right');
    };
    const vert = (x1, y1, x2, y2, st) => { const e = g(st); const my = (y1 + y2) / 2; sv('path', { d: `M${x1} ${y1} C${x1} ${my} ${x2} ${my} ${x2} ${y2}` }, e); head(e, x2, y2, 'down'); };
    const board = rel($('#ovBoard', fig), r0), comp = rel($('#ovComp', fig), r0), prompts = rel($('#ovPrompts', fig), r0), t5 = rel($('#ovT5', fig), r0);
    const recall = rel($('#ovRecall', fig), r0), enc = rel($('#ovEnc', fig), r0), tok = rel($('#ovTok', fig), r0), dit = rel($('#ovDit', fig), r0), cross = rel($('#ovCross', fig), r0), dec = rel($('#ovDec', fig), r0);
    side(board, recall, '4');
    side(comp, prompts, '3');
    side(t5, cross, '6');
    side(enc, { l: tok.l - 4, cy: tok.cy }, '5');
    const recs = $$('.hs.rec', fig).map(h => rel(h, r0)), hist = rel($('#ovHist', fig), r0), busY = hist.b + 8;
    const bus = g('4'), xs = [recall.cx, ...recs.map(a => a.cx)];
    recs.forEach(a => sv('path', { d: `M${a.cx} ${a.b + 3} V${busY}` }, bus));
    sv('path', { d: `M${Math.min(...xs)} ${busY} H${Math.max(...xs)} M${recall.cx} ${busY} V${recall.t - 3}` }, bus);
    head(bus, recall.cx, recall.t - 3, 'down');
    vert(dit.cx - 4, tok.b + 6, dit.cx - 4, dit.t - 3, '5');
    const decPill = rel($('#ovDec .pill', fig), r0);
    vert(decPill.cx, dit.b + 3, decPill.cx, decPill.t - 3, '7');
    applyStage();
  }
  new ResizeObserver(() => requestAnimationFrame(drawLinks)).observe(map);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => requestAnimationFrame(drawLinks));

  /* stages outline what they touch; until decoding, the window being rendered is still noise */
  const tabs = $('#ovTabs', fig), tabBtns = $$('button', tabs), cap = $('#ovCap', fig), playBtn = $('#ovPlay', fig), navBtns = $$('.ov-ctl .ibtn', fig);
  let stage = 0, touring = false, timers = [];
  function applyStage() { $$('[data-st]', fig).forEach(el => el.classList.toggle('on', stage > 0 && el.dataset.st.split(',').includes(String(stage)))); }
  function setStage(n) {
    stage = n;
    fig.dataset.stage = n;
    applyStage();
    setTab(tabs, n ? tabBtns[n - 1] : null);
    cap.innerHTML = n ? `<b>${n}. ${STAGES[n - 1][0]}.</b> ${STAGES[n - 1][1]}` : REST;
    fig.classList.toggle('noisy', n >= 1 && n <= 6);
    if (n === 6 && !RM) $$('.tk-n', fig).forEach((r, i) => r.animate([{ opacity: 0.9 }, { opacity: 0.35 }, { opacity: 0.85 }, { opacity: 0.25 }, { opacity: 0.75 }], { duration: 2600, delay: (i % 5) * 60, easing: 'ease-in-out' }));
    if (!RM) { cap.classList.remove('swap'); void cap.offsetWidth; cap.classList.add('swap'); }
  }
  const setPlay = () => {
    playBtn.innerHTML = touring
      ? '<svg viewBox="0 0 14 14"><path d="M4 2.5v9M10 2.5v9" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>Pause'
      : '<svg viewBox="0 0 14 14"><path d="M3.5 2v10l8.5-5z" fill="currentColor"/></svg>Play';
    playBtn.setAttribute('aria-label', touring ? 'Pause the pass' : 'Play the pass');
  };
  function stopTour() { timers.forEach(clearTimeout); timers = []; touring = false; setPlay(); }
  const DWELL = 3300;
  function tour() {
    stopTour();
    touring = true; setPlay();
    for (let n = 1; n <= 7; n++) timers.push(setTimeout(() => setStage(n), (n - 1) * DWELL));
    timers.push(setTimeout(() => setReader(B), 4 * DWELL + 1600));
    timers.push(setTimeout(() => { touring = false; setPlay(); setReader(A); setStage(0); }, 7 * DWELL + 300));
  }
  tabBtns.forEach(b => b.addEventListener('click', () => { stopTour(); const n = +b.dataset.n; setStage(n === stage ? 0 : n); }));
  navBtns.forEach(b => b.addEventListener('click', () => { stopTour(); const d = +b.dataset.d; setStage(stage === 0 ? (d > 0 ? 1 : 7) : Math.max(1, Math.min(7, stage + d))); }));
  playBtn.addEventListener('click', () => { if (touring) stopTour(); else tour(); });
  fig.addEventListener('keydown', e => { if ((e.key === 'ArrowRight' || e.key === 'ArrowLeft') && e.target.closest('.ov-ctl')) { e.preventDefault(); navBtns[e.key === 'ArrowRight' ? 1 : 0].click(); } });
  setPlay();
  setStage(0);
  onReveal(fig, () => { if (!RM) timers.push(setTimeout(tour, 500)); });
});

/* =============== Benchmark: how CaT-MBench is built and judged (paper panels c and d) =============== */
run('benchFig', function benchFig() {
  const fig = $('#benchFig');
  if (!fig) return;
  const AD = [
    ['Wan2.1 and Wan2.2, per shot', 'one prompt per shot'],
    ['MultiShotMaster', 'story caption and shot captions'],
    ['HoloCine', 'global caption and shot captions'],
    ['StoryMem', 'shot prompt and keyframe memory'],
    ['StoryDiffusion', 'keyframes, then Wan2.2-I2V'],
    ['IC-LoRA', 'three-panel grid, then Wan2.2-I2V'],
  ];
  const Q = [
    ['Character consistency', 'Is C1 at S3 the same person as at S1?', 0],
    ['Object consistency', 'Is O1 at S2 the same object as at S1?'],
    ['Scene consistency', 'Are S10 and S1 in the same place (L1)?'],
    ['State establishment', 'Is O1 folded by the end of S1?'],
    ['State persistence', 'Is O1 still folded in S2?'],
    ['Cut execution', 'S1 → S2: a cut to a new camera setup?', 1],
  ];
  const FR = [
    ['s01_t00', 'S1 0:00'], ['s01_t02', 'S1 0:02'], ['s01_t04', 'S1 0:04'], ['s02_t06', 'S2 0:06'], ['s03_t09', 'S3 0:09'], ['s03_t10', 'S3 0:10'],
    ['s03_t11', 'S3 0:11'], ['s04_t13', 'S4 0:13'], ['s08_t45', 'S8 0:45'], ['s08_t46', 'S8 0:46'], ['s08_t47', 'S8 0:47'], ['s09_t50', 'S9 0:50'],
  ];
  const EX = [
    { q: 'Is C1 at 0:09–0:12 the same person as at 0:00–0:04?', hi: [[0, 1, 2], [4, 5, 6]], steps: [['Judge', 'No: the face differs, though hair, build and clothing match.', 'bad'], ['Cut gate', 'New camera setup confirmed (frame correlation −0.08).', 'ok'], ['Result', 'The gate passes, so the character check counts, and fails.', 'bad']] },
    { q: 'S1 → S2: does the planned cut open a new camera setup?', hi: [[2], [3]], steps: [['TransNetV2', 'Cut detected within 0.5 s.', 'ok'], ['Frame correlation', '0.30', ''], ['Judge', 'New camera setup confirmed.', 'ok'], ['Result', 'Cut executed; the consistency checks across it count.', 'ok']] },
  ];
  fig.innerHTML = `<div class="fig-head"><h3>How CaT-MBench is built and judged</h3><button type="button" class="tbtn" id="benchReplay"></button></div>
    <div class="bf-grid">
      <div class="bf-col">
        <span class="label">Stories, from films no training block uses</span>
        <div class="bf-big"><b>122</b><span>stories of 6–12 shots, up to 60 s each</span></div>
        <div class="mbar" style="margin-top:10px"><i style="flex:30 1 0;background:var(--grey-c)"></i><i style="flex:92 1 0;background:var(--grey-a)"></i></div>
        <div class="mkey"><span><i style="background:var(--grey-c)"></i>30 development</span><span><i style="background:var(--grey-a)"></i>92 test</span></div>
        <div class="bf-req"><span class="label">A shot-list request, with registry identifiers</span><b>C1</b> dark-haired man, grey suit. <b>O1</b> piece of paper. <b>L1</b> wooden kitchen.<br><b>S01</b> Medium shot. C1 writes on the paper, folds it, and hands it to C2.</div>
        <div class="bf-in"><span class="label">Native input, one adapter per baseline</span>${AD.map(([m, f]) => `<p><b>${m}</b><span>${f}</span></p>`).join('')}</div>
      </div>
      <div class="bf-col">
        <span class="label">Question bank, written by GPT-6.1-sol from each shot list</span>
        <div class="bf-q">${Q.map(([t, q, ex], i) => `<button type="button" data-i="${i}"${ex != null ? ` data-ex="${ex}" aria-label="${t}: show it being judged"` : ' tabindex="-1"'}><b>${t}</b><span>${esc(q)}</span></button>`).join('')}</div>
        <div class="bf-nums"><div><b>1,361</b><span>binary questions: 349 development, 1,012 test</span></div><div><b>995</b><span>planned cuts: 255 development, 740 test; 674 of the 740 change place, shot size, angle or cast</span></div></div>
      </div>
    </div>
    <div class="bf-judge">
      <div class="bf-jhead"><span class="label">Judging: Gemini 3.8 Flash reads the generated video at 2 fps, then the cut gate decides what counts</span><div class="tabs" id="jEx" role="group" aria-label="Example"><button type="button" data-e="0" class="on" aria-pressed="true">Character check</button><button type="button" data-e="1" aria-pressed="false">Cut check</button></div></div>
      <p class="bf-rule"><span><b>Visible evidence only</b></span><span>yes or no, with a time and a reason</span><span>unclear counts as no</span></p>
      <div class="jstrip" id="jStrip">${FR.map(([f, l], i) => `<div class="jf" data-i="${i}" style="background-image:url(${MEDIA}f/judge/${f}.jpg)"><span>${l}</span></div>`).join('')}</div>
      <div class="jflow" id="jFlow" aria-live="polite"></div>
    </div>`;
  const ex = $('#jEx', fig), flow = $('#jFlow', fig), strip = $('#jStrip', fig), qBtns = $$('.bf-q button', fig), replay = $('#benchReplay', fig);
  let ei = 0, stepT = [], tourT = [], touring = false;
  function setE(i) {
    ei = i;
    stepT.forEach(clearTimeout); stepT = [];
    setTab(ex, $$('button', ex)[i]);
    qBtns.forEach(b => b.classList.toggle('on', b.dataset.ex === String(i)));
    const E = EX[i];
    $$('.jf', strip).forEach(f => { const k = +f.dataset.i; f.classList.toggle('a', E.hi[0].includes(k)); f.classList.toggle('b', E.hi[1].includes(k)); });
    flow.innerHTML = `<p class="jq">${esc(E.q)}</p>` + E.steps.map(([k, v, c]) => `<div class="jstep ${c}"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('');
    $$('.jstep', flow).forEach((st, n) => { if (RM) st.classList.add('on'); else stepT.push(setTimeout(() => st.classList.add('on'), 250 + n * 450)); });
  }
  const setReplay = () => { replay.innerHTML = touring ? 'Skip' : '<svg viewBox="0 0 14 14"><path d="M2.5 7a4.5 4.5 0 1 0 1.6-3.45M2.5 1.8v2.6h2.6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>Replay'; };
  function finish() { tourT.forEach(clearTimeout); tourT = []; touring = false; fig.classList.remove('touring'); qBtns.forEach(b => b.classList.add('lit')); setReplay(); }
  /* once on first view: the question types light in turn, then each worked example is judged step by step */
  function tour() {
    finish();
    touring = true; setReplay();
    fig.classList.add('touring');
    qBtns.forEach(b => b.classList.remove('lit'));
    qBtns.forEach((b, i) => tourT.push(setTimeout(() => b.classList.add('lit'), 300 + i * 380)));
    tourT.push(setTimeout(() => setE(0), 2700));
    tourT.push(setTimeout(() => setE(1), 7000));
    tourT.push(setTimeout(finish, 9600));
  }
  $$('button', ex).forEach(b => b.addEventListener('click', () => { finish(); setE(+b.dataset.e); }));
  qBtns.forEach(b => { if (b.dataset.ex != null) b.addEventListener('click', () => { finish(); setE(+b.dataset.ex); }); });
  replay.addEventListener('click', () => { if (touring) finish(); else tour(); });
  setE(0); finish();
  onReveal(fig, () => { if (!RM) tour(); });
});

/* =============== Data: from films to annotated blocks (paper panels a and b), one figure =============== */
run('dataFig', function dataFig() {
  const fig = $('#dataFig'), pipe = $('#pipe'), host = $('#blk');
  if (!fig || !pipe || !host) return;
  /* pipeline: each step carries its own number and small chart */
  const SRC = [['CinePile', 65.74, 'var(--grey-a)'], ['CineDance', 24.69, 'var(--grey-b)'], ['MovieChat', 9.57, 'var(--grey-c)']];
  const HB = [15459, 12498, 8388, 5959, 4301, 4242, 5642, 4826, 4864, 4136, 3100, 2046], mx = Math.max(...HB);
  const step = (n, title, big, desc, mini) => `<li class="pp" data-i="${n - 1}"><span class="pp-n">${n}</span><b class="pp-t">${title}</b>${big ? `<strong>${big}</strong>` : ''}<span class="pp-d">${desc}</span><div class="mini">${mini}</div></li>`;
  pipe.innerHTML = [
    step(1, 'Cinematic videos', '19,535', 'source videos', `<div class="mbar">${SRC.map(([nm, v, c]) => `<i style="flex:${v} 1 0;background:${c}" title="${nm} ${v.toFixed(1)}%"></i>`).join('')}</div><div class="mkey">${SRC.map(([nm, v, c]) => `<span><i style="background:${c}"></i>${nm} ${v.toFixed(1)}%</span>`).join('')}</div>`),
    step(2, 'Shot detection', '', 'TransNetV2 finds the cuts; quality checks follow', `<div class="pp-cuts">${[1, 2, 3].map(i => `<i style="background-image:url(${MEDIA}f/ground/s${i}-rgb.jpg)"></i>`).join('')}<svg style="left:calc(33.333% - 2px)" viewBox="0 0 24 24"><use href="#i-scissor"/></svg><svg style="left:calc(66.667% + 2px)" viewBox="0 0 24 24"><use href="#i-scissor"/></svg></div>`),
    step(3, 'Scene grouping', '75,461', 'blocks of up to 12 shots, 358,302 shots and 332.8 hours in all', `<div class="mhist">${HB.map((v, i) => `<i class="${i === 0 ? 'mode' : ''}" data-i="${i}" style="height:${(v / mx * 100).toFixed(1)}%"></i>`).join('')}<span class="med" style="left:${(3.5 / 12 * 100).toFixed(2)}%"><span>median 4</span></span></div><div class="mhist-x"><span>1 shot</span><span>12 shots</span></div>`),
    step(4, 'Shot-first filtering', '+28.1%', 'aesthetic, technical and annotation checks drop single shots, not whole windows', `<div class="myield"><div><p><span>window-first</span><em>227k</em></p><i style="width:${(226648 / 290267 * 100).toFixed(1)}%"></i></div><div><p><span>shot-first</span><em>290k</em></p><i style="width:100%"></i></div></div><div class="mkey"><span>usable shots kept</span></div>`),
    step(5, 'Dataset split', '66,573<small>train</small>', 'blocks, and 916 held out; split by film, so benchmark films never appear in training', `<div class="mbar split"><i style="flex:66573 1 0;background:var(--grey-b)"></i><i style="flex:916 1 0;min-width:4px"></i></div><div class="mkey"><span><i style="background:var(--grey-b)"></i>train 66,573</span><span><i style="background:var(--ours)"></i>held out 916</span></div>`),
  ].join('');
  $$('.mhist i', pipe).forEach(b => {
    const i = +b.dataset.i;
    b.addEventListener('pointermove', e => showTip(`<b>${i + 1} shot${i ? 's' : ''} per block</b><table><tr><td>blocks</td><td>${HB[i].toLocaleString('en-US')}</td></tr><tr><td>share</td><td>${(HB[i] / 75461 * 100).toFixed(1)}%</td></tr></table>`, e));
    b.addEventListener('pointerleave', hideTip);
  });

  /* the annotated block: registry, three shots, and the grounding layers */
  const TRACK = [
    { shot: 1, entities: ['C1', 'O1'], frames: [[[447, 37, 330, 443], [0, 413, 456, 67]], [[445, 34, 327, 446], [0, 407, 453, 73]], [[437, 38, 325, 442], [0, 404, 444, 69]], [[460, 43, 319, 437], [0, 92, 652, 375]], [[459, 39, 327, 441], [0, 342, 468, 119]], [[440, 42, 333, 438], [0, 339, 447, 117]], [[394, 24, 351, 456], [0, 361, 485, 93]], [[364, 0, 379, 480], [0, 352, 470, 104]], [[379, 0, 382, 480], [0, 369, 471, 93]], [[401, 0, 391, 480], [0, 375, 488, 90]], [[409, 1, 406, 479], [7, 395, 492, 85]], [[385, 13, 430, 467], [47, 400, 690, 80]], [[342, 8, 464, 472], [88, 443, 386, 37]], [[372, 6, 448, 474], [126, 456, 349, 24]]] },
    { shot: 2, entities: ['C2', 'O2', 'O3'], frames: [[[145, 71, 303, 409], [250, 0, 425, 480], [408, 80, 133, 393]], [[123, 61, 298, 419], [249, 0, 487, 480], [415, 74, 142, 399]], [[104, 61, 287, 419], [249, 0, 510, 480], [428, 71, 150, 409]], [[96, 54, 305, 426], [242, 0, 517, 480], [424, 66, 171, 414]], [[109, 50, 325, 430], [235, 0, 535, 480], [439, 57, 178, 423]], [[140, 26, 367, 454], [238, 0, 562, 480], [424, 50, 216, 430]], [[162, 5, 399, 475], [206, 0, 626, 480], [426, 65, 226, 408]], [[155, 2, 420, 478], [169, 0, 663, 480], [420, 36, 233, 290]], [[118, 0, 432, 480], [135, 0, 697, 480], [397, 33, 255, 447]], [[66, 0, 445, 480], [113, 0, 719, 480], [386, 29, 267, 451]], [[29, 0, 462, 480], [645, 0, 187, 480], [393, 29, 248, 451]], [[0, 0, 468, 480], [679, 0, 153, 480], [403, 27, 282, 453]], [[0, 0, 444, 480], [466, 0, 366, 480], [407, 28, 315, 452]], [[0, 0, 428, 480], [339, 0, 493, 480], [402, 31, 348, 449]], [[0, 0, 435, 480], [73, 0, 759, 480], [398, 33, 388, 447]], [[0, 0, 454, 480], [73, 0, 759, 480], [399, 33, 416, 447]], [[0, 0, 471, 480], null, [395, 32, 413, 448]], [[0, 0, 470, 480], null, [391, 36, 410, 444]], [[0, 0, 460, 480], null, [388, 42, 413, 438]], [[0, 0, 471, 480], null, [390, 46, 411, 434]]] },
    { shot: 3, entities: ['C1'], frames: [[[315, 36, 517, 444]], [[361, 21, 471, 459]], [[368, 26, 464, 454]], [[333, 54, 499, 426]], [[264, 45, 532, 435]], [[251, 35, 540, 445]], [[284, 38, 540, 442]], [[342, 34, 490, 446]], [[397, 12, 435, 468]], [[405, 13, 427, 467]], [[363, 29, 469, 451]], [[296, 2, 536, 478]], [[283, 8, 549, 472]], [[315, 17, 517, 463]], [[389, 10, 443, 470]], [[430, 3, 402, 477]], [[424, 12, 408, 468]], [[380, 30, 452, 450]], [[310, 21, 522, 459]], [[284, 15, 548, 465]]] },
  ];
  const COL = { C1: '#29B6AA', O1: '#5B8FF9', C2: '#FCB84C', O2: '#B084D8', O3: '#EE7D92' };
  const REG = [
    ['C1', 'an older man with white hair; black suit, white shirt, dark red tie'],
    ['C2', 'a man with dark hair and a lean build; dark grey polo shirt'],
    ['O1', 'a pile of newspapers and magazines'],
    ['O2', 'a metal mesh cage or display case'],
    ['O3', 'a black armored suit and cowl inside the cage'],
  ];
  const SHOT = [
    'Medium shot, static. C1 stands near a pile of newspapers and looks down, then looks up.',
    'Medium shot, tracking. C2 walks towards a metal mesh cage containing a black armored suit.',
    'Medium shot, static. C1 speaks while looking at C2.',
  ];
  const LAYERS = [
    ['rgb', 'Frame', 'three consecutive shots of one scene block at the film’s own frame rate.'],
    ['boxes', 'Boxes', 'Qwen3.8-27B, then tracked; each box keeps its registry identifier, so C1 is the same man in shots 1 and 3.'],
    ['masks', 'Masks', 'SAM 3.1 tracks every registered entity through its shot.'],
    ['depth', 'Depth', 'VGGT-Ω relative depth, lighter is farther, kept on every fourth frame.'],
  ];
  host.innerHTML = `<div class="blk-head"><div data-p="6"><span class="pp-n">6</span><b>Annotate</b>one pass of Qwen3.7-Flash per block</div><div data-p="7"><span class="pp-n">7</span><b>Ground</b><div class="tabs" id="gSeg" role="group" aria-label="Grounding layer">${LAYERS.map(([k, n]) => `<button type="button" data-l="${k}"${k === 'boxes' ? ' class="on" aria-pressed="true"' : ' aria-pressed="false"'}>${n}</button>`).join('')}</div></div></div>
    <span class="label" style="margin-bottom:6px">Entity registry, written with every shot line and state change</span>
    <div class="blk-reg">${REG.map(([id, d]) => `<p data-e="${id}"><i style="background:${COL[id]}"></i><b>${id}</b><span>${esc(d)}</span></p>`).join('')}</div>
    <div class="blk-shots">${TRACK.map((t, i) => `<div class="blk-shot">
      <div class="gfr"><div class="gvid"><video muted loop playsinline preload="none" poster="${MEDIA}v/ground/s${i + 1}-poster.jpg" src="${MEDIA}v/ground/s${i + 1}.mp4" aria-label="Shot ${i + 1} of the annotated block"></video></div>
        <div class="gbx">${t.entities.map(n => `<div class="gb" data-e="${n}" style="border-color:${COL[n]}"><span style="background:${COL[n]}">${n}</span></div>`).join('')}</div>
        <span class="gs">Shot ${i + 1}</span></div>
      <p><b>Shot ${i + 1}</b>${esc(SHOT[i])}</p>${i === 0 ? '<p class="chg"><b>C1 posture</b>looking down → looking up and forward</p>' : ''}
    </div>`).join('')}</div>
    <p class="cap" id="gCap"></p>`;

  const clips = $$('.blk-shot', host).map((el, i) => ({ v: $('video', el), t: TRACK[i], gs: $$('.gb', el), j: -1 }));
  /* masks and depth exist on every fourth 16 fps frame; boxes follow the same index by time */
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
    $('#gCap', host).innerHTML = `<b>${L[1]}</b>: ${L[2]}`;
  }
  $$('.blk-reg p', host).forEach(p => {
    p.addEventListener('pointerenter', () => {
      if (host.dataset.layer !== 'boxes') setL('boxes');
      host.classList.add('focus');
      $$('.gb', host).forEach(g => g.classList.toggle('hl', g.dataset.e === p.dataset.e));
    });
    p.addEventListener('pointerleave', () => { host.classList.remove('focus'); $$('.gb', host).forEach(g => g.classList.remove('hl')); });
  });
  watch(host, v => {
    clips.forEach(c => { if (v) { c.v.preload = 'auto'; c.v.play().catch(() => {}); } else c.v.pause(); });
    cancelAnimationFrame(raf);
    if (v) tick();
  }, 0.15);

  /* once on first view: the pipeline lights step by step, the block is annotated, then grounded layer by layer */
  const steps = $$('.pp', pipe), heads = $$('.blk-head > div', host), replay = $('#dataReplay');
  let timers = [], touring = false;
  const growMini = el => { if (RM) return; grow($$('.mbar i, .mhist i, .myield i', el), $$('.mhist i', el).length ? 'bottom' : 'left', 0, 40); };
  const setReplay = () => { replay.innerHTML = touring ? 'Skip' : '<svg viewBox="0 0 14 14"><path d="M2.5 7a4.5 4.5 0 1 0 1.6-3.45M2.5 1.8v2.6h2.6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>Replay'; };
  function finish() { timers.forEach(clearTimeout); timers = []; touring = false; fig.classList.remove('touring'); [...steps, ...heads].forEach(x => x.classList.add('lit')); setReplay(); }
  function tour() {
    finish();
    touring = true; setReplay();
    fig.classList.add('touring');
    [...steps, ...heads].forEach(x => x.classList.remove('lit'));
    setL('rgb');
    const at = (ms, fn) => timers.push(setTimeout(fn, ms));
    steps.forEach((st, i) => at(300 + i * 700, () => { st.classList.add('lit'); growMini(st); }));
    const t6 = 300 + steps.length * 700;
    at(t6, () => { heads[0].classList.add('lit'); if (!RM) $$('.blk-reg p, .blk-shot p', host).forEach((p, i) => p.animate([{ opacity: 0, transform: 'translateY(4px)' }, { opacity: 1, transform: 'none' }], { duration: 380, delay: i * 90, easing: EASE, fill: 'backwards' })); });
    at(t6 + 1200, () => { heads[1].classList.add('lit'); setL('boxes'); });
    at(t6 + 2700, () => setL('masks'));
    at(t6 + 4200, () => setL('depth'));
    at(t6 + 5700, () => { setL('boxes'); finish(); });
  }
  $$('button', seg).forEach(b => b.addEventListener('click', () => { if (touring) finish(); setL(b.dataset.l); }));
  replay.addEventListener('click', () => { if (touring) { finish(); setL('boxes'); } else tour(); });
  setL('boxes'); finish();
  onReveal(fig, () => { if (!RM) tour(); });
});

/* =============== Data: registered entities per block =============== */
run('entStack', function entStack() {
  const host = $('#entStack');
  if (!host) return;
  const ENT = { People: [8.21, 18.82, 33.10, 39.87], Objects: [5.98, 16.44, 17.26, 60.31], Places: [0.27, 81.90, 15.13, 2.70] };
  const COLS = [['var(--grey-d)', 'var(--paper)'], ['var(--grey-b)', 'var(--ink)'], ['var(--grey-a)', 'var(--ink)'], ['var(--ours)', 'var(--ink)']];
  host.innerHTML = Object.entries(ENT).map(([k, arr]) => `<span>${k}</span><div class="stack">${arr.map((v, i) => `<div data-i="${i}" style="flex:${v} 1 0;background:${COLS[i][0]};color:${COLS[i][1]}">${v >= 9 ? Math.round(v) + '%' : ''}</div>`).join('')}</div>`).join('');
  $$('.stack', host).forEach((st, r) => $$('div', st).forEach((d, i) => {
    const k = Object.keys(ENT)[r], v = ENT[k][i];
    d.addEventListener('pointermove', e => showTip(`<b>${k}</b>: ${['none', 'one', 'two', 'three or more'][i]} per block, ${v.toFixed(1)}%`, e));
    d.addEventListener('pointerleave', hideTip);
  }));
  onReveal(host.closest('.fig'), () => { if (!RM) $$('.stack', host).forEach((st, r) => grow($$('div', st), 'left', r * 120, 70)); });
});

/* =============== Data: composition sunburst (facet bars on a phone) =============== */
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
      ['Evaluation dimension', 'scored questions, development and test', 1361, [['Character', 65.17, 887], ['Object', 6.32, 86], ['Scene', 18.81, 256], ['State', 9.70, 132]]],
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
      <div class="mbar" style="height:12px">${cats.map(([lab, share, count, what]) => `<i style="flex:${share} 1 0;background:${lab ? rgb(catColor(base, cats, lab)) : '#4c5261'}" title="${esc(lab || what)} ${share.toFixed(1)}%"></i>`).join('')}</div>
      <div class="mkey">${cats.filter(c => c[0]).map(([lab, share]) => `<span><i style="background:${rgb(catColor(base, cats, lab))}"></i>${esc(lab)} ${Math.round(share)}%</span>`).join('')}</div></div>`).join('')}</div>`;
    host.innerHTML = `<div class="sbars">${block('Dataset', '288,083 labelled training shots', half('data'))}${block('Benchmark', '1,361 CaT-MBench questions, 122 stories', half('bench'))}</div>`;
    return () => grow($$('.mbar i', host), 'left', 0, 25);
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
    const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Composition sunburst: training data in the upper half, CaT-MBench in the lower half' }, host);
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
        const el = sv('path', { d: wedge(R.o0, R.o1, b0, b1), fill: lab ? rgb(color) : '#4c5261', stroke: 'var(--ink)', 'stroke-width': 1.4, class: 'sb-seg' }, gW);
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
      T(gW, x, y + 4.5, `${Math.round(sg.share)}%`, '', `font:500 13.5px var(--body);font-variant-numeric:tabular-nums;fill:${L < 0.52 ? '#fff' : '#121520'};pointer-events:none`, 'middle').setAttribute('transform', `rotate(${rot} ${x} ${y})`);
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
      const t = sv('text', { x: xt, y: y0, 'text-anchor': anchor, style: 'font:500 14px var(--body);fill:var(--paper)' }, g);
      ls.forEach((w, i) => { const sp = sv('tspan', { x: xt, dy: i ? 16 : 0 }, t); sp.textContent = w; });
      l.lg = g;
    });
    /* centre: no plate behind it, just the two halves named */
    const cg = sv('g', {}, svg);
    sv('circle', { cx, cy, r: R.c - 4, fill: 'var(--ink)', stroke: 'var(--rule)' }, cg);
    const top = sv('g', {}, cg), bot = sv('g', {}, cg);
    sv('path', { d: `M${cx - 48} ${cy - 13}l6 -9l6 9h-4v8h-4v-8z`, fill: 'var(--cut)' }, top);
    T(top, cx - 32, cy - 10, 'Dataset', '', 'font:600 18px var(--body);fill:var(--cut);letter-spacing:-.02em');
    sv('path', { d: `M${cx - 56} ${cy + 13}l6 9l6 -9h-4v-8h-4v8z`, fill: 'var(--ours)' }, bot);
    T(bot, cx - 40, cy + 25, 'Benchmark', '', 'font:600 17px var(--body);fill:var(--ours);letter-spacing:-.02em');
    const info = sv('g', { style: 'opacity:0;transition:opacity .22s' }, cg);
    const iCat = T(info, cx, cy - 26, '', '', 'font:500 10px var(--body);fill:var(--mute)', 'middle');
    const iVal = T(info, cx, cy + 5, '', '', 'font:600 26px var(--body);fill:var(--paper);letter-spacing:-.02em', 'middle');
    const iLab = T(info, cx, cy + 23, '', '', 'font:500 11.5px var(--body);fill:var(--paper)', 'middle');
    const iCnt = T(info, cx, cy + 38, '', '', 'font:500 10px var(--body);font-variant-numeric:tabular-nums;fill:var(--mute)', 'middle');
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
      sg.el.addEventListener('pointerenter', () => focus([sg.el, sg.fac.el], sg.fac.name.toLowerCase(), `${sg.share.toFixed(1)}%`, sg.lab || sg.what, `${sg.count.toLocaleString('en-US')} of ${sg.fac.total.toLocaleString('en-US')}`));
      sg.el.addEventListener('pointerleave', reset);
    });
    facetsAll.forEach(f => {
      f.el.addEventListener('pointerenter', () => focus([f.el, ...f.cats.map(c => c.el)], f.unit, f.total.toLocaleString('en-US'), f.name, `${f.cats.filter(c => c.lab).length} categories`));
      f.el.addEventListener('pointerleave', reset);
    });
    try { const b = svg.getBBox(); if (b.width > 0) svg.setAttribute('viewBox', `${b.x - 8} ${b.y - 8} ${b.width + 16} ${b.height + 16}`); } catch (e) {}
    return () => {
      const all = [...facetsAll.map(f => ({ el: f.el, d: f.d0 })), ...segs.map(s => ({ el: s.el, d: s.b0 }))];
      all.forEach(({ el, d }) => { el.style.transformBox = 'view-box'; el.style.transformOrigin = `${cx}px ${cy}px`; el.animate([{ opacity: 0, transform: 'scale(.9)' }, { opacity: 1, transform: 'none' }], { duration: 500, delay: d * 0.9, easing: EASE, fill: 'backwards' }); });
      labels.forEach(l => l.lg.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 400, delay: 200 + l.mid * 0.9, fill: 'backwards' }));
    };
  }
  chart(host, W => (W < 560 ? bars() : ring()));
});
})();
