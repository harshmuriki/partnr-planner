(() => {
  'use strict';
  const rows = JSON.parse(document.getElementById('vt-tree-data').textContent);
  const canvas = document.getElementById('vt-canvas');
  const detail = document.getElementById('vt-detail');
  const tree = document.getElementById('vt-tree');
  const current = document.getElementById('vt-current');
  const ns = 'http://www.w3.org/2000/svg';
  const styles = {
    start: ['#334155', '#f1f5f9', 'Start'], success: ['#15803d', '#f0fdf4', '✓ Solved'],
    failure: ['#b91c1c', '#fef2f2', '× Failed'], interrupted: ['#0369a1', '#f0f9ff', '↻ Observed / replanned'],
    pending: ['#64748b', '#fff', '○ Pending'], skipped: ['#94a3b8', '#f8fafc', '– Skipped'],
    running: ['#7c3aed', '#f5f3ff', 'In progress']
  };
  let svg, width = 0, height = 0, zoom = 1, selected = null;
  function element(tag, attrs, text) {
    const el = document.createElementNS(ns, tag);
    Object.entries(attrs || {}).forEach(([key, value]) => el.setAttribute(key, value));
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function paragraph(text) {
    const p = document.createElement('p'); p.textContent = text; detail.append(p);
  }
  function shortLabel(goal) {
    if (goal === 'start') return 'Start';
    const match = goal.match(/^([\w-]+)\((.*)\)$/);
    if (!match) return goal;
    const args = match[2].split(',').map(s => s.trim().replace(/_/g, ' '));
    const labels = {
      explore: () => `Explore ${args[0]}`, picked: () => `Pick up ${args[0]}`,
      on: () => `Place ${args[0]} on ${args[1]}`, in: () => `Place ${args[0]} in ${args[1]}`,
      'opened-drawer': () => `Open ${args[0]}`, 'closed-drawer': () => `Close ${args[0]}`,
      'opened-door': () => `Open ${args[0]}`, 'closed-door': () => `Close ${args[0]}`,
      opened: () => `Open ${args[0]}`, closed: () => `Close ${args[0]}`
    };
    return labels[match[1]] ? labels[match[1]]() : goal.replace(/_/g, ' ');
  }
  function select(node) {
    selected = node.id;
    canvas.querySelectorAll('.vt-node').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.id === selected)));
    detail.replaceChildren();
    const title = document.createElement('h2'); title.textContent = node.goal === 'start' ? 'Start' : node.goal; detail.append(title);
    paragraph((styles[node.status] || styles.pending)[2]);
    if (node.refs.length) paragraph(node.refs.join(' · '));
    // Reuse the current report's full details, including local images and prompts.
    const step = [...current.querySelectorAll('details.step[data-node]')].find(el => node.refs.includes(el.dataset.node));
    if (step) {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = 'Open in current view';
      button.onclick = () => { setView('current'); step.open = true; step.scrollIntoView({block: 'center'}); };
      detail.append(button);
      const body = step.querySelector('.step-expand');
      if (body) detail.append(body.cloneNode(true));
    } else {
      if (node.failure) paragraph(node.failure);
      const source = document.getElementById('exec-logs');
      const logs = source ? JSON.parse(source.textContent) : {};
      const pre = document.createElement('pre');
      pre.textContent = (logs[node.id] || 'See the current view for the recorded plan and execution details.').replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '');
      detail.append(pre);
    }
  }
  function scale(value) {
    zoom = Math.max(0.15, Math.min(2, value));
    if (svg) { svg.setAttribute('width', width * zoom); svg.setAttribute('height', height * zoom); }
  }
  function fit() {
    if (svg) {
      scale(Math.min(1, (canvas.clientWidth - 20) / width, (canvas.clientHeight - 20) / height));
      canvas.scrollTo(0, 0);
    }
  }
  function draw() {
    const showSkipped = document.getElementById('vt-skipped').checked;
    const byId = new Map(rows.map(n => [n.id, {...n, children: []}]));
    const root = rows.find(n => !n.parent);
    canvas.replaceChildren();
    if (!root || rows.length < 2) { canvas.textContent = 'No subgoals recorded yet.'; return; }
    byId.forEach(n => { if (byId.has(n.parent)) byId.get(n.parent).children.push(n); });
    let leaf = 0, maxDepth = 0;
    const visible = [];
    function layout(n, depth) {
      n.children = n.children.filter(c => showSkipped || c.status !== 'skipped');
      n.children.forEach(c => layout(c, depth + 1));
      n.x = n.children.length ? (n.children[0].x + n.children[n.children.length - 1].x) / 2 : leaf++ * 250 + 20;
      n.y = depth * 130 + 20; maxDepth = Math.max(maxDepth, depth); visible.push(n);
    }
    layout(byId.get(root.id), 0);
    width = Math.max(1, leaf) * 250 + 20; height = (maxDepth + 1) * 130 + 20;
    svg = element('svg', {viewBox: `0 0 ${width} ${height}`, role: 'group', 'aria-label': 'Subgoals and replanning branches'});
    // Edges first, so labels and nodes always remain readable.
    visible.forEach(n => n.children.forEach(c => {
      const x1 = n.x + 110, y1 = n.y + 82, x2 = c.x + 110, y2 = c.y;
      svg.append(element('path', {d: `M${x1},${y1} V${y1 + 24} H${x2} V${y2}`, fill: 'none', stroke: c.replan ? '#6366f1' : '#cbd5e1', 'stroke-width': 1.5, 'stroke-dasharray': c.replan ? '5 3' : 'none'}));
      if (c.replan) svg.append(element('text', {x: x2 + 6, y: y2 - 7, fill: '#4f46e5', 'font-size': 10}, `Replan · B${c.branch}`));
    }));
    visible.forEach(n => {
      const [stroke, fill, status] = styles[n.status] || styles.pending;
      const group = element('g', {class: 'vt-node', transform: `translate(${n.x},${n.y})`, tabindex: 0, role: 'button', 'aria-label': `${n.goal}: ${status}`, 'aria-pressed': String(n.id === selected), 'data-id': n.id});
      group.append(element('rect', {width: 220, height: 82, rx: 9, fill, stroke}));
      group.append(element('title', {}, n.goal));
      group.append(element('text', {x: 12, y: 19, 'font-size': 10, fill: stroke}, n.refs.length ? n.refs.join(' · ') : 'Initial state'));
      const label = shortLabel(n.goal);
      const lines = label.match(/.{1,29}(?:\s|$)|.{1,29}/g) || [label];
      lines.slice(0, 2).forEach((line, i) => group.append(element('text', {x: 12, y: 38 + i * 16, 'font-size': 12, fill: '#1e293b'}, line.trim() + (i === 1 && lines.length > 2 ? '…' : ''))));
      group.append(element('text', {x: 12, y: 72, 'font-size': 10, fill: stroke}, status));
      group.addEventListener('click', () => select(n));
      group.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(n); } });
      svg.append(group);
    });
    canvas.append(svg); scale(zoom);
  }
  function setView(view) {
    const isTree = view === 'tree'; current.hidden = isTree; tree.hidden = !isTree;
    document.querySelectorAll('[data-vt-view]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.vtView === view)));
    if (isTree && !svg) draw();
    try { localStorage.setItem('vlm-tamp-report-view', view); } catch (_) { /* file URLs may disable storage */ }
  }
  document.querySelectorAll('[data-vt-view]').forEach(b => b.onclick = () => setView(b.dataset.vtView));
  document.getElementById('vt-fit').onclick = fit;
  document.getElementById('vt-actual').onclick = () => scale(1);
  document.getElementById('vt-plus').onclick = () => scale(zoom * 1.2);
  document.getElementById('vt-minus').onclick = () => scale(zoom / 1.2);
  document.getElementById('vt-skipped').onchange = draw;
  try { if (localStorage.getItem('vlm-tamp-report-view') === 'tree') setView('tree'); } catch (_) {}
})();
