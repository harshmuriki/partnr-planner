// Embedded by render(): the exported dashboard remains a single offline HTML file.
(function () {
  const data = DATA.subtasks || {definitions: [], runs: []};
  if (!data.runs.length) {
    $('subtaskCoverage').textContent = 'Per-criterion results have not been extracted for this collection.';
    return;
  }
  for (const [id, field] of [['sTask','task'], ['sMemory','memory'], ['sUnc','uncertainty']]) {
    for (const value of [...new Set(data.runs.map(r => r[field]))].sort((a,b) => String(a).localeCompare(String(b), undefined, {numeric:true}))) {
      const option = document.createElement('option');
      option.value = value; option.textContent = field === 'task' ? `Task ${value}` : value;
      $(id).appendChild(option);
    }
  }
  const baselines = [
    {key:'vlm_tamp_images',label:'VLM-TAMP images',color:'#16a34a'},
    {key:'react_images',label:'ReAct images',color:'#dc2626'},
    {key:'tru_pomdp_no_images',label:'TRU-POMDP',color:'#eab308'}
  ];
  let cells = new Map(), selection = null;
  const cellKey = (criterion, condition) => `${criterion}:${condition}`;
  const counts = entries => entries.reduce((n, e) => {n[e.c.status]++; return n;}, {met:0, unmet:0, unknown:0});
  function renderSubtasks() {
    cells = new Map(); selection = null; $('subtaskDetail').hidden = true;
    const runs = data.runs.filter(r => baselines.some(b=>b.key === r.condition) && String(r.task) === $('sTask').value
      && (!$('sMemory').value || r.memory === $('sMemory').value)
      && (!$('sUnc').value || r.uncertainty === $('sUnc').value));
    const totals = {met:0, unmet:0, unknown:0};
    const present = new Set();
    for (const r of runs) for (const c of r.criteria) {
      const key = cellKey(c.criterion, r.condition);
      if (!cells.has(key)) cells.set(key, []);
      const observed = $('sFinalOnly').checked && r.score_basis === 'last_saved_state' ? {...c,status:'unknown'} : c;
      cells.get(key).push({r,c:observed}); totals[observed.status]++; present.add(c.criterion);
    }
    const definitions = data.definitions.filter(d => present.has(d.id))
      .sort((a,b) => (a.base_order == null) - (b.base_order == null)
        || (a.base_order ?? a.variant_order) - (b.base_order ?? b.variant_order)
        || a.label.localeCompare(b.label));
    const maxCount = Math.max(1, ...[...cells.values()].map(entries=>entries.length));
    $('subtaskCoverage').textContent = `${runs.length.toLocaleString()} runs · ${totals.met.toLocaleString()} succeeded · ${totals.unmet.toLocaleString()} failed · ${totals.unknown.toLocaleString()} unknown subtask outcomes. Full bar width = ${maxCount} outcomes. Each repetition contributes separately. ${runs.filter(r=>r.score_basis==='last_saved_state').length} runs use last saved states${$('sFinalOnly').checked ? ' (shown as unknown in final-only view)' : ' when final scores are missing'}.`;
    let html = `<thead><tr><th>Ordered subtasks</th>${baselines.map(c=>`<th><span class="swatch" style="background:${c.color}"></span>${esc(c.label)}</th>`).join('')}</tr></thead><tbody>`;
    for (const [index,d] of definitions.entries()) {
      html += `<tr data-definition="${d.id}"><td title="${esc(JSON.stringify(d.predicate))}"><b>${index+1}.</b> ${esc(d.label)}${d.base_order == null ? '<small class="sub">Additional variant criterion</small>' : ''}</td>`;
      for (const baseline of baselines) {
        const n = counts(cells.get(cellKey(d.id,baseline.key)) || []);
        const total = n.met + n.unmet + n.unknown;
        const checkpoints = (cells.get(cellKey(d.id,baseline.key)) || []).filter(e=>e.r.score_basis==='last_saved_state').length;
        const label = `${n.met} succeeded · ${n.unmet} failed · ${n.unknown} unknown`;
        const basis = checkpoints ? `<small class="sub">${checkpoints} last saved state${checkpoints===1?'':'s'}</small>` : '';
        html += `<td>${total ? `<button data-criterion="${d.id}" data-condition="${esc(baseline.key)}" aria-label="${esc(baseline.label + ': ' + d.label + ': ' + label)}" title="${label}. Click for individual runs."><small>${label}</small>${basis}<span class="subtask-bar" style="width:${100*total/maxCount}%" aria-hidden="true">${['met','unmet','unknown'].map(status=>`<span class="subtask-${status}" style="width:${100*n[status]/total}%"></span>`).join('')}</span></button>` : '<span class="sub">Not applicable</span>'}</td>`;
      }
      html += '</tr>';
    }
    $('subtaskTable').innerHTML = html + (definitions.length ? '' : '<tr><td colspan="4">No matching criteria.</td></tr>') + '</tbody>';
  }

  function renderDetail() {
    if (!selection) return;
    const d = data.definitions.find(d=>d.id === selection.criterion);
    const baseline = baselines.find(c=>c.key === selection.condition);
    const entries = (cells.get(cellKey(d.id,baseline.key)) || [])
      .filter(e=>!$('sStatus').value || e.c.status === $('sStatus').value)
      .sort((a,b)=>a.r.task_id.localeCompare(b.r.task_id) || a.r.iteration-b.r.iteration);
    $('subtaskDetail').hidden = false;
    $('subtaskDetailTitle').textContent = `T${d.task} · ${d.label} · ${baseline.label} · ${entries.length} outcomes`;
    let html = '<tr><th>Variant / repetition</th><th>Criterion / result</th><th>Scoring evidence</th><th>Artifacts</th></tr>';
    for (const {r,c} of entries) {
      const groups = (r.ambiguous_groups || []).filter(g => g.criterion_ids.includes(c.id));
      const ambiguous = groups.map(g=>`${g.unmet_count} unmet among {${g.criterion_ids.join(', ')}}; individual attribution unresolved.`).join(' ');
      const checkpoint = r.checkpoint;
      const basis = checkpoint ? `Last saved state: ${num(checkpoint.elapsed_seconds,1)} s after logging began; ${checkpoint.sim_step_count ?? '?'} steps; about ${num(checkpoint.seconds_before_process_end,1)} s before process end. This is not a final score.` : '';
      const note = basis || (c.status === 'unknown' ? (ambiguous || 'No attributable final score for this criterion.') : '');
      const links = (r.html_links || []).filter(l=>l.label === 'Trace');
      if (!links.length && r.html_links?.length) links.push(r.html_links[0]);
      html += `<tr><td>${esc(r.task_id)}<br>Repeat ${r.iteration}</td><td>${esc(c.id)} · <b>${esc(c.status)}</b><br><span class="sub">${esc(note)}</span></td><td class="sub">${esc(r.method)}<br>${esc(r.source)}${r.source_event_id ? `<br>${esc(r.source_event_id)}` : ''}<details><summary>Saved evaluator scores</summary><pre>${esc(JSON.stringify(r.metrics,null,2))}</pre></details></td><td>${links.map(l=>`<a href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.label)}</a>`).join('<br>')}${r.folder_url ? `<br><a href="${esc(r.folder_url)}" target="_blank" rel="noopener">Run files</a>` : ''}</td></tr>`;
    }
    $('subtaskDetailTable').innerHTML = html + (entries.length ? '' : '<tr><td colspan="4">No outcomes with this status. Choose another status above.</td></tr>');
  }
  $('subtaskTable').onclick = e => {
    const button = e.target.closest('button[data-criterion]'); if (!button) return;
    selection = {criterion:+button.dataset.criterion, condition:button.dataset.condition};
    $('sStatus').value = '';
    renderDetail(); $('subtaskDetail').scrollIntoView({behavior:'smooth',block:'nearest'});
  };
  ['sTask','sMemory','sUnc','sFinalOnly'].forEach(id=>$(id).onchange=renderSubtasks);
  $('sStatus').onchange=renderDetail;
  renderSubtasks();
})();
