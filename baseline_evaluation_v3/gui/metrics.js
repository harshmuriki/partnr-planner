/* Archive-only reporting: no API mutations or sandbox initialization. */
'use strict';
function parseCSV(text) {
  const rows = []; let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') { if (quoted && text[i + 1] === '"') { field += '"'; i++; } else quoted = !quoted; }
    else if (c === ',' && !quoted) { row.push(field); field = ''; }
    else if ((c === '\n' || c === '\r') && !quoted) {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); if (row.some(Boolean)) rows.push(row); row = []; field = '';
    } else field += c;
  }
  row.push(field); if (row.some(Boolean)) rows.push(row);
  const headers = rows.shift() || [];
  return rows.map(values => Object.fromEntries(headers.map((key, i) => [key, values[i] || ''])));
}
function metricValue(row, metric) {
  if (!row.run) return null;
  if (metric === 'sim_steps' && !row.run.steps_complete) return null;
  if (['video_duration_sec', 'execution_elapsed_seconds', 'action_wall_seconds'].includes(metric)) {
    const duration = row.run[metric];
    return typeof duration === 'number' && Number.isFinite(duration) ? duration : null;
  }
  const value = ['action_count', 'sim_steps'].includes(metric) ? row.run[metric] : row.skills?.[metric] ?? (row.skills ? 0 : null);
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}
// Statuses whose per-variant evidence JSON (CSV `source`) has per-action timers.
const MEASURED = {measured_single_sandbox:'Measured · 1 sandbox', measured_skynet:'Measured · Skynet A40'};
function recordingTiming(run, timing) {
  const missing = {execution_elapsed_seconds:null, action_wall_seconds:null, timing_status:'No matching timing evidence', interval_definition:''};
  if (!timing || timing.run_id !== run.id) return missing;
  const number = raw => typeof raw === 'string' && raw.trim() !== '' && Number.isFinite(Number(raw)) && Number(raw) >= 0 ? Number(raw) : null;
  const valid = ['recovered_current', 'recovered_http_session', ...Object.keys(MEASURED)].includes(timing.timing_status);
  return {execution_elapsed_seconds:valid ? number(timing.execution_elapsed_seconds) : null,
    action_wall_seconds:MEASURED[timing.timing_status] ? number(timing.action_wall_seconds) : null,
    timing_status:timing.timing_status, interval_definition:timing.interval_definition, timing_source:timing.source};
}
function clock(seconds) {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds)) return '—';
  const sign = seconds < 0 ? '-' : '';
  const tenths = Math.round(Math.abs(seconds) * 10);
  const minutes = Math.floor(tenths / 600), rest = (tenths % 600) / 10;
  return sign + (minutes ? `${minutes}m ${rest.toFixed(1).padStart(4, '0')}s` : `${rest.toFixed(1)}s`);
}
function delta(row, rows, metric) {
  const base = rows.find(r => r.id === `${row.task}-${row.mem}-BASE`);
  const a = metricValue(row, metric), b = base ? metricValue(base, metric) : null;
  return a === null || b === null ? null : a - b;
}
function mean(rows, metric) {
  const values = rows.map(r => metricValue(r, metric)).filter(v => v !== null);
  return {n: values.length, value: values.length ? values.reduce((a, b) => a + b, 0) / values.length : null};
}
if (typeof module !== 'undefined') module.exports = {parseCSV, metricValue, delta, mean, clock, recordingTiming};
if (typeof document !== 'undefined') {
  const $ = id => document.getElementById(id);
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = n => n === null ? '—' : n.toLocaleString(undefined, {maximumFractionDigits:1});
  const show = (n, name = $('metric').value) => ['video_duration_sec', 'execution_elapsed_seconds', 'action_wall_seconds'].includes(name) ? clock(n) : fmt(n);
  const axes = ['BASE','ABS','SUB','CON','DIS'];
  let rows = [], loading = false, lastUpdated = '', archiveSignature = null;
  const filtered = () => rows.filter(r => ['task','mem','axis'].every(k => !$(k).value || r[k] === $(k).value));
  const table = (headers, body) => `<table><thead><tr>${headers.map(h => `<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${body.join('')}</tbody></table>`;
  function render() {
    const selected = filtered(), saved = selected.filter(r => r.run), metric = $('metric').value;
    const avg = mean(selected, metric);
    const video = mean(selected, 'video_duration_sec');
    const elapsed = mean(selected, 'execution_elapsed_seconds');
    $('summary').innerHTML = [[`${saved.length} / ${selected.length}`, 'Recordings saved'],[clock(elapsed.value),`Overall avg elapsed time · n=${elapsed.n}/${saved.length}`],[show(avg.value),`Mean ${$('metric').selectedOptions[0].text.toLowerCase()} · n=${avg.n}`],[clock(video.value),`Mean video time · n=${video.n}`]].map(([v,l]) => `<div class="card"><strong>${v}</strong><span>${esc(l)}</span></div>`).join('');
    const timingGroups = [...new Set(selected.map(r => r.task))].map(task => [task, selected.filter(r => r.task === task)]);
    timingGroups.push(['Overall (per recording)', selected]);
    $('task-times').innerHTML = table(['Task','Saved','Avg elapsed time','Elapsed coverage','Avg active skill time','Active coverage','Avg video time','Video coverage'], timingGroups.map(([task, group]) => {
      const wall = mean(group, 'execution_elapsed_seconds'), playback = mean(group, 'video_duration_sec');
      const active = mean(group, 'action_wall_seconds');
      const count = group.filter(r => r.run).length;
      return `<tr><td>${esc(task)}</td><td>${count}</td><td>${clock(wall.value)}</td><td>${wall.n} / ${count}</td><td>${clock(active.value)}</td><td>${active.n} / ${count}</td><td>${clock(playback.value)}</td><td>${playback.n} / ${count}</td></tr>`;
    }));
    const memories = $('mem').value ? [$('mem').value] : ['ACC','INC','OUT'];
    const columnGroups = [
      {name:'Object Availability', columns:[['BASE','Target Exists'],['SUB','Substitute Available'],['ABS','No Suitable Object']]},
      {name:'Object Containment', columns:[['BASE','On Surface'],['CON','Inside Closed Receptacle']]},
      {name:'Distractor Presence', columns:[['BASE','None'],['DIS','Present']]},
    ].map(g => ({...g, columns:g.columns.filter(([axis]) => !$('axis').value || axis === $('axis').value)})).filter(g => g.columns.length);
    const columns = columnGroups.flatMap(g => g.columns);
    const labels = {ACC:'Accurate', INC:'Incomplete', OUT:'Outdated'};
    $('matrix').innerHTML = [...new Set(selected.map(r => r.task))].map(task => {
      const taskRows = selected.filter(r => r.task === task);
      const max = Math.max(1, ...taskRows.map(r => metricValue(r, metric) ?? 0));
      return `<div class="task-matrix"><h3>${task}<span>${esc($('metric').selectedOptions[0].text)}</span></h3><div class="scroll"><table class="comparison"><thead><tr><th rowspan="2" scope="col">Memory</th>${columnGroups.map(g => `<th scope="colgroup" colspan="${g.columns.length}">${g.name}</th>`).join('')}</tr><tr>${columns.map(([,label]) => `<th scope="col">${label}</th>`).join('')}</tr></thead><tbody>${memories.map(mem =>
        `<tr><th scope="row" class="memory ${mem}">${labels[mem]}</th>${columns.map(([axis]) => {
          const row = rows.find(r => r.id === `${task}-${mem}-${axis}`), value = row ? metricValue(row, metric) : null;
          return `<td>${row ? `<a class="metric-cell ${axis === 'BASE' ? 'base-cell' : ''}" href="scenario_viewer.html#${row.id}" aria-label="${row.id}: ${show(value)} ${esc($('metric').selectedOptions[0].text)}"><span class="axis-label">${axis}</span><strong>${show(value)}</strong>${value === null ? `<small>${row.run ? 'Unknown' : 'Not saved'}</small>` : `<div class="bar" style="width:${100 * value / max}%"></div>`}</a>` : '<span class="not-applicable">N/A</span>'}</td>`;
        }).join('')}</tr>`
      ).join('')}</tbody></table></div></div>`;
    }).join('');
    $('records').innerHTML = table(['Variant','Actions','Sim steps','Elapsed time','Active skill time','Timing evidence','Video time','Navigate','Explore','Open / Close','Δ vs BASE'],selected.map(r => {
      const d = delta(r, rows, metric);
      return `<tr><td><a href="scenario_viewer.html#${r.id}">${r.id}</a></td>${['action_count','sim_steps'].map(k => `<td>${fmt(metricValue(r,k))}</td>`).join('')}<td>${clock(metricValue(r,'execution_elapsed_seconds'))}</td><td>${clock(metricValue(r,'action_wall_seconds'))}</td><td>${MEASURED[r.run?.timing_status] || esc(r.run?.interval_definition || r.run?.timing_status || '—')}</td><td>${clock(metricValue(r,'video_duration_sec'))}</td>${['Navigate','Explore'].map(k => `<td>${fmt(metricValue(r,k))}</td>`).join('')}<td>${fmt(metricValue(r,'Open'))} / ${fmt(metricValue(r,'Close'))}</td><td>${d > 0 ? '+' : ''}${show(d)}</td></tr>`;
    }));
    $('action-times').innerHTML = selected.filter(r => r.timing).map(r => {
      const timing = r.timing;
      const body = timing.actions.map(a => `<tr><td>${a.sequence}</td><td>${esc(a.skill)} ${esc(a.target)}</td><td>${clock(a.observed_wall_seconds)}</td><td>${clock(a.result?.timing?.action_wall_seconds ?? null)}</td><td>${clock(a.result?.timing?.environment_step_wall_seconds ?? null)}</td><td>${clock(a.result?.timing?.frame_callback_wall_seconds ?? null)}</td></tr>`);
      return `<details><summary>${r.id} · ${clock(timing.execution_elapsed_seconds)} elapsed</summary><p>Load: ${clock(timing.load_wall_seconds)} · Save: ${clock(timing.save_wall_seconds)} · Worker commands: ${clock(timing.worker_command_wall_seconds)}. No human pauses; automatic sequential runner${timing.status === 'replayed' ? ' (headless Skynet replay of the saved sequence; nothing saved)' : ''}.</p>${table(['#','Action','Request → observed completion','Active skill','Environment step calls','Frame callbacks'],body)}</details>`;
    }).join('') || '<p>No instrumented recordings in the current filters.</p>';
  }
  async function get(path, json = false) {
    const response = await fetch(path, {cache:'no-store'});
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return json ? response.json() : response.text();
  }
  async function refresh(force = false) {
    if (loading) return; loading = true; $('refresh').disabled = true;
    if (!lastUpdated) { $('status').hidden = false; $('status').textContent = 'Loading saved recordings…'; }
    try {
      const index = await get('../ground_truth/index.json', true);
      if (!Array.isArray(index.ground_truths)) throw new Error('Invalid ground-truth index');
      const timingCSV = await get('../ground_truth_timing/current_recordings.csv');
      const timing = new Map(parseCSV(timingCSV).map(r => [r.variant, r]));
      const signature = JSON.stringify(index.ground_truths) + timingCSV;
      if (!force && signature === archiveSignature) { $('status').hidden = true; return; }
      const csv = await get('../variant_object_assets.csv');
      const ids = [...new Set(parseCSV(csv).map(r => r.Variant_ID.trim()).filter(id => /^T[1-7]-(ACC|INC|OUT)-(BASE|ABS|SUB|CON|DIS)$/.test(id)))];
      const runs = new Map(index.ground_truths.map(r => [r.variant,r]));
      runs.forEach(run => Object.assign(run, recordingTiming(run, timing.get(run.variant))));
      const next = ids.map(id => { const [task,mem,axis] = id.split('-'); return {id,task,mem,axis,run:runs.get(id)}; });
      // Bound concurrent reads so recording traffic stays responsive.
      let cursor = 0;
      await Promise.all(Array.from({length:4}, async () => {
        while (cursor < next.length) {
          const row = next[cursor++]; if (!row.run) continue;
          try {
            const actions = parseCSV(await get(`../ground_truth/${row.task}/${row.id}/actions.csv`));
            if (actions.length !== row.run.action_count || actions.some(a => !a.skill)) throw new Error('Archive updating');
            if (MEASURED[row.run.timing_status]) {
              const detail = await get(`../ground_truth_timing/${row.run.timing_source}`, true);
              if (detail.run_id !== row.run.id || !['saved', 'replayed'].includes(detail.status)) throw new Error('Timing evidence updating');
              row.timing = detail;
            }
            row.skills = {};
            actions.forEach(a => { row.skills[a.skill] = (row.skills[a.skill] || 0) + 1; });
          } catch { row.detailError = true; }
        }
      }));
      const durations = await window.groundTruthVideoDurations('../ground_truth/', index.ground_truths);
      next.forEach(row => { if (row.run) row.run.video_duration_sec = durations.get(row.id) ?? null; });
      const unreadable = next.filter(row => row.run?.artifact_status === 'ready' && row.run.video_duration_sec === null).length;
      rows = next.sort((a,b) => Number(a.task.slice(1))-Number(b.task.slice(1)) || axes.indexOf(a.axis)-axes.indexOf(b.axis) || ['ACC','INC','OUT'].indexOf(a.mem)-['ACC','INC','OUT'].indexOf(b.mem));
      for (const k of ['task','axis']) {
        const previous = $(k).value;
        $(k).innerHTML = `<option value="">All ${k === 'task' ? 'tasks' : 'scenarios'}</option>` + [...new Set(rows.map(r => r[k]))].map(v => `<option>${v}</option>`).join('');
        $(k).value = previous;
      }
      lastUpdated = new Date().toLocaleTimeString();
      const errors = rows.filter(r => r.detailError).length;
      archiveSignature = errors || unreadable ? null : signature;
      $('status').className = errors || unreadable ? 'error' : '';
      $('status').textContent = [errors ? `${errors} action breakdown(s) unavailable` : '', unreadable ? `${unreadable} video(s) could not be read` : '']
        .filter(Boolean).join('; ') + (errors || unreadable ? '; retrying automatically.' : '');
      $('status').hidden = !errors && !unreadable;
      $('export').disabled = false; render();
    } catch (error) {
      $('status').hidden = false;
      $('status').className = 'error';
      $('status').textContent = `Could not refresh: ${error.message}.${lastUpdated ? ` Showing data from ${lastUpdated}.` : ' Open this page through the scenario viewer server.'}`;
    } finally { loading = false; $('refresh').disabled = false; }
  }
  ['task','mem','axis','metric'].forEach(k => $(k).addEventListener('change',render));
  $('refresh').onclick = () => refresh(true);
  $('export').onclick = () => {
    const keys = ['action_count','sim_steps','execution_elapsed_seconds','action_wall_seconds','video_duration_sec','Navigate','Explore','Open','Close'];
    const data = [['variant','task','memory','scenario','saved','steps_complete','artifact_status','timing_status','interval_definition',...keys,'delta_metric','delta_vs_base','completed_at'],...filtered().map(r => [r.id,r.task,r.mem,r.axis,Boolean(r.run),r.run?.steps_complete ?? '',r.run?.artifact_status ?? '',r.run?.timing_status ?? '',r.run?.interval_definition ?? '',...keys.map(k => { const v = metricValue(r,k); return k === 'video_duration_sec' && v !== null ? v.toFixed(3) : v; }),$('metric').value,delta(r,rows,$('metric').value),r.run?.completed_at ?? ''])];
    const csv = data.map(line => line.map(v => `"${String(v ?? '').replace(/"/g,'""')}"`).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const a = document.createElement('a'); a.href = url; a.download = 'ground-truth-metrics.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
  };
  setInterval(() => { if (!document.hidden) refresh(); },5000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  refresh();
}
