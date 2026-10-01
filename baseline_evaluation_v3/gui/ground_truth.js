/* Exact-episode sandbox that records live; the run is saved as ground truth without replaying (one per variant). */
(() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let variant = null, state = null, generation = 0, fetching = false, posting = false, frameLoading = false;
  let lastGraph = '', lastRooms = '', lastActions = '', lastVideo = '', lastReports = '', lastSteps = '', lastPacks = '', loadError = false;
  let roomRevision = 0, sandboxView = false, displayedRun = null;
  let chapters = [], lastEvidence = '';
  const roomSaving = new Set(), roomFeedback = new Map();
  const el = id => document.getElementById(id);
  const api = id => `../api/ground-truth/${encodeURIComponent(id)}`;
  async function request(url, payload) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch(url, {cache:'no-store', signal:controller.signal, ...(payload ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)} : {})});
      const isJSON = (response.headers.get('content-type') || '').includes('application/json');
      if (response.status === 404 || !isJSON) {
        throw new Error('This address is serving the static viewer without the ground-truth API. Open the viewer server on port 8000, or start it with python3 scripts/serve_scenario_viewer.py --port 8000.');
      }
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      return data;
    } catch (e) {
      if (e.name === 'AbortError') throw new Error('The viewer server did not respond within 12 seconds. Check that it is running, then retry.');
      throw e;
    } finally { clearTimeout(timeout); }
  }
  function error(message = '') { if (el('gtError')) { el('gtError').textContent = message; el('gtError').hidden = !message; } }
  async function post(operation, payload) {
    const id = variant, token = generation;
    posting = true;
    renderStatus();
    try {
      const sessionId = state?.session?.id;
      await request(`${api(id)}/${operation}`, {...payload, ...(sessionId ? {session_id:sessionId} : {})});
      if (token !== generation) return;
      if (operation === 'sandbox') sandboxView = true;
      if (operation === 'close') sandboxView = false;
      error();
      await poll(true);
    } catch (e) { if (token === generation) error(e.message); }
    finally { posting = false; if (token === generation) renderStatus(); }
  }
  let videoTimes = new Map();
  window.applyVideoDurations = times => { videoTimes = times; if (state) renderStatus(); };
  const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
  const stepLabel = a => `${esc(a.skill)} ${esc(a.skill === 'ReportAbsence' ? '' : a.target)}`;
  function renderStatus() {
    if (!state || !el('gtSandbox')) return;
    const s = state.session, here = s.variant === variant, busy = s.busy || posting;
    const sandbox = here && s.mode === 'sandbox';
    const kept = s.steps.filter(step => step.keep).length, count = here ? s.steps.length : 0;
    const sessions = state.sessions || [], full = sessions.length >= (state.max_sessions || 1);
    el('gtSessions').textContent = state.max_sessions
      ? `${sessions.length}/${state.max_sessions} sandboxes open${sessions.length ? ': ' + sessions.map(item => `${item.variant}${item.busy ? ' (busy)' : ''}`).join(', ') : ''}. Each variant records independently; close a sandbox to free a slot.`
      : 'Parallel sandboxes become available after the viewer server restarts.';
    el('gtClose').hidden = !s.id;
    el('gtClose').disabled = busy;
    el('gtSandbox').disabled = busy || !state.meta.available || (!here && full);
    el('gtSandbox').textContent = here && s.mode ? 'Reset sandbox to the start' : 'Open sandbox';
    el('gtRun').disabled = busy || !sandbox;
    el('gtStatus').textContent = here && s.phase ? s.phase :
      here && s.error ? s.error :
      sandbox ? `● Recording · ${plural(count, 'action')} since the start. Save once the task is complete, or reset to start over.` :
      full && !here ? 'All sandbox slots are in use. Select an open variant and close its sandbox to free a slot.' :
      s.busy ? 'The simulator is busy.' : 'Open the sandbox to load this episode at its exact start.';
    const result = s.last_result;
    el('gtResult').hidden = !result;
    if (result) { el('gtResult').textContent = result.message; el('gtResult').className = result.ok ? 'gt-saved' : 'gt-error'; }
    const evaluation = here ? s.snapshot?.evaluation : null;
    const reported = r => s.steps.some(a => a.skill === 'ReportAbsence' && a.target === r);
    const complete = evaluation?.success && state.meta.reports.every(reported) && !state.meta.unsupported_criteria.length;
    el('gtRecord').disabled = busy || !sandbox || !count || !complete;
    el('gtRecord').textContent = `Save ground truth · ${plural(count, 'action')}`;
    el('gtRecord').title = !sandbox ? 'Open the sandbox to record' : complete ? 'Save every action since the start as this variant\'s ground truth' : 'Available once the task requirements are met';
    const percent = Math.min(100, Math.max(0, Math.round((evaluation?.percent_complete || 0) * 100)));
    el('gtProgress').hidden = !evaluation;
    el('gtProgress').className = complete ? 'gt-completion is-complete' : 'gt-completion';
    el('gtProgressTitle').textContent = complete ? '✓ Success requirements met' : `Task progress · ${percent}%`;
    el('gtProgressDetail').textContent = complete
      ? 'The task requirements are met. Save ground truth to keep this run, or keep going.'
      : state.meta.unsupported_criteria.length ? 'Some requirements cannot yet be evaluated.'
      : evaluation?.success && !state.meta.reports.every(reported) ? 'An absence report is still required.'
      : 'Updates live as you try skills, including during an action.';
    el('gtProgressBar').value = percent;

    const gt = state.ground_truth;
    const duration = videoTimes.get(variant) ?? gt?.video_duration_sec;
    const durationLabel = typeof duration === 'number' ? ` · ${window.formatVideoTime(duration)} video` : '';
    el('gtSaved').innerHTML = gt
      ? `<b>Saved ground truth</b> · ${plural(gt.action_count, 'action')} · ${gt.sim_steps} simulator steps${gt.steps_complete ? '' : ' (partial)'}${durationLabel}${gt.stale ? '<br>The spec or episode changed since this was saved. Record it again.' : ''}`
      : 'No ground truth saved for this variant yet.';
    el('gtSaved').className = gt ? 'gt-saved' : 'gt-help';
    el('gtDownloads').innerHTML = gt ? `<a href="${esc(gt.artifacts.data)}" target="_blank" rel="noopener">Run data (JSON)</a> · <a href="${esc(gt.artifacts.actions)}" download>Actions (CSV)</a> · <a href="${esc(gt.artifacts.files)}" target="_blank" rel="noopener">Folder</a>` : '';
    el('gtVideoStatus').textContent = state.archive_error ? `Archive export needs attention: ${state.archive_error}` :
      gt?.artifact_status === 'error' ? `Video unavailable: ${gt.artifact_error}. The actions are still saved.` : '';
    const video = gt?.artifacts.video || '';
    if (gt?.id && gt.id !== displayedRun) { displayedRun = gt.id; sandboxView = false; }
    const playback = !!video && !sandboxView;
    el('gtWorkspace').hidden = playback;
    el('gtRecord').hidden = playback;
    el('gtPracticeHelp').hidden = playback;
    el('gtStatus').hidden = playback;
    el('gtProgress').hidden = playback || !evaluation;
    el('gtRecording').hidden = !playback;
    el('gtShowVideo').hidden = !video || playback;
    el('gtShowVideo').disabled = busy;
    if (playback) el('gtSandbox').textContent = sandbox ? 'Return to live sandbox' : 'Open sandbox to record again';
    if (!here || !s.snapshot?.has_frame) { el('gtFrame').hidden = true; el('gtFrameHelp').textContent = 'The live camera appears when the sandbox opens.'; }
    if (!playback) el('gtVideo').pause();
    const videoKey = `${gt?.id || ''}:${video}`;
    if (videoKey !== lastVideo) {
      lastVideo = videoKey;
      chapters = [];
      let frame = 0;
      (gt?.actions || []).forEach(a => { chapters.push({action:a, start:frame / 30}); frame += a.result?.recorded_frames || 0; });
      renderChapters();
      if (video) { el('gtVideo').src = video; el('gtDownloadVideo').href = video; }
      else { el('gtVideo').removeAttribute('src'); el('gtVideo').load(); }
      updatePlayback();
    }
    renderEvidence();
    const savedActions = JSON.stringify(gt?.actions || []);
    if (savedActions !== lastActions) {
      lastActions = savedActions;
      el('gtSavedActions').innerHTML = (gt?.actions || []).map(a => `<li><b>${stepLabel(a)}</b><br>${esc(a.result?.response || a.result?.error || '')}${a.result?.skill_steps !== undefined ? ` · ${a.result.skill_steps} simulator steps` : ''}</li>`).join('') || '<li>No ground truth saved yet.</li>';
    }

    const reports = JSON.stringify(state.meta.reports);
    if (reports !== lastReports) {
      lastReports = reports;
      el('gtReports').innerHTML = state.meta.reports.map(r => `<p class="gt-help">${esc(r)}</p><button type="button" data-report="${esc(r)}">Report no suitable object</button>`).join('');
      el('gtReports').querySelectorAll('button').forEach(b => b.onclick = () => post('action', {request_id:crypto.randomUUID(), skill:'ReportAbsence', target:b.dataset.report}));
    }
    el('gtReports').querySelectorAll('button').forEach(button => {
      const done = here && reported(button.dataset.report);
      button.disabled = busy || !sandbox || done;
      button.textContent = done ? 'Absence reported' : 'Report no suitable object';
    });
    el('gtMemory').textContent = state.meta.robot_memory || 'This spec lists no initial robot memory.';
    el('gtCriteria').textContent = state.meta.success_criteria || '';
    el('gtBlocked').hidden = state.meta.available && !state.meta.unsupported_criteria.length;
    el('gtBlocked').textContent = state.meta.blocked_reason || (state.meta.unsupported_criteria.length ? 'These spec requirements cannot yet be evaluated, so a recording cannot be saved: ' + state.meta.unsupported_criteria.join(' ') : '');

    const steps = JSON.stringify([here ? s.steps : [], busy, sandbox]);
    if (steps !== lastSteps) {
      lastSteps = steps;
      const list = here ? s.steps : [];
      const disabled = busy || !sandbox ? ' disabled' : '';
      el('gtStepList').innerHTML = list.map((a, i) => `<li class="${a.keep ? '' : 'gt-unchecked'}">
        <label><input type="checkbox" data-step="${esc(a.id)}"${a.keep ? ' checked' : ''}${disabled} aria-label="Include step ${i + 1} in a pack"><b>${i + 1}. ${stepLabel(a)}</b></label>
        ${a.result?.ok === false || !/success/i.test(a.result?.response || '') && a.skill !== 'ReportAbsence' ? '<span class="gt-help">failed</span>' : ''}
        <div class="gt-help">${esc(a.result?.response || a.result?.error || '')}${a.result?.skill_steps !== undefined ? ` · ${a.result.skill_steps} simulator steps` : ''}</div></li>`).join('')
        || '<li class="gt-help">No actions yet.</li>';
      el('gtStepList').querySelectorAll('[data-step]').forEach(box => box.onchange = () => post('steps', {id:box.dataset.step, keep:box.checked}));
    }
    el('gtPackSave').disabled = busy || !sandbox || !kept;
    const packs = JSON.stringify([state.packs || [], busy, sandbox]);
    if (packs !== lastPacks) {
      lastPacks = packs;
      const disabled = busy || !sandbox ? ' disabled' : '';
      el('gtPackList').innerHTML = (state.packs || []).map(p => `<li>
        <details><summary><b>${esc(p.name)}</b> <span class="gt-help">· ${plural(p.steps.length, 'step')} · saved from ${esc(p.source_variant)}</span></summary>
          <ol class="gt-pack-steps">${p.steps.map(a => `<li>${stepLabel(a)}</li>`).join('')}</ol></details>
        <div class="gt-pack-actions"><button type="button" data-apply="${esc(p.id)}"${disabled} title="${sandbox ? 'Run these steps in this sandbox' : 'Open the sandbox to apply'}">Apply</button>
        <button type="button" class="gt-remove" data-delete-pack="${esc(p.id)}" data-name="${esc(p.name)}"${busy ? ' disabled' : ''} aria-label="Delete pack ${esc(p.name)}" title="Delete pack">✕</button></div></li>`).join('')
        || `<li class="gt-help">No packs for ${esc(variant.split('-')[0])} yet.</li>`;
      el('gtPackList').querySelectorAll('[data-apply]').forEach(b => b.onclick = () => post('packs', {apply:b.dataset.apply}));
      el('gtPackList').querySelectorAll('[data-delete-pack]').forEach(b => b.onclick = () => { if (confirm(`Delete the pack “${b.dataset.name}” for every ${variant.split('-')[0]} variant?`)) post('packs', {delete:b.dataset.deletePack}); });
    }

    const snapshot = here ? s.snapshot || {} : {};
    const graph = JSON.stringify(snapshot.gt_graph || {});
    if (graph !== lastGraph) {
      lastGraph = graph;
      const aliases = Object.entries(snapshot.aliases || {});
      const nodeHTML = node => {
        const children = node.children || [...(node.furniture || []), ...(node.receptacles || []), ...(node.objects || [])];
        const alias = aliases.find(([,runtime]) => runtime === node.name)?.[0];
        const label = alias || node.name;
        const button = `<button type="button" class="gt-entity" data-target="${esc(alias || 'runtime:' + node.name)}">${esc(label)}</button>`;
        return children.length ? `<details open><summary>${button}</summary>${children.map(nodeHTML).join('')}</details>` : `<div>${button}${node.states ? ` <span class="gt-help">${esc(Object.entries(node.states).map(([k,v])=>`${k}: ${v}`).join(', '))}</span>` : ''}</div>`;
      };
      const root = snapshot.gt_graph?.root;
      el('gtTree').innerHTML = root ? nodeHTML(root) : (snapshot.gt_graph?.rooms || []).map(nodeHTML).join('') || 'Open the sandbox to view rooms, furniture, and objects.';
      el('gtTree').querySelectorAll('[data-target]').forEach(b => b.onclick = () => { el('gtTarget').value = b.dataset.target; });
      el('gtEntities').innerHTML = [...Object.keys(snapshot.aliases || {}), ...(snapshot.entities?.entity_names || []).filter(name=>!aliases.some(([,runtime])=>runtime===name)).map(name=>'runtime:'+name)].map(name=>`<option value="${esc(name)}"></option>`).join('');
    }
  }
  function renderEvidence() {
    const gt = state.ground_truth, evidence = gt?.criteria_evidence;
    const recording = state.session.variant === variant && state.session.mode === 'recording';
    const key = JSON.stringify([gt?.id, evidence, gt?.artifacts.video, recording]);
    if (key === lastEvidence) return;
    lastEvidence = key;
    const status = evidence?.status || 'unknown';
    const labels = {pass:'Passed', fail:'Failed', unknown:'Not verified', outdated:'Outdated recording'};
    el('gtEvidenceBadge').textContent = gt ? labels[status] : 'No recording';
    el('gtEvidenceBadge').className = `gt-evidence-badge gt-evidence-${status}`;
    el('gtEvidenceSummary').textContent = !gt ? 'Record a ground truth to see whether it meets every requirement and which steps satisfied them.'
      : !evidence ? 'Saved criteria details are unavailable. Reconnect to the updated viewer server.'
      : status === 'pass' ? `All ${evidence.total} requirements passed in this saved recording.`
      : status === 'outdated' ? `The spec or episode has changed. These results describe the recorded version (${evidence.met}/${evidence.total} passed), not verification of the current version.`
      : status === 'fail' ? `This recording does not meet all requirements (${evidence.met}/${evidence.total} passed).`
      : `Not all requirements can be verified from this recording (${evidence.met}/${evidence.total} passed).`;
    el('gtEvidenceBasis').textContent = evidence ? evidence.basis + ' Select a linked step to pause the video at its evidence.' : '';
    el('gtEvidenceRows').innerHTML = (evidence?.criteria || []).map((row, index) => `<li class="gt-criterion">
      <div class="gt-criterion-label"><span class="gt-evidence-badge gt-evidence-${esc(row.status)}">${esc(labels[row.status])}</span><span>${esc(row.label)}</span></div>
      <div><p class="gt-help">${esc(row.detail)}</p><div class="gt-evidence-links">${row.evidence.map((link, j) => {
        const time = link.video_time;
        const enabled = !!gt.artifacts.video && Number.isFinite(time) && !recording;
        return `<button type="button" data-criterion="${index}" data-evidence="${j}"${enabled ? '' : ' disabled'} title="${enabled ? 'View this recorded evidence' : recording ? 'Available after the current recording finishes' : 'Video timing is unavailable'}">Step ${link.sequence} · ${esc(link.skill)} ${esc(link.target)}${Number.isFinite(time) ? ` <time>${Math.floor(time / 60)}:${(time % 60).toFixed(1).padStart(4, '0')}</time>` : ''}</button>`;
      }).join('')}</div></div></li>`).join('');
    el('gtEvidenceRows').querySelectorAll('button').forEach(button => button.onclick = () => {
      const link = evidence.criteria[Number(button.dataset.criterion)].evidence[Number(button.dataset.evidence)];
      sandboxView = false;
      renderStatus();
      const video = el('gtVideo');
      video.pause();
      const seek = () => { video.currentTime = Math.max(0, Math.min(link.video_time, video.duration - 1 / 30)); updatePlayback(); };
      if (video.readyState >= 1) seek();
      else video.addEventListener('loadedmetadata', seek, {once:true});
      el('gtRecording').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block:'start'});
    });
  }
  function renderChapters() {
    el('gtChapters').innerHTML = chapters.map(({action, start}, i) => `<li><button type="button" data-chapter="${i}"><span class="gt-chapter-number">${i + 1}</span><span>${stepLabel(action)}</span><time>${Math.floor(start / 60)}:${String(Math.floor(start % 60)).padStart(2, '0')}</time></button></li>`).join('');
    el('gtChapters').querySelectorAll('button').forEach(button => button.onclick = () => {
      el('gtVideo').currentTime = chapters[Number(button.dataset.chapter)].start;
      el('gtVideo').play().catch(() => {});
      updatePlayback();
    });
  }
  function updatePlayback() {
    const video = el('gtVideo');
    if (!video || !chapters.length) return;
    let index = 0;
    chapters.forEach((chapter, i) => { if (chapter.start <= video.currentTime + 0.001) index = i; });
    const action = chapters[index].action;
    el('gtNowPlaying').textContent = `Step ${index + 1} of ${chapters.length} · ${action.skill} ${action.target}`;
    el('gtChapters').querySelectorAll('button').forEach((button, i) => {
      if (i === index) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
  }
  let noteTimers = {};
  const noteDirty = new Set();
  const noteTime = note => note ? 'Saved' : '';
  function renderNotes() {
    ['task', 'all'].forEach(scope => {
      const box = el(`gtNote-${scope}`), note = state?.notes?.[scope];
      // Never overwrite text that is being typed or has not been saved yet.
      if (!box || noteDirty.has(scope) || document.activeElement === box) return;
      box.value = note?.text || '';
      el(`gtNoteStatus-${scope}`).textContent = noteTime(note);
    });
  }
  function queueNote(scope) {
    noteDirty.add(scope);
    el(`gtNoteStatus-${scope}`).textContent = 'Unsaved changes…';
    clearTimeout(noteTimers[scope]);
    noteTimers[scope] = setTimeout(() => saveNote(scope), 800);
  }
  async function saveNote(scope) {
    const id = variant, token = generation, box = el(`gtNote-${scope}`), text = box.value;
    el(`gtNoteStatus-${scope}`).textContent = 'Saving…';
    try {
      const saved = await request(`${api(id)}/notes`, {scope, text});
      if (token !== generation) return;
      if (state) state.notes = {...state.notes, [scope]: saved.note};
      if (box.value === text) { noteDirty.delete(scope); el(`gtNoteStatus-${scope}`).textContent = saved.note ? noteTime(saved.note) : 'Cleared'; }
    } catch (e) {
      if (token === generation) el(`gtNoteStatus-${scope}`).textContent = `Not saved: ${e.message}. Keep this page open and edit again to retry.`;
    }
  }
  window.addEventListener('beforeunload', event => { if (noteDirty.size) event.preventDefault(); });
  // Two ranked annotations per target: likely rooms (shared by object name everywhere) and
  // likely furniture (shared by object name within the same apartment).
  const KINDS = {
    room: {choices: 'room_choices', endpoint: 'rooms', field: 'rooms', noun: 'room'},
    furniture: {choices: 'furniture_choices', endpoint: 'furniture', field: 'furniture', noun: 'furniture'},
  };
  function roomFocus(node) {
    return node?.dataset.object ? {object:node.dataset.object, kind:node.dataset.kind, item:node.dataset.item, action:node.dataset.action} : null;
  }
  function furnitureIndex() {
    const index = new Map();
    Object.entries(state.meta.furniture || {}).forEach(([room, entries]) => entries.forEach(entry => {
      const known = index.get(entry.name) || {description: entry.description, rooms: []};
      known.rooms.push(room);
      index.set(entry.name, known);
    }));
    return index;
  }
  function rankedList(kind, obj, items, busy, label) {
    const attributes = item => `data-kind="${kind}" data-object="${esc(obj)}" data-item="${esc(item)}"`;
    const noun = KINDS[kind].noun;
    return `<ol class="gt-room-ranking" aria-label="Likely ${noun} order for ${esc(obj)}">${items.map((item, index) => `<li>
      <span class="gt-room-rank" aria-hidden="true">${index + 1}</span>
      <label><input type="checkbox" ${attributes(item)} data-action="select" checked${busy ? ' disabled' : ''} aria-label="Include ${esc(item)} for ${esc(obj)}"><span>${label(item)}</span></label>
      <div class="gt-room-moves"><button type="button" ${attributes(item)} data-action="up" aria-label="Move ${esc(item)} earlier for ${esc(obj)}" title="Check earlier"${busy || index === 0 ? ' disabled' : ''}>↑</button><button type="button" ${attributes(item)} data-action="down" aria-label="Move ${esc(item)} later for ${esc(obj)}" title="Check later"${busy || index === items.length - 1 ? ' disabled' : ''}>↓</button></div>
    </li>`).join('')}</ol>`;
  }
  function option(kind, obj, item, busy, label) {
    return `<label><input type="checkbox" data-kind="${kind}" data-object="${esc(obj)}" data-item="${esc(item)}" data-action="select"${busy ? ' disabled' : ''} aria-label="Include ${esc(item)} for ${esc(obj)}"><span>${label}</span></label>`;
  }
  function renderRooms(focus = null) {
    const key = JSON.stringify([state.meta.rooms, state.meta.targets, state.meta.furniture, state.room_choices, state.furniture_choices, [...roomSaving], [...roomFeedback]]);
    if (key === lastRooms) return;
    const container = el('gtRooms');
    focus = focus || (container.contains(document.activeElement) ? roomFocus(document.activeElement) : null);
    const expanded = new Map([...container.querySelectorAll('details[data-add]')].map(node => [node.dataset.add, node.open]));
    lastRooms = key;
    const furniture = furnitureIndex();
    container.innerHTML = state.meta.targets.map(obj => {
      const rooms = state.room_choices[obj] || [], chosen = (state.furniture_choices || {})[obj] || [];
      const busy = roomSaving.has(obj);
      const availableRooms = state.meta.rooms.filter(room => !rooms.includes(room));
      const furnitureLabel = name => {
        const info = furniture.get(name);
        return `${esc(name)} <span class="gt-help">· ${esc(info?.rooms.join(', ') || '')}${info?.description ? ' · ' + esc(info.description) : ''}</span>`;
      };
      // Offer furniture grouped by room, following the room search order first.
      const roomOrder = [...rooms.filter(room => state.meta.furniture?.[room]), ...Object.keys(state.meta.furniture || {}).filter(room => !rooms.includes(room))];
      const groups = roomOrder.map(room => {
        const entries = state.meta.furniture[room].filter(entry => !chosen.includes(entry.name));
        if (!entries.length) return '';
        return `<div class="gt-furniture-group"><div class="gt-help"><b>${esc(room)}</b>${rooms.includes(room) ? ` · likely room ${rooms.indexOf(room) + 1}` : ''}</div>
          <div class="gt-room-options">${entries.map(entry => option('furniture', obj, entry.name, busy, `${esc(entry.name)}${entry.description ? ` <span class="gt-help">${esc(entry.description)}</span>` : ''}`)).join('')}</div></div>`;
      }).join('');
      const furnitureCount = [...furniture.keys()].filter(name => !chosen.includes(name)).length;
      const openRooms = expanded.has(`room:${obj}`) ? expanded.get(`room:${obj}`) : !rooms.length;
      const openFurniture = expanded.get(`furniture:${obj}`) || false;
      return `<div class="gt-room-card" data-object="${esc(obj)}" aria-busy="${busy}"><b>${esc(obj)}</b>
        <h4 class="gt-rank-title">Likely rooms</h4>
        ${rooms.length ? rankedList('room', obj, rooms, busy, esc) : '<p class="gt-help gt-room-empty">No rooms selected. Add the rooms worth visiting below.</p>'}
        <details class="gt-room-add" data-add="room:${esc(obj)}"${openRooms ? ' open' : ''}><summary>Add rooms to search (${availableRooms.length})</summary>
          <div class="gt-room-options">${availableRooms.map(room => option('room', obj, room, busy, esc(room))).join('') || '<span class="gt-help">All rooms are in the search order.</span>'}</div>
        </details>
        ${furniture.size ? `<h4 class="gt-rank-title">Likely furniture</h4>
        ${chosen.length ? rankedList('furniture', obj, chosen, busy, furnitureLabel) : '<p class="gt-help gt-room-empty">No furniture selected. Add the furniture it is most likely on or inside.</p>'}
        <details class="gt-room-add" data-add="furniture:${esc(obj)}"${openFurniture ? ' open' : ''}><summary>Add furniture to check (${furnitureCount})</summary>
          ${groups || '<span class="gt-help">All furniture is in the list.</span>'}
        </details>` : ''}
        <span class="gt-help gt-room-feedback" role="status">${esc(roomFeedback.get(obj) || '')}</span></div>`;
    }).join('');
    container.querySelectorAll('input').forEach(input => input.onchange = () => {
      const {kind, object, item} = input.dataset;
      const previous = (state[KINDS[kind].choices] || {})[object] || [];
      saveRanking(kind, object, input.checked ? [...previous, item] : previous.filter(value => value !== item), roomFocus(input));
    });
    container.querySelectorAll('[data-action="up"], [data-action="down"]').forEach(button => button.onclick = () => {
      const {kind, object, item, action} = button.dataset;
      const items = [...((state[KINDS[kind].choices] || {})[object] || [])];
      const index = items.indexOf(item), next = index + (action === 'up' ? -1 : 1);
      if (index < 0 || next < 0 || next >= items.length) return;
      [items[index], items[next]] = [items[next], items[index]];
      saveRanking(kind, object, items, roomFocus(button));
    });
    if (focus) {
      const controls = [...container.querySelectorAll('[data-action]')].filter(node => node.dataset.object === focus.object && node.dataset.kind === focus.kind && !node.disabled);
      const target = controls.find(node => node.dataset.item === focus.item && node.dataset.action === focus.action)
        || controls.find(node => node.dataset.item === focus.item)
        || controls.find(node => node.closest('.gt-room-ranking'));
      if (target && (!target.closest('details') || target.closest('details').open)) target.focus({preventScroll:true});
      else container.querySelector(`details[data-add="${CSS.escape(`${focus.kind}:${focus.object}`)}"] summary`)?.focus({preventScroll:true});
    }
  }
  async function saveRanking(kind, object, items, focus) {
    if (roomSaving.has(object)) return;
    const {choices, endpoint, field} = KINDS[kind];
    state[choices] = state[choices] || {};
    const id = variant, token = generation, previous = [...(state[choices][object] || [])];
    state[choices][object] = items;
    roomSaving.add(object); roomRevision++; roomFeedback.set(object, 'Saving…');
    renderRooms(focus);
    try {
      await request(`${api(id)}/${endpoint}`, {object, [field]: items});
      if (token !== generation) return;
      roomFeedback.set(object, kind === 'room' ? 'Room order saved.' : 'Furniture order saved.');
    } catch (e) {
      if (token !== generation) return;
      state[choices][object] = previous;
      roomFeedback.set(object, `Not saved: ${e.message}`);
    } finally {
      if (token === generation) {
        roomSaving.delete(object); roomRevision++;
        const activeCard = document.activeElement?.closest('.gt-room-card');
        renderRooms(activeCard?.dataset.object === object ? focus : null);
      }
    }
  }
  async function poll(force = false) {
    if (!variant || (fetching && !force)) return;
    const id = variant, token = generation;
    fetching = true;
    const revision = roomRevision;
    try {
      const next = await request(api(id));
      if (token !== generation) return;
      // An older poll must not overwrite a room edit that is saving or just finished.
      if (state && (roomSaving.size || revision !== roomRevision)) { next.room_choices = state.room_choices; next.furniture_choices = state.furniture_choices; }
      // A sandbox already open here is a recording in progress: show it, not the saved video.
      if (!state && next.session.variant === id && next.session.mode) { sandboxView = true; displayedRun = next.ground_truth?.id || null; }
      state = next;
      // Room annotation does not depend on starting or rendering the simulator.
      renderRooms(); renderStatus(); renderNotes();
      if (loadError) error();
      loadError = false;
      el('gtConnection').hidden = true;
    } catch (e) {
      if (token === generation) {
        loadError = true;
        error(`Ground truth unavailable: ${e.message}`);
        el('gtStatus').textContent = 'Ground truth could not be loaded.';
        el('gtConnection').hidden = false;
        el('gtRun').disabled = true; el('gtSandbox').disabled = true; el('gtRecord').disabled = true;
        if (!state) el('gtRooms').textContent = 'Room choices load from the viewer server. Use the connection options above to reconnect.';
      }
    }
    finally { if (token === generation) fetching = false; }
  }
  window.renderGroundTruth = selected => {
    if (variant === selected.id) return;
    if (el('gtFrame')?.dataset.blob) URL.revokeObjectURL(el('gtFrame').dataset.blob);
    noteDirty.forEach(scope => { clearTimeout(noteTimers[scope]); saveNote(scope); }); noteTimers = {}; noteDirty.clear();
    sandboxView = false; displayedRun = null; chapters = []; lastEvidence = '';
    variant = selected.id; generation++; state = null; fetching = false; posting = false; frameLoading = false;
    roomSaving.clear(); roomFeedback.clear(); roomRevision++;
    lastGraph = ''; lastRooms = ''; lastActions = ''; lastVideo = ''; lastReports = ''; lastSteps = ''; lastPacks = ''; loadError = false;
    el('groundTruthBody').innerHTML = `
      <div id="gtError" class="gt-error" role="alert" hidden></div><div id="gtBlocked" class="gt-error" hidden></div>
      <div id="gtConnection" class="gt-toolbar" hidden><button type="button" id="gtRetry">Retry connection</button><a id="gtServerLink">Open viewer server on port 8000</a></div>
      <details class="gt-notes" open><summary>Assumptions</summary>
        <div class="gt-notes-grid">${['task', 'all'].map(scope => `<label>${scope === 'task' ? `This task (${esc(selected.id.split('-')[0])}, every variant)` : 'All tasks (shared by T1–T7)'}
          <textarea id="gtNote-${scope}" data-scope="${scope}" rows="5" maxlength="20000" placeholder="${scope === 'task' ? 'e.g. Assumed the jug counts as full after one Fill at the kitchen sink.' : 'e.g. Assumed the bedroom lamp is the only light that counts.'}"></textarea>
          <span class="gt-help" id="gtNoteStatus-${scope}" role="status"></span></label>`).join('')}</div>
        <p class="gt-help">Saved automatically for everyone using this viewer. <a href="../ground_truth/assumptions.md" target="_blank" rel="noopener">All assumptions</a></p>
      </details>
      <div id="gtSaved" class="gt-help"></div>
      <section class="gt-evidence" aria-label="Saved ground-truth criteria">
        <div class="gt-evidence-heading"><h3>Saved ground-truth criteria</h3><span id="gtEvidenceBadge" class="gt-evidence-badge">Loading</span></div>
        <p id="gtEvidenceSummary" role="status">Loading saved results…</p>
        <ol id="gtEvidenceRows" class="gt-evidence-rows"></ol><p id="gtEvidenceBasis" class="gt-help"></p>
      </section>
      <div id="gtDownloads" class="gt-help"></div><p id="gtVideoStatus" class="gt-help" role="status"></p>
      <section id="gtRecording" class="gt-playback" aria-label="Saved ground-truth playback" hidden>
        <div class="gt-playback-heading"><h3>Ground-truth video</h3><a id="gtDownloadVideo" download>Download MP4</a></div>
        <video id="gtVideo" controls playsinline preload="auto"></video>
        <p id="gtNowPlaying" class="gt-now-playing"></p>
        <details class="gt-timeline" open><summary>Steps in recording order · select a step to play from it</summary><ol id="gtChapters"></ol></details>
      </section>
      <details class="gt-log"><summary>Ground-truth actions</summary><ol id="gtSavedActions" class="gt-history"></ol></details>
      <p class="gt-help"><a href="../ground_truth/index.html" target="_blank" rel="noopener">All saved ground truths</a></p>
      <div class="gt-toolbar"><button id="gtSandbox" type="button" disabled>Open sandbox</button><button id="gtClose" type="button" hidden>Close sandbox</button><button id="gtShowVideo" type="button" hidden>Return to saved video</button><button id="gtRecord" type="button" class="gt-primary" disabled>Record ground truth</button></div>
      <p id="gtSessions" class="gt-help" role="status"></p>
      <p id="gtPracticeHelp" class="gt-help">Opening the sandbox loads the episode at its exact start and starts recording. Every action you run, including failed attempts and applied packs, is part of the recording. Save it once the task requirements are met; it replaces this variant's previous ground truth. Nothing is replayed, so if something goes wrong, reset the sandbox to start over.</p>
      <p id="gtStatus" role="status">Loading ground truth…</p><div id="gtResult" role="status" hidden></div><div id="gtProgress" class="gt-completion" role="status" aria-live="polite" hidden><strong id="gtProgressTitle"></strong><p id="gtProgressDetail"></p><progress id="gtProgressBar" max="100" value="0" aria-label="Task completion"></progress></div>
      <div id="gtWorkspace"><div class="gt-layout"><div><h3>Apartment and objects</h3><div id="gtTree" class="gt-tree"></div><p class="gt-help">Click an entity to use it as the skill target. Names match the spec where available.</p></div>
      <div><img id="gtFrame" class="gt-camera" alt="Live robot and human cameras from this episode" hidden><p id="gtFrameHelp" class="gt-help">The live camera appears when the sandbox opens.</p>
      <div class="gt-controls"><label>Skill<select id="gtSkill">${['Navigate','Explore','Open','Close','Pick','Place','Fill','Pour','Clean','PowerOn','PowerOff'].map(s=>`<option>${s}</option>`).join('')}</select></label>
      <label>Target<input type="text" id="gtTarget" list="gtEntities" placeholder="Object, furniture, or room"><datalist id="gtEntities"></datalist></label>
      <label id="gtPlace" class="gt-wide" hidden>Place arguments<input type="text" id="gtPlaceArgs" placeholder="jug_0,on,table_0,none,none"><span class="gt-help">object, on/within, destination, none/next_to, reference/none</span></label>
      <button type="button" id="gtRun" class="gt-wide" disabled>Try skill in sandbox</button></div><div id="gtReports"></div></div></div>
      <details class="gt-log" open><summary>Recorded actions</summary><p class="gt-help">Every action since the sandbox opened, in order; all of them are saved with the ground truth. Checkboxes only choose which steps go into a pack. Failed attempts start unchecked.</p><ol id="gtStepList" class="gt-steps"></ol></details>
      <details class="gt-log" open><summary>Action packs for ${esc(selected.id.split('-')[0])}</summary><p class="gt-help">Save the checked steps as a named pack, then apply it in the sandbox of any ${esc(selected.id.split('-')[0])} variant, including this one. Applying runs the steps in order after the current ones and stops at the first step that fails. Packs are saved for everyone using this viewer.</p>
      <div class="gt-pack-save"><input type="text" id="gtPackName" maxlength="80" placeholder="Pack name, e.g. Fill jug at the bathroom sink" aria-label="Pack name"><button type="button" id="gtPackSave" disabled>Save checked steps as pack</button></div>
      <ol id="gtPackList" class="gt-steps gt-packs"></ol></details></div>
      <h3>Initial robot memory</h3><pre id="gtMemory" class="gt-criteria"></pre>
      <h3>Completion requirements</h3><pre id="gtCriteria" class="gt-criteria"></pre>
      <h3>Most likely rooms and furniture for each target</h3><p class="gt-help">Choose the rooms worth visiting and the furniture each target is most likely on or inside, then use ↑ and ↓ to set the order. 1 is checked first. Uncheck an entry to drop it. Everything saves automatically. Rooms are shared wherever the target appears. Furniture is shared by every variant in the same apartment.</p><div id="gtRooms" class="gt-rooms">Loading room choices…</div>`;
    el('gtSandbox').onclick = () => {
      const s = state?.session;
      if (!sandboxView && s?.variant === variant && s.mode === 'sandbox') { sandboxView = true; renderStatus(); return; }
      if (s?.variant === variant && s.mode && s.steps.length && !confirm(`Reset to the exact start? The current recording (${plural(s.steps.length, 'action')}) is discarded unless you saved it.`)) return;
      post('sandbox', {});
    };
    el('gtClose').onclick = () => {
      if (confirm('Close this sandbox and free its slot? Unsaved actions will be discarded. Saved ground truth is kept.')) post('close', {});
    };
    el('gtShowVideo').onclick = () => { sandboxView = false; renderStatus(); };
    el('gtVideo').ontimeupdate = updatePlayback;
    el('gtVideo').onseeked = updatePlayback;
    el('gtRecord').onclick = () => {
      const n = state?.session?.steps?.length || 0, replace = state?.ground_truth ? ' It replaces the saved ground truth.' : '';
      if (confirm(`Save these ${plural(n, 'action')} as this variant's ground truth?${replace}`)) post('record', {});
    };
    el('gtPackSave').onclick = async () => {
      const name = el('gtPackName').value.trim();
      if (!name) { error('Name the pack before saving it.'); el('gtPackName').focus(); return; }
      if ((state?.packs || []).some(p => p.name === name) && !confirm(`Replace the existing pack “${name}” with the checked steps?`)) return;
      await post('packs', {save:name});
      if (!el('gtError').textContent) el('gtPackName').value = '';
    };
    document.querySelectorAll('#groundTruthBody textarea[data-scope]').forEach(box => box.oninput = () => queueNote(box.dataset.scope));
    el('gtRetry').onclick = () => poll(true);
    const serverURL = new URL(location.href);
    serverURL.port = '8000';
    serverURL.hash = selected.id;
    el('gtServerLink').href = serverURL.href;
    el('gtSkill').onchange = () => { el('gtPlace').hidden = el('gtSkill').value !== 'Place'; };
    el('gtRun').onclick = () => {
      const skill = el('gtSkill').value, target = (skill === 'Place' ? el('gtPlaceArgs').value : el('gtTarget').value).trim();
      if (!target) { error('Choose a target or enter Place arguments.'); return; }
      post('action', {request_id:crypto.randomUUID(), skill, target});
    };
    poll();
  };
  setInterval(() => { if (!document.hidden) poll(); }, 350);
  setInterval(async () => {
    const s = state?.session;
    if (s?.variant !== variant || !s.snapshot?.has_frame || document.hidden || frameLoading || !el('gtFrame') || el('gtWorkspace').hidden) return;
    frameLoading = true;
    const token = generation;
    try {
      const response = await fetch(`${api(variant)}/frame?t=${Date.now()}${s.id ? '&session_id=' + encodeURIComponent(s.id) : ''}`, {cache:'no-store'});
      if (!response.ok) return;
      const blob = await response.blob();
      if (token !== generation) return;
      const img = el('gtFrame'), previous = img.dataset.blob;
      const objectURL = URL.createObjectURL(blob);
      const decoded = new Image(); decoded.src = objectURL;
      try { await decoded.decode(); } catch (_) { URL.revokeObjectURL(objectURL); return; }
      if (token !== generation || s.id !== state?.session?.id) { URL.revokeObjectURL(objectURL); return; }
      img.src = objectURL; img.dataset.blob = objectURL; img.hidden = false;
      el('gtFrameHelp').textContent = 'Live camera · recording · robot controls use agent 0';
      if (previous) URL.revokeObjectURL(previous);
    } catch (_) { /* Retry on the next frame tick after a connection interruption. */ }
    finally { frameLoading = false; }
  }, 350);
})();
