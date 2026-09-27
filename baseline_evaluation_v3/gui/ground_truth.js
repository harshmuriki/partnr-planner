/* Exact-episode sandbox practice and replayed, server-saved ground truth (one per variant). */
(() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let variant = null, state = null, generation = 0, fetching = false, posting = false, frameLoading = false;
  let lastGraph = '', lastRooms = '', lastActions = '', lastVideo = '', lastReports = '', lastSteps = '', loadError = false;
  let roomRevision = 0;
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
      await request(`${api(id)}/${operation}`, payload);
      if (token !== generation) return;
      error();
      await poll(true);
    } catch (e) { if (token === generation) error(e.message); }
    finally { posting = false; if (token === generation) renderStatus(); }
  }
  const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
  const stepLabel = a => `${esc(a.skill)} ${esc(a.skill === 'ReportAbsence' ? '' : a.target)}`;
  function renderStatus() {
    if (!state || !el('gtSandbox')) return;
    const s = state.session, here = s.variant === variant, busy = s.busy || posting;
    const sandbox = here && s.mode === 'sandbox';
    const kept = s.steps.filter(step => step.keep).length;
    el('gtSandbox').disabled = busy || !state.meta.available;
    el('gtSandbox').textContent = here && s.mode ? 'Reset sandbox to the start' : 'Open sandbox';
    el('gtRecord').disabled = busy || !sandbox || !kept;
    el('gtRecord').textContent = `Record ground truth · replay ${plural(kept, 'checked step')}`;
    el('gtRun').disabled = busy || !sandbox;
    el('gtStatus').textContent = here && s.phase ? s.phase :
      here && s.error ? s.error :
      sandbox ? 'Sandbox: try anything. Nothing is saved until you record.' :
      s.variant && !here ? `The simulator is on ${s.variant}. Open the sandbox here to switch.` :
      s.busy ? 'The simulator is busy.' : 'Open the sandbox to load this episode at its exact start.';
    const result = s.last_result;
    el('gtResult').hidden = !result;
    if (result) { el('gtResult').textContent = result.message; el('gtResult').className = result.ok ? 'gt-saved' : 'gt-error'; }
    const evaluation = here ? s.snapshot?.evaluation : null;
    const reported = r => s.steps.some(a => a.skill === 'ReportAbsence' && a.target === r);
    el('gtProgress').textContent = evaluation ? `Spec evaluation in the simulator now: ${Math.round(evaluation.percent_complete * 100)}%${evaluation.success ? ' · complete' : ''}${state.meta.reports.length ? (state.meta.reports.every(reported) ? ' · absence reported' : ' · absence report also required') : ''}` : '';

    const gt = state.ground_truth;
    el('gtSaved').innerHTML = gt
      ? `<b>Saved ground truth</b> · ${plural(gt.action_count, 'action')} · ${gt.sim_steps} simulator steps${gt.steps_complete ? '' : ' (partial)'} · ${esc(new Date(gt.completed_at).toLocaleString())}${gt.stale ? '<br>The spec or episode changed since this was saved. Record it again.' : ''}`
      : 'No ground truth saved for this variant yet.';
    el('gtSaved').className = gt ? 'gt-saved' : 'gt-help';
    el('gtDownloads').innerHTML = gt ? `<a href="${esc(gt.artifacts.data)}" target="_blank" rel="noopener">Run data (JSON)</a> · <a href="${esc(gt.artifacts.actions)}" download>Actions (CSV)</a> · <a href="${esc(gt.artifacts.files)}" target="_blank" rel="noopener">Folder</a>` : '';
    el('gtVideoStatus').textContent = state.archive_error ? `Archive export needs attention: ${state.archive_error}` :
      gt?.artifact_status === 'error' ? `Video unavailable: ${gt.artifact_error}. The actions are still saved.` : '';
    const video = gt?.artifacts.video || '';
    if (video !== lastVideo) {
      lastVideo = video;
      el('gtRecording').hidden = !video;
      if (video) { el('gtVideo').src = video; el('gtDownloadVideo').href = video; }
      else { el('gtVideo').removeAttribute('src'); el('gtVideo').load(); }
    }
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
      el('gtStepList').innerHTML = list.map((a, i) => `<li class="${a.keep ? '' : 'gt-skipped'}">
        <label><input type="checkbox" data-step="${esc(a.id)}"${a.keep ? ' checked' : ''}${disabled} aria-label="Replay step ${i + 1} when recording"><b>${i + 1}. ${stepLabel(a)}</b></label>
        <button type="button" class="gt-remove" data-remove="${esc(a.id)}"${disabled} aria-label="Remove step ${i + 1}" title="Remove">✕</button>
        <div class="gt-help">${esc(a.result?.response || a.result?.error || '')}${a.result?.skill_steps !== undefined ? ` · ${a.result.skill_steps} simulator steps` : ''}</div></li>`).join('')
        || '<li class="gt-help">No sandbox actions yet.</li>';
      el('gtStepList').querySelectorAll('[data-step]').forEach(box => box.onchange = () => post('steps', {id:box.dataset.step, keep:box.checked}));
      el('gtStepList').querySelectorAll('[data-remove]').forEach(b => b.onclick = () => post('steps', {id:b.dataset.remove, remove:true}));
      el('gtClearSteps').disabled = busy || !sandbox || !list.length;
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
  let noteTimers = {};
  const noteDirty = new Set();
  const noteTime = note => note ? `Saved ${new Date(note.updated_at).toLocaleString()}` : '';
  function renderNotes() {
    ['variant', 'task'].forEach(scope => {
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
    variant = selected.id; generation++; state = null; fetching = false; posting = false; frameLoading = false;
    roomSaving.clear(); roomFeedback.clear(); roomRevision++;
    lastGraph = ''; lastRooms = ''; lastActions = ''; lastVideo = ''; lastReports = ''; lastSteps = ''; loadError = false;
    el('groundTruthBody').innerHTML = `
      <div id="gtError" class="gt-error" role="alert" hidden></div><div id="gtBlocked" class="gt-error" hidden></div>
      <div id="gtConnection" class="gt-toolbar" hidden><button type="button" id="gtRetry">Retry connection</button><a id="gtServerLink">Open viewer server on port 8000</a></div>
      <details class="gt-notes" open><summary>Assumptions</summary>
        <div class="gt-notes-grid">${['variant', 'task'].map(scope => `<label>${scope === 'variant' ? `This variant (${esc(selected.id)})` : `All of ${esc(selected.id.split('-')[0])} (shared by every variant)`}
          <textarea id="gtNote-${scope}" data-scope="${scope}" rows="5" maxlength="20000" placeholder="${scope === 'variant' ? 'e.g. Assumed the jug counts as full after one Fill at the kitchen sink.' : 'e.g. Assumed the bedroom lamp is the only light that counts.'}"></textarea>
          <span class="gt-help" id="gtNoteStatus-${scope}" role="status"></span></label>`).join('')}</div>
        <p class="gt-help">Saved automatically for everyone using this viewer. <a href="../ground_truth/assumptions.md" target="_blank" rel="noopener">All assumptions</a></p>
      </details>
      <div id="gtSaved" class="gt-help"></div>
      <div id="gtDownloads" class="gt-help"></div><p id="gtVideoStatus" class="gt-help" role="status"></p>
      <details id="gtRecording" hidden><summary>Ground-truth video</summary><video id="gtVideo" controls preload="metadata" style="width:100%"></video><a id="gtDownloadVideo" download>Download video (MP4)</a></details>
      <details class="gt-log"><summary>Ground-truth actions</summary><ol id="gtSavedActions" class="gt-history"></ol></details>
      <p class="gt-help"><a href="../ground_truth/index.html" target="_blank" rel="noopener">All saved ground truths</a></p>
      <div class="gt-toolbar"><button id="gtSandbox" type="button" disabled>Open sandbox</button><button id="gtRecord" type="button" class="gt-primary" disabled>Record ground truth</button></div>
      <p class="gt-help">Practice freely in the sandbox. Record resets the episode to its exact start and replays only the checked steps. It saves only if the replay completes the spec, and replaces this variant's previous ground truth.</p>
      <p id="gtStatus" role="status">Loading ground truth…</p><div id="gtResult" role="status" hidden></div><p id="gtProgress" class="gt-help"></p>
      <div class="gt-layout"><div><h3>Apartment and objects</h3><div id="gtTree" class="gt-tree"></div><p class="gt-help">Click an entity to use it as the skill target. Names match the spec where available.</p></div>
      <div><img id="gtFrame" class="gt-camera" alt="Live robot and human cameras from this episode" hidden><p id="gtFrameHelp" class="gt-help">The live camera appears when the sandbox opens.</p>
      <div class="gt-controls"><label>Skill<select id="gtSkill">${['Navigate','Explore','Open','Close','Pick','Place','Fill','Pour','Clean','PowerOn','PowerOff'].map(s=>`<option>${s}</option>`).join('')}</select></label>
      <label>Target<input type="text" id="gtTarget" list="gtEntities" placeholder="Object, furniture, or room"><datalist id="gtEntities"></datalist></label>
      <label id="gtPlace" class="gt-wide" hidden>Place arguments<input type="text" id="gtPlaceArgs" placeholder="jug_0,on,table_0,none,none"><span class="gt-help">object, on/within, destination, none/next_to, reference/none</span></label>
      <button type="button" id="gtRun" class="gt-wide" disabled>Try skill in sandbox</button></div><div id="gtReports"></div></div></div>
      <details class="gt-log" open><summary>Sandbox steps</summary><p class="gt-help">Not saved. Checked steps are replayed, in order, when you record. Failed attempts start unchecked.</p><ol id="gtStepList" class="gt-steps"></ol><button type="button" id="gtClearSteps" disabled>Clear all steps</button></details>
      <h3>Initial robot memory</h3><pre id="gtMemory" class="gt-criteria"></pre>
      <h3>Completion requirements</h3><pre id="gtCriteria" class="gt-criteria"></pre>
      <h3>Most likely rooms and furniture for each target</h3><p class="gt-help">Choose the rooms worth visiting and the furniture each target is most likely on or inside, then use ↑ and ↓ to set the order. 1 is checked first. Uncheck an entry to drop it. Everything saves automatically. Rooms are shared wherever the target appears. Furniture is shared by every variant in the same apartment.</p><div id="gtRooms" class="gt-rooms">Loading room choices…</div>`;
    el('gtSandbox').onclick = () => post('sandbox', {});
    el('gtRecord').onclick = () => {
      const replace = state?.ground_truth ? ' If the replay completes the task, it replaces the saved ground truth.' : '';
      if (confirm(`Reset the episode to its exact start and replay the checked steps?${replace}`)) post('record', {});
    };
    el('gtClearSteps').onclick = () => { if (confirm('Remove every sandbox step?')) post('steps', {clear:true}); };
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
  setInterval(() => { if (!document.hidden) poll(); }, 1500);
  setInterval(async () => {
    const s = state?.session;
    if (s?.variant !== variant || !s.snapshot?.has_frame || document.hidden || frameLoading || !el('gtFrame')) return;
    frameLoading = true;
    const token = generation;
    try {
      const response = await fetch(`${api(variant)}/frame?t=${Date.now()}`, {cache:'no-store'});
      if (!response.ok) return;
      const blob = await response.blob();
      if (token !== generation) return;
      const img = el('gtFrame'), previous = img.dataset.blob;
      const objectURL = URL.createObjectURL(blob);
      img.src = objectURL; img.dataset.blob = objectURL; img.hidden = false;
      el('gtFrameHelp').textContent = s.mode === 'recording' ? 'Live camera · recording replay' : 'Live camera · sandbox (not saved) · robot controls use agent 0';
      if (previous) URL.revokeObjectURL(previous);
    } catch (_) { /* Retry on the next frame tick after a connection interruption. */ }
    finally { frameLoading = false; }
  }, 350);
})();
