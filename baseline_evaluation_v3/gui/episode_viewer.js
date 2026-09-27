/* Saved episode review. Uses actual rendered placements, never object-preview GIFs. */
(() => {
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let request = 0;
  let controller;
  const path = value => '../' + String(value).split('/').map(encodeURIComponent).join('/');
  async function json(url, signal) {
    const response = await fetch(url, {cache:'no-store', signal});
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    return response.json();
  }
  window.renderEpisode = async variant => {
    const token = ++request;
    if (controller) controller.abort();
    controller = new AbortController();
    const signal = controller.signal;
    const body = document.getElementById('episodeBody');
    body.textContent = 'Loading episode review…';
    try {
      const [audit,index] = await Promise.all([
        json('../generation/audit.json',signal), json('../generation/episodes.json',signal)
      ]);
      if (token !== request) return;
      const item = index.variants[variant.id];
      const issues = audit.issues.filter(i => i.status === 'active' && i.variants.includes(variant.id));
      const issuesHTML = issues.length ? `<details class="episode-issues" open><summary>${issues.length} open issue${issues.length === 1 ? '' : 's'}</summary>` +
        '<ul>' + issues.map(i => `<li><b>${escape(i.id)}</b>: ${escape(i.message)}</li>`).join('') + '</ul></details>' : '';
      if (!item || item.status !== 'generated') {
        const label = {blocked:'Generation blocked pending review of the spec issues below.', pending:'Queued for generation.',failed:'Generation failed without changing the spec.'}[item?.status] || 'No generated episode available.';
        body.innerHTML = `<p>${escape(label)}</p>${item?.error ? `<details><summary>Generation error</summary><pre style="white-space:pre-wrap">${escape(item.error)}</pre></details>` : ''}${issuesHTML}<button type="button" id="episodeRefresh">Refresh status</button>`;
        document.getElementById('episodeRefresh').onclick = () => window.renderEpisode(variant);
        return;
      }
      const data = await json(path(item.review),signal);
      if (token !== request) return;
      body.innerHTML = `<p><b>${escape(data.variant)}</b> · ${data.objects.length} objects · saved initial placements</p>` +
        `<div class="episode-links"><a href="${path(data.dataset)}">Download dataset</a><a href="${path(item.review)}" target="_blank" rel="noopener">Placement validation data</a></div>` +
        `<p class="episode-caption"><span class="episode-badge">Generated</span>${item.runtime_verified ? '<span class="episode-badge">Runtime verified</span>' : ''} Assets, initial states, furniture and rooms were checked after physics settling. Select an object to see its saved placement.</p>` +
        `<div class="episode-layout"><div class="episode-objects" aria-label="Episode objects"></div><div><div class="episode-controls"><label>View <select id="episodeAngle"></select></label><button id="episodeZoomIn" type="button" aria-label="Zoom in">+</button><button id="episodeZoomOut" type="button" aria-label="Zoom out">−</button><button id="episodeFit" type="button">Fit</button><a id="episodeImageLink" target="_blank" rel="noopener">Open image</a></div><div class="episode-viewport"><img id="episodeImage" draggable="false" alt=""></div><p class="episode-caption" id="episodeVisibility"></p><div class="episode-object-info"></div></div></div>${issuesHTML}`;
      const list = body.querySelector('.episode-objects');
      const picture = body.querySelector('#episodeImage');
      const viewport = body.querySelector('.episode-viewport');
      const select = body.querySelector('#episodeAngle');
      let current, scale=1, x=0, y=0, drag;
      const transform = () => { picture.style.transform = `translate(${x}px,${y}px) scale(${scale})`; };
      const fit = () => { scale=1; x=0; y=0; transform(); };
      const showImage = () => {
        const view = current.views[Number(select.value)];
        const url = path(data.image_root + '/' + view.image);
        picture.src=url; picture.alt=`${current.entity}: actual initial placement, ${view.label}`;
        body.querySelector('#episodeImageLink').href=url;
        body.querySelector('#episodeVisibility').textContent=view.target_visible ? (view.label.startsWith('Interior') ? 'Camera inside the closed container; saved objects and door positions are unchanged. Scroll to zoom; drag to pan.' : 'Object visible from this camera. Scroll to zoom; drag to pan.') : 'Object may be occluded by furniture from this camera. Try another angle; closed containers remain closed in these images.';
        fit();
      };
      const spatial = s => `${s.relation} ${s.furniture} (${s.room})`;
      const stateWords = states => Object.entries(states || {}).map(([k, val]) => ({
        is_clean: val ? 'clean' : 'dirty', is_filled: val ? 'filled' : 'empty', is_powered_on: val ? 'on' : 'off',
      }[k] || `${k}=${val}`)).join(', ') || 'none';
      for (const object of data.objects) {
        const button=document.createElement('button');
        button.type='button'; button.className='episode-object'; button.setAttribute('aria-pressed','false');
        button.innerHTML=`${escape(object.entity)}<small>${escape(spatial(object.initial))}</small>`;
        button.onclick=() => {
          current=object;
          list.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
          select.innerHTML=object.views.map((v,i)=>`<option value="${i}">${escape(v.label)}${v.target_visible?' · visible':''}</option>`).join('');
          body.querySelector('.episode-object-info').innerHTML=`<h3>${escape(object.entity)}</h3><div><b>Start:</b> ${escape(spatial(object.initial))}</div><div><b>Expected destination:</b> ${escape(spatial(object.expected))}</div><div><b>Saved position (x, y, z):</b> ${object.position.map(n=>n.toFixed(3)).join(', ')}</div><div><b>Surface:</b> ${escape(String(object.receptacle || 'floor').split('|').pop())}</div><div><b>Asset:</b> ${escape(object.asset)}</div><div><b>Initial states:</b> ${escape(stateWords(object.initial.states))}</div>`;
          showImage();
        };
        list.appendChild(button);
      }
      select.onchange=showImage;
      const zoom=amount=>{scale=Math.max(1,Math.min(8,scale*amount));transform();};
      body.querySelector('#episodeZoomIn').onclick=()=>zoom(1.3);
      body.querySelector('#episodeZoomOut').onclick=()=>zoom(1/1.3);
      body.querySelector('#episodeFit').onclick=fit;
      viewport.onwheel=e=>{e.preventDefault();zoom(e.deltaY<0?1.15:1/1.15);};
      viewport.onpointerdown=e=>{drag=[e.clientX-x,e.clientY-y];viewport.setPointerCapture(e.pointerId);};
      viewport.onpointermove=e=>{if(drag){x=e.clientX-drag[0];y=e.clientY-drag[1];transform();}};
      viewport.onpointerup=viewport.onpointercancel=()=>{drag=null;};
      picture.onerror=()=>{body.querySelector('#episodeVisibility').textContent='Saved placement image could not be loaded.';};
      list.querySelector('button')?.click();
    } catch(error) {
      if (error.name==='AbortError' || token!==request) return;
      body.textContent=`Episode review unavailable: ${error.message}`;
    }
  };
})();
