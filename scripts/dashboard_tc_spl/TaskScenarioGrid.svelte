<script>
  import {findingMix} from './finding_mix.js';
  export let data;
  let task='1', metric='success', chosen=null, grids=[], findingCategory='failure';
  const memories=[{key:'ACC',label:'Accurate'},{key:'INC',label:'Incomplete'},{key:'OUT',label:'Outdated'}];
  const groups=[
    {label:'Object Availability',style:'availability',columns:[{label:'Target Exists',axis:'BASE'},{label:'Substitute Available',axis:'SUB'},{label:'No Suitable Object',axis:'ABS'}]},
    {label:'Object Containment',style:'containment',columns:[{label:'On Surface',axis:'BASE'},{label:'Inside Closed Receptacle',axis:'CON'}]},
    {label:'Distractor Presence',style:'distractors',columns:[{label:'None',axis:'BASE'},{label:'Present',axis:'DIS'}]}
  ];
  const columns=groups.flatMap(g=>g.columns);
  const tasks=[...new Set(data.rows.map(r=>r.task))].sort((a,b)=>a-b);
  const pct=v=>v===null?'—':`${(100*v).toFixed(1)}%`;
  const palette=['#7B3294','#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#56B4E9','#8C564B','#555555'];
  $: annotations=new Map((data.rule_findings?.runs??[]).map(r=>[r.id,r]));
  $: findingLabels=[...new Set((data.rule_findings?.runs??[]).flatMap(r=>r.findings.filter(f=>f[0]===findingCategory).map(f=>f[1])))].sort();
  $: findingColors=Object.fromEntries(findingLabels.map((label,i)=>[label,palette[i%palette.length]]));
  const nice=label=>label==='__none'?'No finding':label==='__missing'?'No annotation':label==='placement_error'?'incorrect placement':label.replaceAll('_',' ');
  const color=label=>label==='__none'?'#cbd5e1':label==='__missing'?'#f1f5f9':findingColors[label];
  const evidence=r=>findingCategory==='failure'&&r.success===1?[]:(annotations.get(r.id)?.findings??[]).filter(f=>f[0]===findingCategory);
  const mixTitle=mix=>mix.total?mix.segments.map(s=>`${nice(s.label)}: ${pct(s.share)} allocated share; found in ${s.runs}/${mix.total} runs`).join(' | '):'No scored unsuccessful runs';

  $: metricLabel=metric==='success'?'Task success rate':'Mean task completion';
  function cell(rows,axis,memory) {
    const matching=rows.filter(r=>r.uncertainty===axis&&r.memory===memory);
    const known=matching.filter(r=>r[metric]!=null);
    const value=known.length?known.reduce((sum,r)=>sum+(metric==='success'?Number(r.success===1):r.completion),0)/known.length:null;
    return {mix:findingCategory==='off'?null:findingMix(matching,annotations,findingCategory),rows:matching,n:known.length,value,axis,memory,total:matching.length};
  }
  $: {metric;findingCategory;annotations; grids=(task==='all'?tasks:[Number(task)]).map(t=>({task:t,baselines:data.baselines.map(b=>{
    const rows=data.rows.filter(r=>r.task===t&&r.condition===b.key);
    return {...b,total:rows.length,memories:memories.map(m=>({...m,cells:columns.map(c=>cell(rows,c.axis,m.key))}))};
  })}));}
  $: {task;metric;findingCategory;chosen=null;}
  function open(c,t,b) {chosen={...c,task:t,baseline:b.label};}
</script>
<section id="task-scenario-grid" class="panel">
  <h2>Task results by scenario and initial memory</h2>
  <p class="intro">Compare each baseline using the scenario viewer layout. Each cell averages the available repetitions for one task variant.</p>
  <div class="controls">
    <label>Task <select aria-label="Scenario grid task" bind:value={task}><option value="all">All tasks</option>{#each tasks as t}<option value={String(t)}>Task {t}</option>{/each}</select></label>
    <label>Metric <select aria-label="Scenario grid metric" bind:value={metric}><option value="success">Task success rate</option><option value="completion">Mean task completion</option></select></label>
    <label>Cell findings <select aria-label="Scenario grid findings" bind:value={findingCategory}><option value="off">None</option><option value="failure">Failure findings</option><option value="inefficiency">Inefficiency findings</option></select></label>
    <span class="scale">0% <i></i> 100%</span>
  </div>
  <p class="intro small">Success = fraction of runs that fully succeed. Completion = mean fraction of benchmark criteria achieved. Missing values are excluded; coverage appears in every cell. BASE is repeated under three headings for comparison and is counted only once in run totals.</p>
  {#if findingCategory!=='off'}
    <div class="finding-key" aria-label="Cell finding colors">{#each findingLabels as label}<span><i style={`background:${color(label)}`}></i>{nice(label)}</span>{/each}<span><i style="background:#cbd5e1"></i>No finding</span><span><i style="background:#f1f5f9"></i>No annotation</span></div>
    <p class="intro small">Colored strips show {findingCategory==='failure'?'unsuccessful runs':'all runs'}. Each run is split equally across its distinct {findingCategory} labels; widths total 100%, including runs with no finding. These are associated findings, not causes. Hover for shares and counts.</p>
  {/if}
  {#each grids as g}<div class="task-block"><h3>Task {g.task}</h3>
    {#each g.baselines as b}<article class="scenario-card" data-task={g.task} data-baseline={b.key}>
      <h4>{b.label}<span>{b.total} unique runs · {metricLabel}</span></h4>
      <div class="grid-scroll"><table class="scenario-grid">
        <thead><tr><th rowspan="2" scope="col"><span class="sr-only">Initial memory</span></th>{#each groups as group}<th colspan={group.columns.length} class={group.style} scope="colgroup">{group.label}</th>{/each}</tr><tr>{#each columns as c}<th scope="col" class="column-label">{c.label}</th>{/each}</tr></thead>
        <tbody>{#each b.memories as m}<tr><th scope="row" class={`memory ${m.key}`}>{m.label}</th>{#each m.cells as c}<td><button class="scenario-cell" data-scenario={c.axis} data-memory={m.key} data-value={c.value??''} data-n={c.n} data-total={c.total} disabled={!c.total} on:click={()=>open(c,g.task,b)} style={c.value===null?'':`background:hsl(204 85% ${97-62*c.value}%);color:${c.value>.6?'white':'#172033'}`} aria-label={`Task ${g.task}, ${b.label}, ${m.label}, ${c.axis}: ${pct(c.value)}, ${c.n} of ${c.total} runs`}>
          <span class="axis-code">{c.axis}</span><strong>{pct(c.value)}</strong><small>{c.total?`${c.n}/${c.total} runs`:'No runs'}</small>
          {#if findingCategory!=='off'&&c.total}
            <span class="finding-strip" data-category={findingCategory} data-eligible={c.mix.total} role="img" aria-label={mixTitle(c.mix)} title={mixTitle(c.mix)}>
              {#if c.mix.total}{#each c.mix.segments as segment}<span class="finding-segment" data-label={segment.label} data-share={segment.share} data-count={segment.runs} style={`width:${100*segment.share}%;background:${color(segment.label)}`} title={`${nice(segment.label)}: ${pct(segment.share)} allocated share; found in ${segment.runs}/${c.mix.total} runs`}></span>{/each}{:else}<span class="finding-empty">No unsuccessful runs</span>{/if}
            </span>
          {/if}
        </button></td>{/each}</tr>{/each}</tbody>
      </table></div>
    </article>{/each}
  </div>{/each}
  {#if chosen}<div class="grid-details"><div class="detail-heading"><h3>Task {chosen.task} · {chosen.baseline} · {chosen.memory}-{chosen.axis}</h3><button on:click={()=>chosen=null}>Close details</button></div><p>{metricLabel}: {pct(chosen.value)} · {chosen.n}/{chosen.total} scored runs</p>
    <table class="grid-runs"><thead><tr><th>Repetition</th><th>Execution status</th><th>Task success</th><th>Task completion</th>{#if findingCategory!=='off'}<th>{findingCategory==='failure'?'Failure':'Inefficiency'} evidence</th>{/if}<th>Artifacts</th></tr></thead><tbody>{#each chosen.rows as r}<tr><td>{r.iteration}</td><td>{r.execution_status}</td><td>{r.success==null?'Unknown':r.success===1?'Successful':'Unsuccessful'}</td><td>{pct(r.completion??null)}</td>{#if findingCategory!=='off'}<td class="cell-evidence">{#if findingCategory==='failure'&&r.success===1}Successful run — outside failure strip{:else if !annotations.has(r.id)}No annotation{:else}{#each evidence(r) as f}<p><strong>{nice(f[1])}</strong>: {f[2]}</p>{:else}No finding{/each}{/if}</td>{/if}<td><a href={r.folder_url} target="_blank" rel="noreferrer">Run folder</a></td></tr>{/each}</tbody></table>
  </div>{/if}
</section>
<style>
 .finding-key{max-width:1400px;display:flex;flex-wrap:wrap;justify-content:center;gap:10px 20px;margin:12px auto;font-size:12px;color:#334155}.finding-key>span{display:flex;align-items:center;gap:6px}.finding-key i{width:13px;height:13px;border:1px solid #94a3b8;border-radius:2px}.finding-strip{display:flex;width:100%;height:21px;flex-shrink:0;margin-top:5px;border-top:2px solid white;border-radius:0 0 7px 7px;overflow:hidden;background:#e2e8f0}.finding-segment{display:block;height:100%;flex-shrink:0;box-shadow:inset -1px 0 #ffffff70}.finding-empty{font-size:9px;color:#475569;width:100%;text-align:center;line-height:19px}.cell-evidence{min-width:260px;max-width:650px;white-space:normal}.cell-evidence p{margin:0 0 8px;overflow-wrap:anywhere}
 .panel{margin:28px 0;padding:26px;background:white;border:1px solid #dbe3ef;border-radius:14px;color:#172033}h2,h3{text-align:center}h2{margin:0 0 12px;font-size:24px}.intro{max-width:1100px;margin:8px auto;text-align:center;line-height:1.6;color:#64748b;font-size:14px}.small{font-size:12px}.controls{display:flex;justify-content:center;align-items:center;gap:28px;flex-wrap:wrap;margin:20px}label{display:flex;gap:10px;align-items:center;font-weight:600;font-size:13px}select{font:inherit;background:white;color:#172033;border:1px solid #94a3b8;border-radius:7px;padding:7px}.scale{display:flex;align-items:center;gap:8px;font-size:12px}.scale i{width:140px;height:12px;border-radius:3px;background:linear-gradient(to right,hsl(204 85% 97%),hsl(204 85% 35%))}.task-block{max-width:1500px;margin:24px auto}.scenario-card{border:1px solid #dbe3ef;border-radius:12px;padding:18px;margin:14px 0}h4{display:flex;justify-content:space-between;gap:16px;margin:0 0 14px;font-size:16px}h4 span{font-size:12px;color:#64748b;font-weight:400}.grid-scroll{overflow:auto}.scenario-grid{width:100%;min-width:900px;table-layout:fixed;border-spacing:4px;border-collapse:separate}.scenario-grid th:first-child{width:108px}.scenario-grid th{text-align:center;position:static;font-size:11px;font-weight:600;padding:7px 4px;border-radius:6px}.scenario-grid thead tr:first-child th{font-size:10px;text-transform:uppercase;letter-spacing:.7px}.availability{background:#eff6ff;color:#1e40af}.containment{background:#fff7ed;color:#9a3412}.distractors{background:#f5f3ff;color:#5b21b6}.column-label{color:#475569}.memory{text-transform:uppercase}.ACC{background:#e7f5ed;color:#006c35}.INC{background:#fff4e2;color:#9a4500}.OUT{background:#fdebed;color:#a32138}.scenario-grid td{padding:0}.scenario-cell{width:100%;min-height:78px;padding:7px 0 0;border:1px solid #cbd5e1;border-radius:8px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;font:inherit;cursor:pointer}.scenario-cell:hover:enabled,.scenario-cell:focus-visible{outline:3px solid #172033;outline-offset:1px}.axis-code{font-size:10px;font-weight:700;letter-spacing:.5px}.scenario-cell strong{font-size:19px;font-variant-numeric:tabular-nums}.scenario-cell small{font-size:10px}.scenario-cell:disabled{background:#f8fafc;color:#64748b;border-style:dashed;cursor:default}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}.grid-details{max-width:1500px;margin:20px auto;padding:20px;background:#f8fafc;border-radius:8px;overflow:auto}.detail-heading{display:flex;justify-content:space-between;align-items:center}.detail-heading h3{margin:0;font-size:16px}.detail-heading button{padding:6px 12px;cursor:pointer}.grid-runs{width:100%;border-collapse:collapse;font-size:13px}.grid-runs th,.grid-runs td{text-align:left;padding:10px;border-bottom:1px solid #dbe3ef}.grid-runs a{color:#1d4ed8}@media(max-width:1000px){.panel{padding:14px}.scenario-card{padding:10px}h4{flex-wrap:wrap}}
</style>
