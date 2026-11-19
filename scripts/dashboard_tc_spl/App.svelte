<script>
  import BreakdownLines from './BreakdownLines.svelte';
  import MemoryHeatmap from './MemoryHeatmap.svelte';
  import TaskScenarioGrid from './TaskScenarioGrid.svelte';
  import RuleFindings from './RuleFindings.svelte';
  export let data;
  let task = 'all', memory = 'all', variant = 'all', group = 'memory', selected = null;
  let lineMetric = 'tc', lineTask = 'all', breakdownMetric = 'tca';
  let selectedMetrics = ['sr','spl','cr','tca'];
  $: visibleMetrics = breakdownMetrics.filter(m=>selectedMetrics.includes(m.key));
  $: lineMetricLabel = lineMetrics.find(m=>m.key===lineMetric).label;
  const coverage = (point,key) => key==='sr'?point.srN:key==='cr'?point.crN:point[key+'N']??point.scored;
  const scenarios = [
    {axis:'BASE',label:'Target Exists',lines:['Target','Exists'],group:'availability'},
    {axis:'SUB',label:'Substitute Available',lines:['Substitute','Available'],group:'availability'},
    {axis:'ABS',label:'No Suitable Object',lines:['No Suitable','Object'],group:'availability'},
    {axis:'BASE',label:'On Surface',lines:['On','Surface'],group:'containment'},
    {axis:'CON',label:'Inside Closed Receptacle',lines:['Inside Closed','Receptacle'],group:'containment'},
    {axis:'BASE',label:'None',lines:['None'],group:'distractors'},
    {axis:'DIS',label:'Present',lines:['Present'],group:'distractors'}
  ];
  const memories = [{key:'ACC',label:'Accurate',color:'#0072B2',dash:''},{key:'INC',label:'Incomplete',color:'#D55E00',dash:'9 5'},{key:'OUT',label:'Outdated',color:'#009E73',dash:'2 5'}];
  $: baselineSeries = data.baselines.map(b=>({...b,series:memories.map(m=>({...m,points:scenarios.map(scenario=>({scenario:scenario.axis,label:scenario.label,group:scenario.group,...stats(data.rows.filter(r=>r.condition===b.key&&r.memory===m.key&&r.uncertainty===scenario.axis&&(lineTask==='all'||r.task===Number(lineTask))))}))}))}));
  function linePath(points, key, max) {
    return points.map((p,i)=>p[key]==null?'':`${i===0||points[i-1][key]==null||points[i-1].group!==p.group?'M':'L'} ${115+i*125} ${330-280*p[key]/max}`).join(' ');
  }
  const colors = ['#0072B2','#D55E00','#CC79A7'];
  const metrics = [{key:'sr',label:'Task success'}, {key:'spl',label:'SPL'}, {key:'spl_s',label:'SPL_S'}, {key:'cr',label:'Task completion'}, {key:'tc',label:'TC-Steps'}];
  $: breakdownMetrics = lineMetrics;
  $: breakdownUnit = breakdownMetrics.find(m=>m.key===breakdownMetric).unit;
  const chartValue=(value,unit)=>value==null?null:value*(unit==='percent'?100:1);
  $: comparisonPanels = [
    {title:'Success, efficiency, and completion',unit:'percent',metrics:visibleMetrics.filter(m=>m.unit==='percent')},
    ...visibleMetrics.filter(m=>m.unit!=='percent').map(m=>({title:m.label,unit:m.unit,metrics:[m]}))
  ].filter(p=>p.metrics.length).map(p=>({...p,max:p.unit==='percent'?100:Math.max(1e-6,...overall.flatMap(b=>p.metrics.map(m=>b[m.key]??0)))*1.1,width:p.unit==='percent'?Math.max(960,p.metrics.length*200+85):380}));
  $: breakdownLabel = breakdownMetrics.find(m=>m.key===breakdownMetric).label;
  const extraMetrics = [{key:'tca',label:'TC-SPL',field:'tcae',unit:'percent'}, {key:'sub',label:'% Subgoals Solved',unit:'percent'}, {key:'runtime',label:'Mean Runtime',field:'runtime_s',unit:'seconds'}, {key:'actions',label:'Mean Actions',field:'num_actions',unit:'actions'}, {key:'planning',label:'LLM Planning Time',field:'llm_planning_time_s',unit:'seconds'}, {key:'cost',label:'Mean Cost',field:'llm_usd',unit:'usd'}];
  const lineMetrics = [...metrics.map(m=>({...m,unit:'percent'})),...extraMetrics];
  $: lineUnit = lineMetrics.find(m=>m.key===lineMetric).unit;
  $: lineMax = lineUnit==='percent'?1:Math.max(1e-6,...baselineSeries.flatMap(b=>b.series.flatMap(s=>s.points.map(p=>p[lineMetric]??0))))*1.1;
  const formatMetric = (v,unit) => v==null?'—':unit==='percent'?percent(v):unit==='usd'?'$'+v.toFixed(3):v.toFixed(1)+(unit==='seconds'?' s':'');
  const percent = v => v == null ? '—' : (100*v).toFixed(1)+'%';
  const mean = a => a.length ? a.reduce((x,y)=>x+y,0)/a.length : null;
  function stats(rows) {
    const scored = rows.filter(r=>r.tcspl !== null);
    const extras = {};
    for(const m of extraMetrics.filter(m=>m.field)) {
      const values = rows.filter(r=>r[m.field]!=null).map(r=>r[m.field]);
      extras[m.key]=mean(values);extras[m.key+'N']=values.length;
    }
    const known = rows.filter(r=>r.subtasks_succeeded!=null&&r.num_subtasks>0);
    const total = known.reduce((s,r)=>s+r.num_subtasks,0);
    extras.sub = total?known.reduce((s,r)=>s+r.subtasks_succeeded,0)/total:null;
    extras.subN = known.length;
    return {...extras,n:rows.length, scored:scored.length, srN:rows.filter(r=>r.success!=null).length, crN:rows.filter(r=>r.completion!=null).length, sr:mean(rows.filter(r=>r.success!=null).map(r=>Number(r.success===1))), cr:mean(rows.filter(r=>r.completion!=null).map(r=>r.completion)), spl:mean(rows.filter(r=>r.spl!=null).map(r=>r.spl)), splN:rows.filter(r=>r.spl!=null).length, spl_s:mean(rows.filter(r=>r.spl_s!=null).map(r=>r.spl_s)), spl_sN:rows.filter(r=>r.spl_s!=null).length, tc:mean(scored.map(r=>r.tcspl))};
  }
  $: filtered = data.rows.filter(r=>(task==='all'||r.task===Number(task))&&(memory==='all'||r.memory===memory)&&(variant==='all'||r.uncertainty===variant));
  $: overall = data.baselines.map(b=>({...b,...stats(filtered.filter(r=>r.condition===b.key))}));
  $: keys = [...new Set(filtered.map(r=>String(r[group])))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
  $: groupEntries = group==='uncertainty'?scenarios.map(s=>({key:s.axis,label:s.label,lines:s.lines})):keys.map(key=>({key}));
  $: grouped = groupEntries.map(entry=>({...entry, baselines:data.baselines.map(b=>({...b,...stats(filtered.filter(r=>String(r[group])===entry.key&&r.condition===b.key))}))}));
  $: runs = selected ? filtered.filter(r=>r.condition===selected.baseline&&(selected.groupKey==null||String(r[group])===selected.groupKey)) : [];
  function reset() {selected=null;}
</script>
<div class="tc-panel">
  <h2>Completion and execution efficiency</h2>
  <p>SPL = mean[S × A<sub>ref</sub> / max(A, A<sub>ref</sub>)]. S is binary evaluator success. SPL_S = mean[S × L / max(P, L)], where L and P are human reference and agent simulator-step counts.</p>
  <p><strong>TC-SPL (completion-weighted action efficiency)</strong> = mean[C × A<sub>ref</sub> / max(A, A<sub>ref</sub>)]. A is the run's recorded skill-attempt count and A<sub>ref</sub> is the human reference count. Failed and interrupted attempts count; Explore and Wait each count once. Done and ReportAbsence are excluded from both sides. Any Rearrange is expanded to two Navigate, one Pick, and one Place. Scores retain saved partial progress and missing counts stay unavailable. References are demonstrated executions, not proven minimum-action plans.</p>
  <p class="note">SPL and TC-SPL use high-level action counts; SPL_S and TC-Steps use simulator steps. Both use human reference executions. Partial progress and evaluator success saved at timeout are retained. Missing inputs are excluded with coverage shown; zero recorded steps are valid. Travel or execution shorter than the reference receives no efficiency penalty.</p>
  <div class="controls">
    <label>Task <select aria-label="Completion task" bind:value={task} on:change={reset}><option value="all">All tasks</option>{#each [1,2,3,4,5,6,7] as t}<option value={String(t)}>Task {t}</option>{/each}</select></label>
    <label>Memory <select aria-label="Completion memory" bind:value={memory} on:change={reset}><option value="all">All memories</option>{#each ['ACC','INC','OUT'] as m}<option value={m}>{m}</option>{/each}</select></label>
    <label>Scenario <select aria-label="Completion scenario" bind:value={variant} on:change={reset}><option value="all">All scenarios</option>{#each ['BASE','ABS','SUB','CON','DIS'] as v}<option value={v}>{v}</option>{/each}</select></label>
  </div>
    <fieldset class="metric-picker"><legend>Metrics to display</legend>{#each breakdownMetrics as metric}<label><input type="checkbox" value={metric.key} bind:group={selectedMetrics}/>{metric.label}</label>{/each}</fieldset>
  <div class="comparison-grid">
  {#each comparisonPanels as panel}
  {@const metricWidth=(panel.width-100)/panel.metrics.length}
  {@const compact=panel.unit!=='percent'}
  {@const plotBase=compact?225:330}
  {@const plotHeight=compact?175:280}
  <div class="metric-comparison efficiency-comparison" class:compact-resource={compact} class:percentage-panel={!compact} style={`max-width:${compact?420:Math.max(1100,panel.width)}px`}>
    <h3>{panel.title}</h3>
    <div class="metric-legend">{#each overall as b,i}<span><span class="legend-swatch" style={`background:${b.color}`}></span>{b.label} (n={b.n})</span>{/each}</div>
    <div class="vertical-scroll">
      <svg class="metric-bars" style="min-width:0" viewBox={`0 0 ${panel.width} ${compact?280:410}`} role="img" aria-label={`Comparison of ${panel.metrics.map(m=>m.label).join(', ')} across three baselines, on one shared axis`}>
        <title>Selected metrics on one shared axis; labels show original values</title>
        <text transform={`translate(16 ${compact?135:190}) rotate(-90)`} text-anchor="middle" class="metric-y-label">{panel.unit==='percent'?'Score (%)':panel.unit==='seconds'?'Time (s)':panel.unit==='actions'?'Actions':'Cost (USD)'}</text>
        {#each [0,.25,.5,.75,1] as tick}
          <line x1="80" x2={panel.width-15} y1={plotBase-plotHeight*tick} y2={plotBase-plotHeight*tick} stroke="#dbe3ef"/>
          <text x="70" y={plotBase+5-plotHeight*tick} text-anchor="end" class="axis-text">{panel.unit==='percent'?(tick*panel.max).toFixed(0)+'%':formatMetric(tick*panel.max,panel.unit)}</text>
        {/each}
        {#each panel.metrics as metric,m}
          {@const left=80+m*metricWidth+15}
          {@const right=80+(m+1)*metricWidth-15}
          {@const step=Math.min(55,(right-left)/3.5)}
          {@const center=(left+right)/2}
          <g class="comparison-metric" data-metric={metric.key} data-max={panel.max} data-unit={metric.unit}>
          {#each overall as b,i}
            {@const value=b[metric.key]}
            {@const plotted=chartValue(value,metric.unit)??0}
            {@const x=center-1.4*step+i*step}
            <g class="metric-bar" data-value={value??''} data-plotted={plotted} role="button" tabindex="0" aria-label={`${b.label}, ${metric.label}: ${formatMetric(value,metric.unit)}`} on:click={()=>selected={baseline:b.key,groupKey:null}} on:keydown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selected={baseline:b.key,groupKey:null};}}}>
              <title>{b.label} · {metric.label}: {formatMetric(value,metric.unit)} · {b.n} runs</title>
              <rect x={x} y="40" width={step*0.8} height={plotHeight+10} fill="transparent" />
              <rect class="metric-value-bar" x={x} y={plotBase-plotHeight*plotted/panel.max} width={step*0.8} height={plotHeight*plotted/panel.max} fill={b.color} stroke="#334155" stroke-width="0.5" />
              <text x={x+step*0.4} y={plotBase-10-plotHeight*plotted/panel.max} text-anchor="middle" class="value-text">{formatMetric(value,metric.unit)}</text>
            </g>
          {/each}
          <text x={center} y={plotBase+35} text-anchor="middle" class="metric-label">{metric.label}</text>
          {#if !compact}<text x={center} y={plotBase+58} text-anchor="middle" class="axis-text">{metric.unit==='percent'?'%':metric.unit==='seconds'?'seconds':metric.unit==='actions'?'actions':'USD'}</text>{/if}
          </g>
        {/each}
      </svg>
    </div>
  </div>
  {:else}<p class="note" role="status">Select at least one metric to display the chart.</p>{/each}
  </div>
  <div class="scroll"><table class="tc-overall"><caption>Selected runs · each repetition has equal weight</caption><thead><tr><th>Baseline</th><th>Runs</th><th>Success</th><th>Completion</th><th>SPL</th><th>SPL_S</th><th>TC-SPL</th><th>Step-metric coverage</th></tr></thead><tbody>
    {#each overall as b}<tr><th><button on:click={()=>selected={baseline:b.key,groupKey:null}}>{b.label}</button></th><td>{b.n}</td><td>{percent(b.sr)}</td><td>{percent(b.cr)}</td><td>{percent(b.spl)} <small>(n={b.splN})</small></td><td>{percent(b.spl_s)}</td><td class="emphasis">{percent(b.tca)} <small>(n={b.tcaN})</small></td><td>{b.scored}/{b.n}</td></tr>{/each}
  </tbody></table></div>
  <div class="controls"><label>Compare by <select aria-label="Completion grouping" bind:value={group} on:change={reset}><option value="task">Task</option><option value="memory">Memory</option><option value="uncertainty">Scenario</option></select></label><label>Metric <select aria-label="Breakdown metric" bind:value={breakdownMetric}>{#each breakdownMetrics as m}<option value={m.key}>{m.label}</option>{/each}</select></label></div>
  <h3 class="breakdown-title">{breakdownLabel} by {group==='task'?'task':group==='memory'?'initial memory':'scenario'}</h3>
  <BreakdownLines groups={grouped} baselines={data.baselines} {group} metric={breakdownMetric} label={breakdownLabel} unit={breakdownUnit} onselect={value=>selected=value}/>
  <div class="scroll"><table class="tc-breakdown"><caption>{breakdownLabel} selected · units shown in column headings</caption><thead><tr><th>Group</th><th>Baseline</th><th>Runs</th>{#each breakdownMetrics as m}<th class:emphasis={m.key===breakdownMetric}>{m.label}{m.unit==='seconds'?' (s)':m.unit==='usd'?' (USD)':m.unit==='percent'?' (%)':''}</th>{/each}</tr></thead><tbody>{#each grouped as g}{#each g.baselines as b}<tr><td>{group==='task'?'Task ':''}{g.label??g.key}</td><td>{b.label}</td><td>{b.n}</td>{#each breakdownMetrics as m}<td class:emphasis={m.key===breakdownMetric}>{formatMetric(b[m.key],m.unit)}</td>{/each}</tr>{/each}{/each}</tbody></table></div>
  <details><summary>Ground-truth sources and calculation details</summary><p>References come from <a href="../../baseline_evaluation_v3/ground_truth_timing/combined.html">baseline_evaluation_v3 replay records</a>. T2-ACC-DIS uses 604 archived steps and T4-ACC-ABS uses 394 archived steps because their replays failed evaluation. The remaining 88 variants use successful replay steps. The same reference is used across baselines for each variant.</p><p>Only the newest saved attempt for each run is included. Evaluation criteria are benchmark-defined, not planner-generated subgoals. GT source paths, fields, and SHA-256 hashes are in <a href="tc_spl_data.json">the downloadable metric data</a>.</p></details>
  {#if selected}<div class="drilldown"><h3>Run details · {runs.length} runs</h3><button on:click={()=>selected=null}>Close details</button><div class="scroll"><table class="tc-runs"><thead><tr><th>Variant / repeat</th><th>Execution</th><th>C</th><th>Agent steps P</th><th>Reference steps L</th><th>Step efficiency</th><th>SPL</th><th>SPL_S</th><th>Actions / reference</th><th>TC-SPL</th><th>Evidence</th></tr></thead><tbody>{#each runs as r}<tr><td>{r.task_id} / {r.iteration}</td><td>{r.execution_status}</td><td>{percent(r.completion)}</td><td>{r.sim_steps??'Unknown'}</td><td>{r.reference_steps}{r.gt_fallback?' (archive)':''}</td><td>{percent(r.efficiency)}</td><td>{percent(r.spl)}</td><td>{percent(r.spl_s)}</td><td>{r.action_count??'Unknown'} / {r.reference_actions}</td><td>{percent(r.tcae)}</td><td><a href={r.folder_url}>Run</a> · <a href={r.gt_source_url}>GT record</a> · <a href={r.gt_action_url}>GT actions</a>{#if r.gt_fallback} · <a href={r.gt_archive_url}>Archive</a>{/if}</td></tr>{/each}</tbody></table></div></div>{/if}
</div>
  <section id="scenario-memory" class="metric-comparison memory-comparison">
    <h2>{lineMetricLabel} by scenario and initial memory</h2>
    <div class="controls"><label>Task <select aria-label="Memory line chart task" bind:value={lineTask}><option value="all">All tasks</option>{#each [1,2,3,4,5,6,7] as t}<option value={String(t)}>Task {t}</option>{/each}</select></label><label>Metric <select aria-label="Memory line chart metric" bind:value={lineMetric}>{#each lineMetrics as m}<option value={m.key}>{m.label}</option>{/each}</select></label></div>
    <p class="memory-subtitle">{lineTask==='all'?'All tasks':'Task '+lineTask} · Three baselines · Shared scale</p>
    <div class="metric-legend">{#each memories as m,i}<span><svg width="46" height="18" aria-hidden="true"><line x1="1" x2="45" y1="9" y2="9" stroke={m.color} stroke-width="2" stroke-dasharray={m.dash}/>{#if i===0}<circle cx="23" cy="9" r="4" fill={m.color}/>{:else if i===1}<rect x="19" y="5" width="8" height="8" fill={m.color}/>{:else}<path d="M23 4 L28 13 L18 13 Z" fill={m.color}/>{/if}</svg>{m.label} ({m.key})</span>{/each}</div>
    <div class="memory-chart-grid">
    {#each baselineSeries as baseline}
    {@const memorySeries = baseline.series}
    <article class="baseline-memory-chart" data-baseline={baseline.key}>
    <h4>{baseline.label}</h4>
    <div class="vertical-scroll"><svg class="metric-bars memory-lines" viewBox="-70 0 1030 490" role="img" aria-label={`${lineMetricLabel} across seven scenario columns for accurate, incomplete, and outdated memory`}>
      <title>{lineMetricLabel} by scenario for {baseline.label}</title>
      <g class="scenario-backgrounds" aria-hidden="true">
        <rect x="75" y="40" width="350" height="440" fill="#EFF6FF"/>
        <rect x="425" y="40" width="250" height="440" fill="#FFF7ED"/>
        <rect x="675" y="40" width="235" height="440" fill="#F5F3FF"/>
      </g>
      {#each [0,.25,.5,.75,1] as tick}<line x1="75" x2="910" y1={330-280*tick} y2={330-280*tick} stroke="#dbe3ef"/><text x="62" y={335-280*tick} text-anchor="end" class="axis-text">{formatMetric(tick*lineMax,lineUnit)}</text>{/each}
      <text transform="translate(-45 190) rotate(-90)" text-anchor="middle" class="metric-label">{lineMetricLabel}</text>
      {#each scenarios as scenario,i}<text x={115+i*125} y="355" text-anchor="middle" class="scenario-label">{#each scenario.lines as line,j}<tspan x={115+i*125} dy={j===0?0:28}>{line}</tspan>{/each}</text>{/each}
      <line x1="425" x2="425" y1="40" y2="383" stroke="#94a3b8" stroke-dasharray="3 4"/>
      <line x1="675" x2="675" y1="40" y2="383" stroke="#94a3b8" stroke-dasharray="3 4"/>
      <text x="240" y="433" text-anchor="middle" class="metric-label"><tspan x="240">Object</tspan><tspan x="240" dy="29">Availability</tspan></text>
      <text x="552" y="433" text-anchor="middle" class="metric-label"><tspan x="552">Object</tspan><tspan x="552" dy="29">Containment</tspan></text>
      <text x="802" y="433" text-anchor="middle" class="metric-label"><tspan x="802">Distractor</tspan><tspan x="802" dy="29">Presence</tspan></text>
      {#each memorySeries as series,j}
        <path class="memory-line" data-memory={series.key} d={linePath(series.points, lineMetric, lineMax)} fill="none" stroke={series.color} stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray={series.dash}/>
        {#each series.points as point,i}{#if point[lineMetric]!=null}
          <g class="memory-point" data-baseline={baseline.key} data-memory={series.key} data-scenario={point.scenario} data-value={point[lineMetric]} tabindex="0" role="img" aria-label={`${series.label}, ${point.label}: ${formatMetric(point[lineMetric],lineUnit)}, ${coverage(point,lineMetric)} scored runs`}>
            <title>{series.label} · {point.label}: {formatMetric(point[lineMetric],lineUnit)} · {coverage(point,lineMetric)}/{point.n} scored runs</title>
            {#if j===0}<circle cx={115+i*125} cy={330-280*point[lineMetric]/lineMax} r="6.5" fill={series.color} stroke="white"/>{:else if j===1}<rect x={109+i*125} y={324-280*point[lineMetric]/lineMax} width="12" height="12" fill={series.color} stroke="white"/>{:else}<path d={`M ${115+i*125} ${324-280*point[lineMetric]/lineMax} l 7 13 h -14 Z`} fill={series.color} stroke="white"/>{/if}
          </g>
        {/if}{/each}
      {/each}

    </svg></div>

    </article>
    {/each}
    </div>
    <div class="memory-tables">
      {#each baselineSeries as baseline}
        <article class="memory-table-card">
          <h4>{baseline.label}</h4>
          <table class="memory-line-table">
            <thead><tr><th>Scenario</th>{#each memories as memory}<th title={memory.label}>{memory.key}</th>{/each}</tr></thead>
            <tbody>{#each scenarios as scenario,i}<tr><th>{scenario.label}</th>{#each baseline.series as series}{@const p = series.points[i]}<td>{formatMetric(p[lineMetric],lineUnit)}<small>n={coverage(p,lineMetric)}</small></td>{/each}</tr>{/each}</tbody>
          </table>
        </article>
      {/each}
    </div>
    <details class="memory-methods"><summary>How to read these plots and metric definitions</summary>
    <p class="note">Uses this section's task selector ({lineTask==='all'?'all tasks':'Task '+lineTask}) and always shows all memory conditions and all seven scenario columns. Filters in other sections do not change these plots. All three panels share the same Y-axis scale. Points aggregate available runs using the definitions below. Missing scenarios remain gaps, not zeros. Target Exists, On Surface, and None reuse the same BASE runs; these are reference points, not additional runs.</p>
    <p class="note">Time metrics are seconds per run; actions count recorded skill attempts. Cost is the recorded USD lower bound for the latest attempt. Subgoals solved means total satisfied benchmark criteria divided by total criteria across runs with available counts. Missing values are excluded; n shows coverage. Lower values indicate less resource use for time, actions, and cost.</p>
    </details>
  </section>
<MemoryHeatmap {data}/>
<TaskScenarioGrid {data}/>
{#if data.rule_findings}<RuleFindings data={data.rule_findings}/>{/if}
<style>
 .comparison-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;max-width:1760px;margin:18px auto;align-items:start;justify-items:center}.comparison-grid .percentage-panel{grid-column:1/-1;width:100%;box-sizing:border-box}.comparison-grid .efficiency-comparison{margin:0;min-width:0}.efficiency-comparison .vertical-scroll{overflow:visible}.comparison-grid .compact-resource{width:100%;box-sizing:border-box;min-width:0;padding:10px}.compact-resource h3{font-size:16px;margin:4px 0 8px}.compact-resource .metric-legend{font-size:11px;gap:5px 10px;min-height:34px}.compact-resource .legend-swatch{width:10px;height:10px}.compact-resource .metric-label{font-size:13px}.compact-resource .metric-bars{min-width:0}

 .metric-picker{display:flex;justify-content:center;flex-wrap:wrap;gap:12px 24px;border:0;margin:12px 0;padding:8px}.metric-picker legend{margin:auto;font-size:13px;font-weight:600}.metric-picker label{display:flex;align-items:center;gap:7px;font-size:13px;cursor:pointer}.metric-picker input{accent-color:#0072B2}
 :global(main){max-width:none;padding:20px 16px}
 @media(min-width:1001px){:global(main){min-width:2200px}}

 .memory-chart-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.memory-chart-grid .baseline-memory-chart{min-width:0;margin-top:8px;padding:10px;border:1px solid #dbe3ef;border-radius:8px}.memory-chart-grid .baseline-memory-chart h4{font-size:14px}.memory-chart-grid .memory-lines{min-width:0;width:100%}.memory-chart-grid .axis-text{font-size:23px}.memory-chart-grid .scenario-label{font-size:20px}.memory-chart-grid .metric-label{font-size:25px}@media(max-width:1000px){.memory-chart-grid{grid-template-columns:1fr}}

 .memory-tables{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:22px}.memory-table-card{min-width:0;padding:10px;border:1px solid #dbe3ef;border-radius:8px;background:#fff}.memory-table-card h4{font-size:14px;margin:2px 0 8px}.memory-line-table{font-size:11px;table-layout:fixed}.memory-line-table th,.memory-line-table td{padding:5px 3px;white-space:normal}.memory-line-table th:first-child{width:36%;font-weight:500}.memory-line-table small{display:block;font-size:10px}.memory-line-table tbody tr:nth-child(3),.memory-line-table tbody tr:nth-child(5){border-bottom:2px solid #cbd5e1}@media(max-width:1000px){.memory-tables{grid-template-columns:1fr}.memory-line-table{font-size:12px}}

 .baseline-memory-chart{margin-top:24px;padding-top:14px;border-top:1px solid #cbd5e1}.baseline-memory-chart h4{font-size:18px;margin:4px 0 12px}

 .scenario-label{font-size:12px;fill:#172033;font-weight:600}

 .metric-comparison{background:white;border:1px solid #dbe3ef;border-radius:8px;padding:16px;margin:18px 0}.metric-legend{display:flex;flex-wrap:wrap;gap:12px 24px;font-size:13px;margin:14px 0}.metric-legend>span{display:flex;align-items:center;gap:7px}.legend-swatch{display:inline-block;width:14px;height:14px;border:1px solid #334155}.vertical-scroll{overflow-x:auto}.metric-bars{display:block;width:100%;min-width:680px;height:auto}.axis-text{font-size:12px;fill:#475569}.value-text{font-size:12px;fill:#172033;font-weight:600}.metric-label{font-size:15px;fill:#172033;font-weight:600}.metric-bar{cursor:pointer}.metric-bar:hover .metric-value-bar,.metric-bar:focus .metric-value-bar{stroke:#0f172a;stroke-width:2}

 .tc-panel{color:#172033;background:#f8fafc;padding:20px;border:1px solid #cbd5e1;border-radius:12px}h2{margin-top:0}p{max-width:1100px;line-height:1.55}.note,small{color:#536277;font-size:13px}.controls{display:flex;gap:20px;flex-wrap:wrap;align-items:center;margin:18px 0}label{display:flex;align-items:center;gap:8px;font-weight:600}select,button{font:inherit}select{padding:6px;background:white;color:#172033;border:1px solid #94a3b8;border-radius:5px}.scroll{overflow:auto;max-height:480px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px;border-bottom:1px solid #dbe3ef;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{background:#e2e8f0}caption{text-align:left;padding:8px 0;color:#536277}button{cursor:pointer;border:0;background:transparent;color:#1d4ed8;text-align:left}button:focus-visible{outline:2px solid #2563eb}.emphasis{font-weight:700;color:#1d4ed8}h3{font-size:16px;margin:8px 0}details{margin-top:18px}summary{cursor:pointer;font-weight:600}a{color:#1d4ed8}.drilldown{margin-top:20px}@media(max-width:500px){.tc-panel{padding:10px}}

 .memory-comparison{padding:28px 22px;background:#fff;border-radius:14px}
 .memory-comparison>h2{text-align:center;font-size:24px;letter-spacing:-.4px;margin:4px 0 18px}
 .memory-comparison>.controls{justify-content:center;margin:0 0 12px}
 .memory-comparison>.controls select{min-width:220px;padding:9px 12px;border-radius:8px}
 .memory-subtitle{max-width:none;text-align:center;margin:8px auto 14px;color:#64748b;font-size:13px}
 .memory-comparison>.metric-legend{justify-content:center;gap:32px;margin:16px auto 24px;padding:12px 20px;border-radius:9px;background:#f8fafc;width:fit-content}
 .memory-chart-grid{gap:20px;align-items:stretch}
 .memory-chart-grid .baseline-memory-chart{margin:0;padding:18px 10px 8px;border-color:#e2e8f0;border-radius:12px;background:#fff;box-shadow:0 2px 8px #0f172a04}
 .memory-chart-grid .baseline-memory-chart h4{text-align:center;font-size:19px;font-weight:650;margin:0 0 8px;letter-spacing:-.2px}
 .memory-chart-grid .scenario-label{font-size:20px;font-weight:500}
 .memory-chart-grid .metric-label{font-size:23px;font-weight:600}
 .memory-chart-grid .axis-text{fill:#64748b;font-size:22px}
 .memory-tables{gap:20px;margin-top:20px}
 .memory-table-card{border-radius:10px;padding:12px}
 .memory-table-card h4{text-align:center;font-size:14px;margin:2px 0 12px}
 .memory-line-table td,.memory-line-table thead th:not(:first-child){text-align:center}
 .memory-line-table td{font-variant-numeric:tabular-nums}
 .memory-line-table small{color:#8290a3;margin-top:3px}
 .memory-methods{max-width:1100px;margin:22px auto 0;padding-top:14px;border-top:1px solid #e2e8f0}
 .memory-methods summary{text-align:center;color:#64748b;font-size:12px;font-weight:500}
 @media(max-width:1000px){.memory-comparison{padding:18px 10px}.memory-comparison>h2{font-size:20px}.memory-comparison>.metric-legend{gap:12px;padding:10px}}

 .metric-y-label{font-size:14px;fill:#172033;font-weight:600}
 .efficiency-comparison{max-width:1100px;margin:18px auto;padding:14px}
 .efficiency-comparison h3{text-align:center}
 .efficiency-comparison .metric-legend{justify-content:center;margin:10px 0}
</style>
