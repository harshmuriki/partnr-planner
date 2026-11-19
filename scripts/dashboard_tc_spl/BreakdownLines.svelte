<script>
  export let groups;
  export let baselines;
  export let group;
  export let metric;
  export let label;
  export let unit='percent';
  $: max=unit==='percent'?1:Math.max(1e-6,...groups.flatMap(g=>g.baselines.map(b=>b[metric]??0)))*1.1;
  export let onselect;
  const colors=['#0072B2','#D55E00','#CC79A7'];
  const dashes=['','9 5','2 5'];
  const names={ACC:'Accurate',INC:'Incomplete',OUT:'Outdated',BASE:'Base',SUB:'Substitute',ABS:'Absent',CON:'Contained',DIS:'Distractors'};
  const format=(v,u)=>u==='percent'?(100*v).toFixed(1)+'%':u==='usd'?'$'+v.toFixed(3):v.toFixed(1)+(u==='seconds'?' s':'');
  const coverage=b=>metric==='sr'?b.srN:metric==='cr'?b.crN:b[metric+'N']??b.scored;
  $: axisLabel=group==='task'?'Task':group==='memory'?'Initial memory':'Scenario';
  $: series=baselines.map((b,i)=>({...b,color:b.color,dash:b.dash,points:groups.map((g,j)=>{
    const stats=g.baselines.find(s=>s.key===b.key);
    return {key:g.key,label:g.label??(group==='task'?'Task '+g.key:names[g.key]??g.key),lines:g.lines,x:groups.length===1?480:110+j*740/(groups.length-1),value:stats?.[metric]??null,stats};
  })}));
  function path(points) {
    return points.map((p,i)=>p.value===null?'':`${i===0||points[i-1].value===null?'M':'L'} ${p.x} ${350-280*p.value/max}`).join(' ');
  }
  function choose(point,baseline){onselect({baseline,groupKey:point.key});}
</script>
<div class="breakdown-lines">
  <div class="legend">{#each series as s,i}<span><svg width="48" height="18" aria-hidden="true"><line x1="1" x2="47" y1="9" y2="9" stroke={s.color} stroke-width="2.5" stroke-dasharray={s.dash}/>{#if s.style===0}<circle cx="24" cy="9" r="4" fill={s.color}/>{:else if s.style===1}<rect x="20" y="5" width="8" height="8" fill={s.color}/>{:else}<path d="M24 4 L29 13 L19 13 Z" fill={s.color}/>{/if}</svg>{s.label}</span>{/each}</div>
  {#if !groups.length}<p>No runs match the selected filters.</p>{:else}
  <div class="plot-scroll"><svg class="breakdown-line-plot" viewBox="0 0 960 475" role="img" aria-label={`${label} by ${axisLabel}, one line per baseline`}>
    <title>{label} by {axisLabel}</title>
    {#each [0,.25,.5,.75,1] as tick}<line x1="90" x2="880" y1={350-280*tick} y2={350-280*tick} stroke="#dbe3ef"/><text x="77" y={355-280*tick} text-anchor="end" class="tick">{format(tick*max,unit)}</text>{/each}
    <text class="breakdown-y-label" transform="translate(22 210) rotate(-90)" text-anchor="middle">{label}{unit==='percent'?' (%)':unit==='seconds'?' (s)':unit==='usd'?' (USD)':''}</text>
    {#each series[0].points as p}<text class="group-label" data-label={p.label} x={p.x} y="382" text-anchor="middle">{#each p.lines??[p.label] as line,j}<tspan x={p.x} dy={j===0?0:20}>{line}</tspan>{/each}</text>{/each}
    <text x="480" y="450" class="axis-label" text-anchor="middle">{axisLabel}</text>
    {#each series as s,i}
      <path class="breakdown-line" data-baseline={s.key} d={path(s.points)} fill="none" stroke={s.color} stroke-width="3" stroke-dasharray={s.dash}/>
      {#each s.points as p}{#if p.value!==null}<g class="breakdown-point" data-baseline={s.key} data-group={p.key} data-value={p.value} tabindex="0" role="button" aria-label={`${s.label}, ${p.label}: ${label} ${format(p.value,unit)}, ${coverage(p.stats)}/${p.stats.n} scored runs`} on:click={()=>choose(p,s.key)} on:keydown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(p,s.key);}}}>
        <title>{s.label} · {p.label}: {format(p.value,unit)} · {coverage(p.stats)}/{p.stats.n} scored runs</title>
        <circle cx={p.x} cy={350-280*p.value/max} r="12" fill="transparent"/>
        {#if s.style===0}<circle cx={p.x} cy={350-280*p.value/max} r="6" fill={s.color} stroke="white" stroke-width="1.5"/>{:else if s.style===1}<rect x={p.x-6} y={344-280*p.value/max} width="12" height="12" fill={s.color} stroke="white" stroke-width="1.5"/>{:else}<path d={`M${p.x} ${343-280*p.value/max} l7 13 h-14 Z`} fill={s.color} stroke="white" stroke-width="1.5"/>{/if}
      </g>{/if}{/each}
    {/each}
  </svg></div>{/if}
  {#if group==='uncertainty'}<p>Target Exists, On Surface, and None reuse the same BASE runs; they are reference points, not additional runs.</p>{/if}
</div>
<style>
 .breakdown-lines{max-width:1100px;margin:16px auto 24px;padding:18px;background:white;border:1px solid #dbe3ef;border-radius:10px}.legend{display:flex;justify-content:center;gap:26px;flex-wrap:wrap;font-size:13px;margin-bottom:8px}.legend span{display:flex;align-items:center;gap:8px}.plot-scroll{overflow-x:auto}.breakdown-line-plot{display:block;width:100%;min-width:620px;height:auto}.tick{font-size:13px;fill:#64748b}.group-label{font-size:15px;fill:#172033}.axis-label,.breakdown-y-label{font-size:15px;font-weight:600;fill:#172033}.breakdown-point{cursor:pointer}.breakdown-point:hover,.breakdown-point:focus{outline:2px solid #172033;outline-offset:3px}p{font-size:12px;color:#64748b;text-align:center;line-height:1.5;margin:6px 0}.legend svg{flex-shrink:0}
</style>
