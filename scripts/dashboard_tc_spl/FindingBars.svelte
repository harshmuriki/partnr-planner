<script>
  export let runs;
  export let baselines;
  export let labels;
  export let onselect;
  let view='bars';
  $: panels=['failure','inefficiency'].map(category=>{
    const groups=baselines.map(b=>({...b,runs:runs.filter(r=>r.baseline===b.key)}));
    const rows=labels[category].map(label=>{
      const bars=groups.map(b=>{const n=b.runs.filter(r=>r.findings.some(f=>f[0]===category&&f[1]===label)).length;return {...b,n,total:b.runs.length,pct:b.runs.length?100*n/b.runs.length:null};});
      const known=bars.filter(b=>b.pct!==null);
      return {label,bars,mean:known.length?known.reduce((s,b)=>s+b.pct,0)/known.length:0};
    }).sort((a,b)=>b.mean-a.mean||a.label.localeCompare(b.label));
    return {category,rows,groups};
  });
  const nice=s=>s==='placement_error'?'incorrect placement':s.replaceAll('_',' ');
  const axisMax=100;
  const ticks=[0,20,40,60,80,100];
</script>
<div class="finding-comparison">
  <h3>Finding rates across baselines</h3>
  <label class="view-control">View <select aria-label="Finding rates view" bind:value={view}><option value="bars">Bar plots</option><option value="heatmap">Heatmaps</option></select></label>
  <p class="note">Rows run from highest to lowest average percentage across baselines. Percentages count distinct runs; findings can overlap.</p>
  <div class="panels" class:compact-heatmaps={view==='heatmap'}>{#each panels as p}<article class="rule-chart" data-category={p.category}>
    <h4>{p.category==='failure'?'Failure findings':'Inefficiency findings'}</h4>
    {#if view==='bars'}<div class="legend" aria-label={`${p.category} baseline key`}>{#each baselines as b}<span><i style={`background:${b.color}`}></i>{b.label}</span>{/each}</div>{:else}<div class="heat-key"><span>0%</span><i></i><span>100%</span></div>{/if}
    {#if view==='heatmap'}<p class="note heat-denominator">% of all runs</p>{:else}<p class="note">Percentage of all selected runs · {p.groups.map(b=>`${b.label}: n=${b.runs.length}`).join(' · ')}</p>{/if}
    {#if view==='heatmap'}
    <table class="finding-heatmap" aria-label={`${p.category==='failure'?'Failure':'Inefficiency'} finding rates heatmap`}>
      <thead><tr><th scope="col">Finding</th>{#each baselines as b}<th scope="col" title={b.label}>{b.key==='react'?'ReAct':b.label}<small>n={p.groups.find(g=>g.key===b.key).runs.length}</small></th>{/each}</tr></thead>
      <tbody>{#each p.rows as row}<tr><th scope="row">{nice(row.label)}</th>{#each row.bars as b}<td>
        <button class="finding-heat-cell" data-category={p.category} data-baseline={b.key} data-label={row.label} data-n={b.n} data-total={b.total} data-pct={b.pct??''}
          style={b.pct===null?'':`background:hsl(204 85% ${97-62*b.pct/100}%);color:${b.pct>60?'white':'#172033'}`}
          disabled={b.pct===null} title={`${b.label}: ${b.n} / ${b.total} runs`}
          aria-label={`${nice(row.label)}, ${b.label}: ${b.pct===null?'unavailable':b.pct.toFixed(1)+'%'}`}
          on:click={()=>onselect(row.label,b.key,p.category)}>{b.pct===null?'—':b.pct.toFixed(1)+'%'}</button>
      </td>{/each}</tr>{/each}</tbody>
    </table>
    {:else}
    <div class="axis" aria-label="Percentage axis: zero to one hundred percent"><span></span><div class="axis-track">{#each ticks as tick}<span class="axis-tick" class:start={tick===0} class:end={tick===axisMax} style={`left:${tick}%`}>{tick}%</span>{/each}</div></div>
    <table class="finding-table" aria-label={`${p.category==='failure'?'Failure':'Inefficiency'} findings by baseline`}><colgroup><col class="name-column"/><col/></colgroup><tbody>
    {#each p.rows as row}<tr class="finding-row" data-label={row.label} data-mean={row.mean}>
      <th scope="row" class="finding-label">{nice(row.label)}</th>
      <td class="series">{#each row.bars as b}<button class="rule-bar" data-baseline={b.key} data-label={row.label} data-n={b.n} data-total={b.total} data-pct={b.pct??''} title={`${b.label}: ${b.n} / ${b.total} runs${b.pct===null?' (no eligible runs)':` (${b.pct.toFixed(1)}%)`}`} aria-label={`${nice(row.label)}, ${b.label}: ${b.n} of ${b.total} runs`} on:click={()=>onselect(row.label,b.key,p.category)}>
        <span class="track" style={`background-size:${100*20/axisMax}% 100%`}><span class="fill" style={`width:${100*(b.pct??0)/axisMax}%;background:${b.color}`}></span></span><strong>{b.pct===null?'—':`${b.pct.toFixed(1)}%`}</strong>
      </button>{/each}</td>
    </tr>{/each}</tbody></table>
    {/if}
  </article>{/each}</div>
</div>
<style>
 .compact-heatmaps{max-width:1000px;margin:0 auto;gap:16px}.compact-heatmaps .rule-chart{padding:12px}.compact-heatmaps h4{font-size:16px;margin:4px 0 8px}.compact-heatmaps .heat-key{margin:6px 0;gap:6px;font-size:11px}.compact-heatmaps .heat-key i{width:90px;height:9px}.compact-heatmaps .heat-denominator{margin:5px 0;font-size:11px}.compact-heatmaps .finding-heatmap{border-spacing:3px 2px}.compact-heatmaps .finding-heatmap th{font-size:11px;padding:5px 2px;line-height:1.25}.compact-heatmaps .finding-heatmap th:first-child{width:34%}.compact-heatmaps .finding-heatmap small{display:block;font-size:10px;font-weight:400;color:#64748b;margin-top:3px}.compact-heatmaps .finding-heat-cell{min-height:30px;font-size:12px;padding:4px 2px}

 .view-control{display:flex;align-items:center;justify-content:center;gap:10px;margin:16px;font-size:13px;font-weight:600}select{font:inherit;padding:7px;border:1px solid #94a3b8;background:white;border-radius:6px;color:#172033}.heat-key{display:flex;align-items:center;justify-content:center;gap:10px;font-size:12px;margin:16px}.heat-key i{width:130px;height:12px;background:linear-gradient(to right,hsl(204 85% 97%),hsl(204 85% 35%))}.finding-heatmap{width:100%;border-collapse:separate;border-spacing:4px;table-layout:fixed}.finding-heatmap th{position:static;background:transparent;white-space:normal;border:0;font-size:12px;padding:8px 4px;text-align:center}.finding-heatmap th:first-child{width:32%;text-align:left}.finding-heatmap td{padding:0;border:0}.finding-heat-cell{width:100%;min-height:40px;border:1px solid #dbe3ef;border-radius:4px;font-size:14px;font-weight:600;font-variant-numeric:tabular-nums;cursor:pointer}.finding-heat-cell:hover:enabled,.finding-heat-cell:focus-visible{outline:2px solid #172033;outline-offset:1px}.finding-heat-cell:disabled{background:#f1f5f9;color:#64748b;cursor:default;border-style:dashed}

 h3,h4{text-align:center;margin:12px 0}h4{font-size:18px}.note{text-align:center;color:#64748b;font-size:12px;line-height:1.6;margin:8px auto;max-width:1050px}.legend{display:flex;flex-wrap:wrap;justify-content:center;gap:28px;margin:18px 0;font-size:13px}.legend span{display:flex;align-items:center;gap:8px}i{display:inline-block;width:18px;height:10px;border-radius:2px}.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;align-items:start}.rule-chart{border:1px solid #dbe3ef;border-radius:10px;padding:22px}.finding-table{width:100%;table-layout:fixed;border-collapse:collapse}.name-column{width:169px}.finding-row th,.finding-row td{padding:12px 0;border-bottom:1px solid #edf0f5;vertical-align:middle}.finding-row th{position:static;text-align:left;background:transparent;padding-right:14px}.finding-label{font-size:13px;font-weight:600;align-self:center}.rule-bar{display:grid;grid-template-columns:1fr 60px;gap:10px;align-items:center;width:100%;padding:3px 0;border:0;background:transparent;cursor:pointer;color:#172033}.rule-bar:hover,.rule-bar:focus-visible{background:#eaf1f9;outline:2px solid #64748b;outline-offset:1px}.track{position:relative;display:block;height:13px;background:linear-gradient(to right,#f3f6fa calc(100% - 1px),#cbd5e1 0);border-left:1px solid #cbd5e1;border-right:1px solid #cbd5e1}.fill{height:100%;display:block;border-radius:0 2px 2px 0}strong{font-size:12px;text-align:right;font-variant-numeric:tabular-nums}.axis{display:grid;grid-template-columns:169px 1fr;margin-top:20px;margin-bottom:4px;color:#64748b;font-size:11px}.axis-track{position:relative;margin-right:70px;height:18px}.axis-tick{position:absolute;transform:translateX(-50%)}.axis-tick.start{transform:none}.axis-tick.end{transform:translateX(-100%)}@media(max-width:1000px){.panels{grid-template-columns:1fr}.rule-chart{padding:12px}.legend{flex-wrap:wrap}.name-column{width:134px}.axis{grid-template-columns:134px 1fr}}
</style>
