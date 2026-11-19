<script>
 import {failureGroups} from './failure_groups.js';
 export let runs;
 export let baselines;
 export let category='failure';
 $: clearLabel=category==='failure'?'Successful':'No inefficiency finding';
 $: branchLabel=category==='failure'?'Unsuccessful':'With inefficiency findings';
 $: denominator=category==='failure'?'unsuccessful':'flagged';
 let chosen=null;
 const nice=s=>s==='placement_error'?'incorrect placement':s.replaceAll('_',' ');
 const pct=(n,d)=>d?(100*n/d).toFixed(1)+'%':'—';
 const title=g=>g.kind==='other'?`Other combinations (${g.combinations})`:g.kind==='unlabeled'?'No failure finding':g.labels.map(nice).join(' + ');
 function wrap(text){const lines=[''];for(const word of text.split(' ')){const i=lines.length-1;if(lines[i].length+word.length>44)lines.push(word);else lines[i]+=(lines[i]?' ':'')+word;}return lines;}
 function chart(b){
   const selected=runs.filter(r=>r.baseline===b.key), stats=failureGroups(selected,6,category),scale=260/Math.max(stats.total,1);
   let targetY=80,sourceY=80+stats.succeeded.length*scale+38;
   const failY=sourceY;
   const groups=stats.groups.map(g=>{const h=g.runs.length*scale,lines=wrap(title(g));const result={...g,y:targetY,sourceY,h,lines};targetY+=Math.max(h,lines.length*17+24)+18;sourceY+=h;return result;});
   return {...b,...stats,runs:selected,scale,failY,groups,height:Math.max(425,targetY+10,failY+stats.failed.length*scale+55)};
 }
 let charts=[];
 // Explicit dependency ensures changes to the parent filters update the partition.
 $: {runs; category; charts=baselines.map(chart);chosen=null;}
 const band=(x,y,x2,y2,h)=>`M${x},${y} C${(x+x2)/2},${y} ${(x+x2)/2},${y2} ${x2},${y2} L${x2},${y2+h} C${(x+x2)/2},${y2+h} ${(x+x2)/2},${y+h} ${x},${y+h} Z`;
 function choose(rows,label){chosen={rows,label};}
 function keypress(e,rows,label){if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(rows,label);}}
</script>
<div class="failure-sankeys">
 <h3>{category==='failure'?'From run outcomes to combinations of failure findings':'From all runs to combinations of inefficiency findings'}</h3>
 <p class="explanation">{#if category==='failure'}Each unsuccessful run enters exactly one branch, defined by its complete set of failure labels. Branch percentages are shares of unsuccessful runs.{:else}All selected runs, successful or unsuccessful, are split into runs with and without an inefficiency finding. Flagged runs enter exactly one branch for their complete set of inefficiency labels. Branch percentages are shares of flagged runs. No finding does not prove an execution was efficient.{/if} “A + B” means both labels occurred in the same run. The six most common labeled combinations are shown separately, sorted largest to smallest, with Other combinations last. These are associated findings, not causal attribution.</p>
 <div class="sankey-grid">
 {#each charts as c}<article class="failure-sankey" data-category={category} data-baseline={c.key} data-total={c.total} data-failed={c.failed.length} data-success={c.succeeded.length}>
  <h4>{c.label}</h4>
  {#if c.total===0}<p>No runs match the current filters.</p>{:else}
  <svg viewBox={`0 0 800 ${c.height}`} role="img" aria-label={`${c.label}: ${c.total} runs, ${c.succeeded.length} ${clearLabel}, ${c.failed.length} ${branchLabel}`}>
   <text x="12" y="26" class="heading">Selected runs</text><text x="185" y="26" class="heading">{category==='failure'?'Outcome':'Rule coverage'}</text><text x="395" y="26" class="heading">Associated finding combinations</text>
   <g role="button" tabindex="0" class="hit" on:click={()=>choose(c.runs,c.label+' · All selected runs')} on:keydown={e=>keypress(e,c.runs,c.label+' · All selected runs')} aria-label={`All ${c.total} selected runs`}>
    <title>{c.total} selected runs</title><rect x="15" y="80" width="16" height={c.total*c.scale} fill={c.color}/><text x="12" y="61">{c.total} runs</text>
   </g>
   {#if c.succeeded.length}<g role="button" tabindex="0" class="hit" on:click={()=>choose(c.succeeded,c.label+' · '+clearLabel)} on:keydown={e=>keypress(e,c.succeeded,c.label+' · '+clearLabel)} aria-label={`${c.succeeded.length} ${clearLabel}`}>
    <title>{clearLabel}: {c.succeeded.length} / {c.total} ({pct(c.succeeded.length,c.total)})</title><path d={band(31,80,185,80,c.succeeded.length*c.scale)} fill="#009E73" opacity=".3"/><rect x="185" y="80" width="16" height={c.succeeded.length*c.scale} fill="#009E73"/>
   </g>{/if}
   <text x="185" y="51" class="outcome-label">{clearLabel}</text><text x="185" y="69" class="count">{c.succeeded.length} ({pct(c.succeeded.length,c.total)})</text>
   {#if c.failed.length}<g role="button" tabindex="0" class="hit" on:click={()=>choose(c.failed,c.label+' · '+branchLabel)} on:keydown={e=>keypress(e,c.failed,c.label+' · '+branchLabel)} aria-label={`${c.failed.length} ${branchLabel}`}>
    <title>{branchLabel}: {c.failed.length} / {c.total} ({pct(c.failed.length,c.total)})</title><path d={band(31,80+c.succeeded.length*c.scale,185,c.failY,c.failed.length*c.scale)} fill="#D55E00" opacity=".3"/><rect x="185" y={c.failY} width="16" height={c.failed.length*c.scale} fill="#D55E00"/>
   </g>{/if}
   <text x="180" y={c.failY+c.failed.length*c.scale+20} class="outcome-label">{branchLabel}: {c.failed.length}</text><text x="180" y={c.failY+c.failed.length*c.scale+37} class="count">{pct(c.failed.length,c.total)} of selected runs</text>
   {#each c.groups as g,i}<g class="hit combination-branch" data-count={g.runs.length} data-kind={g.kind} role="button" tabindex="0" aria-label={`${title(g)}: ${g.runs.length} runs`} on:click={()=>choose(g.runs,c.label+' · '+title(g))} on:keydown={e=>keypress(e,g.runs,c.label+' · '+title(g))}>
    <title>{title(g)}: {g.runs.length} / {c.failed.length} {denominator} runs ({pct(g.runs.length,c.failed.length)})</title>
    <path d={band(201,g.sourceY,375,g.y,g.h)} fill={g.kind==='unlabeled'?'#64748b':c.color} opacity=".23"/>
    <rect x="375" y={g.y} width="14" height={g.h} fill={g.kind==='unlabeled'?'#64748b':c.color}/>
    <rect x="395" y={g.y-3} width="390" height={g.lines.length*17+25} fill="transparent"/>
    <text x="400" y={g.y+12}>{#each g.lines as line,j}<tspan x="400" dy={j===0?0:17}>{line}</tspan>{/each}<tspan x="400" dy="19" class="count">{g.runs.length} runs · {pct(g.runs.length,c.failed.length)} of {denominator}</tspan></text>
   </g>{/each}
  </svg>{/if}
 </article>{/each}
 </div>
 {#if chosen}<div class="combination-details"><h4>{chosen.label} · {chosen.rows.length} runs</h4><button on:click={()=>chosen=null}>Close details</button>
 <div class="detail-scroll"><table class="combination-runs"><thead><tr><th>Run</th><th>Outcome</th><th>Exact {category} combination</th><th>Rule evidence</th><th>Trace</th></tr></thead><tbody>{#each chosen.rows as r}<tr><td>{r.variant} · repeat {r.repeat}</td><td>{r.success?'Successful':'Unsuccessful'}<br>{r.ended}</td><td>{[...new Set(r.findings.filter(f=>f[0]===category).map(f=>nice(f[1])))].sort().join(' + ')||`No ${category} finding`}</td><td>{#each r.findings as f}<p><strong>{nice(f[1])}</strong> ({f[0]}): {f[2]}</p>{/each}</td><td>{#if r.trace_html}<a href={r.trace_html} target="_blank" rel="noreferrer">Trace</a>{/if}<br><a href={r.folder} target="_blank" rel="noreferrer">Run folder</a></td></tr>{/each}</tbody></table></div></div>{/if}
</div>
<style>
 .outcome-label{font-size:12px}

 .failure-sankeys{margin:26px 0 30px;padding:22px 0;border-top:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0}h3{text-align:center;font-size:21px;margin:0 0 10px}.explanation{max-width:1100px;text-align:center;margin:0 auto 22px;line-height:1.6;font-size:13px;color:#64748b}.sankey-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;align-items:start}.failure-sankey{padding:14px;border:1px solid #e2e8f0;border-radius:12px;min-width:0}h4{font-size:17px;text-align:center;margin:6px 0 18px}svg{display:block;width:100%;height:auto}text{font:13px system-ui;fill:#172033}.heading{font-weight:700}.count{fill:#64748b;font-size:12px}.hit{cursor:pointer}.hit:hover path,.hit:focus path{opacity:.55}.hit:focus{outline:none}.hit:focus rect{stroke:#172033;stroke-width:2}.combination-details{margin-top:20px}.detail-scroll{max-height:500px;overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top}th{background:#f1f5f9;position:sticky;top:0}td p{white-space:pre-wrap;overflow-wrap:anywhere;margin:0 0 6px}button,a{color:#1d4ed8}button{padding:7px 12px;background:#fff;border:1px solid #cbd5e1;border-radius:6px;cursor:pointer;margin-bottom:10px}@media(max-width:1000px){.sankey-grid{grid-template-columns:1fr}}
</style>
