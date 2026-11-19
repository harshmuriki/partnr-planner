<script>
 export let data;
 export let task='all';
 const memories=[{key:'ACC',label:'Accurate (ACC)',color:'#0072B2'},{key:'INC',label:'Incomplete (INC)',color:'#D55E00'},{key:'OUT',label:'Outdated (OUT)',color:'#009E73'}];
 const scenarios=[{key:'SUB',lines:['Substitute','vs Target']},{key:'ABS',lines:['No Suitable Object','vs Target']},{key:'CON',lines:['Closed Receptacle','vs On Surface']},{key:'DIS',lines:['Present','vs None']}];
 const mean=rs=>rs.reduce((sum,r)=>sum+r.tcae,0)/rs.length;
 const signed=v=>(v>0?'+':'')+v.toFixed(1);
 const barLabel=v=>(Math.round(v)>=0?'+':'')+Math.round(v);
 $: rows=data.rows.filter(r=>(task==='all'||r.task===Number(task))&&Number.isFinite(r.tcae));
 $: memoryBars=scenarios.flatMap((s,si)=>data.baselines.flatMap((b,bi)=>memories.map((m,mi)=>{
   const relevant=rows.filter(r=>r.condition===b.key&&r.memory===m.key);
   const variants=relevant.filter(r=>r.uncertainty===s.key);
   const base=relevant.filter(r=>r.uncertainty==='BASE');
   const common=new Set(variants.filter(r=>base.some(x=>x.task===r.task)).map(r=>r.task));
   const vr=variants.filter(r=>common.has(r.task)), br=base.filter(r=>common.has(r.task));
   return {scenario:s.key,baseline:b.key,label:b.label,memory:m.key,color:m.color,x:88+si*255+bi*78+mi*20,value:vr.length&&br.length?100*(mean(vr)-mean(br)):null,n:vr.length,baseN:br.length};
 })));
 $: bars=scenarios.flatMap((s,si)=>data.baselines.map((b,bi)=>{
   const parts=memoryBars.filter(p=>p.scenario===s.key&&p.baseline===b.key);
   return {scenario:s.key,baseline:b.key,label:b.label,memory:'Average (ACC, INC, OUT)',color:b.color,x:100+si*255+bi*78,width:32,value:parts.every(p=>p.value!==null)?parts.reduce((sum,p)=>sum+p.value,0)/3:null,n:parts.reduce((sum,p)=>sum+p.n,0),baseN:parts.reduce((sum,p)=>sum+p.baseN,0)};
 }));
 $: legend=data.baselines.map((b,i)=>({label:b.label,color:b.color}));
 $: low=Math.min(-10,Math.floor(Math.min(0,...bars.map(b=>b.value??0))/10)*10);
 $: high=Math.max(10,Math.ceil(Math.max(0,...bars.map(b=>b.value??0))/10)*10);
 $: ticks=Array.from({length:Math.round((high-low)/10)+1},(_,i)=>low+i*10);
 $: y=v=>70+(high-v)/(high-low)*330;
</script>
<div class="delta-view">
 <div class="legend">{#each legend as m}<span><i style={`background:${m.color}`}></i>{m.label}</span>{/each}</div>
 <svg viewBox="0 42 1130 508" role="img" aria-label="TC-SPL changes from BASE by scenario, baseline, and initial memory">
  <rect x="75" y="50" width="510" height="370" fill="#edf5fb"/>
  <rect x="585" y="50" width="255" height="370" fill="#faf4e8"/>
  <rect x="840" y="50" width="255" height="370" fill="#f3effa"/>
  {#each ticks as t}<line x1="75" x2="1095" y1={y(t)} y2={y(t)} stroke={t===0?'#172033':'#dce3eb'} stroke-width={t===0?2:1}/><text x="65" y={y(t)+5} text-anchor="end">{t}</text>{/each}
  <text transform="translate(20 235) rotate(-90)" text-anchor="middle" class="axis">Δ TC-SPL<tspan baseline-shift="sub" font-size="11" font-style="italic">act</tspan> (percentage points)</text>
  {#each bars as b}<g class="delta-bar" data-baseline={b.baseline} data-memory={b.memory} data-scenario={b.scenario} data-value={b.value??''}>
   <title>{b.label} · {b.memory} · {b.scenario}: {b.value===null?'Unavailable':signed(b.value)+' pp'} · scenario n={b.n}, BASE n={b.baseN}</title>
   {#if b.value!==null}<rect x={b.x} y={y(Math.max(0,b.value))} width={b.width} height={Math.abs(y(b.value)-y(0))} fill={b.color}/><text x={b.x+b.width/2} y={y(b.value)+(b.value<0?16:-8)} text-anchor="middle" class="value">{barLabel(b.value)}</text>{:else}<text x={b.x+b.width/2} y={y(0)-8} text-anchor="middle">—</text>{/if}
  </g>{/each}
  {#each scenarios as s,si}
   {#if si>0}<line x1={75+si*255} x2={75+si*255} y1="50" y2="475" stroke="#aab5c3" stroke-dasharray="3 3"/>{/if}
   {#each data.baselines.map(b=>b.label.replace(' (images)','').replace('TRU-POMDP','Tru-POMDP')) as name,bi}<text x={116+si*255+bi*78} y="443" text-anchor="middle" class="name">{name}</text>{/each}
   {#each s.lines as line,i}<text x={202.5+si*255} y={471+i*19} text-anchor="middle" class="axis">{line}</text>{/each}
  {/each}
  {#each [{x:75,w:510,label:'Object Availability',color:'#edf5fb'},{x:585,w:255,label:'Object Containment',color:'#faf4e8'},{x:840,w:255,label:'Distractor Presence',color:'#f3effa'}] as group}
   <rect x={group.x+5} y="510" width={group.w-10} height="28" rx="5" fill={group.color}/><text x={group.x+group.w/2} y="529" text-anchor="middle" class="axis">{group.label}</text>
  {/each}
 </svg>
 <p class="method">Each comparison uses the same available tasks for the scenario and BASE, within each baseline and memory condition. Bars are the equal-weight mean of the ACC, INC, and OUT differences; all three are required. Hover for run counts. Missing comparisons remain unavailable.</p>
</div>
<style>
 .delta-view{max-width:1250px;margin:auto}.legend{margin:0;display:flex;justify-content:center;gap:24px;flex-wrap:wrap;font-size:13px}.legend span{display:flex;align-items:center;gap:8px}i{width:20px;height:12px}p{text-align:center;font-size:13px;color:#475569}svg{display:block;width:100%;height:auto}text{font:14px Arial,sans-serif;fill:#172033}.axis{font-weight:600}.value{font-size:11px}.name{font-size:12px}.method{font-size:12px}
</style>
