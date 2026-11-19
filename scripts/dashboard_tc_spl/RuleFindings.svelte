<script>
  import FindingBars from './FindingBars.svelte';
  export let data;
  let task='all',memory='all',scenario='all',category='failure',outcome='all',focus='',baseline='all';
  const baselines=[{key:'react',label:'ReAct (images)',color:'#D55E00'},{key:'vlm-tamp',label:'VLM-TAMP',color:'#0072B2'},{key:'tru-pomdp',label:'TRU-POMDP',color:'#CC79A7'}];
  const nice=s=>s==='placement_error'?'incorrect placement':s.replaceAll('_',' ');
  $: selected=data.runs.filter(r=>(task==='all'||r.task===task)&&(memory==='all'||r.memory===memory)&&(scenario==='all'||r.uncertainty===scenario)&&(outcome==='all'||r.success===(outcome==='successful')));
  $: labels=[...new Set(data.runs.flatMap(r=>r.findings.filter(f=>f[0]===category).map(f=>f[1])))].sort();
  $: allLabels=Object.fromEntries(['failure','inefficiency'].map(c=>[c,[...new Set(data.runs.flatMap(r=>r.findings.filter(f=>f[0]===c).map(f=>f[1])))].sort()]));
  $: findings=selected.filter(r=>baseline==='all'||r.baseline===baseline).flatMap(r=>r.findings.filter(f=>f[0]===category&&(!focus||f[1]===focus)).map(f=>({run:r,label:f[1],evidence:f[2]})));
  function select(label,b,c){category=c;focus=label;baseline=b;}
</script>
<section id="rule-findings" class="rule-panel">
  <h2>Rule-based failure and inefficiency analysis</h2>
  <p class="intro">{data.runs.length.toLocaleString()} matched runs · {data.snapshot.findings.toLocaleString()} findings · Copied {data.snapshot.imported_at.slice(0,10)}</p>
  <p class="intro">These heuristic labels can overlap and do not establish root causes. Failure and inefficiency rates both use all selected runs as the denominator (a failure finding only occurs on an unsuccessful run). Some rules apply only to particular scenarios; see the rule definitions.</p>
  <nav><a href="uncertainty_classifier/rule_viewer.html" target="_blank" rel="noreferrer">Open full rule viewer ↗</a><a href="uncertainty_classifier/source/findings.csv">Findings CSV</a><a href="uncertainty_classifier/rules_data.json">All viewer data</a><a href="uncertainty_classifier/source/PSEUDOCODE.md">Rule definitions</a><a href="uncertainty_classifier/import_manifest.json">Import provenance</a></nav>
  <div class="controls">
    <label>Task <select aria-label="Rules task" bind:value={task}><option value="all">All tasks</option>{#each ['T1','T2','T3','T4','T5','T6','T7'] as t}<option value={t}>{t}</option>{/each}</select></label>
    <label>Memory <select aria-label="Rules memory" bind:value={memory}><option value="all">All memories</option>{#each ['ACC','INC','OUT'] as m}<option value={m}>{m}</option>{/each}</select></label>
    <label>Scenario <select aria-label="Rules scenario" bind:value={scenario}><option value="all">All scenarios</option>{#each ['BASE','SUB','ABS','CON','DIS'] as s}<option value={s}>{s}</option>{/each}</select></label>
    <label>Outcome <select aria-label="Rules outcome" bind:value={outcome}><option value="all">All outcomes</option><option value="successful">Successful</option><option value="unsuccessful">Unsuccessful</option></select></label>
    <label>Category <select aria-label="Rules category" bind:value={category} on:change={()=>{focus='';baseline='all';}}><option value="failure">Failure findings</option><option value="inefficiency">Inefficiency findings</option></select></label>
  </div>
  <FindingBars runs={selected} {baselines} labels={allLabels} onselect={select}/>
  <div class="controls">
    <label>Evidence baseline <select aria-label="Rules evidence baseline" bind:value={baseline}><option value="all">All baselines</option>{#each baselines as b}<option value={b.key}>{b.label}</option>{/each}</select></label>
    <label>Finding <select aria-label="Rules finding" bind:value={focus}><option value="">All findings</option>{#each labels as l}<option value={l}>{nice(l)}</option>{/each}</select></label>
    <span>{findings.length.toLocaleString()} findings · {new Set(findings.map(f=>f.run.id)).size} runs</span>
  </div>
  <div class="rule-table-scroll"><table class="rule-evidence"><thead><tr><th>Run</th><th>Outcome</th><th>Finding</th><th>Recorded evidence</th><th>Artifacts</th></tr></thead><tbody>{#each findings as f}<tr><td>{f.run.baseline}<br>{f.run.variant} · repeat {f.run.repeat}</td><td>{f.run.success?'Successful':'Unsuccessful'}<br>{f.run.ended}</td><td>{nice(f.label)}</td><td class="evidence">{f.evidence}</td><td>{#if f.run.trace_html}<a href={f.run.trace_html} target="_blank" rel="noreferrer">Trace HTML</a><br>{/if}{#if f.run.trace_txt}<a href={f.run.trace_txt} target="_blank" rel="noreferrer">Trace text</a><br>{/if}<a href={f.run.folder} target="_blank" rel="noreferrer">Run folder</a></td></tr>{/each}</tbody></table>{#if !findings.length}<p>No findings match these filters.</p>{/if}</div>
  <details><summary>Full interactive viewer, action sequences, and rule definitions</summary><p>The complete copied viewer includes rule documentation, additional charts, per-run actions, and trace text. Its links point to this workspace.</p><iframe title="Copied uncertainty rule viewer" src="uncertainty_classifier/rule_viewer.html" loading="lazy"></iframe></details>
</section>
<style>
 .rule-panel{margin:28px 0;padding:26px;background:#fff;border:1px solid #dbe3ef;border-radius:14px;color:#172033}h2{text-align:center;font-size:24px;margin:0 0 12px}.intro{text-align:center;color:#64748b;max-width:1100px;margin:8px auto;line-height:1.5}nav,.controls{display:flex;flex-wrap:wrap;justify-content:center;gap:16px;margin:20px 0}a{color:#1d4ed8}label{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:600}select{font:inherit;padding:7px;background:white;border:1px solid #94a3b8;border-radius:7px;color:#172033}.rule-table-scroll{overflow:auto;max-height:520px;border:1px solid #e2e8f0;border-radius:8px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:10px;text-align:left;border-bottom:1px solid #e2e8f0;vertical-align:top}th{position:sticky;top:0;background:#f1f5f9}.evidence{white-space:pre-wrap;min-width:320px;max-width:1000px;overflow-wrap:anywhere}details{margin-top:20px}summary{cursor:pointer;font-weight:600}iframe{width:100%;height:1000px;border:1px solid #cbd5e1;border-radius:8px}@media(max-width:1000px){.rule-panel{padding:14px}}
</style>
