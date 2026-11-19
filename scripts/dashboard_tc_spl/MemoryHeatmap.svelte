<script>
  import ScenarioDelta from './ScenarioDelta.svelte';
  export let data;
  let task='all';
  const tasks=[...new Set(data.rows.map(r=>r.task))].sort((a,b)=>a-b);
</script>
<section id="memory-scenario-heatmap" class="delta-section">
  <h2>TC-SPL change from BASE</h2>
  <div class="controls">
    <label>Task <select aria-label="Scenario delta task" bind:value={task}><option value="all">All tasks</option>{#each tasks as t}<option value={String(t)}>Task {t}</option>{/each}</select></label>
  </div>
  <ScenarioDelta {data} {task}/>
</section>
<style>
 .delta-section{margin:28px 0;padding:26px;background:#fff;color:#172033;border:1px solid #dbe3ef;border-radius:14px}h2{text-align:center;font-size:24px;margin:0 0 20px}.controls{display:flex;justify-content:center;margin:16px 0}label{display:flex;align-items:center;gap:10px;font-size:13px;font-weight:600}select{font:inherit;background:#fff;color:#172033;border:1px solid #94a3b8;border-radius:7px;padding:8px}@media(max-width:1000px){.delta-section{padding:16px}}
</style>
