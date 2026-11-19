// One unit per eligible run, shared equally across its distinct finding labels.
// This is a display allocation for overlapping findings, not causal attribution.
export function findingMix(runs, annotations, category) {
  const eligible=category==='failure'?runs.filter(r=>r.success!=null&&r.success!==1):runs;
  const weights=new Map();
  for(const run of eligible) {
    const annotation=annotations.get(run.id);
    const labels=annotation?[...new Set(annotation.findings.filter(f=>f[0]===category).map(f=>f[1]))]:['__missing'];
    if(!labels.length) labels.push('__none');
    for(const label of labels) {
      const item=weights.get(label)||{label,weight:0,runs:0};
      item.weight+=1/labels.length;item.runs++;
      weights.set(label,item);
    }
  }
  return {total:eligible.length,segments:[...weights.values()].map(s=>({...s,share:s.weight/eligible.length})).sort((a,b)=>Number(a.label.startsWith('__'))-Number(b.label.startsWith('__'))||b.share-a.share||a.label.localeCompare(b.label))};
}
