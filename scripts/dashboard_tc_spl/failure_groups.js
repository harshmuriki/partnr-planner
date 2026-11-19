export function failureGroups(runs, limit=6, category='failure') {
  const branched = r=>category==='failure'?!r.success:r.findings.some(f=>f[0]===category);
  const failed = runs.filter(branched), succeeded = runs.filter(r=>!branched(r));
  const map = new Map();
  for (const run of failed) {
    const labels = [...new Set(run.findings.filter(f=>f[0]===category).map(f=>f[1]))].sort();
    const key = JSON.stringify(labels);
    if (!map.has(key)) map.set(key,{key,labels,runs:[]});
    map.get(key).runs.push(run);
  }
  const exact = [...map.values()].sort((a,b)=>b.runs.length-a.runs.length||a.key.localeCompare(b.key));
  const labeled = exact.filter(g=>g.labels.length);
  const groups = labeled.slice(0,limit).map(g=>({...g,kind:'exact'}));
  if(labeled.length>limit) groups.push({key:'other',labels:[],kind:'other',combinations:labeled.length-limit,runs:labeled.slice(limit).flatMap(g=>g.runs)});
  const unknown = exact.find(g=>!g.labels.length);
  if(unknown) groups.push({...unknown,kind:'unlabeled'});
  groups.sort((a,b)=>(a.kind==='other')-(b.kind==='other')||b.runs.length-a.runs.length||a.key.localeCompare(b.key));
  return {total:runs.length,failed,succeeded,groups,exact};
}
