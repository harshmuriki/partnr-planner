import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {failureGroups} from './failure_groups.js';
const data=JSON.parse(readFileSync('../../results/all_baselines_5reps_20261005_001408/uncertainty_classifier/dashboard_data.json'));
const run=(id,success,labels)=>({id,success,findings:labels.map(x=>['failure',x,'evidence'])});
const sample=[run('1',true,[]),run('2',false,['a','b']),run('3',false,['b','a','a']),run('4',false,[]),run('5',false,['c'])];
const result=failureGroups(sample,1);
assert.deepEqual(result.groups.map(g=>[g.kind,g.runs.length]),[['exact',2],['unlabeled',1],['other',1]]);
assert.equal(failureGroups([],6).groups.length,0);
for(const baseline of ['react','vlm-tamp','tru-pomdp']){
 const rows=data.runs.filter(r=>r.baseline===baseline),c=failureGroups(rows);
 assert.equal(c.total,450);assert.equal(c.failed.length+c.succeeded.length,450);
 assert.equal(c.groups.at(-1).kind,'other');
 const ordered=c.groups.filter(g=>g.kind!=='other');
 assert.ok(ordered.every((g,i)=>i===0||ordered[i-1].runs.length>=g.runs.length));
 const ids=c.groups.flatMap(g=>g.runs.map(r=>r.id));assert.equal(ids.length,c.failed.length);assert.equal(new Set(ids).size,ids.length);
 for(const g of c.groups.filter(g=>g.kind==='exact'))for(const r of g.runs)assert.equal(JSON.stringify([...new Set(r.findings.filter(f=>f[0]==='failure').map(f=>f[1]))].sort()),g.key);
 console.log(baseline,JSON.stringify({success:c.succeeded.length,failed:c.failed.length,combinations:c.exact.length,shown:c.groups.length,unlabeled:c.groups.find(g=>g.kind==='unlabeled')?.runs.length||0}));
}
console.log('PASS: disjoint complete partitions, repeated labels, other combinations, unlabeled and empty inputs');
for(const baseline of ['react','vlm-tamp','tru-pomdp']){
 const rows=data.runs.filter(r=>r.baseline===baseline),c=failureGroups(rows,6,'inefficiency');
 assert.equal(c.total,450);assert.equal(c.failed.length+c.succeeded.length,450);
 assert.ok(c.failed.some(r=>r.success));
 assert.ok(c.succeeded.every(r=>!r.findings.some(f=>f[0]==='inefficiency')));
 const ids=c.groups.flatMap(g=>g.runs.map(r=>r.id));assert.equal(ids.length,c.failed.length);assert.equal(new Set(ids).size,ids.length);
 const ordered=c.groups.filter(g=>g.kind!=='other');assert.ok(ordered.every((g,i)=>i===0||ordered[i-1].runs.length>=g.runs.length));
 if(c.groups.some(g=>g.kind==='other'))assert.equal(c.groups.at(-1).kind,'other');
 console.log(baseline,JSON.stringify({flagged:c.failed.length,unflagged:c.succeeded.length,successfulFlagged:c.failed.filter(r=>r.success).length}));
}
console.log('PASS: inefficiency includes successful runs, exclusive partitions and descending order with Other last');
