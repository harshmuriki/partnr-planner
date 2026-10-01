// Run with a modern Node.js: node scripts/test_ground_truth_metrics.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {parseCSV, metricValue, delta, mean, recordingTiming, clock} = require('../baseline_evaluation_v3/gui/metrics.js');
assert.deepEqual(parseCSV('skill,target\r\nPlace,"jug,on,table"\r\nExplore,"a\n""b"""'), [{skill:'Place',target:'jug,on,table'},{skill:'Explore',target:'a\n"b"'}]);
const missing = {id:'T1-ACC-ABS',task:'T1',mem:'ACC'};
const base = {id:'T1-ACC-BASE',task:'T1',mem:'ACC',run:{action_count:10,sim_steps:100,steps_complete:1},skills:{Navigate:4}};
const other = {...base,id:'T1-ACC-CON',run:{action_count:12,sim_steps:150,steps_complete:0},skills:null};
assert.equal(metricValue(missing,'action_count'),null);
assert.equal(metricValue(base,'Explore'),0);
assert.equal(metricValue(other,'Explore'),null);
assert.equal(metricValue(other,'sim_steps'),null);
assert.equal(delta(other,[base,other],'action_count'),2);
assert.equal(delta(other,[base,other],'sim_steps'),null);
assert.equal(delta(other,[other],'action_count'),null);
assert.deepEqual(mean([base,missing,other],'sim_steps'),{n:1,value:100});
assert.deepEqual(mean([missing],'action_count'),{n:0,value:null});
assert.equal(metricValue({...base,run:{action_count:0}},'action_count'),0);
assert.equal(metricValue({...base,run:{video_duration_sec:12.5}},'video_duration_sec'),12.5);
assert.equal(metricValue({...base,run:{video_duration_sec:null}},'video_duration_sec'),null);
// Verify against actual saved archive totals without mutating it.
const root = 'baseline_evaluation_v3/ground_truth';
const runs = JSON.parse(fs.readFileSync(`${root}/index.json`)).ground_truths;
for (const run of runs) {
 const actions = parseCSV(fs.readFileSync(`${root}/${run.variant.split('-')[0]}/${run.variant}/actions.csv`,'utf8'));
 assert.equal(actions.length,run.action_count,run.variant);
 if (run.steps_complete) assert.equal(actions.reduce((n,a) => n+Number(a.simulator_steps),0),run.sim_steps,run.variant);
}
console.log(`Metric edge cases and ${runs.length} saved recording totals passed.`);

const evidence = {run_id:'current', timing_status:'recovered_current', execution_elapsed_seconds:'2098', interval_definition:'loaded_to_success'};
assert.equal(recordingTiming({id:'current'}, evidence).execution_elapsed_seconds,2098);
assert.equal(recordingTiming({id:'replaced'}, evidence).execution_elapsed_seconds,null);
for (const value of ['', '-1', 'NaN', 'Infinity']) {
 assert.equal(recordingTiming({id:'current'}, {...evidence, execution_elapsed_seconds:value}).execution_elapsed_seconds,null);
}
assert.equal(recordingTiming({id:'current'}, {...evidence, timing_status:'interrupted'}).execution_elapsed_seconds,null);
assert.equal(recordingTiming({id:'current'}, {...evidence, execution_elapsed_seconds:'0'}).execution_elapsed_seconds,0);
assert.equal(recordingTiming({id:'current'}, evidence).action_wall_seconds,null);
const skynet = {...evidence, timing_status:'measured_skynet', action_wall_seconds:'600', source:'skynet_x/T6-OUT-CON.json'};
assert.equal(recordingTiming({id:'current'}, skynet).execution_elapsed_seconds,2098);
assert.equal(recordingTiming({id:'current'}, skynet).action_wall_seconds,600);
assert.equal(recordingTiming({id:'current'}, skynet).timing_source,'skynet_x/T6-OUT-CON.json');
const timings = new Map(parseCSV(fs.readFileSync('baseline_evaluation_v3/ground_truth_timing/current_recordings.csv','utf8')).map(r => [r.variant,r]));
const timed = runs.map(run => ({task:run.variant.split('-')[0],run:{...run,...recordingTiming(run,timings.get(run.variant))}}));
const overall = mean(timed,'execution_elapsed_seconds');
const taskMeans = [...new Set(timed.map(r => r.task))].map(task => mean(timed.filter(r => r.task === task),'execution_elapsed_seconds'));
assert.equal(taskMeans.reduce((n,t) => n+t.n,0),overall.n);
assert.ok(Math.abs(taskMeans.reduce((n,t) => n+(t.value || 0)*t.n,0)/overall.n-overall.value)<1e-9);
assert.equal(clock(2098),'34m 58.0s');
console.log(`Elapsed timings: ${overall.n}/${runs.length}; overall ${clock(overall.value)}; T6 ${clock(mean(timed.filter(r => r.task === 'T6'),'execution_elapsed_seconds').value)}.`);

assert.equal(recordingTiming({id:'current'}, {...evidence, timing_status:'measured_single_sandbox', action_wall_seconds:'2000.25'}).action_wall_seconds,2000.25);
assert.equal(recordingTiming({id:'old'}, {...evidence, timing_status:'measured_single_sandbox', action_wall_seconds:'2000.25'}).action_wall_seconds,null);
