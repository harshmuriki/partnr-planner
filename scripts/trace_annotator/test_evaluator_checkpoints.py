import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def module():
    path = Path(__file__).resolve().parents[2] / 'habitat_llm/utils/annotation_evidence.py'
    spec = importlib.util.spec_from_file_location('checkpoint_evidence', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_step_checkpoint_survives_interruption_without_terminal(tmp_path):
    m = module()
    writer = m.AnnotationEvidenceWriter(tmp_path, 'episode', 'task')
    writer.checkpoint(lambda: {'task_percent_complete': 0}, initial=True)
    for i in range(1, 28):
        writer.checkpoint(lambda: {'task_percent_complete': i/30})
    latest = json.loads((tmp_path/'evaluation-checkpoint.json').read_text())
    assert latest['event'] == 'checkpoint' and latest['sim_step_count'] == 27
    assert latest['evaluator_only_state']['task_percent_complete'] == 27/30
    history = [json.loads(l) for l in (tmp_path/'evaluation-checkpoints.jsonl').read_text().splitlines()]
    assert [r['sim_step_count'] for r in history] == [0,25]
    assert not any(json.loads(l)['event'] == 'terminal' for l in writer.path.read_text().splitlines())


def test_compact_snapshot_does_not_mutate_tracker_or_grow_with_history():
    m = module()
    tracker = {'proposition_satisfied_at':[3], 'state_sequence':[[False]]*1000+[[True]], 'propositions':[]}
    s=m.evaluator_snapshot({'auto_eval_proposition_tracker': SimpleNamespace(get_metric=lambda: tracker)})
    t=s['auto_eval_proposition_tracker']
    assert 'state_sequence' not in t and len(tracker['state_sequence'])==1001
    assert t['last_state']==[True] and t['state_sequence_length']==1001
    assert len(json.dumps(s))<300


def test_checkpoint_recovery_keeps_final_metrics_and_rejects_initial_state(tmp_path):
    import gzip
    from dashboard_checkpoint_results import recover_checkpoint
    prop={'function_name':'is_clean','args':{'object_handles':['plate'],'number':1}}
    (tmp_path/'input').mkdir();(tmp_path/'dataset').mkdir()
    (tmp_path/'input/dataset.json.gz').write_bytes(gzip.compress(json.dumps({'episodes':[{'evaluation_propositions':[prop]}]}).encode()))
    path=tmp_path/'dataset/evaluation-checkpoint.json'
    record={'event':'checkpoint','sim_step_count':0,'wall_time':110,'elapsed_seconds':2,'evaluator_only_state':{
        'auto_eval_proposition_tracker':{'propositions':[prop],'proposition_satisfied_at':[-1],'state_sequence_length':1},
        'task_constraint_validation':[],'task_percent_complete':0}}
    path.write_text(json.dumps(record))
    row={'completion':None,'run_folder':str(tmp_path),'runtime_s':20}
    assert recover_checkpoint(row) is None
    record['sim_step_count']=4;path.write_text(json.dumps(record))
    (tmp_path/'run.json').write_text(json.dumps({'started':100}))
    recovered=recover_checkpoint(row)
    assert recovered['criteria'][0]['status']=='unmet'
    assert recovered['score_basis']=='last_saved_state'
    assert recovered['checkpoint']['seconds_before_process_end']==10
    assert row['completion'] is None
    assert recover_checkpoint(dict(row,completion=1)) is None


def test_reverse_reader_skips_truncated_tail_and_handles_large_records(tmp_path):
    from dashboard_checkpoint_results import reverse_records
    p=tmp_path/'evidence.jsonl'
    p.write_text(json.dumps({'event':'boundary','data':'x'*300})+'\n'+json.dumps({'event':'checkpoint'})+'\n{"broken"')
    records=list(reverse_records(p,block_size=13))
    assert [r['event'] for r in records]==['checkpoint','boundary']


def test_completed_rerun_replaces_dashboard_row_and_retains_old_artifacts(tmp_path, monkeypatch):
    import refresh_dashboard_checkpoints as publisher
    from build_baseline_dashboard import build
    from build_final_baselines import prepare_run
    from test_build_final_baselines import make_job
    html=tmp_path/'html'; html.mkdir()
    dashboard=tmp_path/'baseline_dashboard.html'
    monkeypatch.setattr(publisher,'ROOT',html)
    monkeypatch.setattr(publisher,'DASHBOARD',dashboard)
    old=make_job(tmp_path,1)
    old.update(planner='tru-pomdp',episode_id='T7-ACC-BASE')
    row=prepare_run(old,tmp_path/'old-summary.json',html,tmp_path)
    (html/'baseline_runs_comparison.json').write_text(json.dumps([row]))
    (html/'dashboard_data.json').write_text(json.dumps(build([row],{})))
    new=make_job(tmp_path,2,{'task_percent_complete':1,'task_state_success':1})
    new.update(planner='tru-pomdp',episode_id='T7-ACC-BASE',repetition=1)
    rerun=tmp_path/'rerun'; rerun.mkdir()
    attempt=rerun/'attempt_000';attempt.mkdir()
    (attempt/'summary.json').write_text(json.dumps({'jobs':[new]}))
    (rerun/'batch.json').write_text(json.dumps({'jobs':[{'experiment':str(attempt),'episode_id':'T7-ACC-BASE','repetition':1}]}))
    assert publisher.refresh(rerun)==0
    saved=json.loads((html/'baseline_runs_comparison.json').read_text())
    assert len(saved)==1 and saved[0]['id']=='tru-pomdp/T7-ACC-BASE/repeat_1'
    assert saved[0]['run_folder']==new['output']
    assert saved[0]['supersedes_run_folder']==old['output']
    assert saved[0]['completion']==1
    assert (html/'tru-pomdp/task_7/T7-ACC-BASE/repeat_1/artifacts').resolve()==Path(old['output'])
    assert (html/'rerun/tru-pomdp/task_7/T7-ACC-BASE/repeat_1/artifacts').resolve()==Path(new['output'])
    assert publisher.refresh(rerun)==0
    override=json.loads((html/'rerun_overrides.json').read_text())['jobs'][0]
    assert override['supersedes_run_folder']==old['output']
