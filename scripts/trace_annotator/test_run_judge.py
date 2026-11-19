import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from judge_contract import LABELS, FAILURES, validate
from judge_evidence import action_attempts, catalog, packet_for, simulator
from run_judge import annotate_one, calibration_report, example_set, prepare, request_for


def prop(n=1):
    return {'function_name': 'is_on_top', 'args': {'object_handles': ['obj'], 'receptacle_handles': ['table'], 'number': n}}


def event(sequence, action=None, response='', replanned=False, terminal=False):
    return {'sequence': sequence, 'event': 'terminal' if terminal else 'boundary',
            'planner_observed': {'high_level_actions': {'0': action} if action else {},
                                 'responses': {'0': response}, 'replanned': {'0': replanned}}}


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'run'
    (root / 'input').mkdir(parents=True)
    (root / 'dataset').mkdir()
    ep = {'instruction': 'Place object on table.', 'evaluation_propositions': [prop()], 'evaluation_constraints': []}
    (root / 'input/dataset.json.gz').write_bytes(gzip.compress(json.dumps({'episodes': [ep]}).encode()))
    records = [{'event': 'start', 'sequence': 0},
               event(1, ['Pick', 'obj_0', None]),
               event(2, ['Pick', 'obj_0', None], 'Failed to pick! Object does not exist.'),
               event(3, ['Pick', 'obj_0', None], replanned=True),
               event(4, ['Pick', 'obj_0', None], 'Failed to pick! Object does not exist.'),
               event(5, terminal=True)]
    records[-1]['evaluator_only_state'] = {
        'auto_eval_proposition_tracker': {'propositions': [prop()], 'proposition_satisfied_at': [-1], 'state_sequence': []},
        'task_constraint_validation': [], 'task_percent_complete': 0, 'task_state_success': 0}
    trace_dir = root / 'dataset/traces/0'
    trace_dir.mkdir(parents=True)
    trace_dir.joinpath('trace-test.html').write_text(''.join(
        '<div class="step"><span class="step-action">Pick[obj_0]</span>'
        '<div class="section"><div class="section-label">Result</div>'
        '<div class="section-content">Failed to pick! Object does not exist.</div></div></div>'
        for _ in range(2)))
    evidence = root / 'dataset/evidence-test.jsonl'
    evidence.write_text('\n'.join(map(json.dumps, records))+'\n')
    manifest = tmp_path / 'source.json'
    manifest.write_text(json.dumps({'runs': [{'baseline': 'react', 'episode_id': 'T1-ACC-BASE', 'repetition': 2,
                                             'path': str(root), 'status': 'completed', 'metrics': {}}]}))
    run = catalog(manifest)[0]
    return root, manifest, run


def blank(packet):
    return {'summary': 'Reviewed', 'failures': [], 'inefficiencies': [],
            'label_assessments': [{'label': l, 'status': 'not_applicable' if l in FAILURES and packet['simulator']['outcome'] in ['complete','unavailable'] else 'absent' if packet['coverage']['history_complete'] else 'insufficient_evidence', 'reason': 'Reviewed history'} for l in LABELS],
            'unclassified_failed_criterion_ids': [c['id'] for c in packet['simulator']['criteria'] if c['status']=='unmet'],
            'important_notes': [], 'possible_failure_notes': []}


def incident(label='hallucinated_object'):
    return {'id': 'i1', 'behavior_id': 'b1', 'label': label, 'criterion_ids': [], 'unmet_group_ids': [],
            'attempt_ids': ['a0001'], 'evidence_ids': ['e0_3','e0_4'], 'explanation': 'Retried a missing object',
            'alternative_explanation': 'The initial absence discovery was exempt.', 'confidence_raw': .8,
            'object_name': 'obj_0', 'outdated_memory_object': True,
            'prior_failed_pick_evidence_ids': ['e0_2'], 'unmet_precondition': None}


def with_incident(packet, x, family='inefficiencies'):
    a = blank(packet)
    a[family] = [x]
    next(r for r in a['label_assessments'] if r['label']==x['label'])['status'] = 'present'
    if family == 'failures':a['unclassified_failed_criterion_ids'] = []
    return a


def test_catalog_keeps_explicit_repetition_not_r001(source):
    _, _, run = source
    assert run['id'].endswith('repeat_2')


def test_catalog_rejects_duplicate(source):
    _, p, _ = source
    d=json.loads(p.read_text());d['runs']*=2;p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='Duplicate'):catalog(p)


def test_runtime_unrolling_and_constraints():
    ep={'evaluation_propositions':[prop(2)]}
    state={'event':'terminal','_id':'e1','evaluator_only_state':{'auto_eval_proposition_tracker':{'propositions':[prop(2),prop(1)],'proposition_satisfied_at':[2,1]},'task_constraint_validation':[[False,True]]}}
    result=simulator(ep,[state],{})
    assert [c['status'] for c in result['criteria']]==['unmet','met']
    assert result['outcome']=='partial'


def test_intermediate_scores_never_become_terminal():
    state={'event':'boundary','evaluator_only_state':{'task_percent_complete':0}}
    assert simulator({'evaluation_propositions':[prop()]},[state],{})['outcome']=='unavailable'


def test_aggregate_partial_keeps_instance_ambiguity():
    s=simulator({'evaluation_propositions':[prop(),prop()]},[],{'task_percent_complete':.5})
    assert s['ambiguous_failure_groups']==[{'id':'g0','criterion_ids':['p0','p1'],'unmet_count':1}]
    assert all(c['status']=='unknown' for c in s['criteria'])


def test_attempts_collapse_ticks_but_keep_retry(source):
    packet=packet_for(source[2])
    assert len(packet['attempts'])==2
    assert packet['attempts'][0]['evidence_ids']==['e0_1','e0_2']
    assert packet['attempts'][1]['evidence_ids']==['e0_3','e0_4']


def test_valid_stale_memory_annotation(source):
    p=packet_for(source[2]);a=with_incident(p,incident());assert validate(a,p)==a


@pytest.mark.parametrize('change', ['missing_prior','reverse_order','wrong_object','duplicate','missing_attempt','bad_confidence','precondition'])
def test_invalid_incidents_rejected(source,change):
    p=packet_for(source[2]);x=incident();a=with_incident(p,x)
    if change=='missing_prior':x['prior_failed_pick_evidence_ids']=[]
    if change=='reverse_order':x['prior_failed_pick_evidence_ids']=['e0_4']
    if change=='wrong_object':x['object_name']='other_0'
    if change=='duplicate':a['inefficiencies'].append(copy.deepcopy(x))
    if change=='missing_attempt':x['attempt_ids']=[]
    if change=='bad_confidence':x['confidence_raw']=1.1
    if change=='precondition':
        x['label']='precondition_failures';a=with_incident(p,x)
    with pytest.raises(Exception):validate(a,p)


@pytest.mark.parametrize('outcome',['complete','unavailable'])
def test_no_failures_when_successful_or_unscored(source,outcome):
    p=packet_for(source[2]);p['simulator']['outcome']=outcome
    x=incident('placement');x['criterion_ids']=['p0']
    a=with_incident(p,x,'failures')
    with pytest.raises(ValueError,match='scored'):validate(a,p)


def test_missing_history_cannot_claim_absence(source):
    p=packet_for(source[2]);a=blank(p);p['coverage']['history_complete']=False
    with pytest.raises(ValueError,match='Incomplete'):validate(a,p)


def test_examples_validate_and_version_request(source,tmp_path):
    _,src,run=source;out=tmp_path/'judge';prepare(src,out)
    p=packet_for(run);r,meta=request_for(out,p)
    assert r['response_format']['json_schema']['strict'] is True
    assert run['id'] not in r['messages'][1]['content'][0]['text']
    ex={'enabled':True,'reviewed_by':'human','source_run_ids':[], 'input':p,'expected_output':blank(p)}
    (out/'examples/example.json').write_text(json.dumps(ex))
    assert len(example_set(out/'examples'))==1
    assert request_for(out,p)[1]['request_sha256']!=meta['request_sha256']
    ex['source_run_ids']=[run['id']];(out/'examples/example.json').write_text(json.dumps(ex))
    with pytest.raises(ValueError,match='prompt example'):request_for(out,p)


def test_context_limit_never_truncates(source,tmp_path):
    _,src,run=source;out=tmp_path/'judge';prepare(src,out)
    with pytest.raises(ValueError,match='no history was truncated'):request_for(out,packet_for(run),max_input_tokens=1)


def test_resume_and_change_invalidation(source,tmp_path):
    _,src,run=source;out=tmp_path/'judge';prepare(src,out)
    p=packet_for(run)
    x=incident(); x.update(attempt_ids=['step_2'], evidence_ids=['step_2'], prior_failed_pick_evidence_ids=['step_1'])
    answer=with_incident(p,x);calls=[]
    class Response:
        model='gpt-6-astra'
        choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(answer),refusal=None))]
        usage=None
        def model_dump(self,**kwargs):return {'model':self.model}
    def create(**kwargs):calls.append(kwargs);return Response()
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert annotate_one(out,run,client)['status']=='annotated'
    assert annotate_one(out,run,client)['status']=='cached'
    assert len(calls)==1
    (out/'system_prompt.txt').write_text((out/'system_prompt.txt').read_text()+'\nBe precise.\n')
    assert annotate_one(out,run,client)['status']=='annotated'
    assert len(calls)==2 and list((out/'history').rglob('*.json'))


def test_refusal_is_not_an_annotation(source,tmp_path):
    _,src,run=source;out=tmp_path/'judge';prepare(src,out)
    class Response:
        model='gpt-6-astra';choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(refusal='No'))]
        def model_dump(self,**kwargs):return {'refusal':'No'}
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:Response())))
    assert annotate_one(out,run,client,retries=0)['status']=='error'
    assert not (out/'analyses'/f'{run["id"]}.json').exists()


def test_annotation_credential_isolated(monkeypatch):
    import openai
    import dotenv
    from run_judge import annotation_client
    calls=[]
    monkeypatch.setenv('OPENAI_API_KEY','other-workload-key')
    monkeypatch.setenv('OPENAI_MODEL_PERSONAL','annotation-only-key')
    monkeypatch.setattr(openai,'OpenAI',lambda **kw:calls.append(kw) or object())
    monkeypatch.setattr(dotenv,'dotenv_values',lambda path:{})
    annotation_client()
    assert calls[0]['api_key']=='annotation-only-key'
    import os
    assert os.environ['OPENAI_API_KEY']=='other-workload-key'
    monkeypatch.delenv('OPENAI_MODEL_PERSONAL')
    with pytest.raises(ValueError,match='OPENAI_MODEL_PERSONAL'):annotation_client()
    assert len(calls)==1


def test_json_dedup_is_lossless():
    from judge_evidence import share_json_blocks
    original=[{'id':str(i),'data':{'nested':{'large':'x'*250},'i':i}} for i in range(3)]
    events,blocks=share_json_blocks(original)
    def expand(v):
        if isinstance(v,dict):
            if set(v)=={'json_ref'}:return expand(blocks[v['json_ref']])
            return {k:expand(x) for k,x in v.items()}
        if isinstance(v,list):return [expand(x) for x in v]
        return v
    assert blocks and expand(events)==original


def test_packet_schema_and_unknown_attempt_reference(source):
    from judge_contract import validate_packet
    p=packet_for(source[2]);validate_packet(p)
    p['attempts'][0]['evidence_ids']=['not_an_event']
    with pytest.raises(ValueError,match='unknown'):validate_packet(p)


def test_invalid_json_marks_coverage_incomplete(source):
    f=source[0]/'dataset/evidence-test.jsonl'
    with f.open('a') as s:s.write('{broken')
    p=packet_for(source[2])
    assert not p['coverage']['history_complete'] and p['coverage']['problems']
    assert p['simulator']['outcome']=='no_progress'


def test_legacy_planner_attempts_stay_partial(source):
    root,_,run=source
    (root/'dataset/evidence-test.jsonl').unlink()
    d=root/'dataset/planner-log';d.mkdir()
    (d/'log.json').write_text(json.dumps({'steps':[event(1,['Pick','obj_0',None])['planner_observed']]}))
    p=packet_for(run)
    assert len(p['attempts'])==1 and not p['coverage']['history_complete']
    assert p['simulator']['outcome']=='unavailable'


def test_calibration_report_and_stale_review(source,tmp_path):
    _,src,run=source;out=tmp_path/'judge';prepare(src,out)
    p=packet_for(run);answer=with_incident(p,incident())
    path=out/'analyses'/(run['id']+'.json');path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'provenance':{'request_sha256':'abc'},'analysis':answer}))
    reviews=tmp_path/'reviews.json'
    row={'run_id':run['id'],'incident_id':'i1','request_sha256':'abc','human_accept':True,'reviewer':'human'}
    reviews.write_text(json.dumps([row]))
    report=calibration_report(out,reviews)
    assert report['metrics']['all']['brier']==pytest.approx(.04)
    row['request_sha256']='old';reviews.write_text(json.dumps([row]))
    with pytest.raises(ValueError,match='different annotation'):calibration_report(out,reviews)


def test_example_draft_never_auto_enabled(source,tmp_path):
    from run_judge import make_example
    _,src,run=source;out=tmp_path/'judge';prepare(src,out)
    p=packet_for(run)
    path=out/'analyses'/(run['id']+'.json');path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'analysis':blank(p)}))
    result=make_example(out,run['id'],'my_example')
    assert result['enabled'] is False and not example_set(out/'examples')
    with pytest.raises(ValueError,match='already exists'):make_example(out,run['id'],'my_example')


def test_changed_examples_force_new_request(source,tmp_path):
    _,src,run=source;out=tmp_path/'judge';prepare(src,out)
    p=packet_for(run);_,a=request_for(out,p)
    ex={'enabled':True,'reviewed_by':'human','input':p,'expected_output':blank(p)}
    (out/'examples/test.json').write_text(json.dumps(ex))
    _,b=request_for(out,p)
    assert a['request_sha256']!=b['request_sha256']


def test_catalog_replacements_persist_for_future_batches(source, tmp_path, monkeypatch):
    import shutil
    import run_judge
    root, src, original = source
    replacement = tmp_path/'replacement'
    shutil.copytree(root, replacement)
    overrides = tmp_path/'overrides.json'
    rows = json.loads(src.read_text())['runs']
    rows[0]['path'] = str(replacement)
    overrides.write_text(json.dumps({'runs': rows}))
    current = catalog(src, overrides)
    assert len(current) == 1 and current[0]['id'] == original['id']
    assert current[0]['path'] == str(replacement)
    assert current[0]['supersedes']['path'] == str(root)
    out = tmp_path/'judge'
    prepare(src, out, overrides=overrides)
    prepare(src, out)  # A later preparation retains the configured override.
    assert json.loads((out/'manifest.json').read_text())['overrides'] == str(overrides)
    monkeypatch.setattr(run_judge, 'annotation_client', lambda: pytest.fail('Dry run must not initialize an API client'))
    result = run_judge.run_batch(out, limit=1)
    assert result[0]['status'] == 'dry_run'
    packet = json.loads((out/'packets'/f"{original['id']}.json").read_text())
    assert packet['provenance']['run_path'] == str(replacement)
    assert packet['provenance']['supersedes']['path'] == str(root)


@pytest.mark.parametrize('change', ['duplicate', 'unknown', 'pending', 'missing_path'])
def test_invalid_replacement_manifests_are_rejected(source, tmp_path, change):
    root, src, _ = source
    row = json.loads(src.read_text())['runs'][0]
    if change == 'unknown': row['episode_id'] = 'T9-ACC-BASE'
    if change == 'pending': row['status'] = 'running'
    if change == 'missing_path': row['path'] = str(tmp_path/'missing')
    overrides = tmp_path/'overrides.json'
    overrides.write_text(json.dumps({'runs': [row,row] if change == 'duplicate' else [row]}))
    with pytest.raises(ValueError): catalog(src, overrides)


def checkpoint(step, satisfied=False):
    return {'event': 'checkpoint', 'wall_time': 100+step, 'elapsed_seconds': step,
            'sim_step_count': step, 'evaluator_only_state': {
                'auto_eval_proposition_tracker': {'propositions': [prop()],
                    'proposition_satisfied_at': [step if satisfied else -1],
                    'state_sequence_length': step+1, 'last_state': [{'is_satisfied': satisfied}]},
                'task_constraint_validation': [], 'task_percent_complete': int(satisfied),
                'task_state_success': int(satisfied)}}


def add_checkpoints(root, final=True):
    if not final:
        path=root/'dataset/evidence-test.jsonl'
        lines=[json.loads(l) for l in path.read_text().splitlines()]
        path.write_text('\n'.join(json.dumps(r) for r in lines if r['event']!='terminal')+'\n')
    (root/'dataset/evaluation-checkpoints.jsonl').write_text(json.dumps(checkpoint(0))+'\n'+json.dumps(checkpoint(25))+'\n')
    (root/'dataset/evaluation-checkpoint.json').write_text(json.dumps(checkpoint(26,True))+'\n')
    (root/'run.json').write_text(json.dumps({'started':100,'elapsed_seconds':26.1}))


def test_checkpoints_are_separate_citable_observations_not_final_or_attempts(source):
    from judge_contract import validate_packet
    root, _, run = source
    add_checkpoints(root, final=False)
    packet = validate_packet(packet_for(dict(run,status='timeout')))
    assert packet['simulator']['outcome'] == 'unavailable'
    assert packet['simulator']['criteria'][0]['status'] == 'unknown'
    snap = packet['checkpoint_evaluation']
    assert snap['outcome'] == 'complete'
    assert snap['source_event_id'] == 'checkpoint1_1'
    assert snap['eligible_for_failure_labels'] is False
    assert snap['seconds_before_process_end'] == pytest.approx(.1)
    assert packet['coverage']['checkpoint_count'] == 3
    assert packet['coverage']['history_complete'] is False
    assert len(packet['attempts']) == 2
    assert all(not e.startswith('checkpoint') for a in packet['attempts'] for e in a['evidence_ids'])
    assert all(f'dataset/{name}' in packet['provenance']['files_sha256'] for name in ['evaluation-checkpoints.jsonl','evaluation-checkpoint.json'])
    with pytest.raises(ValueError, match='scored'):
        validate(with_incident(packet, dict(incident('placement'),criterion_ids=['p0']), 'failures'),packet)


def test_final_scores_win_over_conflicting_checkpoint(source):
    root, _, run=source
    add_checkpoints(root)
    packet=packet_for(run)
    assert packet['simulator']['outcome']=='no_progress'
    assert packet['checkpoint_evaluation']['outcome']=='complete'


def test_initial_only_checkpoint_does_not_establish_outcomes(source):
    root, _, run=source
    add_checkpoints(root,final=False)
    (root/'dataset/evaluation-checkpoints.jsonl').unlink()
    (root/'dataset/evaluation-checkpoint.json').write_text(json.dumps(checkpoint(0)))
    packet=packet_for(run)
    assert packet['checkpoint_evaluation'] is None
    assert packet['simulator']['outcome']=='unavailable'


def test_malformed_latest_checkpoint_falls_back_to_history_and_changes_fingerprint(source):
    root, _, run=source
    add_checkpoints(root,final=False)
    before=packet_for(run)
    (root/'dataset/evaluation-checkpoint.json').write_text('{"event":')
    after=packet_for(run)
    assert after['checkpoint_evaluation']['source_event_id']=='checkpoint0_2'
    assert after['checkpoint_evaluation']['outcome']=='no_progress'
    assert after['coverage']['problems']
    assert before['provenance']['evidence_sha256'] != after['provenance']['evidence_sha256']


@pytest.mark.parametrize('initial_only', [False, True])
def test_checkpoint_request_policy_preserves_edited_prompt_and_no_api(source,tmp_path,initial_only):
    from judge_prompt import PROMPT_POLICY
    root,src,run=source
    add_checkpoints(root,final=False)
    if initial_only:
        (root/'dataset/evaluation-checkpoints.jsonl').unlink()
        (root/'dataset/evaluation-checkpoint.json').write_text(json.dumps(checkpoint(0)))
    out=tmp_path/'judge';prepare(src,out)
    (out/'system_prompt.txt').write_text('Custom user instruction.\n')
    request,_=request_for(out,packet_for(run))
    system=request['messages'][0]['content']
    assert system.startswith('Custom user instruction.') and 'checkpoint scores cannot authorize failure labels' in system
    assert (out/'system_prompt.txt').read_text()=='Custom user instruction.\n'


def test_checkpoint_delta_is_lossless_for_add_remove_resize_and_type_changes():
    from judge_evidence import checkpoint_delta
    before={'nodes':[{'name':'robot','pose':[1,2,3],'stale':True}, {'name':'object'}],
            'removed': {'nested':1}, 'resize':[1,2], 'type':{'old':1}}
    after={'nodes':[{'name':'robot','pose':[4,2,3],'new':False}, {'name':'object'}],
           'resize':[1,2,3], 'type':'changed', 'added':None}
    assignments,removals=checkpoint_delta(before,after)
    value=copy.deepcopy(before)
    for path in removals:
        parent=value
        for part in path[:-1]:parent=parent[part]
        del parent[path[-1]]
    for update in assignments:
        path=update['path'];parent=value
        for part in path[:-1]:parent=parent[part]
        if path:parent[path[-1]]=update['value']
        else:value=update['value']
    assert value==after and before['nodes'][0]['pose']==[1,2,3]
    assert checkpoint_delta([1],{'new':1})==([{'path':[],'value':{'new':1}}],[])


def test_packet_checkpoint_deltas_reconstruct_every_saved_state_with_bounded_anchors(source):
    root,_,run=source
    snapshots=[]
    for i in range(103):
        r=checkpoint(i)
        r['evaluator_only_state']['reference_world_graph']=[
            {'name':f'node{n}','properties':{'description':'unchanged '*40,'translation':[i if n==0 else n,0,0]}}
            for n in range(10)]
        snapshots.append(r)
    (root/'dataset/evaluation-checkpoints.jsonl').write_text('\n'.join(json.dumps(r) for r in snapshots)+'\n')
    packet=packet_for(run)
    def expand(value):
        if isinstance(value,dict):
            if set(value)=={'json_ref'}:return expand(packet['json_blocks'][value['json_ref']])
            return {k:expand(v) for k,v in value.items()}
        if isinstance(value,list):return [expand(v) for v in value]
        return value
    reconstructed={};deltas=0
    for e,r in zip([e for e in packet['events'] if e['kind']=='evaluator_checkpoint'],snapshots):
        state=expand(e['data'])['evaluator_only_state']
        if 'checkpoint_delta' in state:
            deltas+=1;patch=state['checkpoint_delta'];state=copy.deepcopy(reconstructed[patch['base_event_id']])
            for path in patch['remove']:
                parent=state
                for part in path[:-1]:parent=parent[part]
                del parent[path[-1]]
            for update in patch['set']:
                parent=state;path=update['path']
                for part in path[:-1]:parent=parent[part]
                if path:parent[path[-1]]=update['value']
                else:state=update['value']
        reconstructed[e['id']]=state
        assert state==r['evaluator_only_state']
    assert deltas==100  # Full states at 0, 50 and 100.


def test_annotation_request_is_text_only_and_rejects_images(source,tmp_path):
    _,src,run=source;out=tmp_path/'judge';prepare(src,out)
    packet=packet_for(run)
    request,meta=request_for(out,packet)
    assert [item['type'] for item in request['messages'][1]['content']]==['text']
    assert meta['image_count']==0
    assert 'No images or video frames are supplied' in request['messages'][0]['content']
    with pytest.raises(ValueError,match='text-only'):
        request_for(out,packet,images=[{'path':'/not/read/frame.png','evidence_ids':['e0_1']}])
    packet['image_inputs']=[{'evidence_ids':['e0_1']}]
    with pytest.raises(ValueError,match='text-only'):
        request_for(out,packet)


def test_image_manifest_rejected_before_client_or_files(tmp_path,monkeypatch):
    import run_judge
    def forbidden():raise AssertionError('Must not construct annotation client')
    monkeypatch.setattr(run_judge,'annotation_client',forbidden)
    with pytest.raises(ValueError,match='text-only'):
        run_judge.run_batch(tmp_path/'no-manifest',execute=True,image_manifest=tmp_path/'no-images.json')


def test_image_based_prompt_example_rejected(source,tmp_path):
    _,src,run=source;out=tmp_path/'judge';prepare(src,out)
    packet=packet_for(run);example_packet=copy.deepcopy(packet)
    example_packet['image_inputs']=[{'evidence_ids':['e0_1']}]
    (out/'examples/image.json').write_text(json.dumps({'enabled':True,'reviewed_by':'human',
        'source_run_ids':[],'input':example_packet,'expected_output':blank(packet)}))
    with pytest.raises(ValueError,match='text-only'):
        request_for(out,packet)
