"""Regression tests for exactly what is sent and what evidence may be cited."""
import copy
import json

import pytest

from judge_html_input import Names, read_trace, build_input, validation_view, compact_observations
from judge_contract import validate
from test_run_judge import source, blank, incident, with_incident
from judge_evidence import packet_for
from run_judge import prepare, request_for


def test_html_steps_keep_retries_and_exclude_markup(source):
    root, _, _ = source
    path = root / 'dataset/traces/0/trace-test.html'
    path.write_text('<style>SECRET_CSS</style><script>SECRET_SCRIPT</script>' + path.read_text() + '<img src="data:image/png;base64,SECRET_IMAGE">')
    trace, hashes = read_trace(root)
    assert len(trace['steps']) == 2
    assert trace['steps'][0]['action'] == trace['steps'][1]['action'] == 'Pick[obj_0]'
    assert trace['steps'][0]['id'] != trace['steps'][1]['id']
    assert 'SECRET_' not in json.dumps(trace)
    assert len(hashes) == 1


def test_explicit_skip_status_reaches_judge_without_hiding_feedback(source):
    root, _, _ = source
    path = root / 'dataset/traces/0/trace-test.html'
    updated = path.read_text().replace('class="step"', 'class="step skipped" data-status="skipped" data-target-known-before-action="true" data-react-skip-policy="robot_sg_before_action"')
    path.write_text(updated)
    trace, _ = read_trace(root)
    for step in trace['steps']:
        assert step['status'] == 'skipped'
        assert step['counts_as_error'] is False
        assert 'Object does not exist.' in step['result']


def test_missing_html_does_not_fall_back_to_raw_logs(source):
    root, _, run = source
    (root / 'dataset/traces/0/trace-test.html').unlink()
    body, _ = build_input(packet_for(run), 'rubric', [])
    assert not body['trace']['available']
    assert not body['trace']['history_complete']
    assert body['trace']['steps'] == []
    assert body['subtask_results']['subtasks'][0]['result'] == 'failed'


def test_scene_collapses_receptacles_and_keeps_states(source):
    _, _, run = source
    packet = packet_for(run)
    graph = [
        {'name': 'room_1', 'sim_handle': 'r', 'properties': {}, 'relations': {'house': 'inside'}},
        {'name': 'table_0', 'sim_handle': 'table', 'properties': {'states': {'is_clean': False}, 'translation': [99,99,99]}, 'relations': {'room_1': 'inside'}},
        {'name': 'rec_table_0', 'sim_handle': 'mesh', 'properties': {}, 'relations': {'table_0': 'joint'}},
        {'name': 'obj_0', 'sim_handle': 'obj', 'properties': {'states': {'is_powered_on': True}}, 'relations': {'rec_table_0': 'on'}},
        {'name': 'agent_0', 'properties': {'initial_room': 'room_1'}, 'relations': {'room_1': 'inside'}},
    ]
    packet['events'][0]['data']['evaluator_only_state'] = {'reference_world_graph': graph}
    body, _ = build_input(packet, 'rubric', [])
    scene = body['initial_scene']
    assert scene['rooms'] == {'room_1': 'table_0(is_clean=false), obj_0(on table_0, is_powered_on=true), agent_0'}
    assert not {'objects', 'agents', 'furniture'} & scene.keys()
    text = json.dumps(body)
    assert 'sim_handle' not in text and 'translation' not in text and 'rec_table_0' not in text
    assert body['subtask_results']['subtasks'][0]['criterion']['objects'] == ['obj_0']


def test_terminal_state_is_not_initial(source):
    _, _, run = source
    packet = packet_for(run)
    packet['events'][-1]['data']['evaluator_only_state']['reference_world_graph'] = [
        {'name':'obj_0', 'sim_handle':'obj', 'properties':{'states':{'is_clean':True}}, 'relations':{}}]
    names = Names(packet)
    assert names.initial is None
    assert names.name('obj') == 'obj_0'  # Stable identity may still be resolved.


def test_compact_scene_keeps_empty_rooms_and_unlocated_entities(source):
    _, _, run = source
    packet = packet_for(run)
    names = Names(packet)
    names.initial = [
        {'name': 'room_1', 'relations': {'house': 'inside'}},
        {'name': 'room_2', 'relations': {'house': 'inside'}},
        {'name': 'agent_0', 'properties': {'initial_room': 'room_2'}},
        {'name': 'obj_0', 'properties': {'states': {'is_clean': False}}},
    ]
    assert names.scene(packet)['rooms'] == {
        'room_1': '', 'room_2': 'agent_0', 'unknown_room': 'obj_0(is_clean=false)'}


def test_only_displayed_ids_validate_and_stale_pick_exemption_survives(source):
    _, _, run = source
    packet = packet_for(run)
    body, _ = build_input(packet, 'rubric', [])
    shown = validation_view(body, packet)
    claim = incident()
    claim.update(attempt_ids=['step_2'], evidence_ids=['step_2'], prior_failed_pick_evidence_ids=['step_1'])
    assert validate(with_incident(shown, claim), shown)
    claim['evidence_ids'] = ['e0_4']
    with pytest.raises(ValueError, match='unknown evidence'):
        validate(with_incident(shown, claim), shown)


def test_html_change_invalidates_request_without_raw_change(source, tmp_path):
    root, src, run = source
    out = tmp_path / 'judge'
    prepare(src, out)
    packet = packet_for(run)
    old, old_meta = request_for(out, packet)
    path = root / 'dataset/traces/0/trace-test.html'
    path.write_text(path.read_text().replace('Object does not exist.', 'Object not found; removed.'))
    new, new_meta = request_for(out, packet)
    assert old_meta['packet_sha256'] == new_meta['packet_sha256']
    assert old_meta['request_sha256'] != new_meta['request_sha256']
    assert old_meta['html_files_sha256'] != new_meta['html_files_sha256']
    assert set(json.loads(new['messages'][1]['content'][0]['text'])) == {
        'scenario', 'initial_scene', 'initial_robot_memory', 'trace', 'subtask_results', 'definitions', 'examples', 'output_format'}


def test_planning_is_context_not_an_attempt_and_source_dumps_excluded(source):
    root, _, run = source
    directory = root / 'vlm_tamp_pddl'
    directory.mkdir()
    directory.joinpath('test_pddl.html').write_text('''<div id="vt-current"><details class="step">
    <span class="step-action">on(obj_0, table_0)</span>
    <div class="section"><div class="section-label">Response</div><div class="section-content">Move obj_0.</div></div>
    <div class="section"><div class="section-label">Matched events</div><div class="section-content">RAW_DUMP</div></div>
    </details></div><div id="vt-tree"><div class="step"><span class="step-action">DUPLICATE_TREE</span></div></div>''')
    packet = packet_for(run)
    body, _ = build_input(packet, 'rubric', [])
    assert body['trace']['planning'] == [{'id':'plan_1', 'subgoal':'on(obj_0, table_0)', 'Response':'Move obj_0.'}]
    assert len(validation_view(body, packet)['attempts']) == 2
    assert 'RAW_DUMP' not in json.dumps(body) and 'DUPLICATE_TREE' not in json.dumps(body)


def test_checkpoint_unknowns_remain_separate(source):
    _, _, run = source
    packet = packet_for(run)
    checkpoint = copy.deepcopy(packet['simulator'])
    packet['checkpoint_evaluation'] = checkpoint
    packet['simulator']['outcome'] = 'unavailable'
    for criterion in packet['simulator']['criteria']:
        criterion['status'] = 'unknown'
    body, _ = build_input(packet, 'rubric', [])
    score = body['subtask_results']
    assert score['score_basis'] == 'unavailable'
    assert score['subtasks'][0]['result'] == 'unknown'
    assert score['last_saved_state']['subtasks'][0]['result'] == 'failed'
    assert validation_view(body, packet)['simulator']['outcome'] == 'unavailable'


def test_no_image_baselines_rejected_before_api_or_manifest(tmp_path, monkeypatch):
    import run_judge
    monkeypatch.setattr(run_judge, 'annotation_client', lambda: pytest.fail('No API client should be constructed'))
    for baseline in ('react-no-image', 'vlm-tamp-no-image'):
        with pytest.raises(ValueError, match='no-image baselines are excluded'):
            run_judge.run_batch(tmp_path, baseline=baseline, execute=True)
        with pytest.raises(ValueError, match='no-image baselines are excluded'):
            run_judge.run_batch(tmp_path, run_id=baseline + '/T1-ACC-BASE/repeat_1', execute=True)


def test_no_html_skips_paid_call(source, tmp_path):
    from run_judge import annotate_one
    root, src, run = source
    (root / 'dataset/traces/0/trace-test.html').unlink()
    out = tmp_path / 'judge'
    prepare(src, out)
    assert annotate_one(out, run, None, execute=True)['status'] == 'unavailable'
    assert not list((out / 'attempts').glob('**/*.json'))


def test_default_batch_selects_only_three_baselines(source, tmp_path, monkeypatch):
    import run_judge
    _, src, run = source
    out = tmp_path / 'judge'
    prepare(src, out)
    all_runs = [dict(run, baseline=b, id=b + '/T1-ACC-BASE/repeat_1') for b in run_judge.BASELINES]
    monkeypatch.setattr(run_judge, 'catalog', lambda *args: all_runs)
    monkeypatch.setattr(run_judge, 'export', lambda *args: None)
    seen = []
    def annotate(out, run, *args, **kwargs):
        seen.append(run['baseline'])
        return {'run_id': run['id'], 'status': 'dry_run'}
    monkeypatch.setattr(run_judge, 'annotate_one', annotate)
    run_judge.run_batch(out, execute=False)
    assert set(seen) == {'react', 'vlm-tamp', 'tru-pomdp'}


def test_new_example_uses_archived_paid_prompt_not_latest_preview(source, tmp_path):
    from run_judge import make_example, example_set
    _, src, run = source
    out = tmp_path / 'judge'
    prepare(src, out)
    packet = packet_for(run)
    request, provenance = request_for(out, packet)
    body = json.loads(request['messages'][1]['content'][0]['text'])
    shown = validation_view(body, packet)
    claim = incident()
    claim.update(attempt_ids=['step_2'], evidence_ids=['step_2'], prior_failed_pick_evidence_ids=['step_1'])
    attempt = out / 'archived_attempt.json'
    attempt.write_text(json.dumps({'request': request}))
    analysis = out / 'analyses' / (run['id'] + '.json')
    analysis.parent.mkdir(parents=True)
    analysis.write_text(json.dumps({'analysis': with_incident(shown, claim),
                                    'provenance': dict(provenance, attempt_file=str(attempt))}))
    draft = make_example(out, run['id'], 'html_example')
    path = __import__('pathlib').Path(draft['example'])
    item = json.loads(path.read_text())
    assert item['model_input'] == body
    assert not item['enabled']
    item.update(enabled=True, reviewed_by='test reviewer')
    path.write_text(json.dumps(item))
    assert len(example_set(out / 'examples')) == 1


def test_observation_compaction_preserves_changes_and_every_attempt():
    steps = [{'id': f'step_{i}', 'action': 'Pick[obj_0]', 'result': 'Failure',
              'observations': {'Observed evidence': {'states': {'obj_0': {'clean': value}}},
                               'Search and belief diagnostics': {'after': {'nodes': 123, 'termination_reason': stop}}}}
             for i, (value, stop) in enumerate([(False, None), (False, None), (True, 'plan_completed')], 1)]
    compact_observations(steps)
    assert len(steps) == 3 and all(s['action'] == 'Pick[obj_0]' for s in steps)
    assert 'observations' not in steps[1]
    assert steps[0]['observations']['Observed evidence']['states']['obj_0']['clean'] is False
    assert steps[2]['observations']['Observed evidence']['states']['obj_0']['clean'] is True
    assert steps[2]['observations']['Termination'] == {'after': 'plan_completed'}
    assert 'nodes' not in json.dumps(steps)


def test_new_unverified_skip_is_not_exempt(source):
    root, _, _ = source
    path = root / 'dataset/traces/0/trace-test.html'
    path.write_text(path.read_text().replace('class="step"', 'class="step skipped" data-status="skipped" data-react-skip-policy="robot_sg_before_action"'))
    trace, _ = read_trace(root)
    assert all(step.get('counts_as_error') is not False for step in trace['steps'])


def test_historical_html_keeps_recorded_skip_classification(source):
    root, _, _ = source
    path = root / 'dataset/traces/0/trace-test.html'
    path.write_text(path.read_text().replace('class="step"', 'class="step skipped" data-status="skipped"'))
    trace, _ = read_trace(root)
    assert all(step['status'] == 'skipped' and step['counts_as_error'] is False for step in trace['steps'])
