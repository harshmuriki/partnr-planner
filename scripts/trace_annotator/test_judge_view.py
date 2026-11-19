"""The view must remove noise without erasing attempts, recoveries or score provenance."""
import copy
import json

import pytest

from judge_evidence import checkpoint_delta, packet_for, share_json_blocks
from judge import share_text_blocks
from judge_view import expand_blocks, model_view, parse_native_record, prompt_world_description
from test_run_judge import source, with_incident, incident
from judge_contract import validate


def node(name, parent, x=0):
    return {'name': name, 'sim_handle': name + '_handle',
            'properties': {'type': 'object', 'translation': [x, 0, 0], 'states': {'is_clean': True}},
            'relations': {parent: 'on'}}


def state(graph, met=False):
    return {'reference_world_graph': graph, 'task_percent_complete': int(met),
            'auto_eval_proposition_tracker': {'last_state': [{'is_satisfied': met}],
                                             'proposition_satisfied_at': [3 if met else -1]}}


def e(eid, data, kind='execution', order=1):
    return {'id': eid, 'kind': kind, 'order': order, 'source': 'source.jsonl', 'locator': eid, 'data': data}


def test_nested_json_and_text_blocks_decode_before_reading_graphs():
    text = 'Furniture:\n' + ('living_room_1: table_0, couch_0, cabinet_0, floor_living_room_1, chair_0\n' * 12)
    original = [e('x', {'curr_graph': {'0': text}}), e('y', {'curr_graph': {'0': text}})]
    packed, texts = share_text_blocks(original)
    packed, blocks = share_json_blocks(packed)
    assert blocks and texts
    assert expand_blocks({'json_blocks': blocks, 'text_blocks': texts}, packed) == original


def test_checkpoints_merge_by_time_retain_relation_changes_and_final_source(source):
    p = packet_for(source[2]);p['attempts'] = []
    first = state([node('box_0', 'wrong_table')])
    moved_coordinates = state([node('box_0', 'wrong_table', 20)])
    corrected = state([node('box_0', 'right_table', 20)], True)
    sets, removes = checkpoint_delta(first, moved_coordinates)
    # Raw packet order has execution before checkpoints, even though it happened later.
    p['events'] = [e('exec', {'wall_time': 2, 'evaluator_only_state': first}),
                   e('c0', {'wall_time': 1, 'evaluator_only_state': first}, 'evaluator_checkpoint'),
                   e('c1', {'wall_time': 3, 'evaluator_only_state': {'checkpoint_delta': {'base_event_id': 'c0', 'set': sets, 'remove': removes}}}, 'evaluator_checkpoint'),
                   e('c2', {'wall_time': 4, 'evaluator_only_state': corrected}, 'evaluator_checkpoint'),
                   e('c3', {'wall_time': 5, 'evaluator_only_state': corrected}, 'evaluator_checkpoint')]
    p['checkpoint_evaluation'] = {'source_event_id': 'c3', 'eligible_for_failure_labels': False}
    before = copy.deepcopy(p)
    view = model_view(p)
    ids = [x['id'] for x in view['events']]
    assert ids == ['c0', 'exec', 'c2', 'c3']
    assert 'translation' not in json.dumps(view['events'])
    correction = next(x for x in view['events'] if x['id'] == 'c2')['data']['evaluator_state_changes']
    assert correction['entities_changed']['box_0']['relations'] == {'right_table': 'on'}
    assert view['simulator'] == p['simulator'] and view['checkpoint_evaluation'] == p['checkpoint_evaluation']
    assert p == before  # The archived audit packet is untouched.


def test_all_attempts_feedback_and_stale_removal_evidence_survive(source):
    p = packet_for(source[2]);v = model_view(p)
    assert v['attempts'] == p['attempts']
    assert len(v['attempts']) == 2
    assert 'does not exist' in json.dumps(v)
    assert validate(with_incident(p, incident()), v)
    for a in v['attempts']:
        assert set(a['evidence_ids']) <= {e['id'] for e in v['events']}


def test_robot_knowledge_changes_are_not_replaced_by_reference_truth(source):
    p = packet_for(source[2])
    p['events'][1]['data']['planner_observed']['curr_graph'] = {'0': 'Objects:\nbox_0: table_0'}
    p['events'][2]['data']['planner_observed']['curr_graph'] = {'0': 'Objects:\n'}
    v = model_view(p)
    observations = [e['data'].get('planner_observed', {}).get('robot_knowledge') for e in v['events']]
    assert {'0': 'Objects:\nbox_0: table_0'} in observations
    assert {'0': 'Objects:\n'} in observations


def test_prompt_extracts_knowledge_and_drops_tutorial_but_keeps_trace(source):
    p = packet_for(source[2])
    prompt = 'You are a planner.\nInitial agent location:\nagent_0: bedroom_1\nFurniture:\nObjects:\nbox_0: table_0\nPossible Actions:\nPick tutorial'
    p['events'] += [e('prompt', prompt, 'saved_prompt'), e('trace', 'Objects: box_0: wrong_table', 'trace')]
    v = model_view(p);text = json.dumps(v)
    assert 'You are a planner' not in text and 'Pick tutorial' not in text
    assert 'agent_0: bedroom_1' in text and 'box_0: wrong_table' in text
    assert prompt_world_description('Unknown prompt structure') is None


def test_native_plans_parse_numpy_scalars_without_executing_code():
    assert parse_native_record("{'plan': [('navigate', np.float64(1.5))]}") == {'plan': [('navigate', 1.5)]}
    assert parse_native_record("{'plan': [Action(name='pick', args=('agent_0', 'box_0'))]}") == {
        'plan': [{'name': 'pick', 'args': ('agent_0', 'box_0')}]}
    evil = "{'plan': __import__('os').system('false')}"
    assert parse_native_record(evil) == {'unparsed_record': evil}


def test_duplicate_legacy_source_removed_without_dropping_cited_example(source):
    p = packet_for(source[2]);duplicate = copy.deepcopy(p['events'][1]);duplicate.update(id='duplicate', kind='planner_context')
    p['events'].append(duplicate)
    assert 'duplicate' not in {e['id'] for e in model_view(p)['events']}
    required_view = model_view(p, required_ids=['duplicate'])
    kept = next(e for e in required_view['events'] if e['id'] == 'duplicate')
    assert kept['data']['planner_observed']['high_level_actions'] == duplicate['data']['planner_observed']['high_level_actions']
    v = model_view(p)
    answer = with_incident(p, incident())
    answer['inefficiencies'][0]['evidence_ids'].append('duplicate')
    with pytest.raises(ValueError, match='unknown'):
        validate(answer, v)


def test_diagnostics_not_attempts_and_score_state_preserved(source):
    p = packet_for(source[2])
    p['events'].append(e('console', [{'text': 'GPU initialization complete', 'first_line': 1},
                                   {'text': 'Could not find object box_0', 'first_line': 2}], 'diagnostics_not_attempts'))
    v = model_view(p)
    assert 'GPU initialization' not in json.dumps(v)
    assert 'Could not find object box_0' in json.dumps(v)
    assert v['attempts'] == p['attempts'] and v['simulator'] == p['simulator']


def test_occupied_or_targeted_receptacles_keep_identity_even_after_emptying(source):
    p=packet_for(source[2]);p['attempts']=[]
    table={'name':'table_0','properties':{},'relations':{'rec_used':'joint','rec_empty':'joint','rec_target':'joint'}}
    used={'name':'rec_used','sim_handle':'used_handle','properties':{'type':'on'},'relations':{'table_0':'joint','box_0':'under'}}
    empty={'name':'rec_empty','sim_handle':'empty_handle','properties':{'type':'on'},'relations':{'table_0':'joint'}}
    target=dict(empty,name='rec_target',sim_handle='target_handle')
    p['simulator']['criteria'][0]['proposition']['args']['receptacle_handles']=['target_handle']
    initial=state([table,used,empty,target,node('box_0','rec_used')]);later=copy.deepcopy(initial)
    later['reference_world_graph'][1]['relations'].pop('box_0')
    later['reference_world_graph'][-1]['relations']={'table_0':'on'}
    p['events']=[e('first',{'evaluator_only_state':initial}),e('later',{'evaluator_only_state':later})]
    view=model_view(p)
    first=view['events'][0]['data']['evaluator_state_changes']['entities_changed']
    assert 'rec_empty' not in first and 'rec_empty' not in first['table_0']['relations']
    assert {'rec_used','rec_target'} <= set(first)
    change=view['events'][1]['data']['evaluator_state_changes']
    assert change['entities_changed']['rec_used']['relations']=={'table_0':'joint'}
    assert 'entities_removed' not in change
