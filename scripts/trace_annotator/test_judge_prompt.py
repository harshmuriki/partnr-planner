import copy
import json

from judge_evidence import packet_for
from judge_prompt import prompt_view
from judge_contract import validate
from test_run_judge import source, with_incident, incident


def test_payload_preserves_attempts_feedback_and_citable_evidence(source):
    packet = packet_for(source[2]); before = copy.deepcopy(packet)
    view = prompt_view(packet)
    assert packet == before
    assert len(view['attempts']) == len(packet['attempts'])
    for a, b in zip(view['attempts'], packet['attempts']):
        assert a == {k: v for k, v in b.items() if k != 'boundary_source'}
    assert validate(with_incident(packet, incident()), view)
    assert 'does not exist' in json.dumps(view)
    assert all('source' not in e and 'locator' not in e for e in view['events'])
    assert all('high_level_actions' not in e['data'].get('planner_observed', {})
               for e in view['events'] if 'attempt_ids' in e['data'])


def test_scenario_drops_generation_instructions_but_keeps_initial_state(source):
    p = packet_for(source[2])
    p['reference_truth_not_robot_knowledge']['spec.md'] = '''# T1-ACC-BASE
## Initial world state
- box_0: on table_0
## Initial robot memory
- box_0: on table_1
## Spawn / planner notes
- SECRET GENERATION INSTRUCTIONS
'''
    view = prompt_view(p)
    assert view['initial_state']['declared_world_state'] == ['box_0: on table_0']
    assert view['scenario']['initial_robot_memory'] == ['box_0: on table_1']
    assert 'SECRET GENERATION' not in json.dumps(view)
    assert 'sha256' not in json.dumps(view)


def test_score_unknowns_stay_unknown_and_checkpoint_separate(source):
    p = packet_for(source[2]); p['simulator']['outcome'] = 'unavailable'
    p['simulator']['criteria'][0]['status'] = 'unknown'
    p['checkpoint_evaluation'] = {'score_basis': 'last_saved_state', 'task_percent_complete': 1}
    view = prompt_view(p)
    assert view['simulator']['criteria'][0]['result'] == 'unknown'
    assert view['simulator']['outcome'] == 'unavailable'
    assert view['checkpoint_evaluation'] == p['checkpoint_evaluation']


def test_initial_observed_state_keeps_original_citation_and_later_changes(source):
    p = packet_for(source[2])
    for index, table in [(1, 'table_0'), (3, 'table_1')]:
        p['events'][index]['data']['evaluator_only_state'] = {'reference_world_graph': [
            {'name': 'box_0', 'properties': {'type': 'object'}, 'relations': {table: 'on'}}]}
    view = prompt_view(p)
    assert view['initial_state']['observed_state_event_id'] == p['events'][1]['id']
    assert view['initial_state']['observed_entities']['box_0']['relations'] == {'table_0': 'on'}
    update = next(e for e in view['events'] if e['id'] == p['events'][3]['id'])
    assert update['data']['evaluator_state_changes']['entities_changed']['box_0']['relations'] == {'table_1': 'on'}


def _duplicate_packet():
    from judge_prompt import duplicate_supplemental_ids
    observed = {'high_level_actions': {'0': ['Pick', 'box_0', None]},
                'responses': {'0': ''}, 'thought': {'0': 'Pick the box.'}}
    finished = copy.deepcopy(observed); finished['responses'] = {'0': 'Successful execution!'}
    execution = [{'id': 'e' + str(i), 'kind': 'execution', 'data': {'planner_observed': copy.deepcopy(data)}}
                 for i, data in enumerate([observed, finished, observed, finished])]
    extra = [{'id': 'p' + str(i), 'kind': 'planner_context', 'source': 'planner.json',
              'data': dict(copy.deepcopy(data), instruction='task', is_done={'0': False})}
             for i, data in enumerate([observed, finished, observed, finished])]
    return {'instruction': 'task', 'events': execution + extra, 'attempts': [],
            'simulator': {}, 'checkpoint_evaluation': None}


def test_duplicate_supplemental_complete_sequence_without_collapsing_repeated_execution():
    from judge_prompt import duplicate_supplemental_ids
    packet = _duplicate_packet(); before = copy.deepcopy(packet)
    assert duplicate_supplemental_ids(packet) == {'p0', 'p1', 'p2', 'p3'}
    assert packet == before
    packet['attempts'] = [{'evidence_ids': ['p0', 'e0']}]
    packet['simulator']['source_event_id'] = 'p3'
    assert duplicate_supplemental_ids(packet) == {'p1', 'p2'}


def test_duplicate_supplemental_unique_knowledge_stop_or_unmatched_order_keeps_sequence():
    from judge_prompt import duplicate_supplemental_ids
    for field, value in [('curr_graph', {'0': 'new object'}), ('is_done', {'0': True}),
                         ('new_planning_decision', 'search elsewhere')]:
        packet = _duplicate_packet(); packet['events'][-1]['data'][field] = value
        assert duplicate_supplemental_ids(packet) == set()
    packet = _duplicate_packet()
    packet['events'][-1]['data']['responses'] = {'0': 'Different failure'}
    assert duplicate_supplemental_ids(packet) == set()
    packet = _duplicate_packet(); packet['events'][0], packet['events'][1] = packet['events'][1], packet['events'][0]
    assert duplicate_supplemental_ids(packet) == set()


def test_duplicate_supplemental_done_requires_explicit_terminal_execution():
    from judge_prompt import duplicate_supplemental_ids
    packet = _duplicate_packet()
    packet['events'][3]['data']['planner_observed']['high_level_actions'] = {'0': ['Done', None, None]}
    packet['events'][-1]['data']['high_level_actions'] = {'0': ['Done', None, None]}
    packet['events'][-1]['data']['is_done'] = {'0': True}
    assert duplicate_supplemental_ids(packet) == set()
    packet['events'][3]['data']['event'] = 'terminal'
    assert duplicate_supplemental_ids(packet) == {'p0', 'p1', 'p2', 'p3'}


def test_archived_identity_maps_keep_actual_scene_info_keys(source):
    packet = packet_for(source[2])
    packet['reference_truth_not_robot_knowledge']['scene_info.json'] = {
        'receptacle_to_handle': {'table_16': 'table-handle'},
        'spec_furniture_aliases': {'center_table': 'table_16'},
        'all_furniture': ['irrelevant catalog'], 'objects': ['irrelevant catalog']}
    view = prompt_view(packet)
    assert view['identity_mapping'] == {
        'receptacle_to_handle': {'table_16': 'table-handle'},
        'spec_furniture_aliases': {'center_table': 'table_16'}}
    assert list(view) == ['view_version', 'instruction', 'scenario', 'initial_state',
                          'identity_mapping', 'simulator', 'checkpoint_evaluation',
                          'attempts', 'events', 'coverage']
