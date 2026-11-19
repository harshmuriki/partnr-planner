import copy
import json

from judge_examples import simple_examples


def example():
    return {'name': 'example.json', 'source_run_ids': ['private/path'], 'input': {
        'instruction': 'Put apple_1 in basket_0.',
        'simulator': {'outcome': 'partial', 'criteria': [
            {'id': 'p0', 'status': 'unknown', 'proposition': {'function_name': 'is_inside'}}],
            'ambiguous_failure_groups': [{'id': 'g0', 'criterion_ids': ['p0'], 'unmet_count': 1}]},
        'reference_truth_not_robot_knowledge': {'spec.md': '## Initial robot memory\n- apple_1: counter_0\n- apple_0: known absent\n\n## Success criteria\n- is_inside(apple_1, basket_0)\n\n## Spawn / planner notes\nsecret-generation-note\n'},
        'attempts': [{'id': 'a0', 'action': 'Navigate', 'arguments': 'counter_0',
                      'responses': ['Successful execution!'], 'evidence_ids': ['e1']}],
        'events': [{'id': 'e1', 'data': {'json_ref': 'j1'}},
                   {'id': 'unused', 'data': {'response': 'irrelevant'}}],
        'json_blocks': {'j1': {'event': 'vlm_predicate_subgoals', 'response': 'on(bottle_0, counter_0)',
                               'wall_time': 123, 'api_response': 'secret-api-envelope'}},
        'text_blocks': {}, 'provenance': {'sha256': 'private-hash'}},
        'expected_output': {'failures': [{'id': 'f0', 'behavior_id': 'omission', 'label': 'no_action_vlm',
            'criterion_ids': [], 'unmet_group_ids': ['g0'], 'evidence_ids': ['e1'],
            'explanation': 'Fruit goal omitted.', 'confidence_raw': .9}], 'inefficiencies': [],
            'label_assessments': [{'label': 'placement', 'status': 'insufficient_evidence'}],
            'important_notes': [], 'possible_failure_notes': []}}


def test_compact_example_preserves_grounding_without_audit_noise():
    source = example(); original = copy.deepcopy(source)
    result = simple_examples([source])[0]
    assert source == original
    assert result['simulator']['criteria'][0]['status'] == 'unknown'
    assert result['simulator']['ambiguous_failure_groups'][0]['id'] == 'g0'
    assert result['expected_incidents']['failures'][0]['unmet_group_ids'] == ['g0']
    assert result['attempts'][0]['responses'] == ['Successful execution!']
    assert result['evidence'] == [{'id': 'e1', 'event': 'vlm_predicate_subgoals',
                                   'response': 'on(bottle_0, counter_0)'}]
    assert 'apple_0: known absent' in result['declared_scenario']['initial_robot_memory']
    text = json.dumps(result)
    for unwanted in ['private/path', 'secret-generation-note', 'private-hash', 'secret-api-envelope', 'label_assessments']:
        assert unwanted not in text


def test_shared_failed_criterion_and_multiple_attempts_are_not_collapsed():
    source = example()
    source['input']['attempts'] *= 2
    other = copy.deepcopy(source['expected_output']['failures'][0]); other['id'] = 'f1'
    other['label'] = 'insufficient_planning'
    source['expected_output']['failures'].append(other)
    result = simple_examples([source])[0]
    assert len(result['attempts']) == 2
    assert len(result['expected_incidents']['failures']) == 2
    assert all(i['unmet_group_ids'] == ['g0'] for i in result['expected_incidents']['failures'])


def test_prior_pick_citation_and_note_evidence_retained():
    source = example()
    source['expected_output']['failures'][0]['prior_failed_pick_evidence_ids'] = ['pick']
    source['expected_output']['important_notes'] = [{'text': 'Counterevidence', 'evidence_ids': ['note']}]
    source['input']['events'] += [{'id': 'pick', 'data': {'event': 'pick', 'response': 'Not found'}},
                                 {'id': 'note', 'data': {'event': 'memory_snapshot', 'known_objects': ['apple_1']}}]
    result = simple_examples([source])[0]
    assert {e['id'] for e in result['evidence']} == {'e1', 'pick', 'note'}


def test_runtime_aliases_and_robot_knowledge_keep_distinct_provenance():
    source = example()
    source['expected_output']['failures'][0]['object_name'] = 'apple_1'
    source['input']['events'] = [{'id': 'e1', 'data': {
        'event': 'boundary',
        'planner_observed': {'curr_graph': {'0': 'Objects:\napple_1: counter_66\nremote_8: sofa_0'},
                             'responses': {'0': 'Not within reach'}},
        'evaluator_only_state': {'reference_world_graph': [
            {'name': 'apple_1', 'sim_handle': 'Apple_4_:0000',
             'properties': {'translation': [1, 2, 3]}}]}}}]
    result = simple_examples([source])[0]
    assert result['handle_aliases'] == {'Apple_4_:0000': 'apple_1'}
    observed = result['evidence'][0]['planner_observed']
    assert observed['robot_knowledge_excerpt'] == {'0': 'apple_1: counter_66'}
    assert observed['responses'] == {'0': 'Not within reach'}
    assert 'translation' not in json.dumps(result)
