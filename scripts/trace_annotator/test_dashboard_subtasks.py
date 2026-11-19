from dashboard_subtasks import chart_data, criterion_display, subtask_record
from judge_evidence import simulator


def prop(handle='a', number=1):
    return {'function_name': 'is_clean', 'args': {'object_handles': [handle], 'number': number}}


def test_missing_final_is_unknown_not_failed():
    ep = {'evaluation_propositions': [prop()]}
    record = subtask_record(ep, {}, simulator(ep, [], {}), 'summary.json')
    assert [c['status'] for c in record['criteria']] == ['unknown']
    assert record['method'] == 'unavailable'


def test_structured_constraints_take_precedence_over_aggregate():
    ep = {'evaluation_propositions': [prop('a'), prop('b')]}
    sim = simulator(ep, [{'_id': 'terminal:1', 'event': 'terminal', 'evaluator_only_state': {
        'auto_eval_proposition_tracker': {'proposition_satisfied_at': [3, 7]},
        'task_constraint_validation': [[True, False]]}}], {'task_percent_complete': 1})
    record = subtask_record(ep, {}, sim, 'events.jsonl')
    assert [c['status'] for c in record['criteria']] == ['met', 'unmet']
    assert record['method'] == 'terminal_simulator_snapshot'


def test_ambiguous_same_class_explanation_not_arbitrarily_assigned(monkeypatch):
    monkeypatch.setattr('dashboard_subtasks.categories', lambda: {'a': 'book', 'b': 'book', 'c': 'plate'})
    ep = {'evaluation_propositions': [prop('a'), prop('b'), prop('c')]}
    sim = simulator(ep, [], {'task_percent_complete': 2/3,
        'task_explanation': 'Episode failed. Missing steps:\n - Object "book" should have been clean'})
    record = subtask_record(ep, {}, sim, 'summary.json')
    assert [c['status'] for c in record['criteria']] == ['unknown', 'unknown', 'met']
    assert record['ambiguous_groups'] == [{'criterion_ids': ['p0', 'p1'], 'unmet_count': 1}]


def test_quantity_thresholds_are_separate_and_all_success_is_met():
    p = {'function_name': 'is_inside', 'args': {
        'object_handles': ['a', 'b'], 'receptacle_handles': ['wardrobe'], 'number': 2}}
    ep = {'evaluation_propositions': [p]}
    record = subtask_record(ep, {}, simulator(ep, [], {'task_percent_complete': 1}), 'summary.json')
    assert len(record['criteria']) == 2
    assert len({c['key'] for c in record['criteria']}) == 2
    assert all(c['status'] == 'met' for c in record['criteria'])
    assert 'at least 1 of [a / b]' in record['criteria'][1]['label']


def test_group_by_predicate_not_shifted_index_or_changed_target():
    a = {'function_name': 'is_on_top', 'args': {'object_handles': ['a'], 'receptacle_handles': ['table1']}}
    b = {'function_name': 'is_on_top', 'args': {'object_handles': ['a'], 'receptacle_handles': ['table2']}}
    assert criterion_display(a, {})[0] != criterion_display(b, {})[0]
    def row(criteria, task=1):
        return {'task': task, 'subtask_results': {'criteria': criteria, 'method': 'test',
            'source': 'test', 'source_event_id': None, 'ambiguous_groups': [], 'metrics': {}}}
    ep = {'evaluation_propositions': [a,b]}
    rec = subtask_record(ep, {}, simulator(ep, [], {}), 'test')
    shifted = dict(rec['criteria'][1], id='p0')
    data = chart_data([row(rec['criteria']), row([shifted]), row([shifted], task=2)])
    assert len(data['definitions']) == 3
    assert data['runs'][0]['criteria'][1]['criterion'] == data['runs'][1]['criteria'][0]['criterion']
    assert data['runs'][1]['criteria'][0]['criterion'] != data['runs'][2]['criteria'][0]['criterion']


def test_base_order_survives_absent_variant_reindexing():
    def make_row(task_id, props):
        ep = {'evaluation_propositions': props}
        return {'task': 1, 'task_id': task_id,
                'subtask_results': subtask_record(ep, {}, simulator(ep, [], {}), 'test')}
    data = chart_data([make_row('T1-ACC-ABS', [prop('lamp')]),
                       make_row('T1-ACC-BASE', [prop('jug'), prop('lamp')]),
                       make_row('T1-ACC-SUB', [prop('pitcher'), prop('lamp')])])
    definitions = {d['label']: d for d in data['definitions']}
    assert definitions['jug: clean']['base_order'] == 0
    assert definitions['lamp: clean']['base_order'] == 1
    assert definitions['lamp: clean']['variant_order'] == 0
    assert definitions['pitcher: clean']['base_order'] is None
