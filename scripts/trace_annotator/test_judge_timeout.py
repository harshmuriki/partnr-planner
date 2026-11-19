import pytest

from judge_contract import FAILURES, LABELS, validate


def packet(status='timeout', outcome='partial', event=None):
    return {'simulator': {'outcome': outcome,
                         'criteria': [{'id': 'p0', 'status': 'unmet'}],
                         'ambiguous_failure_groups': []},
            'events': [{'id': 'e0', 'kind': 'execution', 'order': 0,
                        'data': event or {'event': 'boundary', 'action': 'Navigate'}}],
            'attempts': [], 'json_blocks': {},
            'coverage': {'execution_status': status, 'history_complete': False}}


def annotation(p):
    return {'summary': 'The recorded run timeout cut off the required placement.',
            'failures': [{'id': 'f0', 'behavior_id': 'b0', 'label': 'timeout',
                          'criterion_ids': ['p0'], 'unmet_group_ids': [],
                          'attempt_ids': [], 'evidence_ids': ['e0'],
                          'explanation': 'Execution was cut off while required work remained.',
                          'alternative_explanation': 'No unproductive loop is established.',
                          'confidence_raw': 0.9, 'object_name': None,
                          'outdated_memory_object': False,
                          'prior_failed_pick_evidence_ids': [], 'unmet_precondition': None}],
            'inefficiencies': [], 'unclassified_failed_criterion_ids': [],
            'important_notes': [], 'possible_failure_notes': [],
            'label_assessments': [
                {'label': label, 'status': 'present' if label == 'timeout' else
                 'not_applicable' if label in FAILURES and p['simulator']['outcome'] in
                 ('complete', 'unavailable') else 'insufficient_evidence',
                 'reason': 'Based on saved evidence'} for label in LABELS]}


def test_recorded_manifest_timeout_is_accepted():
    p = packet()
    assert validate(annotation(p), p)['failures'][0]['label'] == 'timeout'


@pytest.mark.parametrize('event', [
    {'event': 'terminal', 'termination_reason': 'timeout'},
    {'event': 'run_end', 'timed_out': True},
    {'event': 'run_timeout'},
])
def test_cited_structured_execution_termination_is_accepted(event):
    p = packet(status='failed', event=event)
    validate(annotation(p), p)


def test_termination_in_shared_json_block_is_supported():
    p = packet(status='failed', event={'json_ref': 'j0'})
    p['json_blocks']['j0'] = {'event': 'terminal', 'reason': 'time_limit_reached'}
    validate(annotation(p), p)


@pytest.mark.parametrize('status', ['failed', 'cancelled', 'preempted', 'completed'])
def test_other_run_statuses_are_not_inferred_as_timeouts(status):
    p = packet(status=status)
    with pytest.raises(ValueError, match='recorded run time-limit'):
        validate(annotation(p), p)


@pytest.mark.parametrize('event', [
    {'event': 'boundary', 'response': 'timeout'},
    {'event': 'tool_error', 'status': 'timeout'},
    {'event': 'terminal', 'reason': 'The planner thought this could timeout'},
    {'event': 'terminal', 'time_limit_seconds': 600, 'elapsed_seconds': 601},
])
def test_log_prose_tool_timeout_and_elapsed_time_are_not_execution_cutoffs(event):
    p = packet(status='failed', event=event)
    with pytest.raises(ValueError, match='recorded run time-limit'):
        validate(annotation(p), p)


def test_planner_context_cannot_assert_execution_timeout():
    p = packet(status='failed', event={'event': 'terminal', 'status': 'timeout'})
    p['events'][0]['kind'] = 'planner_context'
    with pytest.raises(ValueError, match='recorded run time-limit'):
        validate(annotation(p), p)


@pytest.mark.parametrize('outcome', ['complete', 'unavailable'])
def test_timeout_does_not_bypass_final_score_eligibility(outcome):
    p = packet(outcome=outcome)
    with pytest.raises(ValueError, match='scored unsuccessful'):
        validate(annotation(p), p)


def test_timeout_can_link_an_ambiguous_failed_group():
    p = packet()
    p['simulator']['criteria'][0]['status'] = 'unknown'
    p['simulator']['ambiguous_failure_groups'] = [
        {'id': 'g0', 'criterion_ids': ['p0'], 'unmet_count': 1}]
    a = annotation(p)
    a['failures'][0]['criterion_ids'] = []
    a['failures'][0]['unmet_group_ids'] = ['g0']
    validate(a, p)


def test_timeout_event_must_be_cited_when_manifest_has_no_timeout():
    p = packet(status='failed')
    p['events'].append({'id': 'e1', 'kind': 'execution', 'order': 1,
                        'data': {'event': 'terminal', 'reason': 'timeout'}})
    with pytest.raises(ValueError, match='recorded run time-limit'):
        validate(annotation(p), p)
