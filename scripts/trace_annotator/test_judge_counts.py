import copy

import pytest

from judge_counts import annotation_counts


def sim(outcome='partial', statuses=('unmet', 'met')):
    return {'outcome': outcome,
            'criteria': [{'id': f'p{i}', 'status': status} for i, status in enumerate(statuses)],
            'ambiguous_failure_groups': []}


def incident(label, criteria=(), groups=(), attempts=()):
    return {'label': label, 'criterion_ids': list(criteria),
            'unmet_group_ids': list(groups), 'attempt_ids': list(attempts)}


def test_multiple_labels_do_not_multiply_failed_subtasks():
    analysis = {'failures': [incident('insufficient_planning', ['p0']),
                             incident('no_action_vlm', ['p0'])]}
    counts = annotation_counts(analysis, sim())
    assert counts['failures']['incident_count'] == 2
    assert counts['failures']['distinct_criterion_count'] == 1
    assert counts['failures']['criterion_ids'] == ['p0']
    assert counts['failures']['counts_by_label']['no_action_vlm'] == 1
    assert counts['simulator']['failed_count'] == 1
    assert counts['simulator']['success_count'] == 1


def test_same_label_two_objects_counts_two_incidents():
    counts = annotation_counts({'failures': [incident('placement', ['p0']),
                                              incident('placement', ['p1'])]},
                               sim('no_progress', ('unmet', 'unmet')))
    assert counts['failures']['counts_by_label']['placement'] == 2
    assert counts['failures']['distinct_criterion_count'] == 2


def test_ambiguous_groups_count_once_without_asserting_specific_failures():
    simulator = sim('partial', ('unknown', 'unknown', 'unknown'))
    simulator['ambiguous_failure_groups'] = [
        {'id': 'g0', 'criterion_ids': ['p0', 'p1', 'p2'], 'unmet_count': 2},
        {'id': 'g1', 'criterion_ids': ['p0', 'p1'], 'unmet_count': 1}]
    counts = annotation_counts({'failures': [incident('no_action_vlm', groups=['g0']),
                                             incident('placement', groups=['g0', 'g1'])]}, simulator)
    assert counts['failures']['distinct_criterion_count'] == 0
    assert counts['failures']['distinct_ambiguous_group_count'] == 2
    assert [g['unmet_count'] for g in counts['failures']['ambiguous_groups']] == [2, 1]
    assert counts['simulator']['failed_count'] is None
    assert counts['simulator']['success_count'] is None
    assert counts['simulator']['unknown_count'] == 3


def test_inefficiency_incidents_and_distinct_attempts_are_separate():
    counts = annotation_counts({'inefficiencies': [
        incident('repeated_actions', attempts=['a1']),
        incident('precondition_failures', attempts=['a1']),
        incident('repeated_actions', attempts=['a2'])]}, sim('complete', ('met', 'met')))
    assert counts['inefficiencies']['incident_count'] == 3
    assert counts['inefficiencies']['distinct_attempt_count'] == 2
    assert counts['inefficiencies']['counts_by_label']['repeated_actions'] == 2
    assert counts['failures']['incident_count'] == 0
    assert counts['simulator']['success_count'] == 2


def test_unavailable_final_results_do_not_turn_into_zero_scores():
    counts = annotation_counts({'inefficiencies': [incident('repeated_actions', attempts=['a1'])]},
                               sim('unavailable', ('met', 'unmet')))
    assert counts['simulator'] == {
        'outcome': 'unavailable', 'scoring_available': False, 'criterion_count': 2,
        'known_met_count': None, 'known_unmet_count': None, 'unknown_count': 2,
        'success_count': None, 'failed_count': None}
    assert counts['inefficiencies']['incident_count'] == 1


def test_partial_coverage_keeps_known_counts_without_inventing_totals():
    counts = annotation_counts({}, sim('partial', ('met', 'unmet', 'unknown')))
    assert counts['simulator']['known_met_count'] == 1
    assert counts['simulator']['known_unmet_count'] == 1
    assert counts['simulator']['success_count'] is None
    assert counts['simulator']['failed_count'] is None


def test_counts_preserve_assessment_status_and_do_not_mutate_inputs():
    analysis = {'label_assessments': [{'label': 'placement', 'status': 'insufficient_evidence'}]}
    simulator = sim()
    before = copy.deepcopy((analysis, simulator))
    counts = annotation_counts(analysis, simulator)
    assert counts['failures']['counts_by_label']['placement'] == 0
    assert counts['label_assessment_status']['placement'] == 'insufficient_evidence'
    assert (analysis, simulator) == before


@pytest.mark.parametrize('outcome', ['complete', 'unavailable'])
def test_reject_unscored_or_successful_failure_labels(outcome):
    with pytest.raises(ValueError, match='scored unsuccessful'):
        annotation_counts({'failures': [incident('placement', ['p0'])]}, sim(outcome))


def test_reject_unknown_criterion_and_group_links():
    with pytest.raises(ValueError, match='definitely unmet'):
        annotation_counts({'failures': [incident('placement', ['p1'])]}, sim())
    with pytest.raises(ValueError, match='Unknown ambiguous'):
        annotation_counts({'failures': [incident('placement', groups=['g0'])]}, sim())


def test_reject_checkpoint_as_final_score():
    with pytest.raises(ValueError, match='checkpoint'):
        annotation_counts({}, {**sim(), 'score_basis': 'last_saved_state'})
