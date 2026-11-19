"""Deterministic counts of validated annotations, separate from simulator scores.

Zero incidents means no incident was annotated, not proof a label was absent.
Consult label_assessment_status before aggregating coverage across runs. This
module deliberately does not infer missing scores from task_percent_complete.
"""

from judge_contract import FAILURES, INEFFICIENCIES


def annotation_counts(analysis, simulator):
    """Summarize an already validated judge output and its final simulator data.

    Multiple labels may explain one criterion or one inefficient attempt. Those
    count as multiple incidents but only one distinct criterion/attempt. Ambiguous
    groups retain their own unmet_count; overlapping groups are never summed.
    Checkpoint scoring must not be passed as final simulator data.
    """
    if simulator.get('score_basis') == 'last_saved_state':
        raise ValueError('Pass final simulator data, not checkpoint scoring')
    failures = analysis.get('failures', [])
    inefficiencies = analysis.get('inefficiencies', [])
    outcome = simulator.get('outcome', 'unavailable')
    available = outcome != 'unavailable'
    criteria = simulator.get('criteria', [])
    statuses = {c['id']: c['status'] for c in criteria}
    groups = {g['id']: g for g in simulator.get('ambiguous_failure_groups', [])}
    criterion_ids = sorted({cid for item in failures for cid in item['criterion_ids']})
    group_ids = sorted({gid for item in failures for gid in item['unmet_group_ids']})
    attempt_ids = sorted({aid for item in inefficiencies for aid in item['attempt_ids']})

    if failures and outcome not in ('partial', 'no_progress'):
        raise ValueError('Failure counts require scored unsuccessful subtasks')
    if any(statuses.get(cid) != 'unmet' for cid in criterion_ids):
        raise ValueError('Failure criterion counts require definitely unmet criteria')
    if any(gid not in groups for gid in group_ids):
        raise ValueError('Unknown ambiguous failure group')

    unknown = sum(c['status'] == 'unknown' for c in criteria) if available else len(criteria)
    met = sum(c['status'] == 'met' for c in criteria) if available else None
    unmet = sum(c['status'] == 'unmet' for c in criteria) if available else None
    exhaustive = available and unknown == 0

    def counts(items, labels):
        if any(item['label'] not in labels for item in items):
            raise ValueError('Unexpected label family')
        return {label: sum(item['label'] == label for item in items) for label in labels}

    return {
        'schema_version': 1,
        'simulator': {
            'outcome': outcome,
            'scoring_available': available,
            'criterion_count': len(criteria),
            'known_met_count': met,
            'known_unmet_count': unmet,
            'unknown_count': unknown,
            'success_count': met if exhaustive else None,
            'failed_count': unmet if exhaustive else None,
        },
        'failures': {
            'incident_count': len(failures),
            'criterion_ids': criterion_ids,
            'distinct_criterion_count': len(criterion_ids),
            'ambiguous_group_ids': group_ids,
            'distinct_ambiguous_group_count': len(group_ids),
            'ambiguous_groups': [
                {'id': gid, 'criterion_ids': sorted(set(groups[gid]['criterion_ids'])),
                 'unmet_count': groups[gid]['unmet_count']}
                for gid in group_ids
            ],
            'counts_by_label': counts(failures, FAILURES),
        },
        'inefficiencies': {
            'incident_count': len(inefficiencies),
            'attempt_ids': attempt_ids,
            'distinct_attempt_count': len(attempt_ids),
            'counts_by_label': counts(inefficiencies, INEFFICIENCIES),
        },
        'label_assessment_status': {
            item['label']: item['status'] for item in analysis.get('label_assessments', [])
        },
    }
