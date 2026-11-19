"""Per-criterion dashboard data from saved simulator results, never judge labels."""
from __future__ import annotations

from functools import lru_cache
import json

from sim_results import metadata_categories, recover_from_metrics


@lru_cache(maxsize=1)
def categories():
    return metadata_categories()[0]


def criterion_display(prop, aliases):
    """Group by the full normalized predicate, not its position in a variant.

    Absent-object variants can remove/reorder propositions. Quantity thresholds
    and approved alternative targets must remain distinct.
    """
    args = prop.get('args', {})
    normalized = {}
    for key, value in args.items():
        if key.endswith('handles') or key.endswith('handles_a') or key.endswith('handles_b') or key == 'room_ids':
            normalized[key] = sorted(aliases.get(h, h) for h in value)
        else:
            normalized[key] = value
    function = prop['function_name']
    signature = json.dumps({'function_name': function, 'args': normalized}, sort_keys=True)

    def names(key):
        values = normalized.get(key, [])
        if len(values) > 3:
            return f"any of {len(values)} targets"
        return ' / '.join(values) or '?'

    obj = names('object_handles')
    count = args.get('number', 1)
    if len(args.get('object_handles', [])) > 1:
        obj = f"at least {count} of [{obj}]"
    if function in ('is_on_top', 'is_inside'):
        label = f"{obj} {'on' if function == 'is_on_top' else 'inside'} {names('receptacle_handles')}"
    elif function == 'is_waited_inside':
        label = f"Complete Wait with {obj} inside {names('receptacle_handles')}"
    elif function == 'is_next_to':
        label = f"{names('entity_handles_a')} next to {names('entity_handles_b')}"
    elif function == 'is_in_room':
        label = f"{obj} in {names('room_ids')}"
    elif function == 'is_on_floor':
        label = f"{obj} on floor"
    else:
        label = f"{obj}: {function.removeprefix('is_').replace('_', ' ')}"
    if args.get('is_same_receptacle'):
        label += ' (same receptacle)'
    return signature, label


def subtask_record(episode, scene, sim, source):
    aliases = {h: name for name, h in scene.get('receptacle_to_handle', {}).items()}
    aliases.update({h: name for name, h in episode.get('info', {}).get('variant_spec', {}).get('entity_handles', {}).items()})
    aliases.update({h: name for name, h in scene.get('room_to_id', {}).items()})
    criteria = sim['criteria']
    statuses = [c['status'] for c in criteria]
    method = sim['recovery_method']
    groups = sim.get('ambiguous_failure_groups', [])
    # Exact class-level explanations can constrain assignments, but cannot
    # identify one of several identical objects arbitrarily.
    if 'unknown' in statuses and sim['metrics'].get('task_percent_complete') is not None:
        recovered, recovery, ambiguous = recover_from_metrics(
            [c['proposition'] for c in criteria], sim['metrics'], categories())
        if any(s != 'unknown' for s in recovered):
            statuses, method, groups = recovered, recovery, ambiguous
    result = []
    for c, status in zip(criteria, statuses):
        signature, label = criterion_display(c['proposition'], aliases)
        result.append({'id': c['id'], 'key': signature, 'label': label,
                       'status': status, 'proposition': c['proposition']})
    return {'criteria': result, 'method': method, 'source': source,
            'source_event_id': sim.get('source_event_id'), 'ambiguous_groups': groups,
            'metrics': sim['metrics']}


def chart_data(rows):
    """Compact predicate dictionary plus per-run observations for drill-down."""
    definitions, runs = {}, []
    for row in rows:
        record = row.get('checkpoint_results') if row.get('completion') is None else None
        record = record or row.get('subtask_results')
        if record is None:
            continue
        observations = []
        for c in record['criteria']:
            key = (row['task'], c['key'])
            if key not in definitions:
                definitions[key] = {'id': len(definitions), 'task': row['task'],
                                    'label': c['label'], 'predicate': json.loads(c['key']),
                                    'base_order': None, 'variant_order': int(c['id'][1:])}
            definition = definitions[key]
            definition['variant_order'] = min(definition['variant_order'], int(c['id'][1:]))
            if row.get('task_id', '').endswith('-ACC-BASE'):
                definition['base_order'] = int(c['id'][1:])
            observations.append({'criterion': definitions[key]['id'], 'id': c['id'],
                                 'status': c['status']})
        runs.append({k: row.get(k) for k in ('id', 'condition', 'task', 'task_id',
                     'memory', 'uncertainty', 'iteration', 'folder_url', 'html_links')} | {
                     'criteria': observations, 'method': record['method'],
                     'source': record['source'], 'source_event_id': record['source_event_id'],
                     'ambiguous_groups': record['ambiguous_groups'],
                     'metrics': record['metrics']})
        runs[-1]['score_basis'] = record.get('score_basis', 'final' if row.get('completion') is not None else 'unavailable')
        runs[-1]['checkpoint'] = record.get('checkpoint')
    return {'definitions': list(definitions.values()), 'runs': runs}
