"""Small teaching examples, separate from complete run evidence and audit packets."""
import copy
import json
import re

from episode_context import spec_fields
from judge_view import expand_blocks, parse_native_record


_INCIDENT_KEYS = ('id', 'behavior_id', 'label', 'criterion_ids', 'unmet_group_ids',
                  'attempt_ids', 'evidence_ids', 'explanation', 'alternative_explanation',
                  'confidence_raw', 'object_name', 'outdated_memory_object',
                  'prior_failed_pick_evidence_ids', 'unmet_precondition')
_PLANNING_KEYS = ('event', 'reprompt_round', 'response', 'known_objects',
                  'search_counts', 'description_excerpt', 'decision', 'evidence',
                  'status', 'subgoal', 'already_succeeded', 'explored_room')


def _pick(value, keys):
    return {key: copy.deepcopy(value[key]) for key in keys if key in value}


def _graph_excerpt(graph, names):
    if not isinstance(graph, str):
        return None
    lines = [line for line in graph.splitlines()
             if any(re.search(r'(?<![\w])' + re.escape(name) + r'(?![\w])', line)
                    for name in names)]
    return '\n'.join(lines) or None


def simple_examples(examples):
    """Project already validated examples into partial, citable demonstrations.

    All extracted attempts remain present. Selected source events are excerpts;
    neither omitted categories nor missing per-criterion scores become negatives.
    No source-run identity, paths, hashes, editable review notes or raw spec are sent.
    """
    result = []
    for example in examples:
        packet = example['input']; expected = example['expected_output']
        incidents = {family: [_pick(i, _INCIDENT_KEYS) for i in expected.get(family, [])]
                     for family in ('failures', 'inefficiencies')}
        cited = {eid for values in incidents.values() for incident in values
                 for key in ('evidence_ids', 'prior_failed_pick_evidence_ids')
                 for eid in incident.get(key, [])}
        # Notes can contain critical counterevidence or absence qualifications.
        notes = [copy.deepcopy(note) for key in ('important_notes', 'possible_failure_notes')
                 for note in expected.get(key, [])]
        cited.update(eid for note in notes for eid in note.get('evidence_ids', []))
        attempts = [_pick(a, ('id', 'action', 'arguments', 'responses', 'evidence_ids'))
                    for a in packet.get('attempts', [])]
        names = set(re.findall(r'\b[a-z][a-z_]*_\d+\b', json.dumps([incidents, attempts])))
        refs = packet.get('reference_truth_not_robot_knowledge', {})
        fields = spec_fields(refs.get('spec.md', ''))
        events = expand_blocks(packet, packet.get('events', []))
        selected = []; aliases = {}
        for event in events:
            data = event.get('data', {})
            if isinstance(data, dict) and 'raw_record' in data:
                data = parse_native_record(data['raw_record'])
            if isinstance(data, dict):
                for node in data.get('evaluator_only_state', {}).get('reference_world_graph', []):
                    if node.get('name') in names and node.get('sim_handle'):
                        aliases[node['sim_handle']] = node['name']
            if event['id'] not in cited:
                continue
            if not isinstance(data, dict):
                selected.append({'id': event['id'], 'text': copy.deepcopy(data)})
                continue
            excerpt = _pick(data, _PLANNING_KEYS)
            observed = data.get('planner_observed', {})
            if observed:
                view = _pick(observed, ('high_level_actions', 'responses', 'thought'))
                knowledge = {agent: text for agent, graph in observed.get('curr_graph', {}).items()
                             if (text := _graph_excerpt(graph, names))}
                if knowledge:
                    view['robot_knowledge_excerpt'] = knowledge
                if view:
                    excerpt['planner_observed'] = view
            # Existing concise references remain citable; redundant scoring state
            # is represented once below instead of resending complete world graphs.
            selected.append({'id': event['id'], **excerpt})
        simulator = _pick(packet.get('simulator', {}),
                          ('outcome', 'metrics', 'criteria', 'ambiguous_failure_groups'))
        demo = {
            'scope': 'Partial teaching example. All extracted attempts are listed; selected planning records are excerpts. Unlisted labels are unassessed, not absent. Unknown scores remain unknown.',
            'instruction': packet['instruction'],
            'declared_scenario': {'success_criteria': fields['success_criteria'],
                                  'initial_robot_memory': fields['initial_robot_memory']},
            'simulator': simulator,
            'attempts': attempts,
            'evidence': selected,
            'expected_incidents': incidents,
        }
        if aliases:
            demo['handle_aliases'] = aliases
        if notes:
            demo['qualifications'] = notes
        result.append(demo)
    return result
