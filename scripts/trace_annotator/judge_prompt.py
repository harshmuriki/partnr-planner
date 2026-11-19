"""Lean text-only request sections, projected from the auditable evidence view."""
import copy
from episode_context import items
from judge_view import model_view, expand_blocks, clean

PROMPT_VERSION = 'judge-prompt-v4'
PROMPT_POLICY = '''The run contains scenario information, initial_state, simulator subtask
results, ordered action attempts, and events containing observations and planning
records. Event IDs link attempts to their evidence; action fields removed from an
event are represented once in attempts. Events from supplemental sources are not
additional attempts. Use their timestamps/steps when available; do not invent a
cross-source chronology. Initial state and evaluator changes are reference truth,
not proof of robot knowledge. Declared memory is distinct from logged observations.
The scenario's human ground-truth replay timing is not this baseline's timing.
Examples are short partial demonstrations; unlisted labels are unassessed.
Return the strict JSON schema, including numeric confidence_raw for each incident.
Counts per label and distinct failed subtasks/inefficient attempts are calculated
by code after validation, so two labels on one subtask do not become two failures.
Labels are not mutually exclusive. If the SAME attempt independently satisfies
multiple inefficiency definitions, emit one incident per supported label using
the same behavior_id and attempt_id. Do not omit a supported precondition label
because that attempt also qualifies as repeated_actions. Counts per label can
overlap; distinct_attempt_count counts the shared attempt only once.
For new ReAct runs, steps explicitly marked status=skipped and counts_as_error=false
are missing-object picks whose target was verified in the robot scene graph BEFORE
the action. Only those verified skips are exempt. Unknown or absent prior
membership remains an error; the raw failure message alone is not an exemption.
Historical artifacts retain their previously recorded skipped classification.
Do not count recorded skips as action errors or inefficiencies,
even when their preserved raw feedback says "Unexpected failure" or "Failed".
For remaining steps, evaluate explicit precondition feedback on EVERY attempt, including the first
attempt and otherwise justified retries. Exemption of an initial stale-object
absence-discovery pick applies to hallucinated_object only; it does not exempt
an out-of-reach pick from precondition_failures. An initial attempt may be exempt
from repeated_actions while still independently showing a precondition failure.
Review multi-action cycles as well as immediately adjacent repeated actions.
Intervening navigation, exploration, or a new planning call is not automatically
a justified recovery. Look for an evidenced correction, new information, or a
concrete useful recovery purpose. A successful Navigate to the same broad floor
location does not prove object-specific reach or a different approach. Distinguish
an initially reasonable recovery from continued repetitions of an unchanged
unsuccessful strategy. Do not invent movement details omitted from the evidence.
In simulator.criteria, met means succeeded, unmet means failed, and unknown must
remain unknown. Only final simulator scores authorize failure labels. Last saved
state scoring is separate. Cite only supplied event and attempt IDs.
'''


def duplicate_supplemental_ids(packet):
    """Recognize complete duplicate planner-log sequences, never action loops.

    Match original observations before view projection removes unchanged knowledge.
    Unknown substantive fields, unmatched records, and novel stop signals keep the
    whole source sequence. Each match consumes a distinct execution record.
    """
    events = expand_blocks(packet, packet['events'])
    executions = [(e['data'].get('planner_observed', {}), e['data'].get('event')) for e in events
                  if e['kind'] == 'execution' and isinstance(e.get('data'), dict)]
    groups = {}
    for event in events:
        if (event['kind'] == 'planner_context' and isinstance(event.get('data'), dict)
                and 'high_level_actions' in event['data']):
            groups.setdefault(event['source'], []).append(event)
    protected = {eid for attempt in packet['attempts'] for eid in attempt['evidence_ids']}
    protected.update(score['source_event_id'] for score in
                     [packet['simulator'], packet.get('checkpoint_evaluation') or {}]
                     if score.get('source_event_id'))
    omitted = set()
    for group in groups.values():
        cursor = 0; matched = []
        for event in group:
            observed = clean(event['data'])
            if observed.get('instruction') == packet['instruction']:
                observed.pop('instruction')
            # A true done flag is redundant only for an explicit Done action
            # matched to a terminal execution event, never for an ordinary action.
            done = observed.get('is_done')
            terminal_done = False
            if isinstance(done, dict) and done:
                actions = observed.get('high_level_actions', {})
                if all(v is False or (v is True and isinstance(actions.get(a), list)
                       and actions[a] and actions[a][0] == 'Done') for a, v in done.items()):
                    terminal_done = any(done.values())
                    observed.pop('is_done')
            while cursor < len(executions):
                execution, boundary = executions[cursor]; cursor += 1
                if terminal_done and boundary != 'terminal':
                    continue
                if all(key in execution and execution[key] == value
                       for key, value in observed.items()):
                    matched.append(event['id'])
                    break
            else:
                break
        if len(matched) == len(group):
            omitted.update(set(matched) - protected)
    return omitted


def prompt_view(packet):
    view = copy.deepcopy(model_view(packet))
    duplicates = duplicate_supplemental_ids(packet)
    view['events'] = [e for e in view['events'] if e['id'] not in duplicates]
    view['coverage']['duplicate_supplemental_records_removed'] = len(duplicates)
    context = view.pop('episode_context')
    view['scenario'] = {k: context[k] for k in (
        'episode_id', 'memory_condition', 'availability_condition',
        'success_criteria', 'initial_robot_memory', 'warnings')}
    view['scenario']['source'] = context['scenario_source']['kind']
    reference = context['viewer_reference']
    replay = reference.get('ground_truth_replay')
    view['scenario']['viewer_reference'] = {
        'scope': reference['scope'], 'object_summary': reference['object_summary'],
        'human_ground_truth_replay': {k: v for k, v in replay.items() if k != 'source'} if replay else None}
    refs = packet.get('reference_truth_not_robot_knowledge', {})
    scene = refs.get('scene_info.json') or {}
    # Keep identity maps: historical runtime aliases can differ from spec names.
    view['identity_mapping'] = {k: copy.deepcopy(scene[k]) for k in
                               ('object_to_handle', 'name_to_handle', 'recep_to_handle', 'receptacle_to_handle',
                                'spec_furniture_aliases', 'room_to_id') if k in scene}
    view.pop('reference_truth_not_robot_knowledge')
    view['initial_state'] = {
        'declared_world_state': items(refs.get('spec.md', ''), 'Initial world state'),
        'observed_state_event_id': None,
        'note': 'Declared world state is scenario reference. The first logged evaluator state is below; robot knowledge stays in events.'}
    linked = {}
    for attempt in view['attempts']:
        attempt.pop('boundary_source', None)
        for eid in attempt['evidence_ids']:
            linked.setdefault(eid, []).append(attempt['id'])
    for event in view['events']:
        event.pop('source', None)
        event.pop('locator', None)
        data = event['data']
        if event['id'] in linked:
            data['attempt_ids'] = linked[event['id']]
            observed = data.get('planner_observed', {})
            # Actions and their feedback remain on each unique attempt.
            observed.pop('high_level_actions', None)
            observed.pop('responses', None)
            for key in ('replanned', 'replan_required'):
                observed.pop(key, None)
            if isinstance(observed.get('thought'), dict):
                observed['thought'] = {a: t for a, t in observed['thought'].items() if t}
            data['planner_observed'] = {k: v for k, v in observed.items() if v not in ({}, [], None, '')}
            if not data['planner_observed']:
                data.pop('planner_observed')
        state = data.get('evaluator_state_changes')
        if state:
            if view['initial_state']['observed_state_event_id'] is None and state.get('entities_changed'):
                view['initial_state']['observed_state_event_id'] = event['id']
                view['initial_state']['observed_entities'] = state.pop('entities_changed')
            # Cumulative tracker histories repeat all preceding states each time.
            # Preserve only the unseen suffix, per stream, in the block below.
    previous = {}
    for event in view['events']:
        state = event['data'].get('evaluator_state_changes', {})
        history = state.get('state_changes')
        if isinstance(history, list):
            stream = 'merged' if view['coverage']['model_view']['state_streams_merged_by_timestamp'] else event['kind']
            old = previous.get(stream, [])
            if history[:len(old)] == old:
                state['state_changes'] = history[len(old):]
            previous[stream] = history
    for criterion in view['simulator']['criteria']:
        criterion['result'] = {'met': 'succeeded', 'unmet': 'failed', 'unknown': 'unknown'}[criterion['status']]
    view['coverage']['view_warnings'] = view['coverage']['model_view']['warnings']
    view['coverage'].pop('model_view', None)
    view.pop('image_inputs', None)
    view['view_version'] = PROMPT_VERSION
    return {key: view[key] for key in ('view_version', 'instruction', 'scenario',
            'initial_state', 'identity_mapping', 'simulator', 'checkpoint_evaluation',
            'attempts', 'events', 'coverage')}
