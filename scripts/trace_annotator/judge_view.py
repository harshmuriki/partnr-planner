"""Deterministic, readable model input; raw packets remain the audit record."""
import ast
import copy
import json
import re
from collections import Counter
from episode_context import episode_context

VIEW_VERSION = 'judge-view-v1'
VIEW_POLICY = '''The run and example inputs are readable evidence views, not raw logs.
episode_context contains the scenario instruction, written success criteria and
declared initial robot memory. Its viewer_reference timing/steps/actions describe
the human ground-truth replay, not this baseline run. Written requirements do not
create simulator scores. Declared memory is not proof of the actual prompt content;
use logged knowledge for claims about what the planner knew. Respect source warnings.
This annotation is text/data only. No images or video frames are supplied, even
when the evaluated baseline used images. Do not claim direct visual inspection;
distinguish logged planner observations from independently verified visual facts.
Event IDs and attempt IDs refer to original source records. Each event has its
source and locator. Repeated actions are retained as separate attempts.
State updates contain the full new value of changed named entities/fields; omitted
fields are unchanged, not absent. Removed names are explicit. Evaluator observations
are reference truth, not robot knowledge. Only simulator supplies final scores;
checkpoint_evaluation remains last-saved-state evidence and never enables failures.
Coordinates, physics diagnostics, API envelopes, repeated prompt templates and
unchanged snapshots are omitted. Never infer geometric prerequisites from omitted
coordinates, or label absence from an incomplete history. Annotation must cite
only evidence actually supplied in this view. No JSON-reference or patch decoding
is required. Read event timestamps/steps: supplemental sources are not additional
attempts and are not necessarily interleaved in chronological array order.
'''

# These fields are transport/rendering/physics details, not planner decisions.
NOISE = {'api_response', 'cost_metrics', 'llm_cost', 'token_usage', 'image_url',
         'image_paths', 'images', 'image_path', 'action_image', 'exploration_images',
         'image_saved_wall_time', 'agent_positions', 'agent_states', 'agent_collisions',
         'total_step_count', 'log_index', 'replanning_count', 'image_source',
         'image_capture_phase', 'image_count', 'prompts', 'prompt', 'traces'}
DIAGNOSTIC = re.compile(r'error|failed|failure|exception|traceback|timeout|timed out|cannot|could not|does not exist|not found|removed|gave up', re.I)


def expand_blocks(packet, value):
    """Decode JSON blocks before text sequences (which may themselves be interned)."""
    cache = {}
    def expand(v):
        if isinstance(v, dict):
            if set(v) == {'json_ref'}:
                key = v['json_ref']
                if key not in cache:
                    cache[key] = expand(packet['json_blocks'][key])
                return cache[key]
            if set(v) == {'text_block_sequence'}:
                parts = expand(v['text_block_sequence'])
                return ''.join(packet['text_blocks'][x['ref']] if isinstance(x, dict) else x for x in parts)
            return {k: expand(x) for k, x in v.items()}
        if isinstance(v, list):
            return [expand(x) for x in v]
        return v
    return expand(value)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k not in NOISE}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def parse_native_record(text):
    """Read logged Python literals safely; never execute constructors in a log."""
    class Scalars(ast.NodeTransformer):
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Name) and node.func.id == 'Action' and not node.args
                    and {k.arg for k in node.keywords} == {'name', 'args'}):
                return ast.Dict(keys=[ast.Constant(k.arg) for k in node.keywords],
                                values=[self.visit(k.value) for k in node.keywords])
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {'np', 'numpy'} and node.func.attr in {'float32', 'float64', 'int32', 'int64'}
                    and len(node.args) == 1 and not node.keywords):
                return self.visit(node.args[0])
            return node
    try:
        return ast.literal_eval(Scalars().visit(ast.parse(text, mode='eval')))
    except (ValueError, SyntaxError, TypeError):
        return {'unparsed_record': text}


def restore_checkpoint(state, previous):
    if not isinstance(state, dict) or 'checkpoint_delta' not in state:
        return state
    patch = state['checkpoint_delta']
    if previous is None or patch['base_event_id'] != previous[0]:
        raise ValueError('Checkpoint patch does not refer to the preceding saved state')
    result = copy.deepcopy(previous[1])
    for path in patch['remove']:
        target = result
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    for update in patch['set']:
        path = update['path']
        if not path:
            result = copy.deepcopy(update['value'])
            continue
        target = result
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = copy.deepcopy(update['value'])
    return result


def semantic_state(state, receptacles=None):
    """Keep symbolic state/relations and evaluator history; omit geometry/tick noise."""
    result = {}
    for key in ['task_percent_complete', 'task_state_success', 'task_constraint_validation']:
        if key in state:
            result[key] = state[key]
    tracker = state.get('auto_eval_proposition_tracker', {})
    for key in ['proposition_satisfied_at', 'last_state', 'state_changes', 'propositions', 'dependencies', 'constraints']:
        if key in tracker:
            result[key] = tracker[key]
    graph = state.get('reference_world_graph')
    if isinstance(graph, list):
        by_name = {n['name']: n for n in graph if isinstance(n, dict) and 'name' in n
                   and (receptacles is None or not n['name'].startswith('rec_') or n['name'] in receptacles)}
        result['entities'] = {n['name']: {'sim_handle': n.get('sim_handle'),
            'properties': {k: v for k, v in n.get('properties', {}).items()
                           if k not in {'translation', 'rotation', 'position', 'quaternion', 'bbox', 'aabb', 'size'}},
            'relations': {name: relation for name, relation in n.get('relations', {}).items()
                          if (name in by_name or not name.startswith('rec_'))
                          and not (relation == 'contains' and n['name'] in by_name.get(name, {}).get('relations', {}))}}
            for n in by_name.values()}
    return result


def state_changes(current, previous):
    changes = {k: v for k, v in current.items() if k != 'entities' and (k not in previous or v != previous[k])}
    if 'entities' in current:
        before = previous.get('entities', {})
        changed = {name: node for name, node in current['entities'].items() if before.get(name) != node}
        removed = sorted(set(before) - set(current['entities']))
        if changed:
            changes['entities_changed'] = changed
        if removed:
            changes['entities_removed'] = removed
    removed_fields = sorted(set(previous) - set(current) - {'entities'})
    if removed_fields:
        changes['fields_removed'] = removed_fields
    return changes


def prompt_world_description(text):
    """Retain actual supplied initial knowledge, not the planner's tool tutorial."""
    starts = [i for marker in ['Initial agent location:', 'Furniture:'] if (i := text.find(marker)) >= 0]
    if not starts:
        return None
    start = min(starts)
    ends = [i for marker in ['Possible Actions:', 'Available Actions:', '<|eot_id|>', '<|im_end|>']
            if (i := text.find(marker, start)) >= 0]
    if not ends:
        return None  # Do not guess where task data ends and policy/history begins.
    return text[start:min(ends)].strip()


def model_view(packet, required_ids=()):
    """Preserve scores, attempts, reference identities, and citable semantic changes."""
    events = expand_blocks(packet, packet['events'])
    previous_checkpoint = None
    for event in events:
        if event['kind'] == 'evaluator_checkpoint':
            data = event['data']
            state = restore_checkpoint(data.get('evaluator_only_state', {}), previous_checkpoint)
            data['evaluator_only_state'] = state
            previous_checkpoint = (event['id'], state)
    state_events = [e for e in events if isinstance(e['data'], dict) and 'evaluator_only_state' in e['data']]
    merge_states = bool(state_events) and all(isinstance(e['data'].get('wall_time'), (float, int)) for e in state_events)
    if merge_states:
        # Merge observations before computing changes; never compare the first
        # checkpoint to the end of the execution log and invent a state reversal.
        timed = [e for e in events if e['kind'] in {'execution', 'evaluator_checkpoint'}
                 and isinstance(e['data'].get('wall_time'), (float, int))]
        timed_ids = {e['id'] for e in timed}
        events = sorted(timed, key=lambda e: e['data']['wall_time']) + [e for e in events if e['id'] not in timed_ids]
    # Select the union across the whole history. An emptied receptacle must not
    # disappear from the view and masquerade as removal of a nonexistent entity.
    referenced = json.dumps([packet['simulator'], packet['attempts']])
    receptacles = set()
    for event in state_events:
        for node in event['data']['evaluator_only_state'].get('reference_world_graph', []):
            name = node.get('name', '')
            if name.startswith('rec_') and (name in referenced or str(node.get('sim_handle')) in referenced
                    or any(relation != 'joint' for relation in node.get('relations', {}).values())):
                receptacles.add(name)
    required = set(required_ids)
    required.update(e for a in packet['attempts'] for e in a['evidence_ids'])
    for score in [packet['simulator'], packet.get('checkpoint_evaluation') or {}]:
        if score.get('source_event_id'):
            required.add(score['source_event_id'])
    # Legacy planner_context duplicates the execution event at the same source step.
    execution_locations = {(e['source'], e['locator']) for e in events if e['kind'] == 'execution'}
    previous_states = {}; previous_observed = {}; seen_diagnostics = set()
    kept = []; omitted = Counter(); warnings = []
    for event in events:
        kind = event['kind']; data = event['data']; projected = {}; reason = None
        if kind == 'saved_prompt':
            description = prompt_world_description(data) if isinstance(data, str) else None
            if description:
                projected = {'initial_robot_knowledge_excerpt': description}
            else:
                reason = 'planner_prompt_template'
                warnings.append('A saved prompt had no recognized world-description boundaries; raw prompt retained only in audit packet.')
        elif kind == 'diagnostics_not_attempts':
            lines = []
            for line in data if isinstance(data, list) else []:
                text = line.get('text', '')
                normalized = re.sub(r'^\[\d{2}:\d{2}:\d{2}:\d+\]:\s*', '', text)
                # Keep errors once, never turn repeated motor diagnostics into actions.
                if DIAGNOSTIC.search(text) and normalized not in seen_diagnostics:
                    seen_diagnostics.add(normalized); lines.append(line)
            if lines:
                projected = {'diagnostics_not_actions': lines}
            else:
                reason = 'startup_or_repeated_diagnostics'
        elif kind == 'planner_context' and (event['source'], event['locator']) in execution_locations:
            reason = 'duplicate_execution_source'
        elif isinstance(data, dict):
            if 'raw_record' in data:
                data = parse_native_record(data['raw_record'])
            projected = clean(data)
            # Fixed metadata/instruction already appears in the task section.
            for key in ['instruction', 'schema_version', 'sequence', 'evaluator_only']:
                projected.pop(key, None)
            state = projected.pop('evaluator_only_state', None)
            if isinstance(state, dict):
                semantic = semantic_state(state, receptacles)
                stream = 'evaluator' if merge_states else ('checkpoint' if kind == 'evaluator_checkpoint' else event['source'])
                changes = state_changes(semantic, previous_states.get(stream, {}))
                previous_states[stream] = semantic
                if changes:
                    projected['evaluator_state_changes'] = changes
                elif kind == 'evaluator_checkpoint' and event['id'] not in required:
                    reason = 'unchanged_symbolic_checkpoint'
            observed = projected.get('planner_observed')
            if kind == 'planner_context' and 'high_level_actions' in projected:
                observed = projected
            if isinstance(observed, dict):
                observed.pop('instruction', None)
                observed.pop('stats', None)  # Authoritative evaluator observation retained separately.
                graphs = observed.pop('curr_graph', None)
                if isinstance(graphs, dict):
                    changed = {agent: graph for agent, graph in graphs.items() if previous_observed.get(agent) != graph}
                    previous_observed.update(graphs)
                    if changed:
                        observed['robot_knowledge'] = changed
            # Supplemental native logs keep planning/search decisions and responses.
            if 'unparsed_record' in projected:
                warnings.append('Unparsed native planner record retained verbatim; do not infer its fields.')
        else:
            # Text traces contain resulting object locations not always in action logs.
            projected = {'text': data}
        if reason and event['id'] not in required:
            omitted[reason] += 1
            continue
        if reason and not projected:
            # Explicit example/image citations must retain their supporting text,
            # even when that source would otherwise be redundant.
            projected = clean(data) if isinstance(data, dict) else {'text': data}
        kept.append({**{k: event[k] for k in ['id', 'kind', 'order', 'source', 'locator']}, 'data': projected})
    refs = copy.deepcopy(packet['reference_truth_not_robot_knowledge'])
    context = episode_context(packet)
    if 'spec.md' in refs:
        # The requested viewer summary is structured above; avoid sending the
        # same instruction, criteria and memory a second time as Markdown.
        remainder = re.sub(r'^## (?:Base task|Task instruction / prompt given|Success criteria|Initial robot memory)\s*\n.*?(?=^## |\Z)',
                           '', refs.pop('spec.md'), flags=re.M | re.S).strip()
        if remainder:
            refs['additional_spec_context'] = remainder
    scene = refs.get('scene_info.json')
    if isinstance(scene, dict):
        for key in ['objects', 'object_affordances', 'all_furniture', 'all_rooms']:
            scene.pop(key, None)
    coverage = copy.deepcopy(packet['coverage'])
    coverage['model_view'] = {'version': VIEW_VERSION, 'raw_event_count': len(events), 'sent_event_count': len(kept),
        'omitted_event_counts': dict(omitted), 'warnings': sorted(set(warnings)),
        'omitted_fields': ['geometry/poses', 'empty unreferenced simulator receptacle nodes', 'API envelopes and costs', 'image file bookkeeping', 'repeated prompt/tool templates', 'static object-category/affordance catalog'],
        'state_streams_merged_by_timestamp': merge_states,
        'note': 'Raw packet retained locally. Symbolic changes preserve state transitions; no action attempts are dropped. Reciprocal contains edges are shown through the contained entity. Geometric prerequisites require explicit feedback; coordinates are not supplied.'}
    return {'view_version': VIEW_VERSION, 'instruction': packet['instruction'],
        'episode_context': context,
        'simulator': packet['simulator'], 'checkpoint_evaluation': packet.get('checkpoint_evaluation'),
        'reference_truth_not_robot_knowledge': refs, 'attempts': packet['attempts'], 'events': kept,
        'coverage': coverage, 'image_inputs': packet.get('image_inputs', [])}
