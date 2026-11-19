"""Text-only judge input: rendered trace content, named initial state and scores.

Raw packets remain audit artifacts. This module never substitutes raw execution
logs for a missing HTML trace. The validation adapter is local, not model input.
"""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import hashlib
import gzip
import json
import re

from episode_context import episode_context, items
from judge_view import expand_blocks

VERSION = 'html-judge-v3'


class Node:
    def __init__(self, tag='', attrs=(), parent=None):
        self.tag, self.attrs, self.parent, self.children = tag, dict(attrs), parent, []

    def has(self, cls):
        return cls in self.attrs.get('class', '').split()

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()

    def text(self):
        if self.tag in ('script', 'style', 'img', 'svg'):
            return ''
        parts = [c.text() if isinstance(c, Node) else c for c in self.children]
        value = ''.join(parts)
        return '\n' + value + '\n' if self.tag in ('div', 'p', 'pre', 'br', 'li', 'tr') else value


def tidy(text):
    return '\n'.join(line.strip() for line in text.splitlines() if line.strip())


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = self.current = Node()
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.current)
        self.current.children.append(node)
        if tag not in ('area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'):
            self.current = node

    def handle_endtag(self, tag):
        node = self.current
        while node.parent:
            if node.tag == tag:
                self.current = node.parent
                break
            node = node.parent

    def handle_data(self, data):
        self.current.children.append(data)


def nearest(node, cls):
    node = node.parent
    while node:
        if node.has(cls):
            return node
        node = node.parent
    return None


def class_text(node, cls):
    return next((tidy(n.text()) for n in node.walk() if n.has(cls)), '')


def sections(node):
    result = []
    for section in node.walk():
        if not section.has('section') or nearest(section, 'step') not in (None, node):
            continue
        label = class_text(section, 'section-label')
        value = class_text(section, 'section-content')
        if label and value:
            result.append((label, value))
    return result


def parse_value(value):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def compact_observations(steps):
    """Remove bookkeeping and exact repeated observations, never action attempts."""
    by_agent = {}
    for step in steps:
        previous = by_agent.setdefault(step.get('agent', '0'), {})
        observations = step.get('observations', {})
        diagnostics = observations.pop('Search and belief diagnostics', None)
        if isinstance(diagnostics, dict):
            reasons = {phase: value['termination_reason'] for phase, value in diagnostics.items()
                       if isinstance(value, dict) and value.get('termination_reason')}
            if reasons:
                observations['Termination'] = reasons
        for label, value in list(observations.items()):
            if label in previous and previous[label] == value:
                del observations[label]
            else:
                previous[label] = value
        if not observations:
            step.pop('observations', None)


def read_trace(root):
    """Read rendered sections only. Preserve repeated attempts and source order."""
    root = Path(root)
    steps, planning, hashes, warnings = [], [], {}, []
    paths = sorted((root / 'dataset' / 'traces').glob('**/trace-*.html'))
    for path in paths:
        raw = path.read_bytes()
        hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        doc = Document(raw.decode('utf-8', errors='replace')).root
        agent = path.parent.name if path.parent.name.isdigit() else None
        for node in doc.walk():
            if not node.has('step') or nearest(node, 'step'):
                continue
            action = class_text(node, 'step-action')
            if not action:
                continue
            step = {'id': f'step_{len(steps) + 1}', 'action': action}
            if (node.attrs.get('data-status') == 'skipped'
                    and (node.attrs.get('data-react-skip-policy') != 'robot_sg_before_action'
                         or node.attrs.get('data-target-known-before-action') == 'true')):
                step['status'] = 'skipped'
                step['counts_as_error'] = False
                if node.attrs.get('data-target-known-before-action') == 'true':
                    step['target_in_robot_sg_before_action'] = True
            if len(paths) > 1:
                step['agent'] = agent
                step['source_step'] = class_text(node, 'step-number')
            for label, value in sections(node):
                if label in ('Robot camera', 'Image context', 'Image timing'):
                    continue
                if label == 'Result':
                    step['result'] = value
                elif label == 'Thought' and value not in ('No thought', 'None'):
                    step['thought'] = value
                elif label == 'Execution Log':
                    if value.removeprefix('Observation: ').strip() != step.get('result'):
                        step['execution_feedback'] = value
                elif label != 'Thought':
                    step.setdefault('observations', {})[label] = parse_value(value)
            # Previews are only a fallback, never duplicate full feedback.
            if 'result' not in step:
                feedback = class_text(node, 'step-preview')
                if feedback:
                    step['result'] = feedback
            steps.append(step)
    # VLM planning is a separate ordered source, not additional skill attempts.
    # The alternative simple tree and source-record dumps duplicate this content.
    pddl_htmls = [p for d in ('vlm_tamp_pddl', 'rail_planner') for p in sorted((root / d).glob('*_pddl.html'))]
    for path in pddl_htmls:
        raw = path.read_bytes()
        hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        doc = Document(raw.decode('utf-8', errors='replace')).root
        current = next((n for n in doc.walk() if n.attrs.get('id') == 'vt-current'), doc)
        previous_context = None
        for node in current.walk():
            if not node.has('step') or nearest(node, 'step'):
                continue
            record = {'id': f'plan_{len(planning) + 1}', 'subgoal': class_text(node, 'step-action')}
            content = {}
            for label, value in sections(node):
                if label == 'Prompt':
                    # The visible prompt contains a repeated tool tutorial. Only
                    # its scene observation belongs in this evidence view.
                    match = re.search(r'(Furniture:.*?)(?=\n(?:Search/inspection counts:|Task:|Instruction:)|\Z)', value, re.S)
                    if match:
                        content['Observed scene'] = match.group(1).strip()
                elif label in ('Response', 'Planner decision', 'Recorded PDDL plans, including retries',
                               'Recorded status history', 'Execution log', 'Task outcome', 'Node'):
                    content[label] = value
            context = {k: content[k] for k in ('Response', 'Observed scene') if k in content}
            if context and context == previous_context:
                for key in context:
                    content.pop(key)
            elif context:
                previous_context = context
            if content:
                record.update(content)
                planning.append(record)
    compact_observations(steps)
    if not steps:
        warnings.append('No readable saved HTML action trace. Raw logs have not been substituted.')
    if len(paths) > 1:
        warnings.append('Multiple HTML streams: steps retain source order within each agent; cross-agent timing is unavailable.')
    return {'available': bool(steps), 'steps': steps, 'planning': planning,
            'ordering': 'Steps are high-level attempts in HTML order. Planning records are a separate source sequence; do not infer interleaving.',
            'observation_encoding': 'An observation section identical to its last displayed value is omitted. Every action attempt remains. Search timings, tree bounds and counters are excluded; termination reasons are retained.',
            'warnings': warnings}, hashes


class Names:
    def __init__(self, packet):
        self.mapping = {}
        self.unresolved = {}
        scene = packet.get('reference_truth_not_robot_knowledge', {}).get('scene_info.json') or {}
        for key in ('object_to_handle', 'name_to_handle', 'recep_to_handle', 'receptacle_to_handle'):
            for name, handle in scene.get(key, {}).items():
                if isinstance(handle, str):
                    self.mapping[handle] = name
        # Archived generated entity mapping is especially important for legacy
        # runs without simulator graph snapshots. Never infer names by ordering.
        self.dataset = {}
        run_path = packet.get('provenance', {}).get('run_path')
        if run_path:
            for filename in ('dataset.json.gz', 'dataset.json'):
                path = Path(run_path) / 'input' / filename
                if path.is_file():
                    raw = path.read_bytes()
                    data = json.loads(gzip.decompress(raw) if filename.endswith('.gz') else raw)
                    episodes = data.get('episodes', [])
                    if len(episodes) == 1:
                        self.dataset = episodes[0]
                        generated = self.dataset.get('info', {}).get('variant_spec', {}).get('entity_handles', {})
                        self.mapping.update({handle: name for name, handle in generated.items()})
                    break
        # Identity can be recovered from later snapshots without using their
        # changing states as the initial world state.
        for value in packet.get('json_blocks', {}).values():
            if isinstance(value, dict) and 'name' in value and 'sim_handle' in value:
                self.add([value])
            if isinstance(value, dict) and 'reference_world_graph' in value:
                self.add(expand_blocks(packet, value['reference_world_graph']))
        self.initial = None
        first_action = min((a['start_order'] for a in packet.get('attempts', [])), default=float('inf'))
        for event in packet.get('events', []):
            data = event.get('data', {})
            state = data.get('evaluator_only_state', {}) if isinstance(data, dict) else {}
            if state:
                state = expand_blocks(packet, state)
                graph = state.get('reference_world_graph') or []
                self.add(graph)
                if (self.initial is None and graph and event.get('order', float('inf')) <= first_action
                        and data.get('event') != 'terminal' and data.get('sim_step_count') in (None, 0, 0.0)):
                    self.initial = graph

    def add(self, graph):
        if not isinstance(graph, list):
            return
        for node in graph:
            if isinstance(node, dict) and node.get('sim_handle') is not None and not node.get('name', '').startswith('rec_'):
                self.mapping[str(node['sim_handle'])] = node['name']

    def name(self, handle):
        handle = str(handle)
        if handle in self.mapping:
            return self.mapping[handle]
        if handle not in self.unresolved:
            self.unresolved[handle] = f'unresolved_entity_{len(self.unresolved) + 1}'
        return self.unresolved[handle]

    def criterion(self, proposition):
        args = {}
        for key, value in proposition.get('args', {}).items():
            if 'handle' in key:
                args[key.replace('entity_handles', 'entities').replace('_handles', 's').replace('_handle', '')] = [self.name(x) for x in value] if isinstance(value, list) else self.name(value)
            else:
                args[key] = value
        return {'predicate': proposition.get('function_name'), **args}

    def scene(self, packet):
        if self.initial is None:
            return {'id': 'initial_scene', 'source': 'declared scenario; simulator initial snapshot unavailable',
                    'declared_state': items(packet.get('reference_truth_not_robot_knowledge', {}).get('spec.md', ''), 'Initial world state')}
        nodes = {n['name']: n for n in self.initial}
        rooms = {name for name, node in nodes.items() if node.get('relations', {}).get('house') == 'inside'}

        def parent(name):
            for target, relation in nodes.get(name, {}).get('relations', {}).items():
                if relation in ('on', 'inside'):
                    if target.startswith('rec_'):
                        target = next((t for t, rel in nodes.get(target, {}).get('relations', {}).items() if rel == 'joint'), target)
                    return target, relation
            return None, None

        def room(name):
            visited = set()
            while name and name not in visited:
                if name in rooms:
                    return name
                visited.add(name)
                name, _ = parent(name)
            return None

        grouped = {name: [] for name in sorted(rooms)}
        result = {'id': 'initial_scene', 'source': 'simulator initial snapshot', 'rooms': grouped,
                  'state_note': 'Only recorded states are listed; an omitted state is unknown. This reference truth is not robot knowledge.'}
        for name, node in nodes.items():
            if name == 'house' or name in rooms or name.startswith('rec_'):
                continue
            target, relation = parent(name)
            details = []
            if name.startswith('agent_'):
                room_name = node.get('properties', {}).get('initial_room') or room(name)
            else:
                room_name = room(name)
                if target and target not in rooms:
                    details.append(f'{relation} {target}')
            details.extend(f'{key}={json.dumps(value, ensure_ascii=False)}'
                           for key, value in node.get('properties', {}).get('states', {}).items())
            entry = name + ('(' + ', '.join(details) + ')' if details else '')
            grouped.setdefault(room_name or 'unknown_room', []).append(entry)
        result['rooms'] = {name: ', '.join(entries) for name, entries in grouped.items()}
        return result


def scores(score, names):
    return {'outcome': score['outcome'],
            'subtasks': [{'id': c['id'], 'criterion': names.criterion(c['proposition']),
                          'result': {'met': 'succeeded', 'unmet': 'failed', 'unknown': 'unknown'}[c['status']]}
                         for c in score['criteria']],
            'ambiguous_failure_groups': score.get('ambiguous_failure_groups', [])}


def compact_examples(examples):
    """Short reviewed demonstrations, not complete archived example packets."""
    result = []
    for example in examples:
        packet, output = example['input'], example['expected_output']
        shown = validation_view(example['model_input'], packet) if example.get('model_input') else packet
        # Keep reviewed explanations and their linked criteria/attempt excerpts.
        # No boilerplate assessments or raw event/schema copies in demonstrations.
        labels = []
        cited_attempts = set()
        for family in ('failures', 'inefficiencies'):
            for incident in output.get(family, []):
                cited_attempts.update(incident['attempt_ids'])
                labels.append({k: incident[k] for k in ('label', 'criterion_ids', 'unmet_group_ids', 'explanation', 'confidence_raw')})
        names = Names(packet)
        for label in labels:
            for handle, name in sorted(names.mapping.items(), key=lambda pair: -len(pair[0])):
                if len(handle) > 8:
                    label['explanation'] = label['explanation'].replace(handle, name)
        planning = []
        for event in expand_blocks(packet, packet.get('events', [])):
            data = event.get('data', {})
            if isinstance(data, dict) and data.get('event') in ('vlm_english_subgoals', 'vlm_predicate_subgoals', 'planner_decision'):
                planning.append({k: data[k] for k in ('event', 'reprompt_round', 'response', 'decision', 'evidence') if k in data})
        result.append({'instruction': packet['instruction'], 'subtask_results': scores(packet['simulator'], names),
                       'planning_excerpts': planning,
                       'action_excerpts': [{'action': a['action'], 'arguments': a['arguments'], 'feedback': a['responses']}
                                           for a in shown.get('attempts', []) if a['id'] in cited_attempts],
                       'expected_incidents': labels,
                       'scope': 'Partial reviewed example; unlisted labels are unassessed. Unknown scores remain unknown.'})
        if example.get('model_input'):
            cited_evidence = {eid for family in ('failures', 'inefficiencies')
                              for incident in output.get(family, []) for eid in incident['evidence_ids']}
            result[-1]['evidence_excerpts'] = [e['data'] for e in shown['events']
                                               if e['id'] in cited_evidence and e['id'] != 'trace']
    return result


def build_input(packet, definitions, examples):
    context = episode_context(packet)
    names = Names(packet)
    trace, hashes = read_trace(packet['provenance']['run_path'])
    trace['id'] = 'trace'
    trace['execution_status'] = packet['coverage']['execution_status']
    trace['history_complete'] = bool(trace['available'] and packet['coverage']['history_complete'])
    # No unseen raw action may be silently dropped while declaring full coverage.
    if len(trace['steps']) < len(packet.get('attempts', [])):
        trace['history_complete'] = False
        trace['warnings'].append('HTML has fewer attempts than the saved action record; omitted behavior is unavailable.')
    subtask = {'id': 'subtask_results', 'score_basis': 'final' if packet['simulator']['outcome'] != 'unavailable' else 'unavailable',
               **scores(packet['simulator'], names)}
    checkpoint = packet.get('checkpoint_evaluation')
    if checkpoint:
        subtask['last_saved_state'] = {'score_basis': 'checkpoint, not final; cannot authorize failure labels', **scores(checkpoint, names)}
    body = {
        'scenario': {'episode_id': context['episode_id'], 'instruction': packet['instruction'],
                     'memory_condition': context['memory_condition'], 'availability_condition': context['availability_condition'],
                     'written_requirements': context['success_criteria'], 'source': context['scenario_source']['kind'],
                     'warnings': context['warnings']},
        'initial_scene': names.scene(packet),
        'initial_robot_memory': {'id': 'initial_robot_memory', 'source': 'declared scenario memory; actual observations are in the trace',
                                 'entries': context['initial_robot_memory']},
        'trace': trace,
        'subtask_results': subtask,
        'definitions': definitions,
        'examples': compact_examples(examples),
        'output_format': {
            'format': 'Return the strict JSON schema supplied with this request.',
            'references': 'criterion_ids use p IDs; unmet_group_ids use g IDs. Each step_N is both an attempt ID and evidence ID. plan_N, initial_scene, initial_robot_memory, subtask_results, trace are evidence only.',
            'incidents': 'Each incident: label, affected criteria or ambiguous group, attempt/evidence IDs, brief explanation and alternative, numeric confidence_raw (0 to 1). One inefficiency incident per attempt per label.',
            'counts': 'Code computes per-label incident counts and distinct affected attempts/subtasks after validation. Shared attempts count once in distinct totals.',
            'coverage': 'Assess every label. Missing history cannot establish absence. Unclassified failed criteria must be listed; unknown results remain unknown.'},
    }
    if names.unresolved:
        body['scenario']['warnings'].append('Some simulator identities could not be resolved: ' + ', '.join(names.unresolved.values()) + '. Do not infer their identity.')
    return body, hashes


def validation_view(body, packet):
    """Only displayed evidence can be cited; retain canonical scores locally."""
    events, attempts = [], []
    for order, step in enumerate(body['trace']['steps']):
        match = re.match(r'([\w]+)\s*\[(.*)\]', step['action'], re.S)
        if not match:
            match = re.match(r'\(?([\w]+)\s+(.*?)\)?$', step['action'], re.S)
        action, arguments = match.groups() if match else (step['action'], '')
        events.append({'id': step['id'], 'order': order, 'data': step})
        attempts.append({'id': step['id'], 'action': action, 'arguments': arguments,
                         'responses': [step.get('result', '')], 'evidence_ids': [step['id']]})
    for plan in body['trace']['planning']:
        events.append({'id': plan['id'], 'order': None, 'data': plan})
    for key in ('initial_scene', 'initial_robot_memory', 'subtask_results', 'trace'):
        events.append({'id': key, 'order': None, 'data': body[key]})
    return {'events': events, 'attempts': attempts, 'simulator': packet['simulator'],
            'coverage': {'history_complete': body['trace']['history_complete'], 'execution_status': body['trace']['execution_status']}}
