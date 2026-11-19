"""Portable manifest ingestion and evidence packets; no Habitat imports or replay."""
import ast
from collections import Counter
import copy
import gzip
import hashlib
import json
import re
from pathlib import Path

from judge import compact, share_text_blocks
from sim_results import structured_results
from study import atomic_json

BASELINES = ['react', 'react-no-image', 'vlm-tamp', 'vlm-tamp-no-image', 'tru-pomdp']
ALIASES = {'react_images': 'react', 'react_no_images': 'react-no-image',
           'vlm_tamp_images': 'vlm-tamp', 'vlm_tamp_no_images': 'vlm-tamp-no-image',
           'tru_pomdp_no_images': 'tru-pomdp'}
FINISHED = {'completed', 'timeout', 'failed', 'cancelled'}


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def catalog(source, overrides=None):
    """Combined batch summary or portable {runs:[baseline,path,episode_id,repetition,...]}."""
    source = Path(source).resolve()
    if source.is_dir():
        source /= 'summary.json'
    data = load(source)
    rows = data.get('runs', data.get('jobs', []))
    if not rows:
        raise ValueError('Source needs a nonempty runs or jobs array')
    runs, seen = [], set()
    for row in rows:
        baseline = row.get('baseline', row.get('planner', row.get('condition')))
        baseline = ALIASES.get(baseline, baseline)
        if baseline not in BASELINES:
            raise ValueError(f'Unknown baseline {baseline!r}; use one of {BASELINES}')
        episode = row['episode_id'].upper()
        repetition = int(row['repetition'])  # Never infer from _r001 in per-repeat jobs.
        if repetition < 1 or not re.fullmatch(r'T\d+-[A-Z]+-[A-Z]+', episode):
            raise ValueError('Invalid variant or repetition')
        rid = f'{baseline}/{episode}/repeat_{repetition}'
        if rid in seen:
            raise ValueError(f'Duplicate run identity: {rid}')
        seen.add(rid)
        path = row.get('path', row.get('output'))
        if path:
            path = Path(path)
            path = str((source.parent / path).resolve() if not path.is_absolute() else path.resolve())
        runs.append({'id': rid, 'baseline': baseline, 'episode_id': episode,
                     'repetition': repetition, 'path': path,
                     'status': row.get('status', 'unknown'), 'metrics': row.get('metrics') or {},
                     'source_summary': str(source), 'run_id': row.get('run_id', rid),
                     'elapsed_seconds': row.get('elapsed_seconds')})
    if overrides is not None:
        replacements = catalog(overrides)
        by_id = {r['id']: r for r in runs}
        for replacement in replacements:
            rid = replacement['id']
            if rid not in by_id:
                raise ValueError(f'Replacement is not in the source catalog: {rid}')
            if replacement['status'] not in FINISHED or not replacement['path']:
                raise ValueError(f'Replacement must be finished with an output path: {rid}')
            if not Path(replacement['path']).is_dir():
                raise ValueError(f'Replacement output folder is missing: {replacement["path"]}')
            replacement['supersedes'] = {k: by_id[rid][k] for k in ('path', 'run_id', 'source_summary')}
            by_id[rid] = replacement
        runs = [by_id[r['id']] for r in runs]
    return runs


def expanded(props):
    """Match simulator quantity expansion for aggregate-only fallback; no new constraints."""
    result = copy.deepcopy(props)
    for p in props:
        number = p.get('args', {}).get('number', 1)
        if isinstance(number, int):
            for n in range(1, number):
                q = copy.deepcopy(p)
                q['args']['number'] = n
                result.append(q)
    return result


def simulator(episode, records, metrics):
    props = expanded(episode.get('evaluation_propositions', []))
    statuses, source, method = None, None, 'unavailable'
    terminal = next((r for r in reversed(records) if r.get('event') == 'terminal'), None)
    constraints = episode.get('evaluation_constraints', [])
    if terminal:
        state = terminal.get('evaluator_only_state', {})
        tracker = state.get('auto_eval_proposition_tracker', state.get('task_evaluation_log', {}))
        props = tracker.get('propositions', props)
        constraints = tracker.get('constraints', constraints)
        statuses = structured_results(state, len(props))
        metrics = dict(metrics)
        metrics.update({k: state[k] for k in ['task_state_success', 'task_percent_complete', 'task_explanation'] if k in state})
        source = terminal['_id']
        if statuses is not None:
            method = 'terminal_simulator_snapshot'
    groups = []
    # Use aggregate final scores only; do not turn intermediate snapshots into scores.
    percent = metrics.get('task_percent_complete')
    success = metrics.get('task_state_success')
    if statuses is None:
        if success == 1 or percent == 1:
            statuses = ['met'] * len(props)
            method = 'final_aggregate_all_met'
        elif percent == 0:
            statuses = ['unmet'] * len(props)
            method = 'final_aggregate_none_met'
        else:
            statuses = ['unknown'] * len(props)
            if isinstance(percent, (int, float)) and 0 < percent < 1:
                count = len(props) * (1 - percent)
                if abs(count - round(count)) < 1e-5:
                    groups = [{'id': 'g0', 'criterion_ids': [f'p{i}' for i in range(len(props))], 'unmet_count': round(count)}]
                    method = 'final_aggregate_ambiguous_group'
    if method == 'terminal_simulator_snapshot':
        outcome = 'complete' if all(s == 'met' for s in statuses) else 'no_progress' if all(s == 'unmet' for s in statuses) else 'partial'
    else:
        outcome = 'complete' if success == 1 or percent == 1 else 'no_progress' if percent == 0 else 'partial' if isinstance(percent, (int, float)) and 0 < percent < 1 else 'unavailable'
    return {'outcome': outcome, 'metrics': {k: metrics[k] for k in ['task_state_success', 'task_percent_complete', 'task_explanation'] if k in metrics},
            'criteria': [{'id': f'p{i}', 'index': i, 'status': statuses[i], 'proposition': p} for i, p in enumerate(props)],
            'constraints': constraints, 'ambiguous_failure_groups': groups,
            'source_event_id': source, 'recovery_method': method}


def action_attempts(events):
    attempts, current = [], {}
    for e in events:
        observed = e['data'].get('planner_observed', {})
        for agent, action in (observed.get('high_level_actions') or {}).items():
            if not isinstance(action, (list, tuple)) or not action:
                continue
            response = (observed.get('responses') or {}).get(agent, '') or ''
            thought = (observed.get('thought') or {}).get(agent)
            old = current.get(agent)
            signature = action[:2]
            replanned = (observed.get('replanned') or {}).get(agent, False)
            new = old is None or old['_signature'] != signature
            if old and e['data'].get('event') != 'terminal':
                new |= old['_ended'] and (not response or bool(replanned))
                new |= bool(replanned and thought and thought != old['_thought'])
            if new:
                old = {'id': f'a{len(attempts):04d}', 'agent': agent, 'action': str(action[0]),
                       'arguments': action[1] if len(action) > 1 else None,
                       'evidence_ids': [], 'responses': [], 'start_order': e['order'],
                       'boundary_source': 'inferred_from_action_response_and_replan',
                       '_signature': signature, '_ended': False, '_thought': thought}
                attempts.append(old)
                current[agent] = old
            old['evidence_ids'].append(e['id'])
            if response and (not old['responses'] or old['responses'][-1] != response):
                old['responses'].append(response)
            old['_ended'] = bool(response)
            old['_thought'] = thought
    return [{k: v for k, v in a.items() if not k.startswith('_')} for a in attempts]


def checkpoint_evaluation(episode, checkpoints, process_end=None):
    """A separate scored snapshot; never changes final failure eligibility."""
    candidates = []
    for record in checkpoints:
        tracker = record.get('evaluator_only_state', {}).get('auto_eval_proposition_tracker', {})
        advanced = (record.get('sim_step_count') or 0) > 0 or tracker.get('state_sequence_length', 0) > 1
        if not advanced:
            continue
        scored = simulator(episode, [dict(record, event='terminal')], {})
        if scored['recovery_method'] != 'terminal_simulator_snapshot':
            continue
        candidates.append((record, scored))
    if not candidates:
        return None
    record, scored = max(candidates, key=lambda pair: (pair[0].get('wall_time') or 0, pair[0].get('sim_step_count') or 0))
    scored.update(recovery_method='checkpoint_simulator_snapshot', score_basis='last_saved_state',
                  eligible_for_failure_labels=False,
                  recorded_at=record.get('wall_time'), elapsed_seconds=record.get('elapsed_seconds'),
                  sim_step_count=record.get('sim_step_count'),
                  seconds_before_process_end=(max(0, process_end - record['wall_time'])
                      if process_end is not None and record.get('wall_time') is not None else None))
    return scored


def checkpoint_delta(before, after, path=()):
    """Lossless JSON changes; arrays of scalars are replaced as a unit."""
    assignments, removals = [], []
    if before == after:
        return assignments, removals
    if isinstance(before, dict) and isinstance(after, dict):
        removals.extend([*path, key] for key in before if key not in after)
        for key, value in after.items():
            if key not in before:
                assignments.append({'path': [*path, key], 'value': value})
            else:
                updates, deleted = checkpoint_delta(before[key], value, (*path, key))
                assignments.extend(updates)
                removals.extend(deleted)
    elif (isinstance(before, list) and isinstance(after, list) and len(before) == len(after)
          and any(isinstance(v, (dict, list)) for v in before + after)):
        for index, (old, new) in enumerate(zip(before, after)):
            updates, deleted = checkpoint_delta(old, new, (*path, index))
            assignments.extend(updates)
            removals.extend(deleted)
    else:
        assignments.append({'path': list(path), 'value': after})
    return assignments, removals


def share_json_blocks(events):
    """Losslessly intern repeated structured states, graphs and tool payloads."""
    counts, keys = Counter(), {}
    def scan(value):
        if isinstance(value, (dict, list)):
            text = json.dumps(value, sort_keys=True, separators=(',', ':'))
            if len(text) >= 100:
                key = hashlib.sha256(text.encode()).hexdigest()
                keys[id(value)] = key
                counts[key] += 1
            for child in value.values() if isinstance(value, dict) else value:
                scan(child)
    for e in events:
        scan(e['data'])
    blocks, names = {}, {}
    def encode(value):
        key = keys.get(id(value))
        if key and counts[key] > 1:
            if key not in names:
                name = f'j{len(names)}'
                names[key] = name
                blocks[name] = children(value)
            return {'json_ref': names[key]}
        return children(value)
    def children(value):
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [encode(v) for v in value]
        return value
    return [dict(e, data=encode(e['data'])) for e in events], blocks


def packet_for(run):
    if run['status'] not in FINISHED or not run['path']:
        raise ValueError('Run is not finished or lacks an output folder')
    root = Path(run['path'])
    files, problems, records, events = {}, [], [], []

    def read(path):
        raw = path.read_bytes()
        files[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
        return raw

    dataset = root / 'input/dataset.json.gz'
    if dataset.exists():
        episode = json.loads(gzip.decompress(read(dataset)))['episodes'][0]
    else:
        episode = json.loads(read(root / 'input/dataset.json'))['episodes'][0]
    references = {}
    for name in ['scene_info.json', 'initial_robot_memory.json', 'spec.md']:
        f = root / 'input' / name
        if f.exists():
            raw = read(f).decode(errors='replace')
            references[name] = json.loads(raw) if name.endswith('.json') else raw
    evidence_paths = sorted((root / 'dataset').glob('evidence-*.jsonl'))
    for file_index, f in enumerate(evidence_paths):
        for line_number, line in enumerate(read(f).decode(errors='replace').splitlines(), 1):
            try:
                r = json.loads(line)
            except ValueError:
                problems.append(f'{f.name}: invalid JSON at line {line_number}')
                continue
            eid = f'e{file_index}_{r.get("sequence", line_number)}'
            if any(e['id'] == eid for e in events):
                raise ValueError('Duplicate evidence sequence')
            r['_id'] = eid
            records.append(r)
            d = copy.deepcopy(r)
            d.pop('_id')
            state = d.get('evaluator_only_state', {})
            tracker = state.get('auto_eval_proposition_tracker', {})
            history = tracker.pop('state_sequence', [])
            # Preserve every distinct proposition-state transition, with its tick.
            changes, last = [], None
            for tick, value in enumerate(history):
                if value != last:
                    changes.append({'tick': tick, 'state': value})
                    last = value
            if history:
                tracker['state_changes'] = changes
                tracker['state_sequence_length'] = len(history)
            events.append({'id': eid, 'kind': 'execution', 'order': len(events),
                           'source': f.relative_to(root).as_posix(), 'locator': f'line {line_number}', 'data': d})
    sim = simulator(episode, records, run['metrics'])
    attempts = action_attempts(events)
    # These are simulator observations, not actions or terminal records. Their
    # timestamps/steps establish state chronology without inventing attempts.
    checkpoints = []
    previous_checkpoint = None
    checkpoint_files = [root / 'dataset/evaluation-checkpoints.jsonl',
                        root / 'dataset/evaluation-checkpoint.json']
    for file_index, f in enumerate(checkpoint_files):
        if not f.exists():
            continue
        for line_number, line in enumerate(read(f).decode(errors='replace').splitlines(), 1):
            try:
                r = json.loads(line)
                if not isinstance(r, dict) or r.get('event') != 'checkpoint':
                    raise ValueError('Not a checkpoint')
            except ValueError:
                problems.append(f'{f.name}: invalid checkpoint at line {line_number}')
                continue
            eid = f'checkpoint{file_index}_{line_number}'
            r['_id'] = eid
            checkpoints.append(r)
            data = {k: v for k, v in r.items() if k != '_id'}
            # Full anchors bound reconstruction to 49 deltas. Scoring above/below
            # still reads original snapshots, and hashes cover unmodified files.
            if previous_checkpoint is not None and (len(checkpoints) - 1) % 50 != 0:
                assignments, removals = checkpoint_delta(
                    previous_checkpoint.get('evaluator_only_state', {}), r.get('evaluator_only_state', {}))
                encoded = {'checkpoint_delta': {'base_event_id': previous_checkpoint['_id'],
                                               'set': assignments, 'remove': removals}}
                if len(json.dumps(encoded)) < len(json.dumps(r.get('evaluator_only_state', {}))):
                    data['evaluator_only_state'] = encoded
            events.append({'id': eid, 'kind': 'evaluator_checkpoint', 'order': None,
                           'source': f.relative_to(root).as_posix(), 'locator': f'line {line_number}',
                           'data': data})
            previous_checkpoint = r
    process_end = None
    run_metadata = root / 'run.json'
    if checkpoints and run_metadata.exists():
        try:
            metadata = json.loads(read(run_metadata))
            if metadata.get('started') is not None and metadata.get('elapsed_seconds') is not None:
                process_end = metadata['started'] + metadata['elapsed_seconds']
        except (ValueError, TypeError):
            problems.append('run.json: invalid process timing')
    checkpoint = checkpoint_evaluation(episode, checkpoints, process_end)
    # Include planner-native history as supporting evidence, not additional attempts.
    for f in sorted((root / 'dataset/planner-log').glob('*.json')):
        try:
            steps = json.loads(read(f)).get('steps', [])
        except ValueError:
            problems.append(f'{f.name}: invalid planner log')
            continue
        previous = None
        for i, step in enumerate(steps):
            d = compact(step)
            for key in ['total_step_count', 'sim_step_count', 'log_index', 'agent_positions', 'agent_states', 'agent_collisions']:
                d.pop(key, None)
            signature = sha(d)
            if signature == previous:
                continue
            previous = signature
            if not records:
                events.append({'id': f'legacy{len(events)}', 'kind': 'execution', 'order': len(events),
                               'source': f.relative_to(root).as_posix(), 'locator': f'steps[{i}]',
                               'data': {'event': 'boundary', 'planner_observed': d}})
            events.append({'id': f'planner{len(events)}', 'kind': 'planner_context', 'order': None,
                           'source': f.relative_to(root).as_posix(), 'locator': f'steps[{i}]', 'data': d})
    if not records:
        attempts = action_attempts([e for e in events if e['kind'] == 'execution'])
    for f in [p for d in ('vlm_tamp_pddl', 'rail_planner') for p in sorted(root.glob(f'{d}/*log.jsonl'))]:
        for i, line in enumerate(read(f).decode(errors='replace').splitlines(), 1):
            try:
                try:
                    d = json.loads(line)
                except ValueError:
                    d = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                d = {'raw_record': line}
            events.append({'id': f'pddl{len(events)}', 'kind': 'planner_context', 'order': None,
                           'source': f.relative_to(root).as_posix(), 'locator': f'line {i}', 'data': compact(d)})
    for f in sorted((root / 'dataset/prompts').glob('**/*.txt')):
        events.append({'id': f'prompt{len(events)}', 'kind': 'saved_prompt', 'order': None,
                       'source': f.relative_to(root).as_posix(), 'locator': 'full file', 'data': read(f).decode(errors='replace')})
    for f in sorted((root / 'dataset/traces').glob('**/*.txt')):
        events.append({'id': f'trace{len(events)}', 'kind': 'trace', 'order': None,
                       'source': f.relative_to(root).as_posix(), 'locator': 'full file', 'data': read(f).decode(errors='replace')})
    # Fold only identical consecutive console diagnostics, preserving source lines.
    console = root / 'console.log'
    if console.exists():
        folded = []
        for i, line in enumerate(read(console).decode(errors='replace').splitlines(), 1):
            line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line)
            if folded and line == folded[-1]['text']:
                folded[-1]['repeat_count'] += 1
                folded[-1]['last_line'] = i
            else:
                folded.append({'text': line, 'first_line': i, 'last_line': i, 'repeat_count': 1})
        for i in range(0, len(folded), 80):
            block = folded[i:i+80]
            events.append({'id': f'console{len(events)}', 'kind': 'diagnostics_not_attempts', 'order': None,
                           'source': 'console.log', 'locator': f'lines {block[0]["first_line"]}-{block[-1]["last_line"]}', 'data': block})
    terminal = any(r.get('event') == 'terminal' for r in records)
    start = any(r.get('event') == 'start' for r in records)
    if not records:
        problems.append('No incremental action evidence; attempt-level coverage unavailable')
    coverage = {'history_complete': start and terminal and not problems and len(evidence_paths) == 1,
                'has_terminal_snapshot': terminal, 'problems': problems,
                'execution_status': run['status'], 'omitted_substantive_events': False,
                'removed_diagnostics': ['unchanged planner ticks', 'per-tick agent_positions/agent_states/agent_collisions; boundary simulator graph retained'],
                'action_boundary_note': 'Attempts inferred from actions, responses and replanning; supplemental logs are not new attempts.'}
    coverage.update(checkpoint_count=len(checkpoints), has_checkpoint_evaluation=checkpoint is not None,
                    checkpoint_policy='Checkpoints support state observations and notes, not final failure labels. Initial-only states are not scored.')
    packed, blocks = share_text_blocks(events)
    packed, json_blocks = share_json_blocks(packed)
    return {'schema_version': 2, 'instruction': episode.get('instruction', ''),
            'simulator': sim, 'checkpoint_evaluation': checkpoint, 'reference_truth_not_robot_knowledge': references,
            'events': packed, 'attempts': attempts, 'text_blocks': blocks, 'json_blocks': json_blocks,
            'coverage': coverage, 'image_inputs': [],
            'provenance': {'run_id': run['id'], 'files_sha256': files, 'evidence_sha256': sha(files),
                           'run_path': str(root), 'source_summary': run.get('source_summary'),
                           'supersedes': run.get('supersedes')}}
