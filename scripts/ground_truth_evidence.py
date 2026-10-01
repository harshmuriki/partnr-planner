"""Explain saved evaluator results using the recorded spec, identities, and action timing."""
import re

PREDICATE = re.compile(r'(is_\w+)\(([^)]+)\)')
ORDER = re.compile(r'order: (is_\w+\([^)]+\)) before (is_\w+\([^)]+\))')


def normalized(text):
    return re.sub(r'\s+', '', text)


def criteria_evidence(run, frame_times=None):
    """Read-only evidence; never infer task success from a skill's success response."""
    meta = run.get('metadata', {})
    snapshot = run.get('snapshot', {})
    evaluation = snapshot.get('evaluation') or {}
    actions = run.get('actions', [])
    lines = [line.strip().removeprefix('- ').strip()
             for line in meta.get('success_criteria', '').splitlines() if line.strip()]
    aliases = snapshot.get('aliases', {})
    entities = snapshot.get('entities', {})
    names = {}
    for entry in entities.get('objects', []) + entities.get('furniture', []):
        runtime = entry['name']
        choices = [name for name, target in aliases.items() if target == runtime]
        name = next((name for name in choices if re.search(r'\b' + re.escape(name) + r'\b', '\n'.join(lines))),
                    choices[0] if choices else runtime)
        names[entry.get('sim_handle')] = name
    orders = [match.groups() for line in lines if (match := ORDER.fullmatch(line))]
    transient = {normalized(before) for before, _ in orders}

    # Tracker state 0 is the initial scene. Each counted env.step adds one state;
    # a skill covering cumulative steps (start, end] owns timestamps in that range.
    spans, elapsed, frame = [], 0, 0
    for index, action in enumerate(actions):
        result = action.get('result') or {}
        steps = result.get('skill_steps')
        frames = result.get('recorded_frames', 0 if steps == 0 else None)
        valid = type(steps) is int and steps >= 0
        if result.get('ok') is False and result.get('steps_source') != 'environment_step_calls':
            valid = False  # Legacy exception paths did not count all environment calls.
        end = elapsed + steps if elapsed is not None and valid else None
        spans.append((elapsed, end, frame, frames))
        elapsed = end
        frame = frame + frames if frame is not None and type(frames) is int and frames >= 0 else None

    def action_link(index, timestamp=None):
        action = actions[index]
        start, end, first_frame, frames = spans[index]
        video_frame = first_frame
        if timestamp is not None and video_frame is not None and frames == end - start:
            video_frame += timestamp - start - 1
        time = video_frame / 30 if video_frame is not None else None
        if frame_times is not None:
            time = frame_times[video_frame] if video_frame is not None and 0 <= video_frame < len(frame_times) else None
        return {'action_index': index, 'sequence': action.get('sequence', index + 1),
                'skill': action['skill'], 'target': action['target'], 'video_time': time,
                'simulator_step': timestamp}

    def attribution(timestamp):
        if timestamp == 0:
            return [], 'Already satisfied in the initial scene; no action was needed.'
        if type(timestamp) is not int or timestamp < 0:
            return [], 'Not satisfied in the recorded run.'
        for index, (start, end, _, _) in enumerate(spans):
            if start is not None and end is not None and start < timestamp <= end:
                return [action_link(index, timestamp)], 'First satisfied during the recorded step below.'
        return [], 'The evaluator recorded success, but incomplete action timing prevents exact step attribution.'

    timestamps = evaluation.get('satisfied_at', [])
    constraints = evaluation.get('constraint_satisfaction')
    current = evaluation.get('current', [])
    props = meta.get('propositions', [])
    by_label = {}
    for index, prop in enumerate(props):
        args = prop.get('args', {})
        keys = ('entity_handles_a', 'entity_handles_b') if prop['function_name'] == 'is_next_to' else ('object_handles', 'receptacle_handles')
        operands = [names.get(handle, handle) for key in keys for handle in args.get(key, [])]
        label = f"{prop['function_name']}({', '.join(operands)})"
        stamp = timestamps[index] if index < len(timestamps) else None
        status = 'unknown'
        shape_ok = len(timestamps) == len(props) and args.get('number', 1) == 1
        if shape_ok and type(stamp) is int:
            if stamp < 0:
                status = 'fail'
            elif constraints is not None and all(index < len(row) for row in constraints):
                status = 'pass' if all(row[index] for row in constraints) else 'fail'
        final = current[index] if shape_ok and index < len(current) else None
        is_transient = normalized(label) in transient
        if final is False and not is_transient:
            status = 'fail'
        links, detail = attribution(stamp) if shape_ok else ([], 'Saved evaluation data cannot be matched to this predicate.')
        if stamp is not None and stamp >= 0:
            if status == 'fail':
                detail += ' It did not satisfy the final-state or ordering checks.'
            elif final is True:
                detail += ' Still satisfied at the end.'
            elif is_transient and final is False:
                detail += ' This is an intermediate requirement; it need not remain true at the end.'
        by_label[normalized(label)] = {'label': label, 'status': status, 'evidence': links,
                                       'detail': detail, 'first_satisfied_at': stamp,
                                       'satisfied_at_end': final, 'proposition_index': index}

    rows = []
    if not lines:
        lines = [row['label'] for row in by_label.values()]
    for line in lines:
        key = normalized(line)
        if PREDICATE.fullmatch(line):
            row = by_label.get(key, {'status': 'unknown', 'evidence': [],
                                    'detail': 'No matching predicate was found in this saved recording.'})
            rows.append({**row, 'label': line})
        elif match := ORDER.fullmatch(line):
            pair = [by_label.get(normalized(label)) for label in match.groups()]
            status = 'unknown'
            if all(row and row['status'] != 'unknown' for row in pair):
                status = 'pass' if all(row['status'] == 'pass' for row in pair) and pair[0]['first_satisfied_at'] < pair[1]['first_satisfied_at'] else 'fail'
            rows.append({'label': line, 'status': status,
                         'evidence': [link for row in pair if row for link in row['evidence']],
                         'detail': 'The first requirement must be satisfied before the second. Links follow that order.'})
        elif '- ' + line in meta.get('reports', []) or line in meta.get('reports', []):
            indices = [i for i, action in enumerate(actions) if action['skill'] == 'ReportAbsence'
                       and action['target'].removeprefix('- ').strip() == line
                       and (action.get('result') or {}).get('ok') is True]
            rows.append({'label': line, 'status': 'pass' if indices else 'fail',
                         'evidence': [action_link(i) for i in indices],
                         'detail': 'Explicit absence report in the saved action list.' if indices else 'The required absence report is missing.'})
        elif line.startswith(('Using ', 'Distractors (')):
            # These compiler notes are enforced by the exact object handles in the predicates.
            matched = list(by_label.values())
            status = ('fail' if any(row['status'] == 'fail' for row in matched) else
                      'pass' if matched and all(row['status'] == 'pass' for row in matched) else 'unknown')
            links = {link['action_index']: link for row in matched for link in row['evidence']}
            rows.append({'label': line, 'status': status, 'evidence': list(links.values()),
                         'detail': 'Checked by the exact target-object predicates above; this has no separate completion action.'})
        else:
            rows.append({'label': line, 'status': 'unknown', 'evidence': [],
                         'detail': 'This requirement has no recorded automatic evaluation.'})
    # Do not hide compiled checks just because an old recorded spec omitted their text.
    represented = {normalized(row['label']) for row in rows}
    rows.extend(row for key, row in by_label.items() if key not in represented)
    met = sum(row['status'] == 'pass' for row in rows)
    status = ('fail' if evaluation.get('success') is False or any(row['status'] == 'fail' for row in rows) else
              'pass' if rows and met == len(rows) and evaluation.get('success') is True else 'unknown')
    return {'status': 'outdated' if run.get('stale') else status, 'recorded_status': status,
            'met': met, 'total': len(rows), 'criteria': rows,
            'basis': 'Saved evaluator results, including final-state and ordering constraints. Step links mark the first time a requirement became true, not an inferred cause.'}
