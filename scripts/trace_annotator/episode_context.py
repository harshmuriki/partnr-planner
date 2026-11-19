"""Scenario-viewer context with archived-spec and current-reference provenance."""
import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

VIEWER_ROOT = Path(__file__).resolve().parents[2] / 'baseline_evaluation_v3'
MEMORY = {'ACC': 'Accurate', 'INC': 'Incomplete', 'OUT': 'Outdated'}


def section(markdown, heading):
    match = re.search(r'^## ' + re.escape(heading) + r'\s*\n(.*?)(?=^## |\Z)', markdown, re.M | re.S)
    return match.group(1).strip() if match else ''


def items(markdown, heading):
    return [line[2:].strip() for line in section(markdown, heading).splitlines() if line.startswith('- ')]


def spec_fields(markdown):
    return {'instruction': section(markdown, 'Task instruction / prompt given').strip('"'),
            'success_criteria': items(markdown, 'Success criteria'),
            'initial_robot_memory': items(markdown, 'Initial robot memory')}


def episode_context(packet, viewer_root=None):
    root = Path(viewer_root) if viewer_root is not None else VIEWER_ROOT
    refs = packet.get('reference_truth_not_robot_knowledge', {})
    archived = refs.get('spec.md', '')
    match = re.search(r'\b(T\d+)-(ACC|INC|OUT)-([A-Z]+)\b', packet.get('provenance', {}).get('run_id', '') or archived)
    eid = match.group(0) if match else None
    source_hashes = {}; warnings = []
    def read(relative):
        path = root / relative
        if not path.is_file():
            return None
        raw = path.read_bytes()
        source_hashes[relative] = hashlib.sha256(raw).hexdigest()
        return raw.decode('utf-8-sig')
    current = read(f'specs/{match.group(1)}/{eid}.md') if match else None
    if archived:
        spec = archived; spec_source = 'archived_run_input/spec.md'
        spec_hash = hashlib.sha256(archived.encode()).hexdigest()
    elif current:
        spec = current; spec_source = 'current_scenario_viewer_spec_fallback'
        spec_hash = hashlib.sha256(current.encode()).hexdigest()
        warnings.append('Archived spec unavailable; current viewer spec is reference context, not verified historical run input.')
    else:
        spec = ''; spec_source = None; spec_hash = None
        warnings.append('Scenario spec unavailable; written success criteria and declared memory remain unavailable.')
    fields = spec_fields(spec)
    matches_current = spec_fields(archived) == spec_fields(current) if archived and current else None
    if matches_current is False:
        warnings.append('Current viewer and archived scenario summary differ. Use the archived run instruction and simulator criteria for this run.')
    if fields['instruction'] and fields['instruction'] != packet['instruction']:
        warnings.append('Spec instruction differs from the executed dataset instruction; executed instruction is authoritative.')
    objects = None; counts = None
    raw_csv = read('variant_object_assets.csv') if eid else None
    if raw_csv:
        rows = [r for r in csv.DictReader(io.StringIO(raw_csv)) if r.get('Variant_ID', '').strip() == eid
                and re.fullmatch(r'\S+\s*\([^)]+\)', r.get('Object', '').strip())]
        if rows:
            objects = []
            for row in rows:
                states = []
                for family in ['Clean_Dirty', 'Filled_Empty', 'Powered_On_Off']:
                    start = row.get('Start_' + family, '').strip(); end = row.get('End_' + family, '').strip()
                    if (start and start != 'n/a') or (end and end != 'n/a'):
                        states.append({'state_family': family, 'start': start, 'expected_end': end})
                objects.append({'object': row['Object'].strip(), 'label': row.get('Label', '').strip().lower(), 'states': states})
            counts = {'object_count': len(objects),
                      'objects_with_expected_state_change': sum(any(s['start'] != s['expected_end'] for s in o['states']) for o in objects),
                      'label_counts': dict(Counter(o['label'] for o in objects))}
    replay = None
    raw_timing = read('ground_truth_timing/replay_measurements.json') if eid else None
    if raw_timing:
        measured = json.loads(raw_timing).get(eid)
        if measured is not None:
            replay = {k: measured.get(k) for k in ['status', 'execution_elapsed_seconds', 'sim_steps', 'action_count', 'source']}
            replay['measurement_kind'] = 'human_ground_truth_skill_replay'
            replay['includes_model_calls'] = False
            replay['includes_scene_loading'] = False
            replay['is_this_baseline_run'] = False
    return {'episode_id': eid,
            'memory_condition': {'code': match.group(2), 'label': MEMORY[match.group(2)]} if match else None,
            'availability_condition': match.group(3) if match else None,
            'instruction': packet['instruction'], 'spec_instruction': fields['instruction'] or None,
            'success_criteria': fields['success_criteria'] if spec else None,
            'initial_robot_memory': fields['initial_robot_memory'] if spec else None,
            'archived_memory_metadata': refs.get('initial_robot_memory.json'),
            'scenario_source': {'kind': spec_source, 'sha256': spec_hash, 'matches_current_viewer_summary': matches_current},
            'viewer_reference': {'scope': 'current scenario-viewer reference data; not baseline execution results',
                                 'object_summary': counts, 'objects': objects, 'ground_truth_replay': replay,
                                 'source_files_sha256': source_hashes},
            'interpretation': ['Written success criteria include behavioral requirements that may not be simulator-scored. Final scores remain exclusively in simulator.',
                               'Initial robot memory here is the declared scenario memory. Actual prompts and logged observations establish what information the planner received; do not assume every spec statement reached it.',
                               'Object state-change counts describe expected start-to-goal changes, not observed changes during this run.',
                               'Spec and runtime aliases may differ; resolve identity through archived handle mappings.'],
            'warnings': warnings}
