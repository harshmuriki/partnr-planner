"""Recover last saved evaluator states without claiming they are final scores."""
import gzip
import json
from pathlib import Path

from dashboard_subtasks import subtask_record
from judge_evidence import simulator
from sim_results import structured_results


def reverse_records(path, block_size=1024*1024):
    """Yield valid JSON records newest first, tolerating a truncated last write."""
    with Path(path).open('rb') as stream:
        position = stream.seek(0, 2)
        pending = b''
        while position:
            size = min(block_size, position)
            position -= size
            stream.seek(position)
            data = stream.read(size) + pending
            lines = data.split(b'\n')
            pending = lines[0]
            offset = position + len(pending) + 1
            entries = []
            for line in lines[1:]:
                entries.append((offset, line))
                offset += len(line) + 1
            if position == 0:
                entries.insert(0, (0, pending))
            for offset, line in reversed(entries):
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                record['_id'] = f'{Path(path).name}:byte:{offset}'
                yield record


def progressed(record):
    state = record.get('evaluator_only_state', {})
    tracker = state.get('auto_eval_proposition_tracker') or state.get('task_evaluation_log') or {}
    length = tracker.get('state_sequence_length', len(tracker.get('state_sequence', [])))
    return (record.get('sim_step_count') or 0) > 0 or (record.get('total_step_count') or 0) > 0 or length > 1


def find_checkpoint(root):
    dataset = Path(root) / 'dataset'
    candidates = []
    latest = dataset / 'evaluation-checkpoint.json'
    paths = ([latest] if latest.exists() else []) + sorted(dataset.glob('evidence-*.jsonl'))
    for path in paths:
        for record in reverse_records(path):
            state = record.get('evaluator_only_state', {})
            tracker = state.get('auto_eval_proposition_tracker') or state.get('task_evaluation_log') or {}
            times = tracker.get('proposition_satisfied_at')
            if (record.get('event') in ('boundary', 'checkpoint', 'terminal') and progressed(record)
                    and isinstance(times, list) and structured_results(state, len(times)) is not None):
                candidates.append((record, str(path)))
                break
        # The dedicated atomic checkpoint is written on every step and contains
        # the latest state. Avoid re-reading a historical multi-GB boundary log.
        if candidates and path == latest:
            break
    return max(candidates, key=lambda x: x[0].get('wall_time', 0)) if candidates else (None, None)


def recover_checkpoint(row):
    if row.get('completion') is not None:
        return None  # A final score, including ambiguous attribution, wins.
    root = Path(row['run_folder'])
    record, source = find_checkpoint(root)
    if record is None:
        return None
    ep = json.loads(gzip.decompress((root/'input/dataset.json.gz').read_bytes()))['episodes'][0]
    scene_path = root/'input/scene_info.json'
    scene = json.loads(scene_path.read_text()) if scene_path.exists() else {}
    # Reuse the pure scoring reader; the returned data is explicitly relabeled
    # checkpoint-only below and never written into final aggregate run metrics.
    scoring = dict(record, event='terminal')
    sim = simulator(ep, [scoring], {})
    result = subtask_record(ep, scene, sim, source)
    result['method'] = 'last_saved_evaluator_state'
    result['score_basis'] = 'last_saved_state'
    result['checkpoint'] = {k: record.get(k) for k in ('wall_time','elapsed_seconds','sim_step_count','total_step_count')}
    result['checkpoint']['event'] = record.get('event')
    metadata_path = root / 'run.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    started = metadata.get('started')
    result['checkpoint']['seconds_before_process_end'] = (
        max(0, started + row['runtime_s'] - record['wall_time'])
        if started is not None and row.get('runtime_s') is not None and record.get('wall_time') is not None else None)
    return result
