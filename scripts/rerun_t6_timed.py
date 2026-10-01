#!/usr/bin/env python3
"""Replay backed-up, authorized T6 sequences through one viewer sandbox.

Preserves exact skill/target strings. Stops on any failure, leaving that sandbox
open for diagnosis. Run only after preparing queue.json and the before/ backup.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'baseline_evaluation_v3'
OUT = BASE / 'ground_truth_timing/single_sandbox_20260930'
URL = 'http://127.0.0.1:8000/baseline_evaluation_v3/api/ground-truth/'


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def resources():
    memory = {line.split(':')[0]: line.split(':')[1].strip()
              for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemAvailable:', 'SwapFree:'))}
    return {'utc': utc(), 'load_average': os.getloadavg(), **memory}


def log(message):
    print(utc(), message, flush=True)


def api(variant, operation='', payload=None):
    request = urllib.request.Request(URL + variant + ('/' + operation if operation else ''),
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def idle(variant, sid):
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        state = api(variant)
        session = state['session']
        if state['max_sessions'] != 1 or len(state['sessions']) != 1 or session['id'] != sid:
            raise RuntimeError('Single sandbox/session invariant changed')
        if not session['busy']:
            if session['error']:
                raise RuntimeError(session['error'])
            return state
        time.sleep(0.2)
    raise RuntimeError('Operation timed out; preserving sandbox')


def publish_timing(item):
    path = BASE / 'ground_truth_timing/current_recordings.csv'
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = list(reader)
    extra = ['action_wall_seconds', 'worker_command_wall_seconds', 'load_wall_seconds', 'save_wall_seconds']
    fields += [key for key in extra if key not in fields]
    row = next(row for row in rows if row['variant'] == item['variant'])
    row.update(run_id=item['run_id'], timing_status='measured_single_sandbox',
        execution_elapsed_seconds=f"{item['execution_elapsed_seconds']:.6f}",
        execution_elapsed_minutes=f"{item['execution_elapsed_seconds']/60:.6f}",
        interval_definition='first_action_request_to_final_success_observed',
        attempt_id='', source=f"single_sandbox_20260930/{item['variant']}.json")
    row.update({key: f'{item[key]:.6f}' for key in extra})
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    temp = path.with_suffix('.csv.tmp'); temp.write_text(text.getvalue()); temp.replace(path)


def main():
    plans = json.loads((OUT / 'queue.json').read_text())
    status_path = OUT / 'status.json'
    results = json.loads(status_path.read_text()) if status_path.exists() else []
    done = {item['variant'] for item in results if item['status'] == 'saved'}
    for plan in plans:
        variant = plan['variant']
        if variant in done:
            continue
        item = {'variant': variant, 'status': 'running', 'started_at': utc(),
                'old_run_id': plan['old_run_id'], 'actions': [], 'resources_start': resources(),
                'max_sandboxes': 1, 'poll_interval_seconds': 0.2}
        results.append(item)
        def checkpoint():
            write(OUT / f'{variant}.json', item)
            write(status_path, results)
        checkpoint()
        try:
            state = api(variant)
            if state['sessions'] or state['max_sessions'] != 1:
                raise RuntimeError('Expected an empty one-slot server')
            old = state['ground_truth']
            if old['id'] != plan['old_run_id']:
                raise RuntimeError('Saved recording changed since backup')
            if any(state['meta'][key] != plan[key] for key in ['spec_hash','dataset_hash']):
                raise RuntimeError('Spec/dataset changed since backup')
            log(f"START {variant}: {len(plan['steps'])} unchanged actions")
            load_start = time.perf_counter()
            sid = api(variant, 'sandbox', {})['session_id']
            state = idle(variant, sid)
            item['load_wall_seconds'] = time.perf_counter() - load_start
            item['session_id'] = sid
            item['execution_started_at'] = utc()
            start = time.perf_counter()
            for number, step in enumerate(plan['steps'], 1):
                rid = uuid.uuid4().hex
                action = {'sequence': number, **step, 'submitted_at': utc()}
                before = time.perf_counter()
                api(variant, 'action', {'session_id': sid, 'request_id': rid, **step})
                state = idle(variant, sid)
                observed_end = time.perf_counter()
                action.update(observed_wall_seconds=observed_end-before, completed_at=utc())
                actual = state['session']['steps']
                if len(actual) != number or actual[-1]['id'] != rid:
                    raise RuntimeError('Unexpected action history')
                result = actual[-1]['result'] or {}
                action.update(result=result, resources=resources())
                item['actions'].append(action)
                checkpoint()
                log(f"{variant} {number}/{len(plan['steps'])} {step['skill']} {step['target']}: "
                    f"{action['observed_wall_seconds']:.3f}s; worker={result.get('timing',{}).get('worker_command_wall_seconds')}")
                if not actual[-1]['keep'] or result.get('ok') is False or result.get('error'):
                    raise RuntimeError(f'Action failed: {result}')
                if step['skill'] != 'ReportAbsence' and 'timing' not in result:
                    raise RuntimeError('Worker timing instrumentation missing')
            item['execution_elapsed_seconds'] = observed_end - start
            item['execution_completed_at'] = utc()
            if not state['session']['snapshot'].get('evaluation', {}).get('success'):
                raise RuntimeError('Final evaluation failed; preserving existing GT')
            if state['ground_truth']['id'] != plan['old_run_id']:
                raise RuntimeError('Saved ground truth changed during replay')
            for key in ['action_wall_seconds','worker_command_wall_seconds',
                        'environment_step_wall_seconds','frame_callback_wall_seconds','clip_close_wall_seconds']:
                item[key] = sum(a['result'].get('timing', {}).get(key, 0) for a in item['actions'])
            checkpoint()
            save_start = time.perf_counter()
            api(variant, 'record', {'session_id': sid})
            state = idle(variant, sid)
            item['save_wall_seconds'] = time.perf_counter() - save_start
            if not (state['session']['last_result'] or {}).get('ok'):
                raise RuntimeError(f"Save failed: {state['session']['last_result']}")
            run = state['ground_truth']
            item['run_id'] = run['id']
            if run['id'] == plan['old_run_id'] or run['artifact_status'] != 'ready' or run['stale'] or not run['steps_complete']:
                raise RuntimeError('Saved recording verification failed')
            if [{k:a[k] for k in ['skill','target']} for a in run['actions']] != plan['steps']:
                raise RuntimeError('Saved sequence differs from authorized sequence')
            folder = BASE / 'ground_truth/T6' / variant
            exported = json.loads((folder / 'run.json').read_text())
            if exported['id'] != run['id']:
                raise RuntimeError('Archive ID mismatch')
            if hashlib.sha256((folder / 'video.mp4').read_bytes()).hexdigest() != run['video_sha256']:
                raise RuntimeError('Video checksum mismatch')
            item.update(status='saved', completed_at=utc(), sim_steps=run['sim_steps'],
                        evaluation=run['evaluation'], resources_end=resources())
            checkpoint()
            publish_timing(item)
            api(variant, 'close', {'session_id': sid})
            log(f"SAVED {variant}: {item['execution_elapsed_seconds']:.3f}s elapsed; {item['action_wall_seconds']:.3f}s active skills")
        except Exception as error:
            item.update(status='failed', error=str(error), failed_at=utc())
            checkpoint(); log(f'FAILED {variant}: {error}')
            raise
    log('ALL T6 RERUNS COMPLETE')


if __name__ == '__main__':
    main()
