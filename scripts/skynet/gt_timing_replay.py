#!/usr/bin/env python3
"""Replay the saved ground-truth sequences headlessly to measure timing (Skynet / Slurm).

  queue      freeze the current saved sequences into <out>/queue.json and split them into shards
  run        replay one shard sequentially (one Slurm array task = one GPU); fresh worker per variant
  summarize  per-variant, per-task, overall and per-skill timing tables
  publish    (local machine) copy the measurements into ground_truth_timing/current_recordings.csv

Replays never touch the ground-truth DB or archive. A replay counts only when every action
succeeds and the final evaluation succeeds. Timers are the worker instrumentation described in
docs/ground_truth_timing_handoff.md, so fields match the single-sandbox T6 batch.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import sqlite3
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.scenario_ground_truth import (BASE, REPORT_RESULT, EpisodeCatalog, GroundTruthStore,
                                           HabitatWorker, succeeded)

TIMING = BASE / 'ground_truth_timing'
SUMS = ['action_wall_seconds', 'worker_command_wall_seconds', 'environment_step_wall_seconds',
        'frame_callback_wall_seconds', 'clip_close_wall_seconds']
INTERVAL = 'first_action_submit_to_final_action_return_headless'  # no HTTP server or client polling


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def log(message):
    print(utc(), message, flush=True)


def resources():
    memory = {line.split(':')[0]: line.split(':')[1].strip()
              for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemAvailable:', 'SwapFree:'))}
    return {'utc': utc(), 'load_average': os.getloadavg(), **memory}


def cpu_seconds(pid):
    """utime + stime of the worker process (all its threads), from /proc."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
    except (OSError, IndexError, ValueError):
        return None


def conditions():
    def command(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return None
    return {'utc': utc(), 'host': socket.gethostname(), 'platform': platform.platform(),
            'cpus_allowed': len(os.sched_getaffinity(0)), 'cpu_model': command('sh', '-c', "grep -m1 'model name' /proc/cpuinfo"),
            'gpu': command('nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'),
            'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')},
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'omp_num_threads': os.environ.get('OMP_NUM_THREADS'), 'habitat_python': os.environ.get('HABITAT_PYTHON')}


# ---------------------------------------------------------------------------- queue
def build_queue(args):
    out = Path(args.out)
    if (out / 'queue.json').exists():
        raise SystemExit(f'{out}/queue.json exists; use a new --out for a new measurement batch')
    catalog = EpisodeCatalog()
    plans = []
    with sqlite3.connect(BASE / 'ground_truth.sqlite3') as db:
        runs = db.execute("select id, variant, spec_hash, dataset_hash, sim_steps from runs "
                          "where status='completed' order by variant").fetchall()
        for run_id, variant, spec_hash, dataset_hash, sim_steps in runs:
            if args.variants and variant not in args.variants:
                continue
            meta = catalog.get(variant)
            if (meta['spec_hash'], meta['dataset_hash']) != (spec_hash, dataset_hash):
                raise SystemExit(f'{variant}: saved ground truth is stale against its spec/dataset')
            steps = [{'skill': s, 'target': t} for s, t in db.execute(
                'select skill, target from actions where run_id=? order by sequence', (run_id,))]
            plans.append({'variant': variant, 'run_id': run_id, 'spec_hash': spec_hash, 'dataset_hash': dataset_hash,
                          'dataset': str(Path(meta['dataset']).relative_to(ROOT)), 'episode_id': meta['episode_id'],
                          'saved_sim_steps': sim_steps, 'steps': steps})
    if args.variants and len(plans) != len(set(args.variants)):
        raise SystemExit(f'Unknown or unsaved variants: {sorted(set(args.variants) - {p["variant"] for p in plans})}')
    if args.smoke:
        # Shortest saved recording of each task: loads every scene once, quickly.
        shortest = {}
        for plan in plans:
            task = plan['variant'].split('-')[0]
            if task not in shortest or plan['saved_sim_steps'] < shortest[task]['saved_sim_steps']:
                shortest[task] = plan
        plans = sorted(shortest.values(), key=lambda p: p['variant'])
    # Longest-first greedy split by saved simulator steps, so shards finish at similar times.
    units = [(plan['variant'], attempt) for plan in plans for attempt in range(1, args.repeats + 1)]
    weight = {p['variant']: p['saved_sim_steps'] or 1 for p in plans}
    shards = [[] for _ in range(min(args.shards, len(units)))]
    for variant, attempt in sorted(units, key=lambda u: -weight[u[0]]):
        min(shards, key=lambda s: sum(weight[v] for v, _ in s)).append((variant, attempt))
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'queue.json', {'created_at': utc(), 'repeats': args.repeats, 'smoke': args.smoke, 'plans': plans,
                               'shards': [[{'variant': v, 'attempt': a} for v, a in shard] for shard in shards]})
    print(len(shards))


# ---------------------------------------------------------------------------- run
def replay(plan, attempt, path):
    item = {'variant': plan['variant'], 'attempt': attempt, 'run_id': plan['run_id'], 'status': 'running',
            'started_at': utc(), 'actions': [], 'resources_start': resources(), 'host': socket.gethostname(),
            'slurm_job': os.environ.get('SLURM_JOB_ID'), 'interval_definition': INTERVAL}
    write(path, item)
    worker = HabitatWorker()
    try:
        load_start = time.perf_counter()
        snapshot = worker.call({'op': 'load', 'dataset': str(ROOT / plan['dataset']), 'episode_id': plan['episode_id'],
                                'results_dir': str(worker.session_dir), 'load_id': uuid.uuid4().hex})
        item['load_wall_seconds'] = time.perf_counter() - load_start
        item['execution_started_at'] = utc()
        log(f"START {plan['variant']} r{attempt}: {len(plan['steps'])} actions; load {item['load_wall_seconds']:.1f}s")
        start = time.perf_counter()
        for number, step in enumerate(plan['steps'], 1):
            action = {'sequence': number, **step, 'submitted_at': utc()}
            cpu_before, before = cpu_seconds(worker.process.pid), time.perf_counter()
            if step['skill'] == 'ReportAbsence':
                result = dict(REPORT_RESULT)  # The server answers these without a worker call.
            else:
                snapshot = worker.call({'op': 'skill', **step})
                result = snapshot.pop('action_result', None) or {}
            end = time.perf_counter()
            cpu_after = cpu_seconds(worker.process.pid)
            action.update(completed_at=utc(), observed_wall_seconds=end - before, result=result, resources=resources(),
                          worker_cpu_seconds=None if None in (cpu_before, cpu_after) else cpu_after - cpu_before)
            item['actions'].append(action)
            write(path, item)
            log(f"{plan['variant']} r{attempt} {number}/{len(plan['steps'])} {step['skill']} {step['target'][:60]}: "
                f"{action['observed_wall_seconds']:.3f}s; cpu={action['worker_cpu_seconds']}")
            if not succeeded(step['skill'], result):
                raise RuntimeError(f'Action {number} failed: {result.get("error") or result.get("response")}')
        item['execution_elapsed_seconds'] = end - start
        item['execution_completed_at'] = utc()
        for key in SUMS:
            item[key] = sum(a['result'].get('timing', {}).get(key, 0) for a in item['actions'])
        item['worker_cpu_seconds'] = sum(a['worker_cpu_seconds'] or 0 for a in item['actions'])
        item['sim_steps'], _ = GroundTruthStore.step_totals(item['actions'])
        item['saved_sim_steps'] = plan['saved_sim_steps']
        item['evaluation'] = snapshot.get('evaluation', {})
        if not item['evaluation'].get('success'):
            raise RuntimeError('Final evaluation failed')
        item.update(status='replayed', completed_at=utc(), resources_end=resources())
        log(f"DONE {plan['variant']} r{attempt}: {item['execution_elapsed_seconds']:.1f}s elapsed; "
            f"{item['action_wall_seconds']:.1f}s active; sim_steps {item['sim_steps']} (saved {plan['saved_sim_steps']})")
    except Exception as error:
        item.update(status='failed', error=str(error), failed_at=utc())
        log(f"FAILED {plan['variant']} r{attempt}: {error}")
        # Keep the worker log next to the evidence; it is the only Habitat-side trace.
        if (worker.directory / 'worker.log').exists():
            shutil.copy2(worker.directory / 'worker.log', path.with_suffix('.worker.log'))
    finally:
        write(path, item)
        worker.close()
        shutil.rmtree(worker.directory, ignore_errors=True)
    return item


def run_shard(args):
    out = Path(args.out)
    queue = json.loads((out / 'queue.json').read_text())
    plans = {p['variant']: p for p in queue['plans']}
    (out / 'results').mkdir(exist_ok=True)
    (out / 'conditions').mkdir(exist_ok=True)
    write(out / 'conditions' / f'shard_{args.shard}.json', conditions())
    failures = 0
    for unit in queue['shards'][args.shard]:
        path = out / 'results' / f"{unit['variant']}.r{unit['attempt']}.json"
        if path.exists() and json.loads(path.read_text()).get('status') == 'replayed':
            continue  # A requeued array task resumes; partial attempts are rerun from the start.
        failures += replay(plans[unit['variant']], unit['attempt'], path)['status'] != 'replayed'
    log(f'SHARD {args.shard} COMPLETE: {failures} failed')
    sys.exit(1 if failures else 0)


# ---------------------------------------------------------------------------- summarize
def representative(out):
    """Per variant: the replayed attempt with the median elapsed time, plus all attempt times."""
    attempts = {}
    for path in sorted((out / 'results').glob('*.r*.json')):
        item = json.loads(path.read_text())
        attempts.setdefault(item['variant'], []).append(item)
    chosen = {}
    for variant, items in attempts.items():
        ok = sorted((i for i in items if i['status'] == 'replayed'), key=lambda i: i['execution_elapsed_seconds'])
        chosen[variant] = {'item': ok[(len(ok) - 1) // 2] if ok else None, 'attempts': items,
                           'elapsed': [i['execution_elapsed_seconds'] for i in ok]}
    return chosen


def table(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(args):
    out = Path(args.out)
    queue = json.loads((out / 'queue.json').read_text())
    chosen = representative(out)
    rows, skills = [], {}
    for plan in queue['plans']:
        entry = chosen.get(plan['variant'], {'item': None, 'attempts': [], 'elapsed': []})
        item = entry['item']
        status = 'replayed' if item else (entry['attempts'][-1]['status'] if entry['attempts'] else 'not_run')
        rows.append({'variant': plan['variant'], 'task': plan['variant'].split('-')[0], 'status': status,
                     'actions': len(plan['steps']), 'attempts_ok': len(entry['elapsed']),
                     'elapsed_s': item and round(item['execution_elapsed_seconds'], 3),
                     'elapsed_min_max_s': ' / '.join(f'{v:.1f}' for v in (min(entry['elapsed']), max(entry['elapsed']))) if entry['elapsed'] else '',
                     'active_skill_s': item and round(item['action_wall_seconds'], 3),
                     'env_step_s': item and round(item['environment_step_wall_seconds'], 3),
                     'frame_callback_s': item and round(item['frame_callback_wall_seconds'], 3),
                     'worker_cpu_s': item and round(item['worker_cpu_seconds'], 3),
                     'load_s': item and round(item['load_wall_seconds'], 3),
                     'sim_steps': item and item['sim_steps'], 'saved_sim_steps': plan['saved_sim_steps'],
                     'host': item and item['host'], 'error': '' if item else (entry['attempts'][-1].get('error', '') if entry['attempts'] else '')})
        for action in (item or {}).get('actions', []):
            s = skills.setdefault(action['skill'], {'skill': action['skill'], 'count': 0, 'observed_s': 0.0, 'env_step_s': 0.0, 'sim_steps': 0})
            s['count'] += 1
            s['observed_s'] += action['observed_wall_seconds']
            s['env_step_s'] += action['result'].get('timing', {}).get('environment_step_wall_seconds', 0)
            s['sim_steps'] += action['result'].get('skill_steps') or 0
    table(out / 'summary_variants.csv', rows)
    timed = [r for r in rows if r['status'] == 'replayed']
    tasks = []
    for task in sorted({r['task'] for r in rows}):
        values = [r['elapsed_s'] for r in timed if r['task'] == task]
        active = [r['active_skill_s'] for r in timed if r['task'] == task]
        tasks.append({'task': task, 'timed': len(values), 'saved': sum(r['task'] == task for r in rows),
                      'mean_elapsed_s': round(statistics.mean(values), 1) if values else '',
                      'median_elapsed_s': round(statistics.median(values), 1) if values else '',
                      'max_elapsed_s': round(max(values), 1) if values else '',
                      'mean_active_skill_s': round(statistics.mean(active), 1) if active else ''})
    all_values = [r['elapsed_s'] for r in timed]
    # Same weighting as the Metrics page: every recording counts once, not every task average.
    tasks.append({'task': 'ALL', 'timed': len(all_values), 'saved': len(rows),
                  'mean_elapsed_s': round(statistics.mean(all_values), 1) if all_values else '',
                  'median_elapsed_s': round(statistics.median(all_values), 1) if all_values else '',
                  'max_elapsed_s': round(max(all_values), 1) if all_values else '',
                  'mean_active_skill_s': round(statistics.mean(r['active_skill_s'] for r in timed), 1) if timed else ''})
    table(out / 'summary_tasks.csv', tasks)
    for s in skills.values():
        s.update(observed_s=round(s['observed_s'], 1), env_step_s=round(s['env_step_s'], 1),
                 mean_observed_s=round(s['observed_s'] / s['count'], 2),
                 seconds_per_sim_step=round(s['env_step_s'] / s['sim_steps'], 4) if s['sim_steps'] else '')
    if skills:
        table(out / 'summary_skills.csv', sorted(skills.values(), key=lambda s: -s['observed_s']))
    slow = sorted(((a['observed_wall_seconds'], r['variant'], a) for r in timed
                   for a in chosen[r['variant']]['item']['actions']), key=lambda x: -x[0])[:25]
    lines = [f"{seconds:8.1f}s  {variant:12} #{a['sequence']:<3} {a['skill']} {a['target']}  "
             f"(sim_steps {a['result'].get('skill_steps')}, env.step {a['result'].get('timing', {}).get('environment_step_wall_seconds', 0):.1f}s, "
             f"cpu {a['worker_cpu_seconds'] or 0:.1f}s)" for seconds, variant, a in slow]
    (out / 'summary_slowest_actions.txt').write_text('\n'.join(lines) + '\n')
    print(f'{len(timed)}/{len(rows)} variants replayed successfully\n')
    for t in tasks:
        print(f"{t['task']:4} timed {t['timed']:>2}/{t['saved']:<2} mean {t['mean_elapsed_s']!s:>8}s  "
              f"median {t['median_elapsed_s']!s:>8}s  max {t['max_elapsed_s']!s:>8}s")
    failed = [r for r in rows if r['status'] != 'replayed']
    if failed:
        print('\nNot replayed:', *(f"  {r['variant']}: {r['status']} {r['error']}" for r in failed), sep='\n')
    print('\nSlowest actions:', *lines[:10], sep='\n')
    print(f'\nTables: {out}/summary_*.csv, summary_slowest_actions.txt')


# ---------------------------------------------------------------------------- publish
def publish(args):
    out = Path(args.out).resolve()
    if not out.is_relative_to(TIMING.resolve()):
        raise SystemExit(f'Copy the batch folder under {TIMING} first (the UI reads evidence from there)')
    if subprocess.run(['pgrep', '-f', 'rerun_t6_timed.py'], capture_output=True).returncode == 0:
        raise SystemExit('The local T6 timing runner is still writing current_recordings.csv; publish after it finishes')
    queue = json.loads((out / 'queue.json').read_text())
    chosen = representative(out)
    with sqlite3.connect(BASE / 'ground_truth.sqlite3') as db:
        current = {v: (i, s, d) for i, v, s, d in db.execute(
            "select id, variant, spec_hash, dataset_hash from runs where status='completed'")}
    path = TIMING / 'current_recordings.csv'
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reader.fieldnames), list(reader)
    extra = ['action_wall_seconds', 'worker_command_wall_seconds', 'load_wall_seconds', 'save_wall_seconds']
    fields += [key for key in extra if key not in fields]
    by_variant = {row['variant']: row for row in rows}
    published, skipped = [], []
    for plan in queue['plans']:
        variant, item = plan['variant'], (chosen.get(plan['variant']) or {}).get('item')
        if not item:
            skipped.append(f'{variant}: no successful replay')
            continue
        if current.get(variant) != (plan['run_id'], plan['spec_hash'], plan['dataset_hash']):
            skipped.append(f'{variant}: saved recording changed since the queue was built (re-recorded?)')
            continue
        evidence = {**item, 'status': 'replayed', 'attempt_elapsed_seconds': chosen[variant]['elapsed'],
                    'save_wall_seconds': None}
        write(out / f'{variant}.json', evidence)  # The per-action table in the UI reads this file.
        row = by_variant.setdefault(variant, {'variant': variant})
        row.update(run_id=plan['run_id'], timing_status='measured_skynet',
                   execution_elapsed_seconds=f"{item['execution_elapsed_seconds']:.6f}",
                   execution_elapsed_minutes=f"{item['execution_elapsed_seconds'] / 60:.6f}",
                   interval_definition=INTERVAL, attempt_id='', source=f'{out.relative_to(TIMING.resolve())}/{variant}.json',
                   retained_attempts=str(len(chosen[variant]['elapsed'])), save_wall_seconds='')
        row.update({key: f'{item[key]:.6f}' for key in extra if key != 'save_wall_seconds'})
        published.append(variant)
    rows = sorted(by_variant.values(), key=lambda r: r['variant'])
    backup = out / f'current_recordings.before_publish_{datetime.now():%Y%m%d_%H%M%S}.csv'
    shutil.copy2(path, backup)
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=fields, restval='')
    writer.writeheader()
    writer.writerows(rows)
    temp = path.with_suffix('.csv.tmp')
    temp.write_text(text.getvalue())
    temp.replace(path)
    print(f'Published {len(published)} variants to {path} (backup: {backup})')
    if skipped:
        print('Skipped:', *skipped, sep='\n  ')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue')
    q.add_argument('--out', required=True)
    q.add_argument('--shards', type=int, default=8)
    q.add_argument('--repeats', type=int, default=1)
    q.add_argument('--variants', nargs='*')
    q.add_argument('--smoke', action='store_true', help='only the shortest saved variant of each task')
    r = sub.add_parser('run')
    r.add_argument('--out', required=True)
    r.add_argument('--shard', type=int, required=True)
    for name in ('summarize', 'publish'):
        sub.add_parser(name).add_argument('--out', required=True)
    args = parser.parse_args()
    {'queue': build_queue, 'run': run_shard, 'summarize': summarize, 'publish': publish}[args.command](args)


if __name__ == '__main__':
    main()
