"""Add subtask scores to cached final dashboard rows without re-extracting logs."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path

from build_baseline_dashboard import render
from build_final_baselines import last_terminal
from dashboard_subtasks import chart_data, subtask_record
from judge_evidence import simulator

REPO = Path(__file__).resolve().parents[2]


def enrich(row, job):
    root = Path(row['run_folder'])
    episode = json.loads(gzip.decompress((root / 'input/dataset.json.gz').read_bytes()))['episodes'][0]
    scene_path = root / 'input/scene_info.json'
    scene = json.loads(scene_path.read_text()) if scene_path.exists() else {}
    source = Path(row['simulator_source'])
    # Cached provenance already identifies whether a terminal snapshot exists.
    # Do not scan gigabytes of tick logs again for runs with no final record.
    records = []
    if source.suffix == '.jsonl':
        terminal = last_terminal(source)
        if terminal is None:
            raise ValueError(f'Cached terminal source has no terminal record: {source}')
        records.append(terminal)
    sim = simulator(episode, records, job.get('metrics') or {})
    assert len(sim['criteria']) == row['num_subtasks'], row['id']
    row['subtask_results'] = subtask_record(episode, scene, sim, str(source))
    return row


def atomic_write(path, content):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=REPO / 'results/final_baselines_html')
    parser.add_argument('--dashboard', type=Path, default=REPO / 'results/run_comparison_20260929/baseline_dashboard.html')
    args = parser.parse_args()
    rows_path, data_path = args.root / 'baseline_runs_comparison.json', args.root / 'dashboard_data.json'
    rows, data = json.loads(rows_path.read_text()), json.loads(data_path.read_text())
    if not all('subtask_results' in r for r in rows):
        jobs = json.loads(Path(data['source_summary']).read_text())['jobs']
        by_id = {f"{j['planner']}/{j['episode_id']}/repeat_{j['repetition']}": j for j in jobs}
        with ThreadPoolExecutor(max_workers=4) as pool:
            updated = []
            for row in pool.map(lambda r: enrich(r, by_id[r['id']]), rows):
                updated.append(row)
                if len(updated) % 150 == 0:
                    print(f'Recovered criteria for {len(updated)}/{len(rows)} runs', flush=True)
        rows = updated
    data['subtasks'] = chart_data(rows)
    atomic_write(rows_path, json.dumps(rows, indent=2) + '\n')
    atomic_write(data_path, json.dumps(data, indent=2) + '\n')
    atomic_write(args.dashboard, render(data))
    print(json.dumps({'runs': len(rows), 'distinct_criteria': len(data['subtasks']['definitions']),
                      'statuses': dict(Counter(c['status'] for r in rows for c in r['subtask_results']['criteria']))}))
    print(args.dashboard)


if __name__ == '__main__':
    main()
