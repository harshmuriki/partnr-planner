"""Link final-run HTML/assets and rebuild the comparison dashboard without API calls."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import gzip
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote

from build_baseline_dashboard import build, render
from export_run_workbook import extract_metrics
from judge_evidence import simulator
from dashboard_subtasks import subtask_record
from dashboard_checkpoint_results import recover_checkpoint

REPO = Path(__file__).resolve().parents[2]
CONDITIONS = {
    'react': 'react_images', 'react-no-image': 'react_no_images',
    'vlm-tamp': 'vlm_tamp_images', 'vlm-tamp-no-image': 'vlm_tamp_no_images',
    'tru-pomdp': 'tru_pomdp_no_images',
}


def symlink(target, link):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        raise ValueError(f'Existing link has another target: {link}')
    if link.exists():
        raise FileExistsError(link)
    link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=target.is_dir())


def url(path, base):
    return quote(os.path.relpath(path, base).replace(os.sep, '/'), safe='/')


def last_terminal(path, block_size=1024 * 1024):
    """Read backward to the last terminal record without scanning all sim ticks.

    A byte offset identifies the source record. If no terminal record exists,
    all lines are checked and the caller retains the saved aggregate scores.
    """
    pattern = re.compile(rb'"event"\s*:\s*"terminal"')
    with path.open('rb') as stream:
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
                if not pattern.search(line):
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if record.get('event') == 'terminal':
                    record['_id'] = f'{path.name}:byte:{offset}'
                    return record
    return None


def prepare_run(job, summary, html_root, web_root):
    source = Path(job['output'])
    task, memory, uncertainty = job['episode_id'].split('-')
    condition = CONDITIONS[job['planner']]
    rid = f"{job['planner']}/{job['episode_id']}/repeat_{job['repetition']}"
    link_root = html_root / Path(job['replacement_collection']).name if job.get('replacement_collection') else html_root
    destination = link_root / job['planner'] / f'task_{task[1:]}' / job['episode_id'] / f"repeat_{job['repetition']}"
    symlink(source, destination / 'artifacts')
    html_files = sorted(source.rglob('*.html'))
    links = []
    for p in html_files:
        target = destination / p.relative_to(source)
        symlink(p, target)
        label = ('Trace' if 'traces' in p.parts else 'PDDL' if p.name.endswith('_pddl.html')
                 else 'VLM prompts' if p.name == 'vlm_prompts.html'
                 else 'Observation history' if p.name == 'observation_history.html' else p.name)
        links.append({'label': label, 'url': url(target, web_root),
                      'path': str(target), 'source': str(p)})
    # HTML uses relative image/text links. Link each sibling asset directory/file
    # without copying images or changing the archived HTML.
    for parent in {p.parent for p in html_files}:
        for asset in parent.iterdir():
            if asset.suffix.lower() == '.html' or any(asset in p.parents for p in html_files):
                continue
            symlink(asset, destination / asset.relative_to(source))
    ep_path = source / 'input/dataset.json.gz'
    episode = json.loads(gzip.decompress(ep_path.read_bytes()))['episodes'][0]
    records = []
    terminal_source = str(summary)
    for evidence in sorted((source / 'dataset').glob('evidence-*.jsonl')):
        record = last_terminal(evidence)
        if record is not None:
            records.append(record)
            terminal_source = str(evidence)
    sim = simulator(episode, records, job.get('metrics') or {})
    total = len(sim['criteria'])
    completion = sim['metrics'].get('task_percent_complete')
    solved = None
    if all(c['status'] != 'unknown' for c in sim['criteria']):
        solved = sum(c['status'] == 'met' for c in sim['criteria'])
    elif completion is not None and abs(completion * total - round(completion * total)) < 1e-5:
        solved = round(completion * total)
    run = {
        'id': rid, 'condition': condition, 'task': int(task[1:]),
        'episode_id': job['episode_id'], 'memory': memory, 'uncertainty': uncertainty,
        'repetition': job['repetition'], 'run_id': job['run_id'], 'path': str(source),
        'status': job['status'], 'runtime_seconds': job.get('elapsed_seconds'),
        'simulator': dict(sim, total_count=total, satisfied_count=solved,
                          source={'file': terminal_source}),
    }
    row = extract_metrics(run)
    cost = (job.get('metrics') or {}).get('cost_metrics') or {}
    row.update({k: cost.get(k) for k in ('llm_planning_time_s', 'total_tokens', 'llm_usd')})
    row['llm_requests'] = cost.get('llm_requests', cost.get('llm_call_count'))
    row['html_links'] = links
    row['folder_url'] = url(destination / 'artifacts', web_root) + '/'
    row['imported'] = job.get('imported', False)
    row['timeout_seconds'] = job.get('timeout_seconds')
    scene_path = source / 'input/scene_info.json'
    scene = json.loads(scene_path.read_text()) if scene_path.exists() else {}
    row['subtask_results'] = subtask_record(episode, scene, sim, terminal_source)
    checkpoint = recover_checkpoint(row)
    if checkpoint:
        row['checkpoint_results'] = checkpoint
    for key in ('replacement_collection', 'supersedes_run_folder'):
        if job.get(key):
            row[key] = job[key]
    return row


def write_index(html_root, rows, web_root):
    html = ['<!doctype html><meta charset="utf-8"><title>Final baseline HTML</title>',
            '<style>body{font:15px system-ui;max-width:1300px;margin:30px auto;padding:0 20px}td,th{text-align:left;padding:7px;border-bottom:1px solid #ddd}table{border-collapse:collapse;width:100%}a{color:#1656b8}details{margin:12px 0}summary{cursor:pointer;font-weight:600}</style>',
            '<h1>Final baseline HTML</h1>',
            '<p>All seven tasks, five baselines, three repetitions. Saved HTML and supporting assets are symlinked to the final collection. Runs without saved HTML still link to their logs and artifacts.</p>']
    for baseline, condition in CONDITIONS.items():
        subset = [r for r in rows if r['condition'] == condition]
        html.append(f'<h2>{escape(baseline)}</h2>')
        for task in range(1, 8):
            selected = sorted((r for r in subset if r['task'] == task), key=lambda r: (r['task_id'], r['iteration']))
            html.append(f'<details><summary>Task {task} · {len(selected)} runs</summary><table><tr><th>Variant</th><th>Repeat</th><th>Execution</th><th>Simulator outcome</th><th>HTML / artifacts</th></tr>')
            for row in selected:
                links = [f'<a href="{escape(url(Path(l["path"]), html_root))}">{escape(l["label"])}</a>' for l in row['html_links']]
                link_root = html_root / Path(row['replacement_collection']).name if row.get('replacement_collection') else html_root
                artifact = link_root / baseline / f'task_{task}' / row['task_id'] / f'repeat_{row["iteration"]}' / 'artifacts'
                links.append(f'<a href="{escape(url(artifact, html_root))}/">Run files</a>')
                html.append(f'<tr><td>{escape(row["task_id"])}</td><td>{row["iteration"]}</td><td>{row["execution_status"]}</td><td>{row["simulator_outcome"]}</td><td>{" · ".join(links)}</td></tr>')
            html.append('</table></details>')
    (html_root / 'index.html').write_text('\n'.join(html) + '\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, default=REPO / 'results/all_baselines_no_video_20261001_220617/summary.json')
    parser.add_argument('--html-root', type=Path, default=REPO / 'results/final_baselines_html')
    parser.add_argument('--dashboard', type=Path, default=REPO / 'results/run_comparison_20260929/baseline_dashboard.html')
    args = parser.parse_args()
    summary, html_root, dashboard = args.source.resolve(), args.html_root.resolve(), args.dashboard.resolve()
    html_root.mkdir(parents=True, exist_ok=True)
    web_root = dashboard.parent
    symlink(html_root, web_root / html_root.name)
    jobs = json.loads(summary.read_text())['jobs']
    overrides_path = html_root / 'rerun_overrides.json'
    if overrides_path.exists():
        overrides = json.loads(overrides_path.read_text())
        replacements = {(j['planner'], j['episode_id'], j['repetition']): j for j in overrides['jobs']}
        jobs = [replacements.get((j['planner'], j['episode_id'], j['repetition']), j) for j in jobs]
    identities = {(j['planner'], j['episode_id'], j['repetition']) for j in jobs}
    assert len(identities) == len(jobs), 'Duplicate run identities'
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        worker = partial(prepare_run, summary=summary, html_root=html_root, web_root=web_root)
        for row in pool.map(worker, jobs):
            rows.append(row)
            if len(rows) % 150 == 0:
                print(f'Linked HTML and extracted metrics for {len(rows)}/{len(jobs)} runs', flush=True)
    data = build(rows, {})  # Costs joined by full identity above, never repeated _r001 IDs.
    data['source_summary'] = str(summary)
    data['html_index'] = html_root.name + '/index.html'
    data['subtitle'] = f'{len(rows):,} runs · 90 variants × 5 baselines × 3 repetitions · final collection (Task 2/4 imported)'
    missing = sum(r['completion'] is None for r in rows)
    data['notes'] = [
        'Source: all_baselines_no_video_20261001_220617. Tasks 1/3/5/6/7 use the new runs; Tasks 2/4 retain their historical inputs and code behavior.',
        'Run details preserve unavailable simulator results as unavailable. Success rate uses all runs as its denominator; unscored runs do not count as successes.',
        f'For comparison with the previous dashboard, aggregate completion and subgoal averages include zero contributions for the {missing} runs without saved completion scores. This aggregation convention is not a simulator failure label. Saved partial scores are retained, including those from timed-out runs.',
        'SR: share of runs with complete simulator outcomes. CR: mean task_percent_complete under the above missing-score convention. ± is standard deviation across the three repetitions.',
        'Subgoals: simulator evaluation propositions, including quantity expansion. Terminal evaluation snapshots take precedence over aggregate scores. Unscored prose is excluded.',
        'Runtime: measured process wall time including setup, exploration, model calls, execution and shutdown. Time limits: 600 s for Tasks 1–5; 1200 s for Tasks 6/7.',
        'Task 7 uses the archived evaluation criteria of the new runs, including the closet wardrobe target. No historical relaxed rescoring is applied.',
        'Actions, simulator steps, tokens and cost average only available values; n reports coverage. Costs are joined by baseline, variant and repetition.',
        'Action convention: each Rearrange selection adds 2 Navigate + 1 Pick + 1 Place; Done is excluded. Partial logs can provide only lower bounds.',
        'Click a variant/baseline cell for all three repetitions and links to every saved HTML file. Missing HTML is not fabricated; raw artifacts remain accessible.',
    ]
    rows_path = html_root / 'baseline_runs_comparison.json'
    rows_path.write_text(json.dumps(rows, indent=2) + '\n')
    (html_root / 'dashboard_data.json').write_text(json.dumps(data, indent=2) + '\n')
    manifest = {'source_summary': str(summary), 'run_count': len(rows),
                'html_count': sum(len(r['html_links']) for r in rows),
                'runs_without_html': sum(not r['html_links'] for r in rows),
                'runs_per_baseline': dict(Counter(r['condition'] for r in rows)),
                'links': [l for r in rows for l in r['html_links']]}
    (html_root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    write_index(html_root, rows, web_root)
    backup = dashboard.with_name('baseline_dashboard.before_final_collection.html')
    if dashboard.exists() and not backup.exists():
        shutil.copy2(dashboard, backup)
    temporary = dashboard.with_suffix('.tmp')
    temporary.write_text(render(data))
    temporary.replace(dashboard)
    (html_root / 'README.md').write_text('''# Final baseline HTML

HTML files are symlinks grouped as `<baseline>/task_<N>/<variant>/repeat_<N>/`.
Original relative subdirectories are retained so images and related pages resolve.
Each repetition also links to its complete run folder through `artifacts/`.
`manifest.json` records every HTML link and its source. `index.html` lists all runs,
including those with no saved HTML. No simulator or annotation API calls are made.

Rebuild from the repository root:

```bash
python scripts/trace_annotator/build_final_baselines.py
```

The comparison dashboard is updated at
`results/run_comparison_20260929/baseline_dashboard.html`.
Serve that directory on port 8765 to view it at
http://localhost:8765/baseline_dashboard.html.
''')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'links'}, indent=2), flush=True)
    print('Dashboard:', dashboard, flush=True)


if __name__ == '__main__':
    main()
