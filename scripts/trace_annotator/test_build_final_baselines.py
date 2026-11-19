import gzip
import json

from build_final_baselines import prepare_run
from build_baseline_dashboard import build


def make_job(tmp_path, repeat, metrics=None):
    root = tmp_path / f'run{repeat}'
    (root / 'input').mkdir(parents=True)
    episode = {'evaluation_propositions': [
        {'function_name': 'is_clean', 'args': {'object_handles': ['plate'], 'number': 1}}
    ]}
    (root / 'input/dataset.json.gz').write_bytes(gzip.compress(json.dumps({'episodes': [episode]}).encode()))
    (root / 'console.log').write_text('')
    return {'output': str(root), 'planner': 'react', 'episode_id': 'T1-ACC-BASE',
            'repetition': repeat, 'run_id': 'same_r001', 'status': 'completed',
            'metrics': metrics, 'elapsed_seconds': 12}


def test_html_relative_images_and_sibling_pages(tmp_path):
    job = make_job(tmp_path, 1)
    root = tmp_path / 'run1' / 'dataset' / 'traces'
    (root / 'images').mkdir(parents=True)
    (root / 'images/p.png').write_bytes(b'image')
    (root / 'trace.html').write_text('<img src="images/p.png"><a href="other.html">Next</a>')
    (root / 'other.html').write_text('Next')
    row = prepare_run(job, tmp_path / 'summary.json', tmp_path / 'html', tmp_path)
    assert len(row['html_links']) == 2
    destination = tmp_path / 'html/react/task_1/T1-ACC-BASE/repeat_1/dataset/traces'
    assert (destination / 'trace.html').is_symlink()
    assert (destination / 'images/p.png').read_bytes() == b'image'
    assert (destination / 'other.html').read_text() == 'Next'


def test_repeated_run_ids_keep_repetition_costs(tmp_path):
    rows = []
    for repeat, cost in [(1, 1.0), (2, 3.0)]:
        job = make_job(tmp_path, repeat, {'task_state_success': 1, 'task_percent_complete': 1,
                                       'cost_metrics': {'llm_usd': cost, 'llm_call_count': repeat}})
        rows.append(prepare_run(job, tmp_path / 'summary.json', tmp_path / 'html', tmp_path))
    assert len({row['id'] for row in rows}) == 2
    assert [r['llm_requests'] for r in rows] == [1, 2]
    data = build(rows, {})
    assert data['overall']['react_images']['usd_mean'] == 2
    assert [r['usd'] for r in data['variants'][0]['cells']['react_images']['runs']] == [1, 3]


def test_missing_scores_stay_unavailable_and_terminal_wins(tmp_path):
    job = make_job(tmp_path, 1)
    row = prepare_run(job, tmp_path / 'summary.json', tmp_path / 'html', tmp_path)
    assert row['simulator_outcome'] == 'unavailable'
    assert row['subtasks_succeeded'] is None and row['completion'] is None
    root = tmp_path / 'run1/dataset'
    root.mkdir()
    (root / 'evidence-test.jsonl').write_text(json.dumps({
        'event': 'terminal', 'evaluator_only_state': {
            'task_state_success': 1, 'task_percent_complete': 1,
            'auto_eval_proposition_tracker': {'proposition_satisfied_at': [3]},
            'task_constraint_validation': []}}) + '\n')
    row = prepare_run(job, tmp_path / 'summary.json', tmp_path / 'html', tmp_path)
    assert row['simulator_outcome'] == 'complete' and row['subtasks_succeeded'] == 1


def test_reverse_terminal_reader_across_blocks_and_truncation(tmp_path):
    from build_final_baselines import last_terminal
    p = tmp_path / 'events.jsonl'
    early = json.dumps({'event': 'terminal', 'value': 1})
    final = json.dumps({'event': 'terminal', 'value': 2, 'large': 'x' * 100})
    prefix = early + '\n' + json.dumps({'event': 'step'}) + '\n'
    p.write_text(prefix + final + '\n' + '{"event":"terminal",truncated')
    record = last_terminal(p, block_size=13)
    assert record['value'] == 2
    assert record['_id'].endswith(f':byte:{len(prefix)}')
    p.write_text(final)
    assert last_terminal(p, block_size=13)['value'] == 2
    p.write_text('{"event":"step"}\n')
    assert last_terminal(p, block_size=13) is None
    p.write_text('')
    assert last_terminal(p, block_size=13) is None
