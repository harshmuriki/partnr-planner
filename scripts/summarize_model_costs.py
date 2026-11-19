"""Summarize episode model-cost reports or legacy run.json metrics without APIs.

Usage: python scripts/summarize_model_costs.py --root RESULTS --task T3-ACC-BASE --output OUTPUT
Legacy intervals cover recorded usage only: missing/failed requests remain unknown.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_cost_module():
    # Avoid Habitat/GPU imports for offline accounting.
    for name, relative in [('habitat_llm.utils.llm_usage', 'habitat_llm/utils/llm_usage.py'),
                           ('habitat_llm.utils.llm_cost_report', 'habitat_llm/utils/llm_cost_report.py')]:
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


costs = load_cost_module()


def legacy_billing(metrics):
    details = {}
    for key in ('cached_tokens', 'cache_write_tokens'):
        if metrics.get(key) is not None:
            details[key] = metrics[key]
    usage = {'prompt_tokens': metrics.get('prompt_tokens'),
             'completion_tokens': metrics.get('completion_tokens'),
             'prompt_tokens_details': details}
    billing = costs.price_usage(metrics.get('llm_model', ''), usage)
    if (metrics.get('prompt_tokens') or 0) > costs.GPT56_LONG_CONTEXT_TOKENS:
        # Cumulative episode totals cannot identify individual long-context calls.
        billing['lower_bound_usd'] = None
        billing['upper_bound_usd'] = None
        billing['cost_usd'] = dict.fromkeys(costs.COMPONENTS)
        billing['issues'].append('missing_per_request_context_lengths')
    billing['complete'] = False
    billing['total_usd'] = None
    billing['issues'].append('legacy_metrics_do_not_audit_all_http_attempts')
    return billing


def collect(root, task=None):
    rows, covered = [], set()
    task_roots = list(root.glob('*/' + task)) if task else []
    search_roots = task_roots or [root]
    for path in sorted(p for base in search_roots for p in base.glob('**/llm_costs/*.json')):
        data = json.loads(path.read_text())
        if task and str(data.get('episode_id', '')).upper() != task.upper():
            continue
        summary = data['summary']
        run_path = next((p / 'run.json' for p in path.parents if (p / 'run.json').is_file()), None)
        if run_path:
            covered.add(run_path.resolve())
        relative = path.relative_to(root)
        targets = data.get('planner_targets', {})
        baseline = data.get('baseline') or (relative.parts[0] if run_path else ','.join(targets.values()) or root.name)
        rows.append({'baseline': baseline, 'episode_id': data.get('episode_id'),
                     'source': str(path.resolve()), 'kind': 'request_ledger',
                     'status': data['status'], **summary})
    for path in sorted(p for base in search_roots for p in base.glob('**/run.json')):
        if path.resolve() in covered:
            continue
        data = json.loads(path.read_text())
        if task and str(data.get('episode_id', '')).upper() != task.upper():
            continue
        metrics = data.get('metrics', {}).get('cost_metrics', {})
        billing = legacy_billing(metrics)
        summary = costs.summarize_calls([{'status': 'legacy', 'billing': billing}])
        summary['api_attempts'] = metrics.get('llm_calls_with_usage')
        summary['local_cache_hits'] = None
        rows.append({'baseline': path.relative_to(root).parts[0], 'episode_id': data.get('episode_id'),
                     'source': str(path.resolve()), 'kind': 'legacy_aggregate',
                     'status': data.get('status'), 'model': metrics.get('llm_model'), **summary})
    return rows


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['episode_id'], row['baseline']), []).append(row)
    output = []
    for (task, baseline), group in sorted(groups.items()):
        def complete_sum(key):
            values = [r[key] for r in group]
            return sum(values) if all(v is not None for v in values) else None
        output.append({'episode_id': task, 'baseline': baseline, 'runs_or_reports': len(group),
                       'complete': all(r['complete'] and r['status'] == 'completed' for r in group),
                       'total_usd': complete_sum('total_usd'),
                       'recorded_usage_lower_usd': complete_sum('lower_bound_usd'),
                       'recorded_usage_upper_usd': complete_sum('upper_bound_usd'),
                       'calls_with_usage_or_attempts': complete_sum('api_attempts'),
                       'input_tokens': sum(r['tokens']['input'] for r in group) if all(r['tokens']['input'] is not None for r in group) else None,
                       'output_tokens': sum(r['tokens']['output'] for r in group) if all(r['tokens']['output'] is not None for r in group) else None,
                       'cache_read_tokens': sum(r['tokens']['cache_read'] for r in group) if all(r['tokens']['cache_read'] is not None for r in group) else None,
                       'cache_write_tokens': sum(r['tokens']['cache_write'] for r in group) if all(r['tokens']['cache_write'] is not None for r in group) else None,
                       **{k + '_usd': sum(r['cost_usd'][k] for r in group)
                          if all(r['cost_usd'][k] is not None for r in group) else None
                          for k in costs.COMPONENTS},
                       'issues': '; '.join(sorted({i for r in group for i in r['issues']}))})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root.resolve(), args.task)
    if not rows:
        parser.error('No matching cost reports or run.json files')
    summary = aggregate(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'pricing_date': costs.PRICING_DATE, 'pricing_sources': costs.PRICING_SOURCES,
              'scope': str(args.root.resolve()),
              'note': 'All attempts under the supplied root are included. Legacy bounds apply to recorded tokens at standard list prices, not unrecorded requests or account invoices. Null total means full spend cannot be established.',
              'baselines': summary, 'runs': rows}
    (args.output / 'model_costs.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'model_costs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
