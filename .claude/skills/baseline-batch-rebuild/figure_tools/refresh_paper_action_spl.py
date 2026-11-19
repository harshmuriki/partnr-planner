# Batch-agnostic copy of scripts/dashboard_tc_spl/refresh_paper_action_spl.py (paper order ReAct, VLM-TAMP, TRU-POMDP).
# Reads BATCH_DIR / FIG_DIR from the environment; run via rebuild_batch.sh.
"""Refresh paper SPL aggregates from verified action-count scores."""
import csv
import json
import os
from pathlib import Path
from statistics import fmean

root = Path(os.environ['BATCH_DIR'])
data = json.loads((root / 'tc_spl_data.json').read_text())
keys = {b['label']: b['key'] for b in data['baselines']}
path = Path(os.environ['FIG_DIR']) / 'metric_percentages.csv'
with path.open() as stream:
    reader = csv.DictReader(stream)
    fields = list(reader.fieldnames)
    rows = list(reader)
if 'SPL_S_percent' not in fields:
    fields.append('SPL_S_percent')
for row in rows:
    selected = [r for r in data['rows'] if r['condition'] == keys[row['baseline']]
                and (row['grouping'] == 'overall'
                     or row['grouping'] == 'task' and str(r['task']) == row['group']
                     or row['grouping'] == 'memory' and r['memory'] == row['group'])]
    assert len(selected) == int(row['runs'])
    values = []
    for run in selected:
        assert run['reference_actions'] > 0 and run['action_count'] is not None
        value = float(run['success'] == 1) * run['reference_actions'] / max(run['action_count'], run['reference_actions'])
        assert abs(value - run['spl']) < 1e-12
        values.append(value)
    row['SPL_percent'] = 100 * fmean(values)
    row['SPL_S_percent'] = 100 * fmean(r['spl_s'] for r in selected if r['spl_s'] is not None)
    if row['grouping'] == 'overall':
        print(row['baseline'], row['SPL_percent'])
# Paper order: ReAct, VLM-TAMP, TRU-POMDP within each overall/task/memory group.
order = ['ReAct (images)', 'VLM-TAMP', 'TRU-POMDP']
groups = list(dict.fromkeys((r['grouping'], r['group']) for r in rows))
rows.sort(key=lambda r: (groups.index((r['grouping'], r['group'])), order.index(r['baseline'])))
with path.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
