"""Recover saved subtask states and publish completed reruns into the dashboard."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from build_baseline_dashboard import build, render
from build_final_baselines import prepare_run, write_index
from dashboard_checkpoint_results import recover_checkpoint
from dashboard_subtasks import chart_data
from enrich_dashboard_subtasks import atomic_write

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'results/final_baselines_html'
DASHBOARD=REPO/'results/run_comparison_20260929/baseline_dashboard.html'


def recovery(row):
    if row.get('completion') is None and not row.get('checkpoint_results'):
        checkpoint=recover_checkpoint(row)
        if checkpoint:
            row['checkpoint_results']=checkpoint
    return row


def refresh(rerun_root=None, recover_existing=False):
    rows_path=ROOT/'baseline_runs_comparison.json'
    data_path=ROOT/'dashboard_data.json'
    rows=json.loads(rows_path.read_text()); old=json.loads(data_path.read_text())
    if recover_existing:
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows=list(pool.map(recovery,rows))
    by_id={r['id']:r for r in rows}
    remaining=0
    replacements=[]
    if rerun_root:
        batch=json.loads((rerun_root/'batch.json').read_text())
        for entry in batch['jobs']:
            summary=Path(entry['experiment'])/'summary.json'
            if not summary.exists(): remaining+=1; continue
            try: jobs=json.loads(summary.read_text())['jobs']
            except (OSError,ValueError,KeyError): remaining+=1; continue
            if not jobs or jobs[0]['status'] not in ('completed','timeout','failed'):
                remaining+=1; continue
            job=dict(jobs[0],planner='tru-pomdp',episode_id=entry['episode_id'],repetition=entry['repetition'],timeout_seconds=1200)
            rid=f"tru-pomdp/{entry['episode_id']}/repeat_{entry['repetition']}"
            current=by_id[rid]
            job['replacement_collection']=str(rerun_root)
            job['supersedes_run_folder']=current.get('supersedes_run_folder',current['run_folder'])
            replacements.append(job)
            if current['run_folder']==job['output']: continue
            row=prepare_run(job,summary,ROOT,DASHBOARD.parent)
            recovery(row)
            row['supersedes_run_folder']=current['run_folder']
            row['replacement_collection']=str(rerun_root)
            by_id[rid]=row
            print('Published',rid,job['status'],row.get('checkpoint_results',{}).get('method'),flush=True)
        # Reproducible overrides: the original summary remains an immutable source.
        atomic_write(ROOT/'rerun_overrides.json',json.dumps({'collection':str(rerun_root),'jobs':replacements},indent=2)+'\n')
        rows=[by_id[r['id']] for r in rows]
    data=build(rows,{})
    for key in ('source_summary','html_index','subtitle','notes'):
        if key in old:data[key]=old[key]
    data['checkpoint_policy']='Last saved evaluator states are used for subtask charts only when final scores are missing. Overall final-score metrics remain unchanged.'
    data['replacement_collections']=sorted({r['replacement_collection'] for r in rows if r.get('replacement_collection')})
    atomic_write(rows_path,json.dumps(rows,indent=2)+'\n')
    atomic_write(data_path,json.dumps(data,indent=2)+'\n')
    atomic_write(DASHBOARD,render(data))
    if rerun_root:
        write_index(ROOT,rows,DASHBOARD.parent)
    print(json.dumps({'checkpoint_runs':sum(bool(r.get('checkpoint_results')) for r in rows),'subtask_statuses':dict(Counter(c['status'] for r in data['subtasks']['runs'] for c in r['criteria'])),'reruns_remaining':remaining}),flush=True)
    return remaining


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--recover-existing',action='store_true')
    parser.add_argument('--rerun-root',type=Path)
    parser.add_argument('--watch',type=int,default=0)
    args=parser.parse_args()
    first=True
    while True:
        remaining=refresh(args.rerun_root,args.recover_existing and first)
        first=False
        if not args.watch or not remaining:break
        time.sleep(max(10,args.watch))

if __name__=='__main__':main()
