"""Copy a classifier snapshot into the batch and relocate viewer links locally."""
import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from urllib.parse import quote, unquote


# Display names only; data keys (e.g. placement_error) are unchanged.
DISPLAY_PHRASES=[('PLACEMENT ERROR','INCORRECT PLACEMENT'),('Placement Error','Incorrect Placement'),('Placement error','Incorrect placement'),('placement error','incorrect placement')]


def display_names(text):
    for old,new in DISPLAY_PHRASES:
        text=text.replace(old,new)
    return text.replace('const NICE = l => l.replace(/_/g, " ");','const NICE = l => l === "placement_error" ? "incorrect placement" : l.replace(/_/g, " ");')


def import_rules(source, root):
    source,root=source.resolve(),root.resolve()
    dest=root/'uncertainty_classifier'
    originals=dest/'source'
    if originals.exists():
        raise FileExistsError('Snapshot already exists; preserve it before importing another version')
    originals.mkdir(parents=True)
    manifest=[]
    for src in sorted(source.rglob('*')):
        if src.is_file() and '__pycache__' not in src.parts:
            rel=src.relative_to(source);out=originals/rel;out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,out)
            manifest.append({'path':str(rel),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
    html=(originals/'viewer/rule_viewer.html').read_text()
    prefix='const DATA = '
    offset=html.index(prefix)+len(prefix)
    data,length=json.JSONDecoder().raw_decode(html[offset:])
    rows=json.loads((root/'baseline_results.json').read_text());lookup={r['id']:r for r in rows}
    assert len(data['runs'])==len(lookup)==1350
    assert len({r['id'] for r in data['runs']})==len(lookup)
    for run in data['runs']:
        expected=lookup[run['id']]
        assert run['success']==(expected['task_state_success']==1)
        folder=(source/'viewer'/unquote(run['folder'])).resolve()
        assert folder==Path(expected['run_folder']).resolve(),run['id']
        for key in ['trace_html','trace_txt','folder']:
            if run.get(key):
                target=(source/'viewer'/unquote(run[key])).resolve()
                assert target.is_relative_to(root) and target.exists(),str(target)
                run[key]=quote(os.path.relpath(target,dest),safe='/')
    csv_rows=list(csv.DictReader(io.StringIO((originals/'findings.csv').read_text())))
    actual=collections.Counter((r['id'],*f) for r in data['runs'] for f in r['findings'])
    expected=collections.Counter((r['run_id'],r['category'],r['label'],r['evidence']) for r in csv_rows)
    assert actual==expected,'Viewer and CSV disagree'
    imported=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    data['snapshot']={'imported_at':imported,'source_directory':str(source),'files':manifest,
        'matched_runs':len(lookup),'findings':sum(actual.values())}
    (dest/'import_manifest.json').write_text(json.dumps(data['snapshot'],indent=2))
    (dest/'rules_data.json').write_text(display_names(json.dumps(data,indent=2)))
    viewer=html[:offset]+json.dumps(data).replace('<','\\u003c')+html[offset+length:]
    viewer=viewer.replace('then open http://localhost:8010/garavindan3/uncertainty_classifier/viewer/rule_viewer.html', 'open this copied viewer through your repository HTTP server')
    (dest/'rule_viewer.html').write_text(display_names(viewer))
    # Dashboard URLs are relative to the batch HTML instead of the local viewer.
    for run in data['runs']:
        for key in ['trace_html','trace_txt','folder']:
            if run.get(key):
                target=(dest/unquote(run[key])).resolve()
                run[key]=quote(os.path.relpath(target,root),safe='/')
    (dest/'dashboard_data.json').write_text(display_names(json.dumps(data,indent=2)))
    (dest/'README.md').write_text('''# Local uncertainty-classifier snapshot

The complete original classifier folder is copied into `source/`, including
findings.csv, rule code, pseudocode, template, and original viewer. Source files
are preserved byte-for-byte; import_manifest.json records SHA-256 hashes.

Open `rule_viewer.html` for the relocated full viewer. Its trace and folder links
point to this batch in your workspace, without depending on the other user's
server or directory. `rules_data.json` holds all viewer data, definitions and
findings; `dashboard_data.json` uses links relative to baseline_dashboard.html.

The dashboard includes rule-based labels from this snapshot. Multiple labels can
apply to one run; these are heuristic findings, not verified causal attribution.
Success metrics remain the baseline evaluator's results. Import validation checks
all run IDs, latest-attempt folders, success flags and the findings.csv contents.
''')
    print(json.dumps({'destination':str(dest),'files':len(manifest),'runs':len(lookup),'findings':sum(actual.values())}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('root',type=Path);a=p.parse_args();import_rules(a.source,a.root)
