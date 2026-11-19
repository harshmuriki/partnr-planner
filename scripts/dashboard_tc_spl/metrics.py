"""Auditable human-reference action- and step-weighted metrics for saved rows."""
import argparse
import csv
import io
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import fmean
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[2]
BASELINES = [('vlm_tamp_images', 'VLM-TAMP'), ('react_images', 'ReAct (images)'), ('tru_pomdp_no_images', 'TRU-POMDP')]

def action_count(counts):
    """Comparable recorded skill attempts; communication actions are excluded."""
    return sum(n * (4 if skill == 'Rearrange' else 1)
               for skill,n in counts.items() if skill not in {'Done','ReportAbsence'})

def action_efficiency(completion, actions, reference_actions):
    if completion is None or actions is None or reference_actions is None or reference_actions <= 0:
        return None
    if not (0 <= completion <= 1 and math.isfinite(actions) and actions >= 0):
        raise ValueError('Invalid completion or action count')
    return completion * reference_actions / max(actions, reference_actions)

def read_gt():
    directory = REPO / 'baseline_evaluation_v3/ground_truth_timing'
    refs = {}
    # This table selects the successful replay (including retry) for each variant.
    for tr in re.findall(r'<tr>(.*?)</tr>', (directory/'combined.html').read_text(), re.S):
        cells = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)
        if not cells or not re.fullmatch(r'T\d+-[A-Z]+-[A-Z]+', cells[0]):
            continue
        variant = cells[0]
        path = (directory / cells[8]).resolve()
        assert path.is_relative_to(directory)
        raw = path.read_bytes()
        replay = json.loads(raw)
        assert replay['variant'] == variant
        fallback = cells[1] != 'replayed'
        if fallback:
            assert variant in {'T2-ACC-DIS', 'T4-ACC-ABS'}, 'Unexpected failed GT replay'
        else:
            assert replay['status'] == 'replayed' and replay['evaluation']['success']
        steps = replay['saved_sim_steps'] if fallback else replay['sim_steps']
        assert steps == float(cells[5] if fallback else cells[4]) and steps > 0
        archive = REPO / f'baseline_evaluation_v3/ground_truth/{variant.split("-")[0]}/{variant}/run.json'
        action_source = archive.with_name('actions.csv') if fallback else path
        action_raw = action_source.read_bytes()
        action_records = list(csv.DictReader(io.StringIO(action_raw.decode()))) if fallback else replay['actions']
        action_counts = dict(Counter(a['skill'] for a in action_records))
        refs[variant] = {'actions':action_count(action_counts), 'action_counts':action_counts,
            'action_source':str(action_source.relative_to(REPO)), 'action_sha256':hashlib.sha256(action_raw).hexdigest(), 'steps':steps,'source':str(path.relative_to(REPO)),
            'sha256':hashlib.sha256(raw).hexdigest(), 'field':'saved_sim_steps' if fallback else 'sim_steps',
            'archive':str(archive.relative_to(REPO)), 'fallback':fallback}
    assert len(refs) == 90
    return refs

def score(completion, success, steps, reference):
    if completion is None or success is None or steps is None:
        return None, None, None
    if not (math.isfinite(steps) and steps >= 0 and 0 <= completion <= 1 and reference > 0):
        raise ValueError('Invalid completion or step count')
    ratio = reference / max(steps, reference)
    return ratio, completion * ratio, float(success == 1) * ratio

def payload(rows, root):
    refs = read_gt()
    records = []
    for row in rows:
        ref = refs[row['task_id']]
        success = row.get('task_state_success')
        if success is None:
            success = (row.get('subtask_results',{}).get('metrics') or {}).get('task_state_success')
        ratio, tc, spl_s = score(row['completion'],success,row['sim_steps'],ref['steps'])
        normalized_actions = action_count(row['counts']) if row.get('num_actions') is not None else None
        spl = action_efficiency(None if success is None else float(success == 1), normalized_actions, ref['actions'])
        tca = action_efficiency(row['completion'], normalized_actions, ref['actions'])
        records.append({**{k:row[k] for k in ['id','condition','task','task_id','memory','uncertainty','iteration','execution_status','completion','sim_steps','folder_url']},
            **{k:row.get(k) for k in ['runtime_s','num_actions','llm_planning_time_s','llm_usd','subtasks_succeeded','num_subtasks']},
            'action_count':normalized_actions, 'reference_actions':ref['actions'], 'tcae':tca,
            'gt_action_url':quote(str(Path(__import__('os').path.relpath(REPO/ref['action_source'],root))),safe='/'),
            'success':success, 'reference_steps':ref['steps'], 'efficiency':ratio,'tcspl':tc,'spl':spl,'spl_s':spl_s,
            'gt_source_url':quote(str(Path(__import__('os').path.relpath(REPO/ref['source'],root))),safe='/'),
            'gt_archive_url':quote(str(Path(__import__('os').path.relpath(REPO/ref['archive'],root))),safe='/'),
            'gt_field':ref['field'], 'gt_fallback':ref['fallback']})
    action_summary = {key:{'runs':len(rs),'scored':len(values),'mean_tcae':fmean(values) if values else None}
        for key,_ in BASELINES
        for rs in [[r for r in records if r['condition']==key]]
        for values in [[r['tcae'] for r in rs if r['tcae'] is not None]]}
    return {'action_summary':action_summary, 'baselines':[{'key':k,'label':v} for k,v in BASELINES], 'rows':records,'references':refs,
        'metric_labels':{'tcspl':'TC-Steps','tcae':'TC-SPL','spl':'SPL','spl_s':'SPL_S'},
        'spl_definition':'SPL = mean S_i * A_ref_i / max(A_i,A_ref_i); SPL_S = mean S_i * L_i / max(P_i,L_i). S_i is binary evaluator success.',
        'action_definition':'TC-SPL = mean C_i * A_ref_i / max(A_i,A_ref_i). Count recorded skill attempts including failures and interruptions; Explore and Wait each count once; exclude Done and ReportAbsence; expand Rearrange to 2 Navigate + Pick + Place. Human references are not assumed optimal.',
        'definition':'TC-Steps = mean C_i * L_i / max(P_i,L_i); L is human reference simulator steps; P is recorded agent simulator steps.',
        'missing_policy':'Unavailable inputs are excluded from step-weighted averages and coverage is reported. Zero recorded steps are valid; saved completion at timeout is retained.'}

def inject(html, rows, root):
    html = html.replace('vlm_tamp_images:"#16a34a", react_images:"#dc2626", tru_pomdp_no_images:"#eab308"', 'vlm_tamp_images:"#0072B2", react_images:"#D55E00", tru_pomdp_no_images:"#CC79A7"')
    data = payload(rows,root)
    rules = root/'uncertainty_classifier/dashboard_data.json'
    if rules.exists():
        data['rule_findings'] = json.loads(rules.read_text())
    text = json.dumps(data).replace('<','\\u003c')
    bundle = (Path(__file__).parent/'bundle.js').read_text().replace('</script','<\\/script')
    block = '<!-- TC-SPL START --><section id="tc-spl-root"></section><script type="application/json" id="tc-spl-data">'+text+'</script><script>'+bundle+'</script><!-- TC-SPL END -->'
    html = re.sub(r'<!-- TC-SPL START -->.*?<!-- TC-SPL END -->','',html,flags=re.S)
    # Remove superseded legacy views and their DOM event handlers on every build.
    html = re.sub(r'<section>\s*<h2>Breakdowns</h2>.*?</section>', '', html, flags=re.S)
    html = re.sub(r'// Breakdowns.*?(?=// Image ablation)', '', html, flags=re.S)
    html = html.replace('legend($("legend2"));', '')
    html = re.sub(r'<section>\s*<h2>Overall comparison</h2>.*?</section>', '', html, flags=re.S)
    html = re.sub(r'\$\("overallCharts"\)\.innerHTML=.*?(?=// Image ablation)', '', html, flags=re.S)
    html = html.replace('legend($("legend1"));', '')
    html = re.sub(r'<section id="subtaskSection">.*?</section>', '', html, flags=re.S)
    html = re.sub(r'<section>\s*<h2>Per-variant results</h2>.*?</section>', '', html, flags=re.S)
    html = re.sub(r'<section id="detail">.*?</section>', '', html, flags=re.S)
    html = re.sub(r'// Per-variant table.*?(?=\$\("notes"\)\.innerHTML)', '', html, flags=re.S)
    marker = '<section class="notes">'
    assert marker in html
    tmp=root/'tc_spl_data.json.tmp';tmp.write_text(json.dumps(data,indent=2));tmp.replace(root/'tc_spl_data.json')
    return html.replace(marker,block+'\n'+marker,1)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();root=args.root.resolve()
    rows=json.loads((root/'baseline_results.json').read_text())
    html=inject((root/'baseline_dashboard.html').read_text(),rows,root)
    tmp=root/'baseline_dashboard.html.tmp';tmp.write_text(html);tmp.replace(root/'baseline_dashboard.html')
    data=json.loads((root/'tc_spl_data.json').read_text())
    for key,label in BASELINES:
        rs=[r for r in data['rows'] if r['condition']==key];valid=[r for r in rs if r['tcspl'] is not None]
        print(label,len(rs),'coverage',len(valid),'TC-Steps',fmean(r['tcspl'] for r in valid),'SPL',fmean(r['spl'] for r in valid),'TC-SPL',data['action_summary'][key])
