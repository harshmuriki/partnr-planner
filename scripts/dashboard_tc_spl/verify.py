import json
import re
from pathlib import Path
from statistics import fmean
from metrics import REPO, read_gt, score, action_count, action_efficiency
from playwright.sync_api import sync_playwright

root=REPO/'results/all_baselines_5reps_20261005_001408'
data=json.loads((root/'tc_spl_data.json').read_text())
assert len(data['rows'])==1350
assert all(r['tcspl'] is not None for r in data['rows'])
assert score(.5,0,200,100)==(.5,.25,0)
assert score(.5,0,0,100)==(1,.5,0)
assert score(1,1,None,100)==(None,None,None)
assert score(1,1,200,100)==(.5,.5,.5)
assert action_count({'Rearrange':2,'Navigate':1,'Explore':1,'Wait':1,'Done':3,'ReportAbsence':1})==11
assert action_efficiency(.75,20,10)==.375
assert action_efficiency(.75,0,10)==.75
assert action_efficiency(.75,None,10) is None
assert action_efficiency(.75,10,0) is None
refs=read_gt()
assert sum(r['fallback'] for r in refs.values())==2
# Independently reproduce the old chart with these same GT references.
old=REPO/'results/final_baselines_html'
expected=json.loads(re.search(r'const DATA = (.*);',(old/'tc_spl.html').read_text()).group(1))
oldrows=json.loads((old/'baseline_runs_comparison.json').read_text())
for key,stats in expected['overall'].items():
    rows=[r for r in oldrows if r['condition']==key]
    values=[(r['completion'] or 0)*refs[r['task_id']]['steps']/max(r['sim_steps'] or refs[r['task_id']]['steps'],refs[r['task_id']]['steps']) for r in rows]
    assert abs(fmean(values)-stats['tcspl'])<1e-12
for r in data['rows']:
    assert abs(r['tcae']-r['completion']*r['reference_actions']/max(r['action_count'],r['reference_actions']))<1e-12
    assert (root/r['gt_action_url']).is_file()
    assert abs(r['tcspl']-r['completion']*r['reference_steps']/max(r['sim_steps'],r['reference_steps']))<1e-12
    assert (root/r['gt_source_url']).is_file()
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1100})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((root/'baseline_dashboard.html').as_uri(),wait_until='load')
    page.wait_for_selector('#tc-spl-root .tc-overall tbody tr')
    assert page.locator('.tc-overall tbody tr').count()==3
    assert 'TC-AE' not in page.locator('#tc-spl-root').inner_text()
    assert page.locator('select[aria-label="Breakdown metric"] option[value="tc"]').count()==0
    assert 'TC-Steps' not in page.locator('.tc-panel').inner_text()
    assert page.locator('select[aria-label="Breakdown metric"] option[value="tca"]').text_content()=='TC-SPL'
    assert data['metric_labels']['tcspl']=='TC-Steps' and data['metric_labels']['tcae']=='TC-SPL' 
    assert page.locator('.tc-breakdown tbody tr').count()==9
    assert page.locator('.breakdown-line').count()==3
    assert page.locator('.breakdown-lines .group-label').all_text_contents()==['Accurate','Incomplete','Outdated']
    assert page.locator('#sankeySection').count()==0
    assert page.locator('#cards .card').first.evaluate('(el)=>getComputedStyle(el).borderTopColor')=='rgb(0, 114, 178)'
    assert page.locator('.memory-line').count()==9
    assert page.locator('.tc-panel .memory-comparison').count()==0
    assert page.locator('#scenario-memory > h2').count()==1
    page.get_by_label('Completion task',exact=True).select_option('7')
    assert page.get_by_label('Memory line chart task',exact=True).input_value()=='all'
    assert page.locator('.memory-point').count()==63
    page.get_by_label('Completion task',exact=True).select_option('all')
    assert page.locator('.baseline-memory-chart h4').all_text_contents()==[b['label'] for b in data['baselines']]
    boxes=page.locator('.baseline-memory-chart').evaluate_all('(els)=>els.map(e=>({top:e.getBoundingClientRect().top,left:e.getBoundingClientRect().left}))')
    assert len({b['top'] for b in boxes})==1
    assert boxes[0]['left']<boxes[1]['left']<boxes[2]['left']
    assert page.locator('main').evaluate('(el)=>el.getBoundingClientRect().width')>=2200
    assert page.locator('.baseline-memory-chart').first.evaluate('(el)=>el.getBoundingClientRect().width')>=650
    assert page.locator('.memory-line-table').first.locator('thead th').all_text_contents()==['Scenario','ACC','INC','OUT']
    table_boxes=page.locator('.memory-table-card').evaluate_all('(els)=>els.map(e=>({top:e.getBoundingClientRect().top,left:e.getBoundingClientRect().left}))')
    assert len(table_boxes)==3 and len({b['top'] for b in table_boxes})==1
    assert table_boxes[0]['left']<table_boxes[1]['left']<table_boxes[2]['left']
    assert page.locator('.memory-line-table').first.locator('tbody th').all_text_contents()==['Target Exists','Substitute Available','No Suitable Object','On Surface','Inside Closed Receptacle','None','Present']
    page.locator('.memory-tables').screenshot(path=str(root/'compact_memory_tables.png'))
    for key,field,label in [('sr','success','Task success'),('spl','spl','SPL'),('cr','completion','Task completion'),('tc','tcspl','TC-Steps')]:
        page.get_by_label('Memory line chart metric',exact=True).select_option(key)
        page.wait_for_function('(label)=>document.querySelector(".memory-lines title").textContent.startsWith(label+" by")',arg=label)
        assert page.locator('.memory-comparison h2').inner_text().startswith(label)
        points=page.locator('.memory-point').evaluate_all('(els)=>els.map(e=>({...e.dataset}))')
        assert len(points)==63
        for point in points:
            rs=[r for r in data['rows'] if r['condition']==point['baseline'] and r['memory']==point['memory'] and r['uncertainty']==point['scenario']]
            assert abs(float(point['value'])-fmean(r[field] for r in rs))<1e-12
    for key,field in [('tca','tcae'),('runtime','runtime_s'),('actions','num_actions'),('planning','llm_planning_time_s'),('cost','llm_usd'),('sub',None)]:
        page.get_by_label('Memory line chart metric',exact=True).select_option(key)
        page.wait_for_timeout(50)
        points=page.locator('.memory-point').evaluate_all('(els)=>els.map(e=>({...e.dataset}))')
        assert len(points)==63
        for point in points:
            rs=[r for r in data['rows'] if r['condition']==point['baseline'] and r['memory']==point['memory'] and r['uncertainty']==point['scenario']]
            if field:
                expected=fmean(r[field] for r in rs if r[field] is not None)
            else:
                rs=[r for r in rs if r['subtasks_succeeded'] is not None and r['num_subtasks']>0]
                expected=sum(r['subtasks_succeeded'] for r in rs)/sum(r['num_subtasks'] for r in rs)
            assert abs(float(point['value'])-expected)<1e-9
        axes=page.locator('.memory-lines').evaluate_all('(els)=>els.map(e=>Array.from(e.querySelectorAll(".axis-text")).map(t=>t.textContent).join("|"))')
        assert len(set(axes))==1
        positions=page.locator('.memory-point circle').evaluate_all('(els)=>els.map(e=>Number(e.getAttribute("cy")))')
        assert all(49<=v<=331 for v in positions)
    page.get_by_label('Memory line chart metric',exact=True).select_option('tc')
    page.get_by_label('Memory line chart task',exact=True).select_option('7')
    page.wait_for_function('document.querySelectorAll(".memory-point").length===36')
    assert sum(t.count('n=0') for t in page.locator('.memory-line-table').all_text_contents())==27
    page.get_by_label('Memory line chart task',exact=True).select_option('all')
    page.wait_for_function('document.querySelectorAll(".memory-point").length===63')
    page.locator('.memory-comparison').screenshot(path=str(root/'tc_spl_memory_lines.png'))
    assert page.locator('.metric-value-bar').count()==12
    assert page.locator('.metric-y-label').text_content()=='Score (%)'
    assert page.get_by_label('Breakdown metric',exact=True).input_value()=='tca'
    assert page.locator('.breakdown-y-label').text_content()=='TC-SPL (%)'
    heights=page.locator('.metric-value-bar').evaluate_all('(els)=>els.map(e=>Number(e.getAttribute("height")))')
    ordered=[k for k,_ in __import__('metrics').BASELINES]
    for metric_index,metric in enumerate(['success','spl','completion','tcae']):
        for i,key in enumerate(ordered):
            selected_rows=[r for r in data['rows'] if r['condition']==key]
            expected_height=280*fmean(r[metric] for r in selected_rows)
            assert abs(heights[metric_index*3+i]-expected_height)<1e-9
    page.locator('.metric-bar').first.click()
    assert page.locator('.tc-runs tbody tr').count()==450
    page.locator('.drilldown > button').click()
    page.locator('.metric-comparison').first.screenshot(path=str(root/'tc_spl_vertical_bars.png'))
    assert '59.1%' in page.locator('.tc-overall').inner_text()
    for key,field in [('sr','success'),('cr','completion'),('spl','spl'),('tca','tcae')]:
        page.get_by_label('Breakdown metric',exact=True).select_option(key)
        bars=page.locator('.breakdown-point').evaluate_all('(els)=>els.map(e=>({...e.dataset}))')
        assert len(bars)==9
        for bar in bars:
            values=[r[field] for r in data['rows'] if r['condition']==bar['baseline'] and r['memory']==bar['group'] and r[field] is not None]
            assert abs(float(bar['value'])-fmean(values))<1e-10
        assert page.locator('.tc-breakdown thead .emphasis').count()==1
    page.locator('.breakdown-point').first.click()
    assert page.locator('.tc-runs tbody tr').count()==150
    page.locator('.drilldown > button').click()
    page.get_by_label('Breakdown metric',exact=True).select_option('tca')
    page.get_by_label('Completion task',exact=True).select_option('1')
    page.get_by_label('Completion memory',exact=True).select_option('ACC')
    page.get_by_label('Completion scenario',exact=True).select_option('BASE')
    page.locator('.tc-overall button').first.click()
    assert page.locator('.tc-runs tbody tr').count()==5
    page.get_by_label('Completion grouping',exact=True).select_option('memory')
    assert page.locator('.tc-breakdown tbody tr').count()==3
    assert page.locator('.tc-runs').count()==0
    for name in ['task','memory','scenario']:
        page.get_by_label('Completion '+name,exact=True).select_option('all')
    page.get_by_label('Completion grouping',exact=True).select_option('task')
    assert page.locator('.breakdown-point').count()==21
    page.get_by_label('Completion grouping',exact=True).select_option('uncertainty')
    assert page.locator('.breakdown-point').count()==21
    assert page.locator('.breakdown-lines .group-label').evaluate_all('(els)=>els.map(e=>e.dataset.label)')==['Target Exists','Substitute Available','No Suitable Object','On Surface','Inside Closed Receptacle','None','Present']
    for b in data['baselines']:
        points=page.locator('.breakdown-point[data-baseline="'+b['key']+'"]').evaluate_all('(els)=>els.map(e=>({...e.dataset}))')
        assert [p['group'] for p in points]==['BASE','SUB','ABS','BASE','CON','BASE','DIS']
        assert points[0]['value']==points[3]['value']==points[5]['value']
    page.locator('.breakdown-lines').screenshot(path=str(root/'completion_scenario_lines.png'))
    page.get_by_label('Completion grouping',exact=True).select_option('memory')
    page.get_by_label('Breakdown metric',exact=True).select_option('tca')
    page.locator('.breakdown-lines').screenshot(path=str(root/'completion_breakdown_lines.png'))
    page.locator('#tc-spl-root').screenshot(path=str(root/'tc_spl_preview.png'))
    page.set_viewport_size({'width':390,'height':844})
    assert page.locator('.tc-overall').is_visible()
    mobile_boxes=page.locator('.baseline-memory-chart').evaluate_all('(els)=>els.map(e=>e.getBoundingClientRect().top)')
    assert mobile_boxes[0]<mobile_boxes[1]<mobile_boxes[2]
    assert not errors,errors
    browser.close()
print('PASS: source provenance, old-score reproduction, 1350 formulas, zero/missing steps, Svelte filters, drilldowns, and removed Sankey section.')
