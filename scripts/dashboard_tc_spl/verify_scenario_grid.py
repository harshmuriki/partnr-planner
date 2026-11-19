"""Validate every scenario-grid cell against saved run metrics and check interactions."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2]/'results/all_baselines_5reps_20261005_001408'
data=json.loads((ROOT/'tc_spl_data.json').read_text())
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':2300,'height':1400})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((ROOT/'baseline_dashboard.html').as_uri())
    page.wait_for_selector('.scenario-cell')
    assert page.locator('.scenario-card').count()==3
    page.get_by_label('Scenario grid task',exact=True).select_option('all')
    for metric in ['success','completion']:
        page.get_by_label('Scenario grid metric',exact=True).select_option(metric)
        page.wait_for_function('document.querySelectorAll(".scenario-card").length===21')
        cells=page.locator('.scenario-cell').evaluate_all('''els=>els.map(e=>({task:Number(e.closest('.scenario-card').dataset.task),baseline:e.closest('.scenario-card').dataset.baseline,...e.dataset,disabled:e.disabled}))''')
        assert len(cells)==441
        for c in cells:
            rows=[r for r in data['rows'] if r['task']==c['task'] and r['condition']==c['baseline'] and r['uncertainty']==c['scenario'] and r['memory']==c['memory']]
            values=[(int(r['success']==1) if metric=='success' else r['completion']) for r in rows if r[metric] is not None]
            assert int(c['total'])==len(rows) and int(c['n'])==len(values)
            assert c['disabled']==(len(rows)==0)
            if values:
                assert abs(float(c['value'])-sum(values)/len(values))<1e-10
            else:
                assert c['value']==''
        # Every repetition appears once per variant; repeated BASE columns agree.
        print(f'PASS: {metric}, all 441 displayed cells match saved metrics')
    page.get_by_label('Scenario grid task',exact=True).select_option('1')
    first=page.locator('.scenario-cell').first
    first.click()
    assert page.locator('.grid-runs tbody tr').count()==5
    assert page.locator('.grid-runs a').first.get_attribute('href')
    page.get_by_role('button',name='Close details',exact=True).last.click()
    page.locator('#task-scenario-grid').screenshot(path=str(ROOT/'task_scenario_grid.png'))
    page.get_by_label('Scenario grid metric',exact=True).select_option('success')
    assert page.locator('.grid-details').count()==0
    assert not errors,errors
    browser.close()
print('PASS: repetition drilldown, metric switching, missing variants, no browser errors')
