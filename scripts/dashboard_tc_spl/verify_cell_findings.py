"""Check proportional cell findings against distinct saved run labels."""
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[2]/'results/all_baselines_5reps_20261005_001408'
data=json.loads((root/'tc_spl_data.json').read_text())
annotations={r['id']:r for r in data['rule_findings']['runs']}
cohorts=defaultdict(list)
for r in data['rows']:
    cohorts[r['task'],r['condition'],r['memory'],r['uncertainty']].append(r)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':2300,'height':1400})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((root/'baseline_dashboard.html').as_uri())
    page.wait_for_selector('.finding-strip')
    page.get_by_label('Scenario grid task',exact=True).select_option('all')
    for category in ['failure','inefficiency']:
        page.get_by_label('Scenario grid findings',exact=True).select_option(category)
        cells=page.locator('.scenario-cell').evaluate_all('''els=>els.map(e=>({task:Number(e.closest('.scenario-card').dataset.task),baseline:e.closest('.scenario-card').dataset.baseline,...e.dataset,eligible:e.querySelector('.finding-strip')?.dataset.eligible,segments:Array.from(e.querySelectorAll('.finding-segment')).map(s=>({...s.dataset}))}))''')
        assert len(cells)==441
        for cell in cells:
            rows=cohorts[cell['task'],cell['baseline'],cell['memory'],cell['scenario']]
            if not rows:
                assert cell.get('eligible') is None
                continue
            eligible=[r for r in rows if category!='failure' or (r['success'] is not None and r['success']!=1)]
            assert int(cell['eligible'])==len(eligible)
            expected=defaultdict(Fraction);counts=Counter()
            for r in eligible:
                a=annotations.get(r['id'])
                labels=({f[1] for f in a['findings'] if f[0]==category} or {'__none'}) if a else {'__missing'}
                for label in labels:
                    expected[label]+=Fraction(1,len(labels));counts[label]+=1
            actual={s['label']:s for s in cell['segments']}
            assert set(actual)==set(expected)
            for label,weight in expected.items():
                assert abs(float(actual[label]['share'])-float(weight/len(eligible)))<1e-12
                assert int(actual[label]['count'])==counts[label]
            if eligible:
                assert abs(sum(float(s['share']) for s in cell['segments'])-1)<1e-12
        print('PASS:',category,'shares and counts for all 441 cells')
    page.get_by_label('Scenario grid task',exact=True).select_option('1')
    page.locator('#task-scenario-grid').screenshot(path=str(root/'scenario_inefficiency_cells.png'))
    page.locator('.scenario-cell').first.click()
    assert page.locator('.grid-runs .cell-evidence').count()==5
    page.get_by_label('Scenario grid findings',exact=True).select_option('failure')
    assert page.locator('.grid-details').count()==0
    page.locator('#task-scenario-grid').screenshot(path=str(root/'scenario_failure_cells.png'))
    page.get_by_label('Scenario grid findings',exact=True).select_option('off')
    assert page.locator('.finding-strip').count()==0
    assert page.locator('.finding-key').count()==0
    assert page.locator('.scenario-cell').count()==63
    assert not errors,errors
    browser.close()
print('PASS: selectors, detail evidence, empty cells, and 100% conservation with overlapping labels')
