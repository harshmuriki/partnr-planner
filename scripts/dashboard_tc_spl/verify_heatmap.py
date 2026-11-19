"""Check the 9x5 TC-SPL heatmap against per-run action-based scores."""
import json
from pathlib import Path
from statistics import fmean
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[2]/'results/all_baselines_5reps_20261005_001408'
data=json.loads((root/'tc_spl_data.json').read_text())
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':2300,'height':1400})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((root/'baseline_dashboard.html').as_uri())
    page.wait_for_selector('.heatmap-cell')
    assert page.locator('.memory-heatmap tbody tr').count()==9
    assert page.locator('.memory-heatmap .scenario-heading small').all_text_contents()==['BASE','SUB','ABS','CON','DIS']
    assert page.locator('.memory-heatmap .baseline-heading').count()==3
    assert page.locator('.memory-point').count()==63 # original line plots preserved
    for task in ['all',*[str(i) for i in range(1,8)]]:
        page.get_by_label('Heatmap task',exact=True).select_option(task)
        cells=page.locator('.heatmap-cell').evaluate_all('(els)=>els.map(e=>({...e.dataset,disabled:e.disabled,text:e.textContent}))')
        assert len(cells)==45
        assert len({(c['baseline'],c['memory'],c['scenario']) for c in cells})==45
        selected=[r for r in data['rows'] if task=='all' or r['task']==int(task)]
        assert sum(int(c['total']) for c in cells)==len(selected)
        for cell in cells:
            rows=[r for r in selected if r['condition']==cell['baseline'] and r['memory']==cell['memory'] and r['uncertainty']==cell['scenario']]
            values=[r['tcae'] for r in rows if r['tcae'] is not None]
            assert int(cell['total'])==len(rows) and int(cell['n'])==len(values)
            assert cell['disabled']==(len(rows)==0)
            if values:
                assert abs(float(cell['value'])-fmean(values))<1e-12
            else:
                assert cell['value']=='' and cell['text']=='—'
    page.get_by_label('Heatmap task',exact=True).select_option('all')
    cell=page.locator('.heatmap-cell').first
    expected=int(cell.get_attribute('data-total'))
    cell.focus();page.keyboard.press('Enter')
    assert page.locator('.heatmap-runs tbody tr').count()==expected
    page.get_by_role('button',name='Close heatmap details',exact=True).click()
    page.locator('#memory-scenario-heatmap').screenshot(path=str(root/'tc_spl_heatmap.png'))
    cell.click()
    page.get_by_label('Heatmap task',exact=True).select_option('7')
    assert page.locator('.heatmap-details').count()==0
    assert not errors,errors
    browser.close()
print('PASS: 360 heatmap cells across all task selections, unique cohorts, action-based scores, missing values, and keyboard drilldown')
