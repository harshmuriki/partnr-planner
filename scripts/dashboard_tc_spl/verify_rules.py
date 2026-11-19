import collections
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from urllib.parse import unquote
from playwright.sync_api import sync_playwright
from metrics import REPO
root=REPO/'results/all_baselines_5reps_20261005_001408'
snapshot=root/'uncertainty_classifier'
data=json.loads((snapshot/'dashboard_data.json').read_text())
assert len(data['runs'])==1350
assert sum(len(r['findings']) for r in data['runs'])==2673
for f in data['snapshot']['files']:
    assert hashlib.sha256((snapshot/'source'/f['path']).read_bytes()).hexdigest()==f['sha256']
for r in data['runs']:
    for key in ['folder','trace_html','trace_txt']:
        if r.get(key):
            path=(root/unquote(r[key])).resolve()
            assert path.is_relative_to(root) and path.exists()
class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(REPO)))
Thread(target=server.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{server.server_port}/results/{root.name}/'
try:
 with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':2300,'height':1200})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(base+'baseline_dashboard.html')
    page.wait_for_selector('#rule-findings .rule-chart')
    assert page.locator('#rule-findings .rule-chart').count()==2
    assert page.locator('.rule-evidence tbody tr').count()==1479

    for panel in page.locator('.finding-comparison .rule-chart').all():
        category=panel.get_attribute('data-category')
        means=panel.locator('.finding-row').evaluate_all('(els)=>els.map(e=>Number(e.dataset.mean))')
        assert means==sorted(means,reverse=True)
        for bar in panel.locator('.rule-bar').all():
            baseline=bar.get_attribute('data-baseline'); label=bar.get_attribute('data-label')
            eligible=[r for r in data['runs'] if r['baseline']==baseline and (category!='failure' or not r['success'])]
            count=sum(any(f[0]==category and f[1]==label for f in r['findings']) for r in eligible)
            assert int(bar.get_attribute('data-total'))==len(eligible)
            assert int(bar.get_attribute('data-n'))==count
            assert abs(float(bar.get_attribute('data-pct'))-100*count/len(eligible))<1e-8
    page.locator('.finding-comparison').screenshot(path=str(root/'grouped_findings_bars.png'))
    # Cross-category clicks select evidence without stale category filters.
    bar=page.locator('.finding-comparison [data-category="inefficiency"] .rule-bar[data-baseline="react"]').last
    label=bar.get_attribute('data-label');bar.click()
    assert page.get_by_label('Rules category',exact=True).input_value()=='inefficiency'
    assert page.get_by_label('Rules evidence baseline',exact=True).input_value()=='react'
    assert page.get_by_label('Rules finding',exact=True).input_value()==label
    assert page.locator('.rule-evidence tbody tr').count()>0
    page.get_by_label('Rules category',exact=True).select_option('failure')

    assert page.locator('.failure-sankey').count()==3
    for card in page.locator('.failure-sankey').all():
        assert int(card.get_attribute('data-total'))==450
        failed=int(card.get_attribute('data-failed'))
        assert failed+int(card.get_attribute('data-success'))==450
        counts=card.locator('.combination-branch').evaluate_all('(els)=>els.map(e=>Number(e.dataset.count))')
        assert sum(counts)==failed
    branch=page.locator('.combination-branch[data-kind="unlabeled"]').first
    count=int(branch.get_attribute('data-count'));branch.locator('text').click()
    assert page.locator('.combination-runs tbody tr').count()==count
    assert 'No failure finding' in page.locator('.combination-runs').inner_text()
    page.locator('.combination-details > button').click()
    branch=page.locator('.combination-branch[data-kind="other"]').first
    count=int(branch.get_attribute('data-count'));branch.focus();page.keyboard.press('Enter')
    assert page.locator('.combination-runs tbody tr').count()==count
    page.locator('.combination-details > button').click()
    page.locator('.failure-sankeys').screenshot(path=str(root/'failure_combinations_sankey.png'))
    page.get_by_label('Rules outcome',exact=True).select_option('successful')
    page.wait_for_function('document.querySelectorAll(".combination-branch").length===0')
    assert page.locator('.finding-comparison [data-category="failure"] .rule-bar').evaluate_all('(els)=>els.every(e=>e.dataset.total==="0" && e.dataset.pct==="")')
    page.get_by_label('Rules outcome',exact=True).select_option('all')
    page.wait_for_function('document.querySelectorAll(".combination-branch").length>0')

    for category,count in [('inefficiency',1194),('failure',1479)]:
        page.get_by_label('Rules category',exact=True).select_option(category)
        page.wait_for_function('(n)=>document.querySelectorAll(".rule-evidence tbody tr").length===n',arg=count)
    page.get_by_label('Rules category',exact=True).select_option('inefficiency')
    page.wait_for_function('document.querySelector(".failure-sankey").dataset.category==="inefficiency"')
    for card in page.locator('.failure-sankey').all():
        key=card.get_attribute('data-baseline')
        rs=[r for r in data['runs'] if r['baseline']==key]
        flagged=sum(any(f[0]=='inefficiency' for f in r['findings']) for r in rs)
        assert int(card.get_attribute('data-failed'))==flagged
        assert int(card.get_attribute('data-success'))==450-flagged
        counts=card.locator('.combination-branch').evaluate_all('(els)=>els.map(e=>Number(e.dataset.count))')
        assert sum(counts)==flagged
        kinds=card.locator('.combination-branch').evaluate_all('(els)=>els.map(e=>e.dataset.kind)')
        if 'other' in kinds:
            assert kinds[-1]=='other'
        named_counts=[n for n,k in zip(counts,kinds) if k!='other']
        assert named_counts==sorted(named_counts,reverse=True)
    branch=page.locator('.combination-branch[data-kind="exact"]').first
    count=int(branch.get_attribute('data-count'));branch.locator('text').click()
    assert page.locator('.combination-runs tbody tr').count()==count
    assert 'Exact inefficiency combination' in page.locator('.combination-runs thead').inner_text()
    page.locator('.combination-details > button').click()
    page.locator('.failure-sankeys').screenshot(path=str(root/'inefficiency_combinations_sankey.png'))
    page.get_by_label('Rules category',exact=True).select_option('failure')
    page.wait_for_function('document.querySelector(".failure-sankey").dataset.category==="failure"')
    page.get_by_label('Rules task',exact=True).select_option('T1')
    page.get_by_label('Rules memory',exact=True).select_option('OUT')
    selected=[r for r in data['runs'] if r['task']=='T1' and r['memory']=='OUT']
    expected=sum(sum(f[0]=='failure' for f in r['findings']) for r in selected)
    page.wait_for_function('(n)=>document.querySelectorAll(".rule-evidence tbody tr").length===n',arg=expected)
    page.locator('.rule-bar').first.click()
    assert page.get_by_label('Rules evidence baseline',exact=True).input_value()=='vlm-tamp'
    assert page.get_by_label('Rules finding',exact=True).input_value()
    page.get_by_label('Rules task',exact=True).select_option('all')
    page.get_by_label('Rules memory',exact=True).select_option('all')
    page.locator('#rule-findings').screenshot(path=str(root/'rule_findings_preview.png'))
    page.goto(base+'uncertainty_classifier/rule_viewer.html')
    page.wait_for_selector('#tbody tr')
    assert page.evaluate('DATA.runs.length')==1350
    page.locator('#tbody tr').first.click()
    page.locator('#traceBtn').click()
    page.wait_for_function('document.getElementById("traceBox").textContent.length>100')
    assert "Couldn't load" not in page.locator('#traceBox').inner_text()
    assert page.locator('#dHead a').first.get_attribute('href').startswith('../')
    assert not errors,errors
    browser.close()
finally:
 server.shutdown();server.server_close()
print('PASS: all 7 snapshot hashes, 1350 run links, 2673 findings, grouped bar counts and descending order, filters, rule drilldown and local viewer trace fetch.')
