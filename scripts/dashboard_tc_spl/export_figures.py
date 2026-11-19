import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
from statistics import mean
from playwright.sync_api import sync_playwright
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('root',type=Path)
parser.add_argument('output',type=Path)
parser.add_argument('--overwrite',action='store_true')
parser.add_argument('--scale',type=int,default=4)
args=parser.parse_args()
root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=args.overwrite)
data=json.loads((root/'tc_spl_data.json').read_text())
manifest=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch()
 page=browser.new_page(viewport={'width':2300,'height':1400})
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((root/'baseline_dashboard.html').as_uri())
 page.wait_for_selector('.metric-picker input')
 styles=page.locator('style').evaluate_all('(els)=>els.map(e=>e.outerHTML).join("\\n")')
 for label in ['Completion task','Completion memory','Completion scenario','Memory line chart task','Heatmap task','Rules task','Rules memory','Rules scenario','Rules outcome']:
  page.get_by_label(label,exact=True).select_option('all')
 for key in ['sr','spl','cr','tca']:
  page.locator(f'.metric-picker input[value="{key}"]').check()
 page.get_by_label('Memory line chart metric',exact=True).select_option('tca')

 def save(stem,title,selector,width,remove=(),heading=False):
  node=page.locator(selector)
  content=node.evaluate('''(el,remove)=>{const c=el.cloneNode(true);for(const sel of remove)c.querySelectorAll(sel).forEach(n=>n.remove());return c.outerHTML;}''',list(remove))
  ep=browser.new_page(viewport={'width':width,'height':1400},device_scale_factor=args.scale)
  heading_html=f'<h2 class="export-title">{html.escape(title)}</h2>' if heading else ''
  document='<!doctype html><html><head><meta charset="utf-8">'+styles+'''<style>
  html,body{margin:0!important;padding:0!important;min-width:0!important;background:white!important;color:#172033;-webkit-print-color-adjust:exact;print-color-adjust:exact}
  *{animation:none!important;transition:none!important;box-shadow:none!important;text-shadow:none!important}
  #figure .track{background:#f3f6fa!important}#figure{box-sizing:border-box;padding:20px;background:white;width:100%}
  #figure>section,#figure>article,#figure>div{margin:0 auto!important;width:100%;box-sizing:border-box}
  #figure .efficiency-comparison,#figure .breakdown-lines{max-width:none!important}
  .export-title{font:600 24px Arial,sans-serif;text-align:center;margin:6px 0 16px}
  #figure .heatmap-scroll{max-width:none}#figure .heatmap-section{padding:16px}
  #figure .rule-chart{border:1px solid #dbe3ef;border-radius:10px;padding:22px}
  @page{margin:0}
  </style></head><body><div id="figure">'''+heading_html+content+'</div></body></html>'
  ep.set_content(document);ep.evaluate('document.fonts.ready')
  loc=ep.locator('#figure');box=loc.bounding_box();height=int(box['height']+1)
  assert loc.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1'),stem
  png_path=out/(stem+'.png')
  loc.screenshot(path=str(png_path))
  with Image.open(png_path) as im:
   pixels=list(im.size)
   im.save(png_path,dpi=(600,600),optimize=True)
  ep.emulate_media(media='screen')
  ep.pdf(path=str(out/(stem+'.pdf')),width=f'{width}px',height=f'{height}px',print_background=True,margin={'top':'0','bottom':'0','left':'0','right':'0'})
  # Keep a self-contained source for lossless re-export and text selection.
  (out/(stem+'.html')).write_text(document)
  manifest.append({'name':stem,'title':title,'width':width,'height':height,'pixel_dimensions':pixels,'raster_scale':args.scale,'png_dpi':600,'png':stem+'.png','pdf':stem+'.pdf','html':stem+'.html'})
  print('Saved',stem,flush=True)
  ep.close()

 save('01_success_efficiency_completion','Success, efficiency, and completion','.percentage-panel',1150)
 page.get_by_label('Completion grouping',exact=True).select_option('task')
 page.get_by_label('Breakdown metric',exact=True).select_option('spl')
 save('02_spl_by_task','SPL by task across three baselines','.breakdown-lines',1150,heading=True)
 page.get_by_label('Completion grouping',exact=True).select_option('memory')
 page.get_by_label('Breakdown metric',exact=True).select_option('tca')
 save('03_tc_spl_by_initial_memory','TC-SPL by initial memory across three baselines','.breakdown-lines',1150,heading=True)
 save('04_tc_spl_scenario_memory_three_baselines','TC-SPL by scenario and initial memory','#scenario-memory',2200,remove=['.controls','.memory-tables','.memory-methods','.memory-subtitle'])
 save('05_tc_spl_heatmap','TC-SPL by baseline, scenario, and initial memory','#memory-scenario-heatmap',1450,remove=['.controls>label','.definition'])
 save('06_failure_findings','Failure findings across baselines','.finding-comparison [data-category="failure"]',1150)
 save('07_inefficiency_findings','Inefficiency findings across baselines','.finding-comparison [data-category="inefficiency"]',1150)
 assert len(manifest)==7
 assert not errors,errors
 browser.close()

rows=data['rows']
with (out/'metric_percentages.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['grouping','group','baseline','runs','task_success_percent','SPL_percent','task_completion_percent','TC_SPL_action_percent'])
 for grouping,values in [('overall',['all']),('task',sorted(set(r['task'] for r in rows))),('memory',['ACC','INC','OUT'])]:
  for group in values:
   for baseline in data['baselines']:
    rs=[r for r in rows if r['condition']==baseline['key'] and (grouping=='overall' or r[grouping]==group)]
    scores=[]
    for field in ['success','spl','completion','tcae']:
     vals=[r[field] for r in rs if r.get(field) is not None]
     scores.append(100*mean(vals) if vals else '')
    writer.writerow([grouping,group,baseline['label'],len(rs),*scores])
source_files=['baseline_dashboard.html','tc_spl_data.json','uncertainty_classifier/dashboard_data.json']
metadata={'figures':manifest,'source_root':str(root),'filters':{'task':'all','memory':'all','scenario':'all','repetitions':'all five; latest attempt only'},'metric_definition':{'SPL':'binary success weighted by high-level action efficiency (spl)','TC-SPL':'completion weighted by high-level action efficiency (historical tcae)'},'sources':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source_files}}
(out/'manifest.json').write_text(json.dumps(metadata,indent=2))
(out/'README.md').write_text('''# Dashboard figures

Exactly seven figures, each supplied as a high-resolution PNG, vector PDF, and self-contained HTML source. PNGs render at 4x resolution (4,600–8,800 pixels wide) with 600-DPI metadata. PDFs retain vector text and chart geometry for sharp scaling; raster-producing shadows and repeating track backgrounds are removed in these exports. Open index.html for previews and downloads.

All tasks, all memory conditions, all scenarios, all five repetitions; newest saved attempt per run. There are 1,350 runs (450 per baseline). No simulator or model calls were made.

1. Success, efficiency, and completion: task success, SPL, task completion, and TC-SPL.
2. SPL by task: simulator-step-weighted binary success, compared across baselines.
3. Initial memory versus TC-SPL: action-based TC-SPL, compared across baselines.
4. TC-SPL by scenario and initial memory: three baseline line charts combined into one figure.
5. TC-SPL heatmap: 9 baseline/memory rows by 5 unique scenarios.
6. Failure findings: percent of unsuccessful runs with each finding.
7. Inefficiency findings: percent of all selected runs with each finding.

TC-SPL means action-based completion efficiency (JSON tcae); TC-Steps is not exported. SPL uses high-level action counts; the step-based variant is SPL_S (spl_s). The scenario line charts repeat BASE at Target Exists, On Surface, and None; these are the same runs, not extra observations. Findings overlap and their percentages need not sum to 100%; they describe detected findings rather than proven causes.

metric_percentages.csv reports overall, per-task, and per-memory percentages. Missing metric values are excluded rather than replaced with zero. manifest.json records the source hashes and export dimensions.
''')
items=''.join(f'<article><h2>{html.escape(m["title"])}</h2><a href="{m["pdf"]}">PDF</a> · <a href="{m["png"]}">PNG</a> · <a href="{m["html"]}">HTML</a><img src="{m["png"]}" alt="{html.escape(m["title"])}"></article>' for m in manifest)
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Seven dashboard figures</title><style>body{font:16px Arial;background:#f5f7fa;margin:30px auto;max-width:1600px;color:#172033}article{background:white;padding:24px;margin:24px 0;border-radius:12px}img{display:block;width:100%;height:auto;margin-top:16px}a{color:#0072b2}</style><h1>Seven dashboard figures</h1><p>All tasks and repetitions. <a href="metric_percentages.csv">Metric percentages CSV</a></p>'+items)
print('Output:',out,flush=True)
