"""Export grouped per-task SPL bars from the saved paper-figure snapshot."""
import csv
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

out = Path(__file__).resolve().parents[2] / 'results/all_baselines_5reps_20261005_001408/paper_figures_20261006'
name = '02_spl_by_task'
rows = [r for r in csv.DictReader((out/'metric_percentages.csv').open()) if r['grouping'] == 'task']
assert len(rows) == 21
backup = out/'archive_before_task_bars'
backup.mkdir(exist_ok=True)
for ext in ['html','png','pdf']:
    src = out/f'{name}.{ext}'
    if not (backup/src.name).exists():
        shutil.copyfile(src, backup/src.name)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
fig, ax = plt.subplots(figsize=(7.16,3.3))
for i,(baseline,color) in enumerate(zip(['VLM-TAMP','ReAct (images)','TRU-POMDP'],['#0072B2','#D55E00','#CC79A7'])):
    selected = sorted([r for r in rows if r['baseline']==baseline],key=lambda r:int(r['group']))
    assert [int(r['group']) for r in selected] == list(range(1,8))
    values = [float(r['SPL_percent'])/100 for r in selected]
    bars = ax.bar([t+(i-1)*.25 for t in range(1,8)],values,width=.23,color=color,label=baseline,zorder=3)
    assert all(abs(bar.get_height()-v)<1e-12 for bar,v in zip(bars,values))
tasks=json.loads((out/'task_definitions.json').read_text())
labels=['Water\nfetching','Package\npreparation','Book\ncollection','Grocery\norganization','Guest table\npreparation','Meal\ndelivery','Household\ncleanup']
ax.set_xticks(range(1,8),[f'T{i+1}\n{label}' for i,label in enumerate(labels)])
ax.tick_params(axis='x',labelsize=7)
ax.set_xlim(.45,7.55)
ax.set_ylim(0,1)
ax.set_xlabel('Task',labelpad=5)
ax.set_ylabel('SPL',labelpad=6)
ax.yaxis.set_major_locator(MultipleLocator(.25))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.grid(axis='y',color='#e2e8f0',linewidth=.7,zorder=0)
ax.tick_params(axis='both',length=0,pad=5)
for side in ['top','right']:ax.spines[side].set_visible(False)
for side in ['left','bottom']:ax.spines[side].set_color('#cbd5e1')
fig.suptitle('SPL by task',y=.985,fontsize=10,fontweight='bold')
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,fontsize=6,handlelength=1,columnspacing=.9,frameon=False)
fig.subplots_adjust(left=.10,right=.985,bottom=.25,top=.78)
for ext in ['pdf','png','svg']:
    fig.savefig(out/f'{name}.{ext}',dpi=600,facecolor='white')
plt.close(fig)
svg=(out/f'{name}.svg').read_text();svg=svg[svg.index('<svg'):]
(out/f'{name}.html').write_text('<!doctype html><meta charset="utf-8"><title>SPL by task</title><style>body{margin:0;background:white}main{max-width:750px;margin:auto}svg{display:block;width:100%;height:auto}</style><main>'+svg+'</main>')
# Keep the chart PDF compact; supply the task guide below its HTML view.
import html
path=out/f'{name}.html'
markup=path.read_text()
guide='<section style="font:14px/1.5 Arial;padding:20px"><h2>Tasks and goal conditions (BASE)</h2><p><a href="11_task_descriptions_and_goals.pdf">Download task guide PDF</a></p>'
for task in tasks:
    guide+=f'<h3>{task["id"]}: {html.escape(task["name"])}</h3><p>{html.escape(task["description"])}</p><ul>'+''.join(f'<li>{html.escape(goal)}</li>' for goal in task['goals'])+'</ul>'
guide+='<p>These are BASE goals; other variants have variant-specific criteria. See the task guide for temporal-evaluation details and task_definitions.json for source provenance.</p></section>'
path.write_text(markup.replace('</main>',guide+'</main>'))
p=out/'manifest.json';manifest=json.loads(p.read_text())
for item in manifest['figures']:
    if item['name']==name:
        item.update(chart_type='grouped bars',pixel_dimensions=[4296,1980],png_dpi=600,width=687.36,height=316.8,recommended_width_inches=7.16,data_note='Action-based SPL from metric_percentages.csv, refreshed from tc_spl_data.json; displayed on a 0–1 scale.')
        item.pop('raster_scale',None)
p.write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved HTML/PDF/SVG/600-DPI PNG; checked all 21 bar values against the snapshot.')
