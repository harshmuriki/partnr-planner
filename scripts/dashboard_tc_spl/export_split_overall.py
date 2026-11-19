"""Export the combined overall paper figure on a shared 0–1 scale."""
import csv
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator, FormatStrFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/all_baselines_5reps_20261005_001408/paper_figures_20261006'
NAME = '01_success_efficiency_completion'
rows = [r for r in csv.DictReader((OUT / 'metric_percentages.csv').open()) if r['grouping'] == 'overall']
assert len(rows) == 3
colors = ['#0072B2', '#D55E00', '#CC79A7']
labels = ['VLM-TAMP', 'ReAct (images)', 'TRU-POMDP']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.labelsize': 11, 'axes.titlesize': 11,
                     'xtick.labelsize': 10, 'ytick.labelsize': 9,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
backup = OUT / 'archive_before_split_overall'
backup.mkdir(exist_ok=True)
for ext in ['html', 'pdf', 'png']:
    src = OUT / f'{NAME}.{ext}'
    if src.exists() and not (backup / src.name).exists():
        shutil.copyfile(src, backup / src.name)


def panel(ax):
    keys = ['task_success_percent', 'SPL_percent', 'task_completion_percent', 'TC_SPL_action_percent']
    ticks = ['Task success', r'SPL$_{\mathit{act}}$', 'Task completion', r'TC-SPL$_{\mathit{act}}$']
    scale = .01
    for i, (row, color, label) in enumerate(zip(rows, colors, labels)):
        values = [float(row[k]) * scale for k in keys]
        bars = ax.bar([x + (i-1)*.24 for x in range(4)], values, width=.22, color=color, label=label, zorder=3)
        ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=4, fontsize=9, fontweight='medium')
    ax.set_xticks(range(4), ticks)
    ax.set_xlim(-.55, 3.55)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.yaxis.set_major_locator(MultipleLocator(.25))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(axis='y', color='#e2e8f0', linewidth=.8, zorder=0)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#cbd5e1')
    ax.tick_params(axis='both', length=0, pad=7)


def save(fig, name):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=600, facecolor='white')
    svg = (OUT / f'{name}.svg').read_text()
    svg = svg[svg.index('<svg'):]
    (OUT / f'{name}.html').write_text('<!doctype html><meta charset="utf-8"><title>Success, completion, and efficiency</title><style>body{margin:0;background:white}main{max-width:1100px;margin:auto}svg{width:100%;height:auto;display:block}</style><main>'+svg+'</main>')

fig, ax = plt.subplots(figsize=(7.16, 3.1))
panel(ax)
fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .995), ncol=3, frameon=False, fontsize=9, handlelength=1.2)
fig.subplots_adjust(left=.10, right=.985, bottom=.16, top=.87)
save(fig, NAME)
plt.close(fig)
manifest_path = OUT / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
for item in manifest['figures']:
    if item['name'] == NAME:
        item.update(title='Success, completion, and execution efficiency (0–1)',
                    recommended_width_inches=7.16, pixel_dimensions=[4296,1860], png_dpi=600,
                    width=687.36, height=297.6,
                    data_note='SPL and TC-SPL use recorded skill attempts relative to human reference action counts; sourced from tc_spl_data.json via refresh_paper_action_spl.py.')
        item.pop('raster_scale', None)
        item.pop('split_exports', None)
manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
print('Saved combined HTML, PDF, SVG, and 600-DPI PNG figure.')
