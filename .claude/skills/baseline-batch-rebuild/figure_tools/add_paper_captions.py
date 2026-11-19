# Batch-agnostic copy of scripts/dashboard_tc_spl/add_paper_captions.py (paper order ReAct, VLM-TAMP, TRU-POMDP).
# Reads BATCH_DIR / FIG_DIR from the environment; run via rebuild_batch.sh.
"""Add editable captions outside the exported chart images and PDF artwork."""
import os
from pathlib import Path
import html
import re

OUT=Path(os.environ['FIG_DIR'])
CAPTIONS={
'01_success_efficiency_completion':'Task success, completion, and execution efficiency across three baselines (450 episodes each). SPL and TC-SPL are weighted by skill-action counts relative to human reference executions.',
'01a_success_completion':'Task success and mean task completion across three baselines (450 episodes each).',
'01b_spl_tc_spl':'Execution efficiency across three baselines: simulator-step-weighted SPL and skill-action-weighted TC-SPL in this saved snapshot.',
'02_spl_by_task':'Per-task SPL across three baselines, averaged over evaluated variants and repetitions. SPL uses skill-action counts relative to human reference executions.',
'03_tc_spl_by_initial_memory':'Action-weighted TC-SPL under accurate, incomplete, and outdated initial memory, averaged over 150 episodes per baseline and memory condition.',
'04_tc_spl_scenario_memory_three_baselines':'Action-weighted TC-SPL across environment scenarios and initial memory conditions for each baseline. Target exists, on surface, and no distractors denote the shared BASE configuration.',
'05_tc_spl_heatmap':'Action-weighted TC-SPL by baseline, initial memory, and environment scenario; cell values summarize the evaluated episodes for each combination.',
'06_failure_findings':'Rule-based failure finding rates among unsuccessful episodes for each baseline; multiple findings may apply to one episode.',
'07_inefficiency_findings':'Rule-based inefficiency finding rates among all evaluated episodes for each baseline; multiple findings may apply to one episode.',
'08_tc_spl_change_from_base':'Change in action-weighted TC-SPL relative to BASE on matched tasks, averaged equally over three memory conditions and reported in percentage points.',
'09_failure_findings_heatmap':'Percentage of unsuccessful episodes exhibiting each rule-based failure finding, by baseline; findings may overlap.',
'10_inefficiency_findings_heatmap':'Percentage of all evaluated episodes exhibiting each rule-based inefficiency finding, by baseline; findings may overlap.',
'11_task_descriptions_and_goals':'Seven household tasks and their BASE goal conditions, including the spatial, object-state, and execution-history requirements used in evaluation.'}
START='<!-- PAPER CAPTION START -->'
END='<!-- PAPER CAPTION END -->'

def add_captions():
    md=['# Suggested paper captions','', 'Captions are separate from image and PDF artwork. Historical exports retain their saved values and metric definitions.','']
    tex=['% Suggested captions; insert each command inside the matching figure/table environment.','% SPL in figures 01 and 02 uses skill-action counts.','']
    for stem,caption in CAPTIONS.items():
        path=OUT/f'{stem}.html'
        if not path.exists():continue
        text=path.read_text()
        text=re.sub(re.escape(START)+'.*?'+re.escape(END),'',text,flags=re.S)
        block=START+'<aside class="paper-caption" style="max-width:1000px;margin:14px auto;padding:12px 20px;font:14px/1.5 Arial,sans-serif;color:#334155;border-top:1px solid #e2e8f0"><strong>Suggested caption:</strong> '+html.escape(caption)+'</aside>'+END
        # Insert outside #figure and any SVG, so export crops remain caption-free.
        if '</body>' in text:text=text.replace('</body>',block+'</body>')
        else:text+=block
        path.write_text(text)
        (OUT/f'{stem}.caption.txt').write_text(caption+'\n')
        md.extend([f'## {stem}',caption,''])
        tex.extend([f'% {stem}',r'\caption{'+caption.replace('%',r'\%')+'}',''])
    (OUT/'captions.md').write_text('\n'.join(md))
    (OUT/'captions.tex').write_text('\n'.join(tex))
    index=OUT/'index.html';text=index.read_text()
    if 'href="captions.md"' not in text:
        text+='<section><h2>Suggested captions</h2><p><a href="captions.md">All captions</a> · <a href="captions.tex">LaTeX captions</a></p></section>'
    if 'href="11_task_descriptions_and_goals.html"' not in text:
        text+='<section><h2>Task descriptions and goal conditions</h2><a href="11_task_descriptions_and_goals.html">HTML guide</a> · <a href="11_task_descriptions_and_goals.pdf">PDF guide</a></section>'
    index.write_text(text)

if __name__=='__main__':
    add_captions()
    print('Added separate captions to 13 HTML views and exported TXT/Markdown/LaTeX captions.')
