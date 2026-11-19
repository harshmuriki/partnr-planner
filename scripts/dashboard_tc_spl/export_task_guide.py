"""Export named task descriptions and BASE goal conditions for paper readers."""
from pathlib import Path
import gzip
import html
import json
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/all_baselines_5reps_20261005_001408/paper_figures_20261006'
TASKS = [
('Water fetching', 'Bring a full jug of water to a living-room table and turn off the bedroom lights.', ['The jug is on either accepted living-room table and is filled.', 'The bedroom lamp is powered off.']),
('Package preparation', 'Arrange two packages and scissors on the living-room center table.', ['Both packages and the scissors are on the designated center table.', 'The packages are next to each other; the scissors are next to both packages. Opening the packages is not evaluated.']),
('Book collection', 'Collect the two green books on the living-room side table and turn off the laptop.', ['Both target books are on the designated living-room side table.', 'The laptop is powered off.']),
('Grocery organization', 'Put three fruits in the fruit basket and two drink containers on the kitchen counter.', ['All three fruits are inside the basket.', 'The basket and both drink containers are on the kitchen counter.']),
('Guest table preparation', 'Use soap to clean two glasses, fill them, and arrange them with a white plate for guests.', ['Both glasses and the target plate are on the living-room side table; the plate is next to each glass.', 'Both glasses are clean and filled.', 'The soap is next to each glass before that glass becomes clean; the soap need not remain there at the end.']),
('Meal delivery', 'Heat bread and deliver it with a water bottle and clean hand towel to the desk in the starting bedroom; turn off the living-room lights.', ['Bread, bottle, and towel are on the designated bedroom desk; the bottle and towel are each next to the bread.', 'The bottle is filled, the towel is clean, and the living-room lamp is powered off.', 'The bread has been inside the microwave and a completed Wait action has occurred while it remains inside. Microwave power and door state are not additional goal criteria.']),
('Household cleanup', 'Put away the living-room toys, wash and store the dirty plate, and turn off the living-room and kitchen lights.', ['Both target toys are inside the designated closet wardrobe.', 'The target plate is clean and inside any evaluator-approved cabinet.', 'Both the living-room and kitchen lamps are powered off.'])]

def main():
    records=json.loads((OUT.parent/'baseline_results.json').read_text())
    catalog=[]
    for i,(name,description,goals) in enumerate(TASKS,1):
        row=next(r for r in records if r['id'].startswith(f'react/T{i}-ACC-BASE/'))
        path=Path(row['run_folder'])/'input/dataset.json.gz'
        ep=json.loads(gzip.open(path).read())['episodes'][0]
        spec=ROOT/f'baseline_evaluation_v3/specs/T{i}/T{i}-ACC-BASE.md'
        catalog.append(dict(id=f'T{i}',name=name,description=description,goals=goals,
                            specification=str(spec.relative_to(ROOT)),evaluated_episode=str(path.relative_to(ROOT)),
                            propositions=ep['evaluation_propositions'],constraints=ep['evaluation_constraints']))
    (OUT/'task_definitions.json').write_text(json.dumps(catalog,indent=2)+'\n')
    sections=''.join(f'<section><h2>{r["id"]} · {html.escape(r["name"])}</h2><p>{html.escape(r["description"])}</p><ul>'+''.join(f'<li>{html.escape(g)}</li>' for g in r['goals'])+'</ul></section>' for r in catalog)
    note=('These are BASE task goals. SUB and ABS variants use their own goal specifications; hidden-object and distractor variants retain the applicable task goals. All task names map to the existing T1–T7 identifiers. Goal descriptions are checked against saved evaluated episodes, not inferred from plot labels.')
    temporal=('Evaluation detail: the T6 specification describes microwave use before delivery, but the saved BASE episode has no explicit temporal-order edges. The guide therefore lists the microwave-history checks and final delivery conditions separately. T5 explicitly enforces soap proximity before cleaning. Other listed final-state goals must hold at episode end.')
    text=f'''<!doctype html><html><head><meta charset="utf-8"><title>Task descriptions and goal conditions</title><style>
body{{font:13px/1.4 Arial,sans-serif;color:#172033;margin:24px auto;max-width:900px;padding:0 20px}}h1{{font-size:23px;margin:0 0 12px}}h2{{font-size:16px;margin:0 0 5px}}section{{border-top:1px solid #dce3eb;padding:10px 0;break-inside:avoid}}p{{margin:4px 0}}ul{{margin:4px 0;padding-left:20px}}.note{{color:#475569;font-size:11px}}@page{{size:A4;margin:12mm}}@media print{{body{{font-size:8pt;margin:0;padding:0;max-width:none;line-height:1.25}}h1{{font-size:13pt}}h2{{font-size:9pt}}section{{padding:6px 0}}.note{{font-size:7pt}}}}
</style></head><body><h1>Task descriptions and goal conditions</h1><p class="note">{note}</p>{sections}<p class="note">{temporal}</p></body></html>'''
    (OUT/'11_task_descriptions_and_goals.html').write_text(text)
    with sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page();page.set_content(text)
        assert page.locator('section').count()==7
        page.pdf(path=str(OUT/'11_task_descriptions_and_goals.pdf'),format='A4',print_background=True)
        browser.close()
    index=OUT/'index.html';s=index.read_text();start='<!-- TASK GUIDE START -->';end='<!-- TASK GUIDE END -->'
    import re
    s=re.sub(start+'.*?'+end,'',s,flags=re.S)
    block=start+'<section><h2>Task descriptions and goal conditions</h2><p><a href="11_task_descriptions_and_goals.html">Read task guide</a> · <a href="11_task_descriptions_and_goals.pdf">PDF</a></p><ul>'+''.join(f'<li>{r["id"]}: {r["name"]}</li>' for r in catalog)+'</ul></section>'+end
    s=s.replace('</body>',block+'</body>');index.write_text(s)
    print('Exported seven task descriptions, BASE goals, and evaluator provenance.')

if __name__=='__main__':main()
