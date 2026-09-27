"""Audit fixed variant specs without rewriting them or resolving task ambiguities.

The workbook is exported separately as source_cells.json so this audit has no
spreadsheet/network dependency. It also records immutable input hashes.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

STATE_VALUES = {
    'is_clean': ('is_clean', True), 'is_dirty': ('is_clean', False),
    'is_filled': ('is_filled', True), 'is_empty': ('is_filled', False),
    'is_powered_on': ('is_powered_on', True),
    'is_powered_off': ('is_powered_on', False),
}


def section(text, heading):
    match = re.search(r'^## ' + re.escape(heading) + r'\n(.*?)(?=^## |\Z)', text, re.M | re.S)
    if not match:
        raise ValueError('Missing section: ' + heading)
    return match[1].strip()


def parse_spec(path):
    text = path.read_text()
    entities = {}
    for entity, cls, asset in re.findall(r'^- (\w+) \((\w+), asset ([^)]+)\):', section(text, 'Affected object(s)'), re.M):
        entities[entity] = {'class': cls, 'asset': asset}
    states = {}
    for heading in ['Initial world state', 'Final expected world state']:
        states[heading] = {}
        for line in section(text, heading).splitlines():
            m = re.fullmatch(r'- (\w+): (on|within|floor) (\w+) \(([^)]+)\)(?:, (.*))?', line)
            if not m:
                raise ValueError(f'{path}: unsupported {heading} line: {line}')
            entity, relation, furniture, room, rest = m.groups()
            attrs = (rest or '').split(', ') if rest else []
            unknown = [a for a in attrs if a not in STATE_VALUES and not re.fullmatch(r'next to \w+|\w+ starts closed', a)]
            if unknown:
                raise ValueError(f'{path}: unsupported attributes {unknown}')
            states[heading][entity] = dict(relation=relation, furniture=furniture, room=room,
                states=dict(STATE_VALUES[a] for a in attrs if a in STATE_VALUES),
                next_to=[a[8:] for a in attrs if a.startswith('next to ')],
                closed=[a.split()[0] for a in attrs if a.endswith(' starts closed')])
    return dict(variant=path.stem, scene=re.search(r'scene_id:\s*(\S+)', text)[1],
        instruction=section(text, 'Task instruction / prompt given').strip('"'), entities=entities,
        initial=states['Initial world state'], final=states['Final expected world state'],
        memory=section(text, 'Initial robot memory'), success=section(text, 'Success criteria'),
        notes=section(text, 'Spawn / planner notes'), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def sync_episode_issues(root, issues):
    """Keep existing review metadata aligned with the current audit."""
    index_path = root / 'generation' / 'episodes.json'
    if not index_path.exists():
        return
    index = json.loads(index_path.read_text())
    for vid, item in index['variants'].items():
        active = [i['id'] for i in issues if vid in i['variants']]
        blocked = any(i['blocking'] and vid in i['variants'] for i in issues)
        item['issues'] = active
        if item['status'] in {'blocked', 'pending'}:
            item['status'] = 'blocked' if blocked else 'pending'
        if item.get('review'):
            review_path = root / item['review']
            review = json.loads(review_path.read_text())
            review['issues'] = active
            review_path.write_text(json.dumps(review, indent=2) + '\n')
    index_path.write_text(json.dumps(index, indent=2) + '\n')


def audit(root):
    out = root / 'generation'
    source = json.loads((out / 'source_cells.json').read_text())
    rows = defaultdict(list)
    for row in csv.DictReader((root / 'variant_object_assets.csv').open()):
        rows[row['Variant_ID']].append(row)
    specs, issues = {}, []

    def issue(code, variants, message, source_cells='', blocking=True):
        issues.append(dict(id=code, variants=sorted(variants), status='active',
            blocking=blocking, message=message, source_cells=source_cells))

    for path in sorted((root / 'specs').glob('T*/T*.md')):
        spec = parse_spec(path)
        vid = spec['variant']
        specs[vid] = spec
        csv_objects = {re.fullmatch(r'(\w+) \((\w+)\)', r['Object'])[1]:r for r in rows[vid]}
        if set(csv_objects) != set(spec['entities']) or set(csv_objects) != set(spec['initial']):
            issue('OBJECTS-' + vid, [vid], 'CSV / pinned entities / initial world object sets disagree.')
        for entity, obj in spec['entities'].items():
            row = csv_objects.get(entity, {})
            if row and re.fullmatch(r'(\w+) \((\w+)\)', row['Object'])[2] != obj['class']:
                issue('CLASS-' + vid + '-' + entity, [vid], 'Spec object class disagrees with CSV.')
            if row.get('Real_Object_ID') != obj['asset']:
                issue('ASSET-' + vid + '-' + entity, [vid], 'Pinned asset does not match CSV.')
            for phase, prefix in [('initial', 'Start'), ('final', 'End')]:
                expected = dict(STATE_VALUES[row[prefix + '_' + pair]] for pair in
                    ['Clean_Dirty', 'Filled_Empty', 'Powered_On_Off'] if row.get(prefix + '_' + pair) in STATE_VALUES)
                if spec[phase].get(entity, {}).get('states') != expected:
                    issue('STATE-' + vid + '-' + entity + '-' + phase, [vid], 'Spec boolean states disagree with CSV.')
        task = int(vid[1])
        scene = str(source['Tasks']['B' + str(task + 3)]).removesuffix('.0')
        if scene != spec['scene']:
            issue('SCENE-' + vid, [vid], 'Scene differs from task sheet.', f'Tasks!B{task+3}')
        conflicts = []
        for entity, state in spec['final'].items():
            for other in state['next_to']:
                if other not in spec['final'] or state['room'] != spec['final'][other]['room']:
                    conflicts.append(f'{entity} next to {other}, with different final rooms or missing partner')
        if conflicts:
            issue('FINAL-RELATIONS-' + vid, [vid], '; '.join(conflicts))

    if set(rows) != set(specs):
        issue('VARIANT-COVERAGE', set(rows) ^ set(specs), 'CSV and spec variant sets differ.')
    groups = defaultdict(list)
    for vid in specs:
        task, mem, axis = vid.split('-')
        groups[task + '-' + axis].append(vid)
    for group in groups.values():
        physical = [{k:s[k] for k in ['scene','instruction','entities','initial','final','success','notes']}
                    for s in (specs[v] for v in group)]
        if any(p != physical[0] for p in physical):
            issue('MEMORY-INVARIANT-' + group[0], group, 'Memory variants differ in physical/task specification.')

    select = lambda prefix: [v for v in specs if v.startswith(prefix)]
    wrong_basket = [v for v in select('T4-') if any(
        specs[v]['final'][entity]['relation'] != 'within' or specs[v]['final'][entity]['furniture'] != 'basket_0'
        or f'- is_inside({entity}, basket_0)' not in specs[v]['success'].splitlines()
        for entity,obj in specs[v]['entities'].items() if obj['class'] in ['apple','orange'])]
    if wrong_basket:
        issue('T4-BASKET', wrong_basket, 'Fruit must be inside basket_0; final state or success criteria do not express the approved containment requirement.', 'Tasks Reformated!A41; Tasks!C7')
    microwave_step = ['- is_inside(bread_0, microwave_0)', '- order: is_inside(bread_0, microwave_0) before is_on_top(bread_0, table_11)']
    unheated = [v for v in select('T6-') if not all(line in specs[v]['success'].splitlines() for line in microwave_step)]
    if unheated:
        issue('T6-HEATING', unheated, 'Heating must be scored as a microwave visit: bread_0 inside microwave_0 before it is placed on table_11. These specs lack that criterion or its ordering.', 'Tasks Reformated!A66; Tasks!C9')
    soap_steps = ['- ' + line for g in ['glass_0', 'glass_1'] for line in (f'is_next_to(soap_dispenser_0, {g})', f'order: is_next_to(soap_dispenser_0, {g}) before is_clean({g})')]
    unsoaped = [v for v in select('T5-') if not all(line in specs[v]['success'].splitlines() for line in soap_steps)]
    if unsoaped:
        issue('T5-SOAP-SCORING', unsoaped, 'Soap use must be scored as bringing the soap: soap_dispenser_0 next to each glass before that glass becomes clean. These specs lack that criterion or its ordering.', 'Tasks Reformated!A53', False)
    validator_path = out/'validator_results.json'
    if validator_path.exists():
        results = json.loads(validator_path.read_text())
        errors = [e for r in results for e in r['errors']]
        if errors:
            issue('VALIDATOR-ERRORS', [r['variant_id'] for r in results if r['errors']],
                f'Validator reports {len(errors)} errors. Raw results are retained in generation/validator_results.json.', blocking=False)
    index_path = out/'episodes.json'
    if index_path.exists():
        failed_groups = defaultdict(list)
        index = json.loads(index_path.read_text())['variants']
        for vid, entry in index.items():
            if entry['status'] == 'failed':
                failed_groups[(vid.split('-')[0],vid.split('-')[2],entry.get('error',''))].append(vid)
        for (task, axis, error), variants in failed_groups.items():
            if task == 'T1' and axis == 'CON' and (out/'containment_diagnostics/findings.json').exists():
                issue('GENERATION-T1-CON', variants,
                    'The allowed default drawer failed to sample the upright jug. Diagnostic tests in the SAME fridge found a stable lower-compartment placement passing is_inside and navigation accessibility, but it requires overriding an access-filtered receptacle and opening different doors. Runtime Open currently opens default link 6 (drawer). Fix and verify the receptacle/default-link interaction metadata before accepting this alternative; not proof that the jug cannot fit. Upper compartment fits but fails navigation accessibility.',
                    'generation/containment_diagnostics/findings.json', blocking=False)
                continue
            issue('GENERATION-' + task + '-' + axis, variants,
                'Exact-spec generation failed after bounded retries; no dataset was accepted. '+error,
                'generation/logs/' + variants[0] + '.json', blocking=False)
    approved_overrides = [
        '2026-09-15 user confirmed OUT means outdated memory. All T3 OUT variants remember the laptop on bedroom table_1 instead of its actual dining table_2. T5-OUT-CON retains its existing stale memory despite the inconsistent source cell.',
        '2026-09-15 user approved T6-CON locations: bread in kitchen cabinet_1, bottle in garage fridge_0, towel in laundry-room cabinet_23. T6-CON-ROOM is resolved; physical placement and heating support remain separate concerns.',
        '2026-09-15 user confirmed T7 current CSV/spec object set is correct. Do not add airplane, doll, toy animal, second task plate or bowl from the older Tasks!F10 snapshot. T7-OBJECTS is resolved by this decision.',
        'T1 jug and substitute pitcher start empty and must be filled (user decision).',
        'T1 ambiguous instruction retains bedroom lights (user decision, now also in reformatted sheet).',
        'T6 bottle/cup start and end filled; T3 phone stays powered on (user decisions).',
        'T5 spray bottle removed; reformatted sheet now also omits it.',
        'T6 baguette follows reformatted sheet L66/I70; Tasks!S9 still says toy_food.',
        '2026-09-15 user-approved correction: removed conflicting final distractor-to-target adjacency from 18 DIS specs (T1/T2/T3/T5/T6/T7). Initial adjacency, distractor destinations, states and task success criteria are preserved.',
        '2026-09-15 user confirmed T4 fruit must be inside basket_0. All 17 specs use within basket_0 and is_inside goals; basket remains on counter_0. Implementation support remains blocked separately.',
        '2026-09-15 user chose a microwave visit for T6 heating: all 20 T6 specs require is_inside(bread_0, microwave_0) before is_on_top(bread_0, table_11); the bread need not remain in the microwave. No heating state is simulated. T6-HEATING is resolved by this decision.',
        '2026-09-15 user set T6-ACC-CAND hand_towel_0 candidate rooms to laundryroom/mudroom_0 and bathroom_2. World state is unchanged: the towel is on shelves_11 in bathroom_2 (the laundry-room cabinet_23 placement is T6-CON only).',
        '2026-09-15 user requested a soap step like the T6 microwave visit: all 20 T5 specs require is_next_to(soap_dispenser_0, glass_N) before is_clean(glass_N) for both glasses; the soap need not stay there and its final location is not scored. Existing T5 episodes keep their placements; only their evaluation is recompiled. T5-SOAP-SCORING is resolved by this decision.',
        '2026-09-15 user clarified that initial robot memory is used only to initialize the baseline, not inside the episode or runtime planner. MEMORY-RUNTIME is not an issue and is no longer reported.',
        '2026-09-15 user clarified that checking whether the planner reports unavailable objects (ABS) is evaluated on the baseline side, not by dataset propositions. ABS-REPORT-SCORING is not a dataset issue and is no longer reported.',
        '2026-09-15 T4 basket support verified: every generated T4 episode was regenerated with the on-placement interior guard and passed runtime initialization; an oracle rollout on T4-ACC-BASE placed apple_0, apple_1 and orange_0 within basket_0 (all is_inside satisfied, 100% complete) and both world graphs record the fruit inside basket_0 and on the counter. T4-BASKET-VALIDATION is resolved.',
        '2026-09-15 oracle-skill rollouts on ACC-BASE episodes: T1 100% (jug filled at a faucet), T3 100%, T6 100% (bread placed within and taken out of microwave_0 before the desk), T5 100% when soap is placed on the near side of each glass (soap-before-clean ordering satisfied; kitchen cabinet_0 carries faucet markers). Faucet reachability, the microwave visit and soap delivery are executable. T6-MICROWAVE-ROLLOUT, T5-SOAP-ROLLOUT and REACHABILITY are resolved.',
        '2026-09-15 user resolved T7-OBJECT-FIT: plate_0 is put away within the kitchen corner cabinet cabinet_7 (fits; cabinet_2 is too narrow), and stuffed_toy_0 uses Dottie Plush Elephant b3e8be210978ec373be6eb5fcffec36c6a9712e6 (24x28x21 cm; fits inside wardrobe_0 and is stable on couch_0) instead of 7c36beb8e017c9b8c66291ad9c5be9f7270281b4, which fits no interior in the scene. T7 specs and variant_object_assets.csv updated.',
        '2026-09-15 user confirmed the furniture/room grounding choices in the T1, T2, T4 and T5 spec notes (and T3, T6, T7 choices reviewed with them). GROUNDING-REVIEW is resolved. Oracle rollout on regenerated T7-ACC-BASE: 21/21 skills, 100% complete (elephant within wardrobe_0, plate within cabinet_7).',
    ]
    result = dict(source_url='https://docs.google.com/spreadsheets/d/13DyMgKH4ZktYdEakcRW2aqUph4Tup9IIiXDOwTq28gw/edit?gid=486995175#gid=486995175',
        workbook_sha256=hashlib.sha256((out/'source_tasks.xlsx').read_bytes()).hexdigest(),
        csv_sha256=hashlib.sha256((root/'variant_object_assets.csv').read_bytes()).hexdigest(),
        specs=specs, issues=issues, approved_overrides=approved_overrides)
    (out/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    sync_episode_issues(root, issues)
    print(json.dumps({'specs':len(specs),'issues':len(issues),'blocked_variants':len({v for i in issues if i['blocking'] for v in i['variants']})}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('baseline_evaluation_v3'))
    audit(parser.parse_args().root)
