"""Apply the T3-OUT invalid-target PowerOff exception to saved rule findings.

Only removes matching evidence; other unnecessary-action evidence is retained.
The classifier implements the same exception for future classifications.
"""
import argparse
import csv
import json
from pathlib import Path
import re


FRAGMENT = re.compile(r"PowerOff on something it can't act on x\d+ \(first at step \d+\)")


def evidence(run_id, category, label, text):
    if (run_id.split('/')[1].startswith('T3-OUT-')
            and category == 'inefficiency' and label == 'unnecessary_actions'):
        return '; '.join(p for p in text.split('; ') if not FRAGMENT.fullmatch(p))
    return text


def update(data):
    changes = []
    for run in data['runs']:
        findings = []
        for category, label, old in run['findings']:
            new = evidence(run['id'], category, label, old)
            if new != old:
                changes.append({'run_id': run['id'], 'before': old, 'after': new})
            if new:
                findings.append([category, label, new])
        run['findings'] = findings
    return changes


def main(root):
    dest = root / 'uncertainty_classifier'
    for filename in ['dashboard_data.json', 'rules_data.json']:
        path = dest / filename
        data = json.loads(path.read_text())
        changes = update(data)
        if changes:
            path.write_text(json.dumps(data, indent=2))
            if filename == 'dashboard_data.json':
                (dest / 't3_out_poweroff_exemption.json').write_text(json.dumps(changes, indent=2))
                print(f'{len(changes)} runs updated; {sum(not c["after"] for c in changes)} unnecessary-action labels removed')
    for filename in ['rule_viewer.html', 'source/viewer/rule_viewer.html']:
        path = dest / filename
        html = path.read_text()
        start = html.index('const DATA = ') + len('const DATA = ')
        data, length = json.JSONDecoder().raw_decode(html[start:])
        if update(data):
            path.write_text(html[:start] + json.dumps(data).replace('<', '\\u003c') + html[start + length:])
    path = dest / 'source/findings.csv'
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    for row in rows:
        row['evidence'] = evidence(row['run_id'], row['category'], row['label'], row['evidence'])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows if row['evidence'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
