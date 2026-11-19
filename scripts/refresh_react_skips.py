"""Refresh existing ReAct HTML status badges without rebuilding their media."""

import argparse
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil

from habitat_llm.action_result import is_missing_object_skip


class StepSpans(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.offsets = [0]
        for line in text.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.depth = 0
        self.active = None
        self.spans = []
        self.feed(text)

    def source_offset(self):
        line, column = self.getpos()
        return self.offsets[line - 1] + column

    def handle_starttag(self, tag, attrs):
        if tag != 'div':
            return
        self.depth += 1
        attrs = dict(attrs)
        if 'step' in attrs.get('class', '').split():
            self.active = (self.depth, self.source_offset(), attrs.get('data-status'))

    def handle_endtag(self, tag):
        if tag != 'div':
            return
        if self.active and self.active[0] == self.depth:
            _, start, status = self.active
            self.spans.append((start, self.source_offset() + len('</div>'), status))
            self.active = None
        self.depth -= 1


def refresh_html(text):
    """Keep all text, links and media; replace only statuses and summary markup."""
    changes = []
    spans = StepSpans(text).spans
    for start, end, status in spans:
        block = text[start:end]
        # This legacy migration must not override the policy used by new runs.
        if 'data-react-skip-policy="robot_sg_before_action"' in block:
            continue
        result = re.search(
            r'<div class="section-label">\s*Result\s*</div>\s*'
            r'<div class="section-content">(.*?)</div>', block, re.S,
        )
        if status not in {'failure', 'success'} or not result or not is_missing_object_skip(html.unescape(result[1])):
            continue
        block = re.sub(r'class="step [^"]+"', 'class="step skipped"', block, count=1)
        block = re.sub(r'data-status="[^"]+"', 'data-status="skipped"', block, count=1)
        block = re.sub(r'<span class="step-status [^"]+">.*?</span>',
                       '<span class="step-status status-skipped">Skipped</span>', block, count=1)
        changes.append((start, end, block))
    if not changes:
        return text, 0
    for start, end, block in reversed(changes):
        text = text[:start] + block + text[end:]
    statuses = [status for _, _, status in StepSpans(text).spans]
    counts = {s: statuses.count(s) for s in ('success', 'failure', 'skipped')}
    scored = len(spans) - counts['skipped']
    rate = 100 * counts['success'] / scored if scored else 0
    text = re.sub(r'(<span class="chip-label">Success</span><span class="chip-value">)[^<]*',
                  lambda m: m[1] + f"{rate:.1f}% ({counts['success']}/{scored})", text)
    for label, count in [('Failures', counts['failure']), ('Skipped', counts['skipped'])]:
        pattern = rf'(<span class="chip-label">{label}</span><span class="chip-value">)[^<]*'
        if re.search(pattern, text):
            text = re.sub(pattern, lambda m: m[1] + str(count), text)
        else:
            text = text.replace('<div class="chip"><span class="chip-label">Success</span>',
                                f'<div class="chip"><span class="chip-label">{label}</span>'
                                f'<span class="chip-value">{count}</span></div>\n'
                                '<div class="chip"><span class="chip-label">Success</span>', 1)
    if '.status-skipped' not in text:
        text = text.replace('</style>', '.step.skipped { border-left: 3px solid #3b82f6; }\n'
                            '.status-skipped { background: #dbeafe; color: #1e40af; }\n</style>', 1)
    if "filterSteps('skipped')" not in text:
        text = text.replace('<button class="btn filter-btn" onclick="filterSteps(\'failure\')">Failures</button>',
                            '<button class="btn filter-btn" onclick="filterSteps(\'failure\')">Failures</button>\n'
                            '<button class="btn filter-btn" onclick="filterSteps(\'skipped\')">Skipped</button>', 1)
    return text, len(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='JSON list of ReAct HTML paths')
    parser.add_argument('--output-dir', type=Path, required=True, help='Audit and original HTML backups')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = []
    for name in json.loads(args.manifest.read_text()):
        path = Path(name).resolve()
        # Explicitly scope migration to ReAct, including image/no-image variants.
        if not any('react' in part.lower() or 'partnr' in part.lower()
                   for part in path.relative_to(Path.cwd()).parts[:-1]):
            raise ValueError(f'Not a ReAct report: {path}')
        original = path.read_text()
        updated, count = refresh_html(original)
        if not count:
            continue
        backup = args.output_dir / 'backups' / path.relative_to(Path.cwd())
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        temporary = path.with_suffix('.html.tmp')
        temporary.write_text(updated)
        temporary.replace(path)
        audit.append({'path': str(path), 'reclassified_steps': count, 'backup': str(backup)})
    (args.output_dir / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f"Updated {len(audit)} reports; {sum(row['reclassified_steps'] for row in audit)} skipped steps")


if __name__ == '__main__':
    main()
