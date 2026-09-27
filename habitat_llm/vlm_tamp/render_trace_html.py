"""Standalone expandable trace, styled like previews/vlm_tamp_trace_example.html."""
from __future__ import annotations

from collections import Counter
import html
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote


def _escape(value):
    return html.escape(str(value), quote=True)


def _section(title, text):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(text))
    return (f'<div class="section"><div class="section-label">{_escape(title)}</div>'
            f'<pre class="section-content">{_escape(text)}</pre></div>')


def _records(records):
    return ('<details class="source-records"><summary>Source records</summary>'
            + _section('Matched events', json.dumps(records, indent=2, default=str))
            + '</details>')


def latest_run_events(events):
    """Appended logs can contain previous runs with the same branch IDs."""
    start = 0
    for index, event in enumerate(events):
        if event.get('seq_idx') == 0:
            start = index
    return events[start:]


def build_trace(events):
    """Associate records by branch, position AND goal, preserving all retries."""
    branches = []
    context = None
    for event in events:
        if event.get('event') == 'vlm_english_subgoals':
            context = event
        if event.get('event') != 'reprompt_branches_added':
            continue
        for index, goals in zip(event.get('added_indices', []), event.get('added_branches', [])):
            branches.append({'index': index, 'goals': goals, 'context': context,
                             'round': event.get('reprompt_round'), 'source': event})

    audit = []
    terminal = next((e for e in reversed(events) if e.get('event') == 'planner_decision'), None)
    for branch in branches:
        index = branch['index']
        boundary = next((e for e in events if e.get('event') == 'reprompt_started'
                         and e.get('branch', e.get('failed_branch')) == index), None)
        branch['boundary'] = boundary
        branch['nodes'] = []
        for position, goal in enumerate(branch['goals']):
            records = []
            for event in events:
                if (event.get('event') not in {'subgoal_status', 'subgoal_execution', 'pddl_plan'}
                        or event.get('branch') != index or event.get('subgoal_idx') != position):
                    continue
                if event.get('subgoal') != goal:
                    audit.append(event)
                    continue
                records.append(event)
            status_events = [e for e in records if e.get('event') in {'subgoal_status', 'subgoal_execution'}]
            status = status_events[-1].get('status', 'pending') if status_events else 'pending'
            if status in {'success', 'solved', 'already'}:
                status = 'success'
            elif status == 'failed':
                status = 'failure'
            elif status == 'observation_boundary':
                status = 'interrupted'
            elif boundary and not records:
                status = 'skipped'
            elif status == 'started' and not terminal:
                status = 'running'
            elif terminal and not records:
                status = 'pending'
            else:
                status = 'pending'
            branch['nodes'].append({'goal': goal, 'position': position, 'records': records,
                                    'status': status})
    return branches, audit, terminal


def render_trace_html(log_dir, out_path=None, episode_runtime_sec=None, episode_metrics=None):
    from habitat_llm.vlm_tamp.render_pddl_baseline_html import (
        _read_jsonl_events, _default_interactive_pddl_html_basename,
        _load_episode_metrics, _outcome_chips_html, _cost_kv_html,
    )

    root = Path(log_dir).resolve()
    destination = Path(out_path) if out_path else root / _default_interactive_pddl_html_basename(str(root))
    events = latest_run_events(_read_jsonl_events(str(root)))
    branches, audit, terminal = build_trace(events)
    if (root / 'vlm_prompts.txt').exists():
        from habitat_llm.vlm_tamp.render_vlm_prompts_html import render_vlm_prompts_chat_html
        render_vlm_prompts_chat_html(str(root))
    metrics = dict(episode_metrics) if episode_metrics is not None else _load_episode_metrics(str(root))
    metrics_file = root / 'episode_metrics.json'
    start_time = events[0].get('wall_time', 0) if events else 0
    if episode_metrics is None and metrics_file.exists() and metrics_file.stat().st_mtime < start_time:
        metrics = {}  # The previous run's outcome is not this run's outcome.
    if episode_runtime_sec is not None:
        metrics['episode_runtime_sec'] = episode_runtime_sec

    def local_url(path):
        if not path or Path(path).is_absolute():
            return None
        target = (root / path).resolve()
        if root not in target.parents or not target.is_file():
            return None
        return quote(os.path.relpath(target, destination.resolve().parent), safe='/')

    def gallery(items):
        figures = []
        seen = set()
        for path, caption in items:
            url = local_url(path)
            if not url or path in seen:
                continue
            seen.add(path)
            figures.append(f'<figure><a href="{url}" target="_blank" rel="noopener">'
                           f'<img class="step-cam" loading="lazy" src="{url}" alt="{_escape(caption)}"></a>'
                           f'<figcaption>{_escape(caption)} · <a download href="{url}">Save image</a>'
                           '</figcaption></figure>')
        return '<div class="gallery">' + ''.join(figures) + '</div>' if figures else ''

    def request_details(context):
        if not context:
            return ''
        return ('<details><summary>VLM request, response and input images</summary>'
                + _section('Image context', context.get('image_context', 'Planning observation'))
                + gallery([(p, f'VLM input {i + 1}') for i, p in enumerate(context.get('image_paths', []))])
                + _section('Prompt', context.get('prompt', 'Not recorded'))
                + _section('Response', context.get('response', 'Not recorded')) + '</details>')

    def cell(identifier, label, status, preview, body, compact=False):
        badges = {'success': 'Solved', 'failure': 'Failed', 'skipped': 'Skipped',
                  'pending': 'Pending', 'running': 'In progress', 'interrupted': 'Observed / replanned', 'marker': 'Boundary'}
        return (f'<details class="step {status}{" compact" if compact else ""}" data-node="{_escape(identifier)}">'
                f'<summary class="step-action-row"><span class="step-number">{_escape(identifier)}</span>'
                f'<span class="step-action">{_escape(label)}</span><span class="step-preview">{_escape(preview)}</span>'
                f'<span class="step-status status-{status}">{badges[status]}</span>'
                '<span class="step-chevron">▼</span></summary><div class="step-expand">'
                + body + '</div></details>')

    first_context = next((e for e in events if e.get('event') == 'vlm_english_subgoals'), None)
    match = re.search(r"following goal:\s*``(.*?)''", str((first_context or {}).get('prompt', '')), re.S)
    goal = re.sub(r'\s+', ' ', match.group(1)).strip() if match else 'Planning and execution trace'
    counts = Counter(n['status'] for b in branches for n in b['nodes'])
    css = Path(__file__).with_name('trace_style.css').read_text()
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>VLM-TAMP planning and execution</title><style>' + css + '</style></head><body>'
             '<div class="container"><div class="header"><h1>' + _escape(root.parent.name + ' · ' + goal)
             + '</h1><div class="header-path">' + _escape(root) + '</div>'
             + _outcome_chips_html(metrics) + '<div class="outcome-row">']
    for text in [f'{counts["success"]} solved nodes', f'{counts["failure"]} failed nodes',
                 f'{counts["skipped"]} skipped nodes', f'{len(branches)} branches']:
        parts.append('<div class="chip">' + _escape(text) + '</div>')
    parts.append('</div></div><div class="content">')
    if metrics:
        parts.append('<section class="panel"><h2>Episode metrics</h2>' + _cost_kv_html(metrics) + '</section>')
    parts.append('<section class="panel"><h2>Planning and execution</h2>'
                 '<p class="hint">Click a row for status history, execution logs, PDDL plans and images. '
                 'Small indented rows are goals skipped when their parent triggered replanning. '
                 'Start/end markers describe plan boundaries, not task-success verdicts.</p>'
                 '<div class="controls"><button class="btn" id="expand">Expand all</button>'
                 '<button class="btn filter-btn" id="collapse">Collapse all</button>'
                 '<button class="btn filter-btn" id="refresh">Refresh trace</button>')
    for path, label in [('observation_history.html', 'Observation history'), ('vlm_prompts.html', 'VLM prompts')]:
        url = local_url(path)
        if url:
            parts.append(f'<a class="btn filter-btn" href="{url}">{label}</a>')
    parts.append('</div><div class="steps">')
    parts.append(cell('start', 'start', 'marker', 'Initial planning request, before execution.',
                      request_details(first_context)))
    for branch in branches:
        index = branch['index']
        parts.append(f'<div class="round-label branch-heading"><span>Branch {index} · '
                     f'Reprompt round {_escape(branch["round"])}</span>'
                     + request_details(branch['context']) + '</div>')
        in_skipped = False
        for node in branch['nodes']:
            status = node['status']
            if status == 'skipped' and not in_skipped:
                boundary = branch['boundary'] or {}
                parent = boundary.get('subgoal_idx', boundary.get('failed_subgoal_idx', '?'))
                parts.append(f'<div class="skipped-group" data-parent="b{index}:s{parent}">'
                             '<div class="hint">Skipped after parent · '
                             + _escape(boundary.get('reason', 'replanning')) + '</div>')
                in_skipped = True
            elif status != 'skipped' and in_skipped:
                parts.append('</div>')
                in_skipped = False
            records = node['records']
            statuses = [e for e in records if e.get('event') == 'subgoal_status']
            execution = [e for e in records if e.get('event') == 'subgoal_execution']
            plans = [e for e in records if e.get('event') == 'pddl_plan']
            preview = ('Not executed; replaced after observation/replanning.' if status == 'skipped'
                       else 'Click to inspect status, execution, observations and images.')
            if status == 'failure':
                preview = next((e.get('failure_msg') for e in reversed(statuses) if e.get('failure_msg')), preview)
            body = _section('Node', f'Branch {index} · Subgoal {node["position"]}\n{node["goal"]}')
            if statuses:
                body += _section('Recorded status history', '\n'.join(
                    f'Event {e.get("seq_idx", "?")}: {e.get("status")} {e.get("failure_msg", "")}' for e in statuses))
            body += _section('Execution log', '\n\n'.join(
                f'Event {e.get("seq_idx", "?")} · {e.get("status")}\n{e.get("log_text", "")}'
                for e in execution) or ('Not executed.' if status == 'skipped' else 'No matching execution transcript recorded yet.'))
            if plans:
                body += _section('Recorded PDDL plans, including retries', '\n\n'.join(
                    f'Event {e.get("seq_idx", "?")}\n' + str(e.get('plan_raw', e.get('plan'))) for e in plans))
            body += _section('Image timing', 'Execution images below are captured at the recorded status boundary. '
                             'VLM input images are planning context and may be shared by multiple goals.')
            body += gallery([(e.get('image_path'), f'{e.get("status")} · {e.get("image_source", "recorded image")}')
                             for e in statuses if e.get('image_path')])
            selected = []
            for event in events:
                if event.get('event') != 'vlm_english_subgoals':
                    continue
                for record in event.get('exploration_images', []):
                    if record.get('branch') == index and record.get('subgoal_idx') == node['position']:
                        selected.append((record['image_path'], f'Explore tick {record["capture_tick"]} · sent to VLM'))
            if selected:
                body += _section('Exploration images sent to the VLM', 'Chronological samples taken during this Explore.') + gallery(selected)
            body += request_details(branch['context']) + _records(records)
            parts.append(cell(f'b{index}:s{node["position"]}', node['goal'], status, preview, body, status == 'skipped'))
        if in_skipped:
            parts.append('</div>')
    if not branches:
        parts.append('<p class="empty-state">Waiting for the first recorded plan.</p>')
    if terminal:
        parts.append(cell('end', 'Planner stopped: ' + str(terminal.get('decision')), 'marker',
                          terminal.get('evidence', ''), _section('Planner decision', terminal.get('evidence', ''))
                          + _section('Task outcome', 'Use the episode metrics above for task success; a planner stop reason does not establish task success.')
                          + _records([terminal])))
    elif metrics:
        parts.append(cell('end', 'Episode ended', 'marker', 'See recorded episode metrics.', _records([metrics])))
    else:
        parts.append('<p class="hint">No terminal decision recorded yet. Refresh to load newer events.</p>')
    parts.append('</div></section>')
    if audit:
        parts.append('<section class="panel"><h2>Unmatched records</h2><p class="hint">These records name a different goal '
                     'at the same branch/position and are excluded from cell execution logs.</p>' + _records(audit) + '</section>')
    parts.append('</div></div><script>'
                 'document.getElementById("expand").onclick=()=>document.querySelectorAll("details.step").forEach(e=>e.open=true);'
                 'document.getElementById("collapse").onclick=()=>document.querySelectorAll("details[open]").forEach(e=>e.open=false);'
                 'document.getElementById("refresh").onclick=()=>location.reload();'
                 '</script></body></html>')
    # A browser refresh should see either the previous page or the complete new page.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                     prefix='.trace-', suffix='.html', delete=False) as stream:
        stream.write(''.join(parts))
        temporary = stream.name
    os.replace(temporary, destination)
    return str(destination)
