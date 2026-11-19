"""Add a self-contained, simplified planning tree to existing HTML reports."""

import json
from pathlib import Path
import re


def build_simple_tree(events):
    from habitat_llm.vlm_tamp.render_trace_html import build_trace, latest_run_events
    from habitat_llm.vlm_tamp.tree_render_2 import (
        _apply_status_colors, _build_tree_from_events,
    )

    events = latest_run_events(events)
    nodes, root_id, pairs = _build_tree_from_events(events)
    _apply_status_colors(nodes, pairs, events)
    branches, _, _ = build_trace(events)
    trace_nodes = {(b['index'], n['position']): n for b in branches for n in b['nodes']}
    replans = {index for e in events if e.get('event') == 'reprompt_branches_added'
               and e.get('append') for index in e.get('added_indices', [])}
    rows = []
    statuses = {'solved': 'success', 'already': 'success', 'failed': 'failure',
                'blocked': 'skipped', 'default': 'pending', 'started': 'running',
                'observation_boundary': 'interrupted'}
    for key, node in nodes.items():
        # The old renderer inserts synthetic "end" leaves. They do not mean success.
        if key != root_id and not node.pair_refs:
            continue
        status = statuses.get(node.status, node.status)
        refs = [f'b{b}:s{s}' for b, s in node.pair_refs]
        matches = [trace_nodes[p] for p in node.pair_refs if p in trace_nodes
                   and trace_nodes[p]['goal'] == node.action]
        recorded = [n for n in matches if n['records']]
        if recorded:
            status = max(recorded, key=lambda n: max(
                e.get('seq_idx', -1) for e in n['records']))['status']
        elif matches:
            status = matches[0]['status']
        branch, position = node.pair_refs[0] if node.pair_refs else (None, None)
        rows.append({'id': key, 'parent': node.parent_id, 'goal': node.action,
                     'status': 'start' if key == root_id else status,
                     'refs': refs, 'branch': branch, 'position': position,
                     'replan': branch in replans and position == 0,
                     'failure': node.failure_msg})
    return rows


def add_simple_tree(document, events):
    """Preserve the original report and its asset URLs; embed an offline view switch."""
    if 'id="vt-view-switch"' in document:
        return document
    data = json.dumps(build_simple_tree(events), ensure_ascii=True).replace('<', '\\u003c')
    folder = Path(__file__).parent
    css = (folder / 'simple_tree.css').read_text(encoding='utf-8')
    script = (folder / 'simple_tree.js').read_text(encoding='utf-8')
    toolbar = '''<nav id="vt-view-switch" aria-label="Report view">
<span>View</span><button type="button" data-vt-view="current" aria-pressed="true">Current view</button>
<button type="button" data-vt-view="tree" aria-pressed="false">Simple tree</button>
</nav><div id="vt-current">'''
    tree = '''</div><section id="vt-tree" hidden aria-label="Simple planning tree">
<header><h1>Planning tree</h1><p>Follow subgoals from top to bottom. Branches show alternatives and replanning.
Select a node for details. Node status describes a subgoal; see the current view for the episode outcome.</p></header>
<div class="vt-controls"><button type="button" id="vt-fit">Fit tree</button>
<button type="button" id="vt-actual">100%</button>
<button type="button" id="vt-minus" aria-label="Zoom out">−</button>
<button type="button" id="vt-plus" aria-label="Zoom in">+</button>
<label><input type="checkbox" id="vt-skipped"> Show skipped goals</label></div>
<div class="vt-legend"><span>✓ Solved</span><span>× Failed</span><span>↻ Observed / replanned</span>
<span>○ Pending</span><span>– Skipped</span></div>
<div class="vt-workspace"><div id="vt-canvas" tabindex="0" aria-label="Scrollable planning tree"></div>
<aside id="vt-detail" aria-live="polite"><h2>Subgoal details</h2><p>Select a node to inspect it.</p></aside></div>
</section>'''
    document = document.replace('</head>', '<style>' + css + '</style></head>', 1)
    document = re.sub(r'(<body\b[^>]*>)', lambda m: m.group(1) + toolbar,
                      document, count=1, flags=re.IGNORECASE)
    return document.replace('</body>', tree + '<script id="vt-tree-data" type="application/json">'
                            + data + '</script><script>' + script + '</script></body>', 1)
