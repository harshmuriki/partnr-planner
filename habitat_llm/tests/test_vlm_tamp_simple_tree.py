"""Simplified view preserves plan topology, status and the original report."""
import json
import re

from habitat_llm.vlm_tamp.simple_tree import add_simple_tree, build_simple_tree, map_rounds


def test_replan_attaches_after_observation_and_keeps_skipped_tail():
    events = [
        {'event': 'reprompt_branches_added', 'append': False, 'added_indices': [0],
         'added_branches': [['explore(room)', 'picked(box)']]},
        {'event': 'subgoal_status', 'branch': 0, 'subgoal_idx': 0,
         'subgoal': 'explore(room)', 'status': 'solved'},
        {'event': 'reprompt_started', 'branch': 0, 'subgoal_idx': 0,
         'reason': 'explore_refresh'},
        {'event': 'reprompt_branches_added', 'append': True, 'added_indices': [1],
         'added_branches': [['opened(drawer)']]},
        {'event': 'subgoal_status', 'branch': 1, 'subgoal_idx': 0,
         'subgoal': 'opened(drawer)', 'status': 'failed'},
    ]
    rows = build_simple_tree(events)
    by_ref = {ref: node for node in rows for ref in node['refs']}
    assert by_ref['b1:s0']['parent'] == by_ref['b0:s0']['id']
    assert by_ref['b1:s0']['replan']
    assert by_ref['b1:s0']['status'] == 'failure'
    assert by_ref['b0:s1']['status'] == 'skipped'
    assert all(row['goal'] != 'end' for row in rows)


def test_view_preserves_original_body_assets_and_escapes_script_content():
    body = '<main><img src="media/frame.png"><p>Original report</p></main>'
    original = '<html><head><base href="../run/"></head><body>' + body + '</body></html>'
    events = [{'event': 'reprompt_branches_added', 'append': False, 'added_indices': [0],
               'added_branches': [['</script><script>alert(1)</script>']]}]
    result = add_simple_tree(original, events)
    assert body in result
    assert '<base href="../run/">' in result
    assert 'id="vt-view-switch"' in result
    # The simple tree is the default view; the original report stays in the page, hidden.
    assert 'id="vt-current" hidden' in result
    assert 'id="vt-tree" hidden' not in result
    assert '</script><script>alert(1)</script>' not in result
    assert add_simple_tree(result, events) == result


def _map_events():
    return [
        {'event': 'vlm_english_subgoals', 'image_context': 'initial_scene',
         'image_paths': ['vlm_images/vlm_input_0000.png', 'overhead_map/map_0000_turn1.png']},
        {'event': 'reprompt_branches_added', 'append': False, 'added_indices': [0],
         'added_branches': [['picked(box)', 'on(box, table)']]},
        {'event': 'subgoal_status', 'branch': 0, 'subgoal_idx': 0, 'subgoal': 'picked(box)', 'status': 'solved'},
        {'event': 'subgoal_status', 'branch': 0, 'subgoal_idx': 1, 'subgoal': 'on(box, table)', 'status': 'failed'},
        {'event': 'reprompt_started', 'branch': 0, 'subgoal_idx': 1, 'reason': 'failure'},
        {'event': 'vlm_english_subgoals', 'image_context': 'current_failure',
         'image_paths': ['vlm_images/vlm_input_0001.png', 'overhead_map/map_0001_turn1.png']},
        {'event': 'reprompt_branches_added', 'append': True, 'added_indices': [1],
         'added_branches': [['on(box, table_2)']]},
        {'event': 'subgoal_status', 'branch': 1, 'subgoal_idx': 0, 'subgoal': 'on(box, table_2)', 'status': 'solved'},
    ]


def test_overhead_maps_link_to_the_branches_planned_from_them():
    events = _map_events()
    rounds = map_rounds(events)
    assert [(r['round'], r['url'], r['branches'], r['reason']) for r in rounds] == [
        (1, 'overhead_map/map_0000_turn1.png', [0], 'start'),
        (2, 'overhead_map/map_0001_turn1.png', [1], 'current failure'),
    ]
    rows = build_simple_tree(events)
    by_ref = {ref: node for node in rows for ref in node['refs']}
    start = next(r for r in rows if r['status'] == 'start')
    assert start['map_round'] == 1 and start['map_start']
    assert by_ref['b0:s0']['map_round'] == 1 and not by_ref['b0:s0']['map_start']
    assert by_ref['b0:s1']['map_round'] == 1
    assert by_ref['b1:s0']['map_round'] == 2 and by_ref['b1:s0']['map_start']
    # Unresolvable paths (url_for returns None) are dropped; runs without maps get no strip data.
    assert map_rounds(events, url_for=lambda p: None) == []
    assert map_rounds([e for e in events if e['event'] != 'vlm_english_subgoals']) == []


def test_map_strip_data_embedded():
    result = add_simple_tree('<html><head></head><body><p>x</p></body></html>', _map_events())
    data = re.search(r'<script id="vt-map-data" type="application/json">(.*?)</script>', result).group(1)
    assert [r['branches'] for r in json.loads(data)] == [[0], [1]]
    assert 'id="vt-maps" hidden' in result  # shown by the script only when there are maps
