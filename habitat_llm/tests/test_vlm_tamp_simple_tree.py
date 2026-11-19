"""Simplified view preserves plan topology, status and the original report."""
from habitat_llm.vlm_tamp.simple_tree import add_simple_tree, build_simple_tree


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
    assert 'id="vt-tree" hidden' in result
    assert '</script><script>alert(1)</script>' not in result
    assert add_simple_tree(result, events) == result
