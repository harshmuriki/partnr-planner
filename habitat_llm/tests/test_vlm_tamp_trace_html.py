"""Trace identity, run isolation and expandable UI regressions."""
import json
import time

from habitat_llm.vlm_tamp.render_pddl_baseline_html import (
    _safe_literal_eval, render_pddl_baseline_log_dir_to_html,
)
from habitat_llm.vlm_tamp.render_trace_html import build_trace, latest_run_events


def branch(index, goals):
    return {'event': 'reprompt_branches_added', 'added_indices': [index],
            'added_branches': [goals], 'reprompt_round': index}


def test_skipped_tail_follows_parent_and_mismatched_execution_is_audited():
    events = [branch(0, ['explore(room)', 'picked(box)']),
              {'event': 'subgoal_status', 'branch': 0, 'subgoal_idx': 0,
               'subgoal': 'explore(room)', 'status': 'solved'},
              {'event': 'reprompt_started', 'branch': 0, 'subgoal_idx': 0,
               'reason': 'explore_refresh'},
              branch(1, ['opened-drawer(drawer)']),
              {'event': 'subgoal_execution', 'branch': 1, 'subgoal_idx': 0,
               'subgoal': 'picked(box)', 'status': 'solved', 'log_text': 'WRONG CELL'}]
    branches, audit, _ = build_trace(events)
    assert [n['status'] for n in branches[0]['nodes']] == ['success', 'skipped']
    assert branches[1]['nodes'][0]['records'] == []
    assert audit == [events[-1]]


def test_retries_remain_together_but_identical_goals_in_other_branches_do_not():
    events = [branch(0, ['picked(box)'])]
    for status in ['started', 'failed', 'started', 'solved']:
        events.append({'event': 'subgoal_status', 'branch': 0, 'subgoal_idx': 0,
                       'subgoal': 'picked(box)', 'status': status})
    events.append(branch(1, ['picked(box)']))
    branches, _, _ = build_trace(events)
    assert len(branches[0]['nodes'][0]['records']) == 4
    assert branches[0]['nodes'][0]['status'] == 'success'
    assert branches[1]['nodes'][0]['status'] == 'pending'


def test_action_repr_parser_preserves_identity_without_executing_code():
    event = _safe_literal_eval("{'event': 'pddl_plan', 'branch': 2, 'subgoal_idx': 3, "
                              "'plan': [Action(name='open', args=('agent_0', 'drawer'))]}")
    assert event['subgoal_idx'] == 3
    assert event['plan'] == [{'name': 'open', 'args': ('agent_0', 'drawer')}]
    assert _safe_literal_eval("{'plan': __import__('os').getcwd()}") is None


def test_latest_run_does_not_reuse_previous_branch_or_metrics(tmp_path):
    old = dict(branch(0, ['old_goal']), seq_idx=0, wall_time=1)
    current = dict(branch(0, ['new_goal']), seq_idx=0, wall_time=time.time() + 5)
    assert latest_run_events([old, current]) == [current]
    (tmp_path / 'vlm_tamp_pddl_log.jsonl').write_text(repr(old) + '\n' + repr(current) + '\n')
    (tmp_path / 'episode_metrics.json').write_text(json.dumps({'llm_model': 'STALE_MODEL'}))
    output = render_pddl_baseline_log_dir_to_html(str(tmp_path))
    text = open(output).read()
    assert 'new_goal' in text and 'old_goal' not in text
    assert 'STALE_MODEL' not in text
    assert 'No terminal decision recorded yet' in text


def test_html_has_boundaries_safe_text_and_only_local_image_links(tmp_path):
    (tmp_path / 'frame.png').write_bytes(b'fixture')
    events = [dict(branch(0, ['explore(room)', 'picked(box)']), seq_idx=0),
              {'event': 'subgoal_status', 'branch': 0, 'subgoal_idx': 0,
               'subgoal': 'explore(room)', 'status': 'solved', 'image_path': 'frame.png'},
              {'event': 'subgoal_execution', 'branch': 0, 'subgoal_idx': 0,
               'subgoal': 'explore(room)', 'status': 'solved', 'log_text': '\x1b[32m<script>bad()</script>\x1b[0m'},
              {'event': 'reprompt_started', 'branch': 0, 'subgoal_idx': 0, 'reason': 'explore_refresh'},
              {'event': 'planner_decision', 'decision': 'budget_exhausted', 'evidence': 'cycle limit'}]
    (tmp_path / 'vlm_tamp_pddl_log.jsonl').write_text('\n'.join(map(repr, events)))
    output = render_pddl_baseline_log_dir_to_html(str(tmp_path))
    text = open(output).read()
    assert 'data-parent="b0:s0"' in text
    assert 'class="step skipped compact"' in text
    assert 'data-node="start"' in text and 'data-node="end"' in text
    assert 'data-node="b0:end"' not in text
    assert 'Planner stopped: budget_exhausted' in text
    assert '<script>bad()' not in text and '&lt;script&gt;bad()' in text
    assert '\x1b' not in text
    assert 'src="frame.png"' in text
    assert '<svg' not in text
