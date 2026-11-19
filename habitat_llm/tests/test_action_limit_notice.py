"""Action-limit stops are explicit and remain separate from task success."""

import json
from types import SimpleNamespace

import pytest

from habitat_llm.planner.llm_planner import LLMPlanner
from habitat_llm.tests.test_tru_pomdp_planner import make_planner, make_world_graph
from habitat_llm.termination import action_limit_notice, termination_from_trace
from scripts import view_trace_logs
from scripts.run_parallel import result_metrics


def react(count, limit=50, done=False):
    planner = LLMPlanner.__new__(LLMPlanner)
    planner.llm = SimpleNamespace()
    planner._agents = []
    planner.reset()
    planner.planner_config = SimpleNamespace(replanning_threshold=limit)
    planner.replanning_count = count
    planner.check_if_agent_done = lambda response: done
    return planner


@pytest.mark.parametrize('count,stop', [(49, False), (50, True), (51, True)])
def test_react_limit_is_announced_and_saved(count, stop, capsys):
    planner = react(count)
    assert planner._check_termination('Navigate[table_0]') is stop
    output = capsys.readouterr().out
    if stop:
        assert 'Reached max actions (50).' in output
        assert termination_from_trace(planner.trace)['reason'] == 'max_actions'
        assert planner._cost_metrics()['termination']['count'] == count
    else:
        assert output == ''
        assert 'termination' not in planner._cost_metrics()
    planner.reset()
    assert 'termination' not in planner._cost_metrics()


def test_normal_done_does_not_claim_action_limit(capsys):
    planner = react(50, done=True)
    assert planner._check_termination('Done[]')
    assert capsys.readouterr().out == ''
    assert not planner._termination


def test_tru_limit_uses_configured_count(capsys):
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    planner.max_decisions = 2
    planner._num_decisions = 2
    assert planner._plan_next_symbolic_action(graph) is False
    assert 'Reached max actions (2).' in capsys.readouterr().out
    assert planner._planner_info({}, replanned=True)['termination']['limit'] == 2
    assert 'Termination: Reached max actions (2).' in planner._trace


def test_trace_html_and_run_summary_preserve_limit_reason(tmp_path):
    trace = tmp_path / 'trace.txt'
    trace.write_text('Task: demo\nRobot SG before action: ["cup_0"]\nThought: try\nPick[cup_0]\nAssigned!Result: '
                     'Unexpected failure! - Failed to pick! Object does not exist.\n'
                     'Termination: Reached max actions (50).\n')
    parsed = view_trace_logs.parse_trace_file(str(trace))
    assert len(parsed['steps']) == 1
    assert parsed['steps'][0]['skipped']
    assert 'Termination:' not in parsed['steps'][0]['result']
    html = tmp_path / 'trace.html'
    view_trace_logs.generate_html(parsed, str(html))
    assert 'class="termination-notice" role="status">Reached max actions (50).' in html.read_text()
    log = tmp_path / 'dataset/planner-log/episode.json'
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({'steps': [{'stats': {'task_state_success': 0.0},
        'termination': action_limit_notice(50, 50, 'replanning_threshold')}]}))
    metrics = result_metrics(tmp_path)
    assert metrics['termination']['reason'] == 'max_actions'
    assert metrics['task_state_success'] == 0.0
