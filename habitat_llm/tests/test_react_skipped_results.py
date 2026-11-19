"""Missing-object ReAct picks are visible skips, not successful or failed skills."""

import pytest

from habitat_llm.action_result import (is_missing_object_skip, mark_react_skip,
    react_action_evidence, robot_scene_graph_marker, SG_POLICY_MARKER)
from scripts import view_trace_logs
from scripts.refresh_react_skips import StepSpans, refresh_html
from scripts.trace_annotator.catalog import infer_trace_issues, parse_react_trace_file


MISSING = 'Unexpected failure! - Failed to pick! Object does not exist.'


@pytest.mark.parametrize('result', [MISSING, MISSING + ' -- Should be skipped',
                                  'Failed to pick! Object does not exist.'])
def test_parser_and_error_counts_exclude_missing_object(tmp_path, result):
    trace = tmp_path / 'trace.txt'
    trace.write_text('Task: demo\nRobot SG before action: ["cup_0"]\nThought: Try remembered cup\nPick[cup_0]\n'
                     f'Assigned!Result: {result}\nObjects: cup_0: unknown\n'
                     'Thought: Finish\nDone[]\n')
    viewer = view_trace_logs.parse_trace_file(str(trace))
    catalog = parse_react_trace_file(str(trace))
    for parsed in (viewer, catalog):
        step = parsed['steps'][0]
        assert step['skipped'] is True
        assert step['success'] is None
        assert step['result'] == result
        issues = infer_trace_issues(parsed['steps'])
        assert issues['failed_step_count'] == 0
        assert issues['efficiency_errors'] == []


def test_other_pick_errors_are_not_skips():
    assert not is_missing_object_skip('Unexpected failure! - Failed to pick! Object is out of reach.')
    assert not is_missing_object_skip(MISSING + ' Another action failed.')


def test_html_skips_and_migration_preserve_real_errors_and_media(tmp_path):
    output = tmp_path / 'report.html'
    steps = [
        {'action': 'Pick', 'args': 'cup_0', 'result': MISSING, 'success': False, 'image': 'camera.png',
         'target_in_robot_sg_before_action': True, 'react_skip_policy': 'legacy'},
        {'action': 'Navigate', 'result': 'Successful execution!', 'success': True},
        {'action': 'Pick', 'result': 'Failed to pick! Out of reach.', 'success': False},
    ]
    view_trace_logs.generate_html({'task': 'demo', 'steps': steps, 'total_steps': 3}, str(output))
    text = output.read_text()
    assert [s[2] for s in StepSpans(text).spans] == ['skipped', 'success', 'failure']
    assert '50.0% (1/2)' in text
    assert '<span class="chip-label">Failures</span><span class="chip-value">1' in text
    assert '<span class="chip-label">Skipped</span><span class="chip-value">1' in text
    assert 'status-skipped">Skipped' in text
    assert MISSING in text
    # Replay migration of a historical report with the old red classification.
    old = text.replace('class="step skipped" data-status="skipped"',
                       'class="step failure" data-status="failure"')
    old = old.replace('class="step-status status-skipped">Skipped',
                      'class="step-status status-failure">Failed')
    old = old.replace('50.0% (1/2)', '33.3% (1/3)')
    migrated, count = refresh_html(old)
    assert count == 1
    assert migrated == text
    assert 'src="camera.png"' in migrated
    assert refresh_html(migrated) == (migrated, 0)


def test_pddl_classification_is_unchanged(tmp_path):
    output = tmp_path / 'pddl.html'
    view_trace_logs.generate_html({'task': 'demo', 'steps': [
        {'action': 'Pick', 'result': MISSING, 'success': False}], 'total_steps': 1},
        str(output), is_pddl_run=True)
    assert [s[2] for s in StepSpans(output.read_text()).spans] == ['failure']


@pytest.mark.parametrize('before,known,skipped', [
    ('Robot SG before action: ["cup_0"]\n', True, True),
    ('Robot SG before action: ["cup_01"]\n', False, False),
    ('Robot SG before action: []\n', False, False),
    ('Objects: cup_0: on table_0\n', None, False),
    ('Objects: cup_01: on table_0\n', None, False),
    ('Objects: No objects found yet\n', None, False),
    ('', None, False),
    ('Thought: cup_0 exists in my scene graph\n', None, False),
    ('Robot SG before action: invalid json\n', None, False),
])
def test_only_prior_exact_membership_exempts_pick(tmp_path, before, known, skipped):
    from scripts.output_parser import TraceParser
    trace = tmp_path / 'trace.txt'
    trace.write_text('Task: demo\n' + SG_POLICY_MARKER + '\n' + before + 'Thought: Try cup\nPick[cup_0]\n'
                     f'Assigned!Result: {MISSING}\nObjects: cup_0: on table_0\n')
    terminal = TraceParser(str(trace))
    terminal.parse()
    for steps in (view_trace_logs.parse_trace_file(str(trace))['steps'],
                  parse_react_trace_file(str(trace))['steps'], terminal.steps):
        assert len(steps) == 1
        step = steps[0]
        assert step['target_in_robot_sg_before_action'] is known
        assert step['skipped'] is skipped
        assert step['success'] is (None if skipped else False)
        assert infer_trace_issues([step])['failed_step_count'] == (0 if skipped else 1)


def test_removed_object_is_not_exempt_on_retry(tmp_path):
    trace = tmp_path / 'trace.txt'
    trace.write_text('Task: demo\nRobot SG before action: ["cup_0"]\nThought: first\n'
                     f'Pick[cup_0]\nAssigned!Result: {MISSING}\n'
                     'Objects: cup_0: on table_0\nRobot SG before action: []\nThought: retry\n'
                     f'Pick[cup_0]\nAssigned!Result: {MISSING}\n')
    for parsed in (view_trace_logs.parse_trace_file(str(trace)), parse_react_trace_file(str(trace))):
        assert [s['skipped'] for s in parsed['steps']] == [True, False]
        assert infer_trace_issues(parsed['steps'])['failed_step_count'] == 1


def test_historical_trace_keeps_original_classification(tmp_path):
    trace = tmp_path / 'trace.txt'
    trace.write_text(f'Task: demo\nThought: old run\nPick[cup_0]\nAssigned!Result: {MISSING}\n')
    for parsed in (view_trace_logs.parse_trace_file(str(trace)), parse_react_trace_file(str(trace))):
        assert parsed['steps'][0]['skipped'] is True
        assert parsed['steps'][0]['react_skip_policy'] == 'legacy'
        assert parsed['steps'][0]['target_in_robot_sg_before_action'] is None


def test_marker_copies_robot_graph_before_mutation():
    from types import SimpleNamespace
    objects = [SimpleNamespace(name='cup_0')]
    marker = robot_scene_graph_marker(SimpleNamespace(get_all_objects=lambda: objects))
    objects.clear()
    assert react_action_evidence(marker + 'Pick[cup_0]')[0]['target_in_robot_sg_before_action'] is True


def test_new_run_unknown_evidence_is_a_failure_in_html(tmp_path):
    step = {'action': 'Pick', 'args': 'cup_0', 'result': MISSING,
            'success': None, 'skipped': True, 'status': 'skipped',
            'react_skip_policy': 'robot_sg_before_action'}
    assert mark_react_skip(step)['success'] is False
    output = tmp_path / 'report.html'
    view_trace_logs.generate_html({'task': 'demo', 'steps': [step], 'total_steps': 1}, str(output))
    text = output.read_text()
    assert StepSpans(text).spans[0][2] == 'failure'
    assert '0.0% (0/1)' in text
    assert 'data-react-skip-policy="robot_sg_before_action"' in text
    assert refresh_html(text) == (text, 0)


def test_real_planner_dispatch_records_graph_before_pick_removes_object():
    """Exercise the actual get_next_action ordering, with no model or simulator."""
    from types import SimpleNamespace
    from habitat_llm.planner.zero_shot_react_planner import ZeroShotReactPlanner

    planner = ZeroShotReactPlanner.__new__(ZeroShotReactPlanner)
    planner.llm = SimpleNamespace()
    planner._agents = []
    planner.reset()
    planner._agents = [SimpleNamespace(uid=0)]
    planner.curr_prompt = 'existing initial prompt'
    planner.stopword = 'Assigned!'
    planner.planner_config = SimpleNamespace(llm=SimpleNamespace(eot_tag=''))
    planner.replan = lambda *args: {'llm_response': 'Thought: try cup\nPick[cup_0]'}
    planner.parse_thought = lambda response: 'try cup'
    planner._check_termination = lambda response: False
    planner.actions_parser = lambda *args: {0: ('Pick', 'cup_0', None)}
    planner._try_fast_explore_actions = lambda *args: None
    planner._maybe_finish_explore_calibration = lambda *args: None
    planner.get_last_agent_states = lambda: {}
    planner.get_last_agent_positions = lambda: {}
    planner.get_agent_collisions = lambda: {}
    planner._cost_metrics = lambda: {}
    objects = [SimpleNamespace(name='cup_0')]
    graph = SimpleNamespace(get_all_objects=lambda: objects)

    def dispatch(*args):
        assert 'Robot SG before action: ["cup_0"]' in planner.trace
        objects.clear()  # The pick discovers that remembered cup no longer exists.
        return {}, {0: MISSING}

    def feedback(responses):
        planner.trace += f'Result: {responses[0]}\nObjects: No objects found yet\n'
        return ''

    planner.process_high_level_actions = dispatch
    planner._add_responses_to_prompt = feedback
    _, info, _ = planner.get_next_action('demo', {}, {0: graph})
    parsed = view_trace_logs._parse_trace_file_react(info['traces'][0])
    assert parsed['steps'][0]['skipped'] is True
    assert parsed['steps'][0]['pre_action_robot_sg_source'] == 'pre_action_robot_graph'
    assert parsed['steps'][0]['thought'] == 'try cup'
    assert not objects
    assert 'Robot SG before action:' not in planner.curr_prompt


@pytest.mark.parametrize('action,args,skipped', [
    ('Rearrange', 'cup_0, on, table_0, None, None', True),
    ('Rearrange', 'cup_01, on, table_0, None, None', False),
    ('Navigate', 'cup_0', False),
])
def test_legacy_composite_pick_uses_its_object_target(action, args, skipped):
    parsed = view_trace_logs._parse_trace_file_react(
        f'Task: demo\nRobot SG before action: ["cup_0"]\nThought: Try\n{action}[{args}]\n'
        f'Assigned!Result: {MISSING}\n')
    assert parsed['steps'][0]['skipped'] is skipped
