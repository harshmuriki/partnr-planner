"""Trace camera provenance and failure identity regressions (no simulator needed)."""
from types import SimpleNamespace

import numpy as np
from PIL import Image

from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner


def make_planner(tmp_path):
    planner = VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    planner.pddl_baseline_html = True
    planner._get_log_dir = lambda: str(tmp_path)
    planner._trace_observations = {"agent_0_third_rgb": np.zeros((2, 3, 3), dtype=np.uint8)}
    return planner


def test_capture_prefers_fresh_render_and_keeps_retry_images(tmp_path):
    planner = make_planner(tmp_path)
    frame = np.full((2, 3, 3), 127, dtype=np.uint8)
    planner.env_interface = SimpleNamespace(sim=SimpleNamespace(
        get_sensor_observations=lambda: {"agent_0_third_rgb": frame}))
    first = planner._save_subgoal_image(2, 1, "failed")
    second = planner._save_subgoal_image(2, 1, "failed")
    assert first["image_path"] != second["image_path"]
    assert first["image_source"] == "fresh_sensor_render"
    assert np.array_equal(np.asarray(Image.open(tmp_path / first["image_path"])), frame)
    assert (tmp_path / second["image_path"]).is_file()


def test_capture_labels_fallback_and_missing_camera(tmp_path):
    planner = make_planner(tmp_path)
    result = planner._save_subgoal_image(0, 0, "started")
    assert result["image_source"] == "latest_planner_observation"
    planner._trace_observations = {}
    assert "image_unavailable" in planner._save_subgoal_image(0, 0, "solved")


def test_failure_log_flushed_before_branch_switch(tmp_path):
    p = make_planner(tmp_path)
    p.is_done = False
    p._agents = []
    p.trace = ["existing trace"]
    p._episode_banner_printed = True
    p.enable_partial_obs_explore = False
    p.branches = [["on(scissors_0, table_12)"], ["opened-drawer(drawer)"]]
    p.branch_idx = 0
    p.subgoal_idx = 0
    p.current_plan = [("pick", [])]
    p.current_action_idx = 0
    p.last_high_level_actions = {0: ("Pick", "scissors_0", None)}
    p._subgoal_retry_count = 0
    p.ntamp_max_pddl_replan_retries = 0
    p.history = []
    p._subgoal_exec_log_lines = []
    p._reprompt_round = 0
    p.verbose = False
    p._vprint = lambda *a, **k: None
    p._trace_append = lambda *a: None
    p._log_subgoal_status = lambda **k: None
    p.process_high_level_actions = lambda *a: ({}, {0: "Unexpected failure!"})
    p._response_failed = lambda response: True
    records = []
    p._log_event = records.append
    # Stop at the branch transition: it must already have flushed the old identity.
    class TransitionReached(Exception):
        pass
    def advance():
        p.branch_idx = 1
        p.subgoal_idx = 0
        raise TransitionReached
    p._advance_branch = advance
    import pytest
    with pytest.raises(TransitionReached):
        p.get_next_action("test", {}, {0: object()})
    assert len(records) == 1
    assert records[0]["event"] == "subgoal_execution"
    assert records[0]["branch"] == 0
    assert records[0]["subgoal"] == "on(scissors_0, table_12)"
    assert records[0]["status"] == "failed"


def test_failure_reprompt_uses_fresh_frame_before_building_prompt(tmp_path):
    p = make_planner(tmp_path)
    p.use_images = True
    p._reprompt_round = 1
    p.branch_idx = 2
    p.subgoal_idx = 3
    frame = np.full((2, 3, 3), 91, dtype=np.uint8)
    p.env_interface = SimpleNamespace(sim=SimpleNamespace(
        get_sensor_observations=lambda: {"agent_0_third_rgb": frame}))
    events = []
    p._log_event = events.append
    stale = dict(p._trace_observations, other_sensor="preserved")
    class PromptReached(Exception):
        pass
    def inspect(observations, world_graph):
        assert np.array_equal(observations["agent_0_third_rgb"], frame)
        assert observations["other_sensor"] == "preserved"
        assert observations["agent_0_third_rgb"] is not frame
        raise PromptReached
    p._extract_visible_entity_names = inspect
    import pytest
    with pytest.raises(PromptReached):
        p._generate_subgoals("test", object(), stale, refresh_failure_observation=True)
    assert not stale["agent_0_third_rgb"].any()
    assert events[0]["image_available"] is True


def test_failure_camera_error_never_resends_stale_image(tmp_path):
    p = make_planner(tmp_path)
    p.use_images = True
    p._reprompt_round = p.branch_idx = p.subgoal_idx = 0
    def unavailable():
        raise RuntimeError("camera unavailable")
    p.env_interface = SimpleNamespace(sim=SimpleNamespace(get_sensor_observations=unavailable))
    events = []
    p._log_event = events.append
    p._vprint = lambda *a: None
    refreshed = p._observe_failure_state(p._trace_observations)
    assert "agent_0_third_rgb" not in refreshed
    assert events[0]["image_available"] is False
    assert events[0]["error"] == "camera unavailable"


def exploration_planner(tmp_path):
    p = make_planner(tmp_path)
    p.use_images = True
    p.branch_idx = p.subgoal_idx = 0
    p.explore_image_interval = 1
    p._log_event = lambda e: None
    p._collect_gt_projected_boxes = lambda *a: [("table_0", "red", 0, 0, 1, 1)]
    p._draw_gt_box_list_on_image = lambda *a: None
    p._save_vlm_images = lambda images: None
    return p


def test_explore_selects_temporally_spaced_views_and_resets(tmp_path):
    p = exploration_planner(tmp_path)
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    p.env_interface = SimpleNamespace(sim=SimpleNamespace(
        get_sensor_observations=lambda: {"agent_0_third_rgb": frame}))
    p._begin_explore_images({}, object(), "kitchen_0")
    for i in range(1, 9):
        frame[:] = i
        p._capture_explore_image({}, object())
    urls, records = p._explore_vlm_images()
    assert len(urls) == 3
    assert [r["capture_tick"] for r in records] == [0, 4, 8]
    assert all(r["visible_entity_names"] == ["table_0"] for r in records)
    assert all((tmp_path / r["image_path"]).is_file() for r in records)
    p._begin_explore_images({}, object(), "bedroom_0")
    urls, records = p._explore_vlm_images()
    assert len(urls) == 1
    assert records[0]["room"] == "bedroom_0"


def test_explore_deduplicates_stationary_frames_and_forces_final_capture(tmp_path):
    p = exploration_planner(tmp_path)
    p.explore_image_interval = 30
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    p.env_interface = SimpleNamespace(sim=SimpleNamespace(
        get_sensor_observations=lambda: {"agent_0_third_rgb": frame}))
    p._begin_explore_images({}, object(), "room")
    p._capture_explore_image({}, object(), force=True)
    assert len(p._explore_image_records) == 1
    frame[:] = 10
    p._capture_explore_image({}, object())
    assert len(p._explore_image_records) == 1
    p._capture_explore_image({}, object(), force=True)
    urls, records = p._explore_vlm_images()
    assert len(urls) == 2


def test_vlm_request_routes_explore_sequence_and_current_failure_image(tmp_path, monkeypatch):
    import importlib
    import pytest
    module = importlib.import_module("habitat_llm.planner.vlm_tamp_pddl_planner")
    monkeypatch.setattr(module, "build_english_subgoal_prompt", lambda **kw: "Plan the task.")
    p = exploration_planner(tmp_path)
    p._agents = []
    p._reprompt_round = 1
    p._build_objects_by_type = lambda *a, **k: {}
    p._build_scene_description = lambda *a, **k: "scene"
    p._extract_visible_entity_names = lambda *a: set()
    p._vprint = p._vprint_header = lambda *a: None
    p._append_vlm_prompt = lambda **k: None
    p.vlm_max_tokens = 100
    p.vlm_temperature = 0.2
    p.vlm_reasoning_effort = "high"
    calls = []
    class Requested(Exception):
        pass
    def ask(prompt, **kwargs):
        calls.append((prompt, kwargs))
        raise Requested
    p.vlm = SimpleNamespace(new_session=lambda: None, ask=ask)
    p._explore_vlm_images = lambda: (["early", "middle", "late"], [
        {"capture_tick": i, "visible_entity_names": ["table_0"]} for i in [0, 30, 60]])
    wg = SimpleNamespace(get_all_objects=lambda: [])
    with pytest.raises(Requested):
        p._generate_subgoals("task", wg, {}, after_explore=True, explored_room="room")
    assert calls[-1][1]["image_data_urls"] == ["early", "middle", "late"]
    assert "different times/viewpoints" in calls[-1][0]
    fresh = {"agent_0_third_rgb": "fresh"}
    p._observe_failure_state = lambda observations: fresh
    def failure_image(observations, world_graph, *, annotate):
        assert observations is fresh
        assert annotate is False
        return ["current"]
    p._get_annotated_vlm_image_urls = failure_image
    with pytest.raises(Requested):
        p._generate_subgoals("task", wg, {}, refresh_failure_observation=True)
    assert calls[-1][1]["image_data_urls"] == ["current"]
    assert "fresh third-person" in calls[-1][0]


def test_unquoted_one_line_predicates_keep_argument_commas():
    from habitat_llm.vlm_tamp.parse_utils import parse_branch_response
    assert parse_branch_response(
        '[explore(entryway_1), explore(living_room_1), on(box_0, table_12)]'
    ) == [['explore(entryway_1)', 'explore(living_room_1)', 'on(box_0, table_12)']]


def test_vlm_logs_and_viewer_use_flat_run_directory(tmp_path):
    from habitat_llm.examples.planner_demo import _get_vlm_tamp_pddl_log_dir
    from habitat_llm.vlm_tamp.render_pddl_baseline_html import _default_interactive_pddl_html_basename
    run = tmp_path / 't2-inc-con'
    config = SimpleNamespace(paths=SimpleNamespace(results_dir=str(run)))
    env = SimpleNamespace(conf=config)
    p = VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    p.env_interface = env
    p._log_dir = None
    p.log_dir_name = 'vlm_tamp_pddl'
    expected = str(run / 'vlm_tamp_pddl')
    assert p._get_log_dir() == expected
    assert _get_vlm_tamp_pddl_log_dir(config, env) == expected
    assert _default_interactive_pddl_html_basename(expected) == 't2-inc-con_pddl.html'
    assert _default_interactive_pddl_html_basename(expected + '/old_episode') == 't2-inc-con_pddl.html'
    assert not (run / 'vlm_tamp_pddl' / 'unknown').exists()
