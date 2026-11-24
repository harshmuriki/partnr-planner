#!/usr/bin/env python3
# Tests for RailPlanner's object-centric mode (plan_config.subgoal_mode=object_centric): task-list parsing, the
# grounding goal text, and the decompose -> ground one task -> advance -> review flow. No sim, GPU or API.

from types import SimpleNamespace

import pytest

from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner
from habitat_llm.rail_planner import RailPlanner
from habitat_llm.rail_planner import object_centric as oc
from habitat_llm.rail_planner import planner as rail_planner_mod

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_task_list():
    text = ('```json\n{"tasks": [{"objects": "all lights in the bedroom", "requirements": ["turned off"]},'
            ' {"objects": "jug", "requirements": ["filled", "on the living-room table"]}]}\n```')
    tasks, decision, _, parsed = oc.parse_task_list(text)
    assert parsed and decision is None
    assert [t.objects for t in tasks] == ["all lights in the bedroom", "jug"]
    assert tasks[1].requirements == ["filled", "on the living-room table"] and tasks[1].idx == 1
    assert oc.parse_task_list('{"decision": "Complete", "reason": "r"}')[1:3] == ("complete", "r")
    assert oc.parse_task_list("no json here") == ([], None, "", False)
    assert oc.parse_task_list('{"tasks": []}')[3] is False
    # Strings and a single requirement string are accepted; start_idx continues numbering.
    tasks, *_ = oc.parse_task_list('{"tasks": ["the mug", {"object": "plate", "conditions": "clean"}]}', start_idx=3)
    assert [(t.idx, t.objects, t.requirements) for t in tasks] == [(3, "the mug", []), (4, "plate", ["clean"])]


def test_goal_text_and_prompts():
    tasks, *_ = oc.parse_task_list('{"tasks": [{"objects": "lights", "requirements": ["off"]},'
                                   ' {"objects": "jug", "requirements": ["filled"]},'
                                   ' {"objects": "cup", "requirements": ["on table"]}]}')
    tasks[0].status, tasks[1].status = "done", "active"
    text = oc.task_goal_text("Do the chores.", tasks, tasks[1])
    assert text.startswith("Do the chores.") and "[OBJECT TASK 2 of 3]" in text and "jug -> filled" in text
    assert "Done object tasks: lights -> off" in text and "cup -> on table" in text
    assert '"tasks"' in oc.build_decompose_prompt("g", "mem", ["Action failed: x"]) and "Action failed: x" in \
        oc.build_decompose_prompt("g", "mem", ["Action failed: x"])
    review = oc.build_review_prompt("g", tasks, "mem", [], extra="- box_1 is not placed correctly.")
    assert "1. lights -> off [done]" in review and "box_1 is not placed correctly" in review


# ---------------------------------------------------------------------------
# Planner flow (built without __init__; only the state these paths touch is set)
# ---------------------------------------------------------------------------


def _rail(mode="object_centric"):
    p = RailPlanner.__new__(RailPlanner)
    p.subgoal_mode = mode
    p.object_task_max_regrounds, p.object_task_max_reviews = 1, 2
    p._tasks, p._task_advance_pending, p._task_reviews, p._oc_final = None, False, 0, False
    p.two_phase, p._mode = False, "interact"
    p.placement_oracle_mode = "none"
    p.end_of_plan_check, p.max_end_of_plan_replans, p._end_of_plan_replans = "none", 3, 0
    p._end_replan_pending, p._end_feedback = False, []
    p.history, p._already_succeeded_subgoals = [], []
    p.is_done, p._world_graph, p._agents = False, None, []
    p.events = []
    p._log_event = p.events.append
    p._trace_append = p._vprint = p._vprint_header = lambda *a, **k: None
    p.finished = []
    p._oc_scene = lambda wg: "memory"
    return p


def _script_vlm(p, responses):
    """Each _oc_vlm_call pops the next scripted response."""
    responses = list(responses)
    p.oc_prompts = []
    p._oc_vlm_call = lambda kind, prompt, obs, wg: (p.oc_prompts.append((kind, prompt)) or (responses.pop(0), 0))
    p.vlm = SimpleNamespace(get_last_response_metadata=lambda: {})


def _finish_recorder(p, monkeypatch):
    def finish(self, reason, evidence=""):
        self.is_done = True
        self.finished.append(reason)
    monkeypatch.setattr(VlmTampPddlPlanner, "_finish_planning", finish)


TWO_TASKS = ('{"tasks": [{"objects": "lights", "requirements": ["off"]},'
             ' {"objects": "jug", "requirements": ["filled"]}]}')


def test_flat_never_decomposes(monkeypatch):
    p = _rail(mode="flat")
    p._oc_vlm_call = lambda *a: pytest.fail("flat must not call the object-task VLM")
    seen = []
    p._rail_generate_subgoals = lambda instr, wg, obs, history_str="", **kw: seen.append((instr, history_str, kw)) or ["b"]
    assert p._generate_subgoals("task", None, {}, history_str="h", append=True) == ["b"]
    assert seen == [("task", "h", {"append": True})] and p._tasks is None


def test_decompose_then_ground_first_task(monkeypatch):
    p = _rail()
    _script_vlm(p, [TWO_TASKS])
    goals = []
    p._rail_generate_subgoals = lambda goal, *a, **kw: goals.append(goal) or ["branch"]
    assert p._generate_subgoals("Turn off the lights and fill the jug.", None, {}) == ["branch"]
    assert [t.status for t in p._tasks] == ["active", "pending"]
    assert "[OBJECT TASK 1 of 2] Plan ONLY this object task now: lights -> off" in goals[0]
    assert [e["event"] for e in p.events][:2] == ["object_tasks_defined", "object_task_start"]
    # A failure replan re-grounds the same task (no new decomposition).
    p._generate_subgoals("Turn off the lights and fill the jug.", None, {}, refresh_failure_observation=True)
    assert len(p.oc_prompts) == 1 and "lights -> off" in goals[1] and p._tasks[0].regrounds == 1


def test_unparsable_decomposition_falls_back_to_whole_task():
    p = _rail()
    _script_vlm(p, ["sorry"])
    p._rail_generate_subgoals = lambda goal, *a, **kw: ["b"]
    p._generate_subgoals("Do it all.", None, {})
    assert len(p._tasks) == 1 and p._tasks[0].requirements == ["Do it all."] and "fallback" in p._tasks[0].note


def test_plan_completed_advances_to_next_task(monkeypatch):
    p = _rail()
    _finish_recorder(p, monkeypatch)
    _script_vlm(p, [TWO_TASKS])
    goals = []
    p._rail_generate_subgoals = lambda goal, *a, **kw: goals.append(goal) or ["b"]
    p._generate_subgoals("task", None, {})
    p._finish_planning("plan_completed", "branch done")
    assert not p.is_done and p._task_advance_pending and p._tasks[0].status == "done"
    p._task_advance_pending = False
    p._generate_subgoals("task", None, {}, append=True, observation_reason="object_task_done")
    assert p._tasks[1].status == "active" and "[OBJECT TASK 2 of 2]" in goals[-1]
    assert "Done object tasks: lights -> off" in goals[-1]


def test_turn1_complete_closes_task_and_grounds_next_in_same_call(monkeypatch):
    p = _rail()
    _finish_recorder(p, monkeypatch)
    _script_vlm(p, [TWO_TASKS])
    goals = []

    def ground(goal, *a, **kw):
        goals.append(goal)
        if len(goals) == 1:  # task 1 already holds
            assert p._handle_turn1_decision({"decision": "complete", "reason": "lights are off"})
            return None
        return ["b"]

    p._rail_generate_subgoals = ground
    assert p._generate_subgoals("task", None, {}) == ["b"]
    assert [t.status for t in p._tasks] == ["done", "active"] and not p.finished
    assert "OBJECT TASK 2" in goals[1]


def test_reground_cap_skips_task(monkeypatch):
    p = _rail()
    _finish_recorder(p, monkeypatch)
    _script_vlm(p, [TWO_TASKS])
    goals = []
    p._rail_generate_subgoals = lambda goal, *a, **kw: goals.append(goal) or ["b"]
    p._generate_subgoals("task", None, {})
    p._generate_subgoals("task", None, {}, refresh_failure_observation=True)  # reground 1 (cap 1)
    p._generate_subgoals("task", None, {}, refresh_failure_observation=True)  # over the cap -> skip
    assert p._tasks[0].status == "failed" and p._tasks[1].status == "active"
    assert any("skipped" in h for h in p.history)


def test_review_complete_finishes_through_end_of_plan_path(monkeypatch):
    p = _rail()
    _finish_recorder(p, monkeypatch)
    _script_vlm(p, ['{"tasks": [{"objects": "lights", "requirements": ["off"]}]}',
                    '{"decision": "complete", "reason": "all done"}'])
    p._rail_generate_subgoals = lambda goal, *a, **kw: ["b"]
    checks = []
    monkeypatch.setattr(rail_planner_mod, "gt_goal_status", lambda env: checks.append(1) or (True, 1.0))
    p.end_of_plan_check, p.env_interface = "oracle", None
    p._generate_subgoals("task", None, {})
    p._finish_planning("plan_completed")  # closes the only task; no oracle check yet
    assert not checks and p._task_advance_pending
    p._generate_subgoals("task", None, {}, append=True, observation_reason="object_task_done")
    assert p.finished == ["plan_completed"] and checks == [1] and p._task_reviews == 1
    assert [k for k, _ in p.oc_prompts] == ["object_task_decompose", "object_task_review"]


def test_review_adds_tasks_and_continues(monkeypatch):
    p = _rail()
    _finish_recorder(p, monkeypatch)
    _script_vlm(p, ['{"tasks": [{"objects": "box", "requirements": ["on the center table"]}]}',
                    '{"tasks": [{"objects": "box", "requirements": ["next to the scissors"]}]}'])
    goals = []
    p._rail_generate_subgoals = lambda goal, *a, **kw: goals.append(goal) or ["b"]
    p._generate_subgoals("task", None, {})
    p._finish_planning("plan_completed")
    p._generate_subgoals("task", None, {}, append=True, observation_reason="object_task_done")
    assert not p.finished and len(p._tasks) == 2 and p._tasks[1].status == "active"
    assert "[OBJECT TASK 2 of 2]" in goals[-1] and "next to the scissors" in goals[-1]


def test_decompose_terminal_decision_ends_episode(monkeypatch):
    p = _rail()
    _finish_recorder(p, monkeypatch)
    _script_vlm(p, ['{"decision": "unsolvable", "reason": "no jug anywhere"}'])
    p._rail_generate_subgoals = lambda *a, **k: pytest.fail("no grounding after a terminal decision")
    assert p._generate_subgoals("task", None, {}) is None and p.finished == ["unsolvable"]


def test_flat_turn1_decision_unchanged(monkeypatch):
    p = _rail(mode="flat")
    _finish_recorder(p, monkeypatch)
    assert p._handle_turn1_decision({"decision": "complete", "reason": "r"}) and p.finished == ["complete"]
