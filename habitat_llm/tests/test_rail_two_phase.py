#!/usr/bin/env python3
# Tests for RailPlanner's two_phase mode (plan_config.two_phase): mode selector parsing, explore-free
# interaction prompts, the ReAct explorer's action guard, and the explore/interact hand-off. No sim, GPU or API.

from types import SimpleNamespace

import pytest

from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner
from habitat_llm.planner.zero_shot_react_planner import ZeroShotReactPlanner
from habitat_llm.rail_planner import RailPlanner
from habitat_llm.rail_planner.mode_selector import build_selector_prompt, parse_selector_response
from habitat_llm.rail_planner.react_explorer import EXPLORE_TOOLS, ExploreReactPlanner
from habitat_llm.vlm_tamp.prompts_vlm_tamp import (
    build_english_subgoal_prompt,
    build_predicate_translation_prompt,
)

# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


def test_parse_selector_response():
    assert parse_selector_response('{"mode": "explore", "reason": "r", "explore_goal": "find the mug"}') == (
        "explore", "r", "find the mug", True)
    assert parse_selector_response('```json\n{"mode": "INTERACT", "reason": "known"}\n```')[:2] == ("interact", "known")
    assert parse_selector_response("let me think")[0] == "interact"
    assert parse_selector_response('{"mode": "dance"}')[3] is False
    assert parse_selector_response(None)[0] == "interact"


def test_selector_prompt_lists_history_and_phases():
    text = build_selector_prompt("tidy", "mug_0 on table_0", "start of the task", ["Action failed: x"],
                                 ["phase 1 (goal: mug): ..."], 5)
    assert "Action failed: x" in text and "phase 1 (goal: mug)" in text and "Exploration phases left: 5" in text
    assert '"mode"' in text


# ---------------------------------------------------------------------------
# Interaction prompts without explore()
# ---------------------------------------------------------------------------


def test_prompts_without_explore():
    default = build_predicate_translation_prompt({"movable": ["mug_0"]})
    no_explore = build_predicate_translation_prompt({"movable": ["mug_0"]}, allow_explore=False)
    assert "explore(<room>)" in default and "explore(" not in no_explore
    english = build_english_subgoal_prompt("g", {"movable": ["mug_0"]}, "d", allow_explore=False)
    assert '"decision":"explore"' in english and "Use the explore action" not in english
    # Default output is unchanged by the new argument.
    assert build_english_subgoal_prompt("g", {}, "d") == build_english_subgoal_prompt("g", {}, "d", allow_explore=True)


# ---------------------------------------------------------------------------
# RailPlanner hand-off (planner built without __init__; only the two_phase state is set)
# ---------------------------------------------------------------------------


def _rail(two_phase=True, max_skills=2):
    p = RailPlanner.__new__(RailPlanner)
    p.two_phase = two_phase
    p.max_explore_phases = 2
    p.explore_phase_max_skills = max_skills
    p._mode, p._explore_phases, p._explore_summaries, p._phase = "interact", 0, [], None
    p._explorer, p._two_phase_started = None, False
    p.placement_oracle_mode = "none"
    p.history, p._already_succeeded_subgoals = [], []
    p.current_plan, p.current_action_idx, p.last_high_level_actions = [], 0, {}
    p._subgoal_retry_count, p._replan_pending, p._end_replan_pending = 0, False, False
    p._reprompt_round, p._num_actions, p.max_actions = 0, 0, 50
    p.branch_idx, p.subgoal_idx, p.branches, p.is_done = 0, 0, [], False
    p._world_graph = SimpleNamespace(get_all_objects=lambda: [])
    p._agents = []
    p.events = []
    p._log_event = p.events.append
    p._trace_append = p._vprint = p._vprint_header = lambda *a, **k: None
    p._memory_snapshot = lambda wg: {}
    p._observe_failure_state = lambda obs, reason="failure": obs
    p._planner_info = lambda uid, **kw: dict(kw)
    return p


def test_flag_off_never_selects(monkeypatch):
    p = _rail(two_phase=False)
    calls = []
    monkeypatch.setattr(VlmTampPddlPlanner, "_generate_subgoals", lambda self, *a, **k: calls.append(k) or ["b"])
    p._choose_mode = lambda *a: pytest.fail("selector must not run with two_phase=false")
    assert p._generate_subgoals("task", p._world_graph, {}) == ["b"]
    assert p._get_explorer is not None and p._explorer is None
    seen = {}
    monkeypatch.setattr(VlmTampPddlPlanner, "_build_english_prompt", lambda self, **k: seen.update(k) or "")
    p._build_english_prompt(goal="g")
    assert "allow_explore" not in seen


def test_selector_explore_skips_interaction(monkeypatch):
    p = _rail()
    monkeypatch.setattr(VlmTampPddlPlanner, "_generate_subgoals", lambda *a, **k: pytest.fail("no Turn 1 on explore"))
    started = []
    p._choose_mode = lambda *a: started.append(a) or "explore"
    assert p._generate_subgoals("task", p._world_graph, {}) is None and started


def test_two_phase_drops_explore_subgoals_and_handles_explore_decision():
    p = _rail()
    p._objects_by_type = {}
    assert p._validate_subgoal("explore(kitchen_1)") is False
    begun = []
    p._begin_explore_phase = lambda goal, reason, source: begun.append((goal, source))
    assert p._handle_turn1_decision({"decision": "explore", "reason": "find the mug in the bedroom"})
    assert begun == [("find the mug in the bedroom", "interaction_vlm")]
    p._explore_phases = p.max_explore_phases
    finished = []
    p._finish_planning = lambda reason, evidence="": finished.append(reason)
    assert p._handle_turn1_decision({"decision": "explore"}) and finished == ["budget_exhausted"]


class _FakeExplorer:
    """Scripted stand-in for ExploreReactPlanner: each get_next_action pops one (replanned, action, response, done)."""

    def __init__(self, script):
        self.script = list(script)
        self.replan_required, self.is_done, self.rejected = True, False, False
        self._termination, self.curr_prompt, self.last_high_level_actions = {}, "", {}
        self.phase_goal = ""

    def start_phase(self, goal, previous):
        self.phase_goal, self.is_done, self.replan_required = goal, False, True

    def phase_instruction(self, task):
        return task

    def get_next_action(self, instruction, observations, world_graphs):
        replanned, action, response, done = self.script.pop(0)
        self.is_done = done
        self.replan_required = bool(response)
        info = {"replanned": {0: replanned}, "high_level_actions": {0: action} if action else {},
                "responses": {0: response}, "thought": {0: "t"}}
        return ({0: "low"} if not done else {}), info, done


def test_explore_phase_counts_actions_and_replans_when_done(tmp_path):
    p = _rail()
    p._get_log_dir = lambda: str(tmp_path)
    p._explorer = _FakeExplorer([
        (True, ("Navigate", "table_0", None), "", False),
        (False, ("Navigate", "table_0", None), "Successful execution!", False),
        (True, ("Done", None, None), "", True),
    ])
    p._get_explorer = lambda: p._explorer
    replans = []
    p._generate_subgoals = lambda *a, **k: replans.append(k)
    p._begin_explore_phase("find the mug", "r", "selector")
    assert p._mode == "explore"
    low, info, done = p._explore_step("task", {}, {0: p._world_graph}, 0)
    assert low == {0: "low"} and p._num_actions == 1 and info["replanned"] == {0: True}
    p._explore_step("task", {}, {0: p._world_graph}, 0)
    low, info, done = p._explore_step("task", {}, {0: p._world_graph}, 0)
    assert p._mode == "interact" and low == {} and not done
    assert replans and replans[0]["observation_reason"] == "explore_phase_done"
    assert [e["event"] for e in p.events].count("explore_phase_end") == 1
    assert (tmp_path / "explore_phase_01.txt").exists()


def test_explore_phase_stops_at_skill_cap(tmp_path):
    p = _rail(max_skills=1)
    p._get_log_dir = lambda: str(tmp_path)
    p._explorer = _FakeExplorer([(True, ("Explore", "kitchen_1", None), "", False),
                                 (False, ("Explore", "kitchen_1", None), "Successful execution!", False)])
    p._get_explorer = lambda: p._explorer
    p._generate_subgoals = lambda *a, **k: None
    p._begin_explore_phase("g", "r", "selector")
    p._explore_step("task", {}, {0: p._world_graph}, 0)
    p._explore_step("task", {}, {0: p._world_graph}, 0)
    p._explore_step("task", {}, {0: p._world_graph}, 0)  # would ask the LLM again: cap ends the phase
    end = [e for e in p.events if e["event"] == "explore_phase_end"]
    assert end and end[0]["why"] == "skill_cap" and p._num_actions == 1


# ---------------------------------------------------------------------------
# ReAct explorer
# ---------------------------------------------------------------------------


def test_explorer_rejects_manipulation(monkeypatch):
    e = ExploreReactPlanner.__new__(ExploreReactPlanner)
    e._agents = []
    low, responses = e.process_high_level_actions({0: ("Pick", "mug_0", None)}, {})
    assert low == {} and "not allowed" in responses[0] and e.rejected
    monkeypatch.setattr(ZeroShotReactPlanner, "process_high_level_actions", lambda self, hl, obs: ({0: "ok"}, {0: ""}))
    assert e.process_high_level_actions({0: ("Navigate", "table_0", None)}, {}) == ({0: "ok"}, {0: ""})
    assert not e.rejected
    # Parse errors keep the base handling (error message as the result).
    assert e.process_high_level_actions({0: ("Pikc", "x", "bad tool")}, {}) == ({0: "ok"}, {0: ""})


def test_explorer_prompt_lists_only_explore_tools(monkeypatch):
    e = ExploreReactPlanner.__new__(ExploreReactPlanner)
    tools = {n: SimpleNamespace(description=f"{n} desc") for n in ("Explore", "Navigate", "Open", "Close", "Pick", "Place")}
    e._agents = [SimpleNamespace(tools=tools)]
    e.prompt = "{tool_descriptions}"
    monkeypatch.setattr(ZeroShotReactPlanner, "prepare_prompt",
                        lambda self, i, wg, **k: ("all", {"tool_descriptions": "all"}))
    prompt, _ = e.prepare_prompt("task", None)
    assert all(f"- {t}:" in prompt for t in EXPLORE_TOOLS) and "Pick" not in prompt and "Place" not in prompt
