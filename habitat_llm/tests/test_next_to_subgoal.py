#!/usr/bin/env python3
# Tests for the next_to(<movable>, <reference_movable>, <surface_furniture>) subgoal (plan_config.next_to_subgoal):
# Turn-2 prompt, validation, the PDDL goal/action (VLM-TAMP) and the RailPlanner expander, both of which must emit
# Place[obj, on, furn, next_to, ref]. With the flag off nothing changes.

import pytest

from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner
from habitat_llm.pddlstream.problem import build_pddlstream_problem, extract_scope_names
from habitat_llm.rail_planner import RailPlanner
from habitat_llm.rail_planner.subgoal_expanders import SubgoalPlanError, build_scene_state, expand_subgoal
from habitat_llm.tests.test_rail_planner import _graph, _plan


def _planner(cls=VlmTampPddlPlanner, on=True):
    p = cls.__new__(cls)
    p.next_to_subgoal = on
    p._objects_by_type = {"movable": ["mug_0", "jug_0", "cup_0"], "surface_furniture": ["table_0", "bed_0"]}
    p._vprint = lambda *a, **k: None
    p._log_event = lambda *a, **k: None
    return p


# ---------------------------------------------------------------------------
# Prompt and validation (shared by VLM-TAMP and RailPlanner)
# ---------------------------------------------------------------------------


def test_prompt_lists_next_to_only_when_enabled():
    objs = {"movable": ["mug_0"]}
    assert "next_to(" not in _planner(on=False)._build_predicate_prompt(objs)
    on = _planner(on=True)._build_predicate_prompt(objs)
    assert "'next_to(<movable>, <reference_movable>, <surface_furniture>)'" in on
    # RailPlanner keeps its own extra subgoal (waited_in) and adds next_to after it.
    rail = _planner(RailPlanner, on=True)
    rail.two_phase = False
    text = rail._build_predicate_prompt(objs, extra_subgoals=RailPlanner.EXTRA_SUBGOALS)
    assert "waited_in(" in text and text.index("waited_in(") < text.index("next_to(")


def test_validation():
    p = _planner(on=True)
    assert p._validate_subgoal("next_to(mug_0, jug_0, table_0)")
    assert not p._validate_subgoal("next_to(mug_0, mug_0, table_0)")      # same object
    assert not p._validate_subgoal("next_to(mug_0, jug_0, cabinet_0)")    # not a surface
    assert not p._validate_subgoal("next_to(mug_0, table_0, table_0)")    # reference is furniture
    assert not p._validate_subgoal("next_to(mug_0, jug_0)")               # missing furniture
    assert not _planner(on=False)._validate_subgoal("next_to(mug_0, jug_0, table_0)")


# ---------------------------------------------------------------------------
# VLM-TAMP: goal literal, PDDL problem, Place call
# ---------------------------------------------------------------------------


def test_goal_literal_and_place_tool():
    p = _planner(on=True)
    p._restore_name = lambda n: n
    names = {"mug_0": "mug_0", "jug_0": "jug_0", "table_0": "table_0"}
    assert p._subgoal_to_goal_literal("next_to(mug_0, jug_0, table_0)", "agent_0", names) == (
        "next_to", "mug_0", "jug_0", "table_0")
    assert _planner(on=False)._subgoal_to_goal_literal("next_to(mug_0, jug_0, table_0)", "agent_0", names) is None
    step = ("place_on_next_to", ("agent_0", "mug_0", "jug_0", "table_0", "kitchen"))
    assert p._action_to_tool(step) == ("Place", "mug_0, on, table_0, next_to, jug_0")
    nav = ("navigate", ("agent_0", "mug_0", "kitchen"))
    assert p._action_to_tool(nav, next_action=step) == ("Navigate", "table_0")


def test_pddl_problem_enables_next_to_only_for_next_to_goals():
    wg = _graph(held="mug_0")
    goal = ("next_to", "mug_0", "jug_0", "table_0")
    scope = extract_scope_names(wg, 0, goal)
    assert {"mug_0", "jug_0", "table_0"} <= scope
    _problem, init, _n, _r = build_pddlstream_problem(wg, 0, goal, scope_names=scope)
    assert ("nextto_enabled",) in init and ("on", "jug_0", "table_0") in init
    _problem, init, _n, _r = build_pddlstream_problem(wg, 0, ("on", "mug_0", "table_0"))
    assert ("nextto_enabled",) not in init


def _solve(goal, held="mug_0"):
    from habitat_llm.pddlstream.solve import solve_pddlstream_problem

    wg = _graph(held=held)
    scope = extract_scope_names(wg, 0, goal)
    problem, _init, _n, _r = build_pddlstream_problem(wg, 0, goal, scope_names=scope)
    solution = solve_pddlstream_problem(problem, max_time=30)
    return solution[0] if solution is not None else None


def test_pddl_solver_uses_place_on_next_to():
    baseline = _solve(("on", "mug_0", "bed_0"))
    if baseline is None:
        pytest.skip("PDDLStream / Fast Downward not available here")
    assert all(step[0] != "place_on_next_to" for step in baseline)
    plan = _solve(("next_to", "mug_0", "jug_0", "table_0"))
    assert plan is not None and plan[-1][0] == "place_on_next_to"
    assert tuple(plan[-1][1][1:4]) == ("mug_0", "jug_0", "table_0")


# ---------------------------------------------------------------------------
# RailPlanner expander
# ---------------------------------------------------------------------------


def test_rail_expander():
    # Holding the mug; the jug is on the table.
    assert _plan(_graph(held="mug_0"), "next_to(mug_0, jug_0, table_0)") == [
        ("navigate", ("table_0",)), ("place_next_to", ("mug_0", "table_0", "kitchen"))]
    # Not holding: pick the mug first. Never "already satisfied" (memory has no distances).
    plan = _plan(_graph(), "next_to(mug_0, jug_0, table_0)")
    assert [a for a, _ in plan] == ["navigate", "pick", "navigate", "place_next_to"]


def test_rail_expander_needs_reference_on_furniture():
    with pytest.raises(SubgoalPlanError, match="put it there first"):
        _plan(_graph(held="mug_0"), "next_to(mug_0, jug_0, bed_0)")
    with pytest.raises(SubgoalPlanError, match="put it there first"):
        _plan(_graph(held="jug_0"), "next_to(mug_0, jug_0, table_0)")  # reference still in hand
    with pytest.raises(SubgoalPlanError):
        expand_subgoal(build_scene_state(_graph(), 0), "next_to", ["mug_0", "mug_0", "table_0"])


def test_rail_place_tool():
    p = RailPlanner.__new__(RailPlanner)
    p._restore_name = lambda n: n
    action = ("place_next_to", ("agent_0", "mug_0", "table_0", "kitchen", "jug_0"))
    assert p._action_to_tool(action) == ("Place", "mug_0, on, table_0, next_to, jug_0")
