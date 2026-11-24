#!/usr/bin/env python3
# Tests for RailPlanner's deployable end-of-plan check (end_of_plan_check=verify_symbolic):
# rail_planner.goal_verifier (ledger parsing + symbolic check) and the planner hook around it.

import json

from habitat_llm.rail_planner import RailPlanner
from habitat_llm.rail_planner.goal_verifier import (
    build_ledger_prompt,
    check_ledger,
    parse_ledger,
)
from habitat_llm.rail_planner.subgoal_expanders import build_scene_state
from habitat_llm.world_model import Furniture, Object, Receptacle, Room, SpotRobot
from habitat_llm.world_model.world_graph import WorldGraph


def _graph(held=None, cabinet_open=None):
    """kitchen: table_0, table_1, cabinet (articulated, with receptacle), lamp (on);
    bedroom: bed. mug + jug on table_0 (0.3 m apart), plate on table_1, cup inside cabinet."""
    wg = WorldGraph()
    kitchen = Room("kitchen", {"type": "room"})
    bedroom = Room("bedroom", {"type": "room"})
    table = Furniture("table_0", {"type": "table", "translation": [0.0, 0.0, 0.0]})
    table1 = Furniture("table_1", {"type": "table", "translation": [3.0, 0.0, 0.0]})
    cab_states = {} if cabinet_open is None else {"is_open": cabinet_open}
    cabinet = Furniture("cabinet_0", {"type": "cabinet", "is_articulated": True, "states": cab_states})
    cab_rec = Receptacle("rec_cabinet_0", {"type": "within"})
    lamp = Furniture("lamp_0", {"type": "lamp", "states": {"is_powered_on": True}})
    bed = Furniture("bed_0", {"type": "bed"})
    mug = Object("mug_0", {"type": "mug", "states": {}, "translation": [0.0, 0.8, 0.0]})
    jug = Object("jug_0", {"type": "jug", "states": {"is_filled": True}, "translation": [0.3, 0.8, 0.0]})
    plate = Object("plate_0", {"type": "plate", "states": {}, "translation": [3.0, 0.8, 0.0]})
    cup = Object("cup_0", {"type": "cup", "states": {}})
    agent = SpotRobot("agent_0", {"type": "agent"})
    for n in [kitchen, bedroom, table, table1, cabinet, cab_rec, lamp, bed, mug, jug, plate, cup, agent]:
        wg.add_node(n)
    for f in [table, table1, cabinet, lamp]:
        wg.add_edge(f, kitchen, "inside", "contains")
    wg.add_edge(bed, bedroom, "inside", "contains")
    wg.add_edge(cab_rec, cabinet, "joint", "joint")
    wg.add_edge(mug, table, "on", "under")
    wg.add_edge(jug, table, "on", "under")
    wg.add_edge(plate, table1, "on", "under")
    wg.add_edge(cup, cab_rec, "inside", "contains")
    wg.add_edge(agent, kitchen, "inside", "contains")
    if held is not None:
        agent.properties["last_held_object"] = {"mug_0": mug, "jug_0": jug}[held]
    return wg


def _cond(pred, *args, alternatives=()):
    return {"pred": pred, "args": list(args), "alternatives": list(alternatives)}


def _check(conditions, missing=(), unexplored=(), seen=None, **graph_kw):
    state = build_scene_state(_graph(**graph_kw), 0)
    return check_ledger(state, conditions, missing, unexplored, set() if seen is None else seen)


# ---------------------------------------------------------------------------
# Ledger prompt + parsing
# ---------------------------------------------------------------------------


def test_prompt_has_task_and_memory_but_no_plan_fields():
    prompt = build_ledger_prompt("Put the mug on the bed.", "kitchen: table_0")
    assert "Put the mug on the bed." in prompt and "kitchen: table_0" in prompt
    assert '"conditions"' in prompt and '"missing"' in prompt


def test_parse_valid_and_fenced_ledger():
    text = "```json\n" + json.dumps({
        "conditions": [
            {"pred": "on", "args": ["mug 0", "table_1"], "alternatives": ["table_0", "table_1"]},
            {"pred": "Powered-Off", "args": ["lamp_0"]},
            {"pred": "inside", "args": ["cup_0", "cabinet_0"]},
            {"pred": "on", "args": ["mug 0", "table_1"], "alternatives": ["table_0"]},
        ],
        "missing": [{"type": "bottle", "why": "needs two"}, "spoon"],
    }) + "\n```"
    conditions, missing, dropped = parse_ledger(text)
    assert conditions == [
        _cond("on", "mug_0", "table_1", alternatives=["table_0"]),
        _cond("powered_off", "lamp_0"),
        _cond("in", "cup_0", "cabinet_0"),
    ]
    assert missing == [{"type": "bottle", "why": "needs two"}, {"type": "spoon", "why": ""}]
    assert dropped == []


def test_parse_drops_bad_conditions_and_survives_garbage():
    text = json.dumps({"conditions": [
        {"pred": "picked", "args": ["mug_0"]},
        {"pred": "on", "args": ["mug_0"]},
        "on(mug_0, table_0)",
        {"pred": "filled", "args": ["jug_0"]},
    ]})
    conditions, missing, dropped = parse_ledger(text)
    assert conditions == [_cond("filled", "jug_0")] and missing == []
    assert len(dropped) == 3
    for bad in ("", None, "no json here", "[1, 2]", "{broken"):
        assert parse_ledger(bad) == ([], [], [])


# ---------------------------------------------------------------------------
# Symbolic check
# ---------------------------------------------------------------------------


def test_met_conditions_produce_no_lines():
    unmet, soft, detail = _check([
        _cond("on", "mug_0", "table_0"),
        _cond("in", "cup_0", "cabinet_0"),
        _cond("filled", "jug_0"),
        _cond("powered_on", "lamp_0"),
        _cond("closed", "cabinet_0"),
        _cond("next_to", "mug_0", "jug_0"),
    ], cabinet_open=False)
    assert unmet == [] and soft == []
    assert [d["status"] for d in detail] == ["met"] * 6


def test_unmet_conditions_use_oracle_style_wording():
    unmet, soft, _ = _check([
        _cond("on", "mug_0", "table_1"),
        _cond("in", "mug_0", "cabinet_0"),
        _cond("filled", "mug_0"),
        _cond("cleaned", "table_0"),
        _cond("powered_off", "lamp_0"),
        _cond("closed", "cabinet_0"),  # open state unknown counts as not closed
        _cond("waited_in", "mug_0", "cabinet_0"),
        _cond("next_to", "mug_0", "plate_0"),
    ])
    assert soft == []
    assert unmet == [
        "mug_0 is not placed correctly (expected on table_1).",
        "mug_0 is not placed correctly (expected inside cabinet_0).",
        "mug_0 is not filled as required.",
        "table_0 is not clean as required.",
        "lamp_0 is not powered off as required.",
        "cabinet_0 is not closed as required.",
        "mug_0 has not been left to wait inside cabinet_0 as required (waited_in).",
        "mug_0 is not placed correctly next to plate_0.",
    ]


def test_waited_in_met_after_wait():
    state = build_scene_state(_graph(), 0)
    state.waited.add(("mug_0", "cabinet_0"))
    assert check_ledger(state, [_cond("waited_in", "mug_0", "cabinet_0")], [], [], set())[:2] == ([], [])


def test_held_object_is_not_placed():
    unmet, _, _ = _check([_cond("on", "mug_0", "table_0"), _cond("next_to", "mug_0", "jug_0")], held="mug_0")
    assert unmet == [
        "mug_0 is not placed correctly (expected on table_0).",
        "mug_0 is not placed correctly next to jug_0.",
    ]


def test_next_to_without_positions_is_soft_and_issued_once():
    seen = set()
    cond = [_cond("next_to", "cup_0", "mug_0")]  # cup_0 has no position
    unmet, soft, detail = _check(cond, seen=seen)
    assert unmet == [] and soft == ["Could not confirm cup_0 is next to mug_0; check it."]
    assert detail[0]["status"] == "unverifiable"
    unmet, soft, detail = _check(cond, seen=seen)
    assert unmet == [] and soft == [] and detail[0]["status"] == "unverifiable_suppressed"


def test_unknown_entity_is_not_found():
    cond = [_cond("on", "bottle_3", "table_0"), _cond("on", "mug_0", "shelf_9")]
    unmet, soft, detail = _check(cond, unexplored=["bedroom", "attic"])
    assert soft == []
    assert unmet == [
        "Not found yet: a required bottle; unexplored rooms: attic, bedroom.",
        "Not found yet: a required shelf; unexplored rooms: attic, bedroom.",
    ]
    # Repeats while there is somewhere left to look, then once more and never again.
    seen = set()
    assert _check(cond[:1], unexplored=["bedroom"], seen=seen)[0] == _check(cond[:1], unexplored=["bedroom"], seen=seen)[0]
    assert _check(cond[:1], seen=seen)[0] == ["Not found yet: a required bottle."]
    unmet, _, detail = _check(cond[:1], seen=seen)
    assert unmet == [] and detail[0]["status"] == "not_found_suppressed"


def test_ambiguous_referent_rejects_softly_once():
    seen = set()
    cond = [_cond("on", "mug_0", "table_0", alternatives=["table_1", "table_99"])]
    unmet, soft, detail = _check(cond, seen=seen)
    assert unmet == []
    assert soft == ["Could not confirm mug_0 is on the intended furniture (table_0 or table_1); check it."]
    assert detail[0]["status"] == "ambiguous"
    unmet, soft, detail = _check(cond, seen=seen)
    assert unmet == [] and soft == [] and detail[0]["status"] == "ambiguous_suppressed"
    # On none of the candidates: a hard line every time.
    cond = [_cond("on", "mug_0", "table_1", alternatives=["bed_0"])]
    for _ in range(2):
        unmet, soft, _ = _check(cond, seen=seen)
        assert unmet == ["mug_0 is not placed correctly (expected on table_1 or bed_0)."] and soft == []
    # Alternatives that are all unknown leave a plain condition.
    assert _check([_cond("on", "mug_0", "table_0", alternatives=["table_99"])])[:2] == ([], [])


def test_missing_objects_need_an_unexplored_room():
    missing = [{"type": "bottle", "why": "task needs two"}]
    unmet, soft, detail = _check([], missing=missing, unexplored=["bedroom"])
    assert unmet == ["Not found yet: a required bottle; unexplored rooms: bedroom."] and soft == []
    unmet, soft, detail = _check([], missing=missing)
    assert unmet == [] and soft == [] and detail[0]["status"] == "not_found_suppressed"


# ---------------------------------------------------------------------------
# Planner hook
# ---------------------------------------------------------------------------


class _FakeVlm:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.sessions = 0

    def new_session(self):
        self.sessions += 1

    def ask(self, prompt, **kwargs):
        self.prompts.append(prompt)
        out = self.responses.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    def get_last_response_metadata(self):
        return {}


def _ledger(*conditions, missing=()):
    return json.dumps({"conditions": list(conditions), "missing": list(missing)})


def _planner(monkeypatch, responses, shadow=True, max_replans=3, gt=(False, 0.5)):
    import habitat_llm.rail_planner.planner as rail_mod

    planner = RailPlanner.__new__(RailPlanner)
    planner.end_of_plan_check = "verify_symbolic"
    planner.max_end_of_plan_replans = max_replans
    planner.log_gt_shadow = shadow
    planner._end_of_plan_replans = 0
    planner._end_replan_pending = False
    planner._replan_pending = False
    planner._end_feedback = []
    planner.is_done = False
    planner.events = []
    planner._log_event = planner.events.append
    planner._trace_append = lambda *a: None
    planner._vprint = lambda *a, **k: None
    planner._vprint_header = lambda *a, **k: None
    planner._append_vlm_prompt = lambda **k: None
    planner.env_interface = None
    planner._agents = []
    planner._world_graph = _graph()
    planner._instruction = "Put the mug on the other table."
    planner._waited = set()
    planner._explored_rooms = set()
    planner._ledger_cache = {}
    planner._verifier_soft_seen = set()
    planner._verifier_calls = 0
    planner._verifier_wall_s = 0.0
    planner._vlm_cycles = 0
    planner._vlm_calls = 0
    planner._vlm_wall_s = 0.0
    planner.vlm = _FakeVlm(responses)
    planner.vlm_max_tokens, planner.vlm_temperature, planner.vlm_reasoning_effort = 100, 0.0, None
    monkeypatch.setattr(rail_mod, "gt_goal_status", lambda env: gt)
    monkeypatch.setattr(rail_mod, "unmet_goal_feedback", lambda env, known: ["gt line"])
    return planner


def _check_event(planner):
    return [e for e in planner.events if e.get("event") == "end_of_plan_check"][-1]


def test_verifier_refuses_unmet_ledger_and_passes_feedback(monkeypatch):
    planner = _planner(monkeypatch, [_ledger(_cond("on", "mug_0", "table_1"), _cond("filled", "jug_0"))])
    planner._finish_planning("plan_completed", "all subgoals done")
    assert not planner.is_done and planner._end_replan_pending and planner._end_of_plan_replans == 1
    assert planner._end_feedback == ["mug_0 is not placed correctly (expected on table_1)."]
    event = _check_event(planner)
    assert event["check"] == "verify_symbolic" and event["replan"] and event["goal_satisfied"] is False
    assert event["feedback"] == planner._end_feedback and event["ledger_cached"] is False
    assert [c["status"] for c in event["conditions"]] == ["unmet", "met"]
    # Ground truth is logged next to the verdict, and never becomes feedback.
    assert event["gt_goal_satisfied_shadow"] is False and event["gt_percent_shadow"] == 0.5
    assert event["gt_feedback_shadow"] == ["gt line"] and "gt line" not in planner._end_feedback
    # The ledger writer sees the task and the memory, in a fresh session, outside the VLM cycle budget.
    prompt = planner.vlm.prompts[0]
    assert planner._instruction in prompt and "mug_0" in prompt
    assert planner.vlm.sessions == 1 and planner._vlm_cycles == 0
    assert planner._verifier_calls == 1 and planner._vlm_calls == 1
    assert any(e.get("event") == "goal_ledger" for e in planner.events)


def test_verifier_accepts_met_ledger_even_if_ground_truth_disagrees(monkeypatch):
    planner = _planner(monkeypatch, [_ledger(_cond("on", "mug_0", "table_0"))], gt=(False, 0.0))
    planner._finish_planning("plan_completed")
    assert planner.is_done and not planner._end_replan_pending and planner._end_feedback == []
    event = _check_event(planner)
    assert event["goal_satisfied"] is True and event["gt_goal_satisfied_shadow"] is False


def test_verifier_soft_feedback_rejects_once(monkeypatch):
    ledger = _ledger(_cond("on", "mug_0", "table_0", alternatives=["table_1"]))
    planner = _planner(monkeypatch, [ledger])
    planner._finish_planning("plan_completed")
    assert planner._end_replan_pending and planner._end_feedback == [
        "Could not confirm mug_0 is on the intended furniture (table_0 or table_1); check it."
    ]
    planner._end_replan_pending = False
    planner._finish_planning("plan_completed")  # same objects: cached ledger, no second VLM call
    assert planner.is_done and len(planner.vlm.prompts) == 1
    assert _check_event(planner)["ledger_cached"] is True


def test_unusable_ledger_or_vlm_error_accepts_the_plan(monkeypatch):
    for response in ("I cannot answer", _ledger(), RuntimeError("api down")):
        planner = _planner(monkeypatch, [response])
        planner._finish_planning("plan_completed")
        assert planner.is_done and not planner._end_replan_pending
        event = _check_event(planner)
        assert event["goal_satisfied"] is None and event["replan"] is False and event["verifier_error"]
        assert planner._ledger_cache == {}


def test_verifier_respects_cap_reason_and_shadow_flag(monkeypatch):
    unmet = _ledger(_cond("on", "mug_0", "table_1"))
    capped = _planner(monkeypatch, [unmet], max_replans=1)
    capped._end_of_plan_replans = 1
    capped._finish_planning("plan_completed")
    assert capped.is_done and capped.vlm.prompts == []
    other = _planner(monkeypatch, [unmet])
    other._finish_planning("unsolvable")
    assert other.is_done and other.vlm.prompts == []
    quiet = _planner(monkeypatch, [unmet], shadow=False)
    quiet._finish_planning("plan_completed")
    assert not any("shadow" in key for key in _check_event(quiet))


def test_new_object_invalidates_cached_ledger(monkeypatch):
    unmet = _ledger(_cond("on", "mug_0", "table_1"))
    planner = _planner(monkeypatch, [unmet, unmet])
    planner._finish_planning("plan_completed")
    wg = planner._world_graph
    spoon = Object("spoon_0", {"type": "spoon", "states": {}})
    wg.add_node(spoon)
    wg.add_edge(spoon, wg.get_node_from_name("table_0"), "on", "under")
    planner._end_replan_pending = False
    planner._finish_planning("plan_completed")
    assert len(planner.vlm.prompts) == 2 and "spoon_0" in planner.vlm.prompts[1]


def test_explored_rooms_feed_the_not_found_line(monkeypatch):
    ledger = _ledger(missing=[{"type": "bottle", "why": "task needs one"}])
    planner = _planner(monkeypatch, [ledger])
    planner._note_explored(("Explore", "kitchen", None), "Successful execution!")
    planner._note_explored(("Explore", "bedroom", None), "Unexpected failure! - could not reach")
    planner._note_explored(("Navigate", "bedroom", None), "Successful execution!")
    planner._note_explored(("Explore", "bedroom", None), None)
    assert planner._explored_rooms == {"kitchen"}
    planner._finish_planning("plan_completed")
    assert planner._end_feedback == ["Not found yet: a required bottle; unexplored rooms: bedroom."]
    assert _check_event(planner)["unexplored_rooms"] == ["bedroom"]
    # Everything explored: nothing left to do about a missing object, so the plan is accepted.
    planner._explored_rooms.add("bedroom")
    planner._end_replan_pending = False
    planner._finish_planning("plan_completed")
    assert planner.is_done


def test_feedback_replan_records_history_for_the_mode_selector(monkeypatch):
    planner = _planner(monkeypatch, [])
    planner._end_feedback = ["Not found yet: a required bottle; unexplored rooms: bedroom."]
    planner.history = []
    planner._reprompt_round = 0
    planner.branch_idx = planner.subgoal_idx = 0
    planner._already_succeeded_subgoals = ["on(mug_0, table_0)"]
    planner._observe_failure_state = lambda obs, reason="failure": obs
    calls = []
    planner._generate_subgoals = lambda *a, **k: calls.append(k)
    planner._replan_with_feedback("task", {}, planner._world_graph)
    assert planner.history == ["End-of-plan check: Not found yet: a required bottle; unexplored rooms: bedroom."]
    assert "- Not found yet: a required bottle" in calls[0]["history_str"]
    assert calls[0]["observation_reason"] == "plan_finished"


def test_verifier_cost_is_reported_separately(monkeypatch):
    from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner

    planner = _planner(monkeypatch, [])
    planner._verifier_calls, planner._verifier_wall_s = 2, 1.5
    planner.two_phase = False
    monkeypatch.setattr(VlmTampPddlPlanner, "_planner_info", lambda self, uid, **kw: {"cost_metrics": {"llm_requests": 7}})
    cost = planner._planner_info(0)["cost_metrics"]
    assert cost == {"llm_requests": 7, "verifier_llm_requests": 2, "verifier_llm_time_s": 1.5}
    planner.end_of_plan_check = "oracle_feedback"
    assert planner._planner_info(0)["cost_metrics"] == {"llm_requests": 7}
