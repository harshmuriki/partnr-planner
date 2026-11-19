#!/usr/bin/env python3
# Tests for rail_planner.subgoal_expanders: WorldGraph -> SceneState extraction and the
# skill sequence each VLM-TAMP subgoal type expands to.

import pytest

from habitat_llm.rail_planner.subgoal_expanders import (
    SubgoalPlanError,
    build_scene_state,
    expand_subgoal,
)
from habitat_llm.rail_planner import RailPlanner
from habitat_llm.world_model import Furniture, Object, Receptacle, Room, SpotRobot
from habitat_llm.world_model.world_graph import WorldGraph


def _graph(held=None, cabinet_open=None, jug_states=None):
    """kitchen: table, cabinet (articulated, with receptacle), sink (faucet), lamp;
    bedroom: bed. mug on table, cup inside cabinet, jug on table."""
    wg = WorldGraph()
    kitchen = Room("kitchen", {"type": "room"})
    bedroom = Room("bedroom", {"type": "room"})
    table = Furniture("table_0", {"type": "table", "is_articulated": False})
    cab_states = {} if cabinet_open is None else {"is_open": cabinet_open}
    cabinet = Furniture(
        "cabinet_0", {"type": "cabinet", "is_articulated": True, "states": cab_states}
    )
    cab_rec = Receptacle("rec_cabinet_0", {"type": "within"})
    sink = Furniture("sink_0", {"type": "sink", "components": ["faucet"]})
    lamp = Furniture("lamp_0", {"type": "lamp", "states": {"is_powered_on": True}})
    bed = Furniture("bed_0", {"type": "bed"})
    mug = Object("mug_0", {"type": "mug", "states": {}})
    cup = Object("cup_0", {"type": "cup", "states": {}})
    jug = Object("jug_0", {"type": "jug", "states": jug_states or {}})
    agent = SpotRobot("agent_0", {"type": "agent"})
    for n in [kitchen, bedroom, table, cabinet, cab_rec, sink, lamp, bed, mug, cup, jug, agent]:
        wg.add_node(n)
    for f in [table, cabinet, sink, lamp]:
        wg.add_edge(f, kitchen, "inside", "contains")
    wg.add_edge(bed, bedroom, "inside", "contains")
    wg.add_edge(cab_rec, cabinet, "joint", "joint")
    wg.add_edge(mug, table, "on", "under")
    wg.add_edge(jug, table, "on", "under")
    wg.add_edge(cup, cab_rec, "inside", "contains")
    wg.add_edge(agent, kitchen, "inside", "contains")
    if held is not None:
        agent.properties["last_held_object"] = {"mug_0": mug, "jug_0": jug, "cup_0": cup}[held]
    return wg


def _plan(wg, subgoal):
    planner = RailPlanner.__new__(RailPlanner)
    pred, args = planner._parse_subgoal(subgoal)
    plan = expand_subgoal(build_scene_state(wg, 0), pred, args)
    return [(a.name, a.args[1:-1]) for a in plan]


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------


def test_state_extraction():
    s = build_scene_state(_graph(cabinet_open=False), 0)
    assert s.agent == "agent_0" and s.room("agent_0") == "kitchen"
    assert s.holding is None
    assert s.placement["mug_0"] == ("table_0", "on")
    assert s.placement["cup_0"] == ("cabinet_0", "inside")
    assert s.containers == {"cabinet_0"}
    assert s.faucets == {"sink_0"}
    assert s.known_closed == {"cabinet_0"} and not s.open
    assert "lamp_0" in s.powered_on


def test_held_object_has_no_placement():
    s = build_scene_state(_graph(held="mug_0"), 0)
    assert s.holding == "mug_0"
    assert "mug_0" not in s.placement


def test_names_with_spaces_resolve():
    wg = _graph()
    wg.add_node(Furniture("kitchen table", {"type": "table"}))
    assert _plan(wg, "on(mug_0, kitchen table)")[-1] == ("place_on", ("mug_0", "kitchen table"))


# ---------------------------------------------------------------------------
# Expanders
# ---------------------------------------------------------------------------


def test_picked():
    assert _plan(_graph(), "picked(mug_0)") == [("navigate", ("mug_0",)), ("pick", ("mug_0",))]
    assert _plan(_graph(held="mug_0"), "picked(mug_0)") == []


def test_picked_while_holding_other_puts_it_down_first():
    assert _plan(_graph(held="jug_0"), "picked(mug_0)") == [
        ("navigate", ("table_0",)),
        ("place_on", ("jug_0", "table_0")),
        ("navigate", ("mug_0",)),
        ("pick", ("mug_0",)),
    ]


def test_pick_from_closed_container_opens_it():
    assert _plan(_graph(cabinet_open=False), "picked(cup_0)") == [
        ("navigate", ("cabinet_0",)),
        ("open", ("cabinet_0",)),
        ("navigate", ("cup_0",)),
        ("pick", ("cup_0",)),
    ]
    assert _plan(_graph(cabinet_open=True), "picked(cup_0)")[0] == ("navigate", ("cup_0",))


def test_on():
    assert _plan(_graph(), "on(mug_0, bed_0)") == [
        ("navigate", ("mug_0",)),
        ("pick", ("mug_0",)),
        ("navigate", ("bed_0",)),
        ("place_on", ("mug_0", "bed_0")),
    ]
    assert _plan(_graph(held="mug_0"), "on(mug_0, bed_0)") == [
        ("navigate", ("bed_0",)),
        ("place_on", ("mug_0", "bed_0")),
    ]
    assert _plan(_graph(), "on(mug_0, table_0)") == []


def test_on_faucet_furniture_is_allowed():
    assert _plan(_graph(), "on(mug_0, sink_0)")[-1] == ("place_on", ("mug_0", "sink_0"))


def test_in_opens_closed_container_first():
    assert _plan(_graph(cabinet_open=False), "in(mug_0, cabinet_0)") == [
        ("navigate", ("cabinet_0",)),
        ("open", ("cabinet_0",)),
        ("navigate", ("mug_0",)),
        ("pick", ("mug_0",)),
        ("navigate", ("cabinet_0",)),
        ("place_in", ("mug_0", "cabinet_0")),
    ]


def test_in_unknown_open_state_counts_as_closed():
    assert ("open", ("cabinet_0",)) in _plan(_graph(cabinet_open=None), "in(mug_0, cabinet_0)")


def test_in_open_container_skips_open():
    plan = _plan(_graph(cabinet_open=True), "in(mug_0, cabinet_0)")
    assert ("open", ("cabinet_0",)) not in plan and len(plan) == 4


def test_in_already_satisfied():
    assert _plan(_graph(), "in(cup_0, cabinet_0)") == []


def test_in_non_container_fails():
    with pytest.raises(SubgoalPlanError, match="no receptacle"):
        _plan(_graph(), "in(mug_0, table_0)")


def test_open_and_close_use_real_state():
    assert _plan(_graph(cabinet_open=False), "opened-door(cabinet_0)") == [
        ("navigate", ("cabinet_0",)),
        ("open", ("cabinet_0",)),
    ]
    assert _plan(_graph(cabinet_open=True), "opened-drawer(cabinet_0)") == []
    assert _plan(_graph(cabinet_open=True), "closed-door(cabinet_0)") == [
        ("navigate", ("cabinet_0",)),
        ("close", ("cabinet_0",)),
    ]
    assert _plan(_graph(cabinet_open=False), "closed-drawer(cabinet_0)") == []
    with pytest.raises(SubgoalPlanError, match="not articulated"):
        _plan(_graph(), "opened-door(table_0)")


def test_power():
    assert _plan(_graph(), "powered_off(lamp_0)") == [
        ("navigate", ("lamp_0",)),
        ("power_off", ("lamp_0",)),
    ]
    assert _plan(_graph(), "powered_on(lamp_0)") == []
    assert _plan(_graph(), "powered_on(mug_0)")[-1] == ("power_on_object", ("mug_0",))


def test_filled():
    assert _plan(_graph(), "filled(jug_0)") == [
        ("navigate", ("jug_0",)),
        ("pick", ("jug_0",)),
        ("navigate", ("sink_0",)),
        ("fill_held", ("jug_0", "sink_0")),
    ]
    assert _plan(_graph(held="jug_0"), "filled(jug_0)") == [
        ("navigate", ("sink_0",)),
        ("fill_held", ("jug_0", "sink_0")),
    ]
    assert _plan(_graph(jug_states={"is_filled": True}), "filled(jug_0)") == []


def test_poured_into():
    assert _plan(_graph(held="jug_0", jug_states={"is_filled": True}), "poured-into(mug_0)") == [
        ("navigate", ("mug_0",)),
        ("pour", ("jug_0", "mug_0")),
    ]
    assert _plan(_graph(held="jug_0"), "poured-into(mug_0)") == [
        ("navigate", ("sink_0",)),
        ("fill_held", ("jug_0", "sink_0")),
        ("navigate", ("mug_0",)),
        ("pour", ("jug_0", "mug_0")),
    ]
    with pytest.raises(SubgoalPlanError, match="held source"):
        _plan(_graph(), "poured-into(mug_0)")


def test_cleaned():
    assert _plan(_graph(), "cleaned(mug_0)") == [
        ("navigate", ("mug_0",)),
        ("pick", ("mug_0",)),
        ("navigate", ("sink_0",)),
        ("clean_object", ("mug_0", "sink_0")),
    ]
    assert _plan(_graph(), "cleaned(table_0)") == [
        ("navigate", ("table_0",)),
        ("clean_furniture", ("table_0",)),
    ]


def test_unknown_entities_and_predicates_fail():
    with pytest.raises(SubgoalPlanError, match="not a known movable"):
        _plan(_graph(), "picked(ghost_0)")
    with pytest.raises(SubgoalPlanError, match="unsupported"):
        _plan(_graph(), "teleported(mug_0)")


# ---------------------------------------------------------------------------
# Planner integration: steps map to the expected Habitat tools
# ---------------------------------------------------------------------------


def test_steps_map_to_tools():
    planner = RailPlanner.__new__(RailPlanner)
    planner._reverse_name_map = {}
    pred, args = planner._parse_subgoal("in(mug_0, cabinet_0)")
    plan = expand_subgoal(build_scene_state(_graph(cabinet_open=False), 0), pred, args)
    tools = [planner._action_to_tool(a) for a in plan]
    assert tools == [
        ("Navigate", "cabinet_0"),
        ("Open", "cabinet_0"),
        ("Navigate", "mug_0"),
        ("Pick", "mug_0"),
        ("Navigate", "cabinet_0"),
        ("Place", "mug_0, within, cabinet_0, None, None"),
    ]
    assert repr(plan[0]).startswith("Action(name='navigate', args=")


def test_fill_uses_closest_faucet_for_held_object():
    wg = _graph(held="jug_0")
    laundry = Room("laundry", {"type": "room"})
    far_sink = Furniture("sink_1", {"type": "sink", "components": ["faucet"], "translation": [9, 0, 9]})
    wg.add_node(laundry)
    wg.add_node(far_sink)
    wg.add_edge(far_sink, laundry, "inside", "contains")
    wg.get_node_from_name("sink_0").properties["translation"] = [1, 0, 1]
    wg.get_node_from_name("agent_0").properties["translation"] = [0, 0, 0]
    assert _plan(wg, "filled(jug_0)") == [
        ("navigate", ("sink_0",)),
        ("fill_held", ("jug_0", "sink_0")),
    ]


# ---------------------------------------------------------------------------
# Sync with the simulator (stale world-graph state)
# ---------------------------------------------------------------------------


class _Grasp:
    def __init__(self, is_grasped):
        self.is_grasped = is_grasped
        self.snap_idx = 0


def _planner_with_gripper(is_grasped):
    from types import SimpleNamespace

    planner = RailPlanner.__new__(RailPlanner)
    planner._vprint = lambda *a, **k: None
    sim = SimpleNamespace(agents_mgr={0: SimpleNamespace(grasp_mgr=_Grasp(is_grasped))})
    planner.env_interface = SimpleNamespace(sim=sim)
    return planner


def test_empty_gripper_clears_stale_held_object():
    wg = _graph(held="mug_0")
    state = build_scene_state(wg, 0)
    _planner_with_gripper(False)._sync_holding_with_gripper(state, wg, 0)
    assert state.holding is None
    assert "last_held_object" not in wg.get_node_from_name("agent_0").properties
    assert expand_subgoal(state, "on", ["cup_0", "table_0"])[-1].name == "place_on"


def test_grasping_gripper_keeps_held_object():
    wg = _graph(held="mug_0")
    state = build_scene_state(wg, 0)
    _planner_with_gripper(True)._sync_holding_with_gripper(state, wg, 0)
    assert state.holding == "mug_0"


def test_gripper_object_overrides_wrong_held_object():
    from types import SimpleNamespace

    wg = _graph(held="mug_0")
    state = build_scene_state(wg, 0)
    planner = _planner_with_gripper(True)
    planner.env_interface.sim.get_rigid_object_manager = lambda: SimpleNamespace(
        get_object_handle_by_id=lambda idx: "cup_handle"
    )
    planner.env_interface.full_world_graph = SimpleNamespace(
        get_node_from_sim_handle=lambda h: SimpleNamespace(name="cup_0")
    )
    planner._sync_holding_with_gripper(state, wg, 0)
    assert state.holding == "cup_0" and "cup_0" not in state.placement
    assert wg.get_node_from_name("agent_0").properties["last_held_object"].name == "cup_0"
    # Placing the cup no longer emits a Place for the mug first.
    assert [a.name for a in expand_subgoal(state, "on", ["cup_0", "bed_0"])] == ["navigate", "place_on"]


def test_open_state_read_from_sim(monkeypatch):
    import habitat_llm.rail_planner.planner as rail_mod

    planner = RailPlanner.__new__(RailPlanner)
    planner.env_interface = None
    # World graph says closed, sim says open: close is needed, open is not.
    wg = _graph(cabinet_open=False)
    state = build_scene_state(wg, 0)
    monkeypatch.setattr(rail_mod, "sim_is_open", lambda node, env: True)
    planner._sync_open_with_sim(state, wg)
    assert expand_subgoal(state, "opened", ["cabinet_0"]) == []
    assert [a.name for a in expand_subgoal(state, "closed", ["cabinet_0"])][-1] == "close"


# ---------------------------------------------------------------------------
# Replan triggers
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from habitat_llm.rail_planner.replan_trigger import (  # noqa: E402
    Snapshot,
    action_entities,
    compute_delta,
    goal_entities,
    should_replan,
    take_snapshot,
)


def _snap(placement, states=None, holding=None, extra_objects=()):
    return Snapshot(
        objects=set(placement) | set(extra_objects) | ({holding} if holding else set()),
        placement=dict(placement),
        holding=holding,
        states={k: frozenset(v) for k, v in (states or {}).items()},
    )


def test_snapshot_from_graph():
    snap = take_snapshot(_graph(cabinet_open=True), 0)
    assert snap.placement["mug_0"] == ("table_0", "on")
    assert snap.placement["cup_0"] == ("cabinet_0", "inside")
    assert snap.states["cabinet_0"] == frozenset({"open"})
    assert snap.states["lamp_0"] == frozenset({"powered_on"})


def test_delta_kinds():
    prev = _snap({"mug_0": ("table_0", "on"), "cup_0": ("table_0", "on")})
    cur = _snap(
        {"mug_0": ("counter_0", "on"), "jug_0": ("sink_0", "on")},
        states={"lamp_0": {"powered_on"}},
        extra_objects=["cup_0"],
    )
    kinds = {(c.kind, c.entity) for c in compute_delta(prev, cur)}
    assert kinds == {
        ("new_object", "jug_0"),
        ("moved", "mug_0"),
        ("vanished", "cup_0"),
        ("state", "lamp_0"),
    }


def test_delta_ignores_own_action_and_held_object():
    prev = _snap({"mug_0": ("table_0", "on")}, states={"cabinet_0": {"known_closed"}})
    cur = _snap({}, states={"cabinet_0": {"open"}}, holding="mug_0")
    assert compute_delta(prev, cur, action_entities(("Open", "cabinet_0", None))) == []
    assert compute_delta(None, cur) == []


def test_delta_ignores_object_put_down_since_last_check():
    # Held at the last check, now on the counter: the robot's own Place.
    prev = _snap({}, holding="apple_0")
    cur = _snap({"apple_0": ("counter_0", "on")})
    assert compute_delta(prev, cur) == []


def test_snapshot_trusts_gripper_over_stale_held_object():
    wg = _graph(held="mug_0")
    stale = take_snapshot(wg, 0)
    assert stale.holding == "mug_0" and "mug_0" not in stale.placement
    agent = wg.get_node_from_name("agent_0")

    def gripper_empty(state):
        state.holding = None
        agent.properties.pop("last_held_object", None)

    synced = take_snapshot(wg, 0, gripper_empty)
    assert synced.holding is None
    assert synced.placement["mug_0"] == ("table_0", "on")


def test_action_entities():
    assert action_entities(("Place", "mug_0, on, table_0, None, None", None)) >= {"mug_0", "table_0"}
    assert action_entities(None) == set()


def test_simple_triggers():
    delta = compute_delta(_snap({}), _snap({"jug_0": ("sink_0", "on")}))
    assert should_replan("failure", delta, set(), set(), set()) == (False, None)
    assert should_replan("every_step", [], set(), set(), set())[0]
    assert should_replan("any_change", [], set(), set(), set()) == (False, None)
    assert should_replan("any_change", delta, set(), set(), set())[0]


def test_oracle_rules_and_dedupe():
    prev = _snap({"mug_0": ("table_0", "on")})
    goal, plan = {"jug_0", "table_0"}, {"mug_0", "table_0"}
    seen = set()
    # R1: the true goal object is seen and the plan does not use it.
    found_jug = _snap({"mug_0": ("table_0", "on"), "jug_0": ("sink_0", "on")})
    assert should_replan("oracle", compute_delta(prev, found_jug), goal, plan, seen) == (
        True,
        "R1_unused_goal_entity",
    )
    assert should_replan("oracle", compute_delta(prev, found_jug), goal, plan, seen) == (False, None)
    # R2: a plan object is not where the plan assumed.
    moved = _snap({"mug_0": ("bed_0", "on")})
    assert should_replan("oracle", compute_delta(prev, moved), goal, plan, seen) == (
        True,
        "R2_plan_assumption_changed",
    )
    # A distractor is ignored.
    distractor = _snap({"mug_0": ("table_0", "on"), "vase_0": ("bed_0", "on")})
    assert should_replan("oracle", compute_delta(prev, distractor), goal, plan, set()) == (False, None)


def test_goal_entities_from_propositions():
    names = {"h_jug": "jug_0", "h_table": "table_0"}

    def lookup(handle):
        if handle not in names:
            raise KeyError(handle)
        return SimpleNamespace(name=names[handle])

    episode = SimpleNamespace(
        evaluation_propositions=[
            SimpleNamespace(args={"object_handles": ["h_jug"], "receptacle_handles": ["h_table", "h_missing"]}),
            {"args": {"object_handles": ["h_jug"]}},
        ]
    )
    env = SimpleNamespace(
        env=SimpleNamespace(env=SimpleNamespace(env=SimpleNamespace(_env=SimpleNamespace(current_episode=episode)))),
        full_world_graph=SimpleNamespace(get_node_from_sim_handle=lookup),
    )
    assert goal_entities(env) == {"jug_0", "table_0"}
    assert goal_entities(SimpleNamespace()) == set()


def _trigger_planner(trigger, monkeypatch, snapshots):
    import habitat_llm.rail_planner.planner as rail_mod

    planner = RailPlanner.__new__(RailPlanner)
    planner.replan_trigger = trigger
    planner.max_proactive_replans = 1
    planner._trigger_prev_snapshot = snapshots.pop(0)
    planner._trigger_seen = set()
    planner._goal_entities = None
    planner._proactive_replans = 0
    planner._replan_pending = False
    planner.last_high_level_actions = {}
    planner.branches, planner.branch_idx, planner.subgoal_idx = [["on(mug_0, table_0)"]], 0, 0
    planner.events = []
    planner._log_event = planner.events.append
    planner._trace_append = lambda *a: None
    planner.env_interface = None
    monkeypatch.setattr(rail_mod, "take_snapshot", lambda wg, uid, sync=None: snapshots.pop(0))
    return planner


def test_hook_sets_pending_replan_after_completed_skill(monkeypatch):
    prev = _snap({"mug_0": ("table_0", "on")})
    snaps = [prev, _snap({"mug_0": ("table_0", "on"), "vase_0": ("bed_0", "on")}), prev]
    planner = _trigger_planner("any_change", monkeypatch, snaps)
    info = {"responses": {0: "Successful execution!"}, "high_level_actions": {}}
    planner._check_replan_trigger(info, ("Navigate", "table_0", None), None, 0)
    assert planner._replan_pending and planner.events[-1]["fired"]
    # Capped at max_proactive_replans.
    planner._replan_pending = False
    planner._trigger_prev_snapshot = _snap({})
    planner._check_replan_trigger(info, ("Navigate", "table_0", None), None, 0)
    assert not planner._replan_pending and planner.events[-1]["capped"]


def test_hook_skips_running_skill_and_explore(monkeypatch):
    prev = _snap({})
    planner = _trigger_planner("every_step", monkeypatch, [prev, prev])
    planner._check_replan_trigger({"responses": {0: ""}}, ("Navigate", "table_0", None), None, 0)
    planner._check_replan_trigger(
        {"responses": {0: "Successful execution!"}}, ("Explore", "kitchen", None), None, 0
    )
    assert not planner._replan_pending and planner.events == []


# ---------------------------------------------------------------------------
# Placement oracle
# ---------------------------------------------------------------------------

from habitat_llm.rail_planner.placement_oracle import (  # noqa: E402
    format_hint,
    place_constraint,
    placement_goals,
)


def _oracle_env(props, constraints=None):
    names = {
        "h_a": "box_0", "h_b": "box_1", "h_s": "scissors_0",
        "h_t0": "table_0", "h_t1": "table_1", "h_bk": "basket_0",
        "h_toy0": "toy_0", "h_toy1": "toy_1",
    }

    def lookup(handle):
        return SimpleNamespace(name=names[handle])

    episode = SimpleNamespace(evaluation_propositions=props, evaluation_constraints=constraints or [])
    return SimpleNamespace(
        env=SimpleNamespace(env=SimpleNamespace(env=SimpleNamespace(_env=SimpleNamespace(current_episode=episode)))),
        full_world_graph=SimpleNamespace(get_node_from_sim_handle=lookup),
    )


_T2_PROPS = [
    {"function_name": "is_on_top", "args": {"object_handles": ["h_a"], "receptacle_handles": ["h_t0"]}},
    {"function_name": "is_on_top", "args": {"object_handles": ["h_b"], "receptacle_handles": ["h_t0", "h_t1", "h_missing"]}},
    {"function_name": "is_next_to", "args": {"entity_handles_a": ["h_b"], "entity_handles_b": ["h_a"]}},
    {"function_name": "is_next_to", "args": {"entity_handles_a": ["h_s"], "entity_handles_b": ["h_a"]}},
    {"function_name": "is_next_to", "args": {"entity_handles_a": ["h_s"], "entity_handles_b": ["h_b"]}},
    {"function_name": "is_inside", "args": {"object_handles": ["h_toy0", "h_toy1"], "receptacle_handles": ["h_bk"]}},
    {"function_name": "is_powered_off", "args": {"object_handles": ["h_a"]}},
]


def test_placement_goals_from_propositions():
    goals = placement_goals(_oracle_env(_T2_PROPS))
    assert (goals["box_0"].relation, goals["box_0"].receptacles) == ("on", ["table_0"])
    assert goals["box_1"].receptacles == ["table_0", "table_1"]
    assert goals["box_0"].next_to == {"box_1", "scissors_0"}
    assert goals["scissors_0"].relation is None and goals["scissors_0"].next_to == {"box_0", "box_1"}
    assert [(goals[t].relation, goals[t].receptacles) for t in ("toy_0", "toy_1")] == [
        ("within", ["basket_0"])
    ] * 2
    assert placement_goals(SimpleNamespace()) == {}


def test_hint_only_covers_known_objects():
    goals = placement_goals(_oracle_env(_T2_PROPS))
    hint = format_hint(goals, {"box_1", "toy_0"})
    assert "- box_1 must end up on table_0 or table_1, next to box_0 and scissors_0" in hint
    assert "- toy_0 must end up inside basket_0" in hint
    assert "box_0 must" not in hint and "scissors_0 must" not in hint
    assert format_hint(goals, set()) == ""


def test_place_constraint_picks_partner_on_destination():
    goals = placement_goals(_oracle_env(_T2_PROPS))
    assert place_constraint(goals["box_0"], ["mug_0"]) is None
    assert place_constraint(goals["box_1"], ["box_0", "mug_0"]) == "box_0"
    assert place_constraint(goals["scissors_0"], ["box_0", "box_1"], ["box_1", "box_0"]) == "box_0"
    assert place_constraint(None, ["box_0"]) is None


def _place_planner(mode, wg):
    planner = RailPlanner.__new__(RailPlanner)
    planner.placement_oracle_mode = mode
    planner._placement_goals = None
    planner._placed_order = []
    planner._reverse_name_map = {}
    planner._agents = []
    planner.events = []
    planner._log_event = planner.events.append
    env = _oracle_env([
        {"function_name": "is_next_to", "args": {"entity_handles_a": ["h_b"], "entity_handles_b": ["h_a"]}},
    ])
    env.world_graph = {0: wg}
    planner.env_interface = env
    return planner


def _boxes_graph():
    wg = WorldGraph()
    table = Furniture("table_0", {"type": "table"})
    agent = SpotRobot("agent_0", {"type": "agent"})
    box0 = Object("box_0", {"type": "box", "states": {}})
    box1 = Object("box_1", {"type": "box", "states": {}})
    for n in [table, agent, box0, box1]:
        wg.add_node(n)
    wg.add_edge(box0, table, "on", "under")
    agent.properties["last_held_object"] = box1
    return wg


def test_place_tool_adds_next_to_only_when_enabled():
    from habitat_llm.rail_planner.subgoal_expanders import Action

    step = Action("place_on", ("agent_0", "box_1", "table_0", "kitchen"))
    on = _place_planner("place", _boxes_graph())
    assert on._action_to_tool(step) == ("Place", "box_1, on, table_0, next_to, box_0")
    assert on.events[-1]["next_to"] == "box_0"
    for mode in ("none", "hint"):
        off = _place_planner(mode, _boxes_graph())
        assert off._action_to_tool(step) == ("Place", "box_1, on, table_0, None, None")
    inside = Action("place_in", ("agent_0", "box_1", "table_0", "kitchen"))
    assert _place_planner("full", _boxes_graph())._action_to_tool(inside)[1].startswith("box_1, within, table_0")


# ---------------------------------------------------------------------------
# End-of-plan check
# ---------------------------------------------------------------------------


def _end_check_planner(check, monkeypatch, satisfied=False, max_replans=3):
    import habitat_llm.rail_planner.planner as rail_mod

    planner = RailPlanner.__new__(RailPlanner)
    planner.end_of_plan_check = check
    planner.max_end_of_plan_replans = max_replans
    planner._end_of_plan_replans = 0
    planner._end_replan_pending = False
    planner._replan_pending = False
    planner.is_done = False
    planner.events = []
    planner._log_event = planner.events.append
    planner._trace_append = lambda *a: None
    planner.env_interface = None
    monkeypatch.setattr(rail_mod, "gt_goal_status", lambda env: (satisfied, 0.5))
    return planner


def test_end_of_plan_oracle_refuses_unmet_goal(monkeypatch):
    planner = _end_check_planner("oracle", monkeypatch, satisfied=False)
    planner._finish_planning("plan_completed", "all subgoals done")
    assert not planner.is_done and planner._end_replan_pending
    event = planner.events[-1]
    assert event["event"] == "end_of_plan_check" and event["replan"] and event["goal_satisfied"] is False


def test_end_of_plan_oracle_accepts_met_goal_and_respects_cap(monkeypatch):
    met = _end_check_planner("oracle", monkeypatch, satisfied=True)
    met._finish_planning("plan_completed")
    assert met.is_done and not met._end_replan_pending
    capped = _end_check_planner("oracle", monkeypatch, satisfied=False, max_replans=1)
    capped._end_of_plan_replans = 1
    capped._finish_planning("plan_completed")
    assert capped.is_done


def test_end_of_plan_check_only_intercepts_plan_completed(monkeypatch):
    for check, reason in (("none", "plan_completed"), ("oracle", "unsolvable"), ("oracle", "budget_exhausted")):
        planner = _end_check_planner(check, monkeypatch, satisfied=False)
        planner._finish_planning(reason)
        assert planner.is_done and not planner._end_replan_pending
        assert not any(e.get("event") == "end_of_plan_check" for e in planner.events)


def test_refused_plan_completed_keeps_episode_running_then_replans(monkeypatch):
    from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner

    planner = _end_check_planner("oracle", monkeypatch, satisfied=False)
    planner._agents = []
    planner.last_high_level_actions = {}
    planner.replan_trigger = "failure"
    planner._planner_info = lambda uid, **kw: dict(kw)
    planner._vprint_header = lambda *a, **k: None
    planner._trigger_snapshot = lambda wg, uid: None
    replans = []
    planner._observe_and_continue = lambda instr, obs, wg, reason: replans.append(reason)

    def base_step(self, instruction, observations, world_graphs):
        self._finish_planning("plan_completed", "all subgoals done")
        return {}, {"is_done": {0: True}}, True

    monkeypatch.setattr(VlmTampPddlPlanner, "get_next_action", base_step)
    _, info, done = planner.get_next_action("task", {}, {0: None})
    assert done is False and "is_done" not in info
    _, info, done = planner.get_next_action("task", {}, {0: None})
    assert replans == ["plan_finished"] and info["replanned"] == {0: True} and done is False


def _feedback_env(props, satisfied, names, types=None):
    from habitat_llm.agent.env.evaluation.evaluation_functions import EvaluationProposition
    from habitat_llm.agent.env.evaluation.predicate_wrappers import PropositionResult

    types = types or {}
    tracker = SimpleNamespace(get_metric=lambda: {
        "propositions": [EvaluationProposition(fn, args) for fn, args in props],
        "state_sequence": [[PropositionResult(s) for s in satisfied]],
    })
    measures = {"auto_eval_proposition_tracker": tracker}
    task = SimpleNamespace(measurements=SimpleNamespace(measures=measures))
    graph = SimpleNamespace(
        get_node_from_sim_handle=lambda h: SimpleNamespace(name=names[h]),
        get_node_from_name=lambda n: SimpleNamespace(properties={"type": types.get(n)}),
    )
    env = SimpleNamespace(env=SimpleNamespace(env=SimpleNamespace(env=SimpleNamespace(_env=SimpleNamespace(task=task)))))
    env.full_world_graph = graph
    return env


def test_unmet_goal_feedback_names_known_objects_only():
    from habitat_llm.rail_planner.replan_trigger import unmet_goal_feedback

    names = {"hb0": "box_0", "hb1": "box_1", "hs": "scissors_0", "ht": "table_16", "hj": "jug_0"}
    props = [
        ("is_on_top", {"object_handles": ["hb0"], "receptacle_handles": ["ht"]}),
        ("is_on_top", {"object_handles": ["hb1"], "receptacle_handles": ["ht"]}),
        ("is_next_to", {"entity_handles_a": ["hs"], "entity_handles_b": ["hb0"]}),
        ("is_filled", {"object_handles": ["hj"]}),
        ("is_on_top", {"object_handles": ["hj"], "receptacle_handles": ["ht"]}),
    ]
    env = _feedback_env(props, [False, True, False, False, False], names, {"jug_0": "jug"})
    lines = unmet_goal_feedback(env, known={"box_0", "box_1", "scissors_0"})
    assert lines == [
        "box_0 is not placed correctly.",
        "scissors_0 is not placed correctly next to box_0.",
        "Not found yet: a required jug.",
    ]
    assert not any("table_16" in line or "jug_0" in line for line in lines)
    env_known = _feedback_env(props[3:4], [False], names)
    assert unmet_goal_feedback(env_known, known={"jug_0"}) == ["jug_0 is not filled as required."]
    assert unmet_goal_feedback(SimpleNamespace(), known=set()) == []


def test_feedback_check_passes_unmet_lines_to_the_replan(monkeypatch):
    import habitat_llm.rail_planner.planner as rail_mod

    planner = _end_check_planner("oracle_feedback", monkeypatch, satisfied=False)
    planner._world_graph = _graph()
    monkeypatch.setattr(rail_mod, "unmet_goal_feedback", lambda env, known: [f"{len(known)} known"])
    planner._finish_planning("plan_completed")
    assert planner._end_replan_pending and planner._end_feedback == ["3 known"]
    assert planner.events[-1]["feedback"] == ["3 known"]
    # Plain oracle never builds feedback.
    plain = _end_check_planner("oracle", monkeypatch, satisfied=False)
    plain._world_graph = _graph()
    plain._finish_planning("plan_completed")
    assert plain._end_feedback == []


# ---------------------------------------------------------------------------
# Movable containers (e.g. fruit basket)
# ---------------------------------------------------------------------------


def _container_state(**kw):
    state = build_scene_state(_graph(**kw), 0)
    state.object_containers = {"jug_0"}  # treat the jug like an annotated basket
    return state


def _steps(state, subgoal):
    pred, args = RailPlanner.__new__(RailPlanner)._parse_subgoal(subgoal)
    return [(a.name, a.args[1:-1]) for a in expand_subgoal(state, pred, args)]


def test_in_movable_container():
    assert _steps(_container_state(), "in(mug_0, jug_0)") == [
        ("navigate", ("mug_0",)),
        ("pick", ("mug_0",)),
        ("navigate", ("jug_0",)),
        ("place_in", ("mug_0", "jug_0")),
    ]
    # Holding something else: put it down where the container stands first.
    assert _steps(_container_state(held="cup_0"), "in(mug_0, jug_0)")[:2] == [
        ("navigate", ("table_0",)),
        ("place_on", ("cup_0", "table_0")),
    ]


def test_in_movable_container_satisfied_and_errors():
    state = _container_state()
    state.contained_in["mug_0"] = "jug_0"
    assert _steps(state, "in(mug_0, jug_0)") == []
    with pytest.raises(SubgoalPlanError, match="holding 'jug_0'"):
        _steps(_container_state(held="jug_0"), "in(mug_0, jug_0)")
    with pytest.raises(SubgoalPlanError, match="inside itself"):
        _steps(_container_state(), "in(jug_0, jug_0)")
    # Not annotated: still rejected as a destination.
    with pytest.raises(SubgoalPlanError):
        _steps(build_scene_state(_graph(), 0), "in(mug_0, jug_0)")


def test_validate_accepts_movable_container(monkeypatch):
    from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner

    planner = RailPlanner.__new__(RailPlanner)
    planner._objects_by_type = {"movable": ["mug_0", "jug_0", "cup_0"]}
    planner._world_graph = object()
    planner._container_cache = {}
    monkeypatch.setattr(RailPlanner, "_movable_containers", lambda self, wg, names: {"jug_0"} & set(names))
    monkeypatch.setattr(VlmTampPddlPlanner, "_validate_subgoal", lambda self, sg: False)
    assert planner._validate_subgoal("in(mug_0, jug_0)")
    assert not planner._validate_subgoal("in(mug_0, cup_0)")
    assert not planner._validate_subgoal("in(jug_0, jug_0)")
    assert not planner._validate_subgoal("on(mug_0, jug_0)")


def test_movable_containers_listed_for_in(monkeypatch):
    from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner

    planner = RailPlanner.__new__(RailPlanner)
    monkeypatch.setattr(
        VlmTampPddlPlanner, "_build_objects_by_type",
        lambda self, wg, visible_object_names=None: {"movable": ["apple_0", "basket_0"], "container_furniture": ["cabinet_0"]},
    )
    monkeypatch.setattr(RailPlanner, "_movable_containers", lambda self, wg, names: {"basket_0"} & set(names))
    out = planner._build_objects_by_type(None)
    assert out["container_furniture"] == ["basket_0", "cabinet_0"]
    assert out["movable"] == ["apple_0", "basket_0"]


# ---------------------------------------------------------------------------
# waited_in (Wait skill) and intermediate placements in the oracle
# ---------------------------------------------------------------------------


def _appliance_state(**kw):
    state = build_scene_state(_graph(**kw), 0)
    state.powerable.add("cabinet_0")  # treat the cabinet like a microwave
    return state


def test_waited_in_puts_inside_closes_powers_and_waits():
    assert _steps(_appliance_state(cabinet_open=False), "waited_in(mug_0, cabinet_0)") == [
        ("navigate", ("cabinet_0",)),
        ("open", ("cabinet_0",)),
        ("navigate", ("mug_0",)),
        ("pick", ("mug_0",)),
        ("navigate", ("cabinet_0",)),
        ("place_in", ("mug_0", "cabinet_0")),
        ("navigate", ("cabinet_0",)),
        ("close", ("cabinet_0",)),
        ("navigate", ("cabinet_0",)),
        ("power_on", ("cabinet_0",)),
        ("wait", ("mug_0", "cabinet_0")),
    ]


def test_waited_in_object_already_inside():
    # cup_0 starts inside the cabinet: close only if open, no power state -> just wait.
    closed = build_scene_state(_graph(cabinet_open=False), 0)
    assert _steps(closed, "waited_in(cup_0, cabinet_0)") == [("wait", ("cup_0", "cabinet_0"))]
    opened = build_scene_state(_graph(cabinet_open=True), 0)
    assert _steps(opened, "heated_in(cup_0, cabinet_0)") == [
        ("navigate", ("cabinet_0",)),
        ("close", ("cabinet_0",)),
        ("wait", ("cup_0", "cabinet_0")),
    ]
    closed.waited.add(("cup_0", "cabinet_0"))
    assert _steps(closed, "waited_in(cup_0, cabinet_0)") == []
    with pytest.raises(SubgoalPlanError, match="no receptacle"):
        _steps(closed, "waited_in(mug_0, table_0)")


def test_wait_step_maps_to_wait_tool_and_is_remembered_after_success():
    planner = RailPlanner.__new__(RailPlanner)
    planner._reverse_name_map = {}
    planner._waited = set()
    plan = expand_subgoal(build_scene_state(_graph(cabinet_open=False), 0), "waited_in", ["cup_0", "cabinet_0"])
    assert [planner._action_to_tool(a) for a in plan] == [("Wait", "")]
    assert planner._waited == set()  # mapping alone does not count as waiting
    planner._on_skill_succeeded(plan[0])
    assert planner._waited == {("cup_0", "cabinet_0")}


def test_validate_waited_in(monkeypatch):
    from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner

    planner = RailPlanner.__new__(RailPlanner)
    planner._objects_by_type = {"movable": ["bread_0"], "container_furniture": ["microwave_0"]}
    planner._vprint = lambda *a, **k: None
    monkeypatch.setattr(VlmTampPddlPlanner, "_validate_subgoal", lambda self, sg: False)
    assert planner._validate_subgoal("waited_in(bread_0, microwave_0)")
    assert not planner._validate_subgoal("waited_in(bread_0, table_0)")
    assert not planner._validate_subgoal("waited_in(microwave_0, microwave_0)")


def test_extra_subgoals_only_in_rail_prompt():
    from habitat_llm.vlm_tamp.prompts_vlm_tamp import build_predicate_translation_prompt

    base = build_predicate_translation_prompt({"movable": ["bread_0"]})
    assert "waited_in" not in base and "{extra_subgoals}" not in base
    assert "one argument.\n], " in base  # VLM-TAMP prompt unchanged
    rail = build_predicate_translation_prompt({"movable": ["bread_0"]}, extra_subgoals=RailPlanner.EXTRA_SUBGOALS)
    assert "\n'waited_in(<movable>, <container_furniture>)'" in rail
    assert rail.index("waited_in") < rail.index("], ")


_T6_PROPS = [
    {"function_name": "is_on_top", "args": {"object_handles": ["h_a"], "receptacle_handles": ["h_t0"]}},
    {"function_name": "is_next_to", "args": {"entity_handles_a": ["h_b"], "entity_handles_b": ["h_a"]}},
    {"function_name": "is_inside", "args": {"object_handles": ["h_a"], "receptacle_handles": ["h_bk"]}},
    {"function_name": "is_waited_inside", "args": {"object_handles": ["h_a"], "receptacle_handles": ["h_bk"]}},
]
_T6_TERMINAL = [{"type": "TerminalSatisfactionConstraint", "args": {"proposition_indices": [0, 1]}}]


def test_oracle_separates_final_and_intermediate_placements():
    goals = placement_goals(_oracle_env(_T6_PROPS, _T6_TERMINAL))
    g = goals["box_0"]
    assert (g.relation, g.receptacles) == ("on", ["table_0"])
    assert g.via == [("within", "basket_0")] and g.wait_in == ["basket_0"]
    hint = format_hint(goals, {"box_0", "box_1"})
    assert (
        "- box_0 must end up on table_0, next to box_1; before that it must be inside basket_0 "
        "and waited there (waited_in)"
    ) in hint
    # Without a terminal constraint every placement is final (previous behaviour).
    assert placement_goals(_oracle_env(_T6_PROPS))["box_0"].receptacles == ["table_0", "basket_0"]


def test_oracle_reads_terminal_constraint_objects():
    from habitat_llm.agent.env.evaluation.evaluation_functions import TerminalSatisfactionConstraint
    from habitat_llm.rail_planner.placement_oracle import terminal_indices

    env = _oracle_env(_T6_PROPS, [TerminalSatisfactionConstraint([0, 1], n_propositions=4)])
    assert terminal_indices(env) == {0, 1}
    assert terminal_indices(_oracle_env(_T6_PROPS)) is None
