#!/usr/bin/env python3
# Tests for the PDDL domain: action dispatch, subgoal parsing, init fact
# generation, scope building, and regressions for every bug fixed.

import os

import pytest

from habitat_llm.pddlstream.problem import build_pddlstream_problem, extract_scope_names
from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner
from habitat_llm.world_model import Furniture, Object, Room, SpotRobot
from habitat_llm.world_model.world_graph import WorldGraph

_DOMAIN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "pddlstream", "pddl_domains", "partnr_rearrange_domain.pddl"
)


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------

def _make_basic_graph():
    """Room + table + mug + agent. Covers navigate, pick, place_on."""
    wg = WorldGraph()
    room = Room("kitchen", {"type": "room"})
    table = Furniture("table", {"type": "table", "is_articulated": False})
    mug = Object("mug", {"type": "mug", "states": {}})
    agent = SpotRobot("agent_0", {"type": "agent"})
    for node in [room, table, mug, agent]:
        wg.add_node(node)
    wg.add_edge(table, room, "inside", "contains")
    wg.add_edge(mug, table, "on", "supports")
    wg.add_edge(agent, room, "inside", "contains")
    return wg


def _make_cabinet_graph():
    """Room + cabinet (articulated container) + cup + agent.
    Covers open, close, place_in."""
    wg = WorldGraph()
    room = Room("kitchen", {"type": "room"})
    cabinet = Furniture("cabinet", {"type": "cabinet", "is_articulated": True})
    cup = Object("cup", {"type": "cup", "states": {}})
    agent = SpotRobot("agent_0", {"type": "agent"})
    for node in [room, cabinet, cup, agent]:
        wg.add_node(node)
    wg.add_edge(cabinet, room, "inside", "contains")
    wg.add_edge(cup, cabinet, "on", "supports")
    wg.add_edge(agent, room, "inside", "contains")
    return wg


def _make_faucet_graph():
    """Room + sink (faucet) + bottle + agent. Covers fill_held, clean_object."""
    wg = WorldGraph()
    room = Room("kitchen", {"type": "room"})
    sink = Furniture("sink", {"type": "sink", "is_articulated": False, "components": ["faucet"]})
    bottle = Object("bottle", {"type": "bottle", "states": {}})
    agent = SpotRobot("agent_0", {"type": "agent"})
    for node in [room, sink, bottle, agent]:
        wg.add_node(node)
    wg.add_edge(sink, room, "inside", "contains")
    wg.add_edge(bottle, sink, "on", "supports")
    wg.add_edge(agent, room, "inside", "contains")
    return wg


def _make_lamp_graph():
    """Room + lamp furniture (powered) + lamp object (powered) + agent.
    Covers power_on, power_off, power_on_object, power_off_object."""
    wg = WorldGraph()
    room = Room("bedroom", {"type": "room"})
    lamp_furn = Furniture(
        "lamp_furn",
        {"type": "lamp", "is_articulated": False,
         "states": {"is_powered_on": False}},
    )
    lamp_obj = Object(
        "lamp_obj",
        {"type": "lamp", "states": {"is_powered_on": False}},
    )
    agent = SpotRobot("agent_0", {"type": "agent"})
    for node in [room, lamp_furn, lamp_obj, agent]:
        wg.add_node(node)
    wg.add_edge(lamp_furn, room, "inside", "contains")
    wg.add_edge(lamp_obj, lamp_furn, "on", "supports")
    wg.add_edge(agent, room, "inside", "contains")
    return wg


def _make_pour_graph():
    """Room + sink (faucet) + bottle1 (filled) + bottle2 + agent. Covers pour."""
    wg = WorldGraph()
    room = Room("kitchen", {"type": "room"})
    sink = Furniture("sink", {"type": "sink", "is_articulated": False, "components": ["faucet"]})
    bottle1 = Object("bottle1", {"type": "bottle", "states": {"is_filled": True}})
    bottle2 = Object("bottle2", {"type": "bottle", "states": {}})
    agent = SpotRobot("agent_0", {"type": "agent"})
    for node in [room, sink, bottle1, bottle2, agent]:
        wg.add_node(node)
    wg.add_edge(sink, room, "inside", "contains")
    wg.add_edge(bottle1, sink, "on", "supports")
    wg.add_edge(bottle2, sink, "on", "supports")
    wg.add_edge(agent, room, "inside", "contains")
    return wg


def _make_planner_stub():
    planner = VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    planner._reverse_name_map = {}
    return planner


# ---------------------------------------------------------------------------
# 1. _action_to_tool dispatch
# ---------------------------------------------------------------------------

class TestActionDispatch:
    def setup_method(self):
        self.p = _make_planner_stub()

    def test_navigate_before_pick(self):
        nav = ("navigate", ["agent_0", "mug", "kitchen"])
        nxt = ("pick", ["agent_0", "mug", "kitchen"])
        skill, arg = self.p._action_to_tool(nav, nxt)
        assert skill == "Navigate"
        assert arg == "mug"

    def test_navigate_before_place_on(self):
        nav = ("navigate", ["agent_0", "table", "kitchen"])
        nxt = ("place_on", ["agent_0", "mug", "table", "kitchen"])
        skill, arg = self.p._action_to_tool(nav, nxt)
        assert skill == "Navigate"
        assert arg == "table"

    def test_navigate_before_place_in(self):
        nav = ("navigate", ["agent_0", "cabinet", "kitchen"])
        nxt = ("place_in", ["agent_0", "cup", "cabinet", "kitchen"])
        skill, arg = self.p._action_to_tool(nav, nxt)
        assert skill == "Navigate"
        assert arg == "cabinet"

    def test_navigate_before_fill_held_goes_to_faucet(self):
        nav = ("navigate", ["agent_0", "sink", "kitchen"])
        nxt = ("fill_held", ["agent_0", "bottle", "sink", "kitchen"])
        skill, arg = self.p._action_to_tool(nav, nxt)
        assert skill == "Navigate"
        assert arg == "sink"  # nargs[2] — the faucet, not the bottle

    def test_navigate_before_clean_object_goes_to_faucet(self):
        nav = ("navigate", ["agent_0", "sink", "kitchen"])
        nxt = ("clean_object", ["agent_0", "bottle", "sink", "kitchen"])
        skill, arg = self.p._action_to_tool(nav, nxt)
        assert skill == "Navigate"
        assert arg == "sink"  # nargs[2] — the faucet

    def test_navigate_before_open(self):
        nav = ("navigate", ["agent_0", "cabinet", "kitchen"])
        nxt = ("open", ["agent_0", "cabinet", "kitchen"])
        skill, arg = self.p._action_to_tool(nav, nxt)
        assert skill == "Navigate"
        assert arg == "cabinet"

    def test_navigate_before_power_on(self):
        nav = ("navigate", ["agent_0", "lamp_furn", "bedroom"])
        nxt = ("power_on", ["agent_0", "lamp_furn", "bedroom"])
        skill, arg = self.p._action_to_tool(nav, nxt)
        assert skill == "Navigate"
        assert arg == "lamp_furn"

    def test_navigate_before_clean_furniture(self):
        nav = ("navigate", ["agent_0", "table", "kitchen"])
        nxt = ("clean_furniture", ["agent_0", "table", "kitchen"])
        skill, arg = self.p._action_to_tool(nav, nxt)
        assert skill == "Navigate"
        assert arg == "table"

    def test_pick(self):
        skill, arg = self.p._action_to_tool(("pick", ["agent_0", "mug", "kitchen"]))
        assert skill == "Pick"
        assert arg == "mug"

    def test_place_on(self):
        skill, arg = self.p._action_to_tool(("place_on", ["agent_0", "mug", "table", "kitchen"]))
        assert skill == "Place"
        assert arg == "mug, on, table, None, None"

    def test_place_in(self):
        skill, arg = self.p._action_to_tool(("place_in", ["agent_0", "cup", "cabinet", "kitchen"]))
        assert skill == "Place"
        assert arg == "cup, within, cabinet, None, None"

    def test_open(self):
        skill, arg = self.p._action_to_tool(("open", ["agent_0", "cabinet", "kitchen"]))
        assert skill == "Open"
        assert arg == "cabinet"

    def test_close(self):
        skill, arg = self.p._action_to_tool(("close", ["agent_0", "cabinet", "kitchen"]))
        assert skill == "Close"
        assert arg == "cabinet"

    def test_power_on_furniture(self):
        skill, arg = self.p._action_to_tool(("power_on", ["agent_0", "lamp_furn", "bedroom"]))
        assert skill == "PowerOn"
        assert arg == "lamp_furn"

    def test_power_on_object(self):
        skill, arg = self.p._action_to_tool(("power_on_object", ["agent_0", "lamp_obj", "bedroom"]))
        assert skill == "PowerOn"
        assert arg == "lamp_obj"

    def test_power_off_furniture(self):
        skill, arg = self.p._action_to_tool(("power_off", ["agent_0", "lamp_furn", "bedroom"]))
        assert skill == "PowerOff"
        assert arg == "lamp_furn"

    def test_power_off_object(self):
        skill, arg = self.p._action_to_tool(("power_off_object", ["agent_0", "lamp_obj", "bedroom"]))
        assert skill == "PowerOff"
        assert arg == "lamp_obj"

    def test_fill_held(self):
        skill, arg = self.p._action_to_tool(("fill_held", ["agent_0", "bottle", "sink", "kitchen"]))
        assert skill == "Fill"
        assert arg == "bottle"

    def test_clean_object(self):
        skill, arg = self.p._action_to_tool(("clean_object", ["agent_0", "bottle", "sink", "kitchen"]))
        assert skill == "Clean"
        assert arg == "bottle"

    def test_clean_furniture(self):
        skill, arg = self.p._action_to_tool(("clean_furniture", ["agent_0", "table", "kitchen"]))
        assert skill == "Clean"
        assert arg == "table"

    def test_pour(self):
        skill, arg = self.p._action_to_tool(("pour", ["agent_0", "bottle1", "bottle2", "kitchen"]))
        assert skill == "Pour"
        assert arg == "bottle2"

    def test_unknown_action_returns_none(self):
        skill, arg = self.p._action_to_tool(("unknown_action", ["agent_0", "foo"]))
        assert skill is None
        assert arg is None


# ---------------------------------------------------------------------------
# 2. _subgoal_to_goal_literal parsing
# ---------------------------------------------------------------------------

class TestSubgoalParsing:
    def setup_method(self):
        self.p = _make_planner_stub()
        # Minimal name_map: original name -> pddl name (same here, no spaces)
        self.name_map = {
            "mug": "mug",
            "cup": "cup",
            "bottle": "bottle",
            "table": "table",
            "cabinet": "cabinet",
            "sink": "sink",
            "lamp": "lamp",
            "lamp_obj": "lamp_obj",
            "kitchen": "kitchen",
        }

    def _lit(self, subgoal):
        return self.p._subgoal_to_goal_literal(subgoal, "agent_0", self.name_map)

    def test_on(self):
        assert self._lit("on(mug, table)") == ("on", "mug", "table")

    def test_in(self):
        assert self._lit("in(cup, cabinet)") == ("in", "cup", "cabinet")

    def test_picked(self):
        assert self._lit("picked(mug)") == ("holding", "agent_0", "mug")

    def test_holding(self):
        assert self._lit("holding(mug)") == ("holding", "agent_0", "mug")

    def test_opened_door(self):
        assert self._lit("opened-door(cabinet)") == ("open", "cabinet")

    def test_opened_drawer(self):
        assert self._lit("opened-drawer(cabinet)") == ("open", "cabinet")

    def test_closed_door(self):
        assert self._lit("closed-door(cabinet)") == ("closed", "cabinet")

    def test_closed_drawer(self):
        assert self._lit("closed-drawer(cabinet)") == ("closed", "cabinet")

    def test_powered_on(self):
        assert self._lit("powered_on(lamp_obj)") == ("powered_on", "lamp_obj")

    def test_powered_on_alias(self):
        assert self._lit("powered-on(lamp_obj)") == ("powered_on", "lamp_obj")

    def test_powered_off(self):
        assert self._lit("powered_off(lamp_obj)") == ("powered_off", "lamp_obj")

    def test_filled(self):
        assert self._lit("filled(bottle)") == ("filled", "bottle")

    def test_poured_into(self):
        assert self._lit("poured-into(bottle)") == ("filled", "bottle")

    def test_cleaned(self):
        assert self._lit("cleaned(bottle)") == ("cleaned", "bottle")

    def test_clean_alias(self):
        assert self._lit("clean(bottle)") == ("cleaned", "bottle")

    def test_unknown_returns_none(self):
        assert self._lit("nonexistent(foo)") is None

    def test_unknown_object_returns_none(self):
        assert self._lit("on(no_such_object, table)") is None


# ---------------------------------------------------------------------------
# 3. Init fact generation
# ---------------------------------------------------------------------------

class TestInitFacts:
    def test_basic_graph_init_facts(self):
        wg = _make_basic_graph()
        _, init, name_map, _ = build_pddlstream_problem(wg, 0, ("on", "mug", "table"))
        assert ("room", "kitchen") in init
        assert ("furniture", "table") in init
        assert ("object", "mug") in init
        assert ("agent", "agent_0") in init
        assert ("handempty", "agent_0") in init
        assert ("at", "agent_0", "kitchen") in init
        assert ("inroom", "table", "kitchen") in init
        assert ("inroom", "mug", "kitchen") in init
        assert ("on", "mug", "table") in init

    def test_cabinet_is_joint_and_container(self):
        wg = _make_cabinet_graph()
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("in", "cup", "cabinet"))
        assert ("joint", "cabinet") in init
        assert ("container", "cabinet") in init
        assert ("closed", "cabinet") in init

    def test_wardrobe_is_container(self):
        wg = WorldGraph()
        room = Room("bedroom", {"type": "room"})
        wardrobe = Furniture(
            "wardrobe_56",
            {"type": "wardrobe", "is_articulated": True, "states": {"is_open": False}},
        )
        toy = Object("toy_airplane_0", {"type": "toy", "states": {}})
        agent = SpotRobot("agent_0", {"type": "agent"})
        for node in [room, wardrobe, toy, agent]:
            wg.add_node(node)
        wg.add_edge(wardrobe, room, "inside", "contains")
        wg.add_edge(toy, room, "inside", "contains")
        wg.add_edge(agent, room, "inside", "contains")
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("in", "toy_airplane_0", "wardrobe_56"))
        assert ("container", "wardrobe_56") in init, \
            "wardrobe must be a container so place_in can be used for 'in' goals"
        assert ("joint", "wardrobe_56") in init
        assert ("closed", "wardrobe_56") in init

    def test_faucet_furniture_has_faucet_fact(self):
        wg = _make_faucet_graph()
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("filled", "bottle"))
        assert ("has_faucet", "sink") in init

    def test_powered_off_object_fact(self):
        wg = _make_lamp_graph()
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("powered_on", "lamp_obj"))
        assert ("powered_off", "lamp_obj") in init
        assert ("powered_on", "lamp_obj") not in init

    def test_powered_on_object_fact(self):
        wg = _make_lamp_graph()
        # Mutate state to powered_on
        for node in wg.graph:
            if hasattr(node, "name") and node.name == "lamp_obj":
                node.properties["states"]["is_powered_on"] = True
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("powered_off", "lamp_obj"))
        assert ("powered_on", "lamp_obj") in init
        assert ("powered_off", "lamp_obj") not in init

    def test_powered_off_furniture_fact(self):
        wg = _make_lamp_graph()
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("powered_on", "lamp_furn"))
        assert ("powered_off", "lamp_furn") in init

    def test_filled_object_fact(self):
        wg = _make_pour_graph()
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("filled", "bottle2"))
        assert ("filled", "bottle1") in init
        assert ("filled", "bottle2") not in init

    def test_cleaned_object_fact(self):
        wg = _make_faucet_graph()
        # Mark bottle as already clean
        for node in wg.graph:
            if hasattr(node, "name") and node.name == "bottle":
                node.properties["states"]["is_clean"] = True
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("cleaned", "bottle"))
        assert ("cleaned", "bottle") in init

    def test_uncleaned_object_has_no_cleaned_fact(self):
        wg = _make_faucet_graph()
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("cleaned", "bottle"))
        assert ("cleaned", "bottle") not in init

    def test_name_sanitization(self):
        wg = WorldGraph()
        room = Room("living room", {"type": "room"})
        table = Furniture("coffee table", {"type": "table", "is_articulated": False})
        agent = SpotRobot("agent_0", {"type": "agent"})
        for node in [room, table, agent]:
            wg.add_node(node)
        wg.add_edge(table, room, "inside", "contains")
        wg.add_edge(agent, room, "inside", "contains")
        _, init, name_map, _ = build_pddlstream_problem(wg, 0, ("handempty", "agent_0"))
        assert name_map["coffee table"] == "coffee_table"
        assert name_map["living room"] == "living_room"
        assert ("furniture", "coffee_table") in init


# ---------------------------------------------------------------------------
# 4. Scope builder
# ---------------------------------------------------------------------------

class TestScopeBuilder:
    def test_filled_goal_includes_faucet(self):
        wg = _make_faucet_graph()
        # Build name_map first
        _, _, name_map, _ = build_pddlstream_problem(wg, 0, ("handempty", "agent_0"))
        goal = ("filled", name_map.get("bottle", "bottle"))
        scope = extract_scope_names(wg, 0, goal)
        assert "sink" in scope

    def test_cleaned_goal_includes_faucet(self):
        wg = _make_faucet_graph()
        _, _, name_map, _ = build_pddlstream_problem(wg, 0, ("handempty", "agent_0"))
        goal = ("cleaned", name_map.get("bottle", "bottle"))
        scope = extract_scope_names(wg, 0, goal)
        assert "sink" in scope

    def test_basic_on_goal_does_not_include_unrelated_furniture(self):
        # Graph with two separate furniture; goal only involves one
        wg = WorldGraph()
        room = Room("kitchen", {"type": "room"})
        table = Furniture("table", {"type": "table", "is_articulated": False})
        fridge = Furniture("fridge", {"type": "fridge", "is_articulated": True})
        mug = Object("mug", {"type": "mug", "states": {}})
        agent = SpotRobot("agent_0", {"type": "agent"})
        for node in [room, table, fridge, mug, agent]:
            wg.add_node(node)
        wg.add_edge(table, room, "inside", "contains")
        wg.add_edge(fridge, room, "inside", "contains")
        wg.add_edge(mug, table, "on", "supports")
        wg.add_edge(agent, room, "inside", "contains")

        _, _, name_map, _ = build_pddlstream_problem(wg, 0, ("handempty", "agent_0"))
        goal = ("on", name_map.get("mug", "mug"), name_map.get("table", "table"))
        scope = extract_scope_names(wg, 0, goal)
        assert "mug" in scope
        assert "table" in scope
        assert "fridge" not in scope

    def test_at_goal_includes_furniture_in_target_room(self):
        wg = _make_basic_graph()
        _, _, name_map, _ = build_pddlstream_problem(wg, 0, ("handempty", "agent_0"))
        goal = ("at", "agent_0", name_map.get("kitchen", "kitchen"))
        scope = extract_scope_names(wg, 0, goal)
        # At least one furniture from kitchen must be in scope for navigate ?x grounding
        assert "table" in scope or "kitchen" in scope

    def test_faucet_scope_falls_back_globally(self):
        # Bottle in bedroom, faucet only in kitchen — global fallback should find it
        wg = WorldGraph()
        bedroom = Room("bedroom", {"type": "room"})
        kitchen = Room("kitchen", {"type": "room"})
        sink = Furniture("sink", {"type": "sink", "is_articulated": False, "components": ["faucet"]})
        bottle = Object("bottle", {"type": "bottle", "states": {}})
        agent = SpotRobot("agent_0", {"type": "agent"})
        for node in [bedroom, kitchen, sink, bottle, agent]:
            wg.add_node(node)
        wg.add_edge(sink, kitchen, "inside", "contains")
        wg.add_edge(bottle, bedroom, "inside", "contains")
        wg.add_edge(agent, bedroom, "inside", "contains")

        _, _, name_map, _ = build_pddlstream_problem(wg, 0, ("handempty", "agent_0"))
        goal = ("filled", name_map.get("bottle", "bottle"))
        scope = extract_scope_names(wg, 0, goal)
        assert "sink" in scope


# ---------------------------------------------------------------------------
# Regression tests – one test per bug that was previously broken
# ---------------------------------------------------------------------------

class TestDomainRegressions:
    """Regressions: each test encodes a previously broken behaviour."""

    def setup_method(self):
        self.p = _make_planner_stub()

    # ------------------------------------------------------------------
    # Domain-file structural regressions
    # ------------------------------------------------------------------

    def _domain_text(self):
        with open(_DOMAIN_PATH) as f:
            return f.read()

    def test_fill_action_removed(self):
        """Bug: standalone `fill` action existed, letting planner skip picking
        the object. It was removed; only `fill_held` should remain."""
        text = self._domain_text()
        assert "(:action fill\n" not in text, "standalone 'fill' action must not exist"
        assert "(:action fill_held" in text

    def test_fill_held_requires_holding(self):
        """Bug: original fill allowed filling without holding the object.
        fill_held must require (holding ?a ?o)."""
        text = self._domain_text()
        start = text.index("(:action fill_held")
        block = text[start: text.index("(:action", start + 1)]
        assert "(holding ?a ?o)" in block

    def test_fill_held_uses_near_not_at_faucet(self):
        """Bug: fill/fill_held previously required (at_faucet ?a ?f) which was
        a sticky predicate never cleared, so the planner skipped navigation.
        The precondition must use (near ?a ?f) instead."""
        text = self._domain_text()
        start = text.index("(:action fill_held")
        block = text[start: text.index("(:action", start + 1)]
        assert "(near ?a ?f)" in block
        assert "(at_faucet" not in block

    def test_place_in_requires_open_container(self):
        """Bug: place_in had no check for whether the container was open,
        so the planner would try to place into a closed cabinet."""
        text = self._domain_text()
        start = text.index("(:action place_in")
        block = text[start: text.index("(:action", start + 1)]
        assert "(not (closed ?f))" in block

    def test_power_on_off_object_actions_exist(self):
        """Bug: power_on/power_off only typed ?f as furniture, so lamp objects
        (which are `object` not `furniture`) caused plan failures."""
        text = self._domain_text()
        assert "(:action power_on_object" in text
        assert "(:action power_off_object" in text

    def test_clean_object_requires_holding_and_faucet(self):
        """Bug: clean_object initially only required being near the object (no
        pick), mirroring the old `fill` bug. Must require holding and faucet."""
        text = self._domain_text()
        start = text.index("(:action clean_object")
        block = text[start: text.index("(:action", start + 1)]
        assert "(holding ?a ?o)" in block
        assert "(has_faucet ?f)" in block
        assert "(near ?a ?f)" in block

    def test_clean_object_near_faucet_not_object(self):
        """The faucet (?f, index 2) must be the near target, not the object (?o)."""
        text = self._domain_text()
        start = text.index("(:action clean_object")
        block = text[start: text.index("(:action", start + 1)]
        assert "(near ?a ?f)" in block
        assert "(near ?a ?o)" not in block

    def test_near_predicate_consumed_by_all_actions(self):
        """Bug: `near` was sticky — navigate added it but nothing removed it.
        This let the planner skip re-navigation steps (e.g. navigate(faucet) →
        place_on(obj) → navigate(bedroom) → pick → clean_object without
        re-navigating to faucet, because near(faucet) never got cleared).

        Fix: every action requiring (near ?a ?x) must also negate it in its
        effect so the planner is forced to emit a fresh navigate when needed."""
        text = self._domain_text()
        cases = [
            ("pick",             "(not (near ?a ?o))"),
            ("place_on",         "(not (near ?a ?f))"),
            ("place_in",         "(not (near ?a ?f))"),
            ("open",             "(not (near ?a ?j))"),
            ("close",            "(not (near ?a ?j))"),
            ("power_on_object",  "(not (near ?a ?o))"),
            ("power_off_object", "(not (near ?a ?o))"),
            ("fill_held",        "(not (near ?a ?f))"),
            ("clean_object",     "(not (near ?a ?f))"),
            ("clean_furniture",  "(not (near ?a ?f))"),
            ("pour",             "(not (near ?a ?tgt))"),
        ]
        for action_name, expected_neg in cases:
            start = text.index(f"(:action {action_name}")
            next_act = text.find("(:action", start + 1)
            block = text[start:next_act] if next_act != -1 else text[start:]
            assert expected_neg in block, (
                f"Action '{action_name}' is missing '{expected_neg}' — "
                f"'near' remains sticky for this action."
            )
        # power_on / power_off (furniture variant) — check without matching power_on_object
        for act in ("power_on", "power_off"):
            start = text.index(f"(:action {act}\n")
            next_act = text.find("(:action", start + 1)
            block = text[start:next_act] if next_act != -1 else text[start:]
            assert "(not (near ?a ?f))" in block, (
                f"Action '{act}' (furniture) is missing '(not (near ?a ?f))' — sticky near."
            )

    def test_place_on_blocked_on_faucet_furniture(self):
        """Bug: place_on had no guard against faucet furniture. The planner
        chose cabinet_97 (a faucet/sink cabinet) as a placement surface.
        The simulator placed the object inside the closed cabinet rather than on
        its surface, making it unpickable. Fix: add (not (has_faucet ?f)) to
        place_on so the planner always chooses a non-faucet surface."""
        text = self._domain_text()
        start = text.index("(:action place_on")
        block = text[start: text.index("(:action", start + 1)]
        assert "(not (has_faucet ?f))" in block

    def test_cleaned_predicate_exists(self):
        """Bug: cleaned predicate was missing so any goal with cleaned() would
        be rejected by the planner."""
        text = self._domain_text()
        assert "(cleaned ?x)" in text

    # ------------------------------------------------------------------
    # Dispatch regressions: _action_to_tool
    # ------------------------------------------------------------------

    def test_navigate_before_fill_held_targets_faucet_not_object(self):
        """Bug: navigate lookahead for fill_held used nargs[1] (the object)
        instead of nargs[2] (the faucet). The robot navigated to the bottle
        rather than the sink so fill never happened."""
        # fill_held(?a, ?o, ?f, ?r) — index 2 is the faucet
        nav = ("navigate", ["agent_0", "sink_1", "kitchen"])
        nxt = ("fill_held", ["agent_0", "bottle_1", "sink_1", "kitchen"])
        skill, target = self.p._action_to_tool(nav, nxt)
        assert skill == "Navigate"
        assert target == "sink_1"  # faucet, NOT bottle_1

    def test_navigate_before_fill_held_does_not_target_object(self):
        """Negative companion: target must NOT be the object being filled."""
        nav = ("navigate", ["agent_0", "bottle_1", "kitchen"])
        nxt = ("fill_held", ["agent_0", "bottle_1", "sink_1", "kitchen"])
        skill, target = self.p._action_to_tool(nav, nxt)
        assert target == "sink_1"
        assert target != "bottle_1"

    def test_navigate_before_clean_object_targets_faucet(self):
        """Bug: clean_object lookahead was missing; without it the robot would
        navigate to the object rather than the faucet."""
        nav = ("navigate", ["agent_0", "sink_1", "kitchen"])
        nxt = ("clean_object", ["agent_0", "bottle_1", "sink_1", "kitchen"])
        skill, target = self.p._action_to_tool(nav, nxt)
        assert skill == "Navigate"
        assert target == "sink_1"  # faucet

    def test_power_on_object_dispatches_not_unsupported(self):
        """Bug: power_on_object was added to navigate-lookahead but NOT to the
        dispatch block, causing 'Unsupported action' at runtime."""
        skill, target = self.p._action_to_tool(
            ("power_on_object", ["agent_0", "lamp_3", "living_room"])
        )
        assert skill == "PowerOn"
        assert target == "lamp_3"

    def test_power_off_object_dispatches_not_unsupported(self):
        """Same as above but for power_off_object."""
        skill, target = self.p._action_to_tool(
            ("power_off_object", ["agent_0", "lamp_3", "living_room"])
        )
        assert skill == "PowerOff"
        assert target == "lamp_3"

    def test_navigate_before_power_on_object_targets_lamp(self):
        """Bug: navigate lookahead did not include power_on_object, so navigation
        before power_on_object fell through to a default or wrong target."""
        nav = ("navigate", ["agent_0", "lamp_3", "living_room"])
        nxt = ("power_on_object", ["agent_0", "lamp_3", "living_room"])
        skill, target = self.p._action_to_tool(nav, nxt)
        assert skill == "Navigate"
        assert target == "lamp_3"

    def test_navigate_before_power_off_object_targets_lamp(self):
        nav = ("navigate", ["agent_0", "lamp_3", "living_room"])
        nxt = ("power_off_object", ["agent_0", "lamp_3", "living_room"])
        skill, target = self.p._action_to_tool(nav, nxt)
        assert skill == "Navigate"
        assert target == "lamp_3"

    def test_clean_furniture_dispatches_not_unsupported(self):
        """Bug: clean_furniture was missing from the dispatch block."""
        skill, target = self.p._action_to_tool(
            ("clean_furniture", ["agent_0", "sink_1", "kitchen"])
        )
        assert skill == "Clean"
        assert target == "sink_1"

    def test_clean_object_dispatches_not_unsupported(self):
        """clean_object must dispatch to 'Clean', not raise 'Unsupported action'."""
        skill, target = self.p._action_to_tool(
            ("clean_object", ["agent_0", "bottle_1", "sink_1", "kitchen"])
        )
        assert skill == "Clean"
        assert target == "bottle_1"

    # ------------------------------------------------------------------
    # Subgoal-parsing regressions
    # ------------------------------------------------------------------

    def _lit(self, subgoal, name_map=None):
        nm = name_map or {"cup_1": "cup_1", "bottle_1": "bottle_1"}
        return self.p._subgoal_to_goal_literal(subgoal, "agent_0", nm)

    def test_cleaned_subgoal_parsed(self):
        """Bug: cleaned predicate was not recognised by _subgoal_to_goal_literal
        so VLM subgoals with 'cleaned(object)' were silently dropped."""
        result = self._lit("cleaned(cup_1)")
        assert result is not None
        assert result[0] == "cleaned"

    def test_filled_subgoal_still_parsed_after_fill_removal(self):
        """Regression: filled must still parse after the fill action was removed."""
        result = self._lit("filled(bottle_1)")
        assert result is not None
        assert result[0] == "filled"

    # ------------------------------------------------------------------
    # Scope-builder regressions
    # ------------------------------------------------------------------

    def test_cleaned_scope_includes_faucet(self):
        """Bug: scope builder included faucet for 'filled' goals but not for
        'cleaned' goals, so clean_object could not find a faucet in scope."""
        wg = WorldGraph()
        kitchen = Room("kitchen", {"type": "room"})
        sink = Furniture("sink", {"type": "sink", "is_articulated": False, "components": ["faucet"]})
        sponge = Object("sponge_1", {"type": "sponge", "states": {}})
        agent = SpotRobot("agent_0", {"type": "agent"})
        for node in [kitchen, sink, sponge, agent]:
            wg.add_node(node)
        wg.add_edge(sink, kitchen, "inside", "contains")
        wg.add_edge(sponge, kitchen, "inside", "contains")
        wg.add_edge(agent, kitchen, "inside", "contains")

        _, init, name_map, _ = build_pddlstream_problem(wg, 0, ("handempty", "agent_0"))
        goal = ("cleaned", name_map.get("sponge_1", "sponge_1"))
        scope = extract_scope_names(wg, 0, goal)
        assert "sink" in scope

    # ------------------------------------------------------------------
    # Init-fact regressions
    # ------------------------------------------------------------------

    def test_closed_container_emits_closed_not_open(self):
        """Bug: place_in had no (not (closed ?f)) precondition. Verify that a
        closed cabinet is emitted as 'closed' in init facts (not 'open'), so the
        precondition correctly blocks placing into it without opening first."""
        wg = _make_cabinet_graph()
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("in", "cup", "cabinet"))
        assert ("closed", "cabinet") in init
        assert ("open", "cabinet") not in init

    def test_powered_off_object_emits_init_fact(self):
        """Bug: powered_off init facts were only generated for furniture. Object
        lamps (typed as `object`) were missing the fact, so power_off_object
        goals were treated as already-satisfied or unsolvable."""
        wg = _make_lamp_graph()
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("powered_on", "lamp_obj"))
        assert ("powered_off", "lamp_obj") in init

    def test_powered_on_object_does_not_get_powered_off_fact(self):
        """Companion: a powered-on object must NOT have (powered_off) in init."""
        wg = _make_lamp_graph()
        for node in wg.graph:
            if hasattr(node, "name") and node.name == "lamp_obj":
                node.properties["states"]["is_powered_on"] = True
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("powered_off", "lamp_obj"))
        assert ("powered_on", "lamp_obj") in init
        assert ("powered_off", "lamp_obj") not in init

    def test_cleaned_init_fact_for_dirty_object(self):
        """Regression: a dirty object must NOT have (cleaned ?o) in init facts."""
        wg = _make_faucet_graph()
        # bottle starts without is_clean → should not be marked cleaned
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("cleaned", "bottle"))
        assert ("cleaned", "bottle") not in init

    def test_cleaned_init_fact_for_clean_object(self):
        """A clean object must have (cleaned ?o) in init facts so the planner
        knows it is already clean and doesn't re-clean it."""
        wg = _make_faucet_graph()
        for node in wg.graph:
            if hasattr(node, "name") and node.name == "bottle":
                node.properties["states"]["is_clean"] = True
        _, init, _, _ = build_pddlstream_problem(wg, 0, ("cleaned", "bottle"))
        assert ("cleaned", "bottle") in init
