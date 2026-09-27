#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
PARTNR-side tests: WorldGraph -> observation, symbolic action -> Habitat skills,
and the skill-hold loop.

These build a WorldGraph by hand and stub out ``process_high_level_actions``, so
no simulator, GPU or network is involved.

NOTE on import order: habitat_llm must be imported before habitat in this
environment, otherwise magnum's static plugins are registered twice and the
interpreter segfaults. This is pre-existing and applies to every planner module,
so these tests never import habitat directly.
"""

import json

import pytest

from habitat_llm.planner.tru_pomdp.planner import TruPOMDPPlanner
from habitat_llm.planner.tru_pomdp.scene import (
    HELD,
    Action,
    ActionType,
    GoalAtom,
    SceneState,
)
from habitat_llm.planner.tru_pomdp.search import DespotConfig
from habitat_llm.planner.tru_pomdp.toh import TohConfig
from habitat_llm.world_model import Furniture, Object, Room, SpotRobot
from habitat_llm.world_model.world_graph import WorldGraph


class StubLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, stop=None, max_length=None):
        self.prompts.append(prompt)
        return self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]


class FakeAgent:
    def __init__(self, uid: int = 0) -> None:
        self.uid = uid

    def reset(self) -> None:
        pass


def fenced(payload) -> str:
    return "reasoning\n\n```json\n" + json.dumps(payload) + "\n```\n"


def make_world_graph(robot_at=(0.0, 0.0, 0.0)) -> WorldGraph:
    """kitchen_1 (counter_24, cabinet_3 closed, fridge_0 closed) + living_room_1."""
    graph = WorldGraph()
    kitchen = Room("kitchen_1", {"type": "room"})
    living_room = Room("living_room_1", {"type": "room"})
    counter = Furniture(
        "counter_24",
        {
            "type": "counter",
            "is_articulated": False,
            "translation": [0.0, 0.0, 0.0],
        },
    )
    cabinet = Furniture(
        "cabinet_3",
        {
            "type": "cabinet",
            "is_articulated": True,
            "states": {"is_open": False},
            "translation": [3.0, 0.0, 0.0],
        },
    )
    fridge = Furniture(
        "fridge_0",
        {
            "type": "fridge",
            "is_articulated": True,
            "states": {"is_open": False},
            "translation": [6.0, 0.0, 0.0],
        },
    )
    table = Furniture(
        "table_10",
        {
            "type": "table",
            "is_articulated": False,
            "translation": [0.0, 0.0, 12.0],
        },
    )
    plate = Object("plate_2", {"type": "plate", "states": {}})
    lamp = Object("lamp_0", {"type": "lamp", "states": {"is_powered_on": True}})
    robot = SpotRobot("agent_0", {"type": "agent", "translation": list(robot_at)})

    for node in (kitchen, living_room, counter, cabinet, fridge, table, plate, lamp, robot):
        graph.add_node(node)
    graph.add_edge(counter, kitchen, "inside", "contains")
    graph.add_edge(cabinet, kitchen, "inside", "contains")
    graph.add_edge(fridge, kitchen, "inside", "contains")
    graph.add_edge(table, living_room, "inside", "contains")
    graph.add_edge(plate, counter, "on", "supports")
    graph.add_edge(lamp, table, "on", "supports")
    graph.add_edge(robot, kitchen, "inside", "contains")
    return graph


def make_planner(llm_responses=(), **overrides) -> TruPOMDPPlanner:
    """
    Build a planner without running ``__init__``, which would need a real
    EnvironmentInterface. Mirrors the stub pattern used by test_pddl_domain.
    """
    planner = TruPOMDPPlanner.__new__(TruPOMDPPlanner)
    planner.planner_config = None
    planner.env_interface = None
    planner._agents = [FakeAgent(0)]
    planner.enable_rag = False
    planner.swap_instruction = True

    planner.despot_config = DespotConfig(
        num_scenarios=6,
        max_search_depth=10,
        rollout_depth=6,
        num_trials=30,
        planning_time_s=10.0,
        seed=3,
    )
    planner.toh_config = TohConfig(c1=1, c2=1)
    planner.replenish_threshold = 0.3
    planner.max_decisions = 20
    planner.max_place_targets = 8
    planner.trust_observed_areas = False
    planner.navigation_threshold = 1.5
    planner.verbose = False
    for key, value in overrides.items():
        setattr(planner, key, value)

    planner.reset()
    planner.llm = StubLLM(llm_responses or [""])
    return planner


# ---------------------------------------------------------------------------
# WorldGraph -> Observation
# ---------------------------------------------------------------------------


def test_domain_is_built_from_the_world_graph():
    planner = make_planner()
    planner._sync_domain(make_world_graph())
    domain = planner.domain
    assert domain.furniture_room == {
        "counter_24": "kitchen_1",
        "cabinet_3": "kitchen_1",
        "fridge_0": "kitchen_1",
        "table_10": "living_room_1",
    }
    assert domain.articulated == {"cabinet_3", "fridge_0"}
    assert domain.room_furniture["kitchen_1"] == ["cabinet_3", "counter_24", "fridge_0"]
    # Distances come from the node translations.
    assert abs(domain.distance("counter_24", "cabinet_3") - 3.0) < 1e-6
    # Only the lamp declares object states, so only it is restricted.
    assert domain.state_affordances == {"lamp_0": {"is_powered_on"}}


def test_observation_matches_the_domain_observation_contract():
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    observation = planner._build_observation(graph)

    assert observation.furniture_open == {"cabinet_3": False, "fridge_0": False}
    assert observation.object_parent == {"plate_2": "counter_24", "lamp_0": "table_10"}
    assert observation.object_states["lamp_0"] == {"is_powered_on": True}
    assert observation.robot_area == "counter_24"
    # Areas holding an observed object count as seen but not as deliberately
    # inspected, so a missing object there cannot refute a hypothesis yet.
    assert observation.inspected_areas == {"counter_24", "table_10"}
    assert observation.fully_inspected_areas == set()


def test_deliberate_inspection_makes_an_area_able_to_refute_a_hypothesis():
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    planner._deliberately_inspected.add("cabinet_3")
    observation = planner._build_observation(graph)
    assert "cabinet_3" in observation.fully_inspected_areas


def test_trust_observed_areas_promotes_incidental_sightings():
    planner = make_planner(trust_observed_areas=True)
    graph = make_world_graph()
    planner._sync_domain(graph)
    observation = planner._build_observation(graph)
    assert observation.fully_inspected_areas == {"counter_24", "table_10"}


# ---------------------------------------------------------------------------
# Symbolic action -> Habitat skills
# ---------------------------------------------------------------------------


def test_open_gets_an_implicit_navigate_when_out_of_range():
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    queue = planner._translate(
        Action(ActionType.OPEN, area="cabinet_3"), graph, SceneState()
    )
    assert queue == [("Navigate", "cabinet_3"), ("Open", "cabinet_3")]


def test_no_implicit_navigate_when_already_in_range():
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    # The robot is at the counter already.
    queue = planner._translate(
        Action(ActionType.PICK, area="counter_24", obj="plate_2"), graph, SceneState()
    )
    assert queue == [("Pick", "plate_2")]


def test_place_uses_the_partnr_argument_format():
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    scene = SceneState(object_parent={"plate_2": HELD})

    plain = planner._translate(
        Action(ActionType.PLACE, area="counter_24", relation="on"), graph, scene
    )
    assert plain == [("Place", "plate_2, on, counter_24, None, None")]

    anchored = planner._translate(
        Action(
            ActionType.PLACE, area="counter_24", relation="within", next_to="lamp_0"
        ),
        graph,
        scene,
    )
    assert anchored == [("Place", "plate_2, within, counter_24, next_to, lamp_0")]


def test_explore_takes_a_room_name_and_needs_no_navigate():
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    assert planner._translate(
        Action(ActionType.EXPLORE, area="kitchen_1"), graph, SceneState()
    ) == [("Explore", "kitchen_1")]


def test_object_state_skills_navigate_to_the_object_parent():
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    scene = SceneState(object_parent={"lamp_0": "table_10"})
    assert planner._translate(
        Action(ActionType.POWER_OFF, obj="lamp_0"), graph, scene
    ) == [("Navigate", "table_10"), ("PowerOff", "lamp_0")]


def test_null_maps_to_wait():
    planner = make_planner()
    graph = make_world_graph()
    planner._sync_domain(graph)
    assert planner._translate(Action(ActionType.NULL), graph, SceneState()) == [
        ("Wait", "")
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response,expected",
    [
        ("Successful execution!", True),
        ("Successful execution of Place.", True),
        ("Unexpected failure! - Object is not in the agent's hand", False),
        ("Node with name cup_9 does not exist in the graph", False),
        ("Agent could not reach the target", False),
        ("Skill timeout", False),
    ],
)
def test_response_is_success(response, expected):
    assert TruPOMDPPlanner._response_is_success(response) is expected


# ---------------------------------------------------------------------------
# Skill-hold loop
# ---------------------------------------------------------------------------


def hidden_mug_llm() -> StubLLM:
    return StubLLM(
        [
            fenced(
                {
                    "answer": [
                        {
                            "objects": [{"object": "mug", "target_area": "table_10"}],
                            "probability": 1.0,
                        }
                    ]
                }
            ),
            fenced({"answer": [{"initial_area": "cabinet_3", "probability": 1.0}]}),
        ]
    )


class RecordingExecutor:
    """Stands in for Planner.process_high_level_actions."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, hl_actions, observations):
        self.calls.append(dict(hl_actions))
        response = self.responses.pop(0) if self.responses else "Successful execution!"
        return {0: "low_level"}, {0: response}


def test_planner_holds_the_skill_until_the_response_is_non_empty():
    planner = make_planner()
    planner.llm = hidden_mug_llm()
    graph = make_world_graph()
    executor = RecordingExecutor(["", "", "Successful execution!"])
    planner.process_high_level_actions = executor

    # First call: plan, then issue the implicit Navigate.
    _, info, done = planner.get_next_action("put the mug away", {}, {0: graph})
    assert not done
    assert info["replanned"][0] is True
    assert planner.last_high_level_actions[0][0] == "Navigate"
    first_action = planner.last_high_level_actions[0]

    # Second call: the skill is still running, so the identical action is re-issued
    # and nothing is replanned.
    _, info, done = planner.get_next_action("put the mug away", {}, {0: graph})
    assert info["replanned"][0] is False
    assert planner.last_high_level_actions[0] == first_action
    assert executor.calls[1] == executor.calls[0]

    # Third call: the Navigate reports success, so it is cleared and the area is
    # recorded as deliberately inspected.
    _, _, done = planner.get_next_action("put the mug away", {}, {0: graph})
    assert planner.last_high_level_actions == {}
    assert "cabinet_3" in planner._deliberately_inspected
    # The manipulation is still queued, so no replanning happened.
    assert planner._queue == [("Open", "cabinet_3")]
    assert planner._symbolic_action == Action(ActionType.OPEN, area="cabinet_3")


def test_planner_plans_an_open_for_a_hypothesised_hidden_object():
    planner = make_planner()
    planner.llm = hidden_mug_llm()
    graph = make_world_graph()
    planner.process_high_level_actions = RecordingExecutor([""])

    planner.get_next_action("put the mug away", {}, {0: graph})
    assert planner._symbolic_action == Action(ActionType.OPEN, area="cabinet_3")
    particle = planner.belief.map_particle()
    assert particle.goal_atoms == (GoalAtom("mug", "table_10"),)
    assert particle.scene.object_parent["mug"] == "cabinet_3"
    assert particle.scene.hypothesized == {"mug"}


def test_hybrid_update_runs_once_the_symbolic_action_completes():
    planner = make_planner()
    planner.llm = hidden_mug_llm()
    graph = make_world_graph()
    planner.process_high_level_actions = RecordingExecutor(
        ["Successful execution!", "Successful execution!"]
    )

    # Navigate completes, then Open completes.
    planner.get_next_action("put the mug away", {}, {0: graph})
    assert planner._queue == [("Open", "cabinet_3")]
    planner.get_next_action("put the mug away", {}, {0: graph})
    assert planner._queue == []
    # The symbolic action was consumed by the hybrid belief update.
    assert planner._symbolic_action is None
    assert "Belief:" in planner._trace
    assert planner.belief is not None and len(planner.belief) >= 1


def test_failed_skill_is_recorded_and_clears_the_queue():
    planner = make_planner()
    planner.llm = hidden_mug_llm()
    graph = make_world_graph()
    planner.process_high_level_actions = RecordingExecutor(
        ["Unexpected failure! - Could not navigate to cabinet_3"]
    )

    planner.get_next_action("put the mug away", {}, {0: graph})
    assert planner._queue == []
    assert planner.last_high_level_actions == {}
    assert planner._failed_attempts
    assert "cabinet_3" not in planner._deliberately_inspected


def test_planner_stops_when_the_toh_produces_nothing():
    planner = make_planner()
    planner.llm = StubLLM(["I cannot help with that."])
    graph = make_world_graph()
    planner.process_high_level_actions = RecordingExecutor([""])

    low_level, _, done = planner.get_next_action("do something", {}, {0: graph})
    assert done is True
    assert low_level == {}


class ScriptedWorld:
    """
    Stands in for process_high_level_actions and applies each skill's effect to
    the world graph, so the planner sees real consequences. The mug the TOH
    hypothesises exists as coffee_mug_5 inside the closed cabinet.
    """

    def __init__(self, graph: WorldGraph) -> None:
        self.graph = graph
        self.log = []
        self.hidden = Object("coffee_mug_5", {"type": "mug", "states": {}})

    def __call__(self, hl_actions, observations):
        name, args, _ = hl_actions[0]
        self.log.append((name, args))
        robot = self.graph.get_spot_robot()
        if name == "Navigate":
            target = self.graph.get_node_from_name(args)
            robot.properties["translation"] = list(target.properties["translation"])
        elif name == "Open":
            furniture = self.graph.get_node_from_name(args)
            furniture.properties["states"]["is_open"] = True
            if not self.graph.has_node(self.hidden):
                self.graph.add_node(self.hidden)
                self.graph.add_edge(self.hidden, furniture, "within", "contains")
        elif name == "Pick":
            obj = self.graph.get_node_from_name(args)
            for neighbour in list(self.graph.graph[obj]):
                self.graph.remove_edge(obj, neighbour)
            self.graph.add_edge(obj, robot, "held_by", "holding")
        elif name == "Place":
            parts = [part.strip() for part in args.split(",")]
            obj = self.graph.get_node_from_name(parts[0])
            furniture = self.graph.get_node_from_name(parts[2])
            for neighbour in list(self.graph.graph[obj]):
                self.graph.remove_edge(obj, neighbour)
            self.graph.add_edge(obj, furniture, parts[1], "supports")
        return {0: "low_level"}, {0: "Successful execution!"}


def test_end_to_end_hidden_object_discovery_and_grounding():
    planner = make_planner()
    planner.llm = hidden_mug_llm()
    graph = make_world_graph()
    world = ScriptedWorld(graph)
    planner.process_high_level_actions = world

    done = False
    for _ in range(30):
        _, _, done = planner.get_next_action(
            "put the mug on the living room table", {}, {0: graph}
        )
        if done:
            break
    assert done, "the planner never finished"

    assert world.log == [
        ("Navigate", "cabinet_3"),
        ("Open", "cabinet_3"),
        ("Pick", "coffee_mug_5"),
        ("Navigate", "table_10"),
        ("Place", "coffee_mug_5, on, table_10, None, None"),
    ]

    particle = planner.belief.map_particle()
    # The hypothesised name was grounded onto the real Habitat id, in the scene
    # and in the goal.
    assert particle.scene.grounding == {"mug": "coffee_mug_5"}
    assert particle.goal_atoms == (GoalAtom("coffee_mug_5", "table_10"),)
    assert planner.domain.goal_satisfied(particle.scene, particle.goal_atoms)
    # Three symbolic decisions and only the two initial TOH queries.
    assert planner._num_decisions == 3
    assert len(planner.llm.prompts) == 2


def test_reset_clears_belief_and_bookkeeping():
    planner = make_planner()
    planner.llm = hidden_mug_llm()
    graph = make_world_graph()
    planner.process_high_level_actions = RecordingExecutor(["Successful execution!"])
    planner.get_next_action("put the mug away", {}, {0: graph})
    assert planner.belief is not None

    planner.reset()
    assert planner.belief is None
    assert planner.domain is None
    assert planner._queue == []
    assert planner._num_decisions == 0
    assert planner._deliberately_inspected == set()
    assert planner.is_done is False
