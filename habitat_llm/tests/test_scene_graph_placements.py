"""All three baseline prompts distinguish furniture interiors from surfaces."""

from types import SimpleNamespace

import pytest

from habitat_llm.llm.instruct.utils import get_objects_descr
from habitat_llm.planner.llm_planner import LLMPlanner
from habitat_llm.planner.tru_pomdp.scene import HELD, Action, ActionType
from habitat_llm.planner.tru_pomdp.toh import build_observation_text
from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner
from habitat_llm.tests.test_tru_pomdp_planner import make_planner, make_world_graph
from habitat_llm.world_model import Receptacle, WorldGraph


def placed_graph(edge="on", receptacle_type=None):
    graph = make_world_graph()
    graph.remove_edge("plate_2", "counter_24")
    target = graph.get_node_from_name("cabinet_3")
    if receptacle_type:
        rec = Receptacle("rec_cabinet_3_0", {"type": receptacle_type})
        graph.add_node(rec)
        graph.add_edge(rec, target, "joint", "joint")
        target = rec
    graph.add_edge("plate_2", target, edge, "supports")
    return graph


def render_baseline(baseline, graph):
    if baseline == "react":
        planner = LLMPlanner.__new__(LLMPlanner)
        planner.llm = SimpleNamespace()
        planner._agents = [SimpleNamespace(uid=0, tools={}, reset=lambda: None)]
        planner.prompt = "{world_description}"
        planner.planner_config = SimpleNamespace(
            objects_response_include_states=True, centralized=False
        )
        planner.reset()
        return planner.prepare_prompt("Move the plate", graph)[0]
    if baseline == "vlm-tamp":
        planner = VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
        return planner._build_scene_description(graph, 0)
    planner = make_planner()
    planner._sync_domain(graph)
    return build_observation_text(planner._build_observation(graph), planner.domain)


@pytest.mark.parametrize("baseline", ["react", "vlm-tamp", "tru-pomdp"])
@pytest.mark.parametrize("edge,rec_type,expected", [
    ("on", "within", "inside"),  # Simulator 'on' edge on an interior shelf.
    ("on", "on", "on"),
    ("inside", None, "inside"),
    ("within", None, "inside"),
    ("in", None, "inside"),
    ("on", None, "on"),  # Cabinet tops are not automatically interiors.
])
def test_baseline_placement_prompts(baseline, edge, rec_type, expected):
    graph = placed_graph(edge, rec_type)
    text = render_baseline(baseline, graph)
    prefix = "plate_2 is " if baseline == "tru-pomdp" else "plate_2: "
    assert prefix + expected + " cabinet_3" in text
    assert "plate_2: " + expected + " cabinet_3" in graph.get_world_descr()


def test_states_rooms_held_and_unknown_are_preserved():
    graph = placed_graph("on", "within")
    graph.get_node_from_name("plate_2").properties["states"] = {"is_clean": True}
    text = get_objects_descr(graph, include_room_name=True, add_state_info=True)
    assert "plate_2: inside cabinet_3 in kitchen_1. States:" in text
    graph.remove_edge("plate_2", "rec_cabinet_3_0")
    assert "plate_2: unknown" in get_objects_descr(graph)
    graph.add_edge("plate_2", "agent_0", "on", "holds")
    assert "plate_2: held by the agent" in get_objects_descr(graph)
    assert "plate_2 is held by the robot" in render_baseline("tru-pomdp", graph)
    graph.remove_edge("plate_2", "agent_0")
    graph.add_edge("plate_2", "cabinet_3", "unknown", "unknown")
    assert graph.get_object_furniture_relation(graph.get_node_from_name("plate_2")) is None
    assert "plate_2 is at cabinet_3" in render_baseline("tru-pomdp", graph)


def test_tru_placement_memory_survives_missing_frame_and_updates():
    planner = make_planner()
    graph = placed_graph("on", "within")
    planner._sync_domain(graph)
    planner._remember_sensor_graph(graph)
    planner._fresh_graph = WorldGraph()
    observation = planner._build_observation(graph)
    assert "plate_2 is inside cabinet_3" in build_observation_text(observation, planner.domain)
    # Seeing the same object on the cabinet top replaces the remembered interior.
    graph = placed_graph("on")
    planner._remember_sensor_graph(graph)
    observation = planner._build_observation(graph)
    assert "plate_2 is on cabinet_3" in build_observation_text(observation, planner.domain)
    planner.reset()
    assert planner._observed_placements == {}


@pytest.mark.parametrize("placement,expected", [("within", "inside"), ("on", "on")])
def test_tru_pick_and_place_update_placement_memory(placement, expected):
    planner = make_planner()
    graph = placed_graph("on", "within")
    planner._sync_domain(graph)
    planner._remember_sensor_graph(graph)
    planner._symbolic_action = Action(ActionType.PICK, obj="plate_2")
    planner._on_skill_finished("Successful execution!", graph)
    assert planner._observed_locations["plate_2"] == HELD
    assert planner._observed_placements["plate_2"] is None
    planner._symbolic_action = Action(ActionType.PLACE, area="cabinet_3", relation=placement)
    planner._on_skill_finished("Failed to place object", graph)
    assert planner._observed_locations["plate_2"] == HELD
    planner._on_skill_finished("Successful execution!", graph)
    assert planner._observed_locations["plate_2"] == "cabinet_3"
    assert planner._observed_placements["plate_2"] == expected
