"""Spawn-room context is emitted once per episode across the baselines."""

from types import SimpleNamespace

import pytest

from habitat_llm.llm.instruct.utils import get_initial_agent_descr, get_world_descr
from habitat_llm.perception.perception_sim import PerceptionSim
from habitat_llm.planner.llm_planner import LLMPlanner
from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner
from habitat_llm.world_model import Room, WorldGraph


def make_perception(start_x=0):
    perception = PerceptionSim.__new__(PerceptionSim)
    agent = SimpleNamespace(base_pos=[start_x, 0, 0])
    perception.sim = SimpleNamespace(
        agents_mgr=SimpleNamespace(
            agent_names=["agent_0"],
            _all_agent_data=[SimpleNamespace(articulated_agent=agent)],
        ),
        semantic_scene=SimpleNamespace(
            regions=[
                SimpleNamespace(id="bedroom", contains=lambda pos: pos[0] == 0),
                SimpleNamespace(id="kitchen", contains=lambda pos: pos[0] == 1),
            ]
        ),
    )
    perception.region_id_to_name = {"bedroom": "bedroom_3", "kitchen": "kitchen_1"}
    perception.sim_handle_to_name = {}
    perception.gt_graph = WorldGraph()
    for name in ("bedroom_3", "kitchen_1", "unknown_room"):
        perception.gt_graph.add_node(Room(name, {"type": "room"}))
    perception.add_agents_to_gt_graph()
    return perception, agent


@pytest.mark.parametrize("partial_obs", [False, True])
def test_spawn_room_survives_movement_and_graph_updates(partial_obs):
    perception, agent = make_perception()
    graph = WorldGraph()
    graph.update(perception.gt_graph, partial_obs, "gt")
    agent.base_pos = [1, 0, 0]
    perception.update_agent_room_associations()
    graph.update(perception.gt_graph, partial_obs, "gt")
    assert graph.get_room_for_entity(graph.get_spot_robot()).name == "kitchen_1"
    assert "agent_0: bedroom_3" in get_initial_agent_descr(graph)
    assert "agent_0:" not in get_world_descr(graph)


def test_unknown_spawn_room_is_not_guessed():
    perception, _ = make_perception(start_x=2)
    assert "agent_0: unknown_room" in get_initial_agent_descr(perception.gt_graph)


@pytest.mark.parametrize("baseline", ["react", "vlm_tamp"])
def test_initial_prompt_only_and_episode_reset(baseline):
    if baseline == "react":
        planner = LLMPlanner.__new__(LLMPlanner)
        planner.llm = SimpleNamespace()
        planner._agents = [SimpleNamespace(uid=0, tools={}, reset=lambda: None)]
        planner.prompt = "{world_description}"
        planner.planner_config = SimpleNamespace(
            objects_response_include_states=True, centralized=False
        )
        render = lambda graph: planner.prepare_prompt("Move bread", graph)[0]
    else:
        planner = VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
        planner.vlm = SimpleNamespace()
        render = lambda graph: planner._build_scene_description(graph, 0)
    planner.reset()
    perception, agent = make_perception()
    memory = "Initial robot memory: jug_0 known absent; pitcher_0 on counter_0.\n"
    perception.gt_graph.initial_robot_memory_prompt = memory
    # Exploration before the first model call must not change the spawn room.
    agent.base_pos = [1, 0, 0]
    perception.update_agent_room_associations()
    first = render(perception.gt_graph)
    assert first.count("agent_0: bedroom_3") == 1
    assert first.count(memory) == 1
    assert memory not in render(perception.gt_graph)
    assert "agent_0:" not in render(perception.gt_graph)
    planner.reset()
    next_episode, _ = make_perception(start_x=1)
    next_episode.gt_graph.initial_robot_memory_prompt = "New episode memory\n"
    assert memory not in render(next_episode.gt_graph)
    planner.reset()
    assert "agent_0: kitchen_1" in render(next_episode.gt_graph)
    assert "agent_0:" not in render(next_episode.gt_graph)
