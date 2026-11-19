import json
from pathlib import Path
import pytest

from habitat_llm.utils.initial_robot_memory import (
    RememberedObject,
    apply_outdated_placements,
    load_initial_memory_prompt,
    load_remembered_object_names,
    load_remembered_object_records,
    load_remembered_object_refs,
    parse_outdated_location,
    resolve_initial_robot_memory_path,
    resolve_placement_node,
)
from habitat_llm.world_model import Floor, Furniture, House, Object, Room, WorldGraph
from habitat_llm.world_model.world_graph import flip_edge


def test_load_remembered_object_names_from_episode_folder(tmp_path: Path):
    payload = {
        "variant": "T1-INC-BASE",
        "objects": [
            {"entity": "jug_0", "in_initial_robot_memory": False},
            {"entity": "lamp_0", "in_initial_robot_memory": True},
        ],
    }
    (tmp_path / "initial_robot_memory.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    assert resolve_initial_robot_memory_path(str(tmp_path)) == str(
        tmp_path / "initial_robot_memory.json"
    )
    assert load_remembered_object_names(str(tmp_path)) == ["lamp_0"]
    assert load_remembered_object_names(
        str(tmp_path / "dataset.json.gz")
    ) == ["lamp_0"]


def test_load_remembered_object_refs_uses_variant_handles(tmp_path: Path):
    payload = {
        "variant": "T1-ACC-BASE",
        "objects": [
            {"entity": "lamp_0", "in_initial_robot_memory": True},
            {"entity": "jug_0", "in_initial_robot_memory": False},
        ],
    }
    (tmp_path / "initial_robot_memory.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    dataset = {
        "episodes": [
            {
                "info": {
                    "variant_spec": {
                        "entity_handles": {
                            "lamp_0": "lamp-handle_:0000",
                            "jug_0": "jug-handle_:0000",
                        }
                    }
                }
            }
        ]
    }
    (tmp_path / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")

    assert load_remembered_object_refs(str(tmp_path)) == [
        ("lamp_0", "lamp-handle_:0000")
    ]


def test_missing_memory_file_returns_empty(tmp_path: Path):
    assert resolve_initial_robot_memory_path(str(tmp_path)) is None
    assert load_remembered_object_names(str(tmp_path)) == []
    assert load_remembered_object_names(None) == []
    assert load_initial_memory_prompt(str(tmp_path)) == ""


@pytest.mark.parametrize("mode", ["ACC", "INC", "OUT"])
def test_startup_memory_uses_gt_and_omits_out_block(tmp_path, mode):
    graph, scissors, table, _, box = _memory_graph()
    scissors.properties["states"] = {"is_clean": False}
    payload = {
        "variant": f"T1-{mode}-SUB",
        "objects": [
            {"entity": "scissors_0", "in_initial_robot_memory": True,
             **({"outdated_location": "on stale_table (stale_room)"} if mode == "OUT" else {})},
            {"entity": "box_1", "in_initial_robot_memory": False},
        ],
        "initial_memory": [
            "scissors_0: on viewer_table (viewer_room), is_clean",
            "jug_0: known absent; no jug exists anywhere in the scene",
        ],
    }
    path = tmp_path / "initial_robot_memory.json"
    path.write_text(json.dumps(payload))
    prompt = load_initial_memory_prompt(str(tmp_path), graph)
    assert "box_1" not in prompt and "viewer_" not in prompt
    assert ("jug_0: known absent" in prompt) == (mode == "ACC")
    assert load_remembered_object_names(str(tmp_path)) == ["scissors_0"]
    if mode == "OUT":
        assert prompt == ""
        assert load_initial_memory_prompt(str(tmp_path)) == ""
    else:
        assert "scissors_0: on chest_of_drawers_56 in living_room_1" in prompt
        assert "clean: False" in prompt
        graph.remove_edge(scissors, graph.find_furniture_for_object(scissors))
        graph.add_edge(scissors, table, "inside", "contains")
        assert "inside table_11 in dining_room_1" in load_initial_memory_prompt(str(tmp_path), graph)
        with pytest.raises(ValueError, match="ground-truth graph"):
            load_initial_memory_prompt(str(tmp_path))


def test_accurate_files_match_viewer_and_sync_preserves_other_memories(tmp_path):
    import shutil
    from dataset_generation.benchmark_generation.sync_accurate_initial_memory import memory_items, sync

    root = Path(__file__).resolve().parents[2] / "baseline_evaluation_v3"
    count = 0
    for path in (root / "episodes").glob("task_*/*/initial_robot_memory.json"):
        data = json.loads(path.read_text())
        source_type = "remembered_stale_locations" if "-OUT-" in data["variant"] else "ground_truth_at_start"
        assert data["initial_memory_source"] == source_type
        if "-ACC-" in data["variant"]:
            source = root / "specs" / data["variant"].split("-")[0] / (data["variant"] + ".md")
            assert data["initial_memory"] == [x for x in memory_items(source.read_text()) if ": known absent" in x]
        count += 1
    assert count == 90
    for variant in ("t1-acc-abs", "t1-inc-abs", "t1-out-abs"):
        dest = tmp_path / "episodes/task_1" / variant
        dest.mkdir(parents=True)
        shutil.copy2(root / "episodes/task_1" / variant / "initial_robot_memory.json", dest)
    spec_dir = tmp_path / "specs/T1"
    spec_dir.mkdir(parents=True)
    shutil.copy2(root / "specs/T1/T1-ACC-ABS.md", spec_dir)
    before = {p: p.read_bytes() for p in (tmp_path / "episodes").glob("task_*/*/*.json")}
    assert sync(tmp_path) == 3
    assert all(p.read_bytes() == content for p, content in before.items())


def test_environment_attaches_memory_without_seeding_missing_entities(tmp_path):
    from types import SimpleNamespace
    from habitat_llm.agent.env.environment_interface import EnvironmentInterface

    (tmp_path / "initial_robot_memory.json").write_text(json.dumps({
        "variant": "T1-ACC-ABS", "objects": [],
        "initial_memory": ["jug_0: known absent"],
    }))
    env = EnvironmentInterface.__new__(EnvironmentInterface)
    env.conf = SimpleNamespace(habitat=SimpleNamespace(dataset=SimpleNamespace(data_path=str(tmp_path))))
    env.partial_obs = True
    env.perception = SimpleNamespace(gt_graph=WorldGraph())
    env.world_graph = {0: WorldGraph()}
    env._seed_initial_robot_memory()
    assert "jug_0: known absent" in env.world_graph[0].initial_robot_memory_prompt
    assert env.world_graph[0].get_all_objects() == []


@pytest.mark.parametrize("mode", ["ACC", "INC", "OUT"])
def test_environment_uses_gt_selection_but_preserves_out_ghosts(tmp_path, mode):
    import copy
    from types import SimpleNamespace
    from habitat_llm.agent.env.environment_interface import EnvironmentInterface

    gt, scissors, table, _, _ = _memory_graph()
    agent_graph = copy.deepcopy(gt)
    for obj in list(agent_graph.get_all_objects()):
        agent_graph.remove_node(obj)
    record = {"entity": "scissors_0", "in_initial_robot_memory": True}
    if mode == "OUT":
        record["outdated_location"] = "on table_11 (dining_room_1)"
    (tmp_path / "initial_robot_memory.json").write_text(json.dumps({
        "variant": f"T2-{mode}-BASE", "objects": [record,
            {"entity": "box_1", "in_initial_robot_memory": False}],
    }))
    env = EnvironmentInterface.__new__(EnvironmentInterface)
    env.conf = SimpleNamespace(habitat=SimpleNamespace(dataset=SimpleNamespace(data_path=str(tmp_path))))
    env.partial_obs = True
    env.wm_update_mode = "gt"
    env.perception = SimpleNamespace(gt_graph=gt)
    env.world_graph = {0: agent_graph}
    env._seed_initial_robot_memory()
    objects = agent_graph.get_all_objects()
    assert len(objects) == 1
    expected = "table_11" if mode == "OUT" else "chest_of_drawers_56"
    assert agent_graph.find_furniture_for_object(objects[0]).name == expected
    if mode != "OUT":
        assert expected in agent_graph.initial_robot_memory_prompt
    assert "box_1" not in agent_graph.initial_robot_memory_prompt
    if mode == "OUT":
        assert agent_graph.initial_robot_memory_prompt == ""
        assert objects[0].sim_handle.endswith(".stale_memory")
        from habitat_llm.llm.instruct.utils import get_objects_descr
        assert "on table_11 in dining_room_1" in get_objects_descr(
            agent_graph, include_room_name=True)
    assert gt.find_furniture_for_object(scissors).name == "chest_of_drawers_56"


def test_parse_outdated_location():
    on_table = parse_outdated_location("on table_0 (dining_room_0)")
    assert on_table is not None
    assert on_table.relation == "on"
    assert on_table.name == "table_0"
    assert on_table.room == "dining_room_0"

    floor = parse_outdated_location(
        "floor floor_living_room_0 (living_room_0)"
    )
    assert floor is not None
    assert floor.relation == "floor"
    assert floor.name == "floor_living_room_0"
    assert floor.room == "living_room_0"

    assert parse_outdated_location("") is None


def test_load_outdated_memory_records(tmp_path: Path):
    payload = {
        "variant": "T2-OUT-BASE",
        "objects": [
            {
                "entity": "box_0",
                "in_initial_robot_memory": True,
                "memory_status": "accurate",
            },
            {
                "entity": "scissors_0",
                "in_initial_robot_memory": True,
                "memory_status": "outdated",
                "outdated_location": "on table_0 (dining_room_0)",
            },
        ],
    }
    (tmp_path / "initial_robot_memory.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    records = load_remembered_object_records(str(tmp_path))
    assert records[0].entity == "box_0"
    assert records[0].outdated_location is None
    assert records[1].entity == "scissors_0"
    assert records[1].outdated_location == "on table_0 (dining_room_0)"


def _memory_graph():
    graph = WorldGraph()
    house = House("house", {"type": "root"})
    dining = Room("dining_room_1", {"type": "room"})
    living = Room("living_room_1", {"type": "room"})
    table = Furniture(
        "table_11",
        {"type": "table", "translation": [1.0, 0.0, 1.0]},
        sim_handle="dining-table-handle_:0000",
    )
    chest = Furniture(
        "chest_of_drawers_56",
        {"type": "chest_of_drawers", "translation": [2.0, 0.0, 0.0]},
        sim_handle="chest-handle_:0000",
    )
    living_floor = Floor("floor_living_room_1", {"type": "floor"})
    scissors = Object("scissors_0", {"type": "scissors"})
    box = Object("box_1", {"type": "box"})
    for node in (house, dining, living, table, chest, living_floor, scissors, box):
        graph.add_node(node)
    graph.add_edge(dining, house, "in", "contains")
    graph.add_edge(living, house, "in", "contains")
    graph.add_edge(table, dining, "in", "contains")
    graph.add_edge(chest, living, "in", "contains")
    graph.add_edge(living_floor, living, "inside", flip_edge("inside"))
    graph.add_edge(scissors, chest, "on", flip_edge("on"))
    graph.add_edge(box, living_floor, "on", flip_edge("on"))
    return graph, scissors, table, living_floor, box


def test_resolve_and_apply_outdated_table_placement():
    graph, scissors, table, _, _ = _memory_graph()
    scene_info = {
        "receptacle_to_handle": {"table_0": "dining-table-handle_:0000"}
    }
    placement = resolve_placement_node(
        graph, "on table_0 (dining_room_0)", scene_info
    )
    assert placement is table
    warnings = apply_outdated_placements(
        graph,
        graph,
        [
            RememberedObject(
                entity="scissors_0",
                sim_handle=None,
                outdated_location="on table_0 (dining_room_0)",
            )
        ],
        scene_info,
    )
    assert warnings == []
    assert graph.find_furniture_for_object(scissors) is graph.get_node_from_name(
        "chest_of_drawers_56"
    )
    ghost = graph.get_node_from_name("scissors_1")
    assert isinstance(ghost, Object)
    assert ghost.sim_handle == "scissors_1.stale_memory"
    assert graph.find_furniture_for_object(ghost) is table


def test_apply_outdated_floor_and_ghost_object():
    graph, _, _, living_floor, box = _memory_graph()
    scene_info = {}
    warnings = apply_outdated_placements(
        graph,
        graph,
        [
            RememberedObject(
                entity="box_1",
                sim_handle=None,
                outdated_location="floor floor_living_room_0 (living_room_0)",
            ),
            RememberedObject(
                entity="missing_0",
                sim_handle=None,
                outdated_location="floor floor_living_room_0 (living_room_0)",
            ),
        ],
        scene_info,
    )
    assert warnings == []
    assert graph.find_furniture_for_object(box) is living_floor
    ghost_box = graph.get_node_from_name("box_2")
    assert isinstance(ghost_box, Object)
    assert graph.find_furniture_for_object(ghost_box) is living_floor
    ghost = graph.get_node_from_name("missing_1")
    assert isinstance(ghost, Object)
    assert ghost.sim_handle == "missing_1.stale_memory"
    assert graph.find_furniture_for_object(ghost) is living_floor
