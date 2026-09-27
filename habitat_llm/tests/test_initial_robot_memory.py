import json
from pathlib import Path

from habitat_llm.utils.initial_robot_memory import (
    RememberedObject,
    apply_outdated_placements,
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
