"""Furniture identity survives graph ordering, floors, and partial scene metadata."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import habitat_llm.perception.perception_sim as perception_module
from habitat_llm.perception.perception_sim import PerceptionSim
from habitat_llm.world_model import Floor, Room, WorldGraph


def test_legacy_runtime_name_aliases_the_spec_cabinet(monkeypatch):
    monkeypatch.setattr(perception_module, "get_receptacle_dict", lambda *a: {
        "sink": {"on": [SimpleNamespace(unique_name="sink-surface")]}
    })
    monkeypatch.setattr(perception_module, "get_faucet_points", lambda sim: {})
    monkeypatch.setattr(
        perception_module.sutils,
        "get_obj_from_handle",
        lambda *a: SimpleNamespace(translation=[0, 0, 0], is_articulated=False),
    )
    perception = PerceptionSim.__new__(PerceptionSim)
    perception.sim = SimpleNamespace(receptacles={})
    perception.scene_info = {
        "receptacle_to_handle": {"cabinet_6": "sink"},
        "spec_furniture_aliases": {"cabinet_34": {"runtime_name": "cabinet_6"}},
    }
    perception.metadata_interface = SimpleNamespace(metadata=object(), refresh_scene_caches=Mock())
    perception.get_furniture_property_from_metadata = lambda *a: "cabinet"
    perception.get_room_name = lambda handle: "kitchen_1"
    perception.sim_handle_to_name = {}
    perception.gt_graph = WorldGraph()
    perception.gt_graph.add_node(Room("kitchen_1", {"type": "room"}))
    perception.add_furniture_and_receptacles_to_gt_graph()
    assert perception.gt_graph.get_node_from_name("cabinet_34").name == "cabinet_6"


@pytest.mark.parametrize("saved", [True, False])
def test_room_name_reserves_saved_and_generated_furniture_names(monkeypatch, saved):
    perception = PerceptionSim.__new__(PerceptionSim)
    mapping = {"toilet_1": "toilet-handle"}
    perception.scene_info = {"receptacle_to_handle": mapping} if saved else {}
    refresh = Mock()
    perception.metadata_interface = SimpleNamespace(
        metadata=object(),
        recobj_semname_to_handle=mapping,
        refresh_scene_caches=refresh,
    )
    perception.sim = SimpleNamespace(
        receptacles={},
        semantic_scene=SimpleNamespace(
            regions=[
                SimpleNamespace(
                    id="region-toilet", category=SimpleNamespace(name=lambda: "toilet")
                ),
                SimpleNamespace(
                    id="region-kitchen", category=SimpleNamespace(name=lambda: "kitchen")
                ),
            ]
        )
    )
    perception.gt_graph = WorldGraph()
    perception.add_house_to_graph()
    perception.region_id_to_name = {}
    perception.sim_handle_to_name = {}
    perception.get_furniture_property_from_metadata = lambda *args: "toilet"
    perception.get_room_name = lambda handle: "toilet_room_1"
    monkeypatch.setattr(
        perception_module, "get_receptacle_dict",
        lambda *args: {"toilet-handle": {"on": [SimpleNamespace(unique_name="seat")]}},
    )
    monkeypatch.setattr(perception_module, "get_faucet_points", lambda sim: {})
    monkeypatch.setattr(
        perception_module.sutils, "get_obj_from_handle",
        lambda *args: SimpleNamespace(translation=[0, 0, 0], is_articulated=False),
    )
    monkeypatch.setattr(
        perception_module.sutils, "get_floor_point_in_region", lambda *a: [0, 0, 0]
    )
    perception.add_rooms_to_gt_graph()
    perception.add_floors_to_gt_graph()
    assert perception.region_id_to_name["region-toilet"] == "toilet_room_1"
    assert perception.region_id_to_name["region-kitchen"] == "kitchen_1"
    assert not perception.gt_graph.has_node("toilet_1")
    assert perception.gt_graph.has_node("floor_toilet_room_1")
    perception.add_furniture_and_receptacles_to_gt_graph()
    furniture = perception.gt_graph.get_node_from_sim_handle("toilet-handle")
    assert furniture.name == "toilet_1"
    assert perception.gt_graph.get_room_for_entity(furniture).name == "toilet_room_1"
    assert refresh.call_count == (0 if saved else 1)


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    "saved, expected",
    [
        (
            {"table_11": "desk", "table_7": "dining"},
            {"desk": "table_11", "dining": "table_7"},
        ),
        ({}, {"desk": "table_0", "dining": "table_1"}),
        # The unmapped desk must not take the dining table's saved name,
        # even when the desk is inserted first.
        ({"table_0": "dining"}, {"desk": "table_1", "dining": "table_0"}),
    ],
)
def test_perception_furniture_names(monkeypatch, reverse_order, saved, expected):
    handles = ["desk", "dining"]
    if reverse_order:
        handles.reverse()
    receptacles = {
        handle: {"on": [SimpleNamespace(unique_name=f"{handle}-surface")]}
        for handle in handles
    }
    monkeypatch.setattr(
        perception_module, "get_receptacle_dict", lambda *a: receptacles
    )
    monkeypatch.setattr(perception_module, "get_faucet_points", lambda sim: {})
    monkeypatch.setattr(
        perception_module.sutils,
        "get_obj_from_handle",
        lambda *a: SimpleNamespace(translation=[0, 0, 0], is_articulated=False),
    )
    perception = PerceptionSim.__new__(PerceptionSim)
    perception.sim = SimpleNamespace(receptacles={})
    perception.scene_info = {"receptacle_to_handle": saved}
    refresh = Mock()
    perception.metadata_interface = SimpleNamespace(
        metadata=object(),
        recobj_semname_to_handle={"table_0": "desk", "table_1": "dining"},
        refresh_scene_caches=refresh,
    )
    perception.get_furniture_property_from_metadata = lambda *a: "table"
    perception.get_room_name = lambda handle: "living_room_1"
    perception.sim_handle_to_name = {}
    perception.gt_graph = WorldGraph()
    perception.gt_graph.add_node(Room("living_room_1", {"type": "room"}))
    for index in range(4):
        perception.gt_graph.add_node(Floor(f"floor_{index}", {"type": "floor"}))

    perception.add_furniture_and_receptacles_to_gt_graph()

    for handle, name in expected.items():
        furniture = perception.gt_graph.get_node_from_sim_handle(handle)
        assert furniture.name == name
        assert perception.sim_handle_to_name[handle] == name
        assert (
            perception.gt_graph.get_room_for_entity(furniture).name == "living_room_1"
        )
        rec = perception.gt_graph.get_node_from_sim_handle(f"{handle}-surface")
        assert rec.name == f"rec_{name}_0"
        assert furniture in perception.gt_graph.get_neighbors(rec)
    assert refresh.call_count == (0 if len(saved) == 2 else 1)
