from habitat_llm.utils.episode_entity_names import (
    furniture_names_by_sim_handle,
    graph_object_name,
    object_names_by_sim_handle,
)


def test_saved_furniture_names_override_generation_order():
    scene = {"receptacle_to_handle": {"table_11": "desk", "cabinet_5": "sink"}}
    generated = {"table_0": "desk", "cabinet_0": "sink", "couch_0": "sofa"}
    assert furniture_names_by_sim_handle(scene, generated) == {
        "desk": "table_11",
        "sink": "cabinet_5",
        "sofa": "couch_0",
    }


def test_furniture_names_without_scene_info_use_generation_mapping():
    assert furniture_names_by_sim_handle({}, {"table_0": "desk"}) == {"desk": "table_0"}


def test_furniture_alias_does_not_replace_canonical_name():
    scene = {
        "receptacle_to_handle": {"table_16": "coffee", "table_2": "coffee"},
        "spec_furniture_aliases": {"table_16": {"runtime_name": "table_2"}},
    }
    assert furniture_names_by_sim_handle(scene) == {"coffee": "table_2"}


def test_generated_furniture_cannot_take_saved_name_for_another_handle():
    scene = {"receptacle_to_handle": {"table_0": "saved-desk"}}
    assert furniture_names_by_sim_handle(scene, {"table_0": "new-table"}) == {
        "saved-desk": "table_0"
    }


def test_object_names_from_entity_handles():
    ep_info = {
        "info": {
            "variant_spec": {
                "entity_handles": {
                    "bread_0": "Bread_8_:0000",
                    "bottle_0": "03758534dd2a3a8303e742cf4fc10fedd4c48843_:0000",
                    "hand_towel_0": "Tag_Dishtowel_Green_:0000",
                    "lamp_living_0": "B07HK3PNSK_:0000",
                }
            },
            "sample_configs": {
                "1": {
                    "name": "bottle_0",
                    "object_instances": [
                        "03758534dd2a3a8303e742cf4fc10fedd4c48843"
                    ],
                }
            },
        }
    }
    names = object_names_by_sim_handle(ep_info)
    assert names["03758534dd2a3a8303e742cf4fc10fedd4c48843_:0000"] == "bottle_0"
    assert names["Bread_8_:0000"] == "bread_0"
    assert names["B07HK3PNSK_:0000"] == "lamp_living_0"


def test_graph_object_name_prefers_dataset_id():
    names = {
        "03758534dd2a3a8303e742cf4fc10fedd4c48843_:0000": "bottle_0",
    }
    assert (
        graph_object_name(
            "03758534dd2a3a8303e742cf4fc10fedd4c48843_:0000",
            "bottle",
            1,
            names,
            {"bread_0"},
        )
        == "bottle_0"
    )


def test_graph_object_name_falls_back_to_insertion_index():
    assert graph_object_name("unknown_handle_:0000", "bottle", 1, {}, set()) == "bottle_1"
