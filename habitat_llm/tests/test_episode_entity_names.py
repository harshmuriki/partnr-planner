from habitat_llm.utils.episode_entity_names import (
    graph_object_name,
    object_names_by_sim_handle,
)


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
